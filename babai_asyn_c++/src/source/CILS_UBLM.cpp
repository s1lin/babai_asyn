#include <random>

/** \file
 * \brief Computation of Block Babai Algorithm
 * \author Shilei Lin
 * This file is part of CILS.
 *   CILS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   CILS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR P_A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with CILS.  If not, see <http://www.gnu.org/licenses/>.
 */


namespace cils {
    template<typename scalar, typename index>
    class CILS_UBLM {

    public:
        index m, n, upper, lower, search_iter;
        scalar tolerance;
        b_vector x_hat{}, y{};
        b_matrix A{}, I{};
        std::vector<std::vector<double>> permutation;

        explicit CILS_UBLM(CILS<scalar, index> &cils) {
            this->n = cils.n;
            this->m = cils.m;
            this->upper = cils.upper;
            this->search_iter = cils.search_iter;
            this->y.assign(cils.y);
            this->x_hat.resize(n);
            this->x_hat.clear();
            this->A.resize(m, n, false);
            this->A.assign(cils.A);
            this->I.assign(cils.I);

            //TODO:
//            this->tolerance = cils.tolerance;
            this->permutation = cils.permutation;
            this->upper = cils.upper;
            this->lower = cils.lower;
        }

        /*
         *  // Corresponds to Algorithm 2 (SIC) in Report 10
            //  x_hat = SIC_IP(A, y, n, bound) applies the SIC
            //  initial point method
            //
            //  Inputs:
            //      A - m-by-n real matrix
            //      y - m-dimensional real vector
            //      n - integer scalar
            //      bound - integer scalar for the constraint
            //
            //  Outputs:
            //      x_hat - n-dimensional integer vector for the initial point
            //      rho - real scalar for the norm of the residual vector corresponding to
            //      x_hat A_hat - m-by-n real matrix, permuted A for sub-is_bocb methods
            //      Piv - n-by-n real matrix where A_hat*Piv'=A
         */
        returnType<scalar, index> cgsic() {
            b_matrix C(n, n), P(n, n), A_(this->A);
            b_vector b(n);
            scalar xi, rho, rho_min, rho_t;

            // 'cgsic:20' [~, n] = size(A_);
            // 'cgsic:21' x_hat = zeros(n,1);
            x_hat.clear();
            // 'cgsic:22' P = eye(n);
            P.assign(I);
            scalar time = omp_get_wtime();

            // 'cgsic:23' b= A_'*y;
            b_matrix A_t = trans(A_);
            prod(A_t, y, b);
            // 'cgsic:24' C = A_'*A_;
            prod(A_t, A_, C);
            // 'cgsic:25' k = 0;
            index k = -1;
            // 'cgsic:26' rho = norm(y)^2;
            rho = norm_2(y);
            rho *= rho;
            // 'cgsic:27' for i = n:-1:1
            for (index i = n - 1; i >= 0; i--) {
                // 'cgsic:28' rho_min = inf;
                rho_min = INFINITY;
                // 'cgsic:29' for j = 1:i
                for (index j = 0; j <= i; j++) {
                    // 'cgsic:30' if i ~= n
                    if (i != n - 1) {
                        // 'cgsic:31' b(j) = b(j) - C(j,i+1)*x_hat(i+1);
                        b[j] = b[j] - C(j, i + 1) * x_hat[i + 1];
                    }
                    // 'cgsic:33' xi = round(b(j)/C(j,j));
                    // 'cgsic:34' xi = min(upper, max(xi, lower));
                    xi = std::fmin(upper, std::fmax(std::round(b[j] / C[j + n * j]), lower));
                    // 'cgsic:35' rho_t = rho - 2 * b(j) * xi + C(j,j) * xi^2;
                    rho_t = (rho - 2.0 * b[j] * xi) + C[j + n * j] * (xi * xi);
                    // 'cgsic:36' if rho_t < rho_min
                    if (rho_t < rho_min) {
                        // 'cgsic:37' k = j;
                        k = j;
                        // 'cgsic:38' x_hat(i)=xi;
                        x_hat[i] = xi;
                        // 'cgsic:39' rho_min = rho_t;
                        rho_min = rho_t;
                    }
                }
                // 'cgsic:42' rho = rho_min;
                rho = rho_min;
                // 'cgsic:43' A_(:,[k, i]) = A_(:, [i,k]);
                // 'cgsic:44' P(:,[k, i]) = P(:, [i,k]);
                // 'cgsic:46' C(:,[k, i]) = C(:, [i,k]);
                for (int i1 = 0; i1 < m; i1++) {
                    std::swap(A_(i1, i), A_(i1, k));
                }
                for (int i1 = 0; i1 < n; i1++) {
                    std::swap(P[i1 + i * n], P[i1 + k * n]);
                    std::swap(C[i1 + i * n], C[i1 + k * n]);
                }
                // 'cgsic:45' b([k,i]) = b([i,k]);
                std::swap(b[i], b[k]);
                // 'cgsic:47' C([k, i], :) = C([i,k], :);
                for (int i1 = 0; i1 < n; i1++) {
                    std::swap(C(k, i1), C(i, i1));
                }
            }
            time = omp_get_wtime() - time;
            // 'cgsic:49' A_hat = A_;
            // 'cgsic:50' x_hat = P * x_hat;
            b.assign(x_hat);
            x_hat.clear();
            prod(P, b, x_hat);
            return {{}, time, 0};
        }

        /**
         * Gradient projection method to find a real solutiont to the
           box-constrained real LS problem min_{l<=x<=u}||y-Bx||_2
           The input x is an initial point, we may take x=(l+u)/2.
           max_iter is the maximum number of iterations, say 50.
         * @return
         */
        returnType<scalar, index> gp() {
            b_vector ex;

            sd_vector t_bar, t_seq;
            si_vector r1, r2, r3, r4, r5;
            sb_vector b_x_0, b_x_1(n, 0);

            index i, j, k1, k2, k3;
            scalar v_norm = INFINITY;
            b_vector x_cur(n, 0);

            scalar time = omp_get_wtime();

            b_vector c;
            b_matrix A_T = trans(A);
            prod(A_T, y, c);

            // 'ubils:225' for iter = 1:search_iter
            for (index iter = 0; iter < search_iter; iter++) {
                // 'ubils:227' g = A'*(A*x_cur-y);
                b_vector r0(y.size());
                for (i = 0; i < y.size(); i++) {
                    r0[i] = -y[i];
                }
                b_vector Ax, g;
                prod(A, x_cur, Ax);
                for (index i1 = 0; i1 < Ax.size(); i1++) {
                    r0[i1] += Ax[i1];
                }
                prod(A_T, r0, g);

                //  Check KKT conditions
                // 'ubils:230' if (x_cur==cils.l) == 0
                b_x_0.resize(n, 0);
                for (i = 0; i < n; i++) {
                    b_x_0[i] = !(x_cur[i] == lower);
                }
                if (helper::if_all_x_true<index>(b_x_0)) {
                    // 'ubils:231' k1 = 1;
                    k1 = 1;
                } else {
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (x_cur[i] == lower) {
                            k3++;
                        }
                    }
                    r1.resize(k3, 0);
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (x_cur[i] == lower) {
                            r1[k3] = i + 1;
                            k3++;
                        }
                    }
                    b_x_0.resize(r1.size(), 0);
                    k3 = r1.size();
                    for (i = 0; i < k3; i++) {
                        b_x_0[i] = (g[r1[i] - 1] > -1.0E-5);
                    }
                    if (helper::if_all_x_true<index>(b_x_0)) {
                        // 'ubils:232' elseif (g(x_cur==l) > -1.e-5) == 1
                        // 'ubils:233' k1 = 1;
                        k1 = 1;
                    } else {
                        // 'ubils:234' else
                        // 'ubils:234' k1 = 0;
                        k1 = 0;
                    }
                }
                // 'ubils:236' if (x_cur==u) == 0
                b_x_0.resize(n, 0);
                for (i = 0; i < n; i++) {
                    b_x_0[i] = !(x_cur[i] == upper);
                }
                if (helper::if_all_x_true<index>(b_x_0)) {
                    // 'ubils:237' k2 = 1;
                    k2 = 1;
                } else {
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (x_cur[i] == upper) {
                            k3++;
                        }
                    }
                    r2.resize(k3, 0);
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (x_cur[i] == upper) {
                            r2[k3] = i + 1;
                            k3++;
                        }
                    }
                    b_x_0.resize(r2.size(), 0);
                    k3 = r2.size();
                    for (i = 0; i < k3; i++) {
                        b_x_0[i] = (g[r2[i] - 1] < 1.0E-5);
                    }
                    if (helper::if_all_x_true<index>(b_x_0)) {
                        // 'ubils:238' elseif (g(x_cur==u) < 1.e-5) == 1
                        // 'ubils:239' k2 = 1;
                        k2 = 1;
                    } else {
                        // 'ubils:240' else
                        // 'ubils:240' k2 = 0;
                        k2 = 0;
                    }
                }
                // 'ubils:242' if (l<x_cur & x_cur<u) == 0
                b_x_0.resize(n, 0);
                for (i = 0; i < n; i++) {
                    b_x_0[i] = ((!(lower < x_cur[i])) || (!(x_cur[i] < upper)));
                }
                if (helper::if_all_x_true<index>(b_x_0)) {
                    // 'ubils:243' k3 = 1;
                    k3 = 1;
                } else {
                    b_x_0.resize(n, 0);
                    for (i = 0; i < n; i++) {
                        b_x_0[i] = (lower < x_cur[i]);
                        b_x_1[i] = (x_cur[i] < upper);
                    }
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (b_x_0[i] && b_x_1[i]) {
                            k3++;
                        }
                    }
                    r3.resize(k3, 0);
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (b_x_0[i] && b_x_1[i]) {
                            r3[k3] = i + 1;
                            k3++;
                        }
                    }
                    b_x_0.resize(r3.size(), 0);
                    k3 = r3.size();
                    for (i = 0; i < k3; i++) {
                        b_x_0[i] = (g[r3[i] - 1] < 1.0E-5);
                    }
                    if (helper::if_all_x_true<index>(b_x_0)) {
                        // 'ubils:244' elseif (g(l<x_cur & x_cur<u) < 1.e-5) == 1
                        // 'ubils:245' k3 = 1;
                        k3 = 1;
                    } else {
                        // 'ubils:246' else
                        // 'ubils:246' k3 = 0;
                        k3 = 0;
                    }
                }
                x_hat.assign(x_cur);
                // 'ubils:248' if (k1 & k2 & k3)
                if ((k1 != 0) && (k2 != 0) && (k3 != 0)) {
                    break;
                } else {
                    scalar x_tmp;
                    //  Find the Cauchy poindex
                    // 'ubils:253' t_bar = 1.e5*ones(n,1);
                    t_bar.resize(n, 1e5);
                    // 'ubils:254' t_bar(g<0) = (x_cur(g<0)-u(g<0))./g(g<0);
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (g[i] < 0.0) {
                            k3++;
                        }
                    }
                    r4.resize(k3, 0);
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (g[i] < 0.0) {
                            r4[k3] = i + 1;
                            k3++;
                        }
                    }
                    r0.resize(r4.size());
                    k3 = r4.size();
                    for (i = 0; i < k3; i++) {
                        r0[i] = (x_cur[r4[i] - 1] - upper) / g[r4[i] - 1];
                    }
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (g[i] < 0.0) {
                            t_bar[i] = r0[k3];
                            k3++;
                        }
                    }
                    // 'ubils:255' t_bar(g>0) = (x_cur(g>0)-l(g>0))./g(g>0);
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (g[i] > 0.0) {
                            k3++;
                        }
                    }
                    r5.resize(k3, 0);
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (g[i] > 0.0) {
                            r5[k3] = i + 1;
                            k3++;
                        }
                    }
                    r0.resize(r5.size());
                    k3 = r5.size();
                    for (i = 0; i < k3; i++) {
                        r0[i] = (x_cur[r5[i] - 1] - lower) / g[r5[i] - 1];
                    }
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (g[i] > 0.0) {
                            t_bar[i] = r0[k3];
                            k3++;
                        }
                    }
                    //  Generate the ordered and non-repeated sequence of t_bar
                    // 'ubils:258' t_seq = unique([0;t_bar]);
                    r0.resize(n + 1);
                    r0[0] = 0;
                    for (i = 0; i < n; i++) {
                        r0[i + 1] = t_bar[i];
                    }
                    helper::unique_vector<scalar, index>(r0, t_seq);
                    //  Add 0 to make the implementation easier
                    // 'ubils:259' t = 0;
                    x_tmp = 0.0;
                    //  Search
                    // 'ubils:261' for j = 2:length(t_seq)
                    for (k2 = 0; k2 < t_seq.size() - 1; k2++) {
                        // 'ubils:262' tj_1 = t_seq(j-1);
                        scalar tj_1 = t_seq[k2];
                        // 'ubils:263' tj = t_seq(j);
                        scalar tj = t_seq[k2 + 1];
                        //  Compute x_cur(t_{j-1})
                        // 'ubils:265' xt_j_1 = x_cur - min(tj_1,t_bar).*g;
                        //  Compute teh search direction p_{j-1}
                        // 'ubils:267' pj_1 = zeros(n,1);
                        b_vector pj_1(n, 0);
                        pj_1.clear();
                        // 'ubils:268' pj_1(tj_1<t_bar) = -g(tj_1<t_bar);
                        for (i = 0; i < t_bar.size(); i++) {
                            if (tj_1 < t_bar[i]) {
                                pj_1[i] = -g[i];
                            }
                        }
                        //  Compute coefficients
                        // 'ubils:270' q = A*pj_1;
                        b_vector q;
                        prod(A, pj_1, q);
//                        q.clear();
//                        for (j = 0; j < n; j++) {
//                            for (i = 0; i < m; i++) {
//                                q[i] += A[j * m + i] * pj_1[j];
//                            }
//                        }
                        // 'ubils:271' fj_1d = (A*xt_j_1)'*q - c'*pj_1;
                        ex.resize(t_bar.size());
                        for (j = 0; j < t_bar.size(); j++) {
                            ex[j] = std::fmin(tj_1, t_bar[j]);
                        }
                        ex.resize(n);
                        for (i = 0; i < n; i++) {
                            ex[i] = x_cur[i] - ex[i] * g[i];
                        }
                        r0.resize(m);
                        r0.clear();
                        prod(A, ex, r0);
//                        for (j = 0; j < n; j++) {
//                            for (i = 0; i < m; i++) {
//                                r0[i] += A[j * m + i] * ex[j];
//                            }
//                        }
                        scalar delta_t = 0.0;
                        for (i = 0; i < m; i++) {
                            delta_t += r0[i] * q[i];
                        }
                        scalar fj_1d = 0.0;
                        for (i = 0; i < n; i++) {
                            fj_1d += c[i] * pj_1[i];
                        }
                        fj_1d = delta_t - fj_1d;
                        // 'ubils:272' fj_1dd = q'*q;
                        // 'ubils:273' t = tj;
                        x_tmp = tj;
                        //  Find a local minimizer
                        // 'ubils:275' delta_t = -fj_1d/fj_1dd;
                        delta_t = 0.0;
                        for (i = 0; i < m; i++) {
                            delta_t += q[i] * q[i];
                        }
                        delta_t = -fj_1d / delta_t;
                        // 'ubils:276' if fj_1d >= 0
                        if (fj_1d >= 0.0) {
                            // 'ubils:277' t = tj_1;
                            x_tmp = tj_1;
                            break;
                        } else if (delta_t < x_tmp - tj_1) {
                            // 'ubils:279' elseif delta_t < (tj-tj_1)
                            // 'ubils:280' t = tj_1+delta_t;
                            x_tmp = tj_1 + delta_t;
                            break;
                        }
                    }
                    // 'ubils:285' x_cur = x_cur - min(t,t_bar).*g;
                    ex.resize(t_bar.size());
                    k3 = t_bar.size();
                    for (j = 0; j < k3; j++) {
                        ex[j] = std::fmin(x_tmp, t_bar[j]);
                    }
                    for (i = 0; i < n; i++) {
                        x_cur[i] = x_cur[i] - ex[i] * g[i];
                    }
                }
            }

            time = omp_get_wtime() - time;
            return {{}, time, v_norm};
        }

        /**
         * // Corresponds to Algorithm 5 (Partition Strategy) in Report 10
            //  [A_t, P_tmp, x_tilde, Q_t, R_t, t] = partition_H(A, x, m, n)
            //  permutes and partitions A_t so that the submatrices H_i are
            //  full-column rank
            //
            //  Inputs:
            //      A - m-by-n real matrix
            //      x - n-dimensional integer vector
            //      m - integer scalar
            //      n - integer scalar
            //
            //  Outputs:
            //      P_tmp - n-by-n real matrix, permutation such that A_t*P_tmp=A
            //      x_tilde - n-dimensional integer vector (x permuted to correspond to
            //      A_t) Q_t - m-by-n real matrix (Q factors) R_t - m-by-n
            //      real matrix (R factors) t - 2-by-q integer matrix (indicates
            //      submatrices of A_t)
         * @param A_t
         * @param P_t
         * @param d
         * @return
         */
        void part2(index s, b_matrix &A_bar, b_matrix &P, b_matrix &indicator) {
            b_matrix A_hat;
            b_matrix P_bar;
            b_matrix R;
            b_matrix acc;
            b_vector b_d;
            b_vector bb, d;
            si_vector x;
            double f;
            int N;
            int i;
            unsigned int k;
            int loop_ub;
            int t;
            int vlen;
            // Corresponds to Algorithm 6.4 (Block Partition) in Thesis
            //  [A_bar, P, d] = part2(A, s)
            //  permutes and partitions A so that A_bar submatrices A_bar_i are
            //  full-column rank
            //
            //  Inputs:
            //      A - K-by-N real matrix
            //      s - Number of columns of each block
            //
            //  Outputs:
            //      A_bar - K-by-N real matrix,
            //      P - N-by-N permutation such A*P = A_bar
            //      d - 1-by-k integer matrix (indicates submatrices of A_hat)
            N = n;
            A_bar.resize(m, n);
            loop_ub = m * n;
            for (i = 0; i < loop_ub; i++) {
                A_bar[i] = 0.0;
            }
            A_hat.assign(A);

            t = n;
            P.resize(n, n);
            loop_ub = n * n;
            for (i = 0; i < loop_ub; i++) {
                P[i] = 0.0;
            }
            if (n > 0) {
                for (loop_ub = 0; loop_ub < t; loop_ub++) {
                    P[loop_ub + P.size1() * loop_ub] = 1.0;
                }
            }
            d.resize(n);
            loop_ub = n;
            for (i = 0; i < loop_ub; i++) {
                d[i] = s;
            }
            k = 1U;
            f = 1.0;
            CILS_Reduction<scalar, index> reduction;

            while (f <= N) {
                double r;
                int i1;
                bool empty_non_axis_sizes;
                acc.resize(A_hat.size1(), A_hat.size2());
                loop_ub = A_hat.size1() * A_hat.size2();
                for (i = 0; i < loop_ub; i++) {
                    acc[i] = A_hat[i];
                }

                reduction.reset(acc);
                reduction.mgs_max();//qrmgs_max(acc, R, P_bar);
                R.assign(reduction.R);
                P_bar.assign(reduction.P);
//                f = N + 1;
                if (R.size2() > 1) {
                    t = R.size1();
                    vlen = R.size2();
                    if (t <= vlen) {
                        vlen = t;
                    }
                    b_d.resize(vlen);
                    i = vlen - 1;
                    for (loop_ub = 0; loop_ub <= i; loop_ub++) {
                        b_d[loop_ub] = R[loop_ub + R.size1() * loop_ub];
                    }
                    t = b_d.size();
                    bb.resize(b_d.size());
                    for (loop_ub = 0; loop_ub < t; loop_ub++) {
                        bb[loop_ub] = std::abs(b_d[loop_ub]);
                    }
                    x.resize(bb.size());
                    loop_ub = bb.size();
                    for (i = 0; i < loop_ub; i++) {
                        x[i] = (bb[i] > 1.0E-6);
                    }
                    vlen = x.size();
                    if (x.size() == 0) {
                        t = 0;
                    } else {
                        t = x[0];
                        for (loop_ub = 2; loop_ub <= vlen; loop_ub++) {
                            t += x[loop_ub - 1];
                        }
                    }
                    r = t;
                }
                else {
                    r = (std::abs(R[0]) > 1.0E-6);
                }
                acc.resize(A_hat.size1(), A_hat.size2());
                loop_ub = A_hat.size1() * A_hat.size2() - 1;
                for (i = 0; i <= loop_ub; i++) {
                    acc[i] = A_hat[i];
                }

                prod(acc, P_bar, A_hat);
                // TA_hate case wA_hatere A_hat_p is rank deficient
                if (r < s) {
                    double c_d;
                    // Permute tA_hate columns of A_hat and tA_hate entries of s_bar_output
                    if (r < 1.0) {
                        loop_ub = 0;
                    } else {
                        loop_ub = static_cast<int>(r);
                    }
                    c_d = f + r;
                    if (f > c_d - 1.0) {
                        i = 1;
                    } else {
                        i = static_cast<int>(f);
                    }
                    vlen = A_hat.size1();
                    for (i1 = 0; i1 < loop_ub; i1++) {
                        for (int i2{0}; i2 < vlen; i2++) {
                            A_bar[i2 + A_bar.size1() * ((i + i1) - 1)] =
                                    A_hat[i2 + A_hat.size1() * i1];
                        }
                    }
                    if (r + 1.0 > P_bar.size2()) {
                        i = 0;
                        i1 = 0;
                    } else {
                        i = static_cast<int>(r);
                        i1 = P_bar.size2();
                    }
                    vlen = A_hat.size1() - 1;
                    loop_ub = A_hat.size1();
                    t = i1 - i;
                    for (i1 = 0; i1 < t; i1++) {
                        for (int i2{0}; i2 < loop_ub; i2++) {
                            A_hat[i2 + (vlen + 1) * i1] = A_hat[i2 + A_hat.size1() * (i + i1)];
                        }
                    }
                    A_hat.resize(vlen + 1, t, true);
                    d[static_cast<int>(k) - 1] = r;
                    f = c_d;
                    k++;
                }
                else {
                    double c_d;
                    double j;
                    int i2;
                    j = 0.0;
                    while (r >= s) {
                        double d1;
                        c_d = j * s + 1.0;
                        d1 = (j + 1.0) * s;
                        if (c_d > d1) {
                            i = 0;
                            i1 = 0;
                        } else {
                            i = static_cast<int>(c_d) - 1;
                            i1 = static_cast<int>(d1);
                        }
                        c_d = f + s;
                        if (f > c_d - 1.0) {
                            i2 = 1;
                        } else {
                            i2 = static_cast<int>(f);
                        }
                        loop_ub = A_hat.size1();
                        vlen = i1 - i;
                        for (i1 = 0; i1 < vlen; i1++) {
                            for (t = 0; t < loop_ub; t++) {
                                A_bar[t + A_bar.size1() * ((i2 + i1) - 1)] =
                                        A_hat[t + A_hat.size1() * (i + i1)];
                            }
                        }
                        d[static_cast<int>(k) - 1] = s;
                        f = c_d;
                        r -= s;
                        k++;
                        j++;
                    }
                    c_d = j * s + 1.0;
                    if (c_d > P_bar.size2()) {
                        i = 0;
                        i1 = 0;
                    } else {
                        i = static_cast<int>(c_d) - 1;
                        i1 = P_bar.size2();
                    }
                    vlen = A_hat.size1() - 1;
                    loop_ub = A_hat.size1();
                    t = i1 - i;
                    for (i1 = 0; i1 < t; i1++) {
                        for (i2 = 0; i2 < loop_ub; i2++) {
                            A_hat[i2 + (vlen + 1) * i1] = A_hat[i2 + A_hat.size1() * (i + i1)];
                        }
                    }
                    A_hat.resize(vlen + 1, t, true);
                }
                t = N - P_bar.size2();
                if (t < 0) {
                    t = 0;
                }
                acc.resize(t, t);
                loop_ub = t * t;
                for (i = 0; i < loop_ub; i++) {
                    acc[i] = 0.0;
                }
                if (t > 0) {
                    for (loop_ub = 0; loop_ub < t; loop_ub++) {
                        acc[loop_ub + acc.size1() * loop_ub] = 1.0;
                    }
                }
                if ((acc.size1() != 0) && (acc.size2() != 0)) {
                    loop_ub = acc.size1();
                } else if ((N - P_bar.size2() != 0) && (P_bar.size2() != 0)) {
                    loop_ub = N - P_bar.size2();
                } else {
                    loop_ub = acc.size1();
                    if (N - P_bar.size2() > acc.size1()) {
                        loop_ub = N - P_bar.size2();
                    }
                }
                empty_non_axis_sizes = (loop_ub == 0);
                if (empty_non_axis_sizes || ((acc.size1() != 0) && (acc.size2() != 0))) {
                    vlen = acc.size2();
                } else {
                    vlen = 0;
                }
                if (empty_non_axis_sizes ||
                    ((N - P_bar.size2() != 0) && (P_bar.size2() != 0))) {
                    t = P_bar.size2();
                } else {
                    t = 0;
                }
                R.resize(loop_ub, vlen + t);
                for (i = 0; i < vlen; i++) {
                    for (i1 = 0; i1 < loop_ub; i1++) {
                        R[i1 + R.size1() * i] = acc[i1 + loop_ub * i];
                    }
                }
                for (i = 0; i < t; i++) {
                    for (i1 = 0; i1 < loop_ub; i1++) {
                        R[i1 + R.size1() * (i + vlen)] = 0.0;
                    }
                }
                if ((P_bar.size2() != 0) && (N - P_bar.size2() != 0)) {
                    loop_ub = P_bar.size2();
                } else if ((P_bar.size1() != 0) && (P_bar.size2() != 0)) {
                    loop_ub = P_bar.size1();
                } else {
                    loop_ub = P_bar.size2();
                    if (P_bar.size1() > P_bar.size2()) {
                        loop_ub = P_bar.size1();
                    }
                }
                empty_non_axis_sizes = (loop_ub == 0);
                if (empty_non_axis_sizes ||
                    ((P_bar.size2() != 0) && (N - P_bar.size2() != 0))) {
                    vlen = N - P_bar.size2();
                } else {
                    vlen = 0;
                }
                if (empty_non_axis_sizes ||
                    ((P_bar.size1() != 0) && (P_bar.size2() != 0))) {
                    t = P_bar.size2();
                } else {
                    t = 0;
                }
                acc.resize(loop_ub, vlen + t);
                for (i = 0; i < vlen; i++) {
                    for (i1 = 0; i1 < loop_ub; i1++) {
                        acc[i1 + acc.size1() * i] = 0.0;
                    }
                }
                for (i = 0; i < t; i++) {
                    for (i1 = 0; i1 < loop_ub; i1++) {
                        acc[i1 + acc.size1() * (i + vlen)] = P_bar[i1 + loop_ub * i];
                    }
                }
                if ((R.size1() != 0) && (R.size2() != 0)) {
                    loop_ub = R.size2();
                } else if ((acc.size1() != 0) && (acc.size2() != 0)) {
                    loop_ub = acc.size2();
                } else {
                    loop_ub = R.size2();
                    if (acc.size2() > R.size2()) {
                        loop_ub = acc.size2();
                    }
                }
                empty_non_axis_sizes = (loop_ub == 0);
                if (empty_non_axis_sizes || ((R.size1() != 0) && (R.size2() != 0))) {
                    vlen = R.size1();
                } else {
                    vlen = 0;
                }
                if (empty_non_axis_sizes || ((acc.size1() != 0) && (acc.size2() != 0))) {
                    t = acc.size1();
                } else {
                    t = 0;
                }
                P_bar.resize(vlen + t, loop_ub);
                for (i = 0; i < loop_ub; i++) {
                    for (i1 = 0; i1 < vlen; i1++) {
                        P_bar[i1 + P_bar.size1() * i] = R[i1 + vlen * i];
                    }
                }
                for (i = 0; i < loop_ub; i++) {
                    for (i1 = 0; i1 < t; i1++) {
                        P_bar[(i1 + vlen) + P_bar.size1() * i] = acc[i1 + t * i];
                    }
                }
                acc.resize(P.size1(), P.size2());
                loop_ub = P.size1() * P.size2() - 1;
                for (i = 0; i <= loop_ub; i++) {
                    acc[i] = P[i];
                }
                prod(acc, P_bar, P);
            }
            if (static_cast<int>(k - 1U) < 1) {
                N = 0;
            } else {
                N = static_cast<int>(k - 1U);
            }
//            d.resize(N, true);
            indicator.resize(2, static_cast<int>(k - 1U));
            loop_ub = static_cast<int>(k - 1U) << 1;
            for (i = 0; i < loop_ub; i++) {
                indicator[i] = 0.0;
            }
            f = 1.0;
            i = static_cast<int>(k);
            for (vlen = 0; vlen <= i - 2; vlen++) {
                indicator[2 * vlen] = f;
                f += d[vlen];
                indicator[2 * vlen + 1] = f - 1.0;
            }
            b_matrix A_temp;
            prod(A, P, A_temp);
            scalar sum = 0;
            for (index i = 0; i < m * n; i++){
                sum += A_bar[i] - A_temp[i];
            }
            cout << sum;
            cout.flush();
        }

        returnType<scalar, index> bsic_bcp(index is_bocb, index c) {

            b_vector x_tmp, htx(m, 0), y_hat(m, 0), x_t, x_bar(n, 0);
            b_matrix A_T, A_P, P, d(2, n); //A could be permuted based on the init point
            scalar v_norm = helper::find_residual<scalar, index>(A, x_hat, y);

            scalar time = omp_get_wtime();
            if (v_norm <= tolerance) {
                time = omp_get_wtime() - time;
                return {{}, time, v_norm};
            }

            CILS_Reduction<scalar, index> reduction;
            CILS_OLM<scalar, index> olm;

            index i, j, p, iter, cur_end , cur_1st;

            // Partition:
            part2(c, A_P, P, d);
            b_matrix P_trans = trans(P);
            prod(P_trans, x_hat, x_tmp);
            std::vector<int> pp;
            for (i = 0; i < d.size2(); i++){
                pp.push_back(i);
            }
            cout << A;
            cout << A_P;
            time = omp_get_wtime();
            for (index itr = 0; itr < 2 - 1; itr++) {
//                std::shuffle(pp.begin(), pp.end(), std::mt19937(std::random_device()()));
                prod(A_P, x_tmp, htx);
                for (i = 0; i < m; i++) {
                    y_hat[i] = y[i] - htx[i];
                }

                for (j = 0; j < d.size2(); j++) {
                    cur_1st = d(0, pp[j]) - 1;
                    cur_end = d(1, pp[j]) - 1;
                    index t = cur_end - cur_1st + 1;
                    A_T.resize(m, t);

                    x_t.resize(t);
                    for (index col = cur_1st; col <= cur_end; col++) {
                        for (index row = 0; row < m; row++) {
                            A_T(row, col - cur_1st) = A_P(row, col);
                        }
                        x_t[col - cur_1st] = x_tmp[col];
                    }
                    cout << A_T;
                    prod(A_T, x_t, htx);
                    for (i = 0; i < m; i++) {
                        y_hat[i] = y_hat[i] + htx[i];
                    }

                    b_vector z(t, 0);
                    reduction.reset(A_T, y_hat, upper);
                    if (is_bocb)
                        reduction.aip();
                    else
//                        reduction.aip();
                        reduction.aspl_p();
//                    cout << reduction.R;
                    olm.reset(reduction.R, reduction.y, upper, 5, true);

                    if (is_bocb) {
                        olm.bocb();
                        prod(reduction.P, olm.z_hat, z);
                    } else {
                        olm.bnp();
                        prod(reduction.Z, olm.z_hat, z);
                    }

                    for (index col = cur_1st; col <= cur_end; col++) {
                        x_tmp[col] = z[col - cur_1st];
                    }
                    prod(A_T, z, htx);
                    for (i = 0; i < m; i++) {
                        y_hat[i] = y_hat[i] - htx[i];
                    }
                }

                scalar rho = helper::find_residual<scalar, index>(A_P, x_tmp, y);

                if (rho < v_norm) {
//                    x_tmp.assign(x_bar);
                    if (rho <= tolerance) {
                        break;
                    }
                    v_norm = rho;
                }
            }
            time = omp_get_wtime() - time;
            prod(P, x_tmp, x_hat);
            return {{}, time, v_norm};
        }

        returnType<scalar, index> bsic(index is_bocb, index c) {

            b_vector x_tmp(x_hat), htx(m, 0), y_hat(m, 0), x_t;
            b_matrix A_T, A_P(A), P(I), d(2, n); //A could be permuted based on the init point
            scalar v_norm = helper::find_residual<scalar, index>(A, x_hat, y);

            scalar time = omp_get_wtime();
            if (v_norm <= tolerance) {
                time = omp_get_wtime() - time;
                return {{}, time, v_norm};
            }

            CILS_Reduction<scalar, index> reduction;
            CILS_OLM<scalar, index> olm;
//            CILS_SECH_Search<scalar,index> search;

            index i, j, p, iter = 0;


            // 'SCP_Block_Optimal_2:34' cur_end = n;
            index cur_end = n, cur_1st;
            // 'SCP_Block_Optimal_2:35' i = 1;
            index b_i = 1;
            // 'SCP_Block_Optimal_2:36' while cur_end > 0
            while (cur_end > 0) {
                // 'SCP_Block_Optimal_2:37' cur_1st = max(1, cur_end-m+1);
                cur_1st = std::max(1, (cur_end - c) + 1);
                // 'SCP_Block_Optimal_2:38' indicator(1,i) = cur_1st;
                d[2 * (b_i - 1)] = cur_1st;
                // 'SCP_Block_Optimal_2:39' indicator(2,i) = cur_end;
                d[2 * (b_i - 1) + 1] = cur_end;
                // 'SCP_Block_Optimal_2:40' cur_end = cur_1st - 1;
                cur_end = cur_1st - 1;
                // 'SCP_Block_Optimal_2:41' i = i + 1;
                b_i++;
            }
            d.resize(2, b_i - 1, true);
            int block_size[2] = {};
            if (m == 44) {
                block_size[0] = 11;
                block_size[1] = 12;
            } else {
                block_size[0] = block_size[1] = 5;
            }
            time = omp_get_wtime();
            for (index itr = 0; itr < search_iter - 1; itr++) {
                iter = itr;
                A_P.clear();
                x_tmp.clear();
                for (index col = 0; col < n; col++) {
                    p = permutation[iter][col] - 1;
                    for (index row = 0; row < m; row++) {
                        A_P(row, col) = A(row, p);
                    }
                    x_tmp[col] = x_hat[p];
                }

                //todo: [H_t, Piv_cum, indicator] = part(H_P);

                // 'SCP_Block_Optimal_2:104' per = true;
                // y_hat = y - H_t * x_t;
                prod(A_P, x_tmp, htx);
                for (i = 0; i < m; i++) {
                    y_hat[i] = y[i] - htx[i];
                }

                for (j = 0; j < d.size2(); j++) {
                    cur_1st = d(0, j) - 1;
                    cur_end = d(1, j) - 1;
                    index t = cur_end - cur_1st + 1;
                    A_T.resize(m, t);

                    x_t.resize(t);
                    for (index col = cur_1st; col <= cur_end; col++) {
                        for (index row = 0; row < m; row++) {
                            A_T(row, col - cur_1st) = A_P(row, col);
                        }
                        x_t[col - cur_1st] = x_tmp[col];
                    }

                    prod(A_T, x_t, htx);
                    for (i = 0; i < m; i++) {
                        y_hat[i] = y_hat[i] + htx[i];
                    }

                    b_vector z(t, 0);
                    reduction.reset(A_T, y_hat, upper);
                    if (is_bocb)
                        reduction.aip();
                    else
                        reduction.aspl_p();
                    olm.reset(reduction.R, reduction.y, upper, block_size[j], true);

                    if (is_bocb) {
                        olm.bocb();
                        prod(reduction.P, olm.z_hat, z);
                    } else {
                        olm.bnp();
                        prod(reduction.Z, olm.z_hat, z);
                    }

                    for (index col = cur_1st; col <= cur_end; col++) {
                        x_tmp[col] = z[col - cur_1st];
                    }
                    prod(A_T, z, htx);
                    for (i = 0; i < m; i++) {
                        y_hat[i] = y_hat[i] - htx[i];
                    }
                }

                scalar rho = helper::find_residual<scalar, index>(A_P, x_tmp, y);

                if (rho < v_norm) {
                    b_matrix I_P(n, n);
                    for (i = 0; i < n; i++) {
                        p = permutation[iter][i] - 1;
                        for (index i1 = 0; i1 < n; i1++) {
                            I_P[i1 + P.size1() * i] = I[i1 + I.size1() * p];
                        }
                    }
                    P.assign(I_P);
                    prod(P, x_tmp, x_hat);
                    if (rho <= tolerance) {
                        break;
                    }
                    v_norm = rho;
                }
            }
            time = omp_get_wtime() - time;
//            x_hat.clear();
//            x_hat.assign(x_cur);
            return {{}, time, v_norm};
        }


        returnType<scalar, index> pbsic(index is_bocb, index c, index n_t, index n_c) {

            b_vector x_tmp(x_hat);
            b_matrix P(I), d(2, n); //A could be permuted based on the init point
            scalar v_norm = helper::find_residual<scalar, index>(A, x_hat, y);

            scalar time = omp_get_wtime();
            if (v_norm <= tolerance) {
                time = omp_get_wtime() - time;
                return {{}, time, v_norm};
            }
            cils::returnType<scalar, index> reT;
            index i, j, p, iter = 0;

            // 'SCP_Block_Optimal_2:34' cur_end = n;
            index cur_end = n, cur_1st;
            // 'SCP_Block_Optimal_2:35' i = 1;
            index b_i = 1, t;
            // 'SCP_Block_Optimal_2:36' while cur_end > 0
            while (cur_end > 0) {
                // 'SCP_Block_Optimal_2:37' cur_1st = max(1, cur_end-m+1);
                cur_1st = std::max(1, (cur_end - c) + 1);
                // 'SCP_Block_Optimal_2:38' indicator(1,i) = cur_1st;
                d[2 * (b_i - 1)] = cur_1st;
                // 'SCP_Block_Optimal_2:39' indicator(2,i) = cur_end;
                d[2 * (b_i - 1) + 1] = cur_end;
                // 'SCP_Block_Optimal_2:40' cur_end = cur_1st - 1;
                cur_end = cur_1st - 1;
                // 'SCP_Block_Optimal_2:41' i = i + 1;
                b_i++;
            }
            d.resize(2, b_i - 1, true);
            omp_set_nested(1);
//            index nthreads = 2;
            scalar rho;
            b_matrix A_P(m, n), I_P(n, n), A_T;
            b_vector x_t, htx, y_hat(m, 0);
            int block_size[2] = {};
            if (m == 44) {
                block_size[0] = 11;
                block_size[1] = 12;
            } else {
                block_size[0] = block_size[1] = 8;
            }
            CILS_Reduction<scalar, index> reduction;
            CILS_OLM<scalar, index> olm;

            time = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_t) private(reduction, olm, t, A_T, x_t, htx, i, iter, j) firstprivate(y_hat, A_P, I_P, x_tmp, rho, p, cur_end, cur_1st)
            {
#pragma omp for nowait
                for (iter = 0; iter < search_iter - 1; iter++) {
                    A_P.clear();
                    x_tmp.clear();
                    for (index col = 0; col < n; col++) {
                        p = permutation[iter][col] - 1;
                        for (index row = 0; row < m; row++) {
                            A_P(row, col) = A(row, p);
                        }
                        x_tmp[col] = x_hat[p];
                    }

                    //todo: [H_t, Piv_cum, indicator] = part(H_P);

                    // 'SCP_Block_Optimal_2:104' per = true;
                    // y_hat = y - H_t * x_t;
                    prod(A_P, x_tmp, htx);
                    for (i = 0; i < m; i++) {
                        y_hat[i] = y[i] - htx[i];
                    }

                    for (j = 0; j < d.size2(); j++) {
                        cur_1st = d(0, j) - 1;
                        cur_end = d(1, j) - 1;
                        t = cur_end - cur_1st + 1;
                        A_T.resize(m, t);

                        x_t.resize(t);
                        for (index col = cur_1st; col <= cur_end; col++) {
                            for (index row = 0; row < m; row++) {
                                A_T(row, col - cur_1st) = A_P(row, col);
                            }
                            x_t[col - cur_1st] = x_tmp[col];
                        }

                        prod(A_T, x_t, htx);
                        for (i = 0; i < m; i++) {
                            y_hat[i] = y_hat[i] + htx[i];
                        }

                        b_vector z(t, 0);
                        reduction.reset(A_T, y_hat, upper);
                        reduction.paip(n_c);

                        olm.reset(reduction.R, reduction.y, upper, block_size[j], true);
                        if (is_bocb) {

                            olm.pbocb(n_c, 20, 0);

                        } else {
                            olm.pbnp2(n_c, 20, 0);
                        }
                        prod(reduction.P, olm.z_hat, z);
                        for (index col = cur_1st; col <= cur_end; col++) {
                            x_tmp[col] = z[col - cur_1st];
                        }
                        prod(A_T, z, htx);
                        for (i = 0; i < m; i++) {
                            y_hat[i] = y_hat[i] - htx[i];
                        }

                    }

                    rho = helper::find_residual<scalar, index>(A_P, x_tmp, y);

                    if (rho < v_norm) {
                        I_P.resize(n, n);
                        I_P.clear();
                        for (i = 0; i < n; i++) {
                            p = permutation[iter][i] - 1;
                            for (index i1 = 0; i1 < n; i1++) {
                                I_P[i1 + P.size1() * i] = I[i1 + I.size1() * p];
                            }
                        }
                        P.assign(I_P);
                        prod(P, x_tmp, x_hat);
//                        if (rho <= tolerance) {
//                            break;
//                        }
                        v_norm = rho;
                    }
                }
            }

            time = omp_get_wtime() - time;
            return {{}, time, v_norm};
        }
    };
}