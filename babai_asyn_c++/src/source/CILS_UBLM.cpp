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
            //      x_hat A_hat - m-by-n real matrix, permuted A for sub-optimal methods
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
                // 'ubils:248' if (k1 & k2 & k3)
                if ((k1 != 0) && (k2 != 0) && (k3 != 0)) {
                    k1 = k2 = k3 = 0;
                    for (i = 0; i < n; i++) {
                        x_hat[i] = x_cur[i];
                    }
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
//        returnType<scalar, index> partition(b_matrix &A_t, b_matrix &P_t, b_vector &d) {
//            b_matrix A_bar_p, P_tmp, P_hat, P_tilde, b_A_bar_p, b_A_tilde, b_p, R;
//            b_vector b_d, y_tmp;
//            si_vector b_x;
//            int q, i, input_sizes_idx_1, k, loop_ub, sizes_idx_1, t, vlen;
//
//            //  A_ = A;
//            //  x_ = x;
//            // 'partition:23' [m, n] = size(A);
//            // 'partition:24' A_t = A;
//            A_t.assign(A);
//            // 'partition:25' q = n;
//            q = A.size2();
//            // 'partition:26' P_tmp = eye(n);
//            t = A.size2();
//            P_tmp.assign(I);
//
//            // 'partition:29' d = zeros(n, 1);
//            d.resize(A.size2(), false);
//            // 'partition:30' k = 0;
//            k = 0;
//            // 'partition:32' while q >= 1
//            while (q >= 1) {
//                double c_d;
//                int b_loop_ub, i1, i2, i3, r, p;
//                // 'partition:33' k = k + 1;
//                k++;
//                // 'partition:34' p = max(1, q-m+1);
//                p = std::max(1, q - A.size1() + 1);
//                // 'partition:35' A_bar = A_t(:, p:q);
//                if (p > q) {
//                    i = 0;
//                    i1 = 0;
//                } else {
//                    i = p - 1;
//                    i1 = q;
//                }
//                // 'partition:36' P_tilde = eye(n);
//                t = A.size2();
//                P_tilde.assign(I);
//
//
//                // Find the rank of A_bar
//                // 'partition:39' [Q,R,P_hat]=qr(A_bar);
//                loop_ub = A_t.size1();
//                b_loop_ub = i1 - i;
//                b_A_tilde.resize(A_t.size1(), b_loop_ub);
//                for (i1 = 0; i1 < b_loop_ub; i1++) {
//                    for (i2 = 0; i2 < loop_ub; i2++) {
//                        b_A_tilde[i2 + b_A_tilde.size1() * i1] =
//                                A_t[i2 + A_t.size1() * (i + i1)];
//                    }
//                }
//                helper::qrp(b_A_tilde,  R, b_p, true, true);
//                P_hat.resize(b_p.size1(), b_p.size2());
//                loop_ub = b_p.size1() * b_p.size2();
//                for (i1 = 0; i1 < loop_ub; i1++) {
//                    P_hat[i1] = b_p[i1];
//                }
//                // 'partition:40' if size(R,2)>1
//                if (R.size2() > 1) {
//                    // 'partition:41' r = sum(abs(diag(R)) > 10^(-6));
//                    t = R.size1();
//                    vlen = R.size2();
//                    if (t < vlen) {
//                        vlen = t;
//                    }
//                    b_d.resize(vlen);
//                    i1 = vlen - 1;
//                    for (sizes_idx_1 = 0; sizes_idx_1 <= i1; sizes_idx_1++) {
//                        b_d[sizes_idx_1] = R[sizes_idx_1 + R.size1() * sizes_idx_1];
//                    }
//                    t = b_d.size();
//                    y_tmp.resize(b_d.size());
//                    for (sizes_idx_1 = 0; sizes_idx_1 < t; sizes_idx_1++) {
//                        y_tmp[sizes_idx_1] = std::abs(b_d[sizes_idx_1]);
//                    }
//                    b_x.resize(y_tmp.size());
//                    loop_ub = y_tmp.size();
//                    for (i1 = 0; i1 < loop_ub; i1++) {
//                        b_x[i1] = (y_tmp[i1] > 1.0E-6);
//                    }
//                    vlen = b_x.size();
//                    if (b_x.size() == 0) {
//                        r = 0;
//                    } else {
//                        r = b_x[0];
//                        for (sizes_idx_1 = 2; sizes_idx_1 <= vlen; sizes_idx_1++) {
//                            r += b_x[sizes_idx_1 - 1];
//                        }
//                    }
//                } else {
//                    // 'partition:42' else
//                    // 'partition:43' r = sum(abs(R(1,1)) > 10^(-6));
//                    r = (std::abs(R[0]) > 1.0E-6);
//                }
//                // 'partition:45' A_bar_p = A_bar * P_hat;
//                loop_ub = A_t.size1();
//                b_A_tilde.resize(A_t.size1(), b_loop_ub);
//                for (i1 = 0; i1 < b_loop_ub; i1++) {
//                    for (i2 = 0; i2 < loop_ub; i2++) {
//                        b_A_tilde[i2 + b_A_tilde.size1() * i1] =
//                                A_t[i2 + A_t.size1() * (i + i1)];
//                    }
//                }
//                prod(b_A_tilde, b_p, A_bar_p);
//                // 'partition:46' d(k) = size(A_bar, 2);
//                d[k - 1] = b_loop_ub;
//                // The case where A_bar is rank deficient
//                // 'partition:49' if r < d(k)
//                c_d = d[k - 1];
//                if (r < c_d) {
//                    double b_t;
//                    bool empty_non_axis_sizes;
//                    // Permute the columns of A_t and the entries of x_tilde
//                    // 'partition:51' A_t(:, p:q) = [A_bar_p(:,r+1:d(k)), A_bar_p(:,
//                    // 1:r)];
//                    if (r + 1.0 > c_d) {
//                        i = -1;
//                        i1 = -1;
//                    } else {
//                        i = r - 1;
//                        i1 = static_cast<int>(c_d) - 1;
//                    }
//                    if (1 > r) {
//                        loop_ub = 0;
//                    } else {
//                        loop_ub = r;
//                    }
//                    if (p > q) {
//                        i2 = 0;
//                    } else {
//                        i2 = p - 1;
//                    }
//                    b_loop_ub = i1 - i;
//                    if ((A_bar_p.size1() != 0) && (b_loop_ub != 0)) {
//                        t = A_bar_p.size1();
//                    } else if ((A_bar_p.size1() != 0) && (loop_ub != 0)) {
//                        t = A_bar_p.size1();
//                    } else {
//                        if (A_bar_p.size1() > 0) {
//                            t = A_bar_p.size1();
//                        } else {
//                            t = 0;
//                        }
//                        if (A_bar_p.size1() > t) {
//                            t = A_bar_p.size1();
//                        }
//                    }
//                    empty_non_axis_sizes = (t == 0);
//                    if (empty_non_axis_sizes ||
//                        ((A_bar_p.size1() != 0) && (b_loop_ub != 0))) {
//                        input_sizes_idx_1 = i1 - i;
//                    } else {
//                        input_sizes_idx_1 = 0;
//                    }
//                    if (empty_non_axis_sizes || ((A_bar_p.size1() != 0) && (loop_ub != 0))) {
//                        sizes_idx_1 = loop_ub;
//                    } else {
//                        sizes_idx_1 = 0;
//                    }
//                    vlen = A_bar_p.size1();
//                    b_A_tilde.resize(A_bar_p.size1(), b_loop_ub);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        for (i3 = 0; i3 < vlen; i3++) {
//                            b_A_tilde[i3 + b_A_tilde.size1() * i1] =
//                                    A_bar_p[i3 + A_bar_p.size1() * ((i + i1) + 1)];
//                        }
//                    }
//                    b_loop_ub = A_bar_p.size1();
//                    b_A_bar_p.resize(A_bar_p.size1(), loop_ub);
//                    for (i = 0; i < loop_ub; i++) {
//                        for (i1 = 0; i1 < b_loop_ub; i1++) {
//                            b_A_bar_p[i1 + b_A_bar_p.size1() * i] =
//                                    A_bar_p[i1 + A_bar_p.size1() * i];
//                        }
//                    }
//                    for (i = 0; i < input_sizes_idx_1; i++) {
//                        for (i1 = 0; i1 < t; i1++) {
//                            A_t[i1 + A_t.size1() * (i2 + i)] = b_A_tilde[i1 + t * i];
//                        }
//                    }
//                    for (i = 0; i < sizes_idx_1; i++) {
//                        for (i1 = 0; i1 < t; i1++) {
//                            A_t[i1 + A_t.size1() * ((i2 + i) + input_sizes_idx_1)] =
//                                    b_A_bar_p[i1 + t * i];
//                        }
//                    }
//                    // Update the permutation matrix P_tilde
//                    // 'partition:54' I_d = eye(d(k));
//                    if (c_d < 0.0) {
//                        b_t = 0.0;
//                    } else {
//                        b_t = c_d;
//                    }
//                    t = static_cast<int>(b_t);
//                    P_hat.resize(t, t);
//                    loop_ub = static_cast<int>(b_t) * static_cast<int>(b_t);
//                    for (i = 0; i < loop_ub; i++) {
//                        P_hat[i] = 0.0;
//                    }
//                    if (static_cast<int>(b_t) > 0) {
//                        for (sizes_idx_1 = 0; sizes_idx_1 < t; sizes_idx_1++) {
//                            P_hat[sizes_idx_1 + P_hat.size1() * sizes_idx_1] = 1.0;
//                        }
//                    }
//                    // 'partition:55' P_d = [I_d(:, r+1:d(k)), I_d(:, 1:r)];
//                    if (r + 1.0 > c_d) {
//                        i = -1;
//                        i1 = -1;
//                    } else {
//                        i = r - 1;
//                        i1 = static_cast<int>(d[k - 1]) - 1;
//                    }
//                    if (1 > r) {
//                        loop_ub = 0;
//                    } else {
//                        loop_ub = r;
//                    }
//                    b_loop_ub = i1 - i;
//                    if ((P_hat.size1() != 0) && (b_loop_ub != 0)) {
//                        t = P_hat.size1();
//                    } else if ((P_hat.size1() != 0) && (loop_ub != 0)) {
//                        t = P_hat.size1();
//                    } else {
//                        if (P_hat.size1() > 0) {
//                            t = P_hat.size1();
//                        } else {
//                            t = 0;
//                        }
//                        if (P_hat.size1() > t) {
//                            t = P_hat.size1();
//                        }
//                    }
//                    empty_non_axis_sizes = (t == 0);
//                    if (empty_non_axis_sizes || ((P_hat.size1() != 0) && (b_loop_ub != 0))) {
//                        input_sizes_idx_1 = i1 - i;
//                    } else {
//                        input_sizes_idx_1 = 0;
//                    }
//                    if (empty_non_axis_sizes || ((P_hat.size1() != 0) && (loop_ub != 0))) {
//                        sizes_idx_1 = loop_ub;
//                    } else {
//                        sizes_idx_1 = 0;
//                    }
//                    // 'partition:56' P_hat = P_hat * P_d;
//                    vlen = P_hat.size1() - 1;
//                    b_A_tilde.resize(P_hat.size1(), b_loop_ub);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        for (i2 = 0; i2 <= vlen; i2++) {
//                            b_A_tilde[i2 + b_A_tilde.size1() * i1] =
//                                    P_hat[i2 + P_hat.size1() * ((i + i1) + 1)];
//                        }
//                    }
//                    vlen = P_hat.size1() - 1;
//                    b_A_bar_p.resize(P_hat.size1(), loop_ub);
//                    for (i = 0; i < loop_ub; i++) {
//                        for (i1 = 0; i1 <= vlen; i1++) {
//                            b_A_bar_p[i1 + b_A_bar_p.size1() * i] = P_hat[i1 + P_hat.size1() * i];
//                        }
//                    }
//                    A_bar_p.resize(t, input_sizes_idx_1 + sizes_idx_1);
//                    for (i = 0; i < input_sizes_idx_1; i++) {
//                        for (i1 = 0; i1 < t; i1++) {
//                            A_bar_p[i1 + A_bar_p.size1() * i] = b_A_tilde[i1 + t * i];
//                        }
//                    }
//                    for (i = 0; i < sizes_idx_1; i++) {
//                        for (i1 = 0; i1 < t; i1++) {
//                            A_bar_p[i1 + A_bar_p.size1() * (i + input_sizes_idx_1)] =
//                                    b_A_bar_p[i1 + t * i];
//                        }
//                    }
//                    prod(b_p, A_bar_p, P_hat);
//                    // 'partition:58' d(k) = r;
//                    d[k - 1] = r;
//                } else {
//                    // 'partition:59' else
//                    // Permute the columns of A_t and the entries of x_tilde
//                    // 'partition:61' A_t(:, p:q) = A_bar_p;
//                    if (p > q) {
//                        i = 1;
//                    } else {
//                        i = p;
//                    }
//                    loop_ub = A_bar_p.size2();
//                    for (i1 = 0; i1 < loop_ub; i1++) {
//                        b_loop_ub = A_bar_p.size1();
//                        for (i2 = 0; i2 < b_loop_ub; i2++) {
//                            A_t[i2 + A_t.size1() * ((i + i1) - 1)] =
//                                    A_bar_p[i2 + A_bar_p.size1() * i1];
//                        }
//                    }
//                }
//                // Update the permutation matrix P_tilde
//                // 'partition:64' P_tilde(p:q, p:q) = P_hat;
//                if (p > q) {
//                    i = 1;
//                    i1 = 1;
//                } else {
//                    i = p;
//                    i1 = p;
//                }
//                loop_ub = P_hat.size2();
//                for (i2 = 0; i2 < loop_ub; i2++) {
//                    b_loop_ub = P_hat.size1();
//                    for (i3 = 0; i3 < b_loop_ub; i3++) {
//                        P_tilde[((i + i3) + P_tilde.size1() * ((i1 + i2) - 1)) - 1] =
//                                P_hat[i3 + P_hat.size1() * i2];
//                    }
//                }
//                // 'partition:65' P_tmp = P_tmp * P_tilde;
//                b_A_tilde.resize(P_tmp.size1(), P_tmp.size2());
//                loop_ub = P_tmp.size1() * P_tmp.size2() - 1;
//                for (i = 0; i <= loop_ub; i++) {
//                    b_A_tilde[i] = P_tmp[i];
//                }
//                prod(b_A_tilde, P_tilde, P_tmp);
//                // 'partition:67' p = q - r + 1;
//                p = q - r + 1;
//                // 'partition:68' R_t(:, p:q) = R(:, 1:r);
//                if (1 > r) {
//                    loop_ub = 0;
//                } else {
//                    loop_ub = r;
//                }
//                if (p > q) {
//                    i = 1;
//                } else {
//                    i = p;
//                }
//                b_loop_ub = R.size1();
//                for (i1 = 0; i1 < loop_ub; i1++) {
//                    for (i2 = 0; i2 < b_loop_ub; i2++) {
//                        R_t[i2 + R_t.size1() * ((i + i1) - 1)] = R[i2 + R.size1() * i1];
//                    }
//                }
//                // 'partition:69' Q_t(:, p:q) = Q(:, 1:r);
//                if (1 > r) {
//                    loop_ub = 0;
//                } else {
//                    loop_ub = r;
//                }
//                if (p > q) {
//                    i = 1;
//                } else {
//                    i = p;
//                }
//                b_loop_ub = Q.size1();
//                for (i1 = 0; i1 < loop_ub; i1++) {
//                    for (i2 = 0; i2 < b_loop_ub; i2++) {
//                        Q_t[i2 + Q_t.size1() * ((i + i1) - 1)] = Q[i2 + Q.size1() * i1];
//                    }
//                }
//                // 'partition:71' q = q - r;
//                q -= r;
//            }
//            // 'partition:73' d = d(1:k);
//            if (1 > k) {
//                i = 0;
//            } else {
//                i = k;
//            }
//            d.resize(i);
//            // Remove the extra columns of the d
//
//            //  %Test
//            //  [H, P_H, x_H, Q_H, R_H, i_H] = partition_H(A_, x_, m, n);
//            //  size(A_t)
//            //  size(H)
//            //  AD=norm(A_t-H, 2)
//            //  PD=norm(P_tmp-P_H,2)
//            //  xD=norm(x_H-x_tilde,2)
//            //  QD=norm(Q_t-Q_H,2)
//            //  RD=norm(R_t-R_H,2)
//            //
//
//            //  dk
//            //  i_H
//        }

        returnType<scalar, index> bsic(index optimal) {

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
                cur_1st = std::max(1, (cur_end - m) + 1);
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
                    reduction.aip();
                    if (optimal) {
                        //todo
                    } else {
                        olm.reset(reduction.R, reduction.y, upper, true);
                        olm.bnp();
                        prod(reduction.P, olm.z_hat, z);
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

        scalar pasic(b_matrix &d, b_vector &x_tmp, index optimal, index iter, index nthreads) {
//            cout << iter << "," << omp_get_thread_num() << endl;
            b_matrix A_P(m, n), A_T;
            b_vector x_t, htx, y_hat(m, 0);
            x_tmp.clear();
            CILS_Reduction<scalar, index> reduction;
            CILS_OLM<scalar, index> olm;
            index p, i;
            for (index col = 0; col < n; col++) {
                p = permutation[iter][col] - 1;
                for (index row = 0; row < m; row++) {
                    A_P(row, col) = A(row, p);
                }
                x_tmp[col] = x_hat[p];
            }
            prod(A_P, x_tmp, htx);
            for (i = 0; i < m; i++) {
                y_hat[i] = y[i] - htx[i];
            }

#pragma omp parallel default(shared) num_threads(2) firstprivate(htx) private(A_T, x_t, p, i, reduction, olm)
            {
                for (int u = 0; u < 2; u++) {
#pragma omp for nowait
                    for (int j = 0; j < d.size2(); j++) {
//                        printf("Task %d: thread %d "
//                               "of the %d children of "
//                               "%d: "
//                               "handling iter %d\n",
//                               iter, omp_get_thread_num(),
//                               omp_get_team_size(2),
//                               omp_get_ancestor_thread_num(1),
//                               j);

                        index cur_1st = d(0, j) - 1;
                        index cur_end = d(1, j) - 1;
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
                        reduction.aip();
                        if (optimal) {
                            //todo
                        } else {
                            olm.reset(reduction.R, reduction.y, upper, true);
                            olm.bnp();
                            prod(reduction.P, olm.z_hat, z);
                        }
                        for (index col = cur_1st; col <= cur_end; col++) {
                            x_tmp[col] = z[col - cur_1st];
                        }
                        prod(A_T, z, htx);
                        for (i = 0; i < m; i++) {
                            y_hat[i] = y_hat[i] - htx[i];
                        }
                    }
                }
            }
            scalar rho = helper::find_residual<scalar, index>(A_P, x_tmp, y);
//            cout << x_tmp;
            return rho;
        }

        returnType<scalar, index> pbsic(index optimal, index n_t) {

            b_vector x_tmp(x_hat);
            b_matrix P(I), d(2, n); //A could be permuted based on the init point
            scalar v_norm = helper::find_residual<scalar, index>(A, x_hat, y);

            scalar time = omp_get_wtime();
            if (v_norm <= tolerance) {
                time = omp_get_wtime() - time;
                return {{}, time, v_norm};
            }

            index i, j, p, iter = 0;

            // 'SCP_Block_Optimal_2:34' cur_end = n;
            index cur_end = n, cur_1st;
            // 'SCP_Block_Optimal_2:35' i = 1;
            index b_i = 1;
            // 'SCP_Block_Optimal_2:36' while cur_end > 0
            while (cur_end > 0) {
                // 'SCP_Block_Optimal_2:37' cur_1st = max(1, cur_end-m+1);
                cur_1st = std::max(1, (cur_end - m) + 1);
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
            index nthreads = 2;
            index tasks = search_iter/n_t;
            scalar rho;
            b_matrix I_P(n, n);
            time = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_t) firstprivate(I_P, x_tmp, rho, p)
            {
#pragma omp single nowait
#pragma omp taskloop
                for (index itr = 0; itr < search_iter - 1; itr++) {
                    rho = pasic(d, x_tmp, optimal, itr, nthreads);
                    if (rho < v_norm) {
                        for (i = 0; i < n; i++) {
                            p = permutation[itr][i] - 1;
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