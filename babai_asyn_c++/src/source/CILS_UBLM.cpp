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
//            this->permutation = cils.permutation;
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
                for (index i1 = 0; i1 < Ax.size(); i1++){
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
                }
                else {
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

        returnType<scalar, index> partition(b_matrix &A_t, b_matrix &P_t, b_matrix &Q_t, b_matrix &R_t, b_vector &d) {
            b_matrix A_bar_p, P_tmp, P_hat, P_tilde, Q, R, b_A_bar_p, b_A_tilde, b_p;
            b_vector b_d, y_tmp;
            si_vector b_x;
            int q, i, input_sizes_idx_1, k, loop_ub, sizes_idx_1, t, vlen;
            // Corresponds to Algorithm 5 (Partition Strategy) in Report 10
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
            //  A_ = A;
            //  x_ = x;
            // 'partition:23' [m, n] = size(A);
            // 'partition:24' A_t = A;
            A_t = A;
            // 'partition:25' q = n;
            q = A.size2();
            // 'partition:26' P_tmp = eye(n);
            t = A.size2();
            P_tmp.resize(A.size2(), A.size2(), false);
            for (i = 0; i < t; i++) {
                P_tmp[i + P_tmp.size1() * i] = 1.0;
            }

            // 'partition:27' R_t = zeros(m,n);
            R_t.resize(A.size1(), A.size2(), false);

            // 'partition:28' Q_t = zeros(m,n);
            Q_t.resize(A.size1(), A.size2(), false);

            // 'partition:29' d = zeros(n, 1);
            d.resize(A.size2(), false);
            // 'partition:30' k = 0;
            k = 0;
            // 'partition:32' while q >= 1
            while (q >= 1) {
                double c_d;
                int b_loop_ub, i1, i2, i3, r, p;
                // 'partition:33' k = k + 1;
                k++;
                // 'partition:34' p = max(1, q-m+1);
                p = std::max(1, q - A.size1() + 1);
                // 'partition:35' A_bar = A_t(:, p:q);
                if (p > q) {
                    i = 0;
                    i1 = 0;
                } else {
                    i = p - 1;
                    i1 = q;
                }
                // 'partition:36' P_tilde = eye(n);
                t = A.size2();
                P_tilde.resize(A.size2(), A.size2(), false);
                for (sizes_idx_1 = 0; sizes_idx_1 < t; sizes_idx_1++) {
                    P_tilde[sizes_idx_1 + P_tilde.size1() * sizes_idx_1] = 1.0;
                }

                // Find the rank of A_bar
                // 'partition:39' [Q,R,P_hat]=qr(A_bar);
                loop_ub = A_t.size1();
                b_loop_ub = i1 - i;
                b_A_tilde.resize(A_t.size1(), b_loop_ub);
                for (i1 = 0; i1 < b_loop_ub; i1++) {
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        b_A_tilde[i2 + b_A_tilde.size1() * i1] =
                                A_t[i2 + A_t.size1() * (i + i1)];
                    }
                }
                helper::qrp(b_A_tilde, Q, R, b_p, true, true);
                P_hat.resize(b_p.size1(), b_p.size2());
                loop_ub = b_p.size1() * b_p.size2();
                for (i1 = 0; i1 < loop_ub; i1++) {
                    P_hat[i1] = b_p[i1];
                }
                // 'partition:40' if size(R,2)>1
                if (R.size2() > 1) {
                    // 'partition:41' r = sum(abs(diag(R)) > 10^(-6));
                    t = R.size1();
                    vlen = R.size2();
                    if (t < vlen) {
                        vlen = t;
                    }
                    b_d.resize(vlen);
                    i1 = vlen - 1;
                    for (sizes_idx_1 = 0; sizes_idx_1 <= i1; sizes_idx_1++) {
                        b_d[sizes_idx_1] = R[sizes_idx_1 + R.size1() * sizes_idx_1];
                    }
                    t = b_d.size();
                    y_tmp.resize(b_d.size());
                    for (sizes_idx_1 = 0; sizes_idx_1 < t; sizes_idx_1++) {
                        y_tmp[sizes_idx_1] = std::abs(b_d[sizes_idx_1]);
                    }
                    b_x.resize(y_tmp.size());
                    loop_ub = y_tmp.size();
                    for (i1 = 0; i1 < loop_ub; i1++) {
                        b_x[i1] = (y_tmp[i1] > 1.0E-6);
                    }
                    vlen = b_x.size();
                    if (b_x.size() == 0) {
                        r = 0;
                    } else {
                        r = b_x[0];
                        for (sizes_idx_1 = 2; sizes_idx_1 <= vlen; sizes_idx_1++) {
                            r += b_x[sizes_idx_1 - 1];
                        }
                    }
                } else {
                    // 'partition:42' else
                    // 'partition:43' r = sum(abs(R(1,1)) > 10^(-6));
                    r = (std::abs(R[0]) > 1.0E-6);
                }
                // 'partition:45' A_bar_p = A_bar * P_hat;
                loop_ub = A_t.size1();
                b_A_tilde.resize(A_t.size1(), b_loop_ub);
                for (i1 = 0; i1 < b_loop_ub; i1++) {
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        b_A_tilde[i2 + b_A_tilde.size1() * i1] =
                                A_t[i2 + A_t.size1() * (i + i1)];
                    }
                }
                prod(b_A_tilde, b_p, A_bar_p);
                // 'partition:46' d(k) = size(A_bar, 2);
                d[k - 1] = b_loop_ub;
                // The case where A_bar is rank deficient
                // 'partition:49' if r < d(k)
                c_d = d[k - 1];
                if (r < c_d) {
                    double b_t;
                    bool empty_non_axis_sizes;
                    // Permute the columns of A_t and the entries of x_tilde
                    // 'partition:51' A_t(:, p:q) = [A_bar_p(:,r+1:d(k)), A_bar_p(:,
                    // 1:r)];
                    if (r + 1.0 > c_d) {
                        i = -1;
                        i1 = -1;
                    } else {
                        i = r - 1;
                        i1 = static_cast<int>(c_d) - 1;
                    }
                    if (1 > r) {
                        loop_ub = 0;
                    } else {
                        loop_ub = r;
                    }
                    if (p > q) {
                        i2 = 0;
                    } else {
                        i2 = p - 1;
                    }
                    b_loop_ub = i1 - i;
                    if ((A_bar_p.size1() != 0) && (b_loop_ub != 0)) {
                        t = A_bar_p.size1();
                    } else if ((A_bar_p.size1() != 0) && (loop_ub != 0)) {
                        t = A_bar_p.size1();
                    } else {
                        if (A_bar_p.size1() > 0) {
                            t = A_bar_p.size1();
                        } else {
                            t = 0;
                        }
                        if (A_bar_p.size1() > t) {
                            t = A_bar_p.size1();
                        }
                    }
                    empty_non_axis_sizes = (t == 0);
                    if (empty_non_axis_sizes ||
                        ((A_bar_p.size1() != 0) && (b_loop_ub != 0))) {
                        input_sizes_idx_1 = i1 - i;
                    } else {
                        input_sizes_idx_1 = 0;
                    }
                    if (empty_non_axis_sizes || ((A_bar_p.size1() != 0) && (loop_ub != 0))) {
                        sizes_idx_1 = loop_ub;
                    } else {
                        sizes_idx_1 = 0;
                    }
                    vlen = A_bar_p.size1();
                    b_A_tilde.resize(A_bar_p.size1(), b_loop_ub);
                    for (i1 = 0; i1 < b_loop_ub; i1++) {
                        for (i3 = 0; i3 < vlen; i3++) {
                            b_A_tilde[i3 + b_A_tilde.size1() * i1] =
                                    A_bar_p[i3 + A_bar_p.size1() * ((i + i1) + 1)];
                        }
                    }
                    b_loop_ub = A_bar_p.size1();
                    b_A_bar_p.resize(A_bar_p.size1(), loop_ub);
                    for (i = 0; i < loop_ub; i++) {
                        for (i1 = 0; i1 < b_loop_ub; i1++) {
                            b_A_bar_p[i1 + b_A_bar_p.size1() * i] =
                                    A_bar_p[i1 + A_bar_p.size1() * i];
                        }
                    }
                    for (i = 0; i < input_sizes_idx_1; i++) {
                        for (i1 = 0; i1 < t; i1++) {
                            A_t[i1 + A_t.size1() * (i2 + i)] = b_A_tilde[i1 + t * i];
                        }
                    }
                    for (i = 0; i < sizes_idx_1; i++) {
                        for (i1 = 0; i1 < t; i1++) {
                            A_t[i1 + A_t.size1() * ((i2 + i) + input_sizes_idx_1)] =
                                    b_A_bar_p[i1 + t * i];
                        }
                    }
                    // Update the permutation matrix P_tilde
                    // 'partition:54' I_d = eye(d(k));
                    if (c_d < 0.0) {
                        b_t = 0.0;
                    } else {
                        b_t = c_d;
                    }
                    t = static_cast<int>(b_t);
                    P_hat.resize(t, t);
                    loop_ub = static_cast<int>(b_t) * static_cast<int>(b_t);
                    for (i = 0; i < loop_ub; i++) {
                        P_hat[i] = 0.0;
                    }
                    if (static_cast<int>(b_t) > 0) {
                        for (sizes_idx_1 = 0; sizes_idx_1 < t; sizes_idx_1++) {
                            P_hat[sizes_idx_1 + P_hat.size1() * sizes_idx_1] = 1.0;
                        }
                    }
                    // 'partition:55' P_d = [I_d(:, r+1:d(k)), I_d(:, 1:r)];
                    if (r + 1.0 > c_d) {
                        i = -1;
                        i1 = -1;
                    } else {
                        i = r - 1;
                        i1 = static_cast<int>(d[k - 1]) - 1;
                    }
                    if (1 > r) {
                        loop_ub = 0;
                    } else {
                        loop_ub = r;
                    }
                    b_loop_ub = i1 - i;
                    if ((P_hat.size1() != 0) && (b_loop_ub != 0)) {
                        t = P_hat.size1();
                    } else if ((P_hat.size1() != 0) && (loop_ub != 0)) {
                        t = P_hat.size1();
                    } else {
                        if (P_hat.size1() > 0) {
                            t = P_hat.size1();
                        } else {
                            t = 0;
                        }
                        if (P_hat.size1() > t) {
                            t = P_hat.size1();
                        }
                    }
                    empty_non_axis_sizes = (t == 0);
                    if (empty_non_axis_sizes || ((P_hat.size1() != 0) && (b_loop_ub != 0))) {
                        input_sizes_idx_1 = i1 - i;
                    } else {
                        input_sizes_idx_1 = 0;
                    }
                    if (empty_non_axis_sizes || ((P_hat.size1() != 0) && (loop_ub != 0))) {
                        sizes_idx_1 = loop_ub;
                    } else {
                        sizes_idx_1 = 0;
                    }
                    // 'partition:56' P_hat = P_hat * P_d;
                    vlen = P_hat.size1() - 1;
                    b_A_tilde.resize(P_hat.size1(), b_loop_ub);
                    for (i1 = 0; i1 < b_loop_ub; i1++) {
                        for (i2 = 0; i2 <= vlen; i2++) {
                            b_A_tilde[i2 + b_A_tilde.size1() * i1] =
                                    P_hat[i2 + P_hat.size1() * ((i + i1) + 1)];
                        }
                    }
                    vlen = P_hat.size1() - 1;
                    b_A_bar_p.resize(P_hat.size1(), loop_ub);
                    for (i = 0; i < loop_ub; i++) {
                        for (i1 = 0; i1 <= vlen; i1++) {
                            b_A_bar_p[i1 + b_A_bar_p.size1() * i] = P_hat[i1 + P_hat.size1() * i];
                        }
                    }
                    A_bar_p.resize(t, input_sizes_idx_1 + sizes_idx_1);
                    for (i = 0; i < input_sizes_idx_1; i++) {
                        for (i1 = 0; i1 < t; i1++) {
                            A_bar_p[i1 + A_bar_p.size1() * i] = b_A_tilde[i1 + t * i];
                        }
                    }
                    for (i = 0; i < sizes_idx_1; i++) {
                        for (i1 = 0; i1 < t; i1++) {
                            A_bar_p[i1 + A_bar_p.size1() * (i + input_sizes_idx_1)] =
                                    b_A_bar_p[i1 + t * i];
                        }
                    }
                    prod(b_p, A_bar_p, P_hat);
                    // 'partition:58' d(k) = r;
                    d[k - 1] = r;
                } else {
                    // 'partition:59' else
                    // Permute the columns of A_t and the entries of x_tilde
                    // 'partition:61' A_t(:, p:q) = A_bar_p;
                    if (p > q) {
                        i = 1;
                    } else {
                        i = p;
                    }
                    loop_ub = A_bar_p.size2();
                    for (i1 = 0; i1 < loop_ub; i1++) {
                        b_loop_ub = A_bar_p.size1();
                        for (i2 = 0; i2 < b_loop_ub; i2++) {
                            A_t[i2 + A_t.size1() * ((i + i1) - 1)] =
                                    A_bar_p[i2 + A_bar_p.size1() * i1];
                        }
                    }
                }
                // Update the permutation matrix P_tilde
                // 'partition:64' P_tilde(p:q, p:q) = P_hat;
                if (p > q) {
                    i = 1;
                    i1 = 1;
                } else {
                    i = p;
                    i1 = p;
                }
                loop_ub = P_hat.size2();
                for (i2 = 0; i2 < loop_ub; i2++) {
                    b_loop_ub = P_hat.size1();
                    for (i3 = 0; i3 < b_loop_ub; i3++) {
                        P_tilde[((i + i3) + P_tilde.size1() * ((i1 + i2) - 1)) - 1] =
                                P_hat[i3 + P_hat.size1() * i2];
                    }
                }
                // 'partition:65' P_tmp = P_tmp * P_tilde;
                b_A_tilde.resize(P_tmp.size1(), P_tmp.size2());
                loop_ub = P_tmp.size1() * P_tmp.size2() - 1;
                for (i = 0; i <= loop_ub; i++) {
                    b_A_tilde[i] = P_tmp[i];
                }
                prod(b_A_tilde, P_tilde, P_tmp);
                // 'partition:67' p = q - r + 1;
                p = q - r + 1;
                // 'partition:68' R_t(:, p:q) = R(:, 1:r);
                if (1 > r) {
                    loop_ub = 0;
                } else {
                    loop_ub = r;
                }
                if (p > q) {
                    i = 1;
                } else {
                    i = p;
                }
                b_loop_ub = R.size1();
                for (i1 = 0; i1 < loop_ub; i1++) {
                    for (i2 = 0; i2 < b_loop_ub; i2++) {
                        R_t[i2 + R_t.size1() * ((i + i1) - 1)] = R[i2 + R.size1() * i1];
                    }
                }
                // 'partition:69' Q_t(:, p:q) = Q(:, 1:r);
                if (1 > r) {
                    loop_ub = 0;
                } else {
                    loop_ub = r;
                }
                if (p > q) {
                    i = 1;
                } else {
                    i = p;
                }
                b_loop_ub = Q.size1();
                for (i1 = 0; i1 < loop_ub; i1++) {
                    for (i2 = 0; i2 < b_loop_ub; i2++) {
                        Q_t[i2 + Q_t.size1() * ((i + i1) - 1)] = Q[i2 + Q.size1() * i1];
                    }
                }
                // 'partition:71' q = q - r;
                q -= r;
            }
            // 'partition:73' d = d(1:k);
            if (1 > k) {
                i = 0;
            } else {
                i = k;
            }
            d.resize(i);
            // Remove the extra columns of the d

            //  %Test
            //  [H, P_H, x_H, Q_H, R_H, i_H] = partition_H(A_, x_, m, n);
            //  size(A_t)
            //  size(H)
            //  AD=norm(A_t-H, 2)
            //  PD=norm(P_tmp-P_H,2)
            //  xD=norm(x_H-x_tilde,2)
            //  QD=norm(Q_t-Q_H,2)
            //  RD=norm(R_t-R_H,2)
            //

            //  dk
            //  i_H
        }

//        returnType <scalar, index>
//        bsic(b_vector &x_cur, scalar v_norm_cur, index mode) {
//
//            b_vector x_tmp(x_cur), x_per(n, 0), y_bar(m, 0);
//            b_matrix A_T, A_P(A), P_tmp, P_par(I), Q_t, R_t, P_fin; //A could be permuted based on the init point
//
//            b_vector stopping(3, 0);
//
//            b_vector z(n, 0);
//            scalar time = omp_get_wtime();
//
//            index b_count = 0;
//
//            if (v_norm_cur <= tolerance) {
//                stopping[0] = 1;
//                time = omp_get_wtime() - time;
//                return {{}, time, v_norm_cur};
//            }
//
//            index cur_1st, cur_end, i, j, ll, k1, per = -1, best_per = -1, iter = 0, p;
//            scalar v_norm = v_norm_cur;
//            P_tmp.resize(A_P.size2(), A_P.size2(), false);
//
//            time = omp_get_wtime();
//            for (index itr = 0; itr < 3000; itr++) {
//                iter = itr; //mode ? rand() % search_iter : itr;
//
//                A_P.clear();
//                x_tmp.clear();
//                for (i = 0; i < n; i++) {
//                    p = permutation[iter][i] - 1;
//                    column(A_P, i) = column(A, p);
//                    x_tmp[i] = x_per[p];
//                }
//
//                // 'SCP_Block_Optimal_2:104' per = true;
//                P_tmp.clear();
//                P_tmp.assign(I);
//                si_vector d;
//                partition(A_P, P_tmp, Q_t, R_t, d);
//                b_vector x_t;
//                prod(P_tmp, x_tmp, x_t);
//                per = iter;
//                //  dk = zeros(2,k);
//                //  s = 0;
//                //  for i=1:k
//                //      dk(2,i)=n-s;
//                //      dk(1,i)=n-s-t(i)+1;
//                //      s = s + t(i);
//                //  end
//                // 'SCP_Block_Optimal_2:56' for j = 1:t.size2()
//                index s = 0;
//                for (j = 0; j < d.size(); j++) {
//                    cur_1st = n - s - 1;//t(0, j);
//                    cur_end = n - s - d(j) + 1;//t(1, j);
//                    s += d(j);
//                    y_bar.clear();
//                    if (cur_end == n - 1) {
//                        auto a1 = subrange(A_P, 0, m, 0, cur_1st);
//                        y_bar = y - prod(a1, subrange(x_tmp, 0, cur_1st));
//                    } else if (cur_1st == 0) {
//                        auto a2 = subrange(A_P, 0, m, cur_end + 1, n);
//                        y_bar = y - prod(a2, subrange(x_tmp, cur_end + 1, n));
//                    } else {
//                        auto a1 = subrange(A_P, 0, m, 0, cur_1st);
//                        auto a2 = subrange(A_P, 0, m, cur_end + 1, n);
//                        y_bar = y - prod(a1, subrange(x_tmp, 0, cur_1st)) -
//                                prod(a2, subrange(x_tmp, cur_end + 1, n));
//                    }
//
//                    //  Compute optimal solution
//                    ll = cur_end - cur_1st + 1;
//                    A_T.resize(m, ll, false);
//
//                    for (i = cur_1st; i <= cur_end; i++) {
//                        column(A_T, i - cur_1st) = column(A_P, i);
//                    }
//
//                    auto Q_j = subrange(Q_t, 0, m, cur_1st, cur_end + 1);
////                    helper::display<scalar, index>(Q_t, "Q_t");
////                    helper::display<scalar, index>(Q_j, "Q_j");
//                    auto y_t = prod(trans(Q_j), y_bar);
//                    auto R_j = subrange(R_t, 0, y_t.size(), cur_1st, cur_end + 1);
////                    helper::display<scalar, index>(R_j, "R_j");
////                    z.resize(ll, false);
////                    CILS_Reduction<scalar, index> reduction(A_T, y_bar, 0, upper);
////                    reduction.aip();
////                    helper::display<scalar, index>(reduction.R, "R_R");
////                    helper::display<scalar, index>(reduction.y, "y_R");
////                    helper::display<scalar, index>(y_t, "y_t");
////                    reduction.aspl_p_serial();
//
//                    CILS_SECH_Search<scalar, index> ils(ll, ll, qam);
////                    ils.obils_search(0, ll, 1, reduction.R, reduction.y, z);
//                    ils.obils_search(0, ll, 1, R_j, y_t, z);
//
////                    subrange(x_tmp, cur_1st, cur_1st + ll) = z;
////                    axpy_prod(reduction.P, z, x, true);
//                    for (i = 0; i < ll; i++) {
//                        x_tmp[cur_1st + i] = z[i];
//                    }
//
//                }
//
//                v_norm_cur = norm_2(y - prod(A_P, x_tmp));
//
//                if (v_norm_cur < v_norm) {
//                    // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
//                    x_per.clear();
//                    for (ll = 0; ll < n; ll++) {
//                        x_per[ll] = x_tmp[ll];
//                    }
//                    P_par.assign(P_tmp);
//                    // 'SCP_Block_Optimal_2:88' v_norm_cur = v_norm_temp;
//                    //  v_norm_cur = v_norm_temp;
//                    // 'SCP_Block_Optimal_2:89' if per
//                    best_per = per;
//                    // 'SCP_Block_Optimal_2:94' if v_norm_cur <= tolerance
//                    if (v_norm_cur <= tolerance) {
//                        // 'SCP_Block_Optimal_2:95' stopping(2)=1;
//                        stopping[1] = 1;
//                    }
//                    // 'SCP_Block_Optimal_2:99' v_norm = v_norm_cur;
//                    v_norm = v_norm_cur;
//                }
//                // If we don't decrease the residual, keep trying permutations
//            }
//            // 'SCP_Block_Optimal_2:44' P_tmp = eye(n);
//            b_matrix P_cur(I);
//
//            //I(:, permutation(:, best_per))
//            if (best_per >= 0) {
//                for (i = 0; i < n; i++) {
//                    column(P_cur, i) = column(I, permutation[best_per][i] - 1);
//                }
//
//                P_tmp.assign(I);
//                axpy_prod(P_cur, P_par, P_tmp, true);
//                axpy_prod(P_tmp, x_per, x_cur, true);
//            }
//
//            time = omp_get_wtime() - time;
//            return {{}, time, v_norm_cur};
//        }
//
//        returnType <scalar, index>
//        pbsic(b_vector &x_cur, scalar v_norm_cur, index n_proc, index mode) {
//            // 'SCP_Block_Optimal_2:24' stopping=zeros(1,3);
//            b_vector stopping(3, 0);
//
//            scalar time = omp_get_wtime();
//            //  Subfunctions: SCP_opt
//
//            // 'SCP_Block_Optimal_2:25' b_count = 0;
//            index b_count = 0;
//            index best_proc = -1;
//            // 'SCP_Block_Optimal_2:27' if v_norm_cur <= tolerance
//            if (v_norm_cur <= tolerance) {
//                // 'SCP_Block_Optimal_2:28' stopping(1)=1;
//                stopping[0] = 1;
//                time = omp_get_wtime() - time;
//                return {{}, time, v_norm_cur};
//            }
//            index cur_1st, cur_end, i, t, k1;
//            // 'SCP_Block_Optimal_2:32' t.size2() = ceil(n/m);
//            // 'SCP_Block_Optimal_2:33' indicator = zeros(2, t.size2());
//
//            scalar v_norm = v_norm_cur;
//            // 'SCP_Block_Optimal_2:50' per = false;
//            std::vector<b_vector> x(n_proc, b_vector(x_cur));
//            b_vector v_norm_cur_proc(n_proc, v_norm), x_per(x_cur);
//            si_vector best_per_proc(n_proc, 0);
//
//            // private variables
//            b_vector z, z_z; //z_z = Z * z;
//            b_vector A_T, A_P(m * n, 0), x_tmp(x_cur), y_bar(m, 0), y(m, 0);
//
//            time = omp_get_wtime();
//#pragma omp parallel default(shared) num_threads(n_proc) private(z, z_z, A_T, cur_1st, cur_end, i, t, k1) firstprivate(A_P, x_tmp, v_norm, y_bar, y)
//            {
//                index t_num = omp_get_thread_num(), per = -1;
//                index start = omp_get_thread_num() * search_iter / n_proc;
//                index end = (omp_get_thread_num() + 1) * search_iter / n_proc;
//
//                for (index itr = start; itr < end; itr++) {
//                    index iter = mode ? rand()// search_iter : itr;
//                    // H_Apply permutation strategy to update x_cur and v_norm_cur
//                    // [x_tmp, v_norm_temp] = block_opt(A_P, y, x_tmp, n, indicator);
//                    // Corresponds to H_Algorithm 12 (Block Optimal) in Report 10
//                    // 'SCP_Block_Optimal_2:56' for j = 1:t.size2()
//                    for (k1 = 0; k1 < n; k1++) {
//                        for (i = 0; i < m; i++) {
//                            A_P[i + m * k1] = H[i + m * (permutation[iter][k1] - 1)];
//                        }
//                        x_tmp[k1] = x[t_num][permutation[iter][k1] - 1];
//                    }
//                    // 'SCP_Block_Optimal_2:104' per = true;
//                    per = iter;
//
//                    for (index j = 0; j < t.size2(); j++) {
//                        cur_1st = indicator[2 * j];
//                        cur_end = indicator[2 * j + 1];
//                        y_bar.clear();
//                        for (k1 = 0; k1 <= cur_1st - 2; k1++) {
//                            for (i = 0; i < m; i++) {
//                                t = k1 * m + i;
//                                y_bar[i] += A_P[i + m * (t / m)] * x_tmp[k1];
//                            }
//                        }
//                        for (k1 = 0; k1 < n - cur_end; k1++) {
//                            for (i = 0; i < m; i++) {
//                                t = k1 * m + i;
//                                y_bar[i] += A_P[i + m * (cur_end + t / m)] * x_tmp[cur_end + k1];
//                            }
//                        }
//                        for (t = 0; t < m; t++) {
//                            y_bar[t] = y[t] - y_bar[t];
//                        }
//                        l = cur_end - cur_1st + 1;
//                        A_T.resize(m * t);
//                        for (k1 = 0; k1 < t; k1++) {
//                            for (i = 0; i < m; i++) {
//                                A_T[i + m * k1] = A_P[i + m * (cur_1st - 1 + k1)];
//                            }
//                        }
//                        // 'SCP_Block_Optimal_2:80' z = obils(H_adj, y_bar, l, u);
//                        z.resize(l);
//                        z_z.resize(t);
//                        z.assign(t, 0);
//                        z_z.assign(t, 0);
//
//                        CILS_Reduction<scalar, index> reduction(m, t, 0, upper, 0, 0);
//                        reduction.aip(A_T, y_bar);
//
//                        CILS_SECH_Search<scalar, index> ils(t, t, qam);
//                        ils.obils_search2(reduction.R_Q, reduction.y_q, z_z);
//
//                        // 'SCP_Block_Optimal_2:81' x_tmp(cur_1st:cur_end) = 2 * z + e_vec;
//                        for (i = 0; i < t; i++) {
//                            z[reduction.p[i] - 1] = z_z[i];
//                            //reduction.y_q[i];//z_z[reduction.p[i] - 1];//2.0 * z_z[i] + 1;
//                        }
//                        for (i = 0; i < t; i++) {
//                            x_tmp[cur_1st + i - 1] = z[i];
//                            //reduction.y_q[i];//z_z[reduction.p[i] - 1];//2.0 * z_z[i] + 1;
//                        }
//                    }
//
//                    // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(y - A_P * x_tmp);
//                    v_norm_cur_proc[t_num] = helper::find_residual(m, n, A_P.data(), x_tmp.data(), y.data());
//                    // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
//                    if (v_norm_cur_proc[t_num] < v_norm) {
//                        // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
//                        for (l = 0; l < n; l++) {
//                            x_per[t] = x_tmp[t];
//                            x[t_num][t] = x_tmp[t];
//                        }
//                        // 'SCP_Block_Optimal_2:89' if per
//                        best_per_proc[t_num] = per;
//                        per = -1;
//                        // 'SCP_Block_Optimal_2:94' if v_norm_cur <= tolerance
//                        if (v_norm_cur_proc[t_num] <= tolerance) {
//                            // 'SCP_Block_Optimal_2:95' stopping(2)=1;
//                            stopping[1] = 1;
//                            iter = search_iter;
//                            best_proc = t_num;
//                        }
//                        v_norm = v_norm_cur_proc[t_num];
//                    }
//                    if (stopping[1])
//                        iter = search_iter;
//                }
//            }
//
//            b_vector P_cur(n * n, 0), P_tmp(n * n, 0);
//            // 'SCP_Block_Optimal_2:44' P_tmp = eye(n);
//
//            b_vector x_a(n, 0);
//
//            if (best_proc < 0) {
//                scalar v_min = INFINITY;
//                for (k1 = 0; k1 < n_proc; k1++) {
//                    if (v_norm_cur_proc[k1] < v_min) {
//                        v_min = v_norm_cur_proc[k1];
//                        best_proc = k1;
//                    }
//                }
//            }
//
//            for (k1 = 0; k1 < n; k1++) {
//                for (i = 0; i < n; i++) {
//                    P_cur[i + n * k1] = P_tmp[i + n * (permutation[best_per_proc[best_proc]][k1] - 1)];
//                }
//            }
//
//            // 'SCP_Block_Optimal_2:115' x_cur = P_cur * x_cur;
//            helper::mtimes_Axy<scalar, index>(n, n, P_cur.data(), x[best_proc].data(), x_a.data());
//            for (i = 0; i < n; i++) {
//                x_cur[i] = x_a[i];
//            }
//
//            time = omp_get_wtime() - time;
//            return {stopping, time, v_norm};
//        }
//
//        returnType <scalar, index>
//        pabsic(b_vector &x_cur, scalar *v_norm_cur, index size) {
//
//            // 'SCP_Block_Optimal_2:24' stopping=zeros(1,3);
//            b_vector stopping(3, 0);
//
//            scalar time = omp_get_wtime();
//            //  Subfunctions: SCP_opt
//            scalar flag = 0.0;
//            // 'SCP_Block_Optimal_2:25' b_count = 0;
//            // 'SCP_Block_Optimal_2:27' if v_norm_cur <= tolerance
//            if (v_norm_cur[0] <= tolerance) {
//                // 'SCP_Block_Optimal_2:28' stopping(1)=1;
//                stopping[0] = 1;
//                time = omp_get_wtime() - time;
//                return {{}, time, v_norm_cur[0]};
//            }
//            index cur_1st, cur_end, i, t, k1, per = -1, best_per = -1;
//            // 'SCP_Block_Optimal_2:32' t.size2() = ceil(n/m);
//            // 'SCP_Block_Optimal_2:33' indicator = zeros(2, t.size2());
//            auto v_norm = (double *) malloc(2 * sizeof(double));
//            auto v_norm_rank = (double *) malloc(2 * sizeof(double));
//            v_norm_rank[0] = v_norm_cur[0];
//            v_norm[0] = v_norm_cur[0];
//            v_norm[1] = rank;
//
//            b_vector x_per(x_cur), x(n, 0);
//            if (verbose) {
//                cout << "here: " << rank * search_iter / size << "," << (rank + 1) * search_iter / size
//                     << endl;
//                if (rank != 0) {
//                    helper::display<scalar, index>(m, n, H.data(), "H" + to_string(rank));
//                    helper::display<scalar, index>(1, n, permutation[0].data(), "Per" + to_string(rank));
//                }
//            }
//
////        index start = rank * search_iter / size;
////        index end = (rank + 1) * search_iter / size;
////        index slice = (end - start) / t.size2();
////        index iter = start;
////        b_matrix  A_P(H);
//            if (rank == 0)
//                time = MPI_Wtime();
///*#pragma omp parallel default(shared) num_threads(t.size2()) private(v_norm_cur, cur_1st, cur_end, i, t, k1, per)
//        {
//            vector<scalar> A_T, x_tmp(x_cur), y_bar(m, 0), y(m, 0);
//            vector<scalar> z, z_z; //z_z = Z * z;
//            v_norm_cur = (double *) malloc(1 * sizeof(double));
//            //v_norm[0] = v_norm_rank[0];
//
////            index t_num = omp_get_thread_num();
////            index t_start = start + slice * t_num;
////            index t_end = start + slice * (t_num + 1);
//            while (iter < end) {
//                if (omp_get_thread_num() == 0) {
//                    for (k1 = 0; k1 < n; k1++) {
//                        for (i = 0; i < m; i++) {
//                            A_P[i + m * k1] = H[i + m * (permutation[iter][k1] - 1)];
//                        }
//                        x_tmp[k1] = x_per[permutation[iter][k1] - 1];
//                    }
//                    per = iter;
//                }
//#pragma omp barrier
//                for (index inner = 0; inner < 4; inner++) {
//#pragma omp for schedule(static, 1) nowait
//                    for (index j = 0; j < t.size2(); j++) {
////                        if (rank == 0 && inner == 0)
////                            cout << omp_get_thread_num() << " " << endl;
//                        cur_1st = indicator[2 * j];
//                        cur_end = indicator[2 * j + 1];
//                        y_bar.clear();
//                        for (k1 = 0; k1 <= cur_1st - 2; k1++) {
//                            for (i = 0; i < m; i++) {
//                                t = k1 * m + i;
//                                y_bar[i] += A_P[i + m * (t / m)] * x_tmp[k1];
//                            }
//                        }
//                        for (k1 = 0; k1 < n - cur_end; k1++) {
//                            for (i = 0; i < m; i++) {
//                                t = k1 * m + i;
//                                y_bar[i] += A_P[i + m * (cur_end + t / m)] * x_tmp[cur_end + k1];
//                            }
//                        }
//                        for (t = 0; t < m; t++) {
//                            y_bar[t] = y[t] - y_bar[t];
//                        }
//                        l = cur_end - cur_1st + 1;
//                        A_T.resize(m * t);
//                        for (k1 = 0; k1 < t; k1++) {
//                            for (i = 0; i < m; i++) {
//                                A_T[i + m * k1] = A_P[i + m * (cur_1st - 1 + k1)];//2.0 *
//                            }
//                        }
//                        // 'SCP_Block_Optimal_2:80' z = obils(H_adj, y_bar, l, u);
//                        z.resize(l);
//                        z_z.resize(t);
//                        z.assign(t, 0);
//                        z_z.assign(t, 0);
//
//                        CILS_Reduction<scalar, index> reduction(m, t, 0, upper, 0, 0);
//                        reduction.aip(A_T, y_bar);
//
//                        CILS_SECH_Search<scalar, index> ils(t, t, qam);
//                        ils.obils_search2(reduction.R_Q, reduction.y_q, z_z);
//
//                        // 'SCP_Block_Optimal_2:81' x_tmp(cur_1st:cur_end) = 2 * z + e_vec;
//                        for (i = 0; i < t; i++) {
//                            z[reduction.p[i] - 1] = z_z[i];
//                            //reduction.y_q[i];//z_z[reduction.p[i] - 1];//2.0 * z_z[i] + 1;
//                        }
//                        for (i = 0; i < t; i++) {
//                            x_tmp[cur_1st + i - 1] = z[i];
//                            //reduction.y_q[i];//z_z[reduction.p[i] - 1];//2.0 * z_z[i] + 1;
//                        }
//                    }
//                }
//
//                if (omp_get_thread_num() == 0) {
//                    iter++;
//                    // x_tmp
//                    // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(y - A_P * x_tmp);
//                    v_norm_cur[0] = helper::find_residual<scalar, index>(m, n, A_P.data(), x_tmp.data(), y.data());
//                    // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
//                    if (v_norm_cur[0] < v_norm[0]) {
//                        // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
//                        for (l = 0; l < n; l++) {
//                            x_per[t] = x_tmp[t];
//                        }
//                        // 'SCP_Block_Optimal_2:88' v_norm_cur = v_norm_temp;
//                        // 'SCP_Block_Optimal_2:89' if per
//                        best_per = per;
//                        per = -1;
//                        // 'SCP_Block_Optimal_2:94' if v_norm_cur <= tolerance
//                        if (v_norm_cur[0] <= tolerance) {
//                            // 'SCP_Block_Optimal_2:95' stopping(2)=1;
//                            stopping[1] = 1;
//                            iter = end;
//                        }
//                        v_norm[0] = v_norm_cur[0];
//                    }
//                }
//#pragma omp barrier
//                if (stopping[1])
//                    iter = end;
//            }
//
//        }
////        */
////
////#pragma omp parallel default(shared) num_threads(t.size2()) private(cur_1st, cur_end, i, t, k1)
////            {
////                b_vector x_tmp(x_cur), y_bar(m, 0), y(m, 0);
////                b_matrix  A_P(H);
////                b_vector A_T, z, z_z; //z_z = Z * z;
//////#pragma omp for schedule(static, chunk)
////                index t_num = omp_get_thread_num(), per = -1;
////                index start = omp_get_thread_num() * search_iter / t.size2();
////                index end = (omp_get_thread_num() + 1) * search_iter / t.size2();
////                for (index iter = start; iter < end; iter++) {
////                    for (k1 = 0; k1 < n; k1++) {
////                        for (i = 0; i < m; i++) {
////                            A_P[i + m * k1] = H[i + m * (permutation[iter][k1] - 1)];
////                        }
////                        x_tmp[k1] = x_per[permutation[iter][k1] - 1];
////                    }
////                    per = iter;
////#pragma omp barrier
////                    for (index inner = 0; inner < 4; inner++) {
////#pragma omp for schedule(dynamic) nowait
////                        for (index j = 0; j < t.size2(); j++) {
////                            cur_1st = indicator[2 * j];
////                            cur_end = indicator[2 * j + 1];
////                            y_bar.clear();
////                            for (k1 = 0; k1 <= cur_1st - 2; k1++) {
////                                for (i = 0; i < m; i++) {
////                                    t = k1 * m + i;
////                                    y_bar[i] += A_P[i + m * (t / m)] * x_tmp[k1];
////                                }
////                            }
////                            for (k1 = 0; k1 < n - cur_end; k1++) {
////                                for (i = 0; i < m; i++) {
////                                    t = k1 * m + i;
////                                    y_bar[i] += A_P[i + m * (cur_end + t / m)] * x_tmp[cur_end + k1];
////                                }
////                            }
////                            for (t = 0; t < m; t++) {
////                                y_bar[t] = y[t] - y_bar[t];
////                            }
////                            l = cur_end - cur_1st + 1;
////                            A_T.resize(m * t);
////                            for (k1 = 0; k1 < t; k1++) {
////                                for (i = 0; i < m; i++) {
////                                    A_T[i + m * k1] = A_P[i + m * (cur_1st - 1 + k1)];//2.0 *
////                                }
////                            }
////                            // 'SCP_Block_Optimal_2:80' z = obils(H_adj, y_bar, l, u);
////                            z.resize(l);
////                            z_z.resize(t);
////                            z.assign(t, 0);
////                            z_z.assign(t, 0);
////
////                            CILS_Reduction<scalar, index> reduction(m, t, 0, upper, 0, 0);
////                            reduction.aip(A_T, y_bar);
////
////                            CILS_SECH_Search<scalar, index> ils(t, t, qam);
////                            ils.obils_search2(reduction.R_Q, reduction.y_q, z_z);
////
////                            // 'SCP_Block_Optimal_2:81' x_tmp(cur_1st:cur_end) = 2 * z + e_vec;
////                            for (i = 0; i < t; i++) {
////                                z[reduction.p[i] - 1] = z_z[i];
////                                //reduction.y_q[i];//z_z[reduction.p[i] - 1];//2.0 * z_z[i] + 1;
////                            }
////                            for (i = 0; i < t; i++) {
////                                x_tmp[cur_1st + i - 1] = z[i];
////                                //reduction.y_q[i];//z_z[reduction.p[i] - 1];//2.0 * z_z[i] + 1;
////                            }
////                        }
////                    }
////                    v_norm_cur[0] = helper::find_residual<scalar, index>(m, n, A_P.data(), x_tmp.data(),
////                                                                         y.data());
////                    // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
////                    if (v_norm_cur[0] < v_norm[0]) {
////                        // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
////                        for (l = 0; l < n; l++) {
////                            x_per[t] = x_tmp[t];
////                        }
////                        // 'SCP_Block_Optimal_2:88' v_norm_cur = v_norm_temp;
////                        // 'SCP_Block_Optimal_2:89' if per
////                        best_per = per;
////                        per = -1;
////                        // 'SCP_Block_Optimal_2:94' if v_norm_cur <= tolerance
////                        if (v_norm_cur[0] <= tolerance) {
////                            // 'SCP_Block_Optimal_2:95' stopping(2)=1;
////                            stopping[1] = 1;
////                        }
////                        v_norm[0] = v_norm_cur[0];
////                    }
////                    if (stopping[1]) {
////                        iter = search_iter;
////                    }
////                }
////            }
////            // x_tmp
////            // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(y - A_P * x_tmp);
////
////            MPI_Barrier(MPI_COMM_WORLD);
////            MPI_Allreduce(v_norm, v_norm_rank, 1, MPI_2DOUBLE_PRECISION, MPI_MINLOC, MPI_COMM_WORLD);
////
////            if (rank == v_norm_rank[1]) {
////                b_vector P_cur(n * n, 0), P_tmp(n * n, 0);
////                // 'SCP_Block_Optimal_2:44' P_tmp = eye(n);
////                helper::eye<scalar, index>(n, P_tmp.data());
////                helper::eye<scalar, index>(n, P_cur.data());
////
////                if (best_per >= 0) {
////                    for (k1 = 0; k1 < n; k1++) {
////                        for (i = 0; i < n; i++) {
////                            P_cur[i + n * k1] = P_tmp[i + n * (permutation[best_per][k1] - 1)];
////                        }
////                    }
////                }
////                //'SCP_Block_Optimal_2:115' x_cur = P_cur * x_cur;
////                helper::mtimes_Axy<scalar, index>(n, n, P_cur.data(), x_per.data(), x.data());
////                for (i = 0; i < n; i++) {
////                    x_cur[i] = x[i];
////                }
////            }
////            MPI_Barrier(MPI_COMM_WORLD);
////            MPI_Bcast(&x_cur[0], n, MPI_DOUBLE, v_norm_rank[1], MPI_COMM_WORLD);
////            if (rank == 0)
////                time = MPI_Wtime() - time;
////
////            if (verbose)
////                helper::display(n, x.data(), "x:" + std::to_string(rank));
////
////            return {stopping, time, v_norm[0]};
////        }
//
//        };
    };
}