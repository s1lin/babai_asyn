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
    class CILS_SO_UBILS {
    private:
        CILS<scalar, index> cils;

    public:
        index m, n;
        b_vector z_hat, y;
        b_matrix A, P, t;

        explicit CILS_SO_UBILS(CILS<scalar, index> &cils) {
            this->m = cils.m;
            this->n = cils.n;
            this->z_hat.resize(n);
            this->z_hat.clear();
            this->A.resize(m, n);
            this->A.assign(cils.A);
            this->P.resize(n, n);
            this->P.assign(cils.I);
            this->y.resize(m);
            this->y.assign(cils.y);
            this->cils.tolerance = cils.tolerance;
            this->cils = cils;
        }

        /** 
         *Warning: A is permuted.
         *[s_bar_IP, v_norm, HH, Piv] = SIC_IP(H, y, N, bound) applies the SIC
         *initial point method
         *
         *Inputs:
         *    H - K-by-N real matrix
         *    y - K-dimensional real vector
         *    N - integer scalar
         *    bound - integer scalar for the constraint
         *
         *Outputs:
         *    s_bar_IP - N-dimensional integer vector for the initial point
         *    v_norm - real scalar for the norm of the residual vector
         *    corresponding to s_bar_IP
         *    HH - K-by-N real matrix, permuted H for sub-optimal methods
         *    Piv - N-by-N real matrix where HH*Piv'=H
         * @param x
         * @return
         */

        returnType<scalar, index> sic(b_vector &x) {

            x.clear();

            b_vector y_bar(y);
            //Variable Declarations

            index k = -1, x_est = 0, x_round = 0;
            scalar res, max_res, aiy, aih, x_tmp;

            scalar time = omp_get_wtime();
            for (index j = n - 1; j >= 0; j--) {
                max_res = INFINITY;
                for (index i = 0; i <= j; i++) {
                    auto Ai = column(A, i);
                    aiy = inner_prod(Ai, y_bar);
                    aih = inner_prod(Ai, Ai);
                    x_tmp = aiy / aih;
                    x_round = max(min((index) round(x_tmp), cils.upper), cils.lower);
                    res = norm_2(y_bar - x_round * Ai);
                    if (res < max_res) {
                        k = i;
                        x_est = x_round;
                        max_res = res;
                    }
                }
                if (k != -1) {
                    //HH(:,[k,j])=HH(:,[j,k]);
                    b_vector Aj = column(A, j);
                    column(A, j) = column(A, k);
                    column(A, k) = Aj;

                    //Piv(:,[k,j])=Piv(:,[j,k]);
                    b_vector Pj = column(P, j);
                    column(P, j) = column(P, k);
                    column(P, k) = Pj;

                    //s_bar_IP(j)=s_est;
                    x[j] = x_est;
                }
                y_bar = y_bar - x[j] * column(A, j);
            }

            scalar v_norm = norm_2(y_bar);
            time = omp_get_wtime() - time;

            return {{}, time, v_norm};
        }

        returnType<scalar, index> gp(b_vector &x, const index search_iter) {
            b_vector ex;

            sd_vector t_bar, t_seq;
            si_vector r1, r2, r3, r4, r5;
            sb_vector b_x_0, b_x_1(n, 0);

            index i, j, k1, k2, k3;
            scalar v_norm = INFINITY;
            b_vector x_cur(n, 0);


            scalar time = omp_get_wtime();

            b_vector c = prod(trans(cils.A), y);

            // 'ubils:225' for iter = 1:search_iter
            for (index iter = 0; iter < search_iter; iter++) {
                // 'ubils:227' g = cils.A'*(cils.A*x_cur-y);
                b_vector r0 = -y;
                r0 = r0 + prod(cils.A, x_cur);
                b_vector g = prod(trans(cils.A), r0);

                //  Check KKT conditions
                // 'ubils:230' if (x_cur==cils.l) == 0
                b_x_0.resize(n, 0);
                for (i = 0; i < n; i++) {
                    b_x_0[i] = !(x_cur[i] == cils.l[i]);
                }
                if (helper::if_all_x_true<index>(b_x_0)) {
                    // 'ubils:231' k1 = 1;
                    k1 = 1;
                } else {
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (x_cur[i] == cils.l[i]) {
                            k3++;
                        }
                    }
                    r1.resize(k3, 0);
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (x_cur[i] == cils.l[i]) {
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
                        // 'ubils:232' elseif (g(x_cur==cils.l) > -1.e-5) == 1
                        // 'ubils:233' k1 = 1;
                        k1 = 1;
                    } else {
                        // 'ubils:234' else
                        // 'ubils:234' k1 = 0;
                        k1 = 0;
                    }
                }
                // 'ubils:236' if (x_cur==cils.u) == 0
                b_x_0.resize(n, 0);
                for (i = 0; i < n; i++) {
                    b_x_0[i] = !(x_cur[i] == cils.u[i]);
                }
                if (helper::if_all_x_true<index>(b_x_0)) {
                    // 'ubils:237' k2 = 1;
                    k2 = 1;
                } else {
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (x_cur[i] == cils.u[i]) {
                            k3++;
                        }
                    }
                    r2.resize(k3, 0);
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (x_cur[i] == cils.u[i]) {
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
                        // 'ubils:238' elseif (g(x_cur==cils.u) < 1.e-5) == 1
                        // 'ubils:239' k2 = 1;
                        k2 = 1;
                    } else {
                        // 'ubils:240' else
                        // 'ubils:240' k2 = 0;
                        k2 = 0;
                    }
                }
                // 'ubils:242' if (cils.l<x_cur & x_cur<cils.u) == 0
                b_x_0.resize(n, 0);
                for (i = 0; i < n; i++) {
                    b_x_0[i] = ((!(cils.l[i] < x_cur[i])) || (!(x_cur[i] < cils.u[i])));
                }
                if (helper::if_all_x_true<index>(b_x_0)) {
                    // 'ubils:243' k3 = 1;
                    k3 = 1;
                } else {
                    b_x_0.resize(n, 0);
                    for (i = 0; i < n; i++) {
                        b_x_0[i] = (cils.l[i] < x_cur[i]);
                        b_x_1[i] = (x_cur[i] < cils.u[i]);
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
                        // 'ubils:244' elseif (g(cils.l<x_cur & x_cur<cils.u) < 1.e-5) == 1
                        // 'ubils:245' k3 = 1;
                        k3 = 1;
                    } else {
                        // 'ubils:246' else
                        // 'ubils:246' k3 = 0;
                        k3 = 0;
                    }
                }
                scalar v_norm_cur = helper::find_residual<scalar, index>(cils.A, x_cur, y);
                // 'ubils:248' if (k1 & k2 & k3)
                if ((k1 != 0) && (k2 != 0) && (k3 != 0) && (v_norm > v_norm_cur)) {
                    k1 = k2 = k3 = 0;
                    for (i = 0; i < n; i++) {
                        scalar x_est = round(x_cur[i]);
                        //x_est = 2.0 * std::floor(x_est / 2.0) + 1.0;
                        if (x_est < 0) {
                            x_est = 0;
                        } else if (x_est > cils.upper) {
                            x_est = cils.upper;
                        }
                        x[i] = x_est;
                    }
                    v_norm = v_norm_cur;
                } else {
                    scalar x_tmp;
                    //  Find the Cauchy poindex
                    // 'ubils:253' t_bar = 1.e5*ones(n,1);
                    t_bar.resize(n, 1e5);
                    // 'ubils:254' t_bar(g<0) = (x_cur(g<0)-cils.u(g<0))./g(g<0);
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
                        r0[i] = (x_cur[r4[i] - 1] - cils.u[r4[i] - 1]) / g[r4[i] - 1];
                    }
                    k3 = 0;
                    for (i = 0; i < n; i++) {
                        if (g[i] < 0.0) {
                            t_bar[i] = r0[k3];
                            k3++;
                        }
                    }
                    // 'ubils:255' t_bar(g>0) = (x_cur(g>0)-cils.l(g>0))./g(g>0);
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
                        r0[i] = (x_cur[r5[i] - 1] - cils.l[r5[i] - 1]) / g[r5[i] - 1];
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
                        // 'ubils:270' q = cils.A*pj_1;
                        b_vector q = prod(cils.A, pj_1);
//                        q.clear();
//                        for (j = 0; j < n; j++) {
//                            for (i = 0; i < m; i++) {
//                                q[i] += cils.A[j * m + i] * pj_1[j];
//                            }
//                        }
                        // 'ubils:271' fj_1d = (cils.A*xt_j_1)'*q - c'*pj_1;
                        ex.resize(t_bar.size());
                        for (j = 0; j < t_bar.size(); j++) {
                            ex[j] = std::fmin(tj_1, t_bar[j]);
                        }
                        ex.resize(n);
                        for (i = 0; i < n; i++) {
                            ex[i] = x_cur[i] - ex[i] * g[i];
                        }
                        r0.resize(m);
                        r0 = prod(cils.A, ex);
//                        for (j = 0; j < n; j++) {
//                            for (i = 0; i < m; i++) {
//                                r0[i] += cils.A[j * m + i] * ex[j];
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
            //s_bar4 = round_index(s_bar4_unrounded, -1, 1);
//        for (i = 0; i < n; i++) {
//            scalar x_est = round(x[i]);
//            //x_est = 2.0 * std::floor(x_est / 2.0) + 1.0;
//            if (x_est < 0) {
//                x_est = 0;
//            } else if (x_est > cils.upper) {
//                x_est = cils.upper;
//            }
//            x[i] = x_est;
//        }

            //scalar v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x.data(), y.data());
            time = omp_get_wtime() - time;
            return {{}, time, v_norm};
        }

        returnType<scalar, index> pd(b_matrix &A_t, b_matrix &P_cur, b_matrix &Q_t, b_matrix &R_t, b_matrix &t_i) {
            index l = n, k = 0, f = 0, r = 0;

            Q_t.resize(m, n, false);
            R_t.resize(m, n, false);
            t.resize(2, n, false);

            while (l >= 1) {
                f = max(0, l - m);
                b_matrix A_l = subrange(A_t, 0, m, f, l);

                b_matrix P_l(cils.I);
                b_matrix Q, R, P_hat;

                helper::qrp(A_l, Q, R, P_hat, true, false);

//                helper::display<scalar, index>(R, "R");
//                helper::display<scalar, index>(P_hat, "P_hat");
                r = 0;
                if (R.size2() > 1) {
                    for (index i = 0; i < R.size2(); i++) {
                        r += abs(R(i, i)) > 1e-6;
                    }
                } else
                    r = (abs(R(0)) > 1e-6);

                b_matrix A_P(A_l.size1(), A_l.size2());
                axpy_prod(A_l, P_hat, A_P);
//                helper::display<scalar, index>(A_P, "A_P");

                index d = A_l.size2();

                //The case where A_l is rank deficient
                if (r < d) {
                    //Permute the columns of Ahat and the entries of
                    //Ahat(:, f:f+d-1 -r ) = A_P(:, r+1:d);
                    //Ahat(:, f+d-r: l) = A_P(:, 1:r);
                    subrange(A_t, 0, m, f, f + d - r) = subrange(A_P, 0, m, r, d);
                    subrange(A_t, 0, m, f + d - r, l) = subrange(A_P, 0, m, 0, r);

                    //Update the permutation matrix P_l
                    identity_matrix I_d(d, d);
                    b_matrix P_d(d, d);
                    subrange(P_d, 0, m, 0, r) = subrange(I_d, 0, m, r, d);
                    subrange(P_d, 0, m, r, d) = subrange(I_d, 0, m, 0, r);
                    P_hat = prod(P_hat, P_d);
                } else {
                    //Permute the columns of Ahat and the entries of
                    subrange(A_t, 0, m, f, l) = A_P;
                }
//                helper::display<scalar, index>(A_t, "A_t");

                f = l - r;

                subrange(R_t, 0, R.size1(), f, l) = subrange(R, 0, R.size1(), 0, r);
                subrange(Q_t, 0, Q.size1(), f, l) = subrange(Q, 0, Q.size1(), 0, r);
                subrange(P_l, f, l, f, l) = P_hat;
                P_cur = prod(P_cur, P_l);

                t_i(0, k) = f;
                t_i(1, k) = l - 1;
                k = k + 1;
                l = l - r;
            }
            t.resize(2, k);
            //cils.indicator = subrange(t, 0, 2, 0, k);
            return {{}, 0, 0};
        }

        returnType<scalar, index>
        block_sic_optimal(b_vector &x_cur, scalar v_norm_cur, index mode) {

            b_vector x_tmp(x_cur), x_per(n, 0), y_bar(m, 0);
            b_matrix A_T, A_P(A), P_tmp, P_par(cils.I), Q_t, R_t, P_fin; //A could be permuted based on the init point

            b_vector stopping(3, 0);

            b_vector z(n, 0);
            scalar time = omp_get_wtime();

            index b_count = 0;

            if (v_norm_cur <= cils.tolerance) {
                stopping[0] = 1;
                time = omp_get_wtime() - time;
                return {{}, time, v_norm_cur};
            }

            index cur_1st, cur_end, i, j, l, k1, per = -1, best_per = -1, iter = 0, p;
            scalar v_norm = v_norm_cur;
            P_tmp.resize(A_P.size2(), A_P.size2(), false);

            time = omp_get_wtime();
            for (index itr = 0; itr < 3000; itr++) {
                iter = itr; //mode ? rand() % cils.search_iter : itr;

                A_P.clear();
                x_tmp.clear();
                for (i = 0; i < n; i++) {
                    p = cils.permutation[iter][i] - 1;
                    column(A_P, i) = column(A, p);
                    x_tmp[i] = x_per[p];
                }

                // 'SCP_Block_Optimal_2:104' per = true;
                P_tmp.clear();
                P_tmp.assign(cils.I);
                pd(A_P, P_tmp, Q_t, R_t,t);
                prod(P_tmp, x_tmp);
                per = iter;

                // 'SCP_Block_Optimal_2:56' for j = 1:t.size2()
                for (j = 0; j < t.size2(); j++) {
                    cur_1st = t(0, j);
                    cur_end = t(1, j);
                    y_bar.clear();
                    if (cur_end == n - 1) {
                        auto a1 = subrange(A_P, 0, m, 0, cur_1st);
                        y_bar = y - prod(a1, subrange(x_tmp, 0, cur_1st));
                    } else if (cur_1st == 0) {
                        auto a2 = subrange(A_P, 0, m, cur_end + 1, n);
                        y_bar = y - prod(a2, subrange(x_tmp, cur_end + 1, n));
                    } else {
                        auto a1 = subrange(A_P, 0, m, 0, cur_1st);
                        auto a2 = subrange(A_P, 0, m, cur_end + 1, n);
                        y_bar = y - prod(a1, subrange(x_tmp, 0, cur_1st)) -
                                prod(a2, subrange(x_tmp, cur_end + 1, n));
                    }

                    //  Compute optimal solution
                    l = cur_end - cur_1st + 1;
                    A_T.resize(m, l, false);

                    for (i = cur_1st; i <= cur_end; i++) {
                        column(A_T, i - cur_1st) = column(A_P, i);
                    }

                    auto Q_j = subrange(Q_t, 0, m, cur_1st, cur_end + 1);
//                    helper::display<scalar, index>(Q_t, "Q_t");
//                    helper::display<scalar, index>(Q_j, "Q_j");
                    auto y_t = prod(trans(Q_j), y_bar);
                    auto R_j = subrange(R_t, 0, y_t.size(), cur_1st, cur_end + 1);
//                    helper::display<scalar, index>(R_j, "R_j");
//                    z.resize(l, false);
//                    CILS_Reduction<scalar, index> reduction(A_T, y_bar, 0, cils.upper);
//                    reduction.aip();
//                    helper::display<scalar, index>(reduction.R, "R_R");
//                    helper::display<scalar, index>(reduction.y, "y_R");
//                    helper::display<scalar, index>(y_t, "y_t");
//                    reduction.aspl_p_serial();

                    CILS_SECH_Search<scalar, index> ils(l, l, cils.qam);
//                    ils.obils_search(0, l, 1, reduction.R, reduction.y, z);
                    ils.obils_search(0, l, 1, R_j, y_t, z);

//                    subrange(x_tmp, cur_1st, cur_1st + l) = z;
//                    axpy_prod(reduction.P, z, x, true);
                    for (i = 0; i < l; i++) {
                        x_tmp[cur_1st + i] = z[i];
                    }

                }

                v_norm_cur = norm_2(y - prod(A_P, x_tmp));

                if (v_norm_cur < v_norm) {
                    // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
                    x_per.clear();
                    for (l = 0; l < n; l++) {
                        x_per[l] = x_tmp[l];
                    }
                    P_par.assign(P_tmp);
                    // 'SCP_Block_Optimal_2:88' v_norm_cur = v_norm_temp;
                    //  v_norm_cur = v_norm_temp;
                    // 'SCP_Block_Optimal_2:89' if per
                    best_per = per;
                    // 'SCP_Block_Optimal_2:94' if v_norm_cur <= cils.tolerance
                    if (v_norm_cur <= cils.tolerance) {
                        // 'SCP_Block_Optimal_2:95' stopping(2)=1;
                        stopping[1] = 1;
                    }
                    // 'SCP_Block_Optimal_2:99' v_norm = v_norm_cur;
                    v_norm = v_norm_cur;
                }
                // If we don't decrease the residual, keep trying permutations
            }
            // 'SCP_Block_Optimal_2:44' P_tmp = eye(n);
            b_matrix P_cur(cils.I);

            //cils.I(:, permutation(:, best_per))
            if (best_per >= 0) {
                for (i = 0; i < n; i++) {
                    column(P_cur, i) = column(cils.I, cils.permutation[best_per][i] - 1);
                }

                P_tmp.assign(cils.I);
                axpy_prod(P_cur, P_par, P_tmp, true);
                axpy_prod(P_tmp, x_per, x_cur, true);
            }

            time = omp_get_wtime() - time;
            return {{}, time, v_norm_cur};
        }


//        returnType <scalar, index>
//        block_sic_babai(b_vector &x_cur, scalar v_norm_cur, index mode) {
//            b_vector x_tmp(x_cur), x_per(n, 0), y_bar(m, 0);
//            b_matrix A_T, A_P(A), P_tmp, P_par(cils.I), P_fin; //A could be permuted based on the init point
//
//            b_vector stopping(3, 0);
//
//            b_vector z(n, 0);
//            scalar time = omp_get_wtime();
//
//            index b_count = 0;
//
//            if (v_norm_cur <= cils.tolerance) {
//                stopping[0] = 1;
//                time = omp_get_wtime() - time;
//                return {{}, time, v_norm_cur};
//            }
//
//            index cur_1st, cur_end, i, j, l, k1, per = -1, best_per = -1, iter = 0, p;
//            scalar v_norm = v_norm_cur;
//            P_tmp.resize(A_P.size2(), A_P.size2(), false);
//
//            time = omp_get_wtime();
//            for (index itr = 0; itr < cils.search_iter; itr++) {
//                iter = itr; //mode ? rand() % cils.search_iter : itr;
//
//                A_P.clear();
//                x_tmp.clear();
//                for (i = 0; i < n; i++) {
//                    p = cils.permutation[iter][i] - 1;
//                    column(A_P, i) = column(A, p);
//                    x_tmp[i] = x_per[p];
//                }
//
//                // 'SCP_Block_Optimal_2:104' per = true;
//                P_tmp.clear();
//                P_tmp.assign(cils.I);
//                pd(A_P, P_tmp, t);
//                prod(P_tmp, x_tmp);
//                per = iter;
//
//                // 'SCP_Block_Optimal_2:56' for j = 1:t.size2()
//                for (j = 0; j < t.size2(); j++) {
//                    cur_1st = t(0, j);
//                    cur_end = t(1, j);
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
//                    l = cur_end - cur_1st + 1;
//                    A_T.resize(m, l, false);
//
//                    for (i = cur_1st; i <= cur_end; i++) {
//                        column(A_T, i - cur_1st) = column(A_P, i);
//                    }
//
//                    z.resize(l, false);
//                    CILS_Reduction<scalar, index> reduction(A_T, y_bar, 0, cils.upper);
//                    reduction.aip();
//
//                    //  Find the Babai poindex
//                    for (k1 = 0; k1 < l; k1++) {
//                        index c_i = l - k1, r_kk = (c_i + m * (cur_1st - 2 + c_i)) - 1;
//                        scalar x_est;
//                        // 'Babai_Gen:50' if i==t
//                        if (c_i == l) {
//                            x_est = reduction.y[c_i - 1] / reduction.R[r_kk];
//                        } else {
//                            x_est = 0.0;
//                            for (i = 0; i < l - c_i; i++) {
//                                x_est += reduction.R[c_i + m * (cur_1st + c_i + i - 1) - 1] * x_tmp[(c_i + cur_1st + i) - 1];
//                            }
//                            x_est = (y[c_i - 1] - x_est) / reduction.R[r_kk];
//                        }
//                        x_est = std::round(x_est);
//                        if (x_est < 0.0) {
//                            x_est = 0.0;
//                        } else if (x_est > cils.upper) {
//                            x_est = cils.upper;
//                        }
//                        x_tmp[(c_i + cur_end - l) - 1] = x_est;
//                    }
//
//                }
//
//                if (cils.verbose) {
//                    helper::display<scalar, index>(n, x_tmp.data(), "x_tmp_" + std::to_string(iter));
//                }
//                v_norm_cur = norm_2(y - prod(A_P, x_tmp));
//
//                if (v_norm_cur < v_norm) {
//                    // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
//                    x_per.clear();
//                    for (l = 0; l < n; l++) {
//                        x_per[l] = x_tmp[l];
//                    }
//                    P_par.assign(P_tmp);
//                    // 'SCP_Block_Optimal_2:88' v_norm_cur = v_norm_temp;
//                    //  v_norm_cur = v_norm_temp;
//                    // 'SCP_Block_Optimal_2:89' if per
//                    best_per = per;
//                    // 'SCP_Block_Optimal_2:94' if v_norm_cur <= cils.tolerance
//                    if (v_norm_cur <= cils.tolerance) {
//                        // 'SCP_Block_Optimal_2:95' stopping(2)=1;
//                        stopping[1] = 1;
//                    }
//                    // 'SCP_Block_Optimal_2:99' v_norm = v_norm_cur;
//                    v_norm = v_norm_cur;
//                }
//                // If we don't decrease the residual, keep trying permutations
//            }
//            // 'SCP_Block_Optimal_2:44' P_tmp = eye(n);
//            b_matrix P_cur(cils.I);
//
//            if (best_per >= 0) {
//                for (i = 0; i < n; i++) {
//                    column(P_cur, i) = column(cils.I, cils.permutation[best_per][i] - 1);
//                }
//
//                P_tmp.assign(cils.I);
//                axpy_prod(P_cur, P_par, P_tmp, true);
//                axpy_prod(P_tmp, x_per, x_cur, true);
//            }
//
//            time = omp_get_wtime() - time;
//            return {{}, time, v_norm_cur};
//        }
//
//
//        returnType <scalar, index>
//        block_sic_babai_omp(b_vector &x_cur, scalar v_norm_cur,
//                                 index n_proc, index mode) {
//            b_vector R_t(m * n, 0), Q_t(m * n, 0), Piv_cum(n * n, 0);
//            helper::eye(n, Piv_cum.data());
//
//            // 'SCP_Block_Optimal_2:24' stopping=zeros(1,3);
//            b_vector stopping(3, 0);
//            scalar v_norm = v_norm_cur;
//            // 'SCP_Block_Optimal_2:50' per = false;
//            std::vector<b_vector> x(n_proc, b_vector(x_cur)), best_piv(n_proc, b_vector(Piv_cum));
//            b_vector v_norm_cur_proc(n_proc, v_norm);
//            si_vector best_per_proc(n_proc, 0);
//
//            scalar time = omp_get_wtime();
//            //  Subfunctions: SCP_opt
//
//            // 'SCP_Block_Optimal_2:25' b_count = 0;
//            index b_count = 0;
//            index best_proc = -1;
//            // 'SCP_Block_Optimal_2:27' if v_norm_cur <= cils.tolerance
//            if (v_norm_cur <= cils.tolerance) {
//                // 'SCP_Block_Optimal_2:28' stopping(1)=1;
//                stopping[0] = 1;
//                time = omp_get_wtime() - time;
//                return {stopping, time, v_norm_cur};
//            }
//
//            index cur_1st, cur_end, t, i, j, k1, best_per = -1;
//            b_vector z, z_z; //z_z = Z * z;
//            b_vector A_P(m * n, 0), x_tmp(x_cur), x_per(x_cur), y_bar(m, 0), y(m, 0);
//
//            time = omp_get_wtime();
//#pragma omp parallel default(shared) num_threads(n_proc) private(z, z_z, cur_1st, cur_end, i, j, t, k1) firstprivate(R_t, Q_t, Piv_cum, A_P, x_tmp, v_norm, y_bar, y)
//            {
//                index t_num = omp_get_thread_num(), per = -1;
//                index start = omp_get_thread_num() * cils.search_iter / n_proc;
//                index end = (omp_get_thread_num() + 1) * cils.search_iter / n_proc;
//
//                for (index itr = start; itr < end; itr++) {
//                    index iter = mode ? rand()// cils.search_iter : itr;
//                    // H_Apply cils.permutation strategy to update x_cur and v_norm_cur
//                    // [x_tmp, v_norm_temp] = block_opt(A_P, y, x_tmp, n, cils.indicator);
//                    // Corresponds to H_Algorithm 12 (Block Optimal) in Report 10
//                    for (i = 0; i < n; i++) {
//                        for (j = 0; j < m; j++) {
//                            A_P[j + m * i] = H[j + m * (cils.permutation[iter][i] - 1)];
//                        }
//                        x_tmp[i] = x_per[cils.permutation[iter][i] - 1];
//                    }
//
//                    R_t.assign(m * n, 0);
//                    Q_t.assign(m * n, 0);
//                    Piv_cum.assign(n * n, 0);
//                    pd(x_tmp.data(), Q_t.data(), R_t.data(), A_P.data(), Piv_cum.data());
//
//                    per = iter;
//                    // 'SCP_Block_Optimal_2:56' for j = 1:t.size2()
//                    for (j = 0; j < t.size2(); j++) {
//                        // 'Babai_Gen:27' R = R_t(:, cils.indicator(1, j):cils.indicator(2, j));
//                        cur_1st = cils.indicator[2 * j];
//                        cur_end = cils.indicator[2 * j + 1];
//                        //  Compute y_bar
//                        // 'Babai_Gen:39' if lastCol == n
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
//
//                        // 'Babai_Gen:46' y_bar = Q' * y_bar;
//                        if (cils.verbose)
//                            helper::display(m, y_bar.data(), "y_bar");
//
//                        // 'Babai_Gen:46' y_bar = Q' * y_bar;
//                        // Caution: Only use 1,t entry of the vector y.
//                        t = cur_1st > cur_end ? 0 : cur_end - cur_1st + 1;
//                        y.clear();
//                        for (k1 = 0; k1 < m; k1++) {
//                            for (i = 0; i < t; i++) {
//                                y[i] += Q_t[k1 + m * (cur_1st - 1 + i + k1 / m)] * y_bar[k1];
//                            }
//                        }
//
//                        if (cils.verbose)
//                            helper::display(t, y.data(), "y");
//
//                        for (index p = 0; p < 6; p++) {
//#pragma omp for schedule(dynamic) nowait
//                            for (k1 = 0; k1 < t; k1++) {
//                                index c_i = t - k1, r_kk = (c_i + m * (cur_1st - 2 + c_i)) - 1;
//                                scalar x_est;
//                                // 'Babai_Gen:50' if i==t
//                                if (c_i == t) {
//                                    x_est = y[c_i - 1] / R_t[r_kk];
//                                } else {
//                                    x_est = 0.0;
//                                    for (i = 0; i < t - c_i; i++) {
//                                        x_est +=
//                                                R_t[c_i + m * (cur_1st + c_i + i - 1) - 1] *
//                                                x_tmp[(c_i + cur_1st + i) - 1];
//                                    }
//                                    x_est = (y[c_i - 1] - x_est) / R_t[r_kk];
//                                }
//                                x_est = std::round(x_est);
//                                if (x_est < 0.0) {
//                                    x_est = 0.0;
//                                } else if (x_est > cils.upper) {
//                                    x_est = cils.upper;
//                                }
//                                x_tmp[(c_i + cur_end - t) - 1] = x_est;
//                            }
//                        }
//                        if (cils.verbose)
//                            helper::display(n, x_tmp.data(), "x_tmp");
//                    }
//
//                    if (cils.verbose) {
//                        helper::display<scalar, index>(n, x_tmp.data(), "x_tmp_" + std::to_string(iter));
//                    }
//                    // x_tmp
//                    // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(y - A_P * x_tmp);
//                    v_norm_cur = helper::find_residual(m, n, A_P.data(), x_tmp.data(), y.data());
//                    // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
//                    //            prindexf("v_t:%8.5f, v_n:%8.5f\n", v_norm_temp, v_norm);
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
//                        best_piv[t_num].assign(Piv_cum.begin(), Piv_cum.end());
//                        per = -1;
//                        // 'SCP_Block_Optimal_2:94' if v_norm_cur <= cils.tolerance
//                        if (v_norm_cur_proc[t_num] <= cils.tolerance) {
//                            // 'SCP_Block_Optimal_2:95' stopping(2)=1;
//                            stopping[1] = 1;
//                            iter = cils.search_iter;
//                            best_proc = t_num;
//                        }
//                        v_norm = v_norm_cur_proc[t_num];
//                    }
//                    if (stopping[1])
//                        iter = cils.search_iter;
//                    // If we don't decrease the residual, keep trying permutations
//                }
//
//            }
//
//            // 'SCP_Block_Optimal_2:44' P_tmp = P;
//            b_vector P_cur(Piv_cum), P_tmp(Piv_cum);
//
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
//                    P_cur[i + n * k1] = P_tmp[i + n * (cils.permutation[best_per_proc[best_proc]][k1] - 1)];
//                }
//            }
//
//            // 'SCP_Block_Optimal_2:115' x_cur = P_cur * x_cur;
//            b_vector x_a(n, 0), x_b(n, 0);
//            helper::mtimes_Axy<scalar, index>(n, n, best_piv[best_proc].data(), x[best_proc].data(), x_a.data());
//
//            helper::mtimes_Axy<scalar, index>(n, n, P_cur.data(), x_a.data(), x_b.data());
//
//            for (i = 0; i < n; i++) {
//                x_cur[i] = x_b[i];
//            }
//
//            time = omp_get_wtime() - time;
//            return {stopping, time, v_norm_cur};
//        }
//
//
//        returnType <scalar, index>
//        block_sic_optimal_omp(b_vector &x_cur, scalar v_norm_cur,
//                                   index n_proc, index mode) {
//            // 'SCP_Block_Optimal_2:24' stopping=zeros(1,3);
//            b_vector stopping(3, 0);
//
//            scalar time = omp_get_wtime();
//            //  Subfunctions: SCP_opt
//
//            // 'SCP_Block_Optimal_2:25' b_count = 0;
//            index b_count = 0;
//            index best_proc = -1;
//            // 'SCP_Block_Optimal_2:27' if v_norm_cur <= cils.tolerance
//            if (v_norm_cur <= cils.tolerance) {
//                // 'SCP_Block_Optimal_2:28' stopping(1)=1;
//                stopping[0] = 1;
//                time = omp_get_wtime() - time;
//                return {{}, time, v_norm_cur};
//            }
//            index cur_1st, cur_end, i, t, k1;
//            // 'SCP_Block_Optimal_2:32' t.size2() = ceil(n/m);
//            // 'SCP_Block_Optimal_2:33' cils.indicator = zeros(2, t.size2());
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
//                index start = omp_get_thread_num() * cils.search_iter / n_proc;
//                index end = (omp_get_thread_num() + 1) * cils.search_iter / n_proc;
//
//                for (index itr = start; itr < end; itr++) {
//                    index iter = mode ? rand()// cils.search_iter : itr;
//                    // H_Apply cils.permutation strategy to update x_cur and v_norm_cur
//                    // [x_tmp, v_norm_temp] = block_opt(A_P, y, x_tmp, n, cils.indicator);
//                    // Corresponds to H_Algorithm 12 (Block Optimal) in Report 10
//                    // 'SCP_Block_Optimal_2:56' for j = 1:t.size2()
//                    for (k1 = 0; k1 < n; k1++) {
//                        for (i = 0; i < m; i++) {
//                            A_P[i + m * k1] = H[i + m * (cils.permutation[iter][k1] - 1)];
//                        }
//                        x_tmp[k1] = x[t_num][cils.permutation[iter][k1] - 1];
//                    }
//                    // 'SCP_Block_Optimal_2:104' per = true;
//                    per = iter;
//
//                    for (index j = 0; j < t.size2(); j++) {
//                        cur_1st = cils.indicator[2 * j];
//                        cur_end = cils.indicator[2 * j + 1];
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
//                        CILS_Reduction<scalar, index> reduction(m, t, 0, cils.upper, 0, 0);
//                        reduction.aip(A_T, y_bar);
//
//                        CILS_SECH_Search<scalar, index> ils(t, t, cils.qam);
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
//                        // 'SCP_Block_Optimal_2:94' if v_norm_cur <= cils.tolerance
//                        if (v_norm_cur_proc[t_num] <= cils.tolerance) {
//                            // 'SCP_Block_Optimal_2:95' stopping(2)=1;
//                            stopping[1] = 1;
//                            iter = cils.search_iter;
//                            best_proc = t_num;
//                        }
//                        v_norm = v_norm_cur_proc[t_num];
//                    }
//                    if (stopping[1])
//                        iter = cils.search_iter;
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
//                    P_cur[i + n * k1] = P_tmp[i + n * (cils.permutation[best_per_proc[best_proc]][k1] - 1)];
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
//
//        returnType <scalar, index>
//        block_sic_suboptimal_omp(b_vector &x_cur, scalar v_norm_cur,
//                                      index n_proc, index mode) {
//            // 'SCP_Block_Optimal_2:24' stopping=zeros(1,3);
//            b_vector stopping(3, 0);
//
//            scalar time = omp_get_wtime();
//            //  Subfunctions: SCP_opt
//
//            // 'SCP_Block_Optimal_2:25' b_count = 0;
//            index b_count = 0;
//            index best_proc = -1;
//            // 'SCP_Block_Optimal_2:27' if v_norm_cur <= cils.tolerance
//            if (v_norm_cur <= cils.tolerance) {
//                // 'SCP_Block_Optimal_2:28' stopping(1)=1;
//                stopping[0] = 1;
//                time = omp_get_wtime() - time;
//                return {{}, time, v_norm_cur};
//            }
//            index cur_1st, cur_end, i, t, k1, n_dx_q_2, n_dx_q_0, check = 0;
//            index diff = 0, num_iter = 0, flag = 0, temp;
//
//            // 'SCP_Block_Optimal_2:32' t.size2() = ceil(n/m);
//            // 'SCP_Block_Optimal_2:33' cils.indicator = zeros(2, t.size2());
//
//            scalar v_norm = v_norm_cur, sum;
//            // 'SCP_Block_Optimal_2:50' per = false;
//            std::vector<b_vector> x(n_proc, b_vector(x_cur));
//            b_vector v_norm_cur_proc(n_proc, v_norm), x_per(x_cur);
//            si_vector best_per_proc(n_proc, 0), R_S_1(n, 0);
//
//            // private variables
//            b_vector z(n, 0), z_z(n, 0); //z_z = Z * z;
//            b_vector A_T, A_P(m * n, 0), x_tmp(x_cur), y_bar(m, 0), y(m, 0);
//
//            time = omp_get_wtime();
//#pragma omp parallel default(shared) num_threads(n_proc) private(sum, n_dx_q_2, n_dx_q_0, A_T, cur_1st, cur_end, i, t, k1) firstprivate(check, z, z_z, A_P, x_tmp, v_norm, y_bar, y)
//            {
//                index t_num = omp_get_thread_num(), per = -1;
//                index start = omp_get_thread_num() * cils.search_iter / n_proc;
//                index end = (omp_get_thread_num() + 1) * cils.search_iter / n_proc;
//                //#pragma omp for schedule(static, chunk)
//                for (index itr = start; itr < end; itr++) {
////            for (index itr = 0; itr < cils.search_iter; itr++) {
//                    index iter = mode ? rand()// cils.search_iter : itr;
//                    // H_Apply cils.permutation strategy to update x_cur and v_norm_cur
//                    // [x_tmp, v_norm_temp] = block_opt(A_P, y, x_tmp, n, cils.indicator);
//                    // Corresponds to H_Algorithm 12 (Block Optimal) in Report 10
//                    // 'SCP_Block_Optimal_2:56' for j = 1:t.size2()
//                    for (k1 = 0; k1 < n; k1++) {
//                        for (i = 0; i < m; i++) {
//                            A_P[i + m * k1] = H[i + m * (cils.permutation[iter][k1] - 1)];
//                        }
//                        x_tmp[k1] = x[t_num][cils.permutation[iter][k1] - 1];
//                    }
//                    // 'SCP_Block_Optimal_2:104' per = true;
//                    per = iter;
//
//                    for (index j = 0; j < t.size2(); j++) {
//                        cur_1st = cils.indicator[2 * j];
//                        cur_end = cils.indicator[2 * j + 1];
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
//                        z.assign(t, 0);
//                        z_z.assign(t, 0);
//
//                        CILS_Reduction<scalar, index> reduction(m, t, 0, cils.upper, 0, 0);
//                        reduction.aip(A_T, y_bar);
//
//                        CILS_SECH_Search<scalar, index> ils(t, t, cils.qam);
//
//                        index ds = t / cils.block_size;
//                        if (t// cils.block_size != 0 || ds == 1) {
//                            ils.obils_search2(reduction.R_Q, reduction.y_q, z_z);
//                        } else {
//                            si_vector d(ds, cils.block_size);
//                            b_vector y_b(reduction.y_q);
//                            for (index s = d.size() - 2; s >= 0; s--) {
//                                d[s] += d[s + 1];
//                            }
//                            R_S_1.assign(n, 0);
//                            diff = flag = 0;
//
////                        #pragma omp barrier
//                            for (index jj = 0; jj < 10; jj++) {
//#pragma omp for schedule(dynamic) nowait
//                                for (index s = 0; s < ds; s++) {
////                                if (!flag) {
//                                    n_dx_q_2 = d[s];
//                                    n_dx_q_0 = s == ds - 1 ? 0 : d[s + 1];
//
//                                    for (index row = n_dx_q_0; row < n_dx_q_2; row++) {
//                                        sum = 0;
//                                        for (index col = n_dx_q_2; col < t; col++) {
//                                            sum += reduction.R_Q[col * t + row] * z_z[col];
//                                        }
//                                        y_b[row] = reduction.y_q[row] - sum;
//                                    }
//
//                                    R_S_1[s] = ils.obils_search(n_dx_q_0, n_dx_q_2, 1, reduction.R_Q.data(), y_b.data(),
//                                                                z_z);
//#pragma omp atomic
//                                    diff += R_S_1[s];
//                                    flag = ((diff) >= ds) && jj > 0;
////                                }
//                                }
//                            }
//                        }
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
//                        // 'SCP_Block_Optimal_2:94' if v_norm_cur <= cils.tolerance
//                        if (v_norm_cur_proc[t_num] <= cils.tolerance) {
//                            // 'SCP_Block_Optimal_2:95' stopping(2)=1;
//                            stopping[1] = 1;
//                            iter = cils.search_iter;
//                            best_proc = t_num;
//                        }
//                        v_norm = v_norm_cur_proc[t_num];
//                    }
//                    if (stopping[1])
//                        iter = cils.search_iter;
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
//            //        cout << best_proc << " " << stopping[1] << " ";
//            //        helper::display<scalar, index>(n_proc, v_norm_cur_proc.data(), "v_norm_cur_proc");
//            for (k1 = 0; k1 < n; k1++) {
//                for (i = 0; i < n; i++) {
//                    P_cur[i + n * k1] = P_tmp[i + n * (cils.permutation[best_per_proc[best_proc]][k1] - 1)];
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
//
        returnType <scalar, index>
        block_sic_optimal_mpi(b_vector &x_cur, scalar *v_norm_cur,
                                   index size, index rank) {

            // 'SCP_Block_Optimal_2:24' stopping=zeros(1,3);
            b_vector stopping(3, 0);

            scalar time = omp_get_wtime();
            //  Subfunctions: SCP_opt
            scalar flag = 0.0;
            // 'SCP_Block_Optimal_2:25' b_count = 0;
            // 'SCP_Block_Optimal_2:27' if v_norm_cur <= cils.tolerance
            if (v_norm_cur[0] <= cils.tolerance) {
                // 'SCP_Block_Optimal_2:28' stopping(1)=1;
                stopping[0] = 1;
                time = omp_get_wtime() - time;
                return {{}, time, v_norm_cur[0]};
            }
            index cur_1st, cur_end, i, t, k1, per = -1, best_per = -1;
            // 'SCP_Block_Optimal_2:32' t.size2() = ceil(n/m);
            // 'SCP_Block_Optimal_2:33' cils.indicator = zeros(2, t.size2());
            auto v_norm = (double *) malloc(2 * sizeof(double));
            auto v_norm_rank = (double *) malloc(2 * sizeof(double));
            v_norm_rank[0] = v_norm_cur[0];
            v_norm[0] = v_norm_cur[0];
            v_norm[1] = rank;

            b_vector x_per(x_cur), x(n, 0);
            if (cils.verbose) {
                cout << "here: " << rank * cils.search_iter / size << "," << (rank + 1) * cils.search_iter / size
                     << endl;
                if (rank != 0) {
                    helper::display<scalar, index>(m, n, H.data(), "H" + to_string(rank));
                    helper::display<scalar, index>(1, n, cils.permutation[0].data(), "Per" + to_string(rank));
                }
            }

//        index start = rank * cils.search_iter / size;
//        index end = (rank + 1) * cils.search_iter / size;
//        index slice = (end - start) / t.size2();
//        index iter = start;
//        b_matrix A_P(H);
            if (rank == 0)
                time = MPI_Wtime();
/*#pragma omp parallel default(shared) num_threads(t.size2()) private(v_norm_cur, cur_1st, cur_end, i, t, k1, per)
        {
            vector<scalar> A_T, x_tmp(x_cur), y_bar(m, 0), y(m, 0);
            vector<scalar> z, z_z; //z_z = Z * z;
            v_norm_cur = (double *) malloc(1 * sizeof(double));
            //v_norm[0] = v_norm_rank[0];

//            index t_num = omp_get_thread_num();
//            index t_start = start + slice * t_num;
//            index t_end = start + slice * (t_num + 1);
            while (iter < end) {
                if (omp_get_thread_num() == 0) {
                    for (k1 = 0; k1 < n; k1++) {
                        for (i = 0; i < m; i++) {
                            A_P[i + m * k1] = H[i + m * (cils.permutation[iter][k1] - 1)];
                        }
                        x_tmp[k1] = x_per[cils.permutation[iter][k1] - 1];
                    }
                    per = iter;
                }
#pragma omp barrier
                for (index inner = 0; inner < 4; inner++) {
#pragma omp for schedule(static, 1) nowait
                    for (index j = 0; j < t.size2(); j++) {
//                        if (rank == 0 && inner == 0)
//                            cout << omp_get_thread_num() << " " << endl;
                        cur_1st = cils.indicator[2 * j];
                        cur_end = cils.indicator[2 * j + 1];
                        y_bar.clear();
                        for (k1 = 0; k1 <= cur_1st - 2; k1++) {
                            for (i = 0; i < m; i++) {
                                t = k1 * m + i;
                                y_bar[i] += A_P[i + m * (t / m)] * x_tmp[k1];
                            }
                        }
                        for (k1 = 0; k1 < n - cur_end; k1++) {
                            for (i = 0; i < m; i++) {
                                t = k1 * m + i;
                                y_bar[i] += A_P[i + m * (cur_end + t / m)] * x_tmp[cur_end + k1];
                            }
                        }
                        for (t = 0; t < m; t++) {
                            y_bar[t] = y[t] - y_bar[t];
                        }
                        l = cur_end - cur_1st + 1;
                        A_T.resize(m * t);
                        for (k1 = 0; k1 < t; k1++) {
                            for (i = 0; i < m; i++) {
                                A_T[i + m * k1] = A_P[i + m * (cur_1st - 1 + k1)];//2.0 *
                            }
                        }
                        // 'SCP_Block_Optimal_2:80' z = obils(H_adj, y_bar, l, u);
                        z.resize(l);
                        z_z.resize(t);
                        z.assign(t, 0);
                        z_z.assign(t, 0);

                        CILS_Reduction<scalar, index> reduction(m, t, 0, cils.upper, 0, 0);
                        reduction.aip(A_T, y_bar);

                        CILS_SECH_Search<scalar, index> ils(t, t, cils.qam);
                        ils.obils_search2(reduction.R_Q, reduction.y_q, z_z);

                        // 'SCP_Block_Optimal_2:81' x_tmp(cur_1st:cur_end) = 2 * z + e_vec;
                        for (i = 0; i < t; i++) {
                            z[reduction.p[i] - 1] = z_z[i];
                            //reduction.y_q[i];//z_z[reduction.p[i] - 1];//2.0 * z_z[i] + 1;
                        }
                        for (i = 0; i < t; i++) {
                            x_tmp[cur_1st + i - 1] = z[i];
                            //reduction.y_q[i];//z_z[reduction.p[i] - 1];//2.0 * z_z[i] + 1;
                        }
                    }
                }

                if (omp_get_thread_num() == 0) {
                    iter++;
                    // x_tmp
                    // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(y - A_P * x_tmp);
                    v_norm_cur[0] = helper::find_residual<scalar, index>(m, n, A_P.data(), x_tmp.data(), y.data());
                    // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
                    if (v_norm_cur[0] < v_norm[0]) {
                        // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
                        for (l = 0; l < n; l++) {
                            x_per[t] = x_tmp[t];
                        }
                        // 'SCP_Block_Optimal_2:88' v_norm_cur = v_norm_temp;
                        // 'SCP_Block_Optimal_2:89' if per
                        best_per = per;
                        per = -1;
                        // 'SCP_Block_Optimal_2:94' if v_norm_cur <= cils.tolerance
                        if (v_norm_cur[0] <= cils.tolerance) {
                            // 'SCP_Block_Optimal_2:95' stopping(2)=1;
                            stopping[1] = 1;
                            iter = end;
                        }
                        v_norm[0] = v_norm_cur[0];
                    }
                }
#pragma omp barrier
                if (stopping[1])
                    iter = end;
            }

        }
//        */
//
//#pragma omp parallel default(shared) num_threads(t.size2()) private(cur_1st, cur_end, i, t, k1)
//            {
//                b_vector x_tmp(x_cur), y_bar(m, 0), y(m, 0);
//                b_matrix A_P(H);
//                b_vector A_T, z, z_z; //z_z = Z * z;
////#pragma omp for schedule(static, chunk)
//                index t_num = omp_get_thread_num(), per = -1;
//                index start = omp_get_thread_num() * cils.search_iter / t.size2();
//                index end = (omp_get_thread_num() + 1) * cils.search_iter / t.size2();
//                for (index iter = start; iter < end; iter++) {
//                    for (k1 = 0; k1 < n; k1++) {
//                        for (i = 0; i < m; i++) {
//                            A_P[i + m * k1] = H[i + m * (cils.permutation[iter][k1] - 1)];
//                        }
//                        x_tmp[k1] = x_per[cils.permutation[iter][k1] - 1];
//                    }
//                    per = iter;
//#pragma omp barrier
//                    for (index inner = 0; inner < 4; inner++) {
//#pragma omp for schedule(dynamic) nowait
//                        for (index j = 0; j < t.size2(); j++) {
//                            cur_1st = cils.indicator[2 * j];
//                            cur_end = cils.indicator[2 * j + 1];
//                            y_bar.clear();
//                            for (k1 = 0; k1 <= cur_1st - 2; k1++) {
//                                for (i = 0; i < m; i++) {
//                                    t = k1 * m + i;
//                                    y_bar[i] += A_P[i + m * (t / m)] * x_tmp[k1];
//                                }
//                            }
//                            for (k1 = 0; k1 < n - cur_end; k1++) {
//                                for (i = 0; i < m; i++) {
//                                    t = k1 * m + i;
//                                    y_bar[i] += A_P[i + m * (cur_end + t / m)] * x_tmp[cur_end + k1];
//                                }
//                            }
//                            for (t = 0; t < m; t++) {
//                                y_bar[t] = y[t] - y_bar[t];
//                            }
//                            l = cur_end - cur_1st + 1;
//                            A_T.resize(m * t);
//                            for (k1 = 0; k1 < t; k1++) {
//                                for (i = 0; i < m; i++) {
//                                    A_T[i + m * k1] = A_P[i + m * (cur_1st - 1 + k1)];//2.0 *
//                                }
//                            }
//                            // 'SCP_Block_Optimal_2:80' z = obils(H_adj, y_bar, l, u);
//                            z.resize(l);
//                            z_z.resize(t);
//                            z.assign(t, 0);
//                            z_z.assign(t, 0);
//
//                            CILS_Reduction<scalar, index> reduction(m, t, 0, cils.upper, 0, 0);
//                            reduction.aip(A_T, y_bar);
//
//                            CILS_SECH_Search<scalar, index> ils(t, t, cils.qam);
//                            ils.obils_search2(reduction.R_Q, reduction.y_q, z_z);
//
//                            // 'SCP_Block_Optimal_2:81' x_tmp(cur_1st:cur_end) = 2 * z + e_vec;
//                            for (i = 0; i < t; i++) {
//                                z[reduction.p[i] - 1] = z_z[i];
//                                //reduction.y_q[i];//z_z[reduction.p[i] - 1];//2.0 * z_z[i] + 1;
//                            }
//                            for (i = 0; i < t; i++) {
//                                x_tmp[cur_1st + i - 1] = z[i];
//                                //reduction.y_q[i];//z_z[reduction.p[i] - 1];//2.0 * z_z[i] + 1;
//                            }
//                        }
//                    }
//                    v_norm_cur[0] = helper::find_residual<scalar, index>(m, n, A_P.data(), x_tmp.data(),
//                                                                         y.data());
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
//                        // 'SCP_Block_Optimal_2:94' if v_norm_cur <= cils.tolerance
//                        if (v_norm_cur[0] <= cils.tolerance) {
//                            // 'SCP_Block_Optimal_2:95' stopping(2)=1;
//                            stopping[1] = 1;
//                        }
//                        v_norm[0] = v_norm_cur[0];
//                    }
//                    if (stopping[1]) {
//                        iter = cils.search_iter;
//                    }
//                }
//            }
//            // x_tmp
//            // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(y - A_P * x_tmp);
//
//            MPI_Barrier(MPI_COMM_WORLD);
//            MPI_Allreduce(v_norm, v_norm_rank, 1, MPI_2DOUBLE_PRECISION, MPI_MINLOC, MPI_COMM_WORLD);
//
//            if (rank == v_norm_rank[1]) {
//                b_vector P_cur(n * n, 0), P_tmp(n * n, 0);
//                // 'SCP_Block_Optimal_2:44' P_tmp = eye(n);
//                helper::eye<scalar, index>(n, P_tmp.data());
//                helper::eye<scalar, index>(n, P_cur.data());
//
//                if (best_per >= 0) {
//                    for (k1 = 0; k1 < n; k1++) {
//                        for (i = 0; i < n; i++) {
//                            P_cur[i + n * k1] = P_tmp[i + n * (cils.permutation[best_per][k1] - 1)];
//                        }
//                    }
//                }
//                //'SCP_Block_Optimal_2:115' x_cur = P_cur * x_cur;
//                helper::mtimes_Axy<scalar, index>(n, n, P_cur.data(), x_per.data(), x.data());
//                for (i = 0; i < n; i++) {
//                    x_cur[i] = x[i];
//                }
//            }
//            MPI_Barrier(MPI_COMM_WORLD);
//            MPI_Bcast(&x_cur[0], n, MPI_DOUBLE, v_norm_rank[1], MPI_COMM_WORLD);
//            if (rank == 0)
//                time = MPI_Wtime() - time;
//
//            if (cils.verbose)
//                helper::display(n, x.data(), "x:" + std::to_string(rank));
//
//            return {stopping, time, v_norm[0]};
//        }

    };
}