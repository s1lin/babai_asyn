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
#include "CILS_SECH_Search.cpp"

namespace cils {
    template<typename scalar, typename index>
    class CILS_Block_SIC {
    private:
        CILS<scalar, index> cils;

    public:
        b_vector z_hat;

        explicit CILS_Block_SIC(CILS<scalar, index> &cils) {
            this->cils = cils;
            this->z_hat.resize(cils.n);
            this->z_hat.clear();
        }

        returnType<scalar, index>
        cils_scp_block_optimal_serial(b_matrix H, b_vector &x_cur, scalar v_norm_cur, index mode) {
            b_vector x_tmp(x_cur), x_per(x_cur), y_bar(cils.m, 0), y(cils.m, 0);
            b_matrix H_A, H_P(H);
            cils.init_indicator();
            // 'SCP_Block_Optimal_2:24' stopping=zeros(1,3);
            b_vector stopping(3, 0);
            // 'SCP_Block_Optimal_2:44' P_tmp = eye(cils.n);

            b_vector z(cils.n, 0), z_z(cils.n, 0); //z_z = Z * z;
            scalar time = omp_get_wtime();

            // 'SCP_Block_Optimal_2:25' b_count = 0;
            index b_count = 0;
            // 'SCP_Block_Optimal_2:27' if v_norm_cur <= cils.tolerance
            if (v_norm_cur <= cils.tolerance) {
                // 'SCP_Block_Optimal_2:28' stopping(1)=1;
                stopping[0] = 1;
                time = omp_get_wtime() - time;
                return {{}, time, v_norm_cur};
            }
            index cur_1st, cur_end, i, t, j, k1, per = -1, best_per = -1, iter = 0;
            scalar v_norm = v_norm_cur;

            time = omp_get_wtime();
            for (index itr = 0; itr < cils.search_iter; itr++) {
                iter = itr; //mode ? rand() % cils.search_iter : itr;

                //H_P = H(:,permutation(:, i));
                //x_tmp = x_per(permutation(:, i));
                H_P.clear();
                x_tmp.clear();
                for (i = 0; i < cils.n; i++) {
                    column(H_P, i) = column(H, cils.permutation[iter][i] - 1);
                    x_tmp[i] = x_per[cils.permutation[iter][i] - 1];
                }
                // 'SCP_Block_Optimal_2:104' per = true;

                per = iter;

                // 'SCP_Block_Optimal_2:56' for j = 1:cils.q
                for (j = 0; j < cils.q; j++) {
                    // cur_1st refers to the column of H_C where the current block starts
                    // cur_end refers to the column of H_C where the current block ends
                    // 'SCP_Block_Optimal_2:60' cur_1st = cils.indicator(1, j);
                    cur_1st = cils.indicator[2 * j] - 1;
                    // 'SCP_Block_Optimal_2:61' cur_end = cils.indicator(2, j);
                    cur_end = cils.indicator[2 * j + 1] - 1;

                    y_bar.clear();

                    if (cur_end == cils.n - 1) {
                        auto h1 = subrange(H_P, 0, cils.m, 0, cur_1st);
                        y_bar = cils.y_a - prod(h1, subrange(x_tmp, 0, cur_1st));
                    } else if (cur_1st == 0) {
                        auto h2 = subrange(H_P, 0, cils.m, cur_end + 1, cils.n);
                        y_bar = cils.y_a - prod(h2, subrange(x_tmp, cur_end + 1, cils.n));
                    } else {
                        auto h1 = subrange(H_P, 0, cils.m, 0, cur_1st);
                        auto h2 = subrange(H_P, 0, cils.m, cur_end + 1, cils.n);
                        y_bar = cils.y_a - prod(h1, subrange(x_tmp, 0, cur_1st)) -
                                prod(h2, subrange(x_tmp, cur_end + 1, cils.n));
                    }

                    //  Compute optimal solution
                    t = cur_end - cur_1st + 1;
                    H_A.resize(cils.m, t, false);

                    for (i = cur_1st; i <= cur_end; i++) {
                        column(H_A, i - cur_1st) = column(H_P, i);
                    }

                    // 'SCP_Block_Optimal_2:80' z = obils(H_adj, y_bar, l, u);
                    z.resize(t);
                    z.clear();

                    CILS_Reduction<scalar, index> reduction(H_A, y_bar, 0, cils.upper);
                    reduction.cils_aip_reduction();

                    CILS_SECH_Search<scalar, index> ils(t, t, cils.qam);
                    ils.obils_search(0, t, 1, reduction.R_Q, reduction.y_q, z);
                    z = prod(reduction.P, z);
//                    helper::display_vector<scalar, index>(z.size(), z.data(), "z");
                    for (i = 0; i < t; i++) {
                        x_tmp[cur_1st + i] = z[i];
                    }
                }
//                helper::display_vector<scalar, index>(x_tmp.size(), x_tmp.data(), "x_tmp");
                //v_norm_temp = norm(cils.y_a - H_P * x_tmp);
                v_norm_cur = norm_2(cils.y_a - prod(H_P, x_tmp));

                if (v_norm_cur < v_norm) {
                    // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
                    for (t = 0; t < cils.n; t++) {
                        x_per[t] = x_tmp[t];
                    }
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
            // 'SCP_Block_Optimal_2:44' P_tmp = eye(cils.n);
            b_matrix P_tmp(cils.I);

            //I(:, permutation(:, best_per))
            if (best_per >= 0) {
                for (i = 0; i < cils.n; i++) {
                    column(P_tmp, i) = column(cils.I, cils.permutation[best_per][i] - 1);
                }
            }
            // 'SCP_Block_Optimal_2:115' x_cur = P_cur * x_cur;
            x_cur = prod(P_tmp, x_per);

            time = omp_get_wtime() - time;
            return {{}, time, v_norm_cur};
        }


//        returnType <scalar, index>
//        cils_scp_block_babai_serial(b_vector &x_cur, scalar v_norm_cur, index mode) {
//            b_vector R_t(cils.m * cils.n, 0), Q_t(cils.m * cils.n, 0), Piv_cum(cils.n * cils.n, 0), best_piv(cils.n * cils.n, 0);
//            helper::eye(cils.n, best_piv.data());
////        cils.verbose = true;
//
//            // 'SCP_Block_Optimal_2:24' stopping=zeros(1,3);
//            b_vector y_bar(cils.m, 0), y(cils.m, 0);
//            b_vector stopping(3, 0);
//            b_vector H_P(cils.m * cils.n, 0);//, H_H(H);
//            b_vector z, z_z, x_tmp(cils.n, 0), x_per(x_cur); //z_z = Z * z;
//
//            if (cils.verbose) {
//                helper::display_matrix<scalar, index>(cils.m, cils.n, H.data(), "H");
//                helper::display_matrix<scalar, index>(cils.n, 1, x_tmp.data(), "x_tmp");
//            }
//
//            scalar time = omp_get_wtime();
//            //  Subfunctions: SCP_opt
//
//            // 'SCP_Block_Optimal_2:25' b_count = 0;
//            index b_count = 0;
//            // 'SCP_Block_Optimal_2:27' if v_norm_cur <= cils.tolerance
//            if (v_norm_cur <= cils.tolerance) {
//                // 'SCP_Block_Optimal_2:28' stopping(1)=1;
//                stopping[0] = 1;
//                time = omp_get_wtime() - time;
//                return {stopping, time, v_norm_cur};
//            }
//
//            index cur_1st, cur_end, t, i, j, k1, per = -1, best_per = -1;
//            scalar v_norm = v_norm_cur;
//            index iter = 0;
//            time = omp_get_wtime();
//
//            for (index itr = 0; itr < cils.search_iter; itr++) {
//                iter = mode ? rand() % cils.search_iter : itr;
//                H_P.assign(cils.m * cils.n, 0);
//                for (k1 = 0; k1 < cils.n; k1++) {
//                    for (j = 0; j < cils.m; j++) {
//                        H_P[j + cils.m * k1] = H[j + cils.m * (cils.permutation[iter][k1] - 1)];
//                    }
//                    x_tmp[k1] = x_per[cils.permutation[iter][k1] - 1];
//                }
//                if (cils.verbose) {
//                    helper::display_matrix<scalar, index>(cils.m, cils.n, H_P.data(), "H_P");
//                    helper::display_matrix<scalar, index>(cils.n, 1, x_tmp.data(), "x_tmp");
//                }
//
//                R_t.assign(cils.m * cils.n, 0);
//                Q_t.assign(cils.m * cils.n, 0);
//                Piv_cum.assign(cils.n * cils.n, 0);
//                cils_partition_deficient(x_tmp.data(), Q_t.data(), R_t.data(), H_P.data(), Piv_cum.data());
//
//                if (cils.verbose) {
//                    helper::display_matrix<scalar, index>(cils.m, cils.n, H_P.data(), "H_P");
//                    helper::display_matrix<scalar, index>(cils.m, cils.n, Q_t.data(), "Q_t");
//                    helper::display_matrix<scalar, index>(cils.m, cils.n, R_t.data(), "R_t");
//                    helper::display_matrix<scalar, index>(cils.n, cils.n, Piv_cum.data(), "Piv_cum");
//                    helper::display_matrix<scalar, index>(cils.n, 1, x_tmp.data(), "x_tmp");
//                }
//                // 'SCP_Block_Optimal_2:104' per = true;
//                per = iter;
//
//                // 'SCP_Block_Optimal_2:56' for j = 1:cils.q
//                for (j = 0; j < cils.q; j++) {
//
//                    // Get the QR factorization corresponding to block j
//                    cur_1st = cils.indicator[2 * j];
//                    cur_end = cils.indicator[2 * j + 1];
//
//                    //  Compute y_bar
//                    y_bar.clear();
//                    for (k1 = 0; k1 <= cur_1st - 2; k1++) {
//                        for (i = 0; i < cils.m; i++) {
//                            t = k1 * cils.m + i;
//                            y_bar[i] += H_P[i + cils.m * (t / cils.m)] * x_tmp[k1];
//                        }
//                    }
//                    for (k1 = 0; k1 < cils.n - cur_end; k1++) {
//                        for (i = 0; i < cils.m; i++) {
//                            t = k1 * cils.m + i;
//                            y_bar[i] += H_P[i + cils.m * (cur_end + t / cils.m)] * x_tmp[cur_end + k1];
//                        }
//                    }
//                    for (k1 = 0; k1 < cils.m; k1++) {
//                        y_bar[k1] = cils.y_a[k1] - y_bar[k1];
//                    }
//
//                    if (cils.verbose)
//                        helper::display_vector(cils.m, y_bar.data(), "y_bar");
//
//                    // 'Babai_Gen:46' y_bar = Q' * y_bar;
//                    // Caution: Only use 1,t entry of the vector y.
//                    t = cur_1st > cur_end ? 0 : cur_end - cur_1st + 1;
//                    y.clear();
//                    for (k1 = 0; k1 < cils.m; k1++) {
//                        for (i = 0; i < t; i++) {
//                            y[i] += Q_t[k1 + cils.m * (cur_1st - 1 + i + k1 / cils.m)] * y_bar[k1];
//                        }
//                    }
//
//                    if (cils.verbose)
//                        helper::display_vector(t, y.data(), "y");
//
//                    //  Find the Babai point
//                    for (k1 = 0; k1 < t; k1++) {
//                        index c_i = t - k1, r_kk = (c_i + cils.m * (cur_1st - 2 + c_i)) - 1;
//                        scalar x_est;
//                        // 'Babai_Gen:50' if i==t
//                        if (c_i == t) {
//                            x_est = y[c_i - 1] / R_t[r_kk];
//                        } else {
//                            x_est = 0.0;
//                            for (i = 0; i < t - c_i; i++) {
//                                x_est += R_t[c_i + cils.m * (cur_1st + c_i + i - 1) - 1] * x_tmp[(c_i + cur_1st + i) - 1];
//                            }
//                            x_est = (y[c_i - 1] - x_est) / R_t[r_kk];
//                        }
//                        x_est = std::round(x_est);
//                        if (x_est < 0.0) {
//                            x_est = 0.0;
//                        } else if (x_est > cils.upper) {
//                            x_est = cils.upper;
//                        }
//                        x_tmp[(c_i + cur_end - t) - 1] = x_est;
//                    }
//                    if (cils.verbose)
//                        helper::display_vector(cils.n, x_tmp.data(), "x_tmp");
//                }
//
//                if (cils.verbose) {
//                    helper::display_vector<scalar, index>(cils.n, x_tmp.data(), "x_tmp_" + std::to_string(iter));
//                }
//                // x_tmp
//                // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(cils.y_a - H_P * x_tmp);
//                v_norm_cur = helper::find_residual(cils.m, cils.n, H_P.data(), x_tmp.data(), cils.y_a.data());
//                // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
//                //            printf("v_t:%8.5f, v_n:%8.5f\cils.n", v_norm_temp, v_norm);
//                if (v_norm_cur < v_norm) {
//                    // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
//                    for (t = 0; t < cils.n; t++) {
//                        x_per[t] = x_tmp[t];
//                    }
//                    // 'SCP_Block_Optimal_2:88' v_norm_cur = v_norm_temp;
//                    //  v_norm_cur = v_norm_temp;
//                    // 'SCP_Block_Optimal_2:89' if per
//                    best_per = per;
//                    best_piv.assign(Piv_cum);
//                    // 'SCP_Block_Optimal_2:94' if v_norm_cur <= cils.tolerance
//                    if (v_norm_cur <= cils.tolerance) {
//                        // 'SCP_Block_Optimal_2:95' stopping(2)=1;
//                        stopping[1] = 1;
//                        if (cils.qam == 3)
//                            iter = cils.search_iter;
//                    }
//                    // 'SCP_Block_Optimal_2:99' v_norm = v_norm_cur;
//                    v_norm = v_norm_cur;
//                }
//                // If we don't decrease the residual, keep trying permutations
//            }
//
//            // 'SCP_Block_Optimal_2:44' P_cur = P;
//            b_vector P_cur(cils.n * cils.n, 0), P_tmp(cils.n * cils.n, 0);
//            helper::eye(cils.n, P_cur.data());
//            helper::eye(cils.n, P_tmp.data());
//            if (best_per >= 0) {
//                for (k1 = 0; k1 < cils.n; k1++) {
//                    for (i = 0; i < cils.n; i++) {
//                        P_cur[i + cils.n * k1] = P_tmp[i + cils.n * (cils.permutation[best_per][k1] - 1)];
//                    }
//                }
//            }
//
//            // 'SCP_Block_Optimal_2:115' x_cur = P_cur * x_cur;
//            if (cils.verbose)
//                helper::display_matrix<scalar, index>(cils.n, 1, x_per.data(), "x_per");
//
//            x_tmp.assign(cils.n, 0);
//            helper::mtimes_Axy<scalar, index>(cils.n, cils.n, best_piv.data(), x_per.data(), x_tmp.data());
//
//            x_per.assign(cils.n, 0);
//            helper::mtimes_Axy<scalar, index>(cils.n, cils.n, P_cur.data(), x_tmp.data(), x_per.data());
//
//            for (i = 0; i < cils.n; i++) {
//                x_cur[i] = x_per[i];
//            }
//
//            time = omp_get_wtime() - time;
//
//            return {stopping, time, v_norm_cur};
//        }
//
//
//        returnType <scalar, index>
//        cils_scp_block_babai_omp(b_vector &x_cur, scalar v_norm_cur,
//                                 index n_proc, index mode) {
//            b_vector R_t(cils.m * cils.n, 0), Q_t(cils.m * cils.n, 0), Piv_cum(cils.n * cils.n, 0);
//            helper::eye(cils.n, Piv_cum.data());
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
//            b_vector H_P(cils.m * cils.n, 0), x_tmp(x_cur), x_per(x_cur), y_bar(cils.m, 0), y(cils.m, 0);
//
//            time = omp_get_wtime();
//#pragma omp parallel default(shared) num_threads(n_proc) private(z, z_z, cur_1st, cur_end, i, j, t, k1) firstprivate(R_t, Q_t, Piv_cum, H_P, x_tmp, v_norm, y_bar, y)
//            {
//                index t_num = omp_get_thread_num(), per = -1;
//                index start = omp_get_thread_num() * cils.search_iter / n_proc;
//                index end = (omp_get_thread_num() + 1) * cils.search_iter / n_proc;
//
//                for (index itr = start; itr < end; itr++) {
//                    index iter = mode ? rand() % cils.search_iter : itr;
//                    // H_Apply cils.permutation strategy to update x_cur and v_norm_cur
//                    // [x_tmp, v_norm_temp] = block_opt(H_P, cils.y_a, x_tmp, cils.n, cils.indicator);
//                    // Corresponds to H_Algorithm 12 (Block Optimal) in Report 10
//                    for (i = 0; i < cils.n; i++) {
//                        for (j = 0; j < cils.m; j++) {
//                            H_P[j + cils.m * i] = H[j + cils.m * (cils.permutation[iter][i] - 1)];
//                        }
//                        x_tmp[i] = x_per[cils.permutation[iter][i] - 1];
//                    }
//
//                    R_t.assign(cils.m * cils.n, 0);
//                    Q_t.assign(cils.m * cils.n, 0);
//                    Piv_cum.assign(cils.n * cils.n, 0);
//                    cils_partition_deficient(x_tmp.data(), Q_t.data(), R_t.data(), H_P.data(), Piv_cum.data());
//
//                    per = iter;
//                    // 'SCP_Block_Optimal_2:56' for j = 1:cils.q
//                    for (j = 0; j < cils.q; j++) {
//                        // 'Babai_Gen:27' R = R_t(:, cils.indicator(1, j):cils.indicator(2, j));
//                        cur_1st = cils.indicator[2 * j];
//                        cur_end = cils.indicator[2 * j + 1];
//                        //  Compute y_bar
//                        // 'Babai_Gen:39' if lastCol == cils.n
//                        y_bar.clear();
//                        for (k1 = 0; k1 <= cur_1st - 2; k1++) {
//                            for (i = 0; i < cils.m; i++) {
//                                t = k1 * cils.m + i;
//                                y_bar[i] += H_P[i + cils.m * (t / cils.m)] * x_tmp[k1];
//                            }
//                        }
//                        for (k1 = 0; k1 < cils.n - cur_end; k1++) {
//                            for (i = 0; i < cils.m; i++) {
//                                t = k1 * cils.m + i;
//                                y_bar[i] += H_P[i + cils.m * (cur_end + t / cils.m)] * x_tmp[cur_end + k1];
//                            }
//                        }
//                        for (t = 0; t < cils.m; t++) {
//                            y_bar[t] = cils.y_a[t] - y_bar[t];
//                        }
//
//                        // 'Babai_Gen:46' y_bar = Q' * y_bar;
//                        if (cils.verbose)
//                            helper::display_vector(cils.m, y_bar.data(), "y_bar");
//
//                        // 'Babai_Gen:46' y_bar = Q' * y_bar;
//                        // Caution: Only use 1,t entry of the vector y.
//                        t = cur_1st > cur_end ? 0 : cur_end - cur_1st + 1;
//                        y.clear();
//                        for (k1 = 0; k1 < cils.m; k1++) {
//                            for (i = 0; i < t; i++) {
//                                y[i] += Q_t[k1 + cils.m * (cur_1st - 1 + i + k1 / cils.m)] * y_bar[k1];
//                            }
//                        }
//
//                        if (cils.verbose)
//                            helper::display_vector(t, y.data(), "y");
//
//                        for (index p = 0; p < 6; p++) {
//#pragma omp for schedule(dynamic) nowait
//                            for (k1 = 0; k1 < t; k1++) {
//                                index c_i = t - k1, r_kk = (c_i + cils.m * (cur_1st - 2 + c_i)) - 1;
//                                scalar x_est;
//                                // 'Babai_Gen:50' if i==t
//                                if (c_i == t) {
//                                    x_est = y[c_i - 1] / R_t[r_kk];
//                                } else {
//                                    x_est = 0.0;
//                                    for (i = 0; i < t - c_i; i++) {
//                                        x_est +=
//                                                R_t[c_i + cils.m * (cur_1st + c_i + i - 1) - 1] *
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
//                            helper::display_vector(cils.n, x_tmp.data(), "x_tmp");
//                    }
//
//                    if (cils.verbose) {
//                        helper::display_vector<scalar, index>(cils.n, x_tmp.data(), "x_tmp_" + std::to_string(iter));
//                    }
//                    // x_tmp
//                    // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(cils.y_a - H_P * x_tmp);
//                    v_norm_cur = helper::find_residual(cils.m, cils.n, H_P.data(), x_tmp.data(), cils.y_a.data());
//                    // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
//                    //            printf("v_t:%8.5f, v_n:%8.5f\cils.n", v_norm_temp, v_norm);
//                    // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(cils.y_a - H_P * x_tmp);
//                    v_norm_cur_proc[t_num] = helper::find_residual(cils.m, cils.n, H_P.data(), x_tmp.data(), cils.y_a.data());
//                    // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
//                    if (v_norm_cur_proc[t_num] < v_norm) {
//                        // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
//                        for (t = 0; t < cils.n; t++) {
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
//            for (k1 = 0; k1 < cils.n; k1++) {
//                for (i = 0; i < cils.n; i++) {
//                    P_cur[i + cils.n * k1] = P_tmp[i + cils.n * (cils.permutation[best_per_proc[best_proc]][k1] - 1)];
//                }
//            }
//
//            // 'SCP_Block_Optimal_2:115' x_cur = P_cur * x_cur;
//            b_vector x_a(cils.n, 0), x_b(cils.n, 0);
//            helper::mtimes_Axy<scalar, index>(cils.n, cils.n, best_piv[best_proc].data(), x[best_proc].data(), x_a.data());
//
//            helper::mtimes_Axy<scalar, index>(cils.n, cils.n, P_cur.data(), x_a.data(), x_b.data());
//
//            for (i = 0; i < cils.n; i++) {
//                x_cur[i] = x_b[i];
//            }
//
//            time = omp_get_wtime() - time;
//            return {stopping, time, v_norm_cur};
//        }
//
//
//        returnType <scalar, index>
//        cils_scp_block_optimal_omp(b_vector &x_cur, scalar v_norm_cur,
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
//            // 'SCP_Block_Optimal_2:32' cils.q = ceil(cils.n/cils.m);
//            // 'SCP_Block_Optimal_2:33' cils.indicator = zeros(2, cils.q);
//
//            scalar v_norm = v_norm_cur;
//            // 'SCP_Block_Optimal_2:50' per = false;
//            std::vector<b_vector> x(n_proc, b_vector(x_cur));
//            b_vector v_norm_cur_proc(n_proc, v_norm), x_per(x_cur);
//            si_vector best_per_proc(n_proc, 0);
//
//            // private variables
//            b_vector z, z_z; //z_z = Z * z;
//            b_vector H_A, H_P(cils.m * cils.n, 0), x_tmp(x_cur), y_bar(cils.m, 0), y(cils.m, 0);
//
//            time = omp_get_wtime();
//#pragma omp parallel default(shared) num_threads(n_proc) private(z, z_z, H_A, cur_1st, cur_end, i, t, k1) firstprivate(H_P, x_tmp, v_norm, y_bar, y)
//            {
//                index t_num = omp_get_thread_num(), per = -1;
//                index start = omp_get_thread_num() * cils.search_iter / n_proc;
//                index end = (omp_get_thread_num() + 1) * cils.search_iter / n_proc;
//
//                for (index itr = start; itr < end; itr++) {
//                    index iter = mode ? rand() % cils.search_iter : itr;
//                    // H_Apply cils.permutation strategy to update x_cur and v_norm_cur
//                    // [x_tmp, v_norm_temp] = block_opt(H_P, cils.y_a, x_tmp, cils.n, cils.indicator);
//                    // Corresponds to H_Algorithm 12 (Block Optimal) in Report 10
//                    // 'SCP_Block_Optimal_2:56' for j = 1:cils.q
//                    for (k1 = 0; k1 < cils.n; k1++) {
//                        for (i = 0; i < cils.m; i++) {
//                            H_P[i + cils.m * k1] = H[i + cils.m * (cils.permutation[iter][k1] - 1)];
//                        }
//                        x_tmp[k1] = x[t_num][cils.permutation[iter][k1] - 1];
//                    }
//                    // 'SCP_Block_Optimal_2:104' per = true;
//                    per = iter;
//
//                    for (index j = 0; j < cils.q; j++) {
//                        cur_1st = cils.indicator[2 * j];
//                        cur_end = cils.indicator[2 * j + 1];
//                        y_bar.clear();
//                        for (k1 = 0; k1 <= cur_1st - 2; k1++) {
//                            for (i = 0; i < cils.m; i++) {
//                                t = k1 * cils.m + i;
//                                y_bar[i] += H_P[i + cils.m * (t / cils.m)] * x_tmp[k1];
//                            }
//                        }
//                        for (k1 = 0; k1 < cils.n - cur_end; k1++) {
//                            for (i = 0; i < cils.m; i++) {
//                                t = k1 * cils.m + i;
//                                y_bar[i] += H_P[i + cils.m * (cur_end + t / cils.m)] * x_tmp[cur_end + k1];
//                            }
//                        }
//                        for (t = 0; t < cils.m; t++) {
//                            y_bar[t] = cils.y_a[t] - y_bar[t];
//                        }
//                        t = cur_end - cur_1st + 1;
//                        H_A.resize(cils.m * t);
//                        for (k1 = 0; k1 < t; k1++) {
//                            for (i = 0; i < cils.m; i++) {
//                                H_A[i + cils.m * k1] = H_P[i + cils.m * (cur_1st - 1 + k1)];
//                            }
//                        }
//                        // 'SCP_Block_Optimal_2:80' z = obils(H_adj, y_bar, l, u);
//                        z.resize(t);
//                        z_z.resize(t);
//                        z.assign(t, 0);
//                        z_z.assign(t, 0);
//
//                        CILS_Reduction<scalar, index> reduction(cils.m, t, 0, cils.upper, 0, 0);
//                        reduction.cils_aip_reduction(H_A, y_bar);
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
//                    // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(cils.y_a - H_P * x_tmp);
//                    v_norm_cur_proc[t_num] = helper::find_residual(cils.m, cils.n, H_P.data(), x_tmp.data(), cils.y_a.data());
//                    // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
//                    if (v_norm_cur_proc[t_num] < v_norm) {
//                        // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
//                        for (t = 0; t < cils.n; t++) {
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
//            b_vector P_cur(cils.n * cils.n, 0), P_tmp(cils.n * cils.n, 0);
//            // 'SCP_Block_Optimal_2:44' P_tmp = eye(cils.n);
//
//            b_vector x_a(cils.n, 0);
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
//            for (k1 = 0; k1 < cils.n; k1++) {
//                for (i = 0; i < cils.n; i++) {
//                    P_cur[i + cils.n * k1] = P_tmp[i + cils.n * (cils.permutation[best_per_proc[best_proc]][k1] - 1)];
//                }
//            }
//
//            // 'SCP_Block_Optimal_2:115' x_cur = P_cur * x_cur;
//            helper::mtimes_Axy<scalar, index>(cils.n, cils.n, P_cur.data(), x[best_proc].data(), x_a.data());
//            for (i = 0; i < cils.n; i++) {
//                x_cur[i] = x_a[i];
//            }
//
//            time = omp_get_wtime() - time;
//            return {stopping, time, v_norm};
//        }
//
//
//        returnType <scalar, index>
//        cils_scp_block_suboptimal_omp(b_vector &x_cur, scalar v_norm_cur,
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
//            // 'SCP_Block_Optimal_2:32' cils.q = ceil(cils.n/cils.m);
//            // 'SCP_Block_Optimal_2:33' cils.indicator = zeros(2, cils.q);
//
//            scalar v_norm = v_norm_cur, sum;
//            // 'SCP_Block_Optimal_2:50' per = false;
//            std::vector<b_vector> x(n_proc, b_vector(x_cur));
//            b_vector v_norm_cur_proc(n_proc, v_norm), x_per(x_cur);
//            si_vector best_per_proc(n_proc, 0), R_S_1(cils.n, 0);
//
//            // private variables
//            b_vector z(cils.n, 0), z_z(cils.n, 0); //z_z = Z * z;
//            b_vector H_A, H_P(cils.m * cils.n, 0), x_tmp(x_cur), y_bar(cils.m, 0), y(cils.m, 0);
//
//            time = omp_get_wtime();
//#pragma omp parallel default(shared) num_threads(n_proc) private(sum, n_dx_q_2, n_dx_q_0, H_A, cur_1st, cur_end, i, t, k1) firstprivate(check, z, z_z, H_P, x_tmp, v_norm, y_bar, y)
//            {
//                index t_num = omp_get_thread_num(), per = -1;
//                index start = omp_get_thread_num() * cils.search_iter / n_proc;
//                index end = (omp_get_thread_num() + 1) * cils.search_iter / n_proc;
//                //#pragma omp for schedule(static, chunk)
//                for (index itr = start; itr < end; itr++) {
////            for (index itr = 0; itr < cils.search_iter; itr++) {
//                    index iter = mode ? rand() % cils.search_iter : itr;
//                    // H_Apply cils.permutation strategy to update x_cur and v_norm_cur
//                    // [x_tmp, v_norm_temp] = block_opt(H_P, cils.y_a, x_tmp, cils.n, cils.indicator);
//                    // Corresponds to H_Algorithm 12 (Block Optimal) in Report 10
//                    // 'SCP_Block_Optimal_2:56' for j = 1:cils.q
//                    for (k1 = 0; k1 < cils.n; k1++) {
//                        for (i = 0; i < cils.m; i++) {
//                            H_P[i + cils.m * k1] = H[i + cils.m * (cils.permutation[iter][k1] - 1)];
//                        }
//                        x_tmp[k1] = x[t_num][cils.permutation[iter][k1] - 1];
//                    }
//                    // 'SCP_Block_Optimal_2:104' per = true;
//                    per = iter;
//
//                    for (index j = 0; j < cils.q; j++) {
//                        cur_1st = cils.indicator[2 * j];
//                        cur_end = cils.indicator[2 * j + 1];
//                        y_bar.clear();
//                        for (k1 = 0; k1 <= cur_1st - 2; k1++) {
//                            for (i = 0; i < cils.m; i++) {
//                                t = k1 * cils.m + i;
//                                y_bar[i] += H_P[i + cils.m * (t / cils.m)] * x_tmp[k1];
//                            }
//                        }
//                        for (k1 = 0; k1 < cils.n - cur_end; k1++) {
//                            for (i = 0; i < cils.m; i++) {
//                                t = k1 * cils.m + i;
//                                y_bar[i] += H_P[i + cils.m * (cur_end + t / cils.m)] * x_tmp[cur_end + k1];
//                            }
//                        }
//                        for (t = 0; t < cils.m; t++) {
//                            y_bar[t] = cils.y_a[t] - y_bar[t];
//                        }
//                        t = cur_end - cur_1st + 1;
//                        H_A.resize(cils.m * t);
//                        for (k1 = 0; k1 < t; k1++) {
//                            for (i = 0; i < cils.m; i++) {
//                                H_A[i + cils.m * k1] = H_P[i + cils.m * (cur_1st - 1 + k1)];
//                            }
//                        }
//                        // 'SCP_Block_Optimal_2:80' z = obils(H_adj, y_bar, l, u);
//                        z.assign(t, 0);
//                        z_z.assign(t, 0);
//
//                        CILS_Reduction<scalar, index> reduction(cils.m, t, 0, cils.upper, 0, 0);
//                        reduction.cils_aip_reduction(H_A, y_bar);
//
//                        CILS_SECH_Search<scalar, index> ils(t, t, cils.qam);
//
//                        index ds = t / cils.block_size;
//                        if (t % cils.block_size != 0 || ds == 1) {
//                            ils.obils_search2(reduction.R_Q, reduction.y_q, z_z);
//                        } else {
//                            si_vector d(ds, cils.block_size);
//                            b_vector y_b(reduction.y_q);
//                            for (index s = d.size() - 2; s >= 0; s--) {
//                                d[s] += d[s + 1];
//                            }
//                            R_S_1.assign(cils.n, 0);
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
//                    // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(cils.y_a - H_P * x_tmp);
//                    v_norm_cur_proc[t_num] = helper::find_residual(cils.m, cils.n, H_P.data(), x_tmp.data(), cils.y_a.data());
//                    // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
//                    if (v_norm_cur_proc[t_num] < v_norm) {
//                        // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
//                        for (t = 0; t < cils.n; t++) {
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
//            b_vector P_cur(cils.n * cils.n, 0), P_tmp(cils.n * cils.n, 0);
//            // 'SCP_Block_Optimal_2:44' P_tmp = eye(cils.n);
//
//            b_vector x_a(cils.n, 0);
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
//            //        helper::display_vector<scalar, index>(n_proc, v_norm_cur_proc.data(), "v_norm_cur_proc");
//            for (k1 = 0; k1 < cils.n; k1++) {
//                for (i = 0; i < cils.n; i++) {
//                    P_cur[i + cils.n * k1] = P_tmp[i + cils.n * (cils.permutation[best_per_proc[best_proc]][k1] - 1)];
//                }
//            }
//
//            // 'SCP_Block_Optimal_2:115' x_cur = P_cur * x_cur;
//            helper::mtimes_Axy<scalar, index>(cils.n, cils.n, P_cur.data(), x[best_proc].data(), x_a.data());
//            for (i = 0; i < cils.n; i++) {
//                x_cur[i] = x_a[i];
//            }
//
//            time = omp_get_wtime() - time;
//            return {stopping, time, v_norm};
//        }
//
//
//        returnType <scalar, index>
//        cils_scp_block_optimal_mpi(b_vector &x_cur, scalar *v_norm_cur,
//                                   index size, index rank) {
//
//            // 'SCP_Block_Optimal_2:24' stopping=zeros(1,3);
//            b_vector stopping(3, 0);
//
//            scalar time = omp_get_wtime();
//            //  Subfunctions: SCP_opt
//            scalar flag = 0.0;
//            // 'SCP_Block_Optimal_2:25' b_count = 0;
//            // 'SCP_Block_Optimal_2:27' if v_norm_cur <= cils.tolerance
//            if (v_norm_cur[0] <= cils.tolerance) {
//                // 'SCP_Block_Optimal_2:28' stopping(1)=1;
//                stopping[0] = 1;
//                time = omp_get_wtime() - time;
//                return {{}, time, v_norm_cur[0]};
//            }
//            index cur_1st, cur_end, i, t, k1, per = -1, best_per = -1;
//            // 'SCP_Block_Optimal_2:32' cils.q = ceil(cils.n/cils.m);
//            // 'SCP_Block_Optimal_2:33' cils.indicator = zeros(2, cils.q);
//            auto v_norm = (double *) malloc(2 * sizeof(double));
//            auto v_norm_rank = (double *) malloc(2 * sizeof(double));
//            v_norm_rank[0] = v_norm_cur[0];
//            v_norm[0] = v_norm_cur[0];
//            v_norm[1] = rank;
//
//            b_vector x_per(x_cur), x(cils.n, 0);
//            if (cils.verbose) {
//                cout << "here: " << rank * cils.search_iter / size << "," << (rank + 1) * cils.search_iter / size
//                     << endl;
//                if (rank != 0) {
//                    helper::display_matrix<scalar, index>(cils.m, cils.n, H.data(), "H" + to_string(rank));
//                    helper::display_matrix<scalar, index>(1, cils.n, cils.permutation[0].data(), "Per" + to_string(rank));
//                }
//            }
//
////        index start = rank * cils.search_iter / size;
////        index end = (rank + 1) * cils.search_iter / size;
////        index slice = (end - start) / cils.q;
////        index iter = start;
////        b_matrix H_P(H);
//            if (rank == 0)
//                time = MPI_Wtime();
///*#pragma omp parallel default(shared) num_threads(cils.q) private(v_norm_cur, cur_1st, cur_end, i, t, k1, per)
//        {
//            vector<scalar> H_A, x_tmp(x_cur), y_bar(cils.m, 0), y(cils.m, 0);
//            vector<scalar> z, z_z; //z_z = Z * z;
//            v_norm_cur = (double *) malloc(1 * sizeof(double));
//            //v_norm[0] = v_norm_rank[0];
//
////            index t_num = omp_get_thread_num();
////            index t_start = start + slice * t_num;
////            index t_end = start + slice * (t_num + 1);
//            while (iter < end) {
//                if (omp_get_thread_num() == 0) {
//                    for (k1 = 0; k1 < cils.n; k1++) {
//                        for (i = 0; i < cils.m; i++) {
//                            H_P[i + cils.m * k1] = H[i + cils.m * (cils.permutation[iter][k1] - 1)];
//                        }
//                        x_tmp[k1] = x_per[cils.permutation[iter][k1] - 1];
//                    }
//                    per = iter;
//                }
//#pragma omp barrier
//                for (index inner = 0; inner < 4; inner++) {
//#pragma omp for schedule(static, 1) nowait
//                    for (index j = 0; j < cils.q; j++) {
////                        if (rank == 0 && inner == 0)
////                            cout << omp_get_thread_num() << " " << endl;
//                        cur_1st = cils.indicator[2 * j];
//                        cur_end = cils.indicator[2 * j + 1];
//                        y_bar.clear();
//                        for (k1 = 0; k1 <= cur_1st - 2; k1++) {
//                            for (i = 0; i < cils.m; i++) {
//                                t = k1 * cils.m + i;
//                                y_bar[i] += H_P[i + cils.m * (t / cils.m)] * x_tmp[k1];
//                            }
//                        }
//                        for (k1 = 0; k1 < cils.n - cur_end; k1++) {
//                            for (i = 0; i < cils.m; i++) {
//                                t = k1 * cils.m + i;
//                                y_bar[i] += H_P[i + cils.m * (cur_end + t / cils.m)] * x_tmp[cur_end + k1];
//                            }
//                        }
//                        for (t = 0; t < cils.m; t++) {
//                            y_bar[t] = cils.y_a[t] - y_bar[t];
//                        }
//                        t = cur_end - cur_1st + 1;
//                        H_A.resize(cils.m * t);
//                        for (k1 = 0; k1 < t; k1++) {
//                            for (i = 0; i < cils.m; i++) {
//                                H_A[i + cils.m * k1] = H_P[i + cils.m * (cur_1st - 1 + k1)];//2.0 *
//                            }
//                        }
//                        // 'SCP_Block_Optimal_2:80' z = obils(H_adj, y_bar, l, u);
//                        z.resize(t);
//                        z_z.resize(t);
//                        z.assign(t, 0);
//                        z_z.assign(t, 0);
//
//                        CILS_Reduction<scalar, index> reduction(cils.m, t, 0, cils.upper, 0, 0);
//                        reduction.cils_aip_reduction(H_A, y_bar);
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
//                }
//
//                if (omp_get_thread_num() == 0) {
//                    iter++;
//                    // x_tmp
//                    // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(cils.y_a - H_P * x_tmp);
//                    v_norm_cur[0] = helper::find_residual<scalar, index>(cils.m, cils.n, H_P.data(), x_tmp.data(), cils.y_a.data());
//                    // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
//                    if (v_norm_cur[0] < v_norm[0]) {
//                        // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
//                        for (t = 0; t < cils.n; t++) {
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
//        */
//
//#pragma omp parallel default(shared) num_threads(cils.q) private(cur_1st, cur_end, i, t, k1)
//            {
//                b_vector x_tmp(x_cur), y_bar(cils.m, 0), y(cils.m, 0);
//                b_matrix H_P(H);
//                b_vector H_A, z, z_z; //z_z = Z * z;
////#pragma omp for schedule(static, chunk)
//                index t_num = omp_get_thread_num(), per = -1;
//                index start = omp_get_thread_num() * cils.search_iter / cils.q;
//                index end = (omp_get_thread_num() + 1) * cils.search_iter / cils.q;
//                for (index iter = start; iter < end; iter++) {
//                    for (k1 = 0; k1 < cils.n; k1++) {
//                        for (i = 0; i < cils.m; i++) {
//                            H_P[i + cils.m * k1] = H[i + cils.m * (cils.permutation[iter][k1] - 1)];
//                        }
//                        x_tmp[k1] = x_per[cils.permutation[iter][k1] - 1];
//                    }
//                    per = iter;
//#pragma omp barrier
//                    for (index inner = 0; inner < 4; inner++) {
//#pragma omp for schedule(dynamic) nowait
//                        for (index j = 0; j < cils.q; j++) {
//                            cur_1st = cils.indicator[2 * j];
//                            cur_end = cils.indicator[2 * j + 1];
//                            y_bar.clear();
//                            for (k1 = 0; k1 <= cur_1st - 2; k1++) {
//                                for (i = 0; i < cils.m; i++) {
//                                    t = k1 * cils.m + i;
//                                    y_bar[i] += H_P[i + cils.m * (t / cils.m)] * x_tmp[k1];
//                                }
//                            }
//                            for (k1 = 0; k1 < cils.n - cur_end; k1++) {
//                                for (i = 0; i < cils.m; i++) {
//                                    t = k1 * cils.m + i;
//                                    y_bar[i] += H_P[i + cils.m * (cur_end + t / cils.m)] * x_tmp[cur_end + k1];
//                                }
//                            }
//                            for (t = 0; t < cils.m; t++) {
//                                y_bar[t] = cils.y_a[t] - y_bar[t];
//                            }
//                            t = cur_end - cur_1st + 1;
//                            H_A.resize(cils.m * t);
//                            for (k1 = 0; k1 < t; k1++) {
//                                for (i = 0; i < cils.m; i++) {
//                                    H_A[i + cils.m * k1] = H_P[i + cils.m * (cur_1st - 1 + k1)];//2.0 *
//                                }
//                            }
//                            // 'SCP_Block_Optimal_2:80' z = obils(H_adj, y_bar, l, u);
//                            z.resize(t);
//                            z_z.resize(t);
//                            z.assign(t, 0);
//                            z_z.assign(t, 0);
//
//                            CILS_Reduction<scalar, index> reduction(cils.m, t, 0, cils.upper, 0, 0);
//                            reduction.cils_aip_reduction(H_A, y_bar);
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
//                    v_norm_cur[0] = helper::find_residual<scalar, index>(cils.m, cils.n, H_P.data(), x_tmp.data(),
//                                                                         cils.y_a.data());
//                    // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
//                    if (v_norm_cur[0] < v_norm[0]) {
//                        // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
//                        for (t = 0; t < cils.n; t++) {
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
//            // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(cils.y_a - H_P * x_tmp);
//
//            MPI_Barrier(MPI_COMM_WORLD);
//            MPI_Allreduce(v_norm, v_norm_rank, 1, MPI_2DOUBLE_PRECISION, MPI_MINLOC, MPI_COMM_WORLD);
//
//            if (rank == v_norm_rank[1]) {
//                b_vector P_cur(cils.n * cils.n, 0), P_tmp(cils.n * cils.n, 0);
//                // 'SCP_Block_Optimal_2:44' P_tmp = eye(cils.n);
//                helper::eye<scalar, index>(cils.n, P_tmp.data());
//                helper::eye<scalar, index>(cils.n, P_cur.data());
//
//                if (best_per >= 0) {
//                    for (k1 = 0; k1 < cils.n; k1++) {
//                        for (i = 0; i < cils.n; i++) {
//                            P_cur[i + cils.n * k1] = P_tmp[i + cils.n * (cils.permutation[best_per][k1] - 1)];
//                        }
//                    }
//                }
//                //'SCP_Block_Optimal_2:115' x_cur = P_cur * x_cur;
//                helper::mtimes_Axy<scalar, index>(cils.n, cils.n, P_cur.data(), x_per.data(), x.data());
//                for (i = 0; i < cils.n; i++) {
//                    x_cur[i] = x[i];
//                }
//            }
//            MPI_Barrier(MPI_COMM_WORLD);
//            MPI_Bcast(&x_cur[0], cils.n, MPI_DOUBLE, v_norm_rank[1], MPI_COMM_WORLD);
//            if (rank == 0)
//                time = MPI_Wtime() - time;
//
//            if (cils.verbose)
//                helper::display_vector(cils.n, x.data(), "x:" + std::to_string(rank));
//
//            return {stopping, time, v_norm[0]};
//        }

    };
}