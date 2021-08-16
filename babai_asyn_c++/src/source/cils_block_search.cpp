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
#include "cils_reduction.cpp"
#include "cils_ils_search.cpp"
#include <mpi.h>

using namespace std;

namespace cils {
    template<typename scalar, typename index, index m, index n>
    returnType<scalar, index>
    cils<scalar, index, m, n>::cils_block_search_serial(const index init, const scalar *R_R, const scalar *y_r,
                                                        const vector<index> *d, vector<scalar> *z_B) {

        index ds = d->size(), n_dx_q_0, n_dx_q_1;
        array<scalar, n> y_b;
        y_b.fill(0);
        if (ds == n) {
            //Find the Babai point
            return cils_babai_search_serial(z_B);
        }

        scalar sum = 0;
        cils_search<scalar, index> ils(m, n, program_def::k);
        scalar start = omp_get_wtime();

        if (init == -1) {
            for (index i = 0; i < ds; i++) {
                n_dx_q_1 = d->at(i);
                n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);

                for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                    sum = 0;
                    for (index col = n_dx_q_1; col < n; col++) {
                        sum += R_R[col * n + row] * z_B->at(col);
                    }
                    y_b[row] = y_r[row] - sum;
                }
                if (is_constrained)
                    ils.obils_search(n_dx_q_0, n_dx_q_1, 0, R_R, y_b, z_B);
                else
                    ils.ils_search(n_dx_q_0, n_dx_q_1, 0, R_R, y_b, z_B);
            }
        }
        start = omp_get_wtime() - start;
//        cils_search<scalar, index, m,  n> ils(program_def::k);

        scalar run_time = omp_get_wtime();

        for (index i = 0; i < ds; i++) {
            n_dx_q_1 = d->at(i);
            n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);

            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                sum = 0;
                for (index col = n_dx_q_1; col < n; col++) {
                    sum += R_R[col * n + row] * z_B->at(col);
                }
                y_b[row] = y_r[row] - sum;
            }
            if (is_constrained)
                ils.obils_search(n_dx_q_0, n_dx_q_1, 1, R_R, y_b, z_B);
            else
                ils.ils_search(n_dx_q_0, n_dx_q_1, 1, R_R, y_b, z_B);
        }

        run_time = omp_get_wtime() - run_time + start;
        //Matlab Partial Reduction needs to do the permutation
        matrix_vector_mult<scalar, index, m, n>(Z, z_B);

        returnType<scalar, index> reT = {{run_time - start}, run_time, 0};
        return reT;
    }

    template<typename scalar, typename index, index m, index n>
    returnType<scalar, index>
    cils<scalar, index, m, n>::cils_block_search_serial_CPUTEST(const scalar *R_R, const scalar y_r,
                                                                const vector<index> *d, vector<scalar> *z_B) {
        index ds = d->size(), n_dx_q_0, n_dx_q_1;
        array<scalar, n> y_b;
        y_b.fill(0);
        vector<scalar> time(2 * ds, 0);
        cils_search<scalar, index> ils(m, n, program_def::k);
        //special cases:
        if (ds == 1) {
            if (d->at(0) == 1) {
                z_B->at(0) = round(y_r[0] / R_R[0]);
                return {{}, 0, 0};
            } else {
                for (index i = 0; i < n; i++) {
                    y_b[i] = y_r[i];
                }
//                if (is_constrained)
//                    ils_search_obils(0, n, y_b, z_B);
//                else
//                    ils_search(0, n, y_b, z_B);
                return {{}, 0, 0};
            }
        } else if (ds == n) {
            //Find the Babai point
            return cils_babai_search_serial(z_B);
        }

        for (index i = 0; i < ds; i++) {
            n_dx_q_1 = d->at(i);
            n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);

            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                scalar sum = 0;
                for (index col = n_dx_q_1; col < n; col++) {
                    sum += R_R[col * n + row] * z_B->at(col);
                }
                y_b[row] = y_r[row] - sum;
            }
        }
//        std::random_shuffle(r_d.begin(), r_d.end());

        scalar start = omp_get_wtime();
        for (index i = 0; i < ds; i++) {
            n_dx_q_1 = d->at(i);
            n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);
            index time_index = i;

            time[time_index] = omp_get_wtime();
            if (is_constrained)
                time[time_index + ds] = ils.obils_search(n_dx_q_0, n_dx_q_1, 1, R_R, y_b, z_B);
            else
                time[time_index + ds] = ils.ils_search(n_dx_q_0, n_dx_q_1, 1, R_R, y_b, z_B);

            time[time_index] = omp_get_wtime() - time[time_index];
        }

        scalar run_time = omp_get_wtime() - start;

//        for (index i = 0; i < ds; i++) {
//            printf("%.5f,", time[i]);
//        }

        //Matlab Partial Reduction needs to do the permutation
        matrix_vector_mult<scalar, index, m, n>(Z, z_B);

        returnType<scalar, index> reT = {time, run_time, 0};
        return reT;
    }

    template<typename scalar, typename index, index m, index n>
    returnType<scalar, index>
    cils<scalar, index, m, n>::cils_block_search_omp(const index n_proc, const index nswp, const index init,
                                                     const scalar *y_r, const vector<index> *d, vector<scalar> *z_B) {
        index ds = d->size();
        if (ds == 1 || ds == n) {
            return cils_block_search_serial(init, d, z_B);
        }

        auto z_x = z_B->data();
        index diff = 0, num_iter = 0, flag = 0, temp, R_S_1[ds] = {}, R_S_2[ds] = {};
        index test, row_n, check = 0, r, _nswp = nswp, end = 0;
        index n_dx_q_2, n_dx_q_1, n_dx_q_0;
        scalar sum = 0, start;
        scalar run_time = 0, run_time3 = 0;

        array<scalar, n> y_B;
        y_B.fill(0);
        cils_search<scalar, index> _ils(m, n, program_def::k);
        omp_set_schedule((omp_sched_t) schedule, chunk_size);
        auto lock = new omp_lock_t[ds]();
        for (index i = 0; i < ds; i++) {
            omp_set_lock(&lock[i]);
        }

        omp_unset_lock(&lock[0]);
        omp_unset_lock(&lock[1]);


#pragma omp parallel default(none) num_threads(n_proc)
        {}
        if (init == -1) {
            start = omp_get_wtime();
            n_dx_q_2 = d->at(0);
            n_dx_q_0 = d->at(1);

            if (program_def::is_constrained)
                _ils.obils_search_omp(n_dx_q_0, n_dx_q_2, 0, 0, R_A, y_r, z_x);
            else
                _ils.ils_search_omp(n_dx_q_0, n_dx_q_2, 0, 0, R_A, y_r, z_x);

            R_S_2[0] = 1;
            end = 1;
#pragma omp parallel default(shared) num_threads(n_proc) private(n_dx_q_2, n_dx_q_1, n_dx_q_0, sum, temp, check, test, row_n)
            {
                for (index j = 0; j < _nswp && !flag; j++) {
#pragma omp for schedule(dynamic) nowait
                    for (index i = 1; i < ds; i++) {
                        if (!flag && end <= i) {// !R_S_1[i] && front >= i &&!R_S_1[i] &&
                            n_dx_q_2 = d->at(i);
                            n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);
                            check = i == end;
                            row_n = (n_dx_q_0 - 1) * (n - n_dx_q_0 / 2);
                            for (index row = n_dx_q_0; row < n_dx_q_2; row++) {
                                sum = 0;
                                row_n += n - row;
                                for (index col = n_dx_q_2; col < n; col++) {
                                    sum += R_A[col + row_n] * z_x[col];
                                }
                                y_B[row] = y_r[row] - sum;
                            }
                            if (program_def::is_constrained)
                                R_S_2[i] = _ils.obils_search_omp(n_dx_q_0, n_dx_q_2, i, 0, R_A, y_B, z_x);
                            else
                                R_S_2[i] = _ils.ils_search_omp(n_dx_q_0, n_dx_q_2, i, 0, R_A, y_B, z_x);
                            if (check) {
                                end = i + 1;
                                R_S_2[i] = 1;
                            }
                            diff += R_S_2[i];
                            if (mode != 0) {
                                flag = ((diff) >= ds - 1) && j > 0;
                            }
                        }
                    }
                }
#pragma omp single
                {
                    if (k != 1) run_time = omp_get_wtime() - start;
                };
            }
            if (k == 1) run_time = omp_get_wtime() - start;
            flag = check = diff = 0;
            _nswp = 3;
        }

        cils_search<scalar, index> ils(m, n, program_def::k);
        scalar run_time2 = omp_get_wtime();
        n_dx_q_2 = d->at(0);
        n_dx_q_0 = d->at(1);

        if (program_def::is_constrained)
            ils.obils_search_omp(n_dx_q_0, n_dx_q_2, 0, 1, R_A, y_r, z_x);
        else
            ils.ils_search_omp(n_dx_q_0, n_dx_q_2, 0, 1, R_A, y_r, z_x);


        R_S_1[0] = 1;
        end = 1;

#pragma omp parallel default(shared) num_threads(n_proc) private(n_dx_q_2, n_dx_q_1, n_dx_q_0, sum, temp, check, test, row_n)
        {

            for (index j = 0; j < _nswp && !flag; j++) {
//                omp_set_lock(&lock[j - 1]);
//                omp_unset_lock(&lock[j - 1]);
#pragma omp for schedule(runtime) nowait
                for (index i = 1; i < ds; i++) {
                    if (!flag && end <= i) {//  front >= i   &&
                        n_dx_q_2 = d->at(i);
                        n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);
                        check = i == end;
                        row_n = (n_dx_q_0 - 1) * (n - n_dx_q_0 / 2);
                        for (index row = n_dx_q_0; row < n_dx_q_2; row++) {
                            sum = 0;
                            row_n += n - row;
#pragma omp simd reduction(+:sum)
                            for (index col = n_dx_q_2; col < n; col++) {
                                sum += R_A[col + row_n] * z_x[col];
                            }
                            y_B[row] = y_r[row] - sum;
                        }
//                        test = 0;
//                        for (index row = 0; row < i; row++){
//                            test += R_S_1[i];
//                        }
//                        omp_set_lock(&lock[i - 1]);

//                        check = check || R_S_1[i - 1];    
                        if (program_def::is_constrained)
                            R_S_1[i] = ils.obils_search_omp(n_dx_q_0, n_dx_q_2, i, check, R_A, y_B, z_x);
                        else
                            R_S_1[i] = ils.ils_search_omp(n_dx_q_0, n_dx_q_2, i, check, R_A, y_B, z_x);
                        omp_unset_lock(&lock[j]);
                        if (check) { //!R_S_1[i] &&
                            end = i + 1;
                            R_S_1[i] = 1;
                        }
#pragma omp atomic
                        diff += R_S_1[i];
                        if (mode != 0) {
                            flag = ((diff) >= ds - stop) && j > 0;
                        }
                    }
                    num_iter = j;

                }
            }

#pragma omp single
            {
                run_time3 = omp_get_wtime() - run_time2;
            }
//#pragma omp flush
        }
        run_time2 = omp_get_wtime() - run_time2;
#pragma parallel omp cancellation point

        returnType<scalar, index> reT;
        matrix_vector_mult<scalar, index, m, n>(Z, z_B);

        scalar time = 0; //(run_time3 + run_time2) * 0.5;
        if (init == -1) {
            time = k == 1 ? run_time2 + run_time : run_time2 + run_time;
        } else {
            time = k == 1 ? run_time2 : run_time2 * 0.5 + run_time3 * 0.5;
        }
        if (mode == 0)
            reT = {{run_time3}, time, (scalar) diff + end};
        else {
            reT = {{run_time3}, time, (scalar) num_iter + 1};
//            cout << "n_proc:" << n_proc << "," << "init:" << init << "," << diff << "," << end << ",Ratio:"
//                 << (index) (run_time2 / run_time3) << "," << run_time << "||";
//            cout.flush();
        }
        for (index i = 0; i < ds; i++) {
            omp_destroy_lock(&lock[i]);
        }
        return reT;
    }


    template<typename scalar, typename index, index m, index n>
    returnType<scalar, index>
    cils<scalar, index, m, n>::cils_scp_block_optimal_serial(vector<scalar> &x_cur, scalar v_norm_cur) {
        vector<scalar> H_A, x_tmp(x_cur), x_per(x_cur), y_bar(m, 0), y(m, 0);
        vector<index> b;
        // 'SCP_Block_Optimal_2:24' stopping=zeros(1,3);
        vector<scalar> stopping(3, 0);
        array<scalar, m * n> H_P(H);
        vector<scalar> z, z_z; //z_z = Z * z;

        scalar time = omp_get_wtime();
        //  Subfunctions: SCP_opt

        // 'SCP_Block_Optimal_2:25' b_count = 0;
        index b_count = 0;
        // 'SCP_Block_Optimal_2:27' if v_norm_cur <= tolerance
        if (v_norm_cur <= tolerance) {
            // 'SCP_Block_Optimal_2:28' stopping(1)=1;
            stopping[0] = 1;
            time = omp_get_wtime() - time;
            return {{}, time, v_norm_cur};
        }
        index cur_1st, cur_end, b_i, i, t, k1, per = -1, best_per = -1;
        // 'SCP_Block_Optimal_2:32' q = ceil(n/m);
        // 'SCP_Block_Optimal_2:33' indicator = zeros(2, q);
        // 'SCP_Block_Optimal_2:47' H_P = H_C;
        // 'SCP_Block_Optimal_2:48' x_tmp = x_cur;
        // 'SCP_Block_Optimal_2:49' v_norm = v_norm_cur;
        scalar v_norm = v_norm_cur;
        // 'SCP_Block_Optimal_2:51' while 1
        // 'SCP_Block_Optimal_2:47' H_P = H_C;
        // 'SCP_Block_Optimal_2:48' x_tmp = x_cur;
        // 'SCP_Block_Optimal_2:49' v_norm = v_norm_cur;
        // 'SCP_Block_Optimal_2:50' per = false;
        // 'SCP_Block_Optimal_2:51' while 1
        time = omp_get_wtime();
        for (index iter = 0; iter < search_iter; iter++) {
            // H_Apply permutation strategy to update x_cur and v_norm_cur
            // [x_tmp, v_norm_temp] = block_opt(H_P, y_a, x_tmp, n, indicator);
            // Corresponds to H_Algorithm 12 (Block Optimal) in Report 10
            // 'SCP_Block_Optimal_2:56' for j = 1:q
            for (index j = 0; j < q; j++) {
                // cur_1st refers to the column of H_C where the current block starts
                // cur_end refers to the column of H_C where the current block ends
                // 'SCP_Block_Optimal_2:60' cur_1st = indicator(1, j);
                cur_1st = indicator[2 * j];
                // 'SCP_Block_Optimal_2:61' cur_end = indicator(2, j);
                cur_end = indicator[2 * j + 1];
                y_bar.assign(m, 0);
                for (k1 = 0; k1 <= cur_1st - 2; k1++) {
                    for (i = 0; i < m; i++) {
                        t = k1 * m + i;
                        y_bar[i] += H_P[t % m + t] * x_tmp[k1];
                    }
                }
                for (k1 = 0; k1 < n - cur_end; k1++) {
                    for (i = 0; i < m; i++) {
                        t = k1 * m + i;
                        y_bar[i] += H_P[t % m + m * (cur_end + t / m)] * x_tmp[cur_end + k1];
                    }
                }
                for (t = 0; t < m; t++) {
                    y_bar[t] = y_a[t] - y_bar[t];
                }
                //                }
                //  Compute optimal solution
                // 'SCP_Block_Optimal_2:73' e_vec = repelem(1, cur_end-cur_1st+1)';
                // 'SCP_Block_Optimal_2:74' y_bar = y_bar - H_P(:, cur_1st:cur_end) * e_vec;
                y.assign(m, 0);
                for (k1 = 0; k1 <= cur_end - cur_1st; k1++) {
                    for (i = 0; i < m; i++) {
                        t = k1 * m + i;
                        y[i] += H_P[t % m + m * (cur_1st - 1 + t / m)];
                    }
                }
                for (t = 0; t < m; t++) {
                    y_bar[t] = y_bar[t] - y[t];
                }

                // 'SCP_Block_Optimal_2:75' H_adj = 2 * H_P(:, cur_1st:cur_end);
                // 'SCP_Block_Optimal_2:76' l = repelem(-1, cur_end-cur_1st+1)';
                // 'SCP_Block_Optimal_2:77' u = repelem(0, cur_end-cur_1st+1)';
                t = cur_end - cur_1st + 1;
                //                if(t < 0) t = 1;
                H_A.resize(m * t);
                for (k1 = 0; k1 < t; k1++) {
                    for (i = 0; i < m; i++) {
                        H_A[i + m * k1] = 2.0 * H_P[i + m * (cur_1st - 1 + k1)];
                    }
                }
                // 'SCP_Block_Optimal_2:80' z = obils(H_adj, y_bar, l, u);
                z.resize(t);
                z_z.resize(t);
                z.assign(t, 0);
                z_z.assign(t, 0);

                cils_reduction<scalar, index> reduction(m, t, 0, 0);
                reduction.cils_qr_serial(H_A.data(), y_bar.data());
                //                reduction.cils_LLL_serial();

                cils_search<scalar, index> ils(m, t, program_def::k);
                ils.obils_search(0, t, 1, reduction.R_Q.data(), reduction.y_q.data(), z_z);

                // 'SCP_Block_Optimal_2:81' x_tmp(cur_1st:cur_end) = 2 * z + e_vec;
                for (i = 0; i < t; i++) {
                    x_tmp[cur_1st + i - 1] = 2.0 * z_z[i] + 1;
                }
            }

            // x_tmp
            // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(y_a - H_P * x_tmp);
            y.assign(m, 0);
            y_bar.assign(m, 0);
            helper::mtimes_Axy(m, n, H_P.data(), x_tmp.data(), y.data());
            for (i = 0; i < m; i++) {
                y[i] = y_a[i] - y[i];
            }
            scalar v_norm_temp = helper::norm<scalar, index>(m, y.data());
            // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
//            printf("v_t:%8.5f, v_n:%8.5f\n", v_norm_temp, v_norm);
            if (v_norm_temp < v_norm) {
                // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
                for (t = 0; t < n; t++) {
                    x_per[t] = x_tmp[t];
                }
                // 'SCP_Block_Optimal_2:88' v_norm_cur = v_norm_temp;
                v_norm_cur = v_norm_temp;
                // 'SCP_Block_Optimal_2:89' if per
                best_per = per;
                per = -1;
                // 'SCP_Block_Optimal_2:94' if v_norm_cur <= tolerance
                if (v_norm_cur <= tolerance) {
                    // 'SCP_Block_Optimal_2:95' stopping(2)=1;
                    stopping[1] = 1;
                    iter = search_iter;
                }
                // 'SCP_Block_Optimal_2:99' v_norm = v_norm_cur;
                v_norm = v_norm_cur;
            }
            if (!stopping[1]) {
                // 'SCP_Block_Optimal_2:100' else
                // 'SCP_Block_Optimal_2:101' permutation = randperm(n);

                // 'SCP_Block_Optimal_2:102' H_P = H_C(:,permutation);
                // 'SCP_Block_Optimal_2:103' x_tmp = x_cur(permutation);
                for (k1 = 0; k1 < n; k1++) {
                    for (i = 0; i < m; i++) {
                        H_P[i + m * k1] = H[i + m * (permutation[iter][k1] - 1)];
                    }
                    x_tmp[k1] = x_cur[permutation[iter][k1] - 1];
                }
                // 'SCP_Block_Optimal_2:104' per = true;
                per = iter;
            }
            // 'SCP_Block_Optimal_2:110' b_count = b_count + 1;
            b_count++;
            // If we don't decrease the residual, keep trying permutations
        }
        if (b_count >= search_iter) {
            // 'SCP_Block_Optimal_2:107' stopping(3)=1;
            stopping[2] = 1;
        }
        vector<scalar> P_cur(n * n, 0), P_tmp(n * n, 0);
        // 'SCP_Block_Optimal_2:44' P_tmp = eye(n);
        helper::eye<scalar, index>(n, P_tmp.data());

        if (best_per >= 0) {
            b.resize(n * n);
            for (k1 = 0; k1 < n; k1++) {
                for (i = 0; i < n; i++) {
                    P_cur[i + n * k1] = P_tmp[i + n * (permutation[best_per][k1] - 1)];
                }
            }
        }
        // 'SCP_Block_Optimal_2:115' x_cur = P_cur * x_cur;
        helper::mtimes_Axy<scalar, index>(n, n, P_cur.data(), x_per.data(), x_tmp.data());
        for (i = 0; i < n; i++) {
            x_cur[i] = x_tmp[i];
        }

        time = omp_get_wtime() - time;
        return {stopping, time, v_norm_cur};
    }


    template<typename scalar, typename index, index m, index n>
    returnType<scalar, index>
    cils<scalar, index, m, n>::cils_scp_block_optimal_omp(vector<scalar> &x_cur, scalar v_norm_cur, index n_proc) {

        vector<index> b;
        // 'SCP_Block_Optimal_2:24' stopping=zeros(1,3);
        vector<scalar> stopping(3, 0);

        scalar time = omp_get_wtime();
        //  Subfunctions: SCP_opt

        // 'SCP_Block_Optimal_2:25' b_count = 0;
        index b_count = 0;
        // 'SCP_Block_Optimal_2:27' if v_norm_cur <= tolerance
        if (v_norm_cur <= tolerance) {
            // 'SCP_Block_Optimal_2:28' stopping(1)=1;
            stopping[0] = 1;
            time = omp_get_wtime() - time;
            return {{}, time, v_norm_cur};
        }
        index cur_1st, cur_end, b_i, i, t, k1, per = -1, best_per = -1;
        // 'SCP_Block_Optimal_2:32' q = ceil(n/m);
        // 'SCP_Block_Optimal_2:33' indicator = zeros(2, q);

        scalar v_norm = v_norm_cur;
        // 'SCP_Block_Optimal_2:50' per = false;
        bool flag = false;
        vector<scalar> x_per(x_cur), x(n, 0);
        // 'SCP_Block_Optimal_2:51' while 1
        #pragma omp parallel default(shared) num_threads(n_proc)
        {}
        time = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(v_norm_cur, cur_1st, cur_end, b_i, i, t, k1, per)
        {
            vector<scalar> H_A, x_tmp(x_cur), y_bar(m, 0), y(m, 0);
            array<scalar, m * n> H_P(H);
            vector<scalar> z, z_z; //z_z = Z * z;
            v_norm_cur = v_norm;
#pragma omp for schedule(static, 1)
            for (index iter = 0; iter < search_iter; iter++) {
                // H_Apply permutation strategy to update x_cur and v_norm_cur
                // [x_tmp, v_norm_temp] = block_opt(H_P, y_a, x_tmp, n, indicator);
                // Corresponds to H_Algorithm 12 (Block Optimal) in Report 10
                // 'SCP_Block_Optimal_2:56' for j = 1:q
                for (index j = 0; j < q; j++) {
                    // cur_1st refers to the column of H_C where the current block starts
                    // cur_end refers to the column of H_C where the current block ends
                    // 'SCP_Block_Optimal_2:60' cur_1st = indicator(1, j);
                    cur_1st = indicator[2 * j];
                    // 'SCP_Block_Optimal_2:61' cur_end = indicator(2, j);
                    cur_end = indicator[2 * j + 1];

                    y_bar.assign(m, 0);
                    for (k1 = 0; k1 <= cur_1st - 2; k1++) {
                        for (i = 0; i < m; i++) {
                            t = k1 * m + i;
                            y_bar[i] += H_P[t % m + t] * x_tmp[k1];
                        }
                    }
                    for (k1 = 0; k1 < n - cur_end; k1++) {
                        for (i = 0; i < m; i++) {
                            t = k1 * m + i;
                            y_bar[i] += H_P[t % m + m * (cur_end + t / m)] * x_tmp[cur_end + k1];
                        }
                    }
                    for (t = 0; t < m; t++) {
                        y_bar[t] = y_a[t] - y_bar[t];
                    }
                    //                }
                    //  Compute optimal solution
                    // 'SCP_Block_Optimal_2:73' e_vec = repelem(1, cur_end-cur_1st+1)';
                    // 'SCP_Block_Optimal_2:74' y_bar = y_bar - H_P(:, cur_1st:cur_end) * e_vec;
                    y.assign(m, 0);
                    for (k1 = 0; k1 <= cur_end - cur_1st; k1++) {
                        for (i = 0; i < m; i++) {
                            t = k1 * m + i;
                            y[i] += H_P[t % m + m * (cur_1st - 1 + t / m)];
                        }
                    }
                    for (t = 0; t < m; t++) {
                        y_bar[t] = y_bar[t] - y[t];
                    }

                    // 'SCP_Block_Optimal_2:75' H_adj = 2 * H_P(:, cur_1st:cur_end);
                    // 'SCP_Block_Optimal_2:76' l = repelem(-1, cur_end-cur_1st+1)';
                    // 'SCP_Block_Optimal_2:77' u = repelem(0, cur_end-cur_1st+1)';
                    t = cur_end - cur_1st + 1;
                    //                if(t < 0) t = 1;
                    H_A.resize(m * t);
                    for (k1 = 0; k1 < t; k1++) {
                        for (i = 0; i < m; i++) {
                            H_A[i + m * k1] = 2.0 * H_P[i + m * (cur_1st - 1 + k1)];
                        }
                    }
                    // 'SCP_Block_Optimal_2:80' z = obils(H_adj, y_bar, l, u);
                    z.resize(t);
                    z_z.resize(t);
                    z.assign(t, 0);
                    z_z.assign(t, 0);

                    cils_reduction<scalar, index> reduction(m, t, 0, 0);
                    reduction.cils_qr_serial(H_A.data(), y_bar.data());
                    //                reduction.cils_LLL_serial();

                    cils_search<scalar, index> ils(m, t, program_def::k);
                    ils.obils_search(0, t, 1, reduction.R_Q.data(), reduction.y_q.data(), z_z);

                    // 'SCP_Block_Optimal_2:81' x_tmp(cur_1st:cur_end) = 2 * z + e_vec;
                    for (i = 0; i < t; i++) {
                        x_tmp[cur_1st + i - 1] = 2.0 * z_z[i] + 1;
                    }
                }

                // x_tmp
                // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(y_a - H_P * x_tmp);
                y.assign(m, 0);
                y_bar.assign(m, 0);
                helper::mtimes_Axy(m, n, H_P.data(), x_tmp.data(), y.data());
                for (i = 0; i < m; i++) {
                    y[i] = y_a[i] - y[i];
                }
                scalar v_norm_temp = helper::norm<scalar, index>(m, y.data());
                // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
                if (v_norm_temp < v_norm) {
                    // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
                    for (t = 0; t < n; t++) {
                        x_per[t] = x_tmp[t];
                    }
                    // 'SCP_Block_Optimal_2:88' v_norm_cur = v_norm_temp;
                    v_norm_cur = v_norm_temp;
                    // 'SCP_Block_Optimal_2:89' if per
                    best_per = per;
                    per = -1;
                    // 'SCP_Block_Optimal_2:94' if v_norm_cur <= tolerance
                    if (v_norm_cur <= tolerance) {
                        // 'SCP_Block_Optimal_2:95' stopping(2)=1;
                        stopping[1] = 1;
                    }
                    v_norm = v_norm_cur;
                }
                if (!stopping[1]) {
                    for (k1 = 0; k1 < n; k1++) {
                        for (i = 0; i < m; i++) {
                            H_P[i + m * k1] = H[i + m * (permutation[iter][k1] - 1)];
                        }
                        x_tmp[k1] = x_cur[permutation[iter][k1] - 1];
                    }
                    per = iter;
                }
                // 'SCP_Block_Optimal_2:110' b_count = b_count + 1;
#pragma omp atomic
                b_count++;

                if (stopping[1]) {
                    iter = search_iter;
                }
                // If we don't decrease the residual, keep trying permutations
            }

        }

        if (b_count >= search_iter) {
            // 'SCP_Block_Optimal_2:107' stopping(3)=1;
            stopping[2] = 1;
        }
        vector<scalar> P_cur(n * n, 0), P_tmp(n * n, 0);
        // 'SCP_Block_Optimal_2:44' P_tmp = eye(n);
        helper::eye<scalar, index>(n, P_tmp.data());

        if (best_per >= 0) {
            b.resize(n * n);
            for (k1 = 0; k1 < n; k1++) {
                for (i = 0; i < n; i++) {
                    P_cur[i + n * k1] = P_tmp[i + n * (permutation[best_per][k1] - 1)];
                }
            }
        }
        // 'SCP_Block_Optimal_2:115' x_cur = P_cur * x_cur;
        helper::mtimes_Axy<scalar, index>(n, n, P_cur.data(), x_per.data(), x.data());
        for (i = 0; i < n; i++) {
            x_cur[i] = x[i];
        }

        time = omp_get_wtime() - time;
        return {stopping, time, v_norm};
    }

    template<typename scalar, typename index, index m, index n>
    returnType<scalar, index>
    cils<scalar, index, m, n>::cils_scp_block_optimal_mpi(vector<scalar> &x_cur, scalar *v_norm_cur,
                                                          index size, index rank) {

        vector<index> b;
        // 'SCP_Block_Optimal_2:24' stopping=zeros(1,3);
        vector<scalar> stopping(3, 0);

        scalar time = omp_get_wtime();
        //  Subfunctions: SCP_opt
        scalar flag = 0.0;
        // 'SCP_Block_Optimal_2:25' b_count = 0;
        index b_count = 0;
        // 'SCP_Block_Optimal_2:27' if v_norm_cur <= tolerance
        if (v_norm_cur[0] <= tolerance) {
            // 'SCP_Block_Optimal_2:28' stopping(1)=1;
            stopping[0] = 1;
            time = omp_get_wtime() - time;
            return {{}, time, v_norm_cur[0]};
        }
        index cur_1st, cur_end, b_i, i, t, k1, per = -1, best_per = -1;
        // 'SCP_Block_Optimal_2:32' q = ceil(n/m);
        // 'SCP_Block_Optimal_2:33' indicator = zeros(2, q);
        auto v_norm = (double *) malloc(2 * sizeof(double));
        auto v_norm_temp = (double *) malloc(1 * sizeof(double));
        auto v_norm_rank = (double *) malloc(2 * sizeof(double));

        v_norm[0] = v_norm_cur[0];
        v_norm[1] = rank;

        v_norm_temp[0] = v_norm_cur[0];


        // 'SCP_Block_Optimal_2:50' per = false;
        vector<scalar> x_per(x_cur), x(n, 0);
        time = MPI_Wtime();
#pragma omp parallel default(shared) num_threads(3) private(v_norm_cur, cur_1st, cur_end, b_i, i, t, k1, per)
        {
            vector<scalar> H_A, x_tmp(x_cur), y_bar(m, 0), y(m, 0);
            array<scalar, m * n> H_P(H);
            vector<scalar> z, z_z; //z_z = Z * z;
            v_norm_cur = v_norm;
#pragma omp for schedule(static, 1) nowait
            for (index iter = rank * search_iter / size; iter < (rank + 1) * search_iter / size; iter++) {
                // H_Apply permutation strategy to update x_cur and v_norm_cur
                // [x_tmp, v_norm_temp] = block_opt(H_P, y_a, x_tmp, n, indicator);
                // Corresponds to H_Algorithm 12 (Block Optimal) in Report 10
                // 'SCP_Block_Optimal_2:56' for j = 1:q
                for (index j = 0; j < q; j++) {
                    // cur_1st refers to the column of H_C where the current block starts
                    // cur_end refers to the column of H_C where the current block ends
                    // 'SCP_Block_Optimal_2:60' cur_1st = indicator(1, j);
                    cur_1st = indicator[2 * j];
                    // 'SCP_Block_Optimal_2:61' cur_end = indicator(2, j);
                    cur_end = indicator[2 * j + 1];

                    y_bar.assign(m, 0);
                    for (k1 = 0; k1 <= cur_1st - 2; k1++) {
                        for (i = 0; i < m; i++) {
                            t = k1 * m + i;
                            y_bar[i] += H_P[t % m + t] * x_tmp[k1];
                        }
                    }
                    for (k1 = 0; k1 < n - cur_end; k1++) {
                        for (i = 0; i < m; i++) {
                            t = k1 * m + i;
                            y_bar[i] += H_P[t % m + m * (cur_end + t / m)] * x_tmp[cur_end + k1];
                        }
                    }
                    for (t = 0; t < m; t++) {
                        y_bar[t] = y_a[t] - y_bar[t];
                    }
                    //                }
                    //  Compute optimal solution
                    // 'SCP_Block_Optimal_2:73' e_vec = repelem(1, cur_end-cur_1st+1)';
                    // 'SCP_Block_Optimal_2:74' y_bar = y_bar - H_P(:, cur_1st:cur_end) * e_vec;
                    y.assign(m, 0);
                    for (k1 = 0; k1 <= cur_end - cur_1st; k1++) {
                        for (i = 0; i < m; i++) {
                            t = k1 * m + i;
                            y[i] += H_P[t % m + m * (cur_1st - 1 + t / m)];
                        }
                    }
                    for (t = 0; t < m; t++) {
                        y_bar[t] = y_bar[t] - y[t];
                    }

                    // 'SCP_Block_Optimal_2:75' H_adj = 2 * H_P(:, cur_1st:cur_end);
                    // 'SCP_Block_Optimal_2:76' l = repelem(-1, cur_end-cur_1st+1)';
                    // 'SCP_Block_Optimal_2:77' u = repelem(0, cur_end-cur_1st+1)';
                    t = cur_end - cur_1st + 1;
                    //                if(t < 0) t = 1;
                    H_A.resize(m * t);
                    for (k1 = 0; k1 < t; k1++) {
                        for (i = 0; i < m; i++) {
                            H_A[i + m * k1] = 2.0 * H_P[i + m * (cur_1st - 1 + k1)];
                        }
                    }
                    // 'SCP_Block_Optimal_2:80' z = obils(H_adj, y_bar, l, u);
                    z.resize(t);
                    z_z.resize(t);
                    z.assign(t, 0);
                    z_z.assign(t, 0);

                    cils_reduction<scalar, index> reduction(m, t, 0, 0);
                    reduction.cils_qr_serial(H_A.data(), y_bar.data());
                    //                reduction.cils_LLL_serial();

                    cils_search<scalar, index> ils(m, t, program_def::k);
                    ils.obils_search(0, t, 1, reduction.R_Q.data(), reduction.y_q.data(), z_z);

                    // 'SCP_Block_Optimal_2:81' x_tmp(cur_1st:cur_end) = 2 * z + e_vec;
                    for (i = 0; i < t; i++) {
                        x_tmp[cur_1st + i - 1] = 2.0 * z_z[i] + 1;
                    }
                }

                // x_tmp
                // 'SCP_Block_Optimal_2:84' v_norm_temp = norm(y_a - H_P * x_tmp);
                y.assign(m, 0);
                y_bar.assign(m, 0);
                helper::mtimes_Axy(m, n, H_P.data(), x_tmp.data(), y.data());
                for (i = 0; i < m; i++) {
                    y[i] = y_a[i] - y[i];
                }
                v_norm_temp[0] = helper::norm<scalar, index>(m, y.data());
                // 'SCP_Block_Optimal_2:86' if v_norm_temp < v_norm
                if (v_norm_temp[0] < v_norm[0]) {
                    // 'SCP_Block_Optimal_2:87' x_cur = x_tmp;
                    for (t = 0; t < n; t++) {
                        x_per[t] = x_tmp[t];
                    }
                    // 'SCP_Block_Optimal_2:88' v_norm_cur = v_norm_temp;
                    v_norm_cur[0] = v_norm_temp[0];
                    // 'SCP_Block_Optimal_2:89' if per
                    best_per = per;
                    per = -1;
                    // 'SCP_Block_Optimal_2:94' if v_norm_cur <= tolerance
                    if (v_norm_cur[0] <= tolerance) {
                        // 'SCP_Block_Optimal_2:95' stopping(2)=1;
                        stopping[1] = 1;
                    }
                    v_norm[0] = v_norm_cur[0];
                }
                if (!stopping[1]) {
                    for (k1 = 0; k1 < n; k1++) {
                        for (i = 0; i < m; i++) {
                            H_P[i + m * k1] = H[i + m * (permutation[iter][k1] - 1)];
                        }
                        x_tmp[k1] = x_cur[permutation[iter][k1] - 1];
                    }
                    per = iter;
                }
                // 'SCP_Block_Optimal_2:110' b_count = b_count + 1;
#pragma omp atomic
                b_count++;

                if (stopping[1]) {
                    iter = search_iter;
                }
                // If we don't decrease the residual, keep trying permutations
            }

        }
//        printf("start:%d, end:%d", rank * search_iter / size, (rank + 1) * search_iter / size);
//        for (iter = rank * search_iter / size + 1; iter < (rank + 1) * search_iter / size; iter++) {
//#pragma omp parallel default(shared) num_threads(q) private(H_A, y_bar, y, cur_1st, cur_end, b_i, i, t, k1)
//            {
//                y_bar.resize(m);
//                y.resize(m);
//                for (index ll = 0; ll < 6; ll++) {
//#pragma omp for schedule(dynamic, 1) nowait
//                    for (index j = 0; j < q; j++) {
//                        // cur_1st refers to the column of H_C where the current block starts
//                        // cur_end refers to the column of H_C where the current block ends
//                        // 'SCP_Block_Optimal_2:60' cur_1st = indicator(1, j);
//                        cur_1st = indicator[2 * j];
//                        // 'SCP_Block_Optimal_2:61' cur_end = indicator(2, j);
//                        cur_end = indicator[2 * j + 1];
//                        y_bar.assign(m, 0);
//                        for (k1 = 0; k1 <= cur_1st - 2; k1++) {
//                            for (i = 0; i < m; i++) {
//                                t = k1 * m + i;
//                                y_bar[i] += H_P[t % m + t] * x_tmp[k1];
//                            }
//                        }
//                        for (k1 = 0; k1 < n - cur_end; k1++) {
//                            for (i = 0; i < m; i++) {
//                                t = k1 * m + i;
//                                y_bar[i] += H_P[t % m + m * (cur_end + t / m)] * x_tmp[cur_end + k1];
//                            }
//                        }
//                        for (t = 0; t < m; t++) {
//                            y_bar[t] = y_a[t] - y_bar[t];
//                        }
//                        //                }
//                        //  Compute optimal solution
//                        // 'SCP_Block_Optimal_2:73' e_vec = repelem(1, cur_end-cur_1st+1)';
//                        // 'SCP_Block_Optimal_2:74' y_bar = y_bar - H_P(:, cur_1st:cur_end) * e_vec;
//                        y.assign(m, 0);
//                        for (k1 = 0; k1 <= cur_end - cur_1st; k1++) {
//                            for (i = 0; i < m; i++) {
//                                t = k1 * m + i;
//                                y[i] += H_P[t % m + m * (cur_1st - 1 + t / m)];
//                            }
//                        }
//                        for (t = 0; t < m; t++) {
//                            y_bar[t] = y_bar[t] - y[t];
//                        }
//
//                        // 'SCP_Block_Optimal_2:75' H_adj = 2 * H_P(:, cur_1st:cur_end);
//                        // 'SCP_Block_Optimal_2:76' l = repelem(-1, cur_end-cur_1st+1)';
//                        // 'SCP_Block_Optimal_2:77' u = repelem(0, cur_end-cur_1st+1)';
//                        t = cur_end - cur_1st + 1;
//                        //                if(t < 0) t = 1;
//                        H_A.resize(m * t);
//                        for (k1 = 0; k1 < t; k1++) {
//                            for (i = 0; i < m; i++) {
//                                H_A[i + m * k1] = 2.0 * H_P[i + m * (cur_1st - 1 + k1)];
//                            }
//                        }
//                        // 'SCP_Block_Optimal_2:80' z = obils(H_adj, y_bar, l, u);
//                        vector<scalar> z(t, 0), z_z(t, 0);
//
//                        cils_reduction<scalar, index> reduction(m, t, 0, 0);
//                        reduction.cils_qr_serial(H_A.data(), y_bar.data());
//                        //                reduction.cils_LLL_serial();
//
//                        cils_search<scalar, index> ils(m, t, program_def::k);
//                        ils.obils_search(0, t, 1, reduction.R_Q.data(), reduction.y_q.data(), z_z);
//
//                        // 'SCP_Block_Optimal_2:81' x_tmp(cur_1st:cur_end) = 2 * z + e_vec;
//                        for (i = 0; i < t; i++) {
//                            x_tmp[cur_1st + i - 1] = 2.0 * z_z[i] + 1;
//                        }
//                    }
//
//                }
//            }
//            y.assign(m, 0);
//            helper::mtimes_Axy(m, n, H_P.data(), x_tmp.data(), y.data());
//            for (i = 0; i < m; i++) {
//                y[i] = y_a[i] - y[i];
//            }
//            v_norm_temp[0] = helper::norm<scalar, index>(m, y.data());
//            if (v_norm_temp[0] < v_norm[0]) {
//                for (t = 0; t < n; t++) {
//                    x_per[t] = x_tmp[t];
//                }
//                v_norm_cur[0] = v_norm_temp[0];
//                best_per = per;
//                per = -1;
//                if (v_norm_cur[0] <= tolerance) {
//                    stopping[1] = 1;
//                    iter = search_iter;
//                }
//                v_norm[0] = v_norm_cur[0];
//            } else {
//                for (k1 = 0; k1 < n; k1++) {
//                    for (i = 0; i < m; i++) {
//                        H_P[i + m * k1] = H[i + m * (permutation[iter][k1] - 1)];
//                    }
//                    x_tmp[k1] = x_cur[permutation[iter][k1] - 1];
//                }
//                per = iter;
//            }
//            //'SCP_Block_Optimal_2:110' b_count = b_count + 1;
//            b_count++;
//        }

//        MPI_Barrier(MPI_COMM_WORLD);
//        printf("v_t:%8.5f, v_n:%8.5f, per:%4d, rank:%d\n", v_norm_cur[0], v_norm[0], best_per, rank);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Allreduce(v_norm, v_norm_rank, 1, MPI_2DOUBLE_PRECISION, MPI_MINLOC, MPI_COMM_WORLD);

        if (rank == v_norm_rank[1]) {
//            printf("v_t:%8.5f, v_n:%8.5f, per:%4d, rank:%8.5f, %8.5f\n", v_norm[0], v_norm[0], best_per, v_norm_rank[0],
//                   v_norm_rank[1]);
            //        MPI_Bcast(&best_per, 1, MPI_INT, (int) v_norm_rank[1], MPI_COMM_WORLD);
            if (b_count >= search_iter) {
                // 'SCP_Block_Optimal_2:107' stopping(3)=1;
                stopping[2] = 1;
            }
            vector<scalar> P_cur(n * n, 0), P_tmp(n * n, 0);
            // 'SCP_Block_Optimal_2:44' P_tmp = eye(n);
            helper::eye<scalar, index>(n, P_tmp.data());
            helper::eye<scalar, index>(n, P_cur.data());

            if (best_per >= 0) {
                b.resize(n * n);
                for (k1 = 0; k1 < n; k1++) {
                    for (i = 0; i < n; i++) {
                        P_cur[i + n * k1] = P_tmp[i + n * (permutation[best_per][k1] - 1)];
                    }
                }
            }
            //'SCP_Block_Optimal_2:115' x_cur = P_cur * x_cur;
            helper::mtimes_Axy<scalar, index>(n, n, P_cur.data(), x_per.data(), x.data());
            for (i = 0; i < n; i++) {
                x_cur[i] = x[i];
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&x_cur[0], n, MPI_DOUBLE, v_norm_rank[1], MPI_COMM_WORLD);

        time = MPI_Wtime() - time;
//        if (rank == 0)
//            helper::display_vector(n, x.data(), "x:" + std::to_string(rank));

        return {stopping, time, v_norm[0]};
    }

}
//                    if (front >= i && end <= i) {
//                        front++;
//                    if (!result[i] && !flag){
//
//                    }

//#pragma omp simd collapse(2)
//                        if(j != 0) {
//                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
////#pragma omp atomic
//                            row_n += n - row;
//                            sum = 0;
//                            for (index col = 0; col < i; col++) {
//                                temp = i - col - 1; //Put values backwards
////                                    if (!result[temp]) {
//                                sum2 = 0;
//#pragma omp simd reduction(+ : sum2)
//                                for (index l = n_dx_q_1 + dx * col; l < n - dx * temp; l++) {
//                                    sum2 += R_A[l + row_n] * z_x[l];
//                                }
////                                    R_S[row * ds + temp] = sum2;
//
//                                sum += sum2;
////                                    sum += R_S[row * ds + temp];
//                                test += result[temp];
//                            }
//                            y_B[row] = y_r[row] - sum;
//                        }
//                    if (first && i > 1) {// front >= i && end <= i!
//                        n_dx_q_2 = d->at(i);
//                        n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);
//
////                        index dx = n_dx_q_2 - n_dx_q_0;
////                        n_dx_q_1 = n_dx_q_0 + 16;
//
//                        row_n = (n_dx_q_0 - 1) * (n - n_dx_q_2 / 2);
//                        for (index row = n_dx_q_0; row < n_dx_q_2; row++) {
//                            sum = 0;
//                            row_n += n - row;
//                            for (index col = n_dx_q_2; col < n; col++) {
//                                sum += R_A[col + row_n] * z_x[col];
//                            }
//                            y_B[row] = y_r[row] - sum;
//                        }
//                        ils.obils_search_omp(n_dx_q_1, n_dx_q_2, i, i == 0, R_A, y_B, z_x);
//
//                        row_n = (n_dx_q_0 - 1) * (n - n_dx_q_0 / 2);
//                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                            sum = 0;
//                            row_n += n - row;
//                            for (index col = n_dx_q_1; col < n; col++) {
//                                sum += R_A[col + row_n] * z_x[col];
//                            }
//                            y_B[row] = y_r[row] - sum;
//                        }
//                        ils.obils_search_omp(n_dx_q_0, n_dx_q_2, i, 0, R_A, y_B, z_x);
//                    } else
//                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                            row_n += n - row;
//                            sum = 0;
//#pragma omp simd reduction(+ : sum)
//                            for (index col = n_dx_q_1; col < n; col++) {
//                                sum += R_A[row_n + col] * z_x[col];
//                            }
//                            y_B[row] = sum;
//                        }
//#pragma omp simd
//                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                            row_n += n - row;
//                            y_B[row] = 0;
//                            for (index col = n_dx_q_1; col < n; col++) {
//                                y_B[row] += R_A[row_n + col] * z_x[col];
//                            }
//                        }
//                        result[i] = ils_search_obils_omp2(n_dx_q_0, n_dx_q_1, i, ds, y_B, z_x);
//void backup(){
// for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//     row_n += n - row;
//     y_B[row] = 0;
//     for (index block = 0; block < i; block++) {
//         R_S[row * ds + block] = 0; //Put values backwards
//         for (index l = n - dx * (i - block); l < n - dx * (i - block - 1); l++) {
//             y_B[row] += R_A[l + row_n] * z_x[l];
///              R_S[row * ds + block] += R_A[l + row_n] * z_x[l];
//         }
///          y_B[row] += R_S[row * ds + block];
//     }
//     z_p[row] = z_x[row];
// }
//    row_n = (n_dx_q_1 - 1) * (n - n_dx_q_1 / 2);
//    for (index row_k = n_dx_q_1 - 1; row_k >= n_dx_q_0;) {
//        y_B[row_k] = 0;
//        for (index block = 0; block < i; block++) {
//            for (index l = n - dx * (i - block); l < n - dx * (i - block - 1); l++) {
//                y_B[row_k] += R_A[l + row_n] * z_x[l];
////                                R_S[row * ds + block] += R_A[l + row_n] * z_x[l];
//            }
//        }
//        z_p[row_k] = z_x[row_k];
//        c[row_k] = (y_r[row_k] - y_B[row_k] - sum[row_k]) / R_A[row_n + row_k];
//        temp = round(c[row_k]);
//        z_p[row_k] = temp < 0 ? 0 : temp > upper ? upper : temp;
//        d[row_k] = c[row_k] > z_p[row_k] ? 1 : -1;
//
//        gamma = R_A[row_n + row_k] * (c[row_k] - z_p[row_k]);
//        newprsd = p[row_k] + gamma * gamma;
//
//        if (row_k != n_dx_q_0) {
//            row_k--;
//            row_n -= (n - row_k - 1);
//            sum[row_k] = 0;
//            for (index col = row_k + 1; col < n_dx_q_1; col++) {
//                sum[row_k] += R_A[row_n + col] * z_p[col];
//            }
//            p[row_k] = newprsd;
//        } else {
//            break;
//        }
//    }
//}
/*
 * index ds = d_s->size(), dx = d_s->at(ds - 1);
        if (ds == 1 || ds == n) {
            return cils_block_search_serial(d_s, z_B);
        }

        auto z_x = z_B->data();
        index upper = pow(2, qam) - 1, n_dx_q_0, n_dx_q_1, z_p[n], iter, dflag;
        index result[ds] = {}, diff = 0, info = 0, flag = 0, row_n, row_k, temp;
        scalar y_B[n] = {}, R_S[n * ds] = {}, sum[n] = {};
        scalar p[n] = {}, c[n] = {};
        index d[n] = {}, l[n] = {}, u[n] = {};
        scalar newprsd, gamma = 0, beta = INFINITY;

        scalar run_time = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(1) private(n_dx_q_0, n_dx_q_1, row_n, temp, row_k, newprsd, gamma, beta, iter, dflag)
        {

            for (index j = 0; j < 1 && !flag; j++) {
//#pragma omp for schedule(static, 1) nowait
                for (index i = 0; i < ds; i++) {
//                    if (flag) continue;
                    n_dx_q_0 = n - (i + 1) * dx;
                    n_dx_q_1 = n - i * dx;
                    gamma = 0;
                    dflag = 1;
                    beta = INFINITY;
                    row_n = (n_dx_q_1 - 1) * (n - n_dx_q_1 / 2);
                    row_k = n_dx_q_1 - 1;for (; row_k >= n_dx_q_0;) {
                        y_B[row_k] = 0;
                        for (index block = 0; block < i; block++) {
                            for (index h = n - dx * (i - block); h < n - dx * (i - block - 1); h++) {
                                y_B[row_k] += R_A[h + row_n] * z_x[h];
//                                R_S[row * ds + block] += R_A[l + row_n] * z_x[l];
                            }
                        }
                        z_p[row_k] = z_x[row_k];
                        c[row_k] = (y_r[row_k] - y_B[row_k] - sum[row_k]) / R_A[row_n + row_k];
                        z_p[row_k] = round(c[row_k]);
                        if (z_p[row_k] <= 0) {
                            z_p[row_k] = u[row_k] = 0; //The lower bound is reached
                            l[row_k] = d[row_k] = 1;
                        } else if (z_p[row_k] >= upper) {
                            z_p[row_k] = upper; //The upper bound is reached
                            u[row_k] = 1;
                            l[row_k] = 0;
                            d[row_k] = -1;
                        } else {
                            l[row_k] = u[row_k] = 0;
                            //  Determine enumeration direction at level block_size
                            d[row_k] = c[row_k] > z_p[row_k] ? 1 : -1;
                        }    gamma = R_A[row_n + row_k] * (c[row_k] - z_p[row_k]);
                        newprsd = p[row_k] + gamma * gamma;    if (row_k != n_dx_q_0) {
                            row_k--;
                            row_n -= (n - row_k - 1);
                            sum[row_k] = 0;
                            for (index col = row_k + 1; col < n_dx_q_1; col++) {
                                sum[row_k] += R_A[row_n + col] * z_p[col];
                            }
                            p[row_k] = newprsd;
                        } else {
                            break;
                        }
                    }

//                    row_n = (n_dx_q_1 - 1) * (n - n_dx_q_1 / 2);
//                    row_k = n_dx_q_1 - 1;
//                    gamma = 0;
//                    dflag = 1;
//                    beta = INFINITY;

//                    for (index count = 0; count < program_def::max_search || iter == 0; count++) {
                    while (true) {
                        if (dflag) {
                            newprsd = p[row_k] + gamma * gamma;
                            if (newprsd < beta) {
                                if (row_k != n_dx_q_0) {
                                    row_k--;
                                    row_n -= (n - row_k - 1);
                                    p[row_k] = newprsd;
                                    c[row_k] = (y_r[row_k] - y_B[row_k] - sum[row_k]) / R_A[row_n + row_k];
                                    z_p[row_k] = round(c[row_k]);
                                    if (z_p[row_k] <= 0) {
                                        z_p[row_k] = u[row_k] = 0;
                                        l[row_k] = d[row_k] = 1;
                                    } else if (z_p[row_k] >= upper) {
                                        z_p[row_k] = upper;
                                        u[row_k] = 1;
                                        l[row_k] = 0;
                                        d[row_k] = -1;
                                    } else {
                                        l[row_k] = u[row_k] = 0;
                                        d[row_k] = c[row_k] > z_p[row_k] ? 1 : -1;
                                    }
                                    gamma = R_A[row_n + row_k] * (c[row_k] - z_p[row_k]);
                                } else {
                                    beta = newprsd;
                                    diff = 0;
//                                    iter++;
#pragma omp simd
                                    for (index h = n_dx_q_0; h < n_dx_q_1; h++) {
                                        diff += z_x[h] == z_p[h];
                                        z_x[h] = z_p[h];
                                    }

//                                    if (iter > program_def::search_iter || diff == dx) {
//                                        break;
//                                    }
                                }
                            } else {
                                dflag = 0;
                            }    } else {
                            if (row_k == n_dx_q_1 - 1) break;
                            else {
                                row_k++;
                                row_n += n - row_k;
                                if (l[row_k] != 1 || u[row_k] != 1) {
                                    z_p[row_k] += d[row_k];
                                    sum[row_k] += R_A[row_n + row_k] * d[row_k];
                                    if (z_p[row_k] == 0) {
                                        l[row_k] = 1;
                                        d[row_k] = -d[row_k] + 1;
                                    } else if (z_p[row_k] == upper) {
                                        u[row_k] = 1;
                                        d[row_k] = -d[row_k] - 1;
                                    } else if (l[row_k] == 1) {
                                        d[row_k] = 1;
                                    } else if (u[row_k] == 1) {
                                        d[row_k] = -1;
                                    } else {
                                        d[row_k] = d[row_k] > 0 ? -d[row_k] - 1 : -d[row_k] + 1;
                                    }
                                    gamma = R_A[row_n + row_k] * (c[row_k] - z_p[row_k]);
                                    dflag = 1;
                                }
                            }
                        }
                    }
//ILS search process
//                        for (index count = 0; count < program_def::max_search || iter == 0; count++) {
//
//                        }
//
//                        result[i] = diff == dx;
//                        if (!result[i]) {
//                        for (index h = 0; h < dx; h++) {
//                            index col = h + n_dx_q_0;
//                            if (z_p[col] != x[h]) {
//                                for (index row = 0; row < ds - i - 1; row++) {
//                                    for (index hh = 0; hh < dx; hh++) {
//                                        temp = row * dx + hh;
//                                        row_n = (n * temp) - ((temp * (temp + 1)) / 2);
//                                        R_S[temp * ds + i] -= R_A[row_n + col] * (z_p[h] - z_x[col]);
//                                    }
//                                }
////                                }
//                            }
//                        }
//                        if (!flag) {
//                            diff += result[i];
//                            info = j;
//                            if (mode != 0)
//                                flag = diff >= ds - stop;
//
//                            if (!result[i] && !flag) {
//                                for (index row = 0; row < ds - i - 1; row++) {
//                                    for (index h = 0; h < dx; h++) {
//                                        temp = row * dx + h;
//                                        sum = 0;
//                                        row_n = (n * temp) - ((temp * (temp + 1)) / 2);
//#pragma omp simd reduction(+:sum)
//                                        for (index col = n_dx_q_0; col < n_dx_q_1; col++) {
////                                  R_S[temp * ds + i] += R_R[temp + n * col] * z_x[col];
//                                            sum += R_A[row_n + col] * z_x[col];
//                                        }
//                                        R_S[temp * ds + i] = sum;
//                                    }
//                                }
//                            }
//                        }
//                    }
//                    for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                        z_x[row] = z_p[row];
//                    }
                }
            }
        }
        scalar run_time2 = omp_get_wtime() - run_time;
#pragma parallel omp cancellation point
#pragma omp flush
        returnType<scalar, index> reT;
        if (mode == 0)
            reT = {z_B, run_time2, diff};
        else {
            reT = {z_B, run_time2, info};
            cout << "diff:" << diff << ", ";
        }
        return reT;
    }
 */