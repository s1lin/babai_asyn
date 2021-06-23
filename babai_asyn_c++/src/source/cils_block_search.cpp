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
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with CILS.  If not, see <http://www.gnu.org/licenses/>.
 */

using namespace std;

namespace cils {
    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_block_search_serial(const index init, const vector<index> *d, coder::array<scalar, 1U> &z_B) {

        index ds = d->size(), n_dx_q_0, n_dx_q_1;
        vector<scalar> y_b(n, 0);
        //special cases:
        if (ds == 1) {
            if (d->at(0) == 1) {
                z_B[0] = round(y_r[0] / R_R[0]);
                return {{}, 0, 0};
            } else {
                for (index i = 0; i < n; i++) {
                    y_b[i] = y_r[i];
                }
//                if (is_constrained)
//                    ils_search_obils(0, n, &y_b, z_B);
//                else
//                    ils_search(0, n, &y_b, z_B);
                return {{}, 0, 0};
            }
        } else if (ds == n) {
            //Find the Babai point
            return cils_babai_search_serial(z_B);
        }

        scalar sum = 0;
        cils_ils<scalar, index, n> ils(program_def::k);
        scalar start = omp_get_wtime();

        if (init == -1) {
            for (index i = 0; i < ds; i++) {
                n_dx_q_1 = d->at(i);
                n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);

                for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                    sum = 0;
                    for (index col = n_dx_q_1; col < n; col++) {
                        sum += R_R[col + row * n] * z_B[col];
                    }
                    y_b[row] = y_r[row] - sum;
                }
                if (is_constrained)
                    ils.obils_serial(n_dx_q_0, n_dx_q_1, 0, R_R, &y_b, z_B);
                else
                    ils_search(n_dx_q_0, n_dx_q_1, &y_b, z_B);
            }
        }
        start = omp_get_wtime() - start;
//        cils_ils<scalar, index, n> ils(program_def::k);

        scalar run_time = omp_get_wtime();

        for (index i = 0; i < ds; i++) {
            n_dx_q_1 = d->at(i);
            n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);

            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                sum = 0;
                for (index col = n_dx_q_1; col < n; col++) {
                    sum += R_R[col + row * n] * z_B[col];
                }
                y_b[row] = y_r[row] - sum;
            }
            if (is_constrained)
                ils.obils_serial(n_dx_q_0, n_dx_q_1, 1, R_R, &y_b, z_B);
            else
                ils_search(n_dx_q_0, n_dx_q_1, &y_b, z_B);
        }

        run_time = omp_get_wtime() - run_time + start;
        //Matlab Partial Reduction needs to do the permutation
//        if (is_matlab)
//            vector_permutation<scalar, index, n>(Z, z_B);

        returnType<scalar, index> reT = {{run_time - start}, run_time, 0};
        return reT;
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_block_search_serial_CPUTEST(const vector<index> *d, coder::array<scalar, 1U> &z_B) {
        index ds = d->size(), n_dx_q_0, n_dx_q_1;
        vector<scalar> y_b(n, 0);
        vector<scalar> time(2 * ds, 0);
        //special cases:
        if (ds == 1) {
            if (d->at(0) == 1) {
                z_B[0] = round(y_r[0] / R_R[0]);
                return {{}, 0, 0};
            } else {
                for (index i = 0; i < n; i++) {
                    y_b[i] = y_r[i];
                }
                if (is_constrained)
                    ils_search_obils(0, n, &y_b, z_B);
                else
                    ils_search(0, n, &y_b, z_B);
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
                    sum += R_R[col + row * n] * z_B[col];
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
                time[time_index + ds] = ils_search_obils(n_dx_q_0, n_dx_q_1, &y_b, z_B);
            else
                ils_search(n_dx_q_0, n_dx_q_1, &y_b, z_B);
            time[time_index] = omp_get_wtime() - time[time_index];
        }

        scalar run_time = omp_get_wtime() - start;

//        for (index i = 0; i < ds; i++) {
//            printf("%.5f,", time[i]);
//        }

        //Matlab Partial Reduction needs to do the permutation
//        if (is_matlab)
//            vector_permutation<scalar, index, n>(Z, z_B);

        returnType<scalar, index> reT = {time, run_time, 0};
        return reT;
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_block_search_omp(const index n_proc, const index nswp, const index init,
                                                  const vector<index> *d, coder::array<scalar, 1U> &z_B) {
        index ds = d->size();
        if (ds == 1 || ds == n) {
            return cils_block_search_serial(init, d, z_B);
        }

        auto z_x = z_B.data();
        index diff = 0, num_iter = 0, flag = 0, temp, R_S_1[ds] = {}, R_S_2[ds] = {};
        index test, row_n, check = 0, r, _nswp = nswp, end = 0;
        index n_dx_q_2, n_dx_q_1, n_dx_q_0;
        scalar sum = 0, y_B[n] = {}, start;
        scalar run_time = 0, run_time3 = 0;

        cils_ils<scalar, index, n> _ils(program_def::k);
        omp_set_schedule((omp_sched_t) schedule, chunk_size);

        if (init == -1) {
            scalar start = omp_get_wtime();
            n_dx_q_2 = d->at(0);
            n_dx_q_0 = d->at(1);
            _ils.obils_omp(n_dx_q_0, n_dx_q_2, 0, 0, R_A, y_r.data(), z_x);
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

                            test = 0;
                            for (index row = n_dx_q_0; row < n_dx_q_2; row++) {
                                sum = 0;
                                row_n += n - row;

                                for (index col = n_dx_q_2; col < n; col++) {
                                    sum += R_A[col + row_n] * z_x[col];
                                }
                                y_B[row] = y_r[row] - sum;
                            }
                            check = check || test >= i;
                            R_S_2[i] = _ils.obils_omp(n_dx_q_0, n_dx_q_2, i, 0, R_A, y_B, z_x);

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
                    run_time = omp_get_wtime() - start;
                };
            }
            flag = check = diff = 0;
            _nswp = k == 1 ? 5 : 2;
        }

        cils_ils<scalar, index, n> ils(program_def::k);
        scalar run_time2 = omp_get_wtime();
        n_dx_q_2 = d->at(0);
        n_dx_q_0 = d->at(1);
        ils.obils_omp(n_dx_q_0, n_dx_q_2, 0, 1, R_A, y_r.data(), z_x);
        R_S_1[0] = 1;
        end = 1;

#pragma omp parallel default(shared) num_threads(n_proc) private(n_dx_q_2, n_dx_q_1, n_dx_q_0, sum, temp, check, test, row_n)
        {
            for (index j = 0; j < _nswp && !flag; j++) {
#pragma omp for schedule(runtime) nowait
                for (index i = 1; i < ds; i++) {
                    if (!flag && end <= i) {//  front >= i &&!R_S_1[i]  &&
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
//                        test = 0;
//                        for (index row = 0; row < i; row++){
//                            test += R_S_1[i];
//                        }

//                        check = check;// || test == i;
                        R_S_1[i] = ils.obils_omp(n_dx_q_0, n_dx_q_2, i, check, R_A, y_B, z_x);

                        if (check) { //!R_S_1[i] &&
                            end = i + 1;
                            R_S_1[i] = 1;
                        }
//                        if (R_S_1[i]) {
//                            for (index row = 0; row < ds - i - 1; row++) {
//                                for (index h = 0; h < dx; h++) {
//                                    temp = row * dx + h;
//                                    sum = 0;
//                                    row_n = (n * temp) - ((temp * (temp + 1)) / 2);
//#pragma omp simd reduction(+ : sum)
//                                    for (index col = n_dx_q_0; col < n_dx_q_1; col++) {
////                                  R_S[temp * ds + i] += R_R[temp + n * col] * z_x[col];
//                                        sum += R_A[row_n + col] * z_x[col];
//                                    }
//                                    R_S[temp * ds + i] = sum;
//                                }
//                            }
//                        }
//
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
            };
//#pragma omp flush
        }
        run_time2 = omp_get_wtime() - run_time2;
#pragma parallel omp cancellation point


//        if (mode == 3) {
//            cout << endl;
//
//            cout << find_residual<scalar, index, n>(R_R, y_r, z_B) << endl;
//            auto _r = find_residual_by_block<scalar, index, n>(R_R, y_r, d, z_B);
//            display_vector<scalar, index>(&_r);
//
//            vector_reverse_permutation<scalar, index, n>(Z, &x_t);
//            _r = find_residual_by_block<scalar, index, n>(R_R, y_r, d, &x_t);
//            display_vector<scalar, index>(&_r);
//
//            auto _b = find_bit_error_rate_by_block<scalar, index, n>(z_B, &x_t, d, k);
//            display_vector<scalar, index>(&_b);
//            vector_permutation<scalar, index, n>(Z, &x_t);
//        }

//        if (is_matlab)
//            vector_permutation<scalar, index, n>(Z, z_B); //Matlab Partial Reduction needs to do the permutation

        returnType<scalar, index> reT;


        scalar time = 0; //(run_time3 + run_time2) * 0.5;
        if (init == -1) {
            time = k == 1 ? run_time2 + run_time : run_time3 + run_time;
        } else {
            time = k == 1 ? run_time2 : run_time2;
        }
        if (mode == 0)
            reT = {{run_time3}, time, (scalar) diff + end};
        else {
            reT = {{run_time3}, time, (scalar) num_iter + 1};
//            cout << "n_proc:" << n_proc << "," << "init:" << init << "," << diff << "," << end << ",Ratio:"
//                 << (index) (run_time2 / run_time3) << "," << run_time << "||";
//            cout.flush();
        }
        return reT;
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_block_search_omp_dynamic_block(const index n_proc, const index nswp, const index init,
                                                                const vector<index> *d, coder::array<scalar, 1U> &z_B) {
        index ds = d->size();
        if (ds == 1 || ds == n) {
            return cils_block_search_serial(init, d, z_B);
        }

        auto z_x = z_B->data();
        index diff = 0, num_iter = 0, flag = 0, flag_2 = 0, temp, result[ds] = {};
        index test, row_n, check = 0, z_p[n] = {};
        index n_dx_q_2 = d->at(0), n_dx_q_1, n_dx_q_0 = d->at(1);
        index front = chunk_size * n_proc, end = 0, first = 1;
        scalar sum = 0, sum2 = 0, y_B[n] = {}, R_S[n * ds] = {};
        scalar run_time3;

        cils_ils<scalar, index, n> ils(program_def::k);
        omp_set_schedule((omp_sched_t) schedule, chunk_size);

        scalar run_time = omp_get_wtime();
        if (k == 3) {
            n_dx_q_2 = d->at(0);
            n_dx_q_0 = d->at(1);
            ils.obils_omp(n_dx_q_0, n_dx_q_2, 0, 1, R_A, y_r, z_x);
            result[0] = 1;
        }

#pragma omp parallel default(shared) num_threads(n_proc) private(first, n_dx_q_2, n_dx_q_1, n_dx_q_0, sum, temp, sum2, check, test, row_n)
        {
//#pragma omp barrier
            for (index j = 0; j < nswp && !flag; j++) {//
#pragma omp for schedule(runtime) nowait
                for (index i = 0; i < n_proc; i++) {
                    for (index l = 0; l < n; l++) {
                        n_dx_q_2 = d->at(i);
                        n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);
                        check = i == end;
                        front++;
                        row_n = (n_dx_q_0 - 1) * (n - n_dx_q_0 / 2);
//#pragma omp simd collapse(2)
                        test = 0;
                        for (index row = n_dx_q_0; row < n_dx_q_2; row++) {
                            sum = 0;
                            row_n += n - row;
                            for (index col = n_dx_q_2; col < n; col++) {
                                sum += R_A[col + row_n] * z_x[col];
                            }
                            y_B[row] = y_r[row] - sum;
                        }
//                        test = test / block_size;
//                        }
                        check = check || test >= i;
                        result[i] = ils.obils_omp(n_dx_q_0, n_dx_q_2, i, check, R_A, y_B, z_x);

                        if (check) { //!result[i] &&
                            end = i + 1;
                            result[i] = 1;
                        }
                        diff += result[i];
                        if (mode != 0) {
                            flag = ((diff) >= ds - stop) && j > 0;
                        }
                    }
                }
                num_iter = j;
                if (mode != 0) {
//                    flag = diff >= ds - stop;
                    if (flag) break;
                }
                first = 0;
            }

#pragma omp single
            {
                run_time3 = omp_get_wtime() - run_time;
            };
//#pragma omp flush
        }
        scalar run_time2 = omp_get_wtime() - run_time;
#pragma parallel omp cancellation point
        //Matlab Partial Reduction needs to do the permutation
//        if (is_matlab)
//            vector_permutation<scalar, index, n>(Z, z_B);

        returnType<scalar, index> reT;
        if (mode == 0)
            reT = {{}, run_time3, (scalar) diff + end};
        else {
            reT = {{}, run_time3, (scalar) num_iter};
            cout << "n_proc:" << n_proc << "," << "init:" << init << "," << diff << "," << end << ",Ratio:"
                 << (index) (run_time2 / run_time3) << "," << num_iter << "||";
            if (mode == 1)
                cout << endl;
            cout.flush();
        }
        return reT;
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
//                        ils.obils_omp(n_dx_q_1, n_dx_q_2, i, i == 0, R_A, y_B, z_x);
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
//                        ils.obils_omp(n_dx_q_0, n_dx_q_2, i, 0, R_A, y_B, z_x);
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
        index result[ds] = {}, diff = 0, num_iter = 0, flag = 0, row_n, row_k, temp;
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
                    row_k = n_dx_q_1 - 1;

                    for (; row_k >= n_dx_q_0;) {
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
                        }

                        gamma = R_A[row_n + row_k] * (c[row_k] - z_p[row_k]);
                        newprsd = p[row_k] + gamma * gamma;

                        if (row_k != n_dx_q_0) {
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
                            }

                        } else {
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
//                            num_iter = j;
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
            reT = {z_B, run_time2, num_iter};
            cout << "diff:" << diff << ", ";
        }
        return reT;
    }
 */