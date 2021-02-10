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
    cils<scalar, index, n>::cils_block_search_serial(const vector<index> *d, vector<index> *z_B) {

        index ds = d->size(), dx = d->at(ds - 1), n_dx_q_0, n_dx_q_1;
        vector<scalar> y_b(n, 0);
        //special cases:
        if (ds == 1) {
            if (d->at(0) == 1) {
                z_B->at(0) = round(y_A->x[0] / R->x[0]);
                return {z_B, 0, 0};
            } else {
                for (index i = 0; i < n; i++) {
                    y_b[i] = y_A->x[i];
                }
                if (is_constrained)
                    ils_search_obils(0, n, &y_b, z_B);
                else
                    ils_search(0, n, &y_b, z_B);
                return {z_B, 0, 0};
            }
        } else if (ds == n) {
            //Find the Babai point
            return cils_babai_search_serial(z_B);
        }
        scalar start = omp_get_wtime();

        for (index i = 0; i < ds; i++) {
            n_dx_q_0 = i == 0 ? n - dx : n - d->at(ds - 1 - i);
            n_dx_q_1 = i == 0 ? n : n - d->at(ds - i);

            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                scalar sum = 0;
                for (index col = n_dx_q_1; col < n; col++) {
                    sum += R->x[col * n + row] * z_B->at(col);
                }
                y_b[row] = y_A->x[row] - sum;
            }

            if (is_constrained)
                ils_search_obils(n_dx_q_0, n_dx_q_1, &y_b, z_B);
            else
                ils_search(n_dx_q_0, n_dx_q_1, &y_b, z_B);
        }

        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {z_B, run_time, 0};
        return reT;
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_block_search_omp(const index n_proc, const index nswp, const index init,
                                                  const vector<index> *d_s, vector<index> *z_B) {
        index ds = d_s->size(), dx = d_s->at(ds - 1);
        if (ds == 1 || ds == n) {
            return cils_block_search_serial(d_s, z_B);
        }
        bool check = false;
        auto z_x = z_B->data();
        index n_dx_q_0, n_dx_q_1, result[ds] = {}, diff = 0, num_iter = 0, flag = 0, row_n, temp;
        index front = 2 * n_proc, end = 1;
        scalar R_S[n * ds] = {}, sum = 0, y_B[n] = {};
        scalar run_time3;

        ils_search_obils_omp2(n - dx, n, y_B, z_x);
        result[0] = 1;
        for (index row = 0; row < ds - 1; row++) {
            for (index h = 0; h < dx; h++) {
                temp = row * dx + h;
                sum = 0;
                row_n = (n * temp) - ((temp * (temp + 1)) / 2);
#pragma omp simd reduction(+ : sum)
                for (index col = n - dx; col < n; col++) {
                    sum += R_A->x[row_n + col] * z_x[col];
                }
                R_S[temp * ds] = sum;
            }
        }
        omp_set_schedule((omp_sched_t) schedule, chunk_size);
        scalar run_time = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(n_dx_q_0, n_dx_q_1, row_n, sum, temp, check)
        {

//#pragma omp barrier
            for (index j = 0; j < nswp && !flag; j++) {
#pragma omp for schedule(runtime) nowait
                for (index i = 1; i < ds; i++) {
                    if (end <= i && !result[i] && !flag) {// front >= i

                        front++;
                        n_dx_q_0 = n - (i + 1) * dx;
                        n_dx_q_1 = n - i * dx;
                        check = i == end && result[end - 1];
                        row_n = (n_dx_q_0 - 1) * (n - n_dx_q_0 / 2);
//#pragma omp simd// collapse(2)
//                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
////#pragma omp atomic
//                            row_n += n - row;
//                            y_B[row] = 0;
//                            for (index col = 0; col < i; col++) {
//                                temp = i - col - 1;
//                                if (!result[temp]) {
//
//                                    sum = 0; //Put values backwards
//#pragma omp simd reduction(+ : sum)
//                                    for (index l = n_dx_q_1 + dx * col; l < n - dx * temp; l++) {
//                                        sum += R_A->x[l + row_n] * z_x[l];
//                                    }
//                                    R_S[row * ds + temp] = sum;
//                                }
//                                y_B[row] += R_S[row * ds + temp];
//                            }
//                        }


                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                            row_n += n - row;
                            sum = 0;
#pragma omp simd reduction(+ : sum)
                            for (index col = n_dx_q_1; col < n; col++) {
                                sum += R_A->x[row_n + col] * z_x[col];
                            }
                            y_B[row] = sum;
                        }

                        result[i] = ils_search_obils_omp2(n_dx_q_0, n_dx_q_1, y_B, z_x);
#pragma omp atomic
                        diff += result[i];

                        if (check) {
#pragma omp atomic
                            end++;
                            result[i] = 1;
                        } else {
//                            end = result[i] ? i - 1 : end;
                        }
                        flag = (end + diff) >= ds - stop;
//                        if (result[i]) {
//                            for (index row = 0; row < ds - i - 1; row++) {
//                                for (index h = 0; h < dx; h++) {
//                                    temp = row * dx + h;
//                                    sum = 0;
//                                    row_n = (n * temp) - ((temp * (temp + 1)) / 2);
//#pragma omp simd reduction(+ : sum)
//                                    for (index col = n_dx_q_0; col < n_dx_q_1; col++) {
////                                  R_S[temp * ds + i] += R->x[temp + n * col] * z_x[col];
//                                        sum += R_A->x[row_n + col] * z_x[col];
//                                    }
//                                    R_S[temp * ds + i] = sum;
//                                }
//                            }
//                        }
                    }

                }
                num_iter = j;
                if (mode != 0) {
//                    flag = diff >= ds - stop;
                    if (flag) break;
                }
            }

#pragma omp single
            {
                run_time3 = omp_get_wtime() - run_time;
            };
//#pragma omp flush
        }
        scalar run_time2 = omp_get_wtime() - run_time;
#pragma parallel omp cancellation point

        returnType<scalar, index> reT;
        if (mode == 0)
            reT = {z_B, run_time2, diff};
        else {
            reT = {z_B, run_time2, num_iter};
            cout << "n_proc:" << n_proc << "," << "init:" << init << "," << diff << "," << end << ",Ratio:"
                 << (int) (run_time2 / run_time3) << ",";
        }
        return reT;
    }
}

//#pragma omp simd
//                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                            row_n += n - row;
//                            y_B[row] = 0;
//                            for (index col = n_dx_q_1; col < n; col++) {
//                                y_B[row] += R_A->x[row_n + col] * z_x[col];
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
//             y_B[row] += R_A->x[l + row_n] * z_x[l];
///              R_S[row * ds + block] += R_A->x[l + row_n] * z_x[l];
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
//                y_B[row_k] += R_A->x[l + row_n] * z_x[l];
////                                R_S[row * ds + block] += R_A->x[l + row_n] * z_x[l];
//            }
//        }
//        z_p[row_k] = z_x[row_k];
//        c[row_k] = (y_A->x[row_k] - y_B[row_k] - sum[row_k]) / R_A->x[row_n + row_k];
//        temp = round(c[row_k]);
//        z_p[row_k] = temp < 0 ? 0 : temp > upper ? upper : temp;
//        d[row_k] = c[row_k] > z_p[row_k] ? 1 : -1;
//
//        gamma = R_A->x[row_n + row_k] * (c[row_k] - z_p[row_k]);
//        newprsd = p[row_k] + gamma * gamma;
//
//        if (row_k != n_dx_q_0) {
//            row_k--;
//            row_n -= (n - row_k - 1);
//            sum[row_k] = 0;
//            for (index col = row_k + 1; col < n_dx_q_1; col++) {
//                sum[row_k] += R_A->x[row_n + col] * z_p[col];
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
                                y_B[row_k] += R_A->x[h + row_n] * z_x[h];
//                                R_S[row * ds + block] += R_A->x[l + row_n] * z_x[l];
                            }
                        }
                        z_p[row_k] = z_x[row_k];
                        c[row_k] = (y_A->x[row_k] - y_B[row_k] - sum[row_k]) / R_A->x[row_n + row_k];
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

                        gamma = R_A->x[row_n + row_k] * (c[row_k] - z_p[row_k]);
                        newprsd = p[row_k] + gamma * gamma;

                        if (row_k != n_dx_q_0) {
                            row_k--;
                            row_n -= (n - row_k - 1);
                            sum[row_k] = 0;
                            for (index col = row_k + 1; col < n_dx_q_1; col++) {
                                sum[row_k] += R_A->x[row_n + col] * z_p[col];
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
                                    c[row_k] = (y_A->x[row_k] - y_B[row_k] - sum[row_k]) / R_A->x[row_n + row_k];
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
                                    gamma = R_A->x[row_n + row_k] * (c[row_k] - z_p[row_k]);
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
                                    sum[row_k] += R_A->x[row_n + row_k] * d[row_k];
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
                                    gamma = R_A->x[row_n + row_k] * (c[row_k] - z_p[row_k]);
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
//                                        R_S[temp * ds + i] -= R_A->x[row_n + col] * (z_p[h] - z_x[col]);
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
////                                  R_S[temp * ds + i] += R->x[temp + n * col] * z_x[col];
//                                            sum += R_A->x[row_n + col] * z_x[col];
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