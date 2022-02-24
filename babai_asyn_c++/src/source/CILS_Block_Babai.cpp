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
using namespace std;

namespace cils {

    template<typename scalar, typename index>
    class CILS_Block_Babai {
    private:
        CILS<scalar, index> cils;
        b_vector R_A;
        index init;
    public:
        b_vector z_hat;

        CILS_Block_Babai(CILS<scalar, index> &cils) {
            this->cils = cils;
            this->z_hat.resize(cils.n);
            this->z_hat.clear();
        }

        void set_R_A(const b_matrix &R) {
            R_A.resize(cils.n / 2 * (1 + cils.n));
            R_A.clear();
            index idx = 0;
            for (index row = 0; row < R.size1(); row++) {
                for (index col = row; col < R.size2(); col++) {
                    R_A[idx] = R(row, col);
                    idx++;
                }
            }
        }


        returnType<scalar, index>
        cils_block_search_serial(const b_matrix &R, const b_vector &y_bar) {
            index ds = cils.d.size(), n_dx_q_0, n_dx_q_1;
            b_vector y_b(cils.n);
            z_hat.clear();
            if (ds == cils.n) {
                //Find the Babai point
                CILS_Babai<scalar, index> babai(cils);
                auto reT = babai.cils_babai_method(R, y_bar);
                z_hat = babai.z_hat;
                return reT;
            }

            scalar sum = 0;
            CILS_SECH_Search<scalar, index> ils(this->cils);
            scalar start = omp_get_wtime();

            if (init == -1) {
                for (index i = 0; i < ds; i++) {
                    n_dx_q_1 = cils.d[i];
                    n_dx_q_0 = i == ds - 1 ? 0 : cils.d[i + 1];

                    for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                        sum = 0;
                        for (index col = n_dx_q_1; col < cils.n; col++) {
                            sum += R(row, col) * z_hat[col];
                        }
                        y_b[row] = y_bar[row] - sum;
                    }
                    if (cils.is_constrained)
                        ils.obils_search(n_dx_q_0, n_dx_q_1, 0, R, y_b, z_hat);
                    else
                        ils.ils_search(n_dx_q_0, n_dx_q_1, 0, R, y_b, z_hat);
                }
            }
            start = omp_get_wtime() - start;
            scalar run_time = omp_get_wtime();

            for (index i = 0; i < ds; i++) {
                n_dx_q_1 = cils.d[i];
                n_dx_q_0 = i == ds - 1 ? 0 : cils.d[i + 1];
                for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                    sum = 0;
                    for (index col = n_dx_q_1; col < cils.n; col++) {
                        sum += R(row, col) * z_hat[col];
                    }
                    y_b[row] = y_bar[row] - sum;
                }

                if (cils.is_constrained)
                    ils.obils_search(n_dx_q_0, n_dx_q_1, 1, R, y_b, z_hat);
                else
                    ils.ils_search(n_dx_q_0, n_dx_q_1, 1, R, y_b, z_hat);
            }
            run_time = omp_get_wtime() - run_time + start;

            returnType<scalar, index> reT = {{run_time - start}, run_time, 0};
            return reT;
        }


        returnType<scalar, index>
        cils_block_search_serial_CPUTEST(const b_matrix &R, const scalar y_bar) {
            index ds = cils.d.size(), n_dx_q_0, n_dx_q_1;
            b_vector y_b(cils.n);

            sd_vector time(2 * ds, 0);
            CILS_SECH_Search<scalar, index> ils(this->cils);
            //special cases:
            if (ds == 1) {
                if (cils.d[0] == 1) {
                    z_hat[0] = round(y_bar[0] / R(0, 0));
                    return {{}, 0, 0};
                } else {
                    for (index i = 0; i < cils.n; i++) {
                        y_b[i] = y_bar[i];
                    }
//                if (cils.is_constrained)
//                    ils_search_obils(0, cils.n, y_b, z_hat);
//                else
//                    ils_search(0, cils.n, y_b, z_hat);
                    return {{}, 0, 0};
                }
            } else if (ds == cils.n) {
                //Find the Babai point
                CILS_Babai<scalar, index> babai(cils);
                auto reT = babai.cils_babai_method(R, y_bar);
                z_hat = babai.z_hat;
                return reT;
            }

            for (index i = 0; i < ds; i++) {
                n_dx_q_1 = cils.d[i];
                n_dx_q_0 = i == ds - 1 ? 0 : cils.d[i + 1];

                for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                    scalar sum = 0;
                    for (index col = n_dx_q_1; col < cils.n; col++) {
                        sum += R(row, col) * z_hat[col];
                    }
                    y_b[row] = y_bar[row] - sum;
                }
            }
//        std::random_shuffle(r_d.begin(), r_d.end());

            scalar start = omp_get_wtime();
            for (index i = 0; i < ds; i++) {
                n_dx_q_1 = cils.d[i];
                n_dx_q_0 = i == ds - 1 ? 0 : cils.d[i + 1];
                index time_index = i;

                time[time_index] = omp_get_wtime();
                if (cils.is_constrained)
                    time[time_index + ds] = ils.obils_search(n_dx_q_0, n_dx_q_1, 1, R, y_b, z_hat);
                else
                    time[time_index + ds] = ils.ils_search(n_dx_q_0, n_dx_q_1, 1, R, y_b, z_hat);

                time[time_index] = omp_get_wtime() - time[time_index];
            }

            scalar run_time = omp_get_wtime() - start;

//        for (index i = 0; i < ds; i++) {
//            printf("%.5f,", time[i]);
//        }

            //Matlab Partial Reduction needs to do the permutation
            returnType<scalar, index> reT = {time, run_time, 0};
            return reT;
        }


        returnType<scalar, index>
        cils_block_search_omp(const b_vector &R, const b_vector &y_bar, const index n_proc,
                              const index nswp, const index init) {
            index ds = cils.d.size();
            if (ds == 1 || ds == cils.n) {
                return cils_block_search_serial(init, cils.d, z_hat);
            }

            auto z_x = z_hat.data();
            index diff = 0, num_iter = 0, flag = 0, temp, R_S_1[ds] = {}, R_S_2[ds] = {};
            index test, row_n, check = 0, r, _nswp = nswp, end = 0;
            index n_dx_q_2, n_dx_q_1, n_dx_q_0;
            scalar sum = 0, start;
            scalar run_time = 0, run_time3 = 0;

            b_vector y_B(cils.n);
            y_B.clear();
            CILS_SECH_Search<scalar, index> _ils(cils.m, cils.n, cils.qam);
//            omp_set_schedule((omp_sched_t) schedule, chunk_size);
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
                n_dx_q_2 = cils.d[0];
                n_dx_q_0 = cils.d[1];

                if (cils.is_constrained)
                    _ils.obils_search_omp(n_dx_q_0, n_dx_q_2, 0, 0, R_A, y_bar, z_x);
                else
                    _ils.ils_search_omp(n_dx_q_0, n_dx_q_2, 0, 0, R_A, y_bar, z_x);

                R_S_2[0] = 1;
                end = 1;
#pragma omp parallel default(shared) num_threads(n_proc) private(n_dx_q_2, n_dx_q_1, n_dx_q_0, sum, temp, check, test, row_n)
                {
                    for (index j = 0; j < _nswp && !flag; j++) {
#pragma omp for schedule(dynamic) nowait
                        for (index i = 1; i < ds; i++) {
                            if (!flag && end <= i) {// !R_S_1[i] && front >= i &&!R_S_1[i] &&
                                n_dx_q_2 = cils.d[i];
                                n_dx_q_0 = i == ds - 1 ? 0 : cils.d[i + 1];
                                check = i == end;
                                row_n = (n_dx_q_0 - 1) * (cils.n - n_dx_q_0 / 2);
                                for (index row = n_dx_q_0; row < n_dx_q_2; row++) {
                                    sum = 0;
                                    row_n += cils.n - row;
                                    for (index col = n_dx_q_2; col < cils.n; col++) {
                                        sum += R_A[col + row_n] * z_x[col];
                                    }
                                    y_B[row] = y_bar[row] - sum;
                                }
                                if (cils.is_constrained)
                                    R_S_2[i] = _ils.obils_search_omp(n_dx_q_0, n_dx_q_2, i, 0, R_A, y_bar, z_x);
                                else
                                    R_S_2[i] = _ils.ils_search_omp(n_dx_q_0, n_dx_q_2, i, 0, R_A, y_B, z_x);
                                if (check) {
                                    end = i + 1;
                                    R_S_2[i] = 1;
                                }
                                diff += R_S_2[i];
//                                if (mode != 0) {
                                flag = ((diff) >= ds - 1) && j > 0;
//                                }
                            }
                        }
                    }
#pragma omp single
                    {
                        if (cils.qam != 1) run_time = omp_get_wtime() - start;
                    };
                }
                if (cils.qam == 1) run_time = omp_get_wtime() - start;
                flag = check = diff = 0;
                _nswp = 3;
            }

            CILS_SECH_Search<scalar, index> ils(this->cils);
            scalar run_time2 = omp_get_wtime();
            n_dx_q_2 = cils.d[0];
            n_dx_q_0 = cils.d[1];

            if (cils.is_constrained)
                ils.obils_search_omp(n_dx_q_0, n_dx_q_2, 0, 1, R_A, y_bar, z_x);
            else
                ils.ils_search_omp(n_dx_q_0, n_dx_q_2, 0, 1, R_A, y_bar, z_x);


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
                            n_dx_q_2 = cils.d[i];
                            n_dx_q_0 = i == ds - 1 ? 0 : cils.d[i + 1];
                            check = i == end;
                            row_n = (n_dx_q_0 - 1) * (cils.n - n_dx_q_0 / 2);
                            for (index row = n_dx_q_0; row < n_dx_q_2; row++) {
                                sum = 0;
                                row_n += cils.n - row;
#pragma omp simd reduction(+:sum)
                                for (index col = n_dx_q_2; col < cils.n; col++) {
                                    sum += R_A[col + row_n] * z_x[col];
                                }
                                y_B[row] = y_bar[row] - sum;
                            }
//                        test = 0;
//                        for (index row = 0; row < i; row++){
//                            test += R_S_1[i];
//                        }
//                        omp_set_lock(&lock[i - 1]);

//                        check = check || R_S_1[i - 1];
                            if (cils.is_constrained)
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
//                            if (mode != 0) {
                            flag = ((diff) >= ds - cils.offset) && j > 0;
//                            }
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
//            matrix_vector_mult<scalar, index, cils.m, cils.n>(Z, z_hat);

            scalar time = 0; //(run_time3 + run_time2) * 0.5;
            if (init == -1) {
                time = cils.qam == 1 ? run_time2 + run_time : run_time2 + run_time;
            } else {
                time = cils.qam == 1 ? run_time2 : run_time2 * 0.5 + run_time3 * 0.5;
            }
//            if (mode == 0)
//                reT = {{run_time3}, time, (scalar) diff + end};
//            else {
            reT = {{run_time3}, time, (scalar) num_iter + 1};
//            cout << "n_proc:" << n_proc << "," << "init:" << init << "," << diff << "," << end << ",Ratio:"
//                 << (index) (run_time2 / run_time3) << "," << run_time << "||";
//            cout.flush();
//            }
            for (index i = 0; i < ds; i++) {
                omp_destroy_lock(&lock[i]);
            }
            return reT;
        }
    };
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
//                            row_n += cils.n - row;
//                            sum = 0;
//                            for (index col = 0; col < i; col++) {
//                                temp = i - col - 1; //Put values backwards
////                                    if (!result[temp]) {
//                                sum2 = 0;
//#pragma omp simd reduction(+ : sum2)
//                                for (index l = n_dx_q_1 + dx * col; l < cils.n - dx * temp; l++) {
//                                    sum2 += R_A[l + row_n] * z_x[l];
//                                }
////                                    R_S[row * ds + temp] = sum2;
//
//                                sum += sum2;
////                                    sum += R_S[row * ds + temp];
//                                test += result[temp];
//                            }
//                            y_B[row] = y_bar[row] - sum;
//                        }
//                    if (first && i > 1) {// front >= i && end <= i!
//                        n_dx_q_2 = cils.d[i];
//                        n_dx_q_0 = i == ds - 1 ? 0 : cils.d[i + 1];
//
////                        index dx = n_dx_q_2 - n_dx_q_0;
////                        n_dx_q_1 = n_dx_q_0 + 16;
//
//                        row_n = (n_dx_q_0 - 1) * (cils.n - n_dx_q_2 / 2);
//                        for (index row = n_dx_q_0; row < n_dx_q_2; row++) {
//                            sum = 0;
//                            row_n += cils.n - row;
//                            for (index col = n_dx_q_2; col < cils.n; col++) {
//                                sum += R_A[col + row_n] * z_x[col];
//                            }
//                            y_B[row] = y_bar[row] - sum;
//                        }
//                        ils.obils_search_omp(n_dx_q_1, n_dx_q_2, i, i == 0, R_A, y_B, z_x);
//
//                        row_n = (n_dx_q_0 - 1) * (cils.n - n_dx_q_0 / 2);
//                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                            sum = 0;
//                            row_n += cils.n - row;
//                            for (index col = n_dx_q_1; col < cils.n; col++) {
//                                sum += R_A[col + row_n] * z_x[col];
//                            }
//                            y_B[row] = y_bar[row] - sum;
//                        }
//                        ils.obils_search_omp(n_dx_q_0, n_dx_q_2, i, 0, R_A, y_B, z_x);
//                    } else
//                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                            row_n += cils.n - row;
//                            sum = 0;
//#pragma omp simd reduction(+ : sum)
//                            for (index col = n_dx_q_1; col < cils.n; col++) {
//                                sum += R_A[row_n + col] * z_x[col];
//                            }
//                            y_B[row] = sum;
//                        }
//#pragma omp simd
//                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                            row_n += cils.n - row;
//                            y_B[row] = 0;
//                            for (index col = n_dx_q_1; col < cils.n; col++) {
//                                y_B[row] += R_A[row_n + col] * z_x[col];
//                            }
//                        }
//                        result[i] = ils_search_obils_omp2(n_dx_q_0, n_dx_q_1, i, ds, y_B, z_x);
//void backup(){
// for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//     row_n += cils.n - row;
//     y_B[row] = 0;
//     for (index block = 0; block < i; block++) {
//         R_S[row * ds + block] = 0; //Put values backwards
//         for (index l = cils.n - dx * (i - block); l < cils.n - dx * (i - block - 1); l++) {
//             y_B[row] += R_A[l + row_n] * z_x[l];
///              R_S[row * ds + block] += R_A[l + row_n] * z_x[l];
//         }
///          y_B[row] += R_S[row * ds + block];
//     }
//     z_p[row] = z_x[row];
// }
//    row_n = (n_dx_q_1 - 1) * (cils.n - n_dx_q_1 / 2);
//    for (index row_k = n_dx_q_1 - 1; row_k >= n_dx_q_0;) {
//        y_B[row_k] = 0;
//        for (index block = 0; block < i; block++) {
//            for (index l = cils.n - dx * (i - block); l < cils.n - dx * (i - block - 1); l++) {
//                y_B[row_k] += R_A[l + row_n] * z_x[l];
////                                R_S[row * ds + block] += R_A[l + row_n] * z_x[l];
//            }
//        }
//        z_p[row_k] = z_x[row_k];
//        c[row_k] = (y_bar[row_k] - y_B[row_k] - sum[row_k]) / R_A[row_n + row_k];
//        temp = round(c[row_k]);
//        z_p[row_k] = temp < 0 ? 0 : temp > upper ? upper : temp;
//        cils.d[row_k] = c[row_k] > z_p[row_k] ? 1 : -1;
//
//        gamma = R_A[row_n + row_k] * (c[row_k] - z_p[row_k]);
//        newprsd = p[row_k] + gamma * gamma;
//
//        if (row_k != n_dx_q_0) {
//            row_k--;
//            row_n -= (cils.n - row_k - 1);
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
        if (ds == 1 || ds == cils.n) {
            return cils_block_search_serial(d_s, z_hat);
        }

        auto z_x = z_hat.data();
        index upper = pow(2, qam) - 1, n_dx_q_0, n_dx_q_1, z_p[cils.n], iter, dflag;
        index result[ds] = {}, diff = 0, info = 0, flag = 0, row_n, row_k, temp;
        scalar y_B[cils.n] = {}, R_S[cils.n * ds] = {}, sum[cils.n] = {};
        scalar p[cils.n] = {}, c[cils.n] = {};
        index cils.d[cils.n] = {}, l[cils.n] = {}, u[cils.n] = {};
        scalar newprsd, gamma = 0, beta = INFINITY;

        scalar run_time = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(1) private(n_dx_q_0, n_dx_q_1, row_n, temp, row_k, newprsd, gamma, beta, iter, dflag)
        {

            for (index j = 0; j < 1 && !flag; j++) {
//#pragma omp for schedule(static, 1) nowait
                for (index i = 0; i < ds; i++) {
//                    if (flag) continue;
                    n_dx_q_0 = cils.n - (i + 1) * dx;
                    n_dx_q_1 = cils.n - i * dx;
                    gamma = 0;
                    dflag = 1;
                    beta = INFINITY;
                    row_n = (n_dx_q_1 - 1) * (cils.n - n_dx_q_1 / 2);
                    row_k = n_dx_q_1 - 1;for (; row_k >= n_dx_q_0;) {
                        y_B[row_k] = 0;
                        for (index block = 0; block < i; block++) {
                            for (index h = cils.n - dx * (i - block); h < cils.n - dx * (i - block - 1); h++) {
                                y_B[row_k] += R_A[h + row_n] * z_x[h];
//                                R_S[row * ds + block] += R_A[l + row_n] * z_x[l];
                            }
                        }
                        z_p[row_k] = z_x[row_k];
                        c[row_k] = (y_bar[row_k] - y_B[row_k] - sum[row_k]) / R_A[row_n + row_k];
                        z_p[row_k] = round(c[row_k]);
                        if (z_p[row_k] <= 0) {
                            z_p[row_k] = u[row_k] = 0; //The lower bound is reached
                            l[row_k] = cils.d[row_k] = 1;
                        } else if (z_p[row_k] >= upper) {
                            z_p[row_k] = upper; //The upper bound is reached
                            u[row_k] = 1;
                            l[row_k] = 0;
                            cils.d[row_k] = -1;
                        } else {
                            l[row_k] = u[row_k] = 0;
                            //  Determine enumeration direction at level block_size
                            cils.d[row_k] = c[row_k] > z_p[row_k] ? 1 : -1;
                        }    gamma = R_A[row_n + row_k] * (c[row_k] - z_p[row_k]);
                        newprsd = p[row_k] + gamma * gamma;    if (row_k != n_dx_q_0) {
                            row_k--;
                            row_n -= (cils.n - row_k - 1);
                            sum[row_k] = 0;
                            for (index col = row_k + 1; col < n_dx_q_1; col++) {
                                sum[row_k] += R_A[row_n + col] * z_p[col];
                            }
                            p[row_k] = newprsd;
                        } else {
                            break;
                        }
                    }

//                    row_n = (n_dx_q_1 - 1) * (cils.n - n_dx_q_1 / 2);
//                    row_k = n_dx_q_1 - 1;
//                    gamma = 0;
//                    dflag = 1;
//                    beta = INFINITY;

//                    for (index count = 0; count < max_search || iter == 0; count++) {
                    while (true) {
                        if (dflag) {
                            newprsd = p[row_k] + gamma * gamma;
                            if (newprsd < beta) {
                                if (row_k != n_dx_q_0) {
                                    row_k--;
                                    row_n -= (cils.n - row_k - 1);
                                    p[row_k] = newprsd;
                                    c[row_k] = (y_bar[row_k] - y_B[row_k] - sum[row_k]) / R_A[row_n + row_k];
                                    z_p[row_k] = round(c[row_k]);
                                    if (z_p[row_k] <= 0) {
                                        z_p[row_k] = u[row_k] = 0;
                                        l[row_k] = cils.d[row_k] = 1;
                                    } else if (z_p[row_k] >= upper) {
                                        z_p[row_k] = upper;
                                        u[row_k] = 1;
                                        l[row_k] = 0;
                                        cils.d[row_k] = -1;
                                    } else {
                                        l[row_k] = u[row_k] = 0;
                                        cils.d[row_k] = c[row_k] > z_p[row_k] ? 1 : -1;
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

//                                    if (iter > search_iter || diff == dx) {
//                                        break;
//                                    }
                                }
                            } else {
                                dflag = 0;
                            }    } else {
                            if (row_k == n_dx_q_1 - 1) break;
                            else {
                                row_k++;
                                row_n += cils.n - row_k;
                                if (l[row_k] != 1 || u[row_k] != 1) {
                                    z_p[row_k] += cils.d[row_k];
                                    sum[row_k] += R_A[row_n + row_k] * cils.d[row_k];
                                    if (z_p[row_k] == 0) {
                                        l[row_k] = 1;
                                        cils.d[row_k] = -cils.d[row_k] + 1;
                                    } else if (z_p[row_k] == upper) {
                                        u[row_k] = 1;
                                        cils.d[row_k] = -cils.d[row_k] - 1;
                                    } else if (l[row_k] == 1) {
                                        cils.d[row_k] = 1;
                                    } else if (u[row_k] == 1) {
                                        cils.d[row_k] = -1;
                                    } else {
                                        cils.d[row_k] = cils.d[row_k] > 0 ? -cils.d[row_k] - 1 : -cils.d[row_k] + 1;
                                    }
                                    gamma = R_A[row_n + row_k] * (c[row_k] - z_p[row_k]);
                                    dflag = 1;
                                }
                            }
                        }
                    }
//ILS search process
//                        for (index count = 0; count < max_search || iter == 0; count++) {
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
//                                        row_n = (cils.n * temp) - ((temp * (temp + 1)) / 2);
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
//                                flag = diff >= ds - offset;
//
//                            if (!result[i] && !flag) {
//                                for (index row = 0; row < ds - i - 1; row++) {
//                                    for (index h = 0; h < dx; h++) {
//                                        temp = row * dx + h;
//                                        sum = 0;
//                                        row_n = (cils.n * temp) - ((temp * (temp + 1)) / 2);
//#pragma omp simd reduction(+:sum)
//                                        for (index col = n_dx_q_0; col < n_dx_q_1; col++) {
////                                  R_S[temp * ds + i] += R[temp + cils.n * col] * z_x[col];
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
            reT = {z_hat, run_time2, diff};
        else {
            reT = {z_hat, run_time2, info};
            cout << "diff:" << diff << ", ";
        }
        return reT;
    }
 */