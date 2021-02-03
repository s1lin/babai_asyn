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
    cils<scalar, index, n>::cils_block_search_omp(const index n_proc, const index nswp,
                                                  const vector<index> *d_s, vector<index> *z_B) {
        index ds = d_s->size(), dx = d_s->at(ds - 1);
        if (ds == 1 || ds == n) {
            return cils_block_search_serial(d_s, z_B);
        }

        auto z_x = z_B->data();
        index upper = pow(2, qam) - 1, n_dx_q_0, n_dx_q_1, z_p[n];
        index result[ds] = {}, diff = {}, num_iter = 0, flag = 0, row_n, temp;
        scalar y_B[n] = {}, R_S[n * ds] = {}, sum = 0;
        scalar run_time3;

        for (index i = 0; i < ds; i++) {
            n_dx_q_0 = n - (i + 1) * dx;
            n_dx_q_1 = n - i * dx;
            temp = (n_dx_q_0 - 1) * (n - n_dx_q_0 / 2);
//            cout << n_dx_q_0 << " " << n_dx_q_1 << " " << i << ": ";
            for (index col = 0; col < i; col++) {
                row_n = temp;
                for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                    R_S[row * ds + i - col - 1] = 0; //Put values backwards
                    row_n += n - row;
                    for (index l = n - dx * (i - col); l < n - dx * (i - col - 1); l++) {
                        R_S[row * ds + i - col - 1] += R_A->x[l + row_n] * z_x[l];
                    }
                }
            }
        }

        scalar run_time = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(n_dx_q_0, n_dx_q_1, row_n, sum, temp)
        {
            for (index j = 0; j < nswp && !flag; j++) {
#pragma omp for schedule(static, 1) nowait
                for (index i = 0; i < ds; i++) {
                    if (flag) continue;
                    if (!flag && !result[i]) {
                        n_dx_q_0 = n - (i + 1) * dx;
                        n_dx_q_1 = n - i * dx;
//                        temp = (n_dx_q_0 - 1) * (n - n_dx_q_0 / 2);

//                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                            y_B[row] = 0;
//                            for (index col = 0; col < i; col++)
//                                y_B[row] += R_S[row * ds + col];
//                        }

                        result[i] = ils_search_obils_omp(n_dx_q_0, n_dx_q_1, i, ds, R_S, z_x);
                        if (!flag) {
                            diff += result[i];
                            num_iter = j;
                            if (mode != 0)
                                flag = diff >= ds - stop;
                        }
//                        if (!result[i]) {
//                            for (index row = 0; row < ds - i - 1; row++) {
//                                for (index h = 0; h < dx; h++) {
//                                    temp = row * dx + h;
//                                    R_S[temp * ds + i] = 0;
//                                    row_n = (n * temp) - ((temp * (temp + 1)) / 2);
//                                    for (index col = n_dx_q_0; col < n_dx_q_1; col++) {
////                                  R_S[temp * ds + i] += R->x[temp + n * col] * z_x[col];
//                                        R_S[temp * ds + i] += R_A->x[row_n + col] * z_x[col];
//                                    }
//                                }
//                            }
//                        }

                    }
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
}
