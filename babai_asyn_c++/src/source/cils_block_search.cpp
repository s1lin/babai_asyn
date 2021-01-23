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
    cils<scalar, index, n>::cils_block_search_serial(const vector<index> *d, vector<index> *z_B, bool is_constrained) {

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
//                else
//                    ils_search(0, n, &y_b, z_B);
                return {z_B, 0, 0};
            }
        } else if (ds == n) {
            //Find the Babai point
            return cils_babai_search_serial(z_B, is_constrained);
        }
        scalar start = omp_get_wtime();
        for (index i = 0; i < n; i++) {
            y_b[i] = y_A->x[i];
        }
        for (index i = 0; i < ds; i++) {
            n_dx_q_0 = i == 0 ? n - dx : n - d->at(ds - 1 - i);
            n_dx_q_1 = i == 0 ? n : n - d->at(ds - i);

            for (index col = n_dx_q_1; col < n; col++) {
                for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                    y_b[row] = y_b[row] - R->x[col * n + row] * z_B->at(col);
                }
            }
//            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                scalar sum = 0;
//                index row_n = (n * row) - ((row * (row + 1)) / 2);
//                for (index col = n_dx_q_1; col < n; col++) {
//                    sum += R_A->x[row_n + col] * z_B->at(col);
//                }
//                y_b[row] = y_A->x[row] - sum;
//            }
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
                                                  const index stop, const index init,
                                                  const vector<index> *d_s,
                                                  vector<index> *z_B,
                                                  const bool is_constrained) {
        index ds = d_s->size(), dx = d_s->at(ds - 1);
        if (ds == 1 || ds == n) {
            return cils_block_search_serial(d_s, z_B, is_constrained);
        }

        auto z_x = z_B->data();
        bool flag = false;
        index num_iter = 0, n_dx_q_0, n_dx_q_1, row_n, iter = 2 * n_proc, upper = pow(2, qam) - 1;
        scalar sum = 0, result = 0, y_b[n];
        index z_p[n], diff[nswp];

        scalar run_time = omp_get_wtime();

#pragma omp parallel default(shared) num_threads(n_proc) private(sum, result, row_n, n_dx_q_0, n_dx_q_1)
        {
            if (init != -1)
#pragma omp for schedule(dynamic) nowait
                for (index i = 0; i < nswp; i++) {
                    sum = 0;
                    n_dx_q_0 = n - 1 - i;
                    n_dx_q_1 = n_dx_q_0 * n - (n_dx_q_0 * (n - i)) / 2;
#pragma omp simd reduction(+ : sum)
                    for (index col = n - i; col < n; col++)
                        sum += R_A->x[n_dx_q_1 + col] * z_x[col];
                    result = round((y_A->x[n_dx_q_0] - sum) / R_A->x[n_dx_q_0 + n_dx_q_1]);
                    z_x[n_dx_q_0] = !is_constrained ? result : result < 0 ? 0 : result > upper ? upper : result;
                    z_p[n_dx_q_0] = z_x[n_dx_q_0];
                }

            for (index j = 0; j < nswp && !flag; j++) {
#pragma omp for schedule(dynamic) nowait
                for (index i = 0; i < ds; i++) {
                    if (flag) continue;
                    n_dx_q_0 = n - (i + 1) * dx;
                    n_dx_q_1 = n - i * dx;
                    //The block operation
                    for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                        sum = 0;
                        row_n = (n * row) - ((row * (row + 1)) / 2);
#pragma omp simd reduction(+ : sum)
                        for (index col = n_dx_q_1; col < n; col++) {
                            sum += R_A->x[row_n + col] * z_x[col];
                        }
                        y_b[row] = y_A->x[row] - sum;
                    }
                    if (is_constrained)
                        ils_search_obils_omp(n_dx_q_0, n_dx_q_1, y_b, z_x);
                    else
                        ils_search_omp(n_dx_q_0, n_dx_q_1, y_b, z_x);
                    if (i == ds - 1) {
                        num_iter = j;
                        sum = 0;
#pragma omp simd reduction(+ : sum)
                        for (index l = 0; l < n; l++) {
                            sum += z_x[l] == z_p[l];
                            z_p[l] = z_x[l];
                        }
                        if (mode != 0 && !flag) {
                            diff[j] = sum;
                            flag = sum > stop;
#pragma omp flush (flag)
                        }
                    }
                }
            }
        }

        run_time = omp_get_wtime() - run_time;

#pragma parallel omp cancellation point
#pragma omp flush
        returnType<scalar, index> reT = {z_B, run_time, num_iter};
        return reT;
    }
}
