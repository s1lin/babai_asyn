/** \file
 * \brief Computation of integer least square problem by constrained non-block Babai Estimator
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

namespace cils {

    template<typename scalar, typename index, index m, index n>
    returnType <scalar, index>
    cils<scalar, index, m, n>::cils_babai_search_omp(const index n_proc, const index nswp, vector<scalar> *z_B) {

        index num_iter = 0, end = 1, x_min = 0, ni, nj, diff = 0, z_p[n] = {}, result[n] = {};
        bool flag = false, check = false;
        auto z_x = z_B->data();
        scalar sum = 0, r, run_time;
        omp_set_schedule((omp_sched_t) schedule, chunk_size);

        scalar start = omp_get_wtime();
        r = round(y_r[n - 1] / R_A[n * (n + 1) / 2 - 1]);
        z_x[n - 1] = r < 0 ? 0 : r > upper ? upper : r;
        result[n - 1] = 1;
#pragma omp parallel default(shared) num_threads(n_proc) private(check, r, sum, ni, nj)
        {
            for (index j = 0; j < nswp && !flag; j++) {
#pragma omp for schedule(runtime) nowait
                for (index i = 1; i < n; i++) {
                    ni = n - 1 - i;
                    if (!flag) {
                        sum = 0;
                        nj = ni * n - (ni * (n - i)) / 2;

#pragma omp simd reduction(+ : sum)
                        for (index col = n - i; col < n; col++) {
                            sum += R_A[nj + col] * z_x[col];
                        }
                        r = round((y_r[ni] - sum) / R_A[nj + ni]);
                        z_x[ni] = !is_constrained ? r : r < 0 ? 0 : r > upper ? upper : r;
                        result[ni] = z_x[ni] == z_p[ni] && j > 2;
                        diff += result[ni];
                        z_p[ni] = z_x[ni];
                    }
                }
                if (j > 2) {
                    num_iter = j;
                    diff = 0;
#pragma omp simd reduction(+ : diff)
                    for (index l = 0; l < n; l++) {
                        diff += result[l];
                    }
                    flag = diff > n - 10;
                }
                if (flag) break;
            }
#pragma omp single
            {
                run_time = omp_get_wtime() - start;
            };
        }
        scalar run_time2 = omp_get_wtime() - start;
#pragma parallel omp cancellation point
#pragma omp flush
        //Matlab Partial Reduction needs to do the permutation
//        if (is_matlab)
//        vector_permutation<scalar, index, m,  n>(Z, z_B);
        cout << end << "," << diff << "," << (index) run_time2 / run_time << ",";
        returnType<scalar, index> reT = {{}, run_time, num_iter};
        return reT;
    }

    template<typename scalar, typename index, index m, index n>
    returnType <scalar, index>
    cils<scalar, index, m, n>::cils_babai_search_serial(vector<scalar> *z_B) {
        scalar sum = 0;
        scalar start = omp_get_wtime();

        index result = round(y_r[n - 1] / R_R[n * n - 1]);
        z_B->at(n - 1) = result < 0 ? 0 : result > upper ? upper : result;

        for (index i = 1; i < n; i++) {
            index k = n - i - 1;
            for (index col = n - i; col < n; col++) {
                sum += R_R[col * n + k] * z_B->at(col);
//                sum += R_R[col * n + k] * z_B[col);
            }
            result = round((y_r[k] - sum) / R_R[k * n + k]);
            z_B->at(k) = !is_constrained ? result : result < 0 ? 0 : result > upper ? upper : result;
            sum = 0;
        }
        scalar run_time = omp_get_wtime() - start;
        //Matlab Partial Reduction needs to do the permutation
//        if (is_matlab)
        vector_permutation<scalar, index, m, n>(Z, z_B);
        returnType<scalar, index> reT = {{}, run_time, 0};
        return reT;
    }

    template<typename scalar, typename index, index m, index n>
    returnType <scalar, index>
    cils<scalar, index, m, n>::cils_back_solve(array<scalar, n> &z_B) {
        scalar sum = 0;

        scalar start = omp_get_wtime();

        z_B[n - 1] = y_r[n - 1] / R_A[((n - 1) * n) / 2 + n - 1];
        for (index i = 1; i < n; i++) {
            index k = n - i - 1;
            for (index col = n - i; col < n; col++) {
                sum += R_R[col * n + k] * z_B[col];
            }
            z_B[k] = (y_r[k] - sum) / R_R[k * n + k];
            sum = 0;
        }

//        for (index i = 0; i < n; i++) {
//            z_B[i) = round(z_B_tmp[i]) < 0 ? 0 : round(z_B_tmp[i]) > upper ? upper : round(z_B_tmp[i]);
//        }

        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {{}, run_time, 0};
        return reT;
    }

    template<typename scalar, typename index, index m, index n>
    returnType <scalar, index>
    cils<scalar, index, m, n>::cils_back_solve_omp(const index n_proc, const index nswp,
                                                   vector<scalar> *z_B) {

        index s = n_proc, x_min = 0, ni, nj, diff;
        bool flag = false, check = false;
        auto z_x = z_B->data();
        scalar z_p[n], sum = 0, result, num_iter = 0;

        scalar start = omp_get_wtime();
        z_x[n - 1] = y_r[n - 1] / R_A[((n - 1) * n) / 2 + n - 1];
#pragma omp parallel default(shared) num_threads(n_proc) private(check, result, sum, diff, ni, nj)
        {
            for (index j = 0; j < nswp; j++) { // && !flag
#pragma omp for schedule(dynamic) nowait
                for (index i = 1; i < n; i++) {
//                    if (flag) continue; //
                    sum = 0;
                    ni = n - 1 - i;
                    nj = ni * n - (ni * (n - i)) / 2;

#pragma omp simd reduction(+ : sum)
                    for (index col = n - i; col < n; col++) {
                        sum += R_A[nj + col] * z_x[col];
                    }
                    z_x[ni] = (y_r[ni] - sum) / R_A[nj + ni];
//                    if (i == n - 1)
//                        check = true;
                }
//                if (j > 0 && check) {
//                    num_iter = j;
//                    check = false;
//                    diff = 0;
//#pragma omp simd reduction(+ : diff)
//                    for (index l = 0; l < n; l++) {
//                        diff += z_x[l] == z_p[l];
//                        z_p[l] = z_x[l];
//                    }
//                    flag = diff > n - stop;
//                }
            }
        }
        scalar run_time = omp_get_wtime() - start;
#pragma parallel omp cancellation point
#pragma omp flush
        vector_permutation<scalar, index, m, n>(Z, z_B);
        returnType<scalar, index> reT = {{}, run_time, num_iter};
        return reT;
    }

}