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

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_babai_search_omp(const index n_proc, const index nswp,
                                                  vector<index> *z_B) {

        index num_iter = 0, s = n_proc, x_min = 0, ni, nj, diff, upper = pow(2, qam) - 1;
        bool flag = false, check = false;
        auto z_x = z_B->data();
        auto z_p = new index[n]();
        scalar sum = 0, result;

        scalar start = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(check, result, sum, diff, ni, nj)
        {
            for (index j = 0; j < nswp && !flag; j++) {
#pragma omp for schedule(dynamic) nowait
                for (index i = 0; i < n; i++) {
                    if (flag) continue; //
                    sum = 0;
                    ni = n - 1 - i;
                    nj = ni * n - (ni * (n - i)) / 2;

#pragma omp simd reduction(+ : sum)
                    for (index col = n - i; col < n; col++) {
                        sum += R_A->x[nj + col] * z_x[col];
                    }
                    result = round((y_A->x[ni] - sum) / R_A->x[nj + ni]);
                    z_x[ni] = !is_constrained ? result : result < 0 ? 0 : result > upper ? upper : result;

                    if (i == n - 1)
                        check = true;
                }
                if (j > 0 && check) {
                    num_iter = j;
                    check = false;
                    diff = 0;
#pragma omp simd reduction(+ : diff)
                    for (index l = 0; l < n; l++) {
                        diff += z_x[l] == z_p[l];
                        z_p[l] = z_x[l];
                    }
                    flag = diff > stop;
                }
            }
        }
        scalar run_time = omp_get_wtime() - start;
#pragma parallel omp cancellation point
#pragma omp flush
        delete[] z_p;
        //Matlab Partial Reduction needs to do the permutation
        if(is_matlab)
            vector_permutation<scalar, index, n>(Z, z_B);
        returnType<scalar, index> reT = {{}, run_time, num_iter};
        return reT;
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_babai_search_serial(vector<index> *z_B) {
        scalar sum = 0, upper = pow(2, qam) - 1;
        scalar start = omp_get_wtime();

        index result = round(y_A->x[n - 1] / R->x[n * n - 1]);
        z_B->at(n - 1) = result < 0 ? 0 : result > upper ? upper : result;

        for (index i = 1; i < n; i++) {
            index k = n - i - 1;
            for (index col = n - i; col < n; col++) {
                sum += R->x[col + k * n] * z_B->at(col);
//                sum += R->x[col * n + k] * z_B->at(col);
            }
            result = round((y_A->x[k] - sum) / R->x[k * n + k]);
            z_B->at(k) = !is_constrained ? result : result < 0 ? 0 : result > upper ? upper : result;
            sum = 0;
        }
        scalar run_time = omp_get_wtime() - start;
        //Matlab Partial Reduction needs to do the permutation
        if(is_matlab)
            vector_permutation<scalar, index, n>(Z, z_B);
        returnType<scalar, index> reT = {{}, run_time, 0};
        return reT;
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_back_solve(vector<index> *z_B) {
        scalar sum = 0, upper = pow(2, qam) - 1;
        vector<scalar> z_B_tmp(n, 0);

        scalar start = omp_get_wtime();
        index result = round(y_A->x[n - 1] / R->x[n * n - 1]);
        z_B->at(n - 1) = result < 0 ? 0 : result > upper ? upper : result;
        for (index i = 1; i < n; i++) {
            index k = n - i - 1;
            for (index col = n - i; col < n; col++) {
                sum += R->x[col + k * n] * z_B->at(col);
            }
            result = round((y_A->x[k] - sum) / R->x[k * n + k]);
            z_B->at(k) = !is_constrained ? result : result < 0 ? 0 : result > upper ? upper : result;
            sum = 0;
        }

//        for (index i = 0; i < n; i++) {
//            z_B->at(i) = round(z_B_tmp[i]) < 0 ? 0 : round(z_B_tmp[i]) > upper ? upper : round(z_B_tmp[i]);
//        }

        scalar run_time = omp_get_wtime() - start;
        //Matlab Partial Reduction needs to do the permutation
//        if(is_matlab)
//            vector_permutation<scalar, index, n>(Z, z_B);

        returnType<scalar, index> reT = {{}, run_time, 0};
        return reT;
    }
}