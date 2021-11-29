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
#include "../include/CILS.h"

namespace cils {

    template<typename scalar, typename index>
    class CILS_Babai {
    private:
        CILS <scalar, index> cils;
    public:
        b_vector z_hat;

        CILS_Babai(CILS <scalar, index> &cils) {
            this->cils = cils;
            this->z_hat.resize(cils.n);
            this->z_hat.clear();
        }

        returnType <scalar, index>
        cils_babai_method_omp(const b_matrix &R, const b_vector &y_bar, const index n_proc, const index nswp) {

            index num_iter = 0, idx = 0, end = 1, ni, nj, diff = 0, i = cils.n - 1, j, c_i;
            index z_p[cils.n] = {}, result[cils.n] = {};
            bool flag = false, check = false;
            z_hat.clear();
            scalar sum = 0, run_time;
            b_vector R_A(cils.n / 2 * (1 + cils.n));
            for (index row = 0; row < R.size1(); row++) {
                for (index col = row; col < R.size2(); col++) {
                    R_A[idx] = R(row, col);
                    idx++;
                }
            }

//            omp_set_schedule((omp_sched_t) this->schedule, this->chunk_size);
#pragma omp parallel default(shared) num_threads(n_proc)
            {}
            scalar start = omp_get_wtime();
            c_i = round(y_bar[i] / R_A[cils.n / 2 * (1 + cils.n) - 1]);
            z_hat[i] = c_i < 0 ? 0 : c_i > cils.upper ? cils.upper : c_i;
            result[i] = 1;

#pragma omp parallel default(shared) num_threads(n_proc) private(check, sum, i, j, ni, nj, c_i)
            {
                for (index t = 0; t < nswp && !flag; t++) {
#pragma omp for schedule(dynamic) nowait
                    for (ni = 1; ni < cils.n; ni++) {
                        if (!flag && !result[ni]) {
                            i = cils.n - ni - 1;
                            nj = i * cils.n - (i * (i + 1)) / 2;

#pragma omp simd reduction(+ : sum)
                            for (index col = cils.n - ni; col < cils.n; col++) {
                                sum += R_A[nj + col] * z_hat[col];
                            }
                            c_i = round((y_bar[i] - sum) / R_A[nj + i]);
                            z_p[i] = z_hat[i];
                            z_hat[i] = !cils.is_constrained ? c_i : c_i < 0 ? 0 : c_i > cils.upper ? cils.upper : c_i;
                            result[ni] = t > 2 && z_p[i] == z_hat[i];
                            sum = 0;
                        }
                    }
#pragma omp single
                    {
                        if (t > 2) {
                            num_iter = t;
                            diff = 0;
#pragma omp simd reduction(+ : diff)
                            for (index l = 0; l < cils.n; l++) {
                                diff += result[i];
                            }
                            flag = diff >= cils.n;
                        }
                    }
                }
#pragma omp single
                {
                    run_time = omp_get_wtime() - start;
                };
            }
            scalar run_time2 = omp_get_wtime() - start;
            cout << num_iter << "," << diff << "," << run_time << ",";
            returnType<scalar, index> reT = {{}, run_time2, num_iter};
            return reT;
        }

        /**
         * Ordinary/Box-Constrained Nearest Plane algorithm
         * Description:
         *  cils_babai_method(R,y)
         *
         * @param R: n-by-n Reduced upper triangular matrix
         * @param y_bar n-dimensional reduced real vector
         * @return returnType: ~, time, ~
         *  Main Reference:
         *  Lin, S. Thesis.
         *  Authors: Lin, Shilei
         *  Copyright (c) 2021. Scientific Computing Lab, McGill University.
         *  Dec 2021. Last revision: Dec 2021
         */
        returnType <scalar, index>
        cils_babai_method(const b_matrix &R, const b_vector &y_bar) {
            scalar sum = 0;
            z_hat.clear();
            scalar time = omp_get_wtime();
            for (index i = cils.n - 1; i >= 0; i--) {
                for (index j = i + 1; j < cils.n; j++) {
                    sum += R(i, j) * z_hat[j];
                }
                scalar c_i = round((y_bar[i] - sum) / R(i, i));
                z_hat[i] = !cils.is_constrained ? c_i : c_i < 0 ? 0 : c_i > cils.upper ? cils.upper : c_i;
                sum = 0;
            }
            time = omp_get_wtime() - time;
            return {{}, time, 0};
        }

        returnType <scalar, index>
        cils_backward_triangular_solve(const b_matrix &R, const b_vector &y_bar) {
            scalar sum = 0;
            scalar time = omp_get_wtime();
            for (index i = cils.n - 1; i >= 0; i--) {
                for (index j = i + 1; j < cils.n; j++) {
                    sum += R(i, j) * z_hat[j];
                }
                z_hat[i] = (y_bar[i] - sum) / R(i, i);
                sum = 0;
            }
            time = omp_get_wtime() - time;
            return {{}, time, 0};
        }
    };
}