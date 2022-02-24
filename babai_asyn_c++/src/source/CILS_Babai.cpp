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
        cils_babai_omp(const b_matrix &R, const b_vector &y_bar, const index n_t, const index nstep,
                       const index init) {

            index num_iter = 0, idx = 0, end = 1, ni, nj, diff = 0, i = cils.n - 1, j, c_i;
            index z_p[cils.n] = {}, delta[cils.n] = {};
            bool flag = false, check = false;
            sd_vector ber(1, 0);

//            z_hat.clear();
//            helper::display_vector<scalar, index>(z_hat.size(), z_hat, "x_ser");

            scalar sum = 0, run_time;
//            b_vector R_A(cils.n / 2 * (1 + cils.n));
//            for (index row = 0; row < R.size1(); row++) {
//                for (index col = row; col < R.size2(); col++) {
//                    R_A[idx] = R(row, col);
//                    idx++;
//                }
//            }
#pragma omp parallel default(shared) num_threads(n_t)
            {}

            scalar start = omp_get_wtime();
            if (nstep != 0) {
//            omp_set_schedule((omp_sched_t) this->schedule, this->chunk_size);

                c_i = round(y_bar[i] / R(i, i));//R_A[cils.n / 2 * (1 + cils.n) - 1]);
                z_hat[i] = !cils.is_constrained ? c_i : max(min((index) c_i, cils.upper), 0);
                delta[i] = 1;

#pragma omp parallel default(shared) num_threads(n_t) private(check, sum, i, j, ni, nj, c_i)
                {
                    for (index t = 0; t < nstep && !flag; t++) {
#pragma omp for schedule(dynamic, 1) nowait
                        for (ni = 1; ni < cils.n; ni++) {
                            if (!flag && !delta[ni]) {//
                                i = cils.n - ni - 1;
//                                nj = i * cils.n - (i * (i + 1)) / 2;

//#pragma omp simd reduction(+ : sum)
//                                for (index col = cils.n - ni; col < cils.n; col++) {
//                                    sum += R_A[nj + col] * z_hat[col];
//                                }
                                for (index col = cils.n - 1; col >= cils.n - ni; col--) {
                                    sum += R(i, col) * z_hat[col];
                                }
//                                for (j = cils.n - ni; j < cils.n; j++) {
//                                    sum += R(i, j) * z_hat[j];
//                                }

//                                c_i = round((y_bar[i] - sum) / R(i, i));

                                c_i = round((y_bar[i] - sum) / R(i, i));//R_A[nj + i]);
                                z_p[i] = z_hat[i];
                                z_hat[i] = !cils.is_constrained ? c_i : max(min((index) c_i, cils.upper), 0);
                                delta[ni] = t > 2 && z_p[i] == z_hat[i];
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
                                    diff += delta[l];
                                }
                                flag = diff >= cils.n - 5;
                            }
                        }
                    }

                }
            }

            scalar run_time2 = omp_get_wtime() - start;
            returnType<scalar, index> reT = {ber, run_time2, num_iter};
            return reT;
        }

        returnType <scalar, index>
        cils_babai_oomp(const b_matrix &R, const b_vector &y_bar, const index n_t, const index nstep,
                        const index init) {

            index num_iter = 0, idx = 0, end = 1, ni, nj, diff = 0, i = cils.n - 1, j, c_i;
            index z_p[cils.n] = {}, count[nstep] = {}, delta[nstep] = {};
            bool flag = false;
            sd_vector ber(1, 0);

//            z_hat.clear();
//            helper::display_vector<scalar, index>(z_hat.size(), z_hat, "x_ser");

            scalar sum = 0;
            b_vector R_A(cils.n / 2 * (1 + cils.n));


            for (index row = 0; row < R.size1(); row++) {
                for (index col = row; col < R.size2(); col++) {
                    R_A[idx] = R(row, col);
                    idx++;
                }
            }

#pragma omp parallel default(shared) num_threads(n_t)
            {}

            scalar run_time = omp_get_wtime();
            if (nstep != 0) {
//            omp_set_schedule((omp_sched_t) this->schedule, this->chunk_size);

                c_i = round(y_bar[i] / R_A[cils.n / 2 * (1 + cils.n) - 1]);//R_A[cils.n / 2 * (1 + cils.n) - 1]);
                z_hat[i] = !cils.is_constrained ? c_i : max(min((index) c_i, cils.upper), 0);
                idx = cils.n - 2;
#pragma omp parallel default(shared) num_threads(n_t) private(sum, i, j, ni, nj, c_i)
                {
                    for (index t = 0; t < nstep && !flag; t++) {
#pragma omp for schedule(dynamic, 1) nowait
                        for (ni = 1; ni < cils.n; ni++) {
                            i = cils.n - ni - 1;
                            if (!flag && i <= idx) {
                                nj = i * cils.n - (i * (i + 1)) / 2;
                                sum = y_bar[i];
#pragma omp simd reduction(- : sum)
                                for (index col = cils.n - 1; col >= cils.n - ni; col--) {
                                    sum -= R_A[nj + col] * z_hat[col];
                                }
                                c_i = round(sum / R_A[nj + i]);
                                z_p[i] = z_hat[i];
                                z_hat[i] = !cils.is_constrained ? c_i : max(min((index) c_i, cils.upper), 0);
#pragma omp atomic
                                delta[t] += z_p[i] == z_hat[i];
                                if (idx == i) idx--;
                            }
#pragma omp atomic
                            count[t]++;
                        }
                        if(!flag) {
                            flag = (delta[t] >= cils.n * 0.6 || idx <= n_t) && count[t] == cils.n - 1;
                            num_iter = t;
                        }
                    }
                }
            }

            run_time = omp_get_wtime() - run_time;
            returnType<scalar, index> reT = {ber, run_time, num_iter};
            return reT;
        }

        /**
         * Ordinary/Box-Constrained Nearest Plane algorithm
         * Description:
         *  cils_babai(R,y)
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
        cils_babai(const b_matrix &R, const b_vector &y_bar) {
            scalar sum = 0;
//            z_hat.clear();
            scalar time = omp_get_wtime();
            for (index i = cils.n - 1; i >= 0; i--) {
                for (index j = i + 1; j < cils.n; j++) {
                    sum += R(i, j) * z_hat[j];
                }
                scalar c_i = round((y_bar[i] - sum) / R(i, i));
                z_hat[i] = !cils.is_constrained ? c_i : max(min((index) c_i, cils.upper), 0);
                sum = 0;
            }
            time = omp_get_wtime() - time;
            return {{}, time, 0};
        }

        returnType <scalar, index>
        cils_babai_sic(const b_matrix &A, const b_vector &y, const index nstep) {
            b_vector sum(cils.n, 0), z(z_hat);
            b_vector a_t(cils.n, 0), y_bar(y);
            for (index i = 0; i < cils.n; i++) {
                b_vector a = column(A, i);
                a_t[i] = inner_prod(a, a);
            }
            index t = 0;
            scalar time = omp_get_wtime();
            for (t = 0; t < nstep; t++) {
                index diff = 0;
                for (index i = cils.n - 1; i >= 0; i--) {
                    for (index j = 0; j < i; j++) {
                        sum += column(A, j) * z_hat[j];
                    }
                    for (index j = i + 1; j < cils.n; j++) {
                        sum += column(A, j) * z_hat[j];
                    }
                    y_bar = y - sum;
                    scalar c_i = round(inner_prod(column(A, i), y_bar) / a_t[i]);
                    z_hat[i] = !cils.is_constrained ? c_i : max(min((index) c_i, cils.upper), 0);
                    diff += z_hat[i] - z[i];
                    sum.clear();
                }
                if (diff == 0) {
                    cout << "t:" << t << " ";
                    cout.flush();
//                    break;
                } else {
                    z.assign(z_hat);
                }
            }

            time = omp_get_wtime() - time;
            return {{}, time, t};
        }

        returnType <scalar, index>
        cils_babai_sic_omp(const b_matrix &A, const b_vector &y, const index nstep, const index n_t) {
            b_vector sum(cils.n, 0), z(z_hat);
            b_vector a_t(cils.n, 0), y_bar(y);
            index diff[nstep], flag = 0, num_iter = 0;
            for (index i = 0; i < cils.n; i++) {
                b_vector a = column(A, i);
                a_t[i] = inner_prod(a, a);
            }

            scalar time = omp_get_wtime();

#pragma omp parallel default(shared) num_threads(n_t) firstprivate(sum, y_bar)
            {
                for (index t = 0; t < nstep && !flag; t++) {
#pragma omp for schedule(dynamic) nowait
                    for (index i = cils.n - 1; i >= 0; i--) {
                        for (index j = 0; j < i; j++) {
                            sum += column(A, j) * z_hat[j];
                        }
                        for (index j = i + 1; j < cils.n; j++) {
                            sum += column(A, j) * z_hat[j];
                        }
                        y_bar = y - sum;
                        scalar c_i = round(inner_prod(column(A, i), y_bar) / a_t[i]);
                        z_hat[i] = !cils.is_constrained ? c_i : max(min((index) c_i, cils.upper), 0);
                        diff[t] += z_hat[i] - z[i];
                        sum.clear();
                    }
                    if (diff[t] == 0 && t > 2) {
                        num_iter = t;
                        flag = 1;
                    } else {
                        z.assign(z_hat);
                    }
                }
            }

            time = omp_get_wtime() - time;
            return {{}, time, num_iter};
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