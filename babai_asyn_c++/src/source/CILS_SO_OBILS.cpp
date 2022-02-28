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

    template<typename scalar, typename index>
    class CILS_SO_OBILS {
    private:

        CILS <scalar, index> cils;

        void init_R_A() {
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

    public:
        b_vector z_hat, R_A, y_bar;
        b_matrix R;

        CILS_SO_OBILS(CILS <scalar, index> &cils, const b_vector &z_hat, const b_matrix &R, const b_vector &y_bar) {
            this->cils = cils;
            this->R = R;
            this->y_bar = y_bar;
            this->z_hat = z_hat;
            init_R_A();
        }

        CILS_SO_OBILS(CILS <scalar, index> &cils, const b_vector &z_hat) {
            this->cils = cils;
            this->z_hat = z_hat;
        }


        returnType <scalar, index>
        pbnp(const index n_t, const index nstep, const index init) {

            index num_iter = 0, idx = 0, end = 1, ni, nj, diff = 0, i = cils.n - 1, j, c_i;
            index z_p[cils.n] = {}, delta[cils.n] = {};
            bool flag = false, check = false;
            sd_vector ber(1, 0);

            scalar sum = 0, run_time;
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
        o_pbnp(const index n_t, const index nstep, const index init) {

            index num_iter = 0, idx = 0, end = 1, ni, nj, diff = 0, i = cils.n - 1, j, c_i;
            index z_p[cils.n] = {}, count[nstep] = {}, delta[nstep] = {};
            bool flag = false;

            scalar sum = 0;

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
                        if (!flag) {
                            flag = (delta[t] >= cils.n * 0.6 || idx <= n_t) && count[t] == cils.n - 1;
                            num_iter = t;
                        }
                    }
                }
            }

            run_time = omp_get_wtime() - run_time;
            returnType<scalar, index> reT = {{}, run_time, num_iter};
            return reT;
        }

        returnType <scalar, index> bnp() {
            scalar sum = 0;
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

        returnType <scalar, index> bbnp(index init) {
            scalar sum = 0;
            index ds = cils.d.size(), n_dx_q_0, n_dx_q_1;
            b_vector y_b(cils.n);

            scalar run_time = omp_get_wtime();

            for (index i = 0; i < ds; i++) {
                n_dx_q_1 = cils.d[i];
                n_dx_q_0 = i == ds - 1 ? 0 : cils.d[i + 1];

                subrange(y_b, n_dx_q_0, n_dx_q_1) = subrange(y_bar, n_dx_q_0, n_dx_q_1) -
                                                    prod(subrange(R, n_dx_q_0, n_dx_q_1, n_dx_q_1, cils.n),
                                                         subrange(z_hat, n_dx_q_1, cils.n));

                for (index row_k = n_dx_q_1 - 1; row_k >= n_dx_q_0; row_k--){
                    sum = 0;
                    for (index col = row_k + 1; col < n_dx_q_1; col++) {
                        sum += R(row_k, col) * z_hat[col];
                    }
                    scalar R_kk = R(row_k, row_k);
                    scalar c_i = y_b[row_k] / R_kk;
                    z_hat[row_k] = !cils.is_constrained ? c_i : max(min((index) c_i, cils.upper), 0);
                }

            }
            run_time = omp_get_wtime() - run_time;
            returnType<scalar, index> reT = {{run_time}, run_time, 0};
            return reT;
        }

        returnType <scalar, index> sic(const index nstep) {
            b_vector sum(cils.n, 0), z(z_hat);
            b_vector a_t(cils.n, 0);
            y_bar.resize(cils.m);
            y_bar.assign(cils.y);
            for (index i = 0; i < cils.n; i++) {
                b_vector a = column(cils.A, i);
                a_t[i] = inner_prod(a, a);
            }
            index t = 0;
            scalar time = omp_get_wtime();
            for (t = 0; t < nstep; t++) {
                index diff = 0;
                for (index i = cils.n - 1; i >= 0; i--) {
                    for (index j = 0; j < i; j++) {
                        sum += column(cils.A, j) * z_hat[j];
                    }
                    for (index j = i + 1; j < cils.n; j++) {
                        sum += column(cils.A, j) * z_hat[j];
                    }
                    y_bar = cils.y - sum;
                    scalar c_i = round(inner_prod(column(cils.A, i), y_bar) / a_t[i]);
                    z_hat[i] = !cils.is_constrained ? c_i : max(min((index) c_i, cils.upper), 0);
                    diff += z_hat[i] - z[i];
                    sum.clear();
                }
                //Converge:
                if (diff == 0) {
                    break;
                } else {
                    z.assign(z_hat);
                }
            }

            time = omp_get_wtime() - time;
            return {{}, time, t};
        }

        returnType <scalar, index> psic(const index nstep, const index n_t) {
            b_vector sum(cils.n, 0), z(z_hat);
            b_vector a_t(cils.n, 0), y_b(cils.y);
            index diff[nstep], flag = 0, num_iter = 0;
            for (index i = 0; i < cils.n; i++) {
                b_vector a = column(cils.A, i);
                a_t[i] = inner_prod(a, a);
            }

            scalar time = omp_get_wtime();

#pragma omp parallel default(shared) num_threads(n_t) firstprivate(sum, y_b)
            {
                for (index t = 0; t < nstep && !flag; t++) {
#pragma omp for schedule(dynamic) nowait
                    for (index i = cils.n - 1; i >= 0; i--) {
                        for (index j = 0; j < i; j++) {
                            sum += column(cils.A, j) * z_hat[j];
                        }
                        for (index j = i + 1; j < cils.n; j++) {
                            sum += column(cils.A, j) * z_hat[j];
                        }
                        y_bar = cils.y - sum;
                        scalar c_i = round(inner_prod(column(cils.A, i), y_bar) / a_t[i]);
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

        returnType <scalar, index> backsolve() {
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

        returnType <scalar, index> bocb(index init) {
            scalar sum = 0;
            index ds = cils.d.size(), n_dx_q_0, n_dx_q_1;
            b_vector y_b(cils.n);

            CILS_SECH_Search<scalar, index> search(this->cils);
            scalar start = omp_get_wtime();

            if (init == -1) {
                for (index i = 0; i < ds; i++) {
                    n_dx_q_1 = cils.d[i];
                    n_dx_q_0 = i == ds - 1 ? 0 : cils.d[i + 1];

                    subrange(y_b, n_dx_q_0, n_dx_q_1) = subrange(y_bar, n_dx_q_0, n_dx_q_1) -
                                                        prod(subrange(R, n_dx_q_0, n_dx_q_1, n_dx_q_1, cils.n),
                                                             subrange(z_hat, n_dx_q_1, cils.n));

                    if (cils.is_constrained)
                        search.ch(n_dx_q_0, n_dx_q_1, 0, R, y_b, z_hat);
                    else
                        search.se(n_dx_q_0, n_dx_q_1, 0, R, y_b, z_hat);
                }
            }
            start = omp_get_wtime() - start;
            scalar run_time = omp_get_wtime();

            for (index i = 0; i < ds; i++) {
                n_dx_q_1 = cils.d[i];
                n_dx_q_0 = i == ds - 1 ? 0 : cils.d[i + 1];

                subrange(y_b, n_dx_q_0, n_dx_q_1) = subrange(y_bar, n_dx_q_0, n_dx_q_1) -
                                                    prod(subrange(R, n_dx_q_0, n_dx_q_1, n_dx_q_1, cils.n),
                                                         subrange(z_hat, n_dx_q_1, cils.n));


                if (cils.is_constrained)
                    search.ch(n_dx_q_0, n_dx_q_1, 1, R, y_b, z_hat);
                else
                    search.se(n_dx_q_0, n_dx_q_1, 1, R, y_b, z_hat);
            }
            run_time = omp_get_wtime() - run_time + start;
            returnType<scalar, index> reT = {{run_time - start}, run_time, 0};
            return reT;
        }


        returnType <scalar, index> pbocb(const index n_proc, const index nstep, const index init) {
            index ds = cils.d.size();

            index diff = 0, num_iter = 0, flag = 0, temp, R_S_1[ds] = {}, R_S_2[ds] = {};
            index test, row_n, check = 0, r, _nswp = nstep, end = 0;
            index n_dx_q_2, n_dx_q_1, n_dx_q_0;
            scalar sum = 0, start;
            scalar run_time = 0, run_time3 = 0;

            b_vector y_B(cils.n);
            y_B.clear();
            CILS_SECH_Search<scalar, index> _ils(cils.m, cils.n, cils.qam);
//            omp_set_schedule((omp_sched_t) schedule, chunk_size);
//            auto lock = new omp_lock_t[ds]();
//            for (index i = 0; i < ds; i++) {
//                omp_set_lock(&lock[i]);
//            }
//
//            omp_unset_lock(&lock[0]);
//            omp_unset_lock(&lock[1]);


#pragma omp parallel default(none) num_threads(n_proc)
            {}
            if (init == -1) {
                start = omp_get_wtime();
                n_dx_q_2 = cils.d[0];
                n_dx_q_0 = cils.d[1];

                if (cils.is_constrained)
                    _ils.mch(n_dx_q_0, n_dx_q_2, 0, 0, R_A, y_bar, z_hat);
//                else
//                    _ils.mse(n_dx_q_0, n_dx_q_2, 0, 0, R_A, y_bar, z_hat);

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
                                        sum += R_A[col + row_n] * z_hat[col];
                                    }
                                    y_B[row] = y_bar[row] - sum;
                                }
                                if (cils.is_constrained)
                                    R_S_2[i] = _ils.mch(n_dx_q_0, n_dx_q_2, i, 0, R_A, y_B, z_hat);
//                                else
//                                    R_S_2[i] = _ils.mse(n_dx_q_0, n_dx_q_2, i, 0, R_A, y_B, z_hat);
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

            CILS_SECH_Search<scalar, index> search(this->cils);
            scalar run_time2 = omp_get_wtime();
            n_dx_q_2 = cils.d[0];
            n_dx_q_0 = cils.d[1];

            if (cils.is_constrained)
                search.mch(n_dx_q_0, n_dx_q_2, 0, 1, R_A, y_bar, z_hat);
//            else
//                search.mse(n_dx_q_0, n_dx_q_2, 0, 1, R_A, y_bar, z_hat);


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
                                    sum += R_A[col + row_n] * z_hat[col];
                                }
                                y_B[row] = y_bar[row] - sum;
                            }
//                        test = 0;
//                        for (index row = 0; row < i; row++){
//                            test += R_S_1[i];
//                        }
//                        omp_set_lock(&lock[i - 1]);

//                        check = check || R_S_1[i - 1];
                            if (cils.is_constrained) {
                                R_S_1[i] = search.mch2(n_dx_q_0, n_dx_q_2, i, check, R_A, y_B, z_hat);
//                                R_S_1[i] = search.ch(n_dx_q_0, n_dx_q_2, check, R, y_B, z_hat);
                            }
//                            else
//                                R_S_1[i] = search.mse(n_dx_q_0, n_dx_q_2, i, check, R_A, y_B, z_hat);
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

        returnType <scalar, index> pbocb_test(const index n_proc, const index nstep, const index init) {

            index ds = cils.d.size();
            index diff = 0, num_iter = 0, flag = 0, temp;
            si_vector R_S_1(ds, 0), count(nstep, 0), delta(nstep, 0);
            index row_n, check = 0, r, end = 0;
            index n_dx_q_1, n_dx_q_0;
            scalar sum = 0, start;
            scalar run_time = 0, run_time3 = 0;

#pragma omp parallel default(none) num_threads(n_proc)
            {}

            CILS_SECH_Search<scalar, index> search(this->cils);
            scalar run_time2 = omp_get_wtime();
            n_dx_q_1 = cils.d[0];
            n_dx_q_0 = cils.d[1];

            if (cils.is_constrained)
                search.mch(n_dx_q_0, n_dx_q_1, 0, 1, R_A, y_bar, z_hat);
            else
                search.mse(n_dx_q_0, n_dx_q_1, 0, 1, R_A, y_bar, z_hat);

            R_S_1[0] = 1;
            end = 1;

#pragma omp parallel default(shared) num_threads(n_proc) private(n_dx_q_1, n_dx_q_0, sum, temp, check, row_n)
            {
                for (index j = 0; j < nstep && !flag; j++) {
#pragma omp for schedule(dynamic) nowait
                    for (index i = 1; i < ds; i++) {
                        if (!flag && end <= i) {
                            n_dx_q_1 = cils.d[i];
                            n_dx_q_0 = i == ds - 1 ? 0 : cils.d[i + 1];
                            check = i == end;
                            //row_n = (n_dx_q_0 - 1) * (cils.n - n_dx_q_0 / 2);

                            if (cils.is_constrained)
//                                if(check)
                                R_S_1[i] = search.mch(n_dx_q_0, n_dx_q_1, i, check, R_A, y_bar, z_hat);
//                                else
//                                    R_S_1[i] = search.mch(n_dx_q_0, n_dx_q_1, i, check, R_A, y_bar, z_hat);
                            else
                                R_S_1[i] = search.mse(n_dx_q_0, n_dx_q_1, i, check, R_A, y_bar, z_hat);

                            if (check) {
                                end = i + 1;
                                R_S_1[i] = 1;
                            }
                        }
#pragma omp atomic
                        delta[j] += R_S_1[i];
#pragma omp atomic
                        count[j]++;
                    }
                    if (!flag) {
                        flag = (delta[j] >= ds - cils.offset) && count[j] == ds - 1;
                        num_iter = j;
                    }

                }

#pragma omp single
                {
                    run_time3 = omp_get_wtime() - run_time2;
                }
            }
            run_time2 = omp_get_wtime() - run_time2;

            scalar time = 0;
            if (init == -1) {
                time = cils.qam == 1 ? run_time2 + run_time : run_time2 + run_time;
            } else {
                time = cils.qam == 1 ? run_time2 : run_time2 * 0.5 + run_time3 * 0.5;
            }
//            if (mode == 0)
//                reT = {{run_time3}, time, (scalar) diff + end};
//            else {
//            helper::display<scalar, index>(delta, nstep, "delta");
//            helper::display<scalar, index>(count, nstep, "count");
//
//            cout << "n_proc:" << n_proc << "," << "init:" << init << ",end:" << end << ",ratio:"
//                 << (index) (run_time2 / run_time3) << "," << run_time << "," << num_iter << "||" << endl;
//            cout.flush();

            return {{run_time3}, time, (scalar) num_iter + 1};
        }


        returnType <scalar, index> bocb_CPU() {
            index ds = cils.d.size(), n_dx_q_0, n_dx_q_1;
            b_vector y_b(cils.n);

            sd_vector time(2 * ds, 0);
            CILS_SECH_Search<scalar, index> search(this->cils);
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
//                    se(0, cils.n, y_b, z_hat);
                    return {{}, 0, 0};
                }
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
                    time[time_index + ds] = search.ch(n_dx_q_0, n_dx_q_1, 1, R, y_b, z_hat);
                else
                    time[time_index + ds] = search.se(n_dx_q_0, n_dx_q_1, 1, R, y_b, z_hat);

                time[time_index] = omp_get_wtime() - time[time_index];
            }

            scalar run_time = omp_get_wtime() - start;


            //Matlab Partial Reduction needs to do the permutation
            returnType<scalar, index> reT = {time, run_time, 0};
            return reT;
        }
    };
}

//                scalar prod_time = omp_get_wtime();
//                prod_time = omp_get_wtime() - prod_time;
//                scalar prod_time2 = omp_get_wtime();
//                for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                    sum = 0;
//                    for (index col = n_dx_q_1; col < cils.n; col++) {
//                        sum += R(row, col) * z_hat[col];
//                    }
//                    y_b[row] = y_bar[row] - sum;
//                }
//                prod_time2 = omp_get_wtime() - prod_time2;
//                cout << "Ratio:" << prod_time / prod_time2 <<" ";