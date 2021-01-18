//
// Created by Shilei Lin on 2020-12-17.
//
#include "../include/cils.h"

namespace cils {
    template<typename scalar, typename index, bool is_read, index n>
    returnType <scalar, index>
    cils<scalar, index, is_read, n>::cils_babai_search_omp(const index n_proc, const index nswp,
                                                           vector<index> *z_B) {

        index count = 0, num_iter = 0, s = n_proc, x_min = 0, ni, nj, diff, z_p[n] = {};
        bool flag = false, check = false;
        auto z_x = z_B->data();
        scalar res[nswp] = {}, sum, sum2;
        auto lock = new omp_lock_t[n]();
        for (index i = 0; i < n; i++) {
            omp_set_lock(&lock[i]);
        }

        scalar start = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(sum, diff, ni, nj)
        {

            if (omp_get_thread_num() == 0) {
                ni = n - 1;
                nj = ni * n - (ni * n) / 2;
                z_x[ni] = round((y_A->x[ni] - sum) / R_A->x[nj + ni]);
                omp_unset_lock(&lock[0]);
            }

            for (index j = 1; j < nswp && !flag; j++) {
//                omp_set_lock(&lock[j - 1]);
//                omp_unset_lock(&lock[j - 1]);
#pragma omp for schedule(dynamic) nowait
                for (index i = 1; i < n; i++) {
                    if (flag) continue;
                    sum = 0;
                    ni = n - 1 - i;
                    nj = ni * n - (ni * (n - i)) / 2;
#pragma omp simd reduction(+:sum)
                    for (index col = n - i; col < n; col++) {
                        sum += R_A->x[nj + col] * z_x[col];
//                        if (col == n - i) {
//                            omp_unset_lock(&lock[j]);
//                        }
                    }
                    z_x[ni] = round((y_A->x[ni] - sum) / R_A->x[nj + ni]);
                    if (i == n - 1)
                        check = true;
                }
                if (mode != 0 && check) {
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


//            for (index j = 0; j < nswp && !flag; j++) {
//#pragma omp for schedule(dynamic) nowait
//                for (index i = 0; i < n; i++) {
//                    if (flag || i > s) continue; //
//                    s++;
//                    sum = 0;
//                    ni = n - 1 - i;
//                    nj = ni * n - (ni * (n - i)) / 2;
//
//#pragma omp simd reduction(+ : sum)
//                    for (index col = n - i; col < n; col++) {
//                        sum += R_A->x[nj + col] * z_x[col];
//                    }
//                    omp_set_lock(&lock[ni]);
//                    z_x[ni] = round((y_A->x[ni] - sum) / R_A->x[nj + ni]);
//                    omp_unset_lock(&lock[ni]);
//                    if (i == n - 1)
//                        check = true;
//                }
//                if (mode != 0 && check) {
//                    num_iter = j;
//                    check = false;
//                    diff = 0;
//#pragma omp simd reduction(+ : diff)
//                    for (index l = 0; l < n; l++) {
//                        diff += z_x[l] == z_p[l];
//                        z_p[l] = z_x[l];
//                    }
//                    flag = diff > stop;
//                }
//            }
        }
        scalar run_time = omp_get_wtime() - start;
//        for (index i = 0; i < nswp; i++)
//            cout << sqrt(res[i]) << " ";
        delete[] lock;
        returnType<scalar, index> reT = {z_B, run_time, num_iter};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType <scalar, index>
    cils<scalar, index, is_read, n>::cils_babai_search_omp_constrained(const index n_proc, const index nswp,
                                                                       vector<index> *z_B) {

        index count = 0, num_iter = 0, s = n_proc, x_min = 0, ni, nj, diff, z_p[n] = {}, upper = pow(2, k) - 1;
        bool flag = false, check = false;
        auto z_x = z_B->data();
        scalar res[nswp] = {}, sum, result;

        scalar start = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(result, sum, diff, ni, nj)
        {
            for (index j = 0; j < nswp && !flag; j++) {
#pragma omp for schedule(dynamic) nowait
                for (index i = 0; i < n; i++) {
                    if (flag || i > s) continue; //
                    s++;
                    sum = 0;
                    ni = n - 1 - i;
                    nj = ni * n - (ni * (n - i)) / 2;

#pragma omp simd reduction(+ : sum)
                    for (index col = n - i; col < n; col++) {
                        sum += R_A->x[nj + col] * z_x[col];
                    }
                    result = round((y_A->x[ni] - sum) / R_A->x[nj + ni]);
                    z_x[ni] = result < 0 ? 0 : result > upper ? upper : result;
                    if (i == n - 1)
                        check = true;
                }
                if (mode != 0 && check) {
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
//        for (index i = 0; i < nswp; i++)
//            cout << sqrt(res[i]) << " ";

        returnType<scalar, index> reT = {z_B, run_time, num_iter};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType <scalar, index>
    cils<scalar, index, is_read, n>::cils_babai_search_serial(vector<index> *z_B, bool is_constrained) {
        scalar sum = 0, upper = pow(2, k) - 1;
        scalar start = omp_get_wtime();

        index result = round(y_A->x[n - 1] / R_A->x[((n - 1) * n) / 2 + n - 1]);
        z_B->at(n - 1) = result < 0 ? 0 : result > upper ? upper : result;

        for (index i = 1; i < n; i++) {
            index k = n - i - 1;
            for (index col = n - i; col < n; col++) {
                sum += R_A->x[k * n - (k * (n - i)) / 2 + col] * z_B->at(col);
            }
            result = round((y_A->x[n - 1 - i] - sum) / R_A->x[k * n - (k * (n - i)) / 2 + n - 1 - i]);
            z_B->at(k) = !is_constrained ? result : result < 0 ? 0 : result > upper ? upper : result;
            sum = 0;
        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {z_B, run_time, 0};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType <scalar, index>
    cils<scalar, index, is_read, n>::cils_back_solve(vector<index> *z_B) {
        scalar sum = 0;
        scalar start = omp_get_wtime();
        vector<scalar> z_B_tmp(n, 0);
        z_B_tmp[n - 1] = y_A->x[n - 1] / R_A->x[((n - 1) * n) / 2 + n - 1];

        for (index i = 1; i < n; i++) {
            index k = n - i - 1;
            for (index col = n - i; col < n; col++) {
                sum += R_A->x[k * n - (k * (n - i)) / 2 + col] * z_B->at(col);
            }
            z_B_tmp[k] = (y_A->x[n - 1 - i] - sum) / R_A->x[k * n - (k * (n - i)) / 2 + n - 1 - i];
            sum = 0;
        }
        for (index i = 0; i < n; i++) {
            z_B->at(i) = round(z_B_tmp[i]);
        }

        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {z_B, run_time, 0};
        return reT;
    }
}