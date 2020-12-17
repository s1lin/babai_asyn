//
// Created by Shilei Lin on 2020-12-17.
//
#include "../include/cils.h"

namespace cils{
    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    cils<scalar, index, is_read, n>::cils_babai_search_omp(const index n_proc, const index nswp,
                                                           vector<index> *z_B) {

        index count = 0, num_iter = 0, x_max = n_proc, x_min = 0;
        auto z_x = z_B->data();
        auto *u_p = new index[n](), *z_p = new index[n]();

        scalar start = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc)
        {
            for (index j = 0; j < nswp; j++) {
#pragma omp for schedule(dynamic, 1) nowait
                for (index i = 0; i < n; i++) {
                    if (i <= min(x_max, n)) {
                        x_max += n_proc;
                        babai_solve_omp(i, z_x, z_p, u_p);
                        count += u_p[n - 1 - i];
                    }
                }
                num_iter = j;
            }
        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {z_B, run_time, num_iter};
        delete[] u_p;
        delete[] z_p;
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    cils<scalar, index, is_read, n>::cils_babai_search_serial(vector<index> *z_B) {
        scalar sum = 0;
        scalar start = omp_get_wtime();

        z_B->at(n - 1) = round(y_A->x[n - 1] / R_A->x[((n - 1) * n) / 2 + n - 1]);
        for (index i = 1; i < n; i++) {
            index k = n - i - 1;
            for (index col = n - i; col < n; col++) {
                sum += R_A->x[k * n - (k * (n - i)) / 2 + col] * z_B->at(col);
            }
            z_B->at(k) = round(
                    (y_A->x[n - 1 - i] - sum) / R_A->x[k * n - (k * (n - i)) / 2 + n - 1 - i]);
            sum = 0;
        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {z_B, run_time, 0};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
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