//
// Created by shilei on 2020-12-05.
//
#include <algorithm>
#include <random>
#include "../include/cils.h"

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
                ils_search(0, n, &y_b, z_B, is_constrained);
                return {z_B, 0, 0};
            }
        } else if (ds == n) {
            //Find the Babai point
            return cils_babai_search_serial(z_B, is_constrained);
        }
        scalar start = omp_get_wtime(), res = 0;
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

            ils_search(n_dx_q_0, n_dx_q_1, &y_b, z_B, is_constrained);
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
        bool flag = false, check = false;
        index num_iter, n_dx_q_0, n_dx_q_1, row_n, iter = 2 * n_proc, diff[100] = {}, z_p[n] = {}, upper =
                pow(2, k) - 1, k, count, row_k;
        scalar sum = 0, result = 0, run_time;
        auto y_b = new scalar[n]();

        scalar start = omp_get_wtime();
        auto lock = new omp_lock_t[ds]();


#pragma omp parallel default(shared) num_threads(n_proc) private(sum, result, row_n, n_dx_q_0, n_dx_q_1, k, count, row_k)
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
                    if (flag || i > iter) continue;
                    iter++;
                    n_dx_q_0 = n - (i + 1) * dx;
                    n_dx_q_1 = n - i * dx;
                    //The block operation
                    for (index row = n_dx_q_0; row < n_dx_q_1 && !flag; row++) {
                        sum = 0;
                        row_n = (n * row) - ((row * (row + 1)) / 2);
#pragma omp simd reduction(+ : sum)
                        for (index col = n_dx_q_1; col < n; col++) {
                            sum += R_A->x[row_n + col] * z_x[col];
                        }
                        y_b[row] = y_A->x[row] - sum;
                    }
                    ils_search_omp(n_dx_q_0, n_dx_q_1, y_b, z_x, is_constrained);
                    if (i == ds - 1) {
                        num_iter = j;
                        check = false;
#pragma omp simd reduction(+ : diff)
                        for (index l = 0; l < n; l++) {
                            diff[j] += z_x[l] == z_p[l];
                            z_p[l] = z_x[l];
                        }
                        if (mode != 0 && !flag) {
                            flag = diff[j] > stop;
#pragma omp flush (flag)
                        }
                    }
                }
            }
        }

        scalar run_time2 = omp_get_wtime() - start;

#pragma parallel omp cancellation point
#pragma omp flush
        returnType<scalar, index> reT = {z_B, run_time2, num_iter};
        delete[] y_b;
        return reT;
    }
}
