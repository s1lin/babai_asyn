//
// Created by shilei on 2020-12-05.
//
#include <algorithm>
#include <random>
#include <chrono>
#include "../include/cils.h"

using namespace std;

namespace cils {
    template<typename scalar, typename index, bool is_read, index n>
    returnType <scalar, index>
    cils<scalar, index, is_read, n>::cils_block_search_serial(const vector<index> *d, vector<index> *z_B) {

        index ds = d->size();
        //special cases:
        if (ds == 1) {
            if (d->at(0) == 1) {
                z_B->at(0) = round(y_A->x[0] / R_A->x[0]);
                return {z_B, 0, 0};
            } else {
                vector<scalar> R_B = find_block_Rii(R_A, 0, n, 0, n, n);
                vector<scalar> y_B = find_block_x(y_A, 0, n);
                ils_search(&R_B, &y_B, z_B);
                return {z_B, 0, 0};
            }
        } else if (ds == n) {
            //Find the Babai point
            return cils_babai_search_serial(z_B);
        }

        //last block:
        index q = d->at(ds - 1);

        scalar start = omp_get_wtime();
        //the last block
        vector<scalar> R_b = find_block_Rii<scalar, index>(R_A, n - q, n, n - q, n, n);
        vector<scalar> y_b = find_block_x<scalar, index>(y_A, n - q, n);
        vector<index> x_b(n - q, 0);
        ils_search(&R_b, &y_b, &x_b);
        for (index l = n - q; l < n; l++) {
            z_B->at(l) = x_b[l - n + q];
        }

        //Therefore, skip the last block, start from the second-last block till the first block.
        for (index i = 0; i < ds - 1; i++) {
            q = ds - 2 - i;
            //accumulate the block size
            y_b = find_block_x<scalar, index>(y_A, n - d->at(q), n - d->at(q + 1));
            x_b = find_block_x<scalar, index>(z_B, n - d->at(q + 1), n);
            y_b = block_residual_vector(R_A, &x_b, &y_b, n - d->at(q), n - d->at(q + 1), n - d->at(q + 1), n);

            R_b = find_block_Rii<scalar, index>(R_A, n - d->at(q), n - d->at(q + 1), n - d->at(q), n - d->at(q + 1), n);

            ils_search(&R_b, &y_b, &x_b);

            for (index l = n - d->at(q); l < n - d->at(q + 1); l++) {
                z_B->at(l) = x_b[l - n + d->at(q)];
            }

        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {z_B, run_time, 0};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType <scalar, index>
    cils<scalar, index, is_read, n>::cils_block_search_omp(const index n_proc, const index nswp,
                                                           const index stop, const index schedule,
                                                           const vector<index> *d,
                                                           vector<index> *z_B) {
        index ds = d->size(), dx = d->at(ds - 1);
        if (ds == 1 || ds == n) {
            return cils_block_search_serial(d, z_B);
        }

        auto z_x = z_B->data();
        auto *z_p = new index[n]();

        index num_iter = 0, n_dx_q_0, n_dx_q_1, s = n_proc;
        scalar nres = INFINITY, sum = 0, res = 0, diff = 20, run_time;

        omp_set_schedule((omp_sched_t) schedule, chunk_size);

        scalar start = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(sum, n_dx_q_0, n_dx_q_1)
        {
            auto *y_b = new scalar[dx]();
            auto *x_b = new index[dx]();
            for (index j = 0; j < nswp && abs(nres) > stop; j++) {
#pragma omp for schedule(runtime) nowait
                for (index i = 0; i < ds; i++) {
                    if (i <= s) {
                        n_dx_q_0 = i == 0 ? n - dx : n - d->at(ds - 1 - i);
                        n_dx_q_1 = i == 0 ? n : n - d->at(ds - i);
                        //The block operation
                        if (i != 0) {
                            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                                sum = 0;
#pragma omp simd reduction(+ : sum)
                                for (index col = n_dx_q_1; col < n; col++) {
                                    sum += R_A->x[(n * row) + col - ((row * (row + 1)) / 2)] * z_x[col];
                                }
                                y_b[row - n_dx_q_0] = y_A->x[row] - sum;
                            }
                        } else {
#pragma omp simd
                            for (index l = n_dx_q_0; l < n_dx_q_1; l++)
                                y_b[l - n_dx_q_0] = y_A->x[l];
                        }

                        ils_search_omp(n_dx_q_0, n_dx_q_1, y_b, x_b);
#pragma omp simd
                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                            z_x[row] = x_b[row - n_dx_q_0];
                            diff += z_x[row] == z_p[row] ? 0 : 1;
                            z_p[row] = x_b[row - n_dx_q_0];
                        }
                        s++;
                    }
                }
#pragma omp master
                {
                    nres = diff;
                    diff = 0;
                    num_iter = j;
                }
            }
            delete[] y_b;
            delete[] x_b;
        }
        run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {z_B, run_time, num_iter};

        delete[] z_p;
        return reT;
    }
}
