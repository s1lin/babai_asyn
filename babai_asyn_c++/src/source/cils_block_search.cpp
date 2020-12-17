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
    returnType<scalar, index>
    cils<scalar, index, is_read, n>::cils_block_search_serial(vector<index> *z_B,
                                                              vector<index> *d) {

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
    returnType<scalar, index>
    cils<scalar, index, is_read, n>::cils_block_search_omp(index n_proc, index nswp, scalar stop,
                                                           vector<index> *z_B,
                                                           vector<index> *d) {
        index ds = d->size(), dx = d->at(ds - 1);
        if (ds == 1 || ds == n) {
            return cils_block_search_serial(z_B, d);
        }
        index num_iter = 0, n_dx_q_0, n_dx_q_1, s = n_proc;
        scalar nres = 10, sum = 0;

        vector<scalar> y_b(dx, 0), y_n(dx, 0);
        vector<index> x(dx, 0), z_B_p(n, 0);

        for (index l = n - dx; l < n; l++) {
            y_n[l - (n - dx)] = y_A->x[l];
        }

        scalar start = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(sum, y, x, n_dx_q_0, n_dx_q_1)
        {
            y_b.assign(dx, 0);
            x.assign(dx, 0);
            for (index j = 0; j < nswp && abs(nres) > stop; j++) {//
#pragma omp for schedule(guided) nowait
                for (index i = 0; i < ds; i++) {
                    if (i <= ds) {
                        n_dx_q_0 = i == 0 ? n - dx : n - d->at(ds - 1 - i);
                        n_dx_q_1 = i == 0 ? n : n - d->at(ds - i);
                        //The block operation
                        if (i != 0) {
                            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                                sum = 0;
#pragma omp simd reduction(+ : sum)
                                for (index col = n_dx_q_1; col < n; col++) {
                                    //Translating the index from R(matrix) to R_B(compressed array).
                                    sum += R_A->x[(n * row) + col - ((row * (row + 1)) / 2)] * z_B->at(col);
                                }
                                y_b[row - n_dx_q_0] = y_A->x[row] - sum;
                            }
//                            ils_search_omp(n_dx_q_0, n_dx_q_1, 1000, &y_b, &x);//&x);
                        } else {
//                            ils_search_omp(n_dx_q_0, n_dx_q_1, 1000, &y_n, &x);//&x);
                        }
                        s++;
#pragma omp simd
                        for (index l = n_dx_q_0; l < n_dx_q_1; l++) {
                            z_B->at(l) = x[l - n_dx_q_0];
                        }
                    }
                }
//                }
#pragma omp master
                {
                    if (num_iter > 0) {
                        nres = 0;
#pragma omp simd reduction(+ : nres)
                        for (index l = 0; l < n; l++) {
                            nres += (z_B_p[l] - z_B->at(l));
                            z_B_p[l] = z_B->at(l);
                        }
                    } else {
#pragma omp simd
                        for (index l = 0; l < n; l++) {
                            z_B_p[l] = z_B->at(l);
                        }
                    }
                    num_iter = j;
                }
            }

        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {z_B, run_time, num_iter};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    cils<scalar, index, is_read, n>::cils_block_search_omp_schedule(index n_proc, index nswp,
                                                                    scalar stop, string schedule,
                                                                    vector<index> *z_B,
                                                                    vector<index> *d) {
        index ds = d->size(), dx = d->at(ds - 1);
        if (ds == 1 || ds == n) {
            return cils_block_search_serial(z_B, d);
        }
        
        auto *z_x = (index *) calloc(n, sizeof(index));
        auto *z_p = (index *) calloc(n, sizeof(index));
        auto *y_n = (scalar *) calloc(dx, sizeof(scalar));

        index num_iter = 0, n_dx_q_0, n_dx_q_1, s = n_proc, *x;
        scalar nres = 10, sum = 0, *y_b;

        for (index l = n - dx; l < n; l++) {
            y_n[l - (n - dx)] = y_A->x[l];
        }

        scalar start = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(sum, y_b, x, n_dx_q_0, n_dx_q_1)
        {
            y_b = (scalar *) calloc(dx, sizeof(scalar));
            x = (index *) calloc(dx, sizeof(index));
            for (index j = 0; j < nswp && abs(nres) > stop; j++) {
#pragma omp for schedule(dynamic) nowait
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
                            ils_search_omp(n_dx_q_0, n_dx_q_1, 4, y_b, x);
                        } else {
                            ils_search_omp(n_dx_q_0, n_dx_q_1, 3, y_n, x);
                        }
#pragma omp simd
                        for (index l = n_dx_q_0; l < n_dx_q_1; l++) {
                            z_x[l] = x[l - n_dx_q_0];
                        }
                        s++;
                    }
                }
#pragma omp master
                {
                    if (num_iter > 0) {
                        nres = 0;
#pragma omp simd reduction(+ : nres)
                        for (index l = 0; l < n; l++) {
                            nres += (z_p[l] - z_x[l]);
                            z_p[l] = z_x[l];
                        }
                    } else {
#pragma omp simd
                        for (index l = 0; l < n; l++) {
                            z_p[l] = z_x[l];
                        }
                    }
                    num_iter = j;
                }
            }
            free(y_b);
            free(x);
        }

        scalar run_time = omp_get_wtime() - start;
        for (index l = 0; l < n; l++)
            z_B->at(l) = z_x[l];
        returnType<scalar, index> reT = {z_B, run_time, num_iter};

        free(z_x);
        free(z_p);
        free(y_n);
        return reT;
    }
}
