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
                return {vector<index>(round(y_A->x[0] / R_A->x[0]), 0), 0, 0, 0};
            } else {
                vector<scalar> R_B = find_block_Rii(R_A, 0, n, 0, n, n);
                vector<scalar> y_B = find_block_x(y_A, 0, n);
                return {ils_search(&R_B, &y_B), 0, 0, 0};
            }
        } else if (ds == n) {
            //Find the Babai point
            return cils_babai_search_serial(z_B);
        }

        //last block:
        index q = d->at(ds - 1);

        scalar start = omp_get_wtime();
        //the last block
        vector<scalar> R_ii = find_block_Rii<scalar, index>(R_A, n - q, n, n - q, n, n);
        vector<scalar> y_b_s = find_block_x<scalar, index>(y_A, n - q, n);
        vector<index> x_b_s = ils_search(&R_ii, &y_b_s);
        for (index l = n - q; l < n; l++) {
            z_B->at(l) = x_b_s[l - n + q];
        }
//        display_scalarType(y_b_s);
//        cout<<  n - q<<","<<n <<endl;
//        display_scalarType(x_b_s);

        //Therefore, skip the last block, start from the second-last block till the first block.
        for (index i = 0; i < ds - 1; i++) {
            q = ds - 2 - i;
            //accumulate the block size
            y_b_s = find_block_x<scalar, index>(y_A, n - d->at(q), n - d->at(q + 1));
            x_b_s = find_block_x<scalar, index>(z_B, n - d->at(q + 1), n);
            y_b_s = block_residual_vector(R_A, &x_b_s, &y_b_s, n - d->at(q), n - d->at(q + 1),
                                          n - d->at(q + 1), n);
//            display_scalarType(y_b_s);
            R_ii = find_block_Rii<scalar, index>(R_A, n - d->at(q), n - d->at(q + 1), n - d->at(q),
                                                 n - d->at(q + 1), n);
//            cout<< n - d->at(q)<<","<<n - d->at(q + 1)<<endl;
            x_b_s = ils_search(&R_ii, &y_b_s);
//            display_array(x_b_s->at, x_b_s->size());

            for (index l = n - d->at(q); l < n - d->at(q + 1); l++) {
                z_B->at(l) = x_b_s[l - n + d->at(q)];
            }


        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {*z_B, run_time, 0, 0};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    cils<scalar, index, is_read, n>::cils_block_search_omp(index n_proc, index nswp, scalar stop,
                                                           vector<index> *z_B,
                                                           vector<index> *d) {
        index ds = d->size(), dx = d->at(ds - 1);
        if (ds == 1) {
            if (d->at(0) == 1) {
                z_B->at(0) = round(y_A->x[0] / R_A->x[0]);
                return {*z_B, 0, 0, 0};
            } else {
                vector<scalar> R_B = find_block_Rii(R_A, 0, n, 0, n, n);
                vector<scalar> y_B = find_block_x(y_A, 0, n);
                return {ils_search(&R_B, &y_B), 0, 0, 0};
            }
        } else if (ds == n) {
            //Find the Babai point
            //todo: Change it to omp version
            return cils_babai_search_serial(z_B);
        }
        index count = 0, num_iter = 0, n_dx_q_0, n_dx_q_1;
        scalar res = 0, nres = 10, sum = 0;

        vector<scalar> y(dx, 0), y_n(dx, 0);
        vector<index> x(dx, 0), z_B_p(n, 0);

        for (index l = n - dx; l < n; l++) {
            y_n[l - (n - dx)] = y_A->x[l];
        }

        scalar start = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(sum, y, x, n_dx_q_0, n_dx_q_1)
        {
            y.assign(dx, 0);
            x.assign(dx, 0);
            for (index j = 0; j < nswp && abs(nres) > stop; j++) {//
#pragma omp for nowait
//#pragma omp for nowait //schedule(dynamic)guided
                for (index i = 0; i < ds; i++) {
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
                            y[row - n_dx_q_0] = y_A->x[row] - sum;
                        }
                        ils_search_omp(n_dx_q_0, n_dx_q_1, &y, &x);
                    } else {
                        ils_search_omp(n_dx_q_0, n_dx_q_1, &y_n, &x);
                    }
#pragma omp simd
                    for (index l = n_dx_q_0; l < n_dx_q_1; l++) {
                        z_B->at(l) = x[l - n_dx_q_0];
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
        returnType<scalar, index> reT = {*z_B, run_time, nres, num_iter};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    cils<scalar, index, is_read, n>::cils_block_search_omp_schedule(index n_proc, index nswp,
                                                                    scalar stop, string schedule,
                                                                    vector<index> *z_B,
                                                                    vector<index> *d) {
        index ds = d->size(), dx = d->at(ds - 1);
        if (ds == 1) {
            if (d->at(0) == 1) {
                z_B->at(0) = round(y_A->x[0] / R_A->x[0]);
                return {*z_B, 0, 0, 0};
            } else {
                vector<scalar> R_B = find_block_Rii(R_A, 0, n, 0, n, n);
                vector<scalar> y_B = find_block_x(y_A, 0, n);
                return {ils_search(&R_B, &y_B), 0, 0, 0};
            }
        } else if (ds == n) {
            //Find the Babai point by OpenMP
            return cils_babai_search_omp(n_proc, nswp, z_B);
        }

        index count = 0, num_iter = 0, n_dx_q_0, n_dx_q_1;
        scalar res = 0, nres = 10, sum = 0;

        vector<scalar> y(dx, 0), y_n(dx, 0), p(ds, 0);
        vector<index> x(dx, 0), z_B_p(n, 0), work(ds, 0);

        for (index l = n - dx; l < n; l++) {
            y_n[l - (n - dx)] = y_A->x[l];
        }

        for (index i = 0; i < ds - 1; i++) {
            index q = ds - 2 - i;
            p[i] = find_success_prob_babai(R_A, n - d->at(q), n - d->at(q + 1), n, sigma);
            cout << n - d->at(q) << "," << n - d->at(q + 1) << "," << "P=" << p[i] << endl;
        }



//        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//        shuffle (work.begin(), work.end(), std::default_random_engine(seed));
//        display_vector<scalar, index>(&work);
//        omp_set_schedule(omp_sched_dynamic, n_proc);
        scalar start = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(sum, y, x, n_dx_q_0, n_dx_q_1)
        {
            y.assign(dx, 0);
            x.assign(dx, 0);
            for (index j = 0; j < nswp && abs(nres) > stop; j++) {//
#pragma omp for schedule(dynamic) nowait//schedule(dynamic) nowait
//#pragma omp for nowait //schedule(dynamic)guided
//                for (index m = 0; m < n_proc; m++) {
//                    for (index i = m; i < ds; i += n_proc) {
                for (index i = 0; i < ds; i++) {
                    n_dx_q_0 = work[i] == 0 ? n - dx : n - d->at(ds - 1 - work[i]);
                    n_dx_q_1 = work[i] == 0 ? n : n - d->at(ds - work[i]);
                    //The block operation
                    if (work[i] != 0) {
                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                            sum = 0;
#pragma omp simd reduction(+ : sum)
                            for (index col = n_dx_q_1; col < n; col++) {
                                //Translating the index from R(matrix) to R_B(compressed array).
                                sum += R_A->x[(n * row) + col - ((row * (row + 1)) / 2)] * z_B->at(col);
                            }
                            y[row - n_dx_q_0] = y_A->x[row] - sum;
                        }
                        ils_search_omp(n_dx_q_0, n_dx_q_1, &y, &x);
                    } else {
                        ils_search_omp(n_dx_q_0, n_dx_q_1, &y_n, &x);
                    }
#pragma omp simd
                    for (index l = n_dx_q_0; l < n_dx_q_1; l++) {
                        z_B->at(l) = x[l - n_dx_q_0];
                    }
                }
//                }
//#pragma omp master
//                {
//                    if (num_iter > 0) {
//                        nres = 0;
//#pragma omp simd reduction(+ : nres)
//                        for (index l = 0; l < n; l++) {
//                            nres += (z_B_p[l] - z_B->at(l));
//                            z_B_p[l] = z_B->at(l);
//                        }
//                    } else {
//#pragma omp simd
//                        for (index l = 0; l < n; l++) {
//                            z_B_p[l] = z_B->at(l);
//                        }
//                    }
//                    num_iter = j;
//                }
            }

        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {*z_B, run_time, nres, num_iter};
        return reT;
    }
}
