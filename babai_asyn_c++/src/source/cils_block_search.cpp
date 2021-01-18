//
// Created by shilei on 2020-12-05.
//
#include <algorithm>
#include <random>
#include "../include/cils.h"

using namespace std;

namespace cils {
    template<typename scalar, typename index, bool is_read, index n>
    returnType <scalar, index>
    cils<scalar, index, is_read, n>::cils_block_search_serial(const vector<index> *d, vector<index> *z_B,
                                                              bool is_constrained) {

        index ds = d->size(), dx = d->at(ds - 1), n_dx_q_0, n_dx_q_1;
        //special cases:
        if (ds == 1) {
            if (d->at(0) == 1) {
                z_B->at(0) = round(y_A->x[0] / R_A->x[0]);
                return {z_B, 0, 0};
            } else {
                vector<scalar> R_B = find_block_Rii(R_A, 0, n, 0, n, n);
                vector<scalar> y_B = find_block_x(y_A, 0, n);
                ils_search(&R_B, &y_B, z_B, is_constrained);
                return {z_B, 0, 0};
            }
        } else if (ds == n) {
            //Find the Babai point
            return cils_babai_search_serial(z_B, is_constrained);
        }

        scalar start = omp_get_wtime(), res = 0;
        vector<scalar> R_b, y_b;
        vector<index> x_b(dx, 0);

        for (index i = 0; i < ds; i++) {
            n_dx_q_0 = i == 0 ? n - dx : n - d->at(ds - 1 - i);
            n_dx_q_1 = i == 0 ? n : n - d->at(ds - i);

            y_b = find_block_x<scalar, index>(y_A, n_dx_q_0, n_dx_q_1);
            R_b = find_block_Rii<scalar, index>(R_A, n_dx_q_0, n_dx_q_1, n_dx_q_0, n_dx_q_1, n);

            if (i != 0) {
                //accumulate the block size
                x_b = find_block_x<scalar, index>(z_B, n_dx_q_1, n);
                y_b = block_residual_vector(R_A, &x_b, &y_b, n_dx_q_0, n_dx_q_1, n_dx_q_1, n);
            }

            res += ils_search(&R_b, &y_b, &x_b, is_constrained);

            for (index l = n_dx_q_0; l < n_dx_q_1; l++) {
                z_B->at(l) = x_b[l - n_dx_q_0];
            }
        }

        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {z_B, run_time, 0};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType <scalar, index>
    cils<scalar, index, is_read, n>::cils_block_search_omp(const index n_proc, const index nswp,
                                                           const index stop, const index init,
                                                           const vector<index> *d,
                                                           vector<index> *z_B,
                                                           const bool is_constrained) {
        index ds = d->size(), dx = d->at(ds - 1);
        if (ds == 1 || ds == n) {
            return cils_block_search_serial(d, z_B, is_constrained);
        }

        auto z_x = z_B->data();
        //index count = 0, search_count = 255;
        bool flag = false, check = false;
        index num_iter, n_dx_q_0, n_dx_q_1, row_n, iter = 2 * n_proc, diff[100] = {}, z_p[n] = {};
        scalar sum = 0, run_time, y_b[n] = {};

//        int gap = ds % n_proc == 0 ? ds / n_proc : ds / n_proc + 1;
//        for (int i = 0; i < n_proc; i++) {
//            for (int j = i * gap; j < (i + 1) * gap - gap / 2 && j < ds; j++) {
//                cout << count << " ";
//                work[j] = count;
//                count++;
//            }
//            for (int j = (i + 1) * gap - gap / 2; j < (i + 1) * gap && j < ds; j++) {
//                cout << search_count << " ";
//                work[j] = search_count;
//                search_count--;
//            }
//        }

        scalar start = omp_get_wtime();
        auto lock = new omp_lock_t[ds]();

#pragma omp parallel default(shared) num_threads(n_proc) private(sum, row_n, n_dx_q_0, n_dx_q_1)
        {
            if (init != -1)
#pragma omp for nowait
                for (index i = 0; i < n; i++) {
                    sum = 0;
                    index ni = n - 1 - i, nj = ni * n - (ni * (n - i)) / 2;
#pragma omp simd reduction(+ : sum)
                    for (index col = n - i; col < n; col++)
                        sum += R_A->x[nj + col] * z_x[col];
                    z_x[ni] = (y_A->x[ni] - sum) / R_A->x[nj + ni];
                    if (i < ds) omp_set_lock(&lock[i]);
                }

//            if (omp_get_thread_num() == 0) {
//                // Calculation of ||A||
//                for (index l = n_dx_q_0; l < n_dx_q_1; l++)
//                    y_b[l] = y_A->x[l];
//
//                ils_search_omp(n_dx_q_0, n_dx_q_1, y_b, z_x, is_constrained);
//                omp_unset_lock(&lock[0]);
//            }

            for (index j = 1; j < nswp && !flag; j++) {
#pragma omp for schedule(dynamic) nowait //
                for (index i = 0; i < ds; i++) {
                    if (flag) continue; // || i > iter
                    iter++;
                    n_dx_q_0 = n - (i + 1) * dx;
                    n_dx_q_1 = n - i * dx;
                    //The block operation
//                    if (i != 0) {
                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                            sum = 0;
                            row_n = (n * row) - ((row * (row + 1)) / 2);
#pragma omp simd reduction(+ : sum)
                            for (index col = n_dx_q_1; col < n; col++) {
                                sum += R_A->x[row_n + col] * z_x[col];
                            }
                            y_b[row] = y_A->x[row] - sum;
                        }
//                    } else
//#pragma omp simd
//                        for (index l = n_dx_q_0; l < n_dx_q_1; l++)
//                            y_b[l] = y_A->x[l];

                    ils_search_omp(n_dx_q_0, n_dx_q_1, y_b, z_x, is_constrained);
//                    omp_set_lock(&lock[i]);
//                    omp_unset_lock(&lock[i]);

                    if (i == ds - 1)
                        check = true;

                }
                if (check) {
                    num_iter = j;
                    check = false;
#pragma omp simd reduction(+ : diff)
                    for (index l = 0; l < n; l++) {
                        diff[j] += z_x[l] == z_p[l];
                        z_p[l] = z_x[l];
                    }
                    if (mode != 0)
                        flag = diff[j] > stop;
                }
            }
#pragma omp master
            {
                run_time = omp_get_wtime() - start;
            }
        }

        scalar run_time2 = omp_get_wtime() - start;

//#ifdef VERBOSE //1
//        printf("%d, %.3f, %.3f, ", diff, run_time, run_time / run_time2);
//#endif
        returnType<scalar, index> reT = {z_B, run_time2, num_iter};
        if (mode == 0)
//            for (index i = 0; i < nswp; i++)
            reT = {z_B, run_time2, diff[nswp - 1]};

        return reT;
    }
}
