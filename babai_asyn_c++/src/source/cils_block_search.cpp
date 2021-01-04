//
// Created by shilei on 2020-12-05.
//
#include <algorithm>
#include <random>
#include <chrono>
#include "../include/cils.h"
#include <boost/numeric/ublas/vector.hpp>

using namespace std;

namespace cils {
    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
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
        scalar res = ils_search(&R_b, &y_b, &x_b);
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

            res += ils_search(&R_b, &y_b, &x_b);

            for (index l = n - d->at(q); l < n - d->at(q + 1); l++) {
                z_B->at(l) = x_b[l - n + d->at(q)];
            }

        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {z_B, run_time, 0};
//        cout << res << ",";
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    cils<scalar, index, is_read, n>::cils_block_search_omp(const index n_proc, const index nswp,
                                                           const index stop, const index init,
                                                           const vector<index> *d,
                                                           vector<index> *z_B) {
        index ds = d->size(), dx = d->at(ds - 1);
        if (ds == 1 || ds == n) {
            return cils_block_search_serial(d, z_B);
        }

        auto z_x = z_B->data();
        index count = 0, search_count = 255;
        bool flag = false;
        index num_iter, n_dx_q_0, n_dx_q_1, row_n, iter = 1.5 * n_proc, pitt = n_proc, work[ds] = {};
        scalar sum = 0, run_time, res[ds] = {}, y_b[n] = {};

//        int gap = ds % n_proc == 0 ? ds / n_proc : ds / n_proc + 1;
//        for (int i = 0; i < n_proc; i++) {
//            for (int j = i * gap; j < (i + 1) * gap - gap / 2 && j < ds; j++) {
////                cout << count << " ";
//                work[j] = count;
//                count++;
//            }
//            for (int j = (i + 1) * gap - gap / 2; j < (i + 1) * gap && j < ds; j++) {
////                cout << search_count << " ";
//                work[j] = search_count;
//                search_count--;
//            }
//        }

        scalar start = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(count, pitt, sum, row_n, n_dx_q_0, n_dx_q_1)
        {
#pragma omp barrier
#pragma omp for nowait
                for (index i = 0; i < n; i++) {
                    sum = 0;
                    index ni = n - 1 - i, nj = ni * n - (ni * (n - i)) / 2;
#pragma omp simd reduction(+ : sum)
                    for (index col = n - i; col < n; col++)
                        sum += R_A->x[nj + col] * z_x[col];
                    z_x[ni] = (y_A->x[ni] - sum) / R_A->x[nj + ni];
                }


            for (index j = 0; j < nswp && !flag; j++) {
#pragma omp for schedule(dynamic, 1) nowait //
                for (index i = 0; i < ds; i++) {
                    if (flag || i > iter) continue; // || i > iter
                    iter++;
                    pitt = i;
//                    pitt = j == 0 ? i : work[i];
                    count = 0;
                    n_dx_q_0 = n - (pitt + 1) * dx;
                    n_dx_q_1 = n - pitt * dx;
                    //The block operation
                    if (pitt != 0) {
                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                            sum = 0;
                            row_n = (n * row) - ((row * (row + 1)) / 2);
#pragma omp simd reduction(+ : sum)
                            for (index col = n_dx_q_1; col < n; col++) {
                                sum += R_A->x[row_n + col] * z_x[col];
                            }
                            y_b[row] = y_A->x[row] - sum;
                        }
                    } else
#pragma omp simd
                        for (index l = n_dx_q_0; l < n_dx_q_1; l++)
                            y_b[l] = y_A->x[l];

                    res[j] += ils_search_omp(n_dx_q_0, n_dx_q_1, 0, y_b, z_x);
                }
#pragma omp master
                {
                    if (j > 0 || init == -1) {
                        num_iter = j;
                        flag = res[j] < stop;
                    }
                }
            }
#pragma omp master
            {
                run_time = omp_get_wtime() - start;
            }
        }

        scalar run_time2 = omp_get_wtime() - start;

//#ifdef VERBOSE //1
//        printf("%d, %.3f, %.3f, ", count, run_time, run_time / run_time2);
//#endif
        returnType<scalar, index> reT = {z_B, run_time2, num_iter};
//        for (index i = 0; i < nswp; i++)
//            cout << res[i] << " ";
//        cout << endl;
        return reT;
    }
}
