//
// Created by shilei on 2021-01-13.
//
#include "../include/cils.h"

namespace cils {
    template<typename scalar, typename index, index n>
    inline scalar cils<scalar, index, n>::ils_search(const vector<scalar> *R_B, const vector<scalar> *y_B,
                                                     vector<index> *x, const bool is_constrained) {

        //variables
        index block_size = y_B->size(), k = block_size - 1, upper = pow(2, k) - 1;

        vector<index> z(block_size, 0), d(block_size, 0);
        vector<scalar> p(block_size, 0), c(block_size, 0);

        scalar newprsd, gamma, beta = INFINITY, counter = 0;

        c[k] = y_B->at(k) / R_B->at(R_B->size() - 1);
        z[k] = !is_constrained ? round(c[k]) : round(c[k]) < 0 ? 0 : round(c[k]) > upper ? upper : round(c[k]);
        gamma = R_B->at(R_B->size() - 1) * (c[k] - z[k]);

        //  Determine enumeration direction at level block_size
        d[k] = c[k] > z[k] ? 1 : -1;

        while (true) {
            newprsd = p[k] + gamma * gamma;
            counter++;
            if (counter > max_thre) {
                for (index l = 0; l < block_size; l++) {
                    x->at(l) = z[l];
                }
                break;
            }
            if (newprsd < beta) {
                if (k != 0) {
                    k--;
                    scalar sum = 0;
                    for (index col = k + 1; col < block_size; col++) {
                        sum += R_B->at((block_size * k) + col - ((k * (k + 1)) / 2)) * z[col];
                    }
                    scalar s = y_B->at(k) - sum;
                    scalar R_kk = R_B->at((block_size * k) + k - ((k * (k + 1)) / 2));
                    p[k] = newprsd;
                    c[k] = s / R_kk;
                    z[k] = !is_constrained ? round(c[k]) : round(c[k]) < 0 ? 0 : round(c[k]) > upper ? upper : round(
                            c[k]);
                    gamma = R_kk * (c[k] - z[k]);
                    d[k] = c[k] > z[k] ? 1 : -1;

                } else {
                    beta = newprsd;

                    for (index l = 0; l < block_size; l++) {
                        x->at(l) = z[l];
                    }

                    z[0] += d[0];
                    gamma = R_B->at(0) * (c[0] - z[0]);
                    d[0] = d[0] > 0 ? -d[0] - 1 : -d[0] + 1;
                }
            } else {
                if (k == block_size - 1) break;
                else {
                    k++;
                    z[k] += d[k];
                    gamma = R_B->at((block_size * k) + k - ((k * (k + 1)) / 2)) * (c[k] - z[k]);
                    d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                }
            }
        }
        return beta;
    }

    template<typename scalar, typename index, index n>
    inline scalar cils<scalar, index, n>::ils_search_omp(const index n_dx_q_0, const index n_dx_q_1,
                                                         const scalar *y_B, index *z_x,
                                                         const bool is_constrained) {

        //variables
        scalar sum, newprsd, gamma, beta = INFINITY;

        index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1, iter = 0, upper = pow(2, k) - 1, result;
        index count = 0, end_1 = n_dx_q_1 - 1, row_k = k + n_dx_q_0;
        index row_kk = (n + 1) * end_1 - ((end_1 * (end_1 + 1)) / 2);

        scalar p[dx], c[dx];
        index z[dx], d[dx], x[dx];

#pragma omp simd
        for (index l = 0; l < dx; l++) {
            x[l] = z_x[l + n_dx_q_0];
        }
        //  Initial squared search radius
        scalar R_kk = R_A->x[row_kk];
        c[k] = y_B[row_k] / R_kk;
        result = round(c[k]);
        z[k] = !is_constrained ? result : result < 0 ? 0 : result > upper ? upper : result;
        gamma = R_kk * (c[k] - z[k]);

        //  Determine enumeration direction at level block_size
        d[k] = c[k] > z[k] ? 1 : -1;

        //ILS search process
        while (true) {
            newprsd = p[k] + gamma * gamma;
#pragma omp atomic
            count++;
            if (count > program_def::max_search) {
#pragma omp simd
                for (index l = 0; l < dx; l++) {
                    x[l] = z[l];
                }
                beta = newprsd;
                break;
            }

            if (newprsd < beta) {
                if (k != 0) {
#pragma omp atomic
                    k--;
#pragma omp atomic
                    row_k--;
                    sum = 0;
                    row_kk = (n * row_k) - ((row_k * (row_k + 1)) / 2);
#pragma omp simd reduction(+ : sum)
                    for (index col = k + 1; col < dx; col++) {
                        sum += R_A->x[row_kk + col + n_dx_q_0] * z[col];
                    }
                    R_kk = R_A->x[row_kk + row_k];

                    p[k] = newprsd;
                    c[k] = (y_B[row_k] - sum) / R_kk;
                    result = round(c[k]);
                    z[k] = !is_constrained ? result : result < 0 ? 0 : result > upper ? upper : result;
                    gamma = R_kk * (c[k] - z[k]);

                    d[k] = c[k] > z[k] ? 1 : -1;

                } else {
                    beta = newprsd;
#pragma omp simd
                    for (index l = 0; l < dx; l++) {
                        x[l] = z[l];
                    }
#pragma omp atomic
                    iter++;
                    if (iter > program_def::search_iter) break;

                    z[0] += d[0];
                    gamma = R_A->x[0] * (c[0] - z[0]);
                    d[0] = d[0] > 0 ? -d[0] - 1 : -d[0] + 1;
                }
            } else {
                if (k == dx - 1) break;
                else {
#pragma omp atomic
                    k++;
#pragma omp atomic
                    row_k++;
#pragma omp atomic
                    z[k] += d[k];
                    gamma = R_A->x[(n * row_k) + row_k - ((row_k * (row_k + 1)) / 2)] * (c[k] - z[k]);
                    d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                }
            }
        }
#pragma omp simd
        for (index l = 0; l < dx; l++) {
            z_x[l + n_dx_q_0] = x[l];
        }
        return beta;
    }

}
