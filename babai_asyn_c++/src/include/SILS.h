/** \file
 * \brief Computation of integer least square problem
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

#ifndef SILS_H
#define SILS_H

#include <iostream>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <ctime>
#include <iomanip>
#include <algorithm>
#include <netcdf.h>

using namespace std;

/**
 * namespace of sils
 */
namespace sils {

    /**
     * Return scalar pointer array along with the size.
     * @tparam scalar
     * @tparam index
     */
    template<typename scalar, typename index>
    struct scalarType {
        scalar *x;
        index size;
    };

    /**
     * Return the result of norm2(y-R*x).
     * @tparam scalar
     * @tparam index
     * @tparam n
     * @param R
     * @param y
     * @param x
     * @return residual
     */
    template<typename scalar, typename index, index n>
    inline scalar find_residual(scalarType<scalar, index> *R,
                                scalarType<scalar, index> *y,
                                scalarType<scalar, index> *x) {
        scalar res = 0;
        for (index i = 0; i < n; i++) {
            scalar sum = 0;
            for (index j = i; j < n; j++) {
                sum += x->x[j] * R->x[(n * i) + j - ((i * (i + 1)) / 2)];
            }
            res += (y->x[i] - sum) * (y->x[i] - sum);
        }
        return std::sqrt(res);
    }

    /**
     * Return the result of norm2(y-R*x).
     * @tparam scalar
     * @tparam index
     * @tparam n
     * @param R
     * @param y
     * @param x
     * @return residual
     */
    template<typename scalar, typename index, index n>
    inline scalar norm(scalarType<scalar, index> *x,
                       scalarType<scalar, index> *y) {
        scalar res = 0;
        for (index i = 0; i < n; i++) {
            res += (y->x[i] - x->x[i]) * (y->x[i] - x->x[i]);
        }
        return std::sqrt(res);
    }

    /**
     * Simple function for displaying the struct scalarType
     * @tparam scalar
     * @tparam index
     * @param x
     */
    template<typename scalar, typename index>
    inline void display_scalarType(scalarType<scalar, index> *x) {
        for (int i = 0; i < x->size; i++) {
            cout << x->x[i] << " ";
        }
        cout << endl;
    }


    /**
     * Simple function for displaying the struct scalarType
     * @tparam scalar
     * @tparam index
     * @param x
     */
    template<typename scalar, typename index>
    inline void display_array(scalar *x, index size) {
        for (int i = 0; i < size; i++) {
            cout << x[i] << " ";
        }
        cout << endl;
    }


    /**
     * Block Operation on R_B (compressed array).
     * @tparam scalar
     * @tparam index
     * @param R_B
     * @param begin: the starting index of the nxn block
     * @param end: the end index of the nxn block
     * @param n: the size of R_B.
     * @return scalarType
     */
    template<typename scalar, typename index>
    inline scalarType<scalar, index> *find_block_Rii(scalarType<scalar, index> *R_B,
                                                     const index row_begin, const index row_end,
                                                     const index col_begin, const index col_end,
                                                     const index block_size) {
        auto *R_b_s = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
        index size = col_end - col_begin;
        index R_b_s_size = size * (1 + size) / 2;
        R_b_s->x = (scalar *) calloc(R_b_s_size, sizeof(scalar));
        R_b_s->size = R_b_s_size;
        index counter = 0, i = 0;

        //The block operation
        for (index row = row_begin; row < row_end; row++) {
            //Translating the index from R(matrix) to R_B(compressed array).
            for (index col = row; col < col_end; col++) {
                i = (block_size * row) + col - ((row * (row + 1)) / 2);
                //Put the value into the R_b_s.x
                R_b_s->x[counter] = R_B->x[i];
                counter++;
            }
        }
        return R_b_s;
    }

    /**
     *
     * @tparam scalar
     * @tparam index
     * @param R_B
     * @param x
     * @param y
     * @param row_begin
     * @param row_end
     * @param col_begin
     * @param col_end
     * @return
     */
    template<typename scalar, typename index>
    inline scalarType<scalar, index> *block_residual_vector(scalarType<scalar, index> *R_B,
                                                            scalarType<scalar, index> *x,
                                                            scalarType<scalar, index> *y,
                                                            const index row_begin, const index row_end,
                                                            const index col_begin, const index col_end) {
        index block_size = row_end - row_begin;
        auto *y_b_s = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
        y_b_s->size = block_size;
        y_b_s->x = (scalar *) calloc(block_size, sizeof(scalar));

        index counter = 0, prev_i = 0, i;
        scalar sum = 0;

        //The block operation
        for (index row = row_begin; row < row_end; row++) {
            //Translating the index from R(matrix) to R_B(compressed array).
            for (index col = col_begin; col < col_end; col++) {
                i = (col_end * row) + col - ((row * (row + 1)) / 2);
                sum += R_B->x[i] * x->x[counter];
                counter++;
            }
            y_b_s->x[prev_i] = y->x[row - row_begin] - sum;
            prev_i++;
            sum = counter = 0;
        }
        return y_b_s;
    }

    /**
     *
     * @tparam scalar
     * @tparam index
     * @param x*
     * @param begin
     * @param end
     * @return scalarType*
     */
    template<typename scalar, typename index>
    inline scalarType<scalar, index> *find_block_x(scalarType<scalar, index> *x,
                                                   const index begin,
                                                   const index end) {
        auto *z = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
        z->size = end - begin;
        z->x = (scalar *) calloc(end - begin, sizeof(scalar));

        for (index i = begin; i < end; i++) {
            z->x[i - begin] = x->x[i];
        }
        return z;
    }

    /**
     *
     * @tparam scalar
     * @tparam index
     * @param x*
     * @param begin
     * @param end
     * @return scalarType*
     */
    template<typename scalar, typename index>
    inline scalarType<scalar, index> *find_block_x_omp(scalarType<scalar, index> *x,
                                                       const index begin,
                                                       const index end) {
        auto *z = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
        z->size = end - begin;
        z->x = (scalar *) calloc(end - begin, sizeof(scalar));
#pragma omp parallel for
        for (index i = begin; i < end; i++) {
            z->x[i - begin] = x->x[i];
        }
        return z;
    }


    template<typename scalar, typename index>
    inline scalarType<scalar, index> *create_scalarType(const index block_size) {
        auto *z = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
        z->size = block_size;
        z->x = (scalar *) calloc(block_size, sizeof(scalar));
        return z;
    }

    /**
     *
     * @tparam scalar
     * @tparam index
     * @param first
     * @param second
     * @return scalarType*
     */
    template<typename scalar, typename index>
    inline scalarType<scalar, index> *concatenate_array(scalarType<scalar, index> *first,
                                                        scalarType<scalar, index> *second) {
        auto *z = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
        index size = first->size + second->size;
        z->x = (scalar *) calloc(size, sizeof(scalar));
        z->size = size;

        for (int i = 0; i < first->size; i++) {
            z->x[i] = first->x[i];
        }
        for (int i = first->size; i < size; i++) {
            z->x[i] = second->x[i - first->size];
        }
        return z;
    }

    /**
     * SILS class object
     * @tparam scalar
     * @tparam index
     * @tparam is_read
     * @tparam is_write
     * @tparam n
     */
    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    class SILS {
    public:
        scalar init_res, noise;
        scalarType<scalar, index> R_A, y_A, x_R, x_tA;
    private:
        /**
         * read the problem from files
         */
        void read();

        /**
         * Write ONLY the R_A array into files.
         */
        void write();

        /**
         * Warning: OPENMP enabled.
         * @param i
         * @param z_B
         * @return
         */
        inline scalar do_solve(const index i, const scalar *z_B) {
            scalar sum = 0;
#pragma omp simd reduction(+ : sum)
            for (index col = n - i; col < n; col++)
                sum += R_A.x[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * z_B[col];

            return round((y_A.x[n - 1 - i] - sum) / R_A.x[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
        }


        inline scalarType<scalar, index> *do_block_solve(const index n_dx_q_0,
                                                         const index n_dx_q_1,
                                                         scalarType<scalar, index> *z_B) {
            scalar sum = 0, newprsd, gamma, beta = INFINITY;
            index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1;
            index end_1 = n_dx_q_1 - 1, row_k = k + n_dx_q_0;

            auto *p = (scalar *) calloc(dx, sizeof(scalar));
            auto *c = (scalar *) calloc(dx, sizeof(scalar));
            auto *z = (scalar *) calloc(dx, sizeof(scalar));
            auto *d = (index *) calloc(dx, sizeof(index));
            auto *y = (scalar *) calloc(dx, sizeof(scalar));

            //The block operation
#pragma omp simd reduction(+ : sum)
            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                //Translating the index from R(matrix) to R_B(compressed array).
                for (index col = n_dx_q_1; col < n; col++) {
                    sum += R_A.x[(n * row) + col - ((row * (row + 1)) / 2)] * z_B->x[col];
                }
                y[row - n_dx_q_0] = y_A.x[row] - sum;
                sum = 0;
            }

            //  Initial squared search radius
            scalar R_kk = R_A.x[(n * end_1) + end_1 - ((end_1 * (end_1 + 1)) / 2)];
            c[k] = y[k] / R_kk;
            z[k] = round(c[k]);
            gamma = R_kk * (c[k] - z[k]);

            //  Determine enumeration direction at level block_size
            d[dx - 1] = c[dx - 1] > z[dx - 1] ? 1 : -1;

            //ILS search process
            while (true) {
                newprsd = p[k] + gamma * gamma;
                if (newprsd < beta) {
                    if (k != 0) {
                        k--;
                        row_k--;
                        sum = 0;

#pragma omp simd reduction(+ : sum)
                        for (index col = k + 1; col < dx; col++) {
                            sum += R_A.x[(n * row_k) + col + n_dx_q_0 - ((row_k * (row_k + 1)) / 2)] * z[col];
                        }
                        R_kk = R_A.x[(n * row_k) + row_k - ((row_k * (row_k + 1)) / 2)];

                        p[k] = newprsd;
                        c[k] = (y[k] - sum) / R_kk;
                        z[k] = round(c[k]);
                        gamma = R_kk * (c[k] - z[k]);

                        d[k] = c[k] > z[k] ? 1 : -1;

                    } else {
                        beta = newprsd;
#pragma omp parallel for
                        for (int l = n_dx_q_0; l < n_dx_q_1; l++) {
                            z_B->x[l] = z[l - n_dx_q_0];
                        }

                        z[0] += d[0];
                        gamma = R_A.x[0] * (c[0] - z[0]);
                        d[0] = d[0] > 0 ? -d[0] - 1 : -d[0] + 1;
                    }
                } else {
                    if (k == dx - 1) break;
                    else {
                        k++;
                        row_k++;
                        z[k] += d[k];
                        gamma = R_A.x[(n * row_k) + row_k - ((row_k * (row_k + 1)) / 2)] * (c[k] - z[k]);
                        d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                    }
                }
            }
            return z_B;
        }


    public:
        explicit SILS(scalar noise);

        ~SILS() {
            free(R_A.x);
            free(x_R.x);
            free(y_A.x);
            free(x_tA.x);
        }

        void init();


        /**
         *
         * @param R_B
         * @param y_B
         * @param z_B
         * @return
         */
        scalarType<scalar, index> *sils_search(scalarType<scalar, index> *R_B,
                                               scalarType<scalar, index> *y_B);

        /**
         *
         * @param n_proc
         * @param nswp
         * @param update
         * @param z_B
         * @param z_B_p
         * @return
         */
        scalarType<scalar, index> *sils_babai_search_omp(index n_proc, index nswp, index *update,
                                                         scalarType<scalar, index> *z_B,
                                                         scalarType<scalar, index> *z_B_p);

        /**
         *
         * @param z_B
         * @return
         */
        scalarType<scalar, index> *sils_babai_search_serial(scalarType<scalar, index> *z_B);

        /**
         * @deprecated
         * @param R_B
         * @param y_B
         * @param z_B
         * @param d
         * @param block_size
         * @return
         */
        scalarType<scalar, index> *
        sils_block_search_serial_recursive(scalarType<scalar, index> *R_B,
                                           scalarType<scalar, index> *y_B,
                                           scalarType<scalar, index> *z_B,
                                           vector<index> d);

        scalarType<scalar, index> *
        sils_block_search_serial(scalarType<scalar, index> *R_B,
                                 scalarType<scalar, index> *y_B,
                                 scalarType<scalar, index> *z_B,
                                 scalarType<index, index> *d);

        /**
         *
         * @param n_proc
         * @param nswp
         * @param R_B
         * @param y_B
         * @param z_B
         * @param d
         * @return
         */
        scalarType<scalar, index> *
        sils_block_search_omp(index n_proc, index nswp,
                              scalarType<scalar, index> *R_B,
                              scalarType<scalar, index> *y_B,
                              scalarType<scalar, index> *z_B,
                              scalarType<index, index> *d);


    };
}
#endif