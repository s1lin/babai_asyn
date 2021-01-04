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

#ifndef CILS_H
#define CILS_H

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
#include <bitset>
#include <math.h>
//#include <Eigen/Dense>
#include "config.h"

using namespace std;
//using Eigen::MatrixXd;
//using Eigen::VectorXd;
/**
 * namespace of cils
 */
namespace cils {
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
     * Return scalar pointer array along with the size.
     * @tparam scalar
     * @tparam index
     */
    template<typename scalar, typename index>
    struct returnType {
        vector<index> *x;
        scalar run_time;
        index num_iter;
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
    inline scalar find_residual(const scalarType<scalar, index> *R,
                                const scalarType<scalar, index> *y,
                                const vector<index> *x) {
        scalar res = 0, sum = 0;
        for (index i = 0; i < n; i++) {
            sum = 0;
            for (index j = i; j < n; j++) {
                sum += x->at(j) * R->x[(n * i) + j - ((i * (i + 1)) / 2)];
            }
            res += (y->x[i] - sum) * (y->x[i] - sum);
        }
        return std::sqrt(res);
    }

    /**
     *
     * @tparam scalar
     * @tparam index
     * @tparam n
     * @param x_b
     * @param x_t
     * @param is_binary
     * @return
     */
    template<typename scalar, typename index, index n>
    inline scalar find_bit_error_rate(const vector<index> *x_b,
                                      const vector<index> *x_t,
                                      const index is_binary) {
        index error = 0, size = is_binary ? 1 : sizeof(index);
        for (index i = 0; i < n; i++) {
            std::string binary_x_b = std::bitset<sizeof(index)>(x_b->at(i)).to_string(); //to binary
            std::string binary_x_t = std::bitset<sizeof(index)>(x_t->at(i)).to_string();
            if (is_binary) {
                if (binary_x_b[sizeof(index) - 1] != binary_x_t[sizeof(index) - 1])
                    error++;
            } else {
                for (index j = 0; j < (index) sizeof(index); j++) {
                    if (binary_x_b[j] != binary_x_t[j])
                        error++;
                }
            }
        }
        return (scalar) error / (n * size);
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
    inline scalar norm(const vector<scalar> *x,
                       const vector<scalar> *y) {
        scalar res = 0;
        for (index i = 0; i < n; i++) {
            res += (y->at(i) - x->at(i)) * (y->at(i) - x->at(i));
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
    inline scalar diff(const vector<scalar> *x,
                       const vector<scalar> *y) {
        scalar d = 0;
#pragma omp simd reduction(+ : d)
        for (index i = 0; i < n; i++) {
            d += (y->at(i) - x->at(i));
        }
        return d;
    }

    /**
     * Simple function for displaying the struct scalarType
     * @tparam scalar
     * @tparam index
     * @param x
     */
    template<typename scalar, typename index>
    void display_scalarType(scalarType<scalar, index> *x) {
        for (index i = 0; i < x->size; i++) {
            cout << x->x[i] << endl;
        }
    }

    /**
     * Simple function for displaying the struct scalarType
     * @tparam scalar
     * @tparam index
     * @param x
     */
    template<typename scalar, typename index>
    inline void display_vector(const vector<index> *x) {
        for (index i = 0; i < x->size(); i++) {
            cout << x->at(i) << " ";
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
    inline vector<scalar> find_block_Rii(const scalarType<scalar, index> *R_B,
                                         const index row_begin, const index row_end,
                                         const index col_begin, const index col_end,
                                         const index block_size) {
        index size = col_end - col_begin;
        vector<scalar> R_b_s(size * (1 + size) / 2, 0);
        index counter = 0, i = 0;

        //The block operation
        for (index row = row_begin; row < row_end; row++) {
            //Translating the index from R(matrix) to R_B(compressed array).
            for (index col = row; col < col_end; col++) {
                i = (block_size * row) + col - ((row * (row + 1)) / 2);
                //Put the value into the R_b_s
                R_b_s[counter] = R_B->x[i];
                counter++;
            }
        }

        return R_b_s;
    }

    /**
     *
     * @tparam scalar
     * @tparam index
     * @return
     */
    template<typename scalar, typename index>
    inline scalar find_success_prob_babai(const scalarType<scalar, index> *R_B,
                                          const index row_begin, const index row_end,
                                          const index block_size, const scalar sigma) {
        scalar p = 1;
        for (index row = row_begin; row < row_end; row++) {
            index i = (block_size * row) + row - ((row * (row + 1)) / 2);
            p = p * erf(abs(R_B->x[i]) / 2 / sigma / sqrt(2));
        }
        return p;
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
    inline vector<scalar> block_residual_vector(const scalarType<scalar, index> *R_B,
                                                const vector<index> *x,
                                                const vector<scalar> *y,
                                                const index row_begin, const index row_end,
                                                const index col_begin, const index col_end) {
        index block_size = row_end - row_begin;
        vector<scalar> y_b_s(block_size, 0);

        index counter = 0, prev_i = 0, i;
        scalar sum = 0;

        //The block operation
        for (index row = row_begin; row < row_end; row++) {
            //Translating the index from R(matrix) to R_B(compressed array).
            for (index col = col_begin; col < col_end; col++) {
                i = (col_end * row) + col - ((row * (row + 1)) / 2);
                sum += R_B->x[i] * x->at(counter);
                counter++;
            }
            y_b_s[prev_i] = y->at(row - row_begin) - sum;
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
    inline vector<index> find_block_x(const vector<index> *x,
                                      const index begin,
                                      const index end) {
        vector<index> z(end - begin, 0);
        for (index i = begin; i < end; i++) {
            z[i - begin] = x->at(i);
        }
        return z;
    }

    /**
     *
     * @tparam scalar
     * @tparam index
     * @param y*
     * @param begin
     * @param end
     * @return scalarType*
     */
    template<typename scalar, typename index>
    inline vector<scalar> find_block_x(const scalarType<scalar, index> *x,
                                       const index begin,
                                       const index end) {
        vector<scalar> z(end - begin, 0);
        for (index i = begin; i < end; i++) {
            z[i - begin] = x->x[i];
        }
        return z;
    }


    /**
     * cils class object
     * @tparam scalar
     * @tparam index
     * @tparam is_read
     * @tparam is_write
     * @tparam n
     */
    template<typename scalar, typename index, bool is_read, index n>
    class cils {

    public:
        index qam, snr;
        scalar init_res, sigma;
        vector<index> x_R, x_t;
        scalarType<scalar, index> *R_A, *y_A, *A;
//        Eigen::MatrixXd A, R, Q;
//        Eigen::VectorXd y, x_tV;
    private:
        /**
         *
         * read the problem from files
         */
        void read_nc(string filename);

        /**
         *
         * read the problem from files
         */
        void read_csv(bool is_qr);

        /**
         *
         * @param n_dx_q_0
         * @param n_dx_q_1
         * @param y
         * @return
         */
        inline scalar ils_search_omp(const index n_dx_q_0, const index n_dx_q_1,
                                     const bool is_first,
                                     const scalar *y_B, index *z_x) {

            //variables
            scalar sum, newprsd, gamma, beta = INFINITY;

            index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1, iter = 0;
            index count = 0, end_1 = n_dx_q_1 - 1, row_k = k + n_dx_q_0;
            index row_kk = (n + 1) * end_1 - ((end_1 * (end_1 + 1)) / 2);

            scalar p[dx], c[dx];
            index z[dx], d[dx], x[dx];

            //  Initial squared search radius
            scalar R_kk = R_A->x[row_kk];
            c[k] = y_B[row_k] / R_kk;
            z[k] = round(c[k]);
            gamma = R_kk * (c[k] - z[k]);

            //  Determine enumeration direction at level block_size
            d[k] = c[k] > z[k] ? 1 : -1;

            //ILS search process
            while (true) {
#pragma omp atomic
                count++;
                if (!is_first && count > program_def::max_search) {
#pragma omp simd
                    for (index l = 0; l < dx; l++) {
                        x[l] = z[l];
                    }
                    break;
                }
                newprsd = p[k] + gamma * gamma;
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
                        z[k] = round(c[k]);
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
            if (is_first) return count;
            return beta;
        }


        /**
         *
         * @param R_B
         * @param y_B
         * @param x
         */
        inline scalar ils_search(const vector<scalar> *R_B, const vector<scalar> *y_B,
                                 vector<index> *x) {

            //variables
            index block_size = y_B->size();

            vector<index> z(block_size, 0), d(block_size, 0);
            vector<scalar> p(block_size, 0), c(block_size, 0);

            scalar newprsd, gamma, beta = INFINITY;
            index k = block_size - 1;

            c[block_size - 1] = y_B->at(block_size - 1) / R_B->at(R_B->size() - 1);
            z[block_size - 1] = round(c[block_size - 1]);
            gamma = R_B->at(R_B->size() - 1) * (c[block_size - 1] - z[block_size - 1]);

            //  Determine enumeration direction at level block_size
            d[block_size - 1] = c[block_size - 1] > z[block_size - 1] ? 1 : -1;

            while (true) {
                newprsd = p[k] + gamma * gamma;
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

                        z[k] = round(c[k]);
                        gamma = R_kk * (c[k] - z[k]);
                        d[k] = c[k] > z[k] ? 1 : -1;

                    } else {
                        beta = newprsd;
                        //Deep Copy of the result
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


        inline vector<index> ils_reduction(const scalarType<scalar, index> *B, const vector<scalar> *y_B) {
            std::array<std::array<index, n>, 2> colNormB;
            for (index j = 0; j < n; j++) {
                scalar b_0 = B[0][j];
                scalar b_1 = B[1][j];
                colNormB[0][j] = b_0 * b_0 + b_1 * b_1;
            }
            index n_dim = n;
            for (index k = 0; k < n_dim; k++) {

            }
        }

    public:
        cils(index qam, index snr) {
            this->R_A = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
            this->y_A = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
            this->R_A->x = new scalar[n * (n + 1) / 2 + 1]();
            this->y_A->x = new scalar[n]();

            this->x_R = vector<index>(n, 0);
            this->x_t = vector<index>(n, 0);

            this->init_res = INFINITY;
            this->qam = qam;
            this->snr = snr;
            this->sigma = (scalar) sqrt(((pow(4, qam) - 1) * log2(n)) / (6 * pow(10, ((scalar) snr / 10.0))));
            this->R_A->size = n * (n + 1) / 2;
            this->y_A->size = n;
        }

        ~cils() {
            free(R_A);
            free(y_A);
        }

        void init(bool is_qr, bool is_nc);

        /**
         *
         * @param n_proc: number of Processors/Threads
         * @param nswp: maximum number of iterations
         * @param z_B: estimation of the true parameter
         * @return
         */
        returnType<scalar, index>
        cils_babai_search_omp(const index n_proc, const index nswp, vector<index> *z_B);

        /**
         *
         * @param n_proc
         * @param nswp
         * @param z_B
         * @return
         */
        returnType<scalar, index>
        cils_babai_search_cuda(const index nswp, vector<index> *z_B);

        /**
         *
         * @param n_proc
         * @param nswp
         * @param update
         * @param z_B
         * @param z_B_p
         * @return
         */
        returnType<scalar, index>
        cils_back_solve(vector<index> *z_B);


        /**
         *
         * @param z_B
         * @return
         */
        returnType<scalar, index>
        cils_babai_search_serial(vector<index> *z_B);

        /**
         *
         * @param z_B
         * @param d
         * @return
         */
        returnType<scalar, index>
        cils_block_search_serial(const vector<index> *d, vector<index> *z_B);

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
        returnType<scalar, index>
        cils_block_search_cuda(index nswp, scalar stop, const vector<index> *d, vector<index> *z_B);


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
        returnType<scalar, index>
        cils_block_search_omp(const index n_proc, const index nswp, const index stop, const index schedule,
                              const vector<index> *d, vector<index> *z_B);

        returnType<scalar, index>
        cils_reduction();


    };
}
#endif