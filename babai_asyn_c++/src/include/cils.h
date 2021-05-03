﻿/** \file
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
#include "config.h"
#include "MatlabDataArray.hpp"
#include "MatlabEngine.hpp"
#include <numeric>

using namespace std;

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
        vector<scalar> x;
        scalar run_time;
        scalar num_iter; //true_res, error
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
    inline scalar find_residual(const scalarType<scalar, index> *A,
                                const scalarType<scalar, index> *y,
                                const vector<scalar> *x) {
        scalar res = 0, sum = 0;
        for (index i = 0; i < n; i++) {
            sum = 0;
            for (index j = 0; j < n; j++) {
                sum += x->at(j) * A->x[i * n + j];
            }
            res += (y->x[i] - sum) * (y->x[i] - sum);
        }
        return std::sqrt(res);
    }

    template<typename scalar, typename index, index n>
    inline vector<scalar> find_residual_by_block(const scalarType<scalar, index> *A,
                                                 const scalarType<scalar, index> *y,
                                                 const vector<index> *d, vector<scalar> *x) {
        index ds = d->size();
        vector<scalar> y_b(n, 0), res(ds, 0);
        for (index i = 0; i < ds; i++) {
            index n_dx_q_1 = d->at(i);
            index n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);

            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                scalar sum = 0;
                for (index col = n_dx_q_1; col < n; col++) {
                    sum += A->x[col + row * n] * x->at(row);
                }
                y_b[row] = y->x[row] - sum;
            }

            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                scalar sum = 0;
                for (index col = n_dx_q_0; col < n_dx_q_1; col++) {
                    sum += A->x[col + row * n] * x->at(row);
                }
                res[i] += (y_b[row] - sum) * (y_b[row] - sum);
            }

            res[i] = std::sqrt(res[i]);
        }
        return res;
    }

    template<typename scalar, typename index, index n>
    inline vector<scalar> find_bit_error_rate_by_block(const vector<scalar> *x_b, const vector<scalar> *x_t,
                                                       const vector<index> *d, const index k) {
        index ds = d->size();
        vector<scalar> ber(ds, 0);
        for (index i = 0; i < ds; i++) {
            index error = 0;
            index n_dx_q_1 = d->at(i);
            index n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);
            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                std::string binary_x_b, binary_x_t;
                switch (k) {
                    case 1:
                        binary_x_b = std::bitset<1>((index) x_b->at(row)).to_string(); //to binary
                        binary_x_t = std::bitset<1>((index) x_t->at(row)).to_string();
                        break;
                    case 2:
                        binary_x_b = std::bitset<2>((index) x_b->at(row)).to_string(); //to binary
                        binary_x_t = std::bitset<2>((index) x_t->at(row)).to_string();
                        break;
                    default:
                        binary_x_b = std::bitset<3>((index) x_b->at(row)).to_string(); //to binary
                        binary_x_t = std::bitset<3>((index) x_t->at(row)).to_string();
                        break;
                }
//                cout << binary_x_b << "-" << binary_x_t << " ";
                for (index j = 0; j < k; j++) {
                    if (binary_x_b[j] != binary_x_t[j])
                        error++;
                }
            }
//            cout << error << ", \n";
            ber[i] = (scalar) error / ((n_dx_q_1 - n_dx_q_0) * k);
        }

        return ber;
    }


    template<typename scalar, typename index, index n>
    inline void vector_permutation(const scalarType<scalar, index> *Z,
                                   vector<scalar> *x) {
        vector<scalar> x_P(n, 0);
        for (index i = 0; i < n; i++) {
            for (index j = 0; j < n; j++) {

                if (Z->x[j * n + i] != 0) {
                    x_P[i] = x->at(j);
                    break;
                }
            }
        }

        for (index i = 0; i < n; i++) {
            x->at(i) = x_P[i];
        }
    }

    template<typename scalar, typename index, index n>
    inline void vector_reverse_permutation(const scalarType<scalar, index> *Z,
                                           vector<scalar> *x) {
        vector<scalar> x_P(n, 0);
        for (index i = 0; i < n; i++) {
            for (index j = 0; j < n; j++) {
                if (Z->x[i * n + j] != 0) {
                    x_P[i] = x->at(j);
                    break;
                }
            }
        }

        for (index i = 0; i < n; i++) {
            x->at(i) = x_P[i];
        }
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
    inline scalar find_bit_error_rate(const vector<scalar> *x_b,
                                      const vector<scalar> *x_t,
                                      const index k) {
        index error = 0;
        for (index i = 0; i < n; i++) {
            std::string binary_x_b, binary_x_t;
            switch (k) {
                case 1:
                    binary_x_b = std::bitset<1>((index) x_b->at(i)).to_string(); //to binary
                    binary_x_t = std::bitset<1>((index) x_t->at(i)).to_string();
                    break;
                case 2:
                    binary_x_b = std::bitset<2>((index) x_b->at(i)).to_string(); //to binary
                    binary_x_t = std::bitset<2>((index) x_t->at(i)).to_string();
                    break;
                default:
                    binary_x_b = std::bitset<3>((index) x_b->at(i)).to_string(); //to binary
                    binary_x_t = std::bitset<3>((index) x_t->at(i)).to_string();
                    break;
            }
//            cout << binary_x_b << "-" << binary_x_t << " ";
            for (index j = 0; j < k; j++) {
                if (binary_x_b[j] != binary_x_t[j])
                    error++;
            }
        }
        return (scalar) error / (n * k);
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
    inline void display_vector(const vector<scalar> *x) {
        scalar sum = 0;
        for (index i = 0; i < x->size(); i++) {
            printf("%.5f, ", x->at(i));
            sum += x->at(i);
        }

        printf("SUM = %.5f\n", sum);
    }

    /**
     * Simple function for displaying the struct scalarType
     * @tparam scalar
     * @tparam index
     * @param x
     */
    template<typename scalar, typename index>
    inline void display_vector_by_block(const vector<index> *d, const vector<scalar> *x) {
        cout << endl;
        index ds = d->size();
        for (index i = 0; i < ds; i++) {
            index n_dx_q_1 = d->at(i);
            index n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);
            for (index h = n_dx_q_0; h < n_dx_q_1; h++)
                cout << x->at(h) << " ";
            cout << endl;
        }
        cout << accumulate(x->begin(), x->end(), 0) << endl;
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
     * Evaluating the decomposition
     * @param eval
     * @param qr_eval
     */
    template<typename scalar, typename index, index n>
    scalar qr_validation(const scalarType<scalar, index> *A,
                         const scalarType<scalar, index> *Q,
                         const scalarType<scalar, index> *R,
                         const scalarType<scalar, index> *R_A,
                         const index eval, const index qr_eval) {
        index i, j, k;
        scalar sum, error = 0;
        auto c = new scalar[n * n]();
        if (eval == 1) {
            printf("\nPrinting A...\n");
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    printf("%.3f ", A->x[j * n + i]);
                }
                printf("\n");
            }

            printf("\nPrinting Q...\n");
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    printf("%.3f ", Q->x[j * n + i]);
                }
                printf("\n");
            }

            printf("\nPrinting QT...\n");
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    printf("%.3f ", Q->x[i * n + j]);
                }
                printf("\n");
            }

            printf("\nPrinting R...\n");
//            cout << setw(10);
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    cout << setw(11) << R->x[j * n + i] << " ";
//                    printf("%-1.3f ", R->x[j * n + i]);
                }
                printf("\n");
            }

//            printf("\nPrinting R_A...\n");
//            for (i = 0; i < n; i++) {
//                for (j = i; j < n; j++) {
//                    printf("%.3f ", R_A->x[(n * i) + j - ((i * (i + 1)) / 2)]);
//                }
//                printf("\n");
//            }
        }
        if (qr_eval == 1) {
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    sum = 0;
                    for (k = 0; k < n; k++) {
                        sum = sum + Q->x[k * n + i] * R->x[j * n + k]; //IF not use PYTHON!!!!
//                        sum = sum + Q->x[i * n + k] * R->x[k * n + j];
                    }
//                    c[i * n + j] = sum;
                    c[j * n + i] = sum;
                }
            }
            if (eval == 1) {
                printf("\nQ*R (Init A matrix) : \n");
                for (i = 0; i < n; i++) {
                    for (j = 0; j < n; j++) {
                        printf("%.3f ", c[j * n + i]);
                    }
                    printf("\n");
                }
            }

            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    error += fabs(c[j * n + i] - A->x[j * n + i]);
                }
            }
            printf(", Error: %e\n", error);
        }

        delete[] c;
        return error;
    }

    /**
     * cils class object
     * @tparam scalar
     * @tparam index
     * @tparam is_read
     * @tparam is_write
     * @tparam n
     */
    template<typename scalar, typename index, index n>
    class cils {

    public:
        index qam, snr, upper, lower;
        scalar init_res, sigma;
        vector<scalar> x_R, x_t;
        scalarType<scalar, index> *R_A, *y_A, *y_L, *A, *R, *Q, *v_A, *Z;
    private:
        /**
         * read the problem from files
         */
        void read_nc(string filename);

        /**
         * read the problem from files
         */
        void read_csv();

        /**
         * ils_search_omp(n_dx_q_0, n_dx_q_1, y_B, z_x) produces the optimal solution to
         * the upper triangular integer least squares problem
         * min_{z}||y-Rz|| by a depth-first search algorithm.
         * @param n_dx_q_0
         * @param n_dx_q_1
         * @param y_B
         * @param z_x
         * @param is_constrained
         * @return
         */
        inline scalar ils_search_omp(const index n_dx_q_0, const index n_dx_q_1,
                                     const scalar *y_B, scalar *z_x);


        /**
         * ils_search(n_dx_q_0, n_dx_q_1, y_B, z_x) produces the optimal solution to
         * the upper triangular integer least squares problem
         * min_{z}||y-Rz|| by a depth-first search algorithm.
         * @param R_B
         * @param y_B
         * @param x
         * @param is_constrained
         * @return
         */
        inline scalar ils_search(const index n_dx_q_0, const index n_dx_q_1,
                                 const vector<scalar> *y_B, vector<scalar> *z_x);


        /**
         * ils_search_obils(n_dx_q_0, n_dx_q_1, y_B, z_x) produces the optimal solution to
         * the upper triangular box-constrained integer least squares problem
         * min_{z}||y-Rz|| s.t. z in [l, u] by a search algorithm.
         * @param n_dx_q_0
         * @param n_dx_q_1
         * @param y_B
         * @param z_x
         * @return
         */
        inline scalar ils_search_obils(const index n_dx_q_0, const index n_dx_q_1,
                                       const vector<scalar> *y_B, vector<scalar> *z_x);


        /**
         * ils_search_obils_omp(n_dx_q_0, n_dx_q_1, y_B, z_x) produces the optimal solution to
         * the upper triangular box-constrained integer least squares problem
         * min_{z}||y-Rz|| s.t. z in [l, u] by a search algorithm.
         * @deprecated
         * @param n_dx_q_0
         * @param n_dx_q_1
         * @param i
         * @param ds
         * @param y_B
         * @param z_x
         * @return
         */
        inline bool ils_search_obils_omp(const index n_dx_q_0, const index n_dx_q_1,
                                         const index i, const index ds, scalar *y_B, scalar *z_x);

        void value_input_helper(matlab::data::TypedArray<scalar> const x, scalarType<scalar, index> *arr);

        void value_input_helper(matlab::data::TypedArray<scalar> const x, vector<scalar> *arr);


    public:
        cils(index qam, index snr) {
            this->R_A = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
            this->y_A = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));

            this->R_A->x = new scalar[n * (n + 1) / 2 + 1]();
            this->y_A->x = new scalar[n]();

            this->x_R = vector<scalar>(n, 0);
            this->x_t = vector<scalar>(n, 0);

            this->init_res = INFINITY;
            this->qam = qam;
            this->snr = snr;
            this->sigma = (scalar) sqrt(((pow(4, qam) - 1) * n) / (6 * pow(10, ((scalar) snr / 10.0))));
            this->upper = pow(2, qam) - 1;

            this->R_A->size = n * (n + 1) / 2;
            this->y_A->size = n;

        }

        ~cils() {
            delete[] R_A->x;
            delete[] y_A->x;
            free(R_A);
            free(y_A);
            if (!program_def::is_read) {
                if (program_def::is_matlab) {
                    delete[] A->x;
                    delete[] R->x;
                    delete[] Z->x;

                    free(A);
                    free(R);
                    free(Z);
                } else {
                    delete[] A->x;
                    delete[] R->x;
                    delete[] v_A->x;
                    delete[] Q->x;
                    delete[] Z->x;
                    free(A);
                    free(R);
                    free(v_A);
                    free(Q);
                    free(Z);
                }
            }
        }

        /**
         * Initialize the problem either reading from files (.csv or .nc) or generating the problem
         * @param is_nc
         */
        void init(bool is_nc);

        /**
         * Only invoke is function when it is not reading from files and after completed qr!
         */
        void init_y();

        void init_R_A_reduction();

        /**
         * Serial version of QR-factorization using modified Gram-Schmidt algorithm
         * @tparam scalar
         * @tparam index
         * @param A
         * @param Q
         * @param R
         * @param n_proc
         * @return
         */
        returnType<scalar, index>
        cils_qr_decomposition_serial(const index eval, const index qr_eval);


        /**
         * Serial version of QR-factorization using modified Gram-Schmidt algorithm
         * @tparam scalar
         * @tparam index
         * @param A
         * @param Q
         * @param R
         * @param n_proc
         * @return
         */
        returnType<scalar, index>
        cils_qr_decomposition_omp(const index eval, const index qr_eval, const index n_proc);

        /**
         *
         * @param eval
         * @param qr_eval
         * @return
         */
        returnType<scalar, index>
        cils_qr_decomposition_py(const index eval, const index qr_eval);

        long cils_qr_decomposition_py_helper();

        returnType<scalar, index>
        cils_qr_decomposition_reduction();

        scalar cils_qr_decomposition_reduction_helper();

        /**
         *
         * @param n_proc
         * @param nswp
         * @param z_B
         * @return
         */
        returnType<scalar, index>
        cils_babai_search_cuda(const index nswp, vector<scalar> *z_B);

        /**
         * Usage Caution: If LLL reduction is applied, please do permutation after getting the result.
         * @param n_proc
         * @param nswp
         * @param update
         * @param z_B
         * @param z_B_p
         * @return
         */
        returnType<scalar, index>
        cils_back_solve(vector<scalar> *z_B);

        /**
        * Serial Babai solver
        * @param z_B
        * @param is_constrained
        * @return
        */
        returnType<scalar, index>
        cils_babai_search_serial(vector<scalar> *z_B);

        /**
        * Serial Babai solver
        * @param z_B
        * @param is_constrained
        * @return
        */
        returnType<scalar, index>
        cils_block_search_serial_CPUTEST(const vector<index> *d, vector<scalar> *z_B);

        /**
         * Constrained version of Parallel Babai solver
         * @param n_proc: number of Processors/Threads
         * @param nswp: maximum number of iterations
         * @param z_B: estimation of the true parameter
         * @return
         */
        returnType<scalar, index>
        cils_babai_search_omp(const index n_proc, const index nswp, vector<scalar> *z_B);

        /**
         * Constrained version of Parallel Babai solver
         * @param n_proc: number of Processors/Threads
         * @param nswp: maximum number of iterations
         * @param z_B: estimation of the true parameter
         * @return
         */
        returnType<scalar, index>
        cils_back_solve_omp(const index n_proc, const index nswp, vector<scalar> *z_B);

        /**
         * Unconstrained serial version of Block Babai solver
         * @param z_B
         * @param d
         * @return
         */
        returnType<scalar, index>
        cils_block_search_serial(const index init, const vector<index> *d, vector<scalar> *z_B);


        /**
         * Unconstrained Parallel version of Block Babai solver
         * @param n_proc
         * @param nswp
         * @param stop
         * @param init
         * @param d
         * @param z_B
         * @return
         */
        returnType<scalar, index>
        cils_block_search_omp(const index n_proc, const index nswp, const index init, const vector<index> *d,
                              vector<scalar> *z_B);


        returnType<scalar, index>
        cils_block_search_omp_dynamic_block(const index n_proc, const index nswp, const index init,
                                            const vector<index> *d, vector<scalar> *z_B);

        /**
         * Unconstrained GPU version of Block Babai solver
         * @param n_proc
         * @param nswp
         * @param R_B
         * @param y_B
         * @param z_B
         * @param d
         * @return
         */
        returnType<scalar, index>
        cils_block_search_cuda(index nswp, scalar stop, const vector<index> *d, vector<scalar> *z_B);


    };
}
#endif