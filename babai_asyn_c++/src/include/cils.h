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
#include "config.h"
#include "MatlabDataArray.hpp"
#include "MatlabEngine.hpp"
#include <numeric>
#include "coder_array.h"
#include "coder_utils.h"

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
    inline scalar find_residual(const coder::array<scalar, 2U> &A,
                                const coder::array<scalar, 1U> &y,
                                scalar *x) {
        scalar res = 0, sum = 0;
        for (index i = 0; i < n; i++) {
            sum = 0;
            for (index j = 0; j < n; j++) {
                sum += x[j] * A[j * n + i];
            }
            res += (y[i] - sum) * (y[i] - sum);
        }
        return std::sqrt(res);
    }

//    template<typename scalar, typename index, index n>
//    inline vector<scalar> find_residual_by_block(const coder::array<scalar, 2U> A,
//                                                 const coder::array<scalar, 1U> y,
//                                                 const vector<index> *d, coder::array<scalar, 1U> x) {
//        index ds = d->size();
//        vector<scalar> y_b(n, 0), res(ds, 0);
//        for (index i = 0; i < ds; i++) {
//            index n_dx_q_1 = d[i];
//            index n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);
//
//            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                scalar sum = 0;
//                for (index col = n_dx_q_1; col < n; col++) {
//                    sum += A[col + row * n] * x[row];
//                }
//                y_b[row] = y[row] - sum;
//            }
//
//            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                scalar sum = 0;
//                for (index col = n_dx_q_0; col < n_dx_q_1; col++) {
//                    sum += A[col + row * n] * x[row];
//                }
//                res[i] += (y_b[row] - sum) * (y_b[row] - sum);
//            }
//
//            res[i] = std::sqrt(res[i]);
//        }
//        return res;
//    }

//    template<typename scalar, typename index, index n>
//    inline vector<scalar> find_bit_error_rate_by_block(const vector<scalar> *x_b, const vector<scalar> *x_t,
//                                                       const vector<index> *d, const index k) {
//        index ds = d->size();
//        vector<scalar> ber(ds, 0);
//        for (index i = 0; i < ds; i++) {
//            index error = 0;
//            index n_dx_q_1 = d[i];
//            index n_dx_q_0 = i == ds - 1 ? 0 : d->at(i + 1);
//            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                std::string binary_x_b, binary_x_t;
//                switch (k) {
//                    case 1:
//                        binary_x_b = std::bitset<1>((index) x_b->at(row)).to_string(); //to binary
//                        binary_x_t = std::bitset<1>((index) x_t->at(row)).to_string();
//                        break;
//                    case 2:
//                        binary_x_b = std::bitset<2>((index) x_b->at(row)).to_string(); //to binary
//                        binary_x_t = std::bitset<2>((index) x_t->at(row)).to_string();
//                        break;
//                    default:
//                        binary_x_b = std::bitset<3>((index) x_b->at(row)).to_string(); //to binary
//                        binary_x_t = std::bitset<3>((index) x_t->at(row)).to_string();
//                        break;
//                }
////                cout << binary_x_b << "-" << binary_x_t << " ";
//                for (index j = 0; j < k; j++) {
//                    if (binary_x_b[j] != binary_x_t[j])
//                        error++;
//                }
//            }
////            cout << error << ", \n";
//            ber[i] = (scalar) error / ((n_dx_q_1 - n_dx_q_0) * k);
//        }
//
//        return ber;
//    }




//    template<typename scalar, typename index, index n>
//    inline void vector_reverse_permutation(const coder::array<scalar, 2U> Z,
//                                           coder::array<scalar, 1U> x) {
//        vector<scalar> x_P(n, 0);
//        for (index i = 0; i < n; i++) {
//            scalar sum = 0;
//            for (index j = 0; j < n; j++) {
////                if (Z[i * n + j] != 0) {
//                sum += Z[i * n + j] * x[j];
////                    break;
////                }
//            }
//            x_P[i] = sum;
//        }
//
//        for (index i = 0; i < n; i++) {
//            x[i] = x_P[i];
//        }
//    }


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
                                      const coder::array<scalar, 1U> &x_t,
                                      const index k) {
        index error = 0;
        for (index i = 0; i < n; i++) {
            std::string binary_x_b, binary_x_t;
            switch (k) {
                case 1:
                    binary_x_b = std::bitset<1>((index) x_b->at(i)).to_string(); //to binary
                    binary_x_t = std::bitset<1>((index) x_t[i]).to_string();
                    break;
                case 2:
                    binary_x_b = std::bitset<2>((index) x_b->at(i)).to_string(); //to binary
                    binary_x_t = std::bitset<2>((index) x_t[i]).to_string();
                    break;
                default:
                    binary_x_b = std::bitset<3>((index) x_b->at(i)).to_string(); //to binary
                    binary_x_t = std::bitset<3>((index) x_t[i]).to_string();
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
     * Simple function for displaying the struct scalarType
     * @tparam scalar
     * @tparam index
     * @param x
     */
    template<typename scalar, typename index>
    void display_2D(const coder::array<scalar, 2U> &x) {
        index n = x.size(0);
        for (index i = 0; i < n; i++) {
            for (index j = 0; j < n; j++) {
                cout << x[j * n + i] << " ";
            }
            cout << "\n";
        }
    }

    /**
     * Simple function for displaying the struct scalarType
     * @tparam scalar
     * @tparam index
     * @param x
     */
    template<typename scalar, typename index>
    inline void display_vector(const coder::array<scalar, 1U> &x) {
        scalar sum = 0;
        for (index i = 0; i < x.size(0); i++) {
            printf("%8.5f ", x[i]);
            sum += x[i];
        }
        printf("SUM = %8.5f\n", sum);
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
            printf("%8.5f ", x->at(i));
            sum += x->at(i);
        }
        printf("SUM = %8.5f\n", sum);
    }


    /**
     *
     * @tparam scalar
     * @tparam index
     * @return
     */
    template<typename scalar, typename index>
    inline scalar find_success_prob_babai(const coder::array<scalar, 2U> &R_B,
                                          const index row_begin, const index row_end,
                                          const index block_size, const scalar sigma) {
        scalar p = 1;
        for (index row = row_begin; row < row_end; row++) {
            index i = (block_size * row) + row - ((row * (row + 1)) / 2);
            p = p * erf(abs(R_B[i]) / 2 / sigma / sqrt(2));
        }
        return p;
    }

//    template<typename scalar>
//    static void value_input_helper(matlab::data::TypedArray<scalar> const x, coder::array<scalar, 1U> &arr);
//
//    template<typename scalar>
//    static void value_input_helper(matlab::data::TypedArray<scalar> const x, coder::array<scalar, 2U> &arr);
//
//    template<typename scalar>
//    static void value_input_helper(matlab::data::TypedArray<scalar> const x, vector<scalar> *arr);

    /**
     * 
     * @tparam scalar 
     * @tparam index 
     * @tparam n 
     */
    template<typename scalar, typename index, index n>
    class cils {

    public:
        index qam, snr, upper, lower;
        scalar init_res, sigma, *R_A;
        //R_A: no zeros, R_R: LLL reduced, R_Q: QR
        coder::array<double, 2U> R_R, R_Q, A, Q, Z;
        //x_r: real solution, x_t: true parameter, y_a: original y, y_r: reduced, y_q: QR 
        coder::array<double, 1U> x_r, x_t, y_a, y_r, y_q, v_a, v_q;

    private:

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


    public:
        cils(index qam, index snr) {
            //R_A: no zeros, R_R: LLL reduced, R_Q: QR
//            coder::array<scalar, 2U> R_A, R_R, R_Q, A, Q, Z;
            //x_r: real solution, x_t: true parameter, y_a: original y, y_r: reduced, y_q: QR
//            coder::array<scalar, 1U> x_r, x_t, y_a, y_r, y_q, v_a;

            this->init_res = INFINITY;
            this->qam = qam;
            this->snr = snr;
            this->sigma = (scalar) sqrt(((pow(4, qam) - 1) * n) / (6 * pow(10, ((scalar) snr / 10.0))));
            this->upper = pow(2, qam) - 1;

            this->R_A = new scalar[n * (n + 1) / 2];

            this->R_R.set_size(n, n);
            this->R_Q.set_size(n, n);
            this->A.set_size(n, n);
            this->Q.set_size(n, n);
            this->Z.set_size(n, n);

            this->x_r.set_size(n);
            this->x_t.set_size(n);
            this->y_a.set_size(n);
            this->y_r.set_size(n);
            this->y_q.set_size(n);
            this->v_a.set_size(n);
            this->v_q.set_size(n);
        }

        ~cils() {
            delete[] R_A;
        }

        /**
         * Initialize the problem either reading from files (.csv or .nc) or generating the problem
         */
        void init();

        /**
         * Only invoke is function when it is not reading from files and after completed qr!
         */
        void init_y();

        /**
         *
         */
        void init_R();

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
        cils_qr_serial(const index eval, const index verbose);


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
        cils_qr_omp(const index eval, const index qr_eval, const index n_proc);

        /**
         *
         * @param eval
         * @param qr_eval
         * @return
         */
        returnType<scalar, index>
        cils_qr_py(const index eval, const index qr_eval);

        long cils_qr_py_helper();

        returnType<scalar, index>
        cils_qr_matlab();

        scalar cils_qr_matlab_helper();

        /**
         *
         * @param n_proc
         * @param eval
         * @return
         */
        returnType<scalar, index> cils_LLL_reduction(const index eval, const index verbose, const index n_proc);

        scalar cils_LLL_serial();

        scalar cils_LLL_omp(const index n_proc);

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
        cils_back_solve(coder::array<scalar, 1U> &z_B);

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