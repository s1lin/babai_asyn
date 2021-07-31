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
#include "helper.h"

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
    template<typename scalar, typename index, index m, index n>
    inline scalar find_residual(const array<scalar, m * n> &A,
                                const array<scalar, n> &y,
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

//    template<typename scalar, typename index, index m, index n>
//    inline vector<scalar> find_residual_by_block(const vector<scalar>A,
//                                                 const vector<scalar> y,
//                                                 const vector<index> *d, vector<scalar> x) {
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

//    template<typename scalar, typename index, index m, index n>
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




//    template<typename scalar, typename index, index m, index n>
//    inline void vector_reverse_permutation(const vector<scalar>Z,
//                                           vector<scalar> x) {
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
    template<typename scalar, typename index, index m, index n>
    inline scalar find_bit_error_rate(const vector<scalar> *x_b,
                                      const array<scalar, n> &x_t,
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
    template<typename scalar, typename index, index m, index n>
    void display_matrix(const array<scalar, m * n> &x) {
        for (index i = 0; i < m; i++) {
            for (index j = 0; j < n; j++) {
                printf("%8.4f ", x[j * m + i]);
            }
            cout << "\n";
        }
    }

    template<typename scalar, typename index, index m, index n>
    void display_2D_T(const array<scalar, m * n> &x) {
        for (index i = 0; i < m * n; i++) {
//            for (index j = 0; j < n; j++) {
            printf("%8.4f ", x[i]);
//            }
//            cout << "\n";
        }
    }

    /**
     * Simple function for displaying the struct scalarType
     * @tparam scalar
     * @tparam index
     * @param x
     */
    template<typename scalar, typename index, index n>
    inline void display_array(const array<scalar, n> &x) {
        scalar sum = 0;
        for (index i = 0; i < n; i++) {
            printf("%8.5f ", x[i]);
            sum += x[i];
        }
        printf("SUM = %8.5f\n", sum);
    }

    template<typename scalar, typename index>
    inline void display_vector(const vector<scalar> &x) {
        scalar sum = 0;
        for (index i = 0; i < x.size(); i++) {
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
    inline scalar find_success_prob_babai(const vector<scalar> &R_B,
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
//    static void value_input_helper(matlab::data::TypedArray<scalar> const x, vector<scalar> &arr);
//
//    template<typename scalar>
//    static void value_input_helper(matlab::data::TypedArray<scalar> const x, vector<scalar>&arr);
//
//    template<typename scalar>
//    static void value_input_helper(matlab::data::TypedArray<scalar> const x, vector<scalar> *arr);

    /**
     * 
     * @tparam scalar 
     * @tparam index 
     * @tparam m
     * @tparam n
     */
    template<typename scalar, typename index, index m, index n>
    class cils {

    public:
        index qam, snr, upper, lower;
        scalar init_res, sigma;
        array<scalar, m * (n + 1) / 2> R_A;
        //R_A: no zeros, R_R: LLL reduced, R_Q: QR
        array<scalar, m * n> R_R, R_Q, A, H;
        array<scalar, m * m> Q;
        array<scalar, n * n> Z;
        //x_r: real solution, x_t: true parameter, y_a: original y, y_r: reduced, y_q: QR
        array<scalar, n> x_r, x_t;
        array<scalar, m> y_a, v_a, v_q, y_r, y_q;

        std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr;
    public:
        cils(index qam, index snr) {

            // Start MATLAB engine synchronously
            this->matlabPtr = matlab::engine::startMATLAB();
            this->init_res = INFINITY;
            this->qam = qam;
            this->snr = snr;
            this->sigma = (scalar) sqrt(((pow(4, qam) - 1) * n) / (6 * pow(10, ((scalar) snr / 10.0))));
            this->upper = pow(2, qam) - 1;

            this->R_A.fill(0);// = new scalar[n * (n + 1) / 2];

            this->R_R.fill(0);//.resize(n * n, 0);
            this->R_Q.fill(0);//.resize(n * n, 0);
            this->A.fill(0);//.resize(n * n, 0);
            this->Q.fill(0);//.resize(n * n, 0);
            this->Z.fill(0);//.resize(n * n, 0);
            this->H.fill(0);//.resize(n * n, 0);

            this->x_r.fill(0);//.resize(n, 0);
            this->x_t.fill(0);//.resize(n, 0);
            this->y_a.fill(0);//.resize(n, 0);
            this->y_r.fill(0);//.resize(n, 0);
            this->y_q.fill(0);//.resize(n, 0);
            this->v_a.fill(0);//.resize(n, 0);
            this->v_q.fill(0);//.resize(n, 0);
        }

        ~cils() {
//            delete[] R_A;
            matlabPtr.get_deleter();
//            matlab::engine::terminateEngineClient();
        }

        /**
         * Initialize the problem either reading from files (.csv or .nc) or generating the problem
         */
        void init();

        /**
         *
         */
        void init_ud();

        /**
         * Only invoke is function when it is not reading from files and after completed qr!
         */
        void init_y();

        /**
         *
         */
        void init_R();


        /**
         * Serial version of QR-factorization using modified Gram-Schmidt algorithm, row-oriented
         * @param eval
         * @param verbose
         * @return
         */
        returnType<scalar, index>
        cils_qr_serial(const index eval, const index verbose);


        /**
         * Parallel version of QR-factorization using modified Gram-Schmidt algorithm, row-oriented
         * @param eval
         * @param verbose
         * @param n_proc
         * @return
         */
        returnType<scalar, index>
        cils_qr_omp(const index eval, const index verbose, const index n_proc);

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

        returnType<scalar, index> cils_LLL_qr_reduction(const index eval, const index verbose, const index n_proc);

        returnType<scalar, index> cils_LLL_serial();

        scalar cils_LLL_omp(const index n_proc);

        returnType<scalar, index> cils_LLL_qr_serial();

        scalar cils_LLL_qr_omp(const index n_proc);

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
        cils_back_solve(array<scalar, n> &z_B);

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
         * Parallel Babai solver
         * @param n_proc: number of Processors/Threads
         * @param nswp: maximum number of iterations
         * @param z_B: estimation of the true parameter
         * @return
         */
        returnType<scalar, index>
        cils_babai_search_omp(const index n_proc, const index nswp, vector<scalar> *z_B);

        /**
         * Parallel Babai solver
         * @param n_proc: number of Processors/Threads
         * @param nswp: maximum number of iterations
         * @param z_B: estimation of the true parameter
         * @return
         */
        returnType<scalar, index>
        cils_back_solve_omp(const index n_proc, const index nswp, vector<scalar> *z_B);

        /**
         * Serial version of Block Babai solver
         * @param z_B
         * @param d
         * @return
         */
        returnType<scalar, index>
        cils_block_search_serial(const index init, const vector<index> *d, vector<scalar> *z_B);


        /**
         * Parallel version of Block Babai solver
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
        cils_sic_serial(vector<scalar> &x, array<scalar, m * n> &A_t, array<scalar, n * n> &P);

        returnType<scalar, index>
        cils_qrp_serial(vector<scalar> &x, array<scalar, m * n> &A_t, array<scalar, n * n> &P);

        returnType<scalar, index>
        cils_sic_subopt(vector<scalar> &z, array<scalar, m> &v_cur, array<scalar, m * n> A_t, scalar v_norm_cur,
                        scalar tolerance, index method);
    };
}
#endif