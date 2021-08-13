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
#include <cmath>
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
    struct returnType {
        vector<scalar> x;
        scalar run_time;
        scalar info; //true_res, error
    };



    template<typename scalar, typename index, index m, index n>
    class cils {

    public:
        index qam, snr, upper, lower;
        scalar init_res, sigma, tolerance;
        array<scalar, m * (n + 1) / 2> R_A;

        array<scalar, m * n> A, H;
        array<scalar, n * n> Z, P;
        //x_r: real solution, x_t: true parameter, y_a: original y, y_r: reduced, y_q: QR
        array<scalar, n> x_r, x_t, l, u;
        array<scalar, m> y_a, v_a, v_q;

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

            this->A.fill(0);//.resize(n * n, 0);
            this->Z.fill(0);//.resize(n * n, 0);
            this->P.fill(0);//.resize(n * n, 0);
            this->H.fill(0);//.resize(n * n, 0);

            this->x_r.fill(0);//.resize(n, 0);
            this->x_t.fill(0);//.resize(n, 0);
            this->y_a.fill(0);//.resize(n, 0);
            this->v_a.fill(0);//.resize(n, 0);
            this->v_q.fill(0);//.resize(n, 0);

            this->l.fill(0);//.resize(n, 0);
            this->u.fill(0);//.resize(n, 0);
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
         * Parallel version of QR-factorization using modified Gram-Schmidt algorithm, row-oriented
         * @param eval
         * @param verbose
         * @param n_proc
         * @return
         */
//        returnType<scalar, index>
//        cils_qr_omp(const index eval, const index verbose, const index n_proc);

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
        cils_block_search_serial_CPUTEST(const scalar *R_R, const scalar y_r,
                                         const vector<index> *d, vector<scalar> *z_B);

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
        cils_block_search_omp(const index n_proc, const index nswp, const index init, const scalar *y_r,
                              const vector<index> *d, vector<scalar> *z_B);


        returnType<scalar, index>
        cils_sic_serial(vector<scalar> &x);

        returnType<scalar, index>
        cils_qrp_serial(vector<scalar> &x);

        returnType<scalar, index>
        cils_grad_proj(vector<scalar> &x, const index max_iter);

        returnType<scalar, index>
        cils_sic_subopt(vector<scalar> &z, array<scalar, m> &v_cur, array<scalar, m * n> A_t, scalar v_norm_cur,
                        scalar tolerance, index method);

        /**
         * Applies the SCP-Block Optimal method to obtain a sub-optimal solution
         * @tparam scalar
         * @tparam index
         * @tparam m - integer scalar
         * @tparam n - integer scalar
         * @param x_cur - n-dimensional integer vector for the sub-optimal solution
         * @param v_norm_cur - real scalar for the norm of the residual vector
         * @param max_Babai - integer scalar, maximum number of calls to block_opt
         * @param stopping - 1-by-3 boolean vector, indicates stopping criterion used
         * @return {}
         */
        returnType<scalar, index>
        cils_scp_block_optimal_serial(vector<scalar> &x_cur, scalar v_norm_cur);

        returnType <scalar, index>
        cils_block_search_serial(const index init, const scalar *R_R, const scalar *y_r, const vector<index> *d,
                                 vector<scalar> *z_B);

        returnType <scalar, index>
        cils_scp_block_optimal_omp(vector<scalar> &x_cur, scalar v_norm_cur);

        returnType <scalar, index>
        cils_scp_block_optimal_mpi(vector<scalar> &x_cur, scalar *v_norm_cur, index size, index rank);
    };
}
#endif