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
#include "MatlabDataArray.hpp"
#include "MatlabEngine.hpp"
#include <numeric>
#include "mpi.h"

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/math/tools/norms.hpp>
#include <boost/program_options.hpp>

using namespace std;

using namespace boost::numeric::ublas;
using namespace boost::program_options;

typedef std::vector<double> sd_vector;
typedef std::vector<int> si_vector;
typedef boost::numeric::ublas::vector<double> b_vector;
typedef boost::numeric::ublas::matrix<double> b_matrix;
typedef boost::numeric::ublas::identity_matrix<double> b_eye_matrix;

#include "helper.h"

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
        std::vector<scalar> x;
        scalar run_time;
        scalar info; //true_res, error
    };


    template<typename scalar, typename index>
    class CILS {

    public:
        index qam, snr, upper, lower, search_iter, q, m, n;
        index block_size, spilt_size, offset, is_constrained, verbose = 0;
        scalar init_res, sigma, tolerance;

        si_vector d, indicator;

        std::vector<std::vector<scalar>> permutation;

        b_vector R_A;

        b_matrix A, H;
        b_eye_matrix I; //b_matrix
        //x_r: real solution, x_t: true parameter, y_a: original y, y_r: reduced, y_q: QR
        b_vector x_r, x_t, l, u;
        b_vector y_a, v_a, v_q;


        /**
        *   omp_sched_static = 0x1,
        *   omp_sched_dynamic = 0x2,
        *   omp_sched_guided = 0x3,
        *   omp_sched_auto = 0x4,
        */
        index schedule, chunk_size;
        bool is_init_success{};

        /**
         * Initialize block vector associated with block Babai(Optimal) algorithm
         * Usage:
         * 1. Call it after calling initialization method.
         * 2. This method only generate a new vector d, but do not update block_size, split_size and offset.
         * @param block_size : Size of the each block
         * @param spilt_size : split size of the R(n-block_size:n,n-block_size:n) (last) block. Default = 0
         * @param offset : offset for the last block. Default = 0.
         */
        void init_d(index new_block_size, index new_spilt_size, index new_offset) {
            if (!is_init_success)
                std::cout << "[INFO: ] You need to initialize the class by calling method init().";
            d.resize(n / new_block_size + new_spilt_size - 1);
            for (int i = 0; i < new_spilt_size; i++) {
                d[i] = new_block_size / new_spilt_size;
            }
            d[0] -= new_offset;
            d[1] += new_offset;

            for (index i = d.size() - 2; i >= 0; i--) {
                d[i] += d[i + 1];
            }
        }

        /**
         * Default initialization method for block vector associated with block Babai(Optimal) algorithm
         * Usage:
         * 1. Call it after calling initialization method.
         * 2. This method generate the vector d, by using class fields block_size, split_size and offset.
         */
        void init_d() {
            if (!is_init_success)
                std::cout << "[INFO: ] You need to initialize the class by calling method init().";
            d.resize(n / block_size);// + spilt_size - 1);
            std::fill(d.begin(), d.end(), block_size);

            for (int i = 0; i < spilt_size; i++) {
                d[i] = block_size / spilt_size;
            }
            d[0] -= offset;
            d[1] += offset;

            cout << n << "," << block_size << endl;
            for (index i = d.size() - 2; i >= 0; i--) {
                d[i] += d[i + 1];
            }
        }

        /**
         * Initialize block indicator matrix associated with underdetermined SCP block Babai(Optimal) algorithm
         * Usage:
         * 1. Call it after calling initialization method.
         * 2. This method generate the indicator matrix.
         */
        void init_indicator() {
            if (!is_init_success)
                std::cout << "[INFO: ] You need to initialize the class by calling method init().";
            // 'SCP_Block_Optimal_2:34' cur_end = n;
            index cur_end = n;
            // 'SCP_Block_Optimal_2:35' i = 1;
            index b_i = 1;
            // 'SCP_Block_Optimal_2:36' while cur_end > 0
            while (cur_end > 0) {
                // 'SCP_Block_Optimal_2:37' cur_1st = max(1, cur_end-m+1);
                index cur_1st = std::fmax(1, (cur_end - m) + 1);
                // 'SCP_Block_Optimal_2:38' indicator(1,i) = cur_1st;
                indicator[2 * (b_i - 1)] = cur_1st;
                // 'SCP_Block_Optimal_2:39' indicator(2,i) = cur_end;
                indicator[2 * (b_i - 1) + 1] = cur_end;
                // 'SCP_Block_Optimal_2:40' cur_end = cur_1st - 1;
                cur_end = cur_1st - 1;
                // 'SCP_Block_Optimal_2:41' i = i + 1;
                b_i++;
            }
        }

        CILS() {

        }

        CILS(index m, index n, index qam, index snr, index search_iter) {
            this->m = m;
            this->n = n;
            this->search_iter = search_iter;
            this->q = static_cast<index>(std::ceil((scalar) n / (scalar) m));
            // Start MATLAB engine synchronously
            this->init_res = INFINITY;
            this->qam = qam;
            this->snr = snr;
            this->sigma = (scalar) sqrt(((pow(4, qam) - 1) * n) / (6 * pow(10, ((scalar) snr / 10.0))));
            this->tolerance = sqrt(m) * this->sigma;
            this->upper = pow(2, qam) - 1;

            this->I.resize(n, n, true);
            this->A.resize(m, n, false);
            this->H.resize(m, n, false);
            this->A.clear();
            this->H.clear();

            this->R_A.resize(m * (n + 1) / 2, false);// = new scalar[n * (n + 1) / 2];
            this->x_r.resize(n, false);
            this->x_t.resize(n, false);
            this->y_a.resize(m, false);
            this->v_a.resize(m, false);
            this->v_q.resize(m, false);
            this->l.resize(n, false);
            this->u.resize(n, false);

            this->R_A.clear();
            this->x_r.clear();
            this->x_t.clear();
            this->y_a.clear();
            this->v_a.clear();
            this->v_q.clear();
            this->l.clear();
            this->u.clear();

            std::fill(u.begin(), u.end(), upper);

            this->is_init_success = false;

            this->block_size = 2;
            this->spilt_size = 0;
            this->offset = 0;
            this->lower = 0;

            this->indicator.resize(this->q);
        }

        /**
         * Generating the problem from Matlab funtion
         */



        /**
         * Usage Caution: If LLL reduction is applied, please do permutation after getting the result.
         * @param n_proc
         * @param nswp
         * @param update
         * @param z_B
         * @return
         */



        /**
        * Serial one dimensional Babai estimation for ordinary (box-constrained) ILS problem.
        * @param z_B: 1-by-n estimation vector.
        * @param is_constrained: indicator for either constrained problem or unconstrained problem.
        * @return
        */



        /**
         * Parallel Serial one dimensional Babai estimation for ordinary (box-constrained) ILS problem.
         * @param n_proc: number of Threads.
         * @param nswp: maximum number of chaotic iterations.
         * @param z_B: 1-by-n estimation vector.
         * @return
         */


        /**
        * Serial Babai solver
        * @param z_B
        * @param is_constrained
        * @return
        */




        /**
         * Parallel Babai solver
         * @param n_proc: number of Processors/Threads
         * @param nswp: maximum number of iterations
         * @param z_B: estimation of the true parameter
         * @return
         */



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



        /**
         *  Corresponds to Algorithm 5 (Partition Strategy) in Report 10
         *  [H_A, P, z, Q_tilde, R_tilde, indicator] =
         *  partition_H(A, z_hat, m, n) permutes and partitions H_A so
         *  that the submatrices H_i are full-column rank
         *
         *  Inputs:
         *      A - m-by-n real matrix
         *      z_hat - n-dimensional integer vector
         *      m - integer scalar
         *      n - integer scalar
         *
         *  Outputs:
         *      P - n-by-n real matrix, permutation such that
         *      H_A*P=A z - n-dimensional integer vector
         *      (z_hat permuted to correspond to H_A) Q_tilde - m-by-n real matrix
         *      (Q factors) R_tilde - m-by-n real matrix (R factors) indicator -
         *      2-by-q integer matrix (indicates submatrices of H_A)
         * @param z_B
         * @param Q_tilde
         * @param R_tilde
         * @param H_A
         * @param Piv_cum
         * @return
         */

    };
}
#endif