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
#include "mpi.h"
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace std;


using namespace boost::numeric::ublas;
typedef boost::numeric::ublas::vector<double> b_vector;
typedef boost::numeric::ublas::matrix<double> b_matrix;
typedef boost::numeric::ublas::identity_matrix<double> b_eye_matrix;

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


    template<typename scalar, typename index, index m, index n>
    class CILS {

    private:
        index qam, snr, upper, lower, search_iter;
        scalar init_res, sigma, tolerance;
        array<scalar, m * (n + 1) / 2> R_A;

        b_matrix A, H, Z, P; //b_matrix
        //x_r: real solution, x_t: true parameter, y_a: original y, y_r: reduced, y_q: QR
        b_vector x_r, x_t, l, u;
        b_vector y_a, v_a, v_q;

        std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr;
    public:
        CILS(index qam, index snr) {

            // Start MATLAB engine synchronously
            this->matlabPtr = matlab::engine::startMATLAB();
            this->init_res = INFINITY;
            this->qam = qam;
            this->snr = snr;
            this->sigma = (scalar) sqrt(((pow(4, qam) - 1) * n) / (6 * pow(10, ((scalar) snr / 10.0))));
            this->tolerance = sqrt(m) * this->sigma;
            this->upper = pow(2, qam) - 1;

            b_eye_matrix I(n, n);
            this->Z.resize(n, n);
            this->P.resize(n, n);
            this->Z.assign(I);
            this->P.assign(I);

            this->R_A.fill(0);// = new scalar[n * (n + 1) / 2];

            this->A.resize(m, n, false);
            this->H.resize(m, n, false);
            this->A.clear();
            this->H.clear();

            this->x_r.resize(n, false);
            this->x_t.resize(n, false);
            this->y_a.resize(n, false);
            this->v_a.resize(n, false);
            this->v_q.resize(n, false);
            this->l.resize(n, false);
            this->u.resize(n, false);

            this->x_r.clear();
            this->x_t.clear();
            this->y_a.clear();
            this->v_a.clear();
            this->v_q.clear();
            this->l.clear();
            this->u.clear();

            std::fill(u.begin(), u.end(), upper);

        }

        ~CILS() {
            matlabPtr.get_deleter();
        }

        /**
         * Generating the problem from Matlab funtion
         */
        void init(index rank) {

            //Create MATLAB data array factory
            scalar *size = (double *) malloc(1 * sizeof(double)), *p;

            if (rank == 0) {

                matlab::data::ArrayFactory factory;

                // Call the MATLAB movsum function
                matlab::data::TypedArray<scalar> k_M = factory.createScalar<scalar>(this->qam);
                matlab::data::TypedArray<scalar> m_M = factory.createScalar<scalar>(m);
                matlab::data::TypedArray<scalar> n_M = factory.createScalar<scalar>(n);
                matlab::data::TypedArray<scalar> SNR = factory.createScalar<scalar>(snr);
                matlab::data::TypedArray<scalar> MIT = factory.createScalar<scalar>(search_iter);
                matlabPtr->setVariable(u"k", std::move(k_M));
                matlabPtr->setVariable(u"m", std::move(m_M));
                matlabPtr->setVariable(u"n", std::move(n_M));
                matlabPtr->setVariable(u"SNR", std::move(SNR));
                matlabPtr->setVariable(u"max_iter", std::move(MIT));

                // Call the MATLAB movsum function
                matlabPtr->eval(
                        u" [A, x_t, v, y, sigma, res, permutation, size_perm] = gen_problem(k, m, n, SNR, max_iter);");

                matlab::data::TypedArray<scalar> const A_A = matlabPtr->getVariable(u"A");
                matlab::data::TypedArray<scalar> const y_M = matlabPtr->getVariable(u"y");
                matlab::data::TypedArray<scalar> const x_M = matlabPtr->getVariable(u"x_t");
                matlab::data::TypedArray<scalar> const res = matlabPtr->getVariable(u"res");
                matlab::data::TypedArray<scalar> const per = matlabPtr->getVariable(u"permutation");
                matlab::data::TypedArray<scalar> const szp = matlabPtr->getVariable(u"size_perm");


                index i = 0;
                for (auto r : A_A) {
                    A[i] = r;
                    ++i;
                }
                i = 0;
                for (auto r : y_M) {
                    y_a[i] = r;
                    ++i;
                }
                i = 0;
                for (auto r : x_M) {
                    x_t[i] = r;
                    ++i;
                }
                i = 0;
                for (auto r : res) {
                    this->init_res = r;
                    ++i;
                }
                i = 0;
                for (auto r : res) {
                    this->init_res = r;
                    ++i;
                }

                i = 0;
                for (auto r : szp) {
                    size[0] = r;
                    ++i;
                }
                p = (scalar *) malloc(n * size[0] * sizeof(scalar));
                i = 0;
                for (auto r : per) {
                    p[i] = r;
                    ++i;
                }
            }

            MPI_Barrier(MPI_COMM_WORLD);
            MPI_Bcast(&size[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            if (rank != 0)
                p = (scalar *) malloc(n * size[0] * sizeof(scalar));

            MPI_Bcast(&p[0], (int) size[0] * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);

            index i = 0;
            index k1 = 0;
            permutation.resize((int) size[0] + 1);
            permutation[k1] = vector<scalar>(n);
            permutation[k1].assign(n, 0);
            for (index iter = 0; iter < (int) size[0] * n; iter++) {
                permutation[k1][i] = p[iter];
                i = i + 1;
                if (i == n) {
                    i = 0;
                    k1++;
                    permutation[k1] = vector<scalar>(n);
                    permutation[k1].assign(n, 0);
                }
            }
            i = 0;
        }


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
        cils_grad_proj_omp(vector<scalar> &x, const index search_iter, const index n_proc);

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
        cils_scp_block_optimal_serial(vector<scalar> &x_cur, scalar v_norm_cur, index mode);

        returnType<scalar, index>
        cils_scp_block_suboptimal_serial(vector<scalar> &x_cur, scalar v_norm_cur, index mode);

        returnType<scalar, index>
        cils_scp_block_suboptimal_omp(vector<scalar> &x_cur, scalar v_norm_cur, index n_proc, index mode);

        returnType<scalar, index>
        cils_scp_block_babai_serial(vector<scalar> &x_cur, scalar v_norm_cur, index mode);

        returnType<scalar, index>
        cils_scp_block_babai_omp(vector<scalar> &x_cur, scalar v_norm_cur, index n_proc, index mode);


        /**
         *  Corresponds to Algorithm 5 (Partition Strategy) in Report 10
         *  [H_A, P, z, Q_tilde, R_tilde, indicator] =
         *  partition_H(A, z_B, m, n) permutes and partitions H_A so
         *  that the submatrices H_i are full-column rank
         *
         *  Inputs:
         *      A - m-by-n real matrix
         *      z_B - n-dimensional integer vector
         *      m - integer scalar
         *      n - integer scalar
         *
         *  Outputs:
         *      P - n-by-n real matrix, permutation such that
         *      H_A*P=A z - n-dimensional integer vector
         *      (z_B permuted to correspond to H_A) Q_tilde - m-by-n real matrix
         *      (Q factors) R_tilde - m-by-n real matrix (R factors) indicator -
         *      2-by-q integer matrix (indicates submatrices of H_A)
         * @param z_B
         * @param Q_tilde
         * @param R_tilde
         * @param H_A
         * @param Piv_cum
         * @return
         */
        returnType<scalar, index>
        cils_partition_deficient(scalar *z_B, scalar *Q_tilde, scalar *R_tilde, scalar *H_A, scalar *Piv_cum);

        returnType<scalar, index>
        cils_block_search_serial(const index init, const scalar *R_R, const scalar *y_r, const vector<index> *d,
                                 vector<scalar> *z_B);


        returnType<scalar, index>
        cils_scp_block_optimal_omp(vector<scalar> &x_cur, scalar v_norm_cur, index n_proc, index mode);

        returnType<scalar, index>
        cils_scp_block_optimal_mpi(vector<scalar> &x_cur, scalar *v_norm_cur, index size, index rank);
    };
}
#endif