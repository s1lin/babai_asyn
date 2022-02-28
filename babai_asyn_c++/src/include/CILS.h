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
//#include <netcdf.h>
#include <bitset>
#include <cmath>
#include "MatlabDataArray.hpp"
#include "MatlabEngine.hpp"
#include <numeric>
//#include "mpi.h"

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
typedef std::vector<bool> sb_vector;
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
        sd_vector x;
        scalar run_time;
        scalar info; //true_res, error
    };


    template<typename scalar, typename index>
    class CILS {

    public:
        index qam, snr, upper, lower, search_iter, m, n;
        index block_size, spilt_size, offset, is_constrained, verbose = 0;
        scalar init_res, sigma, tolerance;

        si_vector d;

        std::vector<std::vector<scalar>> permutation;

        b_matrix A, B; //B is a temporary matrix.
        b_eye_matrix I;
        b_vector x_t, y, l, u;

        //x_t: true parameter, y: original y
        CILS() {}

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

            d.resize(n / block_size);
            if (spilt_size != 0) {
                d.resize(n / block_size + spilt_size - 1);
            }

            std::fill(d.begin(), d.end(), block_size);

            for (int i = 0; i < spilt_size; i++) {
                d[i] = block_size / spilt_size;
            }
//            d[0] -= offset;
//            d[1] += offset;

            cout << n << "," << block_size << endl;
            for (index i = d.size() - 2; i >= 0; i--) {
                d[i] += d[i + 1];
            }

            helper::display<scalar, index>(d, "d");
        }

        /**
         * Initialize block indicator matrix associated with underdetermined SCP block Babai(Optimal) algorithm
         * Usage:
         * 1. Call it after calling initialization method.
         * 2. This method generate the indicator matrix.
         */
//        void init_indicator() {
//            if (!is_init_success)
//                std::cout << "[INFO: ] You need to initialize the class by calling method init().";
//            // 'SCP_Block_Optimal_2:34' cur_end = n;
//            index cur_end = n;
//            // 'SCP_Block_Optimal_2:35' i = 1;
//            index b_i = 1;
//            // 'SCP_Block_Optimal_2:36' while cur_end > 0
//            while (cur_end > 0) {
//                // 'SCP_Block_Optimal_2:37' cur_1st = max(1, cur_end-m+1);
//                index cur_1st = std::fmax(1, (cur_end - m) + 1);
//                // 'SCP_Block_Optimal_2:38' indicator(1,i) = cur_1st;
//                indicator[2 * (b_i - 1)] = cur_1st;
//                // 'SCP_Block_Optimal_2:39' indicator(2,i) = cur_end;
//                indicator[2 * (b_i - 1) + 1] = cur_end;
//                // 'SCP_Block_Optimal_2:40' cur_end = cur_1st - 1;
//                cur_end = cur_1st - 1;
//                // 'SCP_Block_Optimal_2:41' i = i + 1;
//                b_i++;
//            }
//        }

        CILS(index m, index n, index qam, index snr, index search_iter) {
            this->m = m;
            this->n = n;
            this->search_iter = search_iter;

            // Start MATLAB engine synchronously
            this->init_res = INFINITY;
            this->qam = qam;
            this->snr = snr;
            this->sigma = (scalar) sqrt(((pow(4, qam) - 1) * n) / (6 * pow(10, ((scalar) snr / 10.0))));
            this->tolerance = sqrt(m) * this->sigma;
            this->upper = pow(2, qam) - 1;

            this->I.resize(n, n, true);
            this->A.resize(m, n, false);
            this->A.clear();

            this->x_t.resize(n, false);
            this->y.resize(m, false);
            this->l.resize(n, false);
            this->u.resize(n, false);

            this->x_t.clear();
            this->y.clear();
            this->l.clear();
            this->u.clear();

            std::fill(u.begin(), u.end(), upper);

            this->is_init_success = false;
            this->block_size = 32;
            this->spilt_size = 2;
            this->offset = 4;
            this->lower = 0;
        }
    };
}
#endif