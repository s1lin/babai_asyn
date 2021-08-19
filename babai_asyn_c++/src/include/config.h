#ifndef CILS_CONFIG_H
#define CILS_CONFIG_H

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <climits>
#include "helper.h"

const static int M = 12;
const static int N = 18;

using namespace std;

namespace cils::program_def {
    typedef int index;
    typedef double scalar;
    /**
     *   omp_sched_static = 0x1,
     *   omp_sched_dynamic = 0x2,
     *   omp_sched_guided = 0x3,
     *   omp_sched_auto = 0x4,
     */
    index qam = 3;
    index SNR = 35;
    index max_iter = 500;
    index search_iter = (int) 1e5;
    index stop = 3;
    index schedule = 2;
    index chunk_size = 1;
    index block_size = 32;
    index spilt_size = 2;
    index offset = 2;
    index is_constrained = true;
    index is_nc = false;
    index is_matlab = false; //Means LLL reduction
    index is_qr = false;
    index mode = 1; //test mode 3: c++ gen
    index num_trials = 10; //nswp
    index is_local = 1;
    index max_search = 400000;//INT_MAX;
    index min_proc = 2;
    index plot_itr = 1;
    scalar coeff = 17.5;
    index max_proc = 10;
    index max_thre = 400000;//maximum search allowed for serial ils.
    auto q = static_cast<index>(std::ceil((scalar) N / (scalar) M));
    index verbose = false;
    index chunk = 1;

    /*   Parameters for block size 64
     *   index k = 3;
         index SNR = 35;
         index max_iter = 1000;
         index search_iter = 100;
         index stop = 2;
         index schedule = 2;
         index chunk_size = 1;
         index block_size = 64;
         index spilt_size = 2;
         index is_constrained = true;
         index is_read = false;
         index is_matlab = true; //Means LLL reduction
         index is_qr = false;
         index mode = 1; //test mode 3: c++ gen
         index num_trials = 10; //nswp
         index is_local = 1;
         index max_search = 1000000;//INT_MAX;
         index min_proc = 2;
         index plot_itr = 2;

         index max_proc = min(omp_get_max_threads(), N / block_size);
         index max_thre = 1000000;//maximum search allowed for serial ils.
     */

    std::vector<index> d_s(N / block_size + spilt_size - 1, block_size);
    std::vector<index> indicator(2 * q, 0);
//        std::vector<index> d_s(N / block_size, block_size);
    vector<vector<scalar>> permutation(search_iter + 3);

    void init_program_def(int argc, char *argv[]) {
        if (argc != 1) {
            is_local = stoi(argv[1]);
        }
//            max_proc = is_local ? 17 : 30;
        printf("[ INFO: The program settings are:]\n"
               "1. QAM: %d, SNR: %d, epoch: %d, block size: %d;\n"
               "2. ILS-Max number of interger search points: %d;\n"
               "3. Chaotic stop minimum: %d, chatoic number of iteratios(nswp): %d;\n"
               "4. ILS-Max of integer search iterations:%d;\n"
               "5. qr: %d, constrained: %d, matlab for qr/LLL: %d.\n",
               (int) pow(4, qam), SNR, max_iter, block_size,
               search_iter, stop, num_trials, max_search,
               is_local, is_constrained, is_matlab);
//        permutation[0] = vector<scalar>(N);
//        for (index k1 = 0; k1 < N; k1++) {
//            permutation[0][k1] = k1 + 1;
//        }
//        vector<scalar> p_6 = {1, 3, 4, 5, 6, 2};
//        permutation[1] = p_6;
//        for (index k1 = 0; k1 < N; k1++) {
//            cout<<permutation[1][k1]<<" ";
//        }
//        for (index k1 = 0; k1 < N; k1++) {
//            permutation[0][k1] = k1 + 1;
//        }
//
//        for (index k1 = 0; k1 <= search_iter; k1++) {
//            permutation[k1] = vector<scalar>(N);
//            permutation[k1].assign(N, 0);
////            helper::randperm(N, permutation[k1].data());
//        }
        for (int i = 0; i < spilt_size; i++) {
            d_s[i] = block_size / spilt_size;
        }
        d_s[0] -= offset;
        d_s[1] += offset;

        for (index i = d_s.size() - 2; i >= 0; i--) {
            d_s[i] += d_s[i + 1];
        }

        // 'SCP_Block_Optimal_2:34' cur_end = n;
        index cur_end = N;
        // 'SCP_Block_Optimal_2:35' i = 1;
        index b_i = 1;
        // 'SCP_Block_Optimal_2:36' while cur_end > 0
        while (cur_end > 0) {
            // 'SCP_Block_Optimal_2:37' cur_1st = max(1, cur_end-m+1);
            index cur_1st = std::fmax(1, (cur_end - M) + 1);
            // 'SCP_Block_Optimal_2:38' indicator(1,i) = cur_1st;
            indicator[2 * (b_i - 1)] = cur_1st;
            // 'SCP_Block_Optimal_2:39' indicator(2,i) = cur_end;
            indicator[2 * (b_i - 1) + 1] = cur_end;
            // 'SCP_Block_Optimal_2:40' cur_end = cur_1st - 1;
            cur_end = cur_1st - 1;
            // 'SCP_Block_Optimal_2:41' i = i + 1;
            b_i++;
        }

        cout << "6. Block Construction: ";
        for (index d_ : d_s) {
            cout << d_ << ", ";
        }
        cout << "\n";
    }


    template<typename scalar, typename index, index m, index n>
    void init_guess(index init_value, vector<scalar> *z_B, scalar *x_R) {
        if (init_value == 0) {
            for (index i = 0; i < n; i++)
                z_B->at(i) = 0;
        } else if (init_value == 2) {
            for (index i = 0; i < n; i++)
                z_B->at(i) = x_R[i];
        } else if (init_value == 1) {
            for (index i = 0; i < n; i++)
                z_B->at(i) = round(std::pow(2, qam) / 2);
        }

    }
}

#endif //CILS_CONFIG_H
