#ifndef CILS_CONFIG_H
#define CILS_CONFIG_H

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <climits>


const static int M = 7;
const static int N = 20;

using namespace std;

namespace cils {

    namespace program_def {
        typedef int index;
        typedef double scalar;
        /**
         *   omp_sched_static = 0x1,
         *   omp_sched_dynamic = 0x2,
         *   omp_sched_guided = 0x3,
         *   omp_sched_auto = 0x4,
         */
        index k = 3;
        index SNR = 35;
        index max_iter = 1000;
        index search_iter = 1000;
        index stop = 3;
        index schedule = 2;
        index chunk_size = 1;
        index block_size = 32;
        index spilt_size = 2;
        index offset = 2;
        index is_constrained = false;
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
        index max_proc = min(omp_get_max_threads(), N / block_size);
        index max_thre = 400000;//maximum search allowed for serial ils.


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
//        std::vector<index> d_s(N / block_size, block_size);

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
                   (int) pow(k, 4), SNR, max_iter, block_size,
                   search_iter, stop, num_trials, max_search,
                   is_local, is_constrained, is_matlab);

            for (int i = 0; i < spilt_size; i++) {
                d_s[i] = block_size / spilt_size;
            }
            d_s[0] -= offset;
            d_s[1] += offset;

            for (index i = d_s.size() - 2; i >= 0; i--) {
                d_s[i] += d_s[i + 1];
            }
            cout << "6. Block Construction: ";
            for (index d_ : d_s) {
                cout << d_ << ", ";
            }
            cout << "\n";
        }


        template<typename scalar, typename index, index m ,index n>
        void init_guess(index init_value, vector<scalar> *z_B, scalar *x_R) {
            if (init_value == 0) {
                for (index i = 0; i < n; i++)
                    z_B->at(i) = 0;
            } else if (init_value == 2) {
                for (index i = 0; i < n; i++)
                    z_B->at(i) = x_R[i];
            } else if (init_value == 1) {
                for (index i = 0; i < n; i++)
                    z_B->at(i) = round(std::pow(2, k) / 2);
            }

        }
    }


}
#endif //CILS_CONFIG_H
