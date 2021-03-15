#ifndef CILS_CONFIG_H
#define CILS_CONFIG_H

#include <vector>
#include <string>
#include <cmath>
#include <iostream>
#include <climits>

const static int N = 512;

using namespace std;

namespace cils {
    typedef int index;
    typedef double scalar;
    namespace program_def {
        /**
         *   omp_sched_static = 0x1,
         *   omp_sched_dynamic = 0x2,
         *   omp_sched_guided = 0x3,
         *   omp_sched_auto = 0x4,
         */
        index k = 3;
        index SNR = 35;
        index max_iter = 5;
        index search_iter = 100;
        index stop = 0;
        index schedule = 2;
        index chunk_size = 1;
        index block_size = 32;
        index is_constrained = true;
        index is_read = false;
        index mode = 3; //test mode 3: c++ gen
        index num_trials = 10; //nswp
        index is_local = 1;
        index max_search = 100000;//INT_MAX;
        index min_proc = 3;

        index max_proc = omp_get_max_threads();
        index max_thre = 2000000;//maximum search allowed for serial ils.

        string suffix = "" + to_string(N);
        string prefix = is_local ? "../../" : "";
        std::vector<index> d_s(N / block_size, block_size);

        void init_program_def(int argc, char *argv[]) {
            if (argc != 1) {
                k = stoi(argv[1]);
                SNR = stoi(argv[2]);
                max_iter = stoi(argv[3]);
                search_iter = stoi(argv[4]);
                stop = stoi(argv[5]);
                schedule = stoi(argv[6]);
                chunk_size = stoi(argv[7]);
                block_size = stoi(argv[8]);
                is_constrained = stoi(argv[9]);
                is_read = stoi(argv[10]);
                mode = stoi(argv[11]);
                num_trials = stoi(argv[12]);
                is_local = stoi(argv[13]);
                max_search = stoi(argv[14]);
                min_proc = stoi(argv[15]);
            }
            printf("The settings are: k=%d, SNR=%d, max_iter=%d, search_iter=%d, stop=%d, block_size=%d, "
                   "nswp=%d, max_search=%d\n",
                   k, SNR, max_iter, search_iter, stop, block_size, num_trials, max_search);
            suffix += "_" + to_string(SNR) + "_" + to_string(k);
            prefix = is_local ? "../../" : "";
            for (index i = d_s.size() - 2; i >= 0; i--) {
                d_s[i] += d_s[i + 1];
            }
        }

        template<typename scalar, typename index, index n>
        inline scalar diff(const vector<scalar> *x, const vector<scalar> *y) {
            scalar diff = 0;
            for (index i = 0; i < n; i++) {
                diff += (y->at(i) - x->at(i));
            }
            return diff;
        }

        template<typename scalar, typename index, index n>
        void init_guess(index init_value, vector<index> *z_B, vector<index> *x_R) {
            z_B->assign(n, 0);
            if (init_value == -1) {
                for (index i = 0; i < n; i++)
                    z_B->at(i) = x_R->at(i);
            } else if (init_value == 1)
                z_B->assign(n, std::pow(2, k) / 2);
        }
    }


}
#endif //CILS_CONFIG_H
