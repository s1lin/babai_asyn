#ifndef CILS_CONFIG_H
#define CILS_CONFIG_H

#include <vector>
#include <string>
#include <cmath>
#include <iostream>

#define N_4096 (4096)
//#define N_4096 (8192)
#define VERBOSE (0)

using namespace std;

namespace cils {
    typedef int index;
    typedef double scalar;
    namespace program_def {

        index k = 3;
        index SNR = 15;
        index max_iter = 10;
        index search_iter = 3;
        index stop = 2000;
        index schedule = 1;
        index chunk_size = 1;
        index block_size = 16;
        index is_qr = true;
        index is_nc = true;
        index mode = 0; //test mode
        index num_trials = 10; //nswp
        index is_local = 1;
        index max_search = 1000;
        index min_proc = 4;

        index max_proc = omp_get_max_threads();

        string suffix = to_string(N_4096);
        string prefix = is_local ? "../../" : "";
        std::vector<index> d_s(N_4096 / block_size, block_size);

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
                is_qr = stoi(argv[9]);
                is_nc = stoi(argv[10]);
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
        inline scalar diff(const vector<scalar> *x,
                           const vector<scalar> *y) {
            scalar diff = 0;
#pragma omp simd reduction(+ : diff)
            for (index i = 0; i < n; i++) {
                diff += (y->at(i) - x->at(i));
            }
            return diff;
        }

        void init_guess(index init_value, vector<index> *z_B, vector<index> *x_R) {
            z_B->assign(N_4096, 0);
            if (init_value == -1) {
//                cout << "before:" << diff<index, index, N_4096>(z_B, x_R);
                for (index i = 0; i < N_4096; i++)
                    z_B->at(i) = x_R->at(i);
//                cout << " after:" << diff<index, index, N_4096>(z_B, x_R) << endl;
            } else if (init_value == 1)
                z_B->assign(N_4096, std::pow(2, k) / 2);
        }
    }


}
#endif //CILS_CONFIG_H
