#ifndef CILS_CONFIG_H
#define CILS_CONFIG_H

#include <vector>
#include <string>
#include <cmath>
#include <iostream>

#define N_4096 (4096)
#define VERBOSE (0)

using namespace std;

namespace cils {
    typedef int index;
    typedef double scalar;
    namespace program_def {

        index k = 3;
        index SNR = 35;
        index max_iter = 5;
        index search_iter = 2;
        index stop = 170;
        index schedule = 1;
        index chunk_size = 1;
        index block_size = 16;
        index is_qr = true;
        index is_nc = false;
        index mode = 1;
        index num_trials = 10; //nswp
        index is_local = 1;

        index min_proc = 10;
        index max_proc = omp_get_max_threads();
        index max_search = 500;

        string suffix = to_string(N_4096) + "_" + to_string(SNR) + "_" + to_string(k);
        string prefix = is_local ? "../../" : "";
        std::vector<index> d(N_4096 / block_size, block_size), d_s(N_4096 / block_size, block_size);

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
            }
            prefix = is_local ? "../../" : "";
            for (index i = d_s.size() - 2; i >= 0; i--) {
                d_s[i] += d_s[i + 1];
            }
        }

        template<typename scalar, typename index, index n>
        inline scalar diff(const vector<scalar> *x,
                           const vector<scalar> *y) {
            scalar d = 0;
#pragma omp simd reduction(+ : d)
            for (index i = 0; i < n; i++) {
                d += (y->at(i) - x->at(i));
            }
            return d;
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
