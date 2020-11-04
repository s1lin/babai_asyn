#ifndef BABAI_SEARCH_ASYN_H
#define BABAI_SEARCH_ASYN_H

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

using namespace std;

namespace babai {

    template<typename scalar, typename index, index n>
    scalar find_residual(const scalar *R, const scalar *y, const scalar *x);

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    class Babai_search_asyn {
    public:
        index size_R_A;
        scalar init_res, noise, *R_A, *y_A, *x_R, *x_tA;

    private:
        //Utils:
        void read_from_RA();

        void read_x_y();

        void write_R_A();

    public:
        explicit Babai_search_asyn(scalar noise);

        ~Babai_search_asyn() {
            free(R_A);
            free(x_R);
            free(y_A);
        }

        void init();

        inline scalar do_solve(const index i, const scalar *z_B) {
            scalar sum = 0;
#pragma omp simd reduction(+ : sum)
            for (index col = n - i; col < n; col++)
                sum += R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * z_B[col];

            return round((y_A[n - 1 - i] - sum) / R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
        }

        scalar *search_omp(index n_proc, index nswp, index *update, scalar *z_B, scalar *z_B_p);

        vector<scalar> search_vec(vector<scalar> z_B);

        scalar *sils_search(scalar *z_B);
    };


}
#endif