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

namespace sils {

    template<typename scalar, typename index>
    struct scalarType {
        scalar *x;
        index size;
    };

    template<typename scalar, typename index, index n>
    scalar find_residual(const scalar *R, const scalar *y, const scalar *x, double *pDouble);

    template<typename scalar, typename index>
    void display_vector(scalarType<scalar, index> x) {
        for (int i = 0; i < x.size; i++) {
            cout << x.x[i] << endl;
        }
    }

    template<typename scalar, typename index>
    inline scalarType<scalar, index> find_block_R(const scalar *R_B,
                                                  const index begin,
                                                  const index block_size,
                                                  const index n) {
        scalarType<scalar, index> R_b_s;
        R_b_s.size = block_size * (1 + block_size) / 2;
        R_b_s.x = (scalar *) calloc(R_b_s.size, sizeof(scalar));

        index counter = 0, prev_i = 0, i;
        for (index row = begin; row < begin + block_size; row++) {
            i = (n * row) + row - ((row * (row + 1)) / 2);
            for (index col = 0; col < block_size - prev_i; col++) {
                R_b_s.x[counter] = R_B[i + col];
                counter++;
            }
            prev_i++;
        }

        return R_b_s;
    }

    template<typename scalar, typename index>
    inline scalar *concatenate_array(const scalar *first, const scalar *second) {
        index fir_size = sizeof(first) / sizeof(first[0]);
        index sec_size = sizeof(second) / sizeof(second[0]);
        auto *z = (scalar *) calloc(fir_size + sec_size, sizeof(scalar));
        for (int i = 0; i < fir_size; i++) {
            z[i] = first[i];
        }
        for (int i = fir_size; i < fir_size + sec_size; i++) {
            z[i] = second[i];
        }
        return z;
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    class SILS {
    public:
        index size_R_A;
        scalar init_res, noise, *R_A, *y_A, *x_R, *x_tA;

    private:
        //Utils:
        void read_from_RA();

        void read_x_y();

        void write_R_A();

    public:
        explicit SILS(scalar noise);

        ~SILS() {
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


        scalar *sils_search(const scalar *R, const scalar *y, scalar *z_B, index block_size, index block_size_R_A);

        scalar *sils_babai_search_omp(index n_proc, index nswp, index *update, scalar *z_B, scalar *z_B_p);

        scalar *sils_babai_search_serial(scalar *z_B);

        scalar *sils_block_search_serial(scalar *R_B,
                                         scalar *y_B,
                                         scalar *z_B,
                                         vector<index> d,
                                         index block_size);

        scalar *sils_babai_block_search_omp(index n_proc, index nswp, index *update, scalar *z_B, scalar *z_B_p);

    };
}
#endif