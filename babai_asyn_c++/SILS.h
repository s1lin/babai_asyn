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

/**
 * namespace of sils
 */
namespace sils {

    /**
     * Return scalar pointer array along with the size.
     * @tparam scalar
     * @tparam index
     */
    template<typename scalar, typename index>
    struct scalarType {
        scalar *x;
        index size;
    };

    /**
     * Return the result of norm2(y-R*x).
     * @tparam scalar
     * @tparam index
     * @tparam n
     * @param R
     * @param y
     * @param x
     * @param pDouble
     * @return residual
     */
    template<typename scalar, typename index, index n>
    scalar find_residual(const scalar *R, const scalar *y, const scalar *x, double *pDouble);

    /**
     * Simple function for displaying the struct scalarType
     * @tparam scalar
     * @tparam index
     * @param x
     */
    template<typename scalar, typename index>
    void display_vector(scalarType<scalar, index> x) {
        for (int i = 0; i < x.size; i++) {
            cout << x.x[i] << endl;
        }
    }

    /**
     * Block Operation on R_B (compressed array).
     * @tparam scalar
     * @tparam index
     * @param R_B
     * @param begin: the starting index of the nxn block
     * @param end: the end index of the nxn block
     * @param n: the size of R_B.
     * @return scalarType
     */
    template<typename scalar, typename index>
    inline scalarType<scalar, index> find_block_R(const scalar *R_B,
                                                  const index begin,
                                                  const index end,
                                                  const index n) {
        index block_size = end - begin;
        scalarType<scalar, index> R_b_s;
        R_b_s.size = block_size * (1 + block_size) / 2;
        R_b_s.x = (scalar *) calloc(R_b_s.size, sizeof(scalar));

        index counter = 0, prev_i = 0, i;

        //The block operation
        for (index row = begin; row < end; row++) {

            //Translating the index from R(matrix) to R_B(compressed array).
            i = (n * row) + row - ((row * (row + 1)) / 2);
            for (index col = 0; col < block_size - prev_i; col++) {
                //Put the value into the R_b_s.x
                R_b_s.x[counter] = R_B[i + col];
                counter++;
            }
            prev_i++;
        }
        return R_b_s;
    }

    /**
     *
     * @tparam scalar
     * @tparam index
     * @param first
     * @param second
     * @return
     */
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

    /**
     * SILS class object
     * @tparam scalar
     * @tparam index
     * @tparam is_read
     * @tparam is_write
     * @tparam n
     */
    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    class SILS {
    public:
        index size_R_A;
        scalar init_res, noise, *R_A, *y_A, *x_R, *x_tA;

    private:
        //Utils:
        /**
         * read the problem from files
         */
        void read();

        void write_R_A();

    public:
        explicit SILS(scalar noise);

        ~SILS() {
            free(R_A);
            free(x_R);
            free(y_A);
        }

        void init();

        /**
         *
         * @param i
         * @param z_B
         * @return
         */
        inline scalar do_solve(const index i, const scalar *z_B) {
            scalar sum = 0;
#pragma omp simd reduction(+ : sum)
            for (index col = n - i; col < n; col++)
                sum += R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * z_B[col];

            return round((y_A[n - 1 - i] - sum) / R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
        }

        /**
         *
         * @param R
         * @param y
         * @param z_B
         * @param block_size
         * @param block_size_R_A
         * @return
         */
        scalar *sils_search(const scalar *R, const scalar *y, scalar *z_B, index block_size, index block_size_R_A);

        /**
         *
         * @param n_proc
         * @param nswp
         * @param update
         * @param z_B
         * @param z_B_p
         * @return
         */
        scalar *sils_babai_search_omp(index n_proc, index nswp, index *update, scalar *z_B, scalar *z_B_p);

        /**
         *
         * @param z_B
         * @return
         */
        scalar *sils_babai_search_serial(scalar *z_B);

        /**
         *
         * @param R_B
         * @param y_B
         * @param z_B
         * @param d
         * @param block_size
         * @return
         */
        scalar *sils_block_search_serial(scalar *R_B,
                                         scalar *y_B,
                                         scalar *z_B,
                                         vector<index> d,
                                         index block_size);

        /**
         *
         * @param n_proc
         * @param nswp
         * @param update
         * @param z_B
         * @param z_B_p
         * @return
         */
        scalar *sils_babai_block_search_omp(index n_proc, index nswp, index *update, scalar *z_B, scalar *z_B_p);

    };
}
#endif