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

#ifndef SILS_H
#define SILS_H

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
     * Return scalar pointer array along with the size.
     * @tparam scalar
     * @tparam index
     */
    template<typename scalar, typename index>
    struct returnType {
        vector<index> x;
        scalar run_time;
        scalar residual;
        index num_iter;
    };

    /**
     * Return the result of norm2(y-R*x).
     * @tparam scalar
     * @tparam index
     * @tparam n
     * @param R
     * @param y
     * @param x
     * @return residual
     */
    template<typename scalar, typename index, index n>
    inline scalar find_residual(const scalarType<scalar, index> *R,
                                const scalarType<scalar, index> *y,
                                const vector<index> *x) {
        scalar res = 0, sum = 0;
        for (index i = 0; i < n; i++) {
            sum = 0;
            for (index j = i; j < n; j++) {
                sum += x->at(j) * R->x[(n * i) + j - ((i * (i + 1)) / 2)];
            }
            res += (y->x[i] - sum) * (y->x[i] - sum);
        }
        return std::sqrt(res);
    }

    template<typename scalar, typename index, index n>
    inline scalar find_bit_error_rate(const vector<index> *x_b,
                                      const vector<index> *x_t) {
        index error = 0, size = sizeof(index);
        for (index i = 0; i < n; i++) {
            std::string binary_x_b = std::bitset<sizeof(index)>(x_b->at(i)).to_string(); //to binary
            std::string binary_x_t = std::bitset<sizeof(index)>(x_t->at(i)).to_string();
            for (index j = 0; j < size; j++)
                if (binary_x_b[j] != binary_x_t[j])
                    error++;

        }
        return (scalar) error / (n * size);
    }

    /**
     * Return the result of norm2(y-R*x).
     * @tparam scalar
     * @tparam index
     * @tparam n
     * @param R
     * @param y
     * @param x
     * @return residual
     */
    template<typename scalar, typename index, index n>
    inline scalar norm(const vector<scalar> *x,
                       const vector<scalar> *y) {
        scalar res = 0;
        for (index i = 0; i < n; i++) {
            res += (y->at(i) - x->at(i)) * (y->at(i) - x->at(i));
        }
        return std::sqrt(res);
    }


    /**
     * Return the result of norm2(y-R*x).
     * @tparam scalar
     * @tparam index
     * @tparam n
     * @param R
     * @param y
     * @param x
     * @return residual
     */
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

    /**
     * Simple function for displaying the struct scalarType
     * @tparam scalar
     * @tparam index
     * @param x
     */
    template<typename scalar, typename index>
    inline void display_vector(const vector<index> *x) {
        for (int i = 0; i < x->size(); i++) {
            cout << x->at(i) << " ";
        }
        cout << endl;
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
    inline vector<scalar> find_block_Rii(const scalarType<scalar, index> *R_B,
                                         const index row_begin, const index row_end,
                                         const index col_begin, const index col_end,
                                         const index block_size) {
        index size = col_end - col_begin;
        vector<scalar> R_b_s(size * (1 + size) / 2, 0);
        index counter = 0, i = 0;

        //The block operation
        for (index row = row_begin; row < row_end; row++) {
            //Translating the index from R(matrix) to R_B(compressed array).
            for (index col = row; col < col_end; col++) {
                i = (block_size * row) + col - ((row * (row + 1)) / 2);
                //Put the value into the R_b_s
                R_b_s[counter] = R_B->x[i];
                counter++;
            }
        }

        return R_b_s;
    }

    /**
     *
     * @tparam scalar
     * @tparam index
     * @param R_B
     * @param x
     * @param y
     * @param row_begin
     * @param row_end
     * @param col_begin
     * @param col_end
     * @return
     */
    template<typename scalar, typename index>
    inline vector<scalar> block_residual_vector(const scalarType<scalar, index> *R_B,
                                                const vector<index> *x,
                                                const vector<scalar> *y,
                                                const index row_begin, const index row_end,
                                                const index col_begin, const index col_end) {
        index block_size = row_end - row_begin;
        vector<scalar> y_b_s(block_size, 0);

        index counter = 0, prev_i = 0, i;
        scalar sum = 0;
//        cout<<"R_B"<<endl;
        //The block operation
        for (index row = row_begin; row < row_end; row++) {
            //Translating the index from R(matrix) to R_B(compressed array).
            for (index col = col_begin; col < col_end; col++) {
                i = (col_end * row) + col - ((row * (row + 1)) / 2);
                sum += R_B->x[i] * x->at(counter);
                counter++;
//                cout<< R_B->at(i) << " ";
            }
//            cout<<endl;
            y_b_s[prev_i] = y->at(row - row_begin) - sum;
            prev_i++;
            sum = counter = 0;
        }
//        cout<<"R_B_END"<<endl;
        return y_b_s;

    }

    /**
     *
     * @tparam scalar
     * @tparam index
     * @param x*
     * @param begin
     * @param end
     * @return scalarType*
     */
    template<typename scalar, typename index>
    inline vector<index> find_block_x(const vector<index> *x,
                                      const index begin,
                                      const index end) {
        vector<index> z(end - begin, 0);
        for (index i = begin; i < end; i++) {
            z[i - begin] = x->at(i);
        }
        return z;
    }

    /**
     *
     * @tparam scalar
     * @tparam index
     * @param y*
     * @param begin
     * @param end
     * @return scalarType*
     */
    template<typename scalar, typename index>
    inline vector<scalar> find_block_x(const scalarType<scalar, index> *x,
                                       const index begin,
                                       const index end) {
        vector<scalar> z(end - begin, 0);
        for (index i = begin; i < end; i++) {
            z[i - begin] = x->x[i];
        }
        return z;
    }


    /**
     *
     * @tparam scalar
     * @tparam index
     * @param first
     * @param second
     * @return scalarType*
     */
    template<typename scalar, typename index>
    inline vector<scalar> concatenate_array(vector<scalar> *first,
                                            vector<scalar> *second) {
        index size = first->size() + second->size();
        vector<scalar> z(size, 0);

        for (int i = 0; i < first->size; i++) {
            z->at(i) = first->at(i);
        }
        for (int i = first->size; i < size; i++) {
            z->at(i) = second->at(i - first->size());
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
    template<typename scalar, typename index, bool is_read, index n>
    class SILS {

    public:

        index qam, snr;
        scalar init_res;
        vector<index> x_R, x_t;
        scalarType<scalar, index> *A, *R_A, *y_A;

    private:
        /**
         * read the problem from files
         */
        void read();

        /**
         * Warning: OPENMP enabled.
         * @param i
         * @param z_B
         * @return
         */
        inline index babai_solve_omp(const index i, vector<index> *z_B) {
            scalar sum = 0;
            index ni = n - 1 - i;
#pragma omp simd reduction(+ : sum)
            for (index col = n - i; col < n; col++)
                sum += R_A->x[ni * n - (ni * (n - i)) / 2 + col] * z_B->at(col);

            return round((y_A->x[ni] - sum) / R_A->x[ni * n - (ni * (n - i)) / 2 + ni]);
        }

        /**
         *
         * @param n_dx_q_0
         * @param n_dx_q_1
         * @param y
         * @return
         */
        inline vector<index> ils_search_omp(const index n_dx_q_0, const index n_dx_q_1,
                                            const vector<scalar> *y) {

            //variables
            scalar sum, newprsd, gamma, beta = INFINITY;

            index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1, iter = 0;
            index end_1 = n_dx_q_1 - 1, row_k = k + n_dx_q_0;

            vector<scalar> p(dx, 0), c(dx, 0);
            vector<index> z(dx, 0), x(dx, 0), d(dx, 0);

            //  Initial squared search radius
            scalar R_kk = R_A->x[(n * end_1) + end_1 - ((end_1 * (end_1 + 1)) / 2)];
            c[k] = y->at(k) / R_kk;
            z[k] = round(c[k]);
            gamma = R_kk * (c[k] - z[k]);

            //  Determine enumeration direction at level block_size
            d[dx - 1] = c[dx - 1] > z[dx - 1] ? 1 : -1;

            //ILS search process
            while (true) {
//                iter++;
                newprsd = p[k] + gamma * gamma;
                if (newprsd < beta) {
                    if (k != 0) {
#pragma omp atomic
                        k--;
#pragma omp atomic
                        row_k--;
                        sum = 0;

#pragma omp simd reduction(+ : sum)
                        for (index col = k + 1; col < dx; col++) {
                            sum += R_A->x[(n * row_k) + col + n_dx_q_0 - ((row_k * (row_k + 1)) / 2)] * z[col];
                        }
                        R_kk = R_A->x[(n * row_k) + row_k - ((row_k * (row_k + 1)) / 2)];

                        p[k] = newprsd;
                        c[k] = (y->at(k) - sum) / R_kk;
                        z[k] = round(c[k]);
                        gamma = R_kk * (c[k] - z[k]);

                        d[k] = c[k] > z[k] ? 1 : -1;

                    } else {
                        beta = newprsd;
#pragma omp simd
                        for (index l = 0; l < dx; l++) {
                            x[l] = z[l];
                        }
                        //x.assign(z.begin(), z.end());
                        z[0] += d[0];
                        gamma = R_A->x[0] * (c[0] - z[0]);
                        d[0] = d[0] > 0 ? -d[0] - 1 : -d[0] + 1;
                    }
                } else {
                    if (k == dx - 1) break;
                    else {
#pragma omp atomic
                        k++;
#pragma omp atomic
                        row_k++;
                        z[k] += d[k];
                        gamma = R_A->x[(n * row_k) + row_k - ((row_k * (row_k + 1)) / 2)] * (c[k] - z[k]);
                        d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                    }
                }
            }
            return x;
        }


        /**
         *
         * @param R_B
         * @param y_B
         * @return
         */
        inline vector<index> ils_search(const vector<scalar> *R_B, const vector<scalar> *y_B) {

            //variables
            index block_size = y_B->size();

            vector<index> z(block_size, 0), d(block_size, 0), z_H(block_size, 0);
            vector<scalar> p(block_size, 0), c(block_size, 0);

            scalar newprsd, gamma, b_R, beta = INFINITY;
            index k = block_size - 1, i, j;

            c[block_size - 1] = y_B->at(block_size - 1) / R_B->at(R_B->size() - 1);
            z[block_size - 1] = round(c[block_size - 1]);
            gamma = R_B->at(R_B->size() - 1) * (c[block_size - 1] - z[block_size - 1]);

            //  Determine enumeration direction at level block_size
            d[block_size - 1] = c[block_size - 1] > z[block_size - 1] ? 1 : -1;

            while (true) {
                newprsd = p[k] + gamma * gamma;
                if (newprsd < beta) {
                    if (k != 0) {
                        k--;
                        scalar sum = 0;
                        for (index col = k + 1; col < block_size; col++) {
                            sum += R_B->at((block_size * k) + col - ((k * (k + 1)) / 2)) * z[col];
                        }
                        scalar s = y_B->at(k) - sum;
                        scalar R_kk = R_B->at((block_size * k) + k - ((k * (k + 1)) / 2));
                        p[k] = newprsd;
                        c[k] = s / R_kk;

                        z[k] = round(c[k]);
                        gamma = R_kk * (c[k] - z[k]);
                        d[k] = c[k] > z[k] ? 1 : -1;

                    } else {
                        beta = newprsd;
                        //Deep Copy of the result
                        for (index l = 0; l < block_size; l++) {
                            z_H[l] = z[l];
                        }
                        //std::memcpy(z_H, z, sizeof(scalar) * block_size);

                        z[0] += d[0];
                        gamma = R_B->at(0) * (c[0] - z[0]);
                        d[0] = d[0] > 0 ? -d[0] - 1 : -d[0] + 1;
                    }
                } else {
                    if (k == block_size - 1) break;
                    else {
                        k++;
                        z[k] += d[k];
                        gamma = R_B->at((block_size * k) + k - ((k * (k + 1)) / 2)) * (c[k] - z[k]);
                        d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                    }
                }
            }
            for (index l = 0; l < block_size; l++) {
                z[l] = z_H[l];
            }
            return z;
        }


        inline vector<index> ils_reduction(const scalarType<scalar, index> *B, const vector<scalar> *y_B) {
            scalar colNormB = new scalar[2][n];
            for (index j = 0; j < n; j++) {
                scalar b_0 = B[0][j];
                scalar b_1 = B[1][j];
                colNormB[0][j] = b_0 * b_0 + b_1 * b_1;
            }
            index n_dim = n;
            for (index k = 0; k < n_dim; k++) {

            }
        }

    public:
        explicit SILS(index qam, index snr);

        ~SILS() {
            free(R_A);
            free(y_A);
        }

        void init();

        /**
         *
         * @param n_proc
         * @param nswp
         * @param update
         * @param z_B
         * @param z_B_p
         * @return
         */
        returnType<scalar, index>
        sils_babai_search_omp(index n_proc, index nswp, vector<index> *z_B);

        /**
         *
         * @param z_B
         * @return
         */
        returnType<scalar, index>
        sils_babai_search_serial(vector<index> *z_B);


        returnType<scalar, index>
        sils_block_search_serial(vector<index> *z_B, vector<index> *d);

        /**
         *
         * @param n_proc
         * @param nswp
         * @param R_B
         * @param y_B
         * @param z_B
         * @param d
         * @return
         */
        returnType<scalar, index>
        sils_block_search_omp(index n_proc, index nswp, scalar stop, vector<index> *z_B, vector<index> *d);


    };
}
#endif