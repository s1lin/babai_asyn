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

#ifndef CILS_HELPER_H
#define CILS_HELPER_H

#endif //CILS_HELPER_H

#include <cmath>
#include <cstring>

const double ZERO = 3.3121686421112381E-170;
using namespace std;
namespace helper {

    void b_rand(const int n, double *r) {
        unsigned int state[625];
        unsigned int u[2];
        int i = n;
        for (int k{0}; k < i; k++) {
            double b_r;
            // ========================= COPYRIGHT NOTICE ============================
            //  This is a uniform (0,1) pseudorandom number generator based on:
            //
            //  A C-program for MT19937, with initialization improved 2002/1/26.
            //  Coded by Takuji Nishimura and Makoto Matsumoto.
            //
            //  Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
            //  All rights reserved.
            //
            //  Redistribution and use in source and binary forms, with or without
            //  modification, are permitted provided that the following conditions
            //  are met:
            //
            //    1. Redistributions of source code must retain the above copyright
            //       notice, this list of conditions and the following disclaimer.
            //
            //    2. Redistributions in binary form must reproduce the above copyright
            //       notice, this list of conditions and the following disclaimer
            //       in the documentation and/or other materials provided with the
            //       distribution.
            //
            //    3. The names of its contributors may not be used to endorse or
            //       promote products derived from this software without specific
            //       prior written permission.
            //
            //  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
            //  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
            //  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
            //  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
            //  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
            //  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
            //  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
            //  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
            //  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
            //  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
            //  OF THIS  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
            //
            // =============================   END   =================================
            do {
                for (int j{0}; j < 2; j++) {
                    unsigned int mti;
                    unsigned int y;
                    mti = state[624] + 1U;
                    if (state[624] + 1U >= 625U) {
                        int kk;
                        for (kk = 0; kk < 227; kk++) {
                            y = (state[kk] & 2147483648U) | (state[kk + 1] & 2147483647U);
                            if ((y & 1U) == 0U) {
                                y >>= 1U;
                            } else {
                                y = y >> 1U ^ 2567483615U;
                            }
                            state[kk] = state[kk + 397] ^ y;
                        }
                        for (kk = 0; kk < 396; kk++) {
                            y = (state[kk + 227] & 2147483648U) |
                                (state[kk + 228] & 2147483647U);
                            if ((y & 1U) == 0U) {
                                y >>= 1U;
                            } else {
                                y = y >> 1U ^ 2567483615U;
                            }
                            state[kk + 227] = state[kk] ^ y;
                        }
                        y = (state[623] & 2147483648U) | (state[0] & 2147483647U);
                        if ((y & 1U) == 0U) {
                            y >>= 1U;
                        } else {
                            y = y >> 1U ^ 2567483615U;
                        }
                        state[623] = state[396] ^ y;
                        mti = 1U;
                    }
                    y = state[static_cast<int>(mti) - 1];
                    state[624] = mti;
                    y ^= y >> 11U;
                    y ^= y << 7U & 2636928640U;
                    y ^= y << 15U & 4022730752U;
                    u[j] = y ^ y >> 18U;
                }
                u[0] >>= 5U;
                u[1] >>= 6U;
                b_r = 1.1102230246251565E-16 * (static_cast<double>(u[0]) * 6.7108864E+7 +
                                                static_cast<double>(u[1]));
            } while (b_r == 0.0);
            r[k] = b_r;
        }
    }

    void randperm(int n, double *p) {
        std::vector<int> idx;
        std::vector<int> iwork;
        int b_i;
        int b_n;
        int i;
        int qEnd;
        b_rand(n, p);
        b_n = n + 1;
        idx.resize(n);
        i = n;
        for (b_i = 0; b_i < i; b_i++) {
            idx[b_i] = 0;
        }
        if (n != 0) {
            double d;
            int k;
            iwork.resize(n);
            b_i = n - 1;
            for (k = 1; k <= b_i; k += 2) {
                d = p[k];
                if ((p[k - 1] <= d) || std::isnan(d)) {
                    idx[k - 1] = k;
                    idx[k] = k + 1;
                } else {
                    idx[k - 1] = k + 1;
                    idx[k] = k;
                }
            }
            if ((n & 1) != 0) {
                idx[n - 1] = n;
            }
            i = 2;
            while (i < b_n - 1) {
                int i2;
                int j;
                i2 = i << 1;
                j = 1;
                for (int pEnd{i + 1}; pEnd < b_n; pEnd = qEnd + i) {
                    int b_p;
                    int kEnd;
                    int q;
                    b_p = j;
                    q = pEnd - 1;
                    qEnd = j + i2;
                    if (qEnd > b_n) {
                        qEnd = b_n;
                    }
                    k = 0;
                    kEnd = qEnd - j;
                    while (k + 1 <= kEnd) {
                        d = p[idx[q] - 1];
                        b_i = idx[b_p - 1];
                        if ((p[b_i - 1] <= d) || std::isnan(d)) {
                            iwork[k] = b_i;
                            b_p++;
                            if (b_p == pEnd) {
                                while (q + 1 < qEnd) {
                                    k++;
                                    iwork[k] = idx[q];
                                    q++;
                                }
                            }
                        } else {
                            iwork[k] = idx[q];
                            q++;
                            if (q + 1 == qEnd) {
                                while (b_p < pEnd) {
                                    k++;
                                    iwork[k] = idx[b_p - 1];
                                    b_p++;
                                }
                            }
                        }
                        k++;
                    }
                    for (k = 0; k < kEnd; k++) {
                        idx[(j + k) - 1] = iwork[k];
                    }
                    j = qEnd;
                }
                i = i2;
            }
        }
        for (b_i = 0; b_i < n; b_i++) {
            p[b_i] = idx[b_i];
        }
    }

    template<typename scalar, typename index>
    void eye(index n, scalar *A) {
        for (index i = 0; i < n * n; i++) {
            A[i] = 0;
        }
        for (index i = 0; i < n; i++) {
            A[i + n * i] = 1.0;
        }
    }

    template<typename scalar, typename index, index m, index n, index mb>
    void b_mtimes(const array<scalar, m * n> &A_C, const array<scalar, n * mb> &B, array<scalar, m * mb> &C) {
        C.fill(0);
        for (index j = 0; j < mb; j++) {
            index coffset = j * m;
            for (index k = 0; k < n; k++) {
                scalar bkj = B[k * n + j];
                index aoffset = k * m;
                for (index i = 0; i < m; i++) {
                    index t = coffset + i;
                    C[t] = C[t] + A_C[aoffset + i] * bkj;
                }
            }
        }
    }

    //B: m x n
    template<typename scalar, typename index, index m, index n>
    void mtimes(const scalar A_C[4], const array<scalar, m * n> &B, array<scalar, 2 * n> &C) {
        for (index j = 0; j < n; j++) {
            index coffset_tmp = j << 1;
            scalar s_tmp = B[coffset_tmp + 1];
            C[coffset_tmp] = A_C[0] * B[coffset_tmp] + A_C[2] * s_tmp;
            C[coffset_tmp + 1] = A_C[1] * B[coffset_tmp] + A_C[3] * s_tmp;
        }
    }

    template<typename scalar, typename index, index m, index n>
    void mtimes(const array<scalar, m * m> &Q, const array<scalar, m * n> &R, array<scalar, m * n> &A_t) {
        for (index j = 0; j < n; j++) {
            for (index k = 0; k < m; k++) {
                for (index i = 0; i < m; i++) {
                    A_t[j * m + i] += Q[k * m + i] * R[j * m + k];
                }
            }
        }
    }

    template<typename scalar, typename index>
    void mtimes_v(index m, index n, const vector<scalar> &Q, const vector<scalar> &R, vector<scalar> &A_t) {
        for (index j = 0; j < n; j++) {
            for (index k = 0; k < m; k++) {
                for (index i = 0; i < m; i++) {
                    A_t[j * m + i] += Q[k * m + i] * R[j * m + k];
                }
            }
        }
    }

    template<typename scalar, typename index>
    void mtimes_col(index m, index n, const vector<scalar> &Q, const vector<scalar> &R, vector<scalar> &A_t) {
        for (index j = 0; j < n; j++) {
            for (index k = 0; k < m; k++) {
                for (index i = 0; i < m; i++) {
                    A_t[j * m + i] += Q[k * m + i] * R[j * n + k];
                }
            }
        }
    }

    template<typename scalar, typename index>
    void inv(const index K, const index N, const vector<scalar> &x, vector<scalar> &y) {
        vector<scalar> b_x;
        vector<index> ipiv, p;
        if ((K == 0) || (N == 0)) {
            int b_n;
            y.resize(K * N);
            b_n = K * N;
            for (index i = 0; i < b_n; i++) {
                y[i] = x[i];
            }
        } else {
            index b_i, b_n, i, i1, j, k, ldap1, n, u1, yk;
            n = K;
            y.resize(K * N);
            b_n = K * N;
            for (i = 0; i < b_n; i++) {
                y[i] = 0.0;
            }
            b_x.resize(K * N);
            b_n = K * N;
            for (i = 0; i < b_n; i++) {
                b_x[i] = x[i];
            }
            b_n = K;
            ipiv.resize(1 * K);
            ipiv[0] = 1;
            yk = 1;
            for (k = 2; k <= b_n; k++) {
                yk++;
                ipiv[k - 1] = yk;
            }
            ldap1 = K;
            b_n = K - 1;
            u1 = K;
            if (b_n < u1) {
                u1 = b_n;
            }
            for (j = 0; j < u1; j++) {
                scalar smax;
                index jj;
                index jp1j;
                index mmj_tmp = n - j;
                yk = j * (n + 1);
                jj = j * (ldap1 + 1);
                jp1j = yk + 2;
                if (mmj_tmp < 1) {
                    b_n = -1;
                } else {
                    b_n = 0;
                    if (mmj_tmp > 1) {
                        smax = std::abs(b_x[jj]);
                        for (k = 2; k <= mmj_tmp; k++) {
                            double s;
                            s = std::abs(b_x[(yk + k) - 1]);
                            if (s > smax) {
                                b_n = k - 1;
                                smax = s;
                            }
                        }
                    }
                }
                if (b_x[jj + b_n] != 0.0) {
                    if (b_n != 0) {
                        b_n += j;
                        ipiv[j] = b_n + 1;
                        for (k = 0; k < n; k++) {
                            smax = b_x[j + k * n];
                            b_x[j + k * n] = b_x[b_n + k * n];
                            b_x[b_n + k * n] = smax;
                        }
                    }
                    i = jj + mmj_tmp;
                    for (b_i = jp1j; b_i <= i; b_i++) {
                        b_x[b_i - 1] = b_x[b_i - 1] / b_x[jj];
                    }
                }
                b_n = yk + n;
                yk = jj + ldap1;
                for (jp1j = 0; jp1j <= mmj_tmp - 2; jp1j++) {
                    smax = b_x[b_n + jp1j * n];
                    if (b_x[b_n + jp1j * n] != 0.0) {
                        i = yk + 2;
                        i1 = mmj_tmp + yk;
                        for (b_i = i; b_i <= i1; b_i++) {
                            b_x[b_i - 1] = b_x[b_i - 1] + b_x[((jj + b_i) - yk) - 1] * -smax;
                        }
                    }
                    yk += n;
                }
            }
            b_n = K;
            p.resize(1 * K);
            p[0] = 1;
            yk = 1;
            for (k = 2; k <= b_n; k++) {
                yk++;
                p[k - 1] = yk;
            }
            i = K;
            for (k = 0; k < i; k++) {
                i1 = ipiv[k];
                if (i1 > k + 1) {
                    b_n = p[i1 - 1];
                    p[i1 - 1] = p[k];
                    p[k] = b_n;
                }
            }
            for (k = 0; k < n; k++) {
                i = p[k];
                y[k + K * (i - 1)] = 1.0;
                for (j = k + 1; j <= n; j++) {
                    if (y[(j + K * (i - 1)) - 1] != 0.0) {
                        i1 = j + 1;
                        for (b_i = i1; b_i <= n; b_i++) {
                            y[(b_i + K * (i - 1)) - 1] =
                                    y[(b_i + K * (i - 1)) - 1] -
                                    y[(j + K * (i - 1)) - 1] *
                                    b_x[(b_i + K * (j - 1)) - 1];
                        }
                    }
                }
            }
            for (j = 0; j < n; j++) {
                b_n = n * j - 1;
                for (k = n; k >= 1; k--) {
                    yk = n * (k - 1) - 1;
                    i = k + b_n;
                    if (y[i] != 0.0) {
                        y[i] = y[i] / b_x[k + yk];
                        for (b_i = 0; b_i <= k - 2; b_i++) {
                            i1 = (b_i + b_n) + 1;
                            y[i1] = y[i1] - y[i] * b_x[(b_i + yk) + 1];
                        }
                    }
                }
            }
        }
    }

    /**
     * Matrix-vector multiplication: Ax=y, where A is m-by-n matrix
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param m : integer scalar, size of the matrix
     * @param n : integer scalar, size of the matrix
     * @param A : matrix, m-by-n in pointer
     * @param x : vector, n-by-1 in pointer
     * @param y : vector, m-by-1 in pointer, storing result.
     */
    template<typename scalar, typename index>
    void mtimes_Axy(const index m, const index n, const scalar *A, const scalar *x, scalar *y) {
        scalar sum = 0;
        for (index i = 0; i < m; i++) {
            sum = 0;
            for (index j = 0; j < n; j++) {
                sum += A[j * m + i] * x[j];
            }
            y[i] = sum;
        }
    }


    /**
     * 
     * @tparam scalar 
     * @tparam index 
     * @tparam n 
     * @param x 
     * @param y 
     * @return 
     */
    template<typename scalar, typename index, index n>
    index length_nonzeros(const scalar *x, const scalar *y) {
        index diff = 0;
        for (index i = 0; i < n; i++) {
            diff += (x[i] != y[i]);
        }
        return diff;
    }

//    template<typename scalar, typename index, index m, index n, index mb>
//    void mrdiv(array<scalar, m * n> &B, const array<scalar, n * mb> &A) {
//        using namespace matlab::engine;
//
//        // Start MATLAB engine synchronously
//        std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
//
//        //Create MATLAB data array factory
//        matlab::data::ArrayFactory factory;
//
//        // Call the MATLAB movsum function
//        matlab::data::TypedArray<scalar> A_M = factory.createArray(
//                {static_cast<unsigned long>(m), static_cast<unsigned long>(n)}, A.begin(), A.end());
//        matlab::data::TypedArray<scalar> B_M = factory.createArray(
//                {static_cast<unsigned long>(n), static_cast<unsigned long>(mb)}, B.begin(), B.end());
//        matlabPtr->setVariable(u"A", std::move(A_M));
//        matlabPtr->setVariable(u"B", std::move(B_M));
//
//        // Call the MATLAB movsum function
//        matlabPtr->eval(u" R = mldivide(A, B);");
//
//        matlab::data::TypedArray<scalar> const R = matlabPtr->getVariable(u"R");
//        index i = 0;
//        for (auto r : R) {
//            B[i] = r;
//            ++i;
//        }
//    }

    /**
     * Givens plane rotation
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param x : A 2-component column vector.
     * @param G : A 2-by-2 orthogonal matrix G so that y = G*x has y(2) = 0.
     */
    template<typename scalar, typename index>
    void planerot(scalar x[2], scalar G[4]) {
        if (x[1] != 0.0) {
            scalar absxk, r, scale, t;
            scale = ZERO;
            absxk = std::abs(x[0]);
            if (absxk > ZERO) {
                r = 1.0;
                scale = absxk;
            } else {
                t = absxk / ZERO;
                r = t * t;
            }
            absxk = std::abs(x[1]);
            if (absxk > scale) {
                t = scale / absxk;
                r = r * t * t + 1.0;
                scale = absxk;
            } else {
                t = absxk / scale;
                r += t * t;
            }
            r = scale * std::sqrt(r);
            scale = x[0] / r;
            G[0] = scale;
            G[2] = x[1] / r;
            G[1] = -x[1] / r;
            G[3] = scale;
            x[0] = r;
            x[1] = 0.0;
        } else {
            G[1] = 0.0;
            G[2] = 0.0;
            G[0] = 1.0;
            G[3] = 1.0;
        }
    }

    /**
     * The Euclidean norm of vector v. This norm is also called the 2-norm, vector magnitude, or Euclidean length.
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param n : the size of the vector
     * @param v : input vector
     * @return
     */
    template<typename scalar, typename index>
    scalar norm(const index n, const scalar *v) {
        scalar y;
        if (n == 0) {
            return 0;
        } else if (n == 1) {
            return std::abs(v[0]);
        }
        scalar scale = ZERO;
        for (index k = 0; k < n; k++) {
            scalar absxk;
            absxk = std::abs(v[k]);
            if (absxk > scale) {
                scalar t;
                t = scale / absxk;
                y = y * t * t + 1.0;
                scale = absxk;
            } else {
                scalar t;
                t = absxk / scale;
                y += t * t;
            }
        }
        y = scale * std::sqrt(y);
        return y;
    }

    /**
     * Find BER with given two vectors
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param n : integer scalar, size of the vector
     * @param x_b : input vector 1
     * @param x_t : input vector 2
     * @param k : log_4(qam)
     * @return
     */
    template<typename scalar, typename index>
    scalar find_bit_error_rate(const index n, const scalar *x_b, const scalar *x_t, const index k) {
        index error = 0;
        for (index i = 0; i < n; i++) {
            std::string binary_x_b, binary_x_t;
            switch (k) {
                case 1:
                    binary_x_b = std::bitset<1>((index) x_b[i]).to_string(); //to binary
                    binary_x_t = std::bitset<1>((index) x_t[i]).to_string();
                    break;
                case 2:
                    binary_x_b = std::bitset<2>((index) x_b[i]).to_string(); //to binary
                    binary_x_t = std::bitset<2>((index) x_t[i]).to_string();
                    break;
                default:
                    binary_x_b = std::bitset<3>((index) x_b[i]).to_string(); //to binary
                    binary_x_t = std::bitset<3>((index) x_t[i]).to_string();
                    break;
            }
//            cout << binary_x_b << "-" << binary_x_t << " ";
            for (index j = 0; j < k; j++) {
                if (binary_x_b[j] != binary_x_t[j])
                    error++;
            }
        }
        return (scalar) error / (n * k);
    }

    /**
     * Simple function for displaying a m-by-n matrix with name
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param m : integer scalar, size of the matrix
     * @param n : integer scalar, size of the matrix
     * @param x : matrix, in pointer
     * @param name: display name of the matrix
     */
    template<typename scalar, typename index>
    void display_matrix(index m, index n, const scalar *x, string name) {
        cout << name << ": \n";
        for (index i = 0; i < m; i++) {
            for (index j = 0; j < n; j++) {
                printf("%8.4f ", x[j * m + i]);
            }
            cout << "\n";
        }
        cout << endl;
    }

    /**
     * Simple function for displaying the a vector with name
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param n : integer scalar, size of the vector
     * @param x : vector, in pointer
     * @param name: display name of the vector
     */
    template<typename scalar, typename index>
    void display_vector(const index n, const scalar *x, const string &name) {
        cout << name << ": ";
        scalar sum = 0;
        for (index i = 0; i < n; i++) {
            printf("%8.4f ", x[i]);
            sum += x[i];
        }
        printf("SUM = %8.5f\n", sum);
    }

    /**
     * Determine whether all values of x are true by lambda expression.
     * @tparam index : integer type : integer required
     * @param x : Testing vector
     * @return true/false
     */
    template<typename index>
    bool if_all_x_true(const vector<bool> &x) {
        bool y = (!x.empty());
        if (!y) return y;
//        for(index k = 0; k < x.size(); k++){
//            if(!x[k]) {
//                y = false;
//                break;
//            }
//        }
        //If false, which means no false x, then return true.
        y = std::any_of(x.begin(), x.end(), [](const bool &e) { return !e; });
        return !y;
    }

    /**
     * Returns the same data as in a, but with no repetitions. b is in sorted order.
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param a : input vector to be processed
     * @param b : output vector to store the results
     */
    template<typename scalar, typename index>
    void unique_vector(const vector<scalar> &a, vector<scalar> &b) {
        index t, e, i, l, j, k, size_a_1 = a.size() + 1, size_a = a.size(), jj, p, r, q, r_j;
        vector<index> i_x(size_a, 0), j_x(size_a, 0);
        scalar absx;
        for (i = 1; i < size_a; i += 2) {
            if ((a[i - 1] <= a[i]) || std::isnan(a[i])) {
                i_x[i - 1] = i;
                i_x[i] = i + 1;
            } else {
                i_x[i - 1] = i + 1;
                i_x[i] = i;
            }
        }
        if ((size_a & 1) != 0) {
            i_x[size_a - 1] = size_a;
        }
        i = 2;
        while (i < size_a_1 - 1) {
            l = i << 1;
            j = 1;
            for (p = i + 1; p < size_a_1; p = r + i) {
                jj = j;
                q = p - 1;
                r = j + l;
                if (r > size_a_1) {
                    r = size_a_1;
                }
                k = 0;
                r_j = r - j;
                while (k + 1 <= r_j) {
                    absx = a[i_x[q] - 1];
                    t = i_x[jj - 1];
                    if ((a[t - 1] <= absx) || std::isnan(absx)) {
                        j_x[k] = t;
                        jj++;
                        if (jj == p) {
                            while (q + 1 < r) {
                                k++;
                                j_x[k] = i_x[q];
                                q++;
                            }
                        }
                    } else {
                        j_x[k] = i_x[q];
                        q++;
                        if (q + 1 == r) {
                            while (jj < p) {
                                k++;
                                j_x[k] = i_x[jj - 1];
                                jj++;
                            }
                        }
                    }
                    k++;
                }
                for (k = 0; k < r_j; k++) {
                    i_x[(j + k) - 1] = j_x[k];
                }
                j = r;
            }
            i = l;
        }
        b.resize(size_a);
        for (k = 0; k < size_a; k++) {
            b[k] = a[i_x[k] - 1];
        }
        k = 0;
        while ((k + 1 <= size_a) && std::isinf(b[k]) && (b[k] < 0.0)) {
            k++;
        }
        l = k;
        k = size_a;
        while ((k >= 1) && std::isnan(b[k - 1])) {
            k--;
        }
        p = size_a - k;
        bool flag = false;
        while ((!flag) && (k >= 1)) {
            if (std::isinf(b[k - 1]) && (b[k - 1] > 0.0)) {
                k--;
            } else {
                flag = true;
            }
        }
        i = (size_a - k) - p;
        jj = -1;
        if (l > 0) {
            jj = 0;
        }
        while (l + 1 <= k) {
            scalar x;
            x = b[l];
            index exitg2;
            do {
                exitg2 = 0;
                l++;
                if (l + 1 > k) {
                    exitg2 = 1;
                } else {
                    absx = std::abs(x / 2.0);
                    if ((!std::isinf(absx)) && (!std::isnan(absx))) {
                        if (absx <= ZERO) {
                            absx = ZERO;
                        } else {
                            frexp(absx, &e);
                            absx = std::ldexp(1.0, e - 53);
                        }
                    } else {
                        absx = NAN;
                    }
                    if ((!(std::abs(x - b[l]) < absx)) &&
                        ((!std::isinf(b[l])) || (!std::isinf(x)) ||
                         ((b[l] > 0.0) != (x > 0.0)))) {
                        exitg2 = 1;
                    }
                }
            } while (exitg2 == 0);
            jj++;
            b[jj] = x;
        }
        if (i > 0) {
            jj++;
            b[jj] = b[k];
        }
        l = k + i;
        for (j = 0; j < p; j++) {
            b[(jj + j) + 1] = b[l + j];
        }
        jj += p;
        if (1 > jj + 1) {
            t = 0;
        } else {
            t = jj + 1;
        }
        b.resize(t);
    }

    /**
     * Return the result of ||y-A*x||.
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param m : integer scalar, size of the matrix
     * @param n : integer scalar, size of the matrix
     * @param A : matrix, m-by-n in pointer
     * @param x : vector, n-by-1 in pointer
     * @param y : vector, m-by-1 in pointer, storing result.
     * @return residual : l2 norm
     */
    template<typename scalar, typename index>
    inline scalar find_residual(const index m, const index n, const scalar *A, const scalar *x, const scalar *y) {
        vector<scalar> Ax(m, 0);
        mtimes_Axy(m, n, A, x, Ax.data());
        for (index i = 0; i < m; i++) {
            Ax[i] = y[i] - Ax[i];
        }
        return norm(m, Ax.data());
    }
}
