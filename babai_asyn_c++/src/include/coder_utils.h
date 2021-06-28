//
// Created by shilei on 6/23/21.
//

#ifndef CILS_CODER_UTILS_H
#define CILS_CODER_UTILS_H

#endif //CILS_CODER_UTILS_H
#include "coder_array.h"
#include <cmath>
#include <cstring>
#include <time.h>

// Variable Definitions
static unsigned int state[625];

static boolean_T isInitialized_eo_sils_reduction{false};

// Function Declarations
namespace coder {
    static void b_rand(double varargin_1, ::coder::array<double, 1U> &r);

    static double det(const ::coder::array<double, 2U> &x);

    static void eye(double varargin_1, ::coder::array<double, 2U> &b_I);

    static void genrand_uint32_vector(unsigned int mt[625], unsigned int u[2]);

    namespace internal {
        namespace blas {
            static void b_mtimes(const ::coder::array<double, 2U> &A_C,
                                 const ::coder::array<double, 2U> &B,
                                 ::coder::array<double, 2U> &C);

            static void mtimes(const double A_C[4],
                               const ::coder::array<double, 2U> &B,
                               ::coder::array<double, 2U> &C);

            static void mtimes(const ::coder::array<double, 2U> &A_C,
                               const ::coder::array<double, 1U> &B,
                               ::coder::array<double, 1U> &C);

            static void mtimes(const ::coder::array<double, 2U> &A_C,
                               const ::coder::array<double, 2U> &B,
                               ::coder::array<double, 2U> &C);

            static void xgerc(int m, int n, double alpha1, int ix0,
                              const ::coder::array<double, 1U> &y_C,
                              ::coder::array<double, 2U> &A_C, int ia0, int lda);

            static double xnrm2(int n, const ::coder::array<double, 2U> &x, int ix0);

        } // namespace blas
        namespace lapack {
            static void xgeqrf(::coder::array<double, 2U> &A_C,
                               ::coder::array<double, 1U> &tau);

            static void xorgqr(int m, int n, int k, ::coder::array<double, 2U> &A_C, int lda,
                               const ::coder::array<double, 1U> &tau);

        } // namespace lapack
        static void mrdiv(::coder::array<double, 2U> &A_C,
                          const ::coder::array<double, 2U> &B);

        static double now();

        namespace reflapack {
            static void xzgetrf(int m, int n, ::coder::array<double, 2U> &A_C, int lda,
                                ::coder::array<int, 2U> &ipiv, int *info);

            static void xzlarf(int m, int n, int iv0, double tau,
                               ::coder::array<double, 2U> &C, int ic0, int ldc,
                               ::coder::array<double, 1U> &work);

            static double xzlarfg(int n, double *alpha1, ::coder::array<double, 2U> &x,
                                  int ix0);

        } // namespace reflapack
    } // namespace internal
    static void normrnd(double sigma, double varargin_1,
                        ::coder::array<double, 1U> &r);

    static void normrnd(double varargin_1, double Abar,
                        ::coder::array<double, 2U> &r);

    static void planerot(double x[2], double G[4]);

    static void qr(const ::coder::array<double, 2U> &A_C,
                   ::coder::array<double, 2U> &Q, ::coder::array<double, 2U> &R_C);

    static void randi(const double lowhigh[2], double varargin_1,
                      ::coder::array<double, 1U> &r);

    static void randn(const double varargin_1[2], ::coder::array<double, 2U> &r);

    static void rng();

} // namespace coder
static void eml_rand_mt19937ar_stateful_init();

static double rt_hypotd_snf(double u0, double u1);

static double rt_powd_snf(double u0, double u1);

// Function Definitions
//
//
namespace coder {
    static void b_rand(double varargin_1, ::coder::array<double, 1U> &r) {
        unsigned int u[2];
        int i;
        i = static_cast<int>(varargin_1);
        r.set_size(i);
        for (int k{0}; k < i; k++) {
            double b_r;
            // ========================= COPYRIGHT NOTICE ============================
            //  This is a uniform (0,1) pseudorandom number generator based on:
            //
            //  A_C C-program for MT19937, with initialization improved 2002/1/26.
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
            //  A_C PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
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
                genrand_uint32_vector(state, u);
                u[0] >>= 5U;
                u[1] >>= 6U;
                b_r = 1.1102230246251565E-16 * (static_cast<double>(u[0]) * 6.7108864E+7 +
                                                static_cast<double>(u[1]));
            } while (b_r == 0.0);
            r[k] = b_r;
        }
    }

//
//
    static double det(const ::coder::array<double, 2U> &x) {
        array<double, 2U> b_x;
        array<int, 2U> ipiv;
        double y_C;
        int loop_ub;
        if ((x.size(0) == 0) || (x.size(1) == 0)) {
            y_C = 1.0;
        } else {
            int i;
            boolean_T isodd;
            b_x.set_size(x.size(0), x.size(1));
            loop_ub = x.size(0) * x.size(1);
            for (i = 0; i < loop_ub; i++) {
                b_x[i] = x[i];
            }
            internal::reflapack::xzgetrf(x.size(0), x.size(1), b_x, x.size(0), ipiv,
                                         &loop_ub);
            y_C = b_x[0];
            i = b_x.size(0);
            for (loop_ub = 0; loop_ub <= i - 2; loop_ub++) {
                y_C *= b_x[(loop_ub + b_x.size(0) * (loop_ub + 1)) + 1];
            }
            isodd = false;
            i = ipiv.size(1);
            for (loop_ub = 0; loop_ub <= i - 2; loop_ub++) {
                if (ipiv[loop_ub] > loop_ub + 1) {
                    isodd = !isodd;
                }
            }
            if (isodd) {
                y_C = -y_C;
            }
        }
        return y_C;
    }

//
//
    static void eye(double varargin_1, ::coder::array<double, 2U> &b_I) {
        double t;
        int loop_ub;
        int m;
        if (varargin_1 < 0.0) {
            t = 0.0;
        } else {
            t = varargin_1;
        }
        m = static_cast<int>(t);
        b_I.set_size(static_cast<int>(t), static_cast<int>(t));
        loop_ub = static_cast<int>(t) * static_cast<int>(t);
        for (int i{0}; i < loop_ub; i++) {
            b_I[i] = 0.0;
        }
        if (static_cast<int>(t) > 0) {
            for (loop_ub = 0; loop_ub < m; loop_ub++) {
                b_I[loop_ub + b_I.size(0) * loop_ub] = 1.0;
            }
        }
    }

//
//
    static void genrand_uint32_vector(unsigned int mt[625], unsigned int u[2]) {
        for (int j{0}; j < 2; j++) {
            unsigned int mti;
            unsigned int y_C;
            mti = mt[624] + 1U;
            if (mti >= 625U) {
                int kk;
                for (kk = 0; kk < 227; kk++) {
                    y_C = (mt[kk] & 2147483648U) | (mt[kk + 1] & 2147483647U);
                    if ((y_C & 1U) == 0U) {
                        y_C >>= 1U;
                    } else {
                        y_C = y_C >> 1U ^ 2567483615U;
                    }
                    mt[kk] = mt[kk + 397] ^ y_C;
                }
                for (kk = 0; kk < 396; kk++) {
                    y_C = (mt[kk + 227] & 2147483648U) | (mt[kk + 228] & 2147483647U);
                    if ((y_C & 1U) == 0U) {
                        y_C >>= 1U;
                    } else {
                        y_C = y_C >> 1U ^ 2567483615U;
                    }
                    mt[kk + 227] = mt[kk] ^ y_C;
                }
                y_C = (mt[623] & 2147483648U) | (mt[0] & 2147483647U);
                if ((y_C & 1U) == 0U) {
                    y_C >>= 1U;
                } else {
                    y_C = y_C >> 1U ^ 2567483615U;
                }
                mt[623] = mt[396] ^ y_C;
                mti = 1U;
            }
            y_C = mt[static_cast<int>(mti) - 1];
            mt[624] = mti;
            y_C ^= y_C >> 11U;
            y_C ^= y_C << 7U & 2636928640U;
            y_C ^= y_C << 15U & 4022730752U;
            u[j] = y_C ^ y_C >> 18U;
        }
    }

//
//
    namespace internal {
        namespace blas {
            static void b_mtimes(const ::coder::array<double, 2U> &A_C,
                                 const ::coder::array<double, 2U> &B,
                                 ::coder::array<double, 2U> &C) {
                int inner;
                int mc;
                int nc;
                mc = A_C.size(0);
                inner = A_C.size(1);
                nc = B.size(0);
                C.set_size(A_C.size(0), B.size(0));
                for (int j{0}; j < nc; j++) {
                    int coffset;
                    int i;
                    coffset = j * mc;
                    for (i = 0; i < mc; i++) {
                        C[coffset + i] = 0.0;
                    }
                    for (int k{0}; k < inner; k++) {
                        double bkj;
                        int aoffset;
                        aoffset = k * A_C.size(0);
                        bkj = B[k * B.size(0) + j];
                        for (i = 0; i < mc; i++) {
                            int b_i;
                            b_i = coffset + i;
                            C[b_i] = C[b_i] + A_C[aoffset + i] * bkj;
                        }
                    }
                }
            }

//
//
            static void mtimes(const double A_C[4], const ::coder::array<double, 2U> &B,
                               ::coder::array<double, 2U> &C) {
                int n;
                n = B.size(1);
                C.set_size(2, B.size(1));
                for (int j{0}; j < n; j++) {
                    double s_tmp;
                    int coffset_tmp;
                    coffset_tmp = j << 1;
//                    cout<<coffset_tmp<<","<<j<<" ";
                    s_tmp = B[coffset_tmp + 1];
                    C[coffset_tmp] = A_C[0] * B[coffset_tmp] + A_C[2] * s_tmp;
                    C[coffset_tmp + 1] = A_C[1] * B[coffset_tmp] + A_C[3] * s_tmp;
                }
            }
            //
//
            static void mtimes(const ::coder::array<double, 2U> &A_C,
                               const ::coder::array<double, 2U> &B,
                               ::coder::array<double, 2U> &C) {
                int inner;
                int mc;
                int nc;
                mc = A_C.size(0);
                inner = A_C.size(1);
                nc = B.size(1);
                C.set_size(A_C.size(0), B.size(1));
                for (int j{0}; j < nc; j++) {
                    int boffset;
                    int coffset;
                    int i;
                    coffset = j * mc;
                    boffset = j * B.size(0);
                    for (i = 0; i < mc; i++) {
                        C[coffset + i] = 0.0;
                    }
                    for (int k{0}; k < inner; k++) {
                        double bkj;
                        int aoffset;
                        aoffset = k * A_C.size(0);
                        bkj = B[boffset + k];
                        for (i = 0; i < mc; i++) {
                            int b_i;
                            b_i = coffset + i;
                            C[b_i] = C[b_i] + A_C[aoffset + i] * bkj;
                        }
                    }
                }
            }

//
//
            static void mtimes(const ::coder::array<double, 2U> &A_C,
                               const ::coder::array<double, 1U> &B,
                               ::coder::array<double, 1U> &C) {
                int i;
                int inner;
                int mc;
                mc = A_C.size(0) - 1;
                inner = A_C.size(1);
                C.set_size(A_C.size(0));
                for (i = 0; i <= mc; i++) {
                    C[i] = 0.0;
                }
                for (int k{0}; k < inner; k++) {
                    int aoffset;
                    aoffset = k * A_C.size(0);
                    for (i = 0; i <= mc; i++) {
                        C[i] = C[i] + A_C[aoffset + i] * B[k];
                    }
                }
            }

//
//
            static void xgerc(int m, int n, double alpha1, int ix0,
                              const ::coder::array<double, 1U> &y_C,
                              ::coder::array<double, 2U> &A_C, int ia0, int lda) {
                if (!(alpha1 == 0.0)) {
                    int jA;
                    jA = ia0;
                    for (int j{0}; j < n; j++) {
                        if (y_C[j] != 0.0) {
                            double temp;
                            int i;
                            temp = y_C[j] * alpha1;
                            i = m + jA;
                            for (int ijA{jA}; ijA < i; ijA++) {
                                A_C[ijA - 1] = A_C[ijA - 1] + A_C[((ix0 + ijA) - jA) - 1] * temp;
                            }
                        }
                        jA += lda;
                    }
                }
            }

//
//
            static double xnrm2(int n, const ::coder::array<double, 2U> &x, int ix0) {
                double y_C;
                y_C = 0.0;
                if (n >= 1) {
                    if (n == 1) {
                        y_C = std::abs(x[ix0 - 1]);
                    } else {
                        double scale;
                        int kend;
                        scale = 3.3121686421112381E-170;
                        kend = (ix0 + n) - 1;
                        for (int k{ix0}; k <= kend; k++) {
                            double absxk;
                            absxk = std::abs(x[k - 1]);
                            if (absxk > scale) {
                                double t;
                                t = scale / absxk;
                                y_C = y_C * t * t + 1.0;
                                scale = absxk;
                            } else {
                                double t;
                                t = absxk / scale;
                                y_C += t * t;
                            }
                        }
                        y_C = scale * std::sqrt(y_C);
                    }
                }
                return y_C;
            }

//
//
        } // namespace blas
        namespace lapack {
            static void xgeqrf(::coder::array<double, 2U> &A_C,
                               ::coder::array<double, 1U> &tau) {
                array<double, 1U> work;
                double atmp;
                int i;
                int ii;
                int m;
                int minmana;
                int minmn;
                int n;
                m = A_C.size(0);
                n = A_C.size(1);
                ii = A_C.size(0);
                minmana = A_C.size(1);
                if (ii < minmana) {
                    minmana = ii;
                }
                ii = A_C.size(0);
                minmn = A_C.size(1);
                if (ii < minmn) {
                    minmn = ii;
                }
                tau.set_size(minmana);
                for (i = 0; i < minmana; i++) {
                    tau[i] = 0.0;
                }
                if ((A_C.size(0) != 0) && (A_C.size(1) != 0) && (minmn >= 1)) {
                    int lda;
                    lda = A_C.size(0);
                    work.set_size(A_C.size(1));
                    ii = A_C.size(1);
                    for (i = 0; i < ii; i++) {
                        work[i] = 0.0;
                    }
                    for (i = 0; i < minmn; i++) {
                        double d;
                        ii = i * lda + i;
                        minmana = m - i;
                        if (i + 1 < m) {
                            atmp = A_C[ii];
                            d = reflapack::xzlarfg(minmana, &atmp, A_C, ii + 2);
                            tau[i] = d;
                            A_C[ii] = atmp;
                        } else {
                            d = 0.0;
                            tau[i] = 0.0;
                        }
                        if (i + 1 < n) {
                            atmp = A_C[ii];
                            A_C[ii] = 1.0;
                            reflapack::xzlarf(minmana, (n - i) - 1, ii + 1, d, A_C, (ii + lda) + 1,
                                              lda, work);
                            A_C[ii] = atmp;
                        }
                    }
                }
            }

//
//
            static void xorgqr(int m, int n, int k, ::coder::array<double, 2U> &A_C, int lda,
                               const ::coder::array<double, 1U> &tau) {
                array<double, 1U> work;
                if (n >= 1) {
                    int b_i;
                    int b_k;
                    int c_i;
                    int i;
                    int ia;
                    int itau;
                    i = n - 1;
                    for (b_i = k; b_i <= i; b_i++) {
                        ia = b_i * lda;
                        b_k = m - 1;
                        for (c_i = 0; c_i <= b_k; c_i++) {
                            A_C[ia + c_i] = 0.0;
                        }
                        A_C[ia + b_i] = 1.0;
                    }
                    itau = k - 1;
                    b_i = A_C.size(1);
                    work.set_size(b_i);
                    for (i = 0; i < b_i; i++) {
                        work[i] = 0.0;
                    }
                    for (c_i = k; c_i >= 1; c_i--) {
                        int iaii;
                        iaii = c_i + (c_i - 1) * lda;
                        if (c_i < n) {
                            int ic0;
                            int lastc;
                            int lastv;
                            A_C[iaii - 1] = 1.0;
                            ic0 = iaii + lda;
                            if (tau[itau] != 0.0) {
                                boolean_T exitg2;
                                lastv = (m - c_i) + 1;
                                b_i = (iaii + m) - c_i;
                                while ((lastv > 0) && (A_C[b_i - 1] == 0.0)) {
                                    lastv--;
                                    b_i--;
                                }
                                lastc = n - c_i;
                                exitg2 = false;
                                while ((!exitg2) && (lastc > 0)) {
                                    int exitg1;
                                    b_i = ic0 + (lastc - 1) * lda;
                                    ia = b_i;
                                    do {
                                        exitg1 = 0;
                                        if (ia <= (b_i + lastv) - 1) {
                                            if (A_C[ia - 1] != 0.0) {
                                                exitg1 = 1;
                                            } else {
                                                ia++;
                                            }
                                        } else {
                                            lastc--;
                                            exitg1 = 2;
                                        }
                                    } while (exitg1 == 0);
                                    if (exitg1 == 1) {
                                        exitg2 = true;
                                    }
                                }
                            } else {
                                lastv = 0;
                                lastc = 0;
                            }
                            if (lastv > 0) {
                                if (lastc != 0) {
                                    for (b_i = 0; b_i < lastc; b_i++) {
                                        work[b_i] = 0.0;
                                    }
                                    b_i = 0;
                                    i = ic0 + lda * (lastc - 1);
                                    for (int iac{ic0}; lda < 0 ? iac >= i : iac <= i; iac += lda) {
                                        double c;
                                        c = 0.0;
                                        b_k = (iac + lastv) - 1;
                                        for (ia = iac; ia <= b_k; ia++) {
                                            c += A_C[ia - 1] * A_C[((iaii + ia) - iac) - 1];
                                        }
                                        work[b_i] = work[b_i] + c;
                                        b_i++;
                                    }
                                }
                                blas::xgerc(lastv, lastc, -tau[itau], iaii, work, A_C, ic0, lda);
                            }
                        }
                        if (c_i < m) {
                            b_i = iaii + 1;
                            i = (iaii + m) - c_i;
                            for (b_k = b_i; b_k <= i; b_k++) {
                                A_C[b_k - 1] = -tau[itau] * A_C[b_k - 1];
                            }
                        }
                        A_C[iaii - 1] = 1.0 - tau[itau];
                        for (b_i = 0; b_i <= c_i - 2; b_i++) {
                            A_C[(iaii - b_i) - 2] = 0.0;
                        }
                        itau--;
                    }
                }
            }

//
//
        } // namespace lapack
        static void mrdiv(::coder::array<double, 2U> &A_C,
                          const ::coder::array<double, 2U> &B) {
            array<double, 2U> Y, b_A, b_B;
            array<double, 1U> tau, vn1, vn2, work;
            array<int, 2U> jpvt;
            double smax;
            int minmana;
            if ((A_C.size(0) == 0) || (A_C.size(1) == 0) ||
                ((B.size(0) == 0) || (B.size(1) == 0))) {
                int kBcol;
                minmana = A_C.size(0);
                A_C.set_size(A_C.size(0), B.size(0));
                kBcol = B.size(0);
                for (int i{0}; i < kBcol; i++) {
                    for (int ii{0}; ii < minmana; ii++) {
                        A_C[ii + A_C.size(0) * i] = 0.0;
                    }
                }
            } else if (B.size(0) == B.size(1)) {
                int b_i;
                int i;
                int ii;
                int j;
                int k;
                int kBcol;
                int mn;
                int n;
                int nmi;
                n = B.size(1);
                b_A.set_size(B.size(0), B.size(1));
                kBcol = B.size(0) * B.size(1);
                for (i = 0; i < kBcol; i++) {
                    b_A[i] = B[i];
                }
                reflapack::xzgetrf(B.size(1), B.size(1), b_A, B.size(1), jpvt, &minmana);
                nmi = A_C.size(0);
                if ((A_C.size(0) != 0) && (A_C.size(1) != 0)) {
                    for (j = 0; j < n; j++) {
                        minmana = nmi * j - 1;
                        mn = n * j;
                        for (k = 0; k < j; k++) {
                            kBcol = nmi * k;
                            i = k + mn;
                            if (b_A[i] != 0.0) {
                                for (b_i = 0; b_i < nmi; b_i++) {
                                    ii = (b_i + minmana) + 1;
                                    A_C[ii] = A_C[ii] - b_A[i] * A_C[b_i + kBcol];
                                }
                            }
                        }
                        smax = 1.0 / b_A[j + mn];
                        for (b_i = 0; b_i < nmi; b_i++) {
                            i = (b_i + minmana) + 1;
                            A_C[i] = smax * A_C[i];
                        }
                    }
                }
                if ((A_C.size(0) != 0) && (A_C.size(1) != 0)) {
                    for (j = n; j >= 1; j--) {
                        minmana = nmi * (j - 1) - 1;
                        mn = n * (j - 1) - 1;
                        i = j + 1;
                        for (k = i; k <= n; k++) {
                            kBcol = nmi * (k - 1);
                            ii = k + mn;
                            if (b_A[ii] != 0.0) {
                                for (b_i = 0; b_i < nmi; b_i++) {
                                    int pvt;
                                    pvt = (b_i + minmana) + 1;
                                    A_C[pvt] = A_C[pvt] - b_A[ii] * A_C[b_i + kBcol];
                                }
                            }
                        }
                    }
                }
                i = B.size(1) - 1;
                for (j = i; j >= 1; j--) {
                    ii = jpvt[j - 1];
                    if (ii != j) {
                        for (b_i = 0; b_i < nmi; b_i++) {
                            smax = A_C[b_i + A_C.size(0) * (j - 1)];
                            A_C[b_i + A_C.size(0) * (j - 1)] = A_C[b_i + A_C.size(0) * (ii - 1)];
                            A_C[b_i + A_C.size(0) * (ii - 1)] = smax;
                        }
                    }
                }
            } else {
                double d;
                int b_i;
                int i;
                int ii;
                int j;
                int k;
                int kBcol;
                int m;
                int ma;
                int minmn;
                int mn;
                int n;
                int nmi;
                int pvt;
                b_A.set_size(B.size(1), B.size(0));
                kBcol = B.size(0);
                for (i = 0; i < kBcol; i++) {
                    minmana = B.size(1);
                    for (ii = 0; ii < minmana; ii++) {
                        b_A[ii + b_A.size(0) * i] = B[i + B.size(0) * ii];
                    }
                }
                b_B.set_size(A_C.size(1), A_C.size(0));
                kBcol = A_C.size(0);
                for (i = 0; i < kBcol; i++) {
                    minmana = A_C.size(1);
                    for (ii = 0; ii < minmana; ii++) {
                        b_B[ii + b_B.size(0) * i] = A_C[i + A_C.size(0) * ii];
                    }
                }
                m = b_A.size(0);
                n = b_A.size(1);
                kBcol = b_A.size(0);
                minmana = b_A.size(1);
                if (kBcol < minmana) {
                    minmana = kBcol;
                }
                tau.set_size(minmana);
                for (i = 0; i < minmana; i++) {
                    tau[i] = 0.0;
                }
                jpvt.set_size(1, b_A.size(1));
                kBcol = b_A.size(1);
                for (i = 0; i < kBcol; i++) {
                    jpvt[i] = 0;
                }
                for (k = 0; k < n; k++) {
                    jpvt[k] = k + 1;
                }
                ma = b_A.size(0);
                kBcol = b_A.size(0);
                minmn = b_A.size(1);
                if (kBcol < minmn) {
                    minmn = kBcol;
                }
                work.set_size(b_A.size(1));
                kBcol = b_A.size(1);
                for (i = 0; i < kBcol; i++) {
                    work[i] = 0.0;
                }
                vn1.set_size(b_A.size(1));
                kBcol = b_A.size(1);
                for (i = 0; i < kBcol; i++) {
                    vn1[i] = 0.0;
                }
                vn2.set_size(b_A.size(1));
                kBcol = b_A.size(1);
                for (i = 0; i < kBcol; i++) {
                    vn2[i] = 0.0;
                }
                for (j = 0; j < n; j++) {
                    d = blas::xnrm2(m, b_A, j * ma + 1);
                    vn1[j] = d;
                    vn2[j] = d;
                }
                for (b_i = 0; b_i < minmn; b_i++) {
                    double s;
                    int ip1;
                    int mmi;
                    ip1 = b_i + 2;
                    kBcol = b_i * ma;
                    ii = kBcol + b_i;
                    nmi = n - b_i;
                    mmi = m - b_i;
                    if (nmi < 1) {
                        minmana = -1;
                    } else {
                        minmana = 0;
                        if (nmi > 1) {
                            smax = std::abs(vn1[b_i]);
                            for (k = 2; k <= nmi; k++) {
                                s = std::abs(vn1[(b_i + k) - 1]);
                                if (s > smax) {
                                    minmana = k - 1;
                                    smax = s;
                                }
                            }
                        }
                    }
                    pvt = b_i + minmana;
                    if (pvt + 1 != b_i + 1) {
                        minmana = pvt * ma;
                        for (k = 0; k < m; k++) {
                            mn = minmana + k;
                            smax = b_A[mn];
                            i = kBcol + k;
                            b_A[mn] = b_A[i];
                            b_A[i] = smax;
                        }
                        minmana = jpvt[pvt];
                        jpvt[pvt] = jpvt[b_i];
                        jpvt[b_i] = minmana;
                        vn1[pvt] = vn1[b_i];
                        vn2[pvt] = vn2[b_i];
                    }
                    if (b_i + 1 < m) {
                        smax = b_A[ii];
                        d = reflapack::xzlarfg(mmi, &smax, b_A, ii + 2);
                        tau[b_i] = d;
                        b_A[ii] = smax;
                    } else {
                        d = 0.0;
                        tau[b_i] = 0.0;
                    }
                    if (b_i + 1 < n) {
                        smax = b_A[ii];
                        b_A[ii] = 1.0;
                        reflapack::xzlarf(mmi, nmi - 1, ii + 1, d, b_A, (ii + ma) + 1, ma,
                                          work);
                        b_A[ii] = smax;
                    }
                    for (j = ip1; j <= n; j++) {
                        minmana = b_i + (j - 1) * ma;
                        d = vn1[j - 1];
                        if (d != 0.0) {
                            smax = std::abs(b_A[minmana]) / d;
                            smax = 1.0 - smax * smax;
                            if (smax < 0.0) {
                                smax = 0.0;
                            }
                            s = d / vn2[j - 1];
                            s = smax * (s * s);
                            if (s <= 1.4901161193847656E-8) {
                                if (b_i + 1 < m) {
                                    d = blas::xnrm2(mmi - 1, b_A, minmana + 2);
                                    vn1[j - 1] = d;
                                    vn2[j - 1] = d;
                                } else {
                                    vn1[j - 1] = 0.0;
                                    vn2[j - 1] = 0.0;
                                }
                            } else {
                                vn1[j - 1] = d * std::sqrt(smax);
                            }
                        }
                    }
                }
                pvt = 0;
                if (b_A.size(0) < b_A.size(1)) {
                    minmn = b_A.size(0);
                    minmana = b_A.size(1);
                } else {
                    minmn = b_A.size(1);
                    minmana = b_A.size(0);
                }
                if (minmn > 0) {
                    smax = std::fmin(1.4901161193847656E-8,
                                     2.2204460492503131E-15 * static_cast<double>(minmana)) *
                           std::abs(b_A[0]);
                    while ((pvt < minmn) &&
                           (!(std::abs(b_A[pvt + b_A.size(0) * pvt]) <= smax))) {
                        pvt++;
                    }
                }
                nmi = b_B.size(1);
                Y.set_size(b_A.size(1), b_B.size(1));
                kBcol = b_A.size(1) * b_B.size(1);
                for (i = 0; i < kBcol; i++) {
                    Y[i] = 0.0;
                }
                m = b_A.size(0);
                minmana = b_B.size(1);
                kBcol = b_A.size(0);
                mn = b_A.size(1);
                if (kBcol < mn) {
                    mn = kBcol;
                }
                for (j = 0; j < mn; j++) {
                    if (tau[j] != 0.0) {
                        for (k = 0; k < minmana; k++) {
                            smax = b_B[j + b_B.size(0) * k];
                            i = j + 2;
                            for (b_i = i; b_i <= m; b_i++) {
                                smax += b_A[(b_i + b_A.size(0) * j) - 1] *
                                        b_B[(b_i + b_B.size(0) * k) - 1];
                            }
                            smax *= tau[j];
                            if (smax != 0.0) {
                                b_B[j + b_B.size(0) * k] = b_B[j + b_B.size(0) * k] - smax;
                                for (b_i = i; b_i <= m; b_i++) {
                                    b_B[(b_i + b_B.size(0) * k) - 1] =
                                            b_B[(b_i + b_B.size(0) * k) - 1] -
                                            b_A[(b_i + b_A.size(0) * j) - 1] * smax;
                                }
                            }
                        }
                    }
                }
                for (k = 0; k < nmi; k++) {
                    for (b_i = 0; b_i < pvt; b_i++) {
                        Y[(jpvt[b_i] + Y.size(0) * k) - 1] = b_B[b_i + b_B.size(0) * k];
                    }
                    for (j = pvt; j >= 1; j--) {
                        i = jpvt[j - 1];
                        Y[(i + Y.size(0) * k) - 1] =
                                Y[(i + Y.size(0) * k) - 1] / b_A[(j + b_A.size(0) * (j - 1)) - 1];
                        for (b_i = 0; b_i <= j - 2; b_i++) {
                            Y[(jpvt[b_i] + Y.size(0) * k) - 1] =
                                    Y[(jpvt[b_i] + Y.size(0) * k) - 1] -
                                    Y[(jpvt[j - 1] + Y.size(0) * k) - 1] *
                                    b_A[b_i + b_A.size(0) * (j - 1)];
                        }
                    }
                }
                A_C.set_size(Y.size(1), Y.size(0));
                kBcol = Y.size(0);
                for (i = 0; i < kBcol; i++) {
                    minmana = Y.size(1);
                    for (ii = 0; ii < minmana; ii++) {
                        A_C[ii + A_C.size(0) * i] = Y[i + Y.size(0) * ii];
                    }
                }
            }
        }

//
//
        static double now() {
            time_t rawtime;
            struct tm expl_temp;
            double dDateNum;
            short cDaysMonthWise[12];
            cDaysMonthWise[0] = 0;
            cDaysMonthWise[1] = 31;
            cDaysMonthWise[2] = 59;
            cDaysMonthWise[3] = 90;
            cDaysMonthWise[4] = 120;
            cDaysMonthWise[5] = 151;
            cDaysMonthWise[6] = 181;
            cDaysMonthWise[7] = 212;
            cDaysMonthWise[8] = 243;
            cDaysMonthWise[9] = 273;
            cDaysMonthWise[10] = 304;
            cDaysMonthWise[11] = 334;
            time(&rawtime);
            expl_temp = *localtime(&rawtime);
            dDateNum =
                    ((((365.0 * static_cast<double>(expl_temp.tm_year + 1900) +
                        std::ceil(static_cast<double>(expl_temp.tm_year + 1900) / 4.0)) -
                       std::ceil(static_cast<double>(expl_temp.tm_year + 1900) / 100.0)) +
                      std::ceil(static_cast<double>(expl_temp.tm_year + 1900) / 400.0)) +
                     static_cast<double>(cDaysMonthWise[expl_temp.tm_mon])) +
                    static_cast<double>(expl_temp.tm_mday);
            if (expl_temp.tm_mon + 1 > 2) {
                int r;
                boolean_T guard1{false};
                if (expl_temp.tm_year + 1900 == 0) {
                    r = 0;
                } else {
                    r = static_cast<int>(
                            std::fmod(static_cast<double>(expl_temp.tm_year + 1900), 4.0));
                    if ((r != 0) && (expl_temp.tm_year + 1900 < 0)) {
                        r += 4;
                    }
                }
                guard1 = false;
                if (r == 0) {
                    if (expl_temp.tm_year + 1900 == 0) {
                        r = 0;
                    } else {
                        r = static_cast<int>(
                                std::fmod(static_cast<double>(expl_temp.tm_year + 1900), 100.0));
                        if ((r != 0) && (expl_temp.tm_year + 1900 < 0)) {
                            r += 100;
                        }
                    }
                    if (r != 0) {
                        dDateNum++;
                    } else {
                        guard1 = true;
                    }
                } else {
                    guard1 = true;
                }
                if (guard1) {
                    if (expl_temp.tm_year + 1900 == 0) {
                        r = 0;
                    } else {
                        r = static_cast<int>(
                                std::fmod(static_cast<double>(expl_temp.tm_year + 1900), 400.0));
                        if ((r != 0) && (expl_temp.tm_year + 1900 < 0)) {
                            r += 400;
                        }
                    }
                    if (r == 0) {
                        dDateNum++;
                    }
                }
            }
            return dDateNum + ((static_cast<double>(expl_temp.tm_hour) * 3600.0 +
                                static_cast<double>(expl_temp.tm_min) * 60.0) +
                               static_cast<double>(expl_temp.tm_sec)) /
                              86400.0;
        }

//
//
        namespace reflapack {
            static void xzgetrf(int m, int n, ::coder::array<double, 2U> &A_C, int lda,
                                ::coder::array<int, 2U> &ipiv, int *info) {
                int b_n;
                int k;
                int yk;
                if (m < n) {
                    yk = m;
                } else {
                    yk = n;
                }
                if (yk < 1) {
                    b_n = 0;
                } else {
                    b_n = yk;
                }
                ipiv.set_size(1, b_n);
                if (b_n > 0) {
                    ipiv[0] = 1;
                    yk = 1;
                    for (k = 2; k <= b_n; k++) {
                        yk++;
                        ipiv[k - 1] = yk;
                    }
                }
                *info = 0;
                if ((m >= 1) && (n >= 1)) {
                    int u0;
                    u0 = m - 1;
                    if (u0 >= n) {
                        u0 = n;
                    }
                    for (int j{0}; j < u0; j++) {
                        double smax;
                        int b_tmp;
                        int i;
                        int ipiv_tmp;
                        int jA;
                        int jp1j;
                        int mmj;
                        mmj = m - j;
                        b_tmp = j * (lda + 1);
                        jp1j = b_tmp + 2;
                        if (mmj < 1) {
                            yk = -1;
                        } else {
                            yk = 0;
                            if (mmj > 1) {
                                smax = std::abs(A_C[b_tmp]);
                                for (k = 2; k <= mmj; k++) {
                                    double s;
                                    s = std::abs(A_C[(b_tmp + k) - 1]);
                                    if (s > smax) {
                                        yk = k - 1;
                                        smax = s;
                                    }
                                }
                            }
                        }
                        if (A_C[b_tmp + yk] != 0.0) {
                            if (yk != 0) {
                                ipiv_tmp = j + yk;
                                ipiv[j] = ipiv_tmp + 1;
                                for (k = 0; k < n; k++) {
                                    yk = k * lda;
                                    jA = j + yk;
                                    smax = A_C[jA];
                                    i = ipiv_tmp + yk;
                                    A_C[jA] = A_C[i];
                                    A_C[i] = smax;
                                }
                            }
                            i = b_tmp + mmj;
                            for (yk = jp1j; yk <= i; yk++) {
                                A_C[yk - 1] = A_C[yk - 1] / A_C[b_tmp];
                            }
                        } else {
                            *info = j + 1;
                        }
                        b_n = n - j;
                        ipiv_tmp = b_tmp + lda;
                        jA = ipiv_tmp;
                        for (k = 0; k <= b_n - 2; k++) {
                            yk = ipiv_tmp + k * lda;
                            smax = A_C[yk];
                            if (A_C[yk] != 0.0) {
                                i = jA + 2;
                                yk = mmj + jA;
                                for (jp1j = i; jp1j <= yk; jp1j++) {
                                    A_C[jp1j - 1] = A_C[jp1j - 1] + A_C[((b_tmp + jp1j) - jA) - 1] * -smax;
                                }
                            }
                            jA += lda;
                        }
                    }
                    if ((*info == 0) && (m <= n) &&
                        (!(A_C[(m + A_C.size(0) * (m - 1)) - 1] != 0.0))) {
                        *info = m;
                    }
                }
            }

//
//
            static void xzlarf(int m, int n, int iv0, double tau,
                               ::coder::array<double, 2U> &C, int ic0, int ldc,
                               ::coder::array<double, 1U> &work) {
                int i;
                int ia;
                int lastc;
                int lastv;
                if (tau != 0.0) {
                    boolean_T exitg2;
                    lastv = m;
                    i = iv0 + m;
                    while ((lastv > 0) && (C[i - 2] == 0.0)) {
                        lastv--;
                        i--;
                    }
                    lastc = n;
                    exitg2 = false;
                    while ((!exitg2) && (lastc > 0)) {
                        int exitg1;
                        i = ic0 + (lastc - 1) * ldc;
                        ia = i;
                        do {
                            exitg1 = 0;
                            if (ia <= (i + lastv) - 1) {
                                if (C[ia - 1] != 0.0) {
                                    exitg1 = 1;
                                } else {
                                    ia++;
                                }
                            } else {
                                lastc--;
                                exitg1 = 2;
                            }
                        } while (exitg1 == 0);
                        if (exitg1 == 1) {
                            exitg2 = true;
                        }
                    }
                } else {
                    lastv = 0;
                    lastc = 0;
                }
                if (lastv > 0) {
                    if (lastc != 0) {
                        int iy;
                        for (iy = 0; iy < lastc; iy++) {
                            work[iy] = 0.0;
                        }
                        iy = 0;
                        i = ic0 + ldc * (lastc - 1);
                        for (int iac{ic0}; ldc < 0 ? iac >= i : iac <= i; iac += ldc) {
                            double c;
                            int b_i;
                            c = 0.0;
                            b_i = (iac + lastv) - 1;
                            for (ia = iac; ia <= b_i; ia++) {
                                c += C[ia - 1] * C[((iv0 + ia) - iac) - 1];
                            }
                            work[iy] = work[iy] + c;
                            iy++;
                        }
                    }
                    blas::xgerc(lastv, lastc, -tau, iv0, work, C, ic0, ldc);
                }
            }

//
//
            static double xzlarfg(int n, double *alpha1, ::coder::array<double, 2U> &x,
                                  int ix0) {
                double tau;
                tau = 0.0;
                if (n > 0) {
                    double xnorm;
                    xnorm = blas::xnrm2(n - 1, x, ix0);
                    if (xnorm != 0.0) {
                        double beta1;
                        beta1 = rt_hypotd_snf(*alpha1, xnorm);
                        if (*alpha1 >= 0.0) {
                            beta1 = -beta1;
                        }
                        if (std::abs(beta1) < 1.0020841800044864E-292) {
                            int i;
                            int k;
                            int knt;
                            knt = -1;
                            i = (ix0 + n) - 2;
                            do {
                                knt++;
                                for (k = ix0; k <= i; k++) {
                                    x[k - 1] = 9.9792015476736E+291 * x[k - 1];
                                }
                                beta1 *= 9.9792015476736E+291;
                                *alpha1 *= 9.9792015476736E+291;
                            } while (!(std::abs(beta1) >= 1.0020841800044864E-292));
                            beta1 = rt_hypotd_snf(*alpha1, blas::xnrm2(n - 1, x, ix0));
                            if (*alpha1 >= 0.0) {
                                beta1 = -beta1;
                            }
                            tau = (beta1 - *alpha1) / beta1;
                            xnorm = 1.0 / (*alpha1 - beta1);
                            for (k = ix0; k <= i; k++) {
                                x[k - 1] = xnorm * x[k - 1];
                            }
                            for (k = 0; k <= knt; k++) {
                                beta1 *= 1.0020841800044864E-292;
                            }
                            *alpha1 = beta1;
                        } else {
                            int i;
                            tau = (beta1 - *alpha1) / beta1;
                            xnorm = 1.0 / (*alpha1 - beta1);
                            i = (ix0 + n) - 2;
                            for (int k{ix0}; k <= i; k++) {
                                x[k - 1] = xnorm * x[k - 1];
                            }
                            *alpha1 = beta1;
                        }
                    }
                }
                return tau;
            }

//
//
        } // namespace reflapack
    } // namespace internal
    static void normrnd(double sigma, double varargin_1, ::coder::array<double, 1U> &r) {
        array<double, 2U> c_r;
        array<double, 2U> d_r;
        double b_r[2];
        int i;
        int loop_ub;
        b_r[0] = static_cast<int>(varargin_1);
        b_r[1] = 1.0;
        randn(b_r, c_r);
        d_r.set_size(c_r.size(0), c_r.size(1));
        loop_ub = c_r.size(0) * c_r.size(1);
        for (i = 0; i < loop_ub; i++) {
            d_r[i] = c_r[i];
        }
        loop_ub = static_cast<int>(varargin_1);
        r.set_size(static_cast<int>(varargin_1));
        for (i = 0; i < loop_ub; i++) {
            r[i] = d_r[i];
        }
        loop_ub = r.size(0);
        for (i = 0; i < loop_ub; i++) {
            r[i] = r[i] * sigma;
        }
        if (sigma < 0.0) {
            loop_ub = r.size(0);
            r.set_size(loop_ub);
            for (i = 0; i < loop_ub; i++) {
                r[i] = NAN;
            }
        }
    }

//
//
    static void normrnd(double varargin_1, double Abar,
                        ::coder::array<double, 2U> &r) {
        double dv[2];
        int loop_ub;
        dv[0] = static_cast<unsigned int>(static_cast<int>(varargin_1));
        dv[1] = static_cast<unsigned int>(static_cast<int>(Abar));
        randn(dv, r);
        loop_ub = r.size(0) * r.size(1);
        for (int i{0}; i < loop_ub; i++) {
            r[i] = r[i] * 0.70710678118654757;
        }
    }

//
//
    static void planerot(double x[2], double G[4]) {
        if (x[1] != 0.0) {
            double absxk;
            double r;
            double scale;
            double t;
            scale = 3.3121686421112381E-170;
            absxk = std::abs(x[0]);
            if (absxk > 3.3121686421112381E-170) {
                r = 1.0;
                scale = absxk;
            } else {
                t = absxk / 3.3121686421112381E-170;
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

//
//
    static void qr(const ::coder::array<double, 2U> &A_C,
                   ::coder::array<double, 2U> &Q, ::coder::array<double, 2U> &R_C) {
        array<double, 2U> b_A;
        array<double, 1U> tau;
        int m;
        int n;
        m = A_C.size(0) - 1;
        n = A_C.size(1);
        Q.set_size(A_C.size(0), A_C.size(0));
        R_C.set_size(A_C.size(0), A_C.size(1));
        if (A_C.size(0) > A_C.size(1)) {
            int b_i;
            int i;
            int loop_ub;
            for (loop_ub = 0; loop_ub < n; loop_ub++) {
                for (b_i = 0; b_i <= m; b_i++) {
                    Q[b_i + Q.size(0) * loop_ub] = A_C[b_i + A_C.size(0) * loop_ub];
                }
            }
            i = A_C.size(1) + 1;
            for (loop_ub = i; loop_ub <= m + 1; loop_ub++) {
                for (b_i = 0; b_i <= m; b_i++) {
                    Q[b_i + Q.size(0) * (loop_ub - 1)] = 0.0;
                }
            }
            internal::lapack::xgeqrf(Q, tau);
            for (loop_ub = 0; loop_ub < n; loop_ub++) {
                for (b_i = 0; b_i <= loop_ub; b_i++) {
                    R_C[b_i + R_C.size(0) * loop_ub] = Q[b_i + Q.size(0) * loop_ub];
                }
                i = loop_ub + 2;
                for (b_i = i; b_i <= m + 1; b_i++) {
                    R_C[(b_i + R_C.size(0) * loop_ub) - 1] = 0.0;
                }
            }
            internal::lapack::xorgqr(A_C.size(0), A_C.size(0), A_C.size(1), Q, A_C.size(0),
                                     tau);
        } else {
            int b_i;
            int i;
            int loop_ub;
            b_A.set_size(A_C.size(0), A_C.size(1));
            loop_ub = A_C.size(0) * A_C.size(1);
            for (i = 0; i < loop_ub; i++) {
                b_A[i] = A_C[i];
            }
            internal::lapack::xgeqrf(b_A, tau);
            for (loop_ub = 0; loop_ub <= m; loop_ub++) {
                for (b_i = 0; b_i <= loop_ub; b_i++) {
                    R_C[b_i + R_C.size(0) * loop_ub] = b_A[b_i + b_A.size(0) * loop_ub];
                }
                i = loop_ub + 2;
                for (b_i = i; b_i <= m + 1; b_i++) {
                    R_C[(b_i + R_C.size(0) * loop_ub) - 1] = 0.0;
                }
            }
            i = A_C.size(0) + 1;
            for (loop_ub = i; loop_ub <= n; loop_ub++) {
                for (b_i = 0; b_i <= m; b_i++) {
                    R_C[b_i + R_C.size(0) * (loop_ub - 1)] =
                            b_A[b_i + b_A.size(0) * (loop_ub - 1)];
                }
            }
            internal::lapack::xorgqr(A_C.size(0), A_C.size(0), A_C.size(0), b_A, A_C.size(0),
                                     tau);
            for (loop_ub = 0; loop_ub <= m; loop_ub++) {
                for (b_i = 0; b_i <= m; b_i++) {
                    Q[b_i + Q.size(0) * loop_ub] = b_A[b_i + b_A.size(0) * loop_ub];
                }
            }
        }
    }

//
//
    static void randi(const double lowhigh[2], double varargin_1,
                      ::coder::array<double, 1U> &r) {
        double s;
        int i;
        s = (lowhigh[1] - lowhigh[0]) + 1.0;
        b_rand(varargin_1, r);
        i = r.size(0);
        for (int k{0}; k < i; k++) {
            r[k] = lowhigh[0] + std::floor(r[k] * s);
        }
    }

//
//
    static void randn(const double varargin_1[2], ::coder::array<double, 2U> &r) {
        static const double dv[257]{0.0,
                                    0.215241895984875,
                                    0.286174591792068,
                                    0.335737519214422,
                                    0.375121332878378,
                                    0.408389134611989,
                                    0.43751840220787,
                                    0.46363433679088,
                                    0.487443966139235,
                                    0.50942332960209,
                                    0.529909720661557,
                                    0.549151702327164,
                                    0.567338257053817,
                                    0.584616766106378,
                                    0.601104617755991,
                                    0.61689699000775,
                                    0.63207223638606,
                                    0.646695714894993,
                                    0.660822574244419,
                                    0.674499822837293,
                                    0.687767892795788,
                                    0.700661841106814,
                                    0.713212285190975,
                                    0.725446140909999,
                                    0.737387211434295,
                                    0.749056662017815,
                                    0.760473406430107,
                                    0.771654424224568,
                                    0.782615023307232,
                                    0.793369058840623,
                                    0.80392911698997,
                                    0.814306670135215,
                                    0.824512208752291,
                                    0.834555354086381,
                                    0.844444954909153,
                                    0.854189171008163,
                                    0.863795545553308,
                                    0.87327106808886,
                                    0.882622229585165,
                                    0.891855070732941,
                                    0.900975224461221,
                                    0.909987953496718,
                                    0.91889818364959,
                                    0.927710533401999,
                                    0.936429340286575,
                                    0.945058684468165,
                                    0.953602409881086,
                                    0.96206414322304,
                                    0.970447311064224,
                                    0.978755155294224,
                                    0.986990747099062,
                                    0.99515699963509,
                                    1.00325667954467,
                                    1.01129241744,
                                    1.01926671746548,
                                    1.02718196603564,
                                    1.03504043983344,
                                    1.04284431314415,
                                    1.05059566459093,
                                    1.05829648333067,
                                    1.06594867476212,
                                    1.07355406579244,
                                    1.0811144097034,
                                    1.08863139065398,
                                    1.09610662785202,
                                    1.10354167942464,
                                    1.11093804601357,
                                    1.11829717411934,
                                    1.12562045921553,
                                    1.13290924865253,
                                    1.14016484436815,
                                    1.14738850542085,
                                    1.15458145035993,
                                    1.16174485944561,
                                    1.16887987673083,
                                    1.17598761201545,
                                    1.18306914268269,
                                    1.19012551542669,
                                    1.19715774787944,
                                    1.20416683014438,
                                    1.2111537262437,
                                    1.21811937548548,
                                    1.22506469375653,
                                    1.23199057474614,
                                    1.23889789110569,
                                    1.24578749554863,
                                    1.2526602218949,
                                    1.25951688606371,
                                    1.26635828701823,
                                    1.27318520766536,
                                    1.27999841571382,
                                    1.28679866449324,
                                    1.29358669373695,
                                    1.30036323033084,
                                    1.30712898903073,
                                    1.31388467315022,
                                    1.32063097522106,
                                    1.32736857762793,
                                    1.33409815321936,
                                    1.3408203658964,
                                    1.34753587118059,
                                    1.35424531676263,
                                    1.36094934303328,
                                    1.36764858359748,
                                    1.37434366577317,
                                    1.38103521107586,
                                    1.38772383568998,
                                    1.39441015092814,
                                    1.40109476367925,
                                    1.4077782768464,
                                    1.41446128977547,
                                    1.42114439867531,
                                    1.42782819703026,
                                    1.43451327600589,
                                    1.44120022484872,
                                    1.44788963128058,
                                    1.45458208188841,
                                    1.46127816251028,
                                    1.46797845861808,
                                    1.47468355569786,
                                    1.48139403962819,
                                    1.48811049705745,
                                    1.49483351578049,
                                    1.50156368511546,
                                    1.50830159628131,
                                    1.51504784277671,
                                    1.521803020761,
                                    1.52856772943771,
                                    1.53534257144151,
                                    1.542128153229,
                                    1.54892508547417,
                                    1.55573398346918,
                                    1.56255546753104,
                                    1.56939016341512,
                                    1.57623870273591,
                                    1.58310172339603,
                                    1.58997987002419,
                                    1.59687379442279,
                                    1.60378415602609,
                                    1.61071162236983,
                                    1.61765686957301,
                                    1.62462058283303,
                                    1.63160345693487,
                                    1.63860619677555,
                                    1.64562951790478,
                                    1.65267414708306,
                                    1.65974082285818,
                                    1.66683029616166,
                                    1.67394333092612,
                                    1.68108070472517,
                                    1.68824320943719,
                                    1.69543165193456,
                                    1.70264685479992,
                                    1.7098896570713,
                                    1.71716091501782,
                                    1.72446150294804,
                                    1.73179231405296,
                                    1.73915426128591,
                                    1.74654827828172,
                                    1.75397532031767,
                                    1.76143636531891,
                                    1.76893241491127,
                                    1.77646449552452,
                                    1.78403365954944,
                                    1.79164098655216,
                                    1.79928758454972,
                                    1.80697459135082,
                                    1.81470317596628,
                                    1.82247454009388,
                                    1.83028991968276,
                                    1.83815058658281,
                                    1.84605785028518,
                                    1.8540130597602,
                                    1.86201760539967,
                                    1.87007292107127,
                                    1.878180486293,
                                    1.88634182853678,
                                    1.8945585256707,
                                    1.90283220855043,
                                    1.91116456377125,
                                    1.91955733659319,
                                    1.92801233405266,
                                    1.93653142827569,
                                    1.94511656000868,
                                    1.95376974238465,
                                    1.96249306494436,
                                    1.97128869793366,
                                    1.98015889690048,
                                    1.98910600761744,
                                    1.99813247135842,
                                    2.00724083056053,
                                    2.0164337349062,
                                    2.02571394786385,
                                    2.03508435372962,
                                    2.04454796521753,
                                    2.05410793165065,
                                    2.06376754781173,
                                    2.07353026351874,
                                    2.0833996939983,
                                    2.09337963113879,
                                    2.10347405571488,
                                    2.11368715068665,
                                    2.12402331568952,
                                    2.13448718284602,
                                    2.14508363404789,
                                    2.15581781987674,
                                    2.16669518035431,
                                    2.17772146774029,
                                    2.18890277162636,
                                    2.20024554661128,
                                    2.21175664288416,
                                    2.22344334009251,
                                    2.23531338492992,
                                    2.24737503294739,
                                    2.25963709517379,
                                    2.27210899022838,
                                    2.28480080272449,
                                    2.29772334890286,
                                    2.31088825060137,
                                    2.32430801887113,
                                    2.33799614879653,
                                    2.35196722737914,
                                    2.36623705671729,
                                    2.38082279517208,
                                    2.39574311978193,
                                    2.41101841390112,
                                    2.42667098493715,
                                    2.44272531820036,
                                    2.4592083743347,
                                    2.47614993967052,
                                    2.49358304127105,
                                    2.51154444162669,
                                    2.53007523215985,
                                    2.54922155032478,
                                    2.56903545268184,
                                    2.58957598670829,
                                    2.61091051848882,
                                    2.63311639363158,
                                    2.65628303757674,
                                    2.68051464328574,
                                    2.70593365612306,
                                    2.73268535904401,
                                    2.76094400527999,
                                    2.79092117400193,
                                    2.82287739682644,
                                    2.85713873087322,
                                    2.89412105361341,
                                    2.93436686720889,
                                    2.97860327988184,
                                    3.02783779176959,
                                    3.08352613200214,
                                    3.147889289518,
                                    3.2245750520478,
                                    3.32024473383983,
                                    3.44927829856143,
                                    3.65415288536101,
                                    3.91075795952492};
        static const double dv1[257]{1.0,
                                     0.977101701267673,
                                     0.959879091800108,
                                     0.9451989534423,
                                     0.932060075959231,
                                     0.919991505039348,
                                     0.908726440052131,
                                     0.898095921898344,
                                     0.887984660755834,
                                     0.878309655808918,
                                     0.869008688036857,
                                     0.860033621196332,
                                     0.851346258458678,
                                     0.842915653112205,
                                     0.834716292986884,
                                     0.826726833946222,
                                     0.818929191603703,
                                     0.811307874312656,
                                     0.803849483170964,
                                     0.796542330422959,
                                     0.789376143566025,
                                     0.782341832654803,
                                     0.775431304981187,
                                     0.768637315798486,
                                     0.761953346836795,
                                     0.755373506507096,
                                     0.748892447219157,
                                     0.742505296340151,
                                     0.736207598126863,
                                     0.729995264561476,
                                     0.72386453346863,
                                     0.717811932630722,
                                     0.711834248878248,
                                     0.705928501332754,
                                     0.700091918136512,
                                     0.694321916126117,
                                     0.688616083004672,
                                     0.682972161644995,
                                     0.677388036218774,
                                     0.671861719897082,
                                     0.66639134390875,
                                     0.660975147776663,
                                     0.655611470579697,
                                     0.650298743110817,
                                     0.645035480820822,
                                     0.639820277453057,
                                     0.634651799287624,
                                     0.629528779924837,
                                     0.624450015547027,
                                     0.619414360605834,
                                     0.614420723888914,
                                     0.609468064925773,
                                     0.604555390697468,
                                     0.599681752619125,
                                     0.594846243767987,
                                     0.590047996332826,
                                     0.585286179263371,
                                     0.580559996100791,
                                     0.575868682972354,
                                     0.571211506735253,
                                     0.566587763256165,
                                     0.561996775814525,
                                     0.557437893618766,
                                     0.552910490425833,
                                     0.548413963255266,
                                     0.543947731190026,
                                     0.539511234256952,
                                     0.535103932380458,
                                     0.530725304403662,
                                     0.526374847171684,
                                     0.522052074672322,
                                     0.517756517229756,
                                     0.513487720747327,
                                     0.509245245995748,
                                     0.505028667943468,
                                     0.500837575126149,
                                     0.49667156905249,
                                     0.492530263643869,
                                     0.488413284705458,
                                     0.484320269426683,
                                     0.480250865909047,
                                     0.476204732719506,
                                     0.47218153846773,
                                     0.468180961405694,
                                     0.464202689048174,
                                     0.460246417812843,
                                     0.456311852678716,
                                     0.452398706861849,
                                     0.448506701507203,
                                     0.444635565395739,
                                     0.440785034665804,
                                     0.436954852547985,
                                     0.433144769112652,
                                     0.429354541029442,
                                     0.425583931338022,
                                     0.421832709229496,
                                     0.418100649837848,
                                     0.414387534040891,
                                     0.410693148270188,
                                     0.407017284329473,
                                     0.403359739221114,
                                     0.399720314980197,
                                     0.396098818515832,
                                     0.392495061459315,
                                     0.388908860018789,
                                     0.385340034840077,
                                     0.381788410873393,
                                     0.378253817245619,
                                     0.374736087137891,
                                     0.371235057668239,
                                     0.367750569779032,
                                     0.364282468129004,
                                     0.360830600989648,
                                     0.357394820145781,
                                     0.353974980800077,
                                     0.350570941481406,
                                     0.347182563956794,
                                     0.343809713146851,
                                     0.340452257044522,
                                     0.337110066637006,
                                     0.333783015830718,
                                     0.330470981379163,
                                     0.327173842813601,
                                     0.323891482376391,
                                     0.320623784956905,
                                     0.317370638029914,
                                     0.314131931596337,
                                     0.310907558126286,
                                     0.307697412504292,
                                     0.30450139197665,
                                     0.301319396100803,
                                     0.298151326696685,
                                     0.294997087799962,
                                     0.291856585617095,
                                     0.288729728482183,
                                     0.285616426815502,
                                     0.282516593083708,
                                     0.279430141761638,
                                     0.276356989295668,
                                     0.273297054068577,
                                     0.270250256365875,
                                     0.267216518343561,
                                     0.264195763997261,
                                     0.261187919132721,
                                     0.258192911337619,
                                     0.255210669954662,
                                     0.252241126055942,
                                     0.249284212418529,
                                     0.246339863501264,
                                     0.24340801542275,
                                     0.240488605940501,
                                     0.237581574431238,
                                     0.23468686187233,
                                     0.231804410824339,
                                     0.228934165414681,
                                     0.226076071322381,
                                     0.223230075763918,
                                     0.220396127480152,
                                     0.217574176724331,
                                     0.214764175251174,
                                     0.211966076307031,
                                     0.209179834621125,
                                     0.206405406397881,
                                     0.203642749310335,
                                     0.200891822494657,
                                     0.198152586545776,
                                     0.195425003514135,
                                     0.192709036903589,
                                     0.190004651670465,
                                     0.187311814223801,
                                     0.1846304924268,
                                     0.181960655599523,
                                     0.179302274522848,
                                     0.176655321443735,
                                     0.174019770081839,
                                     0.171395595637506,
                                     0.168782774801212,
                                     0.166181285764482,
                                     0.163591108232366,
                                     0.161012223437511,
                                     0.158444614155925,
                                     0.15588826472448,
                                     0.153343161060263,
                                     0.150809290681846,
                                     0.148286642732575,
                                     0.145775208005994,
                                     0.143274978973514,
                                     0.140785949814445,
                                     0.138308116448551,
                                     0.135841476571254,
                                     0.133386029691669,
                                     0.130941777173644,
                                     0.12850872228,
                                     0.126086870220186,
                                     0.123676228201597,
                                     0.12127680548479,
                                     0.11888861344291,
                                     0.116511665625611,
                                     0.114145977827839,
                                     0.111791568163838,
                                     0.109448457146812,
                                     0.107116667774684,
                                     0.104796225622487,
                                     0.102487158941935,
                                     0.10018949876881,
                                     0.0979032790388625,
                                     0.095628536713009,
                                     0.093365311912691,
                                     0.0911136480663738,
                                     0.0888735920682759,
                                     0.0866451944505581,
                                     0.0844285095703535,
                                     0.082223595813203,
                                     0.0800305158146631,
                                     0.0778493367020961,
                                     0.0756801303589272,
                                     0.0735229737139814,
                                     0.0713779490588905,
                                     0.0692451443970068,
                                     0.0671246538277886,
                                     0.065016577971243,
                                     0.0629210244377582,
                                     0.06083810834954,
                                     0.0587679529209339,
                                     0.0567106901062031,
                                     0.0546664613248891,
                                     0.0526354182767924,
                                     0.0506177238609479,
                                     0.0486135532158687,
                                     0.0466230949019305,
                                     0.0446465522512946,
                                     0.0426841449164746,
                                     0.0407361106559411,
                                     0.0388027074045262,
                                     0.0368842156885674,
                                     0.0349809414617162,
                                     0.0330932194585786,
                                     0.0312214171919203,
                                     0.0293659397581334,
                                     0.0275272356696031,
                                     0.0257058040085489,
                                     0.0239022033057959,
                                     0.0221170627073089,
                                     0.0203510962300445,
                                     0.0186051212757247,
                                     0.0168800831525432,
                                     0.0151770883079353,
                                     0.0134974506017399,
                                     0.0118427578579079,
                                     0.0102149714397015,
                                     0.00861658276939875,
                                     0.00705087547137324,
                                     0.00552240329925101,
                                     0.00403797259336304,
                                     0.00260907274610216,
                                     0.0012602859304986,
                                     0.000477467764609386};
        unsigned int u[2];
        int i;
        r.set_size(static_cast<int>(varargin_1[0]), static_cast<int>(varargin_1[1]));
        i = static_cast<int>(varargin_1[0]) * static_cast<int>(varargin_1[1]);
        for (int k{0}; k < i; k++) {
            double b_r;
            int b_i;
            int exitg1;
            do {
                exitg1 = 0;
                genrand_uint32_vector(state, u);
                b_i = static_cast<int>((u[1] >> 24U) + 1U);
                b_r = ((static_cast<double>(u[0] >> 3U) * 1.6777216E+7 +
                        static_cast<double>(static_cast<int>(u[1]) & 16777215)) *
                       2.2204460492503131E-16 -
                       1.0) *
                      dv[b_i];
                if (std::abs(b_r) <= dv[b_i - 1]) {
                    exitg1 = 1;
                } else if (b_i < 256) {
                    double b_u;
                    // ========================= COPYRIGHT NOTICE
                    // ============================
                    //  This is a uniform (0,1) pseudorandom number generator based on:
                    //
                    //  A_C C-program for MT19937, with initialization improved 2002/1/26.
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
                    //    2. Redistributions in binary form must reproduce the above
                    //    copyright
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
                    //  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
                    //  FOR A_C PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE
                    //  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
                    //  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
                    //  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
                    //  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
                    //  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
                    //  LIABILITY, OR TORT (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN
                    //  ANY WAY OUT OF THE USE OF THIS  SOFTWARE, EVEN IF ADVISED OF THE
                    //  POSSIBILITY OF SUCH DAMAGE.
                    //
                    // =============================   END =================================
                    do {
                        genrand_uint32_vector(state, u);
                        u[0] >>= 5U;
                        u[1] >>= 6U;
                        b_u = 1.1102230246251565E-16 *
                              (static_cast<double>(u[0]) * 6.7108864E+7 +
                               static_cast<double>(u[1]));
                    } while (b_u == 0.0);
                    if (dv1[b_i] + b_u * (dv1[b_i - 1] - dv1[b_i]) <
                        std::exp(-0.5 * b_r * b_r)) {
                        exitg1 = 1;
                    }
                } else {
                    double b_u;
                    double x;
                    do {
                        // ========================= COPYRIGHT NOTICE
                        // ============================
                        //  This is a uniform (0,1) pseudorandom number generator based on:
                        //
                        //  A_C C-program for MT19937, with initialization improved 2002/1/26.
                        //  Coded by Takuji Nishimura and Makoto Matsumoto.
                        //
                        //  Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
                        //  All rights reserved.
                        //
                        //  Redistribution and use in source and binary forms, with or without
                        //  modification, are permitted provided that the following conditions
                        //  are met:
                        //
                        //    1. Redistributions of source code must retain the above
                        //    copyright
                        //       notice, this list of conditions and the following disclaimer.
                        //
                        //    2. Redistributions in binary form must reproduce the above
                        //    copyright
                        //       notice, this list of conditions and the following disclaimer
                        //       in the documentation and/or other materials provided with the
                        //       distribution.
                        //
                        //    3. The names of its contributors may not be used to endorse or
                        //       promote products derived from this software without specific
                        //       prior written permission.
                        //
                        //  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
                        //  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
                        //  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
                        //  MERCHANTABILITY AND FITNESS FOR A_C PARTICULAR PURPOSE ARE
                        //  DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
                        //  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
                        //  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
                        //  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
                        //  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
                        //  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
                        //  TORT (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
                        //  OF THE USE OF THIS  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
                        //  OF SUCH DAMAGE.
                        //
                        // =============================   END
                        // =================================
                        do {
                            genrand_uint32_vector(state, u);
                            u[0] >>= 5U;
                            u[1] >>= 6U;
                            b_u = 1.1102230246251565E-16 *
                                  (static_cast<double>(u[0]) * 6.7108864E+7 +
                                   static_cast<double>(u[1]));
                        } while (b_u == 0.0);
                        x = std::log(b_u) * 0.273661237329758;
                        // ========================= COPYRIGHT NOTICE
                        // ============================
                        //  This is a uniform (0,1) pseudorandom number generator based on:
                        //
                        //  A_C C-program for MT19937, with initialization improved 2002/1/26.
                        //  Coded by Takuji Nishimura and Makoto Matsumoto.
                        //
                        //  Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
                        //  All rights reserved.
                        //
                        //  Redistribution and use in source and binary forms, with or without
                        //  modification, are permitted provided that the following conditions
                        //  are met:
                        //
                        //    1. Redistributions of source code must retain the above
                        //    copyright
                        //       notice, this list of conditions and the following disclaimer.
                        //
                        //    2. Redistributions in binary form must reproduce the above
                        //    copyright
                        //       notice, this list of conditions and the following disclaimer
                        //       in the documentation and/or other materials provided with the
                        //       distribution.
                        //
                        //    3. The names of its contributors may not be used to endorse or
                        //       promote products derived from this software without specific
                        //       prior written permission.
                        //
                        //  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
                        //  CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
                        //  INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
                        //  MERCHANTABILITY AND FITNESS FOR A_C PARTICULAR PURPOSE ARE
                        //  DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS
                        //  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
                        //  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
                        //  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
                        //  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
                        //  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR
                        //  TORT (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
                        //  OF THE USE OF THIS  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
                        //  OF SUCH DAMAGE.
                        //
                        // =============================   END
                        // =================================
                        do {
                            genrand_uint32_vector(state, u);
                            u[0] >>= 5U;
                            u[1] >>= 6U;
                            b_u = 1.1102230246251565E-16 *
                                  (static_cast<double>(u[0]) * 6.7108864E+7 +
                                   static_cast<double>(u[1]));
                        } while (b_u == 0.0);
                    } while (!(-2.0 * std::log(b_u) > x * x));
                    if (b_r < 0.0) {
                        b_r = x - 3.65415288536101;
                    } else {
                        b_r = 3.65415288536101 - x;
                    }
                    exitg1 = 1;
                }
            } while (exitg1 == 0);
            r[k] = b_r;
        }
    }

//
//
    static void rng() {
        time_t eTime;
        double s;
        double x;
        unsigned int r;
        x = internal::now() * 8.64E+6;
        x = std::floor(x);
        if (std::isnan(x)) {
            s = NAN;
        } else if (x == 0.0) {
            s = 0.0;
        } else {
            s = std::fmod(x, 2.147483647E+9);
            if (s == 0.0) {
                s = 0.0;
            } else if (x < 0.0) {
                s += 2.147483647E+9;
            }
        }
        eTime = time(nullptr);
        time_t b_eTime;
        int exitg1;
        do {
            exitg1 = 0;
            b_eTime = time(nullptr);
            if ((int) b_eTime <= (int) eTime + 1) {
                double s0;
                x = internal::now() * 8.64E+6;
                x = std::floor(x);
                if (std::isnan(x)) {
                    s0 = NAN;
                } else if (x == 0.0) {
                    s0 = 0.0;
                } else {
                    s0 = std::fmod(x, 2.147483647E+9);
                    if (s0 == 0.0) {
                        s0 = 0.0;
                    } else if (x < 0.0) {
                        s0 += 2.147483647E+9;
                    }
                }
                if (s != s0) {
                    exitg1 = 1;
                }
            } else {
                exitg1 = 1;
            }
        } while (exitg1 == 0);
        if (s < 4.294967296E+9) {
            if (s >= 0.0) {
                r = static_cast<unsigned int>(s);
            } else {
                r = 0U;
            }
        } else {
            r = 0U;
        }
        state[0] = r;
        for (int mti{0}; mti < 623; mti++) {
            r = ((r ^ r >> 30U) * 1812433253U + mti) + 1U;
            state[mti + 1] = r;
        }
        state[624] = 624U;
    }

//
//
} // namespace coder
static void eml_rand_mt19937ar_stateful_init() {
    unsigned int r;
    std::memset(&state[0], 0, 625U * sizeof(unsigned int));
    r = 5489U;
    state[0] = 5489U;
    for (int mti{0}; mti < 623; mti++) {
        r = ((r ^ r >> 30U) * 1812433253U + mti) + 1U;
        state[mti + 1] = r;
    }
    state[624] = 624U;
}

static double rt_hypotd_snf(double u0, double u1) {
    double a;
    double y_C;
    a = std::abs(u0);
    y_C = std::abs(u1);
    if (a < y_C) {
        a /= y_C;
        y_C *= std::sqrt(a * a + 1.0);
    } else if (a > y_C) {
        y_C /= a;
        y_C = a * std::sqrt(y_C * y_C + 1.0);
    } else if (!std::isnan(y_C)) {
        y_C = a * 1.4142135623730951;
    }
    return y_C;
}

static double rt_powd_snf(double u0, double u1) {
    double y_C;
    if (std::isnan(u0) || std::isnan(u1)) {
        y_C = NAN;
    } else {
        double d;
        double d1;
        d = std::abs(u0);
        d1 = std::abs(u1);
        if (std::isinf(u1)) {
            if (d == 1.0) {
                y_C = 1.0;
            } else if (d > 1.0) {
                if (u1 > 0.0) {
                    y_C = INFINITY;
                } else {
                    y_C = 0.0;
                }
            } else if (u1 > 0.0) {
                y_C = 0.0;
            } else {
                y_C = INFINITY;
            }
        } else if (d1 == 0.0) {
            y_C = 1.0;
        } else if (d1 == 1.0) {
            if (u1 > 0.0) {
                y_C = u0;
            } else {
                y_C = 1.0 / u0;
            }
        } else if (u1 == 2.0) {
            y_C = u0 * u0;
        } else if ((u1 == 0.5) && (u0 >= 0.0)) {
            y_C = std::sqrt(u0);
        } else if ((u0 < 0.0) && (u1 > std::floor(u1))) {
            y_C = NAN;
        } else {
            y_C = std::pow(u0, u1);
        }
    }
    return y_C;
}