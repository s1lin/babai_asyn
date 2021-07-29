//
// Created by shilei on 6/23/21.
//

#ifndef CILS_HELPER_H
#define CILS_HELPER_H

#endif //CILS_HELPER_H

#include <cmath>
#include <cstring>

namespace helper {

    template<typename scalar, typename index, index m, index n>
    static void eye(scalar size, std::array<scalar, n * n> &b_I) {
        scalar t;
        index loop_ub;
        index mm;
        if (size < 0.0) {
            t = 0.0;
        } else {
            t = size;
        }
        mm = static_cast<index>(t);
        loop_ub = static_cast<index>(t) * static_cast<index>(t);
        for (index i = 0; i < loop_ub; i++) {
            b_I[i] = 0.0;
        }
        if (static_cast<index>(t) > 0) {
            for (loop_ub = 0; loop_ub < mm; loop_ub++) {
                b_I[loop_ub + size * loop_ub] = 1.0;
            }
        }
    }

    namespace internal {
        namespace blas {

            template<typename scalar, typename index, index m, index n, index mb>
            static void
            b_mtimes(const array<scalar, m * n> &A_C, const array<scalar, n * mb> &B, array<scalar, m * mb> &C) {
                C.resize(m * mb, 0);
                for (index j = 0; j < mb; j++) {
                    index coffset = j * m;
                    for (index k = 0; k < n; k++) {
                        scalar bkj = B[k * n + j];
                        index aoffset = k * m;
                        for (index i = 0; i < m; i++) {
                            index b_i = coffset + i;
                            C[b_i] = C[b_i] + A_C[aoffset + i] * bkj;
                        }
                    }
                }
            }

            //B: m x n
            template<typename scalar, typename index, index m, index n>
            static void mtimes(const scalar A_C[4], const array<scalar, m * n> &B, array<scalar, 2 * n> &C) {
                for (index j = 0; j < n; j++) {
                    index coffset_tmp = j << 1;
                    scalar s_tmp = B[coffset_tmp + 1];
                    C[coffset_tmp] = A_C[0] * B[coffset_tmp] + A_C[2] * s_tmp;
                    C[coffset_tmp + 1] = A_C[1] * B[coffset_tmp] + A_C[3] * s_tmp;
                }
            }

            template<typename scalar, typename index, index m, index n>
            static void mtimes(const array<scalar, m * m> &Q, const array<scalar, m * n> &R, array<scalar, m * n> &A_t) {
                for (index  j = 0; j < n; j++) {
                    for (index k = 0; k < m; k++) {
                        for (index i = 0; i < m; i++) {
                            A_t[j * m + i] += Q[k * m + i] * R[j * m + k];
                        }
                    }
                }
            }
        }

        template<typename scalar, typename index, index m, index n, index mb>
        static void
        mrdiv(array<scalar, m * n> &B, const array<scalar, n * mb> &A) {
            using namespace matlab::engine;

            // Start MATLAB engine synchronously
            std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();

            //Create MATLAB data array factory
            matlab::data::ArrayFactory factory;

            // Call the MATLAB movsum function
            matlab::data::TypedArray<scalar> A_M = factory.createArray(
                    {static_cast<unsigned long>(m), static_cast<unsigned long>(n)}, A.begin(), A.end());
            matlab::data::TypedArray<scalar> B_M = factory.createArray(
                    {static_cast<unsigned long>(n), static_cast<unsigned long>(mb)}, B.begin(), B.end());
            matlabPtr->setVariable(u"A", std::move(A_M));
            matlabPtr->setVariable(u"B", std::move(B_M));

            // Call the MATLAB movsum function
            matlabPtr->eval(u" R = mldivide(A, B);");

            matlab::data::TypedArray<scalar> const R = matlabPtr->getVariable(u"R");
            index i = 0;
            for (auto r : R) {
                B[i] = r;
                ++i;
            }
        }
    }

    template<typename scalar, typename index>
    static void planerot(scalar x[2], scalar G[4]) {
        if (x[1] != 0.0) {
            scalar absxk;
            scalar r;
            scalar scale;
            scalar t;
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

    template<typename scalar, typename index, index n>
    scalar b_norm(const array<scalar, n> &x) {
        scalar y;
        if (n == 0) {
            y = 0.0;
        } else {
            y = 0.0;
            if (n == 1) {
                y = std::abs(x[0]);
            } else {
                scalar scale;
                index kend;
                scale = 3.3121686421112381E-170;
                kend = n;
                for (index k = 0; k < kend; k++) {
                    scalar absxk;
                    absxk = std::abs(x[k]);
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
            }
        }
        return y;
    }

}