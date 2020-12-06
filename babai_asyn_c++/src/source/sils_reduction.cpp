#include <cmath>
#include <cstring>
#include "../include/sils.h"

//// Function Declarations
//static scalar rt_roundd_snf(scalar u);
//
//// Function Definitions
//
////
//// Arguments    : scalar u
//// Return Type  : scalar
////
//static scalar rt_roundd_snf(scalar u)
//{
//  scalar y;
//  if (std::abs(u) < 4.503599627370496E+15) {
//    if (u >= 0.5) {
//      y = std::floor(u + 0.5);
//    } else if (u > -0.5) {
//      y = u * 0.0;
//    } else {
//      y = std::ceil(u - 0.5);
//    }
//  } else {
//    y = u;
//  }
//
//  return y;
//}

//
// [R,Z,y] = sils_reduction(B,y) reduces the general standard indexeger
//  least squares problem to an upper triangular one by the LLL-QRZ
//  factorization Q'*B*Z = [R; 0]. The orthogonal matrix Q
//  is not produced.
//
//  Inputs:
//     B - m-by-n real matrix with full column rank
//     y - m-dimensional real vector to be transformed to Q'*y
//
//  Outputs:
//     R - n-by-n LLL-reduced upper triangular matrix
//     Z - n-by-n unimodular matrix, i.e., an indexeger matrix with |det(Z)|=1
//     y - m-vector transformed from the input y by Q', i.e., y := Q'*y
// Arguments    : const scalar B[10000]
//                scalar y[n]
//                scalar R[10000]
//                scalar Z[10000]
// Return Type  : void
//
namespace sils {

    template<typename scalar, typename index>
    returnType<scalar, index> qrmcp(scalar B[10000], scalar* y, scalar* piv) {
        scalar colNormB[200];
        index j;
        scalar *b;
        index k;
        scalar a;
        scalar scale;
        index n;
        index q;
        scalar absxk;
        index i;
        scalar t;
        scalar varargin_1_data[n];
        index b_k;
        bool exitg1;
        index B_size[1];
        scalar B_data[n - 1];
        index v_size_idx_0;
        index y_size_idx_1;
        scalar v_data[n];
        unsigned char b_q[2];
        scalar b_B_data[n];
        unsigned char c_k[2];
        scalar y_data[n - 1];
        scalar b_B[200];

        //  Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
        //           Xiaohu Xie, Tianyang Zhou
        //  Copyright (c) 2006-2016. Scientific Computing Lab, McGill University.
        //  October 2006; Last revision: June 2016
        //  Initialization
        std::memset(&colNormB[0], 0, 200U * sizeof(scalar));

        //  Compute the 2-norm squared of each column of B
        for (j = 0; j < n; j++) {
            piv[j] = static_cast<scalar>(j) + 1.0;
            a = 0.0;
            scale = -INFINITY;
            for (k = 0; k < n; k++) {
                absxk = std::abs(B[k + n * j]);
                if (absxk > scale) {
                    t = scale / absxk;
                    a = a * t * t + 1.0;
                    scale = absxk;
                } else {
                    t = absxk / scale;
                    a += t * t;
                }
            }

            a = scale * std::sqrt(a);
            colNormB[j << 1] = a * a;
        }

        emxInit_real_T(&b, 2);
        for (k = 0; k < n - 1; k++) {
            //  Find the column with minimum 2-norm in B(k:m,k:n)
            q = n - k;
            j = -k;
            for (i = 0; i <= j + n - 1; i++) {
                n = (k + i) << 1;
                varargin_1_data[i] = colNormB[n] - colNormB[n + 1];
            }

            if (q <= 2) {
                if ((varargin_1_data[0] > varargin_1_data[1]) || (rtIsNaN(varargin_1_data
                                                                          [0]) && (!rtIsNaN(varargin_1_data[1])))) {
                    n = 2;
                } else {
                    n = 1;
                }
            } else {
                if (!rtIsNaN(varargin_1_data[0])) {
                    n = 1;
                } else {
                    n = 0;
                    b_k = 2;
                    exitg1 = false;
                    while ((!exitg1) && (b_k <= q)) {
                        if (!rtIsNaN(varargin_1_data[b_k - 1])) {
                            n = b_k;
                            exitg1 = true;
                        } else {
                            b_k++;
                        }
                    }
                }

                if (n == 0) {
                    n = 1;
                } else {
                    t = varargin_1_data[n - 1];
                    i = n + 1;
                    for (b_k = i; b_k <= q; b_k++) {
                        scale = varargin_1_data[b_k - 1];
                        if (t > scale) {
                            t = scale;
                            n = b_k;
                        }
                    }
                }
            }

            q = (n + k) - 1;

            //  Column indexerchange
            if (q > k) {
                n = static_cast<index>(piv[k]);
                piv[k] = piv[q];
                j = q << 1;
                t = colNormB[j + 1];
                piv[q] = n;
                n = k << 1;
                scale = colNormB[n];
                absxk = colNormB[n + 1];
                colNormB[n] = colNormB[j];
                colNormB[n + 1] = t;
                colNormB[j] = scale;
                colNormB[j + 1] = absxk;
                b_q[0] = static_cast<unsigned char>(q);
                b_q[1] = static_cast<unsigned char>(k);
                c_k[0] = static_cast<unsigned char>(k);
                c_k[1] = static_cast<unsigned char>(q);
                for (i = 0; i < 2; i++) {
                    for (q = 0; q < n; q++) {
                        b_B[q + n * i] = B[q + n * b_q[i]];
                    }
                }

                for (i = 0; i < 2; i++) {
                    for (q = 0; q < n; q++) {
                        B[q + n * c_k[i]] = b_B[q + n * i];
                    }
                }
            }

            //  Compute and apply the Householder transformation  I-tau*v*v'
            B_size[0] = n - 1 - k;
            j = n - 1 - k;
            if (0 <= j - 1) {
                std::memcpy(&B_data[0], &B[k * 101 + 1], j * sizeof(scalar));
            }

            if (b_norm(B_data, B_size) > 0.0) {
                //  A Householder transformation is needed
                v_size_idx_0 = n - k;
                j = -k;
                B_size[0] = n - k;
                if (0 <= j + n - 1) {
                    std::memcpy(&v_data[0], &B[k * 101], (j + n) * sizeof(scalar));
                    std::memcpy(&b_B_data[0], &B[k * 101], (j + n) * sizeof(scalar));
                }

                t = b_norm(b_B_data, B_size);
                i = k + n * k;
                if (B[i] >= 0.0) {
                    t = -t;
                }

                scale = B[i] - t;
                v_data[0] = scale;

                //  B(k,k)+sgn(B(k,k))*norm(B(k:n,k))
                a = -1.0 / (t * scale);
                B[i] = t;
                for (i = 0; i < v_size_idx_0; i++) {
                    b_B_data[i] = a * v_data[i];
                }

                i = b->size[0] * b->size[1];
                b->size[0] = n - k;
                b->size[1] = n - 1 - k;
                emxEnsureCapacity_real_T(b, i);
                j = n - 1 - k;
                for (i = 0; i < j; i++) {
                    n = -k;
                    for (q = 0; q <= n + n - 1; q++) {
                        b->data[q + b->size[0] * i] = B[(k + q) + n * ((k + i) + 1)];
                    }
                }

                n = 98 - k;
                y_size_idx_1 = n - 1 - k;
                for (j = 0; j <= n; j++) {
                    q = j * v_size_idx_0;
                    y_data[j] = 0.0;
                    for (b_k = 0; b_k < v_size_idx_0; b_k++) {
                        y_data[j] += b->data[q + b_k] * v_data[b_k];
                    }
                }

                i = b->size[0] * b->size[1];
                b->size[0] = v_size_idx_0;
                b->size[1] = y_size_idx_1;
                emxEnsureCapacity_real_T(b, i);
                for (i = 0; i < v_size_idx_0; i++) {
                    for (q = 0; q < y_size_idx_1; q++) {
                        b->data[i + b->size[0] * q] = B[(k + i) + n * ((k + q) + 1)] -
                                                      b_B_data[i] * y_data[q];
                    }
                }

                j = b->size[1];
                for (i = 0; i < j; i++) {
                    n = b->size[0];
                    for (q = 0; q < n; q++) {
                        B[(k + q) + n * ((k + i) + 1)] = b->data[q + b->size[0] * i];
                    }
                }

                //  Update y by the Householder transformation
                t = 0.0;
                for (i = 0; i < v_size_idx_0; i++) {
                    t += v_data[i] * y[k + i];
                }

                q = n - k;
                j = n - k;
                for (i = 0; i < j; i++) {
                    varargin_1_data[i] = y[k + i] - b_B_data[i] * t;
                }

                if (0 <= q - 1) {
                    std::memcpy(&y[k], &varargin_1_data[0], q * sizeof(scalar));
                }
            }

            //  Update colnormB(2,k+1:n)
            y_size_idx_1 = n - 1 - k;
            j = n - 1 - k;
            for (i = 0; i < j; i++) {
                n = (k + i) + 1;
                q = k + n * n;
                y_data[i] = colNormB[(n << 1) + 1] + B[q] * B[q];
            }

            for (i = 0; i < y_size_idx_1; i++) {
                colNormB[(((k + i) + 1) << 1) + 1] = y_data[i];
            }
        }

        emxFree_real_T(&b);
        n = 2;
        for (j = 0; j < n - 1; j++) {
            if (n <= n) {
                std::memset(&B[(j * n + n) + -1], 0, (101 - n) * sizeof(scalar));
            }

            n++;
        }
    }

    template<typename scalar, typename index>
    returnType<scalar, index> sils_reduction(const scalar B[10000], scalar y[n], scalar R[10000],
                                             scalar Z[10000]) {
        scalar piv[n];
        index j;
        index k;
        index zeta_tmp_tmp;
        index b_zeta_tmp_tmp;
        scalar zeta;
        scalar alpha;
        index i;
        signed char iv[2];
        index n;
        index b_k[2];
        index R_size_idx_0;
        scalar R_data[98];
        scalar b_R_data[200];
        index c_k[2];
        index tmp_size_idx_1;
        scalar r;
        scalar b_Z[200];
        scalar t;
        scalar G_idx_0;
        scalar G_idx_3;
        scalar b_data[198];
        scalar tmp_data[198];

        //  Subfunction: qrmcp
        //  Main Reference:
        //  X. Xie, X.-W. Chang, and M. Al Borno. Partial LLL Reduction,
        //  Proceedings of IEEE GLOBECOM 2011, 5 pages.
        //  Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
        //           Xiaohu Xie, Tianyang Zhou
        //  Copyright (c) 2006-2016. Scientific Computing Lab, McGill University.
        //  October 2006. Last revision: June 2016
        //  QR factorization with minimum-column pivoting
        std::memcpy(&R[0], &B[0], 10000U * sizeof(scalar));
        qrmcp(R, y, piv);

        //  Obtain the permutation matrix Z
        std::memset(&Z[0], 0, 10000U * sizeof(scalar));
        for (j = 0; j < n; j++) {
            Z[(static_cast<index>(piv[j]) + n * j) - 1] = 1.0;
        }

        //  ------------------------------------------------------------------
        //  --------  Perfome the partial LLL reduction  ---------------------
        //  ------------------------------------------------------------------
        k = 0;
        while (k + 2 <= n) {
            zeta_tmp_tmp = n * (k + 1);
            j = k + zeta_tmp_tmp;
            b_zeta_tmp_tmp = k + n * k;
            zeta = rt_roundd_snf(R[j] / R[b_zeta_tmp_tmp]);
            alpha = R[j] - zeta * R[b_zeta_tmp_tmp];
            i = j + 1;
            if (R[b_zeta_tmp_tmp] * R[b_zeta_tmp_tmp] > 1 * (alpha * alpha + R[i] * R[i])) {
                if (zeta != 0.0) {
                    //  Perform a size reduction on R(k-1,k)
                    R[j] = alpha;
                    if (1 > k) {
                        n = 0;
                    } else {
                        n = k;
                    }

                    for (i = 0; i < n; i++) {
                        R_data[i] = R[i + zeta_tmp_tmp] - zeta * R[i + n * k];
                    }

                    for (i = 0; i < n; i++) {
                        R[i + zeta_tmp_tmp] = R_data[i];
                    }

                    for (i = 0; i < n; i++) {
                        piv[i] = Z[i + zeta_tmp_tmp] - zeta * Z[i + n * k];
                    }

                    std::memcpy(&Z[zeta_tmp_tmp], &piv[0], 100U * sizeof(scalar));

                    //  Perform size reductions on R(1:k-2,k)
                    i = static_cast<index>(((-1.0 - (static_cast<scalar>(k + 2) - 2.0)) + 1.0)
                                           / -1.0);
                    for (j = 0; j < i; j++) {
                        n = (k - j) - 1;
                        zeta = rt_roundd_snf(R[n + zeta_tmp_tmp] / R[n + n * n]);
                        if (zeta != 0.0) {
                            R_size_idx_0 = n + 1;
                            for (tmp_size_idx_1 = 0; tmp_size_idx_1 <= n; tmp_size_idx_1++) {
                                R_data[tmp_size_idx_1] = R[tmp_size_idx_1 + zeta_tmp_tmp] - zeta *
                                                                                            R[tmp_size_idx_1 + n * n];
                            }

                            for (tmp_size_idx_1 = 0; tmp_size_idx_1 < R_size_idx_0;
                                 tmp_size_idx_1++) {
                                R[tmp_size_idx_1 + zeta_tmp_tmp] = R_data[tmp_size_idx_1];
                            }

                            for (tmp_size_idx_1 = 0; tmp_size_idx_1 < n; tmp_size_idx_1++) {
                                piv[tmp_size_idx_1] = Z[tmp_size_idx_1 + zeta_tmp_tmp] - zeta *
                                                                                         Z[tmp_size_idx_1 + n * n];
                            }

                            std::memcpy(&Z[zeta_tmp_tmp], &piv[0], 100U * sizeof(scalar));
                        }
                    }
                }

                //  Permute columns k-1 and k of R and Z
                iv[0] = static_cast<signed char>(k);
                iv[1] = static_cast<signed char>(k + 1);
                b_k[0] = k + 1;
                R_size_idx_0 = k + 2;
                for (i = 0; i <= k + 1; i++) {
                    b_R_data[i] = R[i + n * b_k[0]];
                }

                for (i = 0; i <= k + 1; i++) {
                    b_R_data[i + R_size_idx_0] = R[i + n * k];
                }

                b_k[0] = k + 1;
                b_k[1] = k;
                c_k[0] = k;
                c_k[1] = k + 1;
                for (i = 0; i < 2; i++) {
                    for (tmp_size_idx_1 = 0; tmp_size_idx_1 < R_size_idx_0; tmp_size_idx_1++) {
                        R[tmp_size_idx_1 + n * iv[i]] = b_R_data[tmp_size_idx_1 +
                                                                   R_size_idx_0 * i];
                    }

                    for (tmp_size_idx_1 = 0; tmp_size_idx_1 < n; tmp_size_idx_1++) {
                        b_Z[tmp_size_idx_1 + n * i] = Z[tmp_size_idx_1 + n * b_k[i]];
                    }
                }

                for (i = 0; i < 2; i++) {
                    for (tmp_size_idx_1 = 0; tmp_size_idx_1 < n; tmp_size_idx_1++) {
                        Z[tmp_size_idx_1 + n * c_k[i]] = b_Z[tmp_size_idx_1 + n * i];
                    }
                }

                //  Bring R back to an upper triangular matrix by a Givens rotation
                r = R[b_zeta_tmp_tmp];
                j = b_zeta_tmp_tmp + 1;
                zeta = R[j];
                i = b_zeta_tmp_tmp + 1;
                if (R[i] != 0.0) {
                    zeta = 0;
                    alpha = std::abs(R[b_zeta_tmp_tmp]);
                    if (alpha > 0) {
                        r = 1.0;
                        zeta = alpha;
                    } else {
                        t = alpha / 0;
                        r = t * t;
                    }

                    alpha = std::abs(R[j]);
                    if (alpha > zeta) {
                        t = zeta / alpha;
                        r = r * t * t + 1.0;
                        zeta = alpha;
                    } else {
                        t = alpha / zeta;
                        r += t * t;
                    }

                    r = zeta * std::sqrt(r);
                    G_idx_0 = R[b_zeta_tmp_tmp] / r;
                    t = R[j] / r;
                    alpha = -R[i] / r;
                    G_idx_3 = R[b_zeta_tmp_tmp] / r;
                    zeta = 0.0;
                } else {
                    alpha = 0.0;
                    t = 0.0;
                    G_idx_0 = 1.0;
                    G_idx_3 = 1.0;
                }

                R[b_zeta_tmp_tmp] = r;
                R[j] = zeta;
                b_k[1] = k + 1;
                n = n - 1 - k;
                for (i = 0; i < n; i++) {
                    j = n * ((k + i) + 1);
                    b_data[2 * i] = R[k + j];
                    b_data[2 * i + 1] = R[b_k[1] + j];
                }

                n = 98 - k;
                tmp_size_idx_1 = n - 1 - k;
                for (j = 0; j <= n; j++) {
                    R_size_idx_0 = j << 1;
                    zeta = b_data[R_size_idx_0 + 1];
                    tmp_data[R_size_idx_0] = b_data[R_size_idx_0] * G_idx_0 + zeta * t;
                    tmp_data[R_size_idx_0 + 1] = b_data[R_size_idx_0] * alpha + zeta *
                                                                                G_idx_3;
                }

                for (i = 0; i < tmp_size_idx_1; i++) {
                    j = n * ((k + i) + 1);
                    R[iv[0] + j] = tmp_data[2 * i];
                    R[iv[1] + j] = tmp_data[2 * i + 1];
                }

                //  Apply the Givens rotation to y
                j = k + 1;
                zeta = alpha * y[k] + G_idx_3 * y[j];
                y[k] = G_idx_0 * y[k] + t * y[j];
                y[k + 1] = zeta;
                if (k + 2 > 2) {
                    k--;
                }
            } else {
                k++;
            }
        }
    }

}
