//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: sils_reduction.cpp
//
// MATLAB Coder version            : 4.3
// C/C++ source code generated on  : 06-Dec-2020 02:56:06
//
#include "qrmcp.cpp"
#include <cmath>
#include <cstring>

// Function Declarations
static double rt_roundd_snf(double u);

// Function Definitions

//
// Arguments    : double u
// Return Type  : double
//
static double rt_roundd_snf(double u)
{
  double y;
  if (std::abs(u) < 4.503599627370496E+15) {
    if (u >= 0.5) {
      y = std::floor(u + 0.5);
    } else if (u > -0.5) {
      y = u * 0.0;
    } else {
      y = std::ceil(u - 0.5);
    }
  } else {
    y = u;
  }

  return y;
}

//
// [R,Z,y] = sils_reduction(B,y) reduces the general standard integer
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
//     Z - n-by-n unimodular matrix, i.e., an integer matrix with |det(Z)|=1
//     y - m-vector transformed from the input y by Q', i.e., y := Q'*y
// Arguments    : const double B[10000]
//                double y[100]
//                double R[10000]
//                double Z[10000]
// Return Type  : void
//
void sils_reduction(const double B[10000], double y[100], double R[10000],
                    double Z[10000])
{
  double piv[100];
  int j;
  int k;
  int zeta_tmp_tmp;
  int b_zeta_tmp_tmp;
  double zeta;
  double alpha;
  int i;
  signed char iv[2];
  int n;
  int b_k[2];
  int R_size_idx_0;
  double R_data[98];
  double b_R_data[200];
  int c_k[2];
  int tmp_size_idx_1;
  double r;
  double b_Z[200];
  double t;
  double G_idx_0;
  double G_idx_3;
  double b_data[198];
  double tmp_data[198];
  if (isInitialized_sils_reduction == false) {
    sils_reduction_initialize();
  }

  //  Subfunction: qrmcp
  //  Main Reference:
  //  X. Xie, X.-W. Chang, and M. Al Borno. Partial LLL Reduction,
  //  Proceedings of IEEE GLOBECOM 2011, 5 pages.
  //  Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
  //           Xiaohu Xie, Tianyang Zhou
  //  Copyright (c) 2006-2016. Scientific Computing Lab, McGill University.
  //  October 2006. Last revision: June 2016
  //  QR factorization with minimum-column pivoting
  std::memcpy(&R[0], &B[0], 10000U * sizeof(double));
  qrmcp(R, y, piv);

  //  Obtain the permutation matrix Z
  std::memset(&Z[0], 0, 10000U * sizeof(double));
  for (j = 0; j < 100; j++) {
    Z[(static_cast<int>(piv[j]) + 100 * j) - 1] = 1.0;
  }

  //  ------------------------------------------------------------------
  //  --------  Perfome the partial LLL reduction  ---------------------
  //  ------------------------------------------------------------------
  k = 0;
  while (k + 2 <= 100) {
    zeta_tmp_tmp = 100 * (k + 1);
    j = k + zeta_tmp_tmp;
    b_zeta_tmp_tmp = k + 100 * k;
    zeta = rt_roundd_snf(R[j] / R[b_zeta_tmp_tmp]);
    alpha = R[j] - zeta * R[b_zeta_tmp_tmp];
    i = j + 1;
    if (R[b_zeta_tmp_tmp] * R[b_zeta_tmp_tmp] > 1.0000000001 * (alpha * alpha +
         R[i] * R[i])) {
      if (zeta != 0.0) {
        //  Perform a size reduction on R(k-1,k)
        R[j] = alpha;
        if (1 > k) {
          n = 0;
        } else {
          n = k;
        }

        for (i = 0; i < n; i++) {
          R_data[i] = R[i + zeta_tmp_tmp] - zeta * R[i + 100 * k];
        }

        for (i = 0; i < n; i++) {
          R[i + zeta_tmp_tmp] = R_data[i];
        }

        for (i = 0; i < 100; i++) {
          piv[i] = Z[i + zeta_tmp_tmp] - zeta * Z[i + 100 * k];
        }

        std::memcpy(&Z[zeta_tmp_tmp], &piv[0], 100U * sizeof(double));

        //  Perform size reductions on R(1:k-2,k)
        i = static_cast<int>(((-1.0 - (static_cast<double>(k + 2) - 2.0)) + 1.0)
                             / -1.0);
        for (j = 0; j < i; j++) {
          n = (k - j) - 1;
          zeta = rt_roundd_snf(R[n + zeta_tmp_tmp] / R[n + 100 * n]);
          if (zeta != 0.0) {
            R_size_idx_0 = n + 1;
            for (tmp_size_idx_1 = 0; tmp_size_idx_1 <= n; tmp_size_idx_1++) {
              R_data[tmp_size_idx_1] = R[tmp_size_idx_1 + zeta_tmp_tmp] - zeta *
                R[tmp_size_idx_1 + 100 * n];
            }

            for (tmp_size_idx_1 = 0; tmp_size_idx_1 < R_size_idx_0;
                 tmp_size_idx_1++) {
              R[tmp_size_idx_1 + zeta_tmp_tmp] = R_data[tmp_size_idx_1];
            }

            for (tmp_size_idx_1 = 0; tmp_size_idx_1 < 100; tmp_size_idx_1++) {
              piv[tmp_size_idx_1] = Z[tmp_size_idx_1 + zeta_tmp_tmp] - zeta *
                Z[tmp_size_idx_1 + 100 * n];
            }

            std::memcpy(&Z[zeta_tmp_tmp], &piv[0], 100U * sizeof(double));
          }
        }
      }

      //  Permute columns k-1 and k of R and Z
      iv[0] = static_cast<signed char>(k);
      iv[1] = static_cast<signed char>(k + 1);
      b_k[0] = k + 1;
      R_size_idx_0 = k + 2;
      for (i = 0; i <= k + 1; i++) {
        b_R_data[i] = R[i + 100 * b_k[0]];
      }

      for (i = 0; i <= k + 1; i++) {
        b_R_data[i + R_size_idx_0] = R[i + 100 * k];
      }

      b_k[0] = k + 1;
      b_k[1] = k;
      c_k[0] = k;
      c_k[1] = k + 1;
      for (i = 0; i < 2; i++) {
        for (tmp_size_idx_1 = 0; tmp_size_idx_1 < R_size_idx_0; tmp_size_idx_1++)
        {
          R[tmp_size_idx_1 + 100 * iv[i]] = b_R_data[tmp_size_idx_1 +
            R_size_idx_0 * i];
        }

        for (tmp_size_idx_1 = 0; tmp_size_idx_1 < 100; tmp_size_idx_1++) {
          b_Z[tmp_size_idx_1 + 100 * i] = Z[tmp_size_idx_1 + 100 * b_k[i]];
        }
      }

      for (i = 0; i < 2; i++) {
        for (tmp_size_idx_1 = 0; tmp_size_idx_1 < 100; tmp_size_idx_1++) {
          Z[tmp_size_idx_1 + 100 * c_k[i]] = b_Z[tmp_size_idx_1 + 100 * i];
        }
      }

      //  Bring R back to an upper triangular matrix by a Givens rotation
      r = R[b_zeta_tmp_tmp];
      j = b_zeta_tmp_tmp + 1;
      zeta = R[j];
      i = b_zeta_tmp_tmp + 1;
      if (R[i] != 0.0) {
        zeta = 3.3121686421112381E-170;
        alpha = std::abs(R[b_zeta_tmp_tmp]);
        if (alpha > 3.3121686421112381E-170) {
          r = 1.0;
          zeta = alpha;
        } else {
          t = alpha / 3.3121686421112381E-170;
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
      n = 99 - k;
      for (i = 0; i < n; i++) {
        j = 100 * ((k + i) + 1);
        b_data[2 * i] = R[k + j];
        b_data[2 * i + 1] = R[b_k[1] + j];
      }

      n = 98 - k;
      tmp_size_idx_1 = 99 - k;
      for (j = 0; j <= n; j++) {
        R_size_idx_0 = j << 1;
        zeta = b_data[R_size_idx_0 + 1];
        tmp_data[R_size_idx_0] = b_data[R_size_idx_0] * G_idx_0 + zeta * t;
        tmp_data[R_size_idx_0 + 1] = b_data[R_size_idx_0] * alpha + zeta *
          G_idx_3;
      }

      for (i = 0; i < tmp_size_idx_1; i++) {
        j = 100 * ((k + i) + 1);
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

//
// File trailer for sils_reduction.cpp
//
// [EOF]
//
