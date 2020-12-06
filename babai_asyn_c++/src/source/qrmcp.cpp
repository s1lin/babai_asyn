#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <cstring>

#include "../include/SILS.h"


// Function Definitions

//
// [R,piv,y] = qrmcp(B,y) computes the QR factorization of B with
//              minimum-column pivoting:
//                   Q'BP = R (underdetermined B),
//                   Q'BP = [R; 0] (underdetermined B)
//              and computes Q'*y. The orthogonal matrix Q is not produced.
//
//  Inputs:
//     B - m-by-n real matrix to be factorized
//     y - m-dimensional real vector to be transformed to Q'y
//
//  Outputs:
//     R - m-by-n real upper trapezoidal matrix (m < n)
//         n-by-n real upper triangular matrix (m >= n)
//     piv - n-dimensional permutation vector representing P
//     y - m-vector transformed from the input y by Q, i.e., y := Q'*y
// Arguments    : double B[10000]
//                double y[100]
//                double piv[100]
// Return Type  : void
//
void qrmcp(double B[10000], double y[100], double piv[100])
{
  double colNormB[200];
  int j;
  double *b;
  int k;
  double a;
  double scale;
  int n;
  int q;
  double absxk;
  int i;
  double t;
  double varargin_1_data[100];
  int b_k;
  bool exitg1;
  int B_size[1];
  double B_data[99];
  int v_size_idx_0;
  int y_size_idx_1;
  double v_data[100];
  unsigned char b_q[2];
  double b_B_data[100];
  unsigned char c_k[2];
  double y_data[99];
  double b_B[200];

  //  Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
  //           Xiaohu Xie, Tianyang Zhou
  //  Copyright (c) 2006-2016. Scientific Computing Lab, McGill University.
  //  October 2006; Last revision: June 2016
  //  Initialization
  std::memset(&colNormB[0], 0, 200U * sizeof(double));

  //  Compute the 2-norm squared of each column of B
  for (j = 0; j < 100; j++) {
    piv[j] = static_cast<double>(j) + 1.0;
    a = 0.0;
    scale = 3.3121686421112381E-170;
    for (k = 0; k < 100; k++) {
      absxk = std::abs(B[k + 100 * j]);
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
  for (k = 0; k < 99; k++) {
    //  Find the column with minimum 2-norm in B(k:m,k:n)
    q = 100 - k;
    j = -k;
    for (i = 0; i <= j + 99; i++) {
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

    //  Column interchange
    if (q > k) {
      n = static_cast<int>(piv[k]);
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
        for (q = 0; q < 100; q++) {
          b_B[q + 100 * i] = B[q + 100 * b_q[i]];
        }
      }

      for (i = 0; i < 2; i++) {
        for (q = 0; q < 100; q++) {
          B[q + 100 * c_k[i]] = b_B[q + 100 * i];
        }
      }
    }

    //  Compute and apply the Householder transformation  I-tau*v*v'
    B_size[0] = 99 - k;
    j = 99 - k;
    if (0 <= j - 1) {
      std::memcpy(&B_data[0], &B[k * 101 + 1], j * sizeof(double));
    }

    if (b_norm(B_data, B_size) > 0.0) {
      //  A Householder transformation is needed
      v_size_idx_0 = 100 - k;
      j = -k;
      B_size[0] = 100 - k;
      if (0 <= j + 99) {
        std::memcpy(&v_data[0], &B[k * 101], (j + 100) * sizeof(double));
        std::memcpy(&b_B_data[0], &B[k * 101], (j + 100) * sizeof(double));
      }

      t = b_norm(b_B_data, B_size);
      i = k + 100 * k;
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
      b->size[0] = 100 - k;
      b->size[1] = 99 - k;
      emxEnsureCapacity_real_T(b, i);
      j = 99 - k;
      for (i = 0; i < j; i++) {
        n = -k;
        for (q = 0; q <= n + 99; q++) {
          b->data[q + b->size[0] * i] = B[(k + q) + 100 * ((k + i) + 1)];
        }
      }

      n = 98 - k;
      y_size_idx_1 = 99 - k;
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
          b->data[i + b->size[0] * q] = B[(k + i) + 100 * ((k + q) + 1)] -
            b_B_data[i] * y_data[q];
        }
      }

      j = b->size[1];
      for (i = 0; i < j; i++) {
        n = b->size[0];
        for (q = 0; q < n; q++) {
          B[(k + q) + 100 * ((k + i) + 1)] = b->data[q + b->size[0] * i];
        }
      }

      //  Update y by the Householder transformation
      t = 0.0;
      for (i = 0; i < v_size_idx_0; i++) {
        t += v_data[i] * y[k + i];
      }

      q = 100 - k;
      j = 100 - k;
      for (i = 0; i < j; i++) {
        varargin_1_data[i] = y[k + i] - b_B_data[i] * t;
      }

      if (0 <= q - 1) {
        std::memcpy(&y[k], &varargin_1_data[0], q * sizeof(double));
      }
    }

    //  Update colnormB(2,k+1:n)
    y_size_idx_1 = 99 - k;
    j = 99 - k;
    for (i = 0; i < j; i++) {
      n = (k + i) + 1;
      q = k + 100 * n;
      y_data[i] = colNormB[(n << 1) + 1] + B[q] * B[q];
    }

    for (i = 0; i < y_size_idx_1; i++) {
      colNormB[(((k + i) + 1) << 1) + 1] = y_data[i];
    }
  }

  emxFree_real_T(&b);
  n = 2;
  for (j = 0; j < 99; j++) {
    if (n <= 100) {
      std::memset(&B[(j * 100 + n) + -1], 0, (101 - n) * sizeof(double));
    }

    n++;
  }
}

//
// File trailer for qrmcp.cpp
//
// [EOF]
//
