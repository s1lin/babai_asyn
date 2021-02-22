//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: qrtest.cpp
//
// MATLAB Coder version            : 4.3
// C/C++ source code generated on  : 22-Feb-2021 17:29:08
//

// Include Files
#include "qrtest.h"
#include "qrtest_data.h"
#include "qrtest_initialize.h"
#include "rt_nonfinite.h"
#include "xgeqrf.h"
#include "xorgqr.h"
#include <cstring>

// Function Definitions

//
// QRTEST Summary of this function goes here
//    Detailed explanation goes here
// Arguments    : const double A[4194304]
//                double Q[4194304]
//                double R[4194304]
// Return Type  : void
//
void qrtest(const double A[4194304], double Q[4194304], double R[4194304])
{
  static double b_A[4194304];
  double tau[2048];
  int j;
  int i;
  int R_tmp;
  if (isInitialized_qrtest == false) {
    qrtest_initialize();
  }

  std::memcpy(&b_A[0], &A[0], 4194304U * sizeof(double));
  xgeqrf(b_A, tau);
  for (j = 0; j < 2048; j++) {
    for (i = 0; i <= j; i++) {
      R_tmp = i + (j << 11);
      R[R_tmp] = b_A[R_tmp];
    }

    i = j + 2;
    if (i <= 2048) {
      std::memset(&R[(j * 2048 + i) + -1], 0, (2049 - i) * sizeof(double));
    }
  }

  xorgqr(2048, 2048, 2048, b_A, 1, tau, 1);
  for (j = 0; j < 2048; j++) {
    std::memcpy(&Q[j * 2048], &b_A[j * 2048], 2048U * sizeof(double));
  }
}

//
// File trailer for qrtest.cpp
//
// [EOF]
//
