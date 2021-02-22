//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: xgerc.cpp
//
// MATLAB Coder version            : 4.3
// C/C++ source code generated on  : 22-Feb-2021 17:29:08
//

// Include Files
#include "xgerc.h"
#include "qrtest.h"
#include "rt_nonfinite.h"

// Function Definitions

//
// Arguments    : int m
//                int n
//                double alpha1
//                int ix0
//                const double y[2048]
//                double A[4194304]
//                int ia0
// Return Type  : void
//
void xgerc(int m, int n, double alpha1, int ix0, const double y[2048], double A
           [4194304], int ia0)
{
  int jA;
  int jy;
  int j;
  double temp;
  int ix;
  int i;
  int ijA;
  if (!(alpha1 == 0.0)) {
    jA = ia0;
    jy = 0;
    for (j = 0; j < n; j++) {
      if (y[jy] != 0.0) {
        temp = y[jy] * alpha1;
        ix = ix0;
        i = m + jA;
        for (ijA = jA; ijA < i; ijA++) {
          A[ijA - 1] += A[ix - 1] * temp;
          ix++;
        }
      }

      jy++;
      jA += 2048;
    }
  }
}

//
// File trailer for xgerc.cpp
//
// [EOF]
//
