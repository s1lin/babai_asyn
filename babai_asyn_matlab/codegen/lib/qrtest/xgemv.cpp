//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: xgemv.cpp
//
// MATLAB Coder version            : 4.3
// C/C++ source code generated on  : 22-Feb-2021 17:29:08
//

// Include Files
#include "xgemv.h"
#include "qrtest.h"
#include "rt_nonfinite.h"
#include <cstring>

// Function Definitions

//
// Arguments    : int m
//                int n
//                const double A[4194304]
//                int ia0
//                const double x[4194304]
//                int ix0
//                double y[2048]
// Return Type  : void
//
void xgemv(int m, int n, const double A[4194304], int ia0, const double x
           [4194304], int ix0, double y[2048])
{
  int iy;
  int i;
  int iac;
  int ix;
  double c;
  int i1;
  int ia;
  if ((m != 0) && (n != 0)) {
    if (0 <= n - 1) {
      std::memset(&y[0], 0, n * sizeof(double));
    }

    iy = 0;
    i = ia0 + ((n - 1) << 11);
    for (iac = ia0; iac <= i; iac += 2048) {
      ix = ix0;
      c = 0.0;
      i1 = (iac + m) - 1;
      for (ia = iac; ia <= i1; ia++) {
        c += A[ia - 1] * x[ix - 1];
        ix++;
      }

      y[iy] += c;
      iy++;
    }
  }
}

//
// File trailer for xgemv.cpp
//
// [EOF]
//
