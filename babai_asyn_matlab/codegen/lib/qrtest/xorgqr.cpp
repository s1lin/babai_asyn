//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: xorgqr.cpp
//
// MATLAB Coder version            : 4.3
// C/C++ source code generated on  : 22-Feb-2021 17:29:08
//

// Include Files
#include "xorgqr.h"
#include "qrtest.h"
#include "rt_nonfinite.h"
#include "xgemv.h"
#include "xgerc.h"
#include <cstring>

// Function Definitions

//
// Arguments    : int m
//                int n
//                int k
//                double A[4194304]
//                int ia0
//                const double tau[2048]
//                int itau0
// Return Type  : void
//
void xorgqr(int m, int n, int k, double A[4194304], int ia0, const double tau
            [2048], int itau0)
{
  int lastv;
  int coltop;
  int itau;
  int ia;
  double work[2048];
  int lastc;
  int i;
  int iaii;
  boolean_T exitg2;
  int exitg1;
  if (n >= 1) {
    lastv = n - 1;
    for (coltop = k; coltop <= lastv; coltop++) {
      ia = (ia0 + (coltop << 11)) - 1;
      lastc = m - 1;
      if (0 <= lastc) {
        std::memset(&A[ia], 0, (((lastc + ia) - ia) + 1) * sizeof(double));
      }

      A[ia + coltop] = 1.0;
    }

    itau = (itau0 + k) - 2;
    std::memset(&work[0], 0, 2048U * sizeof(double));
    for (i = k; i >= 1; i--) {
      iaii = ((ia0 + i) + ((i - 1) << 11)) - 2;
      if (i < n) {
        A[iaii] = 1.0;
        lastc = m - i;
        if (tau[itau] != 0.0) {
          lastv = lastc + 1;
          lastc += iaii;
          while ((lastv > 0) && (A[lastc] == 0.0)) {
            lastv--;
            lastc--;
          }

          lastc = n - i;
          exitg2 = false;
          while ((!exitg2) && (lastc > 0)) {
            coltop = (iaii + ((lastc - 1) << 11)) + 2048;
            ia = coltop;
            do {
              exitg1 = 0;
              if (ia + 1 <= coltop + lastv) {
                if (A[ia] != 0.0) {
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
          xgemv(lastv, lastc, A, iaii + 2049, A, iaii + 1, work);
          xgerc(lastv, lastc, -tau[itau], iaii + 1, work, A, iaii + 2049);
        }
      }

      if (i < m) {
        lastc = iaii + 2;
        lastv = (iaii + m) - i;
        for (coltop = lastc; coltop <= lastv + 1; coltop++) {
          A[coltop - 1] *= -tau[itau];
        }
      }

      A[iaii] = 1.0 - tau[itau];
      for (coltop = 0; coltop <= i - 2; coltop++) {
        A[(iaii - coltop) - 1] = 0.0;
      }

      itau--;
    }
  }
}

//
// File trailer for xorgqr.cpp
//
// [EOF]
//
