//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: xgeqrf.cpp
//
// MATLAB Coder version            : 4.3
// C/C++ source code generated on  : 22-Feb-2021 17:29:08
//

// Include Files
#include "xgeqrf.h"
#include "qrtest.h"
#include "rt_nonfinite.h"
#include "xgemv.h"
#include "xgerc.h"
#include "xnrm2.h"
#include <cmath>
#include <cstring>

// Function Declarations
static double rt_hypotd_snf(double u0, double u1);

// Function Definitions

//
// Arguments    : double u0
//                double u1
// Return Type  : double
//
static double rt_hypotd_snf(double u0, double u1)
{
  double y;
  double a;
  a = std::abs(u0);
  y = std::abs(u1);
  if (a < y) {
    a /= y;
    y *= std::sqrt(a * a + 1.0);
  } else if (a > y) {
    y /= a;
    y = a * std::sqrt(y * y + 1.0);
  } else {
    if (!rtIsNaN(y)) {
      y = a * 1.4142135623730951;
    }
  }

  return y;
}

//
// Arguments    : double A[4194304]
//                double tau[2048]
// Return Type  : void
//
void xgeqrf(double A[4194304], double tau[2048])
{
  double work[2048];
  int i;
  int ii;
  double atmp;
  int lastc;
  double xnorm_tmp;
  double beta1;
  int knt;
  int coltop;
  int k;
  boolean_T exitg2;
  int exitg1;
  std::memset(&tau[0], 0, 2048U * sizeof(double));
  std::memset(&work[0], 0, 2048U * sizeof(double));
  for (i = 0; i < 2048; i++) {
    ii = (i << 11) + i;
    if (i + 1 < 2048) {
      atmp = A[ii];
      lastc = ii + 2;
      tau[i] = 0.0;
      xnorm_tmp = xnrm2(2047 - i, A, ii + 2);
      if (xnorm_tmp != 0.0) {
        beta1 = rt_hypotd_snf(A[ii], xnorm_tmp);
        if (A[ii] >= 0.0) {
          beta1 = -beta1;
        }

        if (std::abs(beta1) < 1.0020841800044864E-292) {
          knt = -1;
          coltop = (ii - i) + 2048;
          do {
            knt++;
            for (k = lastc; k <= coltop; k++) {
              A[k - 1] *= 9.9792015476736E+291;
            }

            beta1 *= 9.9792015476736E+291;
            atmp *= 9.9792015476736E+291;
          } while (!(std::abs(beta1) >= 1.0020841800044864E-292));

          beta1 = rt_hypotd_snf(atmp, xnrm2(2047 - i, A, ii + 2));
          if (atmp >= 0.0) {
            beta1 = -beta1;
          }

          tau[i] = (beta1 - atmp) / beta1;
          xnorm_tmp = 1.0 / (atmp - beta1);
          for (k = lastc; k <= coltop; k++) {
            A[k - 1] *= xnorm_tmp;
          }

          for (k = 0; k <= knt; k++) {
            beta1 *= 1.0020841800044864E-292;
          }

          atmp = beta1;
        } else {
          tau[i] = (beta1 - A[ii]) / beta1;
          xnorm_tmp = 1.0 / (A[ii] - beta1);
          coltop = (ii - i) + 2048;
          for (k = lastc; k <= coltop; k++) {
            A[k - 1] *= xnorm_tmp;
          }

          atmp = beta1;
        }
      }

      A[ii] = atmp;
      atmp = A[ii];
      A[ii] = 1.0;
      if (tau[i] != 0.0) {
        knt = 2048 - i;
        lastc = (ii - i) + 2047;
        while ((knt > 0) && (A[lastc] == 0.0)) {
          knt--;
          lastc--;
        }

        lastc = 2047 - i;
        exitg2 = false;
        while ((!exitg2) && (lastc > 0)) {
          coltop = (ii + ((lastc - 1) << 11)) + 2048;
          k = coltop;
          do {
            exitg1 = 0;
            if (k + 1 <= coltop + knt) {
              if (A[k] != 0.0) {
                exitg1 = 1;
              } else {
                k++;
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
        knt = 0;
        lastc = 0;
      }

      if (knt > 0) {
        xgemv(knt, lastc, A, ii + 2049, A, ii + 1, work);
        xgerc(knt, lastc, -tau[i], ii + 1, work, A, ii + 2049);
      }

      A[ii] = atmp;
    } else {
      tau[2047] = 0.0;
    }
  }
}

//
// File trailer for xgeqrf.cpp
//
// [EOF]
//
