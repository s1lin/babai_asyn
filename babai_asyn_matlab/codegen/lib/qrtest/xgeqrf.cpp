//
//  Academic License - for use in teaching, academic research, and meeting
//  course requirements at degree granting institutions only.  Not for
//  government, commercial, or other organizational use.
//
//  xgeqrf.cpp
//
//  Code generation for function 'xgeqrf'
//


// Include files
#include "xgeqrf.h"
#include "rt_nonfinite.h"
#include "xgemv.h"
#include "xgerc.h"
#include "xnrm2.h"
#include "rt_nonfinite.h"
#include <cmath>
#include <cstring>

// Function Declarations
static double rt_hypotd_snf(double u0, double u1);

// Function Definitions
static double rt_hypotd_snf(double u0, double u1)
{
  double a;
  double y;
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

namespace coder
{
  namespace internal
  {
    namespace lapack
    {
      void xgeqrf(double A[4194304], double tau[2048])
      {
        double work[2048];
        double atmp;
        double beta1;
        std::memset(&tau[0], 0, 2048U * sizeof(double));
        std::memset(&work[0], 0, 2048U * sizeof(double));
        for (int i = 0; i < 2048; i++) {
          int ii;
          ii = (i << 11) + i;
          if (i + 1 < 2048) {
            double xnorm_tmp;
            int coltop;
            int knt;
            int lastc;
            int lastv;
            atmp = A[ii];
            lastc = ii + 2;
            tau[i] = 0.0;
            xnorm_tmp = blas::xnrm2(2047 - i, A, ii + 2);
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
                  for (lastv = lastc; lastv <= coltop; lastv++) {
                    A[lastv - 1] *= 9.9792015476736E+291;
                  }

                  beta1 *= 9.9792015476736E+291;
                  atmp *= 9.9792015476736E+291;
                } while (!(std::abs(beta1) >= 1.0020841800044864E-292));

                beta1 = rt_hypotd_snf(atmp, blas::xnrm2(2047 - i, A, ii + 2));
                if (atmp >= 0.0) {
                  beta1 = -beta1;
                }

                tau[i] = (beta1 - atmp) / beta1;
                xnorm_tmp = 1.0 / (atmp - beta1);
                for (lastv = lastc; lastv <= coltop; lastv++) {
                  A[lastv - 1] *= xnorm_tmp;
                }

                for (lastv = 0; lastv <= knt; lastv++) {
                  beta1 *= 1.0020841800044864E-292;
                }

                atmp = beta1;
              } else {
                tau[i] = (beta1 - A[ii]) / beta1;
                xnorm_tmp = 1.0 / (A[ii] - beta1);
                coltop = (ii - i) + 2048;
                for (lastv = lastc; lastv <= coltop; lastv++) {
                  A[lastv - 1] *= xnorm_tmp;
                }

                atmp = beta1;
              }
            }

            A[ii] = atmp;
            atmp = A[ii];
            A[ii] = 1.0;
            if (tau[i] != 0.0) {
              boolean_T exitg2;
              lastv = 2048 - i;
              lastc = (ii - i) + 2047;
              while ((lastv > 0) && (A[lastc] == 0.0)) {
                lastv--;
                lastc--;
              }

              lastc = 2047 - i;
              exitg2 = false;
              while ((!exitg2) && (lastc > 0)) {
                int exitg1;
                coltop = (ii + ((lastc - 1) << 11)) + 2048;
                knt = coltop;
                do {
                  exitg1 = 0;
                  if (knt + 1 <= coltop + lastv) {
                    if (A[knt] != 0.0) {
                      exitg1 = 1;
                    } else {
                      knt++;
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
              blas::xgemv(lastv, lastc, A, ii + 2049, A, ii + 1, work);
              blas::xgerc(lastv, lastc, -tau[i], ii + 1, work, A, ii + 2049);
            }

            A[ii] = atmp;
          } else {
            tau[2047] = 0.0;
          }
        }
      }
    }
  }
}

// End of code generation (xgeqrf.cpp)
