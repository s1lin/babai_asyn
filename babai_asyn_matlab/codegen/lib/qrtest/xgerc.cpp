//
//  Academic License - for use in teaching, academic research, and meeting
//  course requirements at degree granting institutions only.  Not for
//  government, commercial, or other organizational use.
//
//  xgerc.cpp
//
//  Code generation for function 'xgerc'
//


// Include files
#include "xgerc.h"
#include "rt_nonfinite.h"

// Function Definitions
namespace coder
{
  namespace internal
  {
    namespace blas
    {
      void xgerc(int m, int n, double alpha1, int ix0, const double y[2048],
                 double A[4194304], int ia0)
      {
        int jA;
        if (!(alpha1 == 0.0)) {
          int jy;
          jA = ia0;
          jy = 0;
          for (int j = 0; j < n; j++) {
            if (y[jy] != 0.0) {
              double temp;
              int i;
              int ix;
              temp = y[jy] * alpha1;
              ix = ix0;
              i = m + jA;
              for (int ijA = jA; ijA < i; ijA++) {
                A[ijA - 1] += A[ix - 1] * temp;
                ix++;
              }
            }

            jy++;
            jA += 2048;
          }
        }
      }
    }
  }
}

// End of code generation (xgerc.cpp)
