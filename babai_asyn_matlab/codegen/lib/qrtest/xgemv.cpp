//
//  Academic License - for use in teaching, academic research, and meeting
//  course requirements at degree granting institutions only.  Not for
//  government, commercial, or other organizational use.
//
//  xgemv.cpp
//
//  Code generation for function 'xgemv'
//


// Include files
#include "xgemv.h"
#include "rt_nonfinite.h"
#include <cstring>

// Function Definitions
namespace coder
{
  namespace internal
  {
    namespace blas
    {
      void xgemv(int m, int n, const double A[4194304], int ia0, const double x
                 [4194304], int ix0, double y[2048])
      {
        double c;
        int iac;
        if ((m != 0) && (n != 0)) {
          int i;
          int iy;
          if (0 <= n - 1) {
            std::memset(&y[0], 0, n * sizeof(double));
          }

          iy = 0;
          i = ia0 + ((n - 1) << 11);
          for (iac = ia0; iac <= i; iac += 2048) {
            int i1;
            int ix;
            ix = ix0;
            c = 0.0;
            i1 = (iac + m) - 1;
            for (int ia = iac; ia <= i1; ia++) {
              c += A[ia - 1] * x[ix - 1];
              ix++;
            }

            y[iy] += c;
            iy++;
          }
        }
      }
    }
  }
}

// End of code generation (xgemv.cpp)
