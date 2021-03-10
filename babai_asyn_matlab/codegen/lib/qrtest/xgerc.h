//
//  Academic License - for use in teaching, academic research, and meeting
//  course requirements at degree granting institutions only.  Not for
//  government, commercial, or other organizational use.
//
//  xgerc.h
//
//  Code generation for function 'xgerc'
//


#ifndef XGERC_H
#define XGERC_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder
{
  namespace internal
  {
    namespace blas
    {
      void xgerc(int m, int n, double alpha1, int ix0, const double y[2048],
                 double A[4194304], int ia0);
    }
  }
}

#endif

// End of code generation (xgerc.h)
