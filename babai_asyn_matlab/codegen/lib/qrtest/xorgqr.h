//
//  Academic License - for use in teaching, academic research, and meeting
//  course requirements at degree granting institutions only.  Not for
//  government, commercial, or other organizational use.
//
//  xorgqr.h
//
//  Code generation for function 'xorgqr'
//


#ifndef XORGQR_H
#define XORGQR_H

// Include files
#include "rtwtypes.h"
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace coder
{
  namespace internal
  {
    namespace lapack
    {
      void xorgqr(int m, int n, int k, double A[4194304], int ia0, const double
                  tau[2048], int itau0);
    }
  }
}

#endif

// End of code generation (xorgqr.h)
