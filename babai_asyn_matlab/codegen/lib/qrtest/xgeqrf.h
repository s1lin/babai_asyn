//
//  Academic License - for use in teaching, academic research, and meeting
//  course requirements at degree granting institutions only.  Not for
//  government, commercial, or other organizational use.
//
//  xgeqrf.h
//
//  Code generation for function 'xgeqrf'
//


#ifndef XGEQRF_H
#define XGEQRF_H

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
      void xgeqrf(double A[4194304], double tau[2048]);
    }
  }
}

#endif

// End of code generation (xgeqrf.h)
