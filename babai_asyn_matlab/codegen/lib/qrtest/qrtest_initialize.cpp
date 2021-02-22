//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: qrtest_initialize.cpp
//
// MATLAB Coder version            : 4.3
// C/C++ source code generated on  : 22-Feb-2021 17:29:08
//

// Include Files
#include "qrtest_initialize.h"
#include "qrtest.h"
#include "qrtest_data.h"
#include "rt_nonfinite.h"

// Function Definitions

//
// Arguments    : void
// Return Type  : void
//
void qrtest_initialize()
{
  rt_InitInfAndNaN();
  isInitialized_qrtest = true;
}

//
// File trailer for qrtest_initialize.cpp
//
// [EOF]
//
