//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: sils_babai_asyn_initialize.cpp
//
// MATLAB Coder version            : 4.3
// C/C++ source code generated on  : 29-Oct-2020 08:26:23
//

// Include Files
#include "sils_babai_asyn_initialize.h"
#include "rt_nonfinite.h"
#include "sils_babai_asyn.h"
#include "sils_babai_asyn_data.h"

// Function Definitions

//
// Arguments    : void
// Return Type  : void
//
void sils_babai_asyn_initialize()
{
  rt_InitInfAndNaN();
  isInitialized_sils_babai_asyn = true;
}

//
// File trailer for sils_babai_asyn_initialize.cpp
//
// [EOF]
//
