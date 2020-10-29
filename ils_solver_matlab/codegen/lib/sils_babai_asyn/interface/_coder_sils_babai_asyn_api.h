/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: _coder_sils_babai_asyn_api.h
 *
 * MATLAB Coder version            : 4.3
 * C/C++ source code generated on  : 29-Oct-2020 08:26:23
 */

#ifndef _CODER_SILS_BABAI_ASYN_API_H
#define _CODER_SILS_BABAI_ASYN_API_H

/* Include Files */
#include <stddef.h>
#include <stdlib.h>
#include "tmwtypes.h"
#include "mex.h"
#include "emlrt.h"

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

/* Function Declarations */
extern void sils_babai_asyn(real_T R[10000], real_T y[100], real_T Zhat[100],
  real_T z[100]);
extern void sils_babai_asyn_api(const mxArray * const prhs[2], int32_T nlhs,
  const mxArray *plhs[2]);
extern void sils_babai_asyn_atexit(void);
extern void sils_babai_asyn_initialize(void);
extern void sils_babai_asyn_terminate(void);
extern void sils_babai_asyn_xil_shutdown(void);
extern void sils_babai_asyn_xil_terminate(void);

#endif

/*
 * File trailer for _coder_sils_babai_asyn_api.h
 *
 * [EOF]
 */
