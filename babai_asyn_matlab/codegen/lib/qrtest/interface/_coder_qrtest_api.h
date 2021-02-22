/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: _coder_qrtest_api.h
 *
 * MATLAB Coder version            : 4.3
 * C/C++ source code generated on  : 22-Feb-2021 17:29:08
 */

#ifndef _CODER_QRTEST_API_H
#define _CODER_QRTEST_API_H

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
extern void qrtest(real_T A[4194304], real_T Q[4194304], real_T R[4194304]);
extern void qrtest_api(const mxArray * const prhs[1], int32_T nlhs, const
  mxArray *plhs[2]);
extern void qrtest_atexit(void);
extern void qrtest_initialize(void);
extern void qrtest_terminate(void);
extern void qrtest_xil_shutdown(void);
extern void qrtest_xil_terminate(void);

#endif

/*
 * File trailer for _coder_qrtest_api.h
 *
 * [EOF]
 */
