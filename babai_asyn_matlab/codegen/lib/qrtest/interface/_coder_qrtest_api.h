/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_qrtest_api.h
 *
 * Code generation for function 'qrtest'
 *
 */

#ifndef _CODER_QRTEST_API_H
#define _CODER_QRTEST_API_H

/* Include files */
#include "emlrt.h"
#include "tmwtypes.h"
#include <string.h>

/* Variable Declarations */
extern emlrtCTX emlrtRootTLSGlobal;
extern emlrtContext emlrtContextGlobal;

#ifdef __cplusplus

extern "C" {

#endif

  /* Function Declarations */
  void qrtest(real_T A[4194304], real_T Q[4194304], real_T R[4194304]);
  void qrtest_api(const mxArray * const prhs[1], int32_T nlhs, const mxArray
                  *plhs[2]);
  void qrtest_atexit(void);
  void qrtest_initialize(void);
  void qrtest_terminate(void);
  void qrtest_xil_shutdown(void);
  void qrtest_xil_terminate(void);

#ifdef __cplusplus

}
#endif
#endif

/* End of code generation (_coder_qrtest_api.h) */
