/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_qrtest_mex.c
 *
 * Code generation for function '_coder_qrtest_mex'
 *
 */

/* Include files */
#include "_coder_qrtest_mex.h"
#include "_coder_qrtest_api.h"
#include "qrtest.h"
#include "qrtest_data.h"
#include "qrtest_initialize.h"
#include "qrtest_terminate.h"

/* Function Declarations */
MEXFUNCTION_LINKAGE void qrtest_mexFunction(qrtestStackData *SD, int32_T nlhs,
  mxArray *plhs[2], int32_T nrhs, const mxArray *prhs[1]);

/* Function Definitions */
void qrtest_mexFunction(qrtestStackData *SD, int32_T nlhs, mxArray *plhs[2],
  int32_T nrhs, const mxArray *prhs[1])
{
  const mxArray *outputs[2];
  int32_T b_nlhs;
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  st.tls = emlrtRootTLSGlobal;

  /* Check for proper number of arguments. */
  if (nrhs != 1) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:WrongNumberOfInputs", 5, 12, 1, 4, 6,
                        "qrtest");
  }

  if (nlhs > 2) {
    emlrtErrMsgIdAndTxt(&st, "EMLRT:runTime:TooManyOutputArguments", 3, 4, 6,
                        "qrtest");
  }

  /* Call the function. */
  qrtest_api(SD, prhs, nlhs, outputs);

  /* Copy over outputs to the caller. */
  if (nlhs < 1) {
    b_nlhs = 1;
  } else {
    b_nlhs = nlhs;
  }

  emlrtReturnArrays(b_nlhs, plhs, outputs);
}

void mexFunction(int32_T nlhs, mxArray *plhs[], int32_T nrhs, const mxArray
                 *prhs[])
{
  qrtestStackData *qrtestStackDataGlobal = NULL;
  qrtestStackDataGlobal = (qrtestStackData *)emlrtMxCalloc(1, (size_t)1U *
    sizeof(qrtestStackData));
  mexAtExit(qrtest_atexit);

  /* Module initialization. */
  qrtest_initialize();

  /* Dispatch the entry-point. */
  qrtest_mexFunction(qrtestStackDataGlobal, nlhs, plhs, nrhs, prhs);

  /* Module termination. */
  qrtest_terminate();
  emlrtMxFree(qrtestStackDataGlobal);
}

emlrtCTX mexFunctionCreateRootTLS(void)
{
  emlrtCreateRootTLS(&emlrtRootTLSGlobal, &emlrtContextGlobal, NULL, 1);
  return emlrtRootTLSGlobal;
}

/* End of code generation (_coder_qrtest_mex.c) */
