/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: _coder_qrtest_api.c
 *
 * MATLAB Coder version            : 4.3
 * C/C++ source code generated on  : 22-Feb-2021 17:29:08
 */

/* Include Files */
#include "_coder_qrtest_api.h"
#include "_coder_qrtest_mex.h"

/* Variable Definitions */
emlrtCTX emlrtRootTLSGlobal = NULL;
emlrtContext emlrtContextGlobal = { true,/* bFirstTime */
  false,                               /* bInitialized */
  131483U,                             /* fVersionInfo */
  NULL,                                /* fErrorFunction */
  "qrtest",                            /* fFunctionName */
  NULL,                                /* fRTCallStack */
  false,                               /* bDebugMode */
  { 2045744189U, 2170104910U, 2743257031U, 4284093946U },/* fSigWrd */
  NULL                                 /* fSigMem */
};

/* Function Declarations */
static real_T (*b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId))[4194304];
static real_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
  const emlrtMsgIdentifier *msgId))[4194304];
static real_T (*emlrt_marshallIn(const emlrtStack *sp, const mxArray *A, const
  char_T *identifier))[4194304];
static const mxArray *emlrt_marshallOut(const real_T u[4194304]);

/* Function Definitions */

/*
 * Arguments    : const emlrtStack *sp
 *                const mxArray *u
 *                const emlrtMsgIdentifier *parentId
 * Return Type  : real_T (*)[4194304]
 */
static real_T (*b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId))[4194304]
{
  real_T (*y)[4194304];
  y = c_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}
/*
 * Arguments    : const emlrtStack *sp
 *                const mxArray *src
 *                const emlrtMsgIdentifier *msgId
 * Return Type  : real_T (*)[4194304]
 */
  static real_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
  const emlrtMsgIdentifier *msgId))[4194304]
{
  real_T (*ret)[4194304];
  static const int32_T dims[2] = { 2048, 2048 };

  emlrtCheckBuiltInR2012b(sp, msgId, src, "double", false, 2U, dims);
  ret = (real_T (*)[4194304])emlrtMxGetData(src);
  emlrtDestroyArray(&src);
  return ret;
}

/*
 * Arguments    : const emlrtStack *sp
 *                const mxArray *A
 *                const char_T *identifier
 * Return Type  : real_T (*)[4194304]
 */
static real_T (*emlrt_marshallIn(const emlrtStack *sp, const mxArray *A, const
  char_T *identifier))[4194304]
{
  real_T (*y)[4194304];
  emlrtMsgIdentifier thisId;
  thisId.fIdentifier = (const char *)identifier;
  thisId.fParent = NULL;
  thisId.bParentIsCell = false;
  y = b_emlrt_marshallIn(sp, emlrtAlias(A), &thisId);
  emlrtDestroyArray(&A);
  return y;
}
/*
 * Arguments    : const real_T u[4194304]
 * Return Type  : const mxArray *
 */
  static const mxArray *emlrt_marshallOut(const real_T u[4194304])
{
  const mxArray *y;
  const mxArray *m;
  static const int32_T iv[2] = { 0, 0 };

  static const int32_T iv1[2] = { 2048, 2048 };

  y = NULL;
  m = emlrtCreateNumericArray(2, iv, mxDOUBLE_CLASS, mxREAL);
  emlrtMxSetData((mxArray *)m, (void *)&u[0]);
  emlrtSetDimensions((mxArray *)m, *(int32_T (*)[2])&iv1[0], 2);
  emlrtAssign(&y, m);
  return y;
}

/*
 * Arguments    : const mxArray * const prhs[1]
 *                int32_T nlhs
 *                const mxArray *plhs[2]
 * Return Type  : void
 */
void qrtest_api(const mxArray * const prhs[1], int32_T nlhs, const mxArray *
                plhs[2])
{
  real_T (*Q)[4194304];
  real_T (*R)[4194304];
  real_T (*A)[4194304];
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  st.tls = emlrtRootTLSGlobal;
  Q = (real_T (*)[4194304])mxMalloc(sizeof(real_T [4194304]));
  R = (real_T (*)[4194304])mxMalloc(sizeof(real_T [4194304]));

  /* Marshall function inputs */
  A = emlrt_marshallIn(&st, emlrtAlias(prhs[0]), "A");

  /* Invoke the target function */
  qrtest(*A, *Q, *R);

  /* Marshall function outputs */
  plhs[0] = emlrt_marshallOut(*Q);
  if (nlhs > 1) {
    plhs[1] = emlrt_marshallOut(*R);
  }
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void qrtest_atexit(void)
{
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtEnterRtStackR2012b(&st);
  emlrtLeaveRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
  qrtest_xil_terminate();
  qrtest_xil_shutdown();
  emlrtExitTimeCleanup(&emlrtContextGlobal);
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void qrtest_initialize(void)
{
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  mexFunctionCreateRootTLS();
  st.tls = emlrtRootTLSGlobal;
  emlrtClearAllocCountR2012b(&st, false, 0U, 0);
  emlrtEnterRtStackR2012b(&st);
  emlrtFirstTimeR2012b(emlrtRootTLSGlobal);
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void qrtest_terminate(void)
{
  emlrtStack st = { NULL,              /* site */
    NULL,                              /* tls */
    NULL                               /* prev */
  };

  st.tls = emlrtRootTLSGlobal;
  emlrtLeaveRtStackR2012b(&st);
  emlrtDestroyRootTLS(&emlrtRootTLSGlobal);
}

/*
 * File trailer for _coder_qrtest_api.c
 *
 * [EOF]
 */
