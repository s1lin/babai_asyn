/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * _coder_qrtest_api.c
 *
 * Code generation for function '_coder_qrtest_api'
 *
 */

/* Include files */
#include "_coder_qrtest_api.h"
#include "qrtest.h"
#include "qrtest_data.h"
#include "rt_nonfinite.h"

/* Function Declarations */
static real_T (*b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId))[4194304];
static real_T (*c_emlrt_marshallIn(const emlrtStack *sp, const mxArray *src,
  const emlrtMsgIdentifier *msgId))[4194304];
static real_T (*emlrt_marshallIn(const emlrtStack *sp, const mxArray *A, const
  char_T *identifier))[4194304];
static const mxArray *emlrt_marshallOut(const real_T u[4194304]);

/* Function Definitions */
static real_T (*b_emlrt_marshallIn(const emlrtStack *sp, const mxArray *u, const
  emlrtMsgIdentifier *parentId))[4194304]
{
  real_T (*y)[4194304];
  y = c_emlrt_marshallIn(sp, emlrtAlias(u), parentId);
  emlrtDestroyArray(&u);
  return y;
}
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
  static const mxArray *emlrt_marshallOut(const real_T u[4194304])
{
  const mxArray *y;
  const mxArray *m;
  static const int32_T iv[2] = { 0, 0 };

  static const int32_T iv1[2] = { 2048, 2048 };

  y = NULL;
  m = emlrtCreateNumericArray(2, iv, mxDOUBLE_CLASS, mxREAL);
  emlrtMxSetData((mxArray *)m, (void *)&u[0]);
  emlrtSetDimensions((mxArray *)m, iv1, 2);
  emlrtAssign(&y, m);
  return y;
}

void qrtest_api(qrtestStackData *SD, const mxArray * const prhs[1], int32_T nlhs,
                const mxArray *plhs[2])
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
  qrtest(SD, &st, *A, *Q, *R);

  /* Marshall function outputs */
  plhs[0] = emlrt_marshallOut(*Q);
  if (nlhs > 1) {
    plhs[1] = emlrt_marshallOut(*R);
  }
}

/* End of code generation (_coder_qrtest_api.c) */
