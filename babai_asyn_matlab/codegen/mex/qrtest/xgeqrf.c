/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * xgeqrf.c
 *
 * Code generation for function 'xgeqrf'
 *
 */

/* Include files */
#include "xgeqrf.h"
#include "qrtest_data.h"
#include "rt_nonfinite.h"
#include "lapacke.h"
#include <stddef.h>

/* Variable Definitions */
static emlrtRSInfo h_emlrtRSI = { 27,  /* lineNo */
  "xgeqrf",                            /* fcnName */
  "/usr/local/MATLAB/R2020b/toolbox/eml/eml/+coder/+internal/+lapack/xgeqrf.m"/* pathName */
};

static emlrtRSInfo i_emlrtRSI = { 91,  /* lineNo */
  "ceval_xgeqrf",                      /* fcnName */
  "/usr/local/MATLAB/R2020b/toolbox/eml/eml/+coder/+internal/+lapack/xgeqrf.m"/* pathName */
};

/* Function Definitions */
void xgeqrf(const emlrtStack *sp, real_T A[4194304], real_T tau[2048])
{
  static const char_T fname[14] = { 'L', 'A', 'P', 'A', 'C', 'K', 'E', '_', 'd',
    'g', 'e', 'q', 'r', 'f' };

  ptrdiff_t info_t;
  emlrtStack b_st;
  emlrtStack st;
  int32_T i;
  int32_T info;
  boolean_T p;
  st.prev = sp;
  st.tls = sp->tls;
  st.site = &h_emlrtRSI;
  b_st.prev = &st;
  b_st.tls = st.tls;
  info_t = LAPACKE_dgeqrf(102, (ptrdiff_t)2048, (ptrdiff_t)2048, &A[0],
    (ptrdiff_t)2048, &tau[0]);
  b_st.site = &i_emlrtRSI;
  info = (int32_T)info_t;
  if (info != 0) {
    p = true;
    if (info != -4) {
      if (info == -1010) {
        emlrtErrorWithMessageIdR2018a(&b_st, &emlrtRTEI, "MATLAB:nomem",
          "MATLAB:nomem", 0);
      } else {
        emlrtErrorWithMessageIdR2018a(&b_st, &b_emlrtRTEI,
          "Coder:toolbox:LAPACKCallErrorInfo",
          "Coder:toolbox:LAPACKCallErrorInfo", 5, 4, 14, fname, 12, info);
      }
    }
  } else {
    p = false;
  }

  if (p) {
    for (info = 0; info < 2048; info++) {
      for (i = 0; i < 2048; i++) {
        A[(info << 11) + i] = rtNaN;
      }

      tau[info] = rtNaN;
    }
  }
}

/* End of code generation (xgeqrf.c) */
