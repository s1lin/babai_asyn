/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * qrtest.c
 *
 * Code generation for function 'qrtest'
 *
 */

/* Include files */
#include "qrtest.h"
#include "qrtest_types.h"
#include "rt_nonfinite.h"
#include "xgeqrf.h"
#include "xorgqr.h"
#include <string.h>

/* Variable Definitions */
static emlrtRSInfo emlrtRSI = { 4,     /* lineNo */
  "qrtest",                            /* fcnName */
  "/home/shilei/CLionProjects/babai_asyn/babai_asyn_matlab/qrtest.m"/* pathName */
};

static emlrtRSInfo b_emlrtRSI = { 25,  /* lineNo */
  "qr",                                /* fcnName */
  "/usr/local/MATLAB/R2020b/toolbox/eml/lib/matlab/matfun/qr.m"/* pathName */
};

static emlrtRSInfo c_emlrtRSI = { 37,  /* lineNo */
  "eml_qr",                            /* fcnName */
  "/usr/local/MATLAB/R2020b/toolbox/eml/lib/matlab/matfun/private/eml_qr.m"/* pathName */
};

static emlrtRSInfo d_emlrtRSI = { 121, /* lineNo */
  "qr_full",                           /* fcnName */
  "/usr/local/MATLAB/R2020b/toolbox/eml/lib/matlab/matfun/private/eml_qr.m"/* pathName */
};

static emlrtRSInfo e_emlrtRSI = { 124, /* lineNo */
  "qr_full",                           /* fcnName */
  "/usr/local/MATLAB/R2020b/toolbox/eml/lib/matlab/matfun/private/eml_qr.m"/* pathName */
};

static emlrtRSInfo f_emlrtRSI = { 127, /* lineNo */
  "qr_full",                           /* fcnName */
  "/usr/local/MATLAB/R2020b/toolbox/eml/lib/matlab/matfun/private/eml_qr.m"/* pathName */
};

static emlrtRSInfo g_emlrtRSI = { 136, /* lineNo */
  "qr_full",                           /* fcnName */
  "/usr/local/MATLAB/R2020b/toolbox/eml/lib/matlab/matfun/private/eml_qr.m"/* pathName */
};

static emlrtRSInfo j_emlrtRSI = { 189, /* lineNo */
  "unpackQR",                          /* fcnName */
  "/usr/local/MATLAB/R2020b/toolbox/eml/lib/matlab/matfun/private/eml_qr.m"/* pathName */
};

/* Function Definitions */
void qrtest(qrtestStackData *SD, const emlrtStack *sp, const real_T A[4194304],
            real_T Q[4194304], real_T R[4194304])
{
  emlrtStack b_st;
  emlrtStack c_st;
  emlrtStack d_st;
  emlrtStack e_st;
  emlrtStack st;
  real_T tau[2048];
  int32_T R_tmp;
  int32_T i;
  int32_T j;
  st.prev = sp;
  st.tls = sp->tls;
  b_st.prev = &st;
  b_st.tls = st.tls;
  c_st.prev = &b_st;
  c_st.tls = b_st.tls;
  d_st.prev = &c_st;
  d_st.tls = c_st.tls;
  e_st.prev = &d_st;
  e_st.tls = d_st.tls;

  /* QRTEST Summary of this function goes here */
  /*    Detailed explanation goes here */
  st.site = &emlrtRSI;
  b_st.site = &b_emlrtRSI;
  c_st.site = &c_emlrtRSI;
  memcpy(&SD->f0.A[0], &A[0], 4194304U * sizeof(real_T));
  d_st.site = &d_emlrtRSI;
  xgeqrf(&d_st, SD->f0.A, tau);
  for (j = 0; j < 2048; j++) {
    d_st.site = &e_emlrtRSI;
    for (i = 0; i <= j; i++) {
      R_tmp = i + (j << 11);
      R[R_tmp] = SD->f0.A[R_tmp];
    }

    R_tmp = j + 2;
    d_st.site = &f_emlrtRSI;
    if (R_tmp <= 2048) {
      memset(&R[(j * 2048 + R_tmp) + -1], 0, (2049 - R_tmp) * sizeof(real_T));
    }
  }

  d_st.site = &g_emlrtRSI;
  e_st.site = &j_emlrtRSI;
  xorgqr(&e_st, 2048, 2048, 2048, SD->f0.A, 1, tau, 1);
  for (j = 0; j < 2048; j++) {
    memcpy(&Q[j * 2048], &SD->f0.A[j * 2048], 2048U * sizeof(real_T));
  }
}

/* End of code generation (qrtest.c) */
