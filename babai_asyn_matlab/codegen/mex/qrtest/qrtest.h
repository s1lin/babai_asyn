/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * qrtest.h
 *
 * Code generation for function 'qrtest'
 *
 */

#pragma once

/* Include files */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mex.h"
#include "emlrt.h"
#include "rtwtypes.h"
#include "qrtest_types.h"

/* Function Declarations */
void qrtest(qrtestStackData *SD, const emlrtStack *sp, const real_T A[4194304],
            real_T Q[4194304], real_T R[4194304]);

/* End of code generation (qrtest.h) */
