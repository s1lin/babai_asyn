/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 *
 * xorgqr.h
 *
 * Code generation for function 'xorgqr'
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
void xorgqr(const emlrtStack *sp, int32_T m, int32_T n, int32_T k, real_T A
            [4194304], int32_T ia0, const real_T tau[2048], int32_T itau0);

/* End of code generation (xorgqr.h) */
