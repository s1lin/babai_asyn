//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: main.cpp
//
// MATLAB Coder version            : 4.3
// C/C++ source code generated on  : 22-Feb-2021 17:29:08
//

//***********************************************************************
// This automatically generated example C++ main file shows how to call
// entry-point functions that MATLAB Coder generated. You must customize
// this file for your application. Do not modify this file directly.
// Instead, make a copy of this file, modify it, and integrate it into
// your development environment.
//
// This file initializes entry-point function arguments to a default
// size and value before calling the entry-point functions. It does
// not store or use any values returned from the entry-point functions.
// If necessary, it does pre-allocate memory for returned values.
// You can use this file as a starting point for a main function that
// you can deploy in your application.
//
// After you copy the file, and before you deploy it, you must make the
// following changes:
// * For variable-size function arguments, change the example sizes to
// the sizes that your application requires.
// * Change the example values of function arguments to the values that
// your application requires.
// * If the entry-point functions return values, store these values or
// otherwise use them as required by your application.
//
//***********************************************************************

// Include Files
#include "main.h"
#include "qrtest.h"
#include "qrtest_terminate.h"
#include "rt_nonfinite.h"

// Function Declarations
static void argInit_2048x2048_real_T(double result[4194304]);
static double argInit_real_T();
static void main_qrtest();

// Function Definitions

//
// Arguments    : double result[4194304]
// Return Type  : void
//
static void argInit_2048x2048_real_T(double result[4194304])
{
  int idx0;
  int idx1;

  // Loop over the array to initialize each element.
  for (idx0 = 0; idx0 < 2048; idx0++) {
    for (idx1 = 0; idx1 < 2048; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + (idx1 << 11)] = argInit_real_T();
    }
  }
}

//
// Arguments    : void
// Return Type  : double
//
static double argInit_real_T()
{
  return 0.0;
}

//
// Arguments    : void
// Return Type  : void
//
static void main_qrtest()
{
  static double dv[4194304];
  static double Q[4194304];
  static double R[4194304];

  // Initialize function 'qrtest' input arguments.
  // Initialize function input argument 'A'.
  // Call the entry-point 'qrtest'.
  argInit_2048x2048_real_T(dv);
  qrtest(dv, Q, R);
}

//
// Arguments    : int argc
//                const char * const argv[]
// Return Type  : int
//
int main(int, const char * const [])
{
  // The initialize function is being called automatically from your entry-point function. So, a call to initialize is not included here. 
  // Invoke the entry-point functions.
  // You can call entry-point functions multiple times.
  main_qrtest();

  // Terminate the application.
  // You do not need to do this more than one time.
  qrtest_terminate();
  return 0;
}

//
// File trailer for main.cpp
//
// [EOF]
//
