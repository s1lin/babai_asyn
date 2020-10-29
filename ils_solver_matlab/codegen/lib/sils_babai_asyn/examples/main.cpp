//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: main.cpp
//
// MATLAB Coder version            : 4.3
// C/C++ source code generated on  : 29-Oct-2020 08:26:23
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
#include "rt_nonfinite.h"
#include "sils_babai_asyn.h"
#include "sils_babai_asyn_terminate.h"

// Function Declarations
static void argInit_100x100_real_T(double result[10000]);
static void argInit_1x100_real_T(double result[100]);
static double argInit_real_T();
static void main_sils_babai_asyn();

// Function Definitions

//
// Arguments    : double result[10000]
// Return Type  : void
//
static void argInit_100x100_real_T(double result[10000])
{
  int idx0;
  int idx1;

  // Loop over the array to initialize each element.
  for (idx0 = 0; idx0 < 100; idx0++) {
    for (idx1 = 0; idx1 < 100; idx1++) {
      // Set the value of the array element.
      // Change this value to the value that the application requires.
      result[idx0 + 100 * idx1] = argInit_real_T();
    }
  }
}

//
// Arguments    : double result[100]
// Return Type  : void
//
static void argInit_1x100_real_T(double result[100])
{
  int idx1;

  // Loop over the array to initialize each element.
  for (idx1 = 0; idx1 < 100; idx1++) {
    // Set the value of the array element.
    // Change this value to the value that the application requires.
    result[idx1] = argInit_real_T();
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
static void main_sils_babai_asyn()
{
  double dv[10000];
  double dv1[100];
  double Zhat[100];
  double z[100];

  // Initialize function 'sils_babai_asyn' input arguments.
  // Initialize function input argument 'R'.
  // Initialize function input argument 'y'.
  // Call the entry-point 'sils_babai_asyn'.
  argInit_100x100_real_T(dv);
  argInit_1x100_real_T(dv1);
  sils_babai_asyn(dv, dv1, Zhat, z);
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
  main_sils_babai_asyn();

  // Terminate the application.
  // You do not need to do this more than one time.
  sils_babai_asyn_terminate();
  return 0;
}

//
// File trailer for main.cpp
//
// [EOF]
//
