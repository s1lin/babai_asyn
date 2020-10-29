//
// Academic License - for use in teaching, academic research, and meeting
// course requirements at degree granting institutions only.  Not for
// government, commercial, or other organizational use.
// File: sils_babai_asyn.cpp
//
// MATLAB Coder version            : 4.3
// C/C++ source code generated on  : 29-Oct-2020 08:26:23
//

// Include Files
#include "sils_babai_asyn.h"
#include "rt_nonfinite.h"
#include "sils_babai_asyn_data.h"
#include "sils_babai_asyn_initialize.h"
#include <cmath>
#include <cstring>

// Function Declarations
static double rt_roundd_snf(double u);

// Function Definitions

//
// Arguments    : double u
// Return Type  : double
//
static double rt_roundd_snf(double u)
{
  double y;
  if (std::abs(u) < 4.503599627370496E+15) {
    if (u >= 0.5) {
      y = std::floor(u + 0.5);
    } else if (u > -0.5) {
      y = u * 0.0;
    } else {
      y = std::ceil(u - 0.5);
    }
  } else {
    y = u;
  }

  return y;
}

//
// Arguments    : const double R[10000]
//                const double y[100]
//                double Zhat[100]
//                double z[100]
// Return Type  : void
//
void sils_babai_asyn(const double R[10000], const double y[100], double Zhat[100],
                     double z[100])
{
  double c[100];
  double d[100];
  double prsd[100];
  double beta;
  double b_gamma;
  double k;
  int exitg1;
  int exitg2;
  int z_tmp_tmp;
  double b_R;
  int z_tmp;
  int loop_ub;
  int R_tmp;
  if (isInitialized_sils_babai_asyn == false) {
    sils_babai_asyn_initialize();
  }

  //  Current point
  //  c(k)=(y(k)-R(k,k+1:n)*z(k+1:n))/R(k,k)
  //  d(k): left or right search direction at level k
  //  Partial squared residual norm for z
  //  prsd(k) = (norm(y(k+1:n) - R(k+1:n,k+1:n)*z(k+1:n)))^2
  //  The level at which search starts to move up to a higher level
  std::memset(&z[0], 0, 100U * sizeof(double));
  std::memset(&c[0], 0, 100U * sizeof(double));
  std::memset(&d[0], 0, 100U * sizeof(double));
  std::memset(&prsd[0], 0, 100U * sizeof(double));
  std::memset(&Zhat[0], 0, 100U * sizeof(double));

  //  Initial squared search radius
  beta = rtInf;
  c[99] = y[99] / R[9999];
  z[99] = rt_roundd_snf(c[99]);
  b_gamma = R[9999] * (c[99] - z[99]);

  //  Determine enumeration direction at level n
  if (c[99] > z[99]) {
    d[99] = 1.0;
  } else {
    d[99] = -1.0;
  }

  k = 100.0;
  do {
    exitg1 = 0;
    do {
      exitg2 = 0;

      //  Temporary partial squared residual norm at level k
      b_gamma = prsd[static_cast<int>(k) - 1] + b_gamma * b_gamma;
      if (b_gamma < beta) {
        if (k != 1.0) {
          //  move to level k-1
          k--;
          z_tmp_tmp = 101 - static_cast<int>(k + 1.0);
          if (z_tmp_tmp == 1) {
            b_R = 0.0;
            z_tmp = static_cast<int>(k + 1.0);
            loop_ub = 100 - z_tmp;
            for (z_tmp_tmp = 0; z_tmp_tmp <= loop_ub; z_tmp_tmp++) {
              R_tmp = (z_tmp + z_tmp_tmp) - 1;
              b_R += R[(static_cast<int>(k) + 100 * R_tmp) - 1] * z[R_tmp];
            }
          } else {
            b_R = 0.0;
            z_tmp = static_cast<int>(k + 1.0);
            loop_ub = 100 - z_tmp;
            for (z_tmp_tmp = 0; z_tmp_tmp <= loop_ub; z_tmp_tmp++) {
              R_tmp = (z_tmp + z_tmp_tmp) - 1;
              b_R += R[(static_cast<int>(k) + 100 * R_tmp) - 1] * z[R_tmp];
            }
          }

          //  Update the partial squared residual norm
          z_tmp_tmp = static_cast<int>(k) - 1;
          prsd[z_tmp_tmp] = b_gamma;

          //  Find the initial integer
          // c(k) = S(k,k) / R(k,k);
          z_tmp = (static_cast<int>(k) + 100 * (static_cast<int>(k) - 1)) - 1;
          c[z_tmp_tmp] = (y[z_tmp_tmp] - b_R) / R[z_tmp];
          z[z_tmp_tmp] = rt_roundd_snf(c[z_tmp_tmp]);
          b_gamma = R[z_tmp] * (c[z_tmp_tmp] - z[z_tmp_tmp]);
          if (c[static_cast<int>(k) - 1] > z[static_cast<int>(k) - 1]) {
            d[z_tmp_tmp] = 1.0;
          } else {
            d[z_tmp_tmp] = -1.0;
          }
        } else {
          //  A new point is found, update the set of candidate solutions
          beta = b_gamma;
          std::memcpy(&Zhat[0], &z[0], 100U * sizeof(double));
          z[0] += d[0];

          // display(z)
          b_gamma = R[0] * (c[0] - z[0]);
          if (d[0] > 0.0) {
            d[0] = -d[0] - 1.0;
          } else {
            d[0] = -d[0] + 1.0;
          }
        }
      } else {
        exitg2 = 1;
      }
    } while (exitg2 == 0);

    if (k == 100.0) {
      exitg1 = 1;
    } else {
      //  Move back to level k+1
      // if ulevel == 0
      //    ulevel = k;
      // end
      k++;

      //  Find a new integer at level k
      z_tmp_tmp = static_cast<int>(k);
      z_tmp = z_tmp_tmp - 1;
      z[z_tmp] += d[z_tmp];
      b_gamma = R[(z_tmp_tmp + 100 * z_tmp) - 1] * (c[z_tmp] - z[z_tmp]);
      if (d[static_cast<int>(k) - 1] > 0.0) {
        d[z_tmp] = -d[static_cast<int>(k) - 1] - 1.0;
      } else {
        d[z_tmp] = -d[static_cast<int>(k) - 1] + 1.0;
      }
    }
  } while (exitg1 == 0);

  //  The p optimal solutions have been found
}

//
// File trailer for sils_babai_asyn.cpp
//
// [EOF]
//
