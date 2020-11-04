
#include "sils_babai_asyn.h"
#include <cmath>
#include <cstring>

namespace solver {
// Function Declarations
    static double rt_roundd_snf(double u);

// Function Definitions

//
// Arguments    : double u
// Return Type  : double
//
    static double rt_roundd_snf(double u) {
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
//                const double y[n]
//                double x[n]
//                double x[n]
// Return Type  : void
//
    template<typename scalar, typename index, bool use_eigen, bool is_read, bool is_write, index n>
    void sils_search(const scalar* R, const scalar* y, scalar* x, const index size_R) {
        scalar c[n];
        scalar d[n];
        scalar prsd[n];
        scalar b_gamma;
        index k = n;
        index exitg1, exitg2, i, j, r_tmp;
        scalar b_R;
        index loop_ub;

        //  Current point
        //  c(k)=(y(k)-R(k,k+1:n)*x(k+1:n))/R(k,k)
        //  d(k): left or right search direction at level k
        //  Partial squared residual norm for x
        //  prsd(k) = (norm(y(k+1:n) - R(k+1:n,k+1:n)*x(k+1:n)))^2
        //  The level at which search starts to move up to a higher level

        //  Initial squared search radius
        double beta = INFINITY;
        c[n-1] = y[n-1] / R[size_R];
        x[n-1] = rt_roundd_snf(c[n-1]);
        b_gamma = R[size_R] * (c[n-1] - x[n-1]);

        //  Determine enumeration direction at level n
        if (c[n-1] > x[n-1]) {
            d[n-1] = 1.0;
        } else {
            d[n-1] = -1.0;
        }
        
        do {
            exitg1 = 0;
            do {
                exitg2 = 0;

                //  Temporary partial squared residual norm at level k
                b_gamma = prsd[k - 1] + b_gamma * b_gamma;
                if (b_gamma < beta) {
                    if (k != 1) {
                        //  move to level k-1
                        k--;
                        i = n - k;
                        if (i == 1) {
                            b_R = 0.0;
                            j = static_cast<int>(k + 1.0);
                            loop_ub = n - j;
                            for (i = 0; i <= loop_ub; i++) {
                                r_tmp = (j + i) - 1;
                                b_R += R[(k + n * r_tmp) - 1] * x[r_tmp];
                            }
                        } else {
                            b_R = 0.0;
                            j = static_cast<int>(k + 1.0);
                            loop_ub = n - j;
                            for (i = 0; i <= loop_ub; i++) {
                                r_tmp = (j + i) - 1;
                                b_R += R[(k + n * r_tmp) - 1] * x[r_tmp];
                            }
                        }

                        //  Update the partial squared residual norm
                        i = k - 1;
                        prsd[i] = b_gamma;

                        //  Find the initial integer
                        // c(k) = S(k,k) / R(k,k);
                        j = (k + n * (k - 1)) - 1;
                        c[i] = (y[i] - b_R) / R[j];
                        x[i] = rt_roundd_snf(c[i]);
                        b_gamma = R[j] * (c[i] - x[i]);
                        if (c[k - 1] > x[k - 1]) {
                            d[i] = 1.0;
                        } else {
                            d[i] = -1.0;
                        }
                    } else {
                        //  A new point is found, update the set of candidate solutions
                        beta = b_gamma;
                        x[0] += d[0];

                        // display(x)
                        b_gamma = R[0] * (c[0] - x[0]);
                        if (d[0] > 0.0) {
                            d[0] = -d[0] - 1;
                        } else {
                            d[0] = -d[0] + 1;
                        }
                    }
                } else {
                    exitg2 = 1;
                }
            } while (exitg2 == 0);

            if (k == n) {
                exitg1 = 1;
            } else {
                //  Move back to level k+1
                // if ulevel == 0
                //    ulevel = k;
                // end
                k++;

                //  Find a new integer at level k
                i = k;
                j = i - 1;
                x[j] += d[j];
                b_gamma = R[(i + n * j) - 1] * (c[j] - x[j]);
                if (d[k - 1] > 0.0) {
                    d[j] = -d[k - 1] - 1.0;
                } else {
                    d[j] = -d[k - 1] + 1.0;
                }
            }
        } while (exitg1 == 0);
    }
}