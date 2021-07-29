/** \file
 * \brief Computation of indexeger least square problem by constrained non-blocl Babai Estimator
 * \author Shilei Lin
 * This file is part of CILS.
 *   CILS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   CILS is distributed in the hope that it will be useful,
 *   but WITAOUT ANY WARRANTY; without even the implied warranty of
 *   MERCA_tNTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with CILS.  If not, see <http://www.gnu.org/licenses/>.
 */

namespace cils {

    template<typename scalar, typename index, index m, index n>
    returnType <scalar, index>
    cils<scalar, index, m, n>::cils_sic_serial(vector<scalar> &x, array<scalar, m * n> &A_t, array<scalar, n * n> &P) {
        array<scalar, m * 2> b_A_t;
        array<scalar, n * 2> b_P_t;
        array<scalar, m> y_temp;
        scalar s_est = 0.0, res, s_temp, max_res;
        index ij[2], ji[2], i1, i2, b_j, i, bound = 1, l = 0;//pow(2, qam) - 1, l = 0;


        x.resize(n, 0);
        helper::eye<scalar, index, m, n>(n, P);
        for (i = 0; i < m; i++) {
            y_q[i] = y_a[i];
        }
        for (i = 0; i < m * n; i++) {
            A_t[i] = A[i];
        }

        scalar time = omp_get_wtime();
        for (index j = 0; j < n; j++) {
            b_j = n - j;
            max_res = INFINITY;

            //  Determine the j-th column
            for (i = 0; i < b_j; i++) {
                s_temp = res = 0.0;
                for (i2 = 0; i2 < m; i2++) {
                    scalar d;
                    d = A_t[i2 + m * i];
                    s_temp += d * y_q[i2];
                    res += d * d;
                }
                s_temp = 2.0 * std::floor(s_temp / res / 2.0) + 1.0;
                if (s_temp < -bound) {
                    s_temp = -bound;
                } else if (s_temp > bound) {
                    s_temp = bound;
                }

                for (i2 = 0; i2 < m; i2++) {
                    y_temp[i2] = y_q[i2] - s_temp * A_t[i2 + m * i];
                }
                res = helper::b_norm<scalar, index, m>(y_temp);

                if (res < max_res) {
                    l = i + 1;
                    s_est = s_temp;
                    max_res = res;
                }
            }
            if (l != 0) {
                ji[0] = l - 1;
                ji[1] = b_j - 1;
                ij[0] = b_j - 1;
                ij[1] = l - 1;

                for (i1 = 0; i1 < 2; i1++) {
                    for (i2 = 0; i2 < m; i2++) {
                        b_A_t[i2 + m * i1] = A_t[i2 + m * ij[i1]];
                    }
                }

                for (i1 = 0; i1 < 2; i1++) {
                    for (i2 = 0; i2 < m; i2++) {
                        A_t[i2 + m * ji[i1]] = b_A_t[i2 + m * i1];
                    }
                }

                for (i1 = 0; i1 < 2; i1++) {
                    for (i2 = 0; i2 < n; i2++) {
                        b_P_t[i2 + n * i1] = P[i2 + n * ij[i1]];
                    }
                }

                for (i1 = 0; i1 < 2; i1++) {
                    for (i2 = 0; i2 < n; i2++) {
                        P[i2 + n * ji[i1]] = b_P_t[i2 + n * i1];
                    }
                }
                x[b_j - 1] = s_est;
            }

            for (i1 = 0; i1 < m; i1++) {
                y_q[i1] = y_q[i1] - x[b_j - 1] * A_t[i1 + m * (b_j - 1)];
            }
        }
        scalar v_norm = helper::b_norm<scalar, index, m>(y_q);
        time = omp_get_wtime() - time;

        return {{}, time, v_norm};
    }

    template<typename scalar, typename index, index m, index n>
    returnType <scalar, index>
    cils<scalar, index, m, n>::cils_qrp_serial(vector<scalar> &x, array<scalar, m * n> &A_t, array<scalar, n * n> &P) {
        array<scalar, 2 * m> b_A_t, b_R_t;
        array<scalar, 2 * n> b_P_t;
        array<scalar, m> b_y_q;
        index ij[2], ji[2], l, i1, i2, i, j;
        scalar c_i, x_est, b_i, b_m, loop_ub, bound = 1, max_res;

        x.resize(n, 0);
        helper::eye<scalar, index, m, n>(n, P);
        for (i = 0; i < m * n; i++) {
            A_t[i] = A[i];
        }

        l = -1;
        x_est = 0.0;

        for (j = 0; j < n - m; j++) {
            b_m = (n - j) - 1;
            max_res = INFINITY;
            i1 = b_m - m;
            for (i = 0; i  <= i1; i ++) {
                scalar res, x_temp;
                c_i = m + i  + 1;
                x_temp = 2.0 * std::floor(y_q[m - 1] / R_Q[(m + m * (c_i - 1)) - 1] / 2.0) + 1.0;
                if (x_temp < -bound) {
                    x_temp = -bound;
                } else if (x_temp > bound) {
                    x_temp = bound;
                }

                for (i2 = 0; i2 < m; i2++) {
                    b_y_q[i2] = y_q[i2] - x_temp * R_Q[i2 + m * (c_i - 1)];
                }
                res = helper::b_norm<scalar, index, m>(b_y_q);

                if (res < max_res) {
                    l = c_i - 1;
                    x_est = x_temp;
                    max_res = res;
                }
            }
            if (l + 1 != 0) {
                ji[0] = l;
                ji[1] = b_m;
                ij[0] = b_m;
                ij[1] = l;
                b_A_t.fill(0);
                b_R_t.fill(0);
                b_P_t.fill(0);
                for (i1 = 0; i1 < 2; i1++) {
                    for (i2 = 0; i2 < m; i2++) {
                        b_A_t[i2 + m * i1] = A_t[i2 + m * ij[i1]];
                        b_R_t[i2 + m * i1] = R_Q[i2 + m * ij[i1]];
                    }
                }

                for (i1 = 0; i1 < 2; i1++) {
                    for (i2 = 0; i2 < m; i2++) {
                        A_t[i2 + m * ji[i1]] = b_A_t[i2 + m * i1];
                        R_Q[i2 + m * ji[i1]] = b_R_t[i2 + m * i1];
                    }
                }


                for (i1 = 0; i1 < 2; i1++) {
                    for (i2 = 0; i2 < n; i2++) {
                        b_P_t[i2 + n * i1] = P[i2 + n * ij[i1]];
                    }
                }

                for (i1 = 0; i1 < 2; i1++) {
                    for (i2 = 0; i2 < n; i2++) {
                        P[i2 + n * ji[i1]] = b_P_t[i2 + n * i1];
                    }
                }
                x[b_m] = x_est;
                for (i1 = 0; i1 < m; i1++) {
                    y_q[i1] = y_q[i1] - x_est * R_Q[i1 + m * b_m];
                }
            }
        }

        // Compute the Babai point to get the first 1:m entries of x
        for (i = 0; i < m; i++) {
            c_i = m - i;
            if (c_i == m) {
                x_est = y_q[c_i - 1] / R_Q[(c_i + m * (c_i - 1)) - 1];
            } else {
                if (c_i + 1.0 > m) {
                    i1 = i2 = 0;
                    b_i = 1;
                } else {
                    i1 = c_i;
                    i2 = m;
                    b_i = i1 + 1;
                }
                x_est = 0.0;
                loop_ub = i2 - i1;
                for (i2 = 0; i2 < loop_ub; i2++) {
                    x_est += R_Q[(c_i + m * (i1 + i2)) - 1] * x[(b_i + i2) - 1];
                }
                x_est = (y_q[c_i - 1] - x_est) / R_Q[(c_i + m * (c_i - 1)) - 1];
            }
            x_est = 2.0 * std::floor(x_est / 2.0) + 1.0;
            if (x_est < -bound) {
                x_est = -bound;
            } else if (x_est > bound) {
                x_est = bound;
            }
            x[c_i - 1] = x_est;
        }
        b_m = m - 1;
        if (1 > m) {
            i = 0;
        } else {
            i = m;
        }
        loop_ub = i - 1;
        for (b_i = 0; b_i <= b_m; b_i++) {
            b_y_q[b_i] = 0.0;
        }
        for (l = 0; l <= loop_ub; l++) {
            index b = l * m;
            for (b_i = 0; b_i <= b_m; b_i++) {
                i = b + b_i;
                b_y_q[b_i] = b_y_q[b_i] + R_Q[i % m + m * (i / m)] * x[l];
            }
        }

        for (i = 0; i < m; i++) {
            y_q[i] = y_q[i] - b_y_q[i];
        }
        scalar v_norm = helper::b_norm<scalar, index, m>(y_q);
        return {{}, 0, v_norm};
    }
}