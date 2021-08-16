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
    cils<scalar, index, m, n>::cils_sic_serial(vector<scalar> &x) {
        array<scalar, m * 2> b_A_t;
        array<scalar, n * 2> b_P_t;
        array<scalar, m> y_1, y_2;
        scalar s_est = 0.0, res, s_temp, max_res;
        index ij[2], ji[2], i1, i2, b_j, i, bound = 1, t = 0;//pow(2, qam) - 1, t = 0;

        x.resize(n, 0);
        helper::eye<scalar, index>(n, P.data());
        for (i = 0; i < m; i++) {
            y_2[i] = y_a[i];
        }
        for (i = 0; i < m * n; i++) {
            H[i] = A[i];
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
                    d = H[i2 + m * i];
                    s_temp += d * y_2[i2];
                    res += d * d;
                }
                s_temp = 2.0 * std::floor(s_temp / res / 2.0) + 1.0;
                if (s_temp < -bound) {
                    s_temp = -bound;
                } else if (s_temp > bound) {
                    s_temp = bound;
                }

                for (i2 = 0; i2 < m; i2++) {
                    y_1[i2] = y_2[i2] - s_temp * H[i2 + m * i];
                }
                res = helper::norm<scalar, index>(m, y_1.data());

                if (res < max_res) {
                    t = i + 1;
                    s_est = s_temp;
                    max_res = res;
                }
            }
            if (t != 0) {
                ji[0] = t - 1;
                ji[1] = b_j - 1;
                ij[0] = b_j - 1;
                ij[1] = t - 1;

                for (i1 = 0; i1 < 2; i1++) {
                    for (i2 = 0; i2 < m; i2++) {
                        b_A_t[i2 + m * i1] = H[i2 + m * ij[i1]];
                    }
                }

                for (i1 = 0; i1 < 2; i1++) {
                    for (i2 = 0; i2 < m; i2++) {
                        H[i2 + m * ji[i1]] = b_A_t[i2 + m * i1];
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
                y_2[i1] = y_2[i1] - x[b_j - 1] * H[i1 + m * (b_j - 1)];
            }
        }
        scalar v_norm = helper::norm<scalar, index>(m, y_2.data());
        time = omp_get_wtime() - time;

        return {{}, time, v_norm};
    }

    template<typename scalar, typename index, index m, index n>
    returnType <scalar, index>
    cils<scalar, index, m, n>::cils_qrp_serial(vector<scalar> &x) {
        array<scalar, 2 * m> b_A_t, b_R_t;
        array<scalar, 2 * n> b_P_t;
        array<scalar, m> b_y_q;
        index ij[2], ji[2], t = -1, i1, i2, i, j;
        scalar c_i, x_est = 0, b_i, b_m, loop_ub, bound = 1, max_res;
        scalar time = omp_get_wtime();
        //QR:
        cils_reduction<scalar, index> reduction(m, n, 0, 0);
        reduction.cils_qr_serial(A.data(), y_a.data());
//        helper::display_vector<scalar, index>(m, reduction.y_q.data(), "y_q");
        
        x.resize(n, 0);
        helper::eye<scalar, index>(n, P.data());
        for (i = 0; i < m * n; i++) {
            H[i] = A[i];
        }

        for (j = 0; j < n - m; j++) {
            b_m = (n - j) - 1;
            max_res = INFINITY;
            i1 = b_m - m;
            for (i = 0; i <= i1; i++) {
                scalar res, x_temp;
                c_i = m + i + 1;
                x_temp = 2.0 * std::floor(reduction.y_q[m - 1] / reduction.R_Q[(m + m * (c_i - 1)) - 1] / 2.0) + 1.0;
                if (x_temp < -bound) {
                    x_temp = -bound;
                } else if (x_temp > bound) {
                    x_temp = bound;
                }

                for (i2 = 0; i2 < m; i2++) {
                    b_y_q[i2] = reduction.y_q[i2] - x_temp * reduction.R_Q[i2 + m * (c_i - 1)];
                }
                res = helper::norm<scalar, index>(m, b_y_q.data());

                if (res < max_res) {
                    t = c_i - 1;
                    x_est = x_temp;
                    max_res = res;
                }
            }
            if (t + 1 != 0) {
                ji[0] = t;
                ji[1] = b_m;
                ij[0] = b_m;
                ij[1] = t;
                b_A_t.fill(0);
                b_R_t.fill(0);
                b_P_t.fill(0);
                for (i1 = 0; i1 < 2; i1++) {
                    for (i2 = 0; i2 < m; i2++) {
                        b_A_t[i2 + m * i1] = H[i2 + m * ij[i1]];
                        b_R_t[i2 + m * i1] = reduction.R_Q[i2 + m * ij[i1]];
                    }
                }

                for (i1 = 0; i1 < 2; i1++) {
                    for (i2 = 0; i2 < m; i2++) {
                        H[i2 + m * ji[i1]] = b_A_t[i2 + m * i1];
                        reduction.R_Q[i2 + m * ji[i1]] = b_R_t[i2 + m * i1];
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
                    reduction.y_q[i1] = reduction.y_q[i1] - x_est * reduction.R_Q[i1 + m * b_m];
                }
            }
        }

        // Compute the Babai point to get the first 1:m entries of x
        for (i = 0; i < m; i++) {
            c_i = m - i;
            if (c_i == m) {
                x_est = reduction.y_q[c_i - 1] / reduction.R_Q[(c_i + m * (c_i - 1)) - 1];
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
                    x_est += reduction.R_Q[(c_i + m * (i1 + i2)) - 1] * x[(b_i + i2) - 1];
                }
                x_est = (reduction.y_q[c_i - 1] - x_est) / reduction.R_Q[(c_i + m * (c_i - 1)) - 1];
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
        for (t = 0; t <= loop_ub; t++) {
            index b = t * m;
            for (b_i = 0; b_i <= b_m; b_i++) {
                i = b + b_i;
                b_y_q[b_i] = b_y_q[b_i] + reduction.R_Q[i % m + m * (i / m)] * x[t];
            }
        }

        for (i = 0; i < m; i++) {
            reduction.y_q[i] = reduction.y_q[i] - b_y_q[i];
        }
        scalar v_norm = helper::norm<scalar, index>(m, reduction.y_q.data());
        time = omp_get_wtime() - time;
        return {{}, time, v_norm};
    }

    template<typename scalar, typename index, index m, index n>
    returnType <scalar, index>
    cils<scalar, index, m, n>::cils_grad_proj(vector<scalar> &x, const index search_iter) {
        vector<scalar> r0, ex;
        array<scalar, n> c, g, pj_1;
        array<scalar, m> q;
        vector<scalar> t_bar, t_seq;
        vector<index> r1, r2, r3, r4, r5;
        vector<bool> b_x_0, b_x_1(n, 0);
        index i, j, k1, k2, k3;
        //
        //  Find a solution to the box-constrained real least squares problem
        //  min_{l<=x<=u}||y_a-Bx|| by the gradient projection method
        //
        //  Inputs:
        //     x - n-dimensional real vector as an initial point
        //     A - m by n real matrix
        //     y_a - m-dimensional real vector
        //     l - n-dimensional integer vector, lower bound
        //     u - n-dimensional integer vector, upper bound
        //
        //  Output:
        //     x - n-dimensional real vector, a solution
        // 'ubils_reduction:221' n = length(x);
        // 'ubils_reduction:223' c = A'*y_a;
        scalar time = omp_get_wtime();
        c.fill(0);
        for (j = 0; j < m; j++) {
            for (i = 0; i < n; i++) {
                c[i] += A[i * m + j] * y_a[j];
            }
        }

        // 'ubils_reduction:225' for iter = 1:search_iter
        for (index iter = 0; iter < search_iter; iter++) {
            // 'ubils_reduction:227' g = A'*(A*x-y_a);
            r0.resize(m);
            for (i = 0; i < m; i++) {
                r0[i] = -y_a[i];
            }
            for (j = 0; j < n; j++) {
                for (i = 0; i < m; i++) {
                    r0[i] += A[j * m + i] * x[j];
                }
            }
            g.fill(0);
            for (j = 0; j < m; j++) {
                for (i = 0; i < n; i++) {
                    g[i] += A[i * m + j] * r0[j];
                }
            }
            //  Check KKT conditions
            // 'ubils_reduction:230' if (x==l) == 0
            b_x_0.resize(n, 0);
            for (i = 0; i < n; i++) {
                b_x_0[i] = !(x[i] == l[i]);
            }
            if (helper::if_all_x_true<index>(b_x_0)) {
                // 'ubils_reduction:231' k1 = 1;
                k1 = 1;
            } else {
                k3 = 0;
                for (i = 0; i < n; i++) {
                    if (x[i] == l[i]) {
                        k3++;
                    }
                }
                r1.resize(k3, 0);
                k3 = 0;
                for (i = 0; i < n; i++) {
                    if (x[i] == l[i]) {
                        r1[k3] = i + 1;
                        k3++;
                    }
                }
                b_x_0.resize(r1.size(), 0);
                k3 = r1.size();
                for (i = 0; i < k3; i++) {
                    b_x_0[i] = (g[r1[i] - 1] > -1.0E-5);
                }
                if (helper::if_all_x_true<index>(b_x_0)) {
                    // 'ubils_reduction:232' elseif (g(x==l) > -1.e-5) == 1
                    // 'ubils_reduction:233' k1 = 1;
                    k1 = 1;
                } else {
                    // 'ubils_reduction:234' else
                    // 'ubils_reduction:234' k1 = 0;
                    k1 = 0;
                }
            }
            // 'ubils_reduction:236' if (x==u) == 0
            b_x_0.resize(n, 0);
            for (i = 0; i < n; i++) {
                b_x_0[i] = !(x[i] == u[i]);
            }
            if (helper::if_all_x_true<index>(b_x_0)) {
                // 'ubils_reduction:237' k2 = 1;
                k2 = 1;
            } else {
                k3 = 0;
                for (i = 0; i < n; i++) {
                    if (x[i] == u[i]) {
                        k3++;
                    }
                }
                r2.resize(k3, 0);
                k3 = 0;
                for (i = 0; i < n; i++) {
                    if (x[i] == u[i]) {
                        r2[k3] = i + 1;
                        k3++;
                    }
                }
                b_x_0.resize(r2.size(), 0);
                k3 = r2.size();
                for (i = 0; i < k3; i++) {
                    b_x_0[i] = (g[r2[i] - 1] < 1.0E-5);
                }
                if (helper::if_all_x_true<index>(b_x_0)) {
                    // 'ubils_reduction:238' elseif (g(x==u) < 1.e-5) == 1
                    // 'ubils_reduction:239' k2 = 1;
                    k2 = 1;
                } else {
                    // 'ubils_reduction:240' else
                    // 'ubils_reduction:240' k2 = 0;
                    k2 = 0;
                }
            }
            // 'ubils_reduction:242' if (l<x & x<u) == 0
            b_x_0.resize(n, 0);
            for (i = 0; i < n; i++) {
                b_x_0[i] = ((!(l[i] < x[i])) || (!(x[i] < u[i])));
            }
            if (helper::if_all_x_true<index>(b_x_0)) {
                // 'ubils_reduction:243' k3 = 1;
                k3 = 1;
            } else {
                b_x_0.resize(n, 0);
                for (i = 0; i < n; i++) {
                    b_x_0[i] = (l[i] < x[i]);
                    b_x_1[i] = (x[i] < u[i]);
                }
                k3 = 0;
                for (i = 0; i < n; i++) {
                    if (b_x_0[i] && b_x_1[i]) {
                        k3++;
                    }
                }
                r3.resize(k3, 0);
                k3 = 0;
                for (i = 0; i < n; i++) {
                    if (b_x_0[i] && b_x_1[i]) {
                        r3[k3] = i + 1;
                        k3++;
                    }
                }
                b_x_0.resize(r3.size(), 0);
                k3 = r3.size();
                for (i = 0; i < k3; i++) {
                    b_x_0[i] = (g[r3[i] - 1] < 1.0E-5);
                }
                if (helper::if_all_x_true<index>(b_x_0)) {
                    // 'ubils_reduction:244' elseif (g(l<x & x<u) < 1.e-5) == 1
                    // 'ubils_reduction:245' k3 = 1;
                    k3 = 1;
                } else {
                    // 'ubils_reduction:246' else
                    // 'ubils_reduction:246' k3 = 0;
                    k3 = 0;
                }
            }
            // 'ubils_reduction:248' if (k1 & k2 & k3)
            if ((k1 != 0) && (k2 != 0) && (k3 != 0)) {
                break;
            } else {
                scalar x_tmp;
                //  Find the Cauchy point
                // 'ubils_reduction:253' t_bar = 1.e5*ones(n,1);
                t_bar.resize(n, 1e5);
                // 'ubils_reduction:254' t_bar(g<0) = (x(g<0)-u(g<0))./g(g<0);
                k3 = 0;
                for (i = 0; i < n; i++) {
                    if (g[i] < 0.0) {
                        k3++;
                    }
                }
                r4.resize(k3, 0);
                k3 = 0;
                for (i = 0; i < n; i++) {
                    if (g[i] < 0.0) {
                        r4[k3] = i + 1;
                        k3++;
                    }
                }
                r0.resize(r4.size());
                k3 = r4.size();
                for (i = 0; i < k3; i++) {
                    r0[i] = (x[r4[i] - 1] - u[r4[i] - 1]) / g[r4[i] - 1];
                }
                k3 = 0;
                for (i = 0; i < n; i++) {
                    if (g[i] < 0.0) {
                        t_bar[i] = r0[k3];
                        k3++;
                    }
                }
                // 'ubils_reduction:255' t_bar(g>0) = (x(g>0)-l(g>0))./g(g>0);
                k3 = 0;
                for (i = 0; i < n; i++) {
                    if (g[i] > 0.0) {
                        k3++;
                    }
                }
                r5.resize(k3, 0);
                k3 = 0;
                for (i = 0; i < n; i++) {
                    if (g[i] > 0.0) {
                        r5[k3] = i + 1;
                        k3++;
                    }
                }
                r0.resize(r5.size());
                k3 = r5.size();
                for (i = 0; i < k3; i++) {
                    r0[i] = (x[r5[i] - 1] - l[r5[i] - 1]) / g[r5[i] - 1];
                }
                k3 = 0;
                for (i = 0; i < n; i++) {
                    if (g[i] > 0.0) {
                        t_bar[i] = r0[k3];
                        k3++;
                    }
                }
                //  Generate the ordered and non-repeated sequence of t_bar
                // 'ubils_reduction:258' t_seq = unique([0;t_bar]);
                r0.resize(n + 1);
                r0[0] = 0;
                for (i = 0; i < n; i++) {
                    r0[i + 1] = t_bar[i];
                }
                helper::unique_vector<scalar, index>(r0, t_seq);
                //  Add 0 to make the implementation easier
                // 'ubils_reduction:259' t = 0;
                x_tmp = 0.0;
                //  Search
                // 'ubils_reduction:261' for j = 2:length(t_seq)
                for (k2 = 0; k2 < t_seq.size() - 1; k2++) {
                    // 'ubils_reduction:262' tj_1 = t_seq(j-1);
                    scalar tj_1 = t_seq[k2];
                    // 'ubils_reduction:263' tj = t_seq(j);
                    scalar tj = t_seq[k2 + 1];
                    //  Compute x(t_{j-1})
                    // 'ubils_reduction:265' xt_j_1 = x - min(tj_1,t_bar).*g;
                    //  Compute teh search direction p_{j-1}
                    // 'ubils_reduction:267' pj_1 = zeros(n,1);
                    pj_1.fill(0);
                    // 'ubils_reduction:268' pj_1(tj_1<t_bar) = -g(tj_1<t_bar);
                    for (i = 0; i < t_bar.size(); i++) {
                        if (tj_1 < t_bar[i]) {
                            pj_1[i] = -g[i];
                        }
                    }
                    //  Compute coefficients
                    // 'ubils_reduction:270' q = A*pj_1;
                    q.fill(0);
                    for (j = 0; j < n; j++) {
                        for (i = 0; i < m; i++) {
                            q[i] += A[j * m + i] * pj_1[j];
                        }
                    }
                    // 'ubils_reduction:271' fj_1d = (A*xt_j_1)'*q - c'*pj_1;
                    ex.resize(t_bar.size());
                    for (j = 0; j < t_bar.size(); j++) {
                        ex[j] = std::fmin(tj_1, t_bar[j]);
                    }
                    ex.resize(n);
                    for (i = 0; i < n; i++) {
                        ex[i] = x[i] - ex[i] * g[i];
                    }
                    r0.resize(m);
                    for (i = 0; i < m; i++) {
                        r0[i] = 0;
                    }
                    for (j = 0; j < n; j++) {
                        for (i = 0; i < m; i++) {
                            r0[i] += A[j * m + i] * ex[j];
                        }
                    }
                    scalar delta_t = 0.0;
                    for (i = 0; i < m; i++) {
                        delta_t += r0[i] * q[i];
                    }
                    scalar fj_1d = 0.0;
                    for (i = 0; i < n; i++) {
                        fj_1d += c[i] * pj_1[i];
                    }
                    fj_1d = delta_t - fj_1d;
                    // 'ubils_reduction:272' fj_1dd = q'*q;
                    // 'ubils_reduction:273' t = tj;
                    x_tmp = tj;
                    //  Find a local minimizer
                    // 'ubils_reduction:275' delta_t = -fj_1d/fj_1dd;
                    delta_t = 0.0;
                    for (i = 0; i < m; i++) {
                        delta_t += q[i] * q[i];
                    }
                    delta_t = -fj_1d / delta_t;
                    // 'ubils_reduction:276' if fj_1d >= 0
                    if (fj_1d >= 0.0) {
                        // 'ubils_reduction:277' t = tj_1;
                        x_tmp = tj_1;
                        break;
                    } else if (delta_t < x_tmp - tj_1) {
                        // 'ubils_reduction:279' elseif delta_t < (tj-tj_1)
                        // 'ubils_reduction:280' t = tj_1+delta_t;
                        x_tmp = tj_1 + delta_t;
                        break;
                    }
                }
                // 'ubils_reduction:285' x = x - min(t,t_bar).*g;
                ex.resize(t_bar.size());
                k3 = t_bar.size();
                for (j = 0; j < k3; j++) {
                    ex[j] = std::fmin(x_tmp, t_bar[j]);
                }
                for (i = 0; i < n; i++) {
                    x[i] = x[i] - ex[i] * g[i];
                }
            }
        }
        //s_bar4 = round_int(s_bar4_unrounded, -1, 1);
        index bound = 1;
        for(i = 0; i < n; i++){
            scalar x_est = x[i];
            x_est = 2.0 * std::floor(x_est / 2.0) + 1.0;
            if (x_est < -bound) {
                x_est = -bound;
            } else if (x_est > bound) {
                x_est = bound;
            }
            x[i] = x_est;
        }
        //todo: v_norm
        time = omp_get_wtime() - time;
        return {{}, 0, time};
    }


}