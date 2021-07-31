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
 *   MERCA_tNTABILITY or FITNESS FOR A_t PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with CILS.  If not, see <http://www.gnu.org/licenses/>.
 */

namespace cils {
    template<typename scalar, typename index, index m, index n>
    returnType <scalar, index>
    cils<scalar, index, m, n>::cils_sic_subopt(vector<scalar> &z, array<scalar, m> &v_cur,
                                               array<scalar, m * n> A_t, scalar v_norm_cur, scalar tolerance, index method) {

        vector<scalar> stopping(3, 0);
        array<scalar, n> s_bar_temp;
        array<scalar, m> v_best, v_temp;
        index i, j, l, loop_ub, depth = 0;
        // 'SIC_subopt:32' if v_norm_cur <= tolerance
        if (v_norm_cur <= tolerance) {
            stopping[0] = 1;
        } else {
            scalar max_res = 0.0;
            // 'SIC_subopt:38' s_bar_temp = 0;
            s_bar_temp.fill(0);
            // 'SIC_subopt:39' v_temp = 0;
            v_best.fill(0.0);
            // 'SIC_subopt:40' while v_norm_cur > tolerance
            int exitg2;
            while (v_norm_cur > tolerance) {
                scalar H_cur_tmp, b_s_bar_temp, res;
                int i1;
                for (i = 0; i < n; i++) {
                    s_bar_temp[i] = z[i];
                }
                for (i = 0; i < m; i++) {
                    v_best[i] = v_cur[i];
                }
                // 'SIC_subopt:41' if method == 1
                if (method == 1) {
                    // 'SIC_subopt:42' [s_bar_temp, v_norm_temp, v_temp] =
                    // SIC1_update(A_t, v_cur, n, 1, z);
                    for (j = 0; j < n; j++) {
                        i = n - j;
                        // 'SIC_subopt:93' v = v+s_bar(j)*H(:,j);
                        for (i1 = 0; i1 < n; i1++) {
                            v_best[i1] += s_bar_temp[i - 1] * A_t[i1 + m * (i - 1)];
                        }
                        loop_ub = m;
                        b_s_bar_temp = 0.0;
                        res = 0.0;
                        for (i1 = 0; i1 < loop_ub; i1++) {
                            H_cur_tmp = A_t[i1 + m * (i - 1)];
                            b_s_bar_temp += H_cur_tmp * v_best[i1];
                            res += H_cur_tmp * H_cur_tmp;
                        }
                        b_s_bar_temp = 2.0 * std::floor(b_s_bar_temp / res / 2.0) + 1.0;
                        // 'round_int:17' for i = 1:length(rounded_val)
                        // 'round_int:18' if rounded_val(i) < lower
                        if (b_s_bar_temp < -1.0) {
                            // 'round_int:19' rounded_val(i) = lower;
                            b_s_bar_temp = -1.0;
                        } else if (b_s_bar_temp > 1.0) {
                            // 'round_int:20' elseif rounded_val(i) > upper
                            // 'round_int:21' rounded_val(i) = upper;
                            b_s_bar_temp = 1.0;
                        }
                        // 'SIC_subopt:97' v =v- s_bar_temp *H(:,j);
                        for (i1 = 0; i1 < m; i1++) {
                            v_best[i1] -= b_s_bar_temp * A_t[i1 + m * (i - 1)];
                        }
                        // Updates the term for \hat{x}_j in the residual
                        // 'SIC_subopt:98' s_bar(j)=s_bar_temp;
                        s_bar_temp[i - 1] = b_s_bar_temp;
                    }
                    // 'SIC_subopt:100' v_norm = norm(v);
                    max_res = helper::b_norm<scalar, index, m>(v_best);
                }
                // 'SIC_subopt:44' if method == 2
                if (method == 2) {
                    scalar s_est;
                    // 'SIC_subopt:45' [s_bar_temp, v_norm_temp, v_temp] =
                    // SIC2_update(A_t, v_cur, n, 1, z);
                    l = 0;
                    // 'SIC_subopt:125' s_est = 0;
                    s_est = 0.0;
                    // 'SIC_subopt:126' v_best = -inf;
                    v_best.fill(-INFINITY);
                    // 'SIC_subopt:127' max_res = inf;
                    max_res = INFINITY;
                    // 'SIC_subopt:128' for j=n:-1:1
                    for (j = 0; j < m; j++) {
                        i = n - j;
                        // 'SIC_subopt:129' v_temp = v+s_bar(j)*H(:,j);
                        b_s_bar_temp = z[i - 1];
                        for (i1 = 0; i1 < n; i1++) {
                            v_temp[i1] = v_cur[i1] + b_s_bar_temp * A_t[i1 + m * (i - 1)];
                        }
                        loop_ub = m;
                        b_s_bar_temp = 0.0;
                        res = 0.0;
                        for (i1 = 0; i1 < loop_ub; i1++) {
                            H_cur_tmp = A_t[i1 + m * (i - 1)];
                            b_s_bar_temp += H_cur_tmp * v_temp[i1];
                            res += H_cur_tmp * H_cur_tmp;
                        }
                        b_s_bar_temp = 2.0 * std::floor(b_s_bar_temp / res / 2.0) + 1.0;
                        // 'round_int:17' for i = 1:length(rounded_val)
                        // 'round_int:18' if rounded_val(i) < lower
                        if (b_s_bar_temp < -1.0) {
                            // 'round_int:19' rounded_val(i) = lower;
                            b_s_bar_temp = -1.0;
                        } else if (b_s_bar_temp > 1.0) {
                            // 'round_int:20' elseif rounded_val(i) > upper
                            // 'round_int:21' rounded_val(i) = upper;
                            b_s_bar_temp = 1.0;
                        }
                        // 'SIC_subopt:133' v_temp =v_temp- s_bar_temp *H(:,j);
                        for (i1 = 0; i1 < m; i1++) {
                            v_temp[i1] = v_temp[i1] - b_s_bar_temp * A_t[i1 + m * (i - 1)];
                        }
                        // 'SIC_subopt:134' res = norm(v_temp);
                        res = helper::b_norm<scalar, index, m>(v_best);
                        // 'SIC_subopt:135' if res < max_res
                        if (res < max_res) {
                            // 'SIC_subopt:136' l=j;
                            l = i;
                            // 'SIC_subopt:137' s_est=s_bar_temp;
                            s_est = b_s_bar_temp;
                            // 'SIC_subopt:138' v_best = v_temp;
                            for (i1 = 0; i1 < m; i1++) {
                                v_best[i1] = v_temp[i1];
                            }
                            // 'SIC_subopt:139' max_res = res;
                            max_res = res;
                        }
                    }
                    // 'SIC_subopt:142' s_bar(l) = s_est;
                    s_bar_temp[l - 1] = s_est;
                    // 'SIC_subopt:143' v = v_best;
                    // 'SIC_subopt:144' v_norm = max_res;
                }
                // 'SIC_subopt:47' depth = depth+1;
                depth++;
                // 'SIC_subopt:49' if v_norm_temp < 0.99999 * v_norm_cur
                if (max_res < 0.99999 * v_norm_cur) {
                    // 'SIC_subopt:50' z = s_bar_temp;
                    for (i = 0; i < n; i++) {
                        z[i] = s_bar_temp[i];
                        v_cur[i] = v_best[i];
                    }
                    // 'SIC_subopt:52' v_norm_cur = v_norm_temp;
                    v_norm_cur = max_res;
                } else {
                    // 'SIC_subopt:59' stopping(2) = 1;
                    stopping[1] = 1.0;
                    break;
                }
            }
        }
        if(v_norm_cur < tolerance)
            stopping[2] = 1.0;

        return {stopping, 0, v_norm_cur};
    }
}