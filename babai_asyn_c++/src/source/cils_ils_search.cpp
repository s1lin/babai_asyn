/** \file
 * \brief Computation of SS_search Algorithm
 * \author Shilei Lin
 * This file is part of CILS.
 *   CILS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   CILS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with CILS.  If not, see <http://www.gnu.org/licenses/>.
 */

namespace cils {

    template<typename scalar, typename index>
    class cils_search {

    private:
        index upper, lower, m, n;
        vector<scalar> p, c, z;
        vector<index> d, l, u;

    public:

        cils_search(index m, index n, index qam) {

            this->upper = 0;
            this->lower = -1;
            this->m = m;
            this->n = n;
            //pow(2, qam) - 1;
            this->p.resize(m);
            this->p.assign(m, 0);
            this->c.resize(m);
            this->c.assign(m, 0);
            this->z.resize(m);
            this->z.assign(m, 0);
            this->d.resize(m);
            this->d.assign(m, 0);
            this->l.resize(m);
            this->l.assign(m, 0);
            this->u.resize(m);
            this->u.assign(m, 0);
        }

        ~cils_search() {
        }

        inline bool obils_search(const index n_dx_q_0, const index n_dx_q_1, const bool check,
                                 const scalar *R_R, const scalar *y_B, vector<scalar> &z_x) {
            this->z.assign(m, 0);
            // Variables
            scalar sum, newprsd, gamma, beta = INFINITY;

            index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1;
            index end_1 = n_dx_q_1 - 1, row_k = k + n_dx_q_0, diff = 0;
            index dflag = 1, count = 0, iter = 0;

            //Initial squared search radius
            scalar R_kk = R_R[m * end_1 + end_1];
            c[row_k] = y_B[row_k] / R_kk;
            z[row_k] = round(c[row_k]);
            if (z[row_k] <= lower) {
                z[row_k] = u[row_k] = lower; //The lower bound is reached
                l[row_k] = d[row_k] = 1;
            } else if (z[row_k] >= upper) {
                z[row_k] = upper; //The upper bound is reached
                u[row_k] = 1;
                l[row_k] = 0;
                d[row_k] = -1;
            } else {
                l[row_k] = u[row_k] = 0;
                //  Determine enumeration direction at level block_size
                d[row_k] = c[row_k] > z[row_k] ? 1 : -1;
            }

            gamma = R_kk * (c[row_k] - z[row_k]);
            //ILS search process
        while (1) {
//            count++;
//            for (count = 0; count < program_def::max_thre || iter == 0; count++) {
                if (dflag) {
                    newprsd = p[row_k] + gamma * gamma;
                    if (newprsd < beta) {
                        if (k != 0) {
                            k--;
                            row_k--;
                            sum = 0;
                            for (index col = k + 1; col < dx; col++) {
                                sum += R_R[(col + n_dx_q_0) * m + row_k] * z[col + n_dx_q_0];
                            }
                            R_kk = R_R[m * row_k + row_k];
                            p[row_k] = newprsd;
                            c[row_k] = (y_B[row_k] - sum) / R_kk;
                            z[row_k] = round(c[row_k]);
                            if (z[row_k] <= lower) {
                                z[row_k] = lower;
                                l[row_k] = 1;
                                u[row_k] = 0;
                                d[row_k] = 1;
                            } else if (z[row_k] >= upper) {
                                z[row_k] = upper;
                                u[row_k] = 1;
                                l[row_k] = 0;
                                d[row_k] = -1;
                            } else {
                                l[row_k] = 0;
                                u[row_k] = 0;
                                d[row_k] = c[row_k] > z[row_k] ? 1 : -1;
                            }
                            gamma = R_kk * (c[row_k] - z[row_k]);

                        } else {
                            beta = newprsd;
                            diff = 0;
                            iter++;
                            for (index h = n_dx_q_0; h < n_dx_q_1; h++) {
                                diff += z_x[h] == z[h];
                                z_x[h] = z[h];
                            }
//                            if (n_dx_q_1 != n) {
//                                if (diff == dx || iter > program_def::search_iter || !check) {
//                                    break;
//                                }
//                            }
                        }
                    } else {
                        dflag = 0;
                    }

                } else {
                    if (k == dx - 1) break;
                    else {
                        k++;
                        row_k++;
                        if (l[row_k] != 1 || u[row_k] != 1) {
                            z[row_k] += d[row_k];
                            if (z[row_k] == lower) {
                                l[row_k] = 1;
                                d[row_k] = -d[row_k] + 1;
                            } else if (z[row_k] == upper) {
                                u[row_k] = 1;
                                d[row_k] = -d[row_k] - 1;
                            } else if (l[row_k] == 1) {
                                d[row_k] = 1;
                            } else if (u[row_k] == 1) {
                                d[row_k] = -1;
                            } else {
                                d[row_k] = d[row_k] > 0 ? -d[row_k] - 1 : -d[row_k] + 1;
                            }
                            gamma = R_R[m * row_k + row_k] * (c[row_k] - z[row_k]);
                            dflag = 1;
                        }
                    }
                }
            }
            return count;
        }

        inline bool obils_search_omp(const index n_dx_q_0, const index n_dx_q_1, const index i, const index check,
                                     const scalar *R_A, const scalar *y_B, scalar *z_x) {
            index dx = n_dx_q_1 - n_dx_q_0;
            index row_k = n_dx_q_1 - 1, row_kk = row_k * (n - n_dx_q_1 / 2);
            index dflag = 1, iter = 0, diff = 0, k = dx - 1;
            scalar R_kk = R_A[row_kk + row_k], beta = INFINITY;

//#pragma omp simd
            for (index R_R = n_dx_q_0; R_R < n_dx_q_1; R_R++) {
                z[R_R] = z_x[R_R];
            }

            c[row_k] = y_B[row_k] / R_kk;
            z[row_k] = round(c[row_k]);
            if (z[row_k] <= lower) {
                z[row_k] = u[row_k] = lower; //The lower bound is reached
                l[row_k] = d[row_k] = 1;
            } else if (z[row_k] >= upper) {
                z[row_k] = upper; //The upper bound is reached
                u[row_k] = 1;
                l[row_k] = 0;
                d[row_k] = -1;
            } else {
                l[row_k] = u[row_k] = 0;
                d[row_k] = c[row_k] > z[row_k] ? 1 : -1; //Determine enumeration direction at level block_size
            }
            //Initial squared search radius
            scalar gamma = R_kk * (c[row_k] - z[row_k]);
            scalar newprsd, sum;
            //ILS search process
            for (index count = 0; count < program_def::max_search || iter == 0; count++) {
                //while (1) {
                if (dflag) {
                    newprsd = p[row_k] + gamma * gamma;
                    if (newprsd < beta) {
                        if (k != 0) {
                            k--;
                            row_k--;
                            row_kk -= (n - row_k - 1);
                            R_kk = R_A[row_kk + row_k];
                            p[row_k] = newprsd;
                            sum = 0;
#pragma omp simd reduction(+ : sum)
                            for (index col = k + 1; col < dx; col++) {
                                sum += R_A[row_kk + col + n_dx_q_0] * z[col + n_dx_q_0];
                            }
                            c[row_k] = (y_B[row_k] - sum) / R_kk;
                            z[row_k] = round(c[row_k]);
                            if (z[row_k] <= lower) {
                                z[row_k] = u[row_k] = lower;
                                l[row_k] = d[row_k] = 1;
                            } else if (z[row_k] >= upper) {
                                z[row_k] = upper;
                                u[row_k] = 1;
                                l[row_k] = 0;
                                d[row_k] = -1;
                            } else {
                                l[row_k] = u[row_k] = 0;
                                d[row_k] = c[row_k] > z[row_k] ? 1 : -1;
                            }
                            gamma = R_kk * (c[row_k] - z[row_k]);

                        } else {
                            beta = newprsd;
                            diff = 0;
                            iter++;
                            for (index h = n_dx_q_0; h < n_dx_q_1; h++) {
                                diff += z_x[h] == z[h];
                                z_x[h] = z[h];
                            }
                            if (i != 0) {
                                if (diff == dx || iter > program_def::search_iter || !check) {
                                    break;
                                }
                            }
                        }
                    } else {
                        dflag = 0;
                    }

                } else {
                    if (k == dx - 1) break;
                    else {
                        k++;
                        row_k++;
                        row_kk += n - row_k;
                        if (l[row_k] != 1 || u[row_k] != 1) {
                            z[row_k] += d[row_k];
                            if (z[row_k] == 0) {
                                l[row_k] = 1;
                                d[row_k] = -d[row_k] + 1;
                            } else if (z[row_k] == upper) {
                                u[row_k] = 1;
                                d[row_k] = -d[row_k] - 1;
                            } else if (l[row_k] == 1) {
                                d[row_k] = 1;
                            } else if (u[row_k] == 1) {
                                d[row_k] = -1;
                            } else {
                                d[row_k] = d[row_k] > 0 ? -d[row_k] - 1 : -d[row_k] + 1;
                            }
                            gamma = R_A[row_kk + row_k] * (c[row_k] - z[row_k]);
                            dflag = 1;
                        }
                    }
                }
            }
            return diff == dx;
        }

        inline bool ils_search(const index n_dx_q_0, const index n_dx_q_1, const bool check,
                               const scalar *R_R, const scalar *y_B, vector<scalar> *z_x) {

            //variables
            scalar sum, newprsd, gamma, beta = INFINITY;

            index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1;
            index end_1 = n_dx_q_1 - 1, row_k = k + n_dx_q_0, diff;
            index row_kk = n * end_1 + end_1, count, iter = 0;

            //Initial squared search radius
            scalar R_kk = R_R[n * end_1 + end_1];
            c[row_k] = y_B[row_k] / R_kk;
            z[row_k] = round(c[row_k]);
            gamma = R_kk * (c[row_k] - z[row_k]);

            //  Determine enumeration direction at level block_size
            d[row_k] = c[row_k] > z[row_k] ? 1 : -1;

            //ILS search process
            for (count = 0; count < program_def::max_thre || iter == 0; count++) {
                newprsd = p[row_k] + gamma * gamma;
                if (newprsd < beta) {
                    if (k != 0) {
                        k--;
                        row_k--;
                        sum = 0;
                        R_kk = R_R[n * row_k + row_k];
                        p[row_k] = newprsd;
                        for (index col = k + 1; col < dx; col++) {
                            sum += R_R[(col + n_dx_q_0) * n + row_k] * z[col + n_dx_q_0];
                        }

                        c[row_k] = (y_B[row_k] - sum) / R_kk;
                        z[row_k] = round(c[row_k]);
                        gamma = R_kk * (c[row_k] - z[row_k]);
                        d[row_k] = c[row_k] > z[row_k] ? 1 : -1;

                    } else {
                        beta = newprsd;
                        diff = 0;
                        iter++;
                        for (index h = n_dx_q_0; h < n_dx_q_1; h++) {
                            diff += z_x->at(h) == z[h];
                            z_x->at(h) = z[h];
                        }
                        if (n_dx_q_1 != n) {
                            if (diff == dx || iter > program_def::search_iter || !check) {
                                break;
                            }
                        }
                        z[row_k] += d[row_k];
                        gamma = R_R[n * row_k + row_k] * (c[row_k] - z[row_k]);
                        d[row_k] = d[row_k] > row_k ? -d[row_k] - 1 : -d[row_k] + 1;
                    }
                } else {
                    if (k == dx - 1) break;
                    else {
                        k++;
                        row_k++;
                        z[row_k] += d[row_k];
                        gamma = R_R[n * row_k + row_k] * (c[row_k] - z[row_k]);
                        d[row_k] = d[row_k] > 0 ? -d[row_k] - 1 : -d[row_k] + 1;
                    }
                }
            }
            return beta;
        }


        inline bool ils_search_omp(const index n_dx_q_0, const index n_dx_q_1, const index i, const index check,
                                   const scalar *R_A, const scalar *y_B,
                                   scalar *z_x) {

            index dx = n_dx_q_1 - n_dx_q_0;
            index row_k = n_dx_q_1 - 1, row_kk = row_k * (n - n_dx_q_1 / 2);
            index dflag = 1, iter = 0, diff = 0, k = dx - 1;
            scalar R_kk = R_A[row_kk + row_k], beta = INFINITY;

//#pragma omp simd
            for (index R_R = n_dx_q_0; R_R < n_dx_q_1; R_R++) {
                z[R_R] = z_x[R_R];
            }

            c[row_k] = y_B[row_k] / R_kk;
            z[row_k] = round(c[row_k]);

            //  Determine enumeration direction at level block_size
            d[row_k] = c[row_k] > z[row_k] ? 1 : -1;
//Initial squared search radius
            scalar gamma = R_kk * (c[row_k] - z[row_k]);
            scalar newprsd, sum;
            //ILS search process
            for (index count = 0; count < program_def::max_search || iter == 0; count++) {
                newprsd = p[row_k] + gamma * gamma;
                if (newprsd < beta) {
                    if (k != 0) {
                        k--;
                        row_k--;
                        sum = 0;
                        row_kk -= (n - row_k - 1);
                        R_kk = R_A[row_kk + row_k];
                        p[row_k] = newprsd;
#pragma omp simd reduction(+ : sum)
                        for (index col = k + 1; col < dx; col++) {
                            sum += R_A[row_kk + col + n_dx_q_0] * z[col + n_dx_q_0];
                        }

                        c[row_k] = (y_B[row_k] - sum) / R_kk;
                        z[row_k] = round(c[row_k]);
                        gamma = R_kk * (c[row_k] - z[row_k]);
                        d[row_k] = c[row_k] > z[row_k] ? 1 : -1;

                    } else {
                        beta = newprsd;
                        diff = 0;
                        iter++;
                        for (index h = n_dx_q_0; h < n_dx_q_1; h++) {
                            diff += z_x[h] == z[h];
                            z_x[h] = z[h];
                        }
                        if (i != 0) {
                            if (diff == dx || iter > program_def::search_iter || !check) {
                                break;
                            }
                        }

                        z[row_k] += d[row_k];
                        gamma = R_A[row_kk + row_k] * (c[row_k] - z[row_k]);
                        d[row_k] = d[row_k] > 0 ? -d[row_k] - 1 : -d[row_k] + 1;
                    }
                } else {
                    if (k == dx - 1) break;
                    else {
                        k++;
                        row_k++;
                        row_kk += n - row_k;
                        z[row_k] += d[row_k];
                        gamma = R_A[row_kk + row_k] * (c[row_k] - z[row_k]);
                        d[row_k] = d[row_k] > 0 ? -d[row_k] - 1 : -d[row_k] + 1;
                    }
                }
            }
            return diff == dx;
        }

        inline bool ubils_search(const index n_dx_q_0, const index n_dx_q_1, const bool check, scalar beta,
                                 const scalar *R_R, const scalar *y_B,
                                 vector<scalar> &z_x) {
            index i, nx, j;
            scalar gamma;
            vector<scalar> zhat(z_x.size(), 0);
            // 'ubils_search:35' [m,n] = size(R_R);
            //  Point which determins the initial search radius beta
            // 'ubils_search:38' zhat = z0;
            for (i = 0; i < n; i++) {
                zhat[i] = z_x[i];
            }
            //  Current point
            // 'ubils_search:41' z = zeros(n,1);
            for (i = 0; i < n; i++) {
                z[i] = 0.0;
            }
            //  c(k) = (y_B(k)-R_R(k,k+1:n)*z(k+1:n))/R_R(k,k)
            // 'ubils_search:44' c = zeros(n,1);
            for (i = 0; i < nx; i++) {
                c[i] = 0.0;
            }
            //  d(k): left or right search direction at level k
            // 'ubils_search:47' d = zeros(n,1);
            for (i = 0; i < nx; i++) {
                d[i] = 0.0;
            }
            //  l(k) = 1 if the lower bound is reached at level k
            // 'ubils_search:50' l = zeros(size(l));
            for (i = 0; i < n; i++) {
                l[i] = 0.0;
            }
            //  u(k) = 1 if the upper bound is reached at level k
            // 'ubils_search:52' u = l;
            for (i = 0; i < n; i++) {
                u[i] = 0.0;
            }
            //  Partial squared residual norm for z
            //  prsd(k) = (norm(y_B(k+1:n)-R_R(k+1:n,k+1:n)*z(k+1:n)))^2
            // 'ubils_search:59' prsd = zeros(m+1,1);
            vector<scalar> prsd(m, 0);

            //  ------------------------------------------------------------------
            //  --------  Search process  ----------------------------------------
            //  ------------------------------------------------------------------
            //  A transformation for z and bounds
            // 'ubils_search:72' [R_R,y_B,u,l,Z_t,s] = shift(R_R,y_B,l,u);
            //
            //  Perform transformations:
            //     z(j) := -l(j) + z(j) if r_{tj}>=0  for t=m or t=j
            //     z(j) := u(j) - z(j)  if r_{tj}<0   for t=m or t=j
            //  so that z(j) is in {0,1, . . . ,u_j-l_j} and R_R(t,j)>=0
            //
            // 'ubils_search:247' [m,n] = size(R_R);
            // 'ubils_search:268' [m,n] = size(R_R);
            index b_m = m;
            index t;
            // 'ubils_search:269' R_temp = R_R;
            // 'ubils_search:270' Z = zeros(n,n);
            vector<scalar> Z(n * n, 0);
            // 'ubils_search:271' s = zeros(n,1);
            vector<scalar> s(n, 0);
            // 'ubils_search:273' for j = 1:n
            i = n;
            for (j = 0; j < n; j++) {
                // 'ubils_search:274' if j > m
                if (j + 1 > m) {
                    // 'ubils_search:275' t = m;
                    t = b_m - 1;
                } else {
                    // 'ubils_search:276' else
                    // 'ubils_search:277' t = j;
                    t = j;
                }
                // 'ubils_search:280' if R_R(t,j) >= 0
                if (R_R[t + m * j] >= 0.0) {
                    // 'ubils_search:281' Z(j,j) = 1;
                    Z[j + n * j] = 1.0;
                    // 'ubils_search:282' s(j) = l(j);
                    s[j] = l[j];
                } else {
                    // 'ubils_search:283' else
                    // 'ubils_search:284' Z(j,j) = -1;
                    Z[j + n * j] = -1.0;
                    // 'ubils_search:285' s(j) = u(j);
                    s[j] = u[j];
                }
            }
            // 'ubils_search:289' R_R = R_temp * Z;
            t = m;
            b_m = n;
            vector<scalar> b_R(m * n, 0);
            for (j = 0; j < n; j++) {
                int boffset;
                int coffset = j * t;
                boffset = j * n;
                for (int b_i = 0; b_i < t; b_i++) {
                    b_R[coffset + b_i] = 0.0;
                }
                for (k = 0; k < b_m; k++) {
                    int aoffset = k * m;
                    int bkj = Z[boffset + k];
                    for (int b_i = 0; b_i < t; b_i++) {
                        i = coffset + b_i;
                        b_R[i] = b_R[i] + R_R[aoffset + b_i] * bkj;
                    }
                }
            }
            // 'ubils_search:290' y_B = y_B - R_temp*s;
            vector<scalar> C(m, 0);
            for (k = 0; k < n; k++) {
                for (i = 0; i < m; i++) {
                    C[i] = C[i] + R_R[k * m + i] * s[k];
                }
            }
            for (i = 0; i < m; i++) {
                C[i] = y_B[i] - C[i];
            }
            // 'ubils_search:291' u = u - l;
            vector<scalar> b_u(n, 0);
            for (i = 0; i < n; i++) {
                b_u[i] = u[i] - l[i];
            }
            // 'ubils_search:292' l = zeros(size(l));
            for (i = 0; i < n; i++) {
                l[i] = 0.0;
            }
            for (i = 0; i < n; i++) {
                u[i] = b_u[i];
            }
            // 'ubils_search:74' utemp = u;
            // 'ubils_search:76' k = n;
            index b_k = n;
            //  S(k,n) = y_B(k), k=1:m-1, S(k,n) = y_B(m), k=m:n
            //  S(m:n,n) = y_B(m);
            //  S(1:m-1,n) = y_B(1:m-1);
            // 'ubils_search:82' c(n) = y_B(m)/R_R(m,n);
            c[n - 1] = C[m - 1] / b_R[(m + b_m * (n - 1)) - 1];
            //  Compute new bound at level k
            // 'ubils_search:85' [l(k),u(k)] = bound(c(k),R_R,utemp,beta,prsd,k);
            bound(c[n - 1], b_R, b_u, beta, prsd, static_cast<double>(n),
                  &l[n - 1], &u[n - 1]);
            //  Find the initial integer in [l(k), u(k)]
            // 'ubils_search:88' [z(k),d(k),l(k),u(k)] = init(c(k),l(k),u(k));
            init(c[n - 1], l[n - 1], u[n - 1], &z[n - 1],
                 &d[n - 1], &l[n - 1], &u[n - 1]);
            //  dflag for down or up search direction
            // 'ubils_search:91' dflag = 1;
            t = 1;
            //  Intend to move down to a lower level
            // 'ubils_search:93' while 1
            while (1) {
                while (t == 1) {
                    // 'ubils_search:95' if l(k) ~= 1 || u(k) ~= 1
                    if ((l[b_k - 1] != 1.0) || (u[b_k - 1] != 1.0)) {
                        // 'ubils_search:96' if k ~= 1
                        if (b_k != 1) {
                            // 'ubils_search:111' if k <= m
                            if (b_k <= m) {
                                //                    for j = path(k1):-1:k1+1
                                //                        S(k1,j-1) = S(k1,j) - R_R(k1,j)*z(j);
                                //                    end
                                //                    c(k1) = S(k1,k1) / R_R(k1,k1);
                                // 'ubils_search:117' sum = y_B(k1) - R_R(k1,k1+1:n)*z(k1+1:n);
                                i = b_k - 1;
                                // 'ubils_search:118' c(k1) = sum / R(k1,k1);
                                scalar sum = 0.0;
                                for (index h = 0; h < n - b_k + 1; h++) {
                                    sum += b_R[(b_k + m * (i + h)) - 2] * z[(b_k + h) - 1];
                                }
                                c[b_k - 2] = (C[b_k - 2] - sum) / b_R[(b_k + m * (b_k - 2)) - 2];

                            } else {
                                // 'ubils_search:124' else
                                //  S(k,n) = y_B(k), k=1:m-1, S(k,n) = y_B(m), k=m:n
                                // S(m:n,n) = y_B(m);
                                // S(1:m-1,n) = y_B(1:m-1);
                                //                    S(m:k1,k1) = S(k1,k1+1) - R_R(m,k1+1)*z(k1+1);
                                //                    c(k1) = S(k1,k1) / R_R(m,k1);
                                // 'ubils_search:133' sum2 = (y_B(m) - R_R(m,k1+1:n)*z(k1+1:n));
                                index coffset;
                                if (b_k > n) {
                                    i = 0;
                                    coffset = 0;
                                    b_m = 1;
                                } else {
                                    i = b_k - 1;
                                    coffset = n;
                                    b_m = b_k;
                                }
                                // 'ubils_search:134' c(k1) = sum2 / R_R(m,k1);
                                scalar sum = 0.0;
                                for (index h = 0; h < coffset - i; h++) {
                                    sum += b_R[(m + b_m * (i + h)) - 1] * z[(b_m + h) - 1];
                                }
                                c[b_k - 2] = (C[m - 1] - sum) / b_R[(m + b_m * (b_k - 2)) - 1];
                            }
                            //  Compute new bound at level k1
                            // 'ubils_search:143' [l(k1),u(k1)] =
                            // bound(c(k1),R_R,utemp,beta,prsd,k1);
                            bound(c[b_k - 2], b_R, b_u, beta, prsd, b_k - 1.0, &l[b_k - 2], &u[b_k - 2]);
                            //  Find the initial integer in [l(k1), u(k1)]
                            // 'ubils_search:146' if l(k1) > u(k1)
                            if (l[b_k - 2] > u[b_k - 2]) {
                                // 'ubils_search:147' l(k1) = 1;
                                l[b_k - 2] = 1.0;
                                // 'ubils_search:148' u(k1) = 1;
                                u[b_k - 2] = 1.0;
                            } else {
                                // 'ubils_search:149' else
                                // 'ubils_search:150' [z(k1),d(k1),l(k1),u(k1)] =
                                // init(c(k1),l(k1),u(k1));
                                init(c[b_k - 2], l[b_k - 2], u[b_k - 2], &z[b_k - 2],
                                     &d[b_k - 2], &l[b_k - 2], &u[b_k - 2]);
                                // 'ubils_search:152' if k1 <= m
                                if (b_k - 1 <= m) {
                                    // 'ubils_search:153' gamma = R_R(k1,k1)*(c(k1)-z(k1));
                                    gamma = b_R[(b_k + b_m * (b_k - 2)) - 2] * (c[b_k - 2] - z[b_k - 2]);
                                    // 'ubils_search:154' prsd(k1)= prsd(k1+1) + gamma * gamma;
                                    prsd[b_k - 2] = prsd[b_k - 1] + gamma * gamma;
                                }
                            }
                            // 'ubils_search:158' k = k1;
                            b_k = static_cast<unsigned int>(b_k - 1);
                            //                ulevel = 0;
                        } else {
                            // 'ubils_search:160' else
                            //  A valid point is found
                            // 'ubils_search:161' zhat = z;
                            for (i = 0; i < n; i++) {
                                zhat[i] = z[i];
                            }
                            // 'ubils_search:162' beta = sqrt(prsd(1));
                            beta = std::sqrt(prsd[0]);
                            // 'ubils_search:163' for j = 1:n
                            for (j = 0; j < n; j++) {
                                // 'ubils_search:164' [l(j),u(j)] = bound(c(j),R_R,utemp,beta,prsd,j);
                                bound(c[j], b_R, b_u, beta, prsd, static_cast<double>(j) + 1.0,
                                      &l[j], &u[j]);
                            }
                            // 'ubils_search:166' dflag = 0;
                            t = 0;
                        }
                    } else {
                        // 'ubils_search:168' else
                        //  Will move back to a higher level
                        // 'ubils_search:169' dflag = 0;
                        t = 0;
                    }
                    // 'ubils_search:94' if dflag == 1
                }
                // 'ubils_search:171' else
                // 'ubils_search:172' if k == n
                if (b_k == n) {
                    break;
                } else {
                    // 'ubils_search:174' else
                    //  Move back to level k+1
                    //              if ulevel == 0
                    //                 ulevel = k;
                    //              end
                    // 'ubils_search:180' k = k + 1;
                    b_k++;
                    // 'ubils_search:182' if l(k) ~= 1 || u(k) ~= 1
                    if ((l[b_k - 1] != 1.0) || (u[b_k - 1] != 1.0)) {
                        double b_d;
                        double d1;
                        double d2;
                        //  Find a new integer at level k
                        // 'ubils_search:184' [z(k),d(k),l(k),u(k)] = ...
                        // 'ubils_search:185' update(z(k),d(k),l(k),u(k),l(k),u(k));
                        b_d = z[b_k - 1];
                        d1 = d[b_k - 1];
                        d2 = u[b_k - 1];
                        //
                        //  Find a new integer at level k and record it if it hits a boundary.
                        //
                        // 'ubils_search:300' if lflag_k == 0 && uflag_k == 0
                        if ((l[b_k - 1] == 0.0) &&
                            (u[b_k - 1] == 0.0)) {
                            double zk;
                            double zk_tmp;
                            // 'ubils_search:301' zk = z_k + d_k;
                            zk_tmp = d[b_k - 1];
                            zk = z[b_k - 1] + zk_tmp;
                            // 'ubils_search:302' if zk > u_k
                            if (zk > u[b_k - 1]) {
                                // 'ubils_search:303' uflag_k = 1;
                                d2 = 1.0;
                            } else if (zk < l[b_k - 1]) {
                                // 'ubils_search:304' elseif zk < l_k
                                // 'ubils_search:305' lflag_k = 1;
                                l[b_k - 1] = 1.0;
                            } else {
                                // 'ubils_search:306' else
                                // 'ubils_search:307' z_k = zk;
                                b_d = zk;
                                // 'ubils_search:308' if d_k > 0
                                if (zk_tmp > 0.0) {
                                    // 'ubils_search:309' d_k = -d_k - 1;
                                    d1 = -zk_tmp - 1.0;
                                } else {
                                    // 'ubils_search:310' else
                                    // 'ubils_search:311' d_k = -d_k + 1;
                                    d1 = -d[b_k - 1] + 1.0;
                                }
                            }
                        }
                        // 'ubils_search:316' if lflag_k == 1 && uflag_k == 0
                        if ((l[b_k - 1] == 1.0) && (d2 == 0.0)) {
                            // 'ubils_search:317' z_k = z_k + 1;
                            b_d++;
                            // 'ubils_search:318' if z_k > u_k
                            if (b_d > u[b_k - 1]) {
                                // 'ubils_search:319' uflag_k = 1;
                                d2 = 1.0;
                            }
                        } else if ((l[b_k - 1] == 0.0) && (d2 == 1.0)) {
                            // 'ubils_search:321' elseif lflag_k == 0 && uflag_k == 1
                            // 'ubils_search:322' z_k = z_k - 1;
                            b_d--;
                            // 'ubils_search:323' if z_k < l_k
                            if (b_d < l[b_k - 1]) {
                                // 'ubils_search:324' lflag_k = 1;
                                l[b_k - 1] = 1.0;
                            }
                        }
                        z[b_k - 1] = b_d;
                        d[b_k - 1] = d1;
                        u[b_k - 1] = d2;
                        // 'ubils_search:186' if k <= m
                        if (b_k <= m) {
                            // 'ubils_search:187' gamma = R_R(k,k)*(c(k)-z(k));
                            gamma = b_R[(b_k + b_m * (b_k - 1)) - 1] * (c[b_k - 1] - z[b_k - 1]);
                            // 'ubils_search:188' prsd(k)= prsd(k+1) + gamma * gamma;
                            prsd[b_k - 1] = prsd[b_k] + gamma * gamma;
                        }
                        // 'ubils_search:190' dflag = 1;
                        t = 1;
                    }
                }
            }
            //  The optimal solution has been found, terminate
            // 'ubils_search:196' if zhat == z0
            vector<bool> x(n, 0);
            for (i = 0; i < n; i++) {
                x[i] = (zhat[i] == z_x[i]);
            }
            bool b_y = helper::if_all_x_true<index>(x);
            if (b_y) {
                //  z0 is the optimal solution
                // 'ubils_search:197' z = zhat;
                for (i = 0; i < n; i++) {
                    z[i] = zhat[i];
                }
            } else {
                // 'ubils_search:198' else
                //  Shift the optimal solution back
                // 'ubils_search:199' z = Z*zhat + s;
                for (i = 0; i < n; i++) {
                    z[i] = 0.0;
                }
                for (k = 0; k < b_m; k++) {
                    for (i = 0; i < n; i++) {
                        z[i] += Z[k * n + i] * zhat[k];
                    }
                }
                for (i = 0; i < n; i++) {
                    z[i] = z[i] + s[i];
                }
            }
            return false;
        }

        void bound(double c_k, const scalar *R_R, double beta, const scalar *prsd, double k, double *l_k,
                   double *u_k) {
            double lambda_k, lambda_k_tmp, mu_k;
            //  ------------------------------------------------------------------
            //  --------  Local functions  ---------------------------------------
            //  ------------------------------------------------------------------
            //
            //  Compute new lower bound and upper bound for z_k
            //
            // 'ubils_search:190' m = size(R_R,1);
            // 'ubils_search:192' if k > m-1
            if (k > static_cast<double>(m) - 1.0) {
                int i;
                int i1;
                int i2;
                int loop_ub;
                // 'ubils_search:193' lambda_k = c_k - (beta+R_R(m,m:k-1)*u(m:k-1))/R_R(m,k);
                if (m > k - 1.0) {
                    i = 0;
                    i1 = 0;
                    i2 = 1;
                } else {
                    i = m - 1;
                    i1 = static_cast<int>(k - 1.0);
                    i2 = m;
                }
                lambda_k_tmp = 0.0;
                loop_ub = i1 - i;
                for (i1 = 0; i1 < loop_ub; i1++) {
                    lambda_k_tmp += R_R[(m + m * (i + i1)) - 1] * u[(i2 + i1) - 1];
                }
                lambda_k = c_k - (beta + lambda_k_tmp) / R_R[(m + m * (k - 1)) - 1];
                // 'ubils_search:194' mu_k = c_k + beta/R_R(m,k);
                mu_k =
                        c_k + beta / R_R[(m + m * (k - 1)) - 1];
            } else {
                // 'ubils_search:195' else
                // 'ubils_search:196' lambda_k = c_k - sqrt(beta^2-prsd(k+1))/R_R(k,k);
                lambda_k_tmp = std::sqrt(beta * beta - prsd[k]) / R_R[(k + m * (k - 1)) - 1];
                lambda_k = c_k - lambda_k_tmp;
                // 'ubils_search:197' mu_k = c_k + sqrt(beta^2-prsd(k+1))/R_R(k,k);
                mu_k = c_k + lambda_k_tmp;
            }
            // 'ubils_search:200' if lambda_k - floor(lambda_k) < 1e-12 && lambda_k ~= 0
            lambda_k_tmp = std::floor(lambda_k);
            if ((lambda_k - lambda_k_tmp < 1.0E-12) && (lambda_k != 0.0)) {
                // 'ubils_search:201' lambda_k = floor(lambda_k);
                lambda_k = lambda_k_tmp;
            }
            // 'ubils_search:204' if ceil(mu_k) - mu_k < 1e-12 && mu_k ~= u(k)
            lambda_k_tmp = std::ceil(mu_k);
            if ((lambda_k_tmp - mu_k < 1.0E-12) && (mu_k != 1.0)) {
                // 'ubils_search:205' mu_k = ceil(mu_k);
                mu_k = lambda_k_tmp;
            }
            // 'ubils_search:208' l_k = ceil(max(0,lambda_k));
            *l_k = std::ceil(std::fmax(0.0, lambda_k));
            // 'ubils_search:209' u_k = floor(min(u(k),mu_k));
            *u_k = std::floor(std::fmin(1.0, mu_k));
        }


        void init(double c_k, double l_k, double u_k, double *z_k, double *d_k,
                  double *lflag_k, double *uflag_k) {
            //
            //  Find the initial integer and the search direction at level _k
            //
            // 'ubils_search:217' z_k = round(c_k);
            *z_k = std::round(c_k);
            // 'ubils_search:218' if z_k <= l_k
            if (*z_k <= l_k) {
                // 'ubils_search:219' z_k = l_k;
                *z_k = l_k;
                // 'ubils_search:220' lflag_k = 1;
                *lflag_k = 1.0;
                //  The lower bound is reached
                // 'ubils_search:221' uflag_k = 0;
                *uflag_k = 0.0;
                // 'ubils_search:222' d_k = 1;
                *d_k = 1.0;
            } else if (*z_k >= u_k) {
                // 'ubils_search:223' elseif z_k >= u_k
                // 'ubils_search:224' z_k = u_k;
                *z_k = u_k;
                // 'ubils_search:225' uflag_k = 1;
                *uflag_k = 1.0;
                //  The upper bound is reached
                // 'ubils_search:226' lflag_k = 0;
                *lflag_k = 0.0;
                // 'ubils_search:227' d_k = -1;
                *d_k = -1.0;
            } else {
                // 'ubils_search:228' else
                // 'ubils_search:229' lflag_k = 0;
                *lflag_k = 0.0;
                // 'ubils_search:230' uflag_k = 0;
                *uflag_k = 0.0;
                // 'ubils_search:231' if c_k > z_k
                // 'ubils_search:233' else
                // 'ubils_search:234' d_k = -1;
                *d_k = -1.0;
            }
        }
    };
}
