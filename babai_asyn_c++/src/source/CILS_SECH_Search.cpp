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
    class CILS_SECH_Search {

    private:
        index m, n, max_thre = 1e6;
        index lower, upper;
        b_vector p, c, z, d, l, u;
        CILS <scalar, index> cils;

    public:

        explicit CILS_SECH_Search(CILS <scalar, index> &cils) {
            this->cils = cils;
            this->m = cils.m;
            this->n = cils.n;
            //pow(2, qam) - 1;
            this->p.resize(n);
            this->p.clear();
            this->c.resize(n);
            this->c.clear();
            this->z.resize(n);
            this->z.clear();
            this->d.resize(n);
            this->d.clear();
            this->l.resize(n);
            this->u.resize(n);
            this->l.clear();
            this->u.clear();
            this->lower = 0;
            this->upper = cils.upper;
        }

        CILS_SECH_Search(index m, index n, index qam) {
            this->m = cils.m;
            this->n = cils.n;
            //pow(2, qam) - 1;
            this->p.resize(n);
            this->p.clear();
            this->c.resize(n);
            this->c.clear();
            this->z.resize(n);
            this->z.clear();
            this->d.resize(n);
            this->d.clear();
            this->l.resize(n);
            this->u.resize(n);
            this->l.clear();
            this->u.clear();
            this->lower = 0;
            this->upper = pow(2, qam) - 1;
        }


        bool ch(const index n_dx_q_0, const index n_dx_q_1, const bool check,
                const b_matrix &R_R, const b_vector &y_B, b_vector &z_x) {
            this->z.clear();
            // Variables
            scalar sum, newprsd, gamma, beta = INFINITY;
            index dx = n_dx_q_1 - n_dx_q_0;
            index diff = 0, row_k = n_dx_q_1 - 1;
            index dflag = 1, count = 0, iter = 0;

            //Initial squared search radius
            scalar R_kk = R_R(row_k, row_k);
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
            while (true) {
//            count++;
//            for (count = 0; count < cils.search_iter || iter == 0; count++) {
                if (dflag) {
                    newprsd = p[row_k] + gamma * gamma;
                    //cout << "newprsd" << newprsd << endl;
                    if (newprsd < beta) {
                        if (row_k != n_dx_q_0) {
                            row_k--;
                            sum = 0;
                            for (index col = row_k + 1; col < n_dx_q_1; col++) {
                                sum += R_R(row_k, col) * z[col];
                            }
                            R_kk = R_R(row_k, row_k);
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
//                                if (diff == dx || iter > search_iter || !check) {
//                                    break;
//                                }
//                            }
                        }
                    } else {
                        dflag = 0;
                    }

                } else {
                    if (row_k == n_dx_q_1 - 1) break;
                    else {
//                        k++;
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
                            gamma = R_R(row_k, row_k) * (c[row_k] - z[row_k]);
                            dflag = 1;
                        }
                    }
                }
            }

            return diff;
        }

        bool mch(const index n_dx_q_0, const index n_dx_q_1, const index i, const index check,
                 const b_vector &R_A, const b_vector &y_bar, b_vector &z_x) {
            index dx = n_dx_q_1 - n_dx_q_0;
            index row_k = n_dx_q_1 - 1, row_kk = row_k * (n - n_dx_q_1 / 2);
            index dflag = 1, iter = 0, diff = 0, k = dx - 1;
            scalar R_kk = R_A[row_kk + row_k], beta = INFINITY;
            scalar newprsd, sum;
            b_vector y_B(m, 0);

            index row_n = (n_dx_q_0 - 1) * (cils.n - n_dx_q_0 / 2);
            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                sum = 0;
                row_n += cils.n - row;
#pragma omp simd reduction(+:sum)
                for (index col = n_dx_q_1; col < cils.n; col++) {
                    sum += R_A[col + row_n] * z_x[col];
                }
                y_B[row] = y_bar[row] - sum;
            }
            
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

            //ILS search process
//            for (index count = 0; count < cils.search_iter || iter == 0; count++) {
            while (1) {
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

//                            if (i != 0) {
//                                if (diff == dx || iter > cils.search_iter || !check) {
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
            return true;//diff == dx;
        }

        bool mch2(const index n_dx_q_0, const index n_dx_q_1, const index i, const index check,
                  const b_vector &R_A, const b_vector &y_bar, b_vector &z_x) {

            index dx = n_dx_q_1 - n_dx_q_0;
            index row_k = n_dx_q_1 - 1, row_kk = row_k * (n - n_dx_q_1 / 2), row_n;
            index dflag = 1, iter = 0, diff = 0, k = dx - 1;
            scalar R_kk = R_A[row_kk + row_k], beta = INFINITY, y_b;
            b_vector y_B(m, 0);
            scalar newprsd, sum, gamma;
            index row_n_start =
                    (n_dx_q_0 - 1) * (cils.n - n_dx_q_0 / 2) + dx * cils.n - (n_dx_q_0 + n_dx_q_1 - 1) * dx / 2;
            row_n = row_n_start;
            y_b = y_bar[row_k];


            for (index col = n_dx_q_1; col < cils.n; col++) {
                y_b -= R_A[col + row_n] * z_x[col];
            }

            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                z[row] = z_x[row];
            }


            c[row_k] = y_b / R_kk;
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
            gamma = R_kk * (c[row_k] - z[row_k]);


            //ILS search process
            for (index count = 0; count < cils.search_iter || iter == 0; count++) {
//            while (true) {
                if (dflag) {
                    newprsd = p[row_k] + gamma * gamma;
                    if (newprsd < beta) {
                        if (k != 0) {
                            k--;
                            row_n -= n - row_k;
                            row_kk -= n - row_k;
                            row_k--;
                            R_kk = R_A[row_kk + row_k];
                            p[row_k] = newprsd;
                            sum = 0;

                            y_b = y_bar[row_k];

                            for (index col = n_dx_q_1; col < cils.n; col++) {
                                y_b -= R_A[col + row_n] * z_x[col];
                            }

                            for (index col = k + 1 + n_dx_q_0; col < n_dx_q_1; col++) {
                                y_b -= R_A[row_kk + col] * z[col];
                            }

                            c[row_k] = y_b / R_kk;
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

                            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                                diff += z_x[row] == z[row];
                                z_x[row] = z[row];
                            }

                            if (i != 0) {
                                if (diff == dx || (iter > 1 && !check)) {
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
                        row_n += n - row_k;
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


        bool se(const index n_dx_q_0, const index n_dx_q_1, const bool check,
                const b_matrix &R_R, const b_vector &y_B, b_vector &z_x) {

            //variables
            scalar sum, newprsd, gamma, beta = INFINITY;

            index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1;
            index end_1 = n_dx_q_1 - 1, row_k = k + n_dx_q_0, diff;
            index row_kk = n * end_1 + end_1, count, iter = 0;

            //Initial squared search radius
            scalar R_kk = R_R(n * end_1 + end_1);
            c[row_k] = y_B[row_k] / R_kk;
            z[row_k] = round(c[row_k]);
            gamma = R_kk * (c[row_k] - z[row_k]);

            //  Determine enumeration direction at level block_size
            d[row_k] = c[row_k] > z[row_k] ? 1 : -1;

            //ILS search process
            for (count = 0; count < max_thre || iter == 0; count++) {
                newprsd = p[row_k] + gamma * gamma;
                if (newprsd < beta) {
                    if (k != 0) {
                        k--;
                        row_k--;
                        sum = 0;
                        R_kk = R_R(n * row_k + row_k);
                        p[row_k] = newprsd;
                        for (index col = k + 1; col < dx; col++) {
                            sum += R_R((col + n_dx_q_0) * n + row_k) * z[col + n_dx_q_0];
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
                        if (n_dx_q_1 != n) {
                            if (diff == dx || iter > cils.search_iter || !check) {
                                break;
                            }
                        }
                        z[row_k] += d[row_k];
                        gamma = R_R(n * row_k + row_k) * (c[row_k] - z[row_k]);
                        d[row_k] = d[row_k] > row_k ? -d[row_k] - 1 : -d[row_k] + 1;
                    }
                } else {
                    if (k == dx - 1) break;
                    else {
                        k++;
                        row_k++;
                        z[row_k] += d[row_k];
                        gamma = R_R(n * row_k + row_k) * (c[row_k] - z[row_k]);
                        d[row_k] = d[row_k] > 0 ? -d[row_k] - 1 : -d[row_k] + 1;
                    }
                }
            }
            return beta;
        }


        bool mse(const index n_dx_q_0, const index n_dx_q_1, const index i, const index check,
                 const b_vector &R_A, const b_vector &y_B, b_vector &z_x) {

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
            for (index count = 0; count < cils.search_iter || iter == 0; count++) {
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
                            if (diff == dx || iter > cils.search_iter || !check) {
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

    };
}
//Collectors:
/*
 *  row_n = (n_dx_q_0 - 1) * (cils.n - n_dx_q_0 / 2) + dx * cils.n - (n_dx_q_0 + n_dx_q_1 - 1) * dx / 2;
 *  for (index row = row_k; row >= n_dx_q_0; row--) {
 *    sum = 0;
 *    for (index col = n_dx_q_1; col < cils.n; col++) {
 *        sum += R_A[col + row_n] * z_x[col];
 *    }
 *    y_B[row] = y_bar[row] - sum;
 *    cout << row_n << ", ";
 *    row_n -= cils.n - row;
 *  }
 *  cout << endl;
 *  helper::display<scalar, index>(subrange(y_B, n_dx_q_0, n_dx_q_1), "backward");
 */
