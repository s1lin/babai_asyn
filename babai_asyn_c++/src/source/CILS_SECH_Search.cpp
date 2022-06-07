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
        index m, n, max_thre = INT_MAX;
        index lower, upper, search_iter;
        b_vector p, c, z, d, l, u;

    public:

        CILS_SECH_Search(index m, index n, index qam, index search_iter) {
            this->m = m;
            this->n = n;
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
            // Variables
            scalar time = omp_get_wtime();
            scalar sum, newprsd, gamma, beta = INFINITY;
            index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1;
            index end_1 = n_dx_q_1 - 1, row_k = k + n_dx_q_0, diff;
            index row_kk = n * end_1 + end_1, dflag = 1, count = 0, iter = 0;

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
//            cout << z[row_k] << endl;
            gamma = R_kk * (c[row_k] - z[row_k]);
            //ILS search process
            while (iter < max_thre) {
                iter++;
                if (omp_get_wtime() - time > 10) {
                    cout << "BREAK DUE TO RUNTIME";
                    break;
                }
                if (dflag) {
                    newprsd = p[row_k] + gamma * gamma;
                    if (newprsd < beta) {
                        if (k != 0) {
                            k--;
                            row_k--;
                            sum = y_B[row_k];
                            for (index col = k + 1; col < dx; col++) {
                                sum -= R_R[(col + n_dx_q_0) * n + row_k] * z[col + n_dx_q_0];
                            }

                            R_kk = R_R(row_k, row_k);
                            p[row_k] = newprsd;
                            c[row_k] = (sum) / R_kk;
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
//                            cout <<z[row_k]<< ",sum:" << sum << endl;
                        } else {
                            beta = newprsd;
                            diff = 0;

                            for (index h = n_dx_q_0; h < n_dx_q_1; h++) {
                                diff += z_x[h] == z[h];
                                z_x[h] = z[h];
                            }

//                            if (time > 10)
//                                break;
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
                            gamma = R_R(row_k, row_k) * (c[row_k] - z[row_k]);
                            dflag = 1;
                        }
                    }
                }
            }

            return diff;
        }

        searchType<scalar, index>
        mch(const index n_dx_q_0, const index n_dx_q_1, const b_vector &R_A, const b_vector &y_B, b_vector &z_x,
            index T, const index case1, index case2, scalar beta) {
            scalar time = omp_get_wtime();
            index dx = n_dx_q_1 - n_dx_q_0;
            index row_k = n_dx_q_1 - 1, row_kk = (row_k * (2 * n - n_dx_q_1)) / 2;
            index dflag = 1, iter = 0, diff = 0, k = dx - 1;
            scalar R_kk = R_A[row_kk + row_k], newprsd, sum, gamma;
            if (case1) {
                beta = INFINITY;
            }

            for (index R_R = n_dx_q_0; R_R < n_dx_q_1; R_R++) {
                z[R_R] = z_x[R_R];
            }

            if (!case2) {
                //Determine enumeration direction at level block_size
                //Initial squared search radius
                gamma = R_kk * (c[row_k] - z[row_k]);

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
            }
            index iter2 = 0;
            while (iter2 < max_thre) {
                iter2++;
                if (omp_get_wtime() - time > 10) {
                    cout << "BREAK DUE TO RUNTIME";
                    break;
                }
                if (dflag) {
                    newprsd = p[row_k] + gamma * gamma;
                    if (newprsd < beta && !case2) {
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
                            if (!case2) {
                                beta = newprsd;
                                diff = 0;
                                iter++;
                                for (index h = n_dx_q_0; h < n_dx_q_1; h++) {
                                    diff += z_x[h] == z[h];
                                    z_x[h] = z[h];
                                }
                                if (iter == T)
                                    break;
                            }
                            case2 = false;
                            k = 0;
                            row_k = n_dx_q_0;
                            row_kk = (row_k * (2 * n - n_dx_q_1)) / 2;
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
            return {diff == dx, beta, T - iter};
        }

        bool mch2(const index n_dx_q_0, const index n_dx_q_1, const index i, const index check,
                  const b_vector &R_A, const b_vector &y_B, b_vector &z_x) {
            scalar time = omp_get_wtime();
            index dx = n_dx_q_1 - n_dx_q_0;
            index row_k = n_dx_q_1 - 1, row_kk = (row_k * (2 * n - n_dx_q_1))/2;
            index dflag = 1, iter = 0, diff = 0, k = dx - 1;
            scalar R_kk = R_A[row_kk + row_k], newprsd, sum, gamma;
            scalar beta = INFINITY;

            //Determine enumeration direction at level block_size
            //Initial squared search radius
            gamma = R_kk * (c[row_k] - z[row_k]);

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

            while (true) {
                if (omp_get_wtime() - time > 10) {
                    cout << "BREAK DUE TO RUNTIME";
                    break;
                }
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
                            for (index h = n_dx_q_0; h < n_dx_q_1; h++) {
                                diff += z_x[h] == z[h];
                                z_x[h] = z[h];
                            }
//                            k = 0;
//                            row_k = n_dx_q_0;
//                            row_kk = row_k * (n - n_dx_q_1 / 2);
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

        bool se(const index n_dx_q_0, const index n_dx_q_1, const bool check,
                const b_matrix &R_R, const b_vector &y_B, b_vector &z_x) {

            //variables
            scalar sum, newprsd, gamma, beta = INFINITY;

            index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1;
            index end_1 = n_dx_q_1 - 1, row_k = k + n_dx_q_0, diff;
            index row_kk = n * end_1 + end_1, count, iter = 0;

            //Initial squared search radius
            scalar R_kk = R_R[row_kk];
            c[row_k] = y_B[row_k] / R_kk;
            z[row_k] = round(c[row_k]);
            gamma = R_kk * (c[row_k] - z[row_k]);

            //  Determine enumeration direction at level block_size
            d[row_k] = c[row_k] > z[row_k] ? 1 : -1;

            //ILS search process
            while (true) {
//            for (count = 0; count < max_thre || iter == 0; count++) {
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
                            diff += z_x[h] == z[h];
                            z_x[h] = z[h];
                        }
//                        if (n_dx_q_1 != n) {
//                            if (diff == dx || iter > search_iter || !check) {
//                                break;
//                            }
//                        }
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


        searchType<scalar, index>
        mse(const index n_dx_q_0, const index n_dx_q_1, const b_vector &R_A, const b_vector &y_B, b_vector &z_x,
            index T, const index case1, index case2, scalar beta) {
            index dx = n_dx_q_1 - n_dx_q_0;
            index row_k = n_dx_q_1 - 1, row_kk = row_k * (n - n_dx_q_1 / 2);
            index dflag = 1, iter = 0, diff = 0, k = dx - 1;
            scalar R_kk = R_A[row_kk + row_k], newprsd, sum, gamma;
            if (case1) {
                beta = INFINITY;
            }

            for (index R_R = n_dx_q_0; R_R < n_dx_q_1; R_R++) {
                z[R_R] = z_x[R_R];
            }

            if (!case2) {
                c[row_k] = y_B[row_k] / R_kk;
                z[row_k] = round(c[row_k]);

                //  Determine enumeration direction at level block_size
                d[row_k] = c[row_k] > z[row_k] ? 1 : -1;
                gamma = R_kk * (c[row_k] - z[row_k]);
            }

            //ILS search process
            while (true) {
                newprsd = p[row_k] + gamma * gamma;
                if (newprsd < beta && !case2) {
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
                        if (!case2) {
                            beta = newprsd;
                            diff = 0;
                            for (index h = n_dx_q_0; h < n_dx_q_1; h++) {
                                diff += z_x[h] == z[h];
                                z_x[h] = z[h];
                            }
                            iter++;
                            if (iter == T)
                                break;
                        }
                        case2 = false;
                        k = 0;
                        row_k = n_dx_q_0;
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
            return {diff != dx, beta, T - iter};
        }
    };
}
//Collectors:
/*
 *  row_n = (n_dx_q_0 - 1) * (n - n_dx_q_0 / 2) + dx * n - (n_dx_q_0 + n_dx_q_1 - 1) * dx / 2;
 *  for (index row = row_k; row >= n_dx_q_0; row--) {
 *    sum = 0;
 *    for (index col = n_dx_q_1; col < n; col++) {
 *        sum += R_A[col + row_n] * z_x[col];
 *    }
 *    y_B[row] = y_bar[row] - sum;
 *    cout << row_n << ", ";
 *    row_n -= n - row;
 *  }
 *  cout << endl;
 *  helper::display<scalar, index>(subrange(y_B, n_dx_q_0, n_dx_q_1), "backward");
 */
