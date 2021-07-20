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

    template<typename scalar, typename index, index m, index n>
    class cils_ils {

    public:
        index upper;
        scalar *p, *c, *z;
        index *d, *l, *u;

        cils_ils(index qam) {

            this->upper = pow(2, qam) - 1;
            this->p = new scalar[n]();
            this->c = new scalar[n]();
            this->z = new scalar[n]();
            this->d = new index[n]();
            this->l = new index[n]();
            this->u = new index[n]();
        }

        ~cils_ils() {
            delete[] p;
            delete[] c;
            delete[] z;
            delete[] d;
            delete[] l;
            delete[] u;
        }

        inline bool obils_search(const index n_dx_q_0, const index n_dx_q_1, const bool check,
                                 const array<scalar, m * n> &R_R, const array<scalar, 1 * n> &y_B, vector<scalar> *z_x) {

            // Variables
            scalar sum, newprsd, gamma, beta = INFINITY;

            index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1;
            index end_1 = n_dx_q_1 - 1, row_k = k + n_dx_q_0, diff = 0;
            index row_kk = n * end_1 + end_1, dflag = 1, count = 0, iter = 0;

            //Initial squared search radius
            scalar R_kk = R_R[n * end_1 + end_1];
            c[row_k] = y_B[row_k] / R_kk;
            z[row_k] = round(c[row_k]);
            if (z[row_k] <= 0) {
                z[row_k] = u[row_k] = 0; //The lower bound is reached
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
//        while (1) {
//            count++;
            for (count = 0; count < program_def::max_thre || iter == 0; count++) {
                if (dflag) {
                    newprsd = p[row_k] + gamma * gamma;
                    if (newprsd < beta) {
                        if (k != 0) {
                            k--;
                            row_k--;
                            sum = 0;
                            for (index col = k + 1; col < dx; col++) {
                                sum += R_R[(col + n_dx_q_0) * n + row_k] * z[col + n_dx_q_0];
                            }
                            R_kk = R_R[n * row_k + row_k];
                            p[row_k] = newprsd;
                            c[row_k] = (y_B[row_k] - sum) / R_kk;
                            z[row_k] = round(c[row_k]);
                            if (z[row_k] <= 0) {
                                z[row_k] = 0;
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
                                diff += z_x->at(h) == z[h];
                                z_x->at(h) = z[h];
                            }
                            if (n_dx_q_1 != n) {
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
                            gamma = R_R[n * row_k + row_k] * (c[row_k] - z[row_k]);
                            dflag = 1;
                        }
                    }
                }
            }
            return count;
        }

        inline bool obils_search_omp(const index n_dx_q_0, const index n_dx_q_1, const index i, const index check,
                                     const array<scalar, m * (n + 1) / 2> &R_A, const array<scalar, 1 * n> &y_B, scalar *z_x) {
            index dx = n_dx_q_1 - n_dx_q_0;
            index row_k = n_dx_q_1 - 1, row_kk = row_k * (n - n_dx_q_1 / 2);
            index dflag = 1, iter = 0, diff = 0, k = dx - 1;
            scalar R_kk = R_A[row_kk + row_k], beta = INFINITY;

//#pragma omp simd
            for (index r = n_dx_q_0; r < n_dx_q_1; r++) {
                z[r] = z_x[r];
            }

            c[row_k] = y_B[row_k] / R_kk;
            z[row_k] = round(c[row_k]);
            if (z[row_k] <= 0) {
                z[row_k] = u[row_k] = 0; //The lower bound is reached
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
                            if (z[row_k] <= 0) {
                                z[row_k] = u[row_k] = 0;
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
                               const array<scalar, m * n> &R_R, const array<scalar, 1 * n> &y_B, vector<scalar> *z_x) {

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
                        for (index col = k + 1; col < dx; col++) {
                            sum += R_R[(col + n_dx_q_0) * n + row_k] * z[col + n_dx_q_0];
                        }
                        R_kk = R_R[n * row_k + row_k];
                        p[row_k] = newprsd;
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
                        gamma = R_R[row_k] * (c[row_k] - z[row_k]);
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
                                   const array<scalar, m * (n + 1) / 2> &R_A, const array<scalar, 1 * n> &y_B, scalar *z_x) {

            //variables
            index dx = n_dx_q_1 - n_dx_q_0;
            index row_k = n_dx_q_1 - 1, row_kk = row_k * (n - n_dx_q_1 / 2);
            index iter = 0, diff = 0, k = dx - 1;
            scalar R_kk = R_A[row_kk + row_k], beta = INFINITY;

#pragma omp simd
            for (index r = n_dx_q_0; r < n_dx_q_1; r++) {
                z[r] = z_x[r];
            }

            //Initial squared search radius
            c[row_k] = y_B[row_k] / R_kk;
            z[row_k] = round(c[row_k]);
            scalar gamma = R_kk * (c[row_k] - z[row_k]);
            scalar newprsd, sum;

            //  Determine enumeration direction at level block_size
            d[row_k] = c[row_k] > z[row_k] ? 1 : -1;

            //ILS search process
            for (index count = 0; count < program_def::max_search || iter == 0; count++) {
                newprsd = p[row_k] + gamma * gamma;
                if (newprsd < beta) {
                    if (k != 0) {
                        k--;
                        row_k--;
                        sum = 0;
                        row_kk -= (n - row_k - 1);
#pragma omp simd reduction(+ : sum)
                        for (index col = k + 1; col < dx; col++) {
                            sum += R_A[row_kk + col + n_dx_q_0] * z[col + n_dx_q_0];
                        }
                        R_kk = R_A[row_kk + row_k];
                        p[row_k] = newprsd;
                        c[row_k] = ((y_B[row_k] - y_B[row_k - n_dx_q_0]) - sum) / R_kk;
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
                        gamma = R_A[row_k] * (c[row_k] - z[row_k]);
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
