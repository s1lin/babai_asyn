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

    template<typename scalar, typename index, index n>
    inline scalar cils<scalar, index, n>::ils_search_obils(const index n_dx_q_0, const index n_dx_q_1,
                                                           const vector<scalar> *y_B, vector<index> *z_x) {

        // Variables
        scalar sum, newprsd, gamma, beta = INFINITY;

        index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1, upper = pow(2, qam) - 1;
        index end_1 = n_dx_q_1 - 1, row_k = k + n_dx_q_0, diff = 0;
        index row_kk = n * end_1 + end_1, dflag = 1, count = 0, iter = 0;
        vector<scalar> p(dx, 0), c(dx, 0);
        vector<index> z(dx, 0), d(dx, 0), l(dx, 0), u(dx, 0);

        //Initial squared search radius
        scalar R_kk = R->x[n * end_1 + end_1];
        c[k] = y_B->at(row_k) / R_kk;
        z[k] = round(c[k]);
        if (z[k] <= 0) {
            z[k] = u[k] = 0; //The lower bound is reached
            l[k] = d[k] = 1;
        } else if (z[k] >= upper) {
            z[k] = upper; //The upper bound is reached
            u[k] = 1;
            l[k] = 0;
            d[k] = -1;
        } else {
            l[k] = u[k] = 0;
            //  Determine enumeration direction at level block_size
            d[k] = c[k] > z[k] ? 1 : -1;
        }

        gamma = R_kk * (c[k] - z[k]);
        //ILS search process
//        while (1) {
//            count++;
        for (count = 0; count < program_def::max_thre || iter == 0; count++) {
            if (dflag) {
                newprsd = p[k] + gamma * gamma;
                if (newprsd < beta) {
                    if (k != 0) {
                        k--;
                        row_k--;
                        sum = 0;
                        for (index col = k + 1; col < dx; col++) {
//                            sum += R->x[(col + n_dx_q_0) + n * row_k] * z[col];
                            sum += R->x[(col + n_dx_q_0) + n * row_k] * z[col];
                        }
                        R_kk = R->x[n * row_k + row_k];
                        p[k] = newprsd;
                        c[k] = (y_B->at(row_k) - sum) / R_kk;
                        z[k] = round(c[k]);
                        if (z[k] <= 0) {
                            z[k] = 0;
                            l[k] = 1;
                            u[k] = 0;
                            d[k] = 1;
                        } else if (z[k] >= upper) {
                            z[k] = upper;
                            u[k] = 1;
                            l[k] = 0;
                            d[k] = -1;
                        } else {
                            l[k] = 0;
                            u[k] = 0;
                            d[k] = c[k] > z[k] ? 1 : -1;
                        }
                        gamma = R_kk * (c[k] - z[k]);

                    } else {
                        beta = newprsd;
                        diff = 0;
                        iter++;
                        for (index h = 0; h < dx; h++) {
                            diff += z_x->at(h + n_dx_q_0) == z[h];
                            z_x->at(h + n_dx_q_0) = z[h];
                        }
                        if (iter > program_def::search_iter) {
                            break;
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
                    if (l[k] != 1 || u[k] != 1) {
                        z[k] += d[k];
                        if (z[k] == 0) {
                            l[k] = 1;
                            d[k] = -d[k] + 1;
                        } else if (z[k] == upper) {
                            u[k] = 1;
                            d[k] = -d[k] - 1;
                        } else if (l[k] == 1) {
                            d[k] = 1;
                        } else if (u[k] == 1) {
                            d[k] = -1;
                        } else {
                            d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                        }
                        gamma = R->x[n * row_k + row_k] * (c[k] - z[k]);
                        dflag = 1;
                    }
                }
            }
        }
        if (mode == 1) {
            cout << count << "," << iter << ";";
        }
        return count;
    }

    template<typename scalar, typename index, index n>
    inline scalar cils<scalar, index, n>::ils_search(const index n_dx_q_0, const index n_dx_q_1,
                                                     const vector<scalar> *y_B, vector<index> *z_x) {

        //variables
        scalar sum, newprsd, gamma, beta = INFINITY;

        index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1;
        index end_1 = n_dx_q_1 - 1, row_k = k + n_dx_q_0;
        index row_kk = n * end_1 + end_1;

        vector<scalar> p(dx, 0), c(dx, 0);
        vector<index> z(dx, 0), d(dx, 0);

        //Initial squared search radius
        scalar R_kk = R->x[n * end_1 + end_1];
        c[k] = y_B->at(row_k) / R_kk;
        z[k] = round(c[k]);
        gamma = R_kk * (c[k] - z[k]);

        //  Determine enumeration direction at level block_size
        d[k] = c[k] > z[k] ? 1 : -1;

        //ILS search process
        while (true) {
            newprsd = p[k] + gamma * gamma;
            if (newprsd < beta) {
                if (k != 0) {
                    k--;
                    row_k--;
                    sum = 0;
                    for (index col = k + 1; col < dx; col++) {
                        sum += R->x[n * (col + n_dx_q_0) + row_k] * z[col];
                    }
                    R_kk = R->x[n * row_k + row_k];
                    p[k] = newprsd;
                    c[k] = (y_B->at(row_k) - sum) / R_kk;
                    z[k] = round(c[k]);
                    gamma = R_kk * (c[k] - z[k]);

                    d[k] = c[k] > z[k] ? 1 : -1;

                } else {
                    beta = newprsd;
                    for (index l = 0; l < dx; l++) {
                        z_x->at(l + n_dx_q_0) = z[l];
                    }
                    z[0] += d[0];
                    gamma = R->x[0] * (c[0] - z[0]);
                    d[0] = d[0] > 0 ? -d[0] - 1 : -d[0] + 1;
                }
            } else {
                if (k == dx - 1) break;
                else {
                    k++;
                    row_k++;
                    z[k] += d[k];
                    gamma = R->x[n * row_k + row_k] * (c[k] - z[k]);
                    d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                }
            }
        }
        return beta;
    }

    template<typename scalar, typename index, index n>
    inline scalar cils<scalar, index, n>::ils_search_omp(const index n_dx_q_0, const index n_dx_q_1,
                                                         const scalar *y_B, index *z_x) {

        //variables
        scalar sum, newprsd, gamma, beta = INFINITY;

        index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1, iter = 0, count = 0, diff = 0;
        index row_k = k + n_dx_q_0, row_kk = row_k * (n - (row_k + 1) / 2);

        scalar p[dx] = {}, c[dx];
        index z[dx], d[dx] = {};

#pragma omp simd
        for (index col = 0; col < dx; col++) {
            z[col] = z_x[col + n_dx_q_0];
        }

        //Initial squared search radius
        scalar R_kk = R_A->x[row_kk + row_k];
        c[k] = (y_A->x[row_k] - y_B[row_k - n_dx_q_0]) / R_kk;
        z[k] = round(c[k]);
        gamma = R_kk * (c[k] - z[k]);

        //  Determine enumeration direction at level block_size
        d[k] = c[k] > z[k] ? 1 : -1;

        //ILS search process
        for (count = 0; count < program_def::max_search || iter == 0; count++) {
            newprsd = p[k] + gamma * gamma;
            if (newprsd < beta) {
                if (k != 0) {
                    k--;
                    row_k--;
                    sum = 0;
                    row_kk -= (n - row_k - 1);
#pragma omp simd reduction(+ : sum)
                    for (index col = k + 1; col < dx; col++) {
                        sum += R_A->x[row_kk + col + n_dx_q_0] * z[col];
                    }
                    R_kk = R_A->x[row_kk + row_k];
                    p[k] = newprsd;
                    c[k] = ((y_A->x[row_k] - y_B[row_k - n_dx_q_0]) - sum) / R_kk;
                    z[k] = round(c[k]);
                    gamma = R_kk * (c[k] - z[k]);
                    d[k] = c[k] > z[k] ? 1 : -1;

                } else {
                    beta = newprsd;
                    diff = 0;
                    for (index l = 0; l < dx; l++) {
                        diff += z_x[l + n_dx_q_0] == z[l];
                        z_x[l + n_dx_q_0] = z[l];
                    }
                    iter++;
                    if (iter > program_def::search_iter || diff == dx) break;

                    z[0] += d[0];
                    gamma = R_A->x[0] * (c[0] - z[0]);
                    d[0] = d[0] > 0 ? -d[0] - 1 : -d[0] + 1;
                }
            } else {
                if (k == dx - 1) break;
                else {
                    k++;
                    row_k++;
                    row_kk += n - row_k;
                    z[k] += d[k];
                    gamma = R_A->x[row_kk + row_k] * (c[k] - z[k]);
                    d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                }
            }
        }
        return beta;
    }

    template<typename scalar, typename index, index n>
    inline bool cils<scalar, index, n>::ils_search_obils_omp(const index n_dx_q_0, const index n_dx_q_1,
                                                             const index i, const index ds, scalar *R_S, index *z_x) {

        // Variables
        index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1, upper = pow(2, qam) - 1, iter = 0, dflag = 1, count = 0, diff = 0;
        index row_k = n_dx_q_1 - 1, row_kk = row_k * (n - n_dx_q_1 / 2), row_n, temp;

        scalar sum = 0, newprsd, gamma = 0, beta = INFINITY;

        scalar p[dx] = {}, c[dx], R_kk = R_A->x[row_kk + row_k], y_B[dx] = {};
        index z[dx], d[dx] = {}, l[dx] = {}, u[dx] = {};

#pragma omp simd
        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
            z[row - n_dx_q_0] = z_x[row];
            for (index col = 0; col < i; col++)
                y_B[row - n_dx_q_0] += R_S[row * ds + col];
        }

//        for (index row = 0; row < dx; row++) {
        //Initial squared search radius
        c[k] = (y_A->x[row_k] - y_B[k]) / R_kk;
        z[k] = round(c[k]);
        if (z[k] <= 0) {
            z[k] = u[k] = 0; //The lower bound is reached
            l[k] = d[k] = 1;
        } else if (z[k] >= upper) {
            z[k] = upper; //The upper bound is reached
            u[k] = 1;
            l[k] = 0;
            d[k] = -1;
        } else {
            l[k] = u[k] = 0;
            //  Determine enumeration direction at level block_size
            d[k] = c[k] > z[k] ? 1 : -1;
        }
        gamma = R_kk * (c[k] - z[k]);
//            newprsd = p[k] + gamma * gamma;
//            if (k != 0) {
//                k--;
//                row_k--;
//                row_kk -= (n - row_k - 1);
//                sum = 0;
//                for (index col = k + 1; col < dx; col++) {
//                    sum += R_A->x[row_kk + col + n_dx_q_0] * z[col];
//                }
//                R_kk = R_A->x[row_kk + row_k];
//                p[k] = newprsd;
//            }
//        }


        //ILS search process
        for (count = 0; count < program_def::max_search || iter == 0; count++) {
            if (dflag) {
                newprsd = p[k] + gamma * gamma;
                if (newprsd < beta) {
                    if (k != 0) {
                        k--;
                        row_k--;
                        row_kk -= (n - row_k - 1);
                        R_kk = R_A->x[row_kk + row_k];
                        p[k] = newprsd;
                        sum = 0;
                        for (index col = k + 1; col < dx; col++) {
                            sum += R_A->x[row_kk + col + n_dx_q_0] * z[col];
                        }
                        c[k] = (y_A->x[row_k] - y_B[k] - sum) / R_kk;
                        z[k] = round(c[k]);
                        if (z[k] <= 0) {
                            z[k] = u[k] = 0;
                            l[k] = d[k] = 1;
                        } else if (z[k] >= upper) {
                            z[k] = upper;
                            u[k] = 1;
                            l[k] = 0;
                            d[k] = -1;
                        } else {
                            l[k] = u[k] = 0;
                            d[k] = c[k] > z[k] ? 1 : -1;
                        }
                        gamma = R_kk * (c[k] - z[k]);

                    } else {
                        beta = newprsd;
                        diff = 0;
                        iter++;
#pragma omp simd
                        for (index h = 0; h < dx; h++) {
                            diff += z_x[h + n_dx_q_0] == z[h];
                            z_x[h + n_dx_q_0] = z[h];
                        }

                        if (iter > program_def::search_iter || diff == dx) {
                            break;
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
                    if (l[k] != 1 || u[k] != 1) {
                        z[k] += d[k];
                        if (z[k] == 0) {
                            l[k] = 1;
                            d[k] = -d[k] + 1;
                        } else if (z[k] == upper) {
                            u[k] = 1;
                            d[k] = -d[k] - 1;
                        } else if (l[k] == 1) {
                            d[k] = 1;
                        } else if (u[k] == 1) {
                            d[k] = -1;
                        } else {
                            d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                        }
                        gamma = R_A->x[row_kk + row_k] * (c[k] - z[k]);
                        dflag = 1;
                    }
                }
            }
        }
//        if (diff != dx){
//            temp = (n_dx_q_0 - 1) * (n - n_dx_q_0 / 2);
//            for (index row = 0; row < ds - i - 1; row++) {
//                for (index h = 0; h < dx; h++) {
//                    temp = row * dx + h;
//                    R_S[temp * ds + i] = 0;
//                    row_n = (n * temp) - ((temp * (temp + 1)) / 2);
//                    for (index col = n_dx_q_0; col < n_dx_q_1; col++) {
////                                  R_S[temp * ds + i] += R->x[temp + n * col] * z_x[col];
//                        R_S[temp * ds + i] += R_A->x[row_n + col] * z_x[col];
//                    }
//                }
//            }
//
//        }

        return diff == dx;
    }

    template<typename scalar, typename index, index n>
    inline bool cils<scalar, index, n>::ils_search_obils_omp2(const index n_dx_q_0, const index n_dx_q_1,
                                                              const index i, const index ds, const bool check,
                                                              scalar *R_S, scalar *y_B, index *z_x) {

        // Variables
        index dx = program_def::block_size, k = dx - 1, upper = pow(2, qam) - 1, iter = 0, dflag = 1, diff = 0;
        index row_k = n_dx_q_1 - 1, row_kk = row_k * (n - n_dx_q_1 / 2), row_n, temp, col, row, count;

        scalar sum = 0, newprsd, gamma = 0, beta = INFINITY, sum2 = 0;

        scalar p[dx] = {}, c[dx], R_kk = R_A->x[row_kk + row_k];
        index z[dx], d[dx] = {}, l[dx] = {}, u[dx] = {};

#pragma omp simd
        for (row = n_dx_q_0; row < n_dx_q_1; row++) {
            z[row - n_dx_q_0] = z_x[row];
//            y_b[row - n_dx_q_0] = y_A->x[row] - y_B[row];
//            y_B[row] = 0;
        }

        //Initial squared search radius
        c[k] = (y_B[row_k] - sum) / R_kk;
        z[k] = round(c[k]);
        if (z[k] <= 0) {
            z[k] = u[k] = 0; //The lower bound is reached
            l[k] = d[k] = 1;
        } else if (z[k] >= upper) {
            z[k] = upper; //The upper bound is reached
            u[k] = 1;
            l[k] = 0;
            d[k] = -1;
        } else {
            l[k] = u[k] = 0;
            //  Determine enumeration direction at level block_size
            d[k] = c[k] > z[k] ? 1 : -1;
        }
        gamma = R_kk * (c[k] - z[k]);

        //ILS search process
        for (count = 0; count < program_def::max_search || iter == 0; count++) {
//        while (1) {
            if (dflag) {
                newprsd = p[k] + gamma * gamma;
                if (newprsd < beta) {
                    if (k != 0) {
                        k--;
                        row_k--;
                        row_kk -= (n - row_k - 1);
                        R_kk = R_A->x[row_kk + row_k];
                        p[k] = newprsd;
                        sum = 0;
#pragma omp simd reduction(+ : sum)
                        for (col = k + 1; col < dx; col++) {
                            sum += R_A->x[row_kk + col + n_dx_q_0] * z[col];
                        }
                        c[k] = (y_B[row_k] - sum) / R_kk;
                        z[k] = round(c[k]);
                        if (z[k] <= 0) {
                            z[k] = u[k] = 0;
                            l[k] = d[k] = 1;
                        } else if (z[k] >= upper) {
                            z[k] = upper;
                            u[k] = 1;
                            l[k] = 0;
                            d[k] = -1;
                        } else {
                            l[k] = u[k] = 0;
                            d[k] = c[k] > z[k] ? 1 : -1;
                        }
                        gamma = R_kk * (c[k] - z[k]);

                    } else {
                        beta = newprsd;
                        diff = 0;
                        iter++;

                        for (index h = 0; h < dx; h++) {
                            diff += z_x[h + n_dx_q_0] == z[h];
//                            if (z_x[h + n_dx_q_0] != z[h]){
//                                for (row = 0; row < ds - i - 1; row++) {
//                                    for (index ll = 0; ll < dx; ll++) {
//                                        temp = row * dx + ll;
//                                        sum2 = 0;
//                                        row_n = (n * temp) - ((temp * (temp + 1)) / 2);
//#pragma omp simd reduction(+ : sum2)
//                                        for (col = n_dx_q_0; col < n_dx_q_1; col++) {
//                                             sum2 += R_A->x[row_n + col] * z_x[col];
//                                        }
//                                        R_S[temp * ds + i] = sum2;
//                                    }
//                                }
//                            }
                            z_x[h + n_dx_q_0] = z[h];
                        }
//                        R_S[temp * ds + i] = 0;
                        if (diff == dx || iter > program_def::search_iter || (!check)) {// || ||(!check)
                            break;
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
                    if (l[k] != 1 || u[k] != 1) {
                        z[k] += d[k];
                        if (z[k] == 0) {
                            l[k] = 1;
                            d[k] = -d[k] + 1;
                        } else if (z[k] == upper) {
                            u[k] = 1;
                            d[k] = -d[k] - 1;
                        } else if (l[k] == 1) {
                            d[k] = 1;
                        } else if (u[k] == 1) {
                            d[k] = -1;
                        } else {
                            d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                        }
                        gamma = R_A->x[row_kk + row_k] * (c[k] - z[k]);
                        dflag = 1;
                    }
                }
            }
        }
//        if (diff != dx){
//            temp = (n_dx_q_0 - 1) * (n - n_dx_q_0 / 2);
//            for (index row = 0; row < ds - i - 1; row++) {
//                for (index h = 0; h < dx; h++) {
//                    temp = row * dx + h;
//                    R_S[temp * ds + i] = 0;
//                    row_n = (n * temp) - ((temp * (temp + 1)) / 2);
//                    for (index col = n_dx_q_0; col < n_dx_q_1; col++) {
////                                  R_S[temp * ds + i] += R->x[temp + n * col] * z_x[col];
//                        R_S[temp * ds + i] += R_A->x[row_n + col] * z_x[col];
//                    }
//                }
//            }
//
//        }

        return diff == dx;
    }

    template<typename scalar, typename index, index n>
    class cils_ils {

    public:
        index upper;
        scalar *p, *c;
        index *z, *d, *l, *u;

        cils_ils(index qam) {

//            this->beta = new scalar[d_s.size()];
//            for (index h = 0; h < d_s.size(); h++) {
//                beta[h] = INFINITY;
//            }
            this->upper = pow(2, qam) - 1;
            this->p = new scalar[n];
            this->c = new scalar[n];
            this->z = new index[n];
            this->d = new index[n];
            this->l = new index[n];
            this->u = new index[n];
        }

        ~cils_ils() {
            delete[] p;
            delete[] c;
            delete[] z;
            delete[] d;
            delete[] l;
            delete[] u;
        }

        inline bool obils_omp(const index n_dx_q_0, const index n_dx_q_1,
                              const index i, const index check,
                              const scalarType <scalar, index> *R_A,
                              const scalar *y_B, index *z_x) {
            index dx = n_dx_q_1 - n_dx_q_0;
            index row_k = n_dx_q_1 - 1, row_kk = row_k * (n - n_dx_q_1 / 2);
            index dflag = 1, iter = 0, diff = 0, k = dx - 1;
            scalar R_kk = R_A->x[row_kk + row_k], beta = INFINITY;
//            beta[0] = INFINITY;
#pragma omp simd
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
                            R_kk = R_A->x[row_kk + row_k];
                            p[row_k] = newprsd;
                            sum = 0;
#pragma omp simd reduction(+ : sum)
                            for (index col = k + 1; col < dx; col++) {
                                sum += R_A->x[row_kk + col + n_dx_q_0] * z[col + n_dx_q_0];
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
                            if (diff == dx || iter > program_def::search_iter || !check) {// ||
                                break;
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
                            gamma = R_A->x[row_kk + row_k] * (c[row_k] - z[row_k]);
                            dflag = 1;
                        }
                    }
                }
            }
            return diff == dx;
        }
    };
}
