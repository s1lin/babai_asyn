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
 *   MERCA_tNTABILITY or FITNESS FOR cils.A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with CILS.  If not, see <http://www.gnu.org/licenses/>.
 */
namespace cils {

    template<typename scalar, typename index>
    class CILS_Init_Point {
    private:
        CILS <scalar, index> cils;
    public:
        b_matrix P, H;

        explicit CILS_Init_Point(CILS <scalar, index> &cils) {
            this->cils = cils;
            this->H.resize(cils.m, cils.n);
            this->H.assign(cils.A);
            this->P.resize(cils.n, cils.n);
            this->P.assign(cils.I);
        }

        /**
         *[s_bar_IP, v_norm, HH, Piv] = SIC_IP(H, y, N, bound) applies the SIC
         *initial point method
         *
         *Inputs:
         *    H - K-by-N real matrix
         *    y - K-dimensional real vector
         *    N - integer scalar
         *    bound - integer scalar for the constraint
         *
         *Outputs:
         *    s_bar_IP - N-dimensional integer vector for the initial point
         *    v_norm - real scalar for the norm of the residual vector
         *    corresponding to s_bar_IP
         *    HH - K-by-N real matrix, permuted H for sub-optimal methods
         *    Piv - N-by-N real matrix where HH*Piv'=H
         * @param x
         * @return
         */
        returnType <scalar, index> cils_sic_serial(b_vector &x) {

            //s_bar_IP=zeros(N,1);
            x.clear();
            //y_temp=y;
            b_vector y_tmp(cils.y_a);
            //Variable Declarations
            index k = -1, x_est = 0, x_round = 0;
            scalar res, max_res, hiy, hih, x_tmp;

            scalar time = omp_get_wtime();
            for (index j = cils.n - 1; j >= 0; j--) {
                max_res = INFINITY;
                for (index i = 0; i <= j; i++) {
                    auto Hi = column(this->H, i);
                    hiy = inner_prod(Hi, y_tmp);
                    hih = inner_prod(Hi, Hi);
                    x_tmp = hiy / hih;
                    x_round = max(min((int) round(x_tmp), cils.upper), cils.lower);
                    res = norm_2(y_tmp - x_round * Hi);
                    if (res < max_res) {
                        k = i;
                        x_est = x_round;
                        max_res = res;
                    }
                }
                if (k != -1) {
                    //HH(:,[k,j])=HH(:,[j,k]);
                    b_vector Hj = column(this->H, j);
                    column(this->H, j) = column(this->H, k);
                    column(this->H, k) = Hj;

                    //Piv(:,[k,j])=Piv(:,[j,k]);
                    b_vector Pj = column(P, j);
                    column(P, j) = column(P, k);
                    column(P, k) = Pj;

                    //s_bar_IP(j)=s_est;
                    x[j] = x_est;
                }
                y_tmp = y_tmp - x[j] * column(this->H, j);
            }

            scalar v_norm = norm_2(y_tmp);
            time = omp_get_wtime() - time;

            return {{}, time, v_norm};
        }


        returnType <scalar, index>
        cils_grad_proj(b_vector &x, const index search_iter) {
            b_vector ex;
//            std::vector<scalar> pj_1(cils.n, 0);
//            std::vector<scalar> q(cils.m, 0);
            std::vector<scalar> t_bar, t_seq;
            std::vector<index> r1, r2, r3, r4, r5;
            std::vector<bool> b_x_0, b_x_1(cils.n, 0);

            index i, j, k1, k2, k3;
            this->H.clear();
            this->P.assign(cils.I);
            this->H.assign(cils.A);
            scalar v_norm = INFINITY;
            //
            //  Find a solution to the box-constrained real least squares problem
            //  min_{cils.l<=x<=cils.u}||cils.y_a-Bx|| by the gradient projection method
            //
            //  Inputs:
            //     x - cils.n-dimensional real vector as an initial point
            //     cils.A - cils.m by cils.n real matrix
            //     cils.y_a - cils.m-dimensional real vector
            //     cils.l - cils.n-dimensional integer vector, lower bound
            //     cils.u - cils.n-dimensional integer vector, cils.upper bound
            //
            //  Output:
            //     x - cils.n-dimensional real vector, a solution
            // 'ubils_reduction:221' cils.n = length(x);
            // 'ubils_reduction:223' c = cils.A'*cils.y_a;


            b_vector x_cur(cils.n, 0);
            scalar time = omp_get_wtime();

            b_vector c = prod(trans(cils.A), cils.y_a);

            // 'ubils_reduction:225' for iter = 1:search_iter
            for (index iter = 0; iter < search_iter; iter++) {
                // 'ubils_reduction:227' g = cils.A'*(cils.A*x_cur-cils.y_a);
                b_vector r0 = -cils.y_a;
                r0 = r0 + prod(cils.A, x_cur);
                b_vector g = prod(trans(cils.A), r0);

                //  Check KKT conditions
                // 'ubils_reduction:230' if (x_cur==cils.l) == 0
                b_x_0.resize(cils.n, 0);
                for (i = 0; i < cils.n; i++) {
                    b_x_0[i] = !(x_cur[i] == cils.l[i]);
                }
                if (helper::if_all_x_true<index>(b_x_0)) {
                    // 'ubils_reduction:231' k1 = 1;
                    k1 = 1;
                } else {
                    k3 = 0;
                    for (i = 0; i < cils.n; i++) {
                        if (x_cur[i] == cils.l[i]) {
                            k3++;
                        }
                    }
                    r1.resize(k3, 0);
                    k3 = 0;
                    for (i = 0; i < cils.n; i++) {
                        if (x_cur[i] == cils.l[i]) {
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
                        // 'ubils_reduction:232' elseif (g(x_cur==cils.l) > -1.e-5) == 1
                        // 'ubils_reduction:233' k1 = 1;
                        k1 = 1;
                    } else {
                        // 'ubils_reduction:234' else
                        // 'ubils_reduction:234' k1 = 0;
                        k1 = 0;
                    }
                }
                // 'ubils_reduction:236' if (x_cur==cils.u) == 0
                b_x_0.resize(cils.n, 0);
                for (i = 0; i < cils.n; i++) {
                    b_x_0[i] = !(x_cur[i] == cils.u[i]);
                }
                if (helper::if_all_x_true<index>(b_x_0)) {
                    // 'ubils_reduction:237' k2 = 1;
                    k2 = 1;
                } else {
                    k3 = 0;
                    for (i = 0; i < cils.n; i++) {
                        if (x_cur[i] == cils.u[i]) {
                            k3++;
                        }
                    }
                    r2.resize(k3, 0);
                    k3 = 0;
                    for (i = 0; i < cils.n; i++) {
                        if (x_cur[i] == cils.u[i]) {
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
                        // 'ubils_reduction:238' elseif (g(x_cur==cils.u) < 1.e-5) == 1
                        // 'ubils_reduction:239' k2 = 1;
                        k2 = 1;
                    } else {
                        // 'ubils_reduction:240' else
                        // 'ubils_reduction:240' k2 = 0;
                        k2 = 0;
                    }
                }
                // 'ubils_reduction:242' if (cils.l<x_cur & x_cur<cils.u) == 0
                b_x_0.resize(cils.n, 0);
                for (i = 0; i < cils.n; i++) {
                    b_x_0[i] = ((!(cils.l[i] < x_cur[i])) || (!(x_cur[i] < cils.u[i])));
                }
                if (helper::if_all_x_true<index>(b_x_0)) {
                    // 'ubils_reduction:243' k3 = 1;
                    k3 = 1;
                } else {
                    b_x_0.resize(cils.n, 0);
                    for (i = 0; i < cils.n; i++) {
                        b_x_0[i] = (cils.l[i] < x_cur[i]);
                        b_x_1[i] = (x_cur[i] < cils.u[i]);
                    }
                    k3 = 0;
                    for (i = 0; i < cils.n; i++) {
                        if (b_x_0[i] && b_x_1[i]) {
                            k3++;
                        }
                    }
                    r3.resize(k3, 0);
                    k3 = 0;
                    for (i = 0; i < cils.n; i++) {
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
                        // 'ubils_reduction:244' elseif (g(cils.l<x_cur & x_cur<cils.u) < 1.e-5) == 1
                        // 'ubils_reduction:245' k3 = 1;
                        k3 = 1;
                    } else {
                        // 'ubils_reduction:246' else
                        // 'ubils_reduction:246' k3 = 0;
                        k3 = 0;
                    }
                }
                scalar v_norm_cur = helper::find_residual<scalar, index>(cils.A, x_cur, cils.y_a);
                // 'ubils_reduction:248' if (k1 & k2 & k3)
                if ((k1 != 0) && (k2 != 0) && (k3 != 0) && (v_norm > v_norm_cur)) {
                    k1 = k2 = k3 = 0;
                    for (i = 0; i < cils.n; i++) {
                        scalar x_est = round(x_cur[i]);
                        //x_est = 2.0 * std::floor(x_est / 2.0) + 1.0;
                        if (x_est < 0) {
                            x_est = 0;
                        } else if (x_est > cils.upper) {
                            x_est = cils.upper;
                        }
                        x[i] = x_est;
                    }
                    v_norm = v_norm_cur;
                } else {
                    scalar x_tmp;
                    //  Find the Cauchy point
                    // 'ubils_reduction:253' t_bar = 1.e5*ones(cils.n,1);
                    t_bar.resize(cils.n, 1e5);
                    // 'ubils_reduction:254' t_bar(g<0) = (x_cur(g<0)-cils.u(g<0))./g(g<0);
                    k3 = 0;
                    for (i = 0; i < cils.n; i++) {
                        if (g[i] < 0.0) {
                            k3++;
                        }
                    }
                    r4.resize(k3, 0);
                    k3 = 0;
                    for (i = 0; i < cils.n; i++) {
                        if (g[i] < 0.0) {
                            r4[k3] = i + 1;
                            k3++;
                        }
                    }
                    r0.resize(r4.size());
                    k3 = r4.size();
                    for (i = 0; i < k3; i++) {
                        r0[i] = (x_cur[r4[i] - 1] - cils.u[r4[i] - 1]) / g[r4[i] - 1];
                    }
                    k3 = 0;
                    for (i = 0; i < cils.n; i++) {
                        if (g[i] < 0.0) {
                            t_bar[i] = r0[k3];
                            k3++;
                        }
                    }
                    // 'ubils_reduction:255' t_bar(g>0) = (x_cur(g>0)-cils.l(g>0))./g(g>0);
                    k3 = 0;
                    for (i = 0; i < cils.n; i++) {
                        if (g[i] > 0.0) {
                            k3++;
                        }
                    }
                    r5.resize(k3, 0);
                    k3 = 0;
                    for (i = 0; i < cils.n; i++) {
                        if (g[i] > 0.0) {
                            r5[k3] = i + 1;
                            k3++;
                        }
                    }
                    r0.resize(r5.size());
                    k3 = r5.size();
                    for (i = 0; i < k3; i++) {
                        r0[i] = (x_cur[r5[i] - 1] - cils.l[r5[i] - 1]) / g[r5[i] - 1];
                    }
                    k3 = 0;
                    for (i = 0; i < cils.n; i++) {
                        if (g[i] > 0.0) {
                            t_bar[i] = r0[k3];
                            k3++;
                        }
                    }
                    //  Generate the ordered and non-repeated sequence of t_bar
                    // 'ubils_reduction:258' t_seq = unique([0;t_bar]);
                    r0.resize(cils.n + 1);
                    r0[0] = 0;
                    for (i = 0; i < cils.n; i++) {
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
                        //  Compute x_cur(t_{j-1})
                        // 'ubils_reduction:265' xt_j_1 = x_cur - min(tj_1,t_bar).*g;
                        //  Compute teh search direction p_{j-1}
                        // 'ubils_reduction:267' pj_1 = zeros(cils.n,1);
                        b_vector pj_1(cils.n, 0);
                        pj_1.clear();
                        // 'ubils_reduction:268' pj_1(tj_1<t_bar) = -g(tj_1<t_bar);
                        for (i = 0; i < t_bar.size(); i++) {
                            if (tj_1 < t_bar[i]) {
                                pj_1[i] = -g[i];
                            }
                        }
                        //  Compute coefficients
                        // 'ubils_reduction:270' q = cils.A*pj_1;
                        b_vector q = prod(cils.A, pj_1);
//                        q.clear();
//                        for (j = 0; j < cils.n; j++) {
//                            for (i = 0; i < cils.m; i++) {
//                                q[i] += cils.A[j * cils.m + i] * pj_1[j];
//                            }
//                        }
                        // 'ubils_reduction:271' fj_1d = (cils.A*xt_j_1)'*q - c'*pj_1;
                        ex.resize(t_bar.size());
                        for (j = 0; j < t_bar.size(); j++) {
                            ex[j] = std::fmin(tj_1, t_bar[j]);
                        }
                        ex.resize(cils.n);
                        for (i = 0; i < cils.n; i++) {
                            ex[i] = x_cur[i] - ex[i] * g[i];
                        }
                        r0.resize(cils.m);
                        r0 = prod(cils.A, ex);
//                        for (j = 0; j < cils.n; j++) {
//                            for (i = 0; i < cils.m; i++) {
//                                r0[i] += cils.A[j * cils.m + i] * ex[j];
//                            }
//                        }
                        scalar delta_t = 0.0;
                        for (i = 0; i < cils.m; i++) {
                            delta_t += r0[i] * q[i];
                        }
                        scalar fj_1d = 0.0;
                        for (i = 0; i < cils.n; i++) {
                            fj_1d += c[i] * pj_1[i];
                        }
                        fj_1d = delta_t - fj_1d;
                        // 'ubils_reduction:272' fj_1dd = q'*q;
                        // 'ubils_reduction:273' t = tj;
                        x_tmp = tj;
                        //  Find a local minimizer
                        // 'ubils_reduction:275' delta_t = -fj_1d/fj_1dd;
                        delta_t = 0.0;
                        for (i = 0; i < cils.m; i++) {
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
                    // 'ubils_reduction:285' x_cur = x_cur - min(t,t_bar).*g;
                    ex.resize(t_bar.size());
                    k3 = t_bar.size();
                    for (j = 0; j < k3; j++) {
                        ex[j] = std::fmin(x_tmp, t_bar[j]);
                    }
                    for (i = 0; i < cils.n; i++) {
                        x_cur[i] = x_cur[i] - ex[i] * g[i];
                    }
                }
            }
            //s_bar4 = round_int(s_bar4_unrounded, -1, 1);
//        for (i = 0; i < cils.n; i++) {
//            scalar x_est = round(x[i]);
//            //x_est = 2.0 * std::floor(x_est / 2.0) + 1.0;
//            if (x_est < 0) {
//                x_est = 0;
//            } else if (x_est > cils.upper) {
//                x_est = cils.upper;
//            }
//            x[i] = x_est;
//        }

            //scalar v_norm = helper::find_residual<scalar, index>(cils.m, cils.n, cils.A.data(), x.data(), cils.y_a.data());
            time = omp_get_wtime() - time;
            return {{}, time, v_norm};
        }

    };
}