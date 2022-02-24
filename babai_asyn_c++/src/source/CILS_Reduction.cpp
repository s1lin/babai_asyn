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
 *   MERCB_tNTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with CILS.  If not, see <http://www.gnu.org/licenses/>.
 */

namespace cils {

    template<typename scalar, typename index>
    class CILS_Reduction {
    private:
        index m, n, upper, lower;
        bool verbose{}, eval{};

        /**
         * Evaluating the LLL decomposition
         * @return
         */
        returnType <scalar, index> lll_validation() {
            printf("====================[ TEST | LLL_VALIDATE ]==================================\n");
            printf("[ INFO: in LLL validation]\n");
            scalar sum, error = 0, det;
            b_matrix B_T = prod(B, Z);
            b_matrix R_I;

            helper::inv<scalar, index>(R_R, R_I);

            b_matrix Q_Z = prod(B_T, R_I);
            b_vector y_R = prod(trans(Q_Z), y);
            if (verbose && n <= 32) {
                printf("\n[ Print R:]\n");
                helper::display<scalar, index>(R_R, "R_R");
                printf("\n[ Print Z:]\n");
                helper::display<scalar, index>(Z, "Z");
                printf("\n[ Print Q'Q:]\n");
                helper::display<scalar, index>(prod(trans(Q_Z), Q_Z), "Q_Z");
                printf("\n[ Print Q'y:]\n");
                helper::display<scalar, index>(m, y_R, "y_R");
                printf("\n[ Print y_r:]\n");
                helper::display<scalar, index>(m, y_r, "y_r");
            }

            for (index i = 0; i < m; i++) {
                error += fabs(y_R(i) - y_r(i));
            }
            printf("LLL Error: %8.5f\n", error);

            index pass = true;
            std::vector<scalar> fail_index(n, 0);
            index k = 1, k1;
            scalar zeta, alpha;

            while (k < n) {
                k1 = k - 1;
                zeta = round(R_R(k1, k) / R_R(k1, k1));
                alpha = R_R(k1, k) - zeta * R_R(k1, k1);

                if (pow(R_R(k1, k1), 2) > (1 + 1.e-10) * (pow(alpha, 2) + pow(R_R(k, k), 2))) {
                    pass = false;
                    fail_index[k] = 1;
                }
                k++;
            }
            if (!pass) {
                cout << "[ERROR: LLL Failed on index:";
                for (index i = 0; i < n; i++) {
                    if (fail_index[i] != 0)
                        cout << i << ",";
                }
                cout << "]" << endl;
            }

            printf("====================[ END | LLL_VALIDATE ]==================================\n");
            return {fail_index, det, (scalar) pass};
        }

        /**
         * Evaluating the QR decomposition
         * @tparam scalar
         * @tparam index
         * @tparam n
         * @param B
         * @param Q
         * @param R
         * @param eval
         * @return
         */
        scalar qr_validation() {
            printf("====================[ TEST | QR_VALIDATE ]==================================\n");
            printf("[ INFO: in QR validation]\n");
            index i, j, k;
            scalar sum, error = 0;

            if (eval == 1) {
                //b_vector B_T(m * n, 0);
                //helper::mtimes_v<scalar, index>(m, n, Q, R, B_T);
                b_matrix B_T = prod(Q, R);
                if (verbose && n <= 16) {
                    printf("\n[ Print Q:]\n");
                    helper::display<scalar, index>(Q, "Q");
                    printf("\n[ Print R:]\n");
                    helper::display<scalar, index>(R, "R");
                    printf("\n[ Print B:]\n");
                    helper::display<scalar, index>(B, "B");
                    printf("\n[ Print Q*R:]\n");
                    helper::display<scalar, index>(B_T, "Q*R");
                }

                for (i = 0; i < m * n; i++) {
                    error += fabs(B_T(i) - B(i));
                }
            }
            printf("QR Error: %8.5f\n", error);
            return error;
        }

        /**
        * Evaluating the QRP decomposition
        * @tparam scalar
        * @tparam index
        * @tparam n
        * @param B
        * @param Q
        * @param R
        * @param P
        * @param eval
        * @return
        */
        scalar qrp_validation() {
            index i, j, k;
            scalar sum, error = 0;

            if (eval == 1) {
                b_matrix B_T = prod(Q, R);//helper::mtimes_v<scalar, index>(m, n, Q, R, B_T);

                b_matrix B_P = prod(B, P);//helper::mtimes_AP<scalar, index>(m, n, B, P, B_P.data());

                if (verbose) {
                    printf("\n[ Print Q:]\n");
                    helper::display<scalar, index>(Q, "Q");
                    printf("\n[ Print R:]\n");
                    helper::display<scalar, index>(R, "R");
                    printf("\n[ Print P:]\n");
                    helper::display<scalar, index>(P, "P");
                    printf("\n[ Print B:]\n");
                    helper::display<scalar, index>(B, "B");
                    printf("\n[ Print B_P:]\n");
                    helper::display<scalar, index>(B_P, "B*P");
                    printf("\n[ Print Q*R:]\n");
                    helper::display<scalar, index>(B_T, "Q*R");
                }

                for (i = 0; i < m * n; i++) {
                    error += fabs(B_T(i) - B_P[i]);
                }
            }

            return error;
        }


        /**
         * Evaluating the QR decomposition for column orientation
         * @tparam scalar
         * @tparam index
         * @tparam n
         * @param B
         * @param Q
         * @param R
         * @param eval
         * @return
         */
        scalar qr_validation_col() {
            index i, j, k;
            scalar sum, error = 0;

            if (eval == 1) {
                b_matrix B_T = prod(Q, R); //helper::mtimes_col<scalar, index>(m, n, Q, R, B_T);

                if (verbose) {
                    printf("\n[ Print Q:]\n");
                    helper::display<scalar, index>(Q, "Q");
                    printf("\n[ Print R:]\n");
                    helper::display<scalar, index>(R, "R");
                    printf("\n[ Print B:]\n");
                    helper::display<scalar, index>(B, "B");
                    printf("\n[ Print Q*R:]\n");
                    helper::display<scalar, index>(B_T, "Q*R");
                }

                for (i = 0; i < m * n; i++) {
                    error += fabs(B_T(i) - B(i));
                }
            }

            return error;
        }

    public:
        //B --> A
        b_eye_matrix I;
        b_matrix B, R, R_R, Q, Z, P, G;
        b_vector y, y_r, p;

        CILS_Reduction(b_matrix &A, b_vector &_y, index lower, index upper) {
            m = A.size1();
            n = A.size2();
            B.resize(m, n);
            B.assign(A);
            y.resize(_y.size());
            y.assign(_y);

            lower = lower;
            upper = upper;

            R_R.resize(m, n, false);
            R.resize(m, n, false);
            Q.resize(m, m, false);
            y_r.resize(m);
            p.resize(m);

            I.resize(n, n, true);
            Z.resize(n, n, false);
            P.resize(n, n, false);

            Z.assign(I);
            P.assign(I);
        }


        /**
         * Serial version of FULL QR-factorization using modified Gram-Schmidt algorithm, row-oriented
         * Results are stored in the class object.
         */
        returnType <scalar, index> mgs_qr() {
            //Initialize B_t = [A, y]:
            b_matrix B_t(B);
            B_t.resize(m, n + 1);
            column(B_t, n) = y;

            //Clear Variables:
            R.resize(m, n + 1);
            Q.clear();
            R.clear();

            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            scalar t_qr = omp_get_wtime();
            for (index j = 0; j < m; j++) {
                for (index k = j; k < n + 1; k++) {
                    R(j, k) = inner_prod(column(Q, j), column(B_t, k));
                    column(B_t, k) = column(B_t, k) - column(Q, j) * R(j, k);
                    if (k == j) {
                        R(k, k) = norm_2(column(B_t, k));
                        column(Q, k) = column(B_t, k) / R(k, k);
                    }
                }
            }
            t_qr = omp_get_wtime() - t_qr;
            y = column(R, n);
            R.resize(m, n);

            return {{}, t_qr, 0};
        }


        /**
         * Parallel version of FULL QR-factorization using modified Gram-Schmidt algorithm, row-oriented
         * Results are stored in the class object.
         */
        returnType <scalar, index> mgs_qr_omp(const index n_proc) {
            scalar a_norm;
            b_matrix B_t(B);
            B_t.resize(m, n);

            //Clear Variables:
            Q.clear();
            R.clear();
            y.clear();
            Z.assign(I);
            auto lock = new omp_lock_t[n]();
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            scalar t_qr = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(a_norm)
            {
#pragma omp for schedule(static, 1)
                for (index i = 0; i < n; i++) {
                    omp_set_lock(&lock[i]);
                }
                if (omp_get_thread_num() == 0) {
                    // Calculation of ||B||
                    a_norm = norm_2(column(B_t, 0));
                    R(0, 0) = a_norm;
                    column(Q, 0) = column(B_t, 0) / a_norm;
                    omp_unset_lock(&lock[0]);
                }

                for (index j = 1; j < m; j++) {
                    omp_set_lock(&lock[j - 1]);
                    omp_unset_lock(&lock[j - 1]);
#pragma omp for schedule(static, 1)
                    for (index k = 0; k < n; k++) {
                        if (k >= j) {
                            R(j - 1, k) = 0;
                            R(j - 1, k) = inner_prod(column(Q, j - 1), column(B_t, k));
                            column(B_t, k) -= column(Q, j - 1) * R(j - 1, k);
                            if (k == j) {
                                a_norm = norm_2(column(B_t, k));
                                R(k, k) = a_norm;
                                column(Q, k) = column(B_t, k) / a_norm;
                                omp_unset_lock(&lock[j]);
                            }
                        }
                    }
                }
            }
            y = prod(trans(Q), y);
            t_qr = omp_get_wtime() - t_qr;
            qr_validation();
            scalar error = -1;
            if (eval || verbose) {
                error = qr_validation();
                cout << "[  QR ERROR OMP:]" << error << endl;
            }
            for (index i = 0; i < n; i++) {
                omp_destroy_lock(&lock[i]);
            }
            delete[] lock;
            return {{}, t_qr, error};

        }


        /**
         * Serial version of REDUCED QR-factorization using modified Gram-Schmidt algorithm, col-oriented
         * Results are stored in the class object.
         * R is n by n, y is transformed from m by 1 to n by 1
         * @param B : m-by-n input matrix
         * @param y : m-by-1 input right hand vector
         */
        returnType <scalar, index> mgs_qr_col() {

            b_matrix B_t(B);

            R.resize(B.size2(), B.size2());
            R.clear();
            Q.resize(B.size1(), B.size2());
            Q.clear();

            scalar t_qr = omp_get_wtime();
            for (index k = 0; k < n; k++) {
                for (index j = 0; j < k; j++) {
                    R(j, k) = inner_prod(column(Q, j), column(B_t, k));
                    column(B_t, k) = column(B_t, k) - column(Q, j) * R(j, k);
                }
                R(k, k) = norm_2(column(B_t, k));
                column(Q, k) = column(B_t, k) / R(k, k);
            }
            y = prod(trans(Q), y);

            t_qr = omp_get_wtime() - t_qr;

            return {{}, t_qr, 0};
        }

        /**
         * @deprecated
         * @return 
         */
        returnType <scalar, index> aip() {
            scalar alpha;
            scalar t_aip = omp_get_wtime();

            CILS_Reduction _reduction(B, y, lower, upper);
            _reduction.mgs_qr_col();

            b_vector y_0(_reduction.y);
            b_matrix R_0(_reduction.R);

            R_R.resize(B.size2(), B.size2());
            R_R.assign(_reduction.R);
            y_r.resize(_reduction.y.size());
            y_r.assign(_reduction.y);

            //Permutation vector
            p.resize(B.size2());
            p.clear();
            for (index i = 0; i < n; i++) {
                p[i] = i;
            }
            index _n = B.size2();

            //Inverse transpose of R
            helper::inv<scalar, index>(R_0, G);
            G = trans(G);

            scalar dist, dist_i;
            index j = 0, x_j = 0;
            for (index k = n - 1; k >= 1; k--) {
                scalar maxDist = -1;
                for (index i = 0; i <= k; i++) {
                    //alpha = y(i:k)' * G(i:k,i);
                    b_vector gi = subrange(column(G, i), i, k + 1);
                    b_vector y_t = subrange(y_r, i, k + 1);
                    alpha = inner_prod(y_t, gi);
                    index x_i = max(min((int) round(alpha), upper), lower);
                    if (alpha < lower || alpha > upper || alpha == x_i)
                        dist = 1 + abs(alpha - x_i);
                    else
                        dist = 1 - abs(alpha - x_i);
                    dist_i = dist / norm_2(gi);
                    if (dist_i > maxDist) {
                        maxDist = dist_i;
                        j = i;
                        x_j = x_i;
                    }
                }
                //p(j:k) = p([j+1:k,j]);
                scalar pj = p[j];
                subrange(p, j, k) = subrange(p, j + 1, k + 1);
                p[k] = pj;

                //Update y, R and G for the new dimension-reduced problem
                //y(1:k-1) = y(1:k-1) - R(1:k-1,j) * x_j;
                auto y_k = subrange(y_r, 0, k);
                y_k = y_k - subrange(column(R_R, j), 0, k) * x_j;
                //R(:,j) = []
                auto R_temp_2 = subrange(R_R, 0, n, j + 1, _n);
                subrange(R_R, 0, n, j, _n - 1) = R_temp_2;
                R_R.resize(m, _n - 1);
                //G(:,j) = []
                auto G_temp_2 = subrange(G, 0, n, j + 1, _n);
                subrange(G, 0, n, j, _n - 1) = G_temp_2;
                G.resize(m, _n - 1);
                //Size decrease 1
                _n--;
                for (index t = j; t <= k - 1; t++) {
                    index t1 = t + 1;
                    //Bring R back to an upper triangular matrix by a Givens rotation
                    scalar W[4] = {};
                    scalar low_tmp[2] = {R_R(t, t), R_R(t1, t)};
                    b_matrix W_m = helper::planerot<scalar, index>(low_tmp, W);
                    R_R(t, t) = low_tmp[0];
                    R_R(t1, t) = low_tmp[1];

                    //Combined Rotation.
                    auto R_G = subrange(R_R, t, t1 + 1, t1, k);
                    R_G = prod(W_m, R_G);
                    auto G_G = subrange(G, t, t1 + 1, 0, t1);
                    G_G = prod(W_m, G_G);
                    auto y_G = subrange(y_r, t, t1 + 1);
                    y_G = prod(W_m, y_G);
                }
            }

            // Reorder the columns of R0 according to p
            //R0 = R0(:,p);
            R_R.resize(n, n);
            R_R.clear();
            //The permutation matrix
            P.resize(n, n);
            P.assign(I);
            for (index i = 0; i < n; i++) {
                for (index i1 = 0; i1 < n; i1++) {
                    R_R(i1, i) = R_0(i1, p[i]);
                    P(i1, i) = I(i1, p[i]);
                }
            }

            // Compute the QR factorization of R0 and then transform y0
            //[Q, R, y] = qrmgs(R0, y0);
            CILS_Reduction reduction_(R_R, y_0, lower, upper);
            reduction_.mgs_qr_col();
            R = reduction_.R;
            y = reduction_.y;
            Q = reduction_.Q;

            t_aip = omp_get_wtime() - t_aip;
            return {{}, 0, t_aip};
        }

        /**
         * Test caller for reduction method.
         * @param n_proc
         * @return
         */
        returnType <scalar, index>
        plll_tester(const index n_proc) {
            scalar time = 0, det = 0;
            returnType<scalar, index> reT, lll_val;
            eval = true;
            verbose = true;

            cout << "[ In plll_serial]\n";
            reT = plll_serial();
            printf("[ INFO: QR TIME: %8.5f, PLLL TIME: %8.5f]\n",
                   reT.run_time, reT.info);
            time = reT.info;
            lll_val = lll_validation();

            cout << "[ In aspl_serial]\n";
            reT = aspl_serial();
            printf("[ INFO: QR TIME: %8.5f, PLLL TIME: %8.5f]\n",
                   reT.run_time, reT.info);
            scalar t_qr = reT.run_time;
            scalar t_plll = reT.info;
            lll_val = lll_validation();

            for (index i = 2; i <= 4; i += 2) {
                cout << "[ In aspl_omp]\n";
                reT = aspl_omp(i);
                printf("[ INFO: QR TIME: %8.5f, PLLL TIME: %8.5f, QR SPU: %8.5f, LLL SPU:%8.2f]\n",
                       reT.run_time, reT.info, t_qr / reT.run_time, t_plll / reT.info);
                lll_val = lll_validation();
            }


            return {{reT.info, lll_val.info}, time, lll_val.run_time};
        }

        /**
         * Original PLLL algorithm
         * Description:
         * [R,Z,y] = sils(B,y) reduces the general standard integer
         *  least squares problem to an upper triangular one by the LLL-QRZ
         *  factorization Q'*B*Z = [R; 0]. The orthogonal matrix Q
         *  is not produced.
         *
         *  Inputs:
         *     B - m-by-n real matrix with full column rank
         *     y - m-dimensional real vector to be transformed to Q'*y
         *
         *  Outputs:
         *     R - n-by-n LLL-reduced upper triangular matrix
         *     Z - n-by-n unimodular matrix, i.e., an integer matrix with
         *     |det(Z)|=1
         *     y - m-vector transformed from the input y by Q', i.e., y := Q'*y
         *
         *  Main Reference:
         *  X. Xie, X.-W. Chang, and M. Al Borno. Partial LLL Reduction,
         *  Proceedings of IEEE GLOBECOM 2011, 5 pages.
         *  Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
         *           Xiaohu Xie, Tianyang Zhou
         *  Copyright (c) 2006-2016. Scientific Computing Lab, McGill University.
         *  October 2006. Last revision: June 2016
         *  See sils.m
         *  @return returnType: ~, time_qr, time_plll
         */
        returnType <scalar, index> plll_serial() {
            scalar zeta, alpha, s, t_qr, t_plll, sum = 0;

            //Initialize B_t = [A, y]:
            b_matrix B_t(B);
            B_t.resize(m, n + 1);
            column(B_t, n) = y;

            //Clear Variables:
            R_R.resize(m, n + 1);
            Q.clear();
            R_R.clear();
            y_r.clear();
            Z.assign(I);
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            t_qr = omp_get_wtime();

            for (index j = 0; j < m; j++) {
                for (index k = j; k < n + 1; k++) {
                    R_R(j, k) = inner_prod(column(Q, j), column(B_t, k));
                    column(B_t, k) = column(B_t, k) - column(Q, j) * R_R(j, k);
                    if (k == j) {
                        R_R(k, k) = norm_2(column(B_t, k));
                        column(Q, k) = column(B_t, k) / R_R(k, k);
                    }
                }
            }
            t_qr = omp_get_wtime() - t_qr;

            //  ------------------------------------------------------------------
            //  --------  Perform the partial LLL reduction  ---------------------
            //  ------------------------------------------------------------------

            index k = 1, k1, i, i1;
            t_plll = omp_get_wtime();

            while (k < n) {
                k1 = k - 1;
                zeta = round(R_R(k1, k) / R_R(k1, k1));
                alpha = R_R(k1, k) - zeta * R_R(k1, k1);

                if (pow(R_R(k1, k1), 2) > (1 + 1.e-10) * (pow(alpha, 2) + pow(R_R(k, k), 2))) {
                    if (zeta != 0.0) {
                        //Perform a size reduction on R(k-1,k)
                        R_R(k1, k) = alpha;
                        for (i = 0; i <= k - 2; i++) {
                            R_R(i, k) = R_R(i, k) - zeta * R_R(i, k1);
                        }
                        for (i = 0; i < n; i++) {
                            Z(i, k) -= zeta * Z(i, k1);
                        }
                        //Perform size reductions on R(1:k-2,k)
                        for (i = k - 2; i >= 0; i--) {
                            zeta = round(R_R(i, k) / R_R(i, i));
                            if (zeta != 0.0) {
                                for (i1 = 0; i1 <= i; i1++) {
                                    R_R(i1, k) = R_R(i1, k) - zeta * R_R(i1, i);
                                }
                                for (i1 = 0; i1 < n; i1++) {
                                    Z(i1, k) -= zeta * Z(i1, i);
                                }
                            }
                        }
                    }
                    //Permute columns k-1 and k of R and Z
                    b_vector R_k1 = column(R_R, k1), R_k = column(R_R, k);
                    b_vector Z_k1 = column(Z, k1), Z_k = column(Z, k);
                    column(R_R, k1) = R_k;
                    column(R_R, k) = R_k1;
                    column(Z, k1) = Z_k;
                    column(Z, k) = Z_k1;

                    //Bring R back to an upper triangular matrix by a Givens rotation
                    scalar G_a[4] = {};
                    scalar low_tmp[2] = {R_R(k1, k1), R_R(k, k1)};
                    b_matrix G_m = helper::planerot<scalar, index>(low_tmp, G_a);
                    R_R(k1, k1) = low_tmp[0];
                    R_R(k, k1) = low_tmp[1];

                    //Combined Rotation.
                    //R([k1,k],k:n) = G * R([k1,k],k:n);
                    //y([k1,k]) = G * y([k1,k]);
                    auto R_G = subrange(R_R, k1, k + 1, k, n + 1);
                    R_G = prod(G_m, R_G);

                    if (k > 1) k--;

                } else {
                    k++;
                }
            }

            y = column(R_R, n);
            R_R.resize(m, n);
            R.assign(R_R);

            t_plll = omp_get_wtime() - t_plll;
            return {{}, t_qr, t_plll};
        }


        /**
         * All-Swap PLLL algorithm
         * Description:
         * [R,Z,y] = sils(B,y) reduces the general standard integer
         *  least squares problem to an upper triangular one by the LLL-QRZ
         *  factorization Q'*B*Z = [R; 0]. The orthogonal matrix Q
         *  is not produced.
         *
         *  Inputs:
         *     B - m-by-n real matrix with full column rank
         *     y - m-dimensional real vector to be transformed to Q'*y
         *
         *  Outputs:
         *     R - n-by-n LLL-reduced upper triangular matrix
         *     Z - n-by-n unimodular matrix, i.e., an integer matrix with
         *     |det(Z)|=1
         *     y - m-vector transformed from the input y by Q', i.e., y := Q'*y
         *
         *  Main Reference:
         *  Lin, S. Thesis.
         *  Authors: Lin, Shilei
         *  Copyright (c) 2021. Scientific Computing Lab, McGill University.
         *  Dec 2021. Last revision: Dec 2021
         *  @return returnType: ~, time_qr, time_plll
         */
        returnType <scalar, index> aspl_serial() {
            scalar zeta, alpha, s, t_qr, t_plll, sum = 0;

            //Initialize B_t = [A, y]:
            b_matrix B_t(B);
            B_t.resize(m, n + 1);
            column(B_t, n) = y;

            //Clear Variables:
            R_R.resize(m, n + 1);
            Q.clear();
            R_R.clear();
            y_r.clear();
            Z.assign(I);
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            t_qr = omp_get_wtime();
            for (index j = 0; j < m; j++) {
                for (index k = j; k < n + 1; k++) {
                    R_R(j, k) = inner_prod(column(Q, j), column(B_t, k));
                    column(B_t, k) = column(B_t, k) - column(Q, j) * R_R(j, k);
                    if (k == j) {
                        R_R(k, k) = norm_2(column(B_t, k));
                        column(Q, k) = column(B_t, k) / R_R(k, k);
                    }
                }
            }
            t_qr = omp_get_wtime() - t_qr;

            //  ------------------------------------------------------------------
            //  --------  Perform the all-swap partial LLL reduction -------------------
            //  ------------------------------------------------------------------

            index k = 1, k1, i, i1;
            index f = true, swap[n] = {}, start = 1, even = true;
            std::vector<b_matrix> G_v(n);
            t_plll = omp_get_wtime();
            while (f || !even) {
                f = false;
                for (index e = 0; e < n; e++) {
                    for (k = n - 1; k >= n - e; k--) {
                        k1 = k - 1;
                        zeta = round(R_R(k1, k) / R_R(k1, k1));
                        alpha = R_R(k1, k) - zeta * R_R(k1, k1);
                        if (pow(R_R(k1, k1), 2) > (1 + 1.e-10) * (pow(alpha, 2) + pow(R_R(k, k), 2))) {
                            swap[k] = 1;
                            f = true;
                            if (zeta != 0.0) {
                                //Perform a size reduction on R(k-1,k)
                                R_R(k1, k) = alpha;
                                for (i = 0; i <= k - 2; i++) {
                                    R_R(i, k) = R_R(i, k) - zeta * R_R(i, k1);
                                }
                                for (i = 0; i < n; i++) {
                                    Z(i, k) -= zeta * Z(i, k1);
                                }
                                //Perform size reductions on R(1:k-2,k)
                                for (i = k - 2; i >= 0; i--) {
                                    zeta = round(R_R(i, k) / R_R(i, i));
                                    if (zeta != 0.0) {
                                        for (i1 = 0; i1 <= i; i1++) {
                                            R_R(i1, k) = R_R(i1, k) - zeta * R_R(i1, i);
                                        }
                                        for (i1 = 0; i1 < n; i1++) {
                                            Z(i1, k) -= zeta * Z(i1, i);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                for (k = start; k < n; k += 2) {
                    if (swap[k] == 1) {
                        //Permute columns k-1 and k of R and Z
                        k1 = k - 1;
                        b_vector R_k1 = column(R_R, k1), R_k = column(R_R, k);
                        b_vector Z_k1 = column(Z, k1), Z_k = column(Z, k);
                        column(R_R, k1) = R_k;
                        column(R_R, k) = R_k1;
                        column(Z, k1) = Z_k;
                        column(Z, k) = Z_k1;

                        scalar G_a[4] = {};
                        scalar low_tmp[2] = {R_R(k1, k1), R_R(k, k1)};
                        b_matrix G_m = helper::planerot<scalar, index>(low_tmp, G_a);
                        R_R(k1, k1) = low_tmp[0];
                        R_R(k, k1) = low_tmp[1];
                        G_v[k] = G_m;
                    }
                }
                for (k = start; k < n; k += 2) {
                    if (swap[k] == 1) {
                        k1 = k - 1;
                        //Bring R back to an upper triangular matrix by a Givens rotation
                        //Combined Rotation.
                        //R([k1,k],k:n) = G * R([k1,k],k:n);
                        //y([k1,k]) = G * y([k1,k]);
                        auto R_G = subrange(R_R, k1, k + 1, k, n + 1);
                        R_G = prod(G_v[k], R_G);
                        swap[k] = 0;
                    }
                }
                if (even) {
                    even = false;
                    start = 2;
                } else {
                    even = true;
                    start = 1;
                }
            }
            y_r = column(R_R, n);
            R_R.resize(m, n);

            t_plll = omp_get_wtime() - t_plll;
            return {{}, t_qr, t_plll};
        }


        /**
         * All-Swap LLL Permutation algorithm
         * Description:
         * [R,P,y] = sils(B,y) reduces the general standard integer
         *  least squares problem to an upper triangular one by the LLL-QRZ
         *  factorization Q'*B*Z = [R; 0]. The orthogonal matrix Q
         *  is not produced.
         *
         *  Inputs:
         *     B - m-by-n real matrix with full column rank
         *     y - m-dimensional real vector to be transformed to Q'*y
         *
         *  Outputs:
         *     R - n-by-n LLL-reduced upper triangular matrix
         *     P - n-by-n unimodular matrix, i.e., an integer matrix with
         *     |det(Z)|=1
         *     y - m-vector transformed from the input y by Q', i.e., y := Q'*y
         *
         *  Main Reference:
         *  Lin, S. Thesis.
         *  Authors: Lin, Shilei
         *  Copyright (c) 2021. Scientific Computing Lab, McGill University.
         *  Dec 2021. Last revision: Dec 2021
         *  @return returnType: ~, time_qr, time_plll
         */
        returnType <scalar, index> aspl_p_serial() {

            scalar zeta, alpha, s, t_qr, t_plll, sum = 0;

            t_qr = omp_get_wtime();
            CILS_Reduction<scalar, index> _reduction(B, y, lower, upper);
//            if (m >= n) {
//                _reduction.mgs_qr_col();
//            } else {
                _reduction.mgs_qr();
//            }
            R = _reduction.R;
            y = _reduction.y;
            t_qr = omp_get_wtime() - t_qr;
//            helper::display<scalar, index>(R, "R");
            Z.assign(I);

            //  ------------------------------------------------------------------
            //  ---  Perform the all-swap partial LLL permutation reduction ------
            //  ------------------------------------------------------------------
            R.resize(m, n + 1);
            column(R, n) = y;

            index k = 1, k1, i, i1, iter = 0;
            index f = true, swap[n] = {}, start = 1, even = true;
            t_plll = omp_get_wtime();
            while ((f || !even) && (iter < 1000)) {
                f = false;
                for (k = start; k < n; k += 2) {
                    k1 = k - 1;
                    zeta = round(R(k1, k) / R(k1, k1));
                    alpha = R(k1, k) - zeta * R(k1, k1);

                    if (pow(R(k1, k1), 2) > (1 + 1.e-10) * (pow(alpha, 2) + pow(R(k, k), 2))) {
                        swap[k] = 1;
                        f = true;
                    }
                }

                for (k = start; k < n; k += 2) {
                    if (swap[k] == 1) {
                        //Permute columns k-1 and k of R and Z
                        k1 = k - 1;
                        b_vector R_k1 = column(R, k1), R_k = column(R, k);
                        b_vector Z_k1 = column(Z, k1), Z_k = column(Z, k);
                        column(R, k1) = R_k;
                        column(R, k) = R_k1;
                        column(Z, k1) = Z_k;
                        column(Z, k) = Z_k1;

                        scalar G_a[4] = {};
                        scalar low_tmp[2] = {R(k1, k1), R(k, k1)};
                        b_matrix G_m = helper::planerot<scalar, index>(low_tmp, G_a);
                        R(k1, k1) = low_tmp[0];
                        R(k, k1) = low_tmp[1];

                        auto R_G = subrange(R, k1, k + 1, k, n + 1);
                        R_G = prod(G_m, R_G);
                        swap[k] = 0;
                    }
                }
                if (even) {
                    even = false;
                    start = 2;
                } else {
                    even = true;
                    start = 1;
                }
                iter++;
            }
            y = column(R, n);
            R.resize(m, n);
            t_plll = omp_get_wtime() - t_plll;
            return {{}, t_qr, t_plll};
        }

        /**
         * All-Swap PLLL algorithm - parallel
         * Description:
         * [R,Z,y] = sils(B,y) reduces the general standard integer
         *  least squares problem to an upper triangular one by the LLL-QRZ
         *  factorization Q'*B*Z = [R; 0]. The orthogonal matrix Q
         *  is not produced.
         *
         *  Inputs:
         *     B - m-by-n real matrix with full column rank
         *     y - m-dimensional real vector to be transformed to Q'*y
         *
         *  Outputs:
         *     R - n-by-n LLL-reduced upper triangular matrix
         *     Z - n-by-n unimodular matrix, i.e., an integer matrix with
         *     |det(Z)|=1
         *     y - m-vector transformed from the input y by Q', i.e., y := Q'*y
         *
         *  Main Reference:
         *  Lin, S. Thesis.
         *  Authors: Lin, Shilei
         *  Copyright (c) 2021. Scientific Computing Lab, McGill University.
         *  Dec 2021. Last revision: Dec 2021
         *  @return returnType: ~, time_qr, time_plll
         */
        returnType <scalar, index> aspl_omp(index n_proc) {
            scalar zeta, alpha, s, t_qr, t_plll, sum = 0;
            scalar a_norm;
            //Initialize B_t = [A, y]:
            b_matrix B_t(B);
            B_t.resize(m, n);

            //Clear Variables:
            Q.clear();
            R_R.clear();
            y_r.clear();
            Z.assign(I);
            auto lock = new omp_lock_t[n]();
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            t_qr = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(a_norm)
            {
#pragma omp for schedule(static, 1)
                for (index i = 0; i < n; i++) {
                    omp_set_lock(&lock[i]);
                }
                if (omp_get_thread_num() == 0) {
                    // Calculation of ||B||
                    a_norm = norm_2(column(B_t, 0));
                    R_R(0, 0) = a_norm;
                    column(Q, 0) = column(B_t, 0) / a_norm;
                    omp_unset_lock(&lock[0]);
                }

                for (index j = 1; j < m; j++) {
                    omp_set_lock(&lock[j - 1]);
                    omp_unset_lock(&lock[j - 1]);
#pragma omp for schedule(static, 1)
                    for (index k = 0; k < n; k++) {
                        if (k >= j) {
                            R_R(j - 1, k) = 0;
                            R_R(j - 1, k) = inner_prod(column(Q, j - 1), column(B_t, k));
                            column(B_t, k) -= column(Q, j - 1) * R_R(j - 1, k);
                            if (k == j) {
                                a_norm = norm_2(column(B_t, k));
                                R_R(k, k) = a_norm;
                                column(Q, k) = column(B_t, k) / a_norm;
                                omp_unset_lock(&lock[j]);
                            }
                        }
                    }
                }
            }

            t_qr = omp_get_wtime() - t_qr;
            R.assign(R_R);
            qr_validation();
            y_r = prod(trans(Q), y);
            //  ------------------------------------------------------------------
            //  --------  Perform the all-swap partial LLL reduction -------------
            //  ------------------------------------------------------------------

            index k = 1, k1, i, i1, i2;
            index f = true, swap[n] = {}, start = 1, even = true, end = n / 2;
            std::vector<b_matrix> G_v(n);
            b_vector R_k1, Z_k1, R_k, Z_k;
            b_matrix R_G, G_m;
#pragma omp parallel default(shared) num_threads(n_proc)
            {}
            t_plll = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(k1, k, i, i1, i2, zeta, alpha, R_k1, Z_k1, G_m, R_G, R_k, Z_k)
            {
                while (f || !even) {
#pragma omp barrier
#pragma omp atomic write
                    f = false;
                    for (index e = 0; e < n; e++) {
#pragma omp for schedule(static, 1)
                        for (k = n - 1; k >= n - e; k--) {
                            k1 = k - 1;
                            zeta = round(R_R(k1, k) / R_R(k1, k1));
                            alpha = R_R(k1, k) - zeta * R_R(k1, k1);
                            if (pow(R_R(k1, k1), 2) > (1 + 1.e-10) * (pow(alpha, 2) + pow(R_R(k, k), 2))) {
                                swap[k] = 1;
                                f = true;
                                if (zeta != 0.0) {
                                    //Perform a size reduction on R(k-1,k)
                                    R_R(k1, k) = alpha;
                                    for (i = 0; i <= k - 2; i++) {
                                        R_R(i, k) = R_R(i, k) - zeta * R_R(i, k1);
                                    }
                                    for (i = 0; i < n; i++) {
                                        Z(i, k) -= zeta * Z(i, k1);
                                    }
                                    //Perform size reductions on R(1:k-2,k)
                                    for (i = k - 2; i >= 0; i--) {
                                        zeta = round(R_R(i, k) / R_R(i, i));
                                        if (zeta != 0.0) {
                                            for (i1 = 0; i1 <= i; i1++) {
                                                R_R(i1, k) = R_R(i1, k) - zeta * R_R(i1, i);
                                            }
                                            for (i1 = 0; i1 < n; i1++) {
                                                Z(i1, k) -= zeta * Z(i1, i);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

#pragma omp for schedule(static, 1)
                    for (i2 = 0; i2 < end; i2++) {
                        k = i2 * 2 + start;
                        if (swap[k] == 1) {
                            //Permute columns k-1 and k of R and Z
                            k1 = k - 1;
                            R_k1 = column(R_R, k1);
                            R_k = column(R_R, k);
                            Z_k1 = column(Z, k1);
                            Z_k = column(Z, k);
                            column(R_R, k1) = R_k;
                            column(R_R, k) = R_k1;
                            column(Z, k1) = Z_k;
                            column(Z, k) = Z_k1;

                            scalar G_a[4] = {};
                            scalar low_tmp[2] = {R_R(k1, k1), R_R(k, k1)};
                            G_m = helper::planerot<scalar, index>(low_tmp, G_a);
                            R_R(k1, k1) = low_tmp[0];
                            R_R(k, k1) = low_tmp[1];
                            G_v[k] = G_m;
                        }
                    }

#pragma omp for schedule(static, 1)
                    for (i2 = 0; i2 < end; i2++) {
                        k = i2 * 2 + start;
                        if (swap[k] == 1) {
                            k1 = k - 1;
                            //Bring R back to an upper triangular matrix by a Givens rotation
                            //Combined Rotation.
                            //R([k1,k],k:n) = G * R([k1,k],k:n);
                            //y([k1,k]) = G * y([k1,k]);
//                            scalar G_a[4] = {};
//                            scalar low_tmp[2] = {R_R(k1, k1), R_R(k, k1)};
//                            G_m = helper::planerot<scalar, index>(low_tmp, G_a);
//                            R_R(k1, k1) = low_tmp[0];
//                            R_R(k, k1) = low_tmp[1];
//                            G_v[k] = G_m;
                            R_G = subrange(R_R, k1, k + 1, k, n);
                            subrange(R_R, k1, k + 1, k, n) = prod(G_v[k], R_G);
                            subrange(y_r, k1, k + 1) = prod(G_v[k], subrange(y_r, k1, k + 1));
                            swap[k] = 0;
                        }
                    }
#pragma omp master
                    {
                        if (even) {
                            even = false;
                            start = 2;
                            end = n / 2 - 1;
                        } else {
                            even = true;
                            start = 1;
                            end = n / 2;
                        }
                    }
#pragma omp barrier
                }
            }
            t_plll = omp_get_wtime() - t_plll;
            return {{}, t_qr, t_plll};
        }

    };
}