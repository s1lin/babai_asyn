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
        returnType<scalar, index> lll_validation() {
            printf("====================[ TEST | LLL_VALIDATE ]==================================\n");
            printf("[ INFO: in LLL validation]\n");
            scalar sum, error = 0, det;
            b_matrix B_T;
            prod(B, Z, B_T);
            b_matrix R_I;

            helper::inv2<scalar, index>(R, R_I);
            b_matrix Q_Z;
            prod(B_T, R_I, Q_Z);
            b_matrix Q_T = trans(Q_Z);
            b_vector y_R;
            prod(Q_T, y_r, y_R);

            if (verbose || n <= 32) {
                helper::display<scalar, index>(y_R, "y_R");
                helper::display<scalar, index>(y, "y_r");
            }

            for (index i = 0; i < m; i++) {
                error += fabs(y_R[i] - y[i]);
            }
            printf("LLL Error: %8.5f\n", error);

            index pass = true;
            std::vector<scalar> fail_index(n, 0);
            index k = 1, k1;
            scalar zeta, alpha;

            k = 1;
            while (k < n) {
                k1 = k - 1;
                zeta = round(R(k1, k) / R(k1, k1));
                alpha = R(k1, k) - zeta * R(k1, k1);

                if (pow(R(k1, k1), 2) > (1 + 1.e-10) * (pow(alpha, 2) + pow(R(k, k), 2))) {
                    cout << "Failed:" << k1 << endl;
                }
                k++;
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
                b_matrix B_T(m, n);
                prod(Q, R, B_T);
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
                    error += fabs(B_T[i] - B[i]);
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
            printf("====================[ TEST | QRP_VALIDATE ]==================================\n");
            printf("[ INFO: in QRP validation]\n");
            index i, j, k;
            scalar sum, error = 0;


            b_matrix B_T;
            prod(Q, R, B_T);

            b_matrix B_P;
            prod(B, P, B_P);//helper::mtimes_AP<scalar, index>(m, n, B, P, B_P.data());

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
                error += fabs(B_T[i] - B_P[i]);
            }


            printf("QR Error: %8.5f\n", error);
            return error;
        }


    public:
        //B --> A
        b_eye_matrix I{};
        b_matrix B{}, R{}, Q{}, Z{}, P{}, R_A{};
        b_vector y{}, y_r{}, p{};

        CILS_Reduction() {}

        explicit CILS_Reduction(CILS<scalar, index> &cils) : CILS_Reduction(cils.A, cils.y, cils.lower, cils.upper) {}

        CILS_Reduction(b_matrix &A, b_vector &_y, index lower, index upper) {
            m = n;
            n = A.size2();
            B.resize(m, n);
            B.assign(A);
            y.resize(_y.size());
            y.assign(_y);

            lower = lower;
            upper = upper;

            R.resize(m, n, false);
            Q.resize(m, m, false);
            y_r.resize(m);
            p.resize(m);

            I.resize(n, n, false);
            Z.resize(n, n, false);
            P.resize(n, n, false);

            I.reset();
            Z.assign(I);
            P.assign(I);
        }

        void reset(CILS<scalar, index> &cils) {
            m = cils.n;
            n = cils.A.size2();
            B.resize(m, n);
            B.assign(cils.A);
            y.resize(cils.y.size());
            y.assign(cils.y);

            R.resize(m, n, false);
            Q.resize(m, m, false);
            y_r.resize(m);
            p.resize(m);

            I.resize(n, n, false);
            Z.resize(n, n, false);
            P.resize(n, n, false);

            I.reset();
            Z.assign(I);
            P.assign(I);
        }

        void reset(b_matrix &A, b_vector &y_t, index up) {
            m = n;
            n = A.size2();
            B.resize(m, n);
            B.assign(A);
            y.resize(y_t.size());
            y.assign(y_t);

            R.resize(n, n, false);
            Q.resize(m, n, false);
            y_r.resize(m);
            p.resize(n);

            I.resize(n, n, false);
            Z.resize(n, n, false);
            P.resize(n, n, false);

            I.reset();
            Z.assign(I);
            P.assign(I);
            this->upper = up;
        }

        void reset(b_matrix &A) {
            m = n;
            n = A.size2();
            B.resize(m, n);
            B.assign(A);
            y.resize(m);
            y.assign(0);

            R.resize(n, n, false);
            Q.resize(m, n, false);
            y_r.resize(m);
            p.resize(n);

            I.resize(n, n, false);
            Z.resize(n, n, false);
            P.resize(n, n, false);

            I.reset();
            Z.assign(I);
            P.assign(I);
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
                b_matrix B_T;
                prod(Q, R, B_T); //helper::mtimes_col<scalar, index>(m, n, Q, R, B_T);

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
                    error += fabs(B_T[i] - B[i]);
                }
            }

            return error;
        }

        /**
         * Serial version of MGS QR-factorization using modified Gram-Schmidt algorithm, row-oriented
         * Results are stored in the class object.
         */
        returnType<scalar, index> mgs_qr() {
            //Clear Variables:
            R.clear();
            Q.assign(B);
            P.assign(I);
            Z.assign(I);

            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            scalar t_qr = omp_get_wtime();
            for (index k = 0; k < n; k++) {
                scalar sum = 0;
                for (index i = 0; i < n; i++) {
                    sum += pow(Q[i + k * n], 2);
                }
                R(k, k) = sqrt(sum);
                for (index i = 0; i < n; i++) {
                    Q[i + k * n] = Q[i + k * n] / R(k, k);
                }
                for (index j = 0; j < n; j++) {
                    if (j > k) {
                        R(k, j) = 0;
                        for (index i = 0; i < n; i++) {
                            R(k, j) += Q[i + k * n] * Q(i, j);
                        }
                        for (index i = 0; i < n; i++) {
                            Q(i, j) -= R(k, j) * Q[i + k * n];
                        }
                    }
                }
            }

            b_matrix Q_T = trans(Q);
            prod(Q_T, y, y_r);
            y.assign(y_r);
            t_qr = omp_get_wtime() - t_qr;

            return {{}, t_qr, 0};
        }

        returnType<scalar, index> mgs_qrp() {
            //Clear Variables:
            R.resize(n, n);
            Q.assign(B);
            P.assign(I);
            sd_vector s1(n, 0);
            sd_vector s2(n, 0);
            //  ------------------------------------------------------------------
//            cout << "--------  Perform the QR factorization: MGS Row-------------------";
            //  ------------------------------------------------------------------
            scalar t_qr = omp_get_wtime();
            scalar sum = 0;

            for (index k = 0; k < n; k++) {
                sum = 0;
                for (index i = 0; i < n; i++) {
                    sum += pow(Q[i + k * n], 2);
                }
                s1[k] = sum;
                s2[k] = 0;
            }

            index l = 0;
            scalar min_norm = 0;

            for (index k = 0; k < n; k++) {
                min_norm = s1[k] - s2[k];
                l = k;
                for (index i = k + 1; i < n; i++) {
                    scalar cur_norm = s1[i] - s2[i];
                    if (cur_norm <= min_norm) {
                        min_norm = cur_norm;
                        l = i;
                    }
                }
                if (l > k) {
                    scalar temp;
                    for (index i = 0; i < n; i++) {
                        temp = R[i + l * n];
                        R[i + l * n] = R[i + k * n];
                        R[i + k * n] = temp;
                        temp = P[i + l * n];
                        P[i + l * n] = P[i + k * n];
                        P[i + k * n] = temp;
                        temp = Q[i + l * n];
                        Q[i + l * n] = Q[i + k * n];
                        Q[i + k * n] = temp;
                    }
                    std::swap(s1[l], s1[k]);
                    std::swap(s2[l], s2[k]);
                }
                sum = 0;
                for (index i = 0; i < n; i++) {
                    sum += pow(Q[i + k * n], 2);
                }
                R(k, k) = sqrt(sum);
                for (index i = 0; i < n; i++) {
                    Q[i + k * n] = Q[i + k * n] / R(k, k);
                }
                for (index j = k + 1; j < n; j++) {
                    R(k, j) = 0;
                    for (index i = 0; i < n; i++) {
                        R(k, j) += Q[i + k * n] * Q(i, j);
                    }
                    s2[j] = s2[j] + pow(R(k, j), 2);
                    for (index i = 0; i < n; i++) {
                        Q(i, j) -= R(k, j) * Q[i + k * n];
                    }
                }
            }
//            cout << endl;
            b_matrix Q_T = trans(Q);
            prod(Q_T, y, y_r);
            y.assign(y_r);
            t_qr = omp_get_wtime() - t_qr;
//            qrp_validation();
            return {{}, t_qr, 0};
        }

        returnType<scalar, index> pmgs_qrp(index n_proc) {
            //Clear Variables:
            R.resize(n, n);
            Q.assign(B);
            P.assign(I);
            sd_vector s1_v(n, 0);
            sd_vector s2_v(n, 0);
            auto s1 = s1_v.data();
            auto s2 = s2_v.data();
//            auto lock = new omp_lock_t[n]();
//            for (index i = 0; i < n; i++) {
//                omp_init_lock((&lock[i]));
//                omp_set_lock(&lock[i]);
//            }
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            scalar t_qr = omp_get_wtime(), min_norm;
            scalar sum = 0;
            index i, j, k, l;
            auto Q_d = Q.data();
            auto R_d = R.data();
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------

#pragma omp parallel default(shared) num_threads(n_proc) private(sum, i, j, k, l, min_norm)
            {
#pragma omp for schedule(static, 1)
                for (k = 0; k < n; k++) {
                    sum = 0;
                    for (i = 0; i < n; i++) {
                        sum += pow(Q_d[i + k * n], 2);
                    }
                    s1[k] = sum;
                }

                for (k = 0; k < n; k++) {
#pragma omp barrier
#pragma omp master
                    {
                        min_norm = s1[k] - s2[k];
                        l = k;
                        for (i = k + 1; i < n; i++) {
                            scalar cur_norm = s1[i] - s2[i];
                            if (cur_norm <= min_norm) {
                                min_norm = cur_norm;
                                l = i;
                            }
                        }
                        if (l > k) {
                            scalar temp;
                            for (i = 0; i < n; i++) {
                                temp = R_d[i + l * n];
                                R_d[i + l * n] = R_d[i + k * n];
                                R_d[i + k * n] = temp;
                                temp = P[i + l * n];
                                P[i + l * n] = P[i + k * n];
                                P[i + k * n] = temp;
                                temp = Q_d[i + l * n];
                                Q_d[i + l * n] = Q_d[i + k * n];
                                Q_d[i + k * n] = temp;
                            }
                            std::swap(s1[l], s1[k]);
                            std::swap(s2[l], s2[k]);
                        }
                        sum = 0;
                        for (i = 0; i < n; i++) {
                            sum += pow(Q_d[i + k * n], 2);
                        }
                        R_d[k + k * n] = sqrt(sum);
                    }

#pragma omp barrier
#pragma omp for schedule(static, 1)
                    for (i = 0; i < n; i++) {
                        Q_d[i + k * n] = Q_d[i + k * n] / R_d[k + k * n];
                    }
#pragma omp for schedule(static, 1)
                    for (j = 0; j < n; j++) {
                        if (j > k) {
                            R_d[k + j * n] = 0;
                            for (i = 0; i < n; i++) {
                                R_d[k + j * n] += Q_d[i + k * n] * Q_d[i + j * n];
                            }
                            s2[j] = s2[j] + pow(R_d[k + j * n], 2);
                            for (i = 0; i < n; i++) {
                                Q_d[i + j * n] -= R_d[k + j * n] * Q_d[i + k * n];
                            }
                        }
                    }
                }

            }

            b_matrix Q_T = trans(Q);
            prod(Q_T, y, y_r);
            y.assign(y_r);
            t_qr = omp_get_wtime() - t_qr;
//            scalar error = qrp_validation();
            return {{}, t_qr, 0};
        }

        /**
         * Parallel version of FULL QR-factorization using modified Gram-Schmidt algorithm, row-oriented
         * Results are stored in the class object.
         */
        returnType<scalar, index> pmgs_qr_col(const index n_proc) {

            b_matrix B_t(B);

            R.resize(B.size2(), B.size2());
            R.clear();
            Q.resize(B.size1(), B.size2());
            Q.assign(B);

            auto Q_d = Q.data();
            auto R_d = R.data();

            scalar sum = 0;
            index i, j, k;

            scalar t_qr = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(sum, i, j, k)
            {
                for (k = 0; k < n; k++) {
#pragma omp single
                    {
                        sum = 0;
                        for (i = 0; i < m; i++) {
                            sum += pow(Q_d[i + k * m], 2);
                        }
                        R_d[k + k * n] = sqrt(sum);
                        for (i = 0; i < m; i++) {
                            Q_d[i + k * m] = Q_d[i + k * m] / R_d[k + k * n];
                        }
                    }
//#pragma omp barrier
#pragma omp for schedule(static, 1)
                    for (j = 0; j < n; j++) {
                        if (j > k) {
                            R_d[k + j * n] = 0;
                            for (i = 0; i < m; i++) {
                                R_d[k + j * n] += Q_d[i + k * m] * Q_d[i + j * m];
                            }
                            for (i = 0; i < m; i++) {
                                Q_d[i + j * m] -= R_d[k + j * n] * Q_d[i + k * m];
                            }
                        }
                    }
                }
            }
            b_matrix Q_T = trans(Q);
            prod(Q_T, y, y_r);
            y.assign(y_r);

            t_qr = omp_get_wtime() - t_qr;

            return {{}, t_qr, 0};

        }


        /**
         * Serial version of REDUCED QR-factorization using modified Gram-Schmidt algorithm, col-oriented
         * Results are stored in the class object.
         * R is n by n, y is transformed from m by 1 to n by 1
         * @param B : m-by-n input matrix
         * @param y : m-by-1 input right hand vector
         */
        returnType<scalar, index> mgs_qr_col() {
            m = B.size1();
            n = B.size2();
            b_matrix B_t(B);

            R.resize(B.size2(), B.size2());
            R.clear();
            Q.resize(B.size1(), B.size2());
            Q.assign(B);

            scalar t_qr = omp_get_wtime();
            for (index k = 0; k < n; k++) {
                for (index j = 0; j < k; j++) {
                    for (index i = 0; i < m; i++) {
                        R(j, k) += Q(i, j) * Q(i, k);
                    }
                    for (index i = 0; i < m; i++) {
                        Q(i, k) -= R(j, k) * Q(i, j);
                    }
                }
                scalar sum = 0;
                for (index i = 0; i < m; i++) {
                    sum += pow(Q(i, k), 2);
                }
                R(k, k) = sqrt(sum);
                for (index i = 0; i < m; i++) {
                    Q(i, k) = Q(i, k) / R(k, k);
                }

            }
            b_matrix Q_T = trans(Q);
            prod(Q_T, y, y_r);
            y.assign(y_r);

            t_qr = omp_get_wtime() - t_qr;

            return {{}, t_qr, 0};
        }

        returnType<scalar, index> mgs_max()
        {
            b_matrix b_P;
            b_matrix s;
            b_matrix varargin_1;
            b_vector b_A;
            double l;
            int iv[2];
            int i;
            int i1;
            int k;
            int loop_ub;
            int nn;
            int t;
            nn = n;
            R.resize(n, n);
            R.clear();

            s.resize(2, n);
            s.clear();

            t = n;
            P.resize(t, t);
            loop_ub = t * t;
            for (i = 0; i < loop_ub; i++) {
                P[i] = 0.0;
            }
            if (t > 0) {
                for (k = 0; k < t; k++) {
                    P[k + P.size1() * k] = 1.0;
                }
            }
            i = n;
            for (k = 0; k < i; k++) {
                loop_ub = n;
                b_A.resize(loop_ub);
                for (i1 = 0; i1 < loop_ub; i1++) {
                    b_A[i1] = B[i1 + n * k];
                }
                l = norm_2(b_A); //coder::b_norm(b_A);
                s[2 * k] = l * l;
            }
            i = n;
            for (int j{0}; j < i; j++) {
                double s_idx_2;
                int i2;
                int idx;
                if (j + 1 > nn) {
                    i1 = 0;
                    i2 = 0;
                    t = 0;
                    idx = 0;
                } else {
                    i1 = j;
                    i2 = nn;
                    t = j;
                    idx = nn;
                }
                loop_ub = i2 - i1;
                if (loop_ub == idx - t) {
                    varargin_1.resize(1, loop_ub);
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        varargin_1[i2] = s[2 * (i1 + i2)] - s[2 * (t + i2) + 1];
                    }
                } 
                t = varargin_1.size2();
                if (varargin_1.size2() <= 2) {
                    if (varargin_1.size2() == 1) {
                        idx = 1;
                    } else if ((varargin_1[0] < varargin_1[varargin_1.size2() - 1]) ||
                               (std::isnan(varargin_1[0]) &&
                                (!std::isnan(varargin_1[varargin_1.size2() - 1])))) {
                        idx = varargin_1.size2();
                    } else {
                        idx = 1;
                    }
                } else {
                    if (!std::isnan(varargin_1[0])) {
                        idx = 1;
                    } else {
                        bool exitg1;
                        idx = 0;
                        k = 2;
                        exitg1 = false;
                        while ((!exitg1) && (k <= t)) {
                            if (!std::isnan(varargin_1[k - 1])) {
                                idx = k;
                                exitg1 = true;
                            } else {
                                k++;
                            }
                        }
                    }
                    if (idx == 0) {
                        idx = 1;
                    } else {
                        l = varargin_1[idx - 1];
                        i1 = idx + 1;
                        for (k = i1; k <= t; k++) {
                            s_idx_2 = varargin_1[k - 1];
                            if (l < s_idx_2) {
                                l = s_idx_2;
                                idx = k;
                            }
                        }
                    }
                }
                l = (static_cast<double>(idx) + (static_cast<double>(j) + 1.0)) - 1.0;
                if (l > static_cast<double>(j) + 1.0) {
                    int b_l[2];
                    iv[0] = j;
                    idx = static_cast<int>(l) - 1;
                    iv[1] = idx;
                    t = P.size1() - 1;
                    b_l[0] = idx;
                    b_l[1] = j;
                    b_P.resize(P.size1(), 2);
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 <= t; i2++) {
                            b_P[i2 + b_P.size1() * i1] = P[i2 + P.size1() * b_l[i1]];
                        }
                    }
                    loop_ub = b_P.size1();
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 < loop_ub; i2++) {
                            P[i2 + P.size1() * iv[i1]] = b_P[i2 + b_P.size1() * i1];
                        }
                    }
                    double s_idx_3;
                    l = s[2 * idx + 1];
                    s_idx_2 = s[2 * j];
                    s_idx_3 = s[2 * j + 1];
                    s[2 * j] = s[2 * idx];
                    s[2 * j + 1] = l;
                    s[2 * idx] = s_idx_2;
                    s[2 * idx + 1] = s_idx_3;
                    iv[0] = j;
                    iv[1] = idx;
                    t = n - 1;
                    b_l[0] = idx;
                    b_l[1] = j;
                    b_P.resize(t + 1, 2);
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 <= t; i2++) {
                            b_P[i2 + b_P.size1() * i1] = B[i2 + n * b_l[i1]];
                        }
                    }
                    loop_ub = b_P.size1();
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 < loop_ub; i2++) {
                            B[i2 + n * iv[i1]] = b_P[i2 + b_P.size1() * i1];
                        }
                    }
                    iv[0] = j;
                    iv[1] = idx;
                    t = R.size1() - 1;
                    b_l[0] = idx;
                    b_l[1] = j;
                    b_P.resize(R.size1(), 2);
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 <= t; i2++) {
                            b_P[i2 + b_P.size1() * i1] = R[i2 + R.size1() * b_l[i1]];
                        }
                    }
                    loop_ub = b_P.size1();
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 < loop_ub; i2++) {
                            R[i2 + R.size1() * iv[i1]] = b_P[i2 + b_P.size1() * i1];
                        }
                    }
                }
                loop_ub = n;
                b_A.resize(loop_ub);
                for (i1 = 0; i1 < loop_ub; i1++) {
                    b_A[i1] = B[i1 + n * j];
                }
                R[j + R.size1() * j] = norm_2(b_A);
                t = n - 1;
                l = R[j + R.size1() * j];
                b_A.resize(t + 1);
                for (i1 = 0; i1 <= t; i1++) {
                    b_A[i1] = B[i1 + n * j] / l;
                }
                loop_ub = b_A.size();
                for (i1 = 0; i1 < loop_ub; i1++) {
                    B[i1 + n * j] = b_A[i1];
                }
                i1 = nn - j;
                for (k = 0; k <= i1 - 2; k++) {
                    unsigned int b_k;
                    b_k = (static_cast<unsigned int>(j) + k) + 2U;
                    loop_ub = n;
                    l = 0.0;
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        l += B[i2 + n * j] *
                             B[i2 + n * (static_cast<int>(b_k) - 1)];
                    }
                    R[j + R.size1() * (static_cast<int>(b_k) - 1)] = l;
                    l = R[j + R.size1() * (static_cast<int>(b_k) - 1)];
                    s[2 * (static_cast<int>(b_k) - 1) + 1] =
                            s[2 * (static_cast<int>(b_k) - 1) + 1] + l * l;
                    t = n - 1;
                    b_A.resize(t + 1);
                    for (i2 = 0; i2 <= t; i2++) {
                        b_A[i2] = B[i2 + n * (static_cast<int>(b_k) - 1)] -
                                  B[i2 + n * j] * l;
                    }
                    loop_ub = b_A.size();
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        B[i2 + n * (static_cast<int>(b_k) - 1)] = b_A[i2];
                    }
                }
            }

            return {{}, 0, 0};
        }
        
        /**
         * @return
         */
        returnType<scalar, index> aip() {
            scalar time = omp_get_wtime();
            b_matrix C;
            b_matrix G;
            b_matrix R0;
            b_matrix b;
            b_matrix b_R;
            b_matrix c_y;
            b_vector b_G;
            b_vector b_y;
            b_matrix d_y;
            double x_j;
            int i;
            int i1;
            int j;
            int loop_ub;
            int ncols;
            //
            //  [R,y,l,u,p] = obils_reduction(B,y,l,u) reduces the general overdetermined
            //  box-constrained integer least squares problem to an upper triangular one
            //  by the QR factorization with column permutations
            //  Q'*B*P = [R; 0]. The orthogonal matrix Q is not produced.
            //
            //  Inputs:
            //     B - m-by-n real matrix with full column rank
            //     y - m-dimensional real vector
            //     l - n-dimensional integer vector, lower bound
            //     u - n-dimensional integer vector, upper bound
            //
            //  Outputs:
            //     R - n-by-n real nonsingular upper triangular matrix
            //     y - n-dimensional vector transformed from the input y, y:=(Q'*y)(1:n)
            //     l - permuted input lower bound l, i.e., l := P'*l
            //     u - permuted input upper bound u, i.e., u := P'*u
            //     p - n-dimensional permutation vector representing P
            //  Main Reference:
            //  S. Breen and X.-W. Chang. Column Reordering for
            //  Box-Constrained Integer Least Squares Problems,
            //  Proceedings of IEEE GLOBECOM 2011, 6 pages.
            //  Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
            //           Xiangyu Ren
            //  Copyright (c) 2015-2018. Scientific Computing Lab, McGill University.
            //  Last revision: December 2018
            // 'obils_reduction:32' [~,n] = size(B);
            //  Transform B and y by the QR factorization
            //  U = qr([B,y]);
            //  R = triu(U(1:n,1:n));
            //  y = U(1:n,n+1);
            //  R0 = R;
            //  y0 = y;
            // 'obils_reduction:41' [~, R, y] = qrmgs(B,y);
            mgs_qr_col();
            b_y.assign(y);
            R0.assign(R);
            // y = Q' * y;
            // 'obils_reduction:43' R0 = R;
            // 'obils_reduction:44' y0 = y;
            //  Permutation vector
            // 'obils_reduction:47' p = 1:n;
            p.resize(n);
            for (i = 0; i < n; i++) {
                p[i] = i;
            }

            //  Inverse transpose of R
            // 'obils_reduction:50' G = inv(R)';
            helper::inv<scalar, index>(R0, G);
            G = trans(G);
            // 'obils_reduction:51' j = 0;
            j = 0;
            // 'obils_reduction:52' x_j = 0;
            x_j = 0.0;
            //  Determine the column permutatons
            // 'obils_reduction:54' for k = n : -1 : 2
            i = n;
            for (int k{0}; k <= i - 2; k++) {
                double dist;
                double maxDist;
                double x_i;
                int b_i;
                int b_k;
                int i2;
                int nrowx;
                b_k = B.size2() - k;
                // 'obils_reduction:55' maxDist = -1;
                maxDist = -1.0;
                //  Determine the k-th column
                // 'obils_reduction:58' for i = 1:k
                for (b_i = 0; b_i < b_k; b_i++) {
                    // 'obils_reduction:59' alpha = y(i:k)' * G(i:k,i);
                    if (b_i + 1 > b_k) {
                        i1 = 0;
                        i2 = 0;
                        ncols = 0;
                    } else {
                        i1 = b_i;
                        i2 = b_k;
                        ncols = b_i;
                    }
                    dist = 0.0;
                    loop_ub = i2 - i1;
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        dist += y[i1 + i2] * G[(ncols + i2) + G.size1() * b_i];
                    }
                    // 'obils_reduction:60' x_i = max(min(round(alpha),u(i)),l(i));
                    x_i = std::fmax(std::fmin(std::round(dist), upper), 0);
                    // 'obils_reduction:61' if (alpha < l(i) || alpha > u(i) || alpha == x_i)
                    if ((dist < 0) || (dist > upper) || (dist == x_i)) {
                        // 'obils_reduction:62' dist = 1 + abs(alpha - x_i);
                        dist = std::abs(dist - x_i) + 1.0;
                    } else {
                        // 'obils_reduction:63' else
                        // 'obils_reduction:64' dist = 1 - abs(alpha - x_i);
                        dist = 1.0 - std::abs(dist - x_i);
                    }
                    // 'obils_reduction:66' dist_i = dist / norm(G(i:k,i));
                    if (b_i + 1 > b_k) {
                        i1 = 0;
                        i2 = 0;
                    } else {
                        i1 = b_i;
                        i2 = b_k;
                    }
                    loop_ub = i2 - i1;
                    b_G.resize(loop_ub);
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        b_G[i2] = G[(i1 + i2) + G.size1() * b_i];
                    }
                    dist /= norm_2(b_G);
                    // 'obils_reduction:67' if dist_i > maxDist
                    if (dist > maxDist) {
                        // 'obils_reduction:68' maxDist = dist_i;
                        maxDist = dist;
                        // 'obils_reduction:69' j = i;
                        j = b_i + 1;
                        // 'obils_reduction:70' x_j = x_i;
                        x_j = x_i;
                    }
                }
                //  Perform permutations
                // 'obils_reduction:75' p(j:k) = p([j+1:k,j]);
                if (j > b_k) {
                    i1 = 1;
                } else {
                    i1 = j;
                }
                if (b_k < static_cast<double>(j) + 1.0) {
                    c_y.resize(1, 0);
                } else {
                    i2 = b_k - j;
                    c_y.resize(1, i2);
                    loop_ub = i2 - 1;
                    for (i2 = 0; i2 <= loop_ub; i2++) {
                        c_y[i2] = (static_cast<unsigned int>(j) + i2) + 1U;
                    }
                }
                d_y.resize(1, c_y.size2() + 1);
                loop_ub = c_y.size2();
                for (i2 = 0; i2 < loop_ub; i2++) {
                    d_y[i2] = static_cast<int>(c_y[i2]) - 1;
                }
                d_y[c_y.size2()] = j - 1;
                c_y.resize(1, d_y.size2());
                loop_ub = d_y.size2();
                for (i2 = 0; i2 < loop_ub; i2++) {
                    c_y[i2] = p[d_y[i2]];
                }
                loop_ub = c_y.size2();
                for (i2 = 0; i2 < loop_ub; i2++) {
                    p[(i1 + i2) - 1] = c_y[i2];
                }
                // 'obils_reduction:76' l(j:k) = l([j+1:k,j]);
                // 'obils_reduction:77' u(j:k) = u([j+1:k,j]);
                //  Update y, R and G for the new dimension-reduced problem
                // 'obils_reduction:80' y(1:k-1) = y(1:k-1) - R(1:k-1,j) * x_j;
                if (b_k - 1 < 1) {
                    i1 = 0;
                    i2 = 0;
                    loop_ub = 0;
                } else {
                    i1 = b_k - 1;
                    i2 = b_k - 1;
                    loop_ub = b_k - 1;
                }
                if (i1 == i2) {
                    c_y.resize(1, loop_ub);
                    for (i1 = 0; i1 < loop_ub; i1++) {
                        c_y[i1] = y[i1] - R[i1 + R.size1() * (j - 1)] * x_j;
                    }
                    loop_ub = c_y.size2();
                    for (i1 = 0; i1 < loop_ub; i1++) {
                        y[i1] = c_y[i1];
                    }
                }
                // 'obils_reduction:81' R(:,j) = [];
                nrowx = R.size1();
                i1 = R.size2();
                ncols = R.size2() - 1;
                for (loop_ub = j; loop_ub <= ncols; loop_ub++) {
                    for (b_i = 0; b_i < nrowx; b_i++) {
                        R[b_i + R.size1() * (loop_ub - 1)] = R[b_i + R.size1() * loop_ub];
                    }
                }
                if (i1 - 1 < 1) {
                    loop_ub = -1;
                } else {
                    loop_ub = i1 - 2;
                }
                nrowx = R.size1() - 1;
                ncols = R.size1();
                for (i1 = 0; i1 <= loop_ub; i1++) {
                    for (i2 = 0; i2 < ncols; i2++) {
                        R[i2 + (nrowx + 1) * i1] = R[i2 + R.size1() * i1];
                    }
                }
                R.size2(loop_ub + 1);
                // 'obils_reduction:82' G(:,j) = [];
                nrowx = G.size1();
                i1 = G.size2();
                ncols = G.size2() - 1;
                for (loop_ub = j; loop_ub <= ncols; loop_ub++) {
                    for (b_i = 0; b_i < nrowx; b_i++) {
                        G[b_i + G.size1() * (loop_ub - 1)] = G[b_i + G.size1() * loop_ub];
                    }
                }
                if (i1 - 1 < 1) {
                    loop_ub = -1;
                } else {
                    loop_ub = i1 - 2;
                }
                nrowx = G.size1() - 1;
                ncols = G.size1();
                for (i1 = 0; i1 <= loop_ub; i1++) {
                    for (i2 = 0; i2 < ncols; i2++) {
                        G[i2 + (nrowx + 1) * i1] = G[i2 + G.size1() * i1];
                    }
                }
                G.size2(loop_ub + 1);
                // 'obils_reduction:83' for t = j : k - 1
                i1 = b_k - j;
                for (int t{0}; t < i1; t++) {
                    double W_idx_0;
                    double W_idx_2;
                    double W_idx_3;
                    double unnamed_idx_0;
                    double unnamed_idx_1;
                    unsigned int b_t;
                    int i3;
                    b_t = static_cast<unsigned int>(j) + t;
                    //  Triangularize R and G by Givens rotation
                    // 'obils_reduction:85' [W, R([t,t+1],t)] = planerot(R([t,t+1],t));
                    b_i = static_cast<int>(b_t) - 1;
                    unnamed_idx_0 =
                            R[(static_cast<int>(b_t) + R.size1() * (static_cast<int>(b_t) - 1)) -
                              1];
                    unnamed_idx_1 =
                            R[static_cast<int>(b_t) + R.size1() * (static_cast<int>(b_t) - 1)];
                    W_idx_3 =
                            R[static_cast<int>(b_t) + R.size1() * (static_cast<int>(b_t) - 1)];
                    if (W_idx_3 != 0.0) {
                        double r;
                        dist = 3.3121686421112381E-170;
                        maxDist = std::abs(R[(static_cast<int>(b_t) +
                                              R.size1() * (static_cast<int>(b_t) - 1)) -
                                             1]);
                        if (maxDist > 3.3121686421112381E-170) {
                            r = 1.0;
                            dist = maxDist;
                        } else {
                            x_i = maxDist / 3.3121686421112381E-170;
                            r = x_i * x_i;
                        }
                        maxDist = std::abs(
                                R[static_cast<int>(b_t) + R.size1() * (static_cast<int>(b_t) - 1)]);
                        if (maxDist > dist) {
                            x_i = dist / maxDist;
                            r = r * x_i * x_i + 1.0;
                            dist = maxDist;
                        } else {
                            x_i = maxDist / dist;
                            r += x_i * x_i;
                        }
                        r = dist * std::sqrt(r);
                        W_idx_0 = unnamed_idx_0 / r;
                        W_idx_2 = unnamed_idx_1 / r;
                        x_i = -W_idx_3 / r;
                        W_idx_3 = R[(static_cast<int>(b_t) +
                                     R.size1() * (static_cast<int>(b_t) - 1)) -
                                    1] /
                                  r;
                        unnamed_idx_0 = r;
                        unnamed_idx_1 = 0.0;
                    } else {
                        x_i = 0.0;
                        W_idx_2 = 0.0;
                        W_idx_0 = 1.0;
                        W_idx_3 = 1.0;
                    }
                    R[(static_cast<int>(b_t) + R.size1() * (static_cast<int>(b_t) - 1)) - 1] =
                            unnamed_idx_0;
                    R[static_cast<int>(b_t) + R.size1() * (static_cast<int>(b_t) - 1)] =
                            unnamed_idx_1;
                    // 'obils_reduction:86' R([t,t+1],t+1:k-1) = W * R([t,t+1],t+1:k-1);
                    if (static_cast<int>(b_t) + 1 > b_k - 1) {
                        i2 = 0;
                        ncols = 0;
                        i3 = 0;
                    } else {
                        i2 = static_cast<int>(b_t);
                        ncols = b_k - 1;
                        i3 = static_cast<int>(b_t);
                    }
                    loop_ub = ncols - i2;
                    b.resize(2, loop_ub);
                    for (ncols = 0; ncols < loop_ub; ncols++) {
                        nrowx = i2 + ncols;
                        b[2 * ncols] = R[(static_cast<int>(b_t) + R.size1() * nrowx) - 1];
                        b[2 * ncols + 1] = R[static_cast<int>(b_t) + R.size1() * nrowx];
                    }
                    ncols = loop_ub - 1;
                    C.resize(2, loop_ub);
                    for (loop_ub = 0; loop_ub <= ncols; loop_ub++) {
                        nrowx = loop_ub << 1;
                        dist = b[nrowx + 1];
                        C[nrowx] = W_idx_0 * b[nrowx] + W_idx_2 * dist;
                        C[nrowx + 1] = x_i * b[nrowx] + W_idx_3 * dist;
                    }
                    loop_ub = C.size2();
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        nrowx = i3 + i2;
                        R[(static_cast<int>(b_t) + R.size1() * nrowx) - 1] = C[2 * i2];
                        R[static_cast<int>(b_t) + R.size1() * nrowx] = C[2 * i2 + 1];
                    }
                    // 'obils_reduction:87' G([t,t+1],1:t) = W * G([t,t+1],1:t);
                    loop_ub = static_cast<int>(b_t);
                    b.resize(2, static_cast<int>(b_t));
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        b[2 * i2] = G[(static_cast<int>(b_t) + G.size1() * i2) - 1];
                        b[2 * i2 + 1] = G[static_cast<int>(b_t) + G.size1() * i2];
                    }
                    C.resize(2, static_cast<int>(b_t));
                    for (loop_ub = 0; loop_ub <= b_i; loop_ub++) {
                        nrowx = loop_ub << 1;
                        dist = b[nrowx + 1];
                        C[nrowx] = W_idx_0 * b[nrowx] + W_idx_2 * dist;
                        C[nrowx + 1] = x_i * b[nrowx] + W_idx_3 * dist;
                    }
                    loop_ub = C.size2();
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        G[(static_cast<int>(b_t) + G.size1() * i2) - 1] = C[2 * i2];
                        G[static_cast<int>(b_t) + G.size1() * i2] = C[2 * i2 + 1];
                    }
                    //  Apply the Givens rotation W to y
                    // 'obils_reduction:89' y(t:t+1) = W * y(t:t+1);
                    dist = y[static_cast<int>(b_t) - 1];
                    maxDist = y[static_cast<int>(static_cast<double>(b_t) + 1.0) - 1];
                    y[static_cast<int>(b_t) - 1] = W_idx_0 * dist + W_idx_2 * maxDist;
                    y[static_cast<int>(static_cast<double>(b_t) + 1.0) - 1] =
                            x_i * dist + W_idx_3 * maxDist;
                }
            }
            //  Reorder the columns of R0 according to p
            // 'obils_reduction:95' R0 = R0(:,p);

            loop_ub = b_R.size1();
            b_R.assign(R0);
            R0.clear();
            for (i = 0; i < n; i++) {
                for (i1 = 0; i1 < n; i1++) {
                    R0[i1 + R0.size1() * i] = b_R[i1 + b_R.size1() * p[i]];
                    P[i1 + P.size1() * i] = I[i1 + I.size1() * p[i]];
                }
            }

            //  Transform R0 and y0 by the QR factorization
            // 'obils_reduction:98' [Q, R, y] = qrmgs(R0, y0);
            y.resize(m);
            for (i = 0; i < m; i++) {
                y[i] = b_y[i];
            }
            CILS_Reduction reduction_;
            reduction_.reset(R0, y, upper);
            reduction_.mgs_qr_col();
            R.assign(reduction_.R);
            y.assign(reduction_.y);
            time = omp_get_wtime() - time;
            return {{}, time, 0};
        }

        returnType<scalar, index> paip(index n_t) {
            scalar time = omp_get_wtime();
            b_matrix C, G, R0, b, b_R, c_y, d_y;
            b_vector b_y, b_G;
            double x_j;
            int i, i1, j, loop_ub, ncols;
            pmgs_qr_col(n_t);
            b_y.assign(y);
            R0.assign(R);
            // y = Q' * y;
            // 'obils_reduction:43' R0 = R;
            // 'obils_reduction:44' y0 = y;
            //  Permutation vector
            // 'obils_reduction:47' p = 1:n;
            p.resize(n);
            for (i = 0; i < n; i++) {
                p[i] = i;
            }

            //  Inverse transpose of R
            // 'obils_reduction:50' G = inv(R)';
            helper::inv2<scalar, index>(R0, G);
//            cout << omp_get_wtime() - time;
            G = trans(G);
            // 'obils_reduction:51' j = 0;
            j = 0;
            // 'obils_reduction:52' x_j = 0;
            x_j = 0.0;
            //  Determine the column permutatons
            // 'obils_reduction:54' for k = n : -1 : 2
            i = n;
            for (int k{0}; k <= i - 2; k++) {
                double dist;
                double maxDist;
                double x_i;
                int b_i;
                int b_k;
                int i2;
                int nrowx;
                b_k = B.size2() - k;
                // 'obils_reduction:55' maxDist = -1;
                maxDist = -1.0;
                //  Determine the k-th column
                // 'obils_reduction:58' for i = 1:k
                for (b_i = 0; b_i < b_k; b_i++) {
                    // 'obils_reduction:59' alpha = y(i:k)' * G(i:k,i);
                    if (b_i + 1 > b_k) {
                        i1 = 0;
                        i2 = 0;
                        ncols = 0;
                    } else {
                        i1 = b_i;
                        i2 = b_k;
                        ncols = b_i;
                    }
                    dist = 0.0;
                    loop_ub = i2 - i1;
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        dist += y[i1 + i2] * G[(ncols + i2) + G.size1() * b_i];
                    }
                    // 'obils_reduction:60' x_i = max(min(round(alpha),u(i)),l(i));
                    x_i = std::fmax(std::fmin(std::round(dist), upper), 0);
                    // 'obils_reduction:61' if (alpha < l(i) || alpha > u(i) || alpha == x_i)
                    if ((dist < 0) || (dist > upper) || (dist == x_i)) {
                        // 'obils_reduction:62' dist = 1 + abs(alpha - x_i);
                        dist = std::abs(dist - x_i) + 1.0;
                    } else {
                        // 'obils_reduction:63' else
                        // 'obils_reduction:64' dist = 1 - abs(alpha - x_i);
                        dist = 1.0 - std::abs(dist - x_i);
                    }
                    // 'obils_reduction:66' dist_i = dist / norm(G(i:k,i));
                    if (b_i + 1 > b_k) {
                        i1 = 0;
                        i2 = 0;
                    } else {
                        i1 = b_i;
                        i2 = b_k;
                    }
                    loop_ub = i2 - i1;
                    b_G.resize(loop_ub);
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        b_G[i2] = G[(i1 + i2) + G.size1() * b_i];
                    }
                    dist /= norm_2(b_G);
                    // 'obils_reduction:67' if dist_i > maxDist
                    if (dist > maxDist) {
                        // 'obils_reduction:68' maxDist = dist_i;
                        maxDist = dist;
                        // 'obils_reduction:69' j = i;
                        j = b_i + 1;
                        // 'obils_reduction:70' x_j = x_i;
                        x_j = x_i;
                    }
                }
                //  Perform permutations
                // 'obils_reduction:75' p(j:k) = p([j+1:k,j]);
                if (j > b_k) {
                    i1 = 1;
                } else {
                    i1 = j;
                }
                if (b_k < static_cast<double>(j) + 1.0) {
                    c_y.resize(1, 0);
                } else {
                    i2 = b_k - j;
                    c_y.resize(1, i2);
                    loop_ub = i2 - 1;
                    for (i2 = 0; i2 <= loop_ub; i2++) {
                        c_y[i2] = (static_cast<unsigned int>(j) + i2) + 1U;
                    }
                }
                d_y.resize(1, c_y.size2() + 1);
                loop_ub = c_y.size2();
                for (i2 = 0; i2 < loop_ub; i2++) {
                    d_y[i2] = static_cast<int>(c_y[i2]) - 1;
                }
                d_y[c_y.size2()] = j - 1;
                c_y.resize(1, d_y.size2());
                loop_ub = d_y.size2();
                for (i2 = 0; i2 < loop_ub; i2++) {
                    c_y[i2] = p[d_y[i2]];
                }
                loop_ub = c_y.size2();
                for (i2 = 0; i2 < loop_ub; i2++) {
                    p[(i1 + i2) - 1] = c_y[i2];
                }
                // 'obils_reduction:76' l(j:k) = l([j+1:k,j]);
                // 'obils_reduction:77' u(j:k) = u([j+1:k,j]);
                //  Update y, R and G for the new dimension-reduced problem
                // 'obils_reduction:80' y(1:k-1) = y(1:k-1) - R(1:k-1,j) * x_j;
                if (b_k - 1 < 1) {
                    i1 = 0;
                    i2 = 0;
                    loop_ub = 0;
                } else {
                    i1 = b_k - 1;
                    i2 = b_k - 1;
                    loop_ub = b_k - 1;
                }
                if (i1 == i2) {
                    c_y.resize(1, loop_ub);
                    for (i1 = 0; i1 < loop_ub; i1++) {
                        c_y[i1] = y[i1] - R[i1 + R.size1() * (j - 1)] * x_j;
                    }
                    loop_ub = c_y.size2();
                    for (i1 = 0; i1 < loop_ub; i1++) {
                        y[i1] = c_y[i1];
                    }
                }
                // 'obils_reduction:81' R(:,j) = [];
                nrowx = R.size1();
                i1 = R.size2();
                ncols = R.size2() - 1;
                for (loop_ub = j; loop_ub <= ncols; loop_ub++) {
                    for (b_i = 0; b_i < nrowx; b_i++) {
                        R[b_i + R.size1() * (loop_ub - 1)] = R[b_i + R.size1() * loop_ub];
                    }
                }
                if (i1 - 1 < 1) {
                    loop_ub = -1;
                } else {
                    loop_ub = i1 - 2;
                }
                nrowx = R.size1() - 1;
                ncols = R.size1();
                for (i1 = 0; i1 <= loop_ub; i1++) {
                    for (i2 = 0; i2 < ncols; i2++) {
                        R[i2 + (nrowx + 1) * i1] = R[i2 + R.size1() * i1];
                    }
                }
                R.size2(loop_ub + 1);
                // 'obils_reduction:82' G(:,j) = [];
                nrowx = G.size1();
                i1 = G.size2();
                ncols = G.size2() - 1;
                for (loop_ub = j; loop_ub <= ncols; loop_ub++) {
                    for (b_i = 0; b_i < nrowx; b_i++) {
                        G[b_i + G.size1() * (loop_ub - 1)] = G[b_i + G.size1() * loop_ub];
                    }
                }
                if (i1 - 1 < 1) {
                    loop_ub = -1;
                } else {
                    loop_ub = i1 - 2;
                }
                nrowx = G.size1() - 1;
                ncols = G.size1();
                for (i1 = 0; i1 <= loop_ub; i1++) {
                    for (i2 = 0; i2 < ncols; i2++) {
                        G[i2 + (nrowx + 1) * i1] = G[i2 + G.size1() * i1];
                    }
                }
                G.size2(loop_ub + 1);
                // 'obils_reduction:83' for t = j : k - 1
                i1 = b_k - j;
                for (int t{0}; t < i1; t++) {
                    double W_idx_0;
                    double W_idx_2;
                    double W_idx_3;
                    double unnamed_idx_0;
                    double unnamed_idx_1;
                    unsigned int b_t;
                    int i3;
                    b_t = static_cast<unsigned int>(j) + t;
                    //  Triangularize R and G by Givens rotation
                    // 'obils_reduction:85' [W, R([t,t+1],t)] = planerot(R([t,t+1],t));
                    b_i = static_cast<int>(b_t) - 1;
                    unnamed_idx_0 =
                            R[(static_cast<int>(b_t) + R.size1() * (static_cast<int>(b_t) - 1)) -
                              1];
                    unnamed_idx_1 =
                            R[static_cast<int>(b_t) + R.size1() * (static_cast<int>(b_t) - 1)];
                    W_idx_3 =
                            R[static_cast<int>(b_t) + R.size1() * (static_cast<int>(b_t) - 1)];
                    if (W_idx_3 != 0.0) {
                        double r;
                        dist = 3.3121686421112381E-170;
                        maxDist = std::abs(R[(static_cast<int>(b_t) +
                                              R.size1() * (static_cast<int>(b_t) - 1)) -
                                             1]);
                        if (maxDist > 3.3121686421112381E-170) {
                            r = 1.0;
                            dist = maxDist;
                        } else {
                            x_i = maxDist / 3.3121686421112381E-170;
                            r = x_i * x_i;
                        }
                        maxDist = std::abs(
                                R[static_cast<int>(b_t) + R.size1() * (static_cast<int>(b_t) - 1)]);
                        if (maxDist > dist) {
                            x_i = dist / maxDist;
                            r = r * x_i * x_i + 1.0;
                            dist = maxDist;
                        } else {
                            x_i = maxDist / dist;
                            r += x_i * x_i;
                        }
                        r = dist * std::sqrt(r);
                        W_idx_0 = unnamed_idx_0 / r;
                        W_idx_2 = unnamed_idx_1 / r;
                        x_i = -W_idx_3 / r;
                        W_idx_3 = R[(static_cast<int>(b_t) +
                                     R.size1() * (static_cast<int>(b_t) - 1)) -
                                    1] /
                                  r;
                        unnamed_idx_0 = r;
                        unnamed_idx_1 = 0.0;
                    } else {
                        x_i = 0.0;
                        W_idx_2 = 0.0;
                        W_idx_0 = 1.0;
                        W_idx_3 = 1.0;
                    }
                    R[(static_cast<int>(b_t) + R.size1() * (static_cast<int>(b_t) - 1)) - 1] =
                            unnamed_idx_0;
                    R[static_cast<int>(b_t) + R.size1() * (static_cast<int>(b_t) - 1)] =
                            unnamed_idx_1;
                    // 'obils_reduction:86' R([t,t+1],t+1:k-1) = W * R([t,t+1],t+1:k-1);
                    if (static_cast<int>(b_t) + 1 > b_k - 1) {
                        i2 = 0;
                        ncols = 0;
                        i3 = 0;
                    } else {
                        i2 = static_cast<int>(b_t);
                        ncols = b_k - 1;
                        i3 = static_cast<int>(b_t);
                    }
                    loop_ub = ncols - i2;
                    b.resize(2, loop_ub);
                    for (ncols = 0; ncols < loop_ub; ncols++) {
                        nrowx = i2 + ncols;
                        b[2 * ncols] = R[(static_cast<int>(b_t) + R.size1() * nrowx) - 1];
                        b[2 * ncols + 1] = R[static_cast<int>(b_t) + R.size1() * nrowx];
                    }
                    ncols = loop_ub - 1;
                    C.resize(2, loop_ub);
                    for (loop_ub = 0; loop_ub <= ncols; loop_ub++) {
                        nrowx = loop_ub << 1;
                        dist = b[nrowx + 1];
                        C[nrowx] = W_idx_0 * b[nrowx] + W_idx_2 * dist;
                        C[nrowx + 1] = x_i * b[nrowx] + W_idx_3 * dist;
                    }
                    loop_ub = C.size2();
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        nrowx = i3 + i2;
                        R[(static_cast<int>(b_t) + R.size1() * nrowx) - 1] = C[2 * i2];
                        R[static_cast<int>(b_t) + R.size1() * nrowx] = C[2 * i2 + 1];
                    }
                    // 'obils_reduction:87' G([t,t+1],1:t) = W * G([t,t+1],1:t);
                    loop_ub = static_cast<int>(b_t);
                    b.resize(2, static_cast<int>(b_t));
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        b[2 * i2] = G[(static_cast<int>(b_t) + G.size1() * i2) - 1];
                        b[2 * i2 + 1] = G[static_cast<int>(b_t) + G.size1() * i2];
                    }
                    C.resize(2, static_cast<int>(b_t));
                    for (loop_ub = 0; loop_ub <= b_i; loop_ub++) {
                        nrowx = loop_ub << 1;
                        dist = b[nrowx + 1];
                        C[nrowx] = W_idx_0 * b[nrowx] + W_idx_2 * dist;
                        C[nrowx + 1] = x_i * b[nrowx] + W_idx_3 * dist;
                    }
                    loop_ub = C.size2();
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        G[(static_cast<int>(b_t) + G.size1() * i2) - 1] = C[2 * i2];
                        G[static_cast<int>(b_t) + G.size1() * i2] = C[2 * i2 + 1];
                    }
                    //  Apply the Givens rotation W to y
                    // 'obils_reduction:89' y(t:t+1) = W * y(t:t+1);
                    dist = y[static_cast<int>(b_t) - 1];
                    maxDist = y[static_cast<int>(static_cast<double>(b_t) + 1.0) - 1];
                    y[static_cast<int>(b_t) - 1] = W_idx_0 * dist + W_idx_2 * maxDist;
                    y[static_cast<int>(static_cast<double>(b_t) + 1.0) - 1] =
                            x_i * dist + W_idx_3 * maxDist;
                }
            }
            //  Reorder the columns of R0 according to p
            // 'obils_reduction:95' R0 = R0(:,p);

            loop_ub = b_R.size1();
            b_R.assign(R0);
            R0.clear();
            for (i = 0; i < n; i++) {
                for (i1 = 0; i1 < n; i1++) {
                    R0[i1 + R0.size1() * i] = b_R[i1 + b_R.size1() * p[i]];
                    P[i1 + P.size1() * i] = I[i1 + I.size1() * p[i]];
                }
            }

            //  Transform R0 and y0 by the QR factorization
            // 'obils_reduction:98' [Q, R, y] = qrmgs(R0, y0);
//            y.resize(m);
//            for (i = 0; i < m; i++) {
//                y[i] = b_y[i];
//            }
            CILS_Reduction reduction_;
            reduction_.reset(R0, b_y, upper);
            reduction_.pmgs_qr_col(n_t);
//            cout<< reT.run_time;/
            R.assign(reduction_.R);
            y.assign(reduction_.y);
            time = omp_get_wtime() - time;
            return {{}, time, 0};
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
        returnType<scalar, index> plll() {
            scalar zeta, alpha, t_qr, t_plll, sum = 0;

            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            auto reT = mgs_qrp();
            Z.assign(P);
            t_qr = reT.run_time;

            //  ------------------------------------------------------------------
            //  --------  Perform the partial LLL reduction  ---------------------
            //  ------------------------------------------------------------------
            index k = 1, k1, i, j;
            t_plll = omp_get_wtime();

            while (k < n) {
                k1 = k - 1;
                zeta = round(R(k1, k) / R(k1, k1));
                alpha = R(k1, k) - zeta * R(k1, k1);

                if (pow(R(k1, k1), 2) > (1 + 1.e-10) * (pow(alpha, 2) + pow(R(k, k), 2))) {
                    if (zeta != 0) {
                        //Perform a size reduction on R(k-1,k)
                        R(k1, k) = alpha;
                        for (i = 0; i <= k - 2; i++) {
                            R[i + k * n] = R[i + k * n] - zeta * R[i + k1 * n];
                        }
                        for (i = 0; i < n; i++) {
                            Z[i + k * n] -= zeta * Z[i + k1 * n];
                        }
                        //Perform size reductions on R(1:k-2,k)
                        for (i = k - 2; i >= 0; i--) {
                            zeta = round(R[i + k * n] / R(i, i));
                            if (zeta != 0) {
                                for (j = 0; j <= i; j++) {
                                    R(j, k) = R(j, k) - zeta * R(j, i);
                                }
                                for (j = 0; j < n; j++) {
                                    Z(j, k) -= zeta * Z(j, i);
                                }
                            }
                        }
                    }

                    //Permute columns k-1 and k of R and Z
                    for (i = 0; i < n; i++) {
                        std::swap(R[i + k1 * n], R[i + k * n]);
                        std::swap(Z[i + k1 * n], Z[i + k * n]);
                    }

                    //Bring R back to an upper triangular matrix by a Givens rotation
                    scalar G[4] = {};
                    scalar low_tmp[2] = {R(k1, k1), R(k, k1)};
                    helper::planerot<scalar, index>(low_tmp, G);
                    R(k1, k1) = low_tmp[0];
                    R(k, k1) = low_tmp[1];

                    //Combined Rotation.
                    for (i = k; i < n; i++) {
                        low_tmp[0] = R(k1, i);
                        low_tmp[1] = R(k, i);
                        R(k1, i) = G[0] * low_tmp[0] + G[2] * low_tmp[1];
                        R(k, i) = G[1] * low_tmp[0] + G[3] * low_tmp[1];
                    }

                    low_tmp[0] = y[k1];
                    low_tmp[1] = y[k];
                    y[k1] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                    y[k] = G[1] * low_tmp[0] + low_tmp[1] * G[3];

                    if (k > 1)
                        k--;

                } else {
                    k++;
                }
            }

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
        returnType<scalar, index> aspl() {
            scalar zeta, alpha, t_qr, t_plll, sum = 0;
            //Clear Variables:
            y_r.assign(y);
            si_vector s(n, 0);
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            cils::returnType<scalar, index> reT = mgs_qrp();
            Z.assign(P);
            t_qr = reT.run_time;

            //  ------------------------------------------------------------------
            cout << "--------  Perform the all-swap partial LLL reduction -------------------" << endl;
            //  ------------------------------------------------------------------

            index k, k1, i, j, e, b_k;
            index f = true, start = 1, even = true;
            t_plll = omp_get_wtime();
            while (f) {
                f = false;
                for (e = 0; e < n - 1; e++) {
                    for (k = 0; k < e + 1; k++) {
                        b_k = n - k;
                        k1 = b_k - 1;
                        zeta = std::round(R[(b_k + n * (k1)) - 2] /
                                          R[(b_k + n * (b_k - 2)) - 2]);
                        alpha = R[(b_k + n * (k1)) - 2] -
                                zeta * R[(b_k + n * (b_k - 2)) - 2];
                        scalar t = R[(b_k + n * (b_k - 2)) - 2];
                        scalar scale = R[(b_k + n * (k1)) - 1];
                        if ((t * t > 1.0000000001 * (alpha * alpha + scale * scale)) &&
                            (zeta != 0.0)) {
                            for (j = 0; j < k1; j++) {
                                R[j + n * (k1)] -= zeta * R[j + n * (b_k - 2)];
                            }
                            for (j = 0; j < n; j++) {
                                Z[j + n * (k1)] -= zeta * Z[j + n * (b_k - 2)];
                            }
//                            for (int b_i{0}; b_i < b_k - 2; b_i++) {
//                                index b_n = (b_k - b_i) - 3;
//                                zeta = std::round(R[b_n + n * (k1)] / R[b_n + n * b_n]);
//                                if (zeta != 0.0) {
//                                    for (j = 0; j <= b_n; j++) {
//                                        R[j + n * (k1)] -= zeta * R[j + n * b_n];
//                                    }
//                                    for (j = 0; j < n; j++) {
//                                        Z[j + n * (k1)] -= zeta * Z[j + n * b_n];
//                                    }
//                                }
//                            }
                        }
                    }
                }
                for (k = start; k < n; k += 2) {
                    k1 = k - 1;
                    if (pow(R(k1, k1), 2) > (1 + 1e-10) * (pow(R(k1, k), 2) + pow(R(k, k), 2))) {
                        f = true;
                        s[k] = 1;
                        for (i = 0; i < n; i++) {
                            std::swap(R[i + k1 * n], R[i + k * n]);
                            std::swap(Z[i + k1 * n], Z[i + k * n]);
                        }
                    }
                }
                for (k = start; k < n; k += 2) {
                    if (s[k]) {
                        s[k] = 0;
                        k1 = k - 1;
                        //Bring R back to an upper triangular matrix by a Givens rotation
                        scalar G[4] = {};
                        scalar low_tmp[2] = {R(k1, k1), R(k, k1)};
                        helper::planerot<scalar, index>(low_tmp, G);
                        R(k1, k1) = low_tmp[0];
                        R(k, k1) = low_tmp[1];

                        //Combined Rotation.
                        for (i = k; i < n; i++) {
                            low_tmp[0] = R(k1, i);
                            low_tmp[1] = R(k, i);
                            R(k1, i) = G[0] * low_tmp[0] + G[2] * low_tmp[1];
                            R(k, i) = G[1] * low_tmp[0] + G[3] * low_tmp[1];
                        }

                        low_tmp[0] = y[k1];
                        low_tmp[1] = y[k];
                        y[k1] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                        y[k] = G[1] * low_tmp[0] + low_tmp[1] * G[3];
                    }
                }
                if (even) {
                    even = false;
                    start = 2;
                } else {
                    even = true;
                    start = 1;
                }
                if (!f) {
                    for (k = start; k < n; k += 2) {
                        k1 = k - 1;
                        if (pow(R(k1, k1), 2) > (1 + 1e-10) * (pow(R(k1, k), 2) + pow(R(k, k), 2))) {
                            f = true;
                            s[k] = 1;
                            for (i = 0; i < n; i++) {
                                std::swap(R[i + k1 * n], R[i + k * n]);
                                std::swap(Z[i + k1 * n], Z[i + k * n]);
                            }
                        }
                    }
                    for (k = start; k < n; k += 2) {
                        if (s[k]) {
                            s[k] = 0;
                            k1 = k - 1;
                            //Bring R back to an upper triangular matrix by a Givens rotation
                            scalar G[4] = {};
                            scalar low_tmp[2] = {R(k1, k1), R(k, k1)};
                            helper::planerot<scalar, index>(low_tmp, G);
                            R(k1, k1) = low_tmp[0];
                            R(k, k1) = low_tmp[1];

                            //Combined Rotation.
                            for (i = k; i < n; i++) {
                                low_tmp[0] = R(k1, i);
                                low_tmp[1] = R(k, i);
                                R(k1, i) = G[0] * low_tmp[0] + G[2] * low_tmp[1];
                                R(k, i) = G[1] * low_tmp[0] + G[3] * low_tmp[1];
                            }

                            low_tmp[0] = y[k1];
                            low_tmp[1] = y[k];
                            y[k1] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                            y[k] = G[1] * low_tmp[0] + low_tmp[1] * G[3];
                        }
                    }
                }
            }
            t_plll = omp_get_wtime() - t_plll;

//            verbose = true;
//            //lll_validation();
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
        returnType<scalar, index> aspl_p() {
            scalar zeta, alpha, t_qr, t_plll, sum = 0;
            //Clear Variables:
            y_r.assign(y);
            si_vector s(n, 0);
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
//            cils::returnType<scalar, index> reT = mgs_qrp();
            cils::returnType<scalar, index> reT = mgs_qr_col();
//            Z.assign(P);
            Z.assign(I);
            t_qr = reT.run_time;

            //  ------------------------------------------------------------------
//            cout << "--------  Perform the all-swap LLL-P reduction -------------------" << endl;
            //  ------------------------------------------------------------------

            index k, k1, i, j, e, b_k;
            index f = true, start = 1, even = true;
            t_plll = omp_get_wtime();
            while (f) {
                f = false;
                for (k = start; k < n; k += 2) {
                    k1 = k - 1;
                    if (pow(R(k1, k1), 2) > (1 + 1e-10) * (pow(R(k1, k), 2) + pow(R(k, k), 2))) {
                        f = true;
                        s[k] = 1;
                        for (i = 0; i < n; i++) {
                            std::swap(R[i + k1 * n], R[i + k * n]);
                            std::swap(Z[i + k1 * n], Z[i + k * n]);
                        }
                    }
                }
                for (k = start; k < n; k += 2) {
                    if (s[k]) {
                        s[k] = 0;
                        k1 = k - 1;
                        //Bring R back to an upper triangular matrix by a Givens rotation
                        scalar G[4] = {};
                        scalar low_tmp[2] = {R(k1, k1), R(k, k1)};
                        helper::planerot<scalar, index>(low_tmp, G);
                        R(k1, k1) = low_tmp[0];
                        R(k, k1) = low_tmp[1];

                        //Combined Rotation.
                        for (i = k; i < n; i++) {
                            low_tmp[0] = R(k1, i);
                            low_tmp[1] = R(k, i);
                            R(k1, i) = G[0] * low_tmp[0] + G[2] * low_tmp[1];
                            R(k, i) = G[1] * low_tmp[0] + G[3] * low_tmp[1];
                        }

                        low_tmp[0] = y[k1];
                        low_tmp[1] = y[k];
                        y[k1] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                        y[k] = G[1] * low_tmp[0] + low_tmp[1] * G[3];
                    }
                }
                if (even) {
                    even = false;
                    start = 2;
                } else {
                    even = true;
                    start = 1;
                }
                if (!f) {
                    for (k = start; k < n; k += 2) {
                        k1 = k - 1;
                        if (pow(R(k1, k1), 2) > (1 + 1e-10) * (pow(R(k1, k), 2) + pow(R(k, k), 2))) {
                            f = true;
                            s[k] = 1;
                            for (i = 0; i < n; i++) {
                                std::swap(R[i + k1 * n], R[i + k * n]);
                                std::swap(Z[i + k1 * n], Z[i + k * n]);
                            }
                        }
                    }
                    for (k = start; k < n; k += 2) {
                        if (s[k]) {
                            s[k] = 0;
                            k1 = k - 1;
                            //Bring R back to an upper triangular matrix by a Givens rotation
                            scalar G[4] = {};
                            scalar low_tmp[2] = {R(k1, k1), R(k, k1)};
                            helper::planerot<scalar, index>(low_tmp, G);
                            R(k1, k1) = low_tmp[0];
                            R(k, k1) = low_tmp[1];

                            //Combined Rotation.
                            for (i = k; i < n; i++) {
                                low_tmp[0] = R(k1, i);
                                low_tmp[1] = R(k, i);
                                R(k1, i) = G[0] * low_tmp[0] + G[2] * low_tmp[1];
                                R(k, i) = G[1] * low_tmp[0] + G[3] * low_tmp[1];
                            }

                            low_tmp[0] = y[k1];
                            low_tmp[1] = y[k];
                            y[k1] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                            y[k] = G[1] * low_tmp[0] + low_tmp[1] * G[3];
                        }
                    }
                }
            }
            t_plll = omp_get_wtime() - t_plll;

//            verbose = true;
//            //lll_validation();
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
        returnType<scalar, index> paspl(index n_c) {
            scalar zeta, alpha, t_qr, t_plll;
            //Clear Variables:

            y_r.assign(y);
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------

            auto reT = pmgs_qrp(n_c);
            t_qr = reT.run_time;
            Z.assign(P);

//            init_R_A();
            //  ------------------------------------------------------------------
//            cout << "--------  Perform the PASPL reduction -------------------" << endl;
            //  ------------------------------------------------------------------


            auto s = new int[n]();
            scalar G[4] = {};
            auto R_d = R.data();
            auto Z_d = Z.data();
            index k, k1, i, j, e, i1, b_k, k2;
            index f = true, start = 1, even = true;
            index iter = 0;

#pragma omp parallel default(shared) num_threads(n_c)
            {}

            t_plll = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_c) private(zeta, alpha, e, k, k1, i, j, i1, b_k) firstprivate(G)
            {
                while (f && iter < 1e6) {
#pragma omp barrier
#pragma omp atomic write
                    f = false;
#pragma omp atomic
                    iter++;

                    for (e = 0; e < n - 1; e++) {
#pragma omp for schedule(static, 1) nowait
                        for (k = 0; k < e + 1; k++) {
                            b_k = n - k - 2;
                            k1 = b_k + 1;
                            zeta = std::round(R_d[b_k + k1 * n] / R_d[b_k + b_k * n]);
                            alpha = R_d[b_k + k1 * n] - zeta * R_d[b_k + b_k * n];
                            if ((pow(R_d[b_k + b_k * n], 2) >
                                 1.0000000001 * (alpha * alpha + pow(R_d[b_k + 1 + k1 * n], 2))) &&
                                (zeta != 0.0)) {
//                                    omp_set_lock(&lock[k]);
                                for (j = 0; j < k1; j++) {
                                    R_d[j + k1 * n] -= zeta * R_d[j + b_k * n];//[j + n * (b_k - 2)];
                                }
                                for (j = 0; j < n; j++) {
                                    Z_d[j + k1 * n] -= zeta * Z_d[j + b_k * n];
                                }
//                                for (int b_i{0}; b_i < b_k; b_i++) {
//                                    index b_n = (b_k - b_i) - 1;
//                                    zeta = std::round(R_d(b_n, k1) / R_d(b_n, b_n));
//                                    if (zeta != 0.0) {
//                                        for (j = 0; j <= b_n; j++) {
//                                            R_d[j + k1 * n] -= zeta * R_d(j, b_n);
//                                        }
//                                        for (j = 0; j < n; j++) {
//                                            Z_d[j + k1 * n] -= zeta * Z_d(j, b_n);
//                                        }
//                                    }
//                                }
                            }

                        }
                    }

#pragma omp barrier
#pragma omp for schedule(static)
                    for (k = start; k < n; k += 2) {
                        k1 = k - 1;
                        if (pow(R_d[k1 + k1 * n], 2) >
                            (1 + 1.e-10) * (pow(R_d[k1 + k * n], 2) + pow(R_d[k + k * n], 2))) {
                            f = true;
                            s[k] = 1;
                            for (i = 0; i < n; i++) {
                                std::swap(R_d[i + k1 * n], R_d[i + k * n]);
                                std::swap(Z_d[i + k1 * n], Z_d[i + k * n]);
                            }
                        }
                    }

#pragma omp for schedule(static)
                    for (k = start; k < n; k += 2) {
                        if (s[k]) {
                            s[k] = 0;
                            k1 = k - 1;
                            //Bring R_d back to an upper triangular matrix by a Givens rotation

                            scalar low_tmp[2] = {R_d[k1 + k1 * n], R_d[k + k1 * n]};
                            helper::planerot<scalar, index>(low_tmp, G);
                            R_d[k1 + k1 * n] = low_tmp[0];
                            R_d[k + k1 * n] = low_tmp[1];

                            //Combined Rotation.
                            for (i = k; i < n; i++) {
                                low_tmp[0] = R_d[k1 + i * n];
                                low_tmp[1] = R_d[k + i * n];
                                R_d[k1 + i * n] = G[0] * low_tmp[0] + G[2] * low_tmp[1];
                                R_d[k + i * n] = G[1] * low_tmp[0] + G[3] * low_tmp[1];
                            }

                            low_tmp[0] = y[k1];
                            low_tmp[1] = y[k];
                            y[k1] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                            y[k] = G[1] * low_tmp[0] + low_tmp[1] * G[3];
                        }
                    }

#pragma omp single
                    {
                        if (even) {
                            even = false;
                            start = 2;
                        } else {
                            even = true;
                            start = 1;
                        }
                    }

                    if (!f) {
#pragma omp barrier
#pragma omp for schedule(static)
                        for (k = start; k < n; k += 2) {
                            k1 = k - 1;

                            if (pow(R_d[k1 + k1 * n], 2) >
                                (1 + 1.e-10) * (pow(R_d[k1 + k * n], 2) + pow(R_d[k + k * n], 2))) {
                                f = true;
                                s[k] = 1;
                                for (i = 0; i < n; i++) {
                                    std::swap(R_d[i + k1 * n], R_d[i + k * n]);
                                    std::swap(Z_d[i + k1 * n], Z_d[i + k * n]);
                                }
                            }
                        }
#pragma omp for schedule(static)
                        for (k = start; k < n; k += 2) {
                            if (s[k]) {
                                s[k] = 0;
                                k1 = k - 1;
                                //Bring R_d back to an upper triangular matrix by a Givens rotation

                                scalar low_tmp[2] = {R_d[k1 + k1 * n], R_d[k + k1 * n]};
                                helper::planerot<scalar, index>(low_tmp, G);
                                R_d[k1 + k1 * n] = low_tmp[0];
                                R_d[k + k1 * n] = low_tmp[1];

                                //Combined Rotation.
                                for (i = k; i < n; i++) {
                                    low_tmp[0] = R_d[k1 + i * n];
                                    low_tmp[1] = R_d[k + i * n];
                                    R_d[k1 + i * n] = G[0] * low_tmp[0] + G[2] * low_tmp[1];
                                    R_d[k + i * n] = G[1] * low_tmp[0] + G[3] * low_tmp[1];
                                }

                                low_tmp[0] = y[k1];
                                low_tmp[1] = y[k];
                                y[k1] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                                y[k] = G[1] * low_tmp[0] + low_tmp[1] * G[3];
                            }
                        }
                    }
                }
            }

            t_plll = omp_get_wtime() - t_plll;

            k = 1;
            while (k < n) {
                k1 = k - 1;

                if (pow(R(k1, k1), 2) > (1 + 1.e-10) * (pow(R(k1, k), 2) + pow(R(k, k), 2))) {
                    cerr << "Failed:" << k1 << endl;
                }
                k++;
            }
            delete[] s;
//            verbose = true;
//            lll_validation();
            return {{}, t_qr, t_plll};
        }

        returnType<scalar, index> paspl_p(index n_c) {
            scalar zeta, alpha, t_qr, t_plll;
            //Clear Variables:

            y_r.assign(y);
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------

            auto reT = pmgs_qrp(n_c);
//            cout << R.size1() << "," << R.size2();
            t_qr = reT.run_time;
            Z.assign(P);

            //  ------------------------------------------------------------------
//            cout << "--------  Perform the PASPL reduction -------------------" << endl;
            //  ------------------------------------------------------------------

            auto s = new int[n]();
            scalar G[4] = {};
            auto R_d = R.data();
            auto Z_d = Z.data();
            index k, k1, i, j, e, i1, b_k, k2;
            index f = true, start = 1, even = true;
            index iter = 0;

#pragma omp parallel default(shared) num_threads(n_c)
            {}

            t_plll = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_c) private(zeta, alpha, e, k, k1, i, j, i1, b_k) firstprivate(G)
            {
                while (f && iter < 1e6) {
#pragma omp barrier
#pragma omp atomic write
                    f = false;
#pragma omp atomic
                    iter++;
#pragma omp barrier
#pragma omp for schedule(static)
                    for (k = start; k < n; k += 2) {
                        k1 = k - 1;
                        if (pow(R_d[k1 + k1 * n], 2) > (pow(R_d[k1 + k * n], 2) + pow(R_d[k + k * n], 2))) {
                            f = true;
                            s[k] = 1;
                            for (i = 0; i < n; i++) {
                                std::swap(R_d[i + k1 * n], R_d[i + k * n]);
                                std::swap(Z_d[i + k1 * n], Z_d[i + k * n]);
                            }
                        }
                    }

#pragma omp for schedule(static)
                    for (k = start; k < n; k += 2) {
                        if (s[k]) {
                            s[k] = 0;
                            k1 = k - 1;
                            //Bring R_d back to an upper triangular matrix by a Givens rotation

                            scalar low_tmp[2] = {R_d[k1 + k1 * n], R_d[k + k1 * n]};
                            helper::planerot<scalar, index>(low_tmp, G);
                            R_d[k1 + k1 * n] = low_tmp[0];
                            R_d[k + k1 * n] = low_tmp[1];

                            //Combined Rotation.
                            for (i = k; i < n; i++) {
                                low_tmp[0] = R_d[k1 + i * n];
                                low_tmp[1] = R_d[k + i * n];
                                R_d[k1 + i * n] = G[0] * low_tmp[0] + G[2] * low_tmp[1];
                                R_d[k + i * n] = G[1] * low_tmp[0] + G[3] * low_tmp[1];
                            }

                            low_tmp[0] = y[k1];
                            low_tmp[1] = y[k];
                            y[k1] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                            y[k] = G[1] * low_tmp[0] + low_tmp[1] * G[3];
                        }
                    }

#pragma omp single
                    {
                        if (even) {
                            even = false;
                            start = 2;
                        } else {
                            even = true;
                            start = 1;
                        }
                    }

                    if (!f) {
#pragma omp barrier
#pragma omp for schedule(static)
                        for (k = start; k < n; k += 2) {
                            k1 = k - 1;

                            if (pow(R_d[k1 + k1 * n], 2) > (pow(R_d[k1 + k * n], 2) + pow(R_d[k + k * n], 2))) {
                                f = true;
                                s[k] = 1;
                                for (i = 0; i < n; i++) {
                                    std::swap(R_d[i + k1 * n], R_d[i + k * n]);
                                    std::swap(Z_d[i + k1 * n], Z_d[i + k * n]);
                                }
                            }
                        }
#pragma omp for schedule(static)
                        for (k = start; k < n; k += 2) {
                            if (s[k]) {
                                s[k] = 0;
                                k1 = k - 1;
                                //Bring R_d back to an upper triangular matrix by a Givens rotation

                                scalar low_tmp[2] = {R_d[k1 + k1 * n], R_d[k + k1 * n]};
                                helper::planerot<scalar, index>(low_tmp, G);
                                R_d[k1 + k1 * n] = low_tmp[0];
                                R_d[k + k1 * n] = low_tmp[1];

                                //Combined Rotation.
                                for (i = k; i < n; i++) {
                                    low_tmp[0] = R_d[k1 + i * n];
                                    low_tmp[1] = R_d[k + i * n];
                                    R_d[k1 + i * n] = G[0] * low_tmp[0] + G[2] * low_tmp[1];
                                    R_d[k + i * n] = G[1] * low_tmp[0] + G[3] * low_tmp[1];
                                }

                                low_tmp[0] = y[k1];
                                low_tmp[1] = y[k];
                                y[k1] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                                y[k] = G[1] * low_tmp[0] + low_tmp[1] * G[3];
                            }
                        }
                    }
                }
            }

            t_plll = omp_get_wtime() - t_plll;

//            k = 1;
//            while (k < n) {
//                k1 = k - 1;
//
//                if (pow(R(k1, k1), 2) > (pow(R(k1, k), 2) + pow(R(k, k), 2))) {
//                    cerr << "Failed:" << k1 << endl;
//                }
//                k++;
//            }
            delete[] s;
//            verbose = true;
//            lll_validation();
            return {{}, t_qr, t_plll};
        }
    };
}