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

//        void init_R_A() {
//            R_A.resize(n / 2 * (1 + n), false);
//            index idx = 0;
//            for (index row = 0; row < R.size1(); row++) {
//                for (index col = row; col < R.size2(); col++) {
//                    R_A[idx] = R(row, col);
//                    idx++;
//                }
//            }
//        }


        /**
         * Evaluating the LLL decomposition
         * @return
         */
        returnType <scalar, index> lll_validation() {
            printf("====================[ TEST | LLL_VALIDATE ]==================================\n");
            printf("[ INFO: in LLL validation]\n");
            scalar sum, error = 0, det;
            b_matrix B_T;
            prod(B, Z, B_T);
            b_matrix R_I;

            helper::inv<scalar, index>(R, R_I);
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
                    error += fabs(B_T(i) - B(i));
                }
            }

            return error;
        }

    public:
        //B --> A
        b_eye_matrix I{};
        b_matrix B{}, R{}, Q{}, Z{}, P{}, R_A{};
        b_vector y{}, y_r{}, p{};

        explicit CILS_Reduction(CILS <scalar, index> &cils) : CILS_Reduction(cils.A, cils.y, cils.lower, cils.upper) {}

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

        void reset(CILS <scalar, index> &cils) {
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

        /**
         * Serial version of MGS QR-factorization using modified Gram-Schmidt algorithm, row-oriented
         * Results are stored in the class object.
         */
        returnType <scalar, index> mgs_qr() {
            //Clear Variables:
            R.clear();
            Q.assign(B);

            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            scalar t_qr = omp_get_wtime();
            for (index k = 0; k < n; k++) {
                scalar sum = 0;
                for (index i = 0; i < n; i++) {
                    sum += pow(Q(i, k), 2);
                }
                R(k, k) = sqrt(sum);
                for (index i = 0; i < n; i++) {
                    Q(i, k) = Q(i, k) / R(k, k);
                }
                for (index j = 0; j < n; j++) {
                    if (j > k) {
                        R(k, j) = 0;
                        for (index i = 0; i < n; i++) {
                            R(k, j) += Q(i, k) * Q(i, j);
                        }
                        for (index i = 0; i < n; i++) {
                            Q(i, j) -= R(k, j) * Q(i, k);
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

        returnType <scalar, index> mgs_qrp() {
            //Clear Variables:
            R.resize(n, n);
            Q.assign(B);
            P.assign(I);
            sd_vector s1(n, 0);
            sd_vector s2(n, 0);
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            scalar t_qr = omp_get_wtime();
            scalar sum = 0;

            for (index k = 0; k < n; k++) {
                sum = 0;
                for (index i = 0; i < n; i++) {
                    sum += pow(Q(i, k), 2);
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
//                index i1, i2, idx;
//                if (k + 1 > n) {
//                    i1 = 0;
//                    i2 = 0;
//                    idx = 0;
//                } else {
//                    i1 = k;
//                    i2 = n;
//                    idx = k;
//                }
//                index loop_ub = i2 - i1;
//                sd_vector varargin_1(loop_ub, 0);
//                for (i2 = 0; i2 < loop_ub; i2++) {
//                    varargin_1[i2] = s1[i1 + i2] - s2[i1 + i2];
//                }
//                index last = loop_ub;
//                if (loop_ub <= 2) {
//                    if (loop_ub == 1) {
//                        idx = 1;
//                    } else if ((varargin_1[0] > varargin_1[loop_ub - 1]) ||
//                               (std::isnan(varargin_1[0]) &&
//                                (!std::isnan(varargin_1[loop_ub - 1])))) {
//                        idx = loop_ub;
//                    } else {
//                        idx = 1;
//                    }
//                } else {
//                    if (!std::isnan(varargin_1[0])) {
//                        idx = 1;
//                    } else {
//                        bool exitg1;
//                        idx = 0;
//                        index j = 2;
//                        exitg1 = false;
//                        while ((!exitg1) && (j <= last)) {
//                            if (!std::isnan(varargin_1[j - 1])) {
//                                idx = j;
//                                exitg1 = true;
//                            } else {
//                                j++;
//                            }
//                        }
//                    }
//                    if (idx == 0) {
//                        idx = 1;
//                    } else {
//                        l = varargin_1[idx - 1];
//                        i1 = idx + 1;
//                        index j;
//                        for (j = i1; j <= last; j++) {
//                            index s_idx_1 = varargin_1[j - 1];
//                            if (l > s_idx_1) {
//                                l = s_idx_1;
//                                idx = j;
//                            }
//                        }
//                    }
//                }
//                // 'qrmgs_cp:14' l = l + j - 1;
//                l = idx + k - 1;

                if (l > k) {
//                    cout << l << ",";
                    scalar temp;
                    for (index i = 0; i < n; i++) {
                        temp = R(i, l);
                        R(i, l) = R(i, k);
                        R(i, k) = temp;
                        temp = R(i, l);
                        P(i, l) = P(i, k);
                        P(i, k) = temp;
                        temp = Q(i, l);
                        Q(i, l) = Q(i, k);
                        Q(i, k) = temp;
                    }
                    std::swap(s1[l], s1[k]);
                    std::swap(s2[l], s2[k]);
                }
                sum = 0;
                for (index i = 0; i < n; i++) {
                    sum += pow(Q(i, k), 2);
                }
                R(k, k) = sqrt(sum);
                for (index i = 0; i < n; i++) {
                    Q(i, k) = Q(i, k) / R(k, k);
                }
                for (index j = k + 1; j < n; j++) {
                    R(k, j) = 0;
                    for (index i = 0; i < n; i++) {
                        R(k, j) += Q(i, k) * Q(i, j);
                    }
                    s2[j] = s2[j] + pow(R(k, j), 2);
                    for (index i = 0; i < n; i++) {
                        Q(i, j) -= R(k, j) * Q(i, k);
                    }
                }
            }
            cout << endl;
            b_matrix Q_T = trans(Q);
            prod(Q_T, y, y_r);
            y.assign(y_r);
            t_qr = omp_get_wtime() - t_qr;
//            qrp_validation();
            return {{}, t_qr, 0};
        }

        returnType <scalar, index> mgs_qrp2() {
            b_matrix b_Q;
            b_matrix s;
            b_matrix varargin_1;
            b_vector b_A;
            b_matrix piv;
            double l;
            int b_l[2];
            int iv[2];
            int i;
            int i1;
            int idx;
            int j;
            int k;
            int last;
            int loop_ub;

            // 'qrmgs_cp:3' R = zeros(n, n);
            R.resize(n, n);
            loop_ub = n * n;
            for (i = 0; i < loop_ub; i++) {
                R[i] = 0.0;
            }
            // 'qrmgs_cp:4' s = zeros(2, n);
            s.resize(2, n);
            loop_ub = n << 1;
            for (i = 0; i < loop_ub; i++) {
                s[i] = 0.0;
            }
            // 'qrmgs_cp:5' piv = 1:n;
            if (n < 1) {
                piv.resize(1, 0);
            } else {
                piv.resize(1, n);
                loop_ub = n - 1;
                for (i = 0; i <= loop_ub; i++) {
                    piv[i] = i + 1U;
                }
            }
            // 'qrmgs_cp:6' Q = A;
            Q.resize(n, n);
            loop_ub = n * n;
            for (i = 0; i < loop_ub; i++) {
                Q[i] = B[i];
            }
            // 'qrmgs_cp:8' for k = 1:n+
            scalar t_qr = omp_get_wtime();
            i = n;
            for (k = 0; k < i; k++) {
                // 'qrmgs_cp:9' s(1, k) = norm(Q(:,k))^2;
                loop_ub = n;
                b_A.resize(n);
                for (i1 = 0; i1 < loop_ub; i1++) {
                    b_A[i1] = B[i1 + n * k];
                }
                double sum = 0;
                for (index ii = 0; ii < n; ii++) {
                    sum += pow(Q(ii, k), 2);
                }
                s[2 * k] = sum;
            }
            // 'qrmgs_cp:12' for j = 1:n
            i = n;
            for (j = 0; j < i; j++) {
                double s_idx_1;
                int i2;
                unsigned int piv_idx_1;
                // 'qrmgs_cp:13' [~, l] = min(s(1,j:n)-s(2,j:n));
                if (j + 1 > n) {
                    i1 = 0;
                    i2 = 0;
                    idx = 0;
                } else {
                    i1 = j;
                    i2 = n;
                    idx = j;
                }
                loop_ub = i2 - i1;
                varargin_1.resize(1, loop_ub);
                for (i2 = 0; i2 < loop_ub; i2++) {
                    varargin_1[i2] = s[2 * (i1 + i2)] - s[2 * (idx + i2) + 1];
                }
                last = varargin_1.size2();
                if (varargin_1.size2() <= 2) {
                    if (varargin_1.size2() == 1) {
                        idx = 1;
                    } else if ((varargin_1[0] > varargin_1[varargin_1.size2() - 1]) ||
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
                        while ((!exitg1) && (k <= last)) {
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
                        for (k = i1; k <= last; k++) {
                            s_idx_1 = varargin_1[k - 1];
                            if (l > s_idx_1) {
                                l = s_idx_1;
                                idx = k;
                            }
                        }
                    }
                }
                // 'qrmgs_cp:14' l = l + j - 1;
                l = (static_cast<double>(idx) + (static_cast<double>(j) + 1.0)) - 1.0;
                // 'qrmgs_cp:15' if l > j
                if (l > static_cast<double>(j) + 1.0) {
//                    cout << l - 1 << ",";
                    double s_idx_2;
                    double s_idx_3;
                    // 'qrmgs_cp:16' piv([j,l]) = piv([l,j]);
                    piv_idx_1 = piv[j];
                    // 'qrmgs_cp:17' s(:,[j,l]) = s(:,[l,j]);
                    piv[j] = piv[static_cast<int>(static_cast<unsigned int>(l)) - 1];
                    s_idx_1 = s[2 * (static_cast<int>(static_cast<unsigned int>(l)) - 1) + 1];
                    piv[static_cast<int>(static_cast<unsigned int>(l)) - 1] = piv_idx_1;
                    s_idx_2 = s[2 * j];
                    s_idx_3 = s[2 * j + 1];
                    s[2 * j] = s[2 * (static_cast<int>(static_cast<unsigned int>(l)) - 1)];
                    s[2 * j + 1] = s_idx_1;
                    s[2 * (static_cast<int>(static_cast<unsigned int>(l)) - 1)] = s_idx_2;
                    s[2 * (static_cast<int>(static_cast<unsigned int>(l)) - 1) + 1] = s_idx_3;
                    // 'qrmgs_cp:18' Q(:,[j,l]) = Q(:,[l;j]);
                    iv[0] = j;
                    i1 = static_cast<int>(static_cast<unsigned int>(l)) - 1;
                    iv[1] = i1;
                    last = Q.size1() - 1;
                    b_l[0] = i1;
                    b_l[1] = j;
                    b_Q.resize(Q.size1(), 2);
                    for (i2 = 0; i2 < 2; i2++) {
                        for (idx = 0; idx <= last; idx++) {
                            b_Q[idx + b_Q.size1() * i2] = Q[idx + Q.size1() * b_l[i2]];
                        }
                    }
                    loop_ub = b_Q.size1();
                    for (i2 = 0; i2 < 2; i2++) {
                        for (idx = 0; idx < loop_ub; idx++) {
                            Q[idx + Q.size1() * iv[i2]] = b_Q[idx + b_Q.size1() * i2];
                        }
                    }
                    // 'qrmgs_cp:19' R(:,[j,l]) = R(:,[l;j]);
                    iv[0] = j;
                    iv[1] = i1;
                    last = R.size1() - 1;
                    b_l[0] = i1;
                    b_l[1] = j;
                    b_Q.resize(R.size1(), 2);
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 <= last; i2++) {
                            b_Q[i2 + b_Q.size1() * i1] = R[i2 + R.size1() * b_l[i1]];
                        }
                    }
                    loop_ub = b_Q.size1();
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 < loop_ub; i2++) {
                            R[i2 + R.size1() * iv[i1]] = b_Q[i2 + b_Q.size1() * i1];
                        }
                    }
                }
                // 'qrmgs_cp:22' R(j,j) = norm(Q(:,j));
                loop_ub = Q.size1();
                b_A.resize(Q.size1());
                for (i1 = 0; i1 < loop_ub; i1++) {
                    b_A[i1] = Q[i1 + Q.size1() * j];
                }
                double sum = 0;
                for (index ii = 0; ii < n; ii++) {
                    sum += pow(Q(ii, j), 2);
                }

                R[j + R.size1() * j] = sqrt(sum);
                // 'qrmgs_cp:23' Q(:,j) = Q(:,j)/R(j,j);
                last = Q.size1() - 1;
                l = R[j + R.size1() * j];
                b_A.resize(Q.size1());
                for (i1 = 0; i1 <= last; i1++) {
                    b_A[i1] = Q[i1 + Q.size1() * j] / l;
                }
                loop_ub = b_A.size();
                for (i1 = 0; i1 < loop_ub; i1++) {
                    Q[i1 + Q.size1() * j] = b_A[i1];
                }
                // 'qrmgs_cp:24' for k = j+1:n
                i1 = n - j;
                for (k = 0; k <= i1 - 2; k++) {
                    piv_idx_1 = (static_cast<unsigned int>(j) + k) + 2U;
                    // 'qrmgs_cp:25' R(j,k) = Q(:,j)'* Q(:,k);
                    loop_ub = Q.size1();
                    l = 0.0;
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        l += Q[i2 + Q.size1() * j] *
                             Q[i2 + Q.size1() * (static_cast<int>(piv_idx_1) - 1)];
                    }
                    R[j + R.size1() * (static_cast<int>(piv_idx_1) - 1)] = l;
                    // 'qrmgs_cp:26' s(2, k) = s(2, k) + R(j,k)^2;
                    l = R[j + R.size1() * (static_cast<int>(piv_idx_1) - 1)];
                    s[2 * (static_cast<int>(piv_idx_1) - 1) + 1] =
                            s[2 * (static_cast<int>(piv_idx_1) - 1) + 1] + l * l;
                    // 'qrmgs_cp:27' Q(:,k) = Q(:,k) - Q(:,j)*R(j,k);
                    last = Q.size1() - 1;
                    b_A.resize(Q.size1());
                    for (i2 = 0; i2 <= last; i2++) {
                        b_A[i2] = Q[i2 + Q.size1() * (static_cast<int>(piv_idx_1) - 1)] -
                                  Q[i2 + Q.size1() * j] * l;
                    }
                    loop_ub = b_A.size();
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        Q[i2 + Q.size1() * (static_cast<int>(piv_idx_1) - 1)] = b_A[i2];
                    }
                }
            }
            // 'qrmgs_cp:30' d = 1;
            // 'qrmgs_cp:31' y = Q' * y;
            last = Q.size2() - 1;
            idx = Q.size1();
            b_A.resize(m);
            loop_ub = m;
            for (i = 0; i < loop_ub; i++) {
                b_A[i] = y[i];
            }
            y.resize(Q.size2());
            for (loop_ub = 0; loop_ub <= last; loop_ub++) {
                y[loop_ub] = 0.0;
            }
            for (k = 0; k < idx; k++) {
                for (loop_ub = 0; loop_ub <= last; loop_ub++) {
                    y[loop_ub] = y[loop_ub] + Q[loop_ub * Q.size1() + k] * b_A[k];
                }
            }
            // 'qrmgs_cp:32' P = zeros(n, n);
            P.resize(n, n);
            loop_ub = n * n;
            for (i = 0; i < loop_ub; i++) {
                P[i] = 0.0;
            }
            // 'qrmgs_cp:33' for j = 1 : n
            i = n;
            for (j = 0; j < i; j++) {
                // 'qrmgs_cp:34' P(piv(j),j) = 1;
                P[(static_cast<int>(piv[j]) + P.size1() * j) - 1] = 1.0;
            }
            //  d = det(Q'*Q);
            //  if abs(d-1)>1e-3
            //      d
            //      R
            //      [R_,piv,y_] = qrmcp(A,y);
            //      R_
            //  end
            // 'qrmgs_cp:30' d = 1;
            // 'qrmgs_cp:31' y = Q' * y;
            b_matrix Q_T = trans(Q);
            prod(Q_T, y, y_r);
            y.assign(y_r);
            t_qr = omp_get_wtime() - t_qr;
            cout << endl;
//            qrp_validation();
            return {{}, t_qr, 0};

        }

        returnType <scalar, index> pmgs_qrp(index n_proc) {
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
#pragma omp parallel default(shared) num_threads(n_proc)
            {}
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------

#pragma omp parallel default(shared) num_threads(n_proc) private(sum, i, j, k, l, min_norm)
            {
#pragma omp for schedule(static, 1)
                for (k = 0; k < n; k++) {
                    sum = 0;
                    for (i = 0; i < n; i++) {
                        sum += pow(Q(i, k), 2);
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
                                temp = R(i, l);
                                R(i, l) = R(i, k);
                                R(i, k) = temp;
                                temp = R(i, l);
                                P(i, l) = P(i, k);
                                P(i, k) = temp;
                                temp = Q(i, l);
                                Q(i, l) = Q(i, k);
                                Q(i, k) = temp;
                            }
                            std::swap(s1[l], s1[k]);
                            std::swap(s2[l], s2[k]);
                        }
                        sum = 0;
                        for (i = 0; i < n; i++) {
                            sum += pow(Q(i, k), 2);
                        }
                        R(k, k) = sqrt(sum);
                    }

#pragma omp barrier
#pragma omp for schedule(static, 1)
                    for (i = 0; i < n; i++) {
                        Q(i, k) = Q(i, k) / R(k, k);
                    }
#pragma omp for schedule(static, 1)
                    for (j = 0; j < n; j++) {
                        if (j > k) {
                            R(k, j) = 0;
                            for (i = 0; i < n; i++) {
                                R(k, j) += Q(i, k) * Q(i, j);
                            }
                            s2[j] = s2[j] + pow(R(k, j), 2);
                            for (i = 0; i < n; i++) {
                                Q(i, j) -= R(k, j) * Q(i, k);
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
        returnType <scalar, index> pmgs(const index n_proc) {

            R.clear();
            Q.clear();
            Q.assign(B);

            auto lock = new omp_lock_t[n]();
            for (index i = 0; i < n; i++) {
                omp_init_lock((&lock[i]));
                omp_set_lock(&lock[i]);
            }
            index i, j, k;
#pragma omp parallel default(shared) num_threads(n_proc)
            {}
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            scalar t_qr = omp_get_wtime(), sum;
#pragma omp parallel default(shared) num_threads(n_proc) private(sum, i, j, k)
            {
                sum = 0;
                if (omp_get_thread_num() == 0) {
                    // Calculation of ||A||
                    for (i = 0; i < n; i++) {
                        sum += Q[i] * Q[i];
                    }
                    R[0] = sqrt(sum);
                    for (i = 0; i < n; i++) {
                        Q[i] = Q[i] / R[0];
                    }
                    omp_unset_lock(&lock[0]);
                }

                for (k = 1; k < n; k++) {
                    //Check if Q[][i-1] (the previous column) is computed.
                    omp_set_lock(&lock[k - 1]);
                    omp_unset_lock(&lock[k - 1]);
#pragma omp for schedule(static, 1) nowait
                    for (j = 0; j < n; j++) {
                        if (j >= k) {
                            R(k - 1, j) = 0;
                            for (i = 0; i < n; i++) {
                                R(k - 1, j) += Q(i, k - 1) * Q(i, j);
                            }
                            for (i = 0; i < n; i++) {
                                Q(i, j) -= R(k - 1, j) * Q(i, k - 1);
                            }

                            //Only one thread calculates the norm(A)
                            //and unsets the lock for the next column.
                            if (j == k) {
                                sum = 0;
                                for (i = 0; i < n; i++) {
                                    sum = sum + pow(Q(i, j), 2);
                                }
                                R(j, j) = sqrt(sum);
                                for (i = 0; i < n; i++) {
                                    Q(i, j) = Q(i, j) / R(j, j);
                                }
                                omp_unset_lock(&lock[k]);
                            }
                        }
                    }
                }
            }
            b_matrix Q_T = trans(Q);
            prod(Q_T, y, y_r);
            y.assign(y_r);
            t_qr = omp_get_wtime() - t_qr;

            scalar error = -1;
            if (eval || verbose) {
                error = qr_validation();
                cout << "[  QR ERROR OMP:]" << error << endl;
            }
            for (i = 0; i < n; i++) {
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
//        returnType <scalar, index> mgs_qr_col() {
//
//            b_matrix B_t(B);
//
//            R.resize(B.size2(), B.size2());
//            R.clear();
//            Q.resize(B.size1(), B.size2());
//            Q.clear();
//
//            scalar t_qr = omp_get_wtime();
//            for (index k = 0; k < n; k++) {
//                for (index j = 0; j < k; j++) {
//                    R(j, k) = iner_prod(column(Q, j), column(B_t, k));
//                    column(B_t, k) = column(B_t, k) - column(Q, j) * R(j, k);
//                }
//                R(k, k) = norm_2(column(B_t, k));
//                column(Q, k) = column(B_t, k) / R(k, k);
//            }
//            y = prod(trans(Q), y);
//
//            t_qr = omp_get_wtime() - t_qr;
//
//            return {{}, t_qr, 0};
//        }

        /**
         * @deprecated
         * @return
         */
//        returnType <scalar, index> aip() {
//            scalar alpha;
//            scalar t_aip = omp_get_wtime();
//
//            CILS_Reduction _reduction(B, y, lower, upper);
//            _reduction.mgs_qr_col();
//
//            b_vector y_0(_reduction.y);
//            b_matrix R_0(_reduction.R);
//
//            R_R.resize(B.size2(), B.size2());
//            R_R.assign(_reduction.R);
//            y_r.resize(_reduction.y.size());
//            y_r.assign(_reduction.y);
//
//            //Permutation vector
//            p.resize(B.size2());
//            p.clear();
//            for (index i = 0; i < n; i++) {
//                p[i] = i;
//            }
//            index _n = B.size2();
//
//            //Inverse transpose of R
//            helper::inv<scalar, index>(R_0, G);
//            G = trans(G);
//
//            scalar dist, dist_i;
//            index j = 0, x_j = 0;
//            for (index k = n - 1; k >= 1; k--) {
//                scalar maxDist = -1;
//                for (index i = 0; i <= k; i++) {
//                    //alpha = y(i:k)' * G(i:k,i);
//                    b_vector gi = subrange(column(G, i), i, k + 1);
//                    b_vector y_t = subrange(y_r, i, k + 1);
//                    alpha = iner_prod(y_t, gi);
//                    index x_i = max(min((int) round(alpha), upper), lower);
//                    if (alpha < lower || alpha > upper || alpha == x_i)
//                        dist = 1 + abs(alpha - x_i);
//                    else
//                        dist = 1 - abs(alpha - x_i);
//                    dist_i = dist / norm_2(gi);
//                    if (dist_i > maxDist) {
//                        maxDist = dist_i;
//                        j = i;
//                        x_j = x_i;
//                    }
//                }
//                //p(j:k) = p([j+1:k,j]);
//                scalar pj = p[j];
//                subrange(p, j, k) = subrange(p, j + 1, k + 1);
//                p[k] = pj;
//
//                //Update y, R and G for the new dimension-reduced problem
//                //y(1:k-1) = y(1:k-1) - R(1:k-1,j) * x_j;
//                auto y_k = subrange(y_r, 0, k);
//                y_k = y_k - subrange(column(R_R, j), 0, k) * x_j;
//                //R(:,j) = []
//                auto R_temp_2 = subrange(R_R, 0, n, j + 1, _n);
//                subrange(R_R, 0, n, j, _n - 1) = R_temp_2;
//                R_R.resize(m, _n - 1);
//                //G(:,j) = []
//                auto G_temp_2 = subrange(G, 0, n, j + 1, _n);
//                subrange(G, 0, n, j, _n - 1) = G_temp_2;
//                G.resize(m, _n - 1);
//                //Size decrease 1
//                _n--;
//                for (index t = j; t <= k - 1; t++) {
//                    index t1 = t + 1;
//                    //Bring R back to an upper triangular matrix by a Givens rotation
//                    scalar W[4] = {};
//                    scalar low_tmp[2] = {R_R(t, t), R_R(t1, t)};
//                    b_matrix W_m = helper::planerot<scalar, index>(low_tmp, W);
//                    R_R(t, t) = low_tmp[0];
//                    R_R(t1, t) = low_tmp[1];
//
//                    //Combined Rotation.
//                    auto R_G = subrange(R_R, t, t1 + 1, t1, k);
//                    R_G = prod(W_m, R_G);
//                    auto G_G = subrange(G, t, t1 + 1, 0, t1);
//                    G_G = prod(W_m, G_G);
//                    auto y_G = subrange(y_r, t, t1 + 1);
//                    y_G = prod(W_m, y_G);
//                }
//            }
//
//            // Reorder the columns of R0 according to p
//            //R0 = R0(:,p);
//            R_R.resize(n, n);
//            R_R.clear();
//            //The permutation matrix
//            P.resize(n, n);
//            P.assign(I);
//            for (index i = 0; i < n; i++) {
//                for (index j = 0; j < n; j++) {
//                    R_R(j, i) = R_0(j, p[i]);
//                    P(j, i) = I(j, p[i]);
//                }
//            }
//
//            // Compute the QR factorization of R0 and then transform y0
//            //[Q, R, y] = qrmgs(R0, y0);
//            CILS_Reduction reduction_(R_R, y_0, lower, upper);
//            reduction_.mgs_qr_col();
//            R = reduction_.R;
//            y = reduction_.y;
//            Q = reduction_.Q;
//
//            t_aip = omp_get_wtime() - t_aip;
//            return {{}, 0, t_aip};
//        }

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
        returnType <scalar, index> plll() {
            scalar zeta, alpha, t_qr, t_plll, sum = 0;

            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            CILS_Reduction<scalar, index> reduction(B, y, lower, upper);
            auto reT = reduction.mgs_qrp();
            R.assign(reduction.R);
            y.assign(reduction.y);
            Z.assign(reduction.P);
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
                            R(i, k) = R(i, k) - zeta * R(i, k1);
                        }
                        for (i = 0; i < n; i++) {
                            Z(i, k) -= zeta * Z(i, k1);
                        }
                        //Perform size reductions on R(1:k-2,k)
                        for (i = k - 2; i >= 0; i--) {
                            zeta = round(R(i, k) / R(i, i));
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
                        std::swap(R(i, k1), R(i, k));
                        std::swap(Z(i, k1), Z(i, k));
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
        returnType <scalar, index> aspl() {
            scalar zeta, alpha, t_qr, t_plll, sum = 0;
            //Clear Variables:
            y_r.assign(y);
            auto s = new int[n]();
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            CILS_Reduction<scalar, index> reduction(B, y, lower, upper);
            cils::returnType<scalar, index> reT = reduction.mgs_qrp();
            R.assign(reduction.R);
            y.assign(reduction.y);
            Z.assign(reduction.P);
            t_qr = reT.run_time;

            //  ------------------------------------------------------------------
            //  --------  Perform the all-swap partial LLL reduction -------------------
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
                            std::swap(R(i, k1), R(i, k));
                            std::swap(Z(i, k1), Z(i, k));
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
                        zeta = round(R(k1, k) / R(k1, k1));
                        alpha = R(k1, k) - zeta * R(k1, k1);

                        if (pow(R(k1, k1), 2) > (1 + 1e-10) * (pow(alpha, 2) + pow(R(k, k), 2))) {
                            f = true;
                            for (i = 0; i < n; i++) {
                                std::swap(R(i, k1), R(i, k));
                                std::swap(Z(i, k1), Z(i, k));
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
        returnType <scalar, index> aspl_p() {

            scalar zeta, alpha, s, t_qr, t_plll, sum = 0;
            Z.assign(I);

            t_qr = omp_get_wtime();
            CILS_Reduction<scalar, index> reduction(B, y, lower, upper);
            auto ret = reduction.mgs_qr();
            R = reduction.R;
            y = reduction.y;
            t_qr = ret.run_time;


            //  ------------------------------------------------------------------
            //  ---  Perform the all-swap partial LLL permutation reduction ------
            //  ------------------------------------------------------------------
            R.resize(m, n + 1);
            column(R, n) = y;

            index k = 1, k1, i, j, iter = 0;
            index f = true, swap[n] = {}, start = 1, even = true;
            t_plll = omp_get_wtime();
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
        returnType <scalar, index> paspl(index n_c) {
            scalar zeta, alpha, t_qr, t_plll;
            //Clear Variables:

            y_r.assign(y);
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------

            CILS_Reduction<scalar, index> reduction(B, y, lower, upper);
            cils::returnType<scalar, index> reT = reduction.pmgs_qrp(n_c);
            t_qr = reT.run_time;
            R.assign(reduction.R);

            //while (reT.info != 0)
            reT = reduction.mgs_qrp();
            scalar error = 0;
            for (index i = 0; i < m * n; i++) {
                error += fabs(R[i] - reduction.R[i]);
            }
            cout<<error;

            R.assign(reduction.R);
            y.assign(reduction.y);
            Z.assign(reduction.P);

//            init_R_A();
            //  ------------------------------------------------------------------
            cout << "--------  Perform the PASPL reduction -------------------" << endl;
            //  ------------------------------------------------------------------


            auto s = new int[n]();
            scalar G[4] = {};
//            auto R = R.data();
//            auto Z_d = Z.data();
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
                            zeta = std::round(R(b_k, k1) / R(b_k, b_k));
                            alpha = R(b_k, k1) - zeta * R(b_k, b_k);
                            if ((pow(R(b_k, b_k), 2) > 1.0000000001 * (alpha * alpha + pow(R(b_k + 1, k1), 2))) &&
                                (zeta != 0.0)) {
//                                    omp_set_lock(&lock[k]);
                                for (j = 0; j < k1; j++) {
                                    R(j, k1) -= zeta * R(j, b_k);//[j + n * (b_k - 2)];
                                }
                                for (j = 0; j < n; j++) {
                                    Z(j, k1) -= zeta * Z(j, b_k);
                                }
//                                for (int b_i{0}; b_i < b_k; b_i++) {
//                                    index b_n = (b_k - b_i) - 1;
//                                    zeta = std::round(R(b_n, k1) / R(b_n, b_n));
//                                    if (zeta != 0.0) {
//                                        for (j = 0; j <= b_n; j++) {
//                                            R(j, k1) -= zeta * R(j, b_n);
//                                        }
//                                        for (j = 0; j < n; j++) {
//                                            Z(j, k1) -= zeta * Z(j, b_n);
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
                        if (pow(R[k1 + k1 * n], 2) > (1 + 1.e-10) * (pow(R(k1, k), 2) + pow(R(k, k), 2))) {
                            f = true;
                            s[k] = 1;
                            for (i = 0; i < n; i++) {
                                std::swap(R(i, k1), R(i, k));
                                std::swap(Z(i, k1), Z(i, k));
                            }
                        }
                    }
#pragma omp barrier
#pragma omp for schedule(static)
                    for (k = start; k < n; k += 2) {
                        if (s[k]) {
                            s[k] = 0;
                            k1 = k - 1;
                            //Bring R back to an upper triangular matrix by a Givens rotation

                            scalar low_tmp[2] = {R[k1 + k1 * n], R[k + k1 * n]};
                            helper::planerot<scalar, index>(low_tmp, G);
                            R[k1 + k1 * n] = low_tmp[0];
                            R[k + k1 * n] = low_tmp[1];

                            //Combined Rotation.
                            for (i = k; i < n; i++) {
                                low_tmp[0] = R[k1 + i * n];
                                low_tmp[1] = R[k + i * n];
                                R[k1 + i * n] = G[0] * low_tmp[0] + G[2] * low_tmp[1];
                                R[k + i * n] = G[1] * low_tmp[0] + G[3] * low_tmp[1];
                            }

                            low_tmp[0] = y[k1];
                            low_tmp[1] = y[k];
                            y[k1] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                            y[k] = G[1] * low_tmp[0] + low_tmp[1] * G[3];
                        }
                    }
#pragma omp barrier
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

                            if (pow(R[k1 + k1 * n], 2) >
                                (1 + 1.e-10) * (pow(R[k1 + k * n], 2) + pow(R[k + k * n], 2))) {
                                f = true;
                                s[k] = 1;
                                for (i = 0; i < n; i++) {
                                    std::swap(R(i, k1), R(i, k));
                                    std::swap(Z(i, k1), Z(i, k));
                                }
                            }
                        }
#pragma omp barrier
#pragma omp for schedule(static)
                        for (k = start; k < n; k += 2) {
                            if (s[k]) {
                                s[k] = 0;
                                k1 = k - 1;
                                //Bring R back to an upper triangular matrix by a Givens rotation

                                scalar low_tmp[2] = {R[k1 + k1 * n], R[k + k1 * n]};
                                helper::planerot<scalar, index>(low_tmp, G);
                                R[k1 + k1 * n] = low_tmp[0];
                                R[k + k1 * n] = low_tmp[1];

                                //Combined Rotation.
                                for (i = k; i < n; i++) {
                                    low_tmp[0] = R[k1 + i * n];
                                    low_tmp[1] = R[k + i * n];
                                    R[k1 + i * n] = G[0] * low_tmp[0] + G[2] * low_tmp[1];
                                    R[k + i * n] = G[1] * low_tmp[0] + G[3] * low_tmp[1];
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

    };
}