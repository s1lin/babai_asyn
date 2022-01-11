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
 *   MERCA_tNTABILITY or FITNESS FOR this->A PARTICULAR PURPOSE.  See the
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

        static scalar rt_hypotd_snf(scalar u0, scalar u1) {
            scalar a;
            scalar y;
            a = std::abs(u0);
            y = std::abs(u1);
            if (a < y) {
                a /= y;
                y *= std::sqrt(a * a + 1.0);
            } else if (a > y) {
                y /= a;
                y = a * std::sqrt(y * y + 1.0);
            } else if (!std::isnan(y)) {
                y = a * 1.4142135623730951;
            }
            return y;
        }

        static scalar xnrm2(index size_n, const scalar *x, index ix0) {
            scalar y = 0.0;
            if (size_n >= 1) {
                if (size_n == 1) {
                    y = std::abs(x[ix0 - 1]);
                } else {
                    scalar scale = 3.3121686421112381E-170;
                    index kend = (ix0 + size_n) - 1;
                    for (index k = ix0; k <= kend; k++) {
                        scalar absxk;
                        absxk = std::abs(x[k - 1]);
                        if (absxk > scale) {
                            scalar t;
                            t = scale / absxk;
                            y = y * t * t + 1.0;
                            scale = absxk;
                        } else {
                            scalar t;
                            t = absxk / scale;
                            y += t * t;
                        }
                    }
                    y = scale * std::sqrt(y);
                }
            }
            return y;
        }

        void
        xgerc(index size_m, index size_n, scalar alpha1, index ix0, const scalar *y, scalar *B, index ia0, index lda) {
            if (!(alpha1 == 0.0)) {
                int jA;
                jA = ia0;
                for (int j{0}; j < size_n; j++) {
                    if (y[j] != 0.0) {
                        double temp;
                        int i;
                        temp = y[j] * alpha1;
                        i = size_m + jA;
                        for (int ijA{jA}; ijA < i; ijA++) {
                            B[ijA - 1] = B[ijA - 1] + B[((ix0 + ijA) - jA) - 1] * temp;
                        }
                    }
                    jA += lda;
                }
            }
        }

        void xgeqp3(index size_m, index size_n, scalar *B, sd_vector &tau, si_vector &jpvt) {
            std::vector<scalar> vn1, vn2, work;
            int i;
            int knt;
            int minmana;
            bool guard1{false};
            knt = size_m;
            minmana = size_n;
            if (knt < minmana) {
                minmana = knt;
            }
            tau.resize(minmana);
            for (i = 0; i < minmana; i++) {
                tau[i] = 0.0;
            }
            guard1 = false;
            if ((size_m == 0) || (size_n == 0)) {
                guard1 = true;
            } else {
                knt = size_m;
                minmana = size_n;
                if (knt < minmana) {
                    minmana = knt;
                }
                if (minmana < 1) {
                    guard1 = true;
                } else {
                    double smax;
                    int k;
                    int ma;
                    int minmn;
                    jpvt.resize(size_n);
                    minmana = size_n;
                    for (i = 0; i < minmana; i++) {
                        jpvt[i] = 0;
                    }
                    for (k = 0; k < size_n; k++) {
                        jpvt[k] = k + 1;
                    }
                    ma = size_m;
                    knt = size_m;
                    minmn = size_n;
                    if (knt < minmn) {
                        minmn = knt;
                    }
                    work.resize(size_n);
                    minmana = size_n;
                    for (i = 0; i < minmana; i++) {
                        work[i] = 0.0;
                    }
                    vn1.resize(size_n);
                    minmana = size_n;
                    for (i = 0; i < minmana; i++) {
                        vn1[i] = 0.0;
                    }
                    vn2.resize(size_n);
                    minmana = size_n;
                    for (i = 0; i < minmana; i++) {
                        vn2[i] = 0.0;
                    }
                    for (knt = 0; knt < size_n; knt++) {
                        smax = xnrm2(size_m, B, knt * ma + 1);
                        vn1[knt] = smax;
                        vn2[knt] = smax;
                    }
                    for (int b_i{0}; b_i < minmn; b_i++) {
                        double s;
                        double temp2;
                        int ii;
                        int ii_tmp;
                        int ip1;
                        int mmi;
                        int nmi;
                        int pvt;
                        ip1 = b_i + 2;
                        ii_tmp = b_i * ma;
                        ii = ii_tmp + b_i;
                        nmi = size_n - b_i;
                        mmi = size_m - b_i;
                        if (nmi < 1) {
                            minmana = -1;
                        } else {
                            minmana = 0;
                            if (nmi > 1) {
                                smax = std::abs(vn1[b_i]);
                                for (k = 2; k <= nmi; k++) {
                                    s = std::abs(vn1[(b_i + k) - 1]);
                                    if (s > smax) {
                                        minmana = k - 1;
                                        smax = s;
                                    }
                                }
                            }
                        }
                        pvt = b_i + minmana;
                        if (pvt + 1 != b_i + 1) {
                            minmana = pvt * ma;
                            for (k = 0; k < size_m; k++) {
                                knt = minmana + k;
                                smax = B[knt];
                                i = ii_tmp + k;
                                B[knt] = B[i];
                                B[i] = smax;
                            }
                            minmana = jpvt[pvt];
                            jpvt[pvt] = jpvt[b_i];
                            jpvt[b_i] = minmana;
                            vn1[pvt] = vn1[b_i];
                            vn2[pvt] = vn2[b_i];
                        }
                        if (b_i + 1 < size_m) {
                            temp2 = B[ii];
                            minmana = ii + 2;
                            tau[b_i] = 0.0;
                            if (mmi > 0) {
                                smax = xnrm2(mmi - 1, B, ii + 2);
                                if (smax != 0.0) {
                                    s = rt_hypotd_snf(B[ii], smax);
                                    if (B[ii] >= 0.0) {
                                        s = -s;
                                    }
                                    if (std::abs(s) < 1.0020841800044864E-292) {
                                        knt = -1;
                                        i = ii + mmi;
                                        do {
                                            knt++;
                                            for (k = minmana; k <= i; k++) {
                                                B[k - 1] = 9.9792015476736E+291 * B[k - 1];
                                            }
                                            s *= 9.9792015476736E+291;
                                            temp2 *= 9.9792015476736E+291;
                                        } while (!(std::abs(s) >= 1.0020841800044864E-292));
                                        s = rt_hypotd_snf(temp2, xnrm2(mmi - 1, B, ii + 2));
                                        if (temp2 >= 0.0) {
                                            s = -s;
                                        }
                                        tau[b_i] = (s - temp2) / s;
                                        smax = 1.0 / (temp2 - s);
                                        for (k = minmana; k <= i; k++) {
                                            B[k - 1] = smax * B[k - 1];
                                        }
                                        for (k = 0; k <= knt; k++) {
                                            s *= 1.0020841800044864E-292;
                                        }
                                        temp2 = s;
                                    } else {
                                        tau[b_i] = (s - B[ii]) / s;
                                        smax = 1.0 / (B[ii] - s);
                                        i = ii + mmi;
                                        for (k = minmana; k <= i; k++) {
                                            B[k - 1] = smax * B[k - 1];
                                        }
                                        temp2 = s;
                                    }
                                }
                            }
                            B[ii] = temp2;
                        } else {
                            tau[b_i] = 0.0;
                        }
                        if (b_i + 1 < size_n) {
                            int ia;
                            temp2 = B[ii];
                            B[ii] = 1.0;
                            ii_tmp = (ii + ma) + 1;
                            if (tau[b_i] != 0.0) {
                                bool exitg2;
                                pvt = mmi;
                                minmana = (ii + mmi) - 1;
                                while ((pvt > 0) && (B[minmana] == 0.0)) {
                                    pvt--;
                                    minmana--;
                                }
                                knt = nmi - 1;
                                exitg2 = false;
                                while ((!exitg2) && (knt > 0)) {
                                    int exitg1;
                                    minmana = ii_tmp + (knt - 1) * ma;
                                    ia = minmana;
                                    do {
                                        exitg1 = 0;
                                        if (ia <= (minmana + pvt) - 1) {
                                            if (B[ia - 1] != 0.0) {
                                                exitg1 = 1;
                                            } else {
                                                ia++;
                                            }
                                        } else {
                                            knt--;
                                            exitg1 = 2;
                                        }
                                    } while (exitg1 == 0);
                                    if (exitg1 == 1) {
                                        exitg2 = true;
                                    }
                                }
                            } else {
                                pvt = 0;
                                knt = 0;
                            }
                            if (pvt > 0) {
                                if (knt != 0) {
                                    for (k = 0; k < knt; k++) {
                                        work[k] = 0.0;
                                    }
                                    k = 0;
                                    i = ii_tmp + ma * (knt - 1);
                                    for (nmi = ii_tmp; ma < 0 ? nmi >= i : nmi <= i; nmi += ma) {
                                        smax = 0.0;
                                        minmana = (nmi + pvt) - 1;
                                        for (ia = nmi; ia <= minmana; ia++) {
                                            smax += B[ia - 1] * B[(ii + ia) - nmi];
                                        }
                                        work[k] = work[k] + smax;
                                        k++;
                                    }
                                }
                                xgerc(pvt, knt, -tau[b_i], ii + 1, work, B, ii_tmp, ma);
                            }
                            B[ii] = temp2;
                        }
                        for (knt = ip1; knt <= size_n; knt++) {
                            minmana = b_i + (knt - 1) * ma;
                            smax = vn1[knt - 1];
                            if (smax != 0.0) {
                                s = std::abs(B[minmana]) / smax;
                                s = 1.0 - s * s;
                                if (s < 0.0) {
                                    s = 0.0;
                                }
                                temp2 = smax / vn2[knt - 1];
                                temp2 = s * (temp2 * temp2);
                                if (temp2 <= 1.4901161193847656E-8) {
                                    if (b_i + 1 < size_m) {
                                        smax = xnrm2(mmi - 1, B, minmana + 2);
                                        vn1[knt - 1] = smax;
                                        vn2[knt - 1] = smax;
                                    } else {
                                        vn1[knt - 1] = 0.0;
                                        vn2[knt - 1] = 0.0;
                                    }
                                } else {
                                    vn1[knt - 1] = smax * std::sqrt(s);
                                }
                            }
                        }
                    }
                }
            }
            if (guard1) {
                jpvt.resize(size_n);
                minmana = size_n;
                for (i = 0; i < minmana; i++) {
                    jpvt[i] = 0;
                }
                for (knt = 0; knt < size_n; knt++) {
                    jpvt[knt] = knt + 1;
                }
            }
        }

        void xorgqr(index size_m, index size_n, index k, scalar *B, index lda, scalar *tau) {
            std::vector<scalar> work;
            if (size_n >= 1) {
                int b_i;
                int b_k;
                int c_i;
                int i;
                int ia;
                int itau;
                i = size_n - 1;
                for (b_i = k; b_i <= i; b_i++) {
                    ia = b_i * lda;
                    b_k = size_m - 1;
                    for (c_i = 0; c_i <= b_k; c_i++) {
                        B[ia + c_i] = 0.0;
                    }
                    B[ia + b_i] = 1.0;
                }
                itau = k - 1;
                b_i = size_n;
                work.resize(b_i);
                for (i = 0; i < b_i; i++) {
                    work[i] = 0.0;
                }
                for (c_i = k; c_i >= 1; c_i--) {
                    int iaii;
                    iaii = c_i + (c_i - 1) * lda;
                    if (c_i < size_n) {
                        int ic0;
                        int lastc;
                        int lastv;
                        B[iaii - 1] = 1.0;
                        ic0 = iaii + lda;
                        if (tau[itau] != 0.0) {
                            bool exitg2;
                            lastv = (size_m - c_i) + 1;
                            b_i = (iaii + size_m) - c_i;
                            while ((lastv > 0) && (B[b_i - 1] == 0.0)) {
                                lastv--;
                                b_i--;
                            }
                            lastc = size_n - c_i;
                            exitg2 = false;
                            while ((!exitg2) && (lastc > 0)) {
                                int exitg1;
                                b_i = ic0 + (lastc - 1) * lda;
                                ia = b_i;
                                do {
                                    exitg1 = 0;
                                    if (ia <= (b_i + lastv) - 1) {
                                        if (B[ia - 1] != 0.0) {
                                            exitg1 = 1;
                                        } else {
                                            ia++;
                                        }
                                    } else {
                                        lastc--;
                                        exitg1 = 2;
                                    }
                                } while (exitg1 == 0);
                                if (exitg1 == 1) {
                                    exitg2 = true;
                                }
                            }
                        } else {
                            lastv = 0;
                            lastc = 0;
                        }
                        if (lastv > 0) {
                            if (lastc != 0) {
                                for (b_i = 0; b_i < lastc; b_i++) {
                                    work[b_i] = 0.0;
                                }
                                b_i = 0;
                                i = ic0 + lda * (lastc - 1);
                                for (int iac{ic0}; lda < 0 ? iac >= i : iac <= i; iac += lda) {
                                    double c;
                                    c = 0.0;
                                    b_k = (iac + lastv) - 1;
                                    for (ia = iac; ia <= b_k; ia++) {
                                        c += B[ia - 1] * B[((iaii + ia) - iac) - 1];
                                    }
                                    work[b_i] = work[b_i] + c;
                                    b_i++;
                                }
                            }
                            xgerc(lastv, lastc, -tau[itau], iaii, work, B, ic0, lda);
                        }
                    }
                    if (c_i < size_m) {
                        b_i = iaii + 1;
                        i = (iaii + size_m) - c_i;
                        for (b_k = b_i; b_k <= i; b_k++) {
                            B[b_k - 1] = -tau[itau] * B[b_k - 1];
                        }
                    }
                    B[iaii - 1] = 1.0 - tau[itau];
                    for (b_i = 0; b_i <= c_i - 2; b_i++) {
                        B[(iaii - b_i) - 2] = 0.0;
                    }
                    itau--;
                }
            }
        }

        /**
         * Evaluating the LLL decomposition
         * @return
         */
        returnType <scalar, index> lll_validation() {
            printf("====================[ TEST | LLL_VALIDATE ]==================================\n");
            printf("[ INFO: in LLL validation]\n");
            scalar sum, error = 0, det;
            b_matrix A_T = prod(A_R, Z);
            b_matrix R_I;

            helper::inv<scalar, index>(R_R, R_I);

            b_matrix Q_Z = prod(A_T, R_I);
            b_vector y_R = prod(trans(Q_Z), y_a);
            if (verbose && n <= 32) {
                printf("\n[ Print R:]\n");
                helper::display_matrix<scalar, index>(R_R, "R_R");
                printf("\n[ Print Z:]\n");
                helper::display_matrix<scalar, index>(Z, "Z");
                printf("\n[ Print Q'Q:]\n");
                helper::display_matrix<scalar, index>(prod(trans(Q_Z), Q_Z), "Q_Z");
                printf("\n[ Print Q'y:]\n");
                helper::display_vector<scalar, index>(m, y_R, "y_R");
                printf("\n[ Print y_r:]\n");
                helper::display_vector<scalar, index>(m, y_r, "y_r");
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
         * @param A_R
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
                //b_vector A_T(m * n, 0);
                //helper::mtimes_v<scalar, index>(m, n, Q, R_Q, A_T);
                b_matrix A_T = prod(Q, R_Q);
                if (verbose && n <= 16) {
                    printf("\n[ Print Q:]\n");
                    helper::display_matrix<scalar, index>(Q, "Q");
                    printf("\n[ Print R:]\n");
                    helper::display_matrix<scalar, index>(R_Q, "R_Q");
                    printf("\n[ Print A_R:]\n");
                    helper::display_matrix<scalar, index>(A_R, "A_R");
                    printf("\n[ Print Q*R:]\n");
                    helper::display_matrix<scalar, index>(A_T, "Q*R");
                }

                for (i = 0; i < m * n; i++) {
                    error += fabs(A_T(i) - A_R(i));
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
        * @param A_R
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
                b_matrix A_T = prod(Q, R_Q);//helper::mtimes_v<scalar, index>(m, n, Q, R_Q, A_T);

                b_matrix A_P = prod(A_R, P);//helper::mtimes_AP<scalar, index>(m, n, A_R, P, A_P.data());

                if (verbose) {
                    printf("\n[ Print Q:]\n");
                    helper::display_matrix<scalar, index>(Q, "Q");
                    printf("\n[ Print R:]\n");
                    helper::display_matrix<scalar, index>(R_Q, "R_Q");
                    printf("\n[ Print P:]\n");
                    helper::display_matrix<scalar, index>(P, "P");
                    printf("\n[ Print A_R:]\n");
                    helper::display_matrix<scalar, index>(A_R, "A_R");
                    printf("\n[ Print A_P:]\n");
                    helper::display_matrix<scalar, index>(A_P, "A_R*P");
                    printf("\n[ Print Q*R:]\n");
                    helper::display_matrix<scalar, index>(A_T, "Q*R");
                }

                for (i = 0; i < m * n; i++) {
                    error += fabs(A_T(i) - A_P[i]);
                }
            }

            return error;
        }


        /**
         * Evaluating the QR decomposition for column orientation
         * @tparam scalar
         * @tparam index
         * @tparam n
         * @param A_R
         * @param Q
         * @param R
         * @param eval
         * @return
         */
        scalar qr_validation_col() {
            index i, j, k;
            scalar sum, error = 0;

            if (eval == 1) {
                b_matrix A_T = prod(Q, R_Q); //helper::mtimes_col<scalar, index>(m, n, Q, R_Q, A_T);

                if (verbose) {
                    printf("\n[ Print Q:]\n");
                    helper::display_matrix<scalar, index>(Q, "Q");
                    printf("\n[ Print R:]\n");
                    helper::display_matrix<scalar, index>(R_Q, "R_Q");
                    printf("\n[ Print A_R:]\n");
                    helper::display_matrix<scalar, index>(A_R, "A_R");
                    printf("\n[ Print Q*R:]\n");
                    helper::display_matrix<scalar, index>(A_T, "Q*R");
                }

                for (i = 0; i < m * n; i++) {
                    error += fabs(A_T(i) - A_R(i));
                }
            }

            return error;
        }

    public:
        //R_A: no zeros, R_R: LLL reduced, R_Q: QR
        b_matrix A_R, R_Q, R_R, Q, Z, P, I, G;
        b_vector y_a, y_q, y_r, p;
        CILS <scalar, index> cils;

        CILS_Reduction(CILS <scalar, index> &cils) {
            this->cils = cils;
            this->m = cils.m;
            this->n = cils.n;

            this->A_R.resize(m, n);
            this->R_R.resize(m, n);
            this->A_R.assign(cils.A);
            this->R_Q.resize(m, n, false);
            this->Q.resize(m, m, false);
            this->y_q.resize(m);
            this->y_r.resize(m);
            this->y_a.resize(m);
            this->y_a.assign(cils.y_a);
            this->p.resize(m);
            this->I.resize(n, n, false);
            this->Z.resize(n, n, false);
            this->P.resize(n, n, false);
            this->I.assign(cils.I);
            this->Z.assign(cils.I);
            this->P.assign(cils.I);
        }

        CILS_Reduction(index m, index n, b_matrix I) {
            this->m = m;
            this->n = n;
            this->A_R.resize(m, n, false);
            this->R_Q.resize(m, n, false);
            this->Q.resize(m, m, false);
            this->y_q.resize(m);
            this->p.resize(m);
            this->Z.resize(n, n, false);
            this->P.resize(n, n, false);
            this->I.assign(I);
            this->Z.assign(I);
            this->P.assign(I);
        }


        /**
         * Serial version of QR-factorization with column pivoting
         * Results are stored in the class object.
         * @param B : m-by-n input matrix
         */
        returnType <scalar, index>
        cils_eml_qr(const scalar *B) {
            std::vector<scalar> A_t, tau;
            std::vector<index> jpvt, jpvt1;
            index i, j;
            scalar error = -1, time, sum;

            Q.resize(m, m, false);
            R_Q.resize(m, n, false);
            Q.assign(m * m, 0);
            R_Q.resize(m * n, 0);

            index size_m = m - 1;
            index size_n = n - 1;
            if (m > n) {
                for (i = 0; i <= size_n; i++) {
                    for (j = 0; j <= size_m; j++) {
                        Q[j + m * i] = B[j + m * i];
                    }
                }
                for (i = n + 1; i <= size_m + 1; i++) {
                    for (j = 0; j <= size_m; j++) {
                        Q[j + m * (i - 1)] = 0.0;
                    }
                }
                xgeqp3(m, m, Q, tau, jpvt1);
                jpvt.resize(n);
                for (i = 0; i <= size_n; i++) {
                    jpvt[i] = jpvt1[i];
                    for (j = 0; j <= i; j++) {
                        R_Q[j + m * i] = Q[j + m * i];
                    }
                    for (j = i + 2; j <= size_m + 1; j++) {
                        R_Q[(j + m * i) - 1] = 0.0;
                    }
                }
                xorgqr(m, m, n, Q, m, tau.data());
            } else {
                A_t.resize(m, n, false);
                A_t.clear();;
                for (i = 0; i < m * n; i++) {
                    A_t(i) = B[i];
                }
                xgeqp3(m, n, A_t, tau, jpvt);
                for (i = 0; i <= size_m; i++) {
                    for (j = 0; j <= i; j++) {
                        R_Q[j + m * i] = A_t[j + m * i];
                    }
                    for (j = i + 2; j <= size_m + 1; j++) {
                        R_Q[(j + m * i) - 1] = 0.0;
                    }
                }

                for (i = m + 1; i <= size_n + 1; i++) {
                    for (j = 0; j <= size_m; j++) {
                        R_Q[j + m * (i - 1)] = A_t[j + m * (i - 1)];
                    }
                }

                xorgqr(m, m, m, A_t, m, tau.data());
                for (i = 0; i <= size_m; i++) {
                    for (j = 0; j <= size_m; j++) {
                        Q[j + m * i] = A_t[j + m * i];
                    }
                }
            }

            for (i = 0; i < m * n; i++) {
                A_R(i) = B[i];
            }

            //Permutation matrix
            P.resize(n, n, false);
            P.clear();;
            for (i = 0; i < n; i++) {
                P[(jpvt[i] + n * i) - 1] = 1;
            }

            if (eval) {
                error = qrp_validation();
                if (verbose)
                    cout << "[  QR ERROR SER:]" << error << endl;
            }

            return {{}, time, error};
        }

        /**
         * Serial version of FULL QR-factorization using modified Gram-Schmidt algorithm, row-oriented
         * Results are stored in the class object.
         */
        returnType <scalar, index> cils_mgs_qr() {
            //Initialize A_t = [A, y_a]:
            b_matrix A_t(A_R);
            A_t.resize(m, n + 1);
            column(A_t, n) = y_a;

            //Clear Variables:
            R_Q.resize(m, n + 1);
            this->Q.clear();
            this->R_Q.clear();
            this->y_q.clear();
            this->Z.assign(this->I);
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            scalar t_qr = omp_get_wtime();
            for (index j = 0; j < m; j++) {
                for (index k = j; k < n + 1; k++) {
                    R_Q(j, k) = inner_prod(column(Q, j), column(A_t, k));
                    column(A_t, k) = column(A_t, k) - column(Q, j) * R_Q(j, k);
                    if (k == j) {
                        R_Q(k, k) = norm_2(column(A_t, k));
                        column(Q, k) = column(A_t, k) / R_Q(k, k);
                    }
                }
            }
            t_qr = omp_get_wtime() - t_qr;
            y_q = column(R_Q, n);
            R_Q.resize(m, n);

            return {{}, t_qr, 0};
        }


        /**
         * Parallel version of FULL QR-factorization using modified Gram-Schmidt algorithm, row-oriented
         * Results are stored in the class object.
         */
        returnType <scalar, index> cils_mgs_qr_omp(const index n_proc) {
            scalar a_norm;
            b_matrix A_t(A_R);
            A_t.resize(m, n);

            //Clear Variables:
            this->Q.clear();
            this->R_Q.clear();
            this->y_q.clear();
            this->Z.assign(this->I);
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
                    // Calculation of ||A_R||
                    a_norm = norm_2(column(A_t, 0));
                    R_Q(0, 0) = a_norm;
                    column(Q, 0) = column(A_t, 0) / a_norm;
                    omp_unset_lock(&lock[0]);
                }

                for (index j = 1; j < m; j++) {
                    omp_set_lock(&lock[j - 1]);
                    omp_unset_lock(&lock[j - 1]);
#pragma omp for schedule(static, 1)
                    for (index k = 0; k < n; k++) {
                        if (k >= j) {
                            R_Q(j - 1, k) = 0;
                            R_Q(j - 1, k) = inner_prod(column(Q, j - 1), column(A_t, k));
                            column(A_t, k) -= column(Q, j - 1) * R_Q(j - 1, k);
                            if (k == j) {
                                a_norm = norm_2(column(A_t, k));
                                R_Q(k, k) = a_norm;
                                column(Q, k) = column(A_t, k) / a_norm;
                                omp_unset_lock(&lock[j]);
                            }
                        }
                    }
                }
            }
            y_q = prod(trans(Q), y_a);
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
         * R_Q is n by n, y is transformed from m by 1 to n by 1
         * @param B : m-by-n input matrix
         * @param y : m-by-1 input right hand vector
         */
        returnType <scalar, index>
        cils_mgs_qr_col(b_matrix &A, b_vector &y) {

            b_matrix A_t(A);

            this->R_Q.resize(A.size2(), A.size2());
            this->R_Q.clear();
            this->Q.resize(A.size1(), A.size2());
            this->Q.clear();
            this->y_q.resize(y.size());
            this->y_q.clear();

            scalar t_qr = omp_get_wtime();
            for (index k = 0; k < n; k++) {
                for (index j = 0; j < k; j++) {
                    R_Q(j, k) = inner_prod(column(Q, j), column(A_t, k));
                    column(A_t, k) = column(A_t, k) - column(Q, j) * R_Q(j, k);
                }
                R_Q(k, k) = norm_2(column(A_t, k));
                column(Q, k) = column(A_t, k) / R_Q(k, k);
            }
            y_q = prod(trans(Q), y);

            t_qr = omp_get_wtime() - t_qr;

            return {{}, t_qr, 0};
        }

        returnType <scalar, index> cils_aip_reduction() {
            scalar alpha;
            scalar t_aip = omp_get_wtime();

            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Column------------------
            //  ------------------------------------------------------------------
            cils_mgs_qr_col(A_R, y_a);

            b_vector y_0(y_q);
            b_matrix R_0(R_Q);

            R_R.resize(A_R.size2(), A_R.size2());
            R_R.assign(R_Q);
            y_r.resize(y_q.size());
            y_r.assign(y_q);

            //Permutation vector
            p.resize(A_R.size2());
            p.clear();
            for (index i = 0; i < n; i++) {
                p[i] = i;
            }
            index R_n = A_R.size2();

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
                    index x_i = max(min((int) round(alpha), cils.upper), cils.lower);
                    if (alpha < cils.lower || alpha > cils.upper || alpha == x_i)
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
                auto R_temp_2 = subrange(R_R, 0, n, j + 1, R_n);
                subrange(R_R, 0, n, j, R_n - 1) = R_temp_2;
                R_R.resize(m, R_n - 1);
                //G(:,j) = []
                auto G_temp_2 = subrange(G, 0, n, j + 1, R_n);
                subrange(G, 0, n, j, R_n - 1) = G_temp_2;
                G.resize(m, R_n - 1);
                //Size decrease 1
                R_n--;
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
            P.assign(this->I);
            for (index i = 0; i < n; i++) {
                for (index i1 = 0; i1 < n; i1++) {
                    R_R(i1, i) = R_0(i1, p[i]);
                    P(i1, i) = this->I(i1, p[i]);
                }
            }

            // Compute the QR factorization of R0 and then transform y0
            //[Q, R, y] = qrmgs(R0, y0);
            cils_mgs_qr_col(R_R, y_0);

            t_aip = omp_get_wtime() - t_aip;
            return {{}, 0, t_aip};
        }

        /**
         * Test caller for reduction method.
         * @param n_proc
         * @return
         */
        returnType <scalar, index>
        cils_plll_reduction_tester(const index n_proc) {
            scalar time = 0, det = 0;
            returnType<scalar, index> reT, lll_val;
            eval = true;
            verbose = true;

            cout << "[ In cils_plll_reduction_serial]\n";
            reT = cils_plll_reduction_serial();
            printf("[ INFO: QR TIME: %8.5f, PLLL TIME: %8.5f]\n",
                   reT.run_time, reT.info);
            time = reT.info;
            lll_val = lll_validation();

            cout << "[ In cils_aspl_reduction_serial]\n";
            reT = cils_aspl_reduction_serial();
            printf("[ INFO: QR TIME: %8.5f, PLLL TIME: %8.5f]\n",
                   reT.run_time, reT.info);
            scalar t_qr = reT.run_time;
            scalar t_plll = reT.info;
            lll_val = lll_validation();

            for (index i = 2; i <= 4; i += 2) {
                cout << "[ In cils_aspl_reduction_omp]\n";
                reT = cils_aspl_reduction_omp(i);
                printf("[ INFO: QR TIME: %8.5f, PLLL TIME: %8.5f, QR SPU: %8.5f, LLL SPU:%8.2f]\n",
                       reT.run_time, reT.info, t_qr / reT.run_time, t_plll / reT.info);
                lll_val = lll_validation();
            }


            return {{reT.info, lll_val.info}, time, lll_val.run_time};
        }

        /**
         * Original PLLL algorithm
         * Description:
         * [R,Z,y] = sils_reduction(B,y) reduces the general standard integer
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
         *  See sils_reduction.m
         *  @return returnType: ~, time_qr, time_plll
         */
        returnType <scalar, index> cils_plll_reduction_serial() {
            scalar zeta, alpha, s, t_qr, t_plll, sum = 0;

            //Initialize A_t = [A, y_a]:
            b_matrix A_t(A_R);
            A_t.resize(m, n + 1);
            column(A_t, n) = y_a;

            //Clear Variables:
            R_R.resize(m, n + 1);
            this->Q.clear();
            this->R_R.clear();
            this->y_r.clear();
            this->Z.assign(this->I);
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            t_qr = omp_get_wtime();
            for (index j = 0; j < m; j++) {
                for (index k = j; k < n + 1; k++) {
                    R_R(j, k) = inner_prod(column(Q, j), column(A_t, k));
                    column(A_t, k) = column(A_t, k) - column(Q, j) * R_R(j, k);
                    if (k == j) {
                        R_R(k, k) = norm_2(column(A_t, k));
                        column(Q, k) = column(A_t, k) / R_R(k, k);
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

            y_r = column(R_R, n);
            R_R.resize(m, n);

            t_plll = omp_get_wtime() - t_plll;
            return {{}, t_qr, t_plll};
        }


        /**
         * All-Swap PLLL algorithm
         * Description:
         * [R,Z,y] = sils_reduction(B,y) reduces the general standard integer
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
        returnType <scalar, index> cils_aspl_reduction_serial() {
            scalar zeta, alpha, s, t_qr, t_plll, sum = 0;

            //Initialize A_t = [A, y_a]:
            b_matrix A_t(A_R);
            A_t.resize(m, n + 1);
            column(A_t, n) = y_a;

            //Clear Variables:
            R_R.resize(m, n + 1);
            this->Q.clear();
            this->R_R.clear();
            this->y_r.clear();
            this->Z.assign(this->I);
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            t_qr = omp_get_wtime();
            for (index j = 0; j < m; j++) {
                for (index k = j; k < n + 1; k++) {
                    R_R(j, k) = inner_prod(column(Q, j), column(A_t, k));
                    column(A_t, k) = column(A_t, k) - column(Q, j) * R_R(j, k);
                    if (k == j) {
                        R_R(k, k) = norm_2(column(A_t, k));
                        column(Q, k) = column(A_t, k) / R_R(k, k);
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
         * [R,P,y] = sils_reduction(B,y) reduces the general standard integer
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
        returnType <scalar, index> cils_aspl_permutation_serial() {
            scalar zeta, alpha, s, t_qr, t_plll, sum = 0;

            //Initialize A_t = [A, y_a]:
            b_matrix A_t(A_R);
            A_t.resize(m, n + 1);
            column(A_t, n) = y_a;

            //Clear Variables:
            R_R.resize(m, n + 1);
            this->Q.clear();
            this->R_R.clear();
            this->y_r.clear();
            this->Z.assign(this->I);
            //  ------------------------------------------------------------------
            //  --------  Perform the QR factorization: MGS Row-------------------
            //  ------------------------------------------------------------------
            t_qr = omp_get_wtime();
            for (index j = 0; j < m; j++) {
                for (index k = j; k < n + 1; k++) {
                    R_R(j, k) = inner_prod(column(Q, j), column(A_t, k));
                    column(A_t, k) = column(A_t, k) - column(Q, j) * R_R(j, k);
                    if (k == j) {
                        R_R(k, k) = norm_2(column(A_t, k));
                        column(Q, k) = column(A_t, k) / R_R(k, k);
                    }
                }
            }
            t_qr = omp_get_wtime() - t_qr;

            //  ------------------------------------------------------------------
            //  --------  Perform the all-swap partial LLL reduction -------------------
            //  ------------------------------------------------------------------

            index k = 1, k1, i, i1;
            index f = true, swap[n] = {}, start = 1, even = true;
//            std::vector<b_matrix> G_v(n);
            t_plll = omp_get_wtime();
            while (f || !even) {
                f = false;
                for (k = start; k < n; k += 2) {
                    k1 = k - 1;
                    zeta = round(R_R(k1, k) / R_R(k1, k1));
                    alpha = R_R(k1, k) - zeta * R_R(k1, k1);

                    if (pow(R_R(k1, k1), 2) > (1 + 1.e-10) * (pow(alpha, 2) + pow(R_R(k, k), 2))) {
                        swap[k] = 1;
                        f = true;
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
//                        G_v[k] = G_m;
                        auto R_G = subrange(R_R, k1, k + 1, k, n + 1);
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
            }
            y_r = column(R_R, n);
            R_R.resize(m, n);

            t_plll = omp_get_wtime() - t_plll;
            return {{}, t_qr, t_plll};
        }

        /**
         * All-Swap PLLL algorithm - parallel
         * Description:
         * [R,Z,y] = sils_reduction(B,y) reduces the general standard integer
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
        returnType <scalar, index> cils_aspl_reduction_omp(index n_proc) {
            scalar zeta, alpha, s, t_qr, t_plll, sum = 0;
            scalar a_norm;
            //Initialize A_t = [A, y_a]:
            b_matrix A_t(A_R);
            A_t.resize(m, n);

            //Clear Variables:
            this->Q.clear();
            this->R_R.clear();
            this->y_r.clear();
            this->Z.assign(this->I);
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
                    // Calculation of ||A_R||
                    a_norm = norm_2(column(A_t, 0));
                    R_R(0, 0) = a_norm;
                    column(Q, 0) = column(A_t, 0) / a_norm;
                    omp_unset_lock(&lock[0]);
                }

                for (index j = 1; j < m; j++) {
                    omp_set_lock(&lock[j - 1]);
                    omp_unset_lock(&lock[j - 1]);
#pragma omp for schedule(static, 1)
                    for (index k = 0; k < n; k++) {
                        if (k >= j) {
                            R_R(j - 1, k) = 0;
                            R_R(j - 1, k) = inner_prod(column(Q, j - 1), column(A_t, k));
                            column(A_t, k) -= column(Q, j - 1) * R_R(j - 1, k);
                            if (k == j) {
                                a_norm = norm_2(column(A_t, k));
                                R_R(k, k) = a_norm;
                                column(Q, k) = column(A_t, k) / a_norm;
                                omp_unset_lock(&lock[j]);
                            }
                        }
                    }
                }
            }

            t_qr = omp_get_wtime() - t_qr;
            R_Q.assign(R_R);
            qr_validation();
            y_r = prod(trans(Q), y_a);
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