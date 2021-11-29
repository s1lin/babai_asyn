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
            b_matrix R_I(m, n);

            typedef permutation_matrix<std::size_t> pmatrix;
            // create a working copy of the input
            b_matrix M(R_R);
            // create a permutation matrix for the LU-factorization
            pmatrix pm(M.size1());
            // perform LU-factorization
            auto res = lu_factorize(M, pm);

            // create identity matrix of "inverse"
            R_I.assign(identity_matrix<typename b_matrix::value_type>(M.size1()));
            // backsubstitute to get the inverse
            lu_substitute(M, pm, R_I);

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
            cout << y_R << endl;
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
            scalar error = -1;
            if (eval) {
                error = qr_validation();
                if (verbose)
                    cout << "[  QR ERROR SER:]" << error << endl;
            }
            return {{}, t_qr, error};
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
         * @param B : m-by-n input matrix
         * @param y : m-by-1 input right hand vector
         */
        returnType <scalar, index>
        cils_qr_serial_col(const scalar *B, const scalar *y) {
            if (verbose)
                cout << "[ In Serial QR]\n";

            index i, j, k;
            scalar error = -1, time, sum;

            b_matrix A_t(m, n);
            for (i = 0; i < m * n; i++) {
                A_t(i) = B[i];
                A_R(i) = B[i];
            }

            //Clear Variables:
            this->R_Q.resize(n, n, false);
            this->R_Q.clear();;
            this->Q.resize(m, n, false);
            this->Q.clear();;
            this->y_q.resize(n);
            this->y_q.clear();
            //start
            time = omp_get_wtime();
            for (k = 0; k < n; k++) {
                double b_Q;
                int i1;
                for (j = 0; j < k; j++) {
                    b_Q = 0.0;
                    for (i1 = 0; i1 < m; i1++) {
                        b_Q += Q[i1 + m * j] * A_t[i1 + m * k];
                    }
                    R_Q[j + n * k] = b_Q;
                    for (i1 = 0; i1 < m; i1++) {
                        A_t[i1 + m * k] = A_t[i1 + m * k] - Q[i1 + m * j] * b_Q;
                    }
                }
                if (m == 0) {
                    R_Q[k + n * k] = 0.0;
                } else {
                    b_Q = 0.0;
                    if (m == 1) {
                        R_Q[k + n * k] = std::abs(A_t[m * k]);
                    } else {
                        double scale;
                        scale = 3.3121686421112381E-170;
                        i1 = m - 1;
                        for (index b_A = 0; b_A <= i1; b_A++) {
                            double absxk;
                            absxk = std::abs(A_t[b_A + m * k]);
                            if (absxk > scale) {
                                double t;
                                t = scale / absxk;
                                b_Q = b_Q * t * t + 1.0;
                                scale = absxk;
                            } else {
                                double t;
                                t = absxk / scale;
                                b_Q += t * t;
                            }
                        }
                        R_Q[k + n * k] = scale * std::sqrt(b_Q);
                    }
                }
                b_Q = R_Q[k + n * k];
                for (i1 = 0; i1 < m; i1++) {
                    Q[i1 + m * k] = A_t[i1 + m * k] / b_Q;
                }
            }
            //y = Q' * y
            for (k = 0; k < m; k++) {
                for (i = 0; i < n; i++) {
                    y_q[i] = y_q[i] + Q[i * m + k] * y[k];
                }
            }

            time = omp_get_wtime() - time;

            if (eval) {
                error = qr_validation_col();
                cout << "[  QR ERROR SER:]" << error << endl;
            }

            return {{}, time, error};
        }


        returnType <scalar, index>
        cils_aip_reduction(const b_matrix &B, const b_vector &y) {
            b_vector b_y, c_y, b, C;

            cils_qr_serial_col(B, y.data());
            R_R.resize(n, n, false);
            R_R.assign(R_Q);
            y_r.resize(n);
            y_r.assign(y_q);

            //  Permutation vector
            p.resize(n);
            for (index i = 0; i < n; i++) {
                p[i] = i + 1;
            }

            //  Inverse transpose of R
            Q.resize(n, n, false);
            helper::inv<scalar, index>(n, n, R_R, Q);
            G.resize(n, n, false);
            for (index i = 0; i < n; i++) {
                for (index i1 = 0; i1 < n; i1++) {
                    G[i1 + n * i] = Q[i + n * i1];
                }
            }
            if (verbose) {
                helper::display_matrix<scalar, index>(G, "G");
                helper::display_matrix<scalar, index>(R_R, "R0");
            }
            index x_j = 0, j = 0, i1, ncols, loop_ub;
            index i = n - 1;
            for (int k = 0; k < i; k++) {
                double absxk, dist, dist_i, maxDist, scale, t, x_i;
                int b_i, b_k, i2, kend;
                b_k = n - k;
                maxDist = -1.0;
                //  Determine the k-th column
                for (b_i = 0; b_i < b_k; b_i++) {
                    if (b_i + 1 > b_k) {
                        i1 = 0;
                        i2 = 0;
                        ncols = 0;
                    } else {
                        i1 = b_i;
                        i2 = b_k;
                        ncols = b_i;
                    }
                    dist_i = 0.0;
                    loop_ub = i2 - i1;
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        dist_i += y_q[i1 + i2] * G[(ncols + i2) + n * b_i];
                    }
                    x_i = std::fmax(std::fmin(std::round(dist_i), upper), lower);
                    if ((dist_i < lower) || (dist_i > upper) || (dist_i == x_i)) {
                        dist = std::abs(dist_i - x_i) + 1.0;
                    } else {
                        dist = 1.0 - std::abs(dist_i - x_i);
                    }
                    if (b_i + 1 > b_k) {
                        i1 = -1;
                        i2 = -1;
                    } else {
                        i1 = b_i - 1;
                        i2 = b_k - 1;
                    }
                    i2 -= i1;
                    if (i2 == 0) {
                        dist_i = 0.0;
                    } else {
                        dist_i = 0.0;
                        if (i2 == 1) {
                            dist_i = std::abs(G[(i1 + n * b_i) + 1]);
                        } else {
                            scale = 3.3121686421112381E-170;
                            kend = i2 - 1;
                            for (ncols = 0; ncols <= kend; ncols++) {
                                absxk = std::abs(G[((i1 + ncols) + n * b_i) + 1]);
                                if (absxk > scale) {
                                    t = scale / absxk;
                                    dist_i = dist_i * t * t + 1.0;
                                    scale = absxk;
                                } else {
                                    t = absxk / scale;
                                    dist_i += t * t;
                                }
                            }
                            dist_i = scale * std::sqrt(dist_i);
                        }
                    }
                    dist_i = dist / dist_i;
                    if (dist_i > maxDist) {
                        maxDist = dist_i;
                        j = b_i + 1;
                        x_j = x_i;
                    }
                }
                //  Perform permutations
                if (j > b_k) {
                    i1 = 1;
                } else {
                    i1 = j;
                }
                if (b_k < static_cast<double>(j) + 1.0) {
                    b_y.resize(0);
                } else {
                    i2 = b_k - j;
                    b_y.resize(1 * i2);
                    loop_ub = i2 - 1;
                    for (i2 = 0; i2 <= loop_ub; i2++) {
                        b_y[i2] = (static_cast<unsigned int>(j) + i2) + 1U;
                    }
                }
                c_y.resize(1 * (b_y.size() + 1));
                loop_ub = b_y.size();
                for (i2 = 0; i2 < loop_ub; i2++) {
                    c_y[i2] = static_cast<int>(b_y[i2]) - 1;
                }
                c_y[b_y.size()] = j - 1;
                b_y.resize(1 * c_y.size());
                loop_ub = c_y.size();
                for (i2 = 0; i2 < loop_ub; i2++) {
                    b_y[i2] = static_cast<unsigned int>(p[c_y[i2]]);
                }
                loop_ub = b_y.size();
                for (i2 = 0; i2 < loop_ub; i2++) {
                    p[(i1 + i2) - 1] = b_y[i2];
                }

                //  Update y, R and G for the new dimension-reduced problem
                if (1 > b_k - 1) {
                    loop_ub = 0;
                } else {
                    loop_ub = b_k - 1;
                }
                b_y.resize(1 * loop_ub);
                for (i1 = 0; i1 < loop_ub; i1++) {
                    b_y[i1] = y_q[i1] - R_Q[i1 + n * (j - 1)] * x_j;
                }
                loop_ub = b_y.size();
                for (i1 = 0; i1 < loop_ub; i1++) {
                    y_q[i1] = b_y[i1];
                }
                kend = n;
                i1 = n;
                ncols = n - 1;
                for (loop_ub = j; loop_ub <= ncols; loop_ub++) {
                    for (b_i = 0; b_i < kend; b_i++) {
                        R_Q[b_i + n * (loop_ub - 1)] = R_Q[b_i + n * loop_ub];
                    }
                }
                if (1 > i1 - 1) {
                    loop_ub = -1;
                } else {
                    loop_ub = i1 - 2;
                }
                kend = n - 1;
                ncols = n;
                for (i1 = 0; i1 <= loop_ub; i1++) {
                    for (i2 = 0; i2 < ncols; i2++) {
                        R_Q[i2 + (kend + 1) * i1] = R_Q[i2 + n * i1];
                    }
                }
                R_Q.resize((kend + 1) * (loop_ub + 1));
                kend = i1 = n;
                ncols = n - 1;
                for (loop_ub = j; loop_ub <= ncols; loop_ub++) {
                    for (b_i = 0; b_i < kend; b_i++) {
                        G[b_i + n * (loop_ub - 1)] = G[b_i + n * loop_ub];
                    }
                }
                if (1 > i1 - 1) {
                    loop_ub = -1;
                } else {
                    loop_ub = i1 - 2;
                }
                kend = n - 1;
                ncols = n;
                for (i1 = 0; i1 <= loop_ub; i1++) {
                    for (i2 = 0; i2 < ncols; i2++) {
                        G[i2 + (kend + 1) * i1] = G[i2 + n * i1];
                    }
                }
                G.resize((kend + 1) * (loop_ub + 1));
                i1 = b_k - j;
                for (int b_t{0}; b_t < i1; b_t++) {
                    double unnamed_idx_1;
                    unsigned int c_t;
                    int i3;
                    c_t = static_cast<unsigned int>(j) + b_t;
                    //  Triangularize R and G by Givens rotation
                    b_i = static_cast<int>(c_t) - 1;
                    dist =
                            R_Q[(static_cast<int>(c_t) + n * (static_cast<int>(c_t) - 1)) -
                                1];
                    unnamed_idx_1 =
                            R_Q[static_cast<int>(c_t) + n * (static_cast<int>(c_t) - 1)];
                    dist_i =
                            R_Q[static_cast<int>(c_t) + n * (static_cast<int>(c_t) - 1)];
                    if (dist_i != 0.0) {
                        double r;
                        scale = 3.3121686421112381E-170;
                        absxk = std::abs(R_Q[(static_cast<int>(c_t) +
                                              n * (static_cast<int>(c_t) - 1)) -
                                             1]);
                        if (absxk > 3.3121686421112381E-170) {
                            r = 1.0;
                            scale = absxk;
                        } else {
                            t = absxk / 3.3121686421112381E-170;
                            r = t * t;
                        }
                        absxk = std::abs(
                                R_Q[static_cast<int>(c_t) + n * (static_cast<int>(c_t) - 1)]);
                        if (absxk > scale) {
                            t = scale / absxk;
                            r = r * t * t + 1.0;
                            scale = absxk;
                        } else {
                            t = absxk / scale;
                            r += t * t;
                        }
                        r = scale * std::sqrt(r);
                        absxk = dist / r;
                        scale = unnamed_idx_1 / r;
                        maxDist = -dist_i / r;
                        x_i = R_Q[(static_cast<int>(c_t) +
                                   n * (static_cast<int>(c_t) - 1)) -
                                  1] /
                              r;
                        dist = r;
                        unnamed_idx_1 = 0.0;
                    } else {
                        maxDist = 0.0;
                        scale = 0.0;
                        absxk = 1.0;
                        x_i = 1.0;
                    }
                    R_Q[(static_cast<int>(c_t) + n * (static_cast<int>(c_t) - 1)) - 1] =
                            dist;
                    R_Q[static_cast<int>(c_t) + n * (static_cast<int>(c_t) - 1)] =
                            unnamed_idx_1;
                    if (static_cast<int>(c_t) + 1 > b_k - 1) {
                        i2 = 0;
                        ncols = 0;
                        i3 = 0;
                    } else {
                        i2 = static_cast<int>(c_t);
                        ncols = b_k - 1;
                        i3 = static_cast<int>(c_t);
                    }
                    loop_ub = ncols - i2;
                    b.resize(2 * loop_ub);
                    for (ncols = 0; ncols < loop_ub; ncols++) {
                        kend = i2 + ncols;
                        b[2 * ncols] = R_Q[(static_cast<int>(c_t) + n * kend) - 1];
                        b[2 * ncols + 1] = R_Q[static_cast<int>(c_t) + n * kend];
                    }
                    ncols = loop_ub - 1;
                    C.resize(2 * loop_ub);
                    for (loop_ub = 0; loop_ub <= ncols; loop_ub++) {
                        kend = loop_ub << 1;
                        dist_i = b[kend + 1];
                        C[kend] = absxk * b[kend] + scale * dist_i;
                        C[kend + 1] = maxDist * b[kend] + x_i * dist_i;
                    }
                    loop_ub = C.size() / 2;
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        kend = i3 + i2;
                        R_Q[(static_cast<int>(c_t) + n * kend) - 1] = C[2 * i2];
                        R_Q[static_cast<int>(c_t) + n * kend] = C[2 * i2 + 1];
                    }
                    loop_ub = static_cast<int>(c_t);
                    b.resize(2 * static_cast<int>(c_t));
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        b[2 * i2] = G[(static_cast<int>(c_t) + n * i2) - 1];
                        b[2 * i2 + 1] = G[static_cast<int>(c_t) + n * i2];
                    }
                    C.resize(2 * static_cast<int>(c_t));
                    for (loop_ub = 0; loop_ub <= b_i; loop_ub++) {
                        kend = loop_ub << 1;
                        dist_i = b[kend + 1];
                        C[kend] = absxk * b[kend] + scale * dist_i;
                        C[kend + 1] = maxDist * b[kend] + x_i * dist_i;
                    }
                    loop_ub = C.size() / 2;
                    for (i2 = 0; i2 < loop_ub; i2++) {
                        G[(static_cast<int>(c_t) + n * i2) - 1] = C[2 * i2];
                        G[static_cast<int>(c_t) + n * i2] = C[2 * i2 + 1];
                    }
                    //  Apply the Givens rotation W to y
                    dist_i = y_q[static_cast<int>(c_t) - 1];
                    dist = y_q[static_cast<int>(static_cast<double>(c_t) + 1.0) - 1];
                    y_q[static_cast<int>(c_t) - 1] = absxk * dist_i + scale * dist;
                    y_q[static_cast<int>(static_cast<double>(c_t) + 1.0) - 1] =
                            maxDist * dist_i + x_i * dist;
                }
            }

            G.resize(n, n, false);
            G.clear();;
            for (i = 0; i < n; i++) {
                for (i1 = 0; i1 < n; i1++) {
                    G[i1 + n * i] = R_R(i1 + n * (p[i] - 1));
                }
            }
            if (verbose) {
                helper::display_matrix<scalar, index>(G, "R0");
                helper::display_matrix<scalar, index>(R_R, "R_R");
                helper::display_matrix<scalar, index>(1, n, p, "p");
            }

            //  Compute the QR factorization of R0 and then transform y0
            index mm = m;
            this->m = n;
            cils_qr_serial_col(G, y_r.data());
            this->m = mm;
            return {{}, 0, 0};
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

//            cout << "[ In cils_plll_reduction_serial]\n";
//            reT = cils_plll_reduction_serial();
//            printf("[ INFO: QR TIME: %8.5f, PLLL TIME: %8.5f]\n",
//                   reT.run_time, reT.info);
//            time = reT.info;
//            lll_val = lll_validation();
//            if (lll_val.info != 1) {
//                cerr << "0: LLL Failed, index:";
//                for (index i = 0; i < n; i++) {
//                    if (lll_val.x[i] != 0)
//                        cerr << i << ",";
//                }
//                cout << endl;
//            }
            cout << "[ In cils_eo_plll_reduction_omp]\n";
            reT = cils_eo_plll_reduction_omp(16);
            printf("[ INFO: QR TIME: %8.5f, PLLL TIME: %8.5f]\n",
                   reT.run_time, reT.info);
            time = reT.info;


            lll_val = lll_validation();
            if (lll_val.info != 1) {
                cerr << "0: LLL Failed, index:";
                for (index i = 0; i < n; i++) {
                    if (lll_val.x[i] != 0)
                        cerr << i << ",";
                }
                cout << endl;
            }

            cout << "[ In cils_eo_plll_reduction_serial]\n";
            reT = cils_eo_plll_reduction_serial();
            printf("[ INFO: QR TIME: %8.5f, PLLL TIME: %8.5f]\n",
                   reT.run_time, reT.info);
            time = reT.info;
            lll_val = lll_validation();
            if (lll_val.info != 1) {
                cerr << "0: LLL Failed, index:";
                for (index i = 0; i < n; i++) {
                    if (lll_val.x[i] != 0)
                        cerr << i << ",";
                }
                cout << endl;
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
            if (verbose)
                cout << y_r;
            t_plll = omp_get_wtime() - t_plll;
            return {{}, t_qr, t_plll};
        }


        /**
         * Even-Odd PLLL algorithm
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
        returnType <scalar, index> cils_eo_plll_reduction_serial() {
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
            //  --------  Perform the eo partial LLL reduction -------------------
            //  ------------------------------------------------------------------

            index k = 1, k1, i, i1;
            index f = true, swap[n] = {}, start = 1, even = true;
            std::vector<b_matrix> G_v(n);
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
                            for (i = k - 3; i >= 0; i--) {
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
            if (verbose)
                cout << y_r;
            t_plll = omp_get_wtime() - t_plll;
            return {{}, t_qr, t_plll};
        }


        /**
         * Even-Odd PLLL algorithm - parallel
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
        returnType <scalar, index> cils_eo_plll_reduction_omp(index n_proc) {
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
            //  --------  Perform the eo partial LLL reduction -------------------
            //  ------------------------------------------------------------------

            index k = 1, k1, i, i1, i2;
            index f = true, swap[n] = {}, start = 1, even = true, end = n / 2;
            std::vector<b_matrix> G_v(n);
            b_vector R_k1, Z_k1, R_k, Z_k;
            b_matrix R_G, G_m;
            t_plll = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(k1, k, i, i1, i2, zeta, alpha, R_k1, Z_k1, G_m, R_G, R_k, Z_k)
            {
                while (f || !even) {
#pragma omp barrier
#pragma omp atomic write
                    f = false;

#pragma omp for schedule(static, 1)
                    for (i2 = 0; i2 < end; i2++) {
                        k = i2 * 2 + start;
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
                                if (abs(zeta) >= 2) {
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
#pragma omp barrier
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