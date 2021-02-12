#include <cstring>

#include <Python.h>
#include <numpy/arrayobject.h>

namespace cils {

    template<typename scalar, typename index>
    class cils_reduction {
    private:
        index m, n, upper, lower;
        bool verbose, eval;

        /**
         * Evaluating the LLL decomposition
         * @return
         */
        returnType <scalar, index> lll_validation() {
            printf("[ INFO: in LLL validation]\n");
            scalar sum, error = 0, det;

            if (verbose) {
                printf("[ Printing R]\n");
                for (index i = 0; i < n; i++) {
                    for (index j = 0; j < n; j++) {
                        printf("%8.5f ", R_R[j * n + i]);
                    }
                    printf("\n");
                }
                printf("[ Printing Z]\n");
                for (index i = 0; i < n; i++) {
                    for (index j = 0; j < n; j++) {
                        printf("%8.5f ", Z[j * n + i]);
                    }
                    printf("\n");
                }
            }
            std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr = matlab::engine::startMATLAB();

            //Create MATLAB data array factory
            matlab::data::ArrayFactory factory;

            // Call the MATLAB movsum function
            matlab::data::TypedArray<double> R_M = factory.createArray(
                    {static_cast<unsigned long>(m), static_cast<unsigned long>(n)}, R_Q.begin(), R_Q.end());
            matlab::data::TypedArray<double> Z_M = factory.createArray(
                    {static_cast<unsigned long>(m), static_cast<unsigned long>(n)}, Z.begin(), Z.end());
            matlab::data::TypedArray<double> RCM = factory.createArray(
                    {static_cast<unsigned long>(m), static_cast<unsigned long>(n)}, R_R.begin(), R_R.end());
            matlabPtr->setVariable(u"R_M", std::move(R_M));
            matlabPtr->setVariable(u"Z_C", std::move(Z_M));
            matlabPtr->setVariable(u"R_C", std::move(RCM));

            // Call the MATLAB movsum function
            matlabPtr->eval(u" d = det(mldivide(R_M * Z_C, R_C));");
//        matlabPtr->eval(u" A = magic(n);");

            matlab::data::TypedArray<double> const d = matlabPtr->getVariable(u"d");
            int i = 0;
            for (auto r : d) {
                det = r;
                ++i;
            }

            index pass = true;
            vector<scalar> fail_index(n, 0);
            i = 2;
            while (i < n) {

                // 'eo_sils_reduction:50' i1 = i-1;
                // 'eo_sils_reduction:51' zeta = round(R_R(i1,i) / R_R(i1,i1));
                scalar zeta = std::round(R_R[i + n * (i - 1) - 2] / R_R[i + n * (i - 2) - 2]);
                // 'eo_sils_reduction:52' alpha = R_R(i1,i) - zeta * R_R(i1,i1);
                scalar s = R_R[(i + n * (i - 2)) - 2];
                scalar alpha = R_R[(i + n * (i - 1)) - 2] - zeta * s;
                // 'eo_sils_reduction:53' if R_R(i1,i1)^2 > (1 + 1.e-0) * (alpha^2 +
                // R_R(i,i)^2)
                scalar a = R_R[(i + n * (i - 1)) - 1];
                if (s * s > 2.0 * (alpha * alpha + a * a)) {
                    // 'eo_sils_reduction:54' if zeta ~= 0
                    //  Perform a size reduction on R_R(k-1,k)
                    // 'eo_sils_reduction:56' f = true;
                    pass = false;
                    fail_index[i] = 1;
                }
                i++;
            }

            return {fail_index, det, (scalar) pass};
        }

        /**
         * Evaluating the QR decomposition
         * @tparam scalar
         * @tparam index
         * @tparam n
         * @param A
         * @param Q
         * @param R
         * @param eval
         * @return
         */
        scalar qr_validation() {
            index i, j, k;
            scalar sum, error = 0;

            if (eval == 1) {
                vector<scalar> A_T(m * n, 0);
                helper::mtimes_v<scalar, index>(m, n, Q, R_Q, A_T);

                if (verbose) {
                    printf("\n[ Print Q:]\n");
                    helper::display_matrix<scalar, index>(m, m, Q.data(), "Q");
                    printf("\n[ Print R:]\n");
                    helper::display_matrix<scalar, index>(m, n, R_Q.data(), "R_Q");
                    printf("\n[ Print A:]\n");
                    helper::display_matrix<scalar, index>(m, n, A.data(), "A");
                    printf("\n[ Print Q*R:]\n");
                    helper::display_matrix<scalar, index>(m, n, A_T.data(), "Q*R");
                }

                for (i = 0; i < m * n; i++) {
                    error += fabs(A_T[i] - A[i]);
                }
            }

            return error;
        }


        /**
         * Evaluating the QR decomposition for column orientation
         * @tparam scalar
         * @tparam index
         * @tparam n
         * @param A
         * @param Q
         * @param R
         * @param eval
         * @return
         */
        scalar qr_validation_col() {
            index i, j, k;
            scalar sum, error = 0;

            if (eval == 1) {
                vector<scalar> A_T(m * n, 0);
                helper::mtimes_col<scalar, index>(m, n, Q, R_Q, A_T);

                if (verbose) {
                    printf("\n[ Print Q:]\n");
                    helper::display_matrix<scalar, index>(m, n, Q.data(), "Q");
                    printf("\n[ Print R:]\n");
                    helper::display_matrix<scalar, index>(n, n, R_Q.data(), "R_Q");
                    printf("\n[ Print A:]\n");
                    helper::display_matrix<scalar, index>(m, n, A.data(), "A");
                    printf("\n[ Print Q*R:]\n");
                    helper::display_matrix<scalar, index>(m, n, A_T.data(), "Q*R");
                }

                for (i = 0; i < m * n; i++) {
                    error += fabs(A_T[i] - A[i]);
                }
            }

            return error;
        }

    public:
        //R_A: no zeros, R_R: LLL reduced, R_Q: QR
        vector<scalar> A, R_Q, R_R, Q, G;
        vector<scalar> Z, p;
        vector<scalar> y_a, y_q, y_r;


        cils_reduction(index m, index n, index lower, index upper, bool eval, bool verbose) {
            this->m = m;
            this->n = n;
            this->eval = eval;
            this->verbose = verbose;
            this->upper = upper;
            this->lower = lower;

            this->A.resize(m * n);
            this->R_Q.resize(m * n);
            this->Q.resize(m * m);
            this->y_q.resize(m);
            this->p.resize(m);
            this->Z.resize(n * n);
            helper::eye<scalar, index>(n, Z.data());
        }

        /**
         * Serial version of QR-factorization using modified Gram-Schmidt algorithm, row-oriented
         * Results are stored in the class object.
         * @param B : m-by-n input matrix
         * @param y : m-by-1 input right hand vector
         */
        returnType <scalar, index>
        cils_qr_serial(const scalar *B, const scalar *y) {
            if (verbose)
                cout << "[ In Serial QR]\n";

            index i, j, k;
            scalar error = -1, time, sum;

            vector<scalar> A_t(m * (n + 1), 0), R_Qy(m * (n + 1), 0);
            for (i = 0; i < m * n; i++) {
                A_t[i] = B[i];
            }
            for (i = m * n; i < m * (n + 1); i++) {
                A_t[i] = y[i - m * n];
            }

            //Clear Variables:
            this->Q.assign(m * m, 0);
            this->y_q.assign(m, 0);
            //start
            time = omp_get_wtime();
            for (j = 0; j < m; j++) {
                //Check if Q[][i-1] (the previous numn) is computed.
                for (k = j; k < n + 1; k++) {
                    R_Qy[j + m * k] = 0;
                    for (i = 0; i < m; i++) {
                        R_Qy[j + m * k] += Q[i + m * j] * A_t[i + m * k];
                    }
                    for (i = 0; i < m; i++) {
                        A_t[i + m * k] -= Q[i + m * j] * R_Qy[j + m * k];
                    }
                    //Calculate the norm(A)
                    if (j == k) {
                        sum = 0;
                        for (i = 0; i < m; i++) {
                            sum += pow(A_t[i + m * k], 2);
                        }
                        R_Qy[k + m * k] = std::sqrt(sum);
                        for (i = 0; i < m; i++) {
                            Q[i + m * k] = A_t[i + m * k] / R_Qy[k + m * k];
                        }
                    }
                }
            }
            //Obtaining Results:
            for (i = 0; i < n * m; i++) {
                A[i] = B[i];
                R_Q[i] = R_Qy[i];
            }
            for (i = m * n; i < m * (n + 1); i++) {
                y_q[i - m * n] = R_Qy[i];
            }

            time = omp_get_wtime() - time;

            if (eval) {
                error = qr_validation();
                if (verbose)
                    cout << "[  QR ERROR SER:]" << error << endl;
            }

            return {{}, time, error};
        }

        /**
         * Serial version of QR-factorization using modified Gram-Schmidt algorithm, col-oriented
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

            vector<scalar> A_t(m * n, 0);
            for (i = 0; i < m * n; i++) {
                A_t[i] = B[i];
                A[i] = B[i];
            }

            //Clear Variables:
            this->R_Q.resize(n * n);
            this->R_Q.assign(n * n, 0);
            this->Q.resize(m * n);
            this->Q.assign(m * n, 0);
            this->y_q.resize(n);
            this->y_q.assign(n, 0);
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
        cils_obils_matlab(const vector<scalar> &B, const vector<scalar> &y) {
            using namespace matlab::engine;
            // Start MATLAB engine synchronously
            std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();

            //Create MATLAB data array factory
            matlab::data::ArrayFactory factory;

            matlab::data::TypedArray<scalar> B_M = factory.createArray(
                    {static_cast<unsigned long>(m), static_cast<unsigned long>(n)}, B.begin(), B.end());
            matlab::data::TypedArray<scalar> y_M = factory.createArray(
                    {static_cast<unsigned long>(m), static_cast<unsigned long>(1)}, y.begin(), y.end());
            matlab::data::TypedArray<index> t_M = factory.createScalar<index>(n);

            matlabPtr->setVariable(u"B", std::move(B_M));
            matlabPtr->setVariable(u"y", std::move(y_M));
            matlabPtr->setVariable(u"t", std::move(t_M));

            // Call the MATLAB movsum function
//            matlabPtr->eval(u" [R,y,l,u,p] = obils_reduction(B,y,zeros(t,1),7*ones(t,1));");
            matlabPtr->eval(u" x = obils(B,y, zeros(t,1), 7*ones(t,1));");

//            matlab::data::TypedArray<scalar> const R_M = matlabPtr->getVariable(u"R");
//            matlab::data::TypedArray<scalar> const y_m = matlabPtr->getVariable(u"y");
//            matlab::data::TypedArray<scalar> const p_m = matlabPtr->getVariable(u"p");
            matlab::data::TypedArray<scalar> const x_m = matlabPtr->getVariable(u"x");
//            R_Q.resize(n * n);
            y_q.resize(n);
//            p.resize(n);
//            index i = 0;
//            for (auto r : R_M) {
//                R_Q[i] = r;
//                ++i;
//            }
//            i = 0;
//            for (auto r : y_m) {
//                y_q[i] = r;
//                ++i;
//            }
//            i = 0;
//            for (auto r : p_m) {
//                p[i] = r;
//                ++i;
//            }
            index i = 0;
            for (auto r : x_m) {
                y_q[i] = r;
                ++i;
            }
            helper::display_vector<scalar, index>(n, y_q.data(), "z");
            return {{}, 0, 0};
        }

        returnType <scalar, index>
        cils_obils_reduction(const vector<scalar> &B, const vector<scalar> &y) {
            vector<scalar> b_y, c_y, b, C;

            cils_qr_serial_col(B.data(), y.data());
            R_R.resize(n * n);
            R_R.assign(R_Q.begin(), R_Q.end());
            y_r.resize(n);
            y_r.assign(y_q.begin(), y_q.end());

            //  Permutation vector
            p.resize(n);
            for (index i = 0; i < n; i++) {
                p[i] = i + 1;
            }

            //  Inverse transpose of R
            Q.resize(n * n);
            helper::inv<scalar, index>(n, n, R_R, Q);
            G.resize(n * n);
            for (index i = 0; i < n; i++) {
                for (index i1 = 0; i1 < n; i1++) {
                    G[i1 + n * i] = Q[i + n * i1];
                }
            }
            if (verbose) {
                helper::display_matrix<scalar, index>(n, n, G.data(), "G");
                helper::display_matrix<scalar, index>(n, n, R_R.data(), "R0");
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

            G.resize(n * n);
            G.assign(n * n, 0);
            for (i = 0; i < n; i++) {
                for (i1 = 0; i1 < n; i1++) {
                    G[i1 + n * i] = R_R[i1 + n * (p[i] - 1)];
                }
            }
            if (verbose) {
                helper::display_matrix<scalar, index>(n, n, G.data(), "R0");
                helper::display_matrix<scalar, index>(n, n, R_R.data(), "R_R");
                helper::display_matrix<scalar, index>(1, n, p.data(), "p");
            }

            //  Compute the QR factorization of R0 and then transform y0
            index mm = m;
            this->m = n;
            cils_qr_serial_col(G.data(), y_r.data());
            this->m = mm;
            return {{}, 0, 0};
        }

        returnType <scalar, index>
        cils_obils_reduction_matlab(const vector<scalar> &B, const vector<scalar> &y) {
            using namespace matlab::engine;
            // Start MATLAB engine synchronously
            std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();

            //Create MATLAB data array factory
            matlab::data::ArrayFactory factory;

            matlab::data::TypedArray<scalar> B_M = factory.createArray(
                    {static_cast<unsigned long>(m), static_cast<unsigned long>(n)}, B.begin(), B.end());
            matlab::data::TypedArray<scalar> y_M = factory.createArray(
                    {static_cast<unsigned long>(m), static_cast<unsigned long>(1)}, y.begin(), y.end());
            matlab::data::TypedArray<index> t_M = factory.createScalar<index>(n);

            matlabPtr->setVariable(u"B", std::move(B_M));
            matlabPtr->setVariable(u"y", std::move(y_M));
            matlabPtr->setVariable(u"t", std::move(t_M));

            // Call the MATLAB movsum function
            matlabPtr->eval(u" [R,y,l,u,p] = obils_reduction(B,y,zeros(t,1),7*ones(t,1));");
//            matlabPtr->eval(u" x = obils(B,y, zeros(t,1), 7*ones(t,1));");

            matlab::data::TypedArray<scalar> const R_M = matlabPtr->getVariable(u"R");
            matlab::data::TypedArray<scalar> const y_m = matlabPtr->getVariable(u"y");
            matlab::data::TypedArray<scalar> const p_m = matlabPtr->getVariable(u"p");
            //matlab::data::TypedArray<scalar> const x_m = matlabPtr->getVariable(u"x");
            R_Q.resize(n * n);
            y_q.resize(n);
            p.resize(n);
            index i = 0;
            for (auto r : R_M) {
                R_Q[i] = r;
                ++i;
            }
            i = 0;
            for (auto r : y_m) {
                y_q[i] = r;
                ++i;
            }
            i = 0;
            for (auto r : p_m) {
                p[i] = r;
                ++i;
            }
//            helper::display_vector<scalar, index>(n, y_q.data(), "z");
            return {{}, 0, 0};
        }

        returnType <scalar, index> cils_qr_omp(const index n_proc) {
#pragma omp parallel default(shared) num_threads(n_proc)
            {}
            cout << "[ In Parallel QR]\n";
            cout.flush();
            scalar error = -1, time, sum = 0, sum_q = 0;
            auto lock = new omp_lock_t[n]();
//        scalar *A_t = new scalar[n * n];
            vector<scalar> A_t(n * n, 0);
            //Clear Variables:
            for (index i = 0; i < n; i++) {
                for (index j = 0; j < n; j++) {
                    R_Q[i * n + j] = Q[i * n + j] = 0;
                    A_t[i * n + j] = A[i * n + j];
                }
                y_q[i] = y_r[i] = 0;
            }

            time = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(sum)
            {
                sum = 0;
#pragma omp for schedule(static, 1)
                for (index i = 0; i < n; i++) {
                    omp_set_lock(&lock[i]);
                }

                if (omp_get_thread_num() == 0) {
                    // Calculation of ||A||
                    for (index i = 0; i < n; i++) {
                        sum = sum + A_t[i] * A_t[i];
                    }
                    R_Q[0] = sqrt(sum);
                    for (index i = 0; i < n; i++) {
                        Q[i] = A_t[i] / R_Q[0];
                    }
                    omp_unset_lock(&lock[0]);
                }

                for (index k = 1; k < n; k++) {
                    //Check if Q[][i-1] (the previous column) is computed.
                    omp_set_lock(&lock[k - 1]);
                    omp_unset_lock(&lock[k - 1]);
#pragma omp for schedule(static, 1)
                    for (index j = 0; j < n; j++) {
                        if (j >= k) {
                            R_Q[(k - 1) * n + j] = 0;
                            for (index i = 0; i < n; i++) {
                                R_Q[j * n + (k - 1)] += Q[(k - 1) * n + i] * A_t[j * n + i];
                            }
                            for (index i = 0; i < n; i++) {
                                A_t[j * n + i] = A_t[j * n + i] - R_Q[j * n + (k - 1)] * Q[(k - 1) * n + i];
                            }
//Only one thread calculates the norm(A)//and unsets the lock for the next column.
                            if (j == k) {
                                sum = 0;
                                for (index i = 0; i < n; i++) {
                                    sum = sum + A_t[k * n + i] * A_t[k * n + i];
                                }
                                R_Q[k * n + k] = sqrt(sum);
                                for (index i = 0; i < n; i++) {
                                    Q[k * n + i] = A_t[k * n + i] / R_Q[k * n + k];
                                }
                                omp_unset_lock(&lock[k]);
                            }
                        }
                    }
                }

                for (index i = 0; i < n; i++) {
#pragma omp barrier
                    sum_q = 0;
#pragma omp for schedule(static, 1) reduction(+ : sum_q)
                    for (index j = 0; j < n; j++) {
                        sum_q += Q[i * n + j] * y_a[j]; //Q'*y;
                    }
                    y_q[i] = sum_q;
                    y_r[i] = y_q[i];
                }
            }
            time = omp_get_wtime() - time;

            if (eval || verbose) {
                error = qr_validation(R_Q, eval, verbose);
            }
            for (index i = 0; i < n; i++) {
                omp_destroy_lock(&lock[i]);
            }
#pragma parallel omp cancellation point
#pragma omp flush
            delete[] lock;
//        delete[] A_t;
            cout << "[  QR ERROR OMP:]" << error << endl;
            return {{}, time, error};


            /*
             * cout << "[ In Parallel QR]\n";
            cout.flush();
            scalar error = -1, time, sum = 0;
            scalar *A_t = new scalar[n * n];
            //Clear Variables:
            for (index i = 0; i < n; i++) {
                for (index j = 0; j < n; j++) {
                    R_Q[i * n + j] = 0;
                    Q[i * n + j] = 0;
                    A_t[i * n + j] = A[i * n + j];
                }
            }

            time = omp_get_wtime();
    #pragma omp parallel default(shared) num_threads(n_proc) private(sum)
            {
    #pragma omp for schedule(dynamic, 1)
                for (index j = 0; j < n; j++) {
                    for (index k = 0; k <= j; k++) {
                        R_Q[k * n + j] = 0;
                        for (index i = 0; i < n; i++) {
                            R_Q[j * n + k] += Q[k * n + i] * A_t[j * n + i];
                        }
                        for (index i = 0; i < n; i++) {
                            A_t[j * n + i] -= R_Q[j * n + k] * Q[k * n + i];
                        }
                        if (j == k) {
                            sum = 0;
                            for (index i = 0; i < n; i++) {
                                sum = sum + A_t[k * n + i] * A_t[k * n + i];
                            }
                            R_Q[k * n + k] = sqrt(sum);
                            for (index i = 0; i < n; i++) {
                                Q[k * n + i] = A_t[k * n + i] / R_Q[k * n + k];
                            }
                        }

                    }
                }
            }

            time = omp_get_wtime() - time;
            if (eval || verbose) {
                error = qr_validation<scalar, index, m,  n>(A, Q, R_Q, eval, verbose);
            }
    #pragma parallel omp cancellation point
    #pragma omp flush
            delete[] A_t;

            cout << "[  QR ERROR OMP:]" << error << endl;
            return {{}, time, error};
             */
        }


        returnType <scalar, index>
        cils_LLL_qr_reduction(const index n_proc) {
            scalar time = 0, det = 0;
            returnType<scalar, index> reT, lll_val;
            helper::eye<scalar, index>(n, Z);
            if (n_proc <= 1) {
                reT = cils_LLL_qr_serial();
                printf("[ INFO: SER LLL QR TIME: %8.5f, Givens: %.1f]\n",
                       reT.run_time, reT.info);
                time = reT.run_time;
            } else {
                time = cils_LLL_qr_omp(n_proc);
                printf("[ INFO: OMP LLL QR TIME: %8.5f]\n", time);
            }

            if (eval) {
                lll_val = lll_validation(verbose);
                if (lll_val.info != 1) {
                    cerr << "0: LLL Failed, index:";
                    for (index i = 0; i < n; i++) {
                        if (lll_val.x[i] != 0)
                            cerr << i << ",";
                    }
                    cout << endl;
                    for (index l = 0; l < 3; l++) {
                        if (n_proc <= 1) {
                            reT = cils_LLL_qr_serial();
                            time = reT.run_time;
                        } else
                            time = cils_LLL_qr_omp(n_proc);
                        //return {fail_index, det, (scalar) pass};
                        //lll_val.run_time ====> DET
                        lll_val = lll_validation(verbose);
                        if (lll_val.info == 1)
                            break;
                        else
                            cerr << l << " LLL Failed, index:";
                        for (index i = 0; i < n; i++) {
                            if (lll_val.x[i] != 0)
                                cerr << i << ",";
                        }
                    }
                }
            }

            return {{reT.info, lll_val.info}, time, lll_val.run_time};
        }


        returnType <scalar, index>
        cils_LLL_reduction(const index n_proc) {
            scalar time = 0, det = 0;
            returnType<scalar, index> reT, lll_val;

            helper::eye<scalar, index>(n, Z.data());

            if (n_proc <= 1) {
                reT = cils_LLL_serial();
                printf("[ INFO: SER LLL TIME: %8.5f, Givens: %.1f]\n",
                       reT.run_time, reT.info);
                time = reT.run_time;
            } else {
                time = cils_LLL_omp(n_proc);
                printf("[ INFO: OMP LLL TIME: %8.5f]\n", time);
            }

            if (eval) {
                lll_val = lll_validation(verbose);
                if (lll_val.info != 1) {
                    cerr << "0: LLL Failed, index:";
                    for (index i = 0; i < n; i++) {
                        if (lll_val.x[i] != 0)
                            cout << i << ",";
                    }
                    cout << endl;
                    for (index l = 0; l < 3; l++) {
                        if (n_proc <= 1) {
                            reT = cils_LLL_serial();
                            time = reT.run_time;
                        } else
                            time = cils_LLL_omp(n_proc);
                        //return {fail_index, det, (scalar) pass};
                        //lll_val.run_time ====> DET
                        lll_val = lll_validation(verbose);
                        if (lll_val.info == 1)
                            break;
                        else
                            cout << l << " LLL Failed, index:";
                        for (index i = 0; i < n; i++) {
                            if (lll_val.x[i] != 0)
                                cout << i << ",";
                        }
                        cout << endl;
                    }
                }
            }

            return {{reT.info, lll_val.info}, time, lll_val.run_time};
        }


        returnType <scalar, index> cils_LLL_serial() {
            if (verbose)
                cout << "[ In cils_LLL_serial]" << endl;
            bool f = true, givens = false;
            scalar zeta, r_ii, alpha, s;

            index swap[n] = {}, counter = 0, c_i;
            index i1, ci2, c_tmp, tmp, i2, odd = static_cast<int>((n + -1.0) / 2.0);
            scalar b_R[n * n] = {};

            R_R.resize(m * n);
            Z.resize(n * n);
            y_r.resize(m);
            helper::eye<scalar, index>(n, Z.data());
            for (index i = 0; i < n * m; i++) {
                R_R[i] = R_Q[i];
            }
            for (index i = 0; i < m; i++) {
                y_r[i] = y_q[i];
            }

            scalar time = omp_get_wtime();
            while (f) {
                f = false;

                for (index i = 0; i < n / 2; i++) {
                    c_i = static_cast<int>((i << 1) + 2U);
                    zeta = round(R_R[(c_i + m * (c_i - 1)) - 2] / R_R[(c_i + m * (c_i - 2)) - 2]);
                    s = R_R[(c_i + m * (c_i - 2)) - 2];
                    alpha = R_R[(c_i + m * (c_i - 1)) - 2] - zeta * s;
                    r_ii = R_R[(c_i + m * (c_i - 1)) - 1];
                    if (s * s > 2.0 * (alpha * alpha + r_ii * r_ii)) {
                        f = true;
                        swap[c_i - 1] = 1;
                        if (zeta != 0.0) {
                            R_R[(c_i + m * (c_i - 1)) - 2] = alpha;
                            if (1 <= c_i - 2) {
                                ci2 = c_i - 2;
                                for (i1 = 0; i1 < ci2; i1++) {
                                    R_R[i1 + m * (c_i - 1)] -= zeta * R_R[i1 + m * (ci2)];
                                }
                            }
                            for (i1 = 0; i1 < n; i1++) {
                                Z[i1 + n * (c_i - 1)] -= zeta * Z[i1 + n * (c_i - 2)];
                            }
                        }

                        for (i2 = 0; i2 < c_i; i2++) {
                            b_R[i2 + m * (c_i - 1)] = R_R[i2 + m * (c_i - 1)];
                            b_R[i2 + m * (c_i - 2)] = R_R[i2 + m * (c_i - 2)];
                        }

                        for (i2 = 0; i2 < c_i; i2++) {
                            R_R[i2 + m * (c_i - 2)] = b_R[i2 + m * (c_i - 1)];
                            R_R[i2 + m * (c_i - 1)] = b_R[i2 + m * (c_i - 2)];
                        }

                        for (i2 = 0; i2 < n; i2++) {
                            b_R[i2 + n * (c_i - 1)] = Z[i2 + n * (c_i - 1)];
                            b_R[i2 + n * (c_i - 2)] = Z[i2 + n * (c_i - 2)];
                        }

                        for (i2 = 0; i2 < n; i2++) {
                            Z[i2 + n * (c_i - 2)] = b_R[i2 + n * (c_i - 1)];
                            Z[i2 + n * (c_i - 1)] = b_R[i2 + n * (c_i - 2)];
                        }
                    }
                }

                for (index i = 0; i < n / 2; i++) {
                    c_i = static_cast<int>((i << 1) + 2U);
                    // 'eo_sils_reduction:69' i1 = i-1;
                    // 'eo_sils_reduction:70' if swap(i) == 1
                    if (swap[c_i - 1]) {
                        givens = true;
                        // 'eo_sils_reduction:71' [G,R_R([i1,i],i1)] = planerot(R_R([i1,i],i1));
                        scalar G[4] = {};
                        scalar low_tmp[2] = {R_R[(c_i + m * (c_i - 2)) - 2], R_R[(c_i + m * (c_i - 2)) - 1]};
                        helper::planerot<scalar, index>(low_tmp, G);
                        R_R[(c_i + m * (c_i - 2)) - 2] = low_tmp[0];
                        R_R[(c_i + m * (c_i - 2)) - 1] = low_tmp[1];

                        // 'eo_sils_reduction:72' R_R([i1,i],i:n) = G * R_R([i1,i],i:n);
                        if (c_i > n) {
                            i1 = i2 = 0;
                            c_tmp = 1;
                        } else {
                            i1 = c_i - 1;
                            i2 = n;
                            c_tmp = c_i;
                        }
                        ci2 = i2 - i1;
                        scalar b[ci2 * 2] = {};
                        for (i2 = 0; i2 < ci2; i2++) {
                            tmp = i1 + i2;
                            b[2 * i2] = R_R[(c_i + m * tmp) - 2];
                            b[2 * i2 + 1] = R_R[(c_i + m * tmp) - 1];
                        }
//                            scalar r1[ci2 * 2] = {};
//                            for (index j = 0; j < ci2; j++) {
//                                index coffset_tmp = j << 1;
//                                r1[coffset_tmp] = G[0] * b[coffset_tmp] + G[2] * b[coffset_tmp + 1];
//                                r1[coffset_tmp + 1] = G[1] * b[coffset_tmp] + G[3] * b[coffset_tmp + 1];
//                            }
//
////                        b_loop_ub = r1.size(1);
//                            for (i1 = 0; i1 < ci2; i1++) {
//                                tmp = (c_tmp + i1) - 1;
//                                R_R[(c_i + n * tmp) - 2] = r1[2 * i1];
//                                R_R[(c_i + n * tmp) - 1] = r1[2 * i1 + 1];
//                            }
                        for (i2 = 0; i2 < ci2; i2++) {
                            tmp = i1 + i2;
                            R_R[(c_i + m * tmp) - 2] = G[0] * b[2 * i2] + G[2] * b[2 * i2 + 1];
                            R_R[(c_i + m * tmp) - 1] = G[1] * b[2 * i2] + G[3] * b[2 * i2 + 1];
                        }
                        // 'eo_sils_reduction:73' y_L([i1,i]) = G * y_L([i1,i]);
                        low_tmp[0] = y_r[c_i - 2];
                        low_tmp[1] = y_r[c_i - 1];
                        y_r[c_i - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                        y_r[c_i - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
                        // 'eo_sils_reduction:74' swap(i) = 0;
                        swap[c_i - 1] = 0;
                    }
                }

                for (index b_i = 0; b_i < odd; b_i++) {
                    c_i = static_cast<int>((b_i << 1) + 3U);

                    // 'eo_sils_reduction:50' i1 = i-1;
                    // 'eo_sils_reduction:51' zeta = round(R_R(i1,i) / R_R(i1,i1));
                    zeta = std::round(R_R[(c_i + m * (c_i - 1)) - 2] /
                                      R_R[(c_i + m * (c_i - 2)) - 2]);
                    // 'eo_sils_reduction:52' alpha = R_R(i1,i) - zeta * R_R(i1,i1);
                    s = R_R[(c_i + m * (c_i - 2)) - 2];
                    alpha = R_R[(c_i + m * (c_i - 1)) - 2] - zeta * s;
                    // 'eo_sils_reduction:53' if R_R(i1,i1)^2 > (1 + 1.e-0) * (alpha^2 +
                    // R_R(i,i)^2)
                    r_ii = R_R[(c_i + m * (c_i - 1)) - 1];
                    if (s * s > 2.0 * (alpha * alpha + r_ii * r_ii)) {
                        f = true;
                        swap[c_i - 1] = 1;
                        if (zeta != 0.0) {
                            R_R[(c_i + m * (c_i - 1)) - 2] = alpha;
                            if (1 <= c_i - 2) {
                                ci2 = c_i - 2;
                                for (i1 = 0; i1 < ci2; i1++) {
                                    R_R[i1 + m * (c_i - 1)] -= zeta * R_R[i1 + m * (ci2)];
                                }
                            }
                            for (i1 = 0; i1 < n; i1++) {
                                Z[i1 + n * (c_i - 1)] -= zeta * Z[i1 + n * (c_i - 2)];
                            }
                        }

                        for (i2 = 0; i2 < c_i; i2++) {
                            b_R[i2 + m * (c_i - 1)] = R_R[i2 + m * (c_i - 1)];
                            b_R[i2 + m * (c_i - 2)] = R_R[i2 + m * (c_i - 2)];
                        }

                        for (i2 = 0; i2 < c_i; i2++) {
                            R_R[i2 + m * (c_i - 2)] = b_R[i2 + m * (c_i - 1)];
                            R_R[i2 + m * (c_i - 1)] = b_R[i2 + m * (c_i - 2)];
                        }

                        for (i2 = 0; i2 < n; i2++) {
                            b_R[i2 + n * (c_i - 1)] = Z[i2 + n * (c_i - 1)];
                            b_R[i2 + n * (c_i - 2)] = Z[i2 + n * (c_i - 2)];
                        }

                        for (i2 = 0; i2 < n; i2++) {
                            Z[i2 + n * (c_i - 2)] = b_R[i2 + n * (c_i - 1)];
                            Z[i2 + n * (c_i - 1)] = b_R[i2 + n * (c_i - 2)];
                        }
                    }
                }

                for (index b_i = 0; b_i < odd; b_i++) {
                    c_i = static_cast<int>((b_i << 1) + 3U);
                    // 'eo_sils_reduction:69' i1 = i-1;
                    // 'eo_sils_reduction:70' if swap(i) == 1
                    if (swap[c_i - 1]) {
                        // 'eo_sils_reduction:71' [G,R_R([i1,i],i1)] = planerot(R_R([i1,i],i1));
                        scalar G[4] = {};
                        scalar low_tmp[2] = {R_R[(c_i + m * (c_i - 2)) - 2], R_R[(c_i + m * (c_i - 2)) - 1]};
                        helper::planerot<scalar, index>(low_tmp, G);
                        R_R[(c_i + m * (c_i - 2)) - 2] = low_tmp[0];
                        R_R[(c_i + m * (c_i - 2)) - 1] = low_tmp[1];
                        // 'eo_sils_reduction:72' R_R([i1,i],i:n) = G * R_R([i1,i],i:n);
                        if (c_i > n) {
                            i1 = i2 = 0;
                            c_tmp = 1;
                        } else {
                            i1 = c_i - 1;
                            i2 = n;
                            c_tmp = c_i;
                        }
                        ci2 = i2 - i1;
                        scalar b[ci2 * 2] = {};
                        for (i2 = 0; i2 < ci2; i2++) {
                            tmp = i1 + i2;
                            b[2 * i2] = R_R[(c_i + m * tmp) - 2];
                            b[2 * i2 + 1] = R_R[(c_i + m * tmp) - 1];
                        }

                        for (i2 = 0; i2 < ci2; i2++) {
                            tmp = i1 + i2;
                            R_R[(c_i + m * tmp) - 2] = G[0] * b[2 * i2] + G[2] * b[2 * i2 + 1];
                            R_R[(c_i + m * tmp) - 1] = G[1] * b[2 * i2] + G[3] * b[2 * i2 + 1];
                        }
                        // 'eo_sils_reduction:73' y_L([i1,i]) = G * y_L([i1,i]);
                        low_tmp[0] = y_r[c_i - 2];
                        low_tmp[1] = y_r[c_i - 1];
                        y_r[c_i - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                        y_r[c_i - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
                        // 'eo_sils_reduction:74' swap(i) = 0;
                        swap[c_i - 1] = 0;
                    }
                }

            }
            time = omp_get_wtime() - time;

            return {{}, time, (scalar) givens};
        }


        scalar cils_LLL_omp(const index n_proc) {
            cout << "[ In cils_LLL_omp]\n";
            cout.flush();
#pragma omp parallel default(shared) num_threads(n_proc)
            {}
            bool f = true;
            scalar zeta, r_ii, alpha, s;

            index swap[n] = {}, counter = 0, c_i;
            index i1, ci2, c_tmp, tmp, i2, odd = static_cast<int>((n + -1.0) / 2.0);
            scalar b_R[n * n] = {};


            for (index i = 0; i < n; i++) {
                for (index j = 0; j < n; j++) {
                    R_R[i * n + j] = R_Q[i * n + j];
                }
            }
            auto R_RA = R_R.data();
            auto Z_A = Z.data();

            scalar time = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private (c_i, i1, ci2, c_tmp, tmp, i2, zeta, r_ii, alpha, s, counter)
            {
                while (f) {
#pragma omp barrier
#pragma omp atomic write
                    f = false;

#pragma omp for schedule(static, 1)
                    for (index i = 0; i < n / 2; i++) {
                        c_i = static_cast<int>((i << 1) + 2U);
                        zeta = round(R_RA[(c_i + n * (c_i - 1)) - 2] / R_RA[(c_i + n * (c_i - 2)) - 2]);
                        s = R_RA[(c_i + n * (c_i - 2)) - 2];
                        alpha = R_RA[(c_i + n * (c_i - 1)) - 2] - zeta * s;
                        r_ii = R_RA[(c_i + n * (c_i - 1)) - 1];
                        if (s * s > 2.0 * (alpha * alpha + r_ii * r_ii)) {
                            f = true;
                            swap[c_i - 1] = 1;
                            if (zeta != 0.0) {
                                R_RA[(c_i + n * (c_i - 1)) - 2] = alpha;
                                if (1 <= c_i - 2) {
                                    ci2 = c_i - 2;
                                    for (i1 = 0; i1 < ci2; i1++) {
                                        R_RA[i1 + n * (c_i - 1)] -= zeta * R_RA[i1 + n * (ci2)];
                                    }
                                }
                                for (i1 = 0; i1 < n; i1++) {
                                    Z_A[i1 + n * (c_i - 1)] -= zeta * Z_A[i1 + n * (c_i - 2)];
                                }
                            }

                            for (i2 = 0; i2 < c_i; i2++) {
                                b_R[i2 + n * (c_i - 1)] = R_RA[i2 + n * (c_i - 1)];
                                b_R[i2 + n * (c_i - 2)] = R_RA[i2 + n * (c_i - 2)];
                            }

                            for (i2 = 0; i2 < c_i; i2++) {
                                R_RA[i2 + n * (c_i - 2)] = b_R[i2 + n * (c_i - 1)];
                                R_RA[i2 + n * (c_i - 1)] = b_R[i2 + n * (c_i - 2)];
                            }

                            for (i2 = 0; i2 < n; i2++) {
                                b_R[i2 + n * (c_i - 1)] = Z_A[i2 + n * (c_i - 1)];
                                b_R[i2 + n * (c_i - 2)] = Z_A[i2 + n * (c_i - 2)];
                            }

                            for (i2 = 0; i2 < n; i2++) {
                                Z_A[i2 + n * (c_i - 2)] = b_R[i2 + n * (c_i - 1)];
                                Z_A[i2 + n * (c_i - 1)] = b_R[i2 + n * (c_i - 2)];
                            }
                        }
                    }

#pragma omp for schedule(static, 1)
                    for (index i = 0; i < n / 2; i++) {
                        c_i = static_cast<int>((i << 1) + 2U);
                        // 'eo_sils_reduction:69' i1 = i-1;
                        // 'eo_sils_reduction:70' if swap(i) == 1
                        if (swap[c_i - 1]) {
                            // 'eo_sils_reduction:71' [G,R_RA([i1,i],i1)] = planerot(R_RA([i1,i],i1));
                            scalar G[4] = {};
                            scalar low_tmp[2] = {R_RA[(c_i + n * (c_i - 2)) - 2], R_RA[(c_i + n * (c_i - 2)) - 1]};
                            helper::planerot<scalar, index>(low_tmp, G);
                            R_RA[(c_i + n * (c_i - 2)) - 2] = low_tmp[0];
                            R_RA[(c_i + n * (c_i - 2)) - 1] = low_tmp[1];

                            // 'eo_sils_reduction:72' R_RA([i1,i],i:n) = G * R_RA([i1,i],i:n);
                            if (c_i > n) {
                                i1 = i2 = 0;
                                c_tmp = 1;
                            } else {
                                i1 = c_i - 1;
                                i2 = n;
                                c_tmp = c_i;
                            }
                            ci2 = i2 - i1;
                            scalar b[ci2 * 2] = {};
                            for (i2 = 0; i2 < ci2; i2++) {
                                tmp = i1 + i2;
                                b[2 * i2] = R_RA[(c_i + n * tmp) - 2];
                                b[2 * i2 + 1] = R_RA[(c_i + n * tmp) - 1];
                            }
//                            scalar r1[ci2 * 2] = {};
//                            for (index j = 0; j < ci2; j++) {
//                                index coffset_tmp = j << 1;
//                                r1[coffset_tmp] = G[0] * b[coffset_tmp] + G[2] * b[coffset_tmp + 1];
//                                r1[coffset_tmp + 1] = G[1] * b[coffset_tmp] + G[3] * b[coffset_tmp + 1];
//                            }
//
////                        b_loop_ub = r1.size(1);
//                            for (i1 = 0; i1 < ci2; i1++) {
//                                tmp = (c_tmp + i1) - 1;
//                                R_RA[(c_i + n * tmp) - 2] = r1[2 * i1];
//                                R_RA[(c_i + n * tmp) - 1] = r1[2 * i1 + 1];
//                            }
                            for (i2 = 0; i2 < ci2; i2++) {
                                tmp = i1 + i2;
                                R_RA[(c_i + n * tmp) - 2] = G[0] * b[2 * i2] + G[2] * b[2 * i2 + 1];
                                R_RA[(c_i + n * tmp) - 1] = G[1] * b[2 * i2] + G[3] * b[2 * i2 + 1];
                            }
                            // 'eo_sils_reduction:73' y_L([i1,i]) = G * y_L([i1,i]);
                            low_tmp[0] = y_r[c_i - 2];
                            low_tmp[1] = y_r[c_i - 1];
                            y_r[c_i - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                            y_r[c_i - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
                            // 'eo_sils_reduction:74' swap(i) = 0;
                            swap[c_i - 1] = 0;
                        }
                    }

//#pragma omp barrier
//#pragma omp atomic write
//                f = false;
#pragma omp for schedule(static, 1)
                    for (index b_i = 0; b_i < odd; b_i++) {
                        c_i = static_cast<int>((b_i << 1) + 3U);

                        // 'eo_sils_reduction:50' i1 = i-1;
                        // 'eo_sils_reduction:51' zeta = round(R_RA(i1,i) / R_RA(i1,i1));
                        zeta = std::round(R_RA[(c_i + n * (c_i - 1)) - 2] /
                                          R_RA[(c_i + n * (c_i - 2)) - 2]);
                        // 'eo_sils_reduction:52' alpha = R_RA(i1,i) - zeta * R_RA(i1,i1);
                        s = R_RA[(c_i + n * (c_i - 2)) - 2];
                        alpha = R_RA[(c_i + n * (c_i - 1)) - 2] - zeta * s;
                        // 'eo_sils_reduction:53' if R_RA(i1,i1)^2 > (1 + 1.e-0) * (alpha^2 +
                        // R_RA(i,i)^2)
                        r_ii = R_RA[(c_i + n * (c_i - 1)) - 1];
                        if (s * s > 2.0 * (alpha * alpha + r_ii * r_ii)) {
                            f = true;
                            swap[c_i - 1] = 1;
                            if (zeta != 0.0) {
                                R_RA[(c_i + n * (c_i - 1)) - 2] = alpha;
                                if (1 <= c_i - 2) {
                                    ci2 = c_i - 2;
                                    for (i1 = 0; i1 < ci2; i1++) {
                                        R_RA[i1 + n * (c_i - 1)] -= zeta * R_RA[i1 + n * (ci2)];
                                    }
                                }
                                for (i1 = 0; i1 < n; i1++) {
                                    Z_A[i1 + n * (c_i - 1)] -= zeta * Z_A[i1 + n * (c_i - 2)];
                                }
                            }

                            for (i2 = 0; i2 < c_i; i2++) {
                                b_R[i2 + n * (c_i - 1)] = R_RA[i2 + n * (c_i - 1)];
                                b_R[i2 + n * (c_i - 2)] = R_RA[i2 + n * (c_i - 2)];
                            }

                            for (i2 = 0; i2 < c_i; i2++) {
                                R_RA[i2 + n * (c_i - 2)] = b_R[i2 + n * (c_i - 1)];
                                R_RA[i2 + n * (c_i - 1)] = b_R[i2 + n * (c_i - 2)];
                            }

                            for (i2 = 0; i2 < n; i2++) {
                                b_R[i2 + n * (c_i - 1)] = Z_A[i2 + n * (c_i - 1)];
                                b_R[i2 + n * (c_i - 2)] = Z_A[i2 + n * (c_i - 2)];
                            }

                            for (i2 = 0; i2 < n; i2++) {
                                Z_A[i2 + n * (c_i - 2)] = b_R[i2 + n * (c_i - 1)];
                                Z_A[i2 + n * (c_i - 1)] = b_R[i2 + n * (c_i - 2)];
                            }
                        }
                    }

#pragma omp for schedule(static, 1)
                    for (index b_i = 0; b_i < odd; b_i++) {
                        c_i = static_cast<int>((b_i << 1) + 3U);
                        // 'eo_sils_reduction:69' i1 = i-1;
                        // 'eo_sils_reduction:70' if swap(i) == 1
                        if (swap[c_i - 1]) {
                            // 'eo_sils_reduction:71' [G,R_RA([i1,i],i1)] = planerot(R_RA([i1,i],i1));
                            scalar G[4] = {};
                            scalar low_tmp[2] = {R_RA[(c_i + n * (c_i - 2)) - 2], R_RA[(c_i + n * (c_i - 2)) - 1]};
                            helper::planerot<scalar, index>(low_tmp, G);
                            R_RA[(c_i + n * (c_i - 2)) - 2] = low_tmp[0];
                            R_RA[(c_i + n * (c_i - 2)) - 1] = low_tmp[1];
                            // 'eo_sils_reduction:72' R_RA([i1,i],i:n) = G * R_RA([i1,i],i:n);
                            if (c_i > n) {
                                i1 = i2 = 0;
                                c_tmp = 1;
                            } else {
                                i1 = c_i - 1;
                                i2 = n;
                                c_tmp = c_i;
                            }
                            ci2 = i2 - i1;
                            scalar b[ci2 * 2] = {};
                            for (i2 = 0; i2 < ci2; i2++) {
                                tmp = i1 + i2;
                                b[2 * i2] = R_RA[(c_i + n * tmp) - 2];
                                b[2 * i2 + 1] = R_RA[(c_i + n * tmp) - 1];
                            }

                            for (i2 = 0; i2 < ci2; i2++) {
                                tmp = i1 + i2;
                                R_RA[(c_i + n * tmp) - 2] = G[0] * b[2 * i2] + G[2] * b[2 * i2 + 1];
                                R_RA[(c_i + n * tmp) - 1] = G[1] * b[2 * i2] + G[3] * b[2 * i2 + 1];
                            }
                            // 'eo_sils_reduction:73' y_L([i1,i]) = G * y_L([i1,i]);
                            low_tmp[0] = y_r[c_i - 2];
                            low_tmp[1] = y_r[c_i - 1];
                            y_r[c_i - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                            y_r[c_i - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
                            // 'eo_sils_reduction:74' swap(i) = 0;
                            swap[c_i - 1] = 0;
                        }
                    }

                }
            }
            time = omp_get_wtime() - time;
            return time;
        }


        returnType <scalar, index> cils_LLL_qr_serial() {

            cout << "[ In cils_LLL_qr_serial]\n";
            scalar zeta, r_ii, alpha, s;
            bool f = true;
            index swap[n] = {}, givens = 0, c_i;
            index i1, ci2, c_tmp, tmp, i2, odd = static_cast<int>((n + -1.0) / 2.0);
            scalar error = -1, time, sum = 0;

            vector<scalar> A_t(n * n, 0);
            //Clear Variables:
            for (index i = 0; i < n; i++) {
                for (index j = 0; j < n; j++) {
                    Q[i * n + j] = R_R[i * n + j] = 0;
                    A_t[i * n + j] = A[i * n + j];
                }
                y_q[i] = y_r[i] = 0;
            }

            time = omp_get_wtime();
            sum = 0;

            for (index k = 0; k < n; k++) {
                //Check if Q[][i-1] (the previous column) is computed.
                for (index j = k; j < n; j++) {
                    R_R[k * n + j] = 0;
                    for (index i = 0; i < n; i++) {
                        R_R[j * n + k] += Q[k * n + i] * A_t[j * n + i];
                    }
                    for (index i = 0; i < n; i++) {
                        A_t[j * n + i] -= R_R[j * n + k] * Q[k * n + i];
                    }
                    //Calculate the norm(A)
                    if (j == k) {
                        s = sum = 0;
                        for (index i = 0; i < n; i++) {
                            sum += pow(A_t[k * n + i], 2);
                        }
                        R_R[k * n + k] = sqrt(sum);
                        for (index i = 0; i < n; i++) {
                            Q[k * n + i] = A_t[k * n + i] / R_R[k * n + k];
                            s += Q[k * n + i] * y_a[i]; //Q'*y;
                        }
                        y_q[k] = s;
                        y_r[k] = y_q[k];
                    }
                }

                if (k % 2 == 0 && k != 0) {
                    c_i = k;
                    zeta = round(R_R[(c_i + n * (c_i - 1)) - 2] / R_R[(c_i + n * (c_i - 2)) - 2]);
                    s = R_R[(c_i + n * (c_i - 2)) - 2];
                    alpha = R_R[(c_i + n * (c_i - 1)) - 2] - zeta * s;
                    r_ii = R_R[(c_i + n * (c_i - 1)) - 1];
                    if (s * s > 2.0 * (alpha * alpha + r_ii * r_ii)) {
                        if (zeta != 0.0) {
                            R_R[(c_i + n * (c_i - 1)) - 2] = alpha;
                            if (1 <= c_i - 2) {
                                ci2 = c_i - 2;
                                for (i1 = 0; i1 < ci2; i1++) {
                                    R_R[i1 + n * (c_i - 1)] -= zeta * R_R[i1 + n * (ci2)];
                                }
                            }
                            for (i1 = 0; i1 < n; i1++) {
                                Z[i1 + n * (c_i - 1)] -= zeta * Z[i1 + n * (c_i - 2)];
                            }
                        }
                        scalar b_R[n * 2] = {};
                        for (i2 = 0; i2 < c_i; i2++) {
                            b_R[i2] = R_R[i2 + n * (c_i - 1)];
                            b_R[i2 + c_i] = R_R[i2 + n * (c_i - 2)];
                        }

                        for (i2 = 0; i2 < c_i; i2++) {
                            R_R[i2 + n * (c_i - 2)] = b_R[i2];
                            R_R[i2 + n * (c_i - 1)] = b_R[i2 + c_i];
                        }

                        for (i2 = 0; i2 < n; i2++) {
                            b_R[i2] = Z[i2 + n * (c_i - 1)];
                            b_R[i2 + n] = Z[i2 + n * (c_i - 2)];
                        }
                        for (i2 = 0; i2 < n; i2++) {
                            Z[i2 + n * (c_i - 2)] = b_R[i2];
                            Z[i2 + n * (c_i - 1)] = b_R[i2 + n];
                        }

                        scalar G[4] = {};
                        scalar low_tmp[2] = {R_R[(c_i + n * (c_i - 2)) - 2], R_R[(c_i + n * (c_i - 2)) - 1]};
                        helper::planerot<scalar, index>(low_tmp, G);
                        R_R[(c_i + n * (c_i - 2)) - 2] = low_tmp[0];
                        R_R[(c_i + n * (c_i - 2)) - 1] = low_tmp[1];

                        // 'eo_sils_reduction:72' R_R([i1,i],i:n) = G * R_R([i1,i],i:n);
                        if (c_i > n) {
                            i1 = i2 = 0;
                            c_tmp = 1;
                        } else {
                            i1 = c_i - 1;
                            i2 = n;
                            c_tmp = c_i;
                        }
                        ci2 = i2 - i1;
                        scalar b[ci2 * 2] = {};
                        for (i2 = 0; i2 < ci2; i2++) {
                            tmp = i1 + i2;
                            b[2 * i2] = R_R[(c_i + n * tmp) - 2];
                            b[2 * i2 + 1] = R_R[(c_i + n * tmp) - 1];
                        }

                        for (i2 = 0; i2 < ci2; i2++) {
                            tmp = i1 + i2;
                            R_R[(c_i + n * tmp) - 2] = G[0] * b[2 * i2] + G[2] * b[2 * i2 + 1];
                            R_R[(c_i + n * tmp) - 1] = G[1] * b[2 * i2] + G[3] * b[2 * i2 + 1];
                        }

                        low_tmp[0] = y_r[c_i - 2];
                        low_tmp[1] = y_r[c_i - 1];
                        y_r[c_i - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                        y_r[c_i - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];

                    }
                }
            }

            while (f) {
                f = false;
                for (index b_i = 0; b_i < odd; b_i++) {
                    c_i = static_cast<int>((b_i << 1) + 3U);

                    // 'eo_sils_reduction:50' i1 = i-1;
                    // 'eo_sils_reduction:51' zeta = round(R_R(i1,i) / R_R(i1,i1));
                    zeta = std::round(R_R[(c_i + n * (c_i - 1)) - 2] /
                                      R_R[(c_i + n * (c_i - 2)) - 2]);
                    // 'eo_sils_reduction:52' alpha = R_R(i1,i) - zeta * R_R(i1,i1);
                    s = R_R[(c_i + n * (c_i - 2)) - 2];
                    alpha = R_R[(c_i + n * (c_i - 1)) - 2] - zeta * s;
                    // 'eo_sils_reduction:53' if R_R(i1,i1)^2 > (1 + 1.e-0) * (alpha^2 +
                    // R_R(i,i)^2)
                    r_ii = R_R[(c_i + n * (c_i - 1)) - 1];
                    if (s * s > 2.0 * (alpha * alpha + r_ii * r_ii)) {
                        f = true;
                        swap[c_i - 1] = 1;
                        if (zeta != 0.0) {
                            R_R[(c_i + n * (c_i - 1)) - 2] = alpha;
                            if (1 <= c_i - 2) {
                                ci2 = c_i - 2;
                                for (i1 = 0; i1 < ci2; i1++) {
                                    R_R[i1 + n * (c_i - 1)] -= zeta * R_R[i1 + n * (ci2)];
                                }
                            }
                            for (i1 = 0; i1 < n; i1++) {
                                Z[i1 + n * (c_i - 1)] -= zeta * Z[i1 + n * (c_i - 2)];
                            }
                        }
                        scalar b_R[n * 2] = {};
                        for (i2 = 0; i2 < c_i; i2++) {
                            b_R[i2] = R_R[i2 + n * (c_i - 1)];
                            b_R[i2 + c_i] = R_R[i2 + n * (c_i - 2)];
                        }

                        for (i2 = 0; i2 < c_i; i2++) {
                            R_R[i2 + n * (c_i - 2)] = b_R[i2];
                            R_R[i2 + n * (c_i - 1)] = b_R[i2 + c_i];
                        }

                        for (i2 = 0; i2 < n; i2++) {
                            b_R[i2] = Z[i2 + n * (c_i - 1)];
                            b_R[i2 + n] = Z[i2 + n * (c_i - 2)];
                        }

                        for (i2 = 0; i2 < n; i2++) {
                            Z[i2 + n * (c_i - 2)] = b_R[i2];
                            Z[i2 + n * (c_i - 1)] = b_R[i2 + n];
                        }
                    }
                }

                for (index b_i = 0; b_i < odd; b_i++) {
                    c_i = static_cast<int>((b_i << 1) + 3U);
                    // 'eo_sils_reduction:69' i1 = i-1;
                    // 'eo_sils_reduction:70' if swap(i) == 1
                    if (swap[c_i - 1]) {
                        // 'eo_sils_reduction:71' [G,R_R([i1,i],i1)] = planerot(R_R([i1,i],i1));
                        scalar G[4] = {};
                        scalar low_tmp[2] = {R_R[(c_i + n * (c_i - 2)) - 2], R_R[(c_i + n * (c_i - 2)) - 1]};
                        helper::planerot<scalar, index>(low_tmp, G);
                        R_R[(c_i + n * (c_i - 2)) - 2] = low_tmp[0];
                        R_R[(c_i + n * (c_i - 2)) - 1] = low_tmp[1];
                        // 'eo_sils_reduction:72' R_R([i1,i],i:n) = G * R_R([i1,i],i:n);
                        if (c_i > n) {
                            i1 = i2 = 0;
                            c_tmp = 1;
                        } else {
                            i1 = c_i - 1;
                            i2 = n;
                            c_tmp = c_i;
                        }
                        ci2 = i2 - i1;
                        scalar b[ci2 * 2] = {};
                        for (i2 = 0; i2 < ci2; i2++) {
                            tmp = i1 + i2;
                            b[2 * i2] = R_R[(c_i + n * tmp) - 2];
                            b[2 * i2 + 1] = R_R[(c_i + n * tmp) - 1];
                        }

                        for (i2 = 0; i2 < ci2; i2++) {
                            tmp = i1 + i2;
                            R_R[(c_i + n * tmp) - 2] = G[0] * b[2 * i2] + G[2] * b[2 * i2 + 1];
                            R_R[(c_i + n * tmp) - 1] = G[1] * b[2 * i2] + G[3] * b[2 * i2 + 1];
                        }
                        // 'eo_sils_reduction:73' y_L([i1,i]) = G * y_L([i1,i]);
                        low_tmp[0] = y_r[c_i - 2];
                        low_tmp[1] = y_r[c_i - 1];
                        y_r[c_i - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                        y_r[c_i - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
                        // 'eo_sils_reduction:74' swap(i) = 0;
                        swap[c_i - 1] = 0;
                    }
                }

                for (index i = 0; i < n / 2; i++) {
                    c_i = static_cast<int>((i << 1) + 2U);
                    zeta = round(R_R[(c_i + n * (c_i - 1)) - 2] / R_R[(c_i + n * (c_i - 2)) - 2]);
                    s = R_R[(c_i + n * (c_i - 2)) - 2];
                    alpha = R_R[(c_i + n * (c_i - 1)) - 2] - zeta * s;
                    r_ii = R_R[(c_i + n * (c_i - 1)) - 1];
                    if (s * s > 2.0 * (alpha * alpha + r_ii * r_ii)) {
                        f = true;
                        swap[c_i - 1] = 1;
                        if (zeta != 0.0) {
                            R_R[(c_i + n * (c_i - 1)) - 2] = alpha;
                            if (1 <= c_i - 2) {
                                ci2 = c_i - 2;
                                for (i1 = 0; i1 < ci2; i1++) {
                                    R_R[i1 + n * (c_i - 1)] -= zeta * R_R[i1 + n * (ci2)];
                                }
                            }
                            for (i1 = 0; i1 < n; i1++) {
                                Z[i1 + n * (c_i - 1)] -= zeta * Z[i1 + n * (c_i - 2)];
                            }
                        }

                        scalar b_R[n * 2] = {};
                        for (i2 = 0; i2 < c_i; i2++) {
                            b_R[i2] = R_R[i2 + n * (c_i - 1)];
                            b_R[i2 + c_i] = R_R[i2 + n * (c_i - 2)];
                        }

                        for (i2 = 0; i2 < c_i; i2++) {
                            R_R[i2 + n * (c_i - 2)] = b_R[i2];
                            R_R[i2 + n * (c_i - 1)] = b_R[i2 + c_i];
                        }
                        for (i2 = 0; i2 < n; i2++) {
                            b_R[i2] = Z[i2 + n * (c_i - 1)];
                            b_R[i2 + n] = Z[i2 + n * (c_i - 2)];
                        }

                        for (i2 = 0; i2 < n; i2++) {
                            Z[i2 + n * (c_i - 2)] = b_R[i2];
                            Z[i2 + n * (c_i - 1)] = b_R[i2 + n];
                        }
                    }
                }

                for (index i = 0; i < n / 2; i++) {
                    givens = 1;
                    c_i = static_cast<int>((i << 1) + 2U);
                    // 'eo_sils_reduction:69' i1 = i-1;
                    // 'eo_sils_reduction:70' if swap(i) == 1
                    if (swap[c_i - 1]) {
                        // 'eo_sils_reduction:71' [G,R_R([i1,i],i1)] = planerot(R_R([i1,i],i1));
                        scalar G[4] = {};
                        scalar low_tmp[2] = {R_R[(c_i + n * (c_i - 2)) - 2], R_R[(c_i + n * (c_i - 2)) - 1]};
                        helper::planerot<scalar, index>(low_tmp, G);
                        R_R[(c_i + n * (c_i - 2)) - 2] = low_tmp[0];
                        R_R[(c_i + n * (c_i - 2)) - 1] = low_tmp[1];

                        // 'eo_sils_reduction:72' R_R([i1,i],i:n) = G * R_R([i1,i],i:n);
                        if (c_i > n) {
                            i1 = i2 = 0;
                            c_tmp = 1;
                        } else {
                            i1 = c_i - 1;
                            i2 = n;
                            c_tmp = c_i;
                        }
                        ci2 = i2 - i1;
                        scalar b[ci2 * 2] = {};
                        for (i2 = 0; i2 < ci2; i2++) {
                            tmp = i1 + i2;
                            b[2 * i2] = R_R[(c_i + n * tmp) - 2];
                            b[2 * i2 + 1] = R_R[(c_i + n * tmp) - 1];
                        }
//                            scalar r1[ci2 * 2] = {};
//                            for (index j = 0; j < ci2; j++) {
//                                index coffset_tmp = j << 1;
//                                r1[coffset_tmp] = G[0] * b[coffset_tmp] + G[2] * b[coffset_tmp + 1];
//                                r1[coffset_tmp + 1] = G[1] * b[coffset_tmp] + G[3] * b[coffset_tmp + 1];
//                            }
//
////                        b_loop_ub = r1.size(1);
//                            for (i1 = 0; i1 < ci2; i1++) {
//                                tmp = (c_tmp + i1) - 1;
//                                R_R[(c_i + n * tmp) - 2] = r1[2 * i1];
//                                R_R[(c_i + n * tmp) - 1] = r1[2 * i1 + 1];
//                            }
                        for (i2 = 0; i2 < ci2; i2++) {
                            tmp = i1 + i2;
                            R_R[(c_i + n * tmp) - 2] = G[0] * b[2 * i2] + G[2] * b[2 * i2 + 1];
                            R_R[(c_i + n * tmp) - 1] = G[1] * b[2 * i2] + G[3] * b[2 * i2 + 1];
                        }
                        // 'eo_sils_reduction:73' y_L([i1,i]) = G * y_L([i1,i]);
                        low_tmp[0] = y_r[c_i - 2];
                        low_tmp[1] = y_r[c_i - 1];
                        y_r[c_i - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                        y_r[c_i - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
                        // 'eo_sils_reduction:74' swap(i) = 0;
                        swap[c_i - 1] = 0;
                    }
                }
            }

            time = omp_get_wtime() - time;
            error = qr_validation(R_Q, 1, n <= 16);
            printf("[ NEW METHOD, QR ERROR: %.5f]\n", error);
            return {{}, time, (scalar) givens};
        }


        scalar cils_LLL_qr_omp(const index n_proc) {
#pragma omp parallel default(shared) num_threads(n_proc)
            {}

            cout << "[ In cils_LLL_qr_omp]\n";
            cout.flush();

            scalar zeta, r_ii, alpha, s;
            bool f = true, condition = false;
            index swap[n] = {}, counter = 0, c_i;
            index i1, ci2, c_tmp, tmp, i2, odd = static_cast<int>((n + -1.0) / 2.0);
            auto lock = new omp_lock_t[n]();
            scalar error = -1, time, sum = 0, tmp_sum = 0, sum_q = 0;
            scalar *b_R = new scalar[n * n];
            vector<scalar> A_t(A);
            //Clear Variables:
//        b_R.set_size(n, n);
            for (index i = 0; i < n; i++) {
                for (index j = 0; j < n; j++) {
                    Q[i * n + j] = 0;
                    R_R[i * n + j] = 0;
                    b_R[i * n + j] = 0;
                }
                omp_set_lock(&lock[i]);
                y_r[i] = y_q[i] = 0;
            }

            time = omp_get_wtime();

#pragma omp simd reduction(+ : sum)
            for (index i = 0; i < n; i++) {
                sum += A_t[i] * A_t[i];
            }
            R_R[0] = sqrt(sum);

            for (index i = 0; i < n; i++) {
                Q[i] = A_t[i] / R_R[0];
            }

            omp_unset_lock(&lock[0]);


#pragma omp parallel default(shared) num_threads(n_proc) private (c_i, i1, ci2, c_tmp, tmp, i2, zeta, r_ii, alpha, s, sum, counter)
            {

                sum = counter = 0;
//a_1 = q_1
//            if (omp_get_thread_num() == 0) {
//                // Calculation of ||A||
//
//            }


                for (index k = 1; k < n; k++) {
                    //Check if Q[][i-1] (the previous column) is computed.
//#pragma omp barrier
                    omp_set_lock(&lock[k - 1]);
                    omp_unset_lock(&lock[k - 1]);
#pragma omp for schedule(static, 1)
                    for (index j = 0; j < n; j++) {
                        if (j >= k) {
//                        R_Q[(k - 1) * n + j] = 0;
                            sum = j * n + (k - 1);
                            for (index i = 0; i < n; i++) {
                                R_R[sum] += Q[(k - 1) * n + i] * A_t[j * n + i];
//                            R_Q[j * n + (k - 1)] += Q[(k - 1) * n + i] * A_t[j * n + i];
//                            R_R[j * n + (k - 1)] = R_Q[(k - 1) * n + i];
                            }
                            for (index i = 0; i < n; i++) {
                                A_t[j * n + i] -= R_R[j * n + (k - 1)] * Q[(k - 1) * n + i];
//                            A_t[j * n + i] = A_t[j * n + i] - R_Q[j * n + (k - 1)] * Q[(k - 1) * n + i];
                            }
//Only one thread calculate
                            if (j == k) {
                                sum_q = sum = 0;
                                for (index i = 0; i < n; i++) {
                                    sum += pow(A_t[k * n + i], 2);
                                }
//                            R_Q[k * n + k] = sqrt(sum);
                                R_R[k * n + k] = sqrt(sum);
                                for (index i = 0; i < n; i++) {
                                    Q[k * n + i] = A_t[k * n + i] / R_R[k * n + k];
                                    sum_q += Q[k * n + i] * y_a[i]; //Q'*y;
                                }
                                y_q[k] = sum_q;
                                y_r[k] = y_q[k];
                                omp_unset_lock(&lock[k]);
                            }
                        }
//                    printf("column:%d, row:%d, %8.5f \n",j, k, R_R[j * n + (k - 1)]);
                    }
#pragma omp single
                    {

                        if (k % 2 == 0) {
//                        omp_set_lock(&lock[k - 1]);
//                        omp_set_lock(&lock[k]);
                            zeta = round(R_R[(k + n * (k - 1)) - 2] / R_R[(k + n * (k - 2)) - 2]);
                            s = R_R[(k + n * (k - 2)) - 2];
                            alpha = R_R[(k + n * (k - 1)) - 2] - zeta * s;
                            r_ii = R_R[(k + n * (k - 1)) - 1];
                            condition = s * s > 2.0 * (alpha * alpha + r_ii * r_ii);
//                    }
//#pragma omp barrier
                            if (condition) {
//#pragma omp master
//                        {
                                if (zeta != 0.0) {
                                    R_R[(k + n * (k - 1)) - 2] = alpha;
                                    if (1 <= k - 2) {
                                        ci2 = k - 2;
                                        for (i1 = 0; i1 < ci2; i1++) {
                                            R_R[i1 + n * (k - 1)] -= zeta * R_R[i1 + n * (ci2)];
                                        }
                                    }
                                    for (i1 = 0; i1 < n; i1++) {
                                        Z[i1 + n * (k - 1)] -= zeta * Z[i1 + n * (k - 2)];
                                    }
                                }
//                        }
//#pragma omp barrier
//#pragma omp for schedule(static, 1)
                                for (i2 = 0; i2 < k; i2++) {
                                    b_R[i2 + n * (k - 1)] = R_R[i2 + n * (k - 1)];
                                    b_R[i2 + n * (k - 2)] = R_R[i2 + n * (k - 2)];
                                }
//#pragma omp for schedule(static, 1)
                                for (i2 = 0; i2 < k; i2++) {
                                    R_R[i2 + n * (k - 2)] = b_R[i2 + n * (k - 1)];
                                    R_R[i2 + n * (k - 1)] = b_R[i2 + n * (k - 2)];
                                }
//#pragma omp for schedule(static, 1)
                                for (i2 = 0; i2 < n; i2++) {
                                    b_R[i2 + n * (k - 1)] = Z[i2 + n * (k - 1)];
                                    b_R[i2 + n * (k - 2)] = Z[i2 + n * (k - 2)];
                                }
//#pragma omp for schedule(static, 1)
                                for (i2 = 0; i2 < n; i2++) {
                                    Z[i2 + n * (k - 2)] = b_R[i2 + n * (k - 1)];
                                    Z[i2 + n * (k - 1)] = b_R[i2 + n * (k - 2)];
                                }

//#pragma omp single
//                            {
                                scalar G[4] = {};
                                scalar low_tmp[2] = {R_R[(k + n * (k - 2)) - 2], R_R[(k + n * (k - 2)) - 1]};
                                helper::planerot<scalar, index>(low_tmp, G);
                                R_R[(k + n * (k - 2)) - 2] = low_tmp[0];
                                R_R[(k + n * (k - 2)) - 1] = low_tmp[1];

                                // 'eo_sils_reduction:72' R_R([i1,i],i:n) = G * R_R([i1,i],i:n);
                                if (k > n) {
                                    i1 = i2 = 0;
                                    c_tmp = 1;
                                } else {
                                    i1 = k - 1;
                                    i2 = n;
                                    c_tmp = k;
                                }
                                ci2 = i2 - i1;
                                scalar b[ci2 * 2] = {};

                                for (i2 = 0; i2 < ci2; i2++) {
                                    tmp = i1 + i2;
                                    b[2 * i2] = R_R[(k + n * tmp) - 2];
                                    b[2 * i2 + 1] = R_R[(k + n * tmp) - 1];
                                }

                                for (i2 = 0; i2 < ci2; i2++) {
                                    tmp = i1 + i2;
                                    R_R[(k + n * tmp) - 2] = G[0] * b[2 * i2] + G[2] * b[2 * i2 + 1];
                                    R_R[(k + n * tmp) - 1] = G[1] * b[2 * i2] + G[3] * b[2 * i2 + 1];
                                }

                                //                            for (i2 = 0; i2 < ci2; i2++) {
                                //                                tmp = i1 + i2;
                                //                                b[2 * i2] = Q[(k + n * tmp) - 2];
                                //                                b[2 * i2 + 1] = Q[(k + n * tmp) - 1];
                                //                            }
                                //
                                //                            for (i2 = 0; i2 < ci2; i2++) {
                                //                                tmp = i1 + i2;
                                //                                Q[(k + n * tmp) - 2] = G[0] * b[2 * i2] + G[2] * b[2 * i2 + 1];
                                //                                Q[(k + n * tmp) - 1] = G[1] * b[2 * i2] + G[3] * b[2 * i2 + 1];
                                //                            }

                                // 'eo_sils_reduction:73' y_L([i1,i]) = G * y_L([i1,i]);
                                low_tmp[0] = y_r[k - 2];
                                low_tmp[1] = y_r[k - 1];
                                y_r[k - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                                y_r[k - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
                                // 'eo_sils_reduction:74' swap(i) = 0;
//                            swap[k - 1] = 1;
//                            }
                            }
//                        omp_unset_lock(&lock[k - 1]);
//                        omp_unset_lock(&lock[k]);
                        }
                    }

                }
#pragma omp barrier

                while (f && counter < 50) {
#pragma omp barrier
#pragma omp atomic write
                    f = false;

#pragma omp for schedule(static, 1)
                    for (index b_i = 0; b_i < odd; b_i++) {
                        c_i = static_cast<int>((b_i << 1) + 3U);

                        // 'eo_sils_reduction:50' i1 = i-1;
                        // 'eo_sils_reduction:51' zeta = round(R_R(i1,i) / R_R(i1,i1));
                        zeta = std::round(R_R[(c_i + n * (c_i - 1)) - 2] /
                                          R_R[(c_i + n * (c_i - 2)) - 2]);
                        // 'eo_sils_reduction:52' alpha = R_R(i1,i) - zeta * R_R(i1,i1);
                        s = R_R[(c_i + n * (c_i - 2)) - 2];
                        alpha = R_R[(c_i + n * (c_i - 1)) - 2] - zeta * s;
                        // 'eo_sils_reduction:53' if R_R(i1,i1)^2 > (1 + 1.e-0) * (alpha^2 +
                        // R_R(i,i)^2)
                        r_ii = R_R[(c_i + n * (c_i - 1)) - 1];
                        if (s * s > 2.0 * (alpha * alpha + r_ii * r_ii)) {
                            f = true;
                            swap[c_i - 1] = 1;
                            if (zeta != 0.0) {
                                R_R[(c_i + n * (c_i - 1)) - 2] = alpha;
                                if (1 <= c_i - 2) {
                                    ci2 = c_i - 2;
                                    for (i1 = 0; i1 < ci2; i1++) {
                                        R_R[i1 + n * (c_i - 1)] -= zeta * R_R[i1 + n * (ci2)];
                                    }
                                }
                                for (i1 = 0; i1 < n; i1++) {
                                    Z[i1 + n * (c_i - 1)] -= zeta * Z[i1 + n * (c_i - 2)];
                                }
                            }

                            for (i2 = 0; i2 < c_i; i2++) {
                                b_R[i2 + n * (c_i - 1)] = R_R[i2 + n * (c_i - 1)];
                                b_R[i2 + n * (c_i - 2)] = R_R[i2 + n * (c_i - 2)];
                            }

                            for (i2 = 0; i2 < c_i; i2++) {
                                R_R[i2 + n * (c_i - 2)] = b_R[i2 + n * (c_i - 1)];
                                R_R[i2 + n * (c_i - 1)] = b_R[i2 + n * (c_i - 2)];
                            }

                            for (i2 = 0; i2 < n; i2++) {
                                b_R[i2 + n * (c_i - 1)] = Z[i2 + n * (c_i - 1)];
                                b_R[i2 + n * (c_i - 2)] = Z[i2 + n * (c_i - 2)];
                            }
                            for (i2 = 0; i2 < n; i2++) {
                                Z[i2 + n * (c_i - 2)] = b_R[i2 + n * (c_i - 1)];
                                Z[i2 + n * (c_i - 1)] = b_R[i2 + n * (c_i - 2)];
                            }
                        }
                    }

#pragma omp for schedule(static, 1)
                    for (index b_i = 0; b_i < odd; b_i++) {
                        c_i = static_cast<int>((b_i << 1) + 3U);
                        // 'eo_sils_reduction:69' i1 = i-1;
                        // 'eo_sils_reduction:70' if swap(i) == 1
                        if (swap[c_i - 1]) {
                            // 'eo_sils_reduction:71' [G,R_R([i1,i],i1)] = planerot(R_R([i1,i],i1));
                            scalar G[4] = {};
                            scalar low_tmp[2] = {R_R[(c_i + n * (c_i - 2)) - 2], R_R[(c_i + n * (c_i - 2)) - 1]};
                            helper::planerot<scalar, index>(low_tmp, G);
                            R_R[(c_i + n * (c_i - 2)) - 2] = low_tmp[0];
                            R_R[(c_i + n * (c_i - 2)) - 1] = low_tmp[1];
                            // 'eo_sils_reduction:72' R_R([i1,i],i:n) = G * R_R([i1,i],i:n);
                            if (c_i > n) {
                                i1 = i2 = 0;
                                c_tmp = 1;
                            } else {
                                i1 = c_i - 1;
                                i2 = n;
                                c_tmp = c_i;
                            }
                            ci2 = i2 - i1;
                            scalar b[ci2 * 2] = {};
                            for (i2 = 0; i2 < ci2; i2++) {
                                tmp = i1 + i2;
                                b[2 * i2] = R_R[(c_i + n * tmp) - 2];
                                b[2 * i2 + 1] = R_R[(c_i + n * tmp) - 1];
                            }

                            for (i2 = 0; i2 < ci2; i2++) {
                                tmp = i1 + i2;
                                R_R[(c_i + n * tmp) - 2] = G[0] * b[2 * i2] + G[2] * b[2 * i2 + 1];
                                R_R[(c_i + n * tmp) - 1] = G[1] * b[2 * i2] + G[3] * b[2 * i2 + 1];
                            }
                            // 'eo_sils_reduction:73' y_L([i1,i]) = G * y_L([i1,i]);
                            low_tmp[0] = y_r[c_i - 2];
                            low_tmp[1] = y_r[c_i - 1];
                            y_r[c_i - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                            y_r[c_i - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
                            // 'eo_sils_reduction:74' swap(i) = 0;
                            swap[c_i - 1] = 0;
                        }
                    }


#pragma omp for schedule(static, 1)
                    for (index i = 0; i < n / 2; i++) {
                        c_i = static_cast<int>((i << 1) + 2U);
                        zeta = round(R_R[(c_i + n * (c_i - 1)) - 2] / R_R[(c_i + n * (c_i - 2)) - 2]);
                        s = R_R[(c_i + n * (c_i - 2)) - 2];
                        alpha = R_R[(c_i + n * (c_i - 1)) - 2] - zeta * s;
                        r_ii = R_R[(c_i + n * (c_i - 1)) - 1];
                        if (s * s > 2.0 * (alpha * alpha + r_ii * r_ii)) {
                            f = true;
                            swap[c_i - 1] = 1;
                            if (zeta != 0.0) {
                                R_R[(c_i + n * (c_i - 1)) - 2] = alpha;
                                if (1 <= c_i - 2) {
                                    ci2 = c_i - 2;
                                    for (i1 = 0; i1 < ci2; i1++) {
                                        R_R[i1 + n * (c_i - 1)] -= zeta * R_R[i1 + n * (ci2)];
                                    }
                                }
                                for (i1 = 0; i1 < n; i1++) {
                                    Z[i1 + n * (c_i - 1)] -= zeta * Z[i1 + n * (c_i - 2)];
                                }
                            }

                            for (i2 = 0; i2 < c_i; i2++) {
                                b_R[i2 + n * (c_i - 1)] = R_R[i2 + n * (c_i - 1)];
                                b_R[i2 + n * (c_i - 2)] = R_R[i2 + n * (c_i - 2)];
                            }

                            for (i2 = 0; i2 < c_i; i2++) {
                                R_R[i2 + n * (c_i - 2)] = b_R[i2 + n * (c_i - 1)];
                                R_R[i2 + n * (c_i - 1)] = b_R[i2 + n * (c_i - 2)];
                            }

                            for (i2 = 0; i2 < n; i2++) {
                                b_R[i2 + n * (c_i - 1)] = Z[i2 + n * (c_i - 1)];
                                b_R[i2 + n * (c_i - 2)] = Z[i2 + n * (c_i - 2)];
                            }

                            for (i2 = 0; i2 < n; i2++) {
                                Z[i2 + n * (c_i - 2)] = b_R[i2 + n * (c_i - 1)];
                                Z[i2 + n * (c_i - 1)] = b_R[i2 + n * (c_i - 2)];
                            }
                        }
                    }

#pragma omp for schedule(static, 1)
                    for (index i = 0; i < n / 2; i++) {
                        c_i = static_cast<int>((i << 1) + 2U);
                        // 'eo_sils_reduction:69' i1 = i-1;
                        // 'eo_sils_reduction:70' if swap(i) == 1
                        if (swap[c_i - 1]) {
                            // 'eo_sils_reduction:71' [G,R_R([i1,i],i1)] = planerot(R_R([i1,i],i1));
                            scalar G[4] = {};
                            scalar low_tmp[2] = {R_R[(c_i + n * (c_i - 2)) - 2], R_R[(c_i + n * (c_i - 2)) - 1]};
                            helper::planerot<scalar, index>(low_tmp, G);
                            R_R[(c_i + n * (c_i - 2)) - 2] = low_tmp[0];
                            R_R[(c_i + n * (c_i - 2)) - 1] = low_tmp[1];

                            // 'eo_sils_reduction:72' R_R([i1,i],i:n) = G * R_R([i1,i],i:n);
                            if (c_i > n) {
                                i1 = i2 = 0;
                                c_tmp = 1;
                            } else {
                                i1 = c_i - 1;
                                i2 = n;
                                c_tmp = c_i;
                            }
                            ci2 = i2 - i1;
                            scalar b[ci2 * 2] = {};
                            for (i2 = 0; i2 < ci2; i2++) {
                                tmp = i1 + i2;
                                b[2 * i2] = R_R[(c_i + n * tmp) - 2];
                                b[2 * i2 + 1] = R_R[(c_i + n * tmp) - 1];
                            }
//                            scalar r1[ci2 * 2] = {};
//                            for (index j = 0; j < ci2; j++) {
//                                index coffset_tmp = j << 1;
//                                r1[coffset_tmp] = G[0] * b[coffset_tmp] + G[2] * b[coffset_tmp + 1];
//                                r1[coffset_tmp + 1] = G[1] * b[coffset_tmp] + G[3] * b[coffset_tmp + 1];
//                            }
//
////                        b_loop_ub = r1.size(1);
//                            for (i1 = 0; i1 < ci2; i1++) {
//                                tmp = (c_tmp + i1) - 1;
//                                R_R[(c_i + n * tmp) - 2] = r1[2 * i1];
//                                R_R[(c_i + n * tmp) - 1] = r1[2 * i1 + 1];
//                            }
                            for (i2 = 0; i2 < ci2; i2++) {
                                tmp = i1 + i2;
                                R_R[(c_i + n * tmp) - 2] = G[0] * b[2 * i2] + G[2] * b[2 * i2 + 1];
                                R_R[(c_i + n * tmp) - 1] = G[1] * b[2 * i2] + G[3] * b[2 * i2 + 1];
                            }
                            // 'eo_sils_reduction:73' y_L([i1,i]) = G * y_L([i1,i]);
                            low_tmp[0] = y_r[c_i - 2];
                            low_tmp[1] = y_r[c_i - 1];
                            y_r[c_i - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                            y_r[c_i - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
                            // 'eo_sils_reduction:74' swap(i) = 0;
                            swap[c_i - 1] = 0;
                        }
                    }

                    counter++;
                }
            }
            time = omp_get_wtime() - time;
            error = qr_validation(R_Q, 1, n <= 16);
            printf("[ NEW OMP METHOD, QR ERROR: %.5f]\n", error);
            for (index i = 0; i < n; i++) {
                omp_destroy_lock(&lock[i]);
            }
            delete[] lock;
            delete[] b_R;
//        delete[] A_t;
            return time;
        }


        returnType <scalar, index>
        cils_qr_py(const index eval, const index qr_eval) {

            scalar error, time = omp_get_wtime();
//        cils_qr_py_helper();
            time = omp_get_wtime() - time;

            if (eval || qr_eval) {
//            error = qr_validation<scalar, index, m,  n>(A, Q, R_Q, R_A, eval, qr_eval);
            }

            return {{}, time, (index) error};
        }


        long int cils_qr_py_helper() {
            PyObject *pName, *pModule, *pFunc;
            PyObject *pArgs, *pValue, *pVec;
            Py_Initialize();
            if (_import_array() < 0)
                PyErr_Print();

            npy_intp dim[1] = {m};

            pVec = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, A.data());
            if (pVec == NULL) printf("There is a problem.\n");

            PyObject *sys_path = PySys_GetObject("path");
            PyList_Append(sys_path,
                          PyUnicode_FromString("/home/shilei/CLionProjects/babai_asyn/babai_asyn_c++/src/example"));
            pName = PyUnicode_FromString("py_qr");
            pModule = PyImport_Import(pName);

            if (pModule != NULL) {
                pFunc = PyObject_GetAttrString(pModule, "qr");
                if (pFunc && PyCallable_Check(pFunc)) {
                    pArgs = PyTuple_New(2);
                    if (PyTuple_SetItem(pArgs, 0, pVec) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", n)) != 0) {
                        return false;
                    }
                    pValue = PyObject_CallObject(pFunc, pArgs);//Perform QR no return value
                    if (pValue != NULL) {
                        PyArrayObject *q, *r;
                        PyArg_ParseTuple(pValue, "O|O", &q, &r);
                        Q = reinterpret_cast<scalar *>(PyArray_DATA(q));
                        R_R = reinterpret_cast<scalar *>(PyArray_DATA(r));
                    } else {
                        PyErr_Print();
                    }
                } else {
                    if (PyErr_Occurred())
                        PyErr_Print();
                    fprintf(stderr, "Cannot find function qr\n");
                }
            } else {
                PyErr_Print();
                fprintf(stderr, "Failed to load file\n");
            }
            return 0;
        }
    };
}

/*
 * Backup
 *
    scalar cils<scalar, index, m,  n>::cils_LLL_serial() {
        bool f = true;
        helper::array<scalar, 1U> r, vi, vr;
        helper::array<double, 2U> b, b_R, r1;
        scalar G[4], low_tmp[2], zeta;
        index c_result[2], sizes[2], b_loop_ub, i, i1, input_sizes_idx_1 = n, loop_ub, result;

        vi.set_size(n);
        for (i = 0; i < n; i++) {
            vi[i] = 0.0;
        }

        scalar time = omp_get_wtime();
        while (f) {
            scalar a, alpha, s;
            index b_i, i2;
            unsigned int c_i;
            // 'eo_sils_reduction:48' f = false;
            f = false;
            // 'eo_sils_reduction:49' for i = 2:2:n
            i = n / 2;
            for (b_i = 0; b_i < i; b_i++) {
                c_i = (b_i << 1) + 2U;
                // 'eo_sils_reduction:50' i1 = i-1;
                // 'eo_sils_reduction:51' zeta = round(R_R(i1,i) / R_R(i1,i1));
                zeta = std::round(R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] /
                                  R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2]);
                // 'eo_sils_reduction:52' alpha = R_R(i1,i) - zeta * R_R(i1,i1);
                s = R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2];
                alpha = R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] - zeta * s;
                // 'eo_sils_reduction:53' if R_R(i1,i1)^2 > (1 + 1.e-0) * (alpha^2 +
                // R_R(i,i)^2)
                a = R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 1];
                if ((s * s > 2.0 * (alpha * alpha + a * a)) && (zeta != 0.0)) {
                    // 'eo_sils_reduction:54' if zeta ~= 0
                    //  Perform a size reduction on R_R(k-1,k)
                    // 'eo_sils_reduction:56' f = true;
                    f = true;
                    // 'eo_sils_reduction:57' swap(i) = 1;
                    vi[static_cast<int>(c_i) - 1] = 1.0;
                    // 'eo_sils_reduction:58' R_R(i1,i) = alpha;
                    R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] = alpha;
                    // 'eo_sils_reduction:59' R_R(1:i-2,i) = R_R(1:i-2,i) - zeta * R_R(1:i-2,i-1);
                    if (1 > static_cast<int>(c_i) - 2) {
                        b_loop_ub = 0;
                    } else {
                        b_loop_ub = static_cast<int>(c_i) - 2;
                    }
                    vr.set_size(b_loop_ub);
                    for (i1 = 0; i1 < b_loop_ub; i1++) {
                        vr[i1] = R_R[i1 + n * (static_cast<int>(c_i) - 1)] -
                                 zeta * R_R[i1 + n * (static_cast<int>(c_i) - 2)];
                    }
                    b_loop_ub = vn;
                    for (i1 = 0; i1 < b_loop_ub; i1++) {
                        R_R[i1 + n * (static_cast<int>(c_i) - 1)] = vr[i1];
                    }
                    // 'eo_sils_reduction:60' Z(:,i) = Z(:,i) - zeta * Z(:,i-1);
                    input_sizes_idx_1 = n - 1;
                    vr.set_size(n);
                    for (i1 = 0; i1 <= input_sizes_idx_1; i1++) {
                        vr[i1] = Z[i1 + n * (static_cast<int>(c_i) - 1)] -
                                 zeta * Z[i1 + n * (static_cast<int>(c_i) - 2)];
                    }
                    b_loop_ub = vn;
                    for (i1 = 0; i1 < b_loop_ub; i1++) {
                        Z[i1 + n * (static_cast<int>(c_i) - 1)] = vr[i1];
                    }
                    //  Permute columns k-1 and k of R_R and Z
                    // 'eo_sils_reduction:63' R_R(1:i,[i1,i]) = R_R(1:i,[i,i1]);
                    sizes[0] = static_cast<int>(c_i) - 2;
                    sizes[1] = static_cast<int>(c_i) - 1;
                    c_result[0] = static_cast<int>(c_i) - 1;
                    c_result[1] = static_cast<int>(c_i) - 2;
                    b_loop_ub = static_cast<int>(c_i);
                    b_R.set_size(static_cast<int>(c_i), 2);
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 < b_loop_ub; i2++) { b_R[i2 + b_n * i1] = R_R[i2 + n * c_result[i1]]; }
                    }
                    b_loop_ub = b_n;
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 < b_loop_ub; i2++) { R_R[i2 + n * sizes[i1]] = b_R[i2 + b_n * i1]; }
                    }
                    // 'eo_sils_reduction:64' Z(:,[i1,i]) = Z(:,[i,i1]);
                    sizes[0] = static_cast<int>(c_i) - 2;
                    sizes[1] = static_cast<int>(c_i) - 1;
                    input_sizes_idx_1 = n - 1;
                    c_result[0] = static_cast<int>(c_i) - 1;
                    c_result[1] = static_cast<int>(c_i) - 2;
                    b_R.set_size(n, 2);
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 <= input_sizes_idx_1; i2++) {
                            b_R[i2 + b_n * i1] = Z[i2 + n * c_result[i1]];
                        }
                    }
                    b_loop_ub = b_n;
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 < b_loop_ub; i2++) {
                            Z[i2 + n * sizes[i1]] = b_R[i2 + b_n * i1];
                        }
                    }
                }
            }
            // 'eo_sils_reduction:68' for i = 2:2:n
            for (b_i = 0; b_i < i; b_i++) {
                c_i = (b_i << 1) + 2U;
                // 'eo_sils_reduction:69' i1 = i-1;
                // 'eo_sils_reduction:70' if swap(i) == 1
                if (vi[static_cast<int>(c_i) - 1] == 1.0) {
                    // 'eo_sils_reduction:71' [G,R_R([i1,i],i1)] = planerot(R_R([i1,i],i1));
                    low_tmp[0] = R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2];
                    low_tmp[1] = R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 1];
                    helper::planerot<scalar, index>(low_tmp, G);
                    R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2] = low_tmp[0];
                    R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 1] = low_tmp[1];
                    // 'eo_sils_reduction:72' R_R([i1,i],i:n) = G * R_R([i1,i],i:n);
                    if (static_cast<int>(c_i) > static_cast<int>(n)) {
                        i1 = 0;
                        i2 = 0;
                        result = 1;
                    } else {
                        i1 = static_cast<int>(c_i) - 1;
                        i2 = static_cast<int>(n);
                        result = static_cast<int>(c_i);
                    }
                    b_loop_ub = i2 - i1;
                    b.set_size(2, b_loop_ub);
                    for (i2 = 0; i2 < b_loop_ub; i2++) {
                        input_sizes_idx_1 = i1 + i2;
                        b[2 * i2] = R_R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 2];
                        b[2 * i2 + 1] = R_R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 1];
                    }
                    helper::internal::blas::mtimes(G, b, r1);
                    b_loop_ub = r1.size(1);
                    for (i1 = 0; i1 < b_loop_ub; i1++) {
                        input_sizes_idx_1 = (result + i1) - 1;
                        R_R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 2] = r1[2 * i1];
                        R_R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 1] = r1[2 * i1 + 1];
                    }
                    // 'eo_sils_reduction:73' y_L([i1,i]) = G * y_L([i1,i]);
                    low_tmp[0] = y_r[static_cast<int>(c_i) - 2];
                    low_tmp[1] = y_r[static_cast<int>(c_i) - 1];
                    y_r[static_cast<int>(c_i) - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                    y_r[static_cast<int>(c_i) - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
                    // 'eo_sils_reduction:74' swap(i) = 0;
                    vi[static_cast<int>(c_i) - 1] = 0.0;
                }
            }
            // 'eo_sils_reduction:77' for i = 3:2:n
            i = static_cast<int>((n + -1.0) / 2.0);
            for (b_i = 0; b_i < i; b_i++) {
                c_i = static_cast<unsigned int>((b_i << 1) + 3);
                // 'eo_sils_reduction:78' i1 = i-1;
                // 'eo_sils_reduction:79' zeta = round(R_R(i1,i) / R_R(i1,i1));
                zeta = std::round(R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] /
                                  R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2]);
                // 'eo_sils_reduction:80' alpha = R_R(i1,i) - zeta * R_R(i1,i1);
                input_sizes_idx_1 = static_cast<int>(c_i) - 2;
                s = R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2];
                alpha = R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] - zeta * s;
                // 'eo_sils_reduction:81' if R_R(i1,i1)^2 > (1 + 1.e-0) * (alpha^2 +
                // R_R(i,i)^2)
                a = R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 1];
                if ((s * s > 2.0 * (alpha * alpha + a * a)) && (zeta != 0.0)) {
                    // 'eo_sils_reduction:82' if zeta ~= 0
                    // 'eo_sils_reduction:83' f = true;
                    f = true;
                    // 'eo_sils_reduction:84' swap(i) = 1;
                    vi[static_cast<int>(c_i) - 1] = 1.0;
                    //  Perform a size reduction on R_R(k-1,k)
                    // 'eo_sils_reduction:86' R_R(i1,i) = alpha;
                    R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] = alpha;
                    // 'eo_sils_reduction:87' R_R(1:i-2,i) = R_R(1:i-2,i) - zeta * R_R(1:i-2,i-1);
                    vr.set_size(input_sizes_idx_1);
                    for (i1 = 0; i1 < input_sizes_idx_1; i1++) {
                        vr[i1] = R_R[i1 + n * (static_cast<int>(c_i) - 1)] -
                                 zeta * R_R[i1 + n * (static_cast<int>(c_i) - 2)];
                    }
                    b_loop_ub = vn;
                    for (i1 = 0; i1 < b_loop_ub; i1++) {
                        R_R[i1 + n * (static_cast<int>(c_i) - 1)] = vr[i1];
                    }
                    // 'eo_sils_reduction:88' Z(:,i) = Z(:,i) - zeta * Z(:,i-1);
                    input_sizes_idx_1 = n - 1;
                    vr.set_size(n);
                    for (i1 = 0; i1 <= input_sizes_idx_1; i1++) {
                        vr[i1] = Z[i1 + n * (static_cast<int>(c_i) - 1)] -
                                 zeta * Z[i1 + n * (static_cast<int>(c_i) - 2)];
                    }
                    b_loop_ub = vn;
                    for (i1 = 0; i1 < b_loop_ub; i1++) {
                        Z[i1 + n * (static_cast<int>(c_i) - 1)] = vr[i1];
                    }
                    //  Permute columns k-1 and k of R_R and Z
                    // 'eo_sils_reduction:91' R_R(1:i,[i1,i]) = R_R(1:i,[i,i1]);
                    sizes[0] = static_cast<int>(c_i) - 2;
                    sizes[1] = static_cast<int>(c_i) - 1;
                    c_result[0] = static_cast<int>(c_i) - 1;
                    c_result[1] = static_cast<int>(c_i) - 2;
                    b_loop_ub = static_cast<int>(c_i);
                    b_R.set_size(static_cast<int>(c_i), 2);
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 < b_loop_ub; i2++) { b_R[i2 + b_n * i1] = R_R[i2 + n * c_result[i1]]; }
                    }
                    b_loop_ub = b_n;
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 < b_loop_ub; i2++) { R_R[i2 + n * sizes[i1]] = b_R[i2 + b_n * i1]; }
                    }
                    // 'eo_sils_reduction:92' Z(:,[i1,i]) = Z(:,[i,i1]);
                    sizes[0] = static_cast<int>(c_i) - 2;
                    sizes[1] = static_cast<int>(c_i) - 1;
                    input_sizes_idx_1 = n - 1;
                    c_result[0] = static_cast<int>(c_i) - 1;
                    c_result[1] = static_cast<int>(c_i) - 2;
                    b_R.set_size(n, 2);
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 <= input_sizes_idx_1; i2++) {
                            b_R[i2 + b_n * i1] = Z[i2 + n * c_result[i1]];
                        }
                    }
                    b_loop_ub = b_n;
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 < b_loop_ub; i2++) {
                            Z[i2 + n * sizes[i1]] = b_R[i2 + b_n * i1];
                        }
                    }
                }
            }
            // 'eo_sils_reduction:96' for i = 3:2:n
            for (b_i = 0; b_i < i; b_i++) {
                c_i = static_cast<unsigned int>((b_i << 1) + 3);
                // 'eo_sils_reduction:97' i1 = i-1;
                // 'eo_sils_reduction:98' if swap(i) == 1
                if (vi[static_cast<int>(c_i) - 1] == 1.0) {
                    //  Bring R_R baci to an upper triangular matrix by a Givens rotation
                    // 'eo_sils_reduction:100' [G,R_R([i1,i],i1)] = planerot(R_R([i1,i],i1));
                    low_tmp[0] = R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2];
                    low_tmp[1] = R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 1];
                    helper::planerot<scalar, index>(low_tmp, G);
                    R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2] = low_tmp[0];
                    R_R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 1] = low_tmp[1];
                    // 'eo_sils_reduction:101' R_R([i1,i],i:n) = G * R_R([i1,i],i:n);
                    if (static_cast<int>(c_i) > static_cast<int>(n)) {
                        i1 = 0;
                        i2 = 0;
                        result = 1;
                    } else {
                        i1 = static_cast<int>(c_i) - 1;
                        i2 = static_cast<int>(n);
                        result = static_cast<int>(c_i);
                    }
                    b_loop_ub = i2 - i1;
                    b.set_size(2, b_loop_ub);
                    for (i2 = 0; i2 < b_loop_ub; i2++) {
                        input_sizes_idx_1 = i1 + i2;
                        b[2 * i2] = R_R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 2];
                        b[2 * i2 + 1] = R_R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 1];
                    }
                    helper::internal::blas::mtimes(G, b, r1);
                    b_loop_ub = r1.size(1);
                    for (i1 = 0; i1 < b_loop_ub; i1++) {
                        input_sizes_idx_1 = (result + i1) - 1;
                        R_R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 2] = r1[2 * i1];
                        R_R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 1] = r1[2 * i1 + 1];
                    }
                    //  Apply the Givens rotation to y_r
                    // 'eo_sils_reduction:104' y_r([i1,i]) = G * y_r([i1,i]);
                    low_tmp[0] = y_r[static_cast<int>(c_i) - 2];
                    low_tmp[1] = y_r[static_cast<int>(c_i) - 1];
                    y_r[static_cast<int>(c_i) - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
                    y_r[static_cast<int>(c_i) - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
                    // 'eo_sils_reduction:105' swap(i) = 0;
                    vi[static_cast<int>(c_i) - 1] = 0.0;
                }
            }
        }
        time = omp_get_wtime() - time;
        return time;
    }
 */