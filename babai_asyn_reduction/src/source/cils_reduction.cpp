#include <cstring>

#include "cils.cpp"

namespace cils {

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_qr_serial(const index eval, const index verbose) {
        cout << "[ In Serial QR]\n";

        index i, j, k, m;
        scalar error = -1, time, sum;
        //Deep Copy
//        scalar *A_t = new scalar[n * n];
        coder::array<scalar, 2U> A_t(A);
        //Clear Variables:
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                R_Q[i * n + j] = 0;
                Q[i * n + j] = 0;
//                A_t[i * n + j] = A[i * n + j];
            }
        }

        //start
        time = omp_get_wtime();
        for (k = 0; k < n; k++) {
            //Check if Q[][i-1] (the previous column) is computed.
            for (j = k; j < n; j++) {
                R_Q[k * n + j] = 0;
                for (i = 0; i < n; i++) {
                    R_Q[j * n + k] += Q[k * n + i] * A_t[j * n + i];
                }
                for (i = 0; i < n; i++) {
                    A_t[j * n + i] -= R_Q[j * n + k] * Q[k * n + i];
                }
                //Calculate the norm(A)
                if (j == k) {
                    sum = 0;
                    for (i = 0; i < n; i++) {
                        sum += pow(A_t[k * n + i], 2);
                    }
                    R_Q[k * n + k] = sqrt(sum);
                    for (i = 0; i < n; i++) {
                        Q[k * n + i] = A_t[k * n + i] / R_Q[k * n + k];
                    }
                }
            }
        }
        //COLUMN:
//        for (j = 0; j < n; j++) {
//            for (k = 0; k <= j; k++) {
//                //Check if Q[][i-1] (the previous column) is computed.
//                R_Q[k * n + j] = 0;
//                for (i = 0; i < n; i++) {
//                    R_Q[j * n + k] += Q[k * n + i] * A_t[j * n + i];
//                }
//                for (i = 0; i < n; i++) {
//                    A_t[j * n + i] -= R_Q[j * n + k] * Q[k * n + i];
//                }
//
//                //Calculate the norm(A)
//                if (j == k) {
//                    sum = 0;
//                    for (i = 0; i < n; i++) {
//                        sum += pow(A_t[k * n + i], 2);
//                    }
//                    R_Q[k * n + k] = sqrt(sum);
//                    for (i = 0; i < n; i++) {
//                        Q[k * n + i] = A_t[k * n + i] / R_Q[k * n + k];
//                    }
//                }
//            }
//        }

        time = omp_get_wtime() - time;
//        delete[] A_t;
        if (eval) {
            error = qr_validation<scalar, index, n>(A, Q, R_Q, eval, verbose);
        }
        return {{}, time, error};
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_qr_omp(const index eval, const index verbose, const index n_proc) {
#pragma omp parallel default(shared) num_threads(n_proc)
        {}
        cout << "[ In Parallel QR]\n";
        cout.flush();
        scalar error = -1, time, sum = 0;
        auto lock = new omp_lock_t[n]();
//        scalar *A_t = new scalar[n * n];
        coder::array<scalar, 2U> A_t(A);
        //Clear Variables:
        for (index i = 0; i < n; i++) {
            for (index j = 0; j < n; j++) {
                R_Q[i * n + j] = 0;
                Q[i * n + j] = 0;
//                A_t[i * n + j] = A[i * n + j];
            }
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
        }
        time = omp_get_wtime() - time;

        if (eval || verbose) {
            error = qr_validation<scalar, index, n>(A, Q, R_Q, eval, verbose);
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
            error = qr_validation<scalar, index, n>(A, Q, R_Q, eval, verbose);
        }
#pragma parallel omp cancellation point
#pragma omp flush
        delete[] A_t;

        cout << "[  QR ERROR OMP:]" << error << endl;
        return {{}, time, error};
         */
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_LLL_qr_reduction(const index eval, const index verbose, const index n_proc) {
        scalar time = 0, det = 0;
        returnType<scalar, index> reT, lll_val;

        if (n_proc <= 1) {
            reT = cils_LLL_qr_serial();
            printf("[ INFO: SER LLL QR TIME: %8.5f, Givens: %.1f]\n",
                   reT.run_time, reT.num_iter);
            time = reT.run_time;
        } else {
            time = cils_LLL_qr_omp(n_proc);
            printf("[ INFO: OMP LLL QR TIME: %8.5f]\n", time);
        }

        if (eval) {
            lll_val = lll_validation<scalar, index, n>(R_R, R_Q, Z, verbose);
            if (lll_val.num_iter != 1) {
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
                    lll_val = lll_validation<scalar, index, n>(R_R, R_Q, Z, verbose);
                    if (lll_val.num_iter == 1)
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

        return {{reT.num_iter, lll_val.num_iter}, time, lll_val.run_time};
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_LLL_reduction(const index eval, const index verbose, const index n_proc) {
        scalar time = 0, det = 0;
        returnType<scalar, index> reT, lll_val;

        coder::eye(n, Z);

        if (n_proc <= 1) {
            reT = cils_LLL_serial();
            printf("[ INFO: SER LLL TIME: %8.5f, Givens: %.1f]\n",
                   reT.run_time, reT.num_iter);
            time = reT.run_time;
        } else {
            time = cils_LLL_omp(n_proc);
            printf("[ INFO: OMP LLL TIME: %8.5f]\n", time);
        }

        if (eval) {
            lll_val = lll_validation<scalar, index, n>(R_R, R_Q, Z, verbose);
            if (lll_val.num_iter != 1) {
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
                    lll_val = lll_validation<scalar, index, n>(R_R, R_Q, Z, verbose);
                    if (lll_val.num_iter == 1)
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

        return {{reT.num_iter, lll_val.num_iter}, time, lll_val.run_time};
    }


    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_LLL_serial() {
        cout << "[ In cils_LLL_serial]" << endl;
        bool f = true, givens = false;
        scalar zeta, r_ii, alpha, s;

        index swap[n] = {}, counter = 0, c_i;
        index i1, ci2, c_tmp, tmp, i2, odd = static_cast<int>((n + -1.0) / 2.0);
        scalar b_R[n * n] = {};

        for (index i = 0; i < n; i++) {
            for (index j = 0; j < n; j++) {
                R_R[i * n + j] = R_Q[i * n + j];
            }
        }

        scalar time = omp_get_wtime();
        while (f) {
            f = false;

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

            for (index i = 0; i < n / 2; i++) {
                c_i = static_cast<int>((i << 1) + 2U);
                // 'eo_sils_reduction:69' i1 = i-1;
                // 'eo_sils_reduction:70' if swap(i) == 1
                if (swap[c_i - 1]) {
                    givens = true;
                    // 'eo_sils_reduction:71' [G,R_R([i1,i],i1)] = planerot(R_R([i1,i],i1));
                    scalar G[4] = {};
                    scalar low_tmp[2] = {R_R[(c_i + n * (c_i - 2)) - 2], R_R[(c_i + n * (c_i - 2)) - 1]};
                    coder::planerot(low_tmp, G);
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

            for (index b_i = 0; b_i < odd; b_i++) {
                c_i = static_cast<int>((b_i << 1) + 3U);
                // 'eo_sils_reduction:69' i1 = i-1;
                // 'eo_sils_reduction:70' if swap(i) == 1
                if (swap[c_i - 1]) {
                    // 'eo_sils_reduction:71' [G,R_R([i1,i],i1)] = planerot(R_R([i1,i],i1));
                    scalar G[4] = {};
                    scalar low_tmp[2] = {R_R[(c_i + n * (c_i - 2)) - 2], R_R[(c_i + n * (c_i - 2)) - 1]};
                    coder::planerot(low_tmp, G);
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

        }
        time = omp_get_wtime() - time;

        return {{}, time, (scalar) givens};
    }


    template<typename scalar, typename index, index n>
    scalar cils<scalar, index, n>::cils_LLL_omp(const index n_proc) {
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
                        coder::planerot(low_tmp, G);
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
                        coder::planerot(low_tmp, G);
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


    template<typename scalar, typename index, index n>
    returnType <scalar, index> cils<scalar, index, n>::cils_LLL_qr_serial() {

        cout << "[ In cils_LLL_qr_serial]\n";
        scalar zeta, r_ii, alpha, s;
        bool f = true;
        index swap[n] = {}, givens = 0, c_i;
        index i1, ci2, c_tmp, tmp, i2, odd = static_cast<int>((n + -1.0) / 2.0);
        scalar error = -1, time, sum = 0;
//        scalar *A_t = new scalar[n * n];
        coder::array<scalar, 2U> A_t(A);
        //Clear Variables:
//        b_R.set_size(n, n);
        for (index i = 0; i < n; i++) {
            for (index j = 0; j < n; j++) {
                Q[i * n + j] = 0;
                R_R[i * n + j] = 0;
//                R_Q[i * n + j] = 0;
//                A_t[i * n + j] = A[i * n + j];
            }
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
                    sum = 0;
                    for (index i = 0; i < n; i++) {
                        sum += pow(A_t[k * n + i], 2);
                    }
                    R_R[k * n + k] = sqrt(sum);
                    for (index i = 0; i < n; i++) {
                        Q[k * n + i] = A_t[k * n + i] / R_R[k * n + k];
                    }
                }
            }
            if (k % 2 == 0) {
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
                    coder::planerot(low_tmp, G);
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
                    coder::planerot(low_tmp, G);
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
                    coder::planerot(low_tmp, G);
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
        error = qr_validation<scalar, index, n>(A, Q, R_Q, 1, n <= 16);
        printf("[ NEW METHOD, QR ERROR: %.5f]\n", error);
//        delete[] A_t;
        return {{}, time, (scalar) givens};
    }


    template<typename scalar, typename index, index n>
    scalar cils<scalar, index, n>::cils_LLL_qr_omp(const index n_proc) {
#pragma omp parallel default(shared) num_threads(n_proc)
        {}

        cout << "[ In cils_LLL_qr_omp]\n";
        cout.flush();

        scalar zeta, r_ii, alpha, s;
        bool f = true, condition = false;
        index swap[n] = {}, counter = 0, c_i;
        index i1, ci2, c_tmp, tmp, i2, odd = static_cast<int>((n + -1.0) / 2.0);
        auto lock = new omp_lock_t[n]();
        scalar error = -1, time, sum = 0, tmp_sum = 0;
        scalar *b_R = new scalar[n * n];
//        scalar *A_t = new scalar[n * n];
        coder::array<scalar, 2U> A_t(A);
        //Clear Variables:
//        b_R.set_size(n, n);
        for (index i = 0; i < n; i++) {
            for (index j = 0; j < n; j++) {
                Q[i * n + j] = 0;
                R_R[i * n + j] = 0;
                b_R[i * n + j] = 0;
//                R_Q[i * n + j] = 0;
//                A_t[i * n + j] = A[i * n + j];
            }
            omp_set_lock(&lock[i]);
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
                            sum = 0;
                            for (index i = 0; i < n; i++) {
                                sum += pow(A_t[k * n + i], 2);
                            }
//                            R_Q[k * n + k] = sqrt(sum);
                            R_R[k * n + k] = sqrt(sum);
                            for (index i = 0; i < n; i++) {
                                Q[k * n + i] = A_t[k * n + i] / R_R[k * n + k];
//                                Q[k * n + i] = A_t[k * n + i] / R_Q[k * n + k];
                            }
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
                                coder::planerot(low_tmp, G);
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
                        coder::planerot(low_tmp, G);
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
                        coder::planerot(low_tmp, G);
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
        error = qr_validation<scalar, index, n>(A, Q, R_Q, 1, n <= 16);
        printf("[ NEW OMP METHOD, QR ERROR: %.5f]\n", error);
        for (index i = 0; i < n; i++) {
            omp_destroy_lock(&lock[i]);
        }
        delete[] lock;
        delete[] b_R;
//        delete[] A_t;
        return time;
    }


//    template<typename scalar, typename index, index n>
//    returnType <scalar, index>
//    cils<scalar, index, n>::cils_qr_py(const index eval, const index qr_eval) {
//
//        scalar error, time = omp_get_wtime();
////        cils_qr_py_helper();
//        time = omp_get_wtime() - time;
//
//        if (eval || qr_eval) {
////            error = qr_validation<scalar, index, n>(A, Q, R_Q, R_A, eval, qr_eval);
//        }
//
//        return {{}, time, (index) error};
//    }
//
//    template<typename scalar, typename index, index n>
//    long int cils<scalar, index, n>::cils_qr_py_helper() {
//        PyObject *pName, *pModule, *pFunc;
//        PyObject *pArgs, *pValue, *pVec;
//        Py_Initialize();
//        if (_import_array() < 0)
//            PyErr_Print();
//
//        npy_intp dim[1] = {A.size(0)};
//
//        pVec = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, A.data());
//        if (pVec == NULL) printf("There is a problem.\n");
//
//        PyObject *sys_path = PySys_GetObject("path");
//        PyList_Append(sys_path,
//                      PyUnicode_FromString("/home/shilei/CLionProjects/babai_asyn/babai_asyn_c++/src/example"));
//        pName = PyUnicode_FromString("py_qr");
//        pModule = PyImport_Import(pName);
//
//        if (pModule != NULL) {
//            pFunc = PyObject_GetAttrString(pModule, "qr");
//            if (pFunc && PyCallable_Check(pFunc)) {
//                pArgs = PyTuple_New(2);
//                if (PyTuple_SetItem(pArgs, 0, pVec) != 0) {
//                    return false;
//                }
//                if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", n)) != 0) {
//                    return false;
//                }
//                pValue = PyObject_CallObject(pFunc, pArgs);//Perform QR no return value
//                if (pValue != NULL) {
//                    PyArrayObject *q, *r;
//                    PyArg_ParseTuple(pValue, "O|O", &q, &r);
//                    Q = reinterpret_cast<scalar *>(PyArray_DATA(q));
//                    R_R = reinterpret_cast<scalar *>(PyArray_DATA(r));
//                } else {
//                    PyErr_Print();
//                }
//            } else {
//                if (PyErr_Occurred())
//                    PyErr_Print();
//                fprintf(stderr, "Cannot find function qr\n");
//            }
//        } else {
//            PyErr_Print();
//            fprintf(stderr, "Failed to load file\n");
//        }
//        return 0;
//    }


    template<typename scalar, typename index, index n>
    void value_input_helper(matlab::data::TypedArray<scalar> const x, coder::array<scalar, 1U> &arr) {
        index i = 0;
        arr.set_size(n);
        for (auto r : x) {
            arr[i] = r;
            ++i;
        }
    }

    template<typename scalar, typename index, index n>
    void value_input_helper(matlab::data::TypedArray<scalar> const x, coder::array<scalar, 2U> &arr) {
        index i = 0;
        arr.set_size(n, n);
        for (auto r : x) {
            arr[i] = r;
            ++i;
        }
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_qr_matlab() {

        scalar time = omp_get_wtime();
        scalar true_res = cils_qr_matlab_helper();
        time = omp_get_wtime() - time;

        return {{}, time, true_res};
    }

    template<typename scalar, typename index, index n>
    scalar cils<scalar, index, n>::cils_qr_matlab_helper() {

        // Start MATLAB engine synchronously
        using namespace matlab::engine;

        // Start MATLAB engine synchronously
        std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();

        //Create MATLAB data array factory
        matlab::data::ArrayFactory factory;

        // Create variables
        matlab::data::TypedArray<scalar> k_M = factory.createScalar<scalar>(program_def::k);
        matlab::data::TypedArray<scalar> SNR_M = factory.createScalar<scalar>(program_def::SNR);
        matlab::data::TypedArray<scalar> m_M = factory.createScalar<scalar>(n);
        matlab::data::TypedArray<scalar> qr_M = factory.createScalar<scalar>(program_def::is_qr);
        matlabPtr->setVariable(u"k", std::move(k_M));
        matlabPtr->setVariable(u"n", std::move(m_M));
        matlabPtr->setVariable(u"SNR", std::move(SNR_M));
        matlabPtr->setVariable(u"qr", std::move(qr_M));

        // Call the MATLAB movsum function
        matlabPtr->eval(u" [A, R_R, Z, y, y_LLL, x_t, init_res, info] = sils_driver_mex(k, n, SNR, qr);");

        // Get the result
        matlab::data::TypedArray<scalar> const A_A = matlabPtr->getVariable(u"A");
        matlab::data::TypedArray<scalar> const R_N = matlabPtr->getVariable(u"R_R");
        matlab::data::TypedArray<scalar> const Z_N = matlabPtr->getVariable(u"Z");
        matlab::data::TypedArray<scalar> const y_R = matlabPtr->getVariable(u"y");//Reduced
        matlab::data::TypedArray<scalar> const y_N = matlabPtr->getVariable(u"y_LLL");//Original
        matlab::data::TypedArray<scalar> const x_T = matlabPtr->getVariable(u"x_t");
        matlab::data::TypedArray<scalar> const res = matlabPtr->getVariable(u"init_res");
        matlab::data::TypedArray<scalar> const b_n = matlabPtr->getVariable(u"info");
        matlab::data::ArrayDimensions dim = A_A.getDimensions();

        value_input_helper<scalar, index, n>(A_A, A);
        value_input_helper<scalar, index, n>(R_N, R_Q);
        value_input_helper<scalar, index, n>(R_N, R_R);
        value_input_helper<scalar, index, n>(Z_N, Z);
        value_input_helper<scalar, index, n>(y_N, y_a);//Original
        value_input_helper<scalar, index, n>(y_R, y_r);//Reduced
        value_input_helper<scalar, index, n>(y_R, y_q);//Reduced
        value_input_helper<scalar, index, n>(x_T, x_t);

        for (auto r : res) {
            init_res = r;
        }

        scalar info[3];
        index _i = 0;
        for (auto r : b_n) {
            info[_i] = r;
            _i++;
        }
        printf("----------------------\n"
               "MATLAB RESULT: QR/LLL Time: %.5f, Babai Time: %.5f, Babai Res: %.5f.\n"
               "----------------------\n", info[0], info[1], info[2]);
        return init_res;
    }

}

/*
 * Backup
 *  template<typename scalar, typename index, index n>
    scalar cils<scalar, index, n>::cils_LLL_serial() {
        bool f = true;
        coder::array<scalar, 1U> r, vi, vr;
        coder::array<double, 2U> b, b_R, r1;
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
                    b_loop_ub = vr.size(0);
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
                    b_loop_ub = vr.size(0);
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
                        for (i2 = 0; i2 < b_loop_ub; i2++) { b_R[i2 + b_R.size(0) * i1] = R_R[i2 + n * c_result[i1]]; }
                    }
                    b_loop_ub = b_R.size(0);
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 < b_loop_ub; i2++) { R_R[i2 + n * sizes[i1]] = b_R[i2 + b_R.size(0) * i1]; }
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
                            b_R[i2 + b_R.size(0) * i1] = Z[i2 + n * c_result[i1]];
                        }
                    }
                    b_loop_ub = b_R.size(0);
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 < b_loop_ub; i2++) {
                            Z[i2 + n * sizes[i1]] = b_R[i2 + b_R.size(0) * i1];
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
                    coder::planerot(low_tmp, G);
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
                    coder::internal::blas::mtimes(G, b, r1);
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
                    b_loop_ub = vr.size(0);
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
                    b_loop_ub = vr.size(0);
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
                        for (i2 = 0; i2 < b_loop_ub; i2++) { b_R[i2 + b_R.size(0) * i1] = R_R[i2 + n * c_result[i1]]; }
                    }
                    b_loop_ub = b_R.size(0);
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 < b_loop_ub; i2++) { R_R[i2 + n * sizes[i1]] = b_R[i2 + b_R.size(0) * i1]; }
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
                            b_R[i2 + b_R.size(0) * i1] = Z[i2 + n * c_result[i1]];
                        }
                    }
                    b_loop_ub = b_R.size(0);
                    for (i1 = 0; i1 < 2; i1++) {
                        for (i2 = 0; i2 < b_loop_ub; i2++) {
                            Z[i2 + n * sizes[i1]] = b_R[i2 + b_R.size(0) * i1];
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
                    coder::planerot(low_tmp, G);
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
                    coder::internal::blas::mtimes(G, b, r1);
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