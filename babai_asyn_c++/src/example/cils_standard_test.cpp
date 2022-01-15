
#include "../source/CILS.cpp"
#include "../source/CILS_SECH_Search.cpp"
#include "../source/CILS_Block_Babai.cpp"
#include "../source/CILS_Babai.cpp"
#include "../source/CILS_Reduction.cpp"
#include "../source/CILS_Init_Point.cpp"
#include <ctime>

template<typename scalar, typename index, index m, index n>
long plot_run() {
    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | ILS | %s ]==================================\n", time_str);
    for (index SNR = 35; SNR <= 35; SNR += 30) {
        scalar res[3][200][2] = {}, ber[3][200][2] = {}, tim[4][200][2] = {}, spu[4][200][2] = {}, t_spu[4][200][2] = {};
        scalar itr[3][200][2] = {}, stm[3][200][2] = {};
        scalar all_time[7][1000][3][2] = {}; //Method, iter, init, qam

        index verbose = n <= 16, count = 0, qr_l = 2;
        scalar run_time;
        cils::returnType<scalar, index> reT, qr_reT, LLL_reT, LLL_reT_omp, qr_reT_omp, LLL_qr_reT, LLL_qr_reT_omp;

        scalar r, t, b, iter, ser_qrd, ser_time;
        scalar t2, r2, b2, iter2;

        index l = 2; //count for qam.

        b_vector z_B(n, 0);
//        for (k = 1; k <= 3; k += 2) {
        scalar qr_spu[200][2] = {}, lll_spu[200][2] = {}, lll_qr_spu[200][2] = {}, qlll_spu[200][2] = {};
        scalar qrT[200][2] = {}, lll[200][2] = {}, LLL_qr[200][2] = {};
        cils::CILS<scalar, index> cils(k, SNR);
        for (index tt = 0; tt <= 0; tt++) {
            is_qr = tt;
            for (index i = 1; i <= max_iter; i++) {
                ser_qrd = 0;
                l = 2;
                for (k = 3; k >= 1; k -= 2) {
                    count = k == 1 ? 0 : 1;
                    run_time = omp_get_wtime();
                    cils.qam = k;
                    scalar ber_babai, ber_thre3, ber_serial, res_lll, res_qr, ber_qr;
                    printf("[ INIT PHASE]\n++++++++++++++++++++++++++++++++++++++\n");
                    do {
                        //Initialize Problem
                        //cils.init();
                        if (is_matlab)
                            qr_reT = cils.cils_qr_matlab();
                        else {
                            //Matlab gen
                            cils.cils_qr_matlab();
                            //QR Decomposition
//                            qr_reT_omp = cils.cils_qr_omp(1, verbose, 2);
                            qr_reT = cils.cils_qr_serial(1, verbose);

                            //QR - Block Solver Testing
                            coder::eye<scalar, index, m, n>(n, cils.Z);
                            for (index ii = 0; ii < n; ii++) {
                                for (index j = 0; j < n; j++) {
                                    cils.R_R[j * n + ii] = cils.R_Q[j * n + ii];
                                }
                            }

                            init_guess<scalar, index, m, n>(0, &z_B, cils.x_r.data());
                            cils.cils_block_search_serial(0, &d_s, &z_B);
                            ber_qr = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);
                            res_qr = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, z_B.data());
                            if (verbose) {
                                cout << "[ SER:";
                                cils::display_vector<scalar, index>(&z_B);
                                cout << "]\n";
                            }
                            //LLL reduction
                            if (!is_qr) {
//                                LLL_reT = cils.cils_LLL_reduction(1, verbose, 1);
//                                LLL_reT_omp = cils.cils_LLL_reduction(1, verbose, 2);

//                                LLL_qr_reT_omp = cils.cils_LLL_qr_reduction(1, verbose, 2);
                                LLL_qr_reT = cils.cils_LLL_qr_reduction(1, verbose, 1);

//                                printf("[ LLL INITIALIZATION INFO]\n"
//                                       "01.The QR SER Error is %.5f, the QR SER Time is %.5f.\n"
//                                       "02.The QR OMP Error is %.5f, the QR OMP Time is %.5f.\n"
//                                       "03.The LLL DET SER is %.5f, the LLL SER Time is %.5f.\n"
//                                       "04.The LLL DET OMP is %.5f, the LLL DET OMP Time is %.5f.\n"
//                                       "03.The LLLQR DET SER is %.5f, the LLLQR SER Time is %.5f.\n"
//                                       "03.The LLLQR DET OMP is %.5f, the LLLQR OMP Time is %.5f.\n",
//                                       qr_reT.num_iter, qr_reT.run_time,
//                                       qr_reT_omp.num_iter, qr_reT_omp.run_time,
//                                       LLL_reT.num_iter, LLL_reT.run_time,
//                                       LLL_reT_omp.num_iter, LLL_reT_omp.run_time,
//                                       LLL_qr_reT.num_iter, LLL_qr_reT.run_time,
//                                       LLL_qr_reT_omp.num_iter, LLL_qr_reT_omp.run_time);
                            }
                        }

                        //LLL - Validating
                        cils.init_res = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, cils.x_t.data());

                        array<scalar, m * n> ZINV;
                        array<scalar, n> z;
                        ZINV.fill(0);
                        z.fill(0);
                        if (!is_qr) {
                            coder::eye<scalar, index, m, n>(n, ZINV);
                            coder::internal::mrdiv<scalar, index, m, n, n>(ZINV, cils.Z);
                        }
                        //qr: Z is eye Zx_t = x, LLL: Z^{-1}x_t = x
                        coder::internal::blas::mtimes<scalar, index, m, n, 1>(ZINV, cils.x_t, z);
                        scalar R_res = cils::find_residual<scalar, index, m, n>(cils.R_R, cils.y_r, z.data());
                        if (verbose) {
                            cout << "Z*x:";
                            cils::display_vector<scalar, index, n>(z);
                            cout << "x_t:";
                            cils::display_vector<scalar, index, n>(cils.x_t);
                        }

                        //LLL - Block Solver Testing
                        cils.init_R();
                        //Babai
                        init_guess<scalar, index, m, n>(0, &z_B, cils.x_r.data());
                        cils.cils_babai_search_serial(&z_B);
                        ber_babai = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);

                        if (verbose) {
                            cout << "BAB:";
                            cils::display_vector<scalar, index>(&z_B);
                        }

                        //P-Block Solver, 3 threads
                        init_guess<scalar, index, m, n>(0, &z_B, cils.x_r.data());
                        cils.cils_block_search_omp(3, num_trials, 0, &d_s, &z_B);
                        ber_thre3 = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);

                        if (verbose) {
                            cout << "OMP:";
                            cils::display_vector<scalar, index>(&z_B);
                        }

                        //S-Block Solver.
                        init_guess<scalar, index, m, n>(0, &z_B, cils.x_r.data());
                        cils.cils_block_search_serial(0, &d_s, &z_B);
                        ber_serial = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);
                        res_lll = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, z_B.data());
                        if (verbose) {
                            cout << "SER:";
                            cils::display_vector<scalar, index>(&z_B);
                        }

                        printf("[ INITIALIZATION INFO]\n"
                               "01.Init Residual by A is %.5f.\n"
                               "02.Init Residual by R is %.5f.\n"
                               "03.ber_babai by R is %.5f.\n"
                               "04.ber_thre3 by R is %.5f.\n"
                               "05.ber_serial by R is %.5f.\n"
                               "06.Result SER Residual by A is %.5f.\n"
                               "07.QR Result SER Residual by A is %.5f.\n"
                               "08.QR Result BER Residual by A is %.5f.\n",
                               cils.init_res, R_res, ber_babai, ber_thre3, ber_serial, res_lll, res_qr, ber_qr);
                        cout.flush();
                        if (ber_qr == 0 && ber_serial != 0) break;
                    } while (ber_babai < ber_thre3 || ber_babai < ber_serial);
                    cils.cils_back_solve(cils.x_r);
                    ser_qrd += qr_reT.run_time;
                    if(ber_qr == 0 && ber_serial != 0){
                        qr_reT = cils.cils_qr_serial(1, verbose);
                        //QR - Block Solver Testing
                        coder::eye<scalar, index, m, n>(n, cils.Z);
                        for (index ii = 0; ii < n; ii++) {
                            for (index j = 0; j < n; j++) {
                                cils.R_R[j * n + ii] = cils.R_Q[j * n + ii];
                            }
                        }
                        cils.init_R();
                    }
                    for (index init = -1; init <= 1; init++) {
                        printf("[ TRIAL PHASE]++++++++++++++++++++++++++++++++++++++\n");
                        std::cout << "Block, size: " << block_size << std::endl;
                        std::cout << "Init, value: " << init << std::endl;
                        /*
                         * Babai Test
                         */
                        init_guess<scalar, index, m, n>(init, &z_B, cils.x_r.data());
                        reT = cils.cils_babai_search_serial(&z_B);
                        r = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, z_B.data());
                        b = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);
                        t = reT.run_time;
                        res[init + 1][0][count] += r;
                        ber[init + 1][0][count] += b;
                        tim[init + 1][0][count] += t;
                        all_time[0][i][init + 1][count] = t;

                        printf("Method: BAB_SER, AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, "
                               "RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs\n",
                               res[init + 1][0][count] / i, ber[init + 1][0][count] / i, tim[init + 1][0][count] / i,
                               r, b, t);

                        /*
                         * Serial Block Babai Test
                         */
                        init_guess<scalar, index, m, n>(init, &z_B, cils.x_r.data());
                        reT = cils.cils_block_search_serial(init, &d_s, &z_B);
                        r = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, z_B.data());
                        b = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);
                        ser_time = reT.run_time;
                        res[init + 1][1][count] += r;
                        ber[init + 1][1][count] += b;
                        tim[init + 1][1][count] += ser_time;
                        all_time[1][i][init + 1][count] = ser_time;

                        if (init == -1) tim[3][1][count] += reT.x[0];

                        printf("Method: ILS_SER, AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, "
                               "RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs\n",
                               res[init + 1][1][count] / i, ber[init + 1][1][count] / i, tim[init + 1][1][count] / i,
                               r, b, ser_time);

                        /*
                         * BLOCK CPU TEST:
                         */
//                        cils::vector_reverse_permutation<scalar, index, m,  n>(cils.Z, &z_B);
                        reT = cils.cils_block_search_serial_CPUTEST(&d_s, &z_B);

                        scalar block_ils_time = std::accumulate(reT.x.begin(), reT.x.begin() + d_s.size(), 0.0);
                        scalar block_ils_iter = std::accumulate(reT.x.begin() + d_s.size(), reT.x.end(), 0.0);

                        for (index ser_tim_itr = 0; ser_tim_itr < d_s.size(); ser_tim_itr++) {
                            stm[init + 1][ser_tim_itr][count] += (reT.x[ser_tim_itr] / block_ils_time) / max_iter;
                            stm[init + 1][ser_tim_itr + d_s.size()][count] +=
                                    reT.x[ser_tim_itr + d_s.size()] / block_ils_iter / max_iter;
                        }

                        /*
                         * Parallel Block Babai Test
                         */
                        LLL_qr[0][count] += (LLL_qr_reT.run_time / coeff);
                        l = 2;
                        qr_l = 1;
                        scalar prev_t = INFINITY;
                        for (index n_proc = 2; n_proc <= max_proc; n_proc += min_proc) {

                            t = r = b = iter = 0;
                            t2 = r2 = b2 = iter2 = INFINITY;
                            index _ll = 0;

                            LLL_qr_reT_omp = cils.cils_LLL_qr_reduction(1, verbose, n_proc);
                            cils.cils_LLL_qr_reduction(1, verbose, 1);

                            if (init == -1) {
                                LLL_qr[qr_l][count] += (LLL_qr_reT_omp.run_time / coeff);
                                lll_qr_spu[qr_l][count] += LLL_qr_reT.run_time / LLL_qr_reT_omp.run_time;
                            }
                            if(ber_qr == 0 && ber_serial != 0){
                                qr_reT = cils.cils_qr_serial(1, verbose);
                                //QR - Block Solver Testing
                                coder::eye<scalar, index, m, n>(n, cils.Z);
                                for (index ii = 0; ii < n; ii++) {
                                    for (index j = 0; j < n; j++) {
                                        cils.R_R[j * n + ii] = cils.R_Q[j * n + ii];
                                    }
                                }
                                cils.init_R();
                            }
                            while (true) {
                                init_guess<scalar, index, m, n>(init, &z_B, cils.x_r.data());
                                reT = cils.cils_block_search_omp(n_proc, num_trials, init, &d_s, &z_B);
                                iter = reT.num_iter;
                                t = reT.run_time;
                                r = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, z_B.data());
                                b = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);

                                t2 = min(t, t2);
                                r2 = min(r, r2);
                                b2 = min(b, b2);
                                iter2 = min(iter, iter2);
                                _ll++;

                                if (b <= ber_babai && prev_t > t) break;
                                if (_ll == 100) {
                                    r = r2;
                                    b = b2;
                                    t = t2;
                                    iter = iter2;
                                    break; //
                                }
                            }


                            res[init + 1][l][count] += r;
                            ber[init + 1][l][count] += b;
                            tim[init + 1][l][count] += t;
                            all_time[l][i][init + 1][count] = t;

                            if (init == -1) tim[3][l][count] += reT.x[0];
                            itr[init + 1][l][count] += iter;
                            spu[init + 1][l][count] += ser_time / t;

                            t_spu[init + 1][l][count] +=
                                    (ser_time + LLL_qr_reT.run_time / coeff) / (t + LLL_qr_reT_omp.run_time / coeff);
                            printf("Method: ILS_OMP, N_PROC: %2d, AVG RES: %8.5f, AVG BER: %8.5f, "
                                   "AVG TIME: %8.5fs, RES: %8.5f, BER: %8.5f, SER TIME: %8.5f, OMP TIME: %8.5fs, "
                                   "NUM_ITER:%7.3f, SPEEDUP:%7.3f, AVG SPEEDUP: %7.3f, AVG TOTAL SPU: %7.3f, ",
                                   n_proc, res[init + 1][l][count] / i, ber[init + 1][l][count] / i,
                                   tim[init + 1][l][count] / i, r, b, ser_time, t,
                                   iter, ser_time / t, spu[init + 1][l][count] / i, t_spu[init + 1][l][count] / i);
                            printf("LLL_QR SER TIME: %8.5fs, LLL_QR OMP_TIME: %8.5fs, LLL_QR SPEEDUP: %7.3f.\n",
                                   LLL_qr_reT.run_time, LLL_qr_reT_omp.run_time, lll_qr_spu[qr_l][count] / i);

                            qr_l++;
                            l++;
                            prev_t = t;
                        }
                    }
                    run_time = omp_get_wtime() - run_time;
                    printf("++++++++++++++++++++++++++++++++++++++\n Trial %d, Elapsed Time: %.5fs. \n"
                           "++++++++++++++++++++++++++++++++++++++\n", i, run_time);
                    cout.flush();
                }
                printf("\n---------------------\nITER:%d\n---------------------\n", i);
                if (i % plot_itr == 0) {//i % 50 == 0 &&
                    PyObject *pName, *pModule, *pFunc;

                    Py_Initialize();
                    if (_import_array() < 0)
                        PyErr_Print();
                    npy_intp di2[2] = {200, 2};
                    npy_intp dim[3] = {3, 200, 2};
                    npy_intp di4[3] = {4, 200, 2};
                    npy_intp di5[4] = {7, 1000, 3, 2};
                    npy_intp dsd[1] = {static_cast<npy_intp>(d_s.size())};

                    scalar d_s_A[d_s.size()];
                    for (index h = 0; h < d_s.size(); h++) {
                        d_s_A[h] = d_s[h];
                    }
                    scalar proc_nums[l - 2] = {};
                    index ll = 0;

                    for (index n_proc = 2; n_proc <= max_proc; n_proc += min_proc) {
                        proc_nums[ll] = n_proc;
                        ll++;
                    }
                    npy_intp dpc[1] = {ll};

                    PyObject *pRes = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, res);
                    PyObject *pBer = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, ber);
                    PyObject *pTim = PyArray_SimpleNewFromData(3, di4, NPY_DOUBLE, tim);
                    PyObject *pItr = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, itr);
                    PyObject *pSer = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, stm);
                    PyObject *pSpu = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, spu);
                    PyObject *pD_s = PyArray_SimpleNewFromData(1, dsd, NPY_DOUBLE, d_s_A);
                    PyObject *pPrc = PyArray_SimpleNewFromData(1, dpc, NPY_DOUBLE, proc_nums);
                    PyObject *pAtm = PyArray_SimpleNewFromData(4, di5, NPY_DOUBLE, all_time);

                    PyObject *pQRT = PyArray_SimpleNewFromData(2, di2, NPY_DOUBLE, qrT);
                    PyObject *pLLL = PyArray_SimpleNewFromData(2, di2, NPY_DOUBLE, lll);
                    PyObject *plqr = PyArray_SimpleNewFromData(2, di2, NPY_DOUBLE, LLL_qr);
                    PyObject *p_qr = PyArray_SimpleNewFromData(2, di2, NPY_DOUBLE, qr_spu);
                    PyObject *pTsp = PyArray_SimpleNewFromData(2, di2, NPY_DOUBLE, lll_spu);
                    PyObject *plqs = PyArray_SimpleNewFromData(2, di2, NPY_DOUBLE, lll_qr_spu);
                    PyObject *pqls = PyArray_SimpleNewFromData(2, di2, NPY_DOUBLE, qlll_spu);
                    PyObject *pTpu = PyArray_SimpleNewFromData(3, di4, NPY_DOUBLE, t_spu);
                    if (pRes == nullptr) printf("[ ERROR] pRes has a problem.\n");
                    if (pBer == nullptr) printf("[ ERROR] pBer has a problem.\n");
                    if (pTim == nullptr) printf("[ ERROR] pTim has a problem.\n");
                    if (pItr == nullptr) printf("[ ERROR] pItr has a problem.\n");
                    if (pSer == nullptr) printf("[ ERROR] pSer has a problem.\n");
                    if (pD_s == nullptr) printf("[ ERROR] pD_s has a problem.\n");
                    if (pPrc == nullptr) printf("[ ERROR] pPrc has a problem.\n");
                    if (pSpu == nullptr) printf("[ ERROR] pSpu has a problem.\n");
                    if (pAtm == nullptr) printf("[ ERROR] pAtm has a problem.\n");
                    if (pQRT == nullptr) printf("[ ERROR] pQRT has a problem.\n");
                    if (pLLL == nullptr) printf("[ ERROR] pLLL has a problem.\n");
                    if (plqr == nullptr) printf("[ ERROR] plqr has a problem.\n");
                    if (p_qr == nullptr) printf("[ ERROR] p_qr has a problem.\n");
                    if (pTsp == nullptr) printf("[ ERROR] pTsp has a problem.\n");
                    if (plqs == nullptr) printf("[ ERROR] plqs has a problem.\n");
                    if (pqls == nullptr) printf("[ ERROR] pqls has a problem.\n");
                    if (pTpu == nullptr) printf("[ ERROR] pTpu has a problem.\n");

                    PyObject *sys_path = PySys_GetObject("path");
                    PyList_Append(sys_path,
                                  PyUnicode_FromString(
                                          "/home/shilei/CLionProjects/babai_asyn/babai_asyn_c++/src/example"));
                    pName = PyUnicode_FromString("plot_helper");
                    pModule = PyImport_Import(pName);

                    if (pModule != nullptr) {
                        pFunc = PyObject_GetAttrString(pModule, "plot_runtime");
                        if (pFunc && PyCallable_Check(pFunc)) {
                            PyObject *pArgs = PyTuple_New(29);
                            if (PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", n)) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", SNR)) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", k)) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", l)) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 4, Py_BuildValue("i", block_size)) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 5, Py_BuildValue("i", i)) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 6, Py_BuildValue("i", is_qr)) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 7, pRes) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 8, pBer) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 9, pTim) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 10, pItr) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 11, pSer) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 12, pD_s) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 13, pPrc) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 14, pSpu) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 15, pAtm) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 16, Py_BuildValue("i", qr_l)) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 17, Py_BuildValue("i", max_proc)) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 18, Py_BuildValue("i", min_proc)) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 19, pQRT) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 20, pLLL) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 21, plqr) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 22, p_qr) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 23, pTsp) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 24, plqs) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 25, pqls) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 26, pTpu) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 27, Py_BuildValue("i", is_constrained)) != 0) {
                                return false;
                            }
                            if (PyTuple_SetItem(pArgs, 28, Py_BuildValue("i", m)) != 0) {
                                return false;
                            }
                            PyObject *pValue = PyObject_CallObject(pFunc, pArgs);

                        } else {
                            if (PyErr_Occurred())
                                PyErr_Print();
                            fprintf(stderr, "Cannot find function qr\n");
                        }
                    } else {
                        PyErr_Print();
                        fprintf(stderr, "Failed to load file\n");

                    }
                }
            }
            printf("End of current TASK.\n");
            printf("-------------------------------------------\n");
        }

    }
    return 0;
}

template<typename scalar, typename index, index m, index n>
long plot_LLL() {
    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | LLL | %s ]==================================\n", time_str);
    for (SNR = 35; SNR <= 35; SNR += 30) {
        scalar qr_spu[200] = {}, lll_spu[200] = {}, lll_qr_spu[200] = {}, qlll_spu[200] = {};
        scalar qrT[200] = {}, lll[200] = {}, LLL_qr[200] = {};
        index verbose = n <= 16, count = 0, qr_l = 2;
        scalar run_time;
        cils::returnType<scalar, index> qr_reT, LLL_reT, LLL_reT_omp, qr_reT_omp, LLL_qr_reT, LLL_qr_reT_omp;

        for (index i = 1; i <= max_iter; i++) {

            printf("[ PLOT LLL]\n++++++++++++++++++++++++++++++++++++++\n");
            std::cout << "Init, size: " << n << std::endl;
            run_time = omp_get_wtime();
            cils::CILS<scalar, index> cils(k, SNR);

            scalar ber_babai, ber_thre3, ber_serial, res_lll, res_qr, ber_qr;
            printf("[ INIT PHASE]\n++++++++++++++++++++++++++++++++++++++\n");

            //Initialize Problem
            cils.init();

            qr_reT_omp = cils.cils_qr_omp(1, verbose, 2);

            qr_reT = cils.cils_qr_serial(1, verbose);

            LLL_reT = cils.cils_LLL_reduction(1, verbose, 1);
            LLL_reT_omp = cils.cils_LLL_reduction(1, verbose, 2);

            LLL_qr_reT = cils.cils_LLL_qr_reduction(1, verbose, 1);
            LLL_qr_reT_omp = cils.cils_LLL_qr_reduction(1, verbose, 2);

            printf("[ INITIALIZATION INFO]\n"
                   "01.The QR SER Error is %.5f, the QR SER Time is %.5f.\n"
                   "02.The QR OMP Error is %.5f, the QR OMP Time is %.5f.\n"
                   "03.The LLL DET SER is %.5f, the LLL SER Time is %.5f.\n"
                   "04.The LLL DET OMP is %.5f, the LLL DET OMP Time is %.5f.\n"
                   "03.The LLLQR DET SER is %.5f, the LLLQR SER Time is %.5f.\n"
                   "03.The LLLQR DET OMP is %.5f, the LLLQR OMP Time is %.5f.\n",
                   qr_reT.num_iter, qr_reT.run_time,
                   qr_reT_omp.num_iter, qr_reT_omp.run_time,
                   LLL_reT.num_iter, LLL_reT.run_time,
                   LLL_reT_omp.num_iter, LLL_reT_omp.run_time,
                   LLL_qr_reT.num_iter, LLL_qr_reT.run_time,
                   LLL_qr_reT_omp.num_iter, LLL_qr_reT_omp.run_time);

            cout.flush();

            qrT[0] += qr_reT.run_time;
            lll[0] += LLL_reT.run_time;
            LLL_qr[0] += LLL_qr_reT.run_time;

            qr_l = 1;
            for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
                printf("[ QR_LLL Parallel TEST: %d-thread]++++++++++++++++++++++++++++++++\n", n_proc);
                cout.flush();
                qr_reT_omp = cils.cils_qr_omp(1, verbose, n_proc);

                //qr-block Testing
                //LLL reduction

                LLL_qr_reT_omp = cils.cils_LLL_qr_reduction(1, verbose, n_proc);

                LLL_reT_omp = cils.cils_LLL_reduction(1, verbose, n_proc);

                qrT[qr_l] += qr_reT_omp.run_time;
                lll[qr_l] += LLL_reT_omp.run_time;
                LLL_qr[qr_l] += LLL_qr_reT_omp.run_time;
                qr_spu[qr_l] += qrT[0] / qrT[qr_l];
                lll_spu[qr_l] += lll[0] / lll[qr_l];
                lll_qr_spu[qr_l] += LLL_qr[0] / LLL_qr[qr_l];
                qlll_spu[qr_l] += (qrT[0] + lll[0]) / (qrT[qr_l] + lll[qr_l]);

                printf("[ TEST INFO]\n"
                       "        01.The QR Error is %.5f.\n"
                       "        02.The LLL DET is %.5f.\n"
                       "        03.The LLL_QR DET is %.5f.\n",
                       qr_reT_omp.num_iter, LLL_reT_omp.num_iter, LLL_qr_reT_omp.num_iter);
                printf("[ QR_LLL Parallel TEST INFO]++++++++++++++++++++++++++++++++\n");

                printf("Method: QR/LLL_OMP, N_PROC: %2d, QR SER_TIME: %.5fs,"
                       "QR OMP_TIME: %.5fs, QR SPEEDUP: %7.3f, LLL SER_TIME: %.5fs,"
                       "LLL OMP_TIME: %.5fs, LLL SPEEDUP: %7.3f, "
                       "QR+LLL SER_TIME: %.5fs, QR+LLL OMP_TIME: %.5fs, QR+LLL SPEEDUP: %7.3f,"
                       "LLL_QR OMP_TIME: %.5fs, LLL_QR SPEEDUP: %7.3f.\n",
                       n_proc, qrT[0] / i,
                       qrT[qr_l] / i, qr_spu[qr_l] / i, lll[0] / i,
                       lll[qr_l] / i, lll_spu[qr_l] / i,
                       (qrT[0] + lll[0]) / i, (qrT[qr_l] + lll[qr_l]) / i, qlll_spu[qr_l] / i,
                       LLL_qr[qr_l] / i, lll_qr_spu[qr_l] / i);

                qr_l++;
            }


            run_time = omp_get_wtime() - run_time;
            printf("++++++++++++++++++++++++++++++++++++++\n Trial %d, Elapsed Time: %.5fs. \n"
                   "++++++++++++++++++++++++++++++++++++++\n", i, run_time);
            cout.flush();


            printf("\n---------------------\nITER:%d\n---------------------\n", i);
            if (i % plot_itr == 0) {//i % 50 == 0 &&
                PyObject *pName, *pModule, *pFunc;
                PyObject *pArgs, *pValue;
                Py_Initialize();
                if (_import_array() < 0)
                    PyErr_Print();
                npy_intp dim[1] = {200};

                scalar proc_nums[qr_l - 2] = {};
                index ll = 0;

                for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
                    proc_nums[ll] = n_proc;
                    ll++;
                }
                npy_intp dpc[1] = {ll};


                PyObject *pQRT = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, qrT);
                PyObject *pLLL = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, lll);
                PyObject *plqr = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, LLL_qr);
                PyObject *p_qr = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, qr_spu);
                PyObject *pTsp = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, lll_spu);
                PyObject *plqs = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, lll_qr_spu);
                PyObject *pqls = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, qlll_spu);

                if (pQRT == nullptr) printf("[ ERROR] pQRT has a problem.\n");
                if (pLLL == nullptr) printf("[ ERROR] pLLL has a problem.\n");
                if (plqr == nullptr) printf("[ ERROR] plqr has a problem.\n");
                if (p_qr == nullptr) printf("[ ERROR] p_qr has a problem.\n");
                if (pTsp == nullptr) printf("[ ERROR] pTsp has a problem.\n");
                if (plqs == nullptr) printf("[ ERROR] plqs has a problem.\n");
                if (pqls == nullptr) printf("[ ERROR] pqls has a problem.\n");

                PyObject *sys_path = PySys_GetObject("path");

                if (is_local)
                    PyList_Append(sys_path, PyUnicode_FromString(
                            "/home/shilei/CLionProjects/babai_asyn/babai_asyn_c++/src/example"));
                else
                    PyList_Append(sys_path, PyUnicode_FromString("./"));

                pName = PyUnicode_FromString("plot_helper");
                pModule = PyImport_Import(pName);

                if (pModule != nullptr) {
                    pFunc = PyObject_GetAttrString(pModule, "plot_runtime_lll");
                    if (pFunc && PyCallable_Check(pFunc)) {
                        pArgs = PyTuple_New(12);
                        if (PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", n)) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", qr_l)) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", i)) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", max_proc)) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 4, Py_BuildValue("i", min_proc)) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 5, pQRT) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 6, pLLL) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 7, plqr) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 8, p_qr) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 9, pTsp) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 10, plqs) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 11, pqls) != 0) {
                            return false;
                        }
                        pValue = PyObject_CallObject(pFunc, pArgs);

                    } else {
                        if (PyErr_Occurred())
                            PyErr_Print();
                        fprintf(stderr, "Cannot find function qr\n");
                    }
                } else {
                    PyErr_Print();
                    fprintf(stderr, "Failed to load file\n");

                }
            }
        }
        printf("End of current TASK.\n");
        printf("-------------------------------------------\n");

    }

    return 0;
}

template<typename scalar, typename index, index m, index n>
long test_ils_search() {
    std::cout << "Init, size: " << n << std::endl;
    cils::CILS<scalar, index> cils(k, SNR);
    index init = -1;
    scalar error = 0, b, r, ber;
    b_vector z_B(n, 0);
    cils::returnType<scalar, index> reT, back_reT, qr_reT = {{}, 0, 0}, qr_reT_omp = {{}, 0, 0};

    for (index i = 0; i < max_iter; i++) {
        cils.init();
        cils::display_vector<scalar, index, n>(cils.x_t);
        scalar res = 0;
        do {
//                qr_reT = cils.cils_hqr_decomposition_serial(1, 1);
            qr_reT = cils.cils_qr_serial(0, 1);
//                qr_reT = cils.cils_LLL_reduction(0, 1);
//                cils.init_R();
            cils.init_y();
            cils::display_vector<scalar, index, n>(cils.x_t);
//                cils.init_R();


//                cils.init_res = cils::find_residual<scalar, index, m,  n>(cils.A, cils.y_A, cils.x_t);
//                cout << "INIT_RES:" << cils.init_res <<endl;
            cils.init_res = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, cils.x_t.data());
            cout << "INIT_RES:" << cils.init_res << endl;

            //Validating Result is solvable
            init_guess<scalar, index, m, n>(0, &z_B, cils.x_r.data());
            cils.cils_babai_search_serial(&z_B);
            b = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);
            init_guess<scalar, index, m, n>(0, &z_B, cils.x_r.data());
            cils.cils_block_search_omp(3, num_trials, 0, &d_s, &z_B);
//                cils::display_vector<scalar, index>(&z_B);
//                cils::display_vector<scalar, index, n>(cils.x_t);
            b = b - cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);
            r = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, z_B.data());
            cout << "The BER diff is " << b << " and the res is " << r << "\n";
        } while (b < 0);
        cils.cils_back_solve(cils.x_r);

        printf("init_res: %.5f, real_res: %.5f, sigma: %.5f, qr_error: %.1f\n", cils.init_res, res, cils.sigma, error);

        init_guess<scalar, index, m, n>(0, &z_B, cils.x_r.data());
        reT = cils.cils_babai_search_serial(&z_B);
        res = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, z_B.data());
        ber = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);
        printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5fs Total Time: %.5fs\n",
               res, ber, reT.run_time, qr_reT.run_time, qr_reT.run_time + reT.run_time);
        scalar bab_tim_constrained = reT.run_time;

        init_guess<scalar, index, m, n>(0, &z_B, cils.x_r.data());
        reT = cils.cils_block_search_serial(init, &d_s, &z_B);
        res = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, z_B.data());
        ber = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);
        printf("\nMethod: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5fs Total Time: %.5fs\n",
               block_size, res, ber, reT.run_time, qr_reT.run_time, qr_reT.run_time + reT.run_time);
        scalar ils_tim_constrained = reT.run_time;

//        for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
//#pragma omp parallel default(none) num_threads(n_proc)
//            {}
//            init_guess<scalar, index, m,  n>(0, &z_B, cils.x_r.data());
//            reT = cils.cils_babai_search_omp(n_proc > max_proc ? max_proc : n_proc, num_trials, &z_B);
//            res = cils::find_residual<scalar, index, m,  n>(cils.A, cils.y_a, z_B.data());
//            ber = cils::find_bit_error_rate<scalar, index, m,  n>(&z_B, cils.x_t, k);
//            printf("Method: BAB_OMP, n_proc: %d, Res: %.5f, BER: %.5f, Num_iter: %f, "
//                   "Solve Time: %.5fs, Solve SpeedUp: %.3f, "
//                   "QR Error: %.5f, QR Time: %.5fs, QR SpeedUp: %.3f, "
//                   "Total Time: %.5fs, Total SpeedUp: %.3f\n",
//                   n_proc, res, ber, reT.num_iter, reT.run_time, (bab_tim_constrained / reT.run_time),
//                   qr_reT_omp.num_iter, qr_reT_omp.run_time, qr_reT.run_time / qr_reT_omp.run_time,
//                   reT.run_time + qr_reT_omp.run_time,
//                   (bab_tim_constrained + qr_reT.run_time) / (qr_reT_omp.run_time + reT.run_time));
//
//        }
        for (init = -1; init <= 1; init++) {
            for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
#pragma omp parallel default(none) num_threads(n_proc)
                {}
                init_guess<scalar, index, m, n>(init, &z_B, z_B.data());
                reT = cils.cils_block_search_omp(n_proc > max_proc ? max_proc : n_proc, num_trials, init, &d_s, &z_B);
//                cils::display_vector_by_block<scalar, index>(&d_s, &z_B);
                res = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, z_B.data());
                ber = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);
                printf("Method: ILS_OMP, n_proc: %d, Res: %.5f, BER: %.5f, Num_iter: %.1f, "
                       "Solve Time: %.5fs, Solve SpeedUp: %.3f, "
                       "QR Error: %.5f, QR Time: %.5fs, QR SpeedUp: %.3f, "
                       "Total Time: %.5fs, Total SpeedUp: %.3f\n",
                       n_proc, res, ber, reT.num_iter, reT.run_time,
                       ((ils_tim_constrained + bab_tim_constrained) / reT.run_time),
                       qr_reT_omp.num_iter, qr_reT_omp.run_time, qr_reT.run_time / qr_reT_omp.run_time,
                       reT.run_time + qr_reT_omp.run_time,
                       (ils_tim_constrained + qr_reT.run_time) / (qr_reT_omp.run_time + reT.run_time)
                );
            }
        }
    }
    return 0;
}

template<typename scalar, typename index, index m, index n>
void plot_res() {
    for (k = 1; k <= 3; k += 2)
        for (SNR = 15; SNR <= 35; SNR += 20) {
            printf("plot_res-------------------------------------------\n");
            std::cout << "Init, size: " << n << std::endl;
            std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
            std::cout << "Init, SNR: " << SNR << std::endl;

            //bool read_r, bool read_ra, bool read_xy
            cils::CILS<scalar, index> cils(k, SNR);
            cils.init();
            auto reT = cils.cils_qr_matlab();
            cils.init_R();
            printf("init_res: %.5f, sigma: %.5f, qr time: %.5fs\n", cils.init_res, cils.sigma, reT.run_time);

            b_vector z_B(n, 0);
            reT = cils.cils_babai_search_serial(&z_B);
            scalar res = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, z_B.data());
            scalar ber = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);
            printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs\n",
                   res, ber, reT.run_time);

//            &z_B.assign(n, 0);
            reT = cils.cils_block_search_serial(0, &d_s, &z_B);
            res = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, z_B.data());
            ber = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);
            printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs\n",
                   block_size, res, ber, reT.run_time);

            for (index init = -1; init <= 1; init++) {
                cout << "init," << init << "\n";
                init_guess<scalar, index, m, n>(0, &z_B, cils.x_r.data());
                reT = cils.cils_block_search_serial(init, &d_s, &z_B);
                res = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, z_B.data());
                ber = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);

                printf("Method: ILS_SER, Block size: %d, Res: %.5f, Ber: %.5f, Time: %.5fs\n",
                       block_size, res, ber, reT.run_time);
                for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
                    cout << d_s[d_s.size() - 1] << "," << n_proc << ",";
                    std::cout.flush();
                    for (index nswp = 1; nswp < max_iter; nswp++) {
                        init_guess<scalar, index, m, n>(init, &z_B, cils.x_r.data());
                        reT = cils.cils_block_search_omp(n_proc > max_proc ? max_proc : n_proc, nswp, init, &d_s, &z_B);
                        res = cils::find_residual<scalar, index, m, n>(cils.A, cils.y_a, z_B.data());
                        ber = cils::find_bit_error_rate<scalar, index, m, n>(&z_B, cils.x_t, k);
                        printf("diff=%.1f, res=%.5f, ber=%.5f, ",
                               reT.num_iter > (N / block_size) ? (N / block_size) : reT.num_iter, res, ber);
                    }
                    cout << endl;
                }
            }

            printf("End of current TASK.\n");
            printf("-------------------------------------------\n");
        }
}
