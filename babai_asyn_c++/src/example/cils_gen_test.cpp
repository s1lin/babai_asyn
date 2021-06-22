
#include "../source/cils.cpp"
#include "../source/cils_ils_search.cpp"
#include "../source/cils_block_search.cpp"
#include "../source/cils_babai_search.cpp"
#include "../source/cils_reduction.cpp"

template<typename scalar, typename index, index n>
long plot_run() {

    for (SNR = 35; SNR <= 35; SNR += 30) {
        index d_s_size = d_s.size();
        scalar res[3][200][2] = {}, ber[3][200][2] = {}, tim[4][200][2] = {}, spu[4][200][2] = {};
        scalar itr[3][200][2] = {}, stm[3][200][2] = {};
        scalar all_time[7][1000][3][2] = {}; //Method, iter, init, qam
        scalar r, t, b, iter, ser_qrd, run_time, ser_time;
        scalar t2, r2, b2, iter2;

        index count = 0, l = 2; //count for qam.

        vector<scalar> z_B(n, 0);

        cils::returnType<scalar, index> reT, qr_reT, LLL_reT;
//        for (k = 1; k <= 3; k += 2) {
        for (index tt = 0; tt <= 0; tt++) {
            is_qr = tt;
            for (index i = 1; i <= max_iter; i++) {
                ser_qrd = 0;
                l = 2;
                for (k = 3; k >= 1; k -= 2) {
                    printf("plot_res-------------------------------------------\n");
                    std::cout << "Init, size: " << n << std::endl;
                    std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
                    count = k == 1 ? 0 : 1;
                    run_time = omp_get_wtime();
                    cils::cils<scalar, index, n> cils(k, SNR);
//                    cils.init();

                    scalar ber_babai, ber_thre3, ber_serial;

                    if (!is_read) {
                        do {
//                            qr_reT = cils.cils_qr_matlab();
//                            qr_reT = cils.cils_PLLL_reduction_serial();
//                            cils.init_R();
                            cils.init();
                            qr_reT = cils.cils_qr_serial(0, 1);
                            //parameters: eval, n_proc:1 -> serial, >1 OMP
//                            cils::display_scalarType<scalar, index>(cils.R);
                            cils.init_y();
//                            cils::display_scalarType<scalar, index>(cils.R);
                            LLL_reT = cils.cils_LLL_reduction(1, 1);
//                            cils::display_scalarType<scalar, index>(cils.R);
                            cils.init_R();

                            cils.init_res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &cils.x_t);
                            init_guess<scalar, index, n>(2, &z_B, &cils.x_t);
                            cils::vector_permutation<scalar, index, n>(cils.Z, &z_B);
                            cout<<"Z_B:";
                            cils::display_vector<scalar, index>(&z_B);

                            cout<<"x_t:";
                            cils::display_vector<scalar, index>(&cils.x_t);
                            scalar R_res = cils::find_residual<scalar, index, n>(cils.R, cils.y_A, &z_B);

                            //Validating Result is solvable
                            init_guess<scalar, index, n>(0, &z_B, &cils.x_R);
                            cils.cils_babai_search_serial(&z_B);
                            cout<<"BAB:";
                            cils::display_vector<scalar, index>(&z_B);

                            ber_babai = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);

                            init_guess<scalar, index, n>(0, &z_B, &cils.x_R);
                            cils.cils_block_search_omp(2, num_trials, 0, &d_s, &z_B);
                            cout<<"OMP:";
                            cils::display_vector<scalar, index>(&z_B);
                            ber_thre3 = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);

                            init_guess<scalar, index, n>(0, &z_B, &cils.x_R);
                            cils.cils_block_search_serial(0, &d_s, &z_B);
                            cout<<"SER:";
                            cils::display_vector<scalar, index>(&z_B);
                            ber_serial = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);

                            r = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
                            cout << "The SERIAL BER diff is " << ber_babai - ber_serial << " and THRD-BER is "
                                 << ber_babai - ber_thre3 << "\n";
                            printf("[ INITIALIZATION INFO]\n"
                                   "1.The QR Error is %.5f.\n"
                                   "2.The determinant of LLL is %.5f.\n"
                                   "3.Init Residual by A is %.5f.\n"
                                   "4.Init Residual by R is %.5f.\n"
                                   "5.ber_babai by R is %.5f.\n"
                                   "6.ber_thre3 by R is %.5f.\n"
                                   "7.ber_serial by R is %.5f.\n",
                                   qr_reT.num_iter, LLL_reT.num_iter, cils.init_res, R_res, ber_babai, ber_thre3, ber_serial);

                        } while (ber_babai < ber_thre3 || ber_babai < ber_serial);

                        cils.cils_back_solve(&cils.x_R);
                    }


                    ser_qrd += qr_reT.run_time;

                    for (index init = -1; init <= 1; init++) {
                        printf("++++++++++++++++++++++++++++++++++++++\n");
                        std::cout << "Block, size: " << block_size << std::endl;
                        std::cout << "Init, value: " << init << std::endl;
                        /*
                         * Babai Test
                         */
                        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                        reT = cils.cils_babai_search_serial(&z_B);
                        r = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
                        b = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
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
                        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                        reT = cils.cils_block_search_serial(init, &d_s, &z_B);
                        r = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
                        b = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
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
                        cils::vector_reverse_permutation<scalar, index, n>(cils.Z, &z_B);
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
                        l = 2;
                        scalar prev_t = INFINITY;
                        for (index n_proc = 2; n_proc <= max_proc; n_proc += min_proc) {
#pragma omp parallel default(none) num_threads(n_proc)
                            {}
                            t = r = b = iter = 0;
                            t2 = r2 = b2 = iter2 = INFINITY;

                            index _ll = 0;
                            while (true) {
                                init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                                reT = cils.cils_block_search_omp(n_proc, num_trials, init, &d_s, &z_B);
                                iter = reT.num_iter;

                                r = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
                                b = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
                                t = reT.run_time;

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
                            printf("Method: ILS_OMP, N_PROC: %d, AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, "
                                   "RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs, NUM_ITER:%.5f, SPEEDUP: %.3f,\n",
                                   n_proc, res[init + 1][l][count] / i, ber[init + 1][l][count] / i,
                                   tim[init + 1][l][count] / i,
                                   r, b, t, iter, spu[init + 1][l][count] / i);

//                            printf("Method: ILS_OMP, N_PROC: %d, RES :%.5f, BER: %.5f, num_iter: %.5f, Time: %.5fs, Avg Time: %.5fs, "
//                                   "Speed up: %.3f, SER REL TIM: %.5f, REAL Time: %.5fs, REAL SpeedUp: %.3f, Total Time: %.5fs, Total SpeedUp: %.3f\n",
//                                   n_proc, res[init + 1][l][count] / i,
//                                   ber[init + 1][l][count] / i, itr[init + 1][l][count] / i,
//                                   tim[init + 1][l][count] / i, tim[init + 1][l][count] / i,
//                                   tim[init + 1][1][count] / tim[init + 1][l][count],
//                                   tim[3][1][count] / i, tim[3][l][count] / i, tim[3][1][count] / tim[3][l][count],
//                                   (tim[3][1][count] + tim[3][l][count]) / i,
//                                   (ser_qrd + tim[init + 1][1][count]) / (qrd[0][l][count] + tim[init + 1][l][count])
//                            );
                            l++;
                            prev_t = t;
                        }
                        printf("++++++++++++++++++++++++++++++++++++++\n");
                    }
                    run_time = omp_get_wtime() - run_time;
                    printf("\n %d-Time: %.5fs, \n", i, run_time);
                    cout.flush();
                }

//                for (k = 3; k >= 1; k -= 2) {
//                    count = k == 1 ? 0 : 1;
//
//                    for (index init = -1; init <= 1; init++) {
//                        printf("++++++++++++++++++++++++++++++++++++++\n");
//                        std::cout << "Block, size: " << block_size << std::endl;
//                        std::cout << "Init, value: " << init << std::endl;
//
//                        printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5f, Total Time: %.5fs\n",
//                               res[init + 1][0][count] / i, ber[init + 1][0][count] / i, tim[init + 1][0][count],
//                               ser_qrd / i,
//                               (ser_qrd + tim[init + 1][0][count]));
//                        printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5f, Total Time: %.5fs\n",
//                               block_size, res[init + 1][1][count] / i, ber[init + 1][1][count] / i,
//                               tim[init + 1][1][count],
//                               ser_qrd / i, (ser_qrd + tim[init + 1][1][count]));
//                        l = 2;
//                        for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
//                            printf("Method: ILS_OMP, n_proc: %d, Res :%.5f, BER: %.5f, num_iter: %.5f, Time: %.5fs, Avg Time: %.5fs, "
//                                   "Speed up: %.3f, SER REL TIM: %.5f, REAL Time: %.5fs, REAL SpeedUp: %.3f, Total Time: %.5fs, Total SpeedUp: %.3f\n",
//                                   n_proc, res[init + 1][l][count] / i,
//                                   ber[init + 1][l][count] / i, itr[init + 1][l][count] / i,
//                                   tim[init + 1][l][count] / i, tim[init + 1][l][count] / i,
//                                   tim[init + 1][1][count] / tim[init + 1][l][count],
//                                   tim[3][1][count] / i, tim[3][l][count] / i, tim[3][1][count] / tim[3][l][count],
//                                   (tim[3][1][count] + tim[3][l][count]) / i,
//                                   (ser_qrd + tim[init + 1][1][count]) / (qrd[0][l][count] + tim[init + 1][l][count])
//                            );
//                            l++;
//                        }
//                        printf("++++++++++++++++++++++++++++++++++++++\n");
//                    }
//                }
                printf("\n---------------------\nITER:%d\n---------------------\n", i);
                if (i % plot_itr == 0) {//i % 50 == 0 &&
                    PyObject *pName, *pModule, *pFunc;
                    PyObject *pArgs, *pValue, *pRes, *pBer, *pTim, *pItr, *pSer, *pD_s, *pPrc, *pSpu, *pAtm;
                    Py_Initialize();
                    if (_import_array() < 0)
                        PyErr_Print();
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

                    pRes = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, res);
                    pBer = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, ber);
                    pTim = PyArray_SimpleNewFromData(3, di4, NPY_DOUBLE, tim);
                    pItr = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, itr);
                    pSer = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, stm);
                    pSpu = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, spu);
                    pD_s = PyArray_SimpleNewFromData(1, dsd, NPY_DOUBLE, d_s_A);
                    pPrc = PyArray_SimpleNewFromData(1, dpc, NPY_DOUBLE, proc_nums);
                    pAtm = PyArray_SimpleNewFromData(4, di5, NPY_DOUBLE, all_time);
                    if (pRes == nullptr) printf("pRes has a problem.\n");
                    if (pBer == nullptr) printf("pBer has a problem.\n");
                    if (pTim == nullptr) printf("pTim has a problem.\n");
                    if (pItr == nullptr) printf("pItr has a problem.\n");
                    if (pSer == nullptr) printf("pSer has a problem.\n");
                    if (pD_s == nullptr) printf("pD_s has a problem.\n");
                    if (pPrc == nullptr) printf("pPrc has a problem.\n");
                    if (pSpu == nullptr) printf("pSpu has a problem.\n");
                    if (pAtm == nullptr) printf("pAtm has a problem.\n");

                    PyObject *sys_path = PySys_GetObject("path");
                    PyList_Append(sys_path,
                                  PyUnicode_FromString(
                                          "/home/shilei/CLionProjects/babai_asyn/babai_asyn_c++/src/example"));
                    pName = PyUnicode_FromString("plot_helper");
                    pModule = PyImport_Import(pName);

                    if (pModule != nullptr) {
                        pFunc = PyObject_GetAttrString(pModule, "plot_runtime");
                        if (pFunc && PyCallable_Check(pFunc)) {
                            pArgs = PyTuple_New(16);
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

    }
    return 0;
}

template<typename scalar, typename index, index n>
long test_ils_search() {
    std::cout << "Init, size: " << n << std::endl;
    cils::cils<scalar, index, n> cils(k, SNR);
    index init = -1;
    scalar error = 0, b, r, ber;
    vector<scalar> z_B(n, 0);
    cils::returnType<scalar, index> reT, back_reT, qr_reT = {{}, 0, 0}, qr_reT_omp = {{}, 0, 0};

    for (index i = 0; i < max_iter; i++) {
        cils.init();
        cils::display_vector<scalar, index>(&cils.x_t);
        scalar res = 0;
        if (!is_read) {
            do {
//                qr_reT = cils.cils_hqr_decomposition_serial(1, 1);
                qr_reT = cils.cils_qr_serial(0, 1);
                qr_reT = cils.cils_LLL_reduction(0, 1);
//                cils.init_R();
                cils.init_y();
                cils::display_vector<scalar, index>(&cils.x_t);
//                cils.init_R();


//                cils.init_res = cils::find_residual<scalar, index, n>(cils.A, cils.y_A, &cils.x_t);
//                cout << "INIT_RES:" << cils.init_res <<endl;
                cils.init_res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &cils.x_t);
                cout << "INIT_RES:" << cils.init_res <<endl;

                //Validating Result is solvable
                init_guess<scalar, index, n>(0, &z_B, &cils.x_R);
                cils.cils_babai_search_serial(&z_B);
                b = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
                init_guess<scalar, index, n>(0, &z_B, &cils.x_R);
                cils.cils_block_search_omp(3, num_trials, 0, &d_s, &z_B);
//                cils::display_vector<scalar, index>(&z_B);
//                cils::display_vector<scalar, index>(&cils.x_t);
                b = b - cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
                r = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
                cout << "The BER diff is " << b << " and the res is " << r << "\n";
            } while (b < 0);
            cils.cils_back_solve(&cils.x_R);
        }

        printf("init_res: %.5f, real_res: %.5f, sigma: %.5f, qr_error: %.1f\n", cils.init_res, res, cils.sigma, error);

        init_guess<scalar, index, n>(0, &z_B, &cils.x_R);
        reT = cils.cils_babai_search_serial(&z_B);
        res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
        ber = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
        printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5fs Total Time: %.5fs\n",
               res, ber, reT.run_time, qr_reT.run_time, qr_reT.run_time + reT.run_time);
        scalar bab_tim_constrained = reT.run_time;

        init_guess<scalar, index, n>(0, &z_B, &cils.x_R);
        reT = cils.cils_block_search_serial(init, &d_s, &z_B);
        res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
        ber = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
        printf("\nMethod: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5fs Total Time: %.5fs\n",
               block_size, res, ber, reT.run_time, qr_reT.run_time, qr_reT.run_time + reT.run_time);
        scalar ils_tim_constrained = reT.run_time;

//        for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
//#pragma omp parallel default(none) num_threads(n_proc)
//            {}
//            init_guess<scalar, index, n>(0, &z_B, &cils.x_R);
//            reT = cils.cils_babai_search_omp(n_proc > max_proc ? max_proc : n_proc, num_trials, &z_B);
//            res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
//            ber = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
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
                init_guess<scalar, index, n>(init, &z_B, &z_B);
                reT = cils.cils_block_search_omp(n_proc > max_proc ? max_proc : n_proc, num_trials, init, &d_s, &z_B);
//                cils::display_vector_by_block<scalar, index>(&d_s, &z_B);
                res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
                ber = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
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

template<typename scalar, typename index, index n>
void plot_res() {
    for (k = 1; k <= 3; k += 2)
        for (SNR = 15; SNR <= 35; SNR += 20) {
            printf("plot_res-------------------------------------------\n");
            std::cout << "Init, size: " << n << std::endl;
            std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
            std::cout << "Init, SNR: " << SNR << std::endl;

            //bool read_r, bool read_ra, bool read_xy
            cils::cils<scalar, index, n> cils(k, SNR);
            cils.init();
            auto reT = cils.cils_qr_matlab();
            cils.init_R();
            printf("init_res: %.5f, sigma: %.5f, qr time: %.5fs\n", cils.init_res, cils.sigma, reT.run_time);

            vector<scalar> z_B(n, 0);
            reT = cils.cils_babai_search_serial(&z_B);
            scalar res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
            scalar ber = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
            printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs\n",
                   res, ber, reT.run_time);

            z_B.assign(n, 0);
            reT = cils.cils_block_search_serial(0, &d_s, &z_B);
            res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
            ber = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
            printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs\n",
                   block_size, res, ber, reT.run_time);

            for (index init = -1; init <= 1; init++) {
                cout << "init," << init << "\n";
                init_guess<scalar, index, n>(0, &z_B, &cils.x_R);
                reT = cils.cils_block_search_serial(init, &d_s, &z_B);
                res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
                ber = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);

                printf("Method: ILS_SER, Block size: %d, Res: %.5f, Ber: %.5f, Time: %.5fs\n",
                       block_size, res, ber, reT.run_time);
                for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
                    cout << d_s[d_s.size() - 1] << "," << n_proc << ",";
                    std::cout.flush();
                    for (index nswp = 1; nswp < max_iter; nswp++) {
                        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                        reT = cils.cils_block_search_omp(n_proc > max_proc ? max_proc : n_proc, nswp, init, &d_s, &z_B);
                        res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
                        ber = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
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
