//
// Created by shilei on 7/27/21.
//

#include "../source/cils.cpp"

#include "../source/cils_block_search.cpp"
//#include "../source/cils_babai_search.cpp"
//#include "../source/cils_reduction.cpp"
#include "../source/cils_init_point.cpp"
#include "../source/cils_sic_opt.cpp"
#include <ctime>

//template<typename scalar, typename index, index m, index n>
//void init_point_test() {
//    time_t t0 = time(nullptr);
//    struct tm *lt = localtime(&t0);
//    char time_str[20];
//    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
//            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
//            lt->tm_hour, lt->tm_min, lt->tm_sec
//    );
//    printf("====================[ TEST | INIT_POINT | %s ]==================================\n", time_str);
//
//
//    cils::cils<scalar, index, m, n> cils(k, 35);
//    cils.init_ud();
////    cout << "A:" << endl;
////    cils::display_matrix<scalar, index, m, n>(cils.A);
////    cout << "HH_SIC" << endl;
////    cils::display_matrix<scalar, index, m, n>(cils.H);
//
//    vector<scalar> x(n, 0);
//    scalar v_norm;
//    array<scalar, m * n> A_t;
//    array<scalar, n * n> P;
//    cils::returnType<scalar, index> reT;
//
//    reT = cils.cils_sic_serial(x, A_t, P);
////    cout << reT.info << ", " << reT.run_time << endl;
////    cils::display_matrix<scalar, index, m, n>(A_t);
//
////    cils.cils_qr_serial(1, 0);
////    helper::display_vector<scalar, index, m>(cils.y_q);
////    reT = cils.cils_qrp_serial(x, A_t, P);
////    cout << reT.info << ", " << reT.run_time << endl;
////    cils::display_matrix<scalar, index, m, n>(A_t);
//
//
//    reT = cils.cils_grad_proj(x, 100);
//    cout << "x" << endl;
//    helper::display_vector<scalar, index>(n, x.data(), "s_bar");
//}
//
//template<typename scalar, typename index, index m, index n>
//void sic_opt_test() {
//    time_t t0 = time(nullptr);
//    struct tm *lt = localtime(&t0);
//    char time_str[20];
//    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
//            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
//            lt->tm_hour, lt->tm_min, lt->tm_sec
//    );
//    printf("====================[ TEST | SIC_OPT | %s ]==================================\n", time_str);
//
//
//    cils::cils<scalar, index, m, n> cils(k, 35);
//    cils.init_ud();
//
//    vector<scalar> x(n, 0), x_p(n, 0), Ax(m, 0);
//    scalar v_norm;
//    array<scalar, m * n> A_t;
//    array<scalar, n * n> P;
//    array<scalar, m> v_ip;
//    cils::returnType<scalar, index> reT;
//
//    reT = cils.cils_sic_serial(x, A_t, P);
//    helper::display_vector<scalar, index>(n, x.data(), "x_z");
//
//    cils::matrix_vector_mult<scalar, index, m, n>(A_t, x, Ax);
//    helper::vsubtract<scalar, index, m>(cils.y_a, Ax.data(), v_ip);
//    scalar v_norm1 = reT.info;
//    helper::display_vector<scalar, index, m>(v_ip);
//
//    reT = cils.cils_sic_subopt(x, v_ip, A_t, v_norm1, 0, 2);
//    helper::display_vector<scalar, index>(n, x.data(), "x_z");
//    helper::display_vector<scalar, index>(reT.x);
//    cout << "v_norm_cur: " << reT.info << endl;
//}

template<typename scalar, typename index, index m, index n>
void block_optimal_test(int size, int rank) {
    vector<scalar> x_q(n, 0), x_tmp(n, 0), x_ser(n, 0), x_omp(n, 0), x_mpi(n, 0);
    double *v_norm_qr = (double *) calloc(1, sizeof(double));

    scalar v_norm;
    cils::returnType<scalar, index> reT3;
    cils::cils<scalar, index, m, n> cils(k, 35);
    //auto a = (double *)malloc(N * sizeof(double));
    if (rank == 0) {
        cils.init_ud();
        time_t t0 = time(nullptr);
        struct tm *lt = localtime(&t0);
        char time_str[20];
        sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
                lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
                lt->tm_hour, lt->tm_min, lt->tm_sec
        );
        printf("====================[ TEST | SIC_OPT | %s ]==================================\n", time_str);

        cils::returnType<scalar, index> reT, reT2;
        //STEP 1: init point by QRP
        reT = cils.cils_qrp_serial(x_q);
        helper::display_vector<scalar, index>(n, x_q.data(), "x");
        v_norm_qr[0] = reT.info;
        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_q.data(), x_tmp.data());
        index diff = helper::length_nonzeros<scalar, index, n>(x_tmp.data(), cils.x_t.data());
        printf("error_bits: %d, v_norm: %8.4f, time: %8.4f\n", diff, v_norm, reT.run_time);


        x_tmp.assign(n, 0);
        for (index i = 0; i < n; i++) {
            x_ser[i] = x_q[i];
            x_omp[i] = x_q[i];
            x_mpi[i] = x_q[i];
        }


        //STEP 2: Block SCP:
        reT2 = cils.cils_scp_block_optimal_serial(x_ser, v_norm_qr[0]);
        v_norm = reT2.info;
        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_ser.data(), x_tmp.data());

        //Result Validation:
//        helper::display_vector<scalar, index>(n, x_ser.data(), "x_z");
//        helper::display_vector<scalar, index>(n, cils.x_t.data(), "x_t");
//        helper::display_vector<scalar, index>(n, x_tmp.data(), "x_p");
        diff = helper::length_nonzeros<scalar, index, n>(x_tmp.data(), cils.x_t.data());
        printf("error_bits: %d, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
               diff, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);

        //STEP 2: OMP-Block SCP:
        x_tmp.assign(n, 0);
        reT2 = cils.cils_scp_block_optimal_omp(x_omp, v_norm_qr[0]);
        v_norm = reT2.info;
        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_omp.data(), x_tmp.data());

        //Result Validation:
//        helper::display_vector<scalar, index>(n, x_omp.data(), "x_z");
//        helper::display_vector<scalar, index>(n, cils.x_t.data(), "x_t");
//        helper::display_vector<scalar, index>(n, x_tmp.data(), "x_p");
        diff = helper::length_nonzeros<scalar, index, n>(x_tmp.data(), cils.x_t.data());
        printf("error_bits: %d, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
               diff, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);

    }

    //STEP 2: MPI-Block SCP:
    MPI_Bcast(&x_mpi[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&v_norm_qr[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&cils.y_a[0], m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(&cils.H[0], m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    reT3 = cils.cils_scp_block_optimal_mpi(x_mpi, v_norm_qr, size, rank);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
//        v_norm = reT3.info;
        x_tmp.assign(n, 0);
        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_mpi.data(), x_tmp.data());

        //Result Validation:
//        helper::display_vector<scalar, index>(n, x_mpi.data(), "x_z");
//        helper::display_vector<scalar, index>(n, cils.x_t.data(), "x_t");
//        helper::display_vector<scalar, index>(n, x_tmp.data(), "x_p");
        index diff = helper::length_nonzeros<scalar, index, n>(x_tmp.data(), cils.x_t.data());
        printf("error_bits: %d, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f, rank:%d\n",
               diff, reT3.x[0], reT3.x[1], reT3.x[2], v_norm, reT3.run_time, rank);
    }

}

template<typename scalar, typename index, index m, index n>
long plot_run(int size, int rank) {
    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | ILS | %s ]==================================\n", time_str);
    for (SNR = 35; SNR <= 35; SNR += 30) {
        index d_s_size = d_s.size();
        scalar res[3][200][2] = {}, ber[3][200][2] = {}, tim[4][200][2] = {}, spu[4][200][2] = {};//, t_spu[4][200][2] = {};
        //scalar stm[3][200][2] = {};
        //scalar all_time[7][1000][3][2] = {}; //Method, iter, init, qam

        index verbose = n <= 16, count = 0;
        scalar run_time;
        cils::returnType<scalar, index> reT;
        auto *v_norm_qr = (double *) calloc(1, sizeof(double));

        scalar r, t, b, iter, ser_time, t2, r2, b2, iter2;

        index l = 2; //count for qam.

        vector<scalar> z_omp(n, 0), z_ser(n, 0), z_ini(n, 0), z_tmp(n, 0);
        cils::cils<scalar, index, m, n> cils(k, SNR);
        
        for (index i = 1; i <= max_iter; i++) {
            l = 2;
            for (k = 3; k >= 1; k -= 2) {
                count = k == 1 ? 0 : 1;
                run_time = omp_get_wtime();
                cils.qam = k;
                printf("[ INIT PHASE]\n++++++++++++++++++++++++++++++++++++++\n");
                cils.init_ud();

                /**
                 * -1: QRP, 0: SIC, 1:GRAD
                 */
                for (index init = -1; init <= 1; init++) {
                    printf("[ TRIAL PHASE]++++++++++++++++++++++++++++++++++++++\n");
                    /*
                     * INIT POINT
                     */
                    z_ini.assign(n, 0);
                    if (init == -1) {
                        reT = cils.cils_qrp_serial(z_ini);
                        cout << "Method: QRP, ";
                    } else if (init == 0) {
                        reT = cils.cils_sic_serial(z_ini);
                        cout << "Method: SIC, ";
                    } else {
                        reT = cils.cils_grad_proj(z_ini, search_iter);
                        cout << "Method: GRD, ";
                    }
                    r = reT.info;
                    v_norm_qr[0] = r;
                    //Grad_proj, P = eye(n);
                    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ini.data(), z_tmp.data());
                    b = (scalar) helper::length_nonzeros<scalar, index, n>(z_ini.data(), cils.x_t.data()) /
                        (scalar) n;
                    t = reT.run_time;
                    res[init + 1][0][count] += r;
                    ber[init + 1][0][count] += b;
                    tim[init + 1][0][count] += t;
                    //all_time[0][i][init + 1][count] = t;

                    printf("AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs\n",
                           res[init + 1][0][count] / i, ber[init + 1][0][count] / i, tim[init + 1][0][count] / i,
                           r, b, t);

                    /*
                     * Serial Block SCP Test
                     */
                    z_ser.assign(z_ini.begin(), z_ini.end());
                    reT = cils.cils_scp_block_optimal_serial(z_ser, v_norm_qr[0]);
                    r = reT.info;
                    z_tmp.assign(n, 0);
                    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ser.data(), z_tmp.data());
                    b = (scalar) helper::length_nonzeros<scalar, index, n>(z_ser.data(), cils.x_t.data()) /
                        (scalar) n;
                    ser_time = reT.run_time;
                    res[init + 1][1][count] += r;
                    ber[init + 1][1][count] += b;
                    tim[init + 1][1][count] += ser_time;
                    //all_time[1][i][init + 1][count] = ser_time;

                    printf("Method: SCP_SER, AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, "
                           "RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs\n",
                           res[init + 1][1][count] / i, ber[init + 1][1][count] / i, tim[init + 1][1][count] / i,
                           r, b, ser_time);

                    /*
                     * Parallel Block Babai Test
                     */
                    l = 2;
                    scalar prev_t = INFINITY;
                    for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
                        index _ll = 0;
                        t = r = b = 0;
                        t2 = r2 = b2 = INFINITY;
                        while (true) {
                            z_omp.assign(z_ini.begin(), z_ini.end());
                            reT = cils.cils_scp_block_optimal_omp(z_omp, v_norm_qr[0], n_proc);
                            r = reT.info;
                            z_tmp.assign(n, 0);
                            helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_omp.data(), z_tmp.data());
                            t = reT.run_time;
                            b = (scalar) helper::length_nonzeros<scalar, index, n>(z_omp.data(), cils.x_t.data()) /
                                (scalar) n;

                            t2 = min(t, t2);
                            r2 = min(r, r2);
                            b2 = min(b, b2);
                            _ll++;

                            if (prev_t > t) break;
                            if (_ll == 100) {
                                r = r2;
                                b = b2;
                                t = t2;
                                break; //
                            }
                        }

                        res[init + 1][l][count] += r;
                        ber[init + 1][l][count] += b;
                        tim[init + 1][l][count] += t;
                        //all_time[l][i][init + 1][count] = t;
                        spu[init + 1][l][count] += ser_time / t;

//                            t_spu[init + 1][l][count] +=
//                                    (ser_time + LLL_qr_reT.run_time / coeff) / (t + LLL_qr_reT_omp.run_time / coeff);
                        printf("Method: SCP_OMP, N_PROC: %2d, AVG RES: %8.5f, AVG BER: %8.5f, "
                               "AVG TIME: %8.5fs, RES: %8.5f, BER: %8.5f, SER TIME: %8.5f, OMP TIME: %8.5fs, "
                               "SPEEDUP:%7.3f, AVG SPEEDUP: %7.3f.\n",
                               n_proc, res[init + 1][l][count] / i, ber[init + 1][l][count] / i,
                               tim[init + 1][l][count] / i, r, b, ser_time, t,
                               ser_time / t, spu[init + 1][l][count] / i);//, t_spu[init + 1][l][count] / i);
//                            printf("LLL_QR SER TIME: %8.5fs, LLL_QR OMP_TIME: %8.5fs, LLL_QR SPEEDUP: %7.3f.\n",
//                                   LLL_qr_reT.run_time, LLL_qr_reT_omp.run_time, lll_qr_spu[qr_l][count] / i);

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
                npy_intp dim[3] = {3, 200, 2};
                npy_intp di4[3] = {4, 200, 2};

                scalar proc_nums[l - 2] = {};
                index ll = 0;

                for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
                    proc_nums[ll] = n_proc;
                    ll++;
                }
                npy_intp dpc[1] = {ll};

                PyObject *pRes = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, res);
                PyObject *pBer = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, ber);
                PyObject *pTim = PyArray_SimpleNewFromData(3, di4, NPY_DOUBLE, tim);
                PyObject *pSpu = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, spu);
                PyObject *pPrc = PyArray_SimpleNewFromData(1, dpc, NPY_DOUBLE, proc_nums);
                
                if (pRes == nullptr) printf("[ ERROR] pRes has a problem.\n");
                if (pBer == nullptr) printf("[ ERROR] pBer has a problem.\n");
                if (pTim == nullptr) printf("[ ERROR] pTim has a problem.\n");
                if (pPrc == nullptr) printf("[ ERROR] pPrc has a problem.\n");
                if (pSpu == nullptr) printf("[ ERROR] pSpu has a problem.\n");

                PyObject *sys_path = PySys_GetObject("path");
                PyList_Append(sys_path,
                              PyUnicode_FromString(
                                      "/home/shilei/CLionProjects/babai_asyn/babai_asyn_c++/src/example"));
                pName = PyUnicode_FromString("plot_helper");
                pModule = PyImport_Import(pName);

                if (pModule != nullptr) {
                    pFunc = PyObject_GetAttrString(pModule, "plot_runtime_ud");
                    if (pFunc && PyCallable_Check(pFunc)) {
                        PyObject *pArgs = PyTuple_New(14);
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
                        if (PyTuple_SetItem(pArgs, 4, Py_BuildValue("i", i)) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 5, pRes) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 6, pBer) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 7, pTim) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 8, pPrc) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 9, pSpu) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 10, Py_BuildValue("i", max_proc)) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 11, Py_BuildValue("i", min_proc)) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 12, Py_BuildValue("i", is_constrained)) != 0) {
                            return false;
                        }
                        if (PyTuple_SetItem(pArgs, 13, Py_BuildValue("i", m)) != 0) {
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
    return 0;
}