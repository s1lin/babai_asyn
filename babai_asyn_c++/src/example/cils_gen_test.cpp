
#include "../source/cils.cpp"
#include "../source/cils_ils_search.cpp"
#include "../source/cils_block_search.cpp"
#include "../source/cils_babai_search.cpp"
#include "../source/cils_reduction.cpp"

#include <boost/python.hpp>
#include <boost/python/exception_translator.hpp>
//#include <boost/python/numpy.hpp>
#include <numpy/arrayobject.h>

template<typename scalar, typename index, index n>
void plot_run() {
    for (k = 1; k <= 3; k += 2)
        for (SNR = 15; SNR <= 35; SNR += 20) {
            printf("plot_res-------------------------------------------\n");
            std::cout << "Init, size: " << n << std::endl;
            std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
            vector<index> z_B(n, 0);

            scalar ser_qrd = 0, run_time;
            vector<scalar> bab_res(3, 0), bab_tim(3, 0), bab_ber(3, 0);
            vector<scalar> ser_res(3, 0), ser_tim(3, 0), ser_ber(3, 0);
            scalar omp_res[3][50] = {}, omp_ber[3][50] = {}, omp_tim[3][50] = {}, omp_itr[3][50] = {}, omp_qrd[3][50] = {}, omp_err[3][50] = {};
            cils::returnType<scalar, index> reT, qr_reT, qr_reT_omp;

            for (index i = 0; i < max_iter; i++) {
                run_time = omp_get_wtime();
                if (i % 20 == 0 && i != 0) {
                    for (index init = -1; init <= 1; init++) {
                        printf("++++++++++++++++++++++++++++++++++++++\n");
                        std::cout << "Block, size: " << block_size << std::endl;
                        std::cout << "Init, value: " << init << std::endl;
                        printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5f, Total Time: %.5fs\n",
                               bab_res[init + 1] / max_iter, bab_ber[init + 1] / max_iter, bab_tim[init + 1],
                               ser_qrd / max_iter,
                               (ser_qrd + bab_tim[init + 1]) / max_iter);
                        printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5f, Total Time: %.5fs\n",
                               block_size, ser_res[init + 1] / max_iter, ser_ber[init + 1] / max_iter,
                               ser_tim[init + 1],
                               ser_qrd / max_iter, (ser_qrd + ser_tim[init + 1]) / max_iter);
                        index l = 0;
                        for (index n_proc = min_proc; n_proc <= max_proc + 2 * min_proc; n_proc += min_proc) {
                            printf("Method: ILS_OMP, n_proc: %d, Res :%.5f, BER: %.5f, num_iter: %.5f, Time: %.5fs, Avg Time: %.5fs, "
                                   "Speed up: %.3f, QR Error: %.3f, QR Time: %.5fs, QR SpeedUp: %.3f, Total Time: %.5fs, Total SpeedUp: %.3f\n",
                                   n_proc > max_proc ? max_proc : n_proc, omp_res[init + 1][l] / max_iter,
                                   omp_ber[init + 1][l] / max_iter,
                                   omp_itr[init + 1][l] / max_iter,
                                   omp_tim[init + 1][l], omp_tim[init + 1][l] / max_iter,
                                   ser_tim[init + 1] / omp_tim[init + 1][l],
                                   omp_err[0][l] / max_iter, omp_qrd[0][l] / max_iter,
                                   ser_qrd / omp_qrd[0][l],
                                   (omp_qrd[0][l] + omp_tim[init + 1][l]) / max_iter,
                                   (ser_qrd + ser_tim[init + 1]) / (omp_qrd[0][l] + omp_tim[init + 1][l])
                            );
                            l++;
                        }
                        printf("++++++++++++++++++++++++++++++++++++++\n");
                    }
                }
                cils::cils<scalar, index, n> cils(k, SNR);
                if (i == 0) {
                    cils.init(is_read);
                    cils.init_y();
                    cils.cils_qr_decomposition_omp(0, 1, max_proc);
                    init_guess<scalar, index, n>(0, &z_B, &cils.x_R);
                    cils.cils_block_search_omp(max_proc, num_trials, 0, &d_s, &z_B);
                    continue;
                }

                cils.init(is_read);
                qr_reT = cils.cils_qr_decomposition_serial(0, 1);
                ser_qrd += qr_reT.run_time;
                cils.init_y();
                cils.init_res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &cils.x_t);
                cils.cils_back_solve(&cils.x_R);

                for (index init = -1; init <= 1; init++) {
                    init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                    reT = cils.cils_babai_search_serial(&z_B);
                    bab_res[init + 1] += cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
                    bab_ber[init + 1] += cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
                    bab_tim[init + 1] += reT.run_time;

                    init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                    reT = cils.cils_block_search_serial(&d_s, &z_B);
                    ser_res[init + 1] += cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
                    ser_ber[init + 1] += cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
                    ser_tim[init + 1] += reT.run_time;

                    index l = 0;
                    for (index n_proc = min_proc; n_proc <= max_proc + 2 * min_proc; n_proc += min_proc) {
//                if (init == -1) {
//                    qr_reT_omp = cils.cils_qr_decomposition_omp(0, 1, n_proc > max_proc ? max_proc : n_proc);
//                    omp_qrd[init + 1][l] += qr_reT_omp.run_time;
//                    omp_err[init + 1][l] += qr_reT_omp.num_iter;
//                }
                        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                        reT = cils.cils_block_search_omp(n_proc > max_proc ? max_proc : n_proc, num_trials, init, &d_s,
                                                         &z_B);
                        omp_res[init + 1][l] += cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
                        omp_ber[init + 1][l] += cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
                        omp_tim[init + 1][l] += reT.run_time;
                        omp_itr[init + 1][l] += reT.num_iter;
                        l++;
                    }
                }
                run_time = omp_get_wtime() - run_time;
                printf("%d-Time: %.5fs, ", i, run_time);
                cout.flush();
            }
            max_iter--;
            for (index init = -1; init <= 1; init++) {
                printf("++++++++++++++++++++++++++++++++++++++\n");
                std::cout << "Block, size: " << block_size << std::endl;
                std::cout << "Init, value: " << init << std::endl;
                printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Avg Solve Time: %.5fs, qr_time: %.5f, Total Time: %.5fs\n",
                       bab_res[init + 1] / max_iter, bab_ber[init + 1] / max_iter, bab_tim[init + 1] / max_iter,
                       ser_qrd / max_iter,
                       (ser_qrd + bab_tim[init + 1]) / max_iter);
                printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Avg Solve Time: %.5fs, qr_time: %.5f, Total Time: %.5fs\n",
                       block_size, ser_res[init + 1] / max_iter, ser_ber[init + 1] / max_iter,
                       ser_tim[init + 1] / max_iter,
                       ser_qrd / max_iter, (ser_qrd + ser_tim[init + 1]) / max_iter);
                index l = 0;
                for (index n_proc = min_proc; n_proc <= max_proc + 2 * min_proc; n_proc += min_proc) {
                    printf("Method: ILS_OMP, n_proc: %d, Res :%.5f, BER: %.5f, num_iter: %.5f, Time: %.5fs, Avg Time: %.5fs, "
                           "Speed up: %.3f, QR Error: %.3f, QR Time: %.5fs, QR SpeedUp: %.3f, Total Time: %.5fs, Total SpeedUp: %.3f\n",
                           n_proc > max_proc ? max_proc : n_proc, omp_res[init + 1][l] / max_iter,
                           omp_ber[init + 1][l] / max_iter,
                           omp_itr[init + 1][l] / max_iter,
                           omp_tim[init + 1][l], omp_tim[init + 1][l] / max_iter,
                           ser_tim[init + 1] / omp_tim[init + 1][l],
                           omp_err[0][l] / max_iter, omp_qrd[0][l] / max_iter,
                           ser_qrd / omp_qrd[0][l],
                           (omp_qrd[0][l] + omp_tim[init + 1][l]) / max_iter,
                           (ser_qrd + ser_tim[init + 1]) / (omp_qrd[0][l] + omp_tim[init + 1][l])
                    );
                    l++;
                }
                printf("++++++++++++++++++++++++++++++++++++++\n");
            }
            printf("End of current TASK.\n");
            printf("-------------------------------------------\n");
        }
}

template<typename scalar, typename index, index n>
long test_ils_search() {
    std::cout << "Init, size: " << n << std::endl;

    cils::cils<scalar, index, n> cils(k, SNR);
    index init = 0;
    scalar error = 0;
    cils::returnType<scalar, index> reT, qr_reT = {nullptr, 0, 0}, qr_reT_omp = {nullptr, 0, 0};
    for (index i = 0; i < max_iter; i++) {

        cils.init(is_read);

        if (!is_read) {
            scalar time = omp_get_wtime();
//            CPyInstance hInstance;
//
//
//            CPyObject pName = PyUnicode_FromString("pyemb3");
//            CPyObject pModule = PyImport_Import(pName);
//            if (pModule) {
//                CPyObject pFunc = PyObject_GetAttrString(pModule, "qr");
//                if (pFunc && PyCallable_Check(pFunc)) {
//                    import_array1(-1);
//                    int l = 10;
//                    double *h_slopes, *h_result;
//                    h_slopes = (double *)malloc(l * sizeof(double));
//                    h_result = (double *)malloc(l * sizeof(double));
//                    npy_intp dim[] = {l};
//                    for (int ii = 0; ii < l; ii++) h_slopes[ii] = ii;
//                    npy_double data_slopes[l];
//                    PyObject *pVec = PyArray_SimpleNewFromData( 1, dim, NPY_DOUBLE, h_slopes );
//                    if (pVec == NULL) printf("Cannot fill pVec.\n");
//
//                    CPyObject pValue = PyObject_CallObject(pFunc, pVec);
//
//                    printf("C: getInteger() = %ld\n", PyLong_AsLong(pValue));
//                } else {
//                    printf("ERROR: function getInteger()\n");
//                }
//
//            } else {
//                printf("ERROR: Module not imported\n");
//            }
            PyObject * pName, *pModule, *pDict, *pFunc;
            PyObject * pArgs, *pValue, *pVec;

            Py_Initialize();

// this macro is defined by NumPy and must be included
            import_array();

//            int nLenslet = 10;
//            double h_slopes[nLenslet][nLenslet];
//            for (int ii = 0; ii < nLenslet; ii++){
//                h_slopes[ii][ii] = ii;
//            }
            npy_intp dim[1] = {cils.A->size};
            pVec = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, cils.A->x);
            if (pVec == NULL) printf("There is a problem.\n");

// load the python file
            PyObject * pval;
            PyObject * sys_path = PySys_GetObject("path");
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
                        PyObject *q, *r;
                        PyArg_ParseTuple(pValue, "O|O", &q, &r);
////                        PyArg_ParseTuple(pValue, "1", &r);
//                        printf("C: q = %ld\n", PyArray_NDIM(q));
//                        printf("C: r = %ld\n", PyArray_NDIM(r));
//                    }
//                }
//                pFunc = PyObject_GetAttrString(pModule, "get_R");
//                if (pFunc && PyCallable_Check(pFunc)) {
//                    pValue = PyObject_CallObject(pFunc, nullptr);
//                    if (pValue != NULL) {
//                        auto *np_ret = reinterpret_cast<PyArrayObject *>(pValue);
////                        if (PyArray_NDIM(np_ret) != n - 1){
////
////                        }
////                        printf("C: qr = %ld\n", PyLong_AsLong(pValue));
////                        printf("C: qr = %ld\n", PyArray_NDIM(np_ret));
//                        cils.R->x = reinterpret_cast<scalar *>(PyArray_DATA(r));
                        cils.Q->x = reinterpret_cast<scalar *>(PyArray_DATA(q));
                        cils.R->x = reinterpret_cast<scalar *>(PyArray_DATA(r));
//                        cout << "Printing output array" << r1 << r2 << endl;
//                        for (int ii = 0; ii < n; ii++) {
//                            for (int jj = 0; jj < n; jj++) {
//                                cout << cils.R->x[jj + ii * n] << ' ';
//                                cout << cils.Q->x[jj + ii * n] << ' ';
//                            }
//                            cout << endl;
//                        }

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

            time = omp_get_wtime() - time;
            cout << time;
            cout.flush();

//            time = omp_get_wtime();
//            qr_reT = cils.cils_qr_decomposition_omp(0, 0, max_proc);
//            time = omp_get_wtime() - time;
//            cout << time;
//            cout.flush();

            cils.init_y();
//            error = cils::qr_validation<scalar, index, n>(cils.A, cils.Q, cils.R, cils.R_A, 0, 1);
            cils.init_res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &cils.x_t);
            cils.cils_back_solve(&cils.x_R);
        }
        printf("init_res: %.5f, sigma: %.5f, qr_error: %d\n", cils.init_res, cils.sigma, error);
        cout.flush();

        vector<index> z_B(n, 0);
        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);

        reT = cils.cils_babai_search_serial(&z_B);
        auto res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
        auto ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
        printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5f Total Time: %.5fs\n",
               res, ber, reT.run_time, qr_reT.run_time, qr_reT.run_time + reT.run_time);
        scalar bab_tim_constrained = reT.run_time;

//        if(res < 100) continue;

        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
        reT = cils.cils_block_search_serial(&d_s, &z_B);
        res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
        ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
        printf("\nMethod: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5f Total Time: %.5fs\n",
               block_size, res, ber, reT.run_time, qr_reT.run_time, qr_reT.run_time + reT.run_time);
        scalar ils_tim_constrained = reT.run_time;
        cout.flush();

        for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
//            qr_reT_omp = cils.cils_qr_decomposition_omp(0, 1, n_proc > max_proc ? max_proc : n_proc);
//            cils.init_y();
//            cils.init_res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &cils.x_t);
//            init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
//            reT = cils.cils_babai_search_omp(n_proc > max_proc ? max_proc : n_proc, num_trials, &z_B);
//            res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
//            ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
//            printf("Method: BAB_OMP, n_proc: %d, Res: %.5f, BER: %.5f, Num_iter: %d, "
//                   "Solve Time: %.5fs, Solve SpeedUp: %.3f, "
//                   "QR Error: %d, QR Time: %.5fs, QR SpeedUp: %.3f, "
//                   "Total Time: %.5fs, Total SpeedUp: %.3f\n",
//                   n_proc, res, ber, reT.num_iter, reT.run_time, (bab_tim_constrained / reT.run_time),
//                   qr_reT_omp.num_iter, qr_reT_omp.run_time, qr_reT.run_time / qr_reT_omp.run_time,
//                   reT.run_time + qr_reT_omp.run_time,
//                   (bab_tim_constrained + qr_reT.run_time) / (qr_reT_omp.run_time + reT.run_time));

            init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
            reT = cils.cils_block_search_omp(n_proc > max_proc ? max_proc : n_proc, num_trials, init, &d_s, &z_B);
            res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
            ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
            printf("Method: ILS_OMP, n_proc: %d, Res: %.5f, BER: %.5f, Num_iter: %d, "
                   "Solve Time: %.5fs, Solve SpeedUp: %.3f, "
                   "QR Error: %d, QR Time: %.5fs, QR SpeedUp: %.3f, "
                   "Total Time: %.5fs, Total SpeedUp: %.3f\n",
                   n_proc, res, ber, reT.num_iter, reT.run_time, (ils_tim_constrained / reT.run_time),
                   qr_reT_omp.num_iter, qr_reT_omp.run_time, qr_reT.run_time / qr_reT_omp.run_time,
                   reT.run_time + qr_reT_omp.run_time,
                   (ils_tim_constrained + qr_reT.run_time) / (qr_reT_omp.run_time + reT.run_time)
            );
        }
    }

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
            cils.init(is_read);
            auto reT = cils.cils_qr_decomposition_omp(0, 1, max_proc);
            cils.init_y();
            printf("init_res: %.5f, sigma: %.5f, qr time: %.5fs\n", cils.init_res, cils.sigma, reT.run_time);

            vector<index> z_B(n, 0);
            reT = cils.cils_babai_search_serial(&z_B);
            scalar res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
            scalar ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
            printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs\n",
                   res, ber, reT.run_time);

            z_B.assign(n, 0);
            reT = cils.cils_block_search_serial(&d_s, &z_B);
            res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
            ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
            printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs\n",
                   block_size, res, ber, reT.run_time);

            for (index init = -1; init <= 1; init++) {
                cout << "init," << init << "\n";
                init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                reT = cils.cils_block_search_serial(&d_s, &z_B);
                res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
                ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);

                printf("Method: ILS_SER, Block size: %d, Res: %.5f, Ber: %.5f, Time: %.5fs\n",
                       block_size, res, ber, reT.run_time);
                for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
                    cout << d_s[d_s.size() - 1] << "," << n_proc << ",";
                    std::cout.flush();
                    for (index nswp = 1; nswp < max_iter; nswp++) {
                        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                        reT = cils.cils_block_search_omp(n_proc > max_proc ? max_proc : n_proc, nswp, init, &d_s, &z_B);
                        res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
                        ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
                        printf("diff=%d, res=%.5f, ber=%.5f, ",
                               reT.num_iter > (N / block_size) ? (N / block_size) : reT.num_iter, res, ber);
                    }
                    cout << endl;
                }
            }

            printf("End of current TASK.\n");
            printf("-------------------------------------------\n");
        }
}
