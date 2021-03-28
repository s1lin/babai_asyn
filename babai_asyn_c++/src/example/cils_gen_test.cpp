
#include "../source/cils.cpp"
#include "../source/cils_ils_search.cpp"
#include "../source/cils_block_search.cpp"
#include "../source/cils_babai_search.cpp"
#include "../source/cils_reduction.cpp"

template<typename scalar, typename index, index n>
long plot_run() {

    for (SNR = 35; SNR <= 35; SNR += 30) {
        scalar res[3][50][2] = {}, ber[3][50][2] = {}, tim[3][50][2] = {};
        scalar itr[3][50][2] = {}, qrd[3][50][2] = {}, err[3][50][2] = {};
        index count = 0, l = 2;
        for (k = 1; k <= 3; k += 2) {
            printf("plot_res-------------------------------------------\n");
            std::cout << "Init, size: " << n << std::endl;
            std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
            vector<index> z_B(n, 0);

            scalar ser_qrd = 0, run_time;

            cils::returnType<scalar, index> reT, qr_reT, qr_reT_omp;
            l = 2;
            for (index i = 0; i < max_iter; i++) {
                run_time = omp_get_wtime();
                if (i % 20 == 0 && i != 0) {
                    for (index init = -1; init <= 1; init++) {
                        printf("++++++++++++++++++++++++++++++++++++++\n");
                        std::cout << "Block, size: " << block_size << std::endl;
                        std::cout << "Init, value: " << init << std::endl;
                        printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5f, Total Time: %.5fs\n",
                               res[init + 1][0][count] / max_iter, ber[init + 1][0][count] / max_iter, tim[init + 1][0][count],
                               ser_qrd / max_iter,
                               (ser_qrd + tim[init + 1][0][count]) / max_iter);
                        printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5f, Total Time: %.5fs\n",
                               block_size, res[init + 1][1][count] / max_iter, ber[init + 1][1][count] / max_iter,
                               tim[init + 1][1][count],
                               ser_qrd / max_iter, (ser_qrd + tim[init + 1][1][count]) / max_iter);
                        l = 2;
                        for (index n_proc = min_proc; n_proc <= max_proc + 2 * min_proc; n_proc += min_proc) {
                            printf("Method: ILS_OMP, n_proc: %d, Res :%.5f, BER: %.5f, num_iter: %.5f, Time: %.5fs, Avg Time: %.5fs, "
                                   "Speed up: %.3f, QR Error: %.3f, QR Time: %.5fs, QR SpeedUp: %.3f, Total Time: %.5fs, Total SpeedUp: %.3f\n",
                                   n_proc > max_proc ? max_proc : n_proc, res[init + 1][l][count] / max_iter,
                                   ber[init + 1][l][count] / max_iter,
                                   itr[init + 1][l][count] / max_iter,
                                   tim[init + 1][l][count], tim[init + 1][l][count] / max_iter,
                                   tim[init + 1][1][count] / tim[init + 1][l][count],
                                   err[0][l][count] / max_iter, qrd[0][l][count] / max_iter,
                                   ser_qrd / qrd[0][l][count],
                                   (qrd[0][l][count] + tim[init + 1][l][count]) / max_iter,
                                   (ser_qrd + tim[init + 1][1][count]) / (qrd[0][l][count] + tim[init + 1][l][count])
                            );
                            l++;
                        }
                        printf("++++++++++++++++++++++++++++++++++++++\n");
                    }
                }

                cils::cils<scalar, index, n> cils(k, SNR);

                cils.init(is_read);
                if (!is_read) {
                    qr_reT = cils.cils_qr_decomposition_reduction();
                    cils.init_R_A_reduction();

                    cils.init_res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &cils.x_t);
                    cils.cils_back_solve(&cils.x_R);

                }

                ser_qrd += qr_reT.run_time;

                for (index init = -1; init <= 1; init++) {
                    init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                    reT = cils.cils_babai_search_serial(&z_B);
                    res[init + 1][0][count] += cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
                    ber[init + 1][0][count] += cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
                    tim[init + 1][0][count] += reT.run_time;

                    init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                    reT = cils.cils_block_search_serial(&d_s, &z_B);
                    res[init + 1][1][count] += cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
                    ber[init + 1][1][count] += cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
                    tim[init + 1][1][count] += reT.run_time;

                    l = 2;
                    for (index n_proc = min_proc; n_proc <= max_proc + 2 * min_proc; n_proc += min_proc) {
                        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                        reT = cils.cils_block_search_omp(n_proc > max_proc ? max_proc : n_proc,
                                                         num_trials, init, &d_s, &z_B);
                        res[init + 1][l][count] += cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
                        ber[init + 1][l][count] += cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
                        tim[init + 1][l][count] += reT.run_time;
                        itr[init + 1][l][count] += reT.num_iter;
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
                printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5f, Total Time: %.5fs\n",
                       res[init + 1][0][count] / max_iter, ber[init + 1][0][count] / max_iter, tim[init + 1][0][count],
                       ser_qrd / max_iter,
                       (ser_qrd + tim[init + 1][0][count]) / max_iter);
                printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5f, Total Time: %.5fs\n",
                       block_size, res[init + 1][1][count] / max_iter, ber[init + 1][1][count] / max_iter,
                       tim[init + 1][1][count],
                       ser_qrd / max_iter, (ser_qrd + tim[init + 1][1][count]) / max_iter);
                l = 2;
                for (index n_proc = min_proc; n_proc <= max_proc + 2 * min_proc; n_proc += min_proc) {
                    printf("Method: ILS_OMP, n_proc: %d, Res :%.5f, BER: %.5f, num_iter: %.5f, Time: %.5fs, Avg Time: %.5fs, "
                           "Speed up: %.3f, QR Error: %.3f, QR Time: %.5fs, QR SpeedUp: %.3f, Total Time: %.5fs, Total SpeedUp: %.3f\n",
                           n_proc > max_proc ? max_proc : n_proc, res[init + 1][l][count] / max_iter,
                           ber[init + 1][l][count] / max_iter,
                           itr[init + 1][l][count] / max_iter,
                           tim[init + 1][l][count], tim[init + 1][l][count] / max_iter,
                           tim[init + 1][1][count] / tim[init + 1][l][count],
                           err[0][l][count] / max_iter, qrd[0][l][count] / max_iter,
                           ser_qrd / qrd[0][l][count],
                           (qrd[0][l][count] + tim[init + 1][l][count]) / max_iter,
                           (ser_qrd + tim[init + 1][1][count]) / (qrd[0][l][count] + tim[init + 1][l][count])
                    );
                    l++;
                }
                printf("++++++++++++++++++++++++++++++++++++++\n");
            }
            count++;
            max_iter++;
        }
        printf("End of current TASK.\n");
        printf("-------------------------------------------\n");

        PyObject * pName, *pModule, *pFunc;
        PyObject * pArgs, *pValue, *pRes, *pBer, *pTim, *pItr;
        Py_Initialize();
        import_array();
        npy_intp dim[3] = {3, 50, 2};

        pRes = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, res);
        pBer = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, ber);
        pTim = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, tim);
        pItr = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, itr);
        if (pRes == NULL) printf("pRes has a problem.\n");
        if (pBer == NULL) printf("pBer has a problem.\n");
        if (pTim == NULL) printf("pTim has a problem.\n");
        if (pItr == NULL) printf("pItr has a problem.\n");

        PyObject * sys_path = PySys_GetObject("path");
        PyList_Append(sys_path,
                      PyUnicode_FromString("/home/shilei/CLionProjects/babai_asyn/babai_asyn_c++/src/example"));
        pName = PyUnicode_FromString("plot_helper");
        pModule = PyImport_Import(pName);

        if (pModule != NULL) {
            pFunc = PyObject_GetAttrString(pModule, "plot_runtime");
            if (pFunc && PyCallable_Check(pFunc)) {
                pArgs = PyTuple_New(8);
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
                if (PyTuple_SetItem(pArgs, 4, pRes) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 5, pBer) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 6, pTim) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 7, pItr) != 0) {
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
//            qr_reT = cils.cils_qr_decomposition_omp(0, 1, max_proc);
//            qr_reT = cils.cils_qr_decomposition_py(0, 1);
//            cils.init_y();
            qr_reT = cils.cils_qr_decomposition_reduction();
            cils.init_R_A_reduction();

            cils.init_res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &cils.x_t);
            cils.cils_back_solve(&cils.x_R);

        }
        printf("init_res: %.5f, sigma: %.5f, qr_error: %.1f\n", cils.init_res, cils.sigma, error);
        cout.flush();

        vector<index> z_B(n, 0);
        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);

        reT = cils.cils_babai_search_serial(&z_B);
//        cils::vector_permutation<scalar, index, n>(cils.Z, &z_B);
        auto res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
        auto ber = cils::find_bit_error_rate<scalar, index, n>(&z_B, &cils.x_t, k);
        printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5fs Total Time: %.5fs\n",
               res, ber, reT.run_time, qr_reT.run_time, qr_reT.run_time + reT.run_time);
        scalar bab_tim_constrained = reT.run_time;
//        cils::display_vector<scalar, index>(&z_B);
//        cils::display_vector<scalar, index>(&cils.x_t);


        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
        reT = cils.cils_block_search_serial(&d_s, &z_B);
//        cils::vector_permutation<scalar, index, n>(cils.Z, reT.x);
        res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, reT.x);
        ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);


        printf("\nMethod: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5fs Total Time: %.5fs\n",
               block_size, res, ber, reT.run_time, qr_reT.run_time, qr_reT.run_time + reT.run_time);
        scalar ils_tim_constrained = reT.run_time;
        cout.flush();

        for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
//            qr_reT_omp = cils.cils_qr_decomposition_omp(0, 1, n_proc > max_proc ? max_proc : n_proc);
//            cils.init_y();
//            cils.init_res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &cils.x_t);
//            init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
//            reT = cils.cils_babai_search_omp(n_proc > max_proc ? max_proc : n_proc, num_trials, &z_B);
//            res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
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
//            cils::vector_permutation<scalar, index, n>(cils.Z, reT.x);
            res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, reT.x);
            ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
//            cout<<endl;
//            cils::display_vector<scalar, index>(&z_B);
//            cils::display_vector<scalar, index>(&cils.x_t);
            printf("Method: ILS_OMP, n_proc: %d, Res: %.5f, BER: %.5f, Num_iter: %.1f, "
                   "Solve Time: %.5fs, Solve SpeedUp: %.3f, "
                   "QR Error: %.5f, QR Time: %.5fs, QR SpeedUp: %.3f, "
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
            auto reT = cils.cils_qr_decomposition_reduction();
            cils.init_R_A_reduction();
            printf("init_res: %.5f, sigma: %.5f, qr time: %.5fs\n", cils.init_res, cils.sigma, reT.run_time);

            vector<index> z_B(n, 0);
            reT = cils.cils_babai_search_serial(&z_B);
            scalar res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
            scalar ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
            printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs\n",
                   res, ber, reT.run_time);

            z_B.assign(n, 0);
            reT = cils.cils_block_search_serial(&d_s, &z_B);
            res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
            ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
            printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs\n",
                   block_size, res, ber, reT.run_time);

            for (index init = -1; init <= 1; init++) {
                cout << "init," << init << "\n";
                init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                reT = cils.cils_block_search_serial(&d_s, &z_B);
                res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
                ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);

                printf("Method: ILS_SER, Block size: %d, Res: %.5f, Ber: %.5f, Time: %.5fs\n",
                       block_size, res, ber, reT.run_time);
                for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
                    cout << d_s[d_s.size() - 1] << "," << n_proc << ",";
                    std::cout.flush();
                    for (index nswp = 1; nswp < max_iter; nswp++) {
                        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                        reT = cils.cils_block_search_omp(n_proc > max_proc ? max_proc : n_proc, nswp, init, &d_s, &z_B);
                        res = cils::find_residual<scalar, index, n>(cils.A, cils.y_L, &z_B);
                        ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
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
