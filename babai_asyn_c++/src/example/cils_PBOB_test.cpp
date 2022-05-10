#include <Python.h>
#include <numpy/arrayobject.h>

#include "../source/CILS.cpp"
#include "../source/CILS_Reduction.cpp"
#include "../source/CILS_SECH_Search.cpp"
#include "../source/CILS_OLM.cpp"

void init_z_hat(b_vector &z_hat, b_vector &x, int init, double mean) {
    z_hat.clear();
    if (init == 1) std::fill(z_hat.x.begin(), z_hat.x.end(), round(mean));
    if (init == 2) z_hat.assign(x);
}

template<typename scalar, typename index>
long test_PBNP(int size_n, bool is_local) {

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | BNP | %s ]==================================\n", time_str);
    cout.flush();

    index d = 0, l = 0, num_trial = 200, k, c = 0, n = size_n;
    scalar t_qr[6][200][6][2] = {}, t_aspl[6][200][6][2] = {}, t_itr[6][200][6][2] = {};
    scalar t_bnp[6][200][6][2], t_ber[6][200][6][2] = {}, run_time;

    cils::CILS<scalar, index> cils;
    cils::CILS_Reduction<scalar, index> reduction(cils), reduction2(cils);

    cils.is_local = is_local;
    b_vector x_ser, x_lll, x_r;
    for (int t = 0; t < num_trial; t++) {
        run_time = omp_get_wtime();
        index s = 0;
        for (int snr = 0; snr <= 50; snr += 10) {
            k = 0;
            for (int qam = 1; qam <= 3; qam += 2) {
                x_ser.resize(n, false);
                x_lll.resize(n, false);

                printf("--------ITER: %d, SNR: %d, QAM: %d, n: %d-------\n", t + 1, snr, (int) pow(4, qam), n);
                cils::init_PBNP(cils, n, snr, qam, c);
                cils::returnType<scalar, index> reT;
                reduction.reset(cils);
                reT = reduction.aspl();
                t_qr[s][t][0][k] = reT.run_time;
                t_aspl[s][t][0][k] = reT.info;
                printf("ASPL: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                       reT.run_time, reT.info, reT.info + reT.run_time);

                reduction.reset(cils);
                reT = reduction.plll();
                t_qr[s][t][1][k] = reT.run_time;
                t_aspl[s][t][1][k] = reT.info;
                printf("PLLL: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                       reT.run_time, reT.info, reT.info + reT.run_time);

                cils::CILS_OLM<scalar, index> olm(cils, x_ser, reduction.R, reduction.y);

                l = 1;
                for (index n_proc = 5; n_proc <= 25; n_proc += 5) {
                    l++;
                    reduction2.reset(cils);
                    reT = reduction2.paspl(n_proc < 10 ? n_proc : 10);
                    t_qr[s][t][l][k] = reT.run_time;
                    t_aspl[s][t][l][k] = reT.info;
                    printf("PASPL: CORE: %3d, QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f, "
                           "SPUQR: %8.4f, SPUASPL: %8.4f, SPUPLLL: %8.4f, SPUTOTAL:%8.4f\n",
                           n_proc < 10 ? n_proc : 10, reT.run_time, reT.info, reT.info + reT.run_time,
                           t_qr[s][t][0][k] / reT.run_time, t_aspl[s][t][0][k] / reT.info,
                           t_aspl[s][t][1][k] / reT.info,
                           (t_qr[s][t][0][k] + t_aspl[s][t][0][k]) / (reT.run_time + reT.info)
                    );
                }

                scalar r = helper::find_residual<scalar, index>(cils.A, cils.x_t, cils.y);
                init_z_hat(olm.z_hat, x_r, 1, (scalar) cils.upper / 2.0);

                reT = olm.bnp();
                projection(reduction.Z, olm.z_hat, x_lll, 0, cils.upper);
                t_ber[s][t][0][k] = helper::find_bit_error_rate<scalar, index>(x_lll, cils.x_t, cils.qam);
                t_bnp[s][t][0][k] = reT.run_time;
                scalar res = helper::find_residual<scalar, index>(cils.A, x_lll, cils.y);
                printf("BNP: BER: %8.5f, RES: %8.4f, TIME: %8.4f\n", t_ber[s][t][0][k], res,
                       t_bnp[s][t][0][k]);

                l = 0;
                scalar total = t_bnp[s][t][0][k] + t_qr[s][t][0][k] + t_aspl[s][t][0][k];
                for (index n_proc = 5; n_proc <= 25; n_proc += 5) {
                    l++;
                    init_z_hat(olm.z_hat, x_r, 1, (int) cils.upper / 2);
                    reT = olm.pbnp2(n_proc < 10 ? n_proc : 10, 10, 1);
                    projection(reduction.Z, olm.z_hat, x_lll, 0, cils.upper);
                    t_ber[s][t][l][k] = helper::find_bit_error_rate<scalar, index>(x_lll, cils.x_t, cils.qam);
                    t_bnp[s][t][l][k] = reT.run_time;
                    t_itr[s][t][l][k] = reT.info;
                    res = helper::find_residual<scalar, index>(cils.A, x_lll, cils.y);
                    printf("PBNP: CORE: %3d, ITER: %4d, BER: %8.5f, RES: %8.4f, TIME: %8.4f, "
                           "BNP SPU: %8.4f, TOTAL SPU: %8.4f\n",
                           n_proc < 10 ? n_proc : 10, (int) reT.info, t_ber[s][t][l][k], res,
                           t_bnp[s][t][l][k],
                           t_bnp[s][t][0][k] / t_bnp[s][t][l][k],
                           total / (t_bnp[s][t][l][k] + t_qr[s][t][l][k] + t_aspl[s][t][l][k]));
                }
                k++;
            }
            s++;
        }
        run_time = omp_get_wtime() - run_time;
        printf("++++++++++++++++++++++++++++++++++++++\n Trial %d, Elapsed Time: %.5fs. \n"
               "++++++++++++++++++++++++++++++++++++++\n", t, run_time);
        cout.flush();
        printf("\n---------------------\nITER:%d\n---------------------\n", t);
        //if (t % 10 == 0) {//
        PyObject *pName, *pModule, *pFunc;
        PyObject *pArgs, *pValue;
        Py_Initialize();
        if (_import_array() < 0)
            PyErr_Print();

        npy_intp di5[5] = {6, 200, 6, 2};

        PyObject *pQRT = PyArray_SimpleNewFromData(4, di5, NPY_DOUBLE, t_qr);
        PyObject *pLLL = PyArray_SimpleNewFromData(4, di5, NPY_DOUBLE, t_aspl);
        PyObject *pBNP = PyArray_SimpleNewFromData(4, di5, NPY_DOUBLE, t_bnp);
        PyObject *pBER = PyArray_SimpleNewFromData(4, di5, NPY_DOUBLE, t_ber);
        PyObject *pITR = PyArray_SimpleNewFromData(4, di5, NPY_DOUBLE, t_itr);

        if (pQRT == nullptr) printf("[ ERROR] pQRT has a problem.\n");
        if (pLLL == nullptr) printf("[ ERROR] pLLL has a problem.\n");
        if (pBNP == nullptr) printf("[ ERROR] pBNP has a problem.\n");
        if (pBER == nullptr) printf("[ ERROR] pBER has a problem.\n");
        if (pITR == nullptr) printf("[ ERROR] pITR has a problem.\n");

        PyObject *sys_path = PySys_GetObject("path");
        if (cils.is_local)
            PyList_Append(sys_path, PyUnicode_FromString(
                    "/home/shilei/CLionProjects/Reference/babai_asyn/babai_asyn_c++/src/plot"));
        else
            PyList_Append(sys_path, PyUnicode_FromString("./"));

        pName = PyUnicode_FromString("plot_bnp");
        pModule = PyImport_Import(pName);

        if (pModule != nullptr) {
            pFunc = PyObject_GetAttrString(pModule, "save_data");
//                if (cils.is_local)
//                    pFunc = PyObject_GetAttrString(pModule, "plot_bnp");

            if (pFunc && PyCallable_Check(pFunc)) {
                pArgs = PyTuple_New(9);
                if (PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", n)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", t)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", 0)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", 0)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 4, pQRT) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 5, pLLL) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 6, pBNP) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 7, pBER) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 8, pITR) != 0) {
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
        //}

    }

    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");

    return 0;

}

template<typename scalar, typename index>
long test_PBOB(int n, int nob, int c, bool is_local) {

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | BOB | %s ]==================================\n", time_str);
    cout.flush();

    index d = 0, l = 0, num_trial = 200, constrain = c != 0, qam = 3;
    scalar t_qr[5][200][10][2] = {}, t_aspl[5][200][10][2] = {}, t_bnp[5][200][10][2] = {}, t_ber[5][200][10][2] = {}, run_time;
    index SNRs[4] = {10, 20, 30, 40};

    cils::CILS<scalar, index> cils;
    cils::CILS_Reduction<scalar, index> reduction(cils), reduction2(cils);
    cils::returnType<scalar, index> reT;
    cils.is_local = is_local;
    cils.is_constrained = constrain;
    cils.block_size = n / nob;
    cils.spilt_size = 0;
    if(cils.block_size == 20)
        cils.spilt_size = 2;

    b_vector x_ser, x_lll, x_r;

    for (int t = 0; t < num_trial; t++) {
        for (int s = 0; s < 4; s++) {
            run_time = omp_get_wtime();

            for (int k = 0; k < 1; k++) {
                cils::init_PBNP(cils, n, SNRs[s], qam, c);
                cils.init_d();
                x_ser.resize(n, false);
                x_lll.resize(n, false);

                printf("--------ITER: %d, SNR: %d, QAM: %d, SIZE: %d-------\n", t + 1, SNRs[s], qam, n);

                reduction.reset(cils);
                if (constrain)
                    reT = reduction.aspl_p();
                else
                    reT = reduction.aspl();
                t_qr[s][t][0][k] = reT.run_time;
                t_aspl[s][t][0][k] = reT.info;
//                reduction.reset(cils);
//                reduction.mgs_qr();
                if (constrain)
                    printf("ASPL-P: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                           reT.run_time, reT.info, reT.info + reT.run_time);
                else
                    printf("ASPL: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                           reT.run_time, reT.info, reT.info + reT.run_time);

                cils::CILS_OLM<scalar, index> olm(cils, x_ser, reduction.R, reduction.y);

                l = 0;
                for (index n_proc = 5; n_proc <= 15; n_proc += 5) {
                    l++;
                    reduction2.reset(cils);
                    if (constrain)
                        reT = reduction2.paspl_p(n_proc == 0 ? 1 : n_proc);
                    else
                        reT = reduction2.paspl(n_proc);

                    t_qr[s][t][l][k] = reT.run_time;
                    t_aspl[s][t][l][k] = reT.info;
                    if (constrain)
                        printf("PASPL-P: CORE: %3d, QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f, "
                               "SPUQR: %8.4f, SPUASPL: %8.4f, 3, SPUTOTAL:%8.4f\n",
                               n_proc, reT.run_time, reT.info, reT.info + reT.run_time,
                               t_qr[s][t][0][k] / reT.run_time, t_aspl[s][t][0][k] / reT.info,
                               t_aspl[s][t][1][k] / reT.info,
                               (t_qr[s][t][0][k] + t_aspl[s][t][0][k]) / (reT.run_time + reT.info));
                    else
                        printf("PASPL: CORE: %3d, QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f, "
                               "SPUQR: %8.4f, SPUASPL: %8.4f, SPUPLLL: %8.4f, SPUTOTAL:%8.4f\n",
                               n_proc, reT.run_time, reT.info, reT.info + reT.run_time,
                               t_qr[s][t][0][k] / reT.run_time, t_aspl[s][t][0][k] / reT.info,
                               t_aspl[s][t][1][k] / reT.info,
                               (t_qr[s][t][0][k] + t_aspl[s][t][0][k]) / (reT.run_time + reT.info));
                }
                cout.flush();
//                scalar r = helper::find_residual<scalar, index>(cils.A, cils.x_t, cils.y);

                init_z_hat(olm.z_hat, x_r, 1, (scalar) cils.upper / 2.0);
                reT = olm.bnp();
                projection(reduction.Z, olm.z_hat, x_lll, 0, cils.upper);
                scalar ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_lll, qam);
                t_bnp[s][t][0][k] = reT.run_time;
                t_ber[s][t][0][k] = ber;
                printf("BNP: BER: %8.4f, TIME: %8.4f\n", ber, t_bnp[s][t][0][k]);
                cout.flush();

                init_z_hat(olm.z_hat, x_r, 1, (scalar) cils.upper / 2.0);
                reT = olm.pbocb_test(1, 10, 0);
                projection(reduction.Z, olm.z_hat, x_lll, 0, cils.upper);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_lll, qam);
                t_bnp[s][t][1][k] = reT.run_time;
                t_ber[s][t][1][k] = ber;
                printf("BOB: BER: %8.4f, TIME: %8.4f\n", ber, t_bnp[s][t][1][k]);
                cout.flush();

                l = 1;
                scalar total = t_bnp[s][t][1][k] + t_qr[s][t][0][k] + t_aspl[s][t][0][k];
                for (index n_proc = 5; n_proc <= 15; n_proc += 5) {
                    l++;
                    init_z_hat(olm.z_hat, x_r, 1, (int) cils.upper / 2);
                    reT = olm.pbocb_test(n_proc, 10, 0);
                    projection(reduction.Z, olm.z_hat, x_lll, 0, cils.upper);
                    ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_lll, qam);
                    t_bnp[s][t][l][k] = reT.run_time;
                    t_ber[s][t][l][k] = ber;
                    printf("PBOB: CORE: %3d, ITER: %4d, BER: %8.4f, TIME: %8.4f, "
                           "BOB SPU: %8.4f, TOTAL SPU: %8.4f\n",
                           n_proc, (int) reT.info, ber,
                           t_bnp[s][t][l][k], t_bnp[s][t][1][k] / t_bnp[s][t][l][k],
                           total / (t_bnp[s][t][l][k] + t_qr[s][t][l-1][k] + t_aspl[s][t][l-1][k]));
                }
            }
        }
        run_time = omp_get_wtime() - run_time;
        printf("++++++++++++++++++++++++++++++++++++++\n Trial %d, Elapsed Time: %.5fs. \n"
               "++++++++++++++++++++++++++++++++++++++\n", t, run_time);
        cout.flush();
        printf("\n---------------------\nITER:%d\n---------------------\n", t);
        //if (t % 10 == 0) {//
        PyObject *pName, *pModule, *pFunc;
        PyObject *pArgs, *pValue;
        Py_Initialize();
        if (_import_array() < 0)
            PyErr_Print();

        npy_intp dim[4] = {5, 200, 10, 2};

        PyObject *pQRT = PyArray_SimpleNewFromData(4, dim, NPY_DOUBLE, t_qr);
        PyObject *pLLL = PyArray_SimpleNewFromData(4, dim, NPY_DOUBLE, t_aspl);
        PyObject *pBNP = PyArray_SimpleNewFromData(4, dim, NPY_DOUBLE, t_bnp);
        PyObject *pBER = PyArray_SimpleNewFromData(4, dim, NPY_DOUBLE, t_ber);

        if (pQRT == nullptr) printf("[ ERROR] pQRT has a problem.\n");
        if (pLLL == nullptr) printf("[ ERROR] pLLL has a problem.\n");
        if (pBNP == nullptr) printf("[ ERROR] pBNP has a problem.\n");
        if (pBER == nullptr) printf("[ ERROR] pBER has a problem.\n");

        PyObject *sys_path = PySys_GetObject("path");
        if (cils.is_local)
            PyList_Append(sys_path, PyUnicode_FromString(
                    "/home/shilei/CLionProjects/Reference/babai_asyn/babai_asyn_c++/src/plot"));
        else
            PyList_Append(sys_path, PyUnicode_FromString("./"));

        pName = PyUnicode_FromString("plot_bob");
        pModule = PyImport_Import(pName);

        if (pModule != nullptr) {
            pFunc = PyObject_GetAttrString(pModule, "save_data");
            if (pFunc && PyCallable_Check(pFunc)) {
                pArgs = PyTuple_New(8);
                if (PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", 20)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", t)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", c)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", 0)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 4, pQRT) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 5, pLLL) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 6, pBNP) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 7, pBER) != 0) {
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
    //}



    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");

    return 0;

}

template<typename scalar, typename index>
long test_CH(int n, int nob, int c, bool is_local){

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | BOB | %s ]==================================\n", time_str);
    cout.flush();

    index d = 0, l = 0, num_trial = 200, constrain = c != 0, qam = 3;
    cils::CILS<scalar, index> cils;
    cils::returnType<scalar, index> reT;

    cils.is_local = is_local;
    cils.is_constrained = constrain;
//    cils.block_size = n / nob;
//    cils.spilt_size = 2;

    b_vector x_ser, x_lll, x_r;
    cils::init_PBNP(cils, n, 0, qam, 1);

    x_ser.resize(n, false);
    x_lll.resize(n, false);

    printf("--------ITER: %d, SNR: %d, QAM: %d, SIZE: %d-------\n", 0, 0, qam, n);

    cils::CILS_SECH_Search<scalar, index> search(n, n, qam, 1e6);
    cils::CILS_OLM<scalar, index> olm(cils, x_ser, cils.B, cils.y);
    cout<<cils.y;
    init_z_hat(x_ser, x_r, 1, (scalar) cils.upper / 2.0);
    search.ch(0, n, 1, cils.B, cils.y, x_ser);
    projection(cils.A, x_ser, x_lll, 0, cils.upper);
    scalar ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_lll, qam);
    cout<<x_lll;
    cout<<ber;
    //    printf("CH: BER: %8.4f, TIME: %8.4f\n", ber, reT.run_time);
    cout.flush();

    init_z_hat(x_ser, x_r, 1, (scalar) cils.upper / 2.0);
    search.mch(0, n, olm.R_A, cils.y,x_ser, (int) INFINITY, true, false, INFINITY);
    projection(cils.A, x_ser, x_lll, 0, cils.upper);
    ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_lll, qam);
    cout<<x_lll;
    cout<<cils.x_t;
    cout<<ber;
//    printf("MCH: BER: %8.4f, TIME: %8.4f\n", ber, reT.run_time);
    cout.flush();

    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");

    return 0;

}
