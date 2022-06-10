
#include <Python.h>
#include <numpy/arrayobject.h>

#include "../source/CILS.cpp"
#include "../source/CILS_Reduction.cpp"
#include "../source/CILS_SECH_Search.cpp"
#include "../source/CILS_OLM.cpp"
#include "../source/CILS_UBLM.cpp"


template<typename scalar, typename index>
bool test_init_pt() {

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | INIT_POINT | %s ]==================================\n", time_str);

    index m = 50, n = 60, qam = 3, snr = 40;
    scalar res;

    cils::CILS<scalar, index> cils;
    cils.is_local = true;
    cils.is_constrained = true;
    cils.search_iter = 300;

    cils::init_ublm(cils, m, n, snr, qam, 1);

//    scalar r = helper::find_residual<scalar, index>(cils.A, cils.x_t, cils.y);
//    printf("[ INIT COMPLETE, RES:%8.5f, RES:%8.5f]\n", cils.init_res, r);

    cils::returnType<scalar, index> reT;
    //----------------------INIT POINT (SERIAL)--------------------------------//
    cils::CILS_Reduction<scalar, index> reduction, reduction1;
//    cout << cils.A;
//    reduction.reset(cils.A, cils.y, 7);
//    b_matrix A_S, A_T;
//    reT = reduction.aip();
//    prod(reduction.Q, reduction.R, A_S);
////    cout << A_T;
//    scalar qr_time = reT.run_time;
//    reduction.reset(cils.A, cils.y, 7);
//
//    reT = reduction.paip(10);
//    prod(reduction.Q, reduction.R, A_T);
//    scalar error;
//    for (index i = 0; i < m * n; i++) {
//        error += fabs(A_S[i] - A_T[i]);
//    }
//    cout << "ERROR:" << error << ", ";
//    printf("QR: SER: %8.4f, PAR: %8.4f, SPU: %8.4f\n", qr_time, reT.run_time, qr_time / reT.run_time);

//    cout << A_T;
//    cout << reduction.R;
//    cout << reduction.Q;
    if (n > 20) {


        cils::CILS_UBLM<scalar, index> ublm(cils);

        CGSIC:
        cout << cils.x_t;
        reT = ublm.cgsic();
        res = helper::find_residual<scalar, index>(cils.A, ublm.x_hat, cils.y);
        scalar ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
        b_vector x_cgsic(ublm.x_hat);
        cout << ublm.x_hat;
        printf("CGSIC: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT.run_time);
//
//    ublm.x_hat.clear();
//    reT = ublm.gp();
//    res = helper::find_residual<scalar, index>(cils.A, ublm.x_hat, cils.y);
//    ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
//    cout << ublm.x_hat;
//    printf("GP: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT.run_time);

        ublm.x_hat.assign(x_cgsic);
        reT = ublm.bsic(false, m);
        res = helper::find_residual<scalar, index>(cils.A, ublm.x_hat, cils.y);
        ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
        cout << ublm.x_hat;
        cout.flush();
        scalar time = reT.run_time;
        printf("BSIC_BNP: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT.run_time);


        ublm.x_hat.assign(x_cgsic);
        reT = ublm.bsic(true, m);
        res = helper::find_residual<scalar, index>(cils.A, ublm.x_hat, cils.y);
        ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
        cout << ublm.x_hat;
        cout.flush();
        time = reT.run_time;
        printf("BSIC_BBB: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT.run_time);

        ublm.x_hat.assign(x_cgsic);
        reT = ublm.pbsic(true, m, 5);
        res = helper::find_residual<scalar, index>(cils.A, ublm.x_hat, cils.y);
        ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
        cout << ublm.x_hat;

        printf("PBSIC: ber: %8.5f, v_norm: %8.4f, time: %8.4f, speedup: %8.4f\n",
               ber, res, reT.run_time, time / reT.run_time);
    }

    return true;

}

template<typename scalar, typename index>
long test_init(int size_m, bool is_local) {

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | INITPT | %s ]==================================\n", time_str);
    cout.flush();

    index num_trial = 200, m = size_m, n = 64;
    scalar t_init[200][4][2][2] = {}, t_beri[200][4][2][2] = {}, run_time, ber, bergp;
    index s = 0, qam = 3;

    cils::CILS<scalar, index> cils;
    cils.is_local = true;
    cils.is_constrained = true;
    cils.search_iter = 2e3;
    cils::returnType<scalar, index> reT;

    cils.is_local = is_local;
    b_vector x_gp, x_cgsic1, x_gpt, x_gp1;
    for (int t = 0; t < num_trial; t++) {
        run_time = omp_get_wtime();
        s = 0;
        for (int snr = 10; snr <= 40; snr += 10) {
            for (int c = 1; c <= 2; c++) {
                printf("------------- CASE: %d -------------\n", c);
                x_gp.resize(n, false);
                x_cgsic1.resize(n, false);
                x_gpt.resize(n, false);

                cils::init_ublm(cils, m, n, snr, qam, c);
                cils::CILS_UBLM<scalar, index> ublm(cils);

                x_cgsic1.assign_col(cils.B, 0);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_cgsic1, cils.qam);
                printf("CGSICM: ber: %8.5f\n", ber);

                x_gpt.assign_col(cils.B, 1);
                cils::projection(cils.I, x_gpt, x_gp1, 0, cils.upper);
                bergp = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_gp1, cils.qam);
                printf("GPM: ber: %8.5f\n", bergp);

                reT = ublm.cgsic();
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                t_init[t][s][c - 1][0] = reT.run_time;
                t_beri[t][s][c - 1][0] = ber;
                printf("CGSIC: ber: %8.5f, time: %8.4f\n", ber, reT.run_time);

                ublm.x_hat.clear();
                reT = ublm.gp();
                cils::projection(cils.I, ublm.x_hat, x_gp, 0, cils.upper);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_gp, cils.qam);
                t_init[t][s][c - 1][1] = reT.run_time;
                t_beri[t][s][c - 1][1] = fmin(ber, bergp);
                printf("GP: ber: %8.5f, time: %8.4f\n", ber, reT.run_time);
            }
            s++;
        }
        run_time = omp_get_wtime() - run_time;
        printf("++++++++++++++++++++++++++++++++++++++\n Trial %d, Elapsed Time: %.5fs. \n"
               "++++++++++++++++++++++++++++++++++++++\n", t, run_time);
        cout.flush();
        printf("\n---------------------\nITER:%d\n---------------------\n", t);

        PyObject * pName, *pModule, *pFunc;
        PyObject * pArgs, *pValue;
        Py_Initialize();
        if (_import_array() < 0)
            PyErr_Print();

        npy_intp di5[4] = {200, 4, 2, 2};

        PyObject * pT = PyArray_SimpleNewFromData(4, di5, NPY_DOUBLE, t_init);
        PyObject * pB = PyArray_SimpleNewFromData(4, di5, NPY_DOUBLE, t_beri);

        if (pT == nullptr) printf("[ ERROR] pT has a problem.\n");
        if (pB == nullptr) printf("[ ERROR] pB has a problem.\n");

        PyObject * sys_path = PySys_GetObject("path");
        if (cils.is_local)
            PyList_Append(sys_path, PyUnicode_FromString(
                    "/home/shilei/CLionProjects/babai_asyn/babai_asyn_c++/src/plot"));
        else
            PyList_Append(sys_path, PyUnicode_FromString("./"));

        pName = PyUnicode_FromString("plot_init");
        pModule = PyImport_Import(pName);

        if (pModule != nullptr) {
            pFunc = PyObject_GetAttrString(pModule, "save_data");
            if (pFunc && PyCallable_Check(pFunc)) {
                pArgs = PyTuple_New(6);
                if (PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", m)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", n)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", t + 1)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", 0)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 4, pT) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 5, pB) != 0) {
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

    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");

    return 0;

}

template<typename scalar, typename index>
long test_pbsic(int size_m, bool is_local) {

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | INITPT | %s ]==================================\n", time_str);
    cout.flush();

    index num_trial = 1, m = size_m, n = 64;
    scalar t_pbsic[200][4][6][2] = {}, t_ber[200][4][6][2] = {}, run_time, ber, berm, bergp, bsic_time;
    index s = 0, qam = 3;

    cils::CILS<scalar, index> cils;
    cils.is_local = true;
    cils.is_constrained = true;
    cils.search_iter = 2e3;
    cils::returnType<scalar, index> reT;

    cils.is_local = is_local;
    b_vector x_bsicm, x_gp;
    for (int t = 0; t < num_trial; t++) {
        run_time = omp_get_wtime();
        s = 0;
        for (int snr = 10; snr <= 40; snr += 10) {
            for (int c = 1; c <= 2; c++) {
                printf("------------- CASE: %d SNR: %d -------------\n", c, snr);
                cils::init_ublm(cils, m, n, snr, qam, c);
                cils::CILS_UBLM<scalar, index> ublm(cils);

                x_bsicm.assign_col(cils.B, 3);
                berm = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_bsicm, cils.qam);
                printf("BSIC_BNPM: ber: %8.5f\n", ber);

                ublm.x_hat.clear();
                reT = ublm.gp();
                cils::projection(cils.I, ublm.x_hat, x_gp, 0, cils.upper);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_gp, cils.qam);
                t_pbsic[t][s][0][c - 1] = reT.run_time;
                t_ber[t][s][0][c - 1] = ber;
                printf("GP: ber: %8.5f, time: %8.4f\n", ber, reT.run_time);

                ublm.x_hat.assign(x_gp);
                reT = ublm.bsic(false, m);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                t_pbsic[t][s][1][c - 1] = reT.run_time;
                t_ber[t][s][1][c - 1] = fmax(ber, berm);
                printf("BSIC_BNP: ber: %8.5f, time: %8.4f\n", ber, reT.run_time);

                ublm.x_hat.assign(x_gp);
                reT = ublm.bsic(true, m);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                bsic_time = reT.run_time;
                t_pbsic[t][s][2][c - 1] = reT.run_time;
                t_ber[t][s][2][c - 1] = ber;
                printf("BSIC_BBB: ber: %8.5f, time: %8.4f\n", ber, reT.run_time);

                ublm.x_hat.assign(x_gp);
                reT = ublm.pbsic(true, m, 5, 2);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                printf("PBSIC|5-2: ber: %8.5f, time: %8.4f, speedup: %8.4f\n",
                       ber, reT.run_time, bsic_time / reT.run_time);
                t_pbsic[t][s][3][c - 1] = reT.run_time;
                t_ber[t][s][3][c - 1] = ber;
                cout.flush();

                ublm.x_hat.assign(x_gp);
                reT = ublm.pbsic(true, m, 10, 2);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                printf("PBSIC|10-2: ber: %8.5f, time: %8.4f, speedup: %8.4f\n",
                       ber, reT.run_time, bsic_time / reT.run_time);
                t_pbsic[t][s][4][c - 1] = reT.run_time;
                t_ber[t][s][4][c - 1] = ber;
                cout.flush();

                ublm.x_hat.assign(x_gp);
                reT = ublm.pbsic(true, m, 5, 4);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                printf("PBSIC|5-4: ber: %8.5f, time: %8.4f, speedup: %8.4f\n",
                       ber, reT.run_time, bsic_time / reT.run_time);
                t_pbsic[t][s][5][c - 1] = reT.run_time;
                t_ber[t][s][5][c - 1] = ber;
                cout.flush();
            }
            s++;
        }
        run_time = omp_get_wtime() - run_time;
        printf("++++++++++++++++++++++++++++++++++++++\n Trial %d, Elapsed Time: %.5fs. \n"
               "++++++++++++++++++++++++++++++++++++++\n", t, run_time);
        cout.flush();
        printf("\n---------------------\nITER:%d\n---------------------\n", t);

        PyObject * pName, *pModule, *pFunc;
        PyObject * pArgs, *pValue;
        Py_Initialize();
        if (_import_array() < 0)
            PyErr_Print();

        npy_intp di5[4] = {200, 4, 6, 2};

        PyObject * pT = PyArray_SimpleNewFromData(4, di5, NPY_DOUBLE, t_pbsic);
        PyObject * pB = PyArray_SimpleNewFromData(4, di5, NPY_DOUBLE, t_ber);

        if (pT == nullptr) printf("[ ERROR] pT has a problem.\n");
        if (pB == nullptr) printf("[ ERROR] pB has a problem.\n");

        PyObject * sys_path = PySys_GetObject("path");
        if (cils.is_local)
            PyList_Append(sys_path, PyUnicode_FromString(
                    "/home/shilei/CLionProjects/babai_asyn/babai_asyn_c++/src/plot"));
        else
            PyList_Append(sys_path, PyUnicode_FromString("./"));

        pName = PyUnicode_FromString("plot_bsic");
        pModule = PyImport_Import(pName);

        if (pModule != nullptr) {
            pFunc = PyObject_GetAttrString(pModule, "save_data");
            if (pFunc && PyCallable_Check(pFunc)) {
                pArgs = PyTuple_New(6);
                if (PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", m)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", n)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", t + 1)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", 0)) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 4, pT) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 5, pB) != 0) {
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

    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");

    return 0;

}