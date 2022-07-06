
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

    index m = 56, n = 64, qam = 3, snr = 40;
    scalar res;

    cils::CILS<scalar, index> cils;
    cils.is_local = true;
    cils.is_constrained = true;
    cils.search_iter = 500;

    cils::init_ublm(cils, m, n, snr, qam, 1);

//    scalar r = helper::find_residual<scalar, index>(cils.A, cils.x_t, cils.y);
//    printf("[ INIT COMPLETE, RES:%8.5f, RES:%8.5f]\n", cils.init_res, r);

    cils::returnType<scalar, index> reT;
    //----------------------INIT POINT (SERIAL)--------------------------------//
    b_vector x_gp;
    cils::CILS_UBLM<scalar, index> ublm(cils);
    ublm.x_hat.clear();
    reT = ublm.gp();
    cils::projection(cils.I, ublm.x_hat, x_gp, 0, cils.upper);
    scalar ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_gp, cils.qam);
    printf("GP: ber: %8.5f, time: %8.4f\n", ber, reT.run_time);

//    ublm.x_hat.assign(x_gp);
//    reT = ublm.bsic_bcp(false, 16, true);
//    ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
//    printf("BSIC_BCP_PER: ber: %8.5f, time: %8.4f\n", ber, reT.run_time);
//
//    ublm.x_hat.assign(x_gp);
//    reT = ublm.bsic_bcp(false, 16, false);
//    ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
//    printf("BSIC_BCP_NOP: ber: %8.5f, time: %8.4f\n", ber, reT.run_time);


//    ublm.x_hat.assign(x_gp);
//    reT = ublm.bsic(false, m, true, true);
//    ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
//    printf("BSIC_SCP_RBB: ber: %8.5f, time: %8.4f, itr: %8.4f\n", ber, reT.run_time, reT.info);
////
//    ublm.x_hat.assign(x_gp);
//    reT = ublm.bsic(true, m, true, true);
//    ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
//    printf("BSIC_SCP_BBB: ber: %8.5f, time: %8.4f, itr: %8.4f\n", ber, reT.run_time, reT.info);
//
//
//    ublm.x_hat.assign(x_gp);
//    reT = ublm.pbsic2(true, m, 10, 1);
//    ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
//    printf("BSIC_SCP_PBBB: ber: %8.5f, time: %8.4f, itr: %8.4f\n", ber, reT.run_time, reT.info);

    ublm.x_hat.assign(x_gp);
    reT = ublm.bsic(false, m, true, true);
    ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
    printf("BSIC_SCP_RBB: ber: %8.5f, time: %8.4f, itr: %8.4f\n", ber, reT.run_time, reT.info);

    ublm.x_hat.assign(x_gp);
    reT = ublm.pbsic(false, m, 5, 2, true, false);
    ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
    printf("BSIC_SCP_PRBB1: ber: %8.5f, time: %8.4f, itr: %8.4f\n", ber, reT.run_time, reT.info);

    ublm.x_hat.assign(x_gp);
    reT = ublm.pbsic2(false, m, 5, 2);
    ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
    printf("BSIC_SCP_PRBB2: ber: %8.5f, time: %8.4f, itr: %8.4f\n", ber, reT.run_time, reT.info);

    ublm.x_hat.assign(x_gp);
    reT = ublm.pbsic(false, m, 5, 1, true, false);
    ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
    printf("BSIC_SCP_PRBB: ber: %8.5f, time: %8.4f, itr: %8.4f\n", ber, reT.run_time, reT.info);

    return true;

}

template<typename scalar, typename index>
long test_pbsic(int size_m, bool is_local, index info, index block_size, index sec) {

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | INITPT | %s ]==================================\n", time_str);
    cout.flush();

    index num_trial = 200, m = size_m, n = 64, qam = 3, s = 0;
    scalar t_pbsic[200][4][12][2] = {}, t_ber[200][4][12][2] = {}, run_time, ber, berm, bergp, bsic_time;

    cils::CILS<scalar, index> cils;
    cils.is_local = true;
    cils.is_constrained = true;
    cils.search_iter = info;
    cils::returnType<scalar, index> reT;

    cils.is_local = is_local;
    b_vector x_cgsic1, x_bsicm(n, 0), x_gp;
    for (int t = 0; t < num_trial; t++) {
        run_time = omp_get_wtime();
        s = 0;
        for (int snr = 10; snr <= 40; snr += 10) {
            for (int c = 1; c <= 2; c++) {
                printf("------------- CASE: %d SNR: %d -------------\n", c, snr);
                cils::init_ublm(cils, m, n, snr, qam, c);
                cils::CILS_UBLM<scalar, index> ublm(cils);

//                x_bsicm.assign_col(cils.B, 3);
//                berm = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_bsicm, cils.qam);
//                printf("BSIC_BNPM: ber: %8.5f\n", ber);

                ublm.x_hat.clear();
                reT = ublm.cgsic();
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                t_pbsic[t][s][0][c - 1] = reT.run_time;
                t_ber[t][s][0][c - 1] = ber;
                printf("CGSIC: ber: %8.5f, time: %8.4f\n", ber, reT.run_time);

                ublm.x_hat.clear();
                reT = ublm.gp();
                cils::projection(cils.I, ublm.x_hat, x_gp, 0, cils.upper);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_gp, cils.qam);
                t_pbsic[t][s][1][c - 1] = reT.run_time;
                t_ber[t][s][1][c - 1] = ber;
                printf("GP: ber: %8.5f, time: %8.4f\n", ber, reT.run_time);

                ublm.x_hat.assign(x_gp);
                reT = ublm.bsic(false, block_size, true, false, 1);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                t_pbsic[t][s][10][c - 1] = reT.run_time;
                t_ber[t][s][10][c - 1] = fmax(ber, berm);
                printf("BSIC_RBB-1: ber: %8.5f, time: %8.4f\n", ber, reT.run_time);

                ublm.x_hat.assign(x_gp);
                reT = ublm.bsic(false, block_size, true, false, 4);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                t_pbsic[t][s][2][c - 1] = reT.run_time;
                t_ber[t][s][2][c - 1] = fmax(ber, berm);
                printf("BSIC_RBB-10: ber: %8.5f, time: %8.4f\n", ber, reT.run_time);

                ublm.x_hat.assign(x_gp);
                reT = ublm.bsic(true, block_size, true, false);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                bsic_time = reT.run_time;
                t_pbsic[t][s][3][c - 1] = reT.run_time;
                t_ber[t][s][3][c - 1] = ber;
                printf("BSIC_BBB: ber: %8.5f, time: %8.4f\n", ber, reT.run_time);

                ublm.x_hat.assign(x_gp);
                reT = ublm.pbsic2(false, block_size, 5, 2);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                printf("PBSIC-PRBB|05-02: ber: %8.5f, time: %8.4f, speedup: %8.4f\n",
                       ber, reT.run_time, bsic_time / reT.run_time);
                t_pbsic[t][s][7][c - 1] = reT.run_time;
                t_ber[t][s][7][c - 1] = ber;
                cout.flush();

                ublm.x_hat.assign(x_gp);
                reT = ublm.pbsic2(false, block_size, 10, 2);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                printf("PBSIC-PRBB|10-02: ber: %8.5f, time: %8.4f, speedup: %8.4f\n",
                       ber, reT.run_time, bsic_time / reT.run_time);
                t_pbsic[t][s][8][c - 1] = reT.run_time;
                t_ber[t][s][8][c - 1] = ber;
                cout.flush();

                ublm.x_hat.assign(x_gp);
                reT = ublm.pbsic2(false, block_size, 5, 4);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                printf("PBSIC-PRBB|05-04: ber: %8.5f, time: %8.4f, speedup: %8.4f\n",
                       ber, reT.run_time, bsic_time / reT.run_time);
                t_pbsic[t][s][9][c - 1] = reT.run_time;
                t_ber[t][s][9][c - 1] = ber;
                cout.flush();

                ublm.x_hat.assign(x_gp);
                reT = ublm.pbsic2(true, block_size, 5, 2);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                printf("PBSIC-PBBB|05-02: ber: %8.5f, time: %8.4f, speedup: %8.4f\n",
                       ber, reT.run_time, bsic_time / reT.run_time);
                t_pbsic[t][s][4][c - 1] = reT.run_time;
                t_ber[t][s][4][c - 1] = ber;
                cout.flush();

                ublm.x_hat.assign(x_gp);
                reT = ublm.pbsic2(true, block_size, 10, 2);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                printf("PBSIC-PBBB|10-02: ber: %8.5f, time: %8.4f, speedup: %8.4f\n",
                       ber, reT.run_time, bsic_time / reT.run_time);
                t_pbsic[t][s][5][c - 1] = reT.run_time;
                t_ber[t][s][5][c - 1] = ber;
                cout.flush();

                ublm.x_hat.assign(x_gp);
                reT = ublm.pbsic2(true, block_size, 5, 4);
                ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, ublm.x_hat, cils.qam);
                printf("PBSIC-PBBB|05-04: ber: %8.5f, time: %8.4f, speedup: %8.4f\n",
                       ber, reT.run_time, bsic_time / reT.run_time);
                t_pbsic[t][s][6][c - 1] = reT.run_time;
                t_ber[t][s][6][c - 1] = ber;
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

        npy_intp di5[4] = {200, 4, 12, 2};

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
                if (PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", info + sec)) != 0) {
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