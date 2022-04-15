#include <Python.h>
#include <numpy/arrayobject.h>

#include "../source/CILS.cpp"
#include "../source/CILS_Reduction.cpp"
#include "../source/CILS_SECH_Search.cpp"
#include "../source/CILS_OLM.cpp"

void init_z_hat(b_vector &z_hat, b_vector &x, int init, int mean) {
    z_hat.clear();
    if (init == 1) z_hat.assign(mean);
    if (init == 2) z_hat.assign(x);
}

template<typename scalar, typename index>
long test_PBNP() {

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | LLL | %s ]==================================\n", time_str);
    cout.flush();

    index d = 0, l = 0, num_trial = 2, snr = 35, k, n = 512;
    scalar t_qr[200][20][2] = {}, t_aspl[200][20][2] = {}, t_itr[3][200][20][2] = {};
    scalar t_bnp[3][200][20][2], t_ber[3][200][20][2] = {}, run_time;
    scalar ber, res, bnp_spu = 0, spu = 0, ser_ber = 0, omp_ber = 0, bnp_ber = 0, pbnp_ber = 0;
    scalar ser_rt = 0, omp_rt = 0, bnp_rt = 0, pbnp_rt = 0;
    cils::CILS<scalar, index> cils;
    cils::CILS_Reduction<scalar, index> reduction(cils), reduction2(cils);

    cils.is_local = 1;

    for (int t = 0; t < num_trial; t++) {
        run_time = omp_get_wtime();
        b_vector x_ser(n, 0), x_lll(n, 0), x_r(n, 0);
        k = 0;
        for (int qam = 1; qam <= 3; qam += 2) {
            printf("--------QAM: %d-------\n", (int) pow(4, qam));
            cils::init_PBNP(cils, n, snr, qam);
            cils::returnType<scalar, index> reT;
            reduction.reset(cils);
            reT = reduction.aspl();
            t_qr[t][0][k] = reT.run_time;
            t_aspl[t][0][k] = reT.info;
            printf("ASPL: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                   reT.run_time, reT.info, reT.info + reT.run_time);

            cils::CILS_OLM<scalar, index> olm(cils, x_ser, reduction.R, reduction.y);

            l = 0;
            for (index n_proc = 2; n_proc <= 32; n_proc *= 2) {
                l++;
                reduction2.reset(cils);
                reT = reduction2.paspl(n_proc < 20? n_proc:16);
                t_qr[t][l][k] = reT.run_time;
                t_aspl[t][l][k] = reT.info;
                printf("PASPL: CORE: %3d, QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f, "
                       "SPUQR: %8.4f, SPULLL: %8.4f, SPUTOTAL:%8.4f\n",
                       n_proc < 20? n_proc:16, reT.run_time, reT.info, reT.info + reT.run_time,
                       t_qr[t][0][k] / reT.run_time, t_aspl[t][0][k] / reT.info,
                       (t_qr[t][0][k] + t_aspl[t][0][k]) / (reT.run_time + reT.info)
                );
            }

            scalar r = helper::find_residual<scalar, index>(cils.A, cils.x_t, cils.y);
            olm.z_hat.clear();
            olm.backsolve();
            x_r.assign(olm.z_hat);

            for (index init = 0; init < 3; init++) {
                printf("+++++++++++ Init value %d ++++++++++++\n", init);

                init_z_hat(olm.z_hat, x_r, init, round(cils.upper / 2));

                reT = olm.bnp();
                projection(reduction.Z, olm.z_hat, x_lll, 0, cils.upper);
                t_ber[init][t][0][k] = helper::find_bit_error_rate<scalar, index>(x_lll, cils.x_t, cils.qam);
                t_bnp[init][t][0][k] = reT.run_time;
                res = helper::find_residual<scalar, index>(cils.A, x_lll, cils.y);
                printf("BNP: BER: %8.5f, RES: %8.4f, TIME: %8.4f\n", t_ber[init][t][0][k], res,
                       t_bnp[init][t][0][k]);

                l = 0;
                scalar total = t_bnp[init][t][0][k] + t_qr[t][0][k] + t_aspl[t][0][k];
                for (index n_proc = 2; n_proc <= 32; n_proc *= 2) {
                    l++;
                    init_z_hat(olm.z_hat, x_r, init, (int) cils.upper / 2);
                    reT = olm.pbnp2(n_proc < 20? n_proc:16, 20);
                    projection(reduction.Z, olm.z_hat, x_lll, 0, cils.upper);
                    t_ber[init][t][l][k] = helper::find_bit_error_rate<scalar, index>(x_lll, cils.x_t, cils.qam);
                    t_bnp[init][t][l][k] = reT.run_time;
                    t_itr[init][t][l][k] = reT.info;
                    res = helper::find_residual<scalar, index>(cils.A, x_lll, cils.y);
                    printf("PBNP: CORE: %3d, ITER: %4d, BER: %8.5f, RES: %8.4f, TIME: %8.4f, "
                           "BNP SPU: %8.4f, TOTAL SPU: %8.4f\n",
                           n_proc < 20? n_proc:16, (int) reT.info, t_ber[init][t][l][k], res, t_bnp[init][t][l][k],
                           t_bnp[init][t][0][k] / t_bnp[init][t][l][k],
                           total / (t_bnp[init][t][l][k] + t_qr[t][l][k] + t_aspl[t][l][k]));
                }
            }
            k++;

        }
        run_time = omp_get_wtime() - run_time;
        printf("++++++++++++++++++++++++++++++++++++++\n Trial %d, Elapsed Time: %.5fs. \n"
               "++++++++++++++++++++++++++++++++++++++\n", t, run_time);
        cout.flush();
        printf("\n---------------------\nITER:%d\n---------------------\n", t);
        if (true) {//t % 10 == 0
            PyObject *pName, *pModule, *pFunc;
            PyObject *pArgs, *pValue;
            Py_Initialize();
            if (_import_array() < 0)
                PyErr_Print();
            npy_intp di4[4] = {3, 200, 20, 2};
            npy_intp di3[3] = {200, 20, 2};

            PyObject *pQRT = PyArray_SimpleNewFromData(3, di3, NPY_DOUBLE, t_qr);
            PyObject *pLLL = PyArray_SimpleNewFromData(3, di3, NPY_DOUBLE, t_aspl);
            PyObject *pBNP = PyArray_SimpleNewFromData(4, di4, NPY_DOUBLE, t_bnp);
            PyObject *pBER = PyArray_SimpleNewFromData(4, di4, NPY_DOUBLE, t_ber);
            PyObject *pITR = PyArray_SimpleNewFromData(4, di4, NPY_DOUBLE, t_itr);

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
                if (cils.is_local)
                    pFunc = PyObject_GetAttrString(pModule, "plot_bnp");

                if (pFunc && PyCallable_Check(pFunc)) {
                    pArgs = PyTuple_New(9);
                    if (PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", n)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", t)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", 20)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", 5)) != 0) {
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
        }

    }

    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");

    return 0;

}

