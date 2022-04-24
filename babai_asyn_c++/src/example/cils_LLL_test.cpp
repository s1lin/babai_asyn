#include <Python.h>
#include <numpy/arrayobject.h>

#include "../source/CILS.cpp"
#include "../source/CILS_Reduction.cpp"

template<typename scalar, typename index>
long plot_LLL() {

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | LLL | %s ]==================================\n", time_str);
    cout.flush();

    index d = 0, l = 0, num_trial = 200;
    scalar t_qr[4][200][20][2] = {}, t_aspl[4][200][20][2] = {}, t_total[4][200][20][2] = {}, run_time;
    cils::CILS<scalar, index> cils;
    cils::CILS_Reduction<scalar, index> reduction(cils);
    cils.is_local = 1;

    for (int t = 0; t < num_trial; t++) {
        d = 0;
        run_time = omp_get_wtime();
        for (int n = 50; n <= 200; n += 50) {
            printf("+++++++++++ Dimension %d ++++++++++++++++++++\n", n);
            for (int k = 0; k <= 1; k++) {
                printf("+++++++++++ Case %d ++++++++++++++++++++\n", k + 1);
                l = 0;
                cils::init_LLL(cils, n, k);
                cout.flush();

                cils::returnType<scalar, index> reT;
//                reduction.reset(cils);
//                reT = reduction.mgs_qrp();
//                cout<<reduction.R;
//
//                reduction.reset(cils);
//                reT = reduction.pmgs_qrp(15);
//
//                cout<<reduction.R;

                reduction.reset(cils);
                reT = reduction.plll();
                t_qr[d][t][0][k] = reT.run_time;
                t_aspl[d][t][0][k] = reT.info;
                t_total[d][t][0][k] = t_qr[d][t][0][k] + t_aspl[d][t][0][k];
                printf("PLLL: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                       reT.run_time, reT.info, reT.info + reT.run_time);

                l++;
                reduction.reset(cils);
                reT = reduction.aspl();
                t_qr[d][t][l][k] = reT.run_time;
                t_aspl[d][t][l][k] = reT.info;
                t_total[d][t][l][k] = t_qr[d][t][l][k] + t_aspl[d][t][l][k];
                printf("ASPL: QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f\n",
                       reT.run_time, reT.info, reT.info + reT.run_time);


                for (index n_proc = 5; n_proc <= 30; n_proc += 5) {
                    l++;
                    reduction.reset(cils);
                    index n_c = n_proc;
                    if (n_proc == 20)
                        n_c = 15;
                    if (n_proc == 25)
                        n_c = 20;
                    if (n_proc == 30)
                        n_c = 20;
                    reT = reduction.paspl(n_c);
                    t_qr[d][t][l][k] = reT.run_time;
                    t_aspl[d][t][l][k] = reT.info;
                    t_total[d][t][l][k] = t_qr[d][t][l][k] + t_aspl[d][t][l][k];
                    printf("PASPL: CORE: %3d, QR: %8.4f, LLL: %8.4f, TOTAL:%8.4f, "
                           "SPUQR: %8.4f, SPULLL: %8.4f, SPUTOTAL:%8.4f,"
                           "SPUPL: %8.4f, SPUTOTAL2:%8.4f\n",
                           n_c, reT.run_time, reT.info, reT.info + reT.run_time,
                           t_qr[d][t][0][k] / reT.run_time, t_aspl[d][t][1][k] / reT.info,
                           t_total[d][t][1][k] / t_total[d][t][l][k],
                           t_aspl[d][t][0][k] / reT.info,
                           t_total[d][t][0][k] / t_total[d][t][l][k]
                    );
                }

            }
            d++;
        }
        run_time = omp_get_wtime() - run_time;
        printf("++++++++++++++++++++++++++++++++++++++\n Trial %d, Elapsed Time: %.5fs. \n"
               "++++++++++++++++++++++++++++++++++++++\n", t, run_time);
        cout.flush();
        printf("\n---------------------\nITER:%d\n---------------------\n", t);
        if ((t + 1) % 10 == 0) {
            PyObject *pName, *pModule, *pFunc;
            PyObject *pArgs, *pValue;
            Py_Initialize();
            if (_import_array() < 0)
                PyErr_Print();
            npy_intp dim[4] = {4, num_trial, 20, 2};

            PyObject *pQRT = PyArray_SimpleNewFromData(4, dim, NPY_DOUBLE, t_qr);
            PyObject *pLLL = PyArray_SimpleNewFromData(4, dim, NPY_DOUBLE, t_aspl);
            PyObject *pTOT = PyArray_SimpleNewFromData(4, dim, NPY_DOUBLE, t_total);

            if (pQRT == nullptr) printf("[ ERROR] pQRT has a problem.\n");
            if (pLLL == nullptr) printf("[ ERROR] pLLL has a problem.\n");
            if (pTOT == nullptr) printf("[ ERROR] pTOT has a problem.\n");

            PyObject *sys_path = PySys_GetObject("path");
            if (cils.is_local)
                PyList_Append(sys_path, PyUnicode_FromString(
                        "/home/shilei/CLionProjects/Reference/babai_asyn/babai_asyn_c++/src/plot"));
            else
                PyList_Append(sys_path, PyUnicode_FromString("./"));

            pName = PyUnicode_FromString("plot_lll");
            pModule = PyImport_Import(pName);

            if (pModule != nullptr) {
                pFunc = PyObject_GetAttrString(pModule, "save_data");
                if (cils.is_local)
                    pFunc = PyObject_GetAttrString(pModule, "plot_lll");

                if (pFunc && PyCallable_Check(pFunc)) {
                    pArgs = PyTuple_New(7);
                    if (PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", 5)) != 0) {
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
                    if (PyTuple_SetItem(pArgs, 6, pTOT) != 0) {
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