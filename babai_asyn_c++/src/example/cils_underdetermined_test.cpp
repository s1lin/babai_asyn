//#include <Python.h>
//#include <numpy/arrayobject.h>
//#include <boost/python.hpp>
//
#include "../source/CILS.cpp"
#include "../source/CILS_Reduction.cpp"
#include "../source/CILS_SECH_Search.cpp"
//#include "../source/CILS_SO_UBILS.cpp"
#include "../source/CILS_OLM.cpp"
//
#include <ctime>

template<typename scalar, typename index>
bool block_babai_test(int size, int rank) {

    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | BLOCBABAI | %s ]==================================\n", time_str);
    cout.flush();

    index n = 512, m = 512, qam = 3, snr = 35, num_trials = 6;
    scalar bnp_spu = 0, spu = 0, ser_ber = 0, omp_ber = 0, bnp_ber = 0, pbnp_ber = 0;
    scalar ser_rt = 0, omp_rt = 0, bnp_rt = 0, pbnp_rt = 0;

    cils::CILS<scalar, index> cils(m, n, qam, snr, 20000);

    cils.init_d();
    cils.is_constrained = false;

    for (int t = 1; t <= 1; t++) {
        b_vector x_ser(n, 0), x_lll(n, 0);
        cout.flush();
        cils::init(cils);

//        if (rank == 0) {
        scalar r = helper::find_residual<scalar, index>(cils.A, cils.x_t, cils.y);

        scalar ber, runtime, res;
        cils::returnType<scalar, index> reT, reT2;
        cils::CILS_Reduction<scalar, index> reduction(cils);
        reduction.aspl();
        cout << reduction.y;

        cils::CILS_Reduction<scalar, index> reduction2(cils);
        reduction2.aspl_omp(10);
        cout << reduction2.y;

        cils::CILS_OLM<scalar, index> obils(cils, x_ser, reduction.R, reduction.y);
        cils::CILS_OLM<scalar, index> obils2(cils, x_ser, reduction2.R, reduction2.y);

        obils.z_hat.clear();
        reT = obils.bnp();
        projection(reduction.Z, obils.z_hat, x_lll, 0, cils.upper);
        ber = helper::find_bit_error_rate<scalar, index>(x_lll, cils.x_t, cils.qam);
        bnp_ber += ber;
        bnp_rt += reT.run_time;
        res = helper::find_residual<scalar, index>(cils.A, x_lll, cils.y);
        printf("bnp1: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT.run_time);


        obils.z_hat.clear();
        reT = obils2.bnp();
        projection(reduction2.Z, obils.z_hat, x_lll, 0, cils.upper);
        ber = helper::find_bit_error_rate<scalar, index>(x_lll, cils.x_t, cils.qam);
//        bnp_ber += ber;
//        bnp_rt += reT.run_time;
        res = helper::find_residual<scalar, index>(cils.A, x_lll, cils.y);
        printf("bnp2: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT.run_time);

        obils.z_hat.clear();
        reT2 = obils.pbnp2(8, 20);
        projection(reduction.Z, obils.z_hat, x_lll, 0, cils.upper);
        ber = helper::find_bit_error_rate<scalar, index>(x_lll, cils.x_t, cils.qam);
        pbnp_ber += ber;
        pbnp_rt += reT2.run_time;
        res = helper::find_residual<scalar, index>(cils.A, x_lll, cils.y);
        printf("opbnp: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT2.run_time);
        bnp_spu += reT.run_time / reT2.run_time;

        obils.z_hat.clear();
        reT2 = obils.pbnp(8, 20);
        projection(reduction.Z, obils.z_hat, x_lll, 0, cils.upper);
        ber = helper::find_bit_error_rate<scalar, index>(x_lll, cils.x_t, cils.qam);
//        pbnp_ber += ber;
//        pbnp_rt += reT2.run_time;
        res = helper::find_residual<scalar, index>(cils.A, x_lll, cils.y);
        printf("opbnp: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT2.run_time);
//        bnp_spu += reT.run_time / reT2.run_time;

        obils.z_hat.clear();
        reT = obils.bocb(0);
        projection(reduction.Z, obils.z_hat, x_lll, 0, cils.upper);
        ber = helper::find_bit_error_rate<scalar, index>(x_lll, cils.x_t, cils.qam);
        ser_ber += ber;
        ser_rt += reT.run_time;
        res = helper::find_residual<scalar, index>(cils.A, x_lll, cils.y);
        printf("bocb: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT.run_time);
        cout.flush();

        obils.z_hat.clear();
        reT2 = obils.pbocb_test(8, 6, 0);
        projection(reduction.Z, obils.z_hat, x_lll, 0, cils.upper);
        ber = helper::find_bit_error_rate<scalar, index>(x_lll, cils.x_t, cils.qam);
        omp_ber += ber;
        omp_rt += reT2.run_time;
        res = helper::find_residual<scalar, index>(cils.A, x_lll, cils.y);
        printf("pbocb: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT2.run_time);
        spu += reT.run_time / reT2.run_time;
//        }
        printf("ber result: bnp_ber: %8.5f, pbnp_ber: %8.5f, bob_ber: %8.5f, pbob_ber: %8.4f \n",
               bnp_ber / t, pbnp_ber / t, ser_ber / t, omp_ber / t);
        printf("spu result: pbnp_spu: %8.5f, pbob_spu: %8.5f\n", bnp_rt / pbnp_rt, ser_rt / omp_rt);
        printf("spu result: pbnp_spu: %8.5f, pbob_spu: %8.5f\n", bnp_spu / t, spu / t);
    }

    return true;
}






//
// template<typename scalar, typename index>
//bool init_point_test(int size, int rank) {
//    index n = 1024, m = 1024, qam = 3, snr = 35, num_trials = 6;
//
//    b_vector x_q(n, 0), x_tmp(n, 0), x_ser(n, 0), x_omp(n, 0), x_mpi(n, 0);
//    auto *v_norm_qr = (double *) calloc(1, sizeof(double));
//    scalar v_norm;
//    cils::returnType<scalar, index> reT3;
//    cils::CILS<scalar, index> cils(m, n, qam, snr, 1e5);
//    cout.flush();
//    cils::init(cils);
//    cils.is_constrained = true;
//
//    if (rank == 0) {
//        scalar r = helper::find_residual<scalar, index>(cils.A, cils.x_t, cils.y);
//        printf("[ INIT COMPLETE, RES:%8.5f, RES:%8.5f]\n", cils.init_res, r);
//
//        time_t t0 = time(nullptr);
//        struct tm *lt = localtime(&t0);
//        char time_str[20];
//        sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
//                lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
//                lt->tm_hour, lt->tm_min, lt->tm_sec
//        );
//        printf("====================[ TEST | INIT_POINT | %s ]==================================\n", time_str);
//
//        cils::returnType<scalar, index> reT, reT2;
////        //----------------------INIT POINT (SERIAL)--------------------------------//
////        cils::CILS_Init_Point<scalar, index> IP(cils);
////        reT = IP.sic_serial(x_q);
////        x_tmp = prod(IP.P, x_q);
////        v_norm_qr[0] = v_norm = helper::find_residual<scalar, index>(cils.A, x_tmp, cils.y);
////        scalar ber = helper::find_bit_error_rate<scalar, index>(x_tmp, cils.x_t, cils.qam);
////        helper::display<scalar, index>(n, x_tmp.data(), "x_q");
////        printf("INI: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, v_norm, reT.run_time);
////
////
////        cils::CILS_Block_SIC BSIC(cils);
////        reT = BSIC.scp_block_optimal_serial(IP.H, x_q, v_norm, 0);
////        x_tmp = prod(IP.P, x_q);
////        helper::display<scalar, index>(n, x_tmp.data(), "x_q");
////        helper::display<scalar, index>(n, cils.x_t, "x_t");
////        ber = helper::find_bit_error_rate<scalar, index>(x_tmp, cils.x_t, cils.qam);
////        v_norm = helper::find_residual<scalar, index>(cils.A, x_tmp, cils.y);
////        printf("SIC: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, v_norm, reT.run_time);
////        cils::CILS_Reduction<scalar, index> reduction(cils);
////        reduction.plll_reduction_tester(1);
//
////        reduction.aip_reduction();
////        helper::display<scalar, index>(n, reduction.y_q, "y");
////        helper::display<scalar, index>(reduction.R_Q, "R");
////        helper::display<scalar, index>(reduction.Q, "Q");
////        cout << reduction.R_Q.size1() << "x" << reduction.R_Q.size2();
////        cout << cils.R.size1() << "x" << cils.R.size2();
////        cout << "Norm_1" << norm_1(reduction.R_Q - cils.R);
//
//        cils::CILS_Reduction<scalar, index> reduction2(cils.A, cils.y, 0, cils.upper);
//        reduction2.mgs_qr_omp(10);
//        cils::CILS_Babai<scalar, index> babai(cils);
//        x_q.clear();
//        for (index ii = 0; ii < x_q.size(); ii++) {
//            x_q[ii] = 0; //random() % 10 - 5;
//        }
//        babai.z_hat.assign(x_q);
//        reT = babai.babai(reduction2.R_Q, reduction2.y_q);
//        scalar ser_time = reT.run_time;
////        helper::display<scalar, index>(cils.R, "R");
////        helper::display<scalar, index>(reduction2.Q, "Q");
////        helper::display<scalar, index>(m, cils.y, "y");
////        //cout << babai.z_hat;
//        x_ser.assign(babai.z_hat);
//////        helper::display<scalar, index>(n, babai.z_hat, "z_hat");
//////        helper::display<scalar, index>(n, x_ser, "x_ser");
//////        helper::display<scalar, index>(n, cils.x_t, "x_t");
//        scalar ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_ser, cils.qam);
//        scalar res = helper::find_residual<scalar, index>(cils.A, x_ser, cils.y);
//        printf("BABAI: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT.run_time);
//
//        babai.z_hat.assign(x_q);
//        reT = babai.babai_oomp(reduction2.R_Q, reduction2.y_q, 10, num_trials, 0);
////        helper::display<scalar, index>(cils.R, "R");
////        helper::display<scalar, index>(reduction2.Q, "Q");
////        helper::display<scalar, index>(m, cils.y, "y");
////        cout << babai.z_hat;
//        x_ser.assign(babai.z_hat);
////        helper::display<scalar, index>(n, babai.z_hat, "z_hat");
////        helper::display<scalar, index>(n, x_ser, "x_ser");
////        helper::display<scalar, index>(n, cils.x_t, "x_t");
//        ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_ser, cils.qam);
//        res = helper::find_residual<scalar, index>(cils.A, x_ser, cils.y);
//        printf("O_BABAI: ber: %8.5f, v_norm: %8.4f, time: %8.4f, iter %1f, spu: %8.4f\n", ber, res, reT.run_time,
//               reT.info, ser_time / reT.run_time);
//
//        babai.z_hat.assign(x_q);
//        reT = babai.babai_omp(reduction2.R_Q, reduction2.y_q, 10, num_trials, 0);
////        helper::display<scalar, index>(cils.R, "R");
////        helper::display<scalar, index>(reduction2.Q, "Q");
////        helper::display<scalar, index>(m, cils.y, "y");
////        cout << babai.z_hat;
//        x_ser.assign(babai.z_hat);
////        helper::display<scalar, index>(n, babai.z_hat, "z_hat");
////        helper::display<scalar, index>(n, x_ser, "x_ser");
////        helper::display<scalar, index>(n, cils.x_t, "x_t");
//        ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_ser, cils.qam);
//        res = helper::find_residual<scalar, index>(cils.A, x_ser, cils.y);
//        printf("BABAI: ber: %8.5f, v_norm: %8.4f, time: %8.4f, iter %1f, spu: %8.4f\n", ber, res, reT.run_time,
//               reT.info, ser_time / reT.run_time);
////
////        scalar ber_npy[5][3][40] = {};
////        for (index init = 0; init <= 6; init += 3) {
////            for (index l = 0; l < num_trials; l++) {
////                for (index i = 3; i <= 15; i += 3) {
////                    index ll = 0;
////                    for (index nswp = 1000; nswp <= 2000; nswp+=100) {
////
////                        for (index ii = 0; ii < x_q.size(); ii++){
////                            x_q[ii] = 0; //random() % 10 - 5;
////                        }
////                        babai.z_hat.assign(x_q);
////                        reT = babai.babai_sic_omp(cils.A, cils.y, nswp, i);
////                        ber = helper::find_bit_error_rate<scalar, index>(x_ser, babai.z_hat, cils.qam);
//////                        if (nswp == 60 && ber > 1e-5) {
//////                            babai.z_hat.assign(x_q);
//////                            reT = babai.babai_sic_omp(cils.A, cils.y, nswp, i);
//////                            ber = helper::find_bit_error_rate<scalar, index>(x_ser, babai.z_hat, cils.qam);
//////                            if (ber > 1e-5) {
//////                                cout << endl;
//////                                helper::display<scalar, index>(n, babai.z_hat, "z_hat");
//////                                helper::display<scalar, index>(n, x_ser, "x_ser");
//////                                printf("BABAI: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT.run_time);
//////                            }
//////                        }
////                        ll++;
////                        ber_npy[i / 3 - 1][init / 3][ll] += ber / num_trials;
////                    }
////                    printf("l: %4d, n_proc: %4d; ber: %1.2f; ", l, i, ber);
////                }
////                if (l % 20 == 0)
////                    cout << endl;
////            }
////        }
////        PyObject *pName, *pModule, *pFunc;
////
////        Py_Initialize();
////        if (_import_array() < 0)
////            PyErr_Print();
////        npy_intp dim[3] = {5, 3, 40};
////        /*
////         * scalar res[1][200][2] = {}, ber[1][200][2] = {}, tim[1][200][2] = {}, spu[1][200][2] = {}, spu2[1][200][2] = {};
////            scalar tim_total[1][200][2] = {}, spu_total[1][200][2] = {};
////         */
////
////        PyObject *pRes = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, ber_npy);
////
////        if (pRes == nullptr) printf("[ ERROR] pRes has a problem.\n");
////
////        PyObject *sys_path = PySys_GetObject("path");
////        PyList_Append(sys_path, PyUnicode_FromString(
////                "/home/shilei/CLionprojectionects/babai_asyn/babai_asyn_c++/src/example"));
////        pName = PyUnicode_FromString("plot_helper");
////        pModule = PyImport_Import(pName);
////
////        if (pModule != nullptr) {
////            pFunc = PyObject_GetAttrString(pModule, "plot_babai_converge");
////            if (pFunc && PyCallable_Check(pFunc)) {
////                PyObject *pArgs = PyTuple_New(4);
////                if (PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", n)) != 0) {
////                    return false;
////                }
////                if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", 2000)) != 0) {
////                    return false;
////                }
////                if (PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", snr)) != 0) {
////                    return false;
////                }
////                if (PyTuple_SetItem(pArgs, 3, pRes) != 0) {
////                    return false;
////                }
////                PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
////
////            } else {
////                if (PyErr_Occurred())
////                    PyErr_Print();
////                fprintf(stderr, "Cannot find function qr\n");
////            }
////        } else {
////            PyErr_Print();
////            fprintf(stderr, "Failed to load file\n");
////
////        }
//
////        cils.block_size = 16;
////
////        cils.init_d();
////        cils::CILS_Block_Babai<scalar, index> block_babai(cils);
////        reT = block_babai.block_search_serial(reduction2.R_Q, reduction2.y_q);
////        cout << block_babai.z_hat;
////        x_hat = prod(reduction2.Z, block_babai.z_hat);
////        ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_hat, cils.qam);
////        res = helper::find_residual<scalar, index>(cils.A, x_hat, cils.y);
////        printf("BLOCK: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT.run_time);
////
////        reT = babai.babai_method_omp(reduction2.R_Q, reduction2.y_q, 2, 6);
////        x_hat = prod(reduction2.Z, babai.z_hat);
//////        helper::display<scalar, index>(n, x_hat, "x_hat");
//////        helper::display<scalar, index>(n, cils.x_t, "x_t");
////        ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_hat, cils.qam);
////        res = helper::find_residual<scalar, index>(cils.A, x_hat, cils.y);
////        printf("BABAI: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT.run_time);
////
////        cils.block_size = 16;
////        cils.spilt_size = 2;
////        cils.offset = 0;
////        cils.init_d();
////        cils::CILS_Block_Babai<scalar, index> block_babai(cils);
////        reT = block_babai.block_search_serial(0, reduction2.R_Q, reduction2.y_q);
////        x_hat = prod(reduction2.Z, babai.z_hat);
//////        helper::display<scalar, index>(n, x_hat, "x_hat");
//////        helper::display<scalar, index>(n, cils.x_t, "x_t");
////        ber = helper::find_bit_error_rate<scalar, index>(cils.x_t, x_hat, cils.qam);
////        res = helper::find_residual<scalar, index>(cils.A, x_hat, cils.y);
////        printf("BABAI: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, res, reT.run_time);
////
////        //----------------------INIT POINT (OMP)--------------------------------//
////        //STEP 1: init point by QRP
////        reT = cils.grad_projection_omp(x_omp, 1e4, 10);
////        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_omp.data(), x_tmp.data());
////        v_norm_qr[0] = v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(),
////                                                                     cils.y.data());
////        ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
////        helper::display<scalar, index>(n, x_tmp.data(), "x_q");
////        printf("INI: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, v_norm, reT.run_time);
//    }
//    return true;
//}template<typename scalar, typename index>
//bool partition_test(int size, int rank) {
//    index n = 18, m = 12, qam = 3, snr = 35, num_trials = 6;
//
//    b_vector x_q(n, 0), x_tmp(n, 0), x_sic(n, 0), x_omp(n, 0), x_mpi(n, 0);
//    cils::returnType<scalar, index> reT3;
//    cils::CILS<scalar, index> cils(m, n, qam, snr, 3000);
//    cout.flush();
//    cils::init(cils);
//    cils.is_constrained = true;
//
//    if (rank == 0) {
//        scalar r = helper::find_residual<scalar, index>(cils.A, cils.x_t, cils.y);
////        printf("[ INIT COMPLETE, RES:%8.5f, RES:%8.5f]\n", cils.init_res, r);
//
//        time_t t0 = time(nullptr);
//        struct tm *lt = localtime(&t0);
//        char time_str[20];
//        sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
//                lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
//                lt->tm_hour, lt->tm_min, lt->tm_sec
//        );
//        printf("====================[ TEST | Partition | %s ]==================================\n", time_str);
//        cout.flush();
//
//        cils::returnType<scalar, index> reT, reT2;
//        cils::CILS_SO_UBILS<scalar, index> ubils(cils);
//
//        //Init point
//        ubils.sic(x_sic);
//        b_vector x_init = prod(ubils.P, x_sic);
//        helper::display<scalar, index>(x_init, "x_sic");
//        helper::display<scalar, index>(cils.x_t, "x_t");
//
//        r = helper::find_residual<scalar, index>(cils.A, x_init, cils.y);
//        ubils.block_sic_optimal(x_sic, r, 1);
//        cout << endl;
//        helper::display<scalar, index>(prod(ubils.P, x_sic), "xpser");
//        helper::display<scalar, index>(cils.x_t, "x_t");
//    }
//    return true;
//}

//template<typename scalar, typename index, index m, index n>
//void block_optimal_test(int size, int rank) {
//    vector<scalar> x_q(n, 0), x_tmp(n, 0), x_ser(n, 0), x_omp(n, 0), x_mpi(n, 0);
//    auto *v_norm_qr = (double *) calloc(1, sizeof(double));
//
//    scalar v_norm;
//    cils::returnType<scalar, index> reT3;
//    cils::cils<scalar, index, m, n> cils(qam, 35);
//    //auto a = (double *)malloc(N * sizeof(double));
//    cils.qam = qam;
//    cils.upper = pow(2, qam) - 1;
//    cils.u.fill(cils.upper);
//    cout.flush();
//    cils.init(rank);
//
//    if (rank == 0) {
////        cils.init_ud();
//        scalar r = helper::find_residual<scalar, index>(cils.A, cils.x_t, cils.y);
//        printf("[ INIT COMPLETE, RES:%8.5f, RES:%8.5f]\n", cils.init_res, r);
//
//        time_t t0 = time(nullptr);
//        struct tm *lt = localtime(&t0);
//        char time_str[20];
//        sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
//                lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
//                lt->tm_hour, lt->tm_min, lt->tm_sec
//        );
//        printf("====================[ TEST | SIC_OPT | %s ]==================================\n", time_str);
//
//        cils::returnType<scalar, index> reT, reT2;
//        //----------------------INIT POINT (OPTIONAL)--------------------------------//
//        //STEP 1: init point by QRP
//        reT = cils.grad_projection(x_q, 100);
//        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_q.data(), x_tmp.data());
//        v_norm_qr[0] = v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(),
//                                                                     cils.y.data());
//        scalar ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
//        helper::display<scalar, index>(n, x_tmp.data(), "x_q");
//        printf("INI: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, v_norm, reT.run_time);
//
//
//        //----------------------OPTIMAL--------------------------------//
//        //STEP 2: Optimal SERIAL SCP:
////        x_ser.assign(x_q.begin(), x_q.end());
////        reT2 = cils.scp_block_optimal_serial(x_ser, v_norm_qr[0], 0);
////        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_ser.data(), x_tmp.data());
////        v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(), cils.y.data());
////
////        //Result Validation:
////        helper::display<scalar, index>(n, x_ser.data(), "x_z");
////        helper::display<scalar, index>(n, cils.x_t.data(), "x_t");
////        helper::display<scalar, index>(n, x_tmp.data(), "x_p");
////        ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
////        printf("SER_BLOCK1: ber: %8.5f, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
////               ber, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);
//
//
//        //STEP 3: SUBOptimal SERIAL SCP:
//        x_ser.assign(x_q.begin(), x_q.end());
//        reT2 = cils.scp_block_suboptimal_serial(x_ser, v_norm_qr[0], 0);
//        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_ser.data(), x_tmp.data());
//        v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(), cils.y.data());
//
//        //Result Validation:
//        helper::display<scalar, index>(n, x_ser.data(), "x_z");
//        helper::display<scalar, index>(n, cils.x_t.data(), "x_t");
//        helper::display<scalar, index>(n, x_tmp.data(), "x_p");
//        ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
//        printf("SER_BLOCK2: ber: %8.5f, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
//               ber, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);
//        //
//
//
//        //STEP 3: Optimal OMP SCP:
//        x_omp.assign(x_q.begin(), x_q.end());
//        reT2 = cils.scp_block_optimal_omp(x_omp, v_norm_qr[0], 5, false);
//        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_omp.data(), x_tmp.data());
//        v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(), cils.y.data());
//
//        //Result Validation:
//        helper::display<scalar, index>(n, x_omp.data(), "x_z");
//        helper::display<scalar, index>(n, cils.x_t.data(), "x_t");
//        helper::display<scalar, index>(n, x_tmp.data(), "x_p");
//        ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
//        printf("OMP_BLOCK1: ber: %8.5f, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
//               ber, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);
//
//
//        //STEP 3: Optimal OMP SCP:
//        x_omp.assign(x_q.begin(), x_q.end());
//        reT2 = cils.scp_block_suboptimal_omp(x_omp, v_norm_qr[0], 5, false);
//        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_omp.data(), x_tmp.data());
//        v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(), cils.y.data());
//
//        //Result Validation:
//        helper::display<scalar, index>(n, x_omp.data(), "x_z");
//        helper::display<scalar, index>(n, cils.x_t.data(), "x_t");
//        helper::display<scalar, index>(n, x_tmp.data(), "x_p");
//        ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
//        printf("OMP_BLOCK2: ber: %8.5f, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
//               ber, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);
//
//
//        //----------------------BABAI--------------------------------//
//        //STEP 4: Babai Serial SCP:
//        x_ser.assign(x_q.begin(), x_q.end());
//        reT2 = cils.scp_block_babai_serial(x_ser, v_norm_qr[0], 0);
//        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_ser.data(), x_tmp.data());
//        v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(), cils.y.data());
//
//        //Result Validation:
//        helper::display<scalar, index>(n, x_ser.data(), "x_z");
//        helper::display<scalar, index>(n, cils.x_t.data(), "x_t");
//        helper::display<scalar, index>(n, x_tmp.data(), "x_p");
//        ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
//        printf("SER_BABAI: ber: %8.5f, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
//               ber, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);
////
////
////        //STEP 5: Babai OMP SCP:
////        x_omp.assign(x_q.begin(), x_q.end());
////        reT2 = cils.scp_block_babai_omp(x_omp, v_norm_qr[0], 5, false);
////        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_omp.data(), x_tmp.data());
////        v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(), cils.y.data());
////
////        //Result Validation:
////        helper::display<scalar, index>(n, x_omp.data(), "x_z");
////        helper::display<scalar, index>(n, cils.x_t.data(), "x_t");
////        helper::display<scalar, index>(n, x_tmp.data(), "x_p");
////        ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
////        printf("OMP_BABAI: ber: %8.5f, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
////               ber, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);
//
//
//    }
//
////    //STEP 2: MPI-Block SCP:
////    MPI_Bcast(&x_mpi[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
////    MPI_Bcast(&v_norm_qr[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
////    MPI_Bcast(&cils.y[0], m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
////    MPI_Bcast(&cils.H[0], m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
////    MPI_Barrier(MPI_COMM_WORLD);
////    reT3 = {{0, 0, 0}, 0, 0};
////    reT3 = cils.scp_block_optimal_mpi(x_mpi, v_norm_qr, size, rank);
////
////    if (rank == 0) {
//////        v_norm = reT3.info;
////        x_tmp.assign(n, 0);
////        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_mpi.data(), x_tmp.data());
////
////        //Result Validation:
////        helper::display<scalar, index>(n, x_mpi.data(), "x_z");
////        helper::display<scalar, index>(n, cils.x_t.data(), "x_t");
////        helper::display<scalar, index>(n, x_tmp.data(), "x_p");
////        scalar ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
////        printf("MPI: ber: %8.5f, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f, rank:%d\n",
////               ber, reT3.x[0], reT3.x[1], reT3.x[2], v_norm, reT3.run_time, rank);
////    }
////    MPI_Barrier(MPI_COMM_WORLD);
//}
//
//template<typename scalar, typename index, index m, index n>
//long plot_run(int size, int rank) {
//    time_t t0 = time(nullptr);
//    struct tm *lt = localtime(&t0);
//    char time_str[20];
//    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
//            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
//            lt->tm_hour, lt->tm_min, lt->tm_sec
//    );
//    printf("====================[ TEST | ILS | %s ]==================================\n", time_str);
////    for (SNR = 35; SNR >= 5; SNR -= 20) {
//    index d_s_size = d_s.size();
//    scalar res[4][200][2] = {}, ber[4][200][2] = {}, tim[5][200][2] = {}, spu[5][200][2] = {};//, t_spu[4][200][2] = {};
//    //scalar stm[3][200][2] = {};
//    //scalar all_time[7][1000][3][2] = {}; //Method, iter, init, qam
//
//    index count = 0;
//    scalar run_time;
//    cils::returnType<scalar, index> reT;
//    auto *v_norm_qr = (double *) calloc(1, sizeof(double));
//
//    scalar r, t, b, iter, t2, r2, b2, iter2, prev_t_block, prev_t_babai;
//    scalar b_ser_block, b_ser_babai, t_ser_block, t_ser_babai, t_init_pt, t_omp;
//    index l = 0; //count for qam.
//
//    vector<scalar> z_omp(n, 0), z_ser(n, 0), z_ini(n, 0), z_tmp(n, 0);
//    cils::cils<scalar, index, m, n> cils(qam, SNR);
//
//    for (index i = 1; i <= max_iter; i++) {
////            for (index k = 1; k <= 3; k += 2) {
//        for (index k = 3; k >= 1; k -= 2) {
//
//            count = k == 1 ? 0 : 1;
//            run_time = omp_get_wtime();
//            cils.qam = k;
//            cils.upper = pow(2, k) - 1;
//            cils.u.fill(cils.upper);
//            cils.init(0);
//            r = helper::find_residual<scalar, index>(cils.A, cils.x_t, cils.y);
//            printf("[ INIT COMPLETE, RES:%8.5f, RES:%8.5f]\n", cils.init_res, r);
//
//            for (index init = -1; init <= 2; init++) {
//                l = 0;
//                printf("[ TRIAL PHASE]\n");
//                /*
//                 * ------------------------------------------------------
//                 * 1. INIT POINT (OPTIONAL) : a. SIC, b.QRP, c.GRAD, d.0
//                 * ------------------------------------------------------
//                 */
//                z_ini.assign(n, 0);
//                if (init == -1) {
//                    reT = cils.qrp_serial(z_ini);
//                    cout << "1a. Method: INIT_QRP, ";
//                } else if (init == 0) {
//                    reT = cils.sic_serial(z_ini);
//                    cout << "1b. Method: INIT_SIC, ";
//                } else if (init == 1) {
//                    reT = cils.grad_projection(z_ini, search_iter);
//                    cout << "1c. Method: INIT_GRD, ";
//                } else {
//                    cout << "1d. Method: INIT_VT0, ";
//                    cils.H.fill(0);
//                    helper::eye<scalar, index>(n, cils.P.data());
//                    for (index i1 = 0; i1 < m * n; i1++) {
//                        cils.H[i1] = cils.A[i1];
//                    }
//                    reT = {{}, 0, 0};
//                }
//
//                helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ini.data(), z_tmp.data());
//                b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), k);
//                r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y.data());
//                v_norm_qr[0] = r;
//                t_init_pt = reT.run_time;
//                res[init + 1][l][count] += r;
//                ber[init + 1][l][count] += b;
//                tim[init + 1][l][count] += t_init_pt;
//                l++;
//                printf("AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs\n",
//                       res[init + 1][l][count] / i, ber[init + 1][l][count] / i, tim[init + 1][l][count] / i,
//                       r, b, t_init_pt);
//
//
//                /*
//                 * -----------------------------------------------------------------
//                 * 2. Block Optimal Serial-SCP
//                 * -----------------------------------------------------------------
//                 */
//                z_ser.assign(z_ini.begin(), z_ini.end());
//                z_tmp.assign(n, 0);
//                index itr = 0;
//                do {
//                    reT = cils.scp_block_suboptimal_serial(z_ser, v_norm_qr[0], itr > 0);
//                    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ser.data(), z_tmp.data());
//                    b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), cils.qam);
//                    z_ser.assign(z_ini.begin(), z_ini.end());
//                    itr++;
//                } while (b > 0.3 && itr < 5);
//                r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y.data());
//                b_ser_block = b;
//                t_ser_block = reT.run_time + t_init_pt;
//                res[init + 1][l][count] += r;
//                ber[init + 1][l][count] += b;
//                tim[init + 1][l][count] += t_ser_block;
//                l++;
//                printf("2a. Method: SUBOP_SER, AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, "
//                       "RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs\n",
//                       res[init + 1][l][count] / i, ber[init + 1][l][count] / i, tim[init + 1][l][count] / i,
//                       r, b, t_ser_block);
//
//                /*
//                 * -----------------------------------------------------------------
//                 * 2b. Block subOptimal Serial-SCP
//                 * -----------------------------------------------------------------
//                 */
//                z_ser.assign(z_ini.begin(), z_ini.end());
//                z_tmp.assign(n, 0);
//                itr = 0;
//                do {
//                    reT = cils.scp_block_optimal_serial(z_ser, v_norm_qr[0], itr > 0);
//                    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ser.data(), z_tmp.data());
//                    b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), cils.qam);
//                    z_ser.assign(z_ini.begin(), z_ini.end());
//                    itr++;
//                } while (b > 0.3 && itr < 5);
//                r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y.data());
//                printf("2b. Method: BLOPT_SER, AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, "
//                       "RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs\n",
//                       res[init + 1][l][count] / i, ber[init + 1][l][count] / i, tim[init + 1][l][count] / i,
//                       r, b, reT.run_time + t_init_pt);
//
//                /*
//                 * -----------------------------------------------------------------
//                 * 3. Block Babai Serial-SCP
//                 * -----------------------------------------------------------------
//                 */
//                z_ser.assign(z_ini.begin(), z_ini.end());
//                z_tmp.assign(n, 0);
//                itr = 0;
//                do {
//                    reT = cils.scp_block_babai_serial(z_ser, v_norm_qr[0], itr > 0);
//                    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ser.data(), z_tmp.data());
//                    b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), cils.qam);
//                    z_ser.assign(z_ini.begin(), z_ini.end());
//                    itr++;
//                } while (b > 0.3 && itr < 5);
//                r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y.data());
//                b_ser_babai = b;
//                t_ser_babai = reT.run_time + t_init_pt;
//                res[init + 1][l][count] += r;
//                ber[init + 1][l][count] += b;
//                tim[init + 1][l][count] += t_ser_babai;
//                l++;
//                printf("3. Method: BABAI_SER, AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, "
//                       "RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs\n",
//                       res[init + 1][l][count] / i, ber[init + 1][l][count] / i, tim[init + 1][l][count] / i,
//                       r, b, t_ser_babai);
//
//
//                prev_t_babai = prev_t_block = INFINITY;
//                for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
//
//                    /*
//                     * -----------------------------------------------------------------
//                     * 4. Block Optimal Parallel-SCP
//                     * -----------------------------------------------------------------
//                     */
//                    index _ll = 0;
//                    t = r = b = 0;
//                    t2 = r2 = b2 = INFINITY;
//                    cils::program_def::chunk = 1;
//                    while (true) {
//                        z_omp.assign(z_ini.begin(), z_ini.end());
//                        reT = cils.scp_block_suboptimal_omp(z_omp, v_norm_qr[0], n_proc, _ll > 0);
//                        z_tmp.assign(n, 0);
//                        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_omp.data(), z_tmp.data());
//                        t = reT.run_time;
//                        b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), k);
//                        r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(),
//                                                                 cils.y.data());
//
//                        t2 = min(t, t2);
//                        r2 = min(r, r2);
//                        b2 = min(b, b2);
//                        _ll++;
//
//                        if (prev_t_block > t && b - b_ser_block < 0.1) break; //
//                        if (_ll == 5) {
//                            r = r2;
//                            b = b2;
//                            t = t2;
//                            break; //
//                        }
//                    }
//                    t_omp = t + t_init_pt;
//                    res[init + 1][l][count] += r;
//                    ber[init + 1][l][count] += b;
//                    tim[init + 1][l][count] += t_omp;
//                    spu[init + 1][l][count] += t_ser_block / t_omp;
//
//                    printf("4. Method: BLOCK_OMP, N_PROC: %2d, AVG RES: %8.5f, AVG BER: %8.5f, "
//                           "AVG TIME: %8.5fs, RES: %8.5f, BER: %8.5f, SER TIME: %8.5f, OMP TIME: %8.5fs, "
//                           "SPEEDUP:%7.3f, AVG SPEEDUP: %7.3f.\n",
//                           n_proc, res[init + 1][l][count] / i, ber[init + 1][l][count] / i,
//                           tim[init + 1][l][count] / i, r, b, t_ser_block, t_omp,
//                           t_ser_block / t_omp, spu[init + 1][l][count] / i);
//
//                    l++;
//                    prev_t_block = t;
//
//
//                    /* -----------------------------------------------------------------
//                     * 5. Block Babai Parallel-SCP
//                     * -----------------------------------------------------------------
//                     */
//
//                    _ll = 0;
//                    t = r = b = 0;
//                    t2 = r2 = b2 = INFINITY;
//                    cils::program_def::chunk = 1;
//                    while (true) {
//                        z_omp.assign(z_ini.begin(), z_ini.end());
//                        reT = cils.scp_block_babai_omp(z_omp, v_norm_qr[0], n_proc, _ll > 0);
//                        z_tmp.assign(n, 0);
//                        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_omp.data(), z_tmp.data());
//                        t = reT.run_time;
//                        b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), k);
//                        r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(),
//                                                                 cils.y.data());
//
//                        t2 = min(t, t2);
//                        r2 = min(r, r2);
//                        b2 = min(b, b2);
//                        _ll++;
//
//                        if (prev_t_babai > t && b - b_ser_babai < 0.1) break; //
//                        if (_ll == 5) {
//                            r = r2;
//                            b = b2;
//                            t = t2;
//                            break; //
//                        }
//                    }
//                    t_omp = t + t_init_pt;
//                    res[init + 1][l][count] += r;
//                    ber[init + 1][l][count] += b;
//                    tim[init + 1][l][count] += t_omp;
//                    spu[init + 1][l][count] += t_ser_babai / t_omp;
//
//                    printf("5. Method: BABAI_OMP, N_PROC: %2d, AVG RES: %8.5f, AVG BER: %8.5f, "
//                           "AVG TIME: %8.5fs, RES: %8.5f, BER: %8.5f, SER TIME: %8.5f, OMP TIME: %8.5fs, "
//                           "SPEEDUP:%7.3f, AVG SPEEDUP: %7.3f.\n",
//                           n_proc, res[init + 1][l][count] / i, ber[init + 1][l][count] / i,
//                           tim[init + 1][l][count] / i, r, b, t_ser_babai, t_omp,
//                           t_ser_babai / t_omp, spu[init + 1][l][count] / i);
//
//                    l++;
//                    prev_t_babai = t;
//                }
//            }
//            run_time = omp_get_wtime() - run_time;
//            printf("++++++++++++++++++++++++++++++++++++++\n Trial %d, Elapsed Time: %.5fs. \n"
//                   "++++++++++++++++++++++++++++++++++++++\n", i, run_time);
//            cout.flush();
//        }
//        printf("\n---------------------\nITER:%d\n---------------------\n", i);
//        if (i % plot_itr == 0) {//i % 50 == 0 &&
//            PyObject *pName, *pModule, *pFunc;
//
//            Py_Initialize();
//            if (_import_array() < 0)
//                PyErr_Print();
//            npy_intp dim[3] = {4, 200, 2};
//            npy_intp di4[3] = {5, 200, 2};
//
//            scalar proc_nums[l - 2] = {};
//            index ll = 0;
//
//            for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
//                proc_nums[ll] = n_proc;
//                ll++;
//            }
//            npy_intp dpc[1] = {ll};
//
//            PyObject *pRes = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, res);
//            PyObject *pBer = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, ber);
//            PyObject *pTim = PyArray_SimpleNewFromData(3, di4, NPY_DOUBLE, tim);
//            PyObject *pSpu = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, spu);
//            PyObject *pPrc = PyArray_SimpleNewFromData(1, dpc, NPY_DOUBLE, proc_nums);
//
//            if (pRes == nullptr) printf("[ ERROR] pRes has a problem.\n");
//            if (pBer == nullptr) printf("[ ERROR] pBer has a problem.\n");
//            if (pTim == nullptr) printf("[ ERROR] pTim has a problem.\n");
//            if (pPrc == nullptr) printf("[ ERROR] pPrc has a problem.\n");
//            if (pSpu == nullptr) printf("[ ERROR] pSpu has a problem.\n");
//
//            PyObject *sys_path = PySys_GetObject("path");
//            PyList_Append(sys_path, PyUnicode_FromString(
//                    "/home/shilei/CLionprojectionects/babai_asyn/babai_asyn_c++/src/example"));
//            pName = PyUnicode_FromString("plot_helper");
//            pModule = PyImport_Import(pName);
//
//            if (pModule != nullptr) {
//                pFunc = PyObject_GetAttrString(pModule, "plot_runtime_ud");
//                if (pFunc && PyCallable_Check(pFunc)) {
//                    PyObject *pArgs = PyTuple_New(14);
//                    if (PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", n)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", SNR)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", qam)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", l)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 4, Py_BuildValue("i", i)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 5, pRes) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 6, pBer) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 7, pTim) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 8, pPrc) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 9, pSpu) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 10, Py_BuildValue("i", max_proc)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 11, Py_BuildValue("i", min_proc)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 12, Py_BuildValue("i", is_constrained)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 13, Py_BuildValue("i", m)) != 0) {
//                        return false;
//                    }
//
//                    PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
//
//                } else {
//                    if (PyErr_Occurred())
//                        PyErr_Print();
//                    fprintf(stderr, "Cannot find function qr\n");
//                }
//            } else {
//                PyErr_Print();
//                fprintf(stderr, "Failed to load file\n");
//
//            }
//        }
//    }
//
//    printf("End of current TASK.\n");
//    printf("-------------------------------------------\n");
////    }
//    return 0;
//}
//
//template<typename scalar, typename index, index m, index n>
//long plot_run_grad_omp(int size, int rank) {
//    time_t t0 = time(nullptr);
//    struct tm *lt = localtime(&t0);
//    char time_str[20];
//    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
//            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
//            lt->tm_hour, lt->tm_min, lt->tm_sec
//    );
//    printf("====================[ TEST | ILS | %s ]==================================\n", time_str);
//    //    for (SNR = 35; SNR >= 5; SNR -= 20) {
//    index d_s_size = d_s.size();
//    scalar res[4][200][2] = {}, ber[4][200][2] = {}, tim[4][200][2] = {}, spu[4][200][2] = {};
//    scalar tim_total[4][200][2] = {}, spu_total[4][200][2] = {};
//    //scalar stm[3][200][2] = {};
//    //scalar all_time[7][1000][3][2] = {}; //Method, iter, init, qam
//
//    index count = 0;
//    scalar run_time;
//    cils::returnType<scalar, index> reT;
//    auto *v_norm_qr = (double *) calloc(1, sizeof(double));
//
//    scalar r, t, b, iter, t2, r2, b2, iter2, prev_t_block, prev_t_babai;
//    scalar b_ser_block, b_ser_babai, t_ser_block, t_ser_babai, t_ser_block_total;
//    scalar t_ser_babai_total, t_init_pt, t_init_pt_omp, t_omp, t_omp_total;
//    index l = 0; //count for qam.
//
//    vector<scalar> z_omp(n, 0), z_ser(n, 0), z_ini(n, 0), z_tmp(n, 0);
//    cils::cils<scalar, index, m, n> cils(qam, SNR);
//
//    for (index i = 1; i <= max_iter; i++) {
//        //            for (index k = 1; k <= 3; k += 2) {
//        for (index k = 3; k >= 1; k -= 2) {
//
//            count = k == 1 ? 0 : 1;
//            run_time = omp_get_wtime();
//            cils.qam = k;
//            cils.upper = pow(2, k) - 1;
//            cils.u.fill(cils.upper);
//            cils.init(0);
//            r = helper::find_residual<scalar, index>(cils.A, cils.x_t, cils.y);
//            printf("[ INIT COMPLETE, RES:%8.5f, RES:%8.5f]\n", cils.init_res, r);
//
//            for (index init = -1; init <= 2; init++) {
//                l = 0;
//                printf("[ TRIAL PHASE]\n");
//                /*
//                 * ------------------------------------------------------
//                 * 1. INIT POINT (OPTIONAL) : a. SIC, b.QRP, c.GRAD, d.0
//                 * ------------------------------------------------------
//                 */
//                z_ini.assign(n, 0);
//                if (init == -1) {
//                    reT = cils.qrp_serial(z_ini);
//                    cout << "1a. Method: INIT_QRP, ";
//                } else if (init == 0) {
//                    reT = cils.sic_serial(z_ini);
//                    cout << "1b. Method: INIT_SIC, ";
//                } else if (init == 1) {
//                    reT = cils.grad_projection(z_ini, search_iter);
//                    cout << "1c. Method: INIT_GRD, ";
//                } else {
//                    cout << "1d. Method: INIT_VT0, ";
//                    cils.H.fill(0);
//                    helper::eye<scalar, index>(n, cils.P.data());
//                    for (index i1 = 0; i1 < m * n; i1++) {
//                        cils.H[i1] = cils.A[i1];
//                    }
//                    reT = {{}, 0, 0};
//                }
//
//                helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ini.data(), z_tmp.data());
//                b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), k);
//                r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y.data());
//                v_norm_qr[0] = r;
//                t_init_pt = reT.run_time;
//                res[init + 1][l][count] += r;
//                ber[init + 1][l][count] += b;
//                tim[init + 1][l][count] += t_init_pt;
//                l++;//0
//                printf("AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs\n",
//                       res[init + 1][l][count] / i, ber[init + 1][l][count] / i, tim[init + 1][l][count] / i,
//                       r, b, t_init_pt);
//
//
//                /*
//                 * -----------------------------------------------------------------
//                 * 2. Block Optimal Serial-SCP
//                 * -----------------------------------------------------------------
//                 */
//                z_ser.assign(z_ini.begin(), z_ini.end());
//                z_tmp.assign(n, 0);
//                index itr = 0;
//                do {
//                    reT = cils.scp_block_suboptimal_serial(z_ser, v_norm_qr[0], itr > 0);
//                    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ser.data(), z_tmp.data());
//                    b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), cils.qam);
//                    z_ser.assign(z_ini.begin(), z_ini.end());
//                    itr++;
//                } while (b > 0.3 && itr < 5);
//                r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y.data());
//                b_ser_block = b;
//                t_ser_block = reT.run_time;
//                t_ser_block_total = reT.run_time + t_init_pt;
//
//                res[init + 1][l][count] += r;
//                ber[init + 1][l][count] += b;
//                tim[init + 1][l][count] += t_ser_block;
//                tim_total[init + 1][l][count] += t_ser_block_total;
//                l++;//1
//                printf("2a. Method: SUBOP_SER, AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, "
//                       "RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs, TOTAL TIME: %.5fs\n",
//                       res[init + 1][l][count] / i, ber[init + 1][l][count] / i, tim[init + 1][l][count] / i,
//                       r, b, t_ser_block, t_ser_block_total);
//
//                /*
//                 * -----------------------------------------------------------------
//                 * 3. Block Babai Serial-SCP
//                 * -----------------------------------------------------------------
//                 */
//                z_ser.assign(z_ini.begin(), z_ini.end());
//                z_tmp.assign(n, 0);
//                itr = 0;
//                do {
//                    reT = cils.scp_block_babai_serial(z_ser, v_norm_qr[0], itr > 0);
//                    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ser.data(), z_tmp.data());
//                    b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), cils.qam);
//                    z_ser.assign(z_ini.begin(), z_ini.end());
//                    itr++;
//                } while (b > 0.3 && itr < 5);
//                r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y.data());
//                b_ser_babai = b;
//                t_ser_babai = reT.run_time;
//                t_ser_babai_total = reT.run_time + t_init_pt;
//                res[init + 1][l][count] += r;
//                ber[init + 1][l][count] += b;
//                tim[init + 1][l][count] += t_ser_babai;
//                tim_total[init + 1][l][count] += t_ser_babai_total;
//                l++;//2
//                printf("3. Method: BABAI_SER, AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, "
//                       "RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs, TOTAL TIME: %.5fs\n",
//                       res[init + 1][l][count] / i, ber[init + 1][l][count] / i, tim[init + 1][l][count] / i,
//                       r, b, t_ser_babai, t_ser_babai_total);
//
//
//                prev_t_babai = prev_t_block = INFINITY;
//                for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
//                    if (init == 1) {
//                        /*
//                         * -----------------------------------------------------------------
//                         * 1d. INIT POINT GRAD
//                         * -----------------------------------------------------------------
//                         */
//
//                        z_ini.assign(n, 0);
//                        reT = cils.grad_projection_omp(z_ini, search_iter, n_proc);
//                        cout << "1d. Method: INIT_PGP, ";
//
//                        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ini.data(), z_tmp.data());
//                        b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), k);
//                        r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y.data());
//                        v_norm_qr[0] = r;
//                        t_init_pt_omp = reT.run_time;
//                        res[init + 1][l][count] += r;
//                        ber[init + 1][l][count] += b;
//                        tim[init + 1][l][count] += t_init_pt_omp;
//                        spu[init + 1][l][count] += t_init_pt / t_init_pt_omp;
//                        printf("N_PROC: %2d, AVG RES: %8.5f, AVG BER: %8.5f, "
//                               "AVG TIME: %8.5fs, RES: %8.5f, BER: %8.5f, SER TIME: %8.5f, OMP TIME: %8.5fs, "
//                               "SPEEDUP:%7.3f, AVG SPEEDUP: %7.3f.\n",
//                               n_proc, res[init + 1][l][count] / i, ber[init + 1][l][count] / i,
//                               tim[init + 1][l][count] / i, r, b, t_init_pt, t_init_pt_omp,
//                               t_init_pt / t_init_pt_omp, spu[init + 1][l][count] / i);
//                    } else{
//                        t_init_pt_omp = t_init_pt;
//                    }
//                    l++;//3
//                    /*
//                     * -----------------------------------------------------------------
//                     * 4. Block Optimal Parallel-SCP
//                     * -----------------------------------------------------------------
//                     */
//                    index _ll = 0;
//                    t = r = b = 0;
//                    t2 = r2 = b2 = INFINITY;
//                    cils::program_def::chunk = 1;
//                    while (true) {
//                        z_omp.assign(z_ini.begin(), z_ini.end());
//                        reT = cils.scp_block_suboptimal_omp(z_omp, v_norm_qr[0], n_proc, _ll > 0);
//                        z_tmp.assign(n, 0);
//                        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_omp.data(), z_tmp.data());
//                        t = reT.run_time;
//                        b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), k);
//                        r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(),
//                                                                 cils.y.data());
//
//                        t2 = min(t, t2);
//                        r2 = min(r, r2);
//                        b2 = min(b, b2);
//                        _ll++;
//
//                        if (prev_t_block > t && b - b_ser_block < 0.1) break; //
//                        if (_ll == 5) {
//                            r = r2;
//                            b = b2;
//                            t = t2;
//                            break; //
//                        }
//                    }
//                    t_omp = t;
//                    t_omp_total = t + t_init_pt_omp;
//                    res[init + 1][l][count] += r;
//                    ber[init + 1][l][count] += b;
//                    tim[init + 1][l][count] += t_omp;
//                    tim_total[init + 1][l][count] += t_omp_total;
//                    spu[init + 1][l][count] += t_ser_block / t_omp;
//                    spu_total[init + 1][l][count] += t_ser_block_total / t_omp_total;
//
//                    printf("4. Method: BLOCK_OMP, N_PROC: %2d, AVG RES: %8.5f, AVG BER: %8.5f, "
//                           "AVG TIME: %8.5fs, RES: %8.5f, BER: %8.5f, SER TIME: %8.5f, OMP TIME: %8.5fs, "
//                           "SPEEDUP:%7.3f, AVG SPEEDUP: %7.3f, TOTAL TIME: %7.3f, TOTAL SPU: %7.3f.\n",
//                           n_proc, res[init + 1][l][count] / i, ber[init + 1][l][count] / i,
//                           tim[init + 1][l][count] / i, r, b, t_ser_block, t_omp,
//                           t_ser_block / t_omp, spu[init + 1][l][count] / i, t_omp_total, t_ser_block_total / t_omp_total);
//
//                    l++;//4
//                    prev_t_block = t;
//
//
//                    /* -----------------------------------------------------------------
//                     * 5. Block Babai Parallel-SCP
//                     * -----------------------------------------------------------------
//                     */
//
//                    _ll = 0;
//                    t = r = b = 0;
//                    t2 = r2 = b2 = INFINITY;
//                    cils::program_def::chunk = 1;
//                    while (true) {
//                        z_omp.assign(z_ini.begin(), z_ini.end());
//                        reT = cils.scp_block_babai_omp(z_omp, v_norm_qr[0], n_proc, _ll > 0);
//                        z_tmp.assign(n, 0);
//                        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_omp.data(), z_tmp.data());
//                        t = reT.run_time;
//                        b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), k);
//                        r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(),
//                                                                 cils.y.data());
//
//                        t2 = min(t, t2);
//                        r2 = min(r, r2);
//                        b2 = min(b, b2);
//                        _ll++;
//
//                        if (prev_t_babai > t && b - b_ser_babai < 0.1) break; //
//                        if (_ll == 5) {
//                            r = r2;
//                            b = b2;
//                            t = t2;
//                            break; //
//                        }
//                    }
//                    t_omp = t;
//                    t_omp_total = t + t_init_pt_omp;
//                    res[init + 1][l][count] += r;
//                    ber[init + 1][l][count] += b;
//                    tim[init + 1][l][count] += t_omp;
//                    spu[init + 1][l][count] += t_ser_babai / t_omp;
//                    tim_total[init + 1][l][count] += t_omp_total;
//                    spu_total[init + 1][l][count] += t_ser_babai_total / t_omp_total;
//
//                    printf("5. Method: BABAI_OMP, N_PROC: %2d, AVG RES: %8.5f, AVG BER: %8.5f, "
//                           "AVG TIME: %8.5fs, RES: %8.5f, BER: %8.5f, SER TIME: %8.5f, OMP TIME: %8.5fs, "
//                           "SPEEDUP:%7.3f, AVG SPEEDUP: %7.3f, TOTAL TIME: %7.3f, TOTAL SPU: %7.3f.\n",
//                           n_proc, res[init + 1][l][count] / i, ber[init + 1][l][count] / i,
//                           tim[init + 1][l][count] / i, r, b, t_ser_babai, t_omp,
//                           t_ser_babai / t_omp, spu[init + 1][l][count] / i, t_omp_total, t_ser_babai_total / t_omp_total);
//
//                    l++;//5
//                    prev_t_babai = t;
//                }
//            }
//        }
//        run_time = omp_get_wtime() - run_time;
//        printf("++++++++++++++++++++++++++++++++++++++\n Trial %d, Elapsed Time: %.5fs. \n"
//               "++++++++++++++++++++++++++++++++++++++\n", i, run_time);
//        cout.flush();
//
//        printf("\n---------------------\nITER:%d\n---------------------\n", i);
//        if (i % plot_itr == 0) {//i % 50 == 0 &&
//            PyObject *pName, *pModule, *pFunc;
//
//            Py_Initialize();
//            if (_import_array() < 0)
//                PyErr_Print();
//            npy_intp dim[3] = {4, 200, 2};
//            /*
//             * scalar res[1][200][2] = {}, ber[1][200][2] = {}, tim[1][200][2] = {}, spu[1][200][2] = {}, spu2[1][200][2] = {};
//                scalar tim_total[1][200][2] = {}, spu_total[1][200][2] = {};
//             */
//
//            scalar proc_nums[l - 2] = {};
//            index ll = 0;
//
//            for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
//                proc_nums[ll] = n_proc;
//                ll++;
//            }
//            npy_intp dpc[1] = {ll};
//
//            PyObject *pRes = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, res);
//            PyObject *pBer = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, ber);
//            PyObject *pTim = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, tim);
//            PyObject *pSpu = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, spu);
//            PyObject *ptim_total = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, tim_total);
//            PyObject *pspu_total = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, spu_total);
//            PyObject *pPrc = PyArray_SimpleNewFromData(1, dpc, NPY_DOUBLE, proc_nums);
//
//            if (pRes == nullptr) printf("[ ERROR] pRes has a problem.\n");
//            if (pBer == nullptr) printf("[ ERROR] pBer has a problem.\n");
//            if (pTim == nullptr) printf("[ ERROR] pTim has a problem.\n");
//            if (pSpu == nullptr) printf("[ ERROR] pSpu has a problem.\n");
//            if (ptim_total == nullptr) printf("[ ERROR] ptim_total has a problem.\n");
//            if (pspu_total == nullptr) printf("[ ERROR] pspu_total has a problem.\n");
//            if (pPrc == nullptr) printf("[ ERROR] pPrc has a problem.\n");
//
//            PyObject *sys_path = PySys_GetObject("path");
//            PyList_Append(sys_path, PyUnicode_FromString(
//                    "/home/shilei/CLionprojectionects/babai_asyn/babai_asyn_c++/src/example"));
//            pName = PyUnicode_FromString("plot_helper");
//            pModule = PyImport_Import(pName);
//
//            if (pModule != nullptr) {
//                pFunc = PyObject_GetAttrString(pModule, "plot_runtime_ud_grad");
//                if (pFunc && PyCallable_Check(pFunc)) {
//                    PyObject *pArgs = PyTuple_New(16);
//                    if (PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", n)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", SNR)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", qam)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", l)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 4, Py_BuildValue("i", i)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 5, pRes) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 6, pBer) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 7, pTim) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 8, pPrc) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 9, pSpu) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 10, ptim_total) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 11, pspu_total) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 12, Py_BuildValue("i", max_proc)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 13, Py_BuildValue("i", min_proc)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 14, Py_BuildValue("i", is_constrained)) != 0) {
//                        return false;
//                    }
//                    if (PyTuple_SetItem(pArgs, 15, Py_BuildValue("i", m)) != 0) {
//                        return false;
//                    }
//
//                    PyObject *pValue = PyObject_CallObject(pFunc, pArgs);
//
//                } else {
//                    if (PyErr_Occurred())
//                        PyErr_Print();
//                    fprintf(stderr, "Cannot find function qr\n");
//                }
//            } else {
//                PyErr_Print();
//                fprintf(stderr, "Failed to load file\n");
//
//            }
//        }
//    }
//
//    printf("End of current TASK.\n");
//    printf("-------------------------------------------\n");
//    //    }
//    return 0;
//}