
//#include <Python.h>
//#include <numpy/arrayobject.h>

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