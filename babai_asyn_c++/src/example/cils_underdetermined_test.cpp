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
void block_optimal_test() {
    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | SIC_OPT | %s ]==================================\n", time_str);


    cils::cils<scalar, index, m, n> cils(k, 35);
    cils.init_ud();

    vector<scalar> x_q(n, 0), x_tmp(n, 0), x_ser(n, 0), x_omp(n, 0);
    scalar v_norm, v_norm_qr;
    cils::returnType<scalar, index> reT, reT2;

    //STEP 1: init point by QRP
    reT = cils.cils_qrp_serial(x_q);
    helper::display_vector<scalar, index>(n, x_q.data(), "x");
    v_norm_qr = reT.info;
    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_q.data(), x_tmp.data());
    index diff = helper::length_nonzeros<scalar, index, n>(x_tmp.data(), cils.x_t.data());
    printf("error_bits: %d, v_norm: %8.4f, time: %8.4f\n", diff, v_norm, reT.run_time);


    //STEP 2: Block SCP:
    x_ser.assign(x_q.begin(), x_q.end());
    reT2 = cils.cils_scp_block_optimal_serial(x_ser, v_norm_qr, 1e-6, 3000);
    v_norm = reT2.info;
    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_ser.data(), x_tmp.data());

    //Result Validation:
    helper::display_vector<scalar, index>(n, x_ser.data(), "x_z");
    helper::display_vector<scalar, index>(n, cils.x_t.data(), "x_t");
    helper::display_vector<scalar, index>(n, x_tmp.data(), "x_p");
    diff = helper::length_nonzeros<scalar, index, n>(x_tmp.data(), cils.x_t.data());
    printf("error_bits: %d, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
           diff, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);

    //STEP 2: OMP-Block SCP:
    x_omp.assign(x_q.begin(), x_q.end());
    reT2 = cils.cils_scp_block_optimal_omp(x_omp, v_norm_qr, 1e-6, 3000);
    v_norm = reT2.info;
    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_omp.data(), x_tmp.data());

    //Result Validation:
    helper::display_vector<scalar, index>(n, x_omp.data(), "x_z");
    helper::display_vector<scalar, index>(n, cils.x_t.data(), "x_t");
    helper::display_vector<scalar, index>(n, x_tmp.data(), "x_p");
    diff = helper::length_nonzeros<scalar, index, n>(x_tmp.data(), cils.x_t.data());
    printf("error_bits: %d, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
           diff, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);
}