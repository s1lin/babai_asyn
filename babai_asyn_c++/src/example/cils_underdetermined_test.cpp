//
// Created by shilei on 7/27/21.
//

#include "../source/cils.cpp"
//#include "../source/cils_ils_search.cpp"
//#include "../source/cils_block_search.cpp"
//#include "../source/cils_babai_search.cpp"
#include "../source/cils_reduction.cpp"
#include "../source/cils_init_point.cpp"
#include <ctime>

template<typename scalar, typename index, index m, index n>
void init_point_test() {
    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | INIT_POINT | %s ]==================================\n", time_str);


    cils::cils<scalar, index, m, n> cils(k, 35);
    cils.init_ud();
    cout << "A:" << endl;
    cils::display_matrix<scalar, index, m, n>(cils.A);
    cout << "HH_SIC" << endl;
    cils::display_matrix<scalar, index, m, n>(cils.H);

    vector<scalar> x(n, 0);
    scalar v_norm;
    array<scalar, m * n> A_t;
    array<scalar, n * n> P;
    cils::returnType<scalar, index> reT;

    reT = cils.cils_sic_serial(x, A_t, P);
    cout << reT.num_iter << ", " << reT.run_time << endl;
    cils::display_matrix<scalar, index, m, n>(A_t);

    cils.cils_qr_serial(1, 0);
    cils::display_vector<scalar, index, m>(cils.y_q);
    reT = cils.cils_qrp_serial(x, A_t, P);
    cout << reT.num_iter << ", " << reT.run_time << endl;
    cils::display_matrix<scalar, index, m, n>(A_t);

}