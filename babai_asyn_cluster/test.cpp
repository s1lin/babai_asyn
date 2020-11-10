//
// Created by shilei on 2020-11-04.
//
#include "src/example/ils_block_search.cpp"
//#include "src/example/ils_babai_search.cpp"
//#include "../../SILS.cpp"

using namespace std;

const int n = 4096;
//void test_ils_search() {
//    std::cout << "Init, size: " << n << std::endl;
//
//    //bool read_r, bool read_ra, bool read_xy
//    double start = omp_get_wtime();
//    sils::SILS<double, int, true, false, n> bsa(0.1);
//    double end_time = omp_get_wtime() - start;
//    printf("Finish Init, time: %f seconds\n", end_time);
//
//    start = omp_get_wtime();
//    auto z_B = bsa.sils_search(&bsa.R_A, &bsa.y_A);
//    end_time = omp_get_wtime() - start;
//    auto res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, z_B);
//    printf("Thread: ILS, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);
//
//    sils::scalarType<double, int> z_BS = {(double *) calloc(n, sizeof(double)), n};
//    start = omp_get_wtime();
//    z_BS = *bsa.sils_babai_search_serial(&z_BS);
//    end_time = omp_get_wtime() - start;
//    res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_BS);
//    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);
//
//}
//



int main() {
    std::cout << "Maximum Threads: " << omp_get_max_threads() << std::endl;
    //plot_run();

    plot_run<double, int, n>();

    //test_ils_search();

    return 0;
}

