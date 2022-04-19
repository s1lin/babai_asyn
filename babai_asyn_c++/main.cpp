//#include "src/example/cils_standard_test.cpp"
//#include "src/example/cils_underdetermined_test.cpp"
//#include "src/example/cils_LLL_test.cpp"
#include "src/example/cils_PBNP_test.cpp"

using namespace std;
using namespace cils;

//void run_test(int argc, char *argv[], int rank) {
//
//
//    block_optimal_test<scalar, int, M, N>(rank);
//    sic_opt_test<scalar, int, M, N>();
//    switch (program_def::mode) {
//        case 0:
////            plot_LLL<scalar, index, M,  N>();
////            plot_res<scalar, int, M, N>();
//            break;
//        case 1:
////            plot_LLL<scalar, index, M,  N>();
////            plot_run<scalar, int, M, N>();
//            break;
//        case 2:
////            ils_block_search<scalar, int, N_4096>();
//            break;
//        case 3:
//            init_point_test<scalar, int, M, N>();
//        default:
////            test_ils_search<scalar, int, M, N>();
//            break;
//    }
//}

int main(int argc, char *argv[]) {


    printf("\n====================[ Run | cils | Release ]==================================\n");
    double t = omp_get_wtime();

    int start = stoi(argv[1]);
    int end = stoi(argv[2]);
//    CILS cils = cils_driver<double, int>(argc, argv);
//    plot_LLL<double, int>();
    test_PBNP<double, int>(start, end);

    t = omp_get_wtime() - t;

    printf("====================[TOTAL TIME | %2.2fs, %2.2fm, %2.2fh]==================================\n",
           t, t / 60, t / 3600);


    return 0;
}
