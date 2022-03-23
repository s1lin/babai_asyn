//#include "src/example/cils_standard_test.cpp"
#include "src/example/cils_underdetermined_test.cpp"

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
    double t;
//    MPI_Comm comm;
    int rank, size;
//    MPI_Init(nullptr, nullptr);
//    comm = MPI_COMM_WORLD;
    /* Determine the sender and receiver */
//    MPI_Comm_rank(comm, &rank);
//    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        printf("\n====================[ Run | cils | Release ]==================================\n");
        t = omp_get_wtime();

    }
//    CILS cils = cils_driver<double, int>(argc, argv);
    block_babai_test<double, int>(size, rank);

    if (rank == 0) {
        t = omp_get_wtime() - t;

        printf("====================[TOTAL TIME | %2.2fs, %2.2fm, %2.2fh]==================================\n",
               t, t / 60, t / 3600);
    }
    /* do a zero length gather */
//    MPI_Gather(NULL, 0, MPI_BYTE, NULL, 0, MPI_BYTE, 0, MPI_COMM_WORLD);

//    MPI_Finalize();

    return 0;
}
