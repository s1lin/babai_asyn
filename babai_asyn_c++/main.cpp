//#include "src/example/cils_standard_test.cpp"
#include "src/example/cils_underdetermined_test.cpp"

using namespace std;
using namespace cils;

void run_test(int argc, char *argv[], int rank) {


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
}

int main(int argc, char *argv[]) {
    double t;
    MPI_Comm comm;
    int rank, size;
    MPI_Init(nullptr, nullptr);
    comm = MPI_COMM_WORLD;
    /* Determine the sender and receiver */
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (rank == 0) {
        printf("\n====================[ Run | cils | Release ]==================================\n");
        t = omp_get_wtime();

    }
    program_def::init_program_def(argc, argv);
    plot_run<scalar, int, M, N>(size, rank);
//    block_optimal_test<scalar, int, M, N>(size, rank);

    if (rank == 0) {
        t = omp_get_wtime() - t;

        printf("====================[TOTAL TIME | %2.2fs, %2.2fm, %2.2fh]==================================\n",
               t, t / 60, t / 3600);
    }
    /* do a zero length gather */
    MPI_Gather(NULL, 0, MPI_BYTE, NULL, 0, MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}

//int main(int argc, char *argv[]) {
//    int numprocs, rank, namelen;
//    char processor_name[MPI_MAX_PROCESSOR_NAME];
//    int iam = 0, np = 1;
//
//    MPI_Init(&argc, &argv);
//    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//    MPI_Get_processor_name(processor_name, &namelen);
//
//  #pragma omp parallel default(shared) private(iam, np)
//    {
//        np = omp_get_num_threads();
//        iam = omp_get_thread_num();
//        printf("Hello from thread %d out of %d from process %d out of %d on %s\n",
//               iam, np, rank, numprocs, processor_name);
//    }
//
//    MPI_Finalize();
//}