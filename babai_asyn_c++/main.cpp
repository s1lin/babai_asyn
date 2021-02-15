//
// Created by shilei on 2020-11-04.

//#include "src/example/cils_test.cpp"
#include "src/example/cils_gen_test.cpp"
//#include <lapack.h>
//#include <mpi.h>

using namespace std;
//using namespace boost;
using namespace cils;

//template<typename scalar, typename index, index n>
//int mpi_test_2(int argc, char *argv[]) {
//    MPI_Datatype vec;
//    MPI_Comm comm;
//    double *vecin, *vecout;
//    int minsize = 2, count;
//    int root, i, stride, errs = 0;
//    int rank, size;
//
//    MPI_Init(&argc, &argv);
//    comm = MPI_COMM_WORLD;
//    /* Determine the sender and receiver */
//    MPI_Comm_rank(comm, &rank);
//    MPI_Comm_size(comm, &size);
//
//    for (root = 0; root < size; root++) {
//        for (count = 1; count < 65000; count = count * 2) {
////            n = 12;
//            stride = 10;
//            vecin = (double *) malloc(n * stride * size * sizeof(double));
//            vecout = (double *) malloc(size * n * sizeof(double));
//
//            MPI_Type_vector(n, 1, stride, MPI_DOUBLE, &vec);
//            MPI_Type_commit(&vec);
//
//            for (i = 0; i < n * stride; i++) vecin[i] = -2;
//            for (i = 0; i < n; i++) vecin[i * stride] = rank * n + i;
//
//            MPI_Gather(vecin, 1, vec, vecout, n, MPI_DOUBLE, root, comm);
//
//            if (rank == root) {
//                for (i = 0; i < n * size; i++) {
//                    if (vecout[i] != i) {
//                        errs++;
//                        if (errs < 10) {
//                            fprintf(stderr, "vecout[%d]=%d\n", i, (int) vecout[i]);
//                            fflush(stderr);
//                        }
//                    }
//                }
//            }
//            MPI_Type_free(&vec);
//            free(vecin);
//            free(vecout);
//        }
//    }
//
//    /* do a zero length gather */
//    MPI_Gather(NULL, 0, MPI_BYTE, NULL, 0, MPI_BYTE, 0, MPI_COMM_WORLD);
//
//    MPI_Finalize();
//    return 0;
//}


void run_test(int argc, char *argv[]) {

//    program_def::max_proc = 54;
//    program_def::min_proc = max_proc == 46 ? 4 : 2;
    program_def::init_program_def(argc, argv);

    switch (program_def::mode) {
        case 0:
            plot_res<scalar, int, N>();
            break;
        case 1:
            plot_run<scalar, int, N>();
            break;
        case 2:
//            ils_block_search<scalar, int, N_4096>();
            break;
        default:
            test_ils_search<scalar, int, N>();
            break;
    }
}

int main(int argc, char *argv[]) {
//    load_test();
//    qr_test<double, int, 4096>();
    run_test(argc, argv);
//    tiny_test();
//    mpi_test_2<double, int, 32>(argc, argv);
    return 0;
}