//
// Created by shilei on 2020-11-04.
//
#include "src/example/cils_test.cpp"
//#include <lapack.h>
#include <mpi.h>

using namespace std;
//using namespace boost;
using namespace cils;

//template<typename scalar, typename index, index n>
//void qr(scalar *Qx, scalar *Rx, scalar *Ax) {
//    // Maximal rank is used by Lapacke
//    const size_t rank = n;
//    const index N = n;
//
//    // Tmp Array for Lapacke
//    const std::unique_ptr<scalar[]> tau(new scalar[rank]);
//    const std::unique_ptr<scalar[]> work(new scalar[rank]);
//    index info = 0;
//
//    // Calculate QR factorisations
//    dgeqrf_(&N, &N, Ax, &N, tau.get(), work.get(), &N, &info);
//
//    cout<<"here";
//    cout.flush();
//    // Copy the upper triangular Matrix R (rank x _n) into position
//    for (size_t row = 0; row < rank; ++row) {
//        memset(Rx + row * N, 0, row * sizeof(scalar)); // Set starting zeros
//        memcpy(Rx + row * N + row, Ax + row * N + row,(N - row) * sizeof(scalar)); // Copy upper triangular part from Lapack result.
//    }
//
//    // Create orthogonal matrix Q (in tmpA)
//    dorgqr_(&N, &N, &N, Ax, &N, tau.get(), work.get(), &N, &info);
//
//    //Copy Q (_m x rank) into position
//    memcpy(Qx, Ax, sizeof(scalar) * (N * N));
//}
//
//template<typename scalar, typename index, index n>
//void qr_test() {
//    std::random_device rd;
//    std::mt19937 gen(rd());
//    index qam = 3, snr = 35;
//    auto sigma = (scalar) sqrt(((pow(4, qam) - 1) * log2(n)) / (6 * pow(10, ((scalar) snr / 10.0))));
//
//    //mean:0, std:sqrt(1/2). same as matlab.
//    std::normal_distribution<scalar> A_norm_dis(0, sqrt(0.5)), v_norm_dis(0, sigma);
//    //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
//    std::uniform_int_distribution<index> int_dis(0, pow(2, 3) - 1);
//    auto A = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
//    A->x = new scalar[n * n]();
//    A->size = n * n;
//
//    auto v_A = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
//    v_A->x = new scalar[n]();
//    v_A->size = n;
//
//    auto R = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
//    R->x = new scalar[n * n]();
//    R->size = n * n;
//
//    auto Q = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
//    Q->x = new scalar[n * n]();
//    Q->size = n * n;
//
//
//#pragma omp parallel for
//    for (index i = 0; i < n * n; i++) {
//        A->x[i] = A_norm_dis(gen);
//        if (i < n) {
////            x_t[i] = int_dis(gen);
//            v_A->x[i] = v_norm_dis(gen);
//        }
//    }
//    qr<scalar, index, n>(Q->x, R->x, A->x);
////    for (index i = 0; i < n * n; i++) {
////        cout << R->x[i] << " ";
////    }
//    cout << "finished";
//}

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
            plot_res<scalar, int, N_4096>();
            break;
        case 1:
            plot_run<scalar, int, N_4096>();
            break;
        default:
            ils_block_search<scalar, int, N_4096>();
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