//
// Created by shilei on 2020-11-04.
//
#include "src/example/cils_test.cpp"

#include <mpi.h>

using namespace std;
using namespace boost;
using namespace cils;

void load_test() {
    int n = 4096;
    vector<int> d(4096 / 16, 16), d_s(4096 / 16, 16);
    for (int i = d_s.size() - 2; i >= 0; i--) {
        d_s[i] += d_s[i + 1];
    }
    int ds = d.size();
    int n_proc = 10;
    vector<int> iter(300, 0);
#pragma omp parallel default(shared) num_threads(n_proc) //private()
    {
//    for (int j = 0; j < 10; j++) {//
#pragma omp for schedule(static) nowait //schedule(dynamic)
        for (int i = 0; i < ds; i++) {
            cout << omp_get_thread_num() << ",i," << i << endl;
        }
    }
    //    }
    cout << iter.size() << endl;
    for (int m = 0; m < ds; m++) {
        cout << iter[m] << ", ";
    }

}

template<typename scalar, typename index, index n>
int mpi_test_2(int argc, char *argv[]) {
    MPI_Datatype vec;
    MPI_Comm comm;
    double *vecin, *vecout;
    int minsize = 2, count;
    int root, i, stride, errs = 0;
    int rank, size;

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    /* Determine the sender and receiver */
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    for (root = 0; root < size; root++) {
        for (count = 1; count < 65000; count = count * 2) {
//            n = 12;
            stride = 10;
            vecin = (double *) malloc(n * stride * size * sizeof(double));
            vecout = (double *) malloc(size * n * sizeof(double));

            MPI_Type_vector(n, 1, stride, MPI_DOUBLE, &vec);
            MPI_Type_commit(&vec);

            for (i = 0; i < n * stride; i++) vecin[i] = -2;
            for (i = 0; i < n; i++) vecin[i * stride] = rank * n + i;

            MPI_Gather(vecin, 1, vec, vecout, n, MPI_DOUBLE, root, comm);

            if (rank == root) {
                for (i = 0; i < n * size; i++) {
                    if (vecout[i] != i) {
                        errs++;
                        if (errs < 10) {
                            fprintf(stderr, "vecout[%d]=%d\n", i, (int) vecout[i]);
                            fflush(stderr);
                        }
                    }
                }
            }
            MPI_Type_free(&vec);
            free(vecin);
            free(vecout);
        }
    }

    /* do a zero length gather */
    MPI_Gather(NULL, 0, MPI_BYTE, NULL, 0, MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}

void run_test(int argc, char *argv[]) {
    std::cout << "Maximum Threads: " << omp_get_max_threads() << std::endl;
    program_def::max_proc = omp_get_max_threads();
    program_def::min_proc = 12;
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

//void tiny_test() {
//    const int n1 = 16;
//    int SNR = 35;
//    printf("plot_run-------------------------------------------\n");
//    std::cout << "Init, size: " << n1 << std::endl;
//    std::cout << "Init, QAM: " << std::pow(4, 1) << std::endl;
//    std::cout << "Init, SNR: " << SNR << std::endl;
//
//    //bool read_r, bool read_ra, bool read_xy
//
//    cils::cils<double, int, false, n1> cils(1, SNR);
//
//    int size = 8;
//
//    vector<int> z_B(n1, 0);
//    vector<int> d(n1 / size, size), d_s(n1 / size, size);
//    for (int i = d_s.size() - 2; i >= 0; i--) {
//        d_s[i] += d_s[i + 1];
//    }
//    for (int i = 0; i < 100; i++) {
//        cils.init(false,);
//        //auto reT = cils.cils_block_search_omp_schedule(3, 10, 0, "", &z_B, &d_s);
//        auto reT = cils.cils_block_search_serial(&z_B, &d_s);
//        auto res = cils::find_residual<double, int, n1>(cils.R_A, cils.y_A, reT.x);
//        auto brr = cils::find_bit_error_rate<double, int, n1>(reT.x, &cils.x_t, false);
//        printf("Method: ILS_OMP, Num of Threads: %d, Block size: %d, Iter: %d, Res: %.5f, BER: %.5f, Run time: %.5fs\n",
//               3, size, reT.num_iter, res, brr, reT.run_time);
//    }
//}

int main(int argc, char *argv[]) {
//    load_test();
    run_test(argc, argv);
//    tiny_test();
//    mpi_test_2<double, int, 32>(argc, argv);
    return 0;
}
