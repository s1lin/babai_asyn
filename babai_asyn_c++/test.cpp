//
// Created by shilei on 2020-11-04.
//
#include "src/example/ils_block_search.cpp"
//#include "src/example/ils_babai_search.cpp"
#include <mpi.h>

#define BUFMAX 81
using namespace std;
using namespace boost;

const int n1 = 64;
const int n2 = 8192;
const int n3 = 16384;
const int n4 = 32768;

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
#pragma omp for schedule(dynamic) nowait //schedule(dynamic)
        for (int i = 0; i < 2 * n_proc; i++) {
//            if (omp_get_thread_num()==n_proc) {
//            for (int m = i; m < ds; m += n_proc) {
//                iter[m] = i;
//            }
//            }
            if (omp_get_thread_num() == 0) {
                cout << omp_get_thread_num() << ",i," << i << endl;
            }

        }
    }
    //    }
    cout << iter.size() << endl;
    for (int m = 0; m < ds; m++) {
        cout << iter[m] << ", ";
    }

}

template<typename scalar, typename index, index n>
int mpi_test(int argc, char *argv[]) {
    int k = 1, index_2 = 0, stop = 0, mode = 1, max_num_iter = 2000;
    if (argc != 1) {
        k = stoi(argv[1]);
        index_2 = stoi(argv[2]);
        stop = stoi(argv[3]);
        mode = stoi(argv[4]);
        max_num_iter = stoi(argv[5]);
    }

    int rank, n_ranks, numbers_per_rank;
    int my_first, my_last;

    // First call MPI_Init
    MPI_Init(&argc, &argv);
    // Get my rank and the number of ranks
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // Calculate the number of iterations for each rank
    numbers_per_rank = floor(max_num_iter / n_ranks);
    if (max_num_iter % n_ranks > 0) {
        // Add 1 in case the number of ranks doesn't divide the number of numbers
        numbers_per_rank += 1;
    }

    // Figure out the first and the last iteration for this rank
    my_first = rank * numbers_per_rank;
    my_last = my_first + numbers_per_rank;
    printf("my_first: %d, my_last: %d\n", my_first, my_last);
    plot_run_mpi<double, int, n1>(my_first, my_last,
                                  k, 15, 3, 8, my_last - my_first, stop);
    MPI_Finalize();
    if (rank == 0) {
        for (int i = 0; i < max_num_iter; i++) {
            cout << argv[i] << " ";
        }
    }

    // Call finalize at the end
    return 0;
}

void run_test(int argc, char *argv[]) {
    std::cout << "Maximum Threads: " << omp_get_max_threads() << std::endl;
    int max_proc = omp_get_max_threads();
    int min_proc = 6;
    int k = 1, index = 0, stop = 0, mode = 1, max_num_iter = 100;
    if (argc != 1) {
        k = stoi(argv[1]);
        index = stoi(argv[2]);
        stop = stoi(argv[3]);
        mode = stoi(argv[4]);
        max_num_iter = stoi(argv[5]);
    }
    max_proc = max_proc != 64 ? max_proc : 100;
    min_proc = max_proc != 64 ? 6 : 12;

    for (int SNR = 15; SNR <= 35; SNR += 10) {
        switch (index) {
            case 0:
                if (mode == 0)
                    plot_res<double, int, n1>(k, SNR, min_proc, max_proc);
                else if (mode == 1)
                    plot_run<double, int, n1>(k, SNR, min_proc, max_proc, max_num_iter, stop);
                else
                    ils_block_search<double, int, n1>(k, SNR);
                break;
            case 1:
                plot_res<double, int, n2>(k, SNR, min_proc, max_proc);
                plot_run<double, int, n2>(k, SNR, min_proc, max_proc, max_num_iter, stop);
//                ils_block_search<double, int, n2>(k, SNR);
                break;
            case 2:
                plot_run<double, int, n3>(k, SNR, min_proc, max_proc, max_num_iter, stop);
//                ils_block_search<double, int, n3>(k, SNR);
                break;
            default:
                plot_res<double, int, n4>(k, SNR, min_proc, max_proc);
                break;
        }

    }

}

void tiny_test() {
    const int n1 = 16;
    int SNR = 35;
    printf("plot_run-------------------------------------------\n");
    std::cout << "Init, size: " << n1 << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, 1) << std::endl;
    std::cout << "Init, SNR: " << SNR << std::endl;

    //bool read_r, bool read_ra, bool read_xy

    sils::sils<double, int, false, n1> sils(1, SNR);

    int size = 8;

    vector<int> z_B(n1, 0);
    vector<int> d(n1 / size, size), d_s(n1 / size, size);
    for (int i = d_s.size() - 2; i >= 0; i--) {
        d_s[i] += d_s[i + 1];
    }
    for (int i = 0; i < 100; i++) {
        sils.init();
        //auto reT = sils.sils_block_search_omp_schedule(3, 10, 0, "", &z_B, &d_s);
        auto reT = sils.sils_block_search_serial(&z_B, &d_s);
        auto res = sils::find_residual<double, int, n1>(sils.R_A, sils.y_A, &reT.x);
        auto brr = sils::find_bit_error_rate<double, int, n1>(&reT.x, &sils.x_t);
        printf("Method: ILS_OMP, Num of Threads: %d, Block size: %d, Iter: %d, Res: %.5f, BER: %.5f, Run time: %.5fs\n",
               3, size, reT.num_iter, res, brr, reT.run_time);
    }
}

int main(int argc, char *argv[]) {
//    load_test();
//    run_test(argc, argv);
//    tiny_test();
    mpi_test<double, int, 32>(argc, argv);
    return 0;
}

