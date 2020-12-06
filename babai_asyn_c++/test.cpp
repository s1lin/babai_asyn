//
// Created by shilei on 2020-11-04.
//
#include "src/example/ils_block_search.cpp"
//#include "src/example/ils_babai_search.cpp"

using namespace std;

const int n1 = 4096;
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

void run_test(int argc, char *argv[]) {
    std::cout << "Maximum Threads: " << omp_get_max_threads() << std::endl;
    int max_proc = omp_get_max_threads();
    int min_proc = 6;
    int k = 1, index = 0, stop = 0, mode = 1, max_num_iter = 10;
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

void tiny_test(){
//    int n = 40996;
    printf("plot_run-------------------------------------------\n");
    std::cout << "Init, size: " << n1 << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, 1) << std::endl;
    std::cout << "Init, SNR: " << 15 << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    sils::SILS<double, int, true, n1> bsa(1, 15);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %.5f seconds\n", end_time);
    printf("-------------------------------------------\n");
    int size = 16;

    vector<int> z_B(n1, 0);
    vector<int> d(n1 / size, size), d_s(n1 / size, size);
    for (int i = d_s.size() - 2; i >= 0; i--) {
        d_s[i] += d_s[i + 1];
    }
    auto reT = bsa.sils_block_search_omp_schedule(3, 3, 0, "", &z_B, &d_s);
}

int main(int argc, char *argv[]) {
//    load_test();
//    run_test(argc, argv);
    tiny_test();
    return 0;
}

