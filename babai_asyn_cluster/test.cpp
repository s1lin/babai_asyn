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

    vector<int> d(4096 / 16, 16);
    sils::scalarType<int, int> d_s{d.data(), (int) d.size()};
    for (int i = d_s.size - 2; i >= 0; i--) {
        d_s.x[i] += d_s.x[i + 1];
    }
    int ds = d.size();
    int n_proc = 10;
    vector<int> iter(300, 0);
#pragma omp parallel default(shared) num_threads(n_proc) //private()
    {
//    for (int j = 0; j < 10; j++) {//
#pragma omp for nowait //schedule(dynamic)
        for (int i = 0; i < n_proc; i++) {
//            if (omp_get_thread_num()==n_proc) {
            for (int m = i; m < ds; m += n_proc) {
                iter[m] = i;
            }
//            }

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
    int k = 3, index = 0;
    if (argc != 1) {
        k = stoi(argv[1]);
        index = stoi(argv[2]);
    }
    max_proc = max_proc != 64 ? max_proc : 100;
    min_proc = max_proc != 64 ? 6 : 12;

    //plot_run();

    for (int SNR = 15; SNR <= 35; SNR += 20) {
        switch (index) {
            case 0:
                plot_res<double, int, n1>(k, SNR, min_proc, max_proc);
                plot_run<double, int, n1>(k, SNR, min_proc, max_proc, -1);//NON-STOP
                plot_run<double, int, n1>(k, SNR, min_proc, max_proc, 0);//NON-STOP
                plot_run<double, int, n1>(k, SNR, min_proc, max_proc, 1);//NON-STOP
                plot_run<double, int, n1>(k, SNR, min_proc, max_proc, 8);//NON-STOP
//                    ils_block_search<double, int, n1>(k, SNR);
                break;
            case 1:
                plot_res<double, int, n2>(k, SNR, min_proc, max_proc);
                plot_run<double, int, n2>(k, SNR, min_proc, max_proc, -1);//NON-STOP
                plot_run<double, int, n2>(k, SNR, min_proc, max_proc, 0);//NON-STOP
                plot_run<double, int, n2>(k, SNR, min_proc, max_proc, 1);//NON-STOP
                plot_run<double, int, n2>(k, SNR, min_proc, max_proc, 5);//NON-STOP
//                ils_block_search<double, int, n2>(k, SNR);
                break;
            case 2:
//                plot_res<double, int, n3>(k, SNR, min_proc, max_proc);
//                plot_run<double, int, n3>(k, SNR, min_proc, max_proc, -1);//NON-STOP
//                plot_run<double, int, n3>(k, SNR, min_proc, max_proc, 0);//NON-STOP
//                plot_run<double, int, n3>(k, SNR, min_proc, max_proc, 1);//NON-STOP
                plot_run<double, int, n3>(k, SNR, min_proc, max_proc, 5);//NON-STOP
//                ils_block_search<double, int, n3>(k, SNR);
                break;
            default:
                plot_res<double, int, n4>(k, SNR, min_proc, max_proc);
                break;
        }

    }

}

int main(int argc, char *argv[]) {
//    load_test();
    run_test(argc, argv);
    return 0;
}

