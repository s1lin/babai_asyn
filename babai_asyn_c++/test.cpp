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


int main(int argc, char *argv[]) {
    std::cout << "Maximum Threads: " << omp_get_max_threads() << std::endl;
    int max_proc = omp_get_max_threads();
    int k = 3, index = 0;
    if (argc != 1) {
        k = stoi(argv[1]);
        index = stoi(argv[2]);
    }
    max_proc = max_proc != 64? max_proc : 100;
    //plot_run();

    for (int SNR = 15; SNR <= 35; SNR += 20) {
        switch(index){
            case 0:
                plot_res<double, int, n1>(k, SNR, max_proc);
                plot_run<double, int, n1>(k, SNR, max_proc, -1);//NON-STOP
                cout<<"STOP--------------------";
                plot_run<double, int, n1>(k, SNR, max_proc, 0.5);//NON-STOP
//                ils_block_search<double, int, n1>(k, SNR);
                break;
            case 1:
                plot_res<double, int, n2>(k, SNR, max_proc);
                plot_run<double, int, n2>(k, SNR, max_proc, -1);//NON-STOP
                cout<<"STOP--------------------";
                plot_run<double, int, n2>(k, SNR, max_proc, 0.5);//NON-STOP
//                ils_block_search<double, int, n2>(k, SNR);
                break;
            case 2:
                plot_res<double, int, n3>(k, SNR, max_proc);
                plot_run<double, int, n3>(k, SNR, max_proc, -1);//NON-STOP
                cout<<"STOP--------------------";
                plot_run<double, int, n3>(k, SNR, max_proc, 0.5);//NON-STOP
//                ils_block_search<double, int, n3>(k, SNR);
//                plot_run<double, int, n3>(k, SNR);
                break;
            default:
                plot_res<double, int, n4>(k, SNR, max_proc);
//                ils_block_search<double, int, n4>(k, SNR);
//                plot_run<double, int, n4>(k, SNR);
                break;
        }

    }

    return 0;
}

