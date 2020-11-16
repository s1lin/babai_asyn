//
// Created by shilei on 2020-11-04.
//
#include "src/example/ils_block_search.cpp"
//#include "src/example/ils_babai_search.cpp"

using namespace std;

const int n = 4096;
//const int n = 8192;
//const int n = 16384;
//const int n = 32648;


int main() {
    std::cout << "Maximum Threads: " << omp_get_max_threads() << std::endl;
    //plot_run();
    for (int k = 1; k<=3; k++){
        for(int SNR = 15; SNR<=45; SNR+= 10){
            plot_run<double, int, n>(k, SNR);
        }
    }
    return 0;
}

