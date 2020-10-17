#include "Babai_search_asyn.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {

    int n = 8192, init_value = 0, n_proc = 12, nswp = 800;
    std::cout << "Init, size: " << n << std::endl;
    Babai_search_asyn bsa(n);
    bsa.init(true, true, 0.1);
    //bsa.init(false, false, 0.1);

//    std::cout << "Eigen Serial:" << std::endl;
//    bsa.search_eigen(init_value);
//
//    std::cout << "Vector Serial:" << std::endl;
//    bsa.search_vec(init_value);

    std::cout << "OpenMP" << std::endl;
    double time = 0;
    for (int i = 0; i < 20; i++) {
        time += bsa.search_omp(n_proc, nswp, init_value);
    }
    cout << time / 20;

//    for (int n_proc = 100; n_proc <= 200; n_proc += 10) {
//        for (int nswp = 0; nswp < 21; nswp += 1) {
//            double res = bsa.search_omp(n_proc, nswp);
//            //if(res <= bsa.init_res)
//            //    break;
//        }
//    }

    return 0;
}

