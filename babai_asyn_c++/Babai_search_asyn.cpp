#include "Babai_search_asyn.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {

    int n = 8192;
    std::cout << "Init, size: " << n << std::endl;
    Babai_search_asyn bsa(n);
    bsa.init(true, true, 0.1);

    std::cout << "Eigen Serial:" << std::endl;
    bsa.search_eigen();

    std::cout << "Vector Serial:" << std::endl;
    bsa.search_vec();

    std::cout << "OpenMP" << std::endl;
    for (int n_proc = 100; n_proc <= 200; n_proc += 10) {
        for (int nswp = 0; nswp < 21; nswp += 1) {
            double res = bsa.search_omp(n_proc, nswp);
            //if(res <= bsa.init_res)
            //    break;
        }
    }

    return 0;
}

