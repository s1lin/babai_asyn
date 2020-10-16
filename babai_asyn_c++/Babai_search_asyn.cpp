#include "Babai_search_asyn.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {

    int n = 2048;
    std::cout << "Init, size: " << n << std::endl;
    Babai_search_asyn bsa(n);
    bsa.init(true, true, 0.1);

    std::cout << "Eigen Serial:" << std::endl;
    bsa.search_eigen();

    std::cout << "Vector Serial:" << std::endl;
    bsa.search_vec();

    std::cout << "OpenMP" << std::endl;
    for (int n_proc = 5; n_proc <= 50; n_proc += 1) {
        for (int nswp = 0; nswp < 21; nswp += 1)
            bsa.search_omp();
    }

    return 0;
}

