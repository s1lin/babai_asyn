#include "Babai_search_asyn.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
    cout << omp_get_max_threads() << endl;

    int n = 4096;
    std::cout << "Init, size: " << n << std::endl;
    Babai_search_asyn bsa(n);
    bsa.init(true, true, 0.1);

    std::cout << "Vector Serial:" << std::endl;
    bsa.search_vec(0);

    std::cout << "OPENMP:" << std::endl;
    for (int n_proc = 5; n_proc <= 40; n_proc += 5)
        bsa.search_omp(n_proc, 10, 0);

    return 0;
}
