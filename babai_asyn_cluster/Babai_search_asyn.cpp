#include "Babai_search_asyn.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main() {
    cout << omp_get_max_threads() << endl;

    int n = 1024;
    std::cout << "Init, size: " << n << std::endl;
    Babai_search_asyn bsa(n);
    bsa.init(false, false, 0.1);

    std::cout << "Vector Serial:" << std::endl;
    bsa.search_vec(0);

    std::cout << "OPENMP:" << std::endl;
    bsa.search_omp(10, 10, 0);
    bsa.search_omp(20, 10, 0);
    bsa.search_omp(30, 10, 0);

    return 0;
}
