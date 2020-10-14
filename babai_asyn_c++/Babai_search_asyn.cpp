#include "Babai_search_asyn.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(){

	int n = 2048;
    std::cout << "Init, size: " << n << std::endl;
	Babai_search_asyn bsa(n);
	bsa.init(true, true, 0.1);

    std::cout << "Serial:" << std::endl;
	bsa.find_raw_x0();

    std::cout << "OpenMP" << std::endl;
    bsa.find_raw_x0_OMP();

	return 0;
	
}
