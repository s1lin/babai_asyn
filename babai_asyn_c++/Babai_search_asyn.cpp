#include "Babai_search_asyn.h"
#include "matplotlibcpp.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(){
	int MAX_JOB = omp_get_max_threads();
	std::cout<<"Max_threads="<< MAX_JOB <<std::endl;

	int n = 2048, n_jobs = 50;
	Babai_search_asyn bsa(n);
	bsa.init(true, 5);
	VectorXd x_ser = bsa.find_raw_x0();

//    matplotlibcpp::plot({1,3,2,4});
//    matplotlibcpp::show();
//    matplotlibcpp::save("basic.png");

    std::cout << "find_raw_x0_OMP" << std::endl;
	for(int nswp = 5; nswp <= n_jobs; nswp++)
	    for(int j = 10; j <= n_jobs; j++)
		    VectorXd x_par = bsa.find_raw_x0_OMP(j, nswp);


	//std::cout << "X distance=" << (x_par - x_ser).norm() << std::endl;

	//Babai_search_cuda bsa_cuda(n, nswp, 1);
	//VectorXd x_ser = bsa.find_raw_x0();
	//std::cout << "X distance=" << (x_par - x_ser).norm() << std::endl;

	
//	int np = 0, id = 0;
//	omp_set_num_threads(8);
//
//#pragma omp parallel
//	{
//		for (int j = 0; j <= 5; j++) {
//
//#pragma omp for nowait
//			for (int i = 5; i >= 0; i--) {
//				np = omp_get_num_threads();
//				id = omp_get_thread_num();
//
//				printf("Hello from thread % d out of % d threads\n", j, i);
//			}
//		}
//
//	}
	
	return 0;
	
}
