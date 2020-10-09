#include "Babai_search_asyn.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

int main(){
	int MAX_JOB = omp_get_max_threads();
	std::cout<<"Max_threads="<< MAX_JOB <<std::endl;

	int n = 1000, n_jobs = 12;
	Babai_search_asyn bsa(n);
	bsa.init(false, 0.5);
	VectorXd x_ser = bsa.find_raw_x0();
	for(int i = 0; i <= n_jobs; i++)
		VectorXd x_par = bsa.find_raw_x0_OMP(n_jobs, i);
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
