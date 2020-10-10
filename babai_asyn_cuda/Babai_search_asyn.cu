#include "Babai_search_asyn.cuh"
#include <ctime>

using Eigen::MatrixXd;
using Eigen::VectorXd;
__global__ void
find_raw_x0_cuda(int n_proc, int nswp, int n, double *raw_x_A, const double *y_A, const double *R_sA) {

    for (int j = 0; j < nswp; j++) {
        double sum = 0;
        for (int i = 1; i < n; i++) {
            for (int col = n - i; col < n; col++) {
                //sum += R(k, col) * raw_x(col);
                //sum += R_A[k * n + col] * raw_x_A[col];
                sum += R_sA[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * raw_x_A[col];
            }
            raw_x_A[n - 1 - i] = round(
                    (y_A[n - 1 - i] - sum) / R_sA[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
            sum = 0;
        }
    }
}

int main(){

	int n = 1000, n_jobs = 50;
	Babai_search_asyn bsa(n);
	bsa.init(false, 5);
    std::cout << "init_res: " << bsa.init_res << std::endl;

    double *x, *y_A, *R_sA;

    cudaMallocManaged(&x, n*sizeof(int));
    cudaMallocManaged(&y_A, n*sizeof(int));
    cudaMallocManaged(&R_sA, bsa.R_sA.size()*sizeof(int));

    x = bsa.x_A.data();
    y_A = bsa.y_A.data();
    R_sA = bsa.R_sA.data();

    std::clock_t start = std::clock();
	find_raw_x0_cuda<<<1, 1>>>(0, 1000, n, x, y_A, R_sA);
    cudaDeviceSynchronize();
    double time = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    double res = 0.0f;
    VectorXd x_result = VectorXd(n);
    for (int i = 0; i < n; i++) {
        x_result(i) = x[i];
    }

    res = (bsa.y - bsa.R * x_result).norm();
    std::cout << "res: " << res << std::endl;
    std::cout << "time: " << time << std::endl;

    cudaFree(x);
    cudaFree(y_A);
    cudaFree(R_sA);

    return 0;
	
}
