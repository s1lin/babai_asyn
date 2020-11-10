#include "../babai_asyn_c++/src/include/SILS.h"
#include <ctime>

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace plt = matplotlibcpp;
__global__ void
find_raw_x0_cuda(int n, double *x_A, double *x_Next_A, const double *y_A, const double *R_sA) {

//    for (int j = 0; j < nswp; j++) {
//        double sum = 0;
//        for (int i = 1; i < n; i++) {
//            for (int col = n - i; col < n; col++) {
//                sum += R_sA[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * raw_x_A[col];
//            }
//            raw_x_A[n - 1 - i] = round(
//                    (y_A[n - 1 - i] - sum) / R_sA[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
//            sum = 0;
//        }
//    }
//tiling
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        double sum = 0.0;
//
//        // store index in register
//        // Multiplication is not executed in every iteration.
//        //int idx_Ai = idx * n;
        for (int col = idx + 1; col < n; col++)
            sum += R_sA[idx * n + col] * x_A[col];
        x_A[idx] = round((y_A[idx] - sum) / R_sA[idx * n + idx]);


//        int idx_Ai = idx * n;
//        for (int j = 0; j < n; j++)
//            if (idx != j)
//                sum += R_sA[idx_Ai + j] * x_A[j];
//        x_Next_A[idx] = round((y_A[idx] - sum) / R_sA[idx * n + idx]);
    }

}

void testDevice(int devID) {
    // Check if we can run. Maybe do something more...
    cudaDeviceProp deviceProp;

    cudaGetDeviceProperties(&deviceProp, devID);
    if (deviceProp.major == 9999 && deviceProp.minor == 9999) {   /* Simulated device. */
        printf("There is no device supporting CUDA.\n");
        cudaThreadExit();
    } else
        printf("Using GPU device number %d.\n", devID);
}

double run(int n, int nswp, Babai_search_asyn bsa) {
    double *x, *x_A, *x_Next_A, *y_A, *R_sA;

    x = (double *) malloc(n * sizeof(double));

    cudaMallocManaged(&x_A, n * sizeof(double));
    cudaMallocManaged(&x_Next_A, n * sizeof(double));
    cudaMallocManaged(&y_A, n * sizeof(double));
    cudaMallocManaged(&R_sA, bsa.R_A.size() * sizeof(double));

    //x = bsa.x_A.data();
    x[n - 1] = round(bsa.y(n - 1) / bsa.R(n - 1, n - 1));
    for (int i = 0; i < n; i++) {
        y_A[i] = bsa.y_A[i];
    }
    for (int i = 0; i < bsa.R_A.size(); i++) {
        R_sA[i] = bsa.R_A[i];
    }
    x_A[n - 1] = x_Next_A[n - 1] = x[n - 1];

    cudaMemcpy(y_A, bsa.y_A.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(R_sA, bsa.R_A.data(), bsa.R_A.size() * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_A, x, n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(x_Next_A, x, n * sizeof(double), cudaMemcpyHostToDevice);

    int tileSize = 4;
    // Optimized kernel
    int nTiles = n / tileSize + (n % tileSize == 0 ? 0 : 1);
    int gridHeight = n / tileSize + (n % tileSize == 0 ? 0 : 1);
    int gridWidth = n / tileSize + (n % tileSize == 0 ? 0 : 1);
    //printf("w=%d, h=%d\n", gridWidth, gridHeight);
    dim3 dGrid(gridHeight, gridWidth), dBlock(tileSize, tileSize);

    std::clock_t start = std::clock();
    for (int k = 0; k < nswp; k++) {
        //if (k % 2)
            find_raw_x0_cuda<<<nTiles, tileSize>>>(n, x_A, x_Next_A, y_A, R_sA);
        //else
            //find_raw_x0_cuda<<<nTiles, tileSize>>>(n, x_Next_A, x_A, y_A, R_sA);


    }

    cudaDeviceSynchronize();
    double time = (std::clock() - start) / (double) CLOCKS_PER_SEC;

    double res = 0.0f;
    VectorXd x_result = VectorXd(n);

    for (int i = 0; i < n; i++) {
        x_result(i) = x_A[i];
    }

    res = (bsa.y - bsa.R * x_result).norm();

    printf("Sweep: %d, Res: %.5f, Run time: %fs\n", nswp, res, time);

    cudaFree(x_A);
    cudaFree(x_Next_A);
    cudaFree(y_A);
    cudaFree(R_sA);
    free(x);
    return res;
}

int main() {

    testDevice(0);

    int n = 2048, n_jobs = 50;
    Babai_search_asyn bsa(n);

    bsa.init(true, true, 0.1);

    std::cout << "find_raw_x0" << std::endl;
    bsa.search_vec(0);

//    std::cout << "find_raw_x0_OMP" << std::endl;
//    for(int nswp = 5; nswp <= n_jobs; nswp++)
//        for(int j = 10; j <= n_jobs; j++)
//            VectorXd x_par = bsa.find_raw_x0_OMP(j, nswp);

    std::vector<double> nswp_pl(20), res_pl(20), tim_pl(20);
    for (int nswp = 0; nswp < 20; nswp++) {
        nswp_pl.push_back(nswp);
        res_pl.push_back(bsa.init_res);
        tim_pl.push_back(bsa.search_vec(0));
    }

    const std::map<std::string, std::string> keyword_arg{
            {"marker",     "o"},
            {"markersize", "5"},
            {"label",      "Serial"}
    };

    plt::xlim(1, 20);
    plt::plot(nswp_pl, tim_pl, keyword_arg);

    const std::map<std::string, std::string> keyword_arg2{
            {"marker",     "1"},
            {"markersize", "5"},
            {"label",      "Matlab"}
    };

    string tim =
            "/home/shilei/CLionProjects/babai_asyn/data/Res_" + to_string(n) + ".csv";
    string row_string, entry;
    int index = 0;
    vector<double> nswp_pl2(20), tim_pl2(20);
    ifstream f1(tim);
    while (getline(f1, row_string)) {
        double d = stod(row_string);
        nswp_pl2.push_back(index);
        tim_pl2.push_back(d);
        index++;
    }

    plt::xlim(1, 20);
    plt::plot(nswp_pl2, tim_pl2, keyword_arg2);

    std::cout << "find_raw_x0_CUDA" << std::endl;
    vector<double> nswp_pl3(20), tim_pl3(20);
    for (int nswp = 10; nswp <= 200; nswp += 10) {
        double time = run(n, nswp, bsa);
        tim_pl3.push_back(time);
    }


    const std::map<std::string, std::string> keyword_arg3{
            {"marker",     "x"},
            {"markersize", "5"},
            {"label",      "GPU"}
    };

    plt::xlim(1, 20);
    plt::plot(nswp_pl, tim_pl3, keyword_arg3);

    plt::title("Residual with GPU");
    plt::legend();
    plt::xlabel("Num of iterations x 10");
    plt::ylabel("Residual");
    plt::save("./resCUDA.png");

    return 0;

}
