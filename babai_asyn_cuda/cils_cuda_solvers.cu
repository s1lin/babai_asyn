#include "../babai_asyn_c++/src/source/sils.cpp"
#include "cils_cuda_solvers.cuh"
#include <ctime>

namespace sils {

    template<typename scalar, typename index, index n>
    void testDevice(index devID) {
        // Check if we can run. Maybe do something more...
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, devID);
        if (deviceProp.major == 9999 && deviceProp.minor == 9999) {   /* Simulated device. */
            printf("There is no device supporting CUDA.\n");
            cudaDeviceSynchronize();
        } else
            printf("Using GPU device number %d.\n", devID);
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    sils<scalar, index, is_read, n>::sils_babai_search_cuda(index nswp, vector<index> *z_B) {
        scalar *z_B_c, *z_B_p, *y_A_c, *R_A_c;
        scalar *z_B_c_h, *z_B_p_h, *y_A_c_h, *R_A_c_h;

        z_B_c_h = (scalar *) malloc(n * sizeof(scalar));
//        z_B_p_h = (scalar*)malloc(n * sizeof(scalar));
//        y_A_c_h = (scalar*)malloc(n * sizeof(scalar));
        R_A_c_h = (scalar *) malloc(n * n * sizeof(scalar));

        cudaMallocManaged(&z_B_c, n * sizeof(scalar));
        cudaMallocManaged(&z_B_p, n * sizeof(scalar));
        cudaMallocManaged(&y_A_c, n * sizeof(scalar));
        cudaMallocManaged(&R_A_c, n * n * sizeof(scalar));

        cudaMemset(&R_A_c, 0, n * n * sizeof(scalar));

        //x = z_B_c.data();
        index end_1 = n - 1;
        z_B->at(end_1) = round(y_A->x[end_1] / R_A->x[(n * end_1) + end_1 - ((end_1 * (end_1 + 1)) / 2)]);

#pragma omp parallel default(shared) for
        for (index row = 0; row < n; row++) {
            for (index col = row; col < n; col++) {
                index i = (n * row) + col - ((row * (row + 1)) / 2);
                R_A_c_h[n * row + col] = R_A->x[i];
            }
            z_B_c_h[row] = z_B->at(row);
        }

        cudaMemcpy(y_A_c, y_A->x, n * sizeof(scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(R_A_c, R_A_c_h, n * n * sizeof(scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(z_B_c, z_B_c_h, n * sizeof(scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(z_B_p, z_B->data(), n * sizeof(scalar), cudaMemcpyHostToDevice);

        index tileSize = 8;
        // Optimized kernel
        index nTiles = n / tileSize + (n % tileSize == 0 ? 0 : 1);
        index gridHeight = n / tileSize + (n % tileSize == 0 ? 0 : 1);
        index gridWidth = n / tileSize + (n % tileSize == 0 ? 0 : 1);
        dim3 dGrid(gridHeight, gridWidth), dBlock(tileSize, tileSize);

        std::clock_t start = std::clock();
        for (index k = 0; k < nswp; k++) {
            babai_solve_cuda<scalar, index, n><<<nTiles, tileSize>>>(z_B_c, z_B_p, y_A_c, R_A_c);
        }
        cudaDeviceSynchronize();
        scalar run_time = (std::clock() - start) / (scalar) CLOCKS_PER_SEC;
        cudaMemcpy(z_B_c_h, z_B_c, n * sizeof(scalar), cudaMemcpyDeviceToHost);
        for (index row = 0; row < n; row++) {
            z_B->at(row) = z_B_c_h[row];
        }

        cudaFree(z_B_c);
        cudaFree(z_B_p);
        cudaFree(y_A_c);
        cudaFree(R_A_c);
        free(z_B_c_h);
        free(R_A_c_h);

        returnType<scalar, index> reT = {*z_B, run_time, 0, 0};
        return reT;
    }

}
