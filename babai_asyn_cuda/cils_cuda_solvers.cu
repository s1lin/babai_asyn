#include "../babai_asyn_c++/src/source/cils.cpp"
#include "../babai_asyn_c++/src/source/cils_babai_search.cpp"
#include "../babai_asyn_c++/src/source/cils_block_search.cpp"
#include "cils_cuda_solvers.cuh"
#include "cuda_runtime_api.h"
#include <ctime>

namespace cils {

    template<typename scalar, typename index, index n>
    void testDevice(index devID) {
        // Check if we can run. Maybe do something more...
        cudaDeviceProp deviceProp{};
        cudaGetDeviceProperties(&deviceProp, devID);
        if (deviceProp.major == 9999 && deviceProp.minor == 9999) {   /* Simulated device. */
            printf("There is no device supporting CUDA.\n");
        } else {
            index cores = 0;
            index mp = deviceProp.multiProcessorCount;
            switch (deviceProp.major) {
                case 2: // Fermi
                    if (deviceProp.minor == 1) cores = mp * 48;
                    else cores = mp * 32;
                    break;
                case 3: // Kepler
                    cores = mp * 192;
                    break;
                case 5: // Maxwell
                    cores = mp * 128;
                    break;
                case 6: // Pascal
                    if ((deviceProp.minor == 1) || (deviceProp.minor == 2)) cores = mp * 128;
                    else if (deviceProp.minor == 0) cores = mp * 64;
                    else printf("Unknown device type\n");
                    break;
                case 7: // Volta and Turing
                    if ((deviceProp.minor == 0) || (deviceProp.minor == 5)) cores = mp * 64;
                    else printf("Unknown device type\n");
                    break;
                case 8: // Ampere
                    if (deviceProp.minor == 0) cores = mp * 64;
                    else if (deviceProp.minor == 6) cores = mp * 128;
                    else printf("Unknown device type\n");
                    break;
                default:
                    printf("Unknown device type\n");
                    break;
            }
            printf("Using GPU device number %d, with %d CUDA cores.\n", devID, cores);
        }
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    cils<scalar, index, is_read, n>::cils_babai_search_cuda(const index nswp, vector<index> *z_B) {
        scalar *y_A_d, *R_A_d;
        index *z_B_d, *z_B_p;

        auto z_B_h = z_B->data();

        cudaMallocManaged(&z_B_d, n * sizeof(index));
        cudaMallocManaged(&z_B_p, n * sizeof(index));
        cudaMallocManaged(&y_A_d, n * sizeof(scalar));
        cudaMallocManaged(&R_A_d, R_A->size * sizeof(scalar));

        index end_1 = n - 1;
        z_B_h[end_1] = round(y_A->x[end_1] / R_A->x[(n * end_1) + end_1 - ((end_1 * (end_1 + 1)) / 2)]);

        cudaMemcpy(y_A_d, y_A->x, n * sizeof(scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(R_A_d, R_A->x, R_A->size * sizeof(scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(z_B_d, z_B_h, n * sizeof(scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(z_B_p, z_B_h, n * sizeof(scalar), cudaMemcpyHostToDevice);

        index tileSize = 4;
        // Optimized kernel
        index nTiles = n / tileSize + (n % tileSize == 0 ? 0 : 1);
        index gridHeight = n / tileSize + (n % tileSize == 0 ? 0 : 1);
        index gridWidth = n / tileSize + (n % tileSize == 0 ? 0 : 1);
        dim3 dGrid(gridHeight, gridWidth), dBlock(tileSize, tileSize);

        std::clock_t start = std::clock();
        for (index k = 0; k < 10; k++) {
//            if (k % 2 == 0)
                cuda::babai_solve_cuda<scalar, index, n><<<512, 8>>>(R_A_d, y_A_d, z_B_d, z_B_p);
//            else
//                cuda::babai_solve_cuda<scalar, index, n><<<4, 1024>>>(R_A_d, y_A_d, z_B_p, z_B_d);
        }

        cudaDeviceSynchronize();
        scalar run_time = (std::clock() - start) / (scalar) CLOCKS_PER_SEC;
        cudaMemcpy(z_B_h, z_B_d, n * sizeof(index), cudaMemcpyDeviceToHost);

        cudaFree(z_B_d);
        cudaFree(z_B_p);
        cudaFree(y_A_d);
        cudaFree(R_A_d);

        returnType<scalar, index> reT = {z_B, run_time, 0};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    cils<scalar, index, is_read, n>::cils_block_search_cuda(const index nswp, const scalar stop,
                                                            const vector<index> *d, vector<index> *z_B) {

        index ds = d->size(), dx = d->at(ds - 1);
        if (ds == 1 || ds == n) {
            return cils_block_search_serial(d, z_B);
        }

        scalar *y_A_d, *R_A_d;
        index *z_B_d, *d_A_d;

        auto z_B_h = z_B->data();
        auto d_A_h = d->data();

        cudaMallocManaged(&z_B_d, n * sizeof(index));
        cudaMallocManaged(&y_A_d, n * sizeof(scalar));
        cudaMallocManaged(&R_A_d, R_A->size * sizeof(scalar));
        cudaMallocManaged(&d_A_d, ds * sizeof(index));

        cudaMemcpy(y_A_d, y_A->x, n * sizeof(scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(R_A_d, R_A->x, R_A->size * sizeof(scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(z_B_d, z_B_h, n * sizeof(index), cudaMemcpyHostToDevice);
        cudaMemcpy(d_A_d, d_A_h, ds * sizeof(index), cudaMemcpyHostToDevice);

        dim3 dGrid(256), dBlock(1);

        std::clock_t start = std::clock();
        for (index k = 0; k < 10; k++) {
            cuda::block_solve_cuda<scalar, index, n><<<dGrid, dBlock>>>(R_A_d, y_A_d, d_A_d, ds, z_B_d);
        }

        cudaDeviceSynchronize();
        scalar run_time = (std::clock() - start) / (scalar) CLOCKS_PER_SEC;
        cudaMemcpy(z_B_h, z_B_d, n * sizeof(index), cudaMemcpyDeviceToHost);
        cudaFree(z_B_d);
        cudaFree(y_A_d);
        cudaFree(R_A_d);
        cudaFree(d_A_d);

        returnType<scalar, index> reT = {z_B, run_time, 0};
        return reT;
    }
}
