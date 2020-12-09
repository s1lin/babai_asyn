#include "../babai_asyn_c++/src/source/sils.cpp"
#include <ctime>

namespace sils {
    template<typename scalar, typename index, index n>
    __global__ void
    babai_solve_cuda(scalar *z_B_c, scalar *z_B_p, const scalar *y_A, const scalar *R_A_c) {

//    for (index j = 0; j < nswp; j++) {
//        scalar sum = 0;
//        for (index i = 1; i < n; i++) {
//            for (index col = n - i; col < n; col++) {
//                sum += R_A_c[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * raw_z_B_c[col];
//            }
//            raw_z_B_c[n - 1 - i] = round(
//                    (y_A[n - 1 - i] - sum) / R_A_c[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
//            sum = 0;
//        }
//    }
//tiling
        index idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            scalar sum = 0.0;
//
//        // store index in register
//        // Multiplication is not executed in every iteration.
//        //index idz_B_ci = idx * n;
            for (index col = idx + 1; col < n; col++)
                sum += R_A_c[idx * n + col] * z_B_c[col];
            z_B_c[idx] = round((y_A[idx] - sum) / R_A_c[idx * n + idx]);


//        index idz_B_ci = idx * n;
//        for (index j = 0; j < n; j++)
//            if (idx != j)
//                sum += R_A_c[idz_B_ci + j] * z_B_c[j];
//        z_B_p[idx] = round((y_A[idx] - sum) / R_A_c[idx * n + idx]);
        }
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    sils<scalar, index, is_read, n>::sils_babai_search_cuda(index nswp, vector<index> *z_B) {
        scalar *z_B_c, *z_B_p, *y_A_c, *R_A_c;

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
                R_A_c[i] = R_A->x[i];
            }
            y_A_c[row] = y_A->x[row];
        }

        for (index row = 0; row < 100; row++) {
            for (index col = 0; col < 100; col++) {
                cout<<R_A_c[row]<<" ";
            }
            cout<<endl;
        }

        z_B_c[end_1] = z_B_p[end_1] = z_B->at(end_1);

        cudaMemcpy(y_A_c, y_A->x, n * sizeof(scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(R_A_c, R_A->x, n * n * sizeof(scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(z_B_c, z_B->data(), n * sizeof(scalar), cudaMemcpyHostToDevice);
        cudaMemcpy(z_B_p, z_B->data(), n * sizeof(scalar), cudaMemcpyHostToDevice);

        index tileSize = 4;
        // Optimized kernel
        index nTiles = n / tileSize + (n % tileSize == 0 ? 0 : 1);
        index gridHeight = n / tileSize + (n % tileSize == 0 ? 0 : 1);
        index gridWidth = n / tileSize + (n % tileSize == 0 ? 0 : 1);
        //printf("w=%d, h=%d\n", gridWidth, gridHeight);
        dim3 dGrid(gridHeight, gridWidth), dBlock(tileSize, tileSize);

//        std::clock_t start = std::clock();
//        for (index k = 0; k < nswp; k++) {
//            //if (k % 2)
//            find_raw_x0_cuda<<<nTiles, tileSize>>>(n, z_B_c, z_B_p, y_A_c, R_A_c);
//            //else
//            //find_raw_x0_cuda<<<nTiles, tileSize>>>(n, z_B_p, z_B_c, y_A_c, R_A_c);
//        }

//        cudaDeviceSynchronize();
//        scalar time = (std::clock() - start) / (scalar) CLOCKS_PER_SEC;


//        scalar res = find_residual<scalar, index, n>(R_A, y_A, &reT.x)
//        printf("Sweep: %d, Res: %.5f, Run time: %fs\n", nswp, res, time);

        cudaFree(z_B_c);
        cudaFree(z_B_p);
        cudaFree(y_A_c);
        cudaFree(R_A_c);
        return 0;
    }

}
