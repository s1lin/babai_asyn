//
// Created by shilei on 2020-12-09.
//

#ifndef SILS_CUDA_CILS_CUDA_SOLVERS_CUH
#define SILS_CUDA_CILS_CUDA_SOLVERS_CUH
using namespace std;
namespace cils{
    template<typename scalar, typename index, index n>
    __global__ void
    babai_solve_cuda(scalar *z_B_c, scalar *z_B_p, const scalar *y_A_c, const scalar *R_A_c) {
        index idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            scalar sum = 0.0;
            for (index col = idx + 1; col < n; col++)
                sum += R_A_c[idx * n + col] * z_B_c[col];
            z_B_c[idx] = round((y_A_c[idx] - sum) / R_A_c[idx * n + idx]);
        }
    }
}
#endif //SILS_CUDA_CILS_CUDA_SOLVERS_CUH
