//
// Created by shilei on 2020-12-09.
//

#ifndef CILS_CUDA_CILS_CUDA_SOLVERS_CUH
#define CILS_CUDA_CILS_CUDA_SOLVERS_CUH

using namespace std;
namespace cils {
    template<typename scalar, typename index, index n>
    __global__ void babai_solve_cuda(const scalar *R_A_c, const scalar *y_A_c, scalar *z_B_c) {
        index idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            scalar sum = 0.0;
            for (index col = idx + 1; col < n; col++) {
                sum += R_A_c[(n * idx) + col - ((idx * (idx + 1)) / 2)] * z_B_c[col];
            }
            z_B_c[idx] = round((y_A_c[idx] - sum) / R_A_c[(n * idx) + idx - ((idx * (idx + 1)) / 2)]);
        }
    }

    template<typename scalar, typename index, index n>
    __device__ index *ils_search_cuda(const index n_dx_q_0, const index n_dx_q_1,
                                      const scalar *R_A_c, const scalar *y) {

        //variables
        scalar sum, newprsd, gamma, beta = INFINITY;

        index dx = n_dx_q_1 - n_dx_q_0, k = dx - 1, iter = 0;
        index end_1 = n_dx_q_1 - 1, row_k = k + n_dx_q_0;

        auto *p = (scalar *) malloc(dx * sizeof(scalar));
        auto *c = (scalar *) malloc(dx * sizeof(scalar));
        auto *z = (index *) malloc(dx * sizeof(index));
        auto *x = (index *) malloc(dx * sizeof(index));
        auto *d = (index *) malloc(dx * sizeof(index));

        //  Initial squared search radius
        scalar R_kk = R_A_c[(n * end_1) + end_1 - ((end_1 * (end_1 + 1)) / 2)];
        c[k] = y->at(k) / R_kk;
        z[k] = round(c[k]);
        gamma = R_kk * (c[k] - z[k]);

        //  Determine enumeration direction at level block_size
        d[dx - 1] = c[dx - 1] > z[dx - 1] ? 1 : -1;

        //ILS search process
        while (true) {
//                iter++;
            newprsd = p[k] + gamma * gamma;
            if (newprsd < beta) {
                if (k != 0) {
                    k--;
                    row_k--;
                    sum = 0;

                    for (index col = k + 1; col < dx; col++) {
                        sum += R_A_c[(n * row_k) + col + n_dx_q_0 - ((row_k * (row_k + 1)) / 2)] * z[col];
                    }
                    R_kk = R_A_c[(n * row_k) + row_k - ((row_k * (row_k + 1)) / 2)];

                    p[k] = newprsd;
                    c[k] = (y->at(k) - sum) / R_kk;
                    z[k] = round(c[k]);
                    gamma = R_kk * (c[k] - z[k]);

                    d[k] = c[k] > z[k] ? 1 : -1;

                } else {
                    beta = newprsd;
                    for (index l = 0; l < dx; l++) {
                        x[l] = z[l];
                    }
                    //x.assign(z.begin(), z.end());
                    z[0] += d[0];
                    gamma = R_A_c[0] * (c[0] - z[0]);
                    d[0] = d[0] > 0 ? -d[0] - 1 : -d[0] + 1;
                }
            } else {
                if (k == dx - 1) break;
                else {
                    k++;
                    row_k++;
                    z[k] += d[k];
                    gamma = R_A_c[(n * row_k) + row_k - ((row_k * (row_k + 1)) / 2)] * (c[k] - z[k]);
                    d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                }
            }
        }

        free(p);
        free(c);
        free(z);
        free(d);
        return x;
    }

    template<typename scalar, typename index, index n>
    __global__ void block_solve_cuda(const scalar *R_A_c, const scalar *y_A_c, const index dx, scalar *z_B_c) {
        index idx = blockIdx.x * blockDim.x + threadIdx.x;
//        vector<index> x(dx, 0);
//        if (idx < n) {
//            for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
//                sum = 0;
//
//                for (index col = n_dx_q_1; col < n; col++) {
//                    //Translating the index from R(matrix) to R_B(compressed array).
//                    sum += R_A_c[(n * row) + col - ((row * (row + 1)) / 2)] * z_B->at(col);
//                }
//                y[row - n_dx_q_0] = y_A->x[row] - sum;
//            }
//            x = ils_search_cuda(n_dx_q_0, n_dx_q_1, &y);
//        } else {
//            x = ils_search_cuda(n_dx_q_0, n_dx_q_1, &y_n);
//        }
//
//        for (index l = n_dx_q_0; l < n_dx_q_1; l++) {
//            z_B_c[l] = x[l - n_dx_q_0];
//        }
    }


}
#endif //CILS_CUDA_CILS_CUDA_SOLVERS_CUH
