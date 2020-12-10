#include "cils_cuda_solvers.cu"
#include <ctime>

using namespace std;
const int n = 4096;

int main() {

    cils::testDevice<double, int, n>(0);
    int n_jobs = 50, size = 16;
    int k = 1, index = 0, stop = 0, mode = 1, max_num_iter = 10, is_qr = 1, SNR = 35;
    cils::cils<double, int, true, n> cils(k, SNR);
    cils.init(is_qr);

    vector<int> z_B(n, 0);
    vector<int> d(n / size, size), d_s(n / size, size);
    for (int i = d_s.size() - 2; i >= 0; i--) {
        d_s[i] += d_s[i + 1];
    }

    z_B.assign(n, 0);
    auto reT = cils.cils_block_search_serial(&z_B, &d_s);
    auto res = cils::find_residual<double, int, n>(cils.R_A, cils.y_A, &reT.x);
    auto brr = cils::find_bit_error_rate<double, int, n>(&reT.x, &cils.x_t, false);
    printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Run time: %.5fs\n", size, res, brr,
           reT.run_time);

    z_B.assign(n, 0);
    reT = cils.cils_babai_search_omp(9, 10, &z_B);
    res = cils::find_residual<double, int, n>(cils.R_A, cils.y_A, &reT.x);
    brr = cils::find_bit_error_rate<double, int, n>(&reT.x, &cils.x_t, false);
    printf("Method: BAB_OMP, Block size: %d, Res: %.5f, BER: %.5f, Run time: %.5fs\n", size, res, brr,
           reT.run_time);

    z_B.assign(n, 0);
    reT = cils.cils_babai_search_serial(&z_B);
    res = cils::find_residual<double, int, n>(cils.R_A, cils.y_A, &reT.x);
    brr = cils::find_bit_error_rate<double, int, n>(&reT.x, &cils.x_t, false);
    printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Run time: %.5fs\n", res, brr, reT.run_time);

    for (int nswp = 1; nswp <= 1; nswp += 10) {
        z_B.assign(n, 0);
        reT = cils.cils_block_search_cuda(nswp, -1, &z_B, &d_s);
        res = cils::find_residual<double, int, n>(cils.R_A, cils.y_A, &reT.x);
        brr = cils::find_bit_error_rate<double, int, n>(&reT.x, &cils.x_t, false);
        printf("Method: ILS_GPU, Res: %.5f, BER: %.5f, Run time: %.5fs\n", res, brr, reT.run_time);
    }
//    for (int nswp = 0; nswp <= 100; nswp += 10) {
//        z_B.assign(n, 0);
//        reT = cils.cils_babai_search_cuda(nswp, &z_B);
//        res = cils::find_residual<double, int, n>(cils.R_A, cils.y_A, &reT.x);
//        brr = cils::find_bit_error_rate<double, int, n>(&reT.x, &cils.x_t, false);
//        printf("Method: BAB_GPU, Res: %.5f, BER: %.5f, Run time: %.5fs\n", res, brr, reT.run_time);
//    }

    return 0;

}
