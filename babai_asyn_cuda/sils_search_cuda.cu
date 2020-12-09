#include "ils_cuda_solvers.cu"
#include <ctime>

int main() {

    testDevice(0);
    const int n = 4096;
    int n_jobs = 50, size = 16;
    int k = 1, index = 0, stop = 0, mode = 1, max_num_iter = 10, is_qr = 1, SNR = 15;
    sils::sils<double, int, true, n> sils(k, SNR);
    sils.init(is_qr);

    vector<int> z_B(n, 0);
    vector<int> d(n / size, size), d_s(n / size, size);
    for (int i = d_s.size() - 2; i >= 0; i--) {
        d_s[i] += d_s[i + 1];
    }

    z_B.assign(n, 0);
    auto reT = sils.sils_block_search_serial(&z_B, &d_s);
    auto res = sils::find_residual<double, int, n>(sils.R_A, sils.y_A, &reT.x);
    auto brr = sils::find_bit_error_rate<double, int, n>(&reT.x, &sils.x_t, false);
    printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Run time: %.5fs\n", size, res, brr,
           reT.run_time);

    z_B.assign(n, 0);
    reT = sils.sils_babai_search_omp(9, 10, &z_B);
    res = sils::find_residual<double, int, n>(sils.R_A, sils.y_A, &reT.x);
    printf("Method: BAB_OMP, Block size: %d, Res: %.5f, Run time: %.5fs\n", 1, res, reT.run_time);

    z_B.assign(n, 0);
    reT = sils.sils_babai_search_serial(&z_B);
    res = sils::find_residual<double, int, n>(sils.R_A, sils.y_A, &reT.x);
    brr = sils::find_bit_error_rate<double, int, n>(&reT.x, &sils.x_t, false);
    printf("Method: BBI_SER, Res: %.5f, BER: %.5f, Run time: %.5fs\n", res, brr, reT.run_time);

    std::cout << "find_raw_x0_CUDA" << std::endl;
    for (int nswp = 10; nswp <= 200; nswp += 10) {
        double time = run(n, nswp, sils);
        cout << time << ",";
    }


    return 0;

}
