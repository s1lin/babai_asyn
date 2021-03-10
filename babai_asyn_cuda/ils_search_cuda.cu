#include "cils_cuda_solvers.cu"
#include <ctime>

using namespace std;
const int n = 4096;

int main(int argc, char *argv[]) {

    init_program_def(argc, argv);

    cils::testDevice<double, int, n>(0);
    cils::cils<double, int, true, n> cils(k, SNR);
    cils.init(is_qr, 0);
    vector<int> z_B(n, 0);
    int init = 0;

    init_guess(init, &z_B, &cils.x_R);
    auto reT = cils.cils_block_search_serial(&d_s, &z_B);
    auto res = cils::find_residual<double, int, n>(cils.R_A, cils.y_A, reT.x);
    auto ber = cils::find_bit_error_rate<double, int, n>(reT.x, &cils.x_t, cils.qam == 1);
    printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Time: %.5fs\n",
           block_size, res, ber, reT.run_time);
    double ils_tim = reT.run_time;


    init_guess(init, &z_B, &cils.x_R);
    reT = cils.cils_babai_search_omp(9, 10, &z_B);
    res = cils::find_residual<double, int, n>(cils.R_A, cils.y_A, reT.x);
    ber = cils::find_bit_error_rate<double, int, n>(reT.x, &cils.x_t, false);
    printf("Method: BAB_OMP, Block size: %d, Res: %.5f, BER: %.5f, Run time: %.5fs\n", block_size, res, ber,
           reT.run_time);

    init_guess(init, &z_B, &cils.x_R);
    reT = cils.cils_babai_search_serial(&z_B);
    res = cils::find_residual<double, int, n>(cils.R_A, cils.y_A, reT.x);
    ber = cils::find_bit_error_rate<double, int, n>(reT.x, &cils.x_t, false);
    printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Run time: %.5fs\n", res, ber, reT.run_time);


    for (int nswp = 1; nswp <= 10; nswp += 10) {
        init_guess(init, &z_B, &cils.x_R);
        reT = cils.cils_babai_search_cuda(10, &z_B);
        res = cils::find_residual<double, int, n>(cils.R_A, cils.y_A, reT.x);
        ber = cils::find_bit_error_rate<double, int, n>(reT.x, &cils.x_t, false);
        printf("Method: BAB_GPU, Res: %.5f, BER: %.5f, Run time: %.5fs\n", res, ber, reT.run_time);

        init_guess(init, &z_B, &cils.x_R);
        reT = cils.cils_block_search_cuda(nswp, -1, &d_s, &z_B);
        res = cils::find_residual<double, int, n>(cils.R_A, cils.y_A, reT.x);
        ber = cils::find_bit_error_rate<double, int, n>(reT.x, &cils.x_t, false);
        printf("Method: ILS_GPU, Res: %.5f, BER: %.5f, Run time: %.5fs\n", res, ber, reT.run_time);
    }

    return 0;

}
