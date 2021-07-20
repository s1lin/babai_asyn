#include "src/example/cils_gen_test.cpp"

using namespace std;
using namespace cils;

void run_test(int argc, char *argv[]) {

    program_def::init_program_def(argc, argv);

    switch (program_def::mode) {
        case 0:
//            plot_LLL<scalar, index, M,  N>();
            plot_res<scalar, int, M, N>();
            break;
        case 1:
//            plot_LLL<scalar, index, M,  N>();
            plot_run<scalar, int, M, N>();
            break;
        case 2:
//            ils_block_search<scalar, int, N_4096>();
            break;
        default:
            test_ils_search<scalar, int, M, N>();
            break;
    }
}

int main(int argc, char *argv[]) {

    printf("====================[ Run | cils | Release ]==================================\n");

    double t = omp_get_wtime();
    run_test(argc, argv);
    t = omp_get_wtime() - t;

    printf("====================[TOTAL TIME | %2.2fs, %2.2fm, %2.2fh]==================================\n",
           t, t / 60, t / 3600);

    return 0;
}