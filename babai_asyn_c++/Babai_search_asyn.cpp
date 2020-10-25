#include "Babai_search_asyn.h"


int main() {
    cout << omp_get_max_threads() << endl;
    int n = 4096;
    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    babai::Babai_search_asyn<double, int, false, true, false> bsa(n, 0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    start = omp_get_wtime();
    vector<double> z_BV = bsa.search_vec();
    end_time = omp_get_wtime() - start;
    double res = babai::find_residual(bsa.n, bsa.R_A, bsa.y_A, z_BV.data());
    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

    int n_proc[3] = {12, 6, 3};

    auto *z_B = (double *) malloc(n * sizeof(double));
    auto *z_B_p = (double *) malloc(n * sizeof(double));
    auto *update = (int *) malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        z_B[i] = 0;
        z_B_p[i] = 0;
        update[i] = 0;
    }

    start = omp_get_wtime();
    z_B = bsa.search_omp(n_proc[0], 10, update, z_B, z_B_p);
    end_time = omp_get_wtime() - start;

    res = babai::find_residual(bsa.n, bsa.R_A, bsa.y_A, z_B);
    printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", n_proc[0], 0, res, end_time);
    free(z_B);
    free(z_B_p);
    free(update);

    auto *z_B2 = (double *) malloc(n * sizeof(double));
    auto *z_B_p2 = (double *) malloc(n * sizeof(double));
    auto *update2 = (int *) malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        z_B2[i] = 0;
        z_B_p2[i] = 0;
        update2[i] = 0;
    }

    start = omp_get_wtime();
    z_B = bsa.search_omp(n_proc[1], 10, update, z_B, z_B_p);
    end_time = omp_get_wtime() - start;
    res = babai::find_residual(bsa.n, bsa.R_A, bsa.y_A, z_B);
    printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", n_proc[1], 0, res, end_time);
    free(z_B2);
    free(z_B_p2);
    free(update2);

    auto *z_B3 = (double *) malloc(n * sizeof(double));
    auto *z_B_p3 = (double *) malloc(n * sizeof(double));
    auto *update3 = (int *) malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        z_B3[i] = 0;
        z_B_p3[i] = 0;
        update3[i] = 0;
    }

    start = omp_get_wtime();
    z_B = bsa.search_omp(n_proc[2], 10, update, z_B, z_B_p);
    end_time = omp_get_wtime() - start;
    res = babai::find_residual(bsa.n, bsa.R_A, bsa.y_A, z_B);
    printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", n_proc[2], 0, res, end_time);
    free(z_B3);
    free(z_B_p3);
    free(update3);

    return 0;
}
