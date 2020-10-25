#include "Babai_search_asyn.h"

inline double do_solve(const int n, const int i, const double *R_A, const double *y_A, const double *z_B) {
    double sum = 0;
#pragma omp simd reduction(+ : sum)
    for (int col = n - i; col < n; col++) {
        sum += R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * z_B[col];
    }

    return round((y_A[n - 1 - i] - sum) / R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
}

double *search_omp(const int n_proc, const int nswp, const int n, const double *R_A, const double *y_A,
                   const bool eigen, int *update, double *z_B, double *z_B_p) {

    int chunk = std::log2(n);
    z_B[n - 1] = round(y_A[n - 1] / R_A[((n - 1) * n) / 2 + n - 1]);
#pragma omp parallel default(shared) num_threads(n_proc) shared(update)
    {
        for (int j = 0; j < nswp; j++) {
#pragma omp for schedule(dynamic, chunk) nowait
            for (int i = 0; i < n; i++) {
                z_B[n - 1 - i] = do_solve(n, i, R_A, y_A, z_B);
            }
        }
    }
    return z_B;
}

int main() {
    cout << omp_get_max_threads() << endl;
    int n = 32768;
    bool eigen = false;
    std::cout << "Init, size: " << n << std::endl;
    Babai_search_asyn bsa(n, eigen);

    double start = omp_get_wtime();
    bsa.init(false, true, true, 0.1);
    double end_time = omp_get_wtime() - start;
    std::cout << "Finish Init, time: " << end_time << std::endl;


    std::cout << "Vector Serial:" << std::endl;
    start = omp_get_wtime();
    vector<double> z_BV = bsa.search_vec();
    end_time = omp_get_wtime() - start;
    double res = Babai_search_asyn::find_residual(bsa.n, bsa.R_A, bsa.y_A, z_BV.data());
    printf("Res = %.5f, init_res = %.5f %f seconds\n", res, bsa.init_res, end_time);

    std::cout << "OPENMP:" << std::endl;
    auto *z_B = (double *) malloc(n * sizeof(double));
    auto *z_B_p = (double *) malloc(n * sizeof(double));
    auto *update = (int *) malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        z_B[i] = 0;
        z_B_p[i] = 0;
        update[i] = 0;
    }

    start = omp_get_wtime();
    z_B = search_omp(80, 10, bsa.n, bsa.R_A, bsa.y_A, eigen, update, z_B, z_B_p);
    end_time = omp_get_wtime() - start;
    res = Babai_search_asyn::find_residual(bsa.n, bsa.R_A, bsa.y_A, z_B);
    printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", 12, 0, res, end_time);
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
    z_B = search_omp(40, 10, bsa.n, bsa.R_A, bsa.y_A, eigen, update2, z_B2, z_B_p2);
    end_time = omp_get_wtime() - start;
    res = Babai_search_asyn::find_residual(bsa.n, bsa.R_A, bsa.y_A, z_B);
    printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", 6, 0, res, end_time);
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
    z_B = search_omp(20, 10, bsa.n, bsa.R_A, bsa.y_A, eigen, update3, z_B3, z_B_p3);
    end_time = omp_get_wtime() - start;
    res = Babai_search_asyn::find_residual(bsa.n, bsa.R_A, bsa.y_A, z_B);
    printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", 3, 0, res, end_time);
    free(z_B3);
    free(z_B_p3);
    free(update3);


    auto *z_B4 = (double *) malloc(n * sizeof(double));
    auto *z_B_p4 = (double *) malloc(n * sizeof(double));
    auto *update4 = (int *) malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        z_B4[i] = 0;
        z_B_p4[i] = 0;
        update4[i] = 0;
    }

    start = omp_get_wtime();
    z_B = search_omp(10, 10, bsa.n, bsa.R_A, bsa.y_A, eigen, update4, z_B4, z_B_p4);
    end_time = omp_get_wtime() - start;
    res = Babai_search_asyn::find_residual(bsa.n, bsa.R_A, bsa.y_A, z_B);
    printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", 3, 0, res, end_time);
    free(z_B4);
    free(z_B_p4);
    free(update4);


    return 0;
}