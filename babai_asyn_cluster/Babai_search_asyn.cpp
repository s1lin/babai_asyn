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
    int n = 4096;
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
//    int n_proc[4] = {16, 8, 4, 2};

    for(int proc =16; proc>=2; proc/=2) {
        auto *z_B = (double *) malloc(n * sizeof(double));
        auto *z_B_p = (double *) malloc(n * sizeof(double));
        auto *update = (int *) malloc(n * sizeof(int));

        for (int i = 0; i < n; i++) {
            z_B[i] = 0;
            z_B_p[i] = 0;
            update[i] = 0;
        }
        start = omp_get_wtime();
        z_B = search_omp(proc, 10, bsa.n, bsa.R_A, bsa.y_A, eigen, update, z_B, z_B_p);
        end_time = omp_get_wtime() - start;
        res = Babai_search_asyn::find_residual(bsa.n, bsa.R_A, bsa.y_A, z_B);
        printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", proc, 0, res, end_time);
        free(z_B);
        free(z_B_p);
        free(update);
    }

    return 0;
}