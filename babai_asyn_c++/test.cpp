//
// Created by shilei on 2020-11-04.
//
#include "SILS.cpp"

using namespace std;

const int n = 10;

void test_ils_block_search() {
    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    sils::SILS<double, int, true, false, n> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    auto *z_B = (double *) calloc(n, sizeof(double));
    vector<int> d(2, 5);
    start = omp_get_wtime();

    z_B = bsa.sils_block_search_serial(bsa.R_A, bsa.y_A, z_B, d, n);

    end_time = omp_get_wtime() - start;
    double res = sils::find_residual(n, bsa.R_A, bsa.y_A, z_B);
    printf("Thread: ILS, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

    auto *z_BS = (double *) calloc(n, sizeof(double));
    start = omp_get_wtime();
    z_BS = bsa.sils_babai_search_serial(z_BS);
    end_time = omp_get_wtime() - start;
    res = sils::find_residual(n, bsa.R_A, bsa.y_A, z_BS);
    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);
}

void test_ils_search() {
    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    sils::SILS<double, int, true, false, n> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    auto *z_B = (double *) calloc(n, sizeof(double));
    start = omp_get_wtime();
    z_B = bsa.sils_search(bsa.R_A, bsa.y_A, z_B, n, bsa.size_R_A);
    end_time = omp_get_wtime() - start;
    double res = sils::find_residual(n, bsa.R_A, bsa.y_A, z_B);
    printf("Thread: ILS, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

    auto *z_BS = (double *) calloc(n, sizeof(double));
    start = omp_get_wtime();
    z_BS = bsa.sils_babai_search_serial(z_BS);
    end_time = omp_get_wtime() - start;
    res = sils::find_residual(n, bsa.R_A, bsa.y_A, z_BS);
    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

}

void test_run(int init_value) {

    std::cout << "Init, value: " << init_value << std::endl;
    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    sils::SILS<double, int, true, false, n> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    auto *z_BS = (double *) malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        if (init_value != -1) {
            z_BS[i] = init_value;
        } else {
            z_BS[i] = bsa.x_R[i];
        }
    }

    start = omp_get_wtime();
    z_BS = bsa.sils_babai_search_serial(z_BS);
    end_time = omp_get_wtime() - start;
    double res = sils::find_residual(n, bsa.R_A, bsa.y_A, z_BS);
    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

    for (int proc = 80; proc >= 2; proc /= 2) {
        auto *z_B = (double *) malloc(n * sizeof(double));
        auto *z_B_p = (double *) malloc(n * sizeof(double));
        auto *update = (int *) malloc(n * sizeof(int));

        for (int i = 0; i < n; i++) {
            if (init_value != -1) {
                z_B[i] = init_value;
                z_B_p[i] = init_value;

            } else {
                z_B[i] = bsa.x_R[i];
                z_B_p[i] = bsa.x_R[i];
            }
            update[i] = 0;
        }

        start = omp_get_wtime();
        z_B = bsa.sils_babai_search_omp(proc, 10, update, z_B, z_B_p);
        end_time = omp_get_wtime() - start;
        res = sils::find_residual(n, bsa.R_A, bsa.y_A, z_B);
        printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", proc, 0, res, end_time);
        free(z_B);
        free(z_B_p);
        free(update);
    }

}

void plot_run() {

    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    sils::SILS<double, int, true, false, 4096> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    string fx = "res_" + to_string(n) + ".csv";
    ofstream file(fx);
    if (file.is_open()) {
        for (int init_value = -1; init_value <= 1; init_value++) {
            std::cout << "Init, value: " << init_value << std::endl;
            vector<double> res(50, 0), tim(50, 0), itr(50, 0);
            double omp_res = 0, omp_time = 0, num_iter = 0;
            double ser_res = 0, ser_time = 0;

            std::cout << "Vector Serial:" << std::endl;
            for (int i = 0; i < 10; i++) {
                auto *z_BS = (double *) malloc(n * sizeof(double));
                for (int l = 0; l < n; l++) {
                    if (init_value != -1) {
                        z_BS[l] = init_value;
                    } else {
                        z_BS[l] = bsa.x_R[l];
                    }
                }

                start = omp_get_wtime();
                z_BS = bsa.sils_babai_search_serial(z_BS);
                ser_time = omp_get_wtime() - start;
                ser_res = sils::find_residual(n, bsa.R_A, bsa.y_A, z_BS);
                printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", ser_res, ser_time);
                res[0] += ser_res;
                tim[0] += ser_time;
            }

            file << init_value << "," << res[0] / 10 << "," << tim[0] / 10 << ",\n";

            std::cout << "OpenMP" << std::endl;
            int index = 0;
            for (int i = 0; i < 10; i++) {
                for (int n_proc = 80; n_proc >= 2; n_proc /= 2) {
                    auto *z_B = (double *) malloc(n * sizeof(double));
                    auto *z_B_p = (double *) malloc(n * sizeof(double));
                    auto *update = (int *) malloc(n * sizeof(int));

                    for (int m = 0; m < n; m++) {
                        z_B[m] = init_value;
                        z_B_p[m] = init_value;
                        update[m] = init_value;
                    }

                    start = omp_get_wtime();
                    z_B = bsa.sils_babai_search_omp(n_proc, 10, update, z_B, z_B_p);
                    omp_time = omp_get_wtime() - start;
                    omp_res = sils::find_residual(n, bsa.R_A, bsa.y_A, z_B);
                    printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", n_proc, 0, omp_res, omp_time);
                    free(z_B);
                    free(z_B_p);
                    free(update);

                    res[index] += omp_res;
                    tim[index] += omp_time;
                    itr[index] += 10;
                    index++;
                }
                index = 0;
            }
            index = 0;
            for (int n_proc = 80; n_proc >= 2; n_proc /= 2) {
                file << init_value << "," << n_proc << ","
                     << res[index] / 10 << ","
                     << tim[index] / 10 << ","
                     << itr[index] / 10 << ",\n";

                printf("Init value: %d, n_proc: %d, res :%f, num_iter: %f, Average time: %fs\n", init_value, n_proc,
                       res[index] / 10,
                       itr[index] / 10,
                       tim[index] / 10);
                index++;
            }
            file << "Next,\n";
        }
    }
    file.close();

}

int main() {
    std::cout << "Maximum Threads: " << omp_get_max_threads() << std::endl;
    //plot_run();
    test_ils_block_search();
//    test_ils_search();

    return 0;
}

