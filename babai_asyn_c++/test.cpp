//
// Created by shilei on 2020-11-04.
//
#include "SILS.cpp"

using namespace std;

const int n = 8192;

void test_ils_block_search() {
    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    sils::SILS<double, int, true, false, n> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    sils::scalarType<double, int> z_B{(double *) calloc(n, sizeof(double)), n};
    vector<int> d(1024, 8); //256*16=4096
//    vector<int> d(5, 2); //10
//    vector<int> d(5, 6); //30
    sils::scalarType<int, int> d_s{d.data(), (int) d.size()};
    sils::scalarType<double, int> z_BV{(double *) calloc(n, sizeof(double)), n};
    for (int i = d_s.size - 2; i >= 0; i--) {
        d_s.x[i] += d_s.x[i + 1];
    }

    start = omp_get_wtime();
    auto z_B_s = bsa.sils_block_search_serial(&bsa.R_A, &bsa.y_A, &z_B, &d_s);
    end_time = omp_get_wtime() - start;
    auto res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y, z_B_s);
    printf("Thread: ILS_SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

    start = omp_get_wtime();
    z_BV = *bsa.sils_block_search_omp(10, 10, &bsa.R_A, &bsa.y_A, &z_BV, &d_s);
    end_time = omp_get_wtime() - start;
    res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_BV);
    printf("Thread: ILS_OP, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

    sils::scalarType<double, int> z_BS = {(double *) calloc(n, sizeof(double)), n};
    start = omp_get_wtime();
    z_B_s = bsa.sils_babai_search_serial(&z_BS);
    end_time = omp_get_wtime() - start;
    res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, z_B_s);
    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);


//    free(z_B_s);
    free(z_BV.x);
//    free(z_B_o);
//    free(d_s.x);

}

void test_ils_search() {
    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    sils::SILS<double, int, true, false, n> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    start = omp_get_wtime();
    auto z_B = bsa.sils_search(&bsa.R_A, &bsa.y_A);
    end_time = omp_get_wtime() - start;
    auto res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, z_B);
    printf("Thread: ILS, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

    sils::scalarType<double, int> z_BS = {(double *) calloc(n, sizeof(double)), n};
    start = omp_get_wtime();
    z_BS = *bsa.sils_babai_search_serial(&z_BS);
    end_time = omp_get_wtime() - start;
    res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_BS);
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

    sils::scalarType<double, int> z_BS = {(double *) calloc(n, sizeof(double)), n};
    for (int i = 0; i < n; i++) {
        if (init_value != -1) {
            z_BS.x[i] = init_value;
        } else {
            z_BS.x[i] = bsa.x_R.x[i];
        }
    }

    start = omp_get_wtime();
    z_BS = *bsa.sils_babai_search_serial(&z_BS);
    end_time = omp_get_wtime() - start;
    auto res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_BS);
    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

    for (int proc = 80; proc >= 2; proc /= 2) {
        sils::scalarType<double, int> z_B = {(double *) calloc(n, sizeof(double)), n};
        sils::scalarType<double, int> z_B_p = {(double *) calloc(n, sizeof(double)), n};
        auto *update = (int *) malloc(n * sizeof(int));

        for (int i = 0; i < n; i++) {
            if (init_value != -1) {
                z_B.x[i] = init_value;
                z_B_p.x[i] = init_value;

            } else {
                z_B.x[i] = bsa.x_R.x[i];
                z_B_p.x[i] = bsa.x_R.x[i];
            }
            update[i] = 0;
        }

        start = omp_get_wtime();
        z_B = *bsa.sils_babai_search_omp(proc, 10, update, &z_B, &z_B_p);
        end_time = omp_get_wtime() - start;
        res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_B);
        printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", proc, 0, res, end_time);
        free(z_B.x);
        free(z_B_p.x);
        free(update);
    }

}

void plot_run() {

    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    sils::SILS<double, int, true, false, n> bsa(0.1);
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
                sils::scalarType<double, int> z_BS{};
                z_BS.x = (double *) calloc(n, sizeof(double));
                z_BS.size = n;
                for (int l = 0; l < n; l++) {
                    if (init_value != -1) {
                        z_BS.x[l] = init_value;
                    } else {
                        z_BS.x[l] = bsa.x_R.x[l];
                    }
                }
//                sils::display_scalarType(z_BS);
                start = omp_get_wtime();
                z_BS = *bsa.sils_babai_search_serial(&z_BS);
                ser_time = omp_get_wtime() - start;
                ser_res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_BS);
                printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", ser_res, ser_time);
                res[0] += ser_res;
                tim[0] += ser_time;
            }

            file << init_value << "," << res[0] / 10 << "," << tim[0] / 10 << ",\n";

            std::cout << "OpenMP" << std::endl;
            int index = 0;
            for (int i = 0; i < 10; i++) {
                for (int n_proc = 80; n_proc >= 2; n_proc /= 2) {
                    sils::scalarType<double, int> z_B = {(double *) calloc(n, sizeof(double)), n};
                    sils::scalarType<double, int> z_B_p = {(double *) calloc(n, sizeof(double)), n};
                    auto *update = (int *) malloc(n * sizeof(int));

                    for (int m = 0; m < n; m++) {
                        z_B.x[m] = init_value;
                        z_B_p.x[m] = init_value;
                        update[m] = init_value;
                    }

                    start = omp_get_wtime();
                    z_B = *bsa.sils_babai_search_omp(n_proc, 10, update, &z_B, &z_B_p);
                    omp_time = omp_get_wtime() - start;
                    omp_res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_B);
                    printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", n_proc, 0, omp_res, omp_time);
                    free(z_B.x);
                    free(z_B_p.x);
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
    //test_ils_search();

    return 0;
}

