//
////Created by Shilei Lin on 2020-11-09.
//
//#include "../source/cils.cpp"
//
//template <typename scalar, typename index, index n>
//void test_run(int init_value) {
//
//    std::cout << "Init, value: " << init_value << std::endl;
//    std::cout << "Init, size: " << n << std::endl;
//
//    //bool read_r, bool read_ra, bool read_xy
//    double start = omp_get_wtime();
//    cils::cils<double, int, true, n> bsa(0.1);
//    double end_time = omp_get_wtime() - start;
//    printf("Finish Init, time: %f seconds\n", end_time);
//
//    cils::scalarType<double, int> z_BS = {(double *) calloc(n, sizeof(double)), n};
//    for (int i = 0; i < n; i++) {
//        if (init_value != -1) {
//            z_BS.x[i] = init_value;
//        } else {
//            z_BS.x[i] = bsa.x_R.x[i];
//        }
//    }
//
//    start = omp_get_wtime();
//    z_BS = *bsa.cils_babai_search_serial(&z_BS);
//    end_time = omp_get_wtime() - start;
//    auto res = cils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_BS);
//    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);
//
//    for (int proc = 80; proc >= 2; proc /= 2) {
//        cils::scalarType<double, int> z_B = {(double *) calloc(n, sizeof(double)), n};
//        cils::scalarType<double, int> z_B_p = {(double *) calloc(n, sizeof(double)), n};
//        auto *update = (int *) malloc(n * sizeof(int));
//
//        for (int i = 0; i < n; i++) {
//            if (init_value != -1) {
//                z_B.x[i] = init_value;
//                z_B_p.x[i] = init_value;
//
//            } else {
//                z_B.x[i] = bsa.x_R.x[i];
//                z_B_p.x[i] = bsa.x_R.x[i];
//            }
//            update[i] = 0;
//        }
//
//        start = omp_get_wtime();
//        z_B = *bsa.cils_babai_search_omp(proc, 10, update, &z_B, &z_B_p);
//        end_time = omp_get_wtime() - start;
//        res = cils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_B);
//        printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", proc, 0, res, end_time);
//        free(z_B.x);
//        free(z_B_p.x);
//        free(update);
//    }
//
//}
//
//template <typename scalar, typename index, index n>
//void plot_run() {
//
//    std::cout << "Init, size: " << n << std::endl;
//
//    //bool read_r, bool read_ra, bool read_xy
//    double start = omp_get_wtime();
//    cils::cils<double, int, true, false, n> bsa(0.1);
//    double end_time = omp_get_wtime() - start;
//    printf("Finish Init, time: %f seconds\n", end_time);
//
//    string fx = "res_" + to_string(n) + ".csv";
//    ofstream file(fx);
//    if (file.is_open()) {
//        for (int init_value = -1; init_value <= 1; init_value++) {
//            std::cout << "Init, value: " << init_value << std::endl;
//            vector<double> res(50, 0), tim(50, 0), itr(50, 0);
//            double omp_res = 0, omp_time = 0, num_iter = 0;
//            double ser_res = 0, ser_time = 0;
//
//            std::cout << "Vector Serial:" << std::endl;
//            for (int i = 0; i < 10; i++) {
//                cils::scalarType<double, int> z_BS{};
//                z_BS.x = (double *) calloc(n, sizeof(double));
//                z_BS.size = n;
//                for (int l = 0; l < n; l++) {
//                    if (init_value != -1) {
//                        z_BS.x[l] = init_value;
//                    } else {
//                        z_BS.x[l] = bsa.x_R.x[l];
//                    }
//                }
////                cils::display_scalarType(z_BS);
//                start = omp_get_wtime();
//                z_BS = *bsa.cils_babai_search_serial(&z_BS);
//                ser_time = omp_get_wtime() - start;
//                ser_res = cils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_BS);
//                printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", ser_res, ser_time);
//                res[0] += ser_res;
//                tim[0] += ser_time;
//            }
//
//            file << init_value << "," << res[0] / 10 << "," << tim[0] / 10 << ",\n";
//
//            std::cout << "OpenMP" << std::endl;
//            int l = 0;
//            for (int i = 0; i < 10; i++) {
//                for (int n_proc = 80; n_proc >= 2; n_proc /= 2) {
//                    cils::scalarType<double, int> z_B = {(double *) calloc(n, sizeof(double)), n};
//                    cils::scalarType<double, int> z_B_p = {(double *) calloc(n, sizeof(double)), n};
//                    auto *update = (int *) malloc(n * sizeof(int));
//
//                    for (int m = 0; m < n; m++) {
//                        z_B.x[m] = init_value;
//                        z_B_p.x[m] = init_value;
//                        update[m] = init_value;
//                    }
//
//                    start = omp_get_wtime();
//                    z_B = *bsa.cils_babai_search_omp(n_proc, 10, update, &z_B, &z_B_p);
//                    omp_time = omp_get_wtime() - start;
//                    omp_res = cils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_B);
//                    printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", n_proc, 0, omp_res, omp_time);
//                    free(z_B.x);
//                    free(z_B_p.x);
//                    free(update);
//
//                    res[l] += omp_res;
//                    tim[l] += omp_time;
//                    itr[l] += 10;
//                    l++;
//                }
//                l = 0;
//            }
//            l = 0;
//            for (int n_proc = 80; n_proc >= 2; n_proc /= 2) {
//                file << init_value << "," << n_proc << ","
//                     << res[l] / 10 << ","
//                     << tim[l] / 10 << ","
//                     << itr[l] / 10 << ",\n";
//
//                printf("Init value: %d, n_proc: %d, res :%f, num_iter: %f, Average time: %fs\n", init_value, n_proc,
//                       res[l] / 10,
//                       itr[l] / 10,
//                       tim[l] / 10);
//                l++;
//            }
//            file << "Next,\n";
//        }
//    }
//    file.close();
//
//}