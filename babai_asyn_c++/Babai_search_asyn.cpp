#include "Babai_search_asyn.h"
//#include "Babai_search_asyn_massive.h"
//#include "matplotlibcpp.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

//namespace plt = matplotlibcpp;
//void plot_run() {
//    for (int n = 4096; n <= 16384; n *= 2) {
//        std::cout << "Init, size: " << n << std::endl;
//        Babai_search_asyn bsa(n);
//        //Babai_search_asyn_massive bsa(n);
//        bsa.init(true, true, 0.1);
//        std::cout << "Finish Init" << std::endl;
//
//        string fx = "res_" + to_string(n) + ".csv";
//        ofstream file(fx);
//        if (file.is_open()) {
//            for (int init_value = -1; init_value <= 1; init_value++) {
//                vector<double> res(50, 0), tim(50, 0), itr(50, 0);
//                double omp_res = 0, omp_time = 0, num_iter = 0;
////                double eig_res = 0, eig_time = 0;
//                double ser_res = 0, ser_time = 0;
//
//                std::cout << "Vector Serial:" << std::endl;
//                for (int i = 0; i < 10; i++) {
////                    std::cout << "Eigen Serial:" << std::endl;
////                    auto[eig_res, eig_time] = bsa.search_eigen(init_value);
////                    res[0] += eig_res;
////                    tim[0] += eig_time;
//
//                    auto[ser_res, ser_time] = bsa.search_vec(init_value);
//                    res[0] += ser_res;
//                    tim[0] += ser_time;
//                }
//
//                file << init_value << "," << res[0] / 10 << "," << tim[0] / 10 << ",\n";
//                if (n == 4096) {
//                    if (init_value == -1)
//                        file << "-1,6.4299,0.1032,\n";
//                    else if (init_value == 0)
//                        file << "0,6.4299,0.0998,\n";
//                    else
//                        file << "1,6.4299,0.0998,\n";
//                } else if (n == 8192) {
//                    if (init_value == -1)
//                        file << "-1,9.06945,0.4765,\n";
//                    else if (init_value == 0)
//                        file << "0,9.06945,0.4788,\n";
//                    else
//                        file << "1,9.06945,0.4804,\n";
//                } else if (n == 16384) {
//                    if (init_value == -1)
//                        file << "-1,12.8879,2.3311,\n";
//                    else if (init_value == 0)
//                        file << "0,12.8879,2.3205,\n";
//                    else
//                        file << "1,12.8879,2.3278,\n";
//                }
//
//                std::cout << "OpenMP" << std::endl;
//                for (int i = 0; i < 10; i++) {
//                    for (int n_proc = 0; n_proc <= 210; n_proc += 10) {
//                        auto[omp_res, omp_time, num_iter] = bsa.search_omp(n_proc, 1000, init_value);
//                        res[n_proc / 10 + 1] += omp_res;
//                        tim[n_proc / 10 + 1] += omp_time;
//                        itr[n_proc / 10 + 1] += num_iter;
//                    }
//                }
//
//                for (int n_proc = 10; n_proc <= 210; n_proc += 10) {
//                    file << init_value << "," << n_proc << ","
//                         << res[n_proc / 10 + 1] / 10 << ","
//                         << tim[n_proc / 10 + 1] / 10 << ","
//                         << itr[n_proc / 10 + 1] / 10 << ",\n";
//
//                    printf("Init value: %d, n_proc: %d, res :%f, num_iter: %f, Average time: %fs\n", init_value, n_proc,
//                           res[n_proc / 10 + 1] / 10,
//                           itr[n_proc / 10 + 1] / 10,
//                           tim[n_proc / 10 + 1] / 10);
//                }
//                file << "Next,\n";
//            }
//        }
//        file.close();
//    }
//}
//
//void plot_convergence() {
//    for (int n = 4096; n <= 16384; n *= 2) {
//        std::cout << "Init, size: " << n << std::endl;
//        Babai_search_asyn bsa(n);
//        bsa.init(true, true, 0.1);
//        std::cout << "Finish Init" << std::endl;
//
//        std::cout << "OpenMP:" << std::endl;
////        bsa.search_omp_plot();
//    }
//}
inline double do_solve(const int n, const int i, const double *R_A, const double *y_A, const double *z_B) {
    double sum = 0;
#pragma omp simd reduction(+ : sum)
    for (int col = n - i; col < n; col++) {
        sum += R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * z_B[col];
    }

    return round((y_A[n - 1 - i] - sum) / R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
}

void search_omp(const int n_proc, const int nswp, const int n, const double *R_A, const double *y_A,
                int *update, double *z_B, double *z_B_p,
                const MatrixXd &R, const VectorXd &y) {

    int count = 0, num_iter = 0;
    int chunk = std::log2(n);

    double start = omp_get_wtime();
    z_B[n - 1] = round(y_A[n - 1] / R_A[((n - 1) * n) / 2 + n - 1]);
#pragma omp parallel default(shared) num_threads(n_proc) private(count) shared(update)
    {
        for (int j = 0; j < nswp ; j++) {//&& count < 16
            //count = 0;
#pragma omp for schedule(dynamic, chunk) nowait
            for (int i = 0; i < n; i++) {
                z_B[n - 1 - i] = do_solve(n, i, R_A, y_A, z_B);
//                if (x_c != x_p) {
//                    update[n - 1 - i] = 0;
//                    z_B_p[n - 1 - i] = x_c;
//                } else {
//                    update[n - 1 - i] = 1;
//                }
//                sum = 0;
            }
//#pragma omp simd reduction(+ : count)
//            for (int col = 0; col < 32; col++) {
//                count += update[col];
//            }
//            num_iter = j;
//
        }
    }

    double end_time = omp_get_wtime() - start;

    Eigen::Map<Eigen::VectorXd> x_result(z_B, n);
    double res = (y - R * x_result).norm();

    printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", n_proc, num_iter, res, end_time);
    //return {res, end_time, num_iter};
}



int main() {
    cout << omp_get_max_threads() << endl;
    int n = 8192;
    std::cout << "Init, size: " << n << std::endl;
    Babai_search_asyn bsa(n);
    //Babai_search_asyn_massive bsa(n);
    bsa.init(true, true, 0.1);
//    std::cout << "Finish Init" << std::endl;
//
//    std::cout << "Eigen Serial:" << std::endl;
//    auto[eig_res, eig_time] =
//    bsa.search_eigen(0);
//
    std::cout << "Vector Serial:" << std::endl;
//    auto[ser_res, ser_time] =
    bsa.search_vec(0);

    std::cout << "OPENMP:" << std::endl;

    auto *z_B = (double *) malloc(n * sizeof(double));
    auto *z_B_p = (double *) malloc(n * sizeof(double));
    auto *update = (int *) malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        z_B[i] = 0;
        z_B_p[i] = 0;
        update[i] = 0;
    }

    search_omp(12, 10, bsa.n, bsa.R_A, bsa.y_A, update, z_B, z_B_p,
               bsa.R, bsa.y);

    auto *z_B2 = (double *) malloc(n * sizeof(double));
    auto *z_B_p2 = (double *) malloc(n * sizeof(double));
    auto *update2 = (int *) malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        z_B2[i] = 0;
        z_B_p2[i] = 0;
        update2[i] = 0;
    }

    search_omp(9, 10, bsa.n, bsa.R_A, bsa.y_A, update2, z_B2, z_B_p2,
               bsa.R, bsa.y);


    for (int i = 0; i < n; i++) {
        z_B[i] = 0;
        z_B_p[i] = 0;
        update[i] = 0;
    }

    search_omp(6, 10, bsa.n, bsa.R_A, bsa.y_A, update, z_B, z_B_p,
               bsa.R, bsa.y);

    for (int i = 0; i < n; i++) {
        z_B[i] = 0;
        z_B_p[i] = 0;
        update[i] = 0;
    }

    search_omp(3, 10, bsa.n, bsa.R_A, bsa.y_A, update, z_B, z_B_p,
               bsa.R, bsa.y);

    free(z_B);
    free(z_B_p);
    free(update);


//    auto *z_B2 = (double *) malloc(n * sizeof(double));
//    auto *z_B_p2 = (double *) malloc(n * sizeof(double));
//    auto *update2 = (int *) malloc(n * sizeof(int));
//
//    for (int i = 0; i < n; i++) {
//        z_B2[i] = 0;
//        z_B_p2[i] = 0;
//        update2[i] = 0;
//    }
//
//    search_omp(6, 10, bsa.n, bsa.R_A, bsa.y_A, update2, z_B2, z_B_p2,
//               bsa.R, bsa.y);
//
//    free(z_B2);
//    free(z_B_p2);
//    free(update2);
//
//    auto *z_B3 = (double *) malloc(n * sizeof(double));
//    auto *z_B_p3 = (double *) malloc(n * sizeof(double));
//    auto *update3 = (int *) malloc(n * sizeof(int));
//    for (int i = 3; i<=12; i+=3) {
//
//
//
//        for (int i = 0; i < n; i++) {
//            z_B3[i] = 0;
//            z_B_p3[i] = 0;
//            update3[i] = 0;
//        }
//
//        search_omp(i, 10, bsa.n, bsa.R_A, bsa.y_A, update3, z_B3, z_B_p3,
//                   bsa.R, bsa.y);
//
//
//    }
//    free(z_B3);
//    free(z_B_p3);
//    free(update3);
    return 0;
}

//            printf("Init value: %d, n_proc: %d, res :%f, num_iter: %f, Average time: %fs\n", init_value, n_proc,
//                   res / 10, num_iter / 10, time / 10);
//            time = res = num_iter = 0;