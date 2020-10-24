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

int main() {
    cout << omp_get_max_threads() << endl;
    int n = 1024;
    std::cout << "Init, size: " << n << std::endl;
    Babai_search_asyn bsa(n);
    //Babai_search_asyn_massive bsa(n);
    bsa.init(false, false, 0.1);
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
//    auto[ser_res, ser_time] =
    bsa.search_omp(10, 10, 0);
    bsa.search_omp(20, 10, 0);
    bsa.search_omp(30, 10, 0);

//
//    plot_convergence();
//    plot_run();

    return 0;
}

//            printf("Init value: %d, n_proc: %d, res :%f, num_iter: %f, Average time: %fs\n", init_value, n_proc,
//                   res / 10, num_iter / 10, time / 10);
//            time = res = num_iter = 0;