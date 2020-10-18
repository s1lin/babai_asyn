#include "Babai_search_asyn.h"
#include "matplotlibcpp.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
namespace plt = matplotlibcpp;


int main() {
    for (int n = 4096; n <= 32768; n *= 2) {
        std::cout << "Init, size: " << n << std::endl;
        Babai_search_asyn bsa(n);
        bsa.init(true, true, 0.1);

        string fx = "res_" + to_string(n) + ".csv";
        ofstream file(fx);
        if (file.is_open()) {
            for (int init_value = -2; init_value <= 2; init_value++) {
                vector<double> res(3, 0), tim(3, 0), itr(3, 0);
                double omp_res = 0, omp_time = 0, num_iter = 0;
                double eig_res = 0, eig_time = 0;
                double ser_res = 0, ser_time = 0;

                for (int i = 0; i < 10; i++) {
                    std::cout << "Eigen Serial:" << std::endl;
                    auto[eig_res, eig_time] = bsa.search_eigen(init_value);
                    res[0] += eig_res;
                    tim[0] += eig_time;

                    std::cout << "Vector Serial:" << std::endl;
                    auto[ser_res, ser_time] = bsa.search_vec(init_value);
                    res[1] += ser_res;
                    tim[1] += ser_time;
                }

                file << init_value << "," << res[0] / 10 << "," << tim[0] / 10 << ",\n";
                file << init_value << "," << res[1] / 10 << "," << tim[1] / 10 << ",\n";

                std::cout << "OpenMP" << std::endl;
                for (int n_proc = 5; n_proc <= 205; n_proc += 10) {
                    for (int i = 0; i < 10; i++) {
                        auto[omp_res, omp_time, num_iter] = bsa.search_omp(n_proc, 1000, init_value);
                        res[2] += omp_res;
                        tim[2] += omp_time;
                        itr[2] += num_iter;
                    }
                    file << init_value << "," << n_proc << "," << res[2] / 10 << "," << tim[2] / 10 << ","
                         << itr[2] / 10 << ",\n";

                    printf("Init value: %d, n_proc: %d, res :%f, num_iter: %f, Average time: %fs\n", init_value, n_proc,
                           res[2] / 10, itr[2] / 10, tim[2] / 10);
                    res[2] = itr[2] = tim[2] = 0;
                }
                file << "Next,\n";
            }
        }
        file.close();
    }
    return 0;
}

//            printf("Init value: %d, n_proc: %d, res :%f, num_iter: %f, Average time: %fs\n", init_value, n_proc,
//                   res / 10, num_iter / 10, time / 10);
//            time = res = num_iter = 0;