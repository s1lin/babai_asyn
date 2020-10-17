#include <iostream>
#include <Eigen/Dense>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <random>
#include <ctime>
#include <iomanip>
#include "matplotlibcpp.h"

#define EPSILON 0.01

using namespace std;
using namespace Eigen;

namespace plt = matplotlibcpp;

class Babai_search_asyn {
public:
    int n;
    double init_res, tol, max_time;
    double *R_sA, *x_A, *y_A;
    vector<vector<double>> R_V;
    Eigen::MatrixXd A, R;
    Eigen::VectorXd y, x_t;

private:
    void read_from_file(const string &file_name) {
        int i = 0, j = 0;
        vector<double> temp;
        ifstream f(file_name);
        string row_string, entry;
        while (getline(f, row_string)) {
            stringstream row_stream(row_string);
            while (getline(row_stream, entry, ',')) {
                double d = stod(entry);
                if (d != 0) {
                    this->R_sA[i] = d;
                    i++;
                }
                this->R_V[j].push_back(d);
                temp.push_back(d);
            }
            j++;
        }
        this->R = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(temp.data(), n, temp.size() / n);
    }

    void write_to_file(const string &file_name) {
        int i = 0, j = 0;
        const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
        ofstream file(file_name);
        if (file.is_open()) {
            file << R.format(CSVFormat);
            file.close();
        }

        ifstream f(file_name);
        string row_string, entry;
        while (getline(f, row_string)) {
            stringstream row_stream(row_string);
            while (getline(row_stream, entry, ',')) {
                double d = stod(entry);
                if (d != 0) {
                    this->R_sA[i] = d;
                    i++;
                }
                this->R_V[j].push_back(d);
            }
            j++;
        }
    }

    void read_x_y() {
        string fy =
                "/home/shilei/CLionProjects/babai_asyn/data/y_" + to_string(n) + ".csv";
        string fx =
                "/home/shilei/CLionProjects/babai_asyn/data/x_" + to_string(n) + ".csv";
        string row_string, entry;
        int index = 0;
        ifstream f1(fy);
        while (getline(f1, row_string)) {
            //cout<<row_string
            double d = stod(row_string);
            //cout<<setprecision(15)<<d<<endl;
            this->y_A[index] = d;
            this->y(index) = d;
            index++;
        }

        index = 0;
        ifstream f2(fx);
        string row_string2, entry2;
        while (getline(f2, row_string2)) {
            double d = stod(row_string2);
            this->x_t(index) = d;
            index++;
        }
    }

    void write_x_y() {
        const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
        string fy =
                "/home/shilei/CLionProjects/babai_asyn/data/y_" + to_string(n) + ".csv";
        ofstream file2(fy);
        if (file2.is_open()) {
            file2 << y.format(CSVFormat);
            file2.close();
        }

        string fx =
                "/home/shilei/CLionProjects/babai_asyn/data/x_" + to_string(n) + ".csv";
        ofstream file3(fx);
        if (file3.is_open()) {
            file3 << x_t.format(CSVFormat);
            file3.close();
        }
    }

public:
    explicit Babai_search_asyn(int n) {

        this->n = n;
        this->A = MatrixXd::Zero(n, n);
        this->R = Eigen::MatrixXd::Zero(n, n);
        this->R_V = vector<vector<double>>();
        this->R_sA = (double *) calloc(n * n, sizeof(double));
        this->x_A = (double *) calloc(n, sizeof(double));
        this->y_A = (double *) calloc(n, sizeof(double));
        this->x_t = VectorXd::Zero(n);
        this->y = VectorXd::Zero(n);
        this->max_time = INFINITY;
        this->init_res = INFINITY;
        this->tol = INFINITY;
        this->R_V.resize(n * n);
    }

    ~Babai_search_asyn() {
        free(R_sA);
        free(x_A);
        free(y_A);
    }

    void init(bool read_r, bool read_xy, double noise) {
        std::random_device rd;
        std::mt19937 gen(rd());

        //mean:0, std:1. same as matlab.
        std::normal_distribution<double> norm_dis(0, 1);

        //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
        std::uniform_int_distribution<int> int_dis(-n, n);

        string file_name =
                "/home/shilei/CLionProjects/babai_asyn/data/R_" + to_string(n) + ".csv";

        if (read_r) {
            read_from_file(file_name);

        } else {
            this->A = MatrixXd::Zero(n, n).unaryExpr([&](double dummy) { return norm_dis(gen); });
            this->R = A.householderQr().matrixQR().triangularView<Eigen::Upper>();
            write_to_file(file_name);
        }

        if (read_xy) {
            read_x_y();
            this->init_res = (y - R * x_t).norm();
        } else {
            this->x_t = VectorXd::Zero(n).unaryExpr([&](int dummy) { return static_cast<double>(int_dis(gen)); });
            this->y = R * x_t + noise * VectorXd::Zero(n).unaryExpr([&](double dummy) { return norm_dis(gen); });
            this->init_res = (y - R * x_t).norm();
            VectorXd::Map(&y_A[0], n) = y;
            write_x_y();
        }
    }

    double search_omp(int n_proc, int nswp, int init_value) {
        double sum = 0, dist = INFINITY;
        int i = 0, j = 0, count = 0;

        auto *z_B = (double *) calloc(n, sizeof(double));
        auto *z_B_p = (double *) calloc(n, sizeof(double));
        auto *update = (double *) calloc(n, sizeof(double));

#pragma omp for
        for (int i = 0; i < n; i++) {
            z_B[i] = init_value;
            z_B_p[i] = init_value;
            update[i] = 0;
        }
        double start = omp_get_wtime();
        z_B[n - 1] = round(y_A[n - 1] / R_sA[((n - 1) * n) / 2 + n - 1]);
#pragma omp parallel num_threads(n_proc) private(sum, i, j, count) shared(update)
        {
            for (int j = 0; j < nswp; j++) {

#pragma omp for nowait schedule(dynamic)
                for (int i = 0; i < n; i++) {
#pragma omp simd reduction(+ : sum)
                    for (int col = n - i; col < n; col++) {
                        sum += R_sA[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * z_B[col];
                    }
                    int x_c = round(
                            (y_A[n - 1 - i] - sum) / R_sA[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
                    int x_p = z_B_p[n - 1 - i];
                    z_B[n - 1 - i] = x_c;

                    if (x_c != x_p) {
                        update[n - 1 - i] = 0;
                        z_B_p[n - 1 - i] = x_c;
                    }else{
                        update[n - 1 - i] = 1;
                    }
                    sum = 0;

                }
#pragma omp simd reduction(+ : count)
                for (int col = 0; col < n; col++) {
                    count += update[col];
                }
                if(count == n){
                    break;
                }
                count = 0;
            }
        }
        double end_time = omp_get_wtime() - start;

        Eigen::Map<Eigen::VectorXd> x_result(z_B, n);
        double res = (y - R * x_result).norm();

        //if (res - tol < EPSILON)// && end_time < max_time)
        printf("Thread: %d, Sweep: %d, Res: %.5f, count :%d, Run time: %fs\n", n_proc, j, res, count, end_time);
        free(z_B);
        free(z_B_p);
        free(update);
        return end_time;

//            if(n_proc == 0) {
//                for (int nswp = 0; nswp < 20; nswp++) {
//                    nswp_pl.push_back(nswp);
//                    res_pl.push_back(init_res);
//                    tim_pl.push_back(find_raw_x0());
//                }
//
//                const std::map<std::string, std::string> keyword_arg{
//                        {"marker",     "o"},
//                        {"markersize", "5"},
//                        {"label",      "Serial"}
//                };
//
//                plt::xlim(1, 20);
//                plt::plot(nswp_pl, tim_pl, keyword_arg);
//
//                const std::map<std::string, std::string> keyword_arg2{
//                        {"marker",     "1"},
//                        {"markersize", "5"},
//                        {"label",      "Matlab"}
//                };
//
//                string tim =
//                        "/home/shilei/CLionProjects/babai_asyn/data/Res_" + to_string(n) + ".csv";
//                string row_string, entry;
//                int index = 0;
//                vector<double> nswp_pl2(20), tim_pl2(20);
//                ifstream f1(tim);
//                while (getline(f1, row_string)) {
//                    double d = stod(row_string);
//                    nswp_pl2.push_back(index);
//                    tim_pl2.push_back(d);
//                    index++;
//                }
//
//                plt::xlim(1, 20);
//                plt::plot(nswp_pl2, tim_pl2, keyword_arg2);
//
//            }else{
//                const std::map<std::string, std::string> keyword_arg{
//                        {"marker",     "x"},
//                        {"markersize", "5"},
//                        {"label",      "num_thread:" + to_string(n_proc)}
//                };
//
//                plt::xlim(1, 20);
//                plt::plot(nswp_pl, res_pl, keyword_arg);
//            }
//    }
//
//    plt::title("Residual with Threads by OpenMP");
//
//    plt::legend();
//
//    plt::xlabel("Num of iterations");
//    plt::ylabel("Residual");
//    plt::save("./resOMP.png");
    }

    double search_eigen(int init_value) {
        VectorXd z_B;
        switch (init_value) {
            case -1:
                z_B = VectorXd::Zero(n);
                z_B.setConstant(n, -1);
                break;
            case 1:
                z_B = VectorXd::Ones(n);
                break;
            default:
                z_B = VectorXd::Zero(n);//x_t;
        }
        cout << z_B[0];
        double start = omp_get_wtime();

        double s = y(n - 1) / R(n - 1, n - 1);
        z_B(n - 1) = round(s);
        for (int k = n - 2; k >= 0; k--) {
            VectorXd d = R.block(k, k + 1, 1, n - k - 1) * z_B.block(k + 1, 0, n - k - 1, 1);
            s = (y(k) - d(0)) / R(k, k);
            z_B(k) = round(s);
        }
        double time = omp_get_wtime() - start;

        double res = (y - R * z_B).norm();
        printf("For %d steps, res = %.5f, init_res = %.5f %f seconds\n", n, res, init_res, time);
        return time;
    }

    double search_vec(int init_value) {
        vector<double> z_B(n, init_value);
        cout << z_B[0];
        double sum = 0;
        double start = omp_get_wtime();
        z_B[n - 1] = round(y_A[n - 1] / R_V[n - 1][n - 1]);
        for (int i = 1; i < n; i++) {
            int k = n - i - 1;
            for (int col = n - i; col < n; col++)
                sum += R_V[k][col] * z_B[col];

            z_B[k] = round((y_A[k] - sum) / R_V[k][k]);
            sum = 0;
        }
        double time = omp_get_wtime() - start;

        Eigen::Map<Eigen::VectorXd> x_result(z_B.data(), n);
        double res = (y - R * x_result).norm();

        this->tol = res;
        this->max_time = time;

        printf("For %d steps, res = %.5f, init_res = %.5f %f seconds\n", n, res, init_res, max_time);
        return res;

    }


};