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
//#include <netcdfcpp.h>
//#include "matplotlibcpp.h"

#define EPSILON 0.01

using namespace std;
using namespace Eigen;

//namespace plt = matplotlibcpp;

class Babai_search_asyn {
public:
    int n, size_R_A;
    bool eigen;
    double init_res, max_time;
    double *R_A, *y_A, *x_R, *x_tA;
    Eigen::MatrixXd A, R;
    Eigen::VectorXd y, x_t;

private:
    void read_from_RA() {
        string fxR =
                "../../data/R_A_" + to_string(n) + ".csv";
        int index = 0;
        ifstream f3(fxR);
        string row_string3, entry3;
        while (getline(f3, row_string3)) {
            stringstream row_stream(row_string3);
            while (getline(row_stream, entry3, ',')) {
                double d = stod(entry3);
                this->R_A[index] = d;
                index++;
            }
        }
        this->size_R_A = index;
    }

    void read_from_R() {
        string file_name = "../../data/R_" + to_string(n) + ".csv";
        int i = 0, j = 0;
        vector<double> temp;
        ifstream f(file_name);
        string row_string, entry;
        while (getline(f, row_string)) {
            stringstream row_stream(row_string);
            while (getline(row_stream, entry, ',')) {
                double d = stod(entry);
                if (d != 0) {
                    this->R_A[i] = d;
                    i++;
                }
                if (this->eigen)
                    temp.push_back(d);
            }
            j++;
        }
        this->size_R_A = i;
        if (this->eigen)
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
                    this->R_A[i] = d;
                    i++;
                }
            }
            j++;
        }
    }

    void read_x_y() {
        string fy =
                "../../data/y_" + to_string(n) + ".csv";
        string fx =
                "../../data/x_" + to_string(n) + ".csv";
        string fxR =
                "../../data/x_R_" + to_string(n) + ".csv";
        string row_string, entry;
        int index = 0;
        ifstream f1(fy);
        while (getline(f1, row_string)) {
            //cout<<row_string
            double d = stod(row_string);
            //cout<<setprecision(15)<<d<<endl;
            this->y_A[index] = d;
            if (this->eigen)
                this->y(index) = d;
            index++;
        }

        index = 0;
        ifstream f2(fx);
        string row_string2, entry2;
        while (getline(f2, row_string2)) {
            double d = stod(row_string2);
            this->x_tA[index] = d;
            if (this->eigen)
                this->x_t(index) = d;
            index++;
        }

        index = 0;
        ifstream f3(fxR);
        string row_string3, entry3;
        while (getline(f3, row_string3)) {
            double d = stod(row_string3);
            this->x_R[index] = d;
            index++;
        }
    }

    void write_x_y() {
        const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
        string fy =
                "../../data/y_" + to_string(n) + ".csv";
        ofstream file2(fy);
        if (file2.is_open()) {
            file2 << y.format(CSVFormat);
            file2.close();
        }

        string fx =
                "../../data/x_" + to_string(n) + ".csv";
        ofstream file3(fx);
        if (file3.is_open()) {
            file3 << x_t.format(CSVFormat);
            file3.close();
        }
    }

    void write_R_A() const {
        string fR = "../../data/R_A_" + to_string(n) + ".csv";
        ofstream file3(fR);
        if (file3.is_open()) {
            for (int i = 0; i < size_R_A; i++)
                file3 << setprecision(15)<<R_A[i] << ",";
            file3.close();
        }
    }

public:
    explicit Babai_search_asyn(int n, bool eigen) {
        this->n = n;
        this->eigen = eigen;
        if (eigen) {
            this->A = MatrixXd::Zero(n, n);
            this->R = Eigen::MatrixXd::Zero(n, n);
            this->x_t = VectorXd::Zero(n);
            this->y = VectorXd::Zero(n);
        }
        this->R_A = (double *) calloc(n * n, sizeof(double));
        this->x_R = (double *) calloc(n, sizeof(double));
        this->x_tA = (double *) calloc(n, sizeof(double));
        this->y_A = (double *) calloc(n, sizeof(double));

        this->max_time = INFINITY;
        this->init_res = INFINITY;
        this->size_R_A = 0;
    }

    ~Babai_search_asyn() {
        free(R_A);
        free(x_R);
        free(y_A);
    }

    static double find_residual(const int n, const double *R, const double *y, const double *x) {
        double res = 0;
        for (int i = 0; i < n; i++) {
            double sum = 0;
            for (int j = i; j < n; j++) {
                sum += x[j] * R[(n * i) + j - ((i * (i + 1)) / 2)];
            }
            res += (y[i] - sum) * (y[i] - sum);
        }
        return std::sqrt(res);
    }

    void compare_R_RA(){
        string file_name = "../../data/R_" + to_string(n) + ".csv";
        int i = 0;
        vector<double> temp;
        ifstream f(file_name);
        string row_string, entry;
        while (getline(f, row_string)) {
            stringstream row_stream(row_string);
            while (getline(row_stream, entry, ',')) {
                double d = stod(entry);
                if(d != 0){
                    if(R_A[i] - d != 0){
                        cout<<i<<endl;
                    }
                    i++;
                }

            }
        }
    }

    void init(bool read_r, bool read_ra, bool read_xy, double noise) {
        std::random_device rd;
        std::mt19937 gen(rd());
        //mean:0, std:1. same as matlab.
        std::normal_distribution<double> norm_dis(0, 1);
        //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
        std::uniform_int_distribution<int> int_dis(-n, n);

        if (read_r) {
            read_from_R();
            write_R_A();
        } else if (read_ra) {
            read_from_RA();
//            compare_R_RA();
        } else {
            this->A = MatrixXd::Zero(n, n).unaryExpr([&](double dummy) { return norm_dis(gen); });
            this->R = A.householderQr().matrixQR().triangularView<Eigen::Upper>();
            int index = 0;

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    if (this->R(i, j) != 0) {
                        this->R_A[index] = this->R(i, j);
                        index++;
                    }
                }
            }
        }

        if (read_xy) {
            read_x_y();
            if (this->eigen)
                this->init_res = (y - R * x_t).norm();
            else
                this->init_res = find_residual(n, R_A, y_A, x_tA);
            cout << this->init_res << endl;
        } else {
            this->x_t = VectorXd::Zero(n).unaryExpr([&](int dummy) { return static_cast<double>(int_dis(gen)); });
            this->y = R * x_t + noise * VectorXd::Zero(n).unaryExpr([&](double dummy) { return norm_dis(gen); });
            this->init_res = (y - R * x_t).norm();
            VectorXd::Map(&y_A[0], n) = y;
        }
    }

//    tuple<double, double, int>
    void search_omp(int n_proc, int nswp, int init_value) {

        double sum = 0, dist = 0, res = 0;
        int count = 0, num_iter = 0;

        auto *z_B = (double *) calloc(n, sizeof(double));
        auto *z_B_p = (double *) calloc(n, sizeof(double));
        auto *update = (double *) calloc(n, sizeof(double));

        if (init_value == -1) {
            for (int i = 0; i < n; i++) {
                z_B[i] = x_R[i];
                z_B_p[i] = x_R[i];
                update[i] = 0;
            }
        } else {
            for (int i = 0; i < n; i++) {
                z_B[i] = init_value;
                z_B_p[i] = init_value;
                update[i] = 0;
            }
        }
        if (n_proc == 0)
            n_proc = 5;
        double start = omp_get_wtime();
        z_B[n - 1] = round(y_A[n - 1] / R_A[((n - 1) * n) / 2 + n - 1]);
#pragma omp parallel num_threads(n_proc) private(sum, count, dist) shared(update)
        {
            for (int j = 0; j < nswp && count < 16; j++) {
                count = 0;
#pragma omp for nowait schedule(dynamic)
                for (int i = 0; i < n; i++) {
#pragma omp simd reduction(+ : sum)
                    for (int col = n - i; col < n; col++) {
                        sum += R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * z_B[col];
                    }
                    int x_c = round(
                            (y_A[n - 1 - i] - sum) / R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
                    int x_p = z_B_p[n - 1 - i];
                    z_B[n - 1 - i] = x_c;

                    if (x_c != x_p) {
                        update[n - 1 - i] = 0;
                        z_B_p[n - 1 - i] = x_c;
                    } else {
                        update[n - 1 - i] = 1;
                    }
                    sum = 0;
                }
//#pragma omp simd reduction(+ : dist)
//                for (int col = 0; col < n; col++) {
//                    dist += abs(z_B[col] - z_B_p[col]);
//                    z_B_p[col] = z_B[col];
//                }
//                if (dist == 0 && j != 0) {
//                    num_iter = j;
//                    break;
//                }
//                dist = 0;


#pragma omp simd reduction(+ : count)
                for (int col = 0; col < 32; col++) {
                    count += update[col];
                }
                num_iter = j;

            }
        }
        double end_time = omp_get_wtime() - start;
        if (this->eigen) {
            Eigen::Map<Eigen::VectorXd> x_result(z_B, n);
            res = (y - R * x_result).norm();
        } else {
            res = find_residual(n, R_A, y_A, z_B);
        }

        printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", n_proc, num_iter, res, end_time);
        free(z_B);
        free(z_B_p);
        free(update);

        //return {res, end_time, num_iter};
    }

    //tuple<double, double>
//    void search_eigen(int init_value) {
//        VectorXd z_B = VectorXd(n);
//        z_B.setConstant(n, init_value);
//        double start = omp_get_wtime();
//        double s = y(n - 1) / R(n - 1, n - 1);
//
//        for (int k = n - 2; k >= 0; k--) {
//            VectorXd d = R.block(k, k + 1, 1, n - k - 1) * z_B.block(k + 1, 0, n - k - 1, 1);
//            s = (y(k) - d(0)) / R(k, k);
//            cout << d(0) << endl;
//            z_B(k) = round(s);
//        }
//        double time = omp_get_wtime() - start;
//        cout << z_B.transpose();
//        double res = (y - R * z_B).norm();
//        printf("Res = %.5f, init_res = %.5f %f seconds\n", res, init_res, time);
//        //return {res, time};
//    }

//    tuple<double, double>
    void search_vec(int init_value) {

        vector<double> z_B(n, init_value);

        double sum = 0, res = 0;
        if (init_value == -1) {
            for (int i = 0; i < n; i++) {
                z_B[i] = x_R[i];
            }
        } else {
            for (int i = 0; i < n; i++) {
                z_B[i] = init_value;
            }
        }

        double start = omp_get_wtime();
        z_B[n - 1] = round(y_A[n - 1] / R_A[((n - 1) * n) / 2 + n - 1]);

        for (int i = 1; i < n; i++) {
            int k = n - i - 1;
            for (int col = n - i; col < n; col++) {
                sum += R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * z_B[col];
            }
            z_B[k] = round(
                    (y_A[n - 1 - i] - sum) / R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
            sum = 0;
        }
        double time = omp_get_wtime() - start;

        if (this->eigen) {
            Eigen::Map<Eigen::VectorXd> x_result(z_B.data(), n);
            res = (y - R * x_result).norm();
        } else {
            res = find_residual(n, R_A, y_A, z_B.data());
        }


        this->max_time = time;

        printf("Res = %.5f, init_res = %.5f %f seconds\n", res, init_res, max_time);
        //return {res, time};
    }

//    void search_omp_plot() {
//        int max_iter = 16;
//        int n_proc[] = {10, 30, 60, 110, 160, 210};
//        for (int k = 0; k < 6; k++) {
//            for (int init_value = -1; init_value <= 1; init_value++) {
//                vector<double> nswp_pl(max_iter, 0), res_pl(max_iter, 0);
//
//                double sum = 0, dist = 0;
//                int count = 0, num_iter = 0;
//                auto *z_B = (double *) calloc(n, sizeof(double));
//                auto *z_B_p = (double *) calloc(n, sizeof(double));
//                auto *update = (double *) calloc(n, sizeof(double));
//                if(init_value == -1){
//                    for (int i = 0; i < n; i++) {
//                        z_B[i] = x_R[i];
//                        z_B_p[i] = x_R[i];
//                        update[i] = 0;
//                    }
//                } else {
//                    for (int i = 0; i < n; i++) {
//                        z_B[i] = init_value;
//                        z_B_p[i] = init_value;
//                        update[i] = 0;
//                    }
//                }
//                double start = omp_get_wtime();
//                z_B[n - 1] = round(y_A[n - 1] / R_A[((n - 1) * n) / 2 + n - 1]);
//
//#pragma omp parallel num_threads(k) private(sum, count, dist) shared(update)
//                {
//                    for (int j = 0; j < max_iter; j++) {
//                        count = 0;
//#pragma omp for nowait schedule(dynamic)
//                        for (int i = 0; i < n; i++) {
//#pragma omp simd reduction(+ : sum)
//                            for (int col = n - i; col < n; col++) {
//                                sum += R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * z_B[col];
//                            }
//                            int x_c = round(
//                                    (y_A[n - 1 - i] - sum) /
//                                    R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
//                            int x_p = z_B_p[n - 1 - i];
//                            z_B[n - 1 - i] = x_c;
//
//                            if (x_c != x_p) {
//                                update[n - 1 - i] = 0;
//                                z_B_p[n - 1 - i] = x_c;
//                            } else {
//                                update[n - 1 - i] = 1;
//                            }
//                            sum = 0;
//                        }
//#pragma omp single
//                        {
//                            nswp_pl.push_back(j);
//                            Eigen::Map<Eigen::VectorXd> x_result(z_B, n);
//                            res_pl.push_back((y - R * x_result).norm());
//                        }
//                    }
//                }
//                double end_time = omp_get_wtime() - start;
//
//                Eigen::Map<Eigen::VectorXd> x_result(z_B, n);
//                double res = (y - R * x_result).norm();
//
//                //if (res - tol < EPSILON)// && end_time < max_time)
//                //printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", n_proc, num_iter, res, end_time);
//                free(z_B);
//                free(z_B_p);
//                free(update);
//                plt::xlim(0, max_iter);
//
//                if(init_value != -1) {
//                    const std::map<std::string, std::string> keyword_arg{
//                            {"marker",     "x"},
//                            {"markersize", "5"},
//                            {"label",      "Init Guess:" + to_string(init_value)}
//                    };
//                    plt::plot(nswp_pl, res_pl, keyword_arg);
//                }else{
//                    const std::map<std::string, std::string> keyword_arg{
//                            {"marker",     "x"},
//                            {"markersize", "5"},
//                            {"label",      "Init Guess: the round of real solution" }
//                    };
//                    plt::plot(nswp_pl, res_pl, keyword_arg);
//                }
//
//            }
//            plt::title("Convergence of residual with " + to_string(n_proc[k]) + " Threads by OpenMP");
//
//            plt::legend();
//
//            plt::xlabel("Num of iterations");
//            plt::ylabel("Residual");
//            plt::save("./resOMP_" + to_string(n_proc[k]) + "_" + to_string(n) + ".png");
//            plt::close();
//            cout << "./resOMP_" + to_string(n_proc[k]) + "_" + to_string(n) + ".png" << endl;
//        }
//    }
};