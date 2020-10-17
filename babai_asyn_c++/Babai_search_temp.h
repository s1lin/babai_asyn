//
// Created by shilei on 2020-10-13.
//

#ifndef BABAI_ASYN_BABAI_SEARCH_TEMP_H
#define BABAI_ASYN_BABAI_SEARCH_TEMP_H

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

#define EPSILON 0.0000001

using namespace std;
using namespace Eigen;

namespace plt = matplotlibcpp;

template <typename INT, typename DOUBLE>
class Babai_search_temp {
public:
    INT n;
    DOUBLE init_res, tol, max_time;
    vector<DOUBLE> R_A, R_sA, y_A, x_A;
    Eigen::MatrixXd A, R;
    Eigen::VectorXd y, x0;
private:
    void read_from_file(const string &file_name) {

        ifstream f(file_name);
        string row_string, entry;
        while (getline(f, row_string)) {
            stringstream row_stream(row_string);
            while (getline(row_stream, entry, ',')) {
                DOUBLE d = stod(entry);
                if (d != 0) {
                    this->R_sA.push_back(d);
                }
                this->R_A.push_back(d);
            }
        }
        this->R = Map<Matrix<DOUBLE, Dynamic, Dynamic, RowMajor>>(R_A.data(), n, R_A.size() / n);
    }

    void write_to_file(const string &file_name) {
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
                DOUBLE d = stod(entry);
                if (d != 0) {
                    this->R_sA.push_back(d);
                }
                this->R_A.push_back(d);
            }
        }
    }

    void read_x_y() {
        string fy =
                "/home/shilei/CLionProjects/babai_asyn/data/y_" + to_string(n) + ".csv";
        string fx =
                "/home/shilei/CLionProjects/babai_asyn/data/x_" + to_string(n) + ".csv";
        string row_string, entry;
        INT index = 0;
        ifstream f1(fy);
        while (getline(f1, row_string)) {
            //cout<<row_string
            DOUBLE d = stod(row_string);
            //cout<<setprecision(15)<<d<<endl;
            this->y_A[index] = d;
            this->y(index) = d;
            index++;
        }

        index = 0;
        ifstream f2(fx);
        string row_string2, entry2;
        while (getline(f2, row_string2)) {
            DOUBLE d = stod(row_string2);
            this->x_A[index] = d;
            this->x0(index) = d;
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
            file3 << x0.format(CSVFormat);
            file3.close();
        }
    }

public:
    explicit Babai_search_temp(INT n) {

        this->n = n;
        this->A = MatrixXd::Zero(n, n);
        this->R = Eigen::MatrixXd::Zero(n, n);
        this->R_A = vector<DOUBLE>();
        this->R_sA = vector<DOUBLE>();
        this->x0 = VectorXd::Zero(n);//.unaryExpr([&](INT dummy) { return round(dis(gen)); });
        this->y = VectorXd::Zero(n);
        this->y_A.resize(y.size());
        this->x_A.resize(x0.size());
        this->max_time = INFINITY;
        this->init_res = INFINITY;
        this->tol = INFINITY;

    }

    void init(bool read_r, bool read_xy, DOUBLE noise) {
        std::random_device rd;
        std::mt19937 gen(rd());

        //mean:0, std:1. same as matlab.
        std::normal_distribution<DOUBLE> norm_dis(0, 1);

        //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
        std::uniform_int_distribution<INT> int_dis(-n, n);

        string file_name =
                "/home/shilei/CLionProjects/babai_asyn/data/R_" + to_string(n) + ".csv";

        if (read_r) {
            read_from_file(file_name);

        } else {
            this->A = MatrixXd::Zero(n, n).unaryExpr([&](DOUBLE dummy) { return norm_dis(gen); });
            this->R = A.householderQr().matrixQR().triangularView<Eigen::Upper>();
            write_to_file(file_name);
        }

        if (read_xy) {
            read_x_y();
            this->init_res = (y - R * x0).norm();
        } else {
            this->x0 = VectorXd::Zero(n).unaryExpr([&](INT dummy) { return static_cast<DOUBLE>(int_dis(gen)); });
            this->y = R * x0 + noise * VectorXd::Zero(n).unaryExpr([&](DOUBLE dummy) { return norm_dis(gen); });
            this->init_res = (y - R * x0).norm();
            VectorXd::Map(&y_A[0], n) = y;
            write_x_y();
        }
    }

    void find_raw_x0_OMP() {

        for (INT n_proc = 0; n_proc <= 50; n_proc += 10) {
            std::vector<DOUBLE> nswp_pl(20), res_pl(20), tim_pl(20);
            if (n_proc != 0) {
                for (INT nswp = 0; nswp < 20; nswp++) {
                    nswp_pl.push_back(nswp);
                    vector<DOUBLE> raw_x_A = {};
                    raw_x_A.resize(n);
                    raw_x_A[n - 1] = round(y_A[n - 1] / R_sA[((n - 1) * (n)) / 2 + n - 1]);

                    DOUBLE start = omp_get_wtime();
#ifdef _OPENMP
                    DOUBLE sum = 0;
#pragma omp parallel num_threads(n_proc) private(sum) shared(raw_x_A)
                    {
                        for (INT j = 0; j < nswp; j++) {
#pragma omp for nowait //schedule(dynamic, n_proc)
                            for (INT i = 1; i < n; i++) {
#pragma omp simd reduction(+ : sum)
                                for (INT col = n - i; col < n; col++)
                                    sum += R_sA[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * raw_x_A[col];
                                raw_x_A[n - 1 - i] = round(
                                        (y_A[n - 1 - i] - sum) /
                                        R_sA[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
                                sum = 0;
                            }
                        }
                    }
#endif
                    DOUBLE end_time = omp_get_wtime() - start;
                    VectorXd x_result = VectorXd(n);
                    for (INT i = 0; i < n; i++) {
                        x_result(i) = raw_x_A[i];
                    }

                    DOUBLE res = (y - R * x_result).norm();
                    res_pl.push_back(res);
                    tim_pl.push_back(end_time);

                    //if (res - tol < EPSILON && end_time < max_time * 0.5)
                    //printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", n_proc, nswp, res, end_time);

                }
            }

            if(n_proc == 0) {
                for (INT nswp = 0; nswp < 20; nswp++) {
                    nswp_pl.push_back(nswp);
                    res_pl.push_back(init_res);
                    tim_pl.push_back(find_raw_x0());
                }

                const std::map<std::string, std::string> keyword_arg{
                        {"marker",     "o"},
                        {"markersize", "5"},
                        {"label",      "Serial"}
                };

                plt::xlim(1, 20);
                plt::plot(nswp_pl, tim_pl, keyword_arg);

                const std::map<std::string, std::string> keyword_arg2{
                        {"marker",     "1"},
                        {"markersize", "5"},
                        {"label",      "Matlab"}
                };

                string tim =
                        "/home/shilei/CLionProjects/babai_asyn/data/Res_" + to_string(n) + ".csv";
                string row_string, entry;
                INT index = 0;
                vector<DOUBLE> nswp_pl2(20), tim_pl2(20);
                ifstream f1(tim);
                while (getline(f1, row_string)) {
                    DOUBLE d = stod(row_string);
                    nswp_pl2.push_back(index);
                    tim_pl2.push_back(d);
                    index++;
                }

                plt::xlim(1, 20);
                plt::plot(nswp_pl2, tim_pl2, keyword_arg2);

            }else{
                const std::map<std::string, std::string> keyword_arg{
                        {"marker",     "x"},
                        {"markersize", "5"},
                        {"label",      "num_thread:" + to_string(n_proc)}
                };

                plt::xlim(1, 20);
                plt::plot(nswp_pl, res_pl, keyword_arg);
            }
        }

        plt::title("Residual with Threads by OpenMP");
        plt::legend();
        plt::xlabel("Num of iterations");
        plt::ylabel("Residual");
        plt::save("./resOMP.png");
    }

    DOUBLE find_raw_x0_vec() {
        DOUBLE res;
        std::cout << "find_raw_x0" << std::endl;

        DOUBLE s = y(n - 1) / R(n - 1, n - 1);
        VectorXd raw_x0 = VectorXd::Zero(n);//x0;
        raw_x0(n - 1) = round(s);
        DOUBLE start = omp_get_wtime();
        for (INT k = n - 2; k >= 0; k--) {
            VectorXd d = R.block(k, k + 1, 1, n - k - 1) * raw_x0.block(k + 1, 0, n - k - 1, 1);
            s = (y(k) - d(0)) / R(k, k);
            raw_x0(k) = round(s);
        }
        DOUBLE time = omp_get_wtime() - start;

        res = (y - R * raw_x0).norm();
        this->tol = res;
        this->max_time = time;
        //printf("For %d steps, res = %.5f, init_res = %.5f %f seconds\n", n, res, init_res, max_time);
        return time;

    }

    DOUBLE find_raw_x0() {
        vector<DOUBLE> raw_x_A;
        raw_x_A.resize(n);
        raw_x_A[n - 1] = round(y_A[n - 1] / R_sA[((n - 1) * (n)) / 2 + n - 1]);

        DOUBLE start = omp_get_wtime();
        DOUBLE sum = 0;
        for (INT i = 1; i < n; i++) {
            for (INT col = n - i; col < n; col++)
                sum += R_sA[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * raw_x_A[col];
            raw_x_A[n - 1 - i] = round(
                    (y_A[n - 1 - i] - sum) /
                    R_sA[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
            sum = 0;
        }
        DOUBLE time = omp_get_wtime() - start;
        VectorXd x_result = VectorXd(n);
        for (INT i = 0; i < n; i++) {
            x_result(i) = raw_x_A[i];
        }

        DOUBLE res = (y - R * x_result).norm();
        this->tol = res;
        this->max_time = time;

        printf("For %d steps, res = %.5f, init_res = %.5f %f seconds\n", n, res, init_res, max_time);
        return res;

    }
};

#endif //BABAI_ASYN_BABAI_SEARCH_TEMP_H
