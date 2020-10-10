#include <iostream>
#include <Eigen/Dense>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>
#include <random>

#define EPSILON 0.0000001

using namespace std;
using namespace Eigen;

class Babai_search_asyn {
public:
    int n;
    double init_res, tol, max_time;
    vector<double> R_A, R_sA, y_A, x_A;
    Eigen::MatrixXd A, R;
    Eigen::VectorXd y, x0;
private:
    void read_from_file(const string &file_name) {

        ifstream f(file_name);
        string row_string, entry;
        while (getline(f, row_string)) {
            stringstream row_stream(row_string);
            while (getline(row_stream, entry, ',')) {
                double d = stod(entry);
                if (d != 0) {
                    this->R_sA.push_back(d);
                }
                this->R_A.push_back(d);
            }
        }
        this->R = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(R_A.data(), n, R_A.size() / n);
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
                double d = stod(entry);
                if (d != 0) {
                    this->R_sA.push_back(d);
                }
                this->R_A.push_back(d);
            }
        }
    }

public:
    explicit Babai_search_asyn(int n) {

        this->n = n;
        this->A = MatrixXd::Zero(n, n);
        this->R = Eigen::MatrixXd::Zero(n, n);
        this->R_A = vector<double>();
        this->R_sA = vector<double>();
        this->x0 = VectorXd::Zero(n);//.unaryExpr([&](int dummy) { return round(dis(gen)); });
        this->y = VectorXd::Zero(n);
        this->y_A.resize(y.size());
        this->x_A.resize(x0.size());
        this->max_time = INFINITY;
        this->init_res = INFINITY;
        this->tol = INFINITY;

    }

    void init(bool f, double noise) {
        std::random_device rd;
        std::mt19937 gen(rd());  //here you could also set a seed
        std::normal_distribution<double> norm_dis(0, 1);//mean:0, std:1. same as matlab.

        string file_name = "R_" + to_string(n) + ".csv";

        if (f) {
            read_from_file(file_name);
        } else {
            this->A = MatrixXd::Zero(n, n).unaryExpr([&](int dummy) { return norm_dis(gen); });
            this->R = A.householderQr().matrixQR().triangularView<Eigen::Upper>();
            write_to_file(file_name);
        }
        //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
        std::uniform_int_distribution<int> int_dis(-n, n);

        this->x0 = VectorXd::Zero(n).unaryExpr([&](int dummy) { return static_cast<double>(int_dis(gen)); });
        this->y = R * x0 + noise * VectorXd::Zero(n).unaryExpr([&](int dummy) { return norm_dis(gen); });
        this->init_res = (y - R * x0).norm();
        VectorXd::Map(&y_A[0], n) = y;
        VectorXd::Map(&x_A[0], n) = x0;
    }




};