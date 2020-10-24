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

#define EPSILON 0.01

using namespace std;
using namespace Eigen;


class Babai_search_asyn {
public:
    int n;
    double init_res, max_time;
    double *R_A, *y_A, *x_R;
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
                    this->R_A[i] = d;
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
                    this->R_A[i] = d;
                    i++;
                }
                this->R_V[j].push_back(d);
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

public:
    explicit Babai_search_asyn(int n) {
        this->n = n;
        this->A = MatrixXd::Zero(n, n);
        this->R = Eigen::MatrixXd::Zero(n, n);
        this->R_V = vector<vector<double>>();
        this->R_V.resize(n * n);
        this->R_A = (double *) calloc(n * n, sizeof(double));
        this->x_R = (double *) calloc(n, sizeof(double));
        this->y_A = (double *) calloc(n, sizeof(double));
        this->x_t = VectorXd::Zero(n);
        this->y = VectorXd::Zero(n);
        this->max_time = INFINITY;
        this->init_res = INFINITY;
    }

    ~Babai_search_asyn() {
        free(R_A);
        free(x_R);
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
                "../../data/R_" + to_string(n) + ".csv";

        if (read_r) {
            read_from_file(file_name);

        } else {
            this->A = MatrixXd::Zero(n, n).unaryExpr([&](double dummy) { return norm_dis(gen); });
            this->R = A.householderQr().matrixQR().triangularView<Eigen::Upper>();
            int index = 0;

            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    this->R_V[i].push_back(R(i, j));
                    temp.push_back(R(i, j));
                    if (this->R(i, j) != 0) {
                        this->R_A[index] = this->R(i, j);
                        index++;
                    }
                }
            }

        }


        if (read_xy) {
            read_x_y();
            this->init_res = (y - R * x_t).norm();
        } else {
            this->x_t = VectorXd::Zero(n).unaryExpr([&](int dummy) { return static_cast<double>(int_dis(gen)); });
            this->y = R * x_t + noise * VectorXd::Zero(n).unaryExpr([&](double dummy) { return norm_dis(gen); });
            this->init_res = (y - R * x_t).norm();
            VectorXd::Map(&y_A[0], n) = y;
        }
    }

    void search_omp(int n_proc, int nswp, int init_value) {

        double sum = 0, dist = 0;
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

        Eigen::Map<Eigen::VectorXd> x_result(z_B, n);
        double res = (y - R * x_result).norm();

        printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", n_proc, num_iter, res, end_time);
        free(z_B);
        free(z_B_p);
        free(update);

    }

    void search_vec(int init_value) {

        vector<double> z_B(n, init_value);

        double sum = 0;
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
        z_B[n - 1] = round(y_A[n - 1] / R_V[n - 1][n - 1]);

        for (int i = 1; i < n; i++) {
            int k = n - i - 1;
            for (int col = n - i; col < n; col++) {
                sum += R_V[k][col] * z_B[col];
            }
            z_B[k] = round((y_A[k] - sum) / R_V[k][k]);
            sum = 0;
        }
        double time = omp_get_wtime() - start;

        Eigen::Map<Eigen::VectorXd> x_result(z_B.data(), n);
        double res = (y - R * x_result).norm();
        this->max_time = time;
        printf("Res = %.5f, init_res = %.5f %f seconds\n", res, init_res, max_time);
    }
};