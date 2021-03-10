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

using namespace std;
using namespace Eigen;

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
                "data/R_A_" + to_string(n) + ".csv";
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
        string file_name = "data/R_" + to_string(n) + ".csv";
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
                "data/y_" + to_string(n) + ".csv";
        string fx =
                "data/x_" + to_string(n) + ".csv";
        string fxR =
                "data/x_R_" + to_string(n) + ".csv";
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
                "data/y_" + to_string(n) + ".csv";
        ofstream file2(fy);
        if (file2.is_open()) {
            file2 << y.format(CSVFormat);
            file2.close();
        }

        string fx =
                "data/x_" + to_string(n) + ".csv";
        ofstream file3(fx);
        if (file3.is_open()) {
            file3 << x_t.format(CSVFormat);
            file3.close();
        }
    }

    void write_R_A() const {
        string fR = "data/R_A_" + to_string(n) + ".csv";
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
        string file_name = "data/R_" + to_string(n) + ".csv";
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

    vector<double> search_vec() {

        vector<double> z_B(n, 0);
        double sum = 0, res = 0;
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
        return z_B;
    }
};