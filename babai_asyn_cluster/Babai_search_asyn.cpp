#include "Babai_search_asyn.h"
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

using namespace std;
using namespace Eigen;

namespace babai {

    template<typename scalar, typename index>
    scalar find_residual(const index n, const scalar *R, const scalar *y, const scalar *x) {
        scalar res = 0;
        for (index i = 0; i < n; i++) {
            scalar sum = 0;
            for (index j = i; j < n; j++) {
                sum += x[j] * R[(n * i) + j - ((i * (i + 1)) / 2)];
            }
            res += (y[i] - sum) * (y[i] - sum);
        }
        return std::sqrt(res);
    }

    template<typename scalar, typename index, bool use_eigen, bool is_read, bool is_write, index n>
    Babai_search_asyn<scalar, index, use_eigen, is_read, is_write, n>::Babai_search_asyn(scalar noise) {
        if (use_eigen) {
            this->A = MatrixXd::Zero(n, n);
            this->R = Eigen::MatrixXd::Zero(n, n);
            this->x_t = VectorXd::Zero(n);
            this->y = VectorXd::Zero(n);
        }
        this->R_A = (scalar *) calloc(n * n, sizeof(scalar));
        this->x_R = (scalar *) calloc(n, sizeof(scalar));
        this->x_tA = (scalar *) calloc(n, sizeof(scalar));
        this->y_A = (scalar *) calloc(n, sizeof(scalar));

        this->init_res = INFINITY;
        this->noise = noise;
        this->size_R_A = 0;
        this->init();
    }

    template<typename scalar, typename index, bool use_eigen, bool is_read, bool is_write, index n>
    void Babai_search_asyn<scalar, index, use_eigen, is_read, is_write, n>::read_from_RA() {
        string fxR = "data/R_A_" + to_string(n) + ".csv";
        index i = 0;
        ifstream f3(fxR);
        string row_string3, entry3;
        while (getline(f3, row_string3)) {
            stringstream row_stream(row_string3);
            while (getline(row_stream, entry3, ',')) {
                scalar d = stod(entry3);
                this->R_A[i] = d;
                i++;
            }
        }
        this->size_R_A = i;
    }

    template<typename scalar, typename index, bool use_eigen, bool is_read, bool is_write, index n>
    void Babai_search_asyn<scalar, index, use_eigen, is_read, is_write, n>::read_from_R() {
        string file_name = "data/R_" + to_string(n) + ".csv";
        index i = 0, j = 0;
        vector<scalar> temp;
        ifstream f(file_name);
        string row_string, entry;
        while (getline(f, row_string)) {
            stringstream row_stream(row_string);
            while (getline(row_stream, entry, ',')) {
                scalar d = stod(entry);
                if (d != 0) {
                    this->R_A[i] = d;
                    i++;
                }
                if (this->use_eigen)
                    temp.push_back(d);
            }
            j++;
        }
        this->size_R_A = i;
        if (this->use_eigen)
            this->R = Map<Matrix<scalar, Dynamic, Dynamic, RowMajor>>(temp.data(), n, temp.size() / n);
    }

    template<typename scalar, typename index, bool use_eigen, bool is_read, bool is_write, index n>
    void Babai_search_asyn<scalar, index, use_eigen, is_read, is_write, n>::write_to_file(const string &file_name) {
        index i = 0, j = 0;
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
                scalar d = stod(entry);
                if (d != 0) {
                    this->R_A[i] = d;
                    i++;
                }
            }
            j++;
        }
    }

    template<typename scalar, typename index, bool use_eigen, bool is_read, bool is_write, index n>
    void Babai_search_asyn<scalar, index, use_eigen, is_read, is_write, n>::read_x_y() {
        string fy =
                "data/y_" + to_string(n) + ".csv";
        string fx =
                "data/x_" + to_string(n) + ".csv";
        string fxR =
                "data/x_R_" + to_string(n) + ".csv";
        string row_string, entry;
        index i = 0;
        ifstream f1(fy);
        while (getline(f1, row_string)) {
            scalar d = stod(row_string);
            this->y_A[i] = d;
            if (use_eigen)
                this->y(i) = d;
            i++;
        }

        i = 0;
        ifstream f2(fx);
        string row_string2, entry2;
        while (getline(f2, row_string2)) {
            scalar d = stod(row_string2);
            this->x_tA[i] = d;
            if (use_eigen)
                this->x_t(i) = d;
            i++;
        }

        i = 0;
        ifstream f3(fxR);
        string row_string3, entry3;
        while (getline(f3, row_string3)) {
            scalar d = stod(row_string3);
            this->x_R[i] = d;
            i++;
        }
    }

    template<typename scalar, typename index, bool use_eigen, bool is_read, bool is_write, index n>
    void Babai_search_asyn<scalar, index, use_eigen, is_read, is_write, n>::write_x_y() {
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

    template<typename scalar, typename index, bool use_eigen, bool is_read, bool is_write, index n>
    void Babai_search_asyn<scalar, index, use_eigen, is_read, is_write, n>::write_R_A() {
        string fR = "data/R_A_" + to_string(n) + ".csv";
        ofstream file3(fR);
        if (file3.is_open()) {
            for (index i = 0; i < size_R_A; i++)
                file3 << setprecision(15) << R_A[i] << ",";
            file3.close();
        }
    }

    template<typename scalar, typename index, bool use_eigen, bool is_read, bool is_write, index n>
    void Babai_search_asyn<scalar, index, use_eigen, is_read, is_write, n>::init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        //mean:0, std:1. same as matlab.
        std::normal_distribution<scalar> norm_dis(0, 1);
        //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
        std::uniform_int_distribution<index> index_dis(-n, n);

        if (is_read) {
//              read_from_R();
//              write_R_A();
//              compare_R_RA();
            read_from_RA();
            read_x_y();
            if (use_eigen)
                this->init_res = (y - R * x_t).norm();
            else
                this->init_res = find_residual(n, R_A, y_A, x_tA);

            cout << "init_res:" << this->init_res << endl;

        } else {
            assert(!use_eigen && "Error! You have to enable Eigen.");
            this->A = MatrixXd::Zero(n, n).unaryExpr([&](scalar dummy) { return norm_dis(gen); });
//                this->R = A.householderQr().matrixQR().triangularView<Eigen::Upper>();
            index rI = 0;
            for (index i = 0; i < n; i++) {
                for (index j = i; j < n; j++) {
                    if (this->R(i, j) != 0) {
                        this->R_A[rI] = this->R(i, j);
                        rI++;
                    }
                }
            }
            this->x_t = VectorXd::Zero(n).unaryExpr(
                    [&](index dummy) { return static_cast<scalar>(index_dis(gen)); });
            this->y = R * x_t + noise * VectorXd::Zero(n).unaryExpr([&](scalar dummy) { return norm_dis(gen); });
            this->init_res = (y - R * x_t).norm();
            VectorXd::Map(&y_A[0], n) = y;
        }
    }

    template<typename scalar, typename index, bool use_eigen, bool is_read, bool is_write, index n>
    scalar *
    Babai_search_asyn<scalar, index, use_eigen, is_read, is_write, n>::search_omp(const index n_proc, const index nswp,
                                                                                  index *update, scalar *z_B,
                                                                                  scalar *z_B_p) {

        index count = 0, num_iter = 0;
        index chunk = std::log2(n);
        scalar res = 0;


        z_B[n - 1] = round(y_A[n - 1] / R_A[((n - 1) * n) / 2 + n - 1]);
#pragma omp parallel default(shared) num_threads(n_proc) private(count) shared(update)
        {
            for (index j = 0; j < nswp; j++) {//&& count < 16
                count = 0;
#pragma omp for schedule(dynamic) nowait
                for (index i = 0; i < n; i++) {
                    int x_c = do_solve(i, z_B);
//                    int x_p = z_B[n - 1 - i];
                    z_B[n - 1 - i] = x_c;

//                    if (x_c != x_p) {
//                        update[n - 1 - i] = 0;
//                        z_B_p[n - 1 - i] = x_c;
//                    } else {
//                        update[n - 1 - i] = 1;
//                    }
                }
//#pragma omp simd reduction(+ : count)
//                for (index col = 0; col < 32; col++) {
//                    count += update[col];
//                }
                num_iter = j;
//
            }
        }
        cout << num_iter << endl;

        return z_B;
    }

    template<typename scalar, typename index, bool use_eigen, bool is_read, bool is_write, index n>
    vector<scalar> Babai_search_asyn<scalar, index, use_eigen, is_read, is_write, n>::search_vec() {

        vector<scalar> z_B(n, 0);
        scalar sum = 0;

        z_B[n - 1] = round(y_A[n - 1] / R_A[((n - 1) * n) / 2 + n - 1]);
        for (index i = 1; i < n; i++) {
            index k = n - i - 1;
            for (index col = n - i; col < n; col++) {
                sum += R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * z_B[col];
            }
            z_B[k] = round(
                    (y_A[n - 1 - i] - sum) / R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
            sum = 0;
        }
        return z_B;
    }
}

int main() {
    cout << omp_get_max_threads() << endl;
    const int n = 32648;
    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    babai::Babai_search_asyn<double, int, false, true, false, n> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    start = omp_get_wtime();
    vector<double> z_BV = bsa.search_vec();
    end_time = omp_get_wtime() - start;
    double res = babai::find_residual(n, bsa.R_A, bsa.y_A, z_BV.data());
    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

    for (int proc = 16; proc >= 2; proc /= 2) {
        auto *z_B = (double *) malloc(n * sizeof(double));
        auto *z_B_p = (double *) malloc(n * sizeof(double));
        auto *update = (int *) malloc(n * sizeof(int));

        for (int i = 0; i < n; i++) {
            z_B[i] = 0;
            z_B_p[i] = 0;
            update[i] = 0;
        }
        start = omp_get_wtime();
        z_B = bsa.search_omp(proc, 10, update, z_B, z_B_p);
        end_time = omp_get_wtime() - start;
        res = babai::find_residual(n, bsa.R_A, bsa.y_A, z_B);
        printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", proc, 0, res, end_time);
        free(z_B);
        free(z_B_p);
        free(update);
    }

    return 0;
}
