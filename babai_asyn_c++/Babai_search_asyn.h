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

namespace babai {

    template<typename scalar, typename index, index n>
    scalar find_residual(const scalar *R, const scalar *y, const scalar *x);

    template<typename scalar, typename index, bool use_eigen, bool is_read, bool is_write, index n>
    class Babai_search_asyn {
    public:
        index size_R_A;
        scalar init_res, noise, *R_A, *y_A, *x_R, *x_tA;
        Eigen::MatrixXd A, R;
        Eigen::VectorXd y, x_t;

    private:
        //Utils:
        void read_from_RA();
        void read_from_R();
        void write_to_file(const string &file_name);
        void read_x_y();
        void write_x_y();
        void write_R_A();

    public:
        Babai_search_asyn(scalar noise);

        ~Babai_search_asyn() {
            free(R_A);
            free(x_R);
            free(y_A);
        }

        void init();

        inline scalar do_solve(const index i, const scalar *z_B) {
            scalar sum = 0;
#pragma omp simd reduction(+ : sum)
            for (index col = n - i; col < n; col++) {
                sum += R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * z_B[col];
            }

            return round((y_A[n - 1 - i] - sum) / R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
        }

        scalar *search_omp(index n_proc, index nswp, index *update, scalar *z_B, scalar *z_B_p);

        vector<scalar> search_vec();
    };



}