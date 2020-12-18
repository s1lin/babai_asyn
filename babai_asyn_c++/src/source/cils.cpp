#include <cstring>
#include <algorithm>

#include "../include/cils.h"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
#include <chrono>

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

#define VERBOSE_LEVEL = 1;

using namespace std;
namespace cils {

    template<typename scalar, typename index, bool is_read, index n>
    void cils<scalar, index, is_read, n>::read_nc(string filename) {

        index ncid, varid, retval;
        if ((retval = nc_open(filename.c_str(), NC_NOWRITE, &ncid))) ERR(retval);

        /* Get the varid of the data variable, based on its name. */
        if ((retval = nc_inq_varid(ncid, "x_t", &varid))) ERR(retval);

        /* Read the data. */
        if ((retval = nc_get_var_int(ncid, varid, &x_t.data()[0]))) ERR(retval);

        /* Get the varid of the data variable, based on its name. */
        if ((retval = nc_inq_varid(ncid, "y", &varid))) ERR(retval);

        /* Read the data. */
        if ((retval = nc_get_var_double(ncid, varid, &y_A->x[0]))) ERR(retval);

        /* Get the varid of the data variable, based on its name. */
        if ((retval = nc_inq_varid(ncid, "x_R", &varid))) ERR(retval);

        /* Read the data. */
        if ((retval = nc_get_var_int(ncid, varid, &x_R.data()[0]))) ERR(retval);

        /* Get the varid of the data variable, based on its name. */
        if ((retval = nc_inq_varid(ncid, "R_A", &varid))) ERR(retval);

        /* Read the data. */
        if ((retval = nc_get_var_double(ncid, varid, &R_A->x[0]))) ERR(retval);
    }

    template<typename scalar, typename index, bool is_read, index n>
    void cils<scalar, index, is_read, n>::read_csv(bool is_qr) {
        string suffix = to_string(n) + "_" + to_string(snr) + "_" + to_string(qam);
        string fy = is_qr ? "../../data/y_" + suffix + ".csv" : "../../data/new_y_" + suffix + ".csv";
        string fx = is_qr ? "../../data/x_" + suffix + ".csv" : "../../data/new_x_" + suffix + ".csv";
        string fR = is_qr ? "../../data/R_A_" + suffix + ".csv" : "../../data/new_R_A_" + suffix + ".csv";
        string fxR = is_qr ? "../../data/x_R_" + suffix + ".csv" : "../../data/new_x_R_" + suffix + ".csv";

        index i = 0;
        ifstream f(fR), f1(fy), f2(fx), f3(fxR);
        string row_string, entry;
        while (getline(f, row_string)) {
            scalar d = stod(row_string);
            this->R_A->x[i] = d;
            i++;
        }
        this->R_A->size = i;
        f.close();

        i = 0;
        while (getline(f1, row_string)) {
            scalar d = stod(row_string);
            this->y_A->x[i] = d;
            i++;
        }
        f1.close();

        i = 0;
        while (getline(f2, row_string)) {
            scalar d = stoi(row_string);
            this->x_t[i] = d;
            i++;
        }
        f2.close();

        i = 0;
        while (getline(f3, row_string)) {
            scalar d = stoi(row_string);
            this->x_R[i] = d;
            i++;
        }
        f3.close();
    }

    template<typename scalar, typename index, bool is_read, index n>
    void cils<scalar, index, is_read, n>::init(bool is_qr, bool is_nc) {
        scalar start = omp_get_wtime();
        if (is_read) {
            string filename = "../../data/new" + to_string(n) + "_" + to_string(snr) + "_" + to_string(qam) + ".nc";
            if (is_qr) {
                filename = "../../data/" + to_string(n) + "_" + to_string(snr) + "_" + to_string(qam) + ".nc";
            }
            if (is_nc) read_nc(filename);
            else read_csv(is_qr);

            this->init_res = find_residual<scalar, index, n>(R_A, y_A, &x_t);
        }
        printf("init_res: %.5f, sigma: %.5f\n", this->init_res, this->sigma);
        scalar end_time = omp_get_wtime() - start;
        printf("Finish Init, time: %.5f seconds\n", end_time);
//        } else {
//            std::random_device rd;
//            std::mt19937 gen(rd());
//            //mean:0, std:sqrt(1/2). same as matlab.
//            std::normal_distribution<scalar> A_norm_dis(0, sqrt(0.5)), v_norm_dis(0, this->sigma);
//            //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
//            std::uniform_int_distribution<index> int_dis(0, pow(2, qam) - 1);
//
//            this->A = MatrixXd::Zero(n, n).unaryExpr([&](double dummy) { return A_norm_dis(gen); });
////            this->R = A.householderQr().matrixQR().triangularView<Eigen::Upper>();
//            this->Q = A.householderQr().householderQ();
//            this->x_tV = VectorXd::Zero(n).unaryExpr([&](int dummy) { return static_cast<double>(int_dis(gen)); });
//            this->y = R * x_tV +
//                      Q.transpose() * VectorXd::Zero(n).unaryExpr([&](double dummy) { return v_norm_dis(gen); });
//            this->init_res = (y - R * x_tV).norm();
//
//            VectorXd::Map(&y_A->x[0], n) = y;
//            index l = 0;
//            for (index i = 0; i < n; i++) {
//                for (index j = 0; j < n; j++) {
//                    if (this->R(i, j) != 0) {
//                        this->R_A->x[l] = this->R(i, j);
//                        l++;
//                    }
//                }
//                x_t[i] = round(x_tV[i]);
//            }
//            this->x_R = *cils_back_solve(&x_R).x;

    }

}
