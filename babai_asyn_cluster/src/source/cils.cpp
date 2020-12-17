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
    void cils<scalar, index, is_read, n>::read(bool is_qr) {
//        string filename = "../../data/new" + to_string(n) + "_" + to_string(snr) + "_" + to_string(qam) + ".nc";
        string filename = "../../data/" + to_string(n) + "_" + to_string(snr) + "_" + to_string(qam) + ".nc";

        index ncid, varid, retval;
        if ((retval = nc_open(filename.c_str(), NC_NOWRITE, &ncid))) ERR(retval);

        /* Get the varid of the data variable, based on its name. */
        if ((retval = nc_inq_varid(ncid, "x_t", &varid))) ERR(retval);

        /* Read the data. */
        if ((retval = nc_get_var_int(ncid, varid, &x_t.data()[0]))) ERR(retval);

        /* Get the varid of the data variable, based on its name. */
        if (is_qr) {
            if ((retval = nc_inq_varid(ncid, "y", &varid))) ERR(retval);
            if ((retval = nc_get_var_double(ncid, varid, &y_A->x[0]))) ERR(retval);
        } else {
            if ((retval = nc_inq_varid(ncid, "y_LLL", &varid))) ERR(retval);
            if ((retval = nc_get_var_double(ncid, varid, &y_A->x[0]))) ERR(retval);
        }

        /* Get the varid of the data variable, based on its name. */
        if ((retval = nc_inq_varid(ncid, "x_R", &varid))) ERR(retval);

        /* Read the data. */
        if ((retval = nc_get_var_int(ncid, varid, &x_R.data()[0]))) ERR(retval);

        /* Get the varid of the data variable, based on its name. */
        if (is_qr) {
            if ((retval = nc_inq_varid(ncid, "R_A", &varid))) ERR(retval);
            if ((retval = nc_get_var_double(ncid, varid, &R_A->x[0]))) ERR(retval);
        } else {
            if ((retval = nc_inq_varid(ncid, "R_A_LLL", &varid))) ERR(retval);
            if ((retval = nc_get_var_double(ncid, varid, &R_A->x[0]))) ERR(retval);
        }
    }

    template<typename scalar, typename index, bool is_read, index n>
    void cils<scalar, index, is_read, n>::init(bool is_qr) {
        scalar start = omp_get_wtime();
        if (is_read) {
            read(is_qr);
            this->init_res = find_residual<scalar, index, n>(R_A, y_A, &x_t);
        } else {
            std::random_device rd;
            std::mt19937 gen(rd());
            //mean:0, std:sqrt(1/2). same as matlab.
            std::normal_distribution<scalar> A_norm_dis(0, sqrt(0.5)), v_norm_dis(0, this->sigma);
            //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
            std::uniform_int_distribution<index> int_dis(0, pow(2, qam) - 1);

            this->A = MatrixXd::Zero(n, n).unaryExpr([&](double dummy) { return A_norm_dis(gen); });
//            this->R = A.householderQr().matrixQR().triangularView<Eigen::Upper>();
            this->Q = A.householderQr().householderQ();
            this->x_tV = VectorXd::Zero(n).unaryExpr([&](int dummy) { return static_cast<double>(int_dis(gen)); });
            this->y = R * x_tV +
                      Q.transpose() * VectorXd::Zero(n).unaryExpr([&](double dummy) { return v_norm_dis(gen); });
            this->init_res = (y - R * x_tV).norm();

            VectorXd::Map(&y_A->x[0], n) = y;
            index l = 0;
            for (index i = 0; i < n; i++) {
                for (index j = 0; j < n; j++) {
                    if (this->R(i, j) != 0) {
                        this->R_A->x[l] = this->R(i, j);
                        l++;
                    }
                }
                x_t[i] = round(x_tV[i]);
            }
            this->x_R = *cils_back_solve(&x_R).x;
        }
        printf("init_res: %.5f, sigma: %.5f\n", this->init_res, this->sigma);
        scalar end_time = omp_get_wtime() - start;
        printf("Finish Init, time: %.5f seconds\n", end_time);

    }

}
