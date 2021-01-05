#include <cstring>
#include <algorithm>
//#include <lapack.h>

#include "../include/cils.h"

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}


using namespace std;
using namespace cils::program_def;


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
        string fy = is_qr ? prefix + "data/y_" + suffix + ".csv" : prefix + "data/new_y_" + suffix + ".csv";
        string fx = is_qr ? prefix + "data/x_" + suffix + ".csv" : prefix + "data/new_x_" + suffix + ".csv";
        string fR = is_qr ? prefix + "data/R_A_" + suffix + ".csv" : prefix + "data/new_R_A_" + suffix + ".csv";
        string fxR = is_qr ? prefix + "data/x_R_" + suffix + ".csv" : prefix + "data/new_x_R_" + suffix + ".csv";

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
            string filename = prefix + "data/new" + suffix + ".nc";
            if (is_qr) {
                filename = prefix + "data/" + suffix + ".nc";
            }
            if (!is_nc) read_csv(is_qr);
            else read_nc(filename);

            this->init_res = find_residual<scalar, index, n>(R_A, y_A, &x_t);
        } else {
            std::random_device rd;
            std::mt19937 gen(rd());

            //mean:0, std:sqrt(1/2). same as matlab.
            std::normal_distribution<scalar> A_norm_dis(0, sqrt(0.5)), v_norm_dis(0, this->sigma);
            //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
            std::uniform_int_distribution<index> int_dis(0, pow(2, qam) - 1);
            this->A = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
            this->A->x = new scalar[n * n]();
            this->A->size = n * n;

            this->v_A = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
            this->v_A->x = new scalar[n]();
            this->v_A->size = n;

            this->R = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
            this->R->x = new scalar[n * n]();
            this->R->size = n * n;

            this->Q = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
            this->Q->x = new scalar[n * n]();
            this->Q->size = n * n;


#pragma omp parallel for
            for (index i = 0; i < n * n; i++) {
                this->A->x[i] = A_norm_dis(gen);
                if (i < n) {
                    this->x_t[i] = int_dis(gen);
                    this->v_A->x[i] = v_norm_dis(gen);
                }
            }
        }
        printf("init_res: %.5f, sigma: %.5f\n", this->init_res, this->sigma);
        scalar end_time = omp_get_wtime() - start;
        printf("Finish Init, time: %.5f seconds\n", end_time);

    }


}
