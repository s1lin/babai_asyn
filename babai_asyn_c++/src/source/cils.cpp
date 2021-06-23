#include <cstring>
#include <algorithm>

#include "../include/cils.h"

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}


using namespace std;
using namespace cils::program_def;


namespace cils {

//    template<typename scalar, typename index, index n>
//    void cils<scalar, index, n>::read_nc(string filename) {
//
//        cout << filename;
//        index ncid, varid, retval;
//        if ((retval = nc_open(filename.c_str(), NC_NOWRITE, &ncid))) ERR(retval);
//
//        /* Get the varid of the data variable, based on its name. */
//        if ((retval = nc_inq_varid(ncid, "x_t", &varid))) ERR(retval);
//
//        /* Read the data. */
//        if ((retval = nc_get_var_double(ncid, varid, &x_t.data()[0]))) ERR(retval);
//
//        /* Get the varid of the data variable, based on its name. */
//        if ((retval = nc_inq_varid(ncid, "y", &varid))) ERR(retval);
//
//        /* Read the data. */
//        if ((retval = nc_get_var_double(ncid, varid, &y_A[0]))) ERR(retval);
//
//        /* Get the varid of the data variable, based on its name. */
//        if ((retval = nc_inq_varid(ncid, "x_R", &varid))) ERR(retval);
//
//        /* Read the data. */
//        if ((retval = nc_get_var_double(ncid, varid, &x_R.data()[0]))) ERR(retval);
//
//        /* Get the varid of the data variable, based on its name. */
//        if ((retval = nc_inq_varid(ncid, "R_A", &varid))) ERR(retval);
//
//        /* Read the data. */
//        if ((retval = nc_get_var_double(ncid, varid, &R_A[0]))) ERR(retval);
//        this->R = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
//        this->R = new scalar[n * n]();
//        this->R->size = n * n;
//
//        for (index i = 0; i < n; i++) {
//            for (index j = 0; j < n; j++) {
//                if (j >= i) {
//                    index row = (n * i) - ((i * (i + 1)) / 2);
//                    R[j * n + i] = R_A[row + j];
//                }
//            }
//        }
//    }

//    template<typename scalar, typename index, index n>
//    void cils<scalar, index, n>::read_csv() {
//        string fy = prefix + "data/y_" + suffix + ".csv";
//        string fx = prefix + "data/x_" + suffix + ".csv";
//        string fR = prefix + "data/R_A_" + suffix + ".csv";
//        string fxR = prefix + "data/x_R_" + suffix + ".csv";
//
//        index i = 0;
//        ifstream f(fR), f1(fy), f2(fx), f3(fxR);
//        string row_string, entry;
//        while (getline(f, row_string)) {
//            scalar d = stod(row_string);
//            this->R_A[i] = d;
//            i++;
//        }
//        this->R_A->size = i;
//        f.close();
//
//        i = 0;
//        while (getline(f1, row_string)) {
//            scalar d = stod(row_string);
//            this->y_A[i] = d;
//            i++;
//        }
//        f1.close();
//
//        i = 0;
//        while (getline(f2, row_string)) {
//            scalar d = stoi(row_string);
//            this->x_t[i] = d;
//            i++;
//        }
//        f2.close();
//
//        i = 0;
//        while (getline(f3, row_string)) {
//            scalar d = stoi(row_string);
//            this->x_R[i] = d;
//            i++;
//        }
//        f3.close();
//    }


    template<typename scalar, typename index, index n>
    void cils<scalar, index, n>::init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        //mean:0, std:sqrt(1/2). same as matlab.
        std::normal_distribution<scalar> A_norm_dis(0, sqrt(0.5)), v_norm_dis(0, this->sigma);
        //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
        std::uniform_int_distribution<index> int_dis(-pow(2, qam - 1), pow(2, qam - 1) - 1);

        for (index i = 0; i < n / 2; i++) {
            for (index j = 0; j < n / 2; j++) {
                A[j + i * n] = 2 * A_norm_dis(gen);
                A[j + n / 2 + i * n] = -2 * A_norm_dis(gen);
                A[j + n / 2 + (i + n / 2) * n] = A[j + i * n];
                A[j + (i + n / 2) * n] = -A[j + n / 2 + i * n];
            }
            this->x_t[i] = (pow(2, qam) + 2 * int_dis(gen)) / 2;
            this->v_a[i] = v_norm_dis(gen);
            this->x_t[i + n / 2] = (pow(2, qam) + 2 * int_dis(gen)) / 2;
            this->v_a[i + n / 2] = v_norm_dis(gen);
        }
        scalar sum = 0;
        for (index i = 0; i < n; i++) {
            sum = 0;
            for (index j = 0; j < n; j++) {
                sum += this->A[i * n + j] * this->x_t[j];
            }
            y_a[i] = sum + v_a[i];
            Z[i * n + i] = 1;
        }
    }

    template<typename scalar, typename index, index n>
    void cils<scalar, index, n>::init_y() {
        scalar rx = 0, qv = 0;
        index ri, rai;
//        for (index i = 0; i < n; i++) {
//            rx = qv = 0;
//            for (index j = 0; j < n; j++) {
//                if (i <= j) { //For some reason the QR stored in Transpose way?
//                    ri = i * n + j;
//                    rai = ri - (i * (i + 1)) / 2;
//                    this->R_A[rai] = this->R[ri];
//                    rx += this->R[ri] * this->x_t[j];
//                }
//                qv += this->Q[j * n + i] * this->v_a[j]; //Transpose Q
//            }
////            cout << endl;
//            this->y_A[i] = rx + qv;
//        }
//Not use python:
        scalar R_T[n * n] = {};
//        for (index i = 0; i < n; i++) {
//            for (index j = 0; j < n; j++) {
////                swap(this->R[i * n + j], this->R[j * n + i]);
//                cout << this->R[j * n + i] << " ";
//            }
//            cout << endl;
//        }

//        scalar tmp;
//        for (index i = 0; i < n; i++) {
//            for (index j = 0; j < n; j++) {
//                R_T[i * n + j] = this->R[j * n + i];
//            }
//        }
//
//        for (index i = 0; i < n; i++) {
//            for (index j = 0; j < n; j++) {
//                this->R[i * n + j] = R_T[i * n + j];
//            }
//        }

        for (index i = 0; i < n; i++) {
            rx = qv = 0;
            for (index j = 0; j < n; j++) {
                if (i <= j) { //For some reason the QR stored in Transpose way?
//                    this->R[i * n + j] = R_T[i * n + j];
//                    this->R_A[(n * i) + j - ((i * (i + 1)) / 2)] = this->R[i * n + j];
//                    rx += this->R[i * n + j] * this->x_t[j];
                    rx += this->R_Q[j * n + i] * this->x_t[j];
                }
                qv += this->Q[j * n + i] * this->v_a[j]; //Transpose Q
            }
            this->y_q[i] = rx + qv;
            this->y_r[i] = rx + qv;
        }
    }


    template<typename scalar, typename index, index n>
    void cils<scalar, index, n>::init_R() {
        scalar R_T[n * n] = {};
//        for (index i = 0; i < n; i++) {
//            for (index j = 0; j < n; j++) {
////                swap(this->R[i * n + j], this->R[j * n + i]);
//                cout << this->R[j * n + i] << " ";
//            }
//            cout << endl;
//        }

        for (index i = 0; i < n; i++) {
            for (index j = 0; j < n; j++) {
                R_T[i * n + j] = this->R_Q[j * n + i];
                R_R[i * n + j] = R_T[j * n + i];
            }
        }
        for (index i = 0; i < n; i++) {
            for (index j = 0; j < n; j++) { //TO i
                this->R_Q[i * n + j] = R_T[i * n + j];
                if (j >= i)
                    this->R_A[(n * i) + j - ((i * (i + 1)) / 2)] = this->R_Q[i * n + j];
            }
        }

//        for (index i = 0; i < n; i++) {
//            for (index j = i; j < n; j++) {
////                swap(this->R[i * n + j], this->R[j * n + i]);
//                cout << this->R_A[(n * i) + j - ((i * (i + 1)) / 2)]  << " ";
//            }
//            cout << endl;
//        }
    }

}