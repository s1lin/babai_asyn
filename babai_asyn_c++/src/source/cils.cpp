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
        cout << filename;
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
    void cils<scalar, index, is_read, n>::read_csv() {
        string fy = prefix + "data/y_" + suffix + ".csv";
        string fx = prefix + "data/x_" + suffix + ".csv";
        string fR = prefix + "data/R_A_" + suffix + ".csv";
        string fxR = prefix + "data/x_R_" + suffix + ".csv";

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
    void cils<scalar, index, is_read, n>::init(bool is_serial, bool is_nc) {
        scalar start = omp_get_wtime();
        if (is_read) {
            string filename = prefix + "data/" + suffix + ".nc";
            if (!is_nc) read_csv();
            else read_nc(filename);

            this->init_res = find_residual<scalar, index, n>(R_A, y_A, &x_t);
        } else {

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

            if (is_serial) {
                std::random_device rd;
                std::mt19937 gen(rd());
                //mean:0, std:sqrt(1/2). same as matlab.
                std::normal_distribution<scalar> A_norm_dis(0, sqrt(0.5)), v_norm_dis(0, this->sigma);
                //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
                std::uniform_int_distribution<index> int_dis(-pow(2, qam - 1), pow(2, qam - 1) - 1);
                scalar rx = 0, qv = 0;
                index counter = 0;

                for (index i = 0; i < n / 2; i++) {
                    for (index j = 0; j < n / 2; j++) {
                        A->x[j + i * n] = A_norm_dis(gen);
                        A->x[j + n / 2 + i * n] = A_norm_dis(gen);
                        A->x[j + n / 2 + (i + n / 2) * n] = A->x[j + i * n];
                        A->x[j + (i + n / 2) * n] = -A->x[j + n / 2 + i * n];
                    }
                    this->x_t[i] = (pow(2, qam) + 2 * int_dis(gen)) / 2;
                    this->v_A->x[i] = v_norm_dis(gen);
                    this->x_t[i + n / 2] = (pow(2, qam) + 2 * int_dis(gen)) / 2;
                    this->v_A->x[i + n / 2] = v_norm_dis(gen);
                }

                //Run QR decomposition
                qr_decomposition<scalar, index, n>(A, Q, R);

                for (index i = 0; i < n; i++) {
                    rx = qv = 0;
                    for (index j = 0; j < n; j++) {
                        if (i <= j) { //For some reason the QR stored in Transpose way?
                            this->R_A->x[counter] = this->R->x[j * n + i];
                            rx += this->R_A->x[counter] * this->x_t[j];
                            counter++;
                        }
                        qv += this->Q->x[i * n + j] * this->v_A->x[j]; //Transpose Q
                    }
                    this->y_A->x[i] = rx + qv;
                }
            }
            else {
                init_omp(12);
            }
        }
        qr_validation<scalar, index, n>(A, Q, R, 0, 1);
        this->init_res = find_residual<scalar, index, n>(R_A, y_A, &x_t);
        printf("init_res: %.5f, sigma: %.5f\n", this->init_res, this->sigma);
        scalar end_time = omp_get_wtime() - start;
        printf("Finish Init, time: %.5f seconds\n", end_time);
    }

    template<typename scalar, typename index, bool is_read, index n>
    void cils<scalar, index, is_read, n>::init_omp(const index n_proc) {

        std::random_device rd;
        std::mt19937 gen(rd());
        //mean:0, std:sqrt(1/2). same as matlab.
        std::normal_distribution<scalar> A_norm_dis(0, sqrt(0.5)), v_norm_dis(0, this->sigma);
        //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
        std::uniform_int_distribution<index> int_dis(-pow(2, qam - 1), pow(2, qam - 1) - 1);

        index i, j, k, m, counter = 0;
        scalar error, d, time, s_time, r_sum = 0, rx, qv;
        omp_lock_t *lock;
        auto t = new scalar[n * n]();
//        lock = (omp_lock_t *) malloc(n * sizeof(omp_lock_t));

#pragma omp parallel default(shared) num_threads(n_proc) private(i, j, k, r_sum, rx, qv, counter)
        {
#pragma omp for schedule(dynamic, 1)
            for (i = 0; i < n / 2; i++) {
                for (j = 0; j < n / 2; j++) {
                    t[j + i * n] = A->x[j + i * n] = A_norm_dis(gen);
                    t[j + n / 2 + i * n] = A->x[j + n / 2 + i * n] = A_norm_dis(gen);
                    t[j + n / 2 + (i + n / 2) * n] = A->x[j + n / 2 + (i + n / 2) * n] = A->x[j + i * n];
                    t[j + (i + n / 2) * n] = A->x[j + (i + n / 2) * n] = -A->x[j + n / 2 + i * n];
                }
                x_t[i] = (pow(2, qam) + 2 * int_dis(gen)) / 2;
                v_A->x[i] = v_norm_dis(gen);
                x_t[i + n / 2] = (pow(2, qam) + 2 * int_dis(gen)) / 2;
                v_A->x[i + n / 2] = v_norm_dis(gen);
//                omp_init_lock(&lock[i]);
//                omp_init_lock(&lock[i + n / 2]);
            }

            //Run QR decomposition
            //First column of ( Q[][0] )
            //Thread 0 calculates the 1st column
            //and unsets the 1st lock.
            r_sum = 0;
            if (omp_get_thread_num() == 0) {
                // Calculation of ||A||
                for (i = 0; i < n; i++) {
                    r_sum = r_sum + t[0 * n + i] * t[0 * n + i];
                }
                R->x[0 * n + 0] = sqrt(r_sum);
                for (i = 0; i < n; i++) {
                    Q->x[0 * n + i] = t[0 * n + i] / R->x[0 * n + 0];
                }
//                omp_unset_lock(&lock[0]);
            }

            for (k = 1; k < n; k++) {
                //Check if Q[][i-1] (the previous column) is computed.
//                omp_set_lock(&lock[k - 1]);
//                omp_unset_lock(&lock[k - 1]);

#pragma omp for schedule(dynamic) nowait
                for (j = 0; j < n; j++) {
                    if (j >= k) {
                        R->x[(k - 1) * n + j] = 0;
                        for (i = 0; i < n; i++) {
                            R->x[j * n + (k - 1)] += Q->x[(k - 1) * n + i] * t[j * n + i];
                        }
                        for (i = 0; i < n; i++) {
                            t[j * n + i] = t[j * n + i] - R->x[j * n + (k - 1)] * Q->x[(k - 1) * n + i];
                        }

                        //Only one thread calculates the norm ||A||
                        //and unsets the lock for the next column.
                        if (j == k) {
                            r_sum = 0;
                            for (i = 0; i < n; i++) {
                                r_sum = r_sum + t[k * n + i] * t[k * n + i];
                            }
                            R->x[k * n + k] = sqrt(r_sum);

                            //#pragma omp for schedule(static,1) nowait
                            for (i = 0; i < n; i++) {
                                Q->x[k * n + i] = t[k * n + i] / R->x[k * n + k];
                            }
//                            omp_unset_lock(&lock[k]);
                        }
                    }
                }
            }
        }
        counter = 0;
        for (i = 0; i < n; i++) {
            rx = qv = 0;
            for (j = 0; j < n; j++) {
                if (i <= j) { //For some reason the QR stored in Transpose way?
                    this->R_A->x[counter] = this->R->x[j * n + i];
                    rx += this->R->x[j * n + i] * this->x_t[j];
                    counter++;
                }
                qv += this->Q->x[i * n + j] * this->v_A->x[j]; //Transpose Q
            }
            this->y_A->x[i] = rx + qv;
        }
//        free(lock);
    }
}