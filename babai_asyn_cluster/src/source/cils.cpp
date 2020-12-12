#include <cstring>
#include <algorithm>

#include "../include/cils.h"
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

#define VERBOSE_LEVEL = 1;

using namespace std;


namespace cils {

    template<typename scalar, typename index, bool is_read, index n>
    cils<scalar, index, is_read, n>::cils(index k, index snr) {
        this->R_A = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
        this->y_A = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
        this->R_A->x = (scalar *) calloc(n * (n + 1) / 2, sizeof(scalar));
        this->y_A->x = (scalar *) calloc(n, sizeof(scalar));

        this->x_R = vector<index>(n, 0);
        this->x_t = vector<index>(n, 0);

        this->init_res = INFINITY;
        this->qam = k;
        this->snr = snr;
        this->sigma = (scalar) sqrt(((pow(4, k) - 1) * log2(n)) / (6 * pow(10, ((scalar) snr / 20.0))));
        this->R_A->size = n * (n + 1) / 2;
        this->y_A->size = n;
    }

    template<typename scalar, typename index, bool is_read, index n>
    void cils<scalar, index, is_read, n>::read(bool is_qr) {
//        string filename = "../../data/new" + to_string(n) + "_" + to_string(snr) + "_" + to_string(qam) + ".nc";
        string filename = "data/" + to_string(n) + "_" + to_string(snr) + "_" + to_string(qam) + ".nc";

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
        }
        else {
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
        }
        else {
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
            this->R = A.householderQr().matrixQR().triangularView<Eigen::Upper>();
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
            this->x_R = cils_back_solve(&x_R).x;
        }
        printf("init_res: %.5f, sigma: %.5f\n", this->init_res, this->sigma);
        scalar end_time = omp_get_wtime() - start;
        printf("Finish Init, time: %.5f seconds\n", end_time);

    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    cils<scalar, index, is_read, n>::cils_babai_search_omp(const index n_proc, const index nswp,
                                                           vector<index> *z_B) {

        index count = 0, num_iter = 0, x_c = 0, chunk = std::log2(n);
        vector<index> z_B_p(n, 0);
        vector<index> update(n, 0);

        scalar start = omp_get_wtime();
        z_B->at(n - 1) = round(y_A->x[n - 1] / R_A->x[((n - 1) * n) / 2 + n - 1]);
#pragma omp parallel default(shared) num_threads(n_proc) private(count, x_c) shared(update)
        {
            for (index j = 0; j < nswp; j++) {//&& count < 16
                count = 0;
#pragma omp for nowait
                for (index i = 0; i < n; i++) {
//                for (index m = 0; m < n_proc; m++) {
//                    for (index i = m; i < n; i += n_proc) {
                    x_c = babai_solve_omp(i, z_B);
//                    index x_p = z_B[n - 1 - i];
                    z_B->at(n - 1 - i) = x_c;

//                    if (x_c != x_p) {
//                        update[n - 1 - i] = 0;
//                        z_B_p[n - 1 - i] = x_c;
//                    } else {
//                        update[n - 1 - i] = 1;
//                    }
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
        //cout << num_iter << endl;
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {*z_B, run_time, 0, num_iter};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    cils<scalar, index, is_read, n>::cils_babai_search_serial(vector<index> *z_B) {
        scalar sum = 0;
        scalar start = omp_get_wtime();

        z_B->at(n - 1) = round(y_A->x[n - 1] / R_A->x[((n - 1) * n) / 2 + n - 1]);
        for (index i = 1; i < n; i++) {
            index k = n - i - 1;
            for (index col = n - i; col < n; col++) {
                sum += R_A->x[k * n - (k * (n - i)) / 2 + col] * z_B->at(col);
            }
            z_B->at(k) = round(
                    (y_A->x[n - 1 - i] - sum) / R_A->x[k * n - (k * (n - i)) / 2 + n - 1 - i]);
            sum = 0;
        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {*z_B, run_time, 0, 0};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    cils<scalar, index, is_read, n>::cils_back_solve(vector<index> *z_B) {
        scalar sum = 0;
        scalar start = omp_get_wtime();
        vector<scalar> z_B_tmp(n, 0);
        z_B_tmp[n - 1] = y_A->x[n - 1] / R_A->x[((n - 1) * n) / 2 + n - 1];

        for (index i = 1; i < n; i++) {
            index k = n - i - 1;
            for (index col = n - i; col < n; col++) {
                sum += R_A->x[k * n - (k * (n - i)) / 2 + col] * z_B->at(col);
            }
            z_B_tmp[k] = (y_A->x[n - 1 - i] - sum) / R_A->x[k * n - (k * (n - i)) / 2 + n - 1 - i];
            sum = 0;
        }
        for (index i = 0; i < n; i++) {
            z_B->at(i) = round(z_B_tmp[i]);
        }

        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {*z_B, run_time, 0, 0};
        return reT;
    }


    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    cils<scalar, index, is_read, n>::cils_block_search_serial(vector<index> *z_B,
                                                              vector<index> *d) {

        index ds = d->size();
        //special cases:
        if (ds == 1) {
            if (d->at(0) == 1) {
                return {vector<index>(round(y_A->x[0] / R_A->x[0]), 0), 0, 0, 0};
            } else {
                vector<scalar> R_B = find_block_Rii(R_A, 0, n, 0, n, n);
                vector<scalar> y_B = find_block_x(y_A, 0, n);
                return {ils_search(&R_B, &y_B), 0, 0, 0};
            }
        } else if (ds == n) {
            //Find the Babai point
            return cils_babai_search_serial(z_B);
        }

        //last block:
        index q = d->at(ds - 1);

        scalar start = omp_get_wtime();
        //the last block
        vector<scalar> R_ii = find_block_Rii<scalar, index>(R_A, n - q, n, n - q, n, n);
        vector<scalar> y_b_s = find_block_x<scalar, index>(y_A, n - q, n);
        vector<index> x_b_s = ils_search(&R_ii, &y_b_s);
        for (index l = n - q; l < n; l++) {
            z_B->at(l) = x_b_s[l - n + q];
        }
//        display_scalarType(y_b_s);
//        cout<<  n - q<<","<<n <<endl;
//        display_scalarType(x_b_s);

        //Therefore, skip the last block, start from the second-last block till the first block.
        for (index i = 0; i < ds - 1; i++) {
            q = ds - 2 - i;
            //accumulate the block size
            y_b_s = find_block_x<scalar, index>(y_A, n - d->at(q), n - d->at(q + 1));
            x_b_s = find_block_x<scalar, index>(z_B, n - d->at(q + 1), n);
            y_b_s = block_residual_vector(R_A, &x_b_s, &y_b_s, n - d->at(q), n - d->at(q + 1),
                                          n - d->at(q + 1), n);
//            display_scalarType(y_b_s);
            R_ii = find_block_Rii<scalar, index>(R_A, n - d->at(q), n - d->at(q + 1), n - d->at(q),
                                                 n - d->at(q + 1), n);
//            cout<< n - d->at(q)<<","<<n - d->at(q + 1)<<endl;
            x_b_s = ils_search(&R_ii, &y_b_s);
//            display_array(x_b_s->at, x_b_s->size());

            for (index l = n - d->at(q); l < n - d->at(q + 1); l++) {
                z_B->at(l) = x_b_s[l - n + d->at(q)];
            }


        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {*z_B, run_time, 0, 0};
        return reT;
    }

}
