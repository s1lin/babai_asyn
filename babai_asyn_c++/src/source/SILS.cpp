#include <cstring>
#include <algorithm>

#include "../include/SILS.h"

#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

#define VERBOSE_LEVEL = 1;

using namespace std;

namespace sils {

    template<typename scalar, typename index, bool is_read, index n>
    SILS<scalar, index, is_read, n>::SILS(index qam, index snr) {
        this->R_A = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
        this->y_A = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
        this->R_A->x = (scalar *) calloc(n * (n + 1) / 2, sizeof(scalar));
        this->y_A->x = (scalar *) calloc(n, sizeof(scalar));

        this->x_R = vector<index>(n, 0);
        this->x_t = vector<index>(n, 0);

        this->init_res = INFINITY;
        this->qam = qam;
        this->snr = snr;

        this->R_A->size = n * (n + 1) / 2;
        this->y_A->size = n;

        this->init();
    }

    template<typename scalar, typename index, bool is_read, index n>
    void SILS<scalar, index, is_read, n>::read() {
        string filename = "../../data/" + to_string(n) + "_" + to_string(snr) + "_" + to_string(qam) + ".nc";
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
    void SILS<scalar, index, is_read, n>::init() {

        if (is_read) {
            read();
            this->init_res = sils::find_residual<scalar, index, n>(R_A, y_A, &x_t);
            cout << "init_res:" << this->init_res << endl;
        } else {
            std::random_device rd;
            std::mt19937 gen(rd());
            //mean:0, std:1. same as matlab.
            std::normal_distribution<scalar> norm_dis(0, 1 / 4);
            //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
            std::uniform_int_distribution<index> index_dis(-n);
        }
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    SILS<scalar, index, is_read, n>::sils_babai_search_omp(const index n_proc, const index nswp,
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
    SILS<scalar, index, is_read, n>::sils_babai_search_serial(vector<index> *z_B) {
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
    SILS<scalar, index, is_read, n>::sils_block_search_serial(vector<index> *z_B,
                                                              vector<index> *d) {

        index ds = d->size();
        //special cases:
        if (ds == 1) {
            if (d->at(0) == 1) {
                return {vector<index>(round(y_A->x[0] / R_A->x[0]), 0), 0, 0, 0};
            } else {
                vector<scalar> R_B = sils::find_block_Rii(R_A, 0, n, 0, n, n);
                vector<scalar> y_B = sils::find_block_x(y_A, 0, n);
                return {ils_search(&R_B, &y_B), 0, 0, 0};
            }
        } else if (ds == n) {
            //Find the Babai point
            return sils_babai_search_serial(z_B);
        }

        //last block:
        index q = d->at(ds - 1);

        scalar start = omp_get_wtime();
        //the last block
        vector<scalar> R_ii = sils::find_block_Rii<scalar, index>(R_A, n - q, n, n - q, n, n);
        vector<scalar> y_b_s = sils::find_block_x<scalar, index>(y_A, n - q, n);
        vector<index> x_b_s = ils_search(&R_ii, &y_b_s);
        for (index l = n - q; l < n; l++) {
            z_B->at(l) = x_b_s[l - n + q];
        }
//        sils::display_scalarType(y_b_s);
//        cout<<  n - q<<","<<n <<endl;
//        sils::display_scalarType(x_b_s);

        //Therefore, skip the last block, start from the second-last block till the first block.
        for (index i = 0; i < ds - 1; i++) {
            q = ds - 2 - i;
            //accumulate the block size
            y_b_s = sils::find_block_x<scalar, index>(y_A, n - d->at(q), n - d->at(q + 1));
            x_b_s = sils::find_block_x<scalar, index>(z_B, n - d->at(q + 1), n);
            y_b_s = sils::block_residual_vector(R_A, &x_b_s, &y_b_s, n - d->at(q), n - d->at(q + 1),
                                                n - d->at(q + 1), n);
//            sils::display_scalarType(y_b_s);
            R_ii = sils::find_block_Rii<scalar, index>(R_A, n - d->at(q), n - d->at(q + 1), n - d->at(q),
                                                       n - d->at(q + 1), n);
//            cout<< n - d->at(q)<<","<<n - d->at(q + 1)<<endl;
            x_b_s = ils_search(&R_ii, &y_b_s);
//            sils::display_array(x_b_s->at, x_b_s->size());

            for (index l = n - d->at(q); l < n - d->at(q + 1); l++) {
                z_B->at(l) = x_b_s[l - n + d->at(q)];
            }


        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {*z_B, run_time, 0, 0};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType<scalar, index>
    SILS<scalar, index, is_read, n>::sils_block_search_omp(index n_proc, index nswp, scalar stop,
                                                           vector<index> *z_B,
                                                           vector<index> *d) {
        index ds = d->size(), dx = d->at(ds - 1);
//        cout << "Block Size: " << dx << "," << "n_proc: " << n_proc << ",";
//        cout << dx << "," << n_proc << ",";
        //special cases:
        if (ds == 1) {
            if (d->at(0) == 1) {
                z_B->at(0) = round(y_A->x[0] / R_A->x[0]);
                return {*z_B, 0, 0, 0};
            } else {
                vector<scalar> R_B = sils::find_block_Rii(R_A, 0, n, 0, n, n);
                vector<scalar> y_B = sils::find_block_x(y_A, 0, n);
                return {ils_search(&R_B, &y_B), 0, 0, 0};
            }
        } else if (ds == n) {
            //Find the Babai point
            //todo: Change it to omp version
            return sils_babai_search_serial(z_B);
        }
        index count = 0, num_iter = 0, n_dx_q_0, n_dx_q_1;
        scalar res = 0, nres = 10, sum = 0;

        vector<scalar> y(dx, 0), y_n(dx, 0);
        vector<index> x(dx, 0), z_B_p(n, 0);

        for (index l = n - dx; l < n; l++) {
            y_n[l - (n - dx)] = y_A->x[l];
        }

        scalar start = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(sum, y, x, n_dx_q_0, n_dx_q_1)
        {
            y.assign(dx, 0);
            x.assign(dx, 0);
            for (index j = 0; j < nswp && abs(nres) > stop; j++) {//
#pragma omp for schedule(dynamic) nowait
//#pragma omp for nowait //schedule(dynamic)guided
//                for (index m = 0; m < n_proc; m++) {
//                    for (index i = m; i < ds; i += n_proc) {
                for (index i = 0; i < ds; i++) {
                    n_dx_q_0 = i == 0 ? n - dx : n - d->at(ds - 1 - i);
                    n_dx_q_1 = i == 0 ? n : n - d->at(ds - i);
                    //The block operation
                    if (i != 0) {
                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                            sum = 0;
#pragma omp simd reduction(+ : sum)
                            for (index col = n_dx_q_1; col < n; col++) {
                                //Translating the index from R(matrix) to R_B(compressed array).
                                sum += R_A->x[(n * row) + col - ((row * (row + 1)) / 2)] * z_B->at(col);
                            }
                            y[row - n_dx_q_0] = y_A->x[row] - sum;
                        }
                        x = ils_search_omp(n_dx_q_0, n_dx_q_1, &y);
                    } else {
                        x = ils_search_omp(n_dx_q_0, n_dx_q_1, &y_n);
                    }
#pragma omp simd
                    for (index l = n_dx_q_0; l < n_dx_q_1; l++) {
                        z_B->at(l) = x[l - n_dx_q_0];
                    }
                }
//                }
#pragma omp master
                {
                    if (num_iter > 0) {
                        nres = 0;
#pragma omp simd reduction(+ : nres)
                        for (index l = 0; l < n; l++) {
                            nres += (z_B_p[l] - z_B->at(l));
                            z_B_p[l] = z_B->at(l);
                        }
                    } else {
#pragma omp simd
                        for (index l = 0; l < n; l++) {
                            z_B_p[l] = z_B->at(l);
                        }
                    }
                    num_iter = j;
                }
            }

        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {*z_B, run_time, nres, num_iter};
        return reT;
    }
}
