#include <cstring>
#include <algorithm>

#include "../include/SILS.h"

#define FILE_NAME "simple_xy.nc"
#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

#define VERBOSE_LEVEL = 1;

using namespace std;

namespace sils {

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    SILS<scalar, index, is_read, is_write, n>::SILS(index qam, index snr) {
        this->R_A.x = (scalar *) calloc(n * (n + 1) / 2, sizeof(scalar));
        this->x_R.x = (scalar *) calloc(n, sizeof(scalar));
        this->x_tA.x = (scalar *) calloc(n, sizeof(scalar));
        this->y_A.x = (scalar *) calloc(n, sizeof(scalar));

        this->init_res = INFINITY;
        this->qam = qam;
        this->snr = snr;

        this->R_A.size = n * (n + 1) / 2;
        this->x_R.size = n;
        this->x_tA.size = n;
        this->y_A.size = n;

        this->init();
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    void SILS<scalar, index, is_read, is_write, n>::read() {
        string filename = "data/" + to_string(n) + "_" + to_string(snr) + "_" + to_string(qam) + ".nc";
        index ncid, varid, retval;
        if ((retval = nc_open(filename.c_str(), NC_NOWRITE, &ncid))) ERR(retval);

        /* Get the varid of the data variable, based on its name. */
        if ((retval = nc_inq_varid(ncid, "x_t", &varid))) ERR(retval);

        /* Read the data. */
        if ((retval = nc_get_var_double(ncid, varid, &x_tA.x[0]))) ERR(retval);

        /* Get the varid of the data variable, based on its name. */
        if ((retval = nc_inq_varid(ncid, "y", &varid))) ERR(retval);

        /* Read the data. */
        if ((retval = nc_get_var_double(ncid, varid, &y_A.x[0]))) ERR(retval);

        /* Get the varid of the data variable, based on its name. */
        if ((retval = nc_inq_varid(ncid, "x_R", &varid))) ERR(retval);

        /* Read the data. */
        if ((retval = nc_get_var_double(ncid, varid, &x_R.x[0]))) ERR(retval);

        /* Get the varid of the data variable, based on its name. */
        if ((retval = nc_inq_varid(ncid, "R_A", &varid))) ERR(retval);

        /* Read the data. */
        if ((retval = nc_get_var_double(ncid, varid, &R_A.x[0]))) ERR(retval);
    }


    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    void SILS<scalar, index, is_read, is_write, n>::write() {
        string fR = "../../data/R_A_" + to_string(n) + ".csv";
        ofstream f(fR);
        if (f.is_open()) {
            for (index i = 0; i < R_A.size; i++)
                f << setprecision(15) << R_A.x[i] << ",";
            f.close();
        }


    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    void SILS<scalar, index, is_read, is_write, n>::init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        //mean:0, std:1. same as matlab.
        std::normal_distribution<scalar> norm_dis(0, 1);
        //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
        std::uniform_int_distribution<index> index_dis(-n);

        if (is_read) {
            read();
            this->init_res = sils::find_residual<scalar, index, n>(&R_A, &y_A, &x_tA);
            cout << "init_res:" << this->init_res << endl;
        } else {
            //todo: generate problem
        }
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    returnType<scalar, index> SILS<scalar, index, is_read, is_write, n>::sils_babai_search_omp(const index n_proc,
                                                                                               const index nswp,
                                                                                               index *update,
                                                                                               scalarType<scalar, index> *z_B,
                                                                                               scalarType<scalar, index> *z_B_p) {

        index count = 0, num_iter = 0;
        index chunk = std::log2(n);
        scalar res = 0;

        scalar start = omp_get_wtime();
        z_B->x[n - 1] = round(y_A.x[n - 1] / R_A.x[((n - 1) * n) / 2 + n - 1]);
#pragma omp parallel default(shared) num_threads(n_proc) private(count) shared(update)
        {
            for (index j = 0; j < nswp; j++) {//&& count < 16
                count = 0;
#pragma omp for schedule(dynamic) nowait
                for (index i = 0; i < n; i++) {
                    int x_c = do_solve(i, z_B->x);
//                    int x_p = z_B[n - 1 - i];
                    z_B->x[n - 1 - i] = x_c;

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
        //cout << num_iter << endl;
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {z_B, run_time, 0, num_iter};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    returnType<scalar, index>
    SILS<scalar, index, is_read, is_write, n>::sils_babai_search_serial(scalarType<scalar, index> *z_B) {
        scalar sum = 0;
        scalar start = omp_get_wtime();

        z_B->x[n - 1] = round(y_A.x[n - 1] / R_A.x[((n - 1) * n) / 2 + n - 1]);
        for (index i = 1; i < n; i++) {
            index k = n - i - 1;
            for (index col = n - i; col < n; col++) {
                sum += R_A.x[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * z_B->x[col];
            }
            z_B->x[k] = round(
                    (y_A.x[n - 1 - i] - sum) / R_A.x[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
            sum = 0;
        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {z_B, run_time, 0, 0};
        return reT;
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    returnType<scalar, index>
    SILS<scalar, index, is_read, is_write, n>::sils_search(scalarType<scalar, index> *R_B,
                                                           scalarType<scalar, index> *y_B) {


        index block_size = y_B->size;

#ifdef VERBOSE
        cout << "sils_search" << endl;
        cout << "Block_size:" << block_size << endl;
#endif
        auto *z = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
        z->x = (scalar *) calloc(block_size, sizeof(scalar));
        z->size = block_size;

        auto *p = (scalar *) calloc(block_size, sizeof(scalar));
        auto *c = (scalar *) calloc(block_size, sizeof(scalar));
        auto *z_H = (scalar *) calloc(block_size, sizeof(scalar));
        auto *d = (index *) calloc(block_size, sizeof(index));

        scalar newprsd, gamma, b_R;
        index k = block_size - 1, i, j;

        //  Initial squared search radius
        double beta = INFINITY;

        c[block_size - 1] = y_B->x[block_size - 1] / R_B->x[R_B->size - 1];
        z->x[block_size - 1] = round(c[block_size - 1]);
        gamma = R_B->x[R_B->size - 1] * (c[block_size - 1] - z->x[block_size - 1]);

        //  Determine enumeration direction at level block_size
        d[block_size - 1] = c[block_size - 1] > z->x[block_size - 1] ? 1 : -1;
        index iter = 0;
        while (true) {
//            iter++;
            newprsd = p[k] + gamma * gamma;
            if (newprsd < beta) {
                if (k != 0) {
                    k--;
                    double sum = 0;
                    for (index col = k + 1; col < block_size; col++) {
                        sum += R_B->x[(block_size * k) + col - ((k * (k + 1)) / 2)] * z->x[col];
                    }
                    scalar s = y_B->x[k] - sum;
                    scalar R_kk = R_B->x[(block_size * k) + k - ((k * (k + 1)) / 2)];
                    p[k] = newprsd;
                    c[k] = s / R_kk;

                    z->x[k] = round(c[k]);
                    gamma = R_kk * (c[k] - z->x[k]);
                    d[k] = c[k] > z->x[k] ? 1 : -1;

                } else {
                    beta = newprsd;
                    //Deep Copy of the result
                    std::memcpy(z_H, z->x, sizeof(scalar) * block_size);

                    z->x[0] += d[0];
                    gamma = R_B->x[0] * (c[0] - z->x[0]);
                    d[0] = d[0] > 0 ? -d[0] - 1 : -d[0] + 1;
                }
            } else {
                if (k == block_size - 1) break;
                else {
                    k++;
                    z->x[k] += d[k];
                    gamma = R_B->x[(block_size * k) + k - ((k * (k + 1)) / 2)] * (c[k] - z->x[k]);
                    d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                }
            }
        }
//        cout<<iter<<endl;
        std::memcpy(z->x, z_H, sizeof(scalar) * block_size);
        returnType<scalar, index> reT = {z, 0, 0, 0};
        free(z_H);
        free(c);
        free(d);
        free(p);

        return reT;
    }


    /**
     * @deprecated
     * @tparam scalar
     * @tparam index
     * @tparam is_read
     * @tparam is_write
     * @tparam n
     * @param R_B
     * @param y_B
     * @param z_B
     * @param d
     * @return
     */
    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    returnType<scalar, index>
    SILS<scalar, index, is_read, is_write, n>::sils_block_search_serial_recursive(scalarType<scalar, index> *R_B,
                                                                                  scalarType<scalar, index> *y_B,
                                                                                  scalarType<scalar, index> *z_B,
                                                                                  vector<index> d) {
        index ds = d.size();
        index block_size = y_B->size;
//        cout << block_size << endl;

        if (ds == 1) {
            if (d[0] == 1) {
                scalarType<scalar, index> x{(scalar *) calloc(1, sizeof(scalar)), 1};
                x.x[0] = round(y_B->x[0] / R_B->x[0]);
                return &x;
            } else {
                return sils_search(R_B, y_B);
            }
        } else {
            //Find the Babai point
            if (ds == n) {
                return sils_babai_search_serial(z_B);
            } else {
                index q = d[0];
                //new_d = d[2:ds];
                vector<index> new_d(d.begin() + 1, d.end());

                auto R_b_s = sils::find_block_Rii<double, int>(R_B, q, block_size, q, block_size, y_B->size);
//                cout << "R_b" << endl;
//                sils::display_scalarType<double, int>(R_b_s);

                auto y_b_s = sils::find_block_x<double, int>(y_B, q, block_size);
//                cout << "\n y_b" << endl;
//                sils::display_scalarType<double, int>(y_b_s);

                auto z_b_s = sils::find_block_x<double, int>(z_B, q, block_size);

                auto x_b_s = sils_block_search_serial_recursive(R_b_s, y_b_s, z_b_s, new_d);
//                cout << "x_b_s," << z_B->size << endl;
//                sils::display_scalarType<double, int>(x_b_s);
//
//                cout << "y_b_s_2" << endl;
                scalarType<scalar, index> *x_b_s_2, z_b_s_2;
                auto y_b_s_2 = block_residual_vector(R_B, x_b_s, y_B, 0, q, q, block_size);
//                sils::display_scalarType<double, int>(y_b_s_2);

                if (q == 1) {
                    x_b_s_2 = (scalarType<scalar, index> *) malloc(sizeof(scalarType<scalar, index>));
                    x_b_s_2->x = (scalar *) calloc(q, sizeof(scalar));
                    x_b_s_2->size = 1;
                    x_b_s_2->x[0] = round(y_b_s_2->x[0] / R_B->x[0]);
                } else {

                    R_b_s = sils::find_block_Rii<double, int>(R_B, 0, q, 0, q, y_B->size);
//                    cout << "R_b_s" << endl;
//                    sils::display_scalarType<double, int>(R_b_s);

                    x_b_s_2 = sils_search(R_b_s, y_b_s_2);
//                    cout << "x_b_s_2" << endl;
//                    sils::display_scalarType<double, int>(x_b_s_2);
                }

                z_B = sils::concatenate_array<scalar, index>(x_b_s_2, x_b_s);
                free(R_b_s);
                free(y_b_s);
                free(y_b_s_2);
                free(x_b_s);
                free(x_b_s_2);
                returnType<scalar, index> reT = {z_B, 0, 0, 0};
                return reT;
            }
        }
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    returnType<scalar, index>
    SILS<scalar, index, is_read, is_write, n>::sils_block_search_serial(scalarType<scalar, index> *R_B,
                                                                        scalarType<scalar, index> *y_B,
                                                                        scalarType<scalar, index> *z_B,
                                                                        scalarType<index, index> *d) {

        index ds = d->size;
        //special cases:
        if (ds == 1) {
            if (d->x[0] == 1) {
                z_B->x[0] = round(y_B->x[0] / R_B->x[0]);
                return {z_B, 0, 0, 0};
            } else {
                return sils_search(R_B, y_B);
            }
        } else if (ds == n) {
            //Find the Babai point
            return sils_babai_search_serial(z_B);
        }

        //last block:
        index q = d->x[ds - 1];

        scalar start = omp_get_wtime();
        //the last block
        auto R_ii = sils::find_block_Rii<double, int>(R_B, n - q, n, n - q, n, n);
        auto y_b_s = sils::find_block_x<double, int>(y_B, n - q, n);
        auto x_b = sils_search(R_ii, y_b_s);
        for (int l = n - q; l < n; l++) {
            z_B->x[l] = x_b.x->x[l - n + q];
        }
//        sils::display_scalarType(y_b_s);
//        cout<<  n - q<<","<<n <<endl;
//        sils::display_scalarType(x_b_s);

        //Therefore, skip the last block, start from the second-last block till the first block.
        for (index i = 0; i < ds - 1; i++) {
            q = ds - 2 - i;
            //accumulate the block size
            y_b_s = sils::find_block_x<double, int>(y_B, n - d->x[q], n - d->x[q + 1]);
            auto x_b_s = sils::find_block_x<double, int>(z_B, n - d->x[q + 1], n);
            y_b_s = sils::block_residual_vector(R_B, x_b_s, y_b_s, n - d->x[q], n - d->x[q + 1],
                                                n - d->x[q + 1],
                                                n);
//            sils::display_scalarType(y_b_s);
            R_ii = sils::find_block_Rii<double, int>(R_B, n - d->x[q], n - d->x[q + 1], n - d->x[q],
                                                     n - d->x[q + 1], n);
//            cout<< n - d->x[q]<<","<<n - d->x[q + 1]<<endl;
            x_b = sils_search(R_ii, y_b_s);
//            sils::display_array(x_b_s->x, x_b_s->size);

            for (int l = n - d->x[q]; l < n - d->x[q + 1]; l++) {
                z_B->x[l] = x_b.x->x[l - n + d->x[q]];
            }
//            sils::display_scalarType(x_b_s);
#ifdef VERBOSE
            cout << "y_b_s" << endl;
            sils::display_scalarType<double, int>(&y_b_s);
            cout << "R_ii" << endl;
            sils::display_scalarType<double, int>(R_ii);
            cout << "y_b_s_2" << endl;
            sils::display_scalarType<double, int>(&y_b_s_2);
            cout << "x" << endl;
            sils::display_scalarType<double, int>(x);
#endif

        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {z_B, run_time, 0, 0};
        free(R_ii);
        free(y_b_s);
//        free(x_b_s);
        return reT;
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    returnType<scalar, index>
    SILS<scalar, index, is_read, is_write, n>::sils_block_search_omp(index n_proc, index nswp, scalar stop,
                                                                     scalarType<scalar, index> *R_B,
                                                                     scalarType<scalar, index> *y_B,
                                                                     scalarType<scalar, index> *z_B,
                                                                     scalarType<scalar, index> *z_B_p,
                                                                     scalarType<index, index> *d) {
        index ds = d->size, dx = d->x[ds - 1];
//        cout << "Block Size: " << dx << "," << "n_proc: " << n_proc << ",";
//        cout << dx << "," << n_proc << ",";
        //special cases:
        if (ds == 1) {
            if (d->x[0] == 1) {
                z_B->x[0] = round(y_B->x[0] / R_B->x[0]);
                return {z_B, 0, 0, 0};
            } else {
                return sils_search(R_B, y_B);
            }
        } else if (ds == n) {
            //Find the Babai point
            //todo: Change it to omp version
            return sils_babai_search_serial(z_B);
        }
        index count = 0, num_iter = 0, n_dx_q_0, n_dx_q_1;
        scalar res = 0, nres = 10, sum = 0;
        auto *y = (scalar *) calloc(dx, sizeof(scalar));
        auto *y_n = (scalar *) calloc(dx, sizeof(scalar));

        for (index l = n - dx; l < n; l++) {
            y_n[l - (n - dx)] = y_B->x[l];
        }

        scalar start = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(sum, y, n_dx_q_0, n_dx_q_1)
        {
            y = (scalar *) calloc(dx, sizeof(scalar));
            for (index j = 0; j < nswp && abs(nres) > stop; j++) {//
#pragma omp for nowait
                for (index i = 0; i < ds; i++) {
                    n_dx_q_0 = i == 0 ? n - dx : n - d->x[ds - 1 - i];
                    n_dx_q_1 = i == 0 ? n : n - d->x[ds - i];
                    //The block operation
                    if (i != 0) {
                        for (index row = n_dx_q_0; row < n_dx_q_1; row++) {
                            sum = 0;
#pragma omp simd reduction(+ : sum)
                            for (index col = n_dx_q_1; col < n; col++) {
                                //Translating the index from R(matrix) to R_B(compressed array).
                                sum += R_B->x[(n * row) + col - ((row * (row + 1)) / 2)] * z_B->x[col];
                            }
                            y[row - n_dx_q_0] = y_B->x[row] - sum;
                        }
                        y = do_block_solve(n_dx_q_0, n_dx_q_1, y);
                    } else {
                        y = do_block_solve(n_dx_q_0, n_dx_q_1, y_n);
                    }
#pragma omp simd
                    for (index l = n_dx_q_0; l < n_dx_q_1; l++) {
                        z_B->x[l] = y[l - n_dx_q_0];
                    }
                }
#pragma omp master
                {
                    if (num_iter > 0) {
                        nres = 0;
#pragma omp simd reduction(+ : nres)
                        for (index l = 0; l < n; l++) {
                            nres += (z_B_p->x[l] - z_B->x[l]);
                            z_B_p->x[l] = z_B->x[l];
                        }
                    } else {
#pragma omp simd
                        for (index l = 0; l < n; l++) {
                            z_B_p->x[l] = z_B->x[l];
                        }
                    }
                    num_iter = j;
                }
            }

        }
        scalar run_time = omp_get_wtime() - start;
        returnType<scalar, index> reT = {z_B, run_time, nres, num_iter};
        free(y);
        free(y_n);
        return reT;
    }
}
