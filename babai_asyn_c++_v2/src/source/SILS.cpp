#include <cstring>
#include <algorithm>

#include "../include/SILS.h"

#define VERBOSE_LEVEL = 1;

using namespace std;

namespace sils {
    template<typename scalar, typename index>
    scalarType<scalar, index> *sils_search_omp(scalarType<scalar, index> *R_B,
                                               scalarType<scalar, index> *y_B,
                                               scalarType<scalar, index> *z_B,
                                               index begin, index end, index r_block_size) {

        index block_size = y_B->size;

#ifdef VERBOSE
        cout << "sils_search" << endl;
        cout << "Block_size:" << block_size << endl;
#endif
        //partial residual
        auto *p = (scalar *) calloc(block_size, sizeof(scalar));
        auto *c = (scalar *) calloc(block_size, sizeof(scalar));
        auto *z = (scalar *) calloc(block_size, sizeof(scalar));
        auto *d = (index *) calloc(block_size, sizeof(index));

        scalar newprsd, gamma, sum = 0, beta = INFINITY;
        index k = block_size - 1;
        index end_1 = end - 1;
        index row_k = k + begin;

        //  Initial squared search radius
        scalar R_kk = R_B->x[(r_block_size * end_1) + end_1 - ((end_1 * (end_1 + 1)) / 2)];
        c[block_size - 1] = y_B->x[block_size - 1] / R_kk;
        z[block_size - 1] = round(c[block_size - 1]);
        gamma = R_kk * (c[block_size - 1] - z[block_size - 1]);

        //  Determine enumeration direction at level block_size
        d[block_size - 1] = c[block_size - 1] > z[block_size - 1] ? 1 : -1;

        while (true) {
            newprsd = p[k] + gamma * gamma;
            if (newprsd < beta) {
                if (k != 0) {
                    k--;
                    sum = 0;
                    row_k = k + begin;
//#pragma omp simd reduction(+ : sum)
                    for (index col = k + 1; col < block_size; col++) {
                        sum += R_B->x[(r_block_size * row_k) + col + begin - ((row_k * (row_k + 1)) / 2)] * z[col];
                    }
                    R_kk = R_B->x[(r_block_size * row_k) + row_k - ((row_k * (row_k + 1)) / 2)];

                    p[k] = newprsd;
                    c[k] = (y_B->x[k] - sum) / R_kk;
                    z[k] = round(c[k]);
                    gamma = R_kk * (c[k] - z[k]);

                    d[k] = c[k] > z[k] ? 1 : -1;

                } else {
                    beta = newprsd;
                    //Deep Copy of the result
//#pragma omp parallel for
                    for (int l = begin; l < end; l++) {
                        z_B->x[l] = z[l - begin];
                    }

                    z[0] += d[0];
                    gamma = R_B->x[0] * (c[0] - z[0]);
                    d[0] = d[0] > 0 ? -d[0] - 1 : -d[0] + 1;
                }
            } else {
                if (k == block_size - 1) break;
                else {
                    k++;
                    z[k] += d[k];
                    row_k = k + begin;
                    gamma = R_B->x[(r_block_size * row_k) + row_k - ((row_k * (row_k + 1)) / 2)] * (c[k] - z[k]);
                    d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                }
            }
        }
        free(z);
        free(c);
        free(d);
        free(p);

        return z_B;
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    SILS<scalar, index, is_read, is_write, n>::SILS(scalar noise) {
        this->R_A.x = (scalar *) calloc(n * n, sizeof(scalar));
        this->x_R.x = (scalar *) calloc(n, sizeof(scalar));
        this->x_tA.x = (scalar *) calloc(n, sizeof(scalar));
        this->y_A.x = (scalar *) calloc(n, sizeof(scalar));

        this->init_res = INFINITY;
        this->noise = noise;

        this->R_A.size = n * n;
        this->x_R.size = n;
        this->x_tA.size = n;
        this->y_A.size = n;

        this->init();
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    void SILS<scalar, index, is_read, is_write, n>::read() {
        string fy = "/home/shilei/data/y_" + to_string(n) + ".csv";
        string fx = "/home/shilei/data/x_" + to_string(n) + ".csv";
        string fR = "/home/shilei/data/R_A_" + to_string(n) + ".csv";
        string fxR ="/home/shilei/data/x_R_" + to_string(n) + ".csv";

        index i = 0;
        ifstream f(fR), f1(fy), f2(fx), f3(fxR);

        string row_string, entry;
        while (getline(f, row_string)) {
            scalar d = stod(row_string);
            this->R_A.x[i] = d;
            i++;
        }
        this->R_A.size = i;
        f.close();

        i = 0;
        while (getline(f1, row_string)) {
            scalar d = stod(row_string);
            this->y_A.x[i] = d;
            i++;
        }
        f1.close();

        i = 0;
        while (getline(f2, row_string)) {
            scalar d = stod(row_string);
            this->x_tA.x[i] = d;
            i++;
        }
        f2.close();

        i = 0;
        while (getline(f3, row_string)) {
            scalar d = stod(row_string);
            this->x_R.x[i] = d;
            i++;
        }
        f3.close();
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
    scalarType<scalar, index> *SILS<scalar, index, is_read, is_write, n>::sils_babai_search_omp(const index n_proc,
                                                                                                const index nswp,
                                                                                                index *update,
                                                                                                scalarType<scalar, index> *z_B,
                                                                                                scalarType<scalar, index> *z_B_p) {

        index count = 0, num_iter = 0;
        index chunk = std::log2(n);
        scalar res = 0;


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

        return z_B;
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    scalarType<scalar, index> *
    SILS<scalar, index, is_read, is_write, n>::sils_babai_search_serial(scalarType<scalar, index> *z_B) {
        scalar sum = 0;
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
        return z_B;
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    scalarType<scalar, index> *
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

        scalar newprsd, gamma;
        index k = block_size - 1;

        //  Initial squared search radius
        double beta = INFINITY;

        c[block_size - 1] = y_B->x[block_size - 1] / R_B->x[R_B->size - 1];
        z->x[block_size - 1] = round(c[block_size - 1]);
        gamma = R_B->x[R_B->size - 1] * (c[block_size - 1] - z->x[block_size - 1]);

        //  Determine enumeration direction at level block_size
        d[block_size - 1] = c[block_size - 1] > z->x[block_size - 1] ? 1 : -1;

        while (true) {
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
        std::memcpy(z->x, z_H, sizeof(scalar) * block_size);

        free(z_H);
        free(c);
        free(d);
        free(p);

        return z;
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
    scalarType<scalar, index> *
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
                return z_B;
            }
        }
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    scalarType<scalar, index> *
    SILS<scalar, index, is_read, is_write, n>::sils_block_search_serial(scalarType<scalar, index> *R_B,
                                                                        scalarType<scalar, index> *y_B,
                                                                        scalarType<scalar, index> *z_B,
                                                                        scalarType<index, index> *d) {
        index ds = d->size;
        //special cases:
        if (ds == 1) {
            if (d->x[0] == 1) {
                z_B->x[0] = round(y_B->x[0] / R_B->x[0]);
                return z_B;
            } else {
                return sils_search(R_B, y_B);
            }
        } else if (ds == n) {
            //Find the Babai point
            return sils_babai_search_serial(z_B);
        }

        //last block:
        index q = d->x[ds - 1];

        //the last block
        auto R_ii = sils::find_block_Rii<double, int>(R_B, n - q, n, n - q, n, n);
        auto y_b_s = sils::find_block_x<double, int>(y_B, n - q, n);
        auto x_b_s = sils_search(R_ii, y_b_s);
        for (int l = n - q; l < n; l++) {
            z_B->x[l] = x_b_s->x[l - n + q];
        }

        //Therefore, skip the last block, start from the second-last block till the first block.
        for (index i = 0; i < ds - 1; i++) {
            q = ds - 2 - i;
            //accumulate the block size
            y_b_s = sils::find_block_x<double, int>(y_B, n - d->x[q], n - d->x[q + 1]);
            x_b_s = sils::find_block_x<double, int>(z_B, n - d->x[q + 1], n);
            y_b_s = sils::block_residual_vector(R_B, x_b_s, y_b_s, n - d->x[q], n - d->x[q + 1],
                                                n - d->x[q + 1],
                                                n);
            R_ii = sils::find_block_Rii<double, int>(R_B, n - d->x[q], n - d->x[q + 1], n - d->x[q],
                                                     n - d->x[q + 1], n);
            x_b_s = sils_search(R_ii, y_b_s);

            for (int l = n - d->x[q]; l < n - d->x[q + 1]; l++) {
                z_B->x[l] = x_b_s->x[l - n + d->x[q]];
            }
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

        free(R_ii);
        free(y_b_s);
        free(x_b_s);

        return z_B;
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    scalarType<scalar, index> *
    SILS<scalar, index, is_read, is_write, n>::sils_block_search_omp(index n_proc, index nswp,
                                                                     scalarType<scalar, index> *R_B,
                                                                     scalarType<scalar, index> *y_B,
                                                                     scalarType<scalar, index> *z_B,
                                                                     scalarType<index, index> *d) {
        index ds = d->size, dx = d->x[ds - 1];
        //special cases:
        if (ds == 1) {
            if (d->x[0] == 1) {
                z_B->x[0] = round(y_B->x[0] / R_B->x[0]);
                return z_B;
            } else {
                return sils_search(R_B, y_B);
            }
        } else if (ds == n) {
            //Find the Babai point
            return sils_babai_search_serial(z_B);
        }


#pragma omp parallel default(shared) num_threads(n_proc)
        {
            for (index j = 0; j < nswp; j++) {
#pragma omp for schedule(dynamic) nowait
                for (index i = 0; i < ds; i++) {
                    if (i == ds - 1)
                        z_B = do_block_solve(n - dx, n, z_B);
                    else
                        z_B = do_block_solve(n - d->x[ds - 2 - i], n - d->x[ds - 1 - i], z_B);
                }
            }
        }
        return z_B;
    }
}