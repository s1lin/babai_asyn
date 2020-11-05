#include <cstring>
#include <algorithm>

#include "Babai_search_asyn.h"

using namespace std;

namespace babai {

    template<typename scalar, typename index>
    scalar find_residual(const index n, const scalar *R, const scalar *y, const scalar *x) {
        scalar res = 0;
        for (index i = 0; i < n; i++) {
            scalar sum = 0;
            for (index j = i; j < n; j++) {
                sum += x[j] * R[(n * i) + j - ((i * (i + 1)) / 2)];
            }
            res += (y[i] - sum) * (y[i] - sum);
        }
        return std::sqrt(res);
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    Babai_search_asyn<scalar, index, is_read, is_write, n>::Babai_search_asyn(scalar noise) {
        this->R_A = (scalar *) calloc(n * n, sizeof(scalar));
        this->x_R = (scalar *) calloc(n, sizeof(scalar));
        this->x_tA = (scalar *) calloc(n, sizeof(scalar));
        this->y_A = (scalar *) calloc(n, sizeof(scalar));

        this->init_res = INFINITY;
        this->noise = noise;
        this->size_R_A = 0;
        this->init();
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    void Babai_search_asyn<scalar, index, is_read, is_write, n>::read_from_RA() {
        string fxR = "../../data/R_A_" + to_string(n) + ".csv";
        index i = 0;
        ifstream f3(fxR);
        string row_string3, entry3;
        while (getline(f3, row_string3)) {
            scalar d = stod(row_string3);
            this->R_A[i] = d;
            i++;
        }
        this->size_R_A = i;
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    void Babai_search_asyn<scalar, index, is_read, is_write, n>::read_x_y() {
        string fy =
                "../../data/y_" + to_string(n) + ".csv";
        string fx =
                "../../data/x_" + to_string(n) + ".csv";
        string fxR =
                "../../data/x_R_" + to_string(n) + ".csv";
        string row_string, entry;
        index i = 0;
        ifstream f1(fy);
        while (getline(f1, row_string)) {
            scalar d = stod(row_string);
            this->y_A[i] = d;
            i++;
        }

        i = 0;
        ifstream f2(fx);
        string row_string2, entry2;
        while (getline(f2, row_string2)) {
            scalar d = stod(row_string2);
            this->x_tA[i] = d;
            i++;
        }

        i = 0;
        ifstream f3(fxR);
        string row_string3, entry3;
        while (getline(f3, row_string3)) {
            scalar d = stod(row_string3);
            this->x_R[i] = d;
            i++;
        }
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    void Babai_search_asyn<scalar, index, is_read, is_write, n>::write_R_A() {
        string fR = "../../data/R_A_" + to_string(n) + ".csv";
        ofstream file3(fR);
        if (file3.is_open()) {
            for (index i = 0; i < size_R_A; i++)
                file3 << setprecision(15) << R_A[i] << ",";
            file3.close();
        }
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    void Babai_search_asyn<scalar, index, is_read, is_write, n>::init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        //mean:0, std:1. same as matlab.
        std::normal_distribution<scalar> norm_dis(0, 1);
        //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
        std::uniform_int_distribution<index> index_dis(-n);

        if (is_read) {
//              read_from_R();
//              write_R_A();
//              compare_R_RA();
            read_from_RA();
            read_x_y();
            this->init_res = find_residual(n, R_A, y_A, x_tA);

            cout << "init_res:" << this->init_res << endl;

        } else {

        }
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    scalar *Babai_search_asyn<scalar, index, is_read, is_write, n>::sils_babai_search_omp(const index n_proc,
                                                                                          const index nswp,
                                                                                          index *update,
                                                                                          scalar *z_B,
                                                                                          scalar *z_B_p) {

        index count = 0, num_iter = 0;
        index chunk = std::log2(n);
        scalar res = 0;


        z_B[n - 1] = round(y_A[n - 1] / R_A[((n - 1) * n) / 2 + n - 1]);
#pragma omp parallel default(shared) num_threads(n_proc) private(count) shared(update)
        {
            for (index j = 0; j < nswp; j++) {//&& count < 16
                count = 0;
#pragma omp for schedule(dynamic) nowait
                for (index i = 0; i < n; i++) {
                    int x_c = do_solve(i, z_B);
//                    int x_p = z_B[n - 1 - i];
                    z_B[n - 1 - i] = x_c;

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
    scalar *Babai_search_asyn<scalar, index, is_read, is_write, n>::sils_babai_search_serial(scalar *z_B) {
        scalar sum = 0;
        z_B[n - 1] = round(y_A[n - 1] / R_A[((n - 1) * n) / 2 + n - 1]);
        for (index i = 1; i < n; i++) {
            index k = n - i - 1;
            for (index col = n - i; col < n; col++) {
                sum += R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + col] * z_B[col];
            }
            z_B[k] = round(
                    (y_A[n - 1 - i] - sum) / R_A[(n - 1 - i) * n - ((n - 1 - i) * (n - i)) / 2 + n - 1 - i]);
            sum = 0;
        }
        return z_B;
    }

    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    scalar *Babai_search_asyn<scalar, index, is_read, is_write, n>::sils_search(const scalar *R,
                                                                                const scalar *y,
                                                                                scalar *z_B,
                                                                                index block_size,
                                                                                index block_size_R_A) {

        auto *prsd = (scalar *) calloc(block_size, sizeof(scalar));
        auto *c = (scalar *) calloc(block_size, sizeof(scalar));
        auto *z_H = (scalar *) calloc(block_size, sizeof(scalar));
        auto *d = (index *) calloc(block_size, sizeof(index));

        scalar newprsd, gamma, b_R;
        index k = block_size - 1, i, j, r_tmp, loop_ub;

        //  Initial squared search radius
        double beta = INFINITY;

        c[block_size - 1] = y_A[block_size - 1] / R_A[size_R_A - 1];
        z_B[block_size - 1] = round(c[block_size - 1]);
        gamma = R_A[size_R_A - 1] * (c[block_size - 1] - z_B[block_size - 1]);

        //  Determine enumeration direction at level block_size
        d[block_size - 1] = c[block_size - 1] > z_B[block_size - 1] ? 1 : -1;

        while (true) {
            newprsd = prsd[k] + gamma * gamma;
            if (newprsd < beta) {
                if (k != 0) {
                    k--;
                    double sum = 0;
                    for (index col = k + 1; col < block_size; col++) {
                        sum += R_A[(block_size * k) + col - ((k * (k + 1)) / 2)] * z_B[col];
                    }
                    scalar s = y_A[k] - sum;
                    scalar R_kk = R_A[(block_size * k) + k - ((k * (k + 1)) / 2)];
                    prsd[k] = newprsd;
                    c[k] = s / R_kk;

                    z_B[k] = round(c[k]);
                    gamma = R_kk * (c[k] - z_B[k]);
                    d[k] = c[k] > z_B[k] ? 1 : -1;

                } else {
                    beta = newprsd;
                    //Deep Copy of the result
                    std::memcpy(z_H, z_B, sizeof(scalar) * block_size);
                    z_B[0] += d[0];
                    gamma = R_A[0] * (c[0] - z_B[0]);
                    d[0] = d[0] > 0 ? -d[0] - 1 : -d[0] + 1;
                }
            } else {
                if (k == block_size - 1) break;
                else {
                    k++;
                    z_B[k] += d[k];
                    gamma = R_A[(block_size * k) + k - ((k * (k + 1)) / 2)] * (c[k] - z_B[k]);
                    d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                }
            }
        }
        std::memcpy(z_B, z_H, sizeof(scalar) * block_size);
        free(c);
        free(d);
        free(prsd);
        free(z_H);

        return z_B;
    }


    template<typename scalar, typename index, bool is_read, bool is_write, index n>
    scalar *Babai_search_asyn<scalar, index, is_read, is_write, n>::sils_babai_block_search_omp(const index n_proc,
                                                                                                const index nswp,
                                                                                                index *update,
                                                                                                scalar *z_B,
                                                                                                scalar *z_B_p) {

        index count = 0, num_iter = 0;
        index chunk = std::log2(n);
        scalar res = 0;


        z_B[n - 1] = round(y_A[n - 1] / R_A[((n - 1) * n) / 2 + n - 1]);
#pragma omp parallel default(shared) num_threads(n_proc) private(count) shared(update)
        {
            for (index j = 0; j < nswp; j++) {//&& count < 16
                count = 0;
#pragma omp for schedule(dynamic) nowait
                for (index i = 0; i < n; i++) {
                    int x_c = do_solve(i, z_B);
                    //                    int x_p = z_B[n - 1 - i];
                    z_B[n - 1 - i] = x_c;

                    // if (x_c != x_p) {
                    //     update[n - 1 - i] = 0;
                    //     z_B_p[n - 1 - i] = x_c;
                    // } else {
                    //     update[n - 1 - i] = 1;
                    // }
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
    scalar *
    Babai_search_asyn<scalar, index, is_read, is_write, n>::sils_block_search_serial(scalar *R_B,
                                                                                     scalar *y_B,
                                                                                     scalar *z_B,
                                                                                     vector<index> d,
                                                                                     index n_R_B) {
        index ds = d.size();
        if (ds == 1) {
            if (d[0] == 1) {
                return x_R;
            } else {
                //todo: determine the right size
                return sils_search(R_B, y_B, z_B, n, size_R_A);
            }
        } else {
            //Find the Babai point
            if (ds == n) {
                return sils_babai_search_serial(z_B);
            } else {
                index q = d[0];
                //new_d = d[2:ds];
                vector<index> new_d(d.begin() + 1, d.end());

                //todo: create R_B
                auto *R_b = babai::find_block_R(R_B, q, n_R_B, q, n_R_B, n_R_B - q, n_R_B);
                auto *y_b = (scalar *) calloc(n_R_B - q, sizeof(scalar));

                for (int i = q; i < n_R_B; i++) {
                    y_b[i - q] = y_B[i];
                }

                index n_R_b = sizeof(R_b) / sizeof(R_b[0]);
                scalar *xx1 = sils_block_search_serial(R_b, y_b, z_B, new_d, n_R_b);

                auto *y_b_2 = (scalar *) calloc(q, sizeof(scalar));
                auto *xx2 = (scalar *) calloc(q, sizeof(scalar));


                for (int row = 0; row < q; row++) {
                    scalar sum = 0;
                    for (int col = q; col < n_R_B; col++) {
                        int i = (n_R_B * row) + col - ((row * (row + 1)) / 2);
                        sum += R_B[i] * xx1[col - q - 1];
                    }
                    y_b_2[row] = y_B[row] - sum;
                }
                if (q == 1) {
                    xx2[0] = round(y_b_2[0] / R_B[0]);
                } else {
                    auto *R_b = babai::find_block_R(R_B, 0, q, 0, q, q, n_R_B);
                    xx2 = sils_search(R_b, y_b_2, z_B);
                }

                return babai::concatenate_array<scalar, index>(xx2, xx1);
            }
        }
    }
}

