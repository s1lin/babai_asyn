#include <cstring>
#include <algorithm>

#include "SILS.h"

#define VERBOSE_LEVEL = 1;

using namespace std;

namespace sils {

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
        string fy = "data/y_" + to_string(n) + ".csv";
        string fx = "data/x_" + to_string(n) + ".csv";
        string fR = "data/R_A_" + to_string(n) + ".csv";
        string fxR ="data/x_R_" + to_string(n) + ".csv";

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

        auto *prsd = (scalar *) calloc(block_size, sizeof(scalar));
        auto *c = (scalar *) calloc(block_size, sizeof(scalar));
        auto *z_H = (scalar *) calloc(block_size, sizeof(scalar));
        auto *d = (index *) calloc(block_size, sizeof(index));

        scalar newprsd, gamma, b_R;
        index k = block_size - 1, i, j, r_tmp, loop_ub;

        //  Initial squared search radius
        double beta = INFINITY;

        c[block_size - 1] = y_B->x[block_size - 1] / R_B->x[R_B->size - 1];
        z->x[block_size - 1] = round(c[block_size - 1]);
        gamma = R_B->x[R_B->size - 1] * (c[block_size - 1] - z->x[block_size - 1]);

        //  Determine enumeration direction at level block_size
        d[block_size - 1] = c[block_size - 1] > z->x[block_size - 1] ? 1 : -1;

        while (true) {
            newprsd = prsd[k] + gamma * gamma;
            if (newprsd < beta) {
                if (k != 0) {
                    k--;
                    double sum = 0;
                    for (index col = k + 1; col < block_size; col++) {
                        sum += R_B->x[(block_size * k) + col - ((k * (k + 1)) / 2)] * z->x[col];
                    }
                    scalar s = y_B->x[k] - sum;
                    scalar R_kk = R_B->x[(block_size * k) + k - ((k * (k + 1)) / 2)];
                    prsd[k] = newprsd;
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
        free(prsd);

        return z;
    }


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
        for (int l = n - d->x[ds - 1]; l < n; l++) {
            z_B->x[l] = x_b_s->x[l - n + d->x[ds - 1]];
        }

        //Therefore, skip the last block, start from the second-last block till the first block.
        for (index i = 0; i < ds - 1; i++) {
            index q = ds - 2 - i;
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
        cout << "sils_block_search_omp" << endl;
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
        index q = d->x[ds - 1], num_iter = 0;

        //the last block
        auto R_ii = sils::find_block_Rii<double, int>(R_B, n - q, n, n - q, n, n);
        auto y_b_s = sils::find_block_x<double, int>(y_B, n - q, n);
        auto x_b_s = sils_search(R_ii, y_b_s);

        for (int l = n - d->x[ds - 1]; l < n; l++) {
            z_B->x[l] = x_b_s->x[l - n + d->x[ds - 1]];
        }


#pragma omp parallel default(shared) num_threads(n_proc) private(y_b_s, R_ii, x_b_s, q)
        {
            for (index j = 0; j < nswp; j++) {
#pragma omp for schedule(dynamic) nowait
                for (index i = 0; i < ds - 1; i++) {
                    index q = ds - 2 - i;
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
//                    cout << "z_B" << endl;
//                    sils::display_scalarType<double, int>(x_b_s);
                }
//                num_iter = j;
            }
        }
//        cout << num_iter << endl;
//        cout << "z_B" << endl;
//        sils::display_scalarType<double, int>(z_B);
        free(R_ii);
        free(y_b_s);
        free(x_b_s);

        return z_B;
    }
}

const int n = 4096;

void test_ils_block_search() {
    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    sils::SILS<double, int, true, false, n> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    sils::scalarType<double, int> z_B{(double *) calloc(n, sizeof(double)), n};
    vector<int> d(256, 16); //256*16=4096
    sils::scalarType<int, int> d_s{d.data(), (int)d.size()};
    sils::scalarType<double, int> z_BV{(double *) calloc(n, sizeof(double)), n};
    for (int i = d_s.size - 2; i >= 0; i--) {
        d_s.x[i] += d_s.x[i + 1];
    }

//    vector<int> d(5, 2); //256*16=4096

    start = omp_get_wtime();
    auto z_B_s = bsa.sils_block_search_serial(&bsa.R_A, &bsa.y_A, &z_B, &d_s);
    end_time = omp_get_wtime() - start;
//    sils::display_scalarType<double, int>(z_B_s);
    auto res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, z_B_s);
    printf("Thread: ILS_SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

//    start = omp_get_wtime();
//    z_B_s = bsa.sils_block_search_serial_recursive(&bsa.R_A, &bsa.y_A, &z_B, d);
//    end_time = omp_get_wtime() - start;
//    res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, z_B_s);
//    printf("Thread: ILS_RE, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);


    start = omp_get_wtime();
    z_BV = *bsa.sils_block_search_omp(40, 16, &bsa.R_A, &bsa.y_A, &z_BV, &d_s);
    end_time = omp_get_wtime() - start;
    res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_BV);
    printf("Thread: ILS_OP, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

//    sils::scalarType<double, int> z_BS = {(double *) calloc(n, sizeof(double)), n};
//    start = omp_get_wtime();
//    z_B_s = bsa.sils_babai_search_serial(&z_BS);
//    end_time = omp_get_wtime() - start;
//    res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, z_B_s);
//    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);


//    free(z_B_s);
    free(z_BV.x);
//    free(z_B_o);
//    free(d_s.x);

}

void test_ils_search() {
    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    sils::SILS<double, int, true, false, n> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    start = omp_get_wtime();
    auto z_B = bsa.sils_search(&bsa.R_A, &bsa.y_A);
    end_time = omp_get_wtime() - start;
    auto res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, z_B);
    printf("Thread: ILS, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

    sils::scalarType<double, int> z_BS = {(double *) calloc(n, sizeof(double)), n};
    start = omp_get_wtime();
    z_BS = *bsa.sils_babai_search_serial(&z_BS);
    end_time = omp_get_wtime() - start;
    res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_BS);
    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

}

void test_run(int init_value) {

    std::cout << "Init, value: " << init_value << std::endl;
    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    sils::SILS<double, int, true, false, n> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    sils::scalarType<double, int> z_BS = {(double *) calloc(n, sizeof(double)), n};
    for (int i = 0; i < n; i++) {
        if (init_value != -1) {
            z_BS.x[i] = init_value;
        } else {
            z_BS.x[i] = bsa.x_R.x[i];
        }
    }

    start = omp_get_wtime();
    z_BS = *bsa.sils_babai_search_serial(&z_BS);
    end_time = omp_get_wtime() - start;
    auto res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_BS);
    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

    for (int proc = 80; proc >= 2; proc /= 2) {
        sils::scalarType<double, int> z_B = {(double *) calloc(n, sizeof(double)), n};
        sils::scalarType<double, int> z_B_p = {(double *) calloc(n, sizeof(double)), n};
        auto *update = (int *) malloc(n * sizeof(int));

        for (int i = 0; i < n; i++) {
            if (init_value != -1) {
                z_B.x[i] = init_value;
                z_B_p.x[i] = init_value;

            } else {
                z_B.x[i] = bsa.x_R.x[i];
                z_B_p.x[i] = bsa.x_R.x[i];
            }
            update[i] = 0;
        }

        start = omp_get_wtime();
        z_B = *bsa.sils_babai_search_omp(proc, 10, update, &z_B, &z_B_p);
        end_time = omp_get_wtime() - start;
        res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_B);
        printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", proc, 0, res, end_time);
        free(z_B.x);
        free(z_B_p.x);
        free(update);
    }

}

void plot_run() {

    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    sils::SILS<double, int, true, false, n> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    string fx = "res_" + to_string(n) + ".csv";
    ofstream file(fx);
    if (file.is_open()) {
        for (int init_value = -1; init_value <= 1; init_value++) {
            std::cout << "Init, value: " << init_value << std::endl;
            vector<double> res(50, 0), tim(50, 0), itr(50, 0);
            double omp_res = 0, omp_time = 0, num_iter = 0;
            double ser_res = 0, ser_time = 0;

            std::cout << "Vector Serial:" << std::endl;
            for (int i = 0; i < 10; i++) {
                sils::scalarType<double, int> z_BS{};
                z_BS.x = (double *) calloc(n, sizeof(double));
                z_BS.size = n;
                for (int l = 0; l < n; l++) {
                    if (init_value != -1) {
                        z_BS.x[l] = init_value;
                    } else {
                        z_BS.x[l] = bsa.x_R.x[l];
                    }
                }
//                sils::display_scalarType(z_BS);
                start = omp_get_wtime();
                z_BS = *bsa.sils_babai_search_serial(&z_BS);
                ser_time = omp_get_wtime() - start;
                ser_res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_BS);
                printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", ser_res, ser_time);
                res[0] += ser_res;
                tim[0] += ser_time;
            }

            file << init_value << "," << res[0] / 10 << "," << tim[0] / 10 << ",\n";

            std::cout << "OpenMP" << std::endl;
            int index = 0;
            for (int i = 0; i < 10; i++) {
                for (int n_proc = 80; n_proc >= 2; n_proc /= 2) {
                    sils::scalarType<double, int> z_B = {(double *) calloc(n, sizeof(double)), n};
                    sils::scalarType<double, int> z_B_p = {(double *) calloc(n, sizeof(double)), n};
                    auto *update = (int *) malloc(n * sizeof(int));

                    for (int m = 0; m < n; m++) {
                        z_B.x[m] = init_value;
                        z_B_p.x[m] = init_value;
                        update[m] = init_value;
                    }

                    start = omp_get_wtime();
                    z_B = *bsa.sils_babai_search_omp(n_proc, 10, update, &z_B, &z_B_p);
                    omp_time = omp_get_wtime() - start;
                    omp_res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_B);
                    printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", n_proc, 0, omp_res, omp_time);
                    free(z_B.x);
                    free(z_B_p.x);
                    free(update);

                    res[index] += omp_res;
                    tim[index] += omp_time;
                    itr[index] += 10;
                    index++;
                }
                index = 0;
            }
            index = 0;
            for (int n_proc = 80; n_proc >= 2; n_proc /= 2) {
                file << init_value << "," << n_proc << ","
                     << res[index] / 10 << ","
                     << tim[index] / 10 << ","
                     << itr[index] / 10 << ",\n";

                printf("Init value: %d, n_proc: %d, res :%f, num_iter: %f, Average time: %fs\n", init_value, n_proc,
                       res[index] / 10,
                       itr[index] / 10,
                       tim[index] / 10);
                index++;
            }
            file << "Next,\n";
        }
    }
    file.close();

}

int main() {
    std::cout << "Maximum Threads: " << omp_get_max_threads() << std::endl;
    plot_run();
    //test_ils_block_search();
    //test_ils_search();

    return 0;
}
