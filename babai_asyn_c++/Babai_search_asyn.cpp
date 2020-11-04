#include <cstring>
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
    scalar *
    Babai_search_asyn<scalar, index, is_read, is_write, n>::search_omp(const index n_proc, const index nswp,
                                                                       index *update, scalar *z_B,
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
    vector<scalar> Babai_search_asyn<scalar, index, is_read, is_write, n>::search_vec(vector<scalar> z_B) {
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
    scalar *Babai_search_asyn<scalar, index, is_read, is_write, n>::sils_search(scalar *z_B) {

        auto *prsd = (scalar *) calloc(n, sizeof(scalar));
        auto *c = (scalar *) calloc(n, sizeof(scalar));
        auto *z_H = (scalar *) calloc(n, sizeof(scalar));
        auto *d = (index *) calloc(n, sizeof(index));

        scalar newprsd, gamma, b_R;
        index k = n - 1, i, j, r_tmp, loop_ub;

        //  Initial squared search radius
        double beta = INFINITY;

        c[n - 1] = y_A[n - 1] / R_A[size_R_A - 1];
        z_B[n - 1] = round(c[n - 1]);
        gamma = R_A[size_R_A - 1] * (c[n - 1] - z_B[n - 1]);

        //  Determine enumeration direction at level n
        d[n - 1] = c[n - 1] > z_B[n - 1] ? 1 : -1;

        while (true) {
            newprsd = prsd[k] + gamma * gamma;
            if (newprsd < beta) {
                if (k != 0) {
                    k--;
                    double sum = 0;
                    for (index col = k + 1; col < n; col++) {
                        sum += R_A[(n * k) + col - ((k * (k + 1)) / 2)] * z_B[col];
                    }
                    scalar s = y_A[k] - sum;
                    prsd[k] = newprsd;
                    c[k] = s / R_A[(n * k) + k - ((k * (k + 1)) / 2)];

                    z_B[k] = round(c[k]);
                    gamma = R_A[(n * k) + k - ((k * (k + 1)) / 2)] * (c[k] - z_B[k]);
                    d[k] = c[k] > z_B[k] ? 1 : -1;

                } else {
                    beta = newprsd;
                    //Deep Copy of the result
                    std::memcpy(z_H, z_B, sizeof(scalar) * n);
                    z_B[0] += d[0];
                    gamma = R_A[0] * (c[0] - z_B[0]);
                    d[0] = d[0] > 0 ? -d[0] - 1 : -d[0] + 1;
                }
            } else {
                if (k == n - 1) break;
                else {
                    k++;
                    z_B[k] += d[k];
                    gamma = R_A[(n * k) + k - ((k * (k + 1)) / 2)] * (c[k] - z_B[k]);
                    d[k] = d[k] > 0 ? -d[k] - 1 : -d[k] + 1;
                }
            }
        }
        std::memcpy(z_B, z_H, sizeof(scalar) * n);
        free(c);
        free(d);
        free(prsd);
        free(z_H);

        return z_B;
    }
}

void test_run(int init_value);

void plot_run();

void test_ils_search();

const int n = 20;

int main() {
    std::cout << "Maximum Threads: " << omp_get_max_threads() << std::endl;
    //plot_run();
    test_ils_search();

    return 0;
}

void test_ils_search() {
    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    babai::Babai_search_asyn<double, int, true, false, n> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    auto *z_B = (double *) malloc(n * sizeof(double));
    start = omp_get_wtime();
    z_B = bsa.sils_search(z_B);
    end_time = omp_get_wtime() - start;
    double res = babai::find_residual(n, bsa.R_A, bsa.y_A, z_B);
    printf("Thread: ILS, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

    vector<double> z_BV(n, 0);
    start = omp_get_wtime();
    vector<double> z = bsa.search_vec(z_BV);
    end_time = omp_get_wtime() - start;
    res = babai::find_residual(n, bsa.R_A, bsa.y_A, z.data());
    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

}

void test_run(int init_value) {

    std::cout << "Init, value: " << init_value << std::endl;
    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    babai::Babai_search_asyn<double, int, true, false, n> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    vector<double> z_BV(n, init_value);
    if (init_value == -1) {
        for (int i = 0; i < n; i++) {
            z_BV[i] = bsa.x_R[i];
        }
    }

    start = omp_get_wtime();
    vector<double> z = bsa.search_vec(z_BV);
    end_time = omp_get_wtime() - start;
    double res = babai::find_residual(n, bsa.R_A, bsa.y_A, z.data());
    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

    for (int proc = 80; proc >= 2; proc /= 2) {
        auto *z_B = (double *) malloc(n * sizeof(double));
        auto *z_B_p = (double *) malloc(n * sizeof(double));
        auto *update = (int *) malloc(n * sizeof(int));

        for (int i = 0; i < n; i++) {
            if (init_value != -1) {
                z_B[i] = init_value;
                z_B_p[i] = init_value;

            } else {
                z_B[i] = bsa.x_R[i];
                z_B_p[i] = bsa.x_R[i];
            }
            update[i] = 0;
        }

        start = omp_get_wtime();
        z_B = bsa.search_omp(proc, 10, update, z_B, z_B_p);
        end_time = omp_get_wtime() - start;
        res = babai::find_residual(n, bsa.R_A, bsa.y_A, z_B);
        printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", proc, 0, res, end_time);
        free(z_B);
        free(z_B_p);
        free(update);
    }

}

void plot_run() {

    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    babai::Babai_search_asyn<double, int, true, false, 4096> bsa(0.1);
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
                vector<double> z_BV(n, init_value);
                if (init_value == -1) {
                    for (int l = 0; l < n; l++) {
                        z_BV[l] = bsa.x_R[l];
                    }
                }
                start = omp_get_wtime();
                vector<double> z = bsa.search_vec(z_BV);
                ser_time = omp_get_wtime() - start;
                ser_res = babai::find_residual(n, bsa.R_A, bsa.y_A, z.data());
                printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", ser_res, ser_time);
                res[0] += ser_res;
                tim[0] += ser_time;
            }

            file << init_value << "," << res[0] / 10 << "," << tim[0] / 10 << ",\n";

            std::cout << "OpenMP" << std::endl;
            int index = 0;
            for (int i = 0; i < 10; i++) {
                for (int n_proc = 80; n_proc >= 2; n_proc /= 2) {
                    auto *z_B = (double *) malloc(n * sizeof(double));
                    auto *z_B_p = (double *) malloc(n * sizeof(double));
                    auto *update = (int *) malloc(n * sizeof(int));

                    for (int m = 0; m < n; m++) {
                        z_B[m] = init_value;
                        z_B_p[m] = init_value;
                        update[m] = init_value;
                    }

                    start = omp_get_wtime();
                    z_B = bsa.search_omp(n_proc, 10, update, z_B, z_B_p);
                    omp_time = omp_get_wtime() - start;
                    omp_res = babai::find_residual(n, bsa.R_A, bsa.y_A, z_B);
                    printf("Thread: %d, Sweep: %d, Res: %.5f, Run time: %fs\n", n_proc, 0, omp_res, omp_time);
                    free(z_B);
                    free(z_B_p);
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