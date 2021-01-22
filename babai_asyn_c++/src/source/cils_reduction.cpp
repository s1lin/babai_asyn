#include <cmath>
#include <cstring>
#include "../include/cils.h"

namespace cils {
    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_qr_decomposition_serial(const index eval, const index qr_eval) {

        index i, j, k, m, counter = 0;
        scalar error, time, sum;
        auto A_t = new scalar[n * n]();
        R->x = new scalar[n * n]();
        Q->x = new scalar[n * n]();

        time = omp_get_wtime();
        for (i = 0; i < n * n; i++) {
            A_t[i] = A->x[i];
        }

        for (k = 0; k < n; k++) {
            //Check if Q[][i-1] (the previous column) is computed.
            for (j = k; j < n; j++) {
                R->x[k * n + j] = 0;
                for (i = 0; i < n; i++) {
                    R->x[j * n + k] += Q->x[k * n + i] * A_t[j * n + i];
                }
                for (i = 0; i < n; i++) {
                    A_t[j * n + i] -= R->x[j * n + k] * Q->x[k * n + i];
                }
                //Calculate the norm(A)
                if (j == k) {
                    sum = 0;
                    for (i = 0; i < n; i++) {
                        sum += pow(A_t[k * n + i], 2);
                    }
                    R->x[k * n + k] = sqrt(sum);
                    for (i = 0; i < n; i++) {
                        Q->x[k * n + i] = A_t[k * n + i] / R->x[k * n + k];
                    }
                }
            }
        }

        time = omp_get_wtime() - time;

        if (eval || qr_eval) {
            error = qr_validation<scalar, index, n>(A, Q, R, R_A, eval, qr_eval);
//            cout << error << " ";
        }

        delete[] A_t;
        return {nullptr, time, (index) error};
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_qr_decomposition_omp(const index eval, const index qr_eval,
                                                      const index n_proc) {

        scalar error, time, sum = 0;

        auto A_t = new scalar[n * n]();
        R->x = new scalar[n * n]();
        Q->x = new scalar[n * n]();
        auto lock = new omp_lock_t[n]();

        time = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(sum)
        {

#pragma omp for schedule(static, 1)
            for (index i = 0; i < n; i++) {
                for (index j = 0; j < n; j++) {
                    A_t[i * n + j] = A->x[i * n + j];
                }
                omp_init_lock((&lock[i]));
                omp_set_lock(&lock[i]);
            }

            sum = 0;
            if (omp_get_thread_num() == 0) {
                // Calculation of ||A||
                for (index i = 0; i < n; i++) {
                    sum = sum + A_t[i] * A_t[i];
                }
                R->x[0] = sqrt(sum);
                for (index i = 0; i < n; i++) {
                    Q->x[i] = A_t[i] / R->x[0];
                }
                omp_unset_lock(&lock[0]);
            }

            for (index k = 1; k < n; k++) {
                //Check if Q[][i-1] (the previous column) is computed.
                omp_set_lock(&lock[k - 1]);
                omp_unset_lock(&lock[k - 1]);
#pragma omp for schedule(static, 1) nowait
                for (index j = 0; j < n; j++) {
                    if (j >= k) {
                        R->x[(k - 1) * n + j] = 0;
                        for (index i = 0; i < n; i++) {
                            R->x[j * n + (k - 1)] += Q->x[(k - 1) * n + i] * A_t[j * n + i];
                        }
                        for (index i = 0; i < n; i++) {
                            A_t[j * n + i] = A_t[j * n + i] - R->x[j * n + (k - 1)] * Q->x[(k - 1) * n + i];
                        }

                        //Only one thread calculates the norm(A)
                        //and unsets the lock for the next column.
                        if (j == k) {
                            sum = 0;
                            for (index i = 0; i < n; i++) {
                                sum = sum + A_t[k * n + i] * A_t[k * n + i];
                            }
                            R->x[k * n + k] = sqrt(sum);
                            for (index i = 0; i < n; i++) {
                                Q->x[k * n + i] = A_t[k * n + i] / R->x[k * n + k];
                            }
                            omp_unset_lock(&lock[k]);
                        }
                    }
                }
            }
        }

        time = omp_get_wtime() - time;
        if (eval || qr_eval) {
            error = qr_validation<scalar, index, n>(A, Q, R, R_A, eval, qr_eval);
        }
        for (index i = 0; i < n; i++) {
            omp_destroy_lock(&lock[i]);
        }
#pragma parallel omp cancellation point
#pragma omp flush
        delete[] lock;
        delete[] A_t;

        return {nullptr, time, (index) error};
    }

}
