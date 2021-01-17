#include <cmath>
#include <cstring>
#include "../include/cils.h"

namespace cils {
    template<typename scalar, typename index, bool is_read, index n>
    returnType <scalar, index>
    cils<scalar, index, is_read, n>::cils_qr_decomposition_serial(const index eval, const index qr_eval) {

        index i, j, k, m, counter = 0;
        scalar error, time, r_sum;
        auto t = new scalar[n * n]();

        time = omp_get_wtime();
        for (i = 0; i < n * n; i++) {
            t[i] = A->x[i];
        }

        for (k = 0; k < n; k++) {
            //Check if Q[][i-1] (the previous column) is computed.
            for (j = k; j < n; j++) {
                R->x[k * n + j] = 0;
                for (i = 0; i < n; i++) {
                    R->x[j * n + k] += Q->x[k * n + i] * t[j * n + i];
                }
                for (i = 0; i < n; i++) {
                    t[j * n + i] -= R->x[j * n + k] * Q->x[k * n + i];
                }
                //Calculate the norm(A)
                if (j == k) {
                    r_sum = 0;
                    for (i = 0; i < n; i++) {
                        r_sum += pow(t[k * n + i], 2);
                    }
                    R->x[k * n + k] = sqrt(r_sum);
                    for (i = 0; i < n; i++) {
                        Q->x[k * n + i] = t[k * n + i] / R->x[k * n + k];
                    }
                }
            }
        }

        for (i = 0; i < n; i++) {
            for (j = i; j < n; j++) {
                R_A->x[counter] = R->x[j * n + i];
                counter++;
            }
        }

        time = omp_get_wtime() - time;

        if (eval || qr_eval) {
            error = qr_validation<scalar, index, n>(A, Q, R, R_A, eval, qr_eval);
            cout << error << " ";
        }

        delete[] t;
        return {nullptr, time, (index) error};
    }

    template<typename scalar, typename index, bool is_read, index n>
    returnType <scalar, index>
    cils<scalar, index, is_read, n>::cils_qr_decomposition_omp(const index eval, const index qr_eval,
                                                               const index n_proc) {

        index i, j, k, m;
        scalar error, time, r_sum = 0;
        omp_lock_t *lock;
        auto t = new scalar[n * n]();
        lock = (omp_lock_t *) malloc(n * sizeof(omp_lock_t));
        time = omp_get_wtime() - time;

#pragma omp parallel default(shared) num_threads(n_proc) private(i, j, k, r_sum)
        {

#pragma omp for schedule(static, 1)
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    t[i * n + j] = A->x[i * n + j];
                }
                omp_set_lock(&lock[i]);
            }

            //Run QR decomposition
            //First column of ( Q[][0] )
            //Thread 0 calculates the 1st column
            //and unsets the 1st lock.
            r_sum = 0;
            if (omp_get_thread_num() == 0) {
                // Calculation of ||A||
                for (i = 0; i < n; i++) {
                    r_sum = r_sum + t[i] * t[i];
                }
                R->x[0] = sqrt(r_sum);
                for (i = 0; i < n; i++) {
                    Q->x[i] = t[i] / R->x[0];
                }
                omp_unset_lock(&lock[0]);
            }

            for (k = 1; k < n; k++) {
                //Check if Q[][i-1] (the previous column) is computed.
                omp_set_lock(&lock[k - 1]);
                omp_unset_lock(&lock[k - 1]);

#pragma omp for schedule(static,1) nowait
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
                            omp_unset_lock(&lock[k]);
                        }
                    }
                }
            }
        }


        time = omp_get_wtime() - time;
        if (eval || qr_eval) {
            error = qr_validation<scalar, index, n>(A, Q, R, R_A, eval, qr_eval);
            cout << error << " ";
        }

        free(lock);
        delete[] t;

        return {nullptr, time, (index) error};
    }

}
