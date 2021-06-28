#include <algorithm>

#include "../include/cils.h"

using namespace std;
using namespace cils::program_def;

namespace cils {

    /**
     * Evaluating the decomposition
     * @tparam scalar
     * @tparam index
     * @tparam n
     * @param A
     * @param Q
     * @param R
     * @param eval
     * @return
     */
    template<typename scalar, typename index, index n>
    scalar qr_validation(const coder::array<scalar, 2U> &A,
                         const coder::array<scalar, 2U> &Q,
                         const coder::array<scalar, 2U> &R,
                         const index eval, const index verbose) {
        index i, j, k;
        scalar sum, error = 0;

        if (eval == 1) {
            if (verbose) {
                printf("\nPrinting A...\n");
                for (i = 0; i < n; i++) {
                    for (j = 0; j < n; j++) {
                        printf("%8.5f ", A[i * n + j]);
                    }
                    printf("\n");
                }
            }

            coder::array<scalar, 2U> A_T;
            coder::internal::blas::mtimes(Q, R, A_T);

            if (verbose) {
                printf("\nQ*R (Init A matrix) : \n");
                for (i = 0; i < n; i++) {
                    for (j = 0; j < n; j++) {
                        printf("%8.5f ", A_T[i * n + j]);
                    }
                    printf("\n");
                }
            }

            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    error += fabs(A_T[i * n + j] - A[i * n + j]);
                }
            }
        }

        return error;
    }


    template<typename scalar, typename index, index n>
    void cils<scalar, index, n>::init() {
        std::random_device rd;
        std::mt19937 gen(rd());
        //mean:0, std:sqrt(1/2). same as matlab.
        std::normal_distribution<scalar> A_norm_dis(0, sqrt(0.5)), v_norm_dis(0, sigma);
        //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
        std::uniform_int_distribution<index> int_dis(-pow(2, qam - 1), pow(2, qam - 1) - 1);

        for (index i = 0; i < n / 2; i++) {
            for (index j = 0; j < n / 2; j++) {
                A[i + j * n] = 2 * A_norm_dis(gen);
                A[i + n / 2 + j * n] = -2 * A_norm_dis(gen);
                A[i + n / 2 + (j + n / 2) * n] = A[i + j * n];
                A[i + (j + n / 2) * n] = -A[i + n / 2 + j * n];
            }
            x_t[i] = (pow(2, qam) + 2 * int_dis(gen)) / 2;
            v_a[i] = v_norm_dis(gen);
            x_t[i + n / 2] = (pow(2, qam) + 2 * int_dis(gen)) / 2;
            v_a[i + n / 2] = v_norm_dis(gen);
        }

        scalar sum = 0;

        //init y_a
        coder::internal::blas::mtimes(A, x_t, y_a);
        for (index i = 0; i < n; i++) {
            y_a[i] += v_a[i];
        }
        if (n <= 16)
            display_vector<scalar, index>(y_a);

        //Set Z to Eye:
        coder::eye(n, Z);
    }

    template<typename scalar, typename index, index n>
    void cils<scalar, index, n>::init_y() {

        for (index i = 0; i < n; i++) {
            scalar sum = 0;
            for (index j = 0; j < n; j++) {
                sum += Q[i * n + j] * y_a[j]; //Q'*y;
            }
            y_q[i] = sum;
            y_r[i] = y_q[i];
        }
    }


    template<typename scalar, typename index, index n>
    void cils<scalar, index, n>::init_R() {
        if (!is_matlab) {
            if (is_qr) {
                for (index i = 0; i < n; i++) {
                    for (index j = 0; j < n; j++) {
                        R_R[j * n + i] = R_Q[j * n + i];
                    }
                }
                for (index i = 0; i < n; i++) {
                    for (index j = i; j < n; j++) {
                        R_A[(n * i) + j - ((i * (i + 1)) / 2)] = R_Q[j * n + i];
                    }
                }
            } else {
                for (index i = 0; i < n; i++) {
                    for (index j = i; j < n; j++) {
                        R_A[(n * i) + j - ((i * (i + 1)) / 2)] = R_R[j * n + i];
                    }
                }
            }
        } else {
            for (index i = 0; i < n; i++) {
                for (index j = i; j < n; j++) {
                    R_A[(n * i) + j - ((i * (i + 1)) / 2)] = R_Q[j * n + i];
                }
            }
        }
        if (n <= 16) {
            cout << endl;
            for (index i = 0; i < n; i++) {
                for (index j = i; j < n; j++) {
//                swap(R[i * n + j], R[j * n + i]);
                    printf("%8.5f ", R_A[(n * i) + j - ((i * (i + 1)) / 2)]);
                }
                cout << endl;
            }
        }
    }

    template<typename scalar, typename index, index n>
    inline void vector_permutation(const coder::array<scalar, 2U> &Z,
                                   vector<scalar> *x) {
        coder::array<scalar, 1U> x_c, x_z;
        x_z.set_size(n);

        for (index i = 0; i < n; i++) {
            x_z[i] = x->at(i);
        }

        coder::internal::blas::mtimes(Z, x_z, x_c);

        for (index i = 0; i < n; i++) {
            x->at(i) = x_c[i];
        }
    }

}