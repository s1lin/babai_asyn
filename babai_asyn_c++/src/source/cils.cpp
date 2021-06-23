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
                         const index eval) {
        index i, j, k;
        scalar sum, error = 0;

        if (eval == 1) {
            printf("\nPrinting A...\n");
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    printf("%8.5f ", A[i * n + j]);
                }
                printf("\n");
            }

            coder::array<scalar, 2U> A_T;
            A_T.set_size(n, n);
            coder::internal::blas::mtimes(Q, R, A_T);

            printf("\nQ*R (Init A matrix) : \n");
            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    printf("%8.5f ", A_T[j * n + i]);
                }
                printf("\n");
            }

            for (i = 0; i < n; i++) {
                for (j = 0; j < n; j++) {
                    error += fabs(A_T[j * n + i] - A[j * n + i]);
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
                A[j + i * n] = 2 * A_norm_dis(gen);
                A[j + n / 2 + i * n] = -2 * A_norm_dis(gen);
                A[j + n / 2 + (i + n / 2) * n] = A[j + i * n];
                A[j + (i + n / 2) * n] = -A[j + n / 2 + i * n];
            }
            x_t[i] = (pow(2, qam) + 2 * int_dis(gen)) / 2;
            v_a[i] = v_norm_dis(gen);
            x_t[i + n / 2] = (pow(2, qam) + 2 * int_dis(gen)) / 2;
            v_a[i + n / 2] = v_norm_dis(gen);
        }
        scalar sum = 0;
        for (index i = 0; i < n; i++) {
            sum = 0;
            for (index j = 0; j < n; j++) {
                sum += A[i * n + j] * x_t[j];
            }
            y_a[i] = sum + v_a[i];
            Z[i * n + i] = 1;
        }
    }

    template<typename scalar, typename index, index n>
    void cils<scalar, index, n>::init_y() {
        scalar rx = 0, qv = 0;
        index ri, rai;
//        for (index i = 0; i < n; i++) {
//            rx = qv = 0;
//            for (index j = 0; j < n; j++) {
//                if (i <= j) { //For some reason the QR stored in Transpose way?
//                    ri = i * n + j;
//                    rai = ri - (i * (i + 1)) / 2;
//                    R_A[rai] = R[ri];
//                    rx += R[ri] * x_t[j];
//                }
//                qv += Q[j * n + i] * v_a[j]; //Transpose Q
//            }
////            cout << endl;
//            y_A[i] = rx + qv;
//        }
//Not use python:
        scalar R_T[n * n] = {};
//        for (index i = 0; i < n; i++) {
//            for (index j = 0; j < n; j++) {
////                swap(R[i * n + j], R[j * n + i]);
//                cout << R[j * n + i] << " ";
//            }
//            cout << endl;
//        }

//        scalar tmp;
//        for (index i = 0; i < n; i++) {
//            for (index j = 0; j < n; j++) {
//                R_T[i * n + j] = R[j * n + i];
//            }
//        }
//
//        for (index i = 0; i < n; i++) {
//            for (index j = 0; j < n; j++) {
//                R[i * n + j] = R_T[i * n + j];
//            }
//        }

        for (index i = 0; i < n; i++) {
            rx = qv = 0;
            for (index j = 0; j < n; j++) {
                if (i <= j) { //For some reason the QR stored in Transpose way?
//                    R[i * n + j] = R_T[i * n + j];
//                    R_A[(n * i) + j - ((i * (i + 1)) / 2)] = R[i * n + j];
//                    rx += R[i * n + j] * x_t[j];
                    rx += R_Q[j * n + i] * x_t[j];
                }
                qv += Q[j * n + i] * v_a[j]; //Transpose Q
            }
            y_q[i] = rx + qv;
            y_r[i] = rx + qv;
        }
    }


    template<typename scalar, typename index, index n>
    void cils<scalar, index, n>::init_R() {

        for (index i = 0; i < n; i++) {
            for (index j = 0; j < n; j++) {
                R_R[i * n + j] = R_Q[j * n + i];
            }
        }
        for (index i = 0; i < n; i++) {
            for (index j = 0; j < n; j++) { //TO i
                R_Q[i * n + j] = R_R[i * n + j];
                if (j >= i)
                    R_A[(n * i) + j - ((i * (i + 1)) / 2)] = R_Q[i * n + j];
            }
        }

//        for (index i = 0; i < n; i++) {
//            for (index j = i; j < n; j++) {
////                swap(R[i * n + j], R[j * n + i]);
//                cout << R_A[(n * i) + j - ((i * (i + 1)) / 2)]  << " ";
//            }
//            cout << endl;
//        }
    }

}