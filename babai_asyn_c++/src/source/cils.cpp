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
        }

        //Set Z to Eye:
        coder::eye(n, Z);
    }

    template<typename scalar, typename index, index n>
    void cils<scalar, index, n>::init_y() {

        coder::internal::blas::mtimes(R_Q, x_t, y_r);
        coder::internal::blas::mtimes(Q, v_a, v_q);

        for (index i = 0; i < n; i++) {
            y_r[i] += v_q[i];
            y_q[i] = y_r[i];
        }
    }


    template<typename scalar, typename index, index n>
    void cils<scalar, index, n>::init_R() {
        if(!is_matlab) {
            for (index i = 0; i < n; i++) {
                for (index j = 0; j < n; j++) {
                    R_R[i * n + j] = R_Q[j * n + i];
                }
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

    template<typename scalar, typename index, index n>
    inline void vector_permutation(const coder::array<scalar, 2U> &Z,
                                   vector<scalar> *x) {
        vector<scalar> x_t(n, 0);
        for (index i = 0; i < n; i++) {
            scalar sum = 0;
            for (index j = 0; j < n; j++) {
                sum += Z[j * n + i] * x->at(j);
            }
            x_t[i] = sum;
        }
        for (index i = 0; i < n; i++){
            x->at(i) = x_t[i];
        }
    }

}