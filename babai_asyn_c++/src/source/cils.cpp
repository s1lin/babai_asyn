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
    returnType<scalar, index>
    lll_validation(const coder::array<scalar, 2U> &R_R,
                   const coder::array<scalar, 2U> &R_Q,
                   const coder::array<scalar, 2U> &Z,
                   const index verbose) {
        printf("[ INFO: in LLL validation]\n");
        scalar sum, error = 0, det;

        if (verbose) {
            printf("[ Printing R]\n");
            for (index i = 0; i < n; i++) {
                for (index j = 0; j < n; j++) {
                    printf("%8.5f ", R_R[j * n + i]);
                }
                printf("\n");
            }
            printf("[ Printing Z]\n");
            for (index i = 0; i < n; i++) {
                for (index j = 0; j < n; j++) {
                    printf("%8.5f ", Z[j * n + i]);
                }
                printf("\n");
            }
        }
        coder::array<double, 2U> R_Z, Q_C;

        R_Z.set_size(n, n);
        Q_C.set_size(n, n);
        for (index i = 0; i < n; i++) {
            for (index j = 0; j < n; j++) {
                R_Z[i * n + j] = 0;
                Q_C[i * n + j] = 0;
            }
        }
        // 'eo_sils_reduction:109' Q1 = R_*Z_C/R_C;
        coder::internal::blas::mtimes(R_Q, Z, R_Z);
        coder::internal::mrdiv(R_Z, R_R);
        //  Q1
        // 'eo_sils_reduction:111' d = det(Q1*Q1');
        coder::internal::blas::b_mtimes(R_Z, R_Z, Q_C);
        det = coder::det(Q_C);

        index i = 1, pass = true;
        vector<scalar> fail_index(n, 0);
        while (i < n) {

            // 'eo_sils_reduction:50' i1 = i-1;
            // 'eo_sils_reduction:51' zeta = round(R_R(i1,i) / R_R(i1,i1));
            scalar zeta = std::round(R_R[i + n * (i - 1) - 2] / R_R[i + n * (i - 2) - 2]);
            // 'eo_sils_reduction:52' alpha = R_R(i1,i) - zeta * R_R(i1,i1);
            scalar s = R_R[(i + n * (i - 2)) - 2];
            scalar alpha = R_R[(i + n * (i - 1)) - 2] - zeta * s;
            // 'eo_sils_reduction:53' if R_R(i1,i1)^2 > (1 + 1.e-0) * (alpha^2 +
            // R_R(i,i)^2)
            scalar a = R_R[(i + n * (i - 1)) - 1];
            if (s * s > 2.0 * (alpha * alpha + a * a)) {
                // 'eo_sils_reduction:54' if zeta ~= 0
                //  Perform a size reduction on R_R(k-1,k)
                // 'eo_sils_reduction:56' f = true;
                pass = false;
                fail_index[i] = 1;
            }
            i++;
        }

        return {fail_index, det, (scalar) pass};
    }


    template<typename scalar, typename index, index n>
    void cils<scalar, index, n>::init() {
        // Start MATLAB engine synchronously
//        using namespace matlab::engine;
//
//        // Start MATLAB engine synchronously
//        std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();
//
//        //Create MATLAB data array factory
//        matlab::data::ArrayFactory factory;
//
//        // Call the MATLAB movsum function
//        matlabPtr->eval(u" A = randn(512);");
//        matlab::data::TypedArray<scalar> const A_A = matlabPtr->getVariable(u"A");
//
//        index i = 0;
//        A.set_size(n, n);
//        for (auto r : A_A) {
//            A[i] = r;
//            ++i;
//        }

//        coder::qr(A, Q, R_Q);

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
//        for (index i = 0; i < n; i++) {
//            y_a[i] = 0;
//        }
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