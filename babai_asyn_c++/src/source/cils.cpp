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
    template<typename scalar, typename index, index m, index n>
    scalar qr_validation(const array<scalar, m * n> &A,
                         const array<scalar, m * m> &Q,
                         const array<scalar, m * n> &R,
                         const index eval, const index verbose) {
        index i, j, k;
        scalar sum, error = 0;

        if (eval == 1) {
            array<scalar, m * n> A_T;
            A_T.fill(0);
            helper::internal::blas::mtimes<scalar, index, m, n>(Q, R, A_T);

            if (verbose) {
                printf("\n[ Print Q:]\n");
                display_matrix<scalar, index, m, m>(Q);
                printf("\n[ Print R:]\n");
                display_matrix<scalar, index, m, n>(R);
                printf("\n[ Print A:]\n");
                display_matrix<scalar, index, m, n>(A);
                printf("\n[ Print Q*R:]\n");
                display_matrix<scalar, index, m, n>(A_T);
            }

            for (i = 0; i < m * n; i++) {
                error += fabs(A_T[i] - A[i]);
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
    template<typename scalar, typename index, index m, index n>
    returnType<scalar, index>
    lll_validation(const array<scalar, m * n> &R_R, const array<scalar, m * n> &R_Q, const array<scalar, m * n> &Z,
                   const index verbose, std::unique_ptr<matlab::engine::MATLABEngine> &matlabPtr) {
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


        //Create MATLAB data array factory
        matlab::data::ArrayFactory factory;

        // Call the MATLAB movsum function
        matlab::data::TypedArray<double> R_M = factory.createArray(
                {static_cast<unsigned long>(m), static_cast<unsigned long>(n)}, R_Q.begin(), R_Q.end());
        matlab::data::TypedArray<double> Z_M = factory.createArray(
                {static_cast<unsigned long>(m), static_cast<unsigned long>(n)}, Z.begin(), Z.end());
        matlab::data::TypedArray<double> RCM = factory.createArray(
                {static_cast<unsigned long>(m), static_cast<unsigned long>(n)}, R_R.begin(), R_R.end());
        matlabPtr->setVariable(u"R_M", std::move(R_M));
        matlabPtr->setVariable(u"Z_C", std::move(Z_M));
        matlabPtr->setVariable(u"R_C", std::move(RCM));

        // Call the MATLAB movsum function
        matlabPtr->eval(u" d = det(mldivide(R_M * Z_C, R_C));");
//        matlabPtr->eval(u" A = magic(n);");

        matlab::data::TypedArray<double> const d = matlabPtr->getVariable(u"d");
        int i = 0;
        for (auto r : d) {
            det = r;
            ++i;
        }

        index pass = true;
        vector<scalar> fail_index(n, 0);
        i = 2;
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


    template<typename scalar, typename index, index m, index n>
    void cils<scalar, index, m, n>::init() {

        //Create MATLAB data array factory
        matlab::data::ArrayFactory factory;

        // Call the MATLAB movsum function
        matlab::data::TypedArray<scalar> k_M = factory.createScalar<scalar>(program_def::k);
        matlab::data::TypedArray<scalar> SNR_M = factory.createScalar<scalar>(program_def::SNR);
        matlab::data::TypedArray<scalar> m_M = factory.createScalar<scalar>(n);
        matlabPtr->setVariable(u"k", std::move(k_M));
        matlabPtr->setVariable(u"n", std::move(m_M));
        matlabPtr->setVariable(u"SNR", std::move(SNR_M));

        // Call the MATLAB movsum function
        matlabPtr->eval(u" [A, y, x_t] = gen(k, n, SNR);");
//        matlabPtr->eval(u" A = magic(n);");

        matlab::data::TypedArray<scalar> const A_A = matlabPtr->getVariable(u"A");
        matlab::data::TypedArray<scalar> const y_M = matlabPtr->getVariable(u"y");
        matlab::data::TypedArray<scalar> const x_M = matlabPtr->getVariable(u"x_t");

        index i = 0;
        for (auto r : A_A) {
            A[i] = r;
            ++i;
        }
        i = 0;
        for (auto r : y_M) {
            y_a[i] = r;
            ++i;
        }
        i = 0;
        for (auto r : x_M) {
            x_t[i] = r;
            ++i;
        }

//        helper::qr(A, Q, R_Q);

//        std::random_device rd;
//        std::mt19937 gen(rd());
//        //mean:0, std:sqrt(1/2). same as matlab.
//        std::normal_distribution<scalar> A_norm_dis(0, sqrt(0.5)), v_norm_dis(0, sigma);
//        //Returns a new random number that follows the distribution's parameters associated to the object (version 1) or those specified by parm
//        std::uniform_int_distribution<index> int_dis(-pow(2, qam - 1), pow(2, qam - 1) - 1);
//
//        for (index i = 0; i < n / 2; i++) {
//            for (index j = 0; j < n / 2; j++) {
//                A[i + j * n] = 2 * A_norm_dis(gen);
//                A[i + n / 2 + j * n] = -2 * A_norm_dis(gen);
//                A[i + n / 2 + (j + n / 2) * n] = A[i + j * n];
//                A[i + (j + n / 2) * n] = -A[i + n / 2 + j * n];
//            }
//            x_t[i] = (pow(2, qam) + 2 * int_dis(gen)) / 2;
//            v_a[i] = v_norm_dis(gen);
//            x_t[i + n / 2] = (pow(2, qam) + 2 * int_dis(gen)) / 2;
//            v_a[i + n / 2] = v_norm_dis(gen);
//        }
//
//        scalar sum = 0;
//
//        //init y_a
//        helper::internal::blas::mtimes(A, x_t, y_a);
//        for (index i = 0; i < n; i++) {
//            y_a[i] += v_a[i];
//        }
////        for (index i = 0; i < n; i++) {
////            y_a[i] = 0;
////        }
//        if (n <= 16)
//            display_vector<scalar, index>(y_a);
//
//        //Set Z to Eye:
//
//        helper::eye(n, Z);
    }

    template<typename scalar, typename index, index m, index n>
    void cils<scalar, index, m, n>::init_ud() {

        //Create MATLAB data array factory
        matlab::data::ArrayFactory factory;

        // Call the MATLAB movsum function
        matlab::data::TypedArray<scalar> m_M = factory.createScalar<scalar>(m);
        matlab::data::TypedArray<scalar> SNR_M = factory.createScalar<scalar>(program_def::SNR);
        matlab::data::TypedArray<scalar> n_M = factory.createScalar<scalar>(n);
        matlabPtr->setVariable(u"m", std::move(m_M));
        matlabPtr->setVariable(u"n", std::move(n_M));
        matlabPtr->setVariable(u"SNR", std::move(SNR_M));

        // Call the MATLAB movsum function
        matlabPtr->eval(
                u" [s_bar1_for_next, y, H, HH_SIC, Piv_SIC, s_bar1] = simulations_IP(m, SNR, n, 'random', 1);");
//        matlabPtr->eval(u" A = magic(n);");

        matlab::data::TypedArray<scalar> const A_A = matlabPtr->getVariable(u"H");
        matlab::data::TypedArray<scalar> const H_A = matlabPtr->getVariable(u"HH_SIC");
        matlab::data::TypedArray<scalar> const Z_A = matlabPtr->getVariable(u"Piv_SIC");
        matlab::data::TypedArray<scalar> const y_A = matlabPtr->getVariable(u"y");
        matlab::data::TypedArray<scalar> const x_M = matlabPtr->getVariable(u"s_bar1_for_next");
        matlab::data::TypedArray<scalar> const x_M_1 = matlabPtr->getVariable(u"s_bar1");

        index i = 0;

        for (auto r : A_A) {
            A[i] = r;
            i++;
        }

        i = 0;
        for (auto r : H_A) {
            H[i] = r;
            i++;
        }
        i = 0;
        for (auto r : y_A) {
            y_a[i] = r;
            i++;
        }
        i = 0;
//        for (auto r : x_M_1) {
//            x_t[i] = r;
//            i++;
//        }

    }

    template<typename scalar, typename index, index m, index n>
    void cils<scalar, index, m, n>::init_y() {

        for (index i = 0; i < n; i++) {
            scalar sum = 0;
            for (index j = 0; j < n; j++) {
                sum += Q[i * n + j] * y_a[j]; //Q'*y;
            }
            y_q[i] = sum;
            y_r[i] = y_q[i];
        }
    }


    template<typename scalar, typename index, index m, index n>
    void cils<scalar, index, m, n>::init_R() {
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

    template<typename scalar, typename index, index m, index n>
    inline void vector_permutation(const array<scalar, m * n> &Z, vector<scalar> *x) {
        array<scalar, n> x_c, x_z;
        x_c.fill(0);
        x_z.fill(0);
        for (index i = 0; i < n; i++) {
            x_z[i] = x->at(i);
        }

        helper::internal::blas::mtimes<scalar, index, m, n, 1>(Z, x_z, x_c);

        for (index i = 0; i < n; i++) {
            x->at(i) = x_c[i];
        }
    }

}