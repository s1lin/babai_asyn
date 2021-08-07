#include "../include/cils.h"

using namespace std;
using namespace cils::program_def;

namespace cils {

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
//        helper::mtimes(A, x_t, y_a);
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
        matlab::data::TypedArray<scalar> n_M = factory.createScalar<scalar>(n);
        matlabPtr->setVariable(u"m", std::move(m_M));
        matlabPtr->setVariable(u"n", std::move(n_M));
        /*
        matlabPtr->setVariable(u"K", std::move(m_M));
        matlabPtr->setVariable(u"N", std::move(n_M));

        // Call the MATLAB movsum function
        matlabPtr->eval(
                u" [s_bar_cur, s_bar1, s_bar2, y, H, HH, Piv] = "
                "simulations_SIC(K, N, 100, 'random', 1, false);");
        */

        matlabPtr->eval(
                u" [s_bar4, y, H, HH, Piv, s_bar1, s] = "
                "simulations_Block_Optimal(m, n, 3000, 3000, 10000, 'random', 1, false);");

        /*
        matlab::data::TypedArray<scalar> const A_A = matlabPtr->getVariable(u"H");
        matlab::data::TypedArray<scalar> const H_A = matlabPtr->getVariable(u"HH");
        matlab::data::TypedArray<scalar> const Z_A = matlabPtr->getVariable(u"Piv");
        matlab::data::TypedArray<scalar> const y_A = matlabPtr->getVariable(u"y");
        matlab::data::TypedArray<scalar> const x_1 = matlabPtr->getVariable(u"s_bar_cur");//v_a
        matlab::data::TypedArray<scalar> const x_2 = matlabPtr->getVariable(u"s_bar1");//x_t
        matlab::data::TypedArray<scalar> const x_3 = matlabPtr->getVariable(u"s_bar2");//x_r
        */
        matlab::data::TypedArray<scalar> const A_A = matlabPtr->getVariable(u"H");
        matlab::data::TypedArray<scalar> const H_A = matlabPtr->getVariable(u"HH");
        matlab::data::TypedArray<scalar> const Z_A = matlabPtr->getVariable(u"Piv");
        matlab::data::TypedArray<scalar> const y_A = matlabPtr->getVariable(u"y");
        matlab::data::TypedArray<scalar> const x_1 = matlabPtr->getVariable(u"s_bar1");
        matlab::data::TypedArray<scalar> const x_2 = matlabPtr->getVariable(u"s_bar4");
        matlab::data::TypedArray<scalar> const x_s = matlabPtr->getVariable(u"s");
//        matlab::data::TypedArray<scalar> const l_A = matlabPtr->getVariable(u"l");
//        matlab::data::TypedArray<scalar> const u_A = matlabPtr->getVariable(u"u");
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
//        i = 0;
//        for (auto r : x_1) {
//            v_a[i] = r;
//            i++;
//        }
        i = 0;
        for (auto r : x_2) {
            x_r[i] = r;
            i++;
        }
        i = 0;
        for (auto r : x_s) {
            x_t[i] = r;
            i++;
        }
//        i = 0;
//        for (auto r : u_A) {
//            u[i] = r;
//            i++;
//        }
        i = 0;
//        for (auto r : x_3) {
//            x_r[i] = r;
//            i++;
//        }

    }

/*    template<typename scalar, typename index, index m, index n>
//    void cils<scalar, index, m, n>::init_R() {
//        if (!is_matlab) {
//            if (is_qr) {
//                for (index i = 0; i < n; i++) {
//                    for (index j = 0; j < n; j++) {
//                        R_R[j * n + i] = R_Q[j * n + i];
//                    }
//                }
//                for (index i = 0; i < n; i++) {
//                    for (index j = i; j < n; j++) {
//                        R_A[(n * i) + j - ((i * (i + 1)) / 2)] = R_Q[j * n + i];
//                    }
//                }
//            } else {
//                for (index i = 0; i < n; i++) {
//                    for (index j = i; j < n; j++) {
//                        R_A[(n * i) + j - ((i * (i + 1)) / 2)] = R_R[j * n + i];
//                    }
//                }
//            }
//        } else {
//            for (index i = 0; i < n; i++) {
//                for (index j = i; j < n; j++) {
//                    R_A[(n * i) + j - ((i * (i + 1)) / 2)] = R_Q[j * n + i];
//                }
//            }
//        }
//        if (n <= 16) {
//            cout << endl;
//            for (index i = 0; i < n; i++) {
//                for (index j = i; j < n; j++) {
////                swap(R[i * n + j], R[j * n + i]);
//                    printf("%8.5f ", R_A[(n * i) + j - ((i * (i + 1)) / 2)]);
//                }
//                cout << endl;
//            }
//        }
   }
*/
    template<typename scalar, typename index, index m, index n>
    inline void matrix_vector_mult(const array<scalar, m * n> &Z, vector<scalar> *x) {
        array<scalar, n> x_c, x_z;
        x_c.fill(0);
        x_z.fill(0);
        for (index i = 0; i < n; i++) {
            x_z[i] = x->at(i);
        }

        helper::mtimes<scalar, index, n, 1>(Z, x_z, x_c);

        for (index i = 0; i < n; i++) {
            x->at(i) = x_c[i];
        }
    }

    template<typename scalar, typename index, index m, index n>
    inline void matrix_vector_mult(const array<scalar, m * n> &Z, const vector<scalar> &x, vector<scalar> &c) {
        array<scalar, n> x_z;
        array<scalar, m> x_c;
        x_c.fill(0);
        x_z.fill(0);
        for (index i = 0; i < n; i++) {
            x_z[i] = x[i];
        }

        helper::mtimes_Axy<scalar, index, m, n>(Z, x_z, x_c);

        for (index i = 0; i < m; i++) {
            c[i] = x_c[i];
        }
    }

}