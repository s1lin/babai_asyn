#include "../include/cils.h"

using namespace std;
using namespace cils::program_def;

namespace cils {

    template<typename scalar, typename index, index m, index n>
    void cils<scalar, index, m, n>::init(index rank) {
        //Create MATLAB data array factory
        scalar *size = (double *) malloc(1 * sizeof(double)), *p;

        if (rank == 0) {

            matlab::data::ArrayFactory factory;

            // Call the MATLAB movsum function
            matlab::data::TypedArray<scalar> k_M = factory.createScalar<scalar>(this->qam);
            matlab::data::TypedArray<scalar> m_M = factory.createScalar<scalar>(m);
            matlab::data::TypedArray<scalar> n_M = factory.createScalar<scalar>(n);
            matlab::data::TypedArray<scalar> SNR = factory.createScalar<scalar>(snr);
            matlab::data::TypedArray<scalar> MIT = factory.createScalar<scalar>(search_iter);
            matlabPtr->setVariable(u"k", std::move(k_M));
            matlabPtr->setVariable(u"m", std::move(m_M));
            matlabPtr->setVariable(u"n", std::move(n_M));
            matlabPtr->setVariable(u"SNR", std::move(SNR));
            matlabPtr->setVariable(u"max_iter", std::move(MIT));

            // Call the MATLAB movsum function
            matlabPtr->eval(
                    u" [A, x_t, v, y, sigma, res, permutation, size_perm] = gen_problem(k, m, n, SNR, max_iter);");

            matlab::data::TypedArray<scalar> const A_A = matlabPtr->getVariable(u"A");
            matlab::data::TypedArray<scalar> const y_M = matlabPtr->getVariable(u"y");
            matlab::data::TypedArray<scalar> const x_M = matlabPtr->getVariable(u"x_t");
            matlab::data::TypedArray<scalar> const res = matlabPtr->getVariable(u"res");
            matlab::data::TypedArray<scalar> const per = matlabPtr->getVariable(u"permutation");
            matlab::data::TypedArray<scalar> const szp = matlabPtr->getVariable(u"size_perm");


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
            i = 0;
            for (auto r : res) {
                this->init_res = r;
                ++i;
            }
            i = 0;
            for (auto r : res) {
                this->init_res = r;
                ++i;
            }

            i = 0;
            for (auto r : szp) {
                size[0] = r;
                ++i;
            }
            p = (scalar *) malloc(n * size[0] * sizeof(scalar));
            i = 0;
            for (auto r : per) {
                p[i] = r;
                ++i;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Bcast(&size[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank != 0)
            p = (scalar *) malloc(n * size[0] * sizeof(scalar));

        MPI_Bcast(&p[0], (int) size[0] * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);

        index i = 0;
        index k1 = 0;
        permutation.resize((int) size[0] + 1);
        permutation[k1] = vector<scalar>(n);
        permutation[k1].assign(n, 0);
        for (index iter = 0; iter < (int) size[0] * n; iter++) {
            permutation[k1][i] = p[iter];
            i = i + 1;
            if (i == n) {
                i = 0;
                k1++;
                permutation[k1] = vector<scalar>(n);
                permutation[k1].assign(n, 0);
            }
        }
        i = 0;
    }

    template<typename scalar, typename index, index m, index n>
    void cils<scalar, index, m, n>::init_ud() {

        //Create MATLAB data array factory
        matlab::data::ArrayFactory factory;

        // Call the MATLAB movsum function
        matlab::data::TypedArray<scalar> k_M = factory.createScalar<scalar>(this->qam);
        matlab::data::TypedArray<scalar> m_M = factory.createScalar<scalar>(m);
        matlab::data::TypedArray<scalar> n_M = factory.createScalar<scalar>(n);
        matlab::data::TypedArray<scalar> SNR = factory.createScalar<scalar>(snr);
        matlab::data::TypedArray<scalar> MIT = factory.createScalar<scalar>(search_iter);
        matlabPtr->setVariable(u"k", std::move(k_M));
        matlabPtr->setVariable(u"m", std::move(m_M));
        matlabPtr->setVariable(u"n", std::move(n_M));
        matlabPtr->setVariable(u"snr", std::move(SNR));
        matlabPtr->setVariable(u"search_iter", std::move(MIT));
        // Call the MATLAB movsum function
        matlabPtr->eval(
                u" [s_bar4, y, H, HH, Piv, s_bar1, s, tolerance] = "
                "simulations_Block_Optimal(k, snr, m, n, 1, search_iter, true);");

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
        matlab::data::TypedArray<scalar> const tolerance_s = matlabPtr->getVariable(u"tolerance");
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
        for (auto r : tolerance_s) {
            this->tolerance = r;
            i++;
        }

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