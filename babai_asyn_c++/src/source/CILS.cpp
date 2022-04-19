#include "../include/CILS.h"

using namespace std;


namespace cils {

    template<typename scalar, typename index>
    static CILS<scalar, index> cils_driver(int argc, char *argv[]) {
        CILS<scalar, index> cils;

        options_description desc;
        if (argc == 1) {
            throw std::invalid_argument("The number of input is incorrect");
        }
        desc.add_options()
                ("help", "produce help message")
                ("size-m", value<int>(&cils.m)->default_value(20), "Model matrix row dimension")
                ("size-n", value<int>(&cils.n)->default_value(20), "Model matrix column dimension")
                ("qam", value<int>(&cils.qam)->default_value(1), "QAM: Quadrature amplitude modulation")
                ("snr", value<int>(&cils.snr)->default_value(35), "SNR: Signal-to-noise ratio")
                ("block_size", value<int>(&cils.block_size)->default_value(2),
                 "Block size for partitioning model matrix, default = 2")
                ("spilt_size", value<int>(&cils.spilt_size)->default_value(0),
                 "Split size for the last block of partitioned model matrix, default = 0")
                ("offset", value<int>(&cils.offset)->default_value(0),
                 "offset for splitting the last block. Default 0 for even splitting, default = 0")
                ("is_constrained", value<int>(&cils.is_constrained)->default_value(1),
                 "a constrained problem, 1: true; 0: false, default = 1")
                ("nswp", value<int>(&cils.nswp)->default_value(10),
                 "max number of iteration for chaotic relaxation, default = 10")
                ("max_search", value<int>(&cils.nswp)->default_value(10),
                 "max number of iteration for chaotic relaxation, default = 10")
                ("min_proc", value<int>(&cils.min_proc)->default_value(2),
                 "minimum number of multi-threads, default = 2")
                ("max_proc", value<int>(&cils.max_proc)->default_value(10),
                 "maximum number of multi-threads, default = 10")
                ("verbose", value<int>(&cils.verbose)->default_value(0), "verbose 1:true, 0:false, default = 0")
                ("schedule", value<int>(&cils.schedule)->default_value(2),
                 "OMP Scheduler 1:static, 2:dynamic, 3:guided, 4:auto")
                ("chunk_size", value<int>(&cils.chunk_size)->default_value(1), "OMP chunk size, default = 1");
    }

    template<typename scalar, typename index>
    static void init(CILS<scalar, index> &cils) {
        try {
            //Step 1: Initialize Problem by Matlab.
            cils.permutation.resize(cils.search_iter + 3);
            //Create MATLAB data array factory
            scalar *size = (double *) malloc(1 * sizeof(double)), *p;

//                if (rank == 0) {
            matlab::data::ArrayFactory factory;

            // Call the MATLAB movsum function
            matlab::data::TypedArray<scalar> k_M = factory.createScalar<scalar>(cils.qam);
            matlab::data::TypedArray<scalar> m_M = factory.createScalar<scalar>(cils.m);
            matlab::data::TypedArray<scalar> n_M = factory.createScalar<scalar>(cils.n);
            matlab::data::TypedArray<scalar> SNR = factory.createScalar<scalar>(cils.snr);
            matlab::data::TypedArray<scalar> MIT = factory.createScalar<scalar>(cils.search_iter);

            std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr;
            matlabPtr = matlab::engine::startMATLAB();

            matlabPtr->setVariable(u"k", std::move(k_M));
            matlabPtr->setVariable(u"m", std::move(m_M));
            matlabPtr->setVariable(u"n", std::move(n_M));
            matlabPtr->setVariable(u"SNR", std::move(SNR));
            matlabPtr->setVariable(u"max_iter", std::move(MIT));

            // Call the MATLAB addpath function
            matlabPtr->eval(u"addpath('/home/shilei/CLionProjects/Reference/babai_asyn/babai_asyn_matlab/')");
//            matlabPtr->eval(u"addpath('/Users/shileilin/CLionProjects/babai_asyn/babai_asyn_matlab/')");
//            matlabPtr->eval(u" [A, x_t, v, y, sigma, res, permutation, size_perm, R0] = gen_problem_convergence(k, m, n, SNR, max_iter);");
            matlabPtr->eval(
                    u" [A, x_t, v, y, sigma, res, permutation, size_perm, R0] = gen_problem(k, m, n, SNR, max_iter);");

            matlab::data::TypedArray<scalar> const A_A = matlabPtr->getVariable(u"A");
            matlab::data::TypedArray<scalar> const y_M = matlabPtr->getVariable(u"y");
            matlab::data::TypedArray<scalar> const x_M = matlabPtr->getVariable(u"x_t");
            matlab::data::TypedArray<scalar> const res = matlabPtr->getVariable(u"res");
            matlab::data::TypedArray<scalar> const per = matlabPtr->getVariable(u"permutation");
            matlab::data::TypedArray<scalar> const szp = matlabPtr->getVariable(u"size_perm");
            matlab::data::TypedArray<scalar> const R_0 = matlabPtr->getVariable(u"R0");

            std::vector<scalar> A_v(cils.m * cils.n, 0);
            index i = 0;
            for (auto r: A_A) {
                A_v[i] = r;
                i++;
            }
            for (index col = 0; col < cils.n; col++) {
                for (index row = 0; row < cils.m; row++) {
                    cils.A(row, col) = A_v[row + col * cils.m];
                }
            }

            std::vector<scalar> R0(cils.n * cils.n, 0);
            i = 0;
            for (auto r: R_0) {
                R0[i] = r;
                ++i;
            }
            cils.B.resize(cils.n, cils.n);
            cils.B.clear();
            for (index col = 0; col < cils.n; col++) {
                for (index row = 0; row < cils.n; row++) {
                    cils.B(row, col) = R0[row + col * cils.n];
                }
            }

            i = 0;
            for (auto r: y_M) {
                cils.y[i] = r;
                ++i;
            }
            i = 0;
            for (auto r: x_M) {
                cils.x_t[i] = r;
                ++i;
            }
            i = 0;
            for (auto r: res) {
                cils.init_res = r;
                ++i;
            }
            i = 0;
            for (auto r: res) {
                cils.init_res = r;
                ++i;
            }

            i = 0;
            for (auto r: szp) {
                size[0] = r;
                ++i;
            }
            p = (scalar *) malloc(cils.n * size[0] * sizeof(scalar));
            i = 0;
            for (auto r: per) {
                p[i] = r;
                ++i;
            }
//                }
//
//                MPI_Barrier(MPI_COMM_WORLD);
//                MPI_Bcast(&size[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//
//                if (rank != 0)
//                    p = (scalar *) malloc(n * size[0] * sizeof(scalar));

//            MPI_Bcast(&p[0], (int) size[0] * cils.n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//            MPI_Barrier(MPI_COMM_WORLD);

            i = 0;
            index k1 = 0;
            cils.permutation.resize((int) size[0] + 1);
            cils.permutation[k1] = std::vector<scalar>(cils.n);
            cils.permutation[k1].assign(cils.n, 0);
            for (index iter = 0; iter < (int) size[0] * cils.n; iter++) {
                cils.permutation[k1][i] = p[iter];
                i = i + 1;
                if (i == cils.n) {
                    i = 0;
                    k1++;
                    cils.permutation[k1] = std::vector<scalar>(cils.n);
                    cils.permutation[k1].assign(cils.n, 0);
                }
            }
            i = 0;
            cils.is_init_success = true;
            matlabPtr.get_deleter();
            //Step 2: initialize variables:

        } catch (const std::exception &e) {
            std::cout << e.what();
            cils.is_init_success = false;
        }
    }

    template<typename scalar, typename index>
    void init_LLL(CILS<scalar, index> &cils, index n, index k) {
        try {
            cils.m = n;
            cils.n = n;
            cils.A.resize(n, n, false);
            cils.y.resize(n, false);
            cils.y.clear();
            matlab::data::ArrayFactory factory;

            // Call the MATLAB movsum function
            matlab::data::TypedArray<scalar> m_M = factory.createScalar<scalar>(cils.m);
            matlab::data::TypedArray<scalar> n_M = factory.createScalar<scalar>(cils.n);
            matlab::data::TypedArray<scalar> k_M = factory.createScalar<scalar>(k);

            std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr;
            matlabPtr = matlab::engine::startMATLAB();

            matlabPtr->setVariable(u"m", std::move(m_M));
            matlabPtr->setVariable(u"n", std::move(n_M));
            matlabPtr->setVariable(u"k", std::move(k_M));

            // Call the MATLAB addpath function
            if (cils.is_local)
                matlabPtr->eval(u"addpath('/home/shilei/CLionProjects/Reference/babai_asyn/babai_asyn_matlab/')");
            matlabPtr->eval(u" [A, y, R0] = gen_lll_problem(k, m, n);");

            matlab::data::TypedArray<scalar> const A_A = matlabPtr->getVariable(u"A");
            matlab::data::TypedArray<scalar> const y_M = matlabPtr->getVariable(u"y");
            matlab::data::TypedArray<scalar> const R_0 = matlabPtr->getVariable(u"R0");

            std::vector<scalar> A_v(cils.m * cils.n, 0);
            index i = 0;
            for (auto r: A_A) {
                A_v[i] = r;
                i++;
            }
            cils.A.resize(n, n);
            for (index col = 0; col < cils.n; col++) {
                for (index row = 0; row < cils.m; row++) {
                    cils.A(row, col) = A_v[row + col * cils.m];
                }
            }

            std::vector<scalar> R0(cils.n * cils.n, 0);
            i = 0;
            for (auto r: R_0) {
                R0[i] = r;
                ++i;
            }
            cils.B.resize(cils.n, cils.n);
            cils.B.clear();
            for (index col = 0; col < cils.n; col++) {
                for (index row = 0; row < cils.n; row++) {
                    cils.B(row, col) = R0[row + col * cils.n];
                }
            }
            cils.y.clear();
            i = 0;
            for (auto r: y_M) {
                cils.y[i] = r;
                ++i;
            }
            i = 0;
            cils.is_init_success = true;
            matlabPtr.get_deleter();

        } catch (const std::exception &e) {
            std::cout << e.what();
            cils.is_init_success = false;
        }
    }

    template<typename scalar, typename index>
    void init_PBNP(CILS<scalar, index> &cils, index n, index snr, index qam, index c) {
        try {
            cils.m = n;
            cils.n = n;
            cils.A.resize(n, n, false);
            cils.y.resize(n, false);
            cils.y.resize(n, false);
            cils.y.clear();
            cils.x_t.resize(n, false);
            cils.x_t.clear();
            cils.qam = qam;
            cils.snr = snr;
            cils.upper = pow(2, qam) - 1;

            matlab::data::ArrayFactory factory;

            // Call the MATLAB movsum function
            matlab::data::TypedArray<scalar> k_M = factory.createScalar<scalar>(cils.qam);
            matlab::data::TypedArray<scalar> m_M = factory.createScalar<scalar>(cils.m);
            matlab::data::TypedArray<scalar> n_M = factory.createScalar<scalar>(cils.n);
            matlab::data::TypedArray<scalar> SNR = factory.createScalar<scalar>(cils.snr);
            matlab::data::TypedArray<scalar> c_M = factory.createScalar<scalar>(c);


            std::unique_ptr<matlab::engine::MATLABEngine> matlabPtr;
            matlabPtr = matlab::engine::startMATLAB();

            matlabPtr->setVariable(u"qam", std::move(k_M));
            matlabPtr->setVariable(u"m", std::move(m_M));
            matlabPtr->setVariable(u"n", std::move(n_M));
            matlabPtr->setVariable(u"SNR", std::move(SNR));
            matlabPtr->setVariable(u"c", std::move(c_M));


            // Call the MATLAB addpath function
            if (cils.is_local)
                matlabPtr->eval(u"addpath('/home/shilei/CLionProjects/Reference/babai_asyn/babai_asyn_matlab/')");
            matlabPtr->eval(u" [A, x_t, y, R0] = gen_olm_problem(qam, m, n, SNR, c);");

            matlab::data::TypedArray<scalar> const A_A = matlabPtr->getVariable(u"A");
            matlab::data::TypedArray<scalar> const y_M = matlabPtr->getVariable(u"y");
            matlab::data::TypedArray<scalar> const x_M = matlabPtr->getVariable(u"x_t");
            matlab::data::TypedArray<scalar> const R_0 = matlabPtr->getVariable(u"R0");

            std::vector<scalar> A_v(cils.m * cils.n, 0);
            index i = 0;
            for (auto r: A_A) {
                A_v[i] = r;
                i++;
            }
            for (index col = 0; col < cils.n; col++) {
                for (index row = 0; row < cils.m; row++) {
                    cils.A(row, col) = A_v[row + col * cils.m];
                }
            }

            std::vector<scalar> R0(cils.n * cils.n, 0);
            i = 0;
            for (auto r: R_0) {
                R0[i] = r;
                ++i;
            }
            cils.B.resize(cils.n, cils.n);
            cils.B.clear();
            for (index col = 0; col < cils.n; col++) {
                for (index row = 0; row < cils.n; row++) {
                    cils.B(row, col) = R0[row + col * cils.n];
                }
            }

            i = 0;
            for (auto r: y_M) {
                cils.y[i] = r;
                ++i;
            }
            i = 0;
            for (auto r: x_M) {
                cils.x_t[i] = r;
                ++i;
            }
            i = 0;
            cils.is_init_success = true;
            matlabPtr.get_deleter();
            //Step 2: initialize variables:

        } catch (const std::exception &e) {
            std::cout << e.what();
            cils.is_init_success = false;
        }
    }
}