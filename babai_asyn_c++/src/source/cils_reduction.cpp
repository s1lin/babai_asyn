#include <cstring>

#include <Python.h>
#include <numpy/arrayobject.h>
//#include "../include/coder_array.h"


namespace cils {

    template<typename scalar, typename index, index n>
    void value_input_helper(matlab::data::TypedArray<scalar> const x, coder::array<scalar, 1U> &arr) {
        index i = 0;
        arr.set_size(n);
        for (auto r : x) {
            arr[i] = r;
            ++i;
        }
    }

    template<typename scalar, typename index, index n>
    void value_input_helper(matlab::data::TypedArray<scalar> const x, coder::array<scalar, 2U> &arr) {
        index i = 0;
        arr.set_size(n, n);
        for (auto r : x) {
            arr[i] = r;
            ++i;
        }
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_qr_serial(const index eval, const index verbose) {

        index i, j, k, m;
        scalar error = -1, time, sum;
        //Deep Copy
        coder::array<scalar, 2U> A_t(A);
        //Clear Variables:
        R_Q.set_size(n, n);
        Q.set_size(n, n);

        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                R_Q[i * n + j] = 0;
                Q[i * n + j] = 0;
            }
        }

        //start
        time = omp_get_wtime();
        for (k = 0; k < n; k++) {
            //Check if Q[][i-1] (the previous column) is computed.
            for (j = k; j < n; j++) {
                R_Q[k * n + j] = 0;
                for (i = 0; i < n; i++) {
                    R_Q[j * n + k] += Q[k * n + i] * A_t[j * n + i];
                }
                for (i = 0; i < n; i++) {
                    A_t[j * n + i] -= R_Q[j * n + k] * Q[k * n + i];
                }
                //Calculate the norm(A)
                if (j == k) {
                    sum = 0;
                    for (i = 0; i < n; i++) {
                        sum += pow(A_t[k * n + i], 2);
                    }
                    R_Q[k * n + k] = sqrt(sum);
                    for (i = 0; i < n; i++) {
                        Q[k * n + i] = A_t[k * n + i] / R_Q[k * n + k];
                    }
                }
            }
        }

        time = omp_get_wtime() - time;

        if (eval) {
            error = qr_validation<scalar, index, n>(A, Q, R_Q, eval, verbose);
        }

        return {{}, time, error};
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_qr_omp(const index eval, const index qr_eval, const index n_proc) {

        scalar error, time, sum = 0;

        auto A_t = new scalar[n * n]();
        auto lock = new omp_lock_t[n]();

        time = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(sum)
        {

#pragma omp for schedule(static, 1)
            for (index i = 0; i < n; i++) {
                for (index j = 0; j < n; j++) {
                    A_t[i * n + j] = A[i * n + j];
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
                R_Q[0] = sqrt(sum);
                for (index i = 0; i < n; i++) {
                    Q[i] = A_t[i] / R_Q[0];
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
                        R_Q[(k - 1) * n + j] = 0;
                        for (index i = 0; i < n; i++) {
                            R_Q[j * n + (k - 1)] += Q[(k - 1) * n + i] * A_t[j * n + i];
                        }
                        for (index i = 0; i < n; i++) {
                            A_t[j * n + i] = A_t[j * n + i] - R_Q[j * n + (k - 1)] * Q[(k - 1) * n + i];
                        }
//Only one thread calculates the norm(A)//and unsets the lock for the next column.if (j == k) {    sum = 0;    for (index i = 0; i < n; i++) {        sum = sum + A_t[k * n + i] * A_t[k * n + i];    }    R_Q[k * n + k] = sqrt(sum);    for (index i = 0; i < n; i++) {        Q[k * n + i] = A_t[k * n + i] / R_Q[k * n + k];    }    omp_unset_lock(&lock[k]);}
                    }
                }
            }
        }

        time = omp_get_wtime() - time;
//        if (eval || qr_eval) {
//            error = qr_validation<scalar, index, n>(A, Q, R, R_A, eval, qr_eval);
//        }
        for (index i = 0; i < n; i++) {
            omp_destroy_lock(&lock[i]);
        }
#pragma parallel omp cancellation point
#pragma omp flush
        delete[] lock;
        delete[] A_t;

        return {{}, time, error};
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_qr_py(const index eval, const index qr_eval) {

        scalar error, time = omp_get_wtime();
//        cils_qr_py_helper();
        time = omp_get_wtime() - time;

        if (eval || qr_eval) {
//            error = qr_validation<scalar, index, n>(A, Q, R_Q, R_A, eval, qr_eval);
        }

        return {{}, time, (index) error};
    }

//    template<typename scalar, typename index, index n>
//    long int cils<scalar, index, n>::cils_qr_py_helper() {
//        PyObject *pName, *pModule, *pFunc;
//        PyObject *pArgs, *pValue, *pVec;
//        Py_Initialize();
//        if (_import_array() < 0)
//            PyErr_Print();
//
//        npy_intp dim[1] = {A->size};
//
//        pVec = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, A);
//        if (pVec == NULL) printf("There is a problem.\n");
//
//        PyObject *sys_path = PySys_GetObject("path");
//        PyList_Append(sys_path,
//                      PyUnicode_FromString("/home/shilei/CLionProjects/babai_asyn/babai_asyn_c++/src/example"));
//        pName = PyUnicode_FromString("py_qr");
//        pModule = PyImport_Import(pName);
//
//        if (pModule != NULL) {
//            pFunc = PyObject_GetAttrString(pModule, "qr");
//            if (pFunc && PyCallable_Check(pFunc)) {
//                pArgs = PyTuple_New(2);
//                if (PyTuple_SetItem(pArgs, 0, pVec) != 0) {
//                    return false;
//                }
//                if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", n)) != 0) {
//                    return false;
//                }
//                pValue = PyObject_CallObject(pFunc, pArgs);//Perform QR no return value
//                if (pValue != NULL) {
//                    PyArrayObject *q, *r;
//                    PyArg_ParseTuple(pValue, "O|O", &q, &r);
//                    Q = reinterpret_cast<scalar *>(PyArray_DATA(q));
//                    R = reinterpret_cast<scalar *>(PyArray_DATA(r));
//                } else {
//                    PyErr_Print();
//                }
//            } else {
//                if (PyErr_Occurred())
//                    PyErr_Print();
//                fprintf(stderr, "Cannot find function qr\n");
//            }
//        } else {
//            PyErr_Print();
//            fprintf(stderr, "Failed to load file\n");
//        }
//        return 0;
//    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_qr_matlab() {
        scalar time = omp_get_wtime();
        scalar true_res = cils_qr_matlab_helper();
        time = omp_get_wtime() - time;

//        return {{}, time, true_res};
        return {{}, time, 0};
    }

    template<typename scalar, typename index, index n>
    scalar cils<scalar, index, n>::cils_qr_matlab_helper() {
        // Start MATLAB engine synchronously
        using namespace matlab::engine;

        // Start MATLAB engine synchronously
        std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();

        //Create MATLAB data array factory
        matlab::data::ArrayFactory factory;

        // Create variables
        matlab::data::TypedArray<scalar> k_M = factory.createScalar<scalar>(program_def::k);
        matlab::data::TypedArray<scalar> SNR_M = factory.createScalar<scalar>(program_def::SNR);
        matlab::data::TypedArray<scalar> m_M = factory.createScalar<scalar>(n);
        matlab::data::TypedArray<scalar> qr_M = factory.createScalar<scalar>(program_def::is_qr);
        matlabPtr->setVariable(u"k", std::move(k_M));
        matlabPtr->setVariable(u"n", std::move(m_M));
        matlabPtr->setVariable(u"SNR", std::move(SNR_M));
        matlabPtr->setVariable(u"qr", std::move(qr_M));

        // Call the MATLAB movsum function
        matlabPtr->eval(u" [A, R, Z, y, y_LLL, x_t, init_res, info] = sils_driver_mex(k, n, SNR, qr);");

        // Get the result
        matlab::data::TypedArray<scalar> const A_A = matlabPtr->getVariable(u"A");
        matlab::data::TypedArray<scalar> const R_N = matlabPtr->getVariable(u"R");
        matlab::data::TypedArray<scalar> const Z_N = matlabPtr->getVariable(u"Z");
        matlab::data::TypedArray<scalar> const y_R = matlabPtr->getVariable(u"y");//Reduced
        matlab::data::TypedArray<scalar> const y_N = matlabPtr->getVariable(u"y_LLL");//Original
        matlab::data::TypedArray<scalar> const x_T = matlabPtr->getVariable(u"x_t");
        matlab::data::TypedArray<scalar> const res = matlabPtr->getVariable(u"init_res");
        matlab::data::TypedArray<scalar> const b_n = matlabPtr->getVariable(u"info");
        matlab::data::ArrayDimensions dim = A_A.getDimensions();

//        cout << dim[0] << " " << dim[1] << endl;
        // Display the result
        value_input_helper<scalar, index, n>(A_A, A);
        value_input_helper<scalar, index, n>(R_N, R_Q);
        value_input_helper<scalar, index, n>(R_N, R_R);
        value_input_helper<scalar, index, n>(Z_N, Z);
        value_input_helper<scalar, index, n>(y_N, y_a);//Original
        value_input_helper<scalar, index, n>(y_R, y_r);//Reduced
        value_input_helper<scalar, index, n>(y_R, y_q);//Reduced
        value_input_helper<scalar, index, n>(x_T, x_t);



        for (auto r : res) {
            init_res = r;
        }
        scalar info[3];
        index _i = 0;
        for (auto r : b_n) {
            info[_i] = r;
            _i++;
        }
        printf("----------------------\n"
               "MATLAB RESULT: QR/LLL Time: %.5f, Babai Time: %.5f, Babai Res: %.5f.\n"
               "----------------------\n", info[0], info[1], info[2]);
        return init_res;
    }

//    template<typename scalar, typename index, index n>
//    returnType <scalar, index>
//    cils<scalar, index, n>::cils_LLL_reduction(const index eval, const index n_proc) {
//        coder::array<double, 2U> A_C, Ai, Ar, Z_C, R_C, b_result, Q_C;
//        coder::array<double, 1U> y;
//        scalar time = 0, det = 0;
////        if (mode == 1){
////            Ar.set_size(n, n);
////            y.set_size(n);
////            for (index i = 0; i < n; i++) {
////                for (index i1 = 0; i1 < n; i1++) {
////                    Ar[i1 + n * i] = R[i1 + n * i];
////                }
////                y[i] = y_A[i];
////            }
////            scalar ser_time = cils_LLL_serial();
////            for (index i = 0; i < n; i++) {
////                for (index i1 = 0; i1 < n; i1++) {
////                    Ar[i1 + n * i] = R[i1 + n * i];
////                }
////                y[i] = y_A[i];
////            }
////
////            scalar omp_time[10];
////
////
////
////
////        } else {
//        if (eval) {
//            A_C.set_size(n, n);
//            Ar.set_size(n, n);
//            for (index i = 0; i < n; i++) {
//                for (index i1 = 0; i1 < n; i1++) {
//                    Ar[n * i1 + i] = R[n * i1 + i];
//                    A_C[i1 + i * n] = A[i1 + i * n];
//                }
//            }
//            Q_C.set_size(n, n);
//            coder::qr(A_C, R_C, Q_C);
//            for (index i = 0; i < n; i++) {
//                for (index j = i; j < n; j++) {
////                swap(this->R[i * n + j], this->R[j * n + i]);
//                    cout << R_C[n * i + j] << " ";
//                }
//                cout << endl;
//            }
//            for (index i = 0; i < n; i++) {
//                for (index j = 0; j < n; j++) {
//                    R[n * i + j] = R_C[n * i + j];
//                    Q[n * i + j] = Q_C[n * i + j];
//                }
////                cout << endl;
//            }
//
//            cout << qr_validation<scalar, index, n>(A, Q, R, R_A, 0, 1);
//            cout.flush();
//        }
//
//        if (n_proc <= 1)
//            time = cils_LLL_serial();
//        else
//            time = cils_LLL_omp(n_proc);
//
//        if (eval) {
//            // 'eo_sils_reduction:109' Q1 = R_*Z_C/R_C;
//
//            index b_loop_ub = n, loop_ub = n;
//
//            Z_C.set_size(n, n);
//            R_C.set_size(n, n);
//            Ar.set_size(n, loop_ub);
//            for (index i = 0; i < loop_ub; i++) {
//                for (index i1 = 0; i1 < b_loop_ub; i1++) {
//                    Z_C[n * i1 + i] = Z[n * i1 + i];
//                    R_C[n * i1 + i] = R[n * i1 + i];
//                    if (Z_C[n * i1 + i] != 0 && Z_C[n * i1 + i] != 1)
//                        cout << i << "," << i1 << "," << Z_C[n * i1 + i] << " ";
//                }
//            }
//            cout << endl;
//            coder::internal::blas::mtimes(R_C, Z_C, Ai);
//            coder::internal::mrdiv(Ai, Ar);
//            //  Q1
//            // 'eo_sils_reduction:111' d = det(Q1*Q1');
//            coder::internal::blas::b_mtimes(Ai, Ai, b_result);
//            det = coder::det(b_result);
//            det = coder::det(Z_C);
//
//            cout << "DET:" << det << endl;
//        }
////        }
//        return {{}, time, det};
//    }

//    template<typename scalar, typename index, index n>
//    scalar cils<scalar, index, n>::cils_LLL_serial() {
//        bool f = true;
//        coder::array<scalar, 1U> r, vi, vr;
//        coder::array<double, 2U> b, b_R, r1;
//        scalar G[4], low_tmp[2], zeta;
//        index c_result[2], sizes[2], b_loop_ub, i, i1, input_sizes_idx_1 = n, loop_ub, result;
//
//        vi.set_size(n);
//        for (i = 0; i < n; i++) {
//            vi[i] = 0.0;
//        }
//
//        scalar time = omp_get_wtime();
//        while (f) {
//            scalar a, alpha, s;
//            index b_i, i2;
//            unsigned int c_i;
//            // 'eo_sils_reduction:48' f = false;
//            f = false;
//            // 'eo_sils_reduction:49' for i = 2:2:n
//            i = n / 2;
//            for (b_i = 0; b_i < i; b_i++) {
//                c_i = (b_i << 1) + 2U;
//                // 'eo_sils_reduction:50' i1 = i-1;
//                // 'eo_sils_reduction:51' zeta = round(R(i1,i) / R(i1,i1));
//                zeta = std::round(R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] /
//                                  R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2]);
//                // 'eo_sils_reduction:52' alpha = R(i1,i) - zeta * R(i1,i1);
//                s = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2];
//                alpha = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] - zeta * s;
//                // 'eo_sils_reduction:53' if R(i1,i1)^2 > (1 + 1.e-0) * (alpha^2 +
//                // R(i,i)^2)
//                a = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 1];
//                if ((s * s > 2.0 * (alpha * alpha + a * a)) && (zeta != 0.0)) {
//                    // 'eo_sils_reduction:54' if zeta ~= 0
//                    //  Perform a size reduction on R(k-1,k)
//                    // 'eo_sils_reduction:56' f = true;
//                    f = true;
//                    // 'eo_sils_reduction:57' swap(i) = 1;
//                    vi[static_cast<int>(c_i) - 1] = 1.0;
//                    // 'eo_sils_reduction:58' R(i1,i) = alpha;
//                    R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] = alpha;
//                    // 'eo_sils_reduction:59' R(1:i-2,i) = R(1:i-2,i) - zeta * R(1:i-2,i-1);
//                    if (1 > static_cast<int>(c_i) - 2) {
//                        b_loop_ub = 0;
//                    } else {
//                        b_loop_ub = static_cast<int>(c_i) - 2;
//                    }
//                    vr.set_size(b_loop_ub);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        vr[i1] = R[i1 + n * (static_cast<int>(c_i) - 1)] -
//                                 zeta * R[i1 + n * (static_cast<int>(c_i) - 2)];
//                    }
//                    b_loop_ub = vr.size(0);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        R[i1 + n * (static_cast<int>(c_i) - 1)] = vr[i1];
//                    }
//                    // 'eo_sils_reduction:60' Z(:,i) = Z(:,i) - zeta * Z(:,i-1);
//                    input_sizes_idx_1 = n - 1;
//                    vr.set_size(n);
//                    for (i1 = 0; i1 <= input_sizes_idx_1; i1++) {
//                        vr[i1] = Z[i1 + n * (static_cast<int>(c_i) - 1)] -
//                                 zeta * Z[i1 + n * (static_cast<int>(c_i) - 2)];
//                    }
//                    b_loop_ub = vr.size(0);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        Z[i1 + n * (static_cast<int>(c_i) - 1)] = vr[i1];
//                    }
//                    //  Permute columns k-1 and k of R and Z
//                    // 'eo_sils_reduction:63' R(1:i,[i1,i]) = R(1:i,[i,i1]);
//                    sizes[0] = static_cast<int>(c_i) - 2;
//                    sizes[1] = static_cast<int>(c_i) - 1;
//                    c_result[0] = static_cast<int>(c_i) - 1;
//                    c_result[1] = static_cast<int>(c_i) - 2;
//                    b_loop_ub = static_cast<int>(c_i);
//                    b_R.set_size(static_cast<int>(c_i), 2);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 < b_loop_ub; i2++) { b_R[i2 + b_R.size(0) * i1] = R[i2 + n * c_result[i1]]; }
//                    }
//                    b_loop_ub = b_R.size(0);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 < b_loop_ub; i2++) { R[i2 + n * sizes[i1]] = b_R[i2 + b_R.size(0) * i1]; }
//                    }
//                    // 'eo_sils_reduction:64' Z(:,[i1,i]) = Z(:,[i,i1]);
//                    sizes[0] = static_cast<int>(c_i) - 2;
//                    sizes[1] = static_cast<int>(c_i) - 1;
//                    input_sizes_idx_1 = n - 1;
//                    c_result[0] = static_cast<int>(c_i) - 1;
//                    c_result[1] = static_cast<int>(c_i) - 2;
//                    b_R.set_size(n, 2);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 <= input_sizes_idx_1; i2++) {
//                            b_R[i2 + b_R.size(0) * i1] = Z[i2 + n * c_result[i1]];
//                        }
//                    }
//                    b_loop_ub = b_R.size(0);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 < b_loop_ub; i2++) {
//                            Z[i2 + n * sizes[i1]] = b_R[i2 + b_R.size(0) * i1];
//                        }
//                    }
//                }
//            }
//            // 'eo_sils_reduction:68' for i = 2:2:n
//            for (b_i = 0; b_i < i; b_i++) {
//                c_i = (b_i << 1) + 2U;
//                // 'eo_sils_reduction:69' i1 = i-1;
//                // 'eo_sils_reduction:70' if swap(i) == 1
//                if (vi[static_cast<int>(c_i) - 1] == 1.0) {
//                    // 'eo_sils_reduction:71' [G,R([i1,i],i1)] = planerot(R([i1,i],i1));
//                    low_tmp[0] = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2];
//                    low_tmp[1] = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 1];
//                    coder::planerot(low_tmp, G);
//                    R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2] = low_tmp[0];
//                    R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 1] = low_tmp[1];
//                    // 'eo_sils_reduction:72' R([i1,i],i:n) = G * R([i1,i],i:n);
//                    if (static_cast<int>(c_i) > static_cast<int>(n)) {
//                        i1 = 0;
//                        i2 = 0;
//                        result = 1;
//                    } else {
//                        i1 = static_cast<int>(c_i) - 1;
//                        i2 = static_cast<int>(n);
//                        result = static_cast<int>(c_i);
//                    }
//                    b_loop_ub = i2 - i1;
//                    b.set_size(2, b_loop_ub);
//                    for (i2 = 0; i2 < b_loop_ub; i2++) {
//                        input_sizes_idx_1 = i1 + i2;
//                        b[2 * i2] = R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 2];
//                        b[2 * i2 + 1] = R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 1];
//                    }
//                    coder::internal::blas::mtimes(G, b, r1);
//                    b_loop_ub = r1.size(1);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        input_sizes_idx_1 = (result + i1) - 1;
//                        R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 2] = r1[2 * i1];
//                        R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 1] = r1[2 * i1 + 1];
//                    }
//                    // 'eo_sils_reduction:73' y_L([i1,i]) = G * y_L([i1,i]);
//                    low_tmp[0] = y_A[static_cast<int>(c_i) - 2];
//                    low_tmp[1] = y_A[static_cast<int>(c_i) - 1];
//                    y_A[static_cast<int>(c_i) - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
//                    y_A[static_cast<int>(c_i) - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
//                    // 'eo_sils_reduction:74' swap(i) = 0;
//                    vi[static_cast<int>(c_i) - 1] = 0.0;
//                }
//            }
//            // 'eo_sils_reduction:77' for i = 3:2:n
//            i = static_cast<int>((n + -1.0) / 2.0);
//            for (b_i = 0; b_i < i; b_i++) {
//                c_i = static_cast<unsigned int>((b_i << 1) + 3);
//                // 'eo_sils_reduction:78' i1 = i-1;
//                // 'eo_sils_reduction:79' zeta = round(R(i1,i) / R(i1,i1));
//                zeta = std::round(R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] /
//                                  R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2]);
//                // 'eo_sils_reduction:80' alpha = R(i1,i) - zeta * R(i1,i1);
//                input_sizes_idx_1 = static_cast<int>(c_i) - 2;
//                s = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2];
//                alpha = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] - zeta * s;
//                // 'eo_sils_reduction:81' if R(i1,i1)^2 > (1 + 1.e-0) * (alpha^2 +
//                // R(i,i)^2)
//                a = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 1];
//                if ((s * s > 2.0 * (alpha * alpha + a * a)) && (zeta != 0.0)) {
//                    // 'eo_sils_reduction:82' if zeta ~= 0
//                    // 'eo_sils_reduction:83' f = true;
//                    f = true;
//                    // 'eo_sils_reduction:84' swap(i) = 1;
//                    vi[static_cast<int>(c_i) - 1] = 1.0;
//                    //  Perform a size reduction on R(k-1,k)
//                    // 'eo_sils_reduction:86' R(i1,i) = alpha;
//                    R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] = alpha;
//                    // 'eo_sils_reduction:87' R(1:i-2,i) = R(1:i-2,i) - zeta * R(1:i-2,i-1);
//                    vr.set_size(input_sizes_idx_1);
//                    for (i1 = 0; i1 < input_sizes_idx_1; i1++) {
//                        vr[i1] = R[i1 + n * (static_cast<int>(c_i) - 1)] -
//                                 zeta * R[i1 + n * (static_cast<int>(c_i) - 2)];
//                    }
//                    b_loop_ub = vr.size(0);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        R[i1 + n * (static_cast<int>(c_i) - 1)] = vr[i1];
//                    }
//                    // 'eo_sils_reduction:88' Z(:,i) = Z(:,i) - zeta * Z(:,i-1);
//                    input_sizes_idx_1 = n - 1;
//                    vr.set_size(n);
//                    for (i1 = 0; i1 <= input_sizes_idx_1; i1++) {
//                        vr[i1] = Z[i1 + n * (static_cast<int>(c_i) - 1)] -
//                                 zeta * Z[i1 + n * (static_cast<int>(c_i) - 2)];
//                    }
//                    b_loop_ub = vr.size(0);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        Z[i1 + n * (static_cast<int>(c_i) - 1)] = vr[i1];
//                    }
//                    //  Permute columns k-1 and k of R and Z
//                    // 'eo_sils_reduction:91' R(1:i,[i1,i]) = R(1:i,[i,i1]);
//                    sizes[0] = static_cast<int>(c_i) - 2;
//                    sizes[1] = static_cast<int>(c_i) - 1;
//                    c_result[0] = static_cast<int>(c_i) - 1;
//                    c_result[1] = static_cast<int>(c_i) - 2;
//                    b_loop_ub = static_cast<int>(c_i);
//                    b_R.set_size(static_cast<int>(c_i), 2);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 < b_loop_ub; i2++) { b_R[i2 + b_R.size(0) * i1] = R[i2 + n * c_result[i1]]; }
//                    }
//                    b_loop_ub = b_R.size(0);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 < b_loop_ub; i2++) { R[i2 + n * sizes[i1]] = b_R[i2 + b_R.size(0) * i1]; }
//                    }
//                    // 'eo_sils_reduction:92' Z(:,[i1,i]) = Z(:,[i,i1]);
//                    sizes[0] = static_cast<int>(c_i) - 2;
//                    sizes[1] = static_cast<int>(c_i) - 1;
//                    input_sizes_idx_1 = n - 1;
//                    c_result[0] = static_cast<int>(c_i) - 1;
//                    c_result[1] = static_cast<int>(c_i) - 2;
//                    b_R.set_size(n, 2);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 <= input_sizes_idx_1; i2++) {
//                            b_R[i2 + b_R.size(0) * i1] = Z[i2 + n * c_result[i1]];
//                        }
//                    }
//                    b_loop_ub = b_R.size(0);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 < b_loop_ub; i2++) {
//                            Z[i2 + n * sizes[i1]] = b_R[i2 + b_R.size(0) * i1];
//                        }
//                    }
//                }
//            }
//            // 'eo_sils_reduction:96' for i = 3:2:n
//            for (b_i = 0; b_i < i; b_i++) {
//                c_i = static_cast<unsigned int>((b_i << 1) + 3);
//                // 'eo_sils_reduction:97' i1 = i-1;
//                // 'eo_sils_reduction:98' if swap(i) == 1
//                if (vi[static_cast<int>(c_i) - 1] == 1.0) {
//                    //  Bring R baci to an upper triangular matrix by a Givens rotation
//                    // 'eo_sils_reduction:100' [G,R([i1,i],i1)] = planerot(R([i1,i],i1));
//                    low_tmp[0] = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2];
//                    low_tmp[1] = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 1];
//                    coder::planerot(low_tmp, G);
//                    R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2] = low_tmp[0];
//                    R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 1] = low_tmp[1];
//                    // 'eo_sils_reduction:101' R([i1,i],i:n) = G * R([i1,i],i:n);
//                    if (static_cast<int>(c_i) > static_cast<int>(n)) {
//                        i1 = 0;
//                        i2 = 0;
//                        result = 1;
//                    } else {
//                        i1 = static_cast<int>(c_i) - 1;
//                        i2 = static_cast<int>(n);
//                        result = static_cast<int>(c_i);
//                    }
//                    b_loop_ub = i2 - i1;
//                    b.set_size(2, b_loop_ub);
//                    for (i2 = 0; i2 < b_loop_ub; i2++) {
//                        input_sizes_idx_1 = i1 + i2;
//                        b[2 * i2] = R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 2];
//                        b[2 * i2 + 1] = R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 1];
//                    }
//                    coder::internal::blas::mtimes(G, b, r1);
//                    b_loop_ub = r1.size(1);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        input_sizes_idx_1 = (result + i1) - 1;
//                        R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 2] = r1[2 * i1];
//                        R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 1] = r1[2 * i1 + 1];
//                    }
//                    //  Apply the Givens rotation to y_A
//                    // 'eo_sils_reduction:104' y_A([i1,i]) = G * y_A([i1,i]);
//                    low_tmp[0] = y_A[static_cast<int>(c_i) - 2];
//                    low_tmp[1] = y_A[static_cast<int>(c_i) - 1];
//                    y_A[static_cast<int>(c_i) - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
//                    y_A[static_cast<int>(c_i) - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
//                    // 'eo_sils_reduction:105' swap(i) = 0;
//                    vi[static_cast<int>(c_i) - 1] = 0.0;
//                }
//            }
//        }
//        time = omp_get_wtime() - time;
//        return time;
//    }
//
//    template<typename scalar, typename index, index n>
//    scalar cils<scalar, index, n>::cils_LLL_omp(const index n_proc) {
//        bool f = true;
//        coder::array<scalar, 1U> r, vi, vr;
//        coder::array<scalar, 2U> b, b_R, r1;
//        scalar G[4], low_tmp[2], zeta;
//        index c_result[2], sizes[2], b_loop_ub, i, i1, input_sizes_idx_1 = n, loop_ub, result;
//
//        vi.set_size(n);
//        for (i = 0; i < n; i++) {
//            vi[i] = 0;
//        }
////        omp_set_num_threads(n_proc);
//
//        scalar time = omp_get_wtime();
//        while (f) {
//            scalar a, alpha, s;
//            index b_i, i2;
//            unsigned int c_i;
//            // 'eo_sils_reduction:48' f = false;
//            f = false;
//            // 'eo_sils_reduction:49' for i = 2:2:n
//            i = n / 2;
//#pragma omp parallel for
//            for (b_i = 0; b_i < i; b_i++) {
//                c_i = (b_i << 1) + 2U;
//                // 'eo_sils_reduction:50' i1 = i-1;
//                // 'eo_sils_reduction:51' zeta = round(R(i1,i) / R(i1,i1));
//                zeta = std::round(R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] /
//                                  R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2]);
//                // 'eo_sils_reduction:52' alpha = R(i1,i) - zeta * R(i1,i1);
//                s = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2];
//                alpha = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] - zeta * s;
//                // 'eo_sils_reduction:53' if R(i1,i1)^2 > (1 + 1.e-0) * (alpha^2 +
//                // R(i,i)^2)
//                a = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 1];
//                if ((s * s > 2.0 * (alpha * alpha + a * a)) && (zeta != 0.0)) {
//                    // 'eo_sils_reduction:54' if zeta ~= 0
//                    //  Perform a size reduction on R(k-1,k)
//                    // 'eo_sils_reduction:56' f = true;
//                    f = true;
//                    // 'eo_sils_reduction:57' swap(i) = 1;
//                    vi[static_cast<int>(c_i) - 1] = 1.0;
//                    // 'eo_sils_reduction:58' R(i1,i) = alpha;
//                    R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] = alpha;
//                    // 'eo_sils_reduction:59' R(1:i-2,i) = R(1:i-2,i) - zeta * R(1:i-2,i-1);
//                    if (1 > static_cast<int>(c_i) - 2) {
//                        b_loop_ub = 0;
//                    } else {
//                        b_loop_ub = static_cast<int>(c_i) - 2;
//                    }
//                    vr.set_size(b_loop_ub);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        vr[i1] = R[i1 + n * (static_cast<int>(c_i) - 1)] -
//                                 zeta * R[i1 + n * (static_cast<int>(c_i) - 2)];
//                    }
//                    b_loop_ub = vr.size(0);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        R[i1 + n * (static_cast<int>(c_i) - 1)] = vr[i1];
//                    }
//                    // 'eo_sils_reduction:60' Z(:,i) = Z(:,i) - zeta * Z(:,i-1);
//                    input_sizes_idx_1 = n - 1;
//                    vr.set_size(n);
//                    for (i1 = 0; i1 <= input_sizes_idx_1; i1++) {
//                        vr[i1] = Z[i1 + n * (static_cast<int>(c_i) - 1)] -
//                                 zeta * Z[i1 + n * (static_cast<int>(c_i) - 2)];
//                    }
//                    b_loop_ub = vr.size(0);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        Z[i1 + n * (static_cast<int>(c_i) - 1)] = vr[i1];
//                    }
//                    //  Permute columns k-1 and k of R and Z
//                    // 'eo_sils_reduction:63' R(1:i,[i1,i]) = R(1:i,[i,i1]);
//                    sizes[0] = static_cast<int>(c_i) - 2;
//                    sizes[1] = static_cast<int>(c_i) - 1;
//                    c_result[0] = static_cast<int>(c_i) - 1;
//                    c_result[1] = static_cast<int>(c_i) - 2;
//                    b_loop_ub = static_cast<int>(c_i);
//                    b_R.set_size(static_cast<int>(c_i), 2);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 < b_loop_ub; i2++) { b_R[i2 + b_R.size(0) * i1] = R[i2 + n * c_result[i1]]; }
//                    }
//                    b_loop_ub = b_R.size(0);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 < b_loop_ub; i2++) { R[i2 + n * sizes[i1]] = b_R[i2 + b_R.size(0) * i1]; }
//                    }
//                    // 'eo_sils_reduction:64' Z(:,[i1,i]) = Z(:,[i,i1]);
//                    sizes[0] = static_cast<int>(c_i) - 2;
//                    sizes[1] = static_cast<int>(c_i) - 1;
//                    input_sizes_idx_1 = n - 1;
//                    c_result[0] = static_cast<int>(c_i) - 1;
//                    c_result[1] = static_cast<int>(c_i) - 2;
//                    b_R.set_size(n, 2);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 <= input_sizes_idx_1; i2++) {
//                            b_R[i2 + b_R.size(0) * i1] = Z[i2 + n * c_result[i1]];
//                        }
//                    }
//                    b_loop_ub = b_R.size(0);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 < b_loop_ub; i2++) {
//                            Z[i2 + n * sizes[i1]] = b_R[i2 + b_R.size(0) * i1];
//                        }
//                    }
//                }
//            }
//            // 'eo_sils_reduction:68' for i = 2:2:n
//            // GIVENS ROTATION:
//            for (b_i = 0; b_i < i; b_i++) {
//                c_i = (b_i << 1) + 2U;
//                // 'eo_sils_reduction:69' i1 = i-1;
//                // 'eo_sils_reduction:70' if swap(i) == 1
//                if (vi[static_cast<int>(c_i) - 1] == 1.0) {
//                    // 'eo_sils_reduction:71' [G,R([i1,i],i1)] = planerot(R([i1,i],i1));
//                    low_tmp[0] = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2];
//                    low_tmp[1] = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 1];
//                    coder::planerot(low_tmp, G);
//                    R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2] = low_tmp[0];
//                    R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 1] = low_tmp[1];
//                    // 'eo_sils_reduction:72' R([i1,i],i:n) = G * R([i1,i],i:n);
//                    if (static_cast<int>(c_i) > static_cast<int>(n)) {
//                        i1 = 0;
//                        i2 = 0;
//                        result = 1;
//                    } else {
//                        i1 = static_cast<int>(c_i) - 1;
//                        i2 = static_cast<int>(n);
//                        result = static_cast<int>(c_i);
//                    }
//                    b_loop_ub = i2 - i1;
//                    b.set_size(2, b_loop_ub);
//                    for (i2 = 0; i2 < b_loop_ub; i2++) {
//                        input_sizes_idx_1 = i1 + i2;
//                        b[2 * i2] = R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 2];
//                        b[2 * i2 + 1] = R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 1];
//                    }
//                    coder::internal::blas::mtimes(G, b, r1);
//                    b_loop_ub = r1.size(1);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        input_sizes_idx_1 = (result + i1) - 1;
//                        R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 2] = r1[2 * i1];
//                        R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 1] = r1[2 * i1 + 1];
//                    }
//                    // 'eo_sils_reduction:73' y_A([i1,i]) = G * y_A([i1,i]);
//                    low_tmp[0] = y_A[static_cast<int>(c_i) - 2];
//                    low_tmp[1] = y_A[static_cast<int>(c_i) - 1];
//                    y_A[static_cast<int>(c_i) - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
//                    y_A[static_cast<int>(c_i) - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
//                    // 'eo_sils_reduction:74' swap(i) = 0;
//                    vi[static_cast<int>(c_i) - 1] = 0.0;
//                }
//            }
//            // 'eo_sils_reduction:77' for i = 3:2:n
//            i = static_cast<int>((n + -1.0) / 2.0);
//#pragma omp parallel for
//            for (b_i = 0; b_i < i; b_i++) {
//                c_i = static_cast<unsigned int>((b_i << 1) + 3);
//                // 'eo_sils_reduction:78' i1 = i-1;
//                // 'eo_sils_reduction:79' zeta = round(R(i1,i) / R(i1,i1));
//                zeta = std::round(R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] /
//                                  R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2]);
//                // 'eo_sils_reduction:80' alpha = R(i1,i) - zeta * R(i1,i1);
//                input_sizes_idx_1 = static_cast<int>(c_i) - 2;
//                s = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2];
//                alpha = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] - zeta * s;
//                // 'eo_sils_reduction:81' if R(i1,i1)^2 > (1 + 1.e-0) * (alpha^2 +
//                // R(i,i)^2)
//                a = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 1];
//                if ((s * s > 2.0 * (alpha * alpha + a * a)) && (zeta != 0.0)) {
//                    // 'eo_sils_reduction:82' if zeta ~= 0
//                    // 'eo_sils_reduction:83' f = true;
//                    f = true;
//                    // 'eo_sils_reduction:84' swap(i) = 1;
//                    vi[static_cast<int>(c_i) - 1] = 1.0;
//                    //  Perform a size reduction on R(k-1,k)
//                    // 'eo_sils_reduction:86' R(i1,i) = alpha;
//                    R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 1)) - 2] = alpha;
//                    // 'eo_sils_reduction:87' R(1:i-2,i) = R(1:i-2,i) - zeta * R(1:i-2,i-1);
//                    vr.set_size(input_sizes_idx_1);
//                    for (i1 = 0; i1 < input_sizes_idx_1; i1++) {
//                        vr[i1] = R[i1 + n * (static_cast<int>(c_i) - 1)] -
//                                 zeta * R[i1 + n * (static_cast<int>(c_i) - 2)];
//                    }
//                    b_loop_ub = vr.size(0);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        R[i1 + n * (static_cast<int>(c_i) - 1)] = vr[i1];
//                    }
//                    // 'eo_sils_reduction:88' Z(:,i) = Z(:,i) - zeta * Z(:,i-1);
//                    input_sizes_idx_1 = n - 1;
//                    vr.set_size(n);
//                    for (i1 = 0; i1 <= input_sizes_idx_1; i1++) {
//                        vr[i1] = Z[i1 + n * (static_cast<int>(c_i) - 1)] -
//                                 zeta * Z[i1 + n * (static_cast<int>(c_i) - 2)];
//                    }
//                    b_loop_ub = vr.size(0);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        Z[i1 + n * (static_cast<int>(c_i) - 1)] = vr[i1];
//                    }
//                    //  Permute columns k-1 and k of R and Z
//                    // 'eo_sils_reduction:91' R(1:i,[i1,i]) = R(1:i,[i,i1]);
//                    sizes[0] = static_cast<int>(c_i) - 2;
//                    sizes[1] = static_cast<int>(c_i) - 1;
//                    c_result[0] = static_cast<int>(c_i) - 1;
//                    c_result[1] = static_cast<int>(c_i) - 2;
//                    b_loop_ub = static_cast<int>(c_i);
//                    b_R.set_size(static_cast<int>(c_i), 2);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 < b_loop_ub; i2++) { b_R[i2 + b_R.size(0) * i1] = R[i2 + n * c_result[i1]]; }
//                    }
//                    b_loop_ub = b_R.size(0);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 < b_loop_ub; i2++) { R[i2 + n * sizes[i1]] = b_R[i2 + b_R.size(0) * i1]; }
//                    }
//                    // 'eo_sils_reduction:92' Z(:,[i1,i]) = Z(:,[i,i1]);
//                    sizes[0] = static_cast<int>(c_i) - 2;
//                    sizes[1] = static_cast<int>(c_i) - 1;
//                    input_sizes_idx_1 = n - 1;
//                    c_result[0] = static_cast<int>(c_i) - 1;
//                    c_result[1] = static_cast<int>(c_i) - 2;
//                    b_R.set_size(n, 2);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 <= input_sizes_idx_1; i2++) {
//                            b_R[i2 + b_R.size(0) * i1] = Z[i2 + n * c_result[i1]];
//                        }
//                    }
//                    b_loop_ub = b_R.size(0);
//                    for (i1 = 0; i1 < 2; i1++) {
//                        for (i2 = 0; i2 < b_loop_ub; i2++) {
//                            Z[i2 + n * sizes[i1]] = b_R[i2 + b_R.size(0) * i1];
//                        }
//                    }
//                }
//            }
//            // 'eo_sils_reduction:96' for i = 3:2:n
//            // GIVENS ROTATION:
//            for (b_i = 0; b_i < i; b_i++) {
//                c_i = static_cast<unsigned int>((b_i << 1) + 3);
//                // 'eo_sils_reduction:97' i1 = i-1;
//                // 'eo_sils_reduction:98' if swap(i) == 1
//                if (vi[static_cast<int>(c_i) - 1] == 1.0) {
//                    //  Bring R baci to an upper triangular matrix by a Givens rotation
//                    // 'eo_sils_reduction:100' [G,R([i1,i],i1)] = planerot(R([i1,i],i1));
//                    low_tmp[0] = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2];
//                    low_tmp[1] = R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 1];
//                    coder::planerot(low_tmp, G);
//                    R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 2] = low_tmp[0];
//                    R[(static_cast<int>(c_i) + n * (static_cast<int>(c_i) - 2)) - 1] = low_tmp[1];
//                    // 'eo_sils_reduction:101' R([i1,i],i:n) = G * R([i1,i],i:n);
//                    if (static_cast<int>(c_i) > static_cast<int>(n)) {
//                        i1 = 0;
//                        i2 = 0;
//                        result = 1;
//                    } else {
//                        i1 = static_cast<int>(c_i) - 1;
//                        i2 = static_cast<int>(n);
//                        result = static_cast<int>(c_i);
//                    }
//                    b_loop_ub = i2 - i1;
//                    b.set_size(2, b_loop_ub);
//                    for (i2 = 0; i2 < b_loop_ub; i2++) {
//                        input_sizes_idx_1 = i1 + i2;
//                        b[2 * i2] = R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 2];
//                        b[2 * i2 + 1] = R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 1];
//                    }
//                    coder::internal::blas::mtimes(G, b, r1);
//                    b_loop_ub = r1.size(1);
//                    for (i1 = 0; i1 < b_loop_ub; i1++) {
//                        input_sizes_idx_1 = (result + i1) - 1;
//                        R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 2] = r1[2 * i1];
//                        R[(static_cast<int>(c_i) + n * input_sizes_idx_1) - 1] = r1[2 * i1 + 1];
//                    }
//                    //  Apply the Givens rotation to y_A
//                    // 'eo_sils_reduction:104' y_A([i1,i]) = G * y_A([i1,i]);
//                    low_tmp[0] = y_A[static_cast<int>(c_i) - 2];
//                    low_tmp[1] = y_A[static_cast<int>(c_i) - 1];
//                    y_A[static_cast<int>(c_i) - 2] = G[0] * low_tmp[0] + low_tmp[1] * G[2];
//                    y_A[static_cast<int>(c_i) - 1] = low_tmp[0] * G[1] + low_tmp[1] * G[3];
//                    // 'eo_sils_reduction:105' swap(i) = 0;
//                    vi[static_cast<int>(c_i) - 1] = 0.0;
//                }
//            }
//        }
//        time = omp_get_wtime() - time;
//        return time;
//    }
//

//
//    template<typename scalar, typename index, index n>
//    void cils<scalar, index, n>::value_input_helper(matlab::data::TypedArray<scalar> const x, vector<scalar> *arr) {
//        index i = 0;
//        for (auto r : x) {
//            arr->at(i) = r;
//            ++i;
//        }
//    }
}


