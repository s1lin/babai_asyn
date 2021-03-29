#include <cstring>

#include <Python.h>
#include <numpy/arrayobject.h>

namespace cils {
    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_qr_decomposition_serial(const index eval, const index qr_eval) {

        index i, j, k, m, counter = 0;
        scalar error, time, sum;
        auto A_t = new scalar[n * n]();
        R->x = new scalar[n * n]();
        Q->x = new scalar[n * n]();

        time = omp_get_wtime();
        for (i = 0; i < n * n; i++) {
            A_t[i] = A->x[i];
        }

        for (k = 0; k < n; k++) {
            //Check if Q[][i-1] (the previous column) is computed.
            for (j = k; j < n; j++) {
                R->x[k * n + j] = 0;
                for (i = 0; i < n; i++) {
                    R->x[j * n + k] += Q->x[k * n + i] * A_t[j * n + i];
                }
                for (i = 0; i < n; i++) {
                    A_t[j * n + i] -= R->x[j * n + k] * Q->x[k * n + i];
                }
                //Calculate the norm(A)
                if (j == k) {
                    sum = 0;
                    for (i = 0; i < n; i++) {
                        sum += pow(A_t[k * n + i], 2);
                    }
                    R->x[k * n + k] = sqrt(sum);
                    for (i = 0; i < n; i++) {
                        Q->x[k * n + i] = A_t[k * n + i] / R->x[k * n + k];
                    }
                }
            }
        }

        time = omp_get_wtime() - time;

        if (eval || qr_eval) {
            error = qr_validation<scalar, index, n>(A, Q, R, R_A, eval, qr_eval);
//            cout << error << " ";
        }

        delete[] A_t;
        return {{}, time, (index) error};
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_qr_decomposition_omp(const index eval, const index qr_eval, const index n_proc) {

        scalar error, time, sum = 0;

        auto A_t = new scalar[n * n]();
        R->x = new scalar[n * n]();
        Q->x = new scalar[n * n]();
        auto lock = new omp_lock_t[n]();

        time = omp_get_wtime();
#pragma omp parallel default(shared) num_threads(n_proc) private(sum)
        {

#pragma omp for schedule(static, 1)
            for (index i = 0; i < n; i++) {
                for (index j = 0; j < n; j++) {
                    A_t[i * n + j] = A->x[i * n + j];
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
                R->x[0] = sqrt(sum);
                for (index i = 0; i < n; i++) {
                    Q->x[i] = A_t[i] / R->x[0];
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
                        R->x[(k - 1) * n + j] = 0;
                        for (index i = 0; i < n; i++) {
                            R->x[j * n + (k - 1)] += Q->x[(k - 1) * n + i] * A_t[j * n + i];
                        }
                        for (index i = 0; i < n; i++) {
                            A_t[j * n + i] = A_t[j * n + i] - R->x[j * n + (k - 1)] * Q->x[(k - 1) * n + i];
                        }

                        //Only one thread calculates the norm(A)
                        //and unsets the lock for the next column.
                        if (j == k) {
                            sum = 0;
                            for (index i = 0; i < n; i++) {
                                sum = sum + A_t[k * n + i] * A_t[k * n + i];
                            }
                            R->x[k * n + k] = sqrt(sum);
                            for (index i = 0; i < n; i++) {
                                Q->x[k * n + i] = A_t[k * n + i] / R->x[k * n + k];
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
        }
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
    cils<scalar, index, n>::cils_qr_decomposition_py(const index eval, const index qr_eval) {

        scalar error, time = omp_get_wtime();
        cils_qr_decomposition_py_helper();
        time = omp_get_wtime() - time;

        if (eval || qr_eval) {
            error = qr_validation<scalar, index, n>(A, Q, R, R_A, eval, qr_eval);
        }

        return {{}, time, (index) error};
    }

    template<typename scalar, typename index, index n>
    long int cils<scalar, index, n>::cils_qr_decomposition_py_helper() {
        PyObject * pName, *pModule, *pFunc;
        PyObject * pArgs, *pValue, *pVec;
        Py_Initialize();
        if (_import_array() < 0)
            PyErr_Print();

        npy_intp dim[1] = {A->size};

        pVec = PyArray_SimpleNewFromData(1, dim, NPY_DOUBLE, A->x);
        if (pVec == NULL) printf("There is a problem.\n");

        PyObject * sys_path = PySys_GetObject("path");
        PyList_Append(sys_path,
                      PyUnicode_FromString("/home/shilei/CLionProjects/babai_asyn/babai_asyn_c++/src/example"));
        pName = PyUnicode_FromString("py_qr");
        pModule = PyImport_Import(pName);

        if (pModule != NULL) {
            pFunc = PyObject_GetAttrString(pModule, "qr");
            if (pFunc && PyCallable_Check(pFunc)) {
                pArgs = PyTuple_New(2);
                if (PyTuple_SetItem(pArgs, 0, pVec) != 0) {
                    return false;
                }
                if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", n)) != 0) {
                    return false;
                }
                pValue = PyObject_CallObject(pFunc, pArgs);//Perform QR no return value
                if (pValue != NULL) {
                    PyArrayObject *q, *r;
                    PyArg_ParseTuple(pValue, "O|O", &q, &r);
                    Q->x = reinterpret_cast<scalar *>(PyArray_DATA(q));
                    R->x = reinterpret_cast<scalar *>(PyArray_DATA(r));
                } else {
                    PyErr_Print();
                }
            } else {
                if (PyErr_Occurred())
                    PyErr_Print();
                fprintf(stderr, "Cannot find function qr\n");
            }
        } else {
            PyErr_Print();
            fprintf(stderr, "Failed to load file\n");
        }
        return 0;
    }

    template<typename scalar, typename index, index n>
    returnType <scalar, index>
    cils<scalar, index, n>::cils_qr_decomposition_reduction() {
        scalar time = omp_get_wtime();
        scalar true_res = cils_qr_decomposition_reduction_helper();
        time = omp_get_wtime() - time;

        return {{}, time, true_res};
    }

    template<typename scalar, typename index, index n>
    scalar cils<scalar, index, n>::cils_qr_decomposition_reduction_helper() {
        // Start MATLAB engine synchronously
        using namespace matlab::engine;

        // Start MATLAB engine synchronously
        std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();

        //Create MATLAB data array factory
        matlab::data::ArrayFactory factory;

        // Create variables
        matlab::data::TypedArray<scalar> k_M = factory.createScalar<scalar>(program_def::k);
        matlab::data::TypedArray<scalar> SNR_M = factory.createScalar<scalar>(program_def::SNR);
        matlab::data::TypedArray<scalar> m_M = factory.createScalar<scalar>(log2(n));
        matlab::data::TypedArray<scalar> qr_M = factory.createScalar<scalar>(program_def::is_qr);
        matlabPtr->setVariable(u"k", std::move(k_M));
        matlabPtr->setVariable(u"m", std::move(m_M));
        matlabPtr->setVariable(u"SNR", std::move(SNR_M));
        matlabPtr->setVariable(u"qr", std::move(qr_M));

        // Call the MATLAB movsum function
        matlabPtr->eval(u" [A, R, Z, y, y_LLL, x_t, init_res, babai_norm] = sils_driver_mex(k, m, SNR, qr);");

        // Get the result
        matlab::data::TypedArray<scalar> const A_A = matlabPtr->getVariable(u"A");
        matlab::data::TypedArray<scalar> const R_N = matlabPtr->getVariable(u"R");
        matlab::data::TypedArray<scalar> const Z_N = matlabPtr->getVariable(u"Z");
        matlab::data::TypedArray<scalar> const y_R = matlabPtr->getVariable(u"y");//Reduced
        matlab::data::TypedArray<scalar> const y_N = matlabPtr->getVariable(u"y_LLL");//Original
        matlab::data::TypedArray<scalar> const x_T = matlabPtr->getVariable(u"x_t");
        matlab::data::TypedArray<scalar> const res = matlabPtr->getVariable(u"init_res");
        matlab::data::TypedArray<scalar> const b_n = matlabPtr->getVariable(u"babai_norm");
        matlab::data::ArrayDimensions dim = A_A.getDimensions();

//        cout << dim[0] << " " << dim[1] << endl;
        // Display the result
        value_input_helper(A_A, A);
        value_input_helper(R_N, R);
        value_input_helper(Z_N, Z);
        value_input_helper(y_N, y_L);//Original
        value_input_helper(y_R, y_A);//Reduced
        value_input_helper(x_T, &x_t);

        for (auto r : res) {
            init_res = r;
        }
        scalar b;
        for (auto r : b_n) {
            b = r;
        }
        if (b != 0)
            cout <<" The Babai Res from Matlab is :" << b << endl;

        return init_res;
    }

    template<typename scalar, typename index, index n>
    void cils<scalar, index, n>::value_input_helper(matlab::data::TypedArray<scalar> const x,
                                                    scalarType <scalar, index> *arr) {
        index i = 0;
        for (auto r : x) {
            arr->x[i] = r;
            ++i;
        }
    }

    template<typename scalar, typename index, index n>
    void cils<scalar, index, n>::value_input_helper(matlab::data::TypedArray<scalar> const x,
                                                    vector<index> *arr) {
        index i = 0;
        for (auto r : x) {
            arr->at(i) = r;
            ++i;
        }
    }
}
