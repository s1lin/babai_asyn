#include "../source/SILS.cpp"

template<typename scalar, typename index, index n>
void ils_block_search(index k, index SNR) {

    std::cout << "Init, size: " << n << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
    std::cout << "Init, SNR: " << SNR << std::endl;

    scalar start = omp_get_wtime();
    sils::SILS<scalar, index, true, false, n> bsa(k, SNR);
    scalar end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);
    printf("-------------------------------------------\n");

    sils::scalarType<scalar, index> z_B{(scalar *) calloc(n, sizeof(scalar)), n};

    for (index size = 8; size <= 32; size *= 2) {
        //Initialize the block vector
        vector<index> d(n / size, size);
        sils::scalarType<index, index> d_s{d.data(), (index) d.size()};
        for (index i = d_s.size - 2; i >= 0; i--) {
            d_s.x[i] += d_s.x[i + 1];
        }

        for (index i = 0; i < 1; i++) {
            printf("++++++++++++++++++++++++++++++++++++++\n");
            free(z_B.x);
            z_B.x = (scalar *) calloc(n, sizeof(scalar));
            auto reT = bsa.sils_block_search_serial(&bsa.R_A, &bsa.y_A, &z_B, &d_s);
            auto res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, reT.x);
            printf("Method: ILS_SER, Block size: %d, Res: %.5f, Run time: %fs\n", size, res, reT.run_time);
            sils::scalarType<scalar, index> z_B_p{(scalar *) calloc(n, sizeof(scalar)), n};

            for (index n_proc = 2; n_proc <= 8; n_proc *= 2) {
                free(z_B.x);
                z_B.x = (scalar *) calloc(n, sizeof(scalar));
                z_B_p.x = (scalar *) calloc(n, sizeof(scalar));
                for (index t = 0; t < n; t++) {
                    z_B.x[t] = pow(2, k) / 2;
                    z_B_p.x[t] = pow(2, k) / 2;
                }
                index iter = 5;
                reT = bsa.sils_block_search_omp(n_proc, iter, &bsa.R_A, &bsa.y_A, &z_B, &z_B_p, &d_s);
                res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, reT.x);
                printf("Method: ILS_OMP, Num of Threads: %d, Block size: %d, Iter: %d, Res: %.5f, Run time: %fs\n",
                       n_proc,
                       size, reT.num_iter, res, reT.run_time);
            }

            free(z_B.x);
            free(z_B_p.x);
            z_B.x = (scalar *) calloc(n, sizeof(scalar));
            sils::scalarType<scalar, index> z_BS = {(scalar *) calloc(n, sizeof(scalar)), n};
            reT = bsa.sils_babai_search_serial(&z_BS);
            res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, reT.x);
            printf("Method: BBI_SER, Res: %.5f, Run time: %fs\n", res, reT.run_time);
        }
    }

}

template<typename scalar, typename index, index n>
void plot_run(index k, index SNR) {

    std::cout << "Init, size: " << n << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
    std::cout << "Init, SNR: " << SNR << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    scalar start = omp_get_wtime();
    sils::SILS<scalar, index, true, false, n> bsa(k, SNR);
    scalar end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);
    printf("-------------------------------------------\n");

    string fx = "res_" + to_string(n) + "_" + to_string(SNR) + "_" + to_string(k) + ".csv";
    ofstream file(fx);
    if (file.is_open()) {

        sils::scalarType<scalar, index> z_B{(scalar *) calloc(n, sizeof(scalar)), n};
        file << n << "," << std::pow(4, k) << "," << SNR << ",\n";
        for (index init = -1; init <= 1; init++) {
            for (index size = 8; size <= 32; size *= 2) {
                printf("++++++++++++++++++++++++++++++++++++++\n");
                std::cout << "Block, size: " << size << std::endl;
                std::cout << "Init, value: " << init << std::endl;
                vector<index> d(n / size, size);
                sils::scalarType<index, index> d_s{d.data(), (index) d.size()};
                for (index i = d_s.size - 2; i >= 0; i--) {
                    d_s.x[i] += d_s.x[i + 1];
                }

                vector<scalar> min_res(50, INFINITY), res(50, 0), tim(50, 0), itr(50, 0);
                scalar omp_res = 0, omp_time = 0, num_iter = 0;
                scalar ser_res = 0, ser_time = 0;

                std::cout << "Babai Serial:" << std::endl;
                for (index i = 0; i < 10; i++) {
                    free(z_B.x);
                    z_B.x = (scalar *) calloc(n, sizeof(scalar));
                    if (init == -1)
                        for (index t = 0; t < n; t++) {
                            z_B.x[t] = bsa.x_R.x[t];
                        }
                    else if (init == 1)
                        for (index t = 0; t < n; t++) {
                            z_B.x[t] = std::pow(2, k) / 2;
                        }
                    auto reT = bsa.sils_babai_search_serial(&z_B);
                    ser_res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, reT.x);
                    res[0] += ser_res;
                    tim[0] += reT.run_time;
                }
                printf("Thread: SR, Res: %.5f, Run time: %fs\n", res[0] / 10, tim[0] / 10);

                file << size << "," << res[0] / 10 << "," << tim[0] / 10 << ",\n";

                std::cout << "Block Serial:" << std::endl;
                for (index i = 0; i < 10; i++) {
                    free(z_B.x);
                    z_B.x = (scalar *) calloc(n, sizeof(scalar));
                    if (init == -1)
                        for (index t = 0; t < n; t++) {
                            z_B.x[t] = bsa.x_R.x[t];
                        }
                    else if (init == 1)
                        for (index t = 0; t < n; t++) {
                            z_B.x[t] = std::pow(2, k) / 2;
                        }
                    auto reT = bsa.sils_block_search_serial(&bsa.R_A, &bsa.y_A, &z_B, &d_s);
                    ser_res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, reT.x);
                    res[1] += ser_res;
                    tim[1] += reT.run_time;
                }
                printf("Method: ILS_SER, Block size: %d, Res: %.5f, Run time: %fs\n", size, res[1] / 10, tim[1] / 10);
                file << size << "," << res[1] / 10 << "," << tim[1] / 10 << ",\n";

                std::cout << "Block OMP:" << std::endl;
                sils::scalarType<scalar, index> z_B_p{(scalar *) calloc(n, sizeof(scalar)), n};
                index l = 2;
                for (index i = 0; i < 10; i++) {
                    for (index n_proc = 3; n_proc <= 192; n_proc *= 4) {
                        if (n_proc == 192)
                            n_proc = 64;
                        free(z_B.x);
                        z_B.x = (scalar *) calloc(n, sizeof(scalar));
                        if (init == -1)
                            for (index t = 0; t < n; t++) {
                                z_B.x[t] = bsa.x_R.x[t];
                            }
                        else if (init == 1)
                            for (index t = 0; t < n; t++) {
                                z_B.x[t] = std::pow(2, k) / 2;
                            }
                        index iter = 6;
                        if(k == 3) iter = 10;
                        auto reT = bsa.sils_block_search_omp(n_proc, iter, &bsa.R_A, &bsa.y_A, &z_B, &z_B_p, &d_s);
                        omp_res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, reT.x);
                        if (omp_res < min_res[l])
                            min_res[l] = omp_res;
                        res[l] = min_res[l];
                        tim[l] += reT.run_time;
                        itr[l] += reT.num_iter;
                        l++;
                    }
//                    min_res = vector<scalar>(50, INFINITY);
                    l = 2;
                }
//                cout << "min_res:" << min_res << endl;
                l = 2;
                for (index n_proc = 3; n_proc <= 192; n_proc *= 4) {
                    if (n_proc == 192)
                        n_proc = 64;
                    file << size << "," << n_proc << ","
                         << res[l] << ","
                         << tim[l] / 10 << ","
                         << itr[l] / 10 << ",\n";

                    printf("Block Size: %d, n_proc: %d, res :%f, num_iter: %f, Average time: %fs\n",
                           size, n_proc,
                           res[l],
                           itr[l] / 10,
                           tim[l] / 10);
                    l++;
                }
                file << "Next,\n";

            }
        }
    }
    file.close();
    printf("-------------------------------------------\n");
}

template<typename scalar, typename index, index n>
void plot_res(index k, index SNR) {
    printf("-------------------------------------------\n");
    std::cout << "Init, size: " << n << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
    std::cout << "Init, SNR: " << SNR << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    scalar start = omp_get_wtime();
    sils::SILS<scalar, index, true, false, n> bsa(k, SNR);
    scalar end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);
    printf("-------------------------------------------\n");

    for (index init = -1; init <= 1; init++) {
        for (index size = 8; size <= 32; size *= 2) {
            vector<index> d(n / size, size);
            sils::scalarType<index, index> d_s{d.data(), (index) d.size()};
            cout << init << "\n";
            for (index i = d_s.size - 2; i >= 0; i--) {
                d_s.x[i] += d_s.x[i + 1];
            }
            scalar omp_res = 0, omp_time = 0, num_iter = 0;
            sils::scalarType<scalar, index> z_B{(scalar *) calloc(n, sizeof(scalar)), n};
            sils::scalarType<scalar, index> z_B_p{(scalar *) calloc(n, sizeof(scalar)), n};
            sils::scalarType<scalar, index> z_B_s{(scalar *) calloc(n, sizeof(scalar)), n};

            z_B_s.x = (scalar *) calloc(n, sizeof(scalar));
            if (init == -1)
                for (index i = 0; i < n; i++) {
                    z_B.x[i] = bsa.x_R.x[i];
                }
            else if (init == 1)
                for (index i = 0; i < n; i++) {
                    z_B.x[i] = std::pow(2, k) / 2;
                }
            auto reT = bsa.sils_block_search_serial(&bsa.R_A, &bsa.y_A, &z_B_s, &d_s);
            auto res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, reT.x);
            auto y = bsa.y_A;
            auto R = bsa.R_A;
            printf("Method: ILS_SER, Block size: %d, Res: %.5f, Run time: %fs\n", size, res, reT.run_time);
            res = INFINITY;
            for (index n_proc = 3; n_proc <= 48; n_proc *= 4) {
                cout << d_s.x[d_s.size - 1] << "," << n_proc << ",";
                for (index nswp = 0; nswp < 12; nswp++) {
                    for (index t = 0; t < 10; t++) {
                        free(z_B.x);
                        z_B.x = (scalar *) calloc(n, sizeof(scalar));
                        if (init == -1)
                            for (index i = 0; i < n; i++) {
                                z_B.x[i] = bsa.x_R.x[i];
                            }
                        else if (init == 1)
                            for (index i = 0; i < n; i++) {
                                z_B.x[i] = std::pow(2, k) / 2;
                            }

                        reT = bsa.sils_block_search_omp(n_proc, nswp, &bsa.R_A, &bsa.y_A, &z_B, &z_B_p, &d_s);
                        scalar newres = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, reT.x);
                        res = newres < res ? newres : res;
                    }
                    cout << res << ",";
                    res = INFINITY;
                }
                cout << endl;
//                cout << sils::norm<scalar, index, n>(&z_B_s, &z_B) << endl;

            }
        }
//        free(z_B.x);
    }
}

template<typename scalar, typename index, index n>
void test_ils_search() {
    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    sils::SILS<double, int, true, false, n> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);

    start = omp_get_wtime();
    auto z_B = bsa.sils_search(&bsa.R_A, &bsa.y_A);
    end_time = omp_get_wtime() - start;
    auto res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, z_B);
    printf("Thread: ILS, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);

    sils::scalarType<double, int> z_BS = {(double *) calloc(n, sizeof(double)), n};
    start = omp_get_wtime();
    z_BS = *bsa.sils_babai_search_serial(&z_BS);
    end_time = omp_get_wtime() - start;
    res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_BS);
    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %fs\n", res, end_time);
}
