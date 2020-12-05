#include "../source/SILS.cpp"

template<typename scalar, typename index, index n>
void ils_block_search(index k, index SNR) {

    std::cout << "Init, size: " << n << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
    std::cout << "Init, SNR: " << SNR << std::endl;

    scalar start = omp_get_wtime();
    sils::SILS<scalar, index, true, n> bsa(k, SNR);
    scalar end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %.5f seconds\n", end_time);
    printf("-------------------------------------------\n");

    vector<index> z_B(n, 0);

    for (index size = 8; size <= 32; size *= 2) {
        //Initialize the block vector
        vector<index> d(n / size, size), d_s(n / size, size);
        for (index i = d_s.size() - 2; i >= 0; i--) {
            d_s[i] += d_s[i + 1];
        }

        for (index i = 0; i < 1; i++) {
            printf("++++++++++++++++++++++++++++++++++++++\n");
            z_B.assign(n, 0);
//            sils::display_vector<index>(&z_B);
            auto reT = bsa.sils_block_search_serial(&z_B, &d_s);
            auto res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &reT.x);
            auto brr = sils::find_bit_error_rate<scalar, index, n>(&reT.x, &bsa.x_tA);
            printf("Method: ILS_SER, Block size: %d, Res: %.5f, BRR: %.5f, Run time: %.5fs\n", size, res, brr,
                   reT.run_time);

            z_B.assign(n, 0);
            reT = bsa.sils_babai_search_omp(9, 10, &z_B);
            res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &reT.x);
            printf("Method: BAB_OMP, Block size: %d, Res: %.5f, Run time: %.5fs\n", 1, res, reT.run_time);
//            for (index n_proc = 1; n_proc <= 13; n_proc += 4) {
//                n_proc = n_proc == 13 ? 12 : n_proc;
            for (index n_proc = 1; n_proc <= 13; n_proc += 4) {
                z_B.assign(n, 0);
                for (index t = 0; t < n; t++) {
                    z_B[t] = pow(2, k) / 2;
                }

                index iter = n_proc == 1 ? 1 : 10;
                reT = bsa.sils_block_search_omp(n_proc, iter, -1, &z_B, &d_s);
                res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &reT.x);
                brr = sils::find_bit_error_rate<scalar, index, n>(&reT.x, &bsa.x_tA);
                printf("Method: ILS_OMP, Num of Threads: %d, Block size: %d, Iter: %d, Res: %.5f, BRR: %.5f, Run time: %.5fs\n",
                       n_proc, size, reT.num_iter, res, brr, reT.run_time);


            }

            z_B.assign(n, 0);
            reT = bsa.sils_babai_search_serial(&z_B);
            res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &reT.x);
            brr = sils::find_bit_error_rate<scalar, index, n>(&reT.x, &bsa.x_tA);
            printf("Method: BBI_SER, Res: %.5f, BRR: %.5f, Run time: %.5fs\n", res, brr, reT.run_time);
        }
    }

}

template<typename scalar, typename index, index n>
void plot_run(index k, index SNR, index min_proc, index max_proc, scalar stop) {
    printf("plot_run-------------------------------------------\n");
    std::cout << "Init, size: " << n << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
    std::cout << "Init, SNR: " << SNR << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    scalar start = omp_get_wtime();
    sils::SILS<scalar, index, true, n> bsa(k, SNR);
    scalar end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %.5f seconds\n", end_time);
    printf("-------------------------------------------\n");
    index size = 16;

    vector<index> z_B(n, 0);
    vector<index> d(n / size, size), d_s(n / size, size);
    for (index i = d_s.size() - 2; i >= 0; i--) {
        d_s[i] += d_s[i + 1];
    }

    for (index init = -1; init <= 1; init++) {
//        for (index size = 16; size <= 16; size *= 2) {
        printf("++++++++++++++++++++++++++++++++++++++\n");
        std::cout << "Block, size: " << size << std::endl;
        std::cout << "Init, value: " << init << std::endl;
        std::cout << "Babai Serial:" << std::endl;
        scalar omp_res = 0, omp_time = 0, num_iter = 0, omp_brr = 0;
        scalar ser_res = 0, ser_time = 0, ser_brr = 0;
        vector<scalar> min_res(50, INFINITY), min_brr(50, INFINITY), res(50, 0), tim(50, 0), itr(50, 0), brr(50, 0);
        sils::returnType<scalar, index> reT;

        for (index i = 0; i < 5; i++) {
            z_B.assign(n, 0);
            if (init == -1)
                for (index t = 0; t < n; t++) {
                    z_B[t] = bsa.x_R[t];
                }
            else if (init == 1)
                for (index t = 0; t < n; t++) {
                    z_B[t] = std::pow(2, k) / 2;
                }
            reT = bsa.sils_babai_search_serial(&z_B);
            ser_res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &reT.x);
            ser_brr = sils::find_bit_error_rate<scalar, index, n>(&reT.x, &bsa.x_tA);
            res[0] += ser_res;
            brr[0] += ser_brr;
            tim[0] += reT.run_time;
        }
        printf("Method: ILS_SER, Block size: %d, Res: %.5f, BRR: %.5f, Run time: %.5fs\n",
               size, res[0] / 5, brr[0] / 5, tim[0] / 5);

        std::cout << "Block Serial:" << std::endl;
        for (index i = 0; i < 5; i++) {
            z_B.assign(n, 0);
            if (init == -1)
                for (index t = 0; t < n; t++) {
                    z_B[t] = bsa.x_R[t];
                }
            else if (init == 1)
                for (index t = 0; t < n; t++) {
                    z_B[t] = std::pow(2, k) / 2;
                }
            reT = bsa.sils_block_search_serial(&z_B, &d_s);
            ser_res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &reT.x);
            ser_brr = sils::find_bit_error_rate<scalar, index, n>(&reT.x, &bsa.x_tA);
            res[1] += ser_res;
            brr[1] += ser_brr;
            tim[1] += reT.run_time;
        }
        printf("Method: ILS_SER, Block size: %d, Res: %.5f, BRR: %.5f, Run time: %.5fs\n",
               size, res[1] / 5, brr[1] / 5, tim[1] / 5);

        std::cout << "Block OMP, " << "Stopping criteria: " << stop << std::endl;
        index l = 2;

        for (index n_proc = min_proc; n_proc <= max_proc; n_proc *= 2) {
            cout << "Threads:" << n_proc << endl;
            for (index i = 0; i < 2000; i++) {
                n_proc = n_proc == 96 ? 64 : n_proc;
                z_B.assign(n, 0);

                if (init == -1)
                    for (index t = 0; t < n; t++) {
                        z_B[t] = bsa.x_R[t];
                    }
                else if (init == 1)
                    for (index t = 0; t < n; t++) {
                        z_B[t] = std::pow(2, k) / 2;
                    }

                index iter = 10;
                if (k == 3) iter = 15;
//                sils::display_vector<index>(&z_BS);
                reT = bsa.sils_block_search_omp(n_proc, iter, stop, &z_B, &d_s);
                omp_res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &reT.x);
                omp_brr = sils::find_bit_error_rate<scalar, index, n>(&reT.x, &bsa.x_tA);
                if (omp_res < min_res[l]) min_res[l] = omp_res;
                if (omp_brr < min_brr[l]) min_brr[l] = omp_brr;
                res[l] = min_res[l];
                brr[l] = min_brr[l];
                tim[l] += reT.run_time;
                itr[l] += reT.num_iter;
            }
            l++;
        }

        l = 2;
        for (index n_proc = min_proc; n_proc <= max_proc; n_proc *= 2) {
            n_proc = n_proc == 96 ? 64 : n_proc;
            printf("Block Size: %d, n_proc: %d, Res :%.5f, BRR: %.5f, num_iter: %.5f, Average time: %.5fs\n",
                   size, n_proc, res[l], brr[l], itr[l] / 2000, tim[l] / 2000);
            l++;
        }

    }
//    }
    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");
}

template<typename scalar, typename index, index n>
void plot_res(index k, index SNR, index min_proc, index max_proc) {
    printf("plot_res-------------------------------------------\n");
    std::cout << "Init, size: " << n << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
    std::cout << "Init, SNR: " << SNR << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    scalar start = omp_get_wtime();
    sils::SILS<scalar, index, true, n> bsa(k, SNR);
    scalar end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %.5f seconds\n", end_time);
    printf("-------------------------------------------\n");

    index size = 16;

    vector<index> z_B(n, 0);
    vector<index> d(n / size, size), d_s(n / size, size);
    for (index i = d_s.size() - 2; i >= 0; i--) {
        d_s[i] += d_s[i + 1];
    }

    for (index init = -1; init <= 1; init++) {
        cout << init << "\n";
        scalar omp_res = 0, omp_time = 0, num_iter = 0;

        z_B.assign(n, 0);

        if (init == -1)
            for (index i = 0; i < n; i++) {
                z_B[i] = bsa.x_R[i];
            }
        else if (init == 1)
            for (index i = 0; i < n; i++) {
                z_B[i] = std::pow(2, k) / 2;
            }

        auto reT = bsa.sils_block_search_serial(&z_B, &d_s);
        auto res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &reT.x);
        auto brr = sils::find_bit_error_rate<scalar, index, n>(&reT.x, &bsa.x_tA);

        printf("Method: ILS_SER, Block size: %d, Res: %.5f, Brr: %.5f, Run time: %.5fs\n", size, res, brr,
               reT.run_time);
        res = INFINITY;
        for (index n_proc = min_proc; n_proc <= max_proc; n_proc *= 2) {
            n_proc = n_proc == 96 ? 64 : n_proc;
            cout << d_s[d_s.size() - 1] << "," << n_proc << ",";
            for (index nswp = 0; nswp < 100; nswp++) {
                for (index t = 0; t < 3; t++) {
                    z_B.assign(n, 0);
                    if (init == -1)
                        for (index i = 0; i < n; i++) {
                            z_B[i] = bsa.x_R[i];
                        }
                    else if (init == 1)
                        for (index i = 0; i < n; i++) {
                            z_B[i] = std::pow(2, k) / 2;
                        }

                    reT = bsa.sils_block_search_omp(n_proc, nswp, -1, &z_B, &d_s);
                    scalar newres = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &reT.x);
                    scalar newbrr = sils::find_bit_error_rate<scalar, index, n>(&reT.x, &bsa.x_tA);
                    res = newres < res ? newres : res;
                    brr = newbrr < brr ? newbrr : brr;
                }
                printf("res = %.5f, brr =%.5f,", res, brr);
                brr = INFINITY;
                res = INFINITY;
            }
            cout << endl;
        }
    }

    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");
}

template<typename scalar, typename index, index n>
void test_ils_search() {
    std::cout << "Init, size: " << n << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    double start = omp_get_wtime();
    sils::SILS<double, int, true, n> bsa(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %.5f seconds\n", end_time);

    start = omp_get_wtime();
    auto z_B = bsa.sils_search(&bsa.R_A, &bsa.y_A);
    end_time = omp_get_wtime() - start;
    auto res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, z_B);
    printf("Thread: ILS, Sweep: 0, Res: %.5f, Run time: %.5fs\n", res, end_time);

//    sils::scalarType<double, int> z_BS = {(double *) calloc(n, sizeof(double)), n};
//    start = omp_get_wtime();
//    z_BS = *bsa.sils_babai_search_serial(&z_BS);
//    end_time = omp_get_wtime() - start;
//    res = sils::find_residual<double, int, n>(&bsa.R_A, &bsa.y_A, &z_BS);
//    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %.5fs\n", res, end_time);
}
