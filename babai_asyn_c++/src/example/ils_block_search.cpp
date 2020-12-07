#include "../source/sils.cpp"
#include "../source/sils_block_search_omp.cpp"
#include <mpi.h>
template<typename scalar, typename index, index n>
void ils_block_search(index k, index SNR) {

    std::cout << "Init, size: " << n << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
    std::cout << "Init, SNR: " << SNR << std::endl;

    sils::sils<scalar, index, false, n> sils(k, SNR);

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
            auto reT = sils.sils_block_search_serial(&z_B, &d_s);
            auto res = sils::find_residual<scalar, index, n>(sils.R_A, sils.y_A, &reT.x);
            auto brr = sils::find_bit_error_rate<scalar, index, n>(&reT.x, &sils.x_t);
            printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Run time: %.5fs\n", size, res, brr,
                   reT.run_time);

            z_B.assign(n, 0);
            reT = sils.sils_babai_search_omp(9, 10, &z_B);
            res = sils::find_residual<scalar, index, n>(sils.R_A, sils.y_A, &reT.x);
            printf("Method: BAB_OMP, Block size: %d, Res: %.5f, Run time: %.5fs\n", 1, res, reT.run_time);
//            for (index n_proc = 1; n_proc <= 13; n_proc += 4) {
//                n_proc = n_proc == 13 ? 12 : n_proc;
            for (index n_proc = 1; n_proc <= 13; n_proc += 4) {
                z_B.assign(n, 0);
                for (index t = 0; t < n; t++) {
                    z_B[t] = pow(2, k) / 2;
                }

                index iter = n_proc == 1 ? 1 : 10;
                reT = sils.sils_block_search_omp_schedule(n_proc, iter, -1, "dynamic", &z_B, &d_s);
                res = sils::find_residual<scalar, index, n>(sils.R_A, sils.y_A, &reT.x);
                brr = sils::find_bit_error_rate<scalar, index, n>(&reT.x, &sils.x_t);
                printf("Method: ILS_OMP, Num of Threads: %d, Block size: %d, Iter: %d, Res: %.5f, BER: %.5f, Run time: %.5fs\n",
                       n_proc, size, reT.num_iter, res, brr, reT.run_time);


            }

            z_B.assign(n, 0);
            reT = sils.sils_babai_search_serial(&z_B);
            res = sils::find_residual<scalar, index, n>(sils.R_A, sils.y_A, &reT.x);
            brr = sils::find_bit_error_rate<scalar, index, n>(&reT.x, &sils.x_t);
            printf("Method: BBI_SER, Res: %.5f, BER: %.5f, Run time: %.5fs\n", res, brr, reT.run_time);
        }
    }

}

template<typename scalar, typename index, index n>
void plot_run(index k, index SNR, index min_proc, index max_proc, index max_num_iter, scalar stop) {

    printf("plot_run-------------------------------------------\n");

    std::cout << "Init, size: " << n << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
    std::cout << "Init, SNR: " << SNR << std::endl;

    sils::sils<scalar, index, false, n> sils(k, SNR);
    index size = 4, iter = 10;

    vector<index> z_B(n, 0), d(n / size, size), d_s(n / size, size);
    for (index i = d_s.size() - 2; i >= 0; i--) {
        d_s[i] += d_s[i + 1];
    }

    vector<scalar> bab_res(3, 0), bab_tim(3, 0), bab_ber(3, 0);
    vector<scalar> ser_res(3, 0), ser_tim(3, 0), ser_ber(3, 0);
    vector<scalar> omp_res(50, 0), omp_ber(50, 0), omp_tim(50, 0), omp_itr(50, 0);
    sils::returnType<scalar, index> reT;

    for (index p = 0; p < max_num_iter; p++) {
//        printf("%d,", p);
        sils.init();
        if (p == 0){
            printf("init_res: %.5f, sigma: %.5f\n", sils.init_res, sils.sigma);
        }
        if (p % 10 == 0) cout << "-";
        if (p % 500 == 0) cout << endl;
        std::cout.flush();

        for (index init = -1; init <= 1; init++) {
            for (index i = 0; i < 5; i++) {
                z_B.assign(n, 0);
                if (init == -1)
                    copy(z_B.begin(), z_B.end(), sils.x_R.begin());
                else if (init == 1)
                    z_B.assign(n, std::pow(2, k) / 2);

                reT = sils.sils_babai_search_serial(&z_B);
                bab_res[init + 1] += sils::find_residual<scalar, index, n>(sils.R_A, sils.y_A, &reT.x) / 5.0;
                bab_ber[init + 1] += sils::find_bit_error_rate<scalar, index, n>(&reT.x, &sils.x_t) / 5.0;
                bab_tim[init + 1] += reT.run_time / 5.0;

                z_B.assign(n, 0);
                if (init == -1)
                    copy(z_B.begin(), z_B.end(), sils.x_R.begin());
                else if (init == 1)
                    z_B.assign(n, std::pow(2, k) / 2);

                reT = sils.sils_block_search_serial(&z_B, &d_s);
                ser_res[init + 1] += sils::find_residual<scalar, index, n>(sils.R_A, sils.y_A, &reT.x) / 5.0;
                ser_ber[init + 1] += sils::find_bit_error_rate<scalar, index, n>(&reT.x, &sils.x_t) / 5.0;
                ser_tim[init + 1] += reT.run_time / 5.0;
                index l = 0;
                for (index n_proc = min_proc; n_proc <= max_proc; n_proc *= 2) {
                    z_B.assign(n, 0);
                    if (init == -1)
                        copy(z_B.begin(), z_B.end(), sils.x_R.begin());
                    else if (init == 1)
                        z_B.assign(n, std::pow(2, k) / 2);

                    n_proc = n_proc == 96 ? 64 : n_proc;
                    reT = sils.sils_block_search_omp(n_proc, iter, stop, &z_B, &d_s);

                    omp_res[init + 1 + 3 * l] +=
                            sils::find_residual<scalar, index, n>(sils.R_A, sils.y_A, &reT.x) / 5.0;
                    omp_ber[init + 1 + 3 * l] += sils::find_bit_error_rate<scalar, index, n>(&reT.x, &sils.x_t) / 5.0;
                    omp_tim[init + 1 + 3 * l] += reT.run_time / 5.0;
                    omp_itr[init + 1 + 3 * l] += reT.num_iter / 5.0;
                    l++;
                }
            }
        }
    }

    //Print Results:
    for (index init = -1; init <= 1; init++) {
        printf("++++++++++++++++++++++++++++++++++++++\n");
        std::cout << "Block, size: " << size << std::endl;
        std::cout << "Init, value: " << init << std::endl;
        printf("Method: BBA_SER, Block size: %d, Res: %.5f, BER: %.5f, Run time: %.5fs, Avg Run time: %.5fs\n",
               size, bab_res[init + 1] / max_num_iter, bab_ber[init + 1] / max_num_iter, bab_tim[init + 1],
               bab_tim[init + 1] / max_num_iter);
        printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Run time: %.5fs, Avg Run time: %.5fs\n",
               size, ser_res[init + 1] / max_num_iter, ser_ber[init + 1] / max_num_iter, ser_tim[init + 1],
               ser_tim[init + 1] / max_num_iter);
        index l = 0;
        for (index n_proc = min_proc; n_proc <= max_proc; n_proc *= 2) {
            n_proc = n_proc == 96 ? 64 : n_proc;
            printf("Method: ILS_OMP, n_proc: %d, Res :%.5f, BER: %.5f, num_iter: %.5f, Run time: %.5fs, Avg Run time: %.5fs\n",
                   n_proc, omp_res[init + 1 + 3 * l] / max_num_iter, omp_ber[init + 1 + 3 * l] / max_num_iter,
                   omp_itr[init + 1 + 3 * l] / max_num_iter,
                   omp_tim[init + 1 + 3 * l], omp_tim[init + 1 + 3 * l] / max_num_iter);
            l++;
        }
        printf("++++++++++++++++++++++++++++++++++++++\n");
    }

    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");
}

template<typename scalar, typename index, index n>
void plot_run_mpi(int argc, char *argv[], index k, index SNR, index min_proc, index max_proc, index max_num_iter, scalar stop) {

    printf("plot_run-------------------------------------------\n");

    std::cout << "Init, size: " << n << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
    std::cout << "Init, SNR: " << SNR << std::endl;

    sils::sils<scalar, index, false, n> sils(k, SNR);
    index size = 4, iter = 10;

    vector<index> z_B(n, 0), d(n / size, size), d_s(n / size, size);
    for (index i = d_s.size() - 2; i >= 0; i--) {
        d_s[i] += d_s[i + 1];
    }

    vector<scalar> bab_res(3, 0), bab_tim(3, 0), bab_ber(3, 0);
    vector<scalar> ser_res(3, 0), ser_tim(3, 0), ser_ber(3, 0);
    vector<scalar> omp_res(50, 0), omp_ber(50, 0), omp_tim(50, 0), omp_itr(50, 0);
    sils::returnType<scalar, index> reT;

    for (index p = 0; p < max_num_iter; p++) {
//        printf("%d,", p);
        sils.init();
        if (p == 0){
            printf("init_res: %.5f, sigma: %.5f\n", sils.init_res, sils.sigma);
        }
        if (p % 10 == 0) cout << "-";
        if (p % 500 == 0) cout << endl;
        std::cout.flush();

        for (index init = -1; init <= 1; init++) {
            for (index i = 0; i < 5; i++) {
                z_B.assign(n, 0);
                if (init == -1)
                    copy(z_B.begin(), z_B.end(), sils.x_R.begin());
                else if (init == 1)
                    z_B.assign(n, std::pow(2, k) / 2);

                reT = sils.sils_babai_search_serial(&z_B);
                bab_res[init + 1] += sils::find_residual<scalar, index, n>(sils.R_A, sils.y_A, &reT.x) / 5.0;
                bab_ber[init + 1] += sils::find_bit_error_rate<scalar, index, n>(&reT.x, &sils.x_t) / 5.0;
                bab_tim[init + 1] += reT.run_time / 5.0;

                z_B.assign(n, 0);
                if (init == -1)
                    copy(z_B.begin(), z_B.end(), sils.x_R.begin());
                else if (init == 1)
                    z_B.assign(n, std::pow(2, k) / 2);

                reT = sils.sils_block_search_serial(&z_B, &d_s);
                ser_res[init + 1] += sils::find_residual<scalar, index, n>(sils.R_A, sils.y_A, &reT.x) / 5.0;
                ser_ber[init + 1] += sils::find_bit_error_rate<scalar, index, n>(&reT.x, &sils.x_t) / 5.0;
                ser_tim[init + 1] += reT.run_time / 5.0;
            }
        }
    }

    int rank, n_ranks, numbers_per_rank;
    int my_first, my_last;
    int numbers = max_num_iter;

    // First call MPI_Init
    MPI_Init(&argc, &argv);
    // Get my rank and the number of ranks
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // Calculate the number of iterations for each rank
    numbers_per_rank = floor(numbers / n_ranks);
    if (numbers % n_ranks > 0) {
        // Add 1 in case the number of ranks doesn't divide the number of numbers
        numbers_per_rank += 1;
    }

    // Figure out the first and the last iteration for this rank
    my_first = rank * numbers_per_rank;
    my_last = my_first + numbers_per_rank;

//do something
    for (index p = my_first; p < my_last; p++) {
//        printf("%d,", p);
        sils.init();
        if (p == 0){
            printf("init_res: %.5f, sigma: %.5f\n", sils.init_res, sils.sigma);
        }
        if (p % 10 == 0) cout << "-";
        if (p % 500 == 0) cout << endl;
        std::cout.flush();

        for (index init = -1; init <= 1; init++) {
            for (index i = 0; i < 5; i++) {
                index l = 0;
                for (index n_proc = min_proc; n_proc <= max_proc; n_proc *= 2) {
                    z_B.assign(n, 0);
                    if (init == -1)
                        copy(z_B.begin(), z_B.end(), sils.x_R.begin());
                    else if (init == 1)
                        z_B.assign(n, std::pow(2, k) / 2);

                    n_proc = n_proc == 96 ? 64 : n_proc;
                    reT = sils.sils_block_search_omp(n_proc, iter, stop, &z_B, &d_s);

                    omp_res[init + 1 + 3 * l] +=
                            sils::find_residual<scalar, index, n>(sils.R_A, sils.y_A, &reT.x) / 5.0;
                    omp_ber[init + 1 + 3 * l] += sils::find_bit_error_rate<scalar, index, n>(&reT.x, &sils.x_t) / 5.0;
                    omp_tim[init + 1 + 3 * l] += reT.run_time / 5.0;
                    omp_itr[init + 1 + 3 * l] += reT.num_iter / 5.0;
                    l++;
                }
            }
        }
    }
    MPI_Finalize(); //MPI cleanup();

    //Print Results:
    for (index init = -1; init <= 1; init++) {
        printf("++++++++++++++++++++++++++++++++++++++\n");
        std::cout << "Block, size: " << size << std::endl;
        std::cout << "Init, value: " << init << std::endl;
        printf("Method: BBA_SER, Block size: %d, Res: %.5f, BER: %.5f, Run time: %.5fs, Avg Run time: %.5fs\n",
               size, bab_res[init + 1] / max_num_iter, bab_ber[init + 1] / max_num_iter, bab_tim[init + 1],
               bab_tim[init + 1] / max_num_iter);
        printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Run time: %.5fs, Avg Run time: %.5fs\n",
               size, ser_res[init + 1] / max_num_iter, ser_ber[init + 1] / max_num_iter, ser_tim[init + 1],
               ser_tim[init + 1] / max_num_iter);
        index l = 0;
        for (index n_proc = min_proc; n_proc <= max_proc; n_proc *= 2) {
            n_proc = n_proc == 96 ? 64 : n_proc;
            printf("Method: ILS_OMP, n_proc: %d, Res :%.5f, BER: %.5f, num_iter: %.5f, Run time: %.5fs, Avg Run time: %.5fs\n",
                   n_proc, omp_res[init + 1 + 3 * l] / max_num_iter, omp_ber[init + 1 + 3 * l] / max_num_iter,
                   omp_itr[init + 1 + 3 * l] / max_num_iter,
                   omp_tim[init + 1 + 3 * l], omp_tim[init + 1 + 3 * l] / max_num_iter);
            l++;
        }
        printf("++++++++++++++++++++++++++++++++++++++\n");
    }

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
    sils::sils<scalar, index, false, n> sils(k, SNR);

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
                z_B[i] = sils.x_R[i];
            }
        else if (init == 1)
            for (index i = 0; i < n; i++) {
                z_B[i] = std::pow(2, k) / 2;
            }

        auto reT = sils.sils_block_search_serial(&z_B, &d_s);
        auto res = sils::find_residual<scalar, index, n>(sils.R_A, sils.y_A, &reT.x);
        auto brr = sils::find_bit_error_rate<scalar, index, n>(&reT.x, &sils.x_t);

        printf("Method: ILS_SER, Block size: %d, Res: %.5f, Brr: %.5f, Run time: %.5fs\n", size, res, brr,
               reT.run_time);
        res = INFINITY;
        for (index n_proc = min_proc; n_proc <= max_proc; n_proc *= 2) {
            n_proc = n_proc == 96 ? 64 : n_proc;
            cout << d_s[d_s.size() - 1] << "," << n_proc << ",";
            for (index nswp = 0; nswp < 30; nswp++) {
                for (index t = 0; t < 3; t++) {
                    z_B.assign(n, 0);
                    if (init == -1)
                        for (index i = 0; i < n; i++) {
                            z_B[i] = sils.x_R[i];
                        }
                    else if (init == 1)
                        for (index i = 0; i < n; i++) {
                            z_B[i] = std::pow(2, k) / 2;
                        }

                    reT = sils.sils_block_search_omp(n_proc, nswp, -1, &z_B, &d_s);
                    scalar newres = sils::find_residual<scalar, index, n>(sils.R_A, sils.y_A, &reT.x);
                    scalar newbrr = sils::find_bit_error_rate<scalar, index, n>(&reT.x, &sils.x_t);
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
    sils::sils<double, int, true, n> sils(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %.5f seconds\n", end_time);

    start = omp_get_wtime();
    auto z_B = sils.sils_search(&sils.R_A, &sils.y_A);
    end_time = omp_get_wtime() - start;
    auto res = sils::find_residual<double, int, n>(&sils.R_A, &sils.y_A, z_B);
    printf("Thread: ILS, Sweep: 0, Res: %.5f, Run time: %.5fs\n", res, end_time);

//    sils::scalarType<double, int> z_BS = {(double *) calloc(n, sizeof(double)), n};
//    start = omp_get_wtime();
//    z_BS = *sils.sils_babai_search_serial(&z_BS);
//    end_time = omp_get_wtime() - start;
//    res = sils::find_residual<double, int, n>(&sils.R_A, &sils.y_A, &z_BS);
//    printf("Thread: SR, Sweep: 0, Res: %.5f, Run time: %.5fs\n", res, end_time);
}
