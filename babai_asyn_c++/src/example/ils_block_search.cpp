#include "../source/cils.cpp"
#include "../source/cils_block_search.cpp"
#include <mpi.h>

template<typename scalar, typename index>
struct results {
    vector<scalar> omp_res, omp_ber, omp_tim, omp_itr;
};

template<typename scalar, typename index, index n>
void ils_block_search(index k, index SNR) {

    std::cout << "Init, size: " << n << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
    std::cout << "Init, SNR: " << SNR << std::endl;

    cils::cils<scalar, index, true, n> cils(k, SNR);
    cils.init(1);
    vector<index> z_B(n, 0);

    for (index size = 16; size <= 16; size *= 2) {
        //Initialize the block vector
        vector<index> d(n / size, size), d_s(n / size, size);
        for (index i = d_s.size() - 2; i >= 0; i--) {
            d_s[i] += d_s[i + 1];
        }

        for (index i = 0; i < 1; i++) {
            printf("++++++++++++++++++++++++++++++++++++++\n");
            z_B.assign(n, 0);
//            cils::display_vector<index>(&z_B);
            auto reT = cils.cils_block_search_serial(&z_B, &d_s);
            auto res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &reT.x);
            auto ber = cils::find_bit_error_rate<scalar, index, n>(&reT.x, &cils.x_t, false);
            printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Time: %.5fs\n", size, res, ber,
                   reT.run_time);
            scalar ils_tim = reT.run_time;

            z_B.assign(n, 0);
            reT = cils.cils_babai_search_serial(&z_B);
            res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &reT.x);
            ber = cils::find_bit_error_rate<scalar, index, n>(&reT.x, &cils.x_t, false);
            printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Time: %.5fs\n", res, ber, reT.run_time);
            scalar ser_tim = reT.run_time;

            for (index n_proc = 4; n_proc <= 12; n_proc += 4) {
                z_B.assign(n, 0);
                reT = cils.cils_babai_search_omp(n_proc, 3, &z_B);
                res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &reT.x);
                ber = cils::find_bit_error_rate<scalar, index, n>(&reT.x, &cils.x_t, false);
                printf("Method: BAB_OMP, n_proc: %d, Res: %.5f, BER: %.5f, Num_iter: %d, Time: %.5fs, SpeedUp: %.3f\n",
                       n_proc, res, ber, reT.num_iter, reT.run_time, (ser_tim/reT.run_time));

                z_B.assign(n, 0);
//                for (index t = 0; t < n; t++) {
//                    z_B[t] = pow(2, k) / 2;
//                }

                reT = cils.cils_block_search_omp(n_proc, 6, 0, &z_B, &d_s);
                res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &reT.x);
                ber = cils::find_bit_error_rate<scalar, index, n>(&reT.x, &cils.x_t, false);
                printf("Method: ILS_OMP, n_proc: %d, Res: %.5f, BER: %.5f, Num_iter: %d, Time: %.5fs, SpeedUp: %.3f\n",
                       n_proc, res, ber, reT.num_iter, reT.run_time, (ils_tim/reT.run_time));
            }


        }
    }

}

template<typename scalar, typename index, index n>
void plot_run(index k, index SNR, index min_proc, index max_proc, index max_num_iter, scalar stop, bool is_qr) {

    printf("plot_run-------------------------------------------\n");

    std::cout << "Init, size: " << n << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
    std::cout << "Init, SNR: " << SNR << std::endl;

    cils::cils<scalar, index, true, n> cils(k, SNR);
    index size = 16, iter = 10;

    vector<index> z_B(n, 0), d(n / size, size), d_s(n / size, size);
    for (index i = d_s.size() - 2; i >= 0; i--) {
        d_s[i] += d_s[i + 1];
    }

    vector<scalar> bab_res(3, 0), bab_tim(3, 0), bab_ber(3, 0);
    vector<scalar> ser_res(3, 0), ser_tim(3, 0), ser_ber(3, 0);
    vector<scalar> omp_res(50, 0), omp_ber(50, 0), omp_tim(50, 0), omp_itr(50, 0);
    cils::returnType<scalar, index> reT;
    cils.init(is_qr);

    for (index p = 0; p < max_num_iter; p++) {
        if (p == 0) {
            printf("init_res: %.5f, sigma: %.5f\n", cils.init_res, cils.sigma);
        }
        if (p % 10 == 0) cout << "-";
        if (p % 500 == 0) cout << endl;
        std::cout.flush();

        for (index init = -1; init <= 1; init++) {
            z_B.assign(n, 0);
            if (init == -1)
                copy(z_B.begin(), z_B.end(), cils.x_R.begin());
            else if (init == 1)
                z_B.assign(n, std::pow(2, k) / 2);

            reT = cils.cils_babai_search_serial(&z_B);
            bab_res[init + 1] += cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &reT.x);
            bab_ber[init + 1] += cils::find_bit_error_rate<scalar, index, n>(&reT.x, &cils.x_t, cils.qam == 1);
            bab_tim[init + 1] += reT.run_time;

            z_B.assign(n, 0);
            if (init == -1)
                copy(z_B.begin(), z_B.end(), cils.x_R.begin());
            else if (init == 1)
                z_B.assign(n, std::pow(2, k) / 2);

            reT = cils.cils_block_search_serial(&z_B, &d_s);
            ser_res[init + 1] += cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &reT.x);
            ser_ber[init + 1] += cils::find_bit_error_rate<scalar, index, n>(&reT.x, &cils.x_t, cils.qam == 1);
            ser_tim[init + 1] += reT.run_time;
            index l = 0;
            for (index n_proc = min_proc; n_proc <= max_proc; n_proc += 12) {
                z_B.assign(n, 0);
                if (init == -1)
                    copy(z_B.begin(), z_B.end(), cils.x_R.begin());
                else if (init == 1)
                    z_B.assign(n, std::pow(2, k) / 2);

                n_proc = n_proc == 96 ? 64 : n_proc;
                reT = cils.cils_block_search_omp(n_proc, iter, stop, &z_B, &d_s);

                omp_res[init + 1 + 3 * l] +=
                        cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &reT.x);
                omp_ber[init + 1 + 3 * l] += cils::find_bit_error_rate<scalar, index, n>(&reT.x, &cils.x_t, cils.qam == 1);
                omp_tim[init + 1 + 3 * l] += reT.run_time;
                omp_itr[init + 1 + 3 * l] += reT.num_iter;
                l++;
            }

        }
    }

    //Print Results:
    for (index init = -1; init <= 1; init++) {
        printf("++++++++++++++++++++++++++++++++++++++\n");
        std::cout << "Block, size: " << size << std::endl;
        std::cout << "Init, value: " << init << std::endl;
        printf("Method: BBA_SER, Block size: %d, Res: %.5f, BER: %.5f, Time: %.5fs, Avg Time: %.5fs\n",
               size, bab_res[init + 1] / max_num_iter, bab_ber[init + 1] / max_num_iter, bab_tim[init + 1],
               bab_tim[init + 1] / max_num_iter);
        printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Time: %.5fs, Avg Time: %.5fs\n",
               size, ser_res[init + 1] / max_num_iter, ser_ber[init + 1] / max_num_iter, ser_tim[init + 1],
               ser_tim[init + 1] / max_num_iter);
        index l = 0;
        for (index n_proc = min_proc; n_proc <= max_proc; n_proc += 12) {
            n_proc = n_proc == 96 ? 64 : n_proc;
            printf("Method: ILS_OMP, n_proc: %d, Res :%.5f, BER: %.5f, num_iter: %.5f, Time: %.5fs, Avg Time: %.5fs\n",
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
void plot_run_mpi(int argc, char *argv[]) {
    index k = 1, index_2 = 0, stop = 0, mode = 1, max_num_iter = 2000, SNR = 15;

    if (argc != 1) {
        k = stoi(argv[1]);
        index_2 = stoi(argv[2]);
        stop = stoi(argv[3]);
        mode = stoi(argv[4]);
        max_num_iter = stoi(argv[5]);
    }
    double omp_res[2000], omp_ber[2000], omp_tim[2000], omp_itr[2000];
    //rbuf = (int *)malloc(gsize*100*sizeof(int));
    int rank, n_ranks, numbers_per_rank;
    int my_first, my_last;

    // First call MPI_Init
    MPI_Init(&argc, &argv);
    // Get my rank and the number of ranks
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // Calculate the number of iterations for each rank
    numbers_per_rank = floor(max_num_iter / n_ranks);
    if (max_num_iter % n_ranks > 0) {
        // Add 1 in case the number of ranks doesn't divide the number of numbers
        numbers_per_rank += 1;
    }

    printf("plot_run-------------------------------------------\n");

    std::cout << "Init, size: " << n << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
    std::cout << "Init, SNR: " << SNR << std::endl;

    cils::cils<scalar, index, false, n> cils(k, SNR);
    index size = 4, iter = 10;

    vector<index> z_B(n, 0), d(n / size, size), d_s(n / size, size);
    for (index i = d_s.size() - 2; i >= 0; i--) {
        d_s[i] += d_s[i + 1];
    }

    MPI_Type_vector(n, n_ranks, 50, MPI_DOUBLE, &omp_res);
    MPI_Type_vector(n, n_ranks, 50, MPI_DOUBLE, &omp_ber);
    MPI_Type_vector(n, n_ranks, 50, MPI_DOUBLE, &omp_tim);
    MPI_Type_vector(n, n_ranks, 50, MPI_DOUBLE, &omp_itr);

//    MPI_Type_commit(&omp_res);
//    MPI_Type_commit(&omp_ber);
//    MPI_Type_commit(&omp_tim);
//    MPI_Type_commit(&omp_itr);


    cils::returnType<scalar, index> reT;
    index first = rank * numbers_per_rank;
    index last = my_first + numbers_per_rank;
    index min_proc = 2, max_proc = 8;
//do something
    for (index p = first; p < last; p++) {
//        printf("%d,", p);
        cils.init(false);
        if (p % 500 == 0) {
            printf("init_res: %.5f, sigma: %.5f\n", cils.init_res, cils.sigma);
        }
        std::cout.flush();

        for (index init = -1; init <= 1; init++) {
            for (index i = 0; i < 5; i++) {
                index l = 0;
                for (index n_proc = min_proc; n_proc <= max_proc; n_proc += 12) {
                    z_B.assign(n, 0);
                    if (init == -1)
                        copy(z_B.begin(), z_B.end(), cils.x_R.begin());
                    else if (init == 1)
                        z_B.assign(n, std::pow(2, k) / 2);

                    n_proc = n_proc == 96 ? 64 : n_proc;
                    reT = cils.cils_block_search_omp(n_proc, iter, stop, &z_B, &d_s);

                    omp_res[init + 1 + 3 * l] +=
                            cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &reT.x);
                    omp_ber[init + 1 + 3 * l] += cils::find_bit_error_rate<scalar, index, n>(&reT.x, &cils.x_t);
                    omp_tim[init + 1 + 3 * l] += reT.run_time;
                    omp_itr[init + 1 + 3 * l] += reT.num_iter;
                    l++;
                }
            }
        }
    }

//    MPI_Gather(vecin, 1, omp_res, vecout, n, MPI_DOUBLE, first, MPI_COMM_WORLD);


    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");
    /* do a zero length gather */
    MPI_Gather(NULL, 0, MPI_BYTE, NULL, 0, MPI_BYTE, 0, MPI_COMM_WORLD);

    MPI_Finalize();

}

template<typename scalar, typename index, index n>
void plot_res(index k, index SNR, index min_proc, index max_proc, bool is_qr) {
    printf("plot_res-------------------------------------------\n");
    std::cout << "Init, size: " << n << std::endl;
    std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
    std::cout << "Init, SNR: " << SNR << std::endl;

    //bool read_r, bool read_ra, bool read_xy
    cils::cils<scalar, index, true, n> cils(k, SNR);
    cils.init(is_qr);
    printf("init_res: %.5f, sigma: %.5f\n", cils.init_res, cils.sigma);
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
                z_B[i] = cils.x_R[i];
            }
        else if (init == 1)
            for (index i = 0; i < n; i++) {
                z_B[i] = std::pow(2, k) / 2;
            }

        auto reT = cils.cils_block_search_serial(&z_B, &d_s);
        auto res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &reT.x);
        auto ber = cils::find_bit_error_rate<scalar, index, n>(&reT.x, &cils.x_t, false);

        printf("Method: ILS_SER, Block size: %d, Res: %.5f, Brr: %.5f, Time: %.5fs\n", size, res, ber,
               reT.run_time);
        res = INFINITY;
        ber = INFINITY;
        for (index n_proc = min_proc; n_proc <= max_proc; n_proc += 12) {
            n_proc = n_proc == 96 ? 64 : n_proc;
            cout << d_s[d_s.size() - 1] << "," << n_proc << ",";
            std::cout.flush();
            for (index nswp = 0; nswp < 30; nswp++) {
                for (index t = 0; t < 3; t++) {
                    z_B.assign(n, 0);
                    if (init == -1)
                        for (index i = 0; i < n; i++) {
                            z_B[i] = cils.x_R[i];
                        }
                    else if (init == 1)
                        for (index i = 0; i < n; i++) {
                            z_B[i] = std::pow(2, k) / 2;
                        }

                    reT = cils.cils_block_search_omp(n_proc, nswp, -1, &z_B, &d_s);
                    scalar newres = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &reT.x);
                    scalar newbrr = cils::find_bit_error_rate<scalar, index, n>(&reT.x, &cils.x_t, cils.qam == 1);
                    res = newres < res ? newres : res;
                    ber = newbrr < ber ? newbrr : ber;
                }
                printf("res = %.5f, ber =%.5f,", res, ber);
                ber = INFINITY;
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
    cils::cils<double, int, true, n> cils(0.1);
    double end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %.5f seconds\n", end_time);

    start = omp_get_wtime();
    auto z_B = cils.cils_search(&cils.R_A, &cils.y_A);
    end_time = omp_get_wtime() - start;
    auto res = cils::find_residual<double, int, n>(&cils.R_A, &cils.y_A, z_B);
    printf("Thread: ILS, Sweep: 0, Res: %.5f, Time: %.5fs\n", res, end_time);

//    cils::scalarType<double, int> z_BS = {(double *) calloc(n, sizeof(double)), n};
//    start = omp_get_wtime();
//    z_BS = *cils.cils_babai_search_serial(&z_BS);
//    end_time = omp_get_wtime() - start;
//    res = cils::find_residual<double, int, n>(&cils.R_A, &cils.y_A, &z_BS);
//    printf("Thread: SR, Sweep: 0, Res: %.5f, Time: %.5fs\n", res, end_time);
}
