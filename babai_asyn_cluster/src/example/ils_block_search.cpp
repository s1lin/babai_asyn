#include "../source/SILS.cpp"

template<typename scalar, typename index, index n>
void ils_block_search() {

    std::cout << "Init, size: " << n << std::endl;

    scalar start = omp_get_wtime();
    sils::SILS<scalar, index, true, false, n> bsa(0.1);
    scalar end_time = omp_get_wtime() - start;
    printf("Finish Init, time: %f seconds\n", end_time);


    sils::scalarType<scalar, index> z_B{(scalar *) calloc(n, sizeof(scalar)), n};

    for (index size = 8; size <= 32; size *= 2) {
        //Initialize the block vector
        vector<index> d(n / size, size);
        sils::scalarType<index, index> d_s{d.data(), (index) d.size()};
        for (index i = d_s.size - 2; i >= 0; i--) {
            d_s.x[i] += d_s.x[i + 1];
        }

        for (index i = 0; i < 10; i++) {

            free(z_B.x);
            z_B.x = (scalar *) calloc(n, sizeof(scalar));
            start = omp_get_wtime();
            z_B = *bsa.sils_block_search_serial(&bsa.R_A, &bsa.y_A, &z_B, &d_s);
            end_time = omp_get_wtime() - start;
            auto res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &z_B);
            printf("Method: ILS_SER, Block size: %d, Res: %.5f, Run time: %fs\n", size, res, end_time);

            for (index n_proc = 12; n_proc >= 3; n_proc /= 2) {
                free(z_B.x);
                z_B.x = (scalar *) calloc(n, sizeof(scalar));
                index iter = 10;
                start = omp_get_wtime();
                z_B = *bsa.sils_block_search_omp(n_proc, iter, &bsa.R_A, &bsa.y_A, &z_B, &d_s);
                end_time = omp_get_wtime() - start;
//                res = sils::norm<scalar, index, n>(&bsa.x_tA, &z_B);
                res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &z_B);
                printf("Method: ILS_OMP, Num of Threads: %d, Block size: %d, Iter: %d, Res: %.5f, Run time: %fs\n",
                       n_proc,
                       size, iter, res, end_time);
            }

            free(z_B.x);
            z_B.x = (scalar *) calloc(n, sizeof(scalar));
            sils::scalarType<scalar, index> z_BS = {(scalar *) calloc(n, sizeof(scalar)), n};
            start = omp_get_wtime();
            z_B = *bsa.sils_babai_search_serial(&z_BS);
            end_time = omp_get_wtime() - start;
            res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &z_B);
            printf("Method: BBI_SER, Res: %.5f, Run time: %fs\n", res, end_time);
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

        for (index size = 8; size <= 32; size *= 2) {
            std::cout << "Init, size: " << size << std::endl;
            vector<index> d(n / size, size);
            sils::scalarType<index, index> d_s{d.data(), (index) d.size()};
            for (index i = d_s.size - 2; i >= 0; i--) {
                d_s.x[i] += d_s.x[i + 1];
            }

            vector<scalar> res(50, 0), tim(50, 0), itr(50, 0);
            scalar omp_res = 0, omp_time = 0, num_iter = 0;
            scalar ser_res = 0, ser_time = 0;

            std::cout << "Babai Serial:" << std::endl;
            for (index i = 0; i < 10; i++) {
                free(z_B.x);
                z_B.x = (scalar *) calloc(n, sizeof(scalar));
                start = omp_get_wtime();
                z_B = *bsa.sils_babai_search_serial(&z_B);
                ser_time = omp_get_wtime() - start;
                ser_res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &z_B);
                res[0] += ser_res;
                tim[0] += ser_time;
            }
            printf("Thread: SR, Res: %.5f, Run time: %fs\n", res[0] / 10, tim[0] / 10);

            file << size << "," << res[0] / 10 << "," << tim[0] / 10 << ",\n";

            std::cout << "Block Serial:" << std::endl;
            for (index i = 0; i < 10; i++) {
                free(z_B.x);
                z_B.x = (scalar *) calloc(n, sizeof(scalar));
                start = omp_get_wtime();
                z_B = *bsa.sils_block_search_serial(&bsa.R_A, &bsa.y_A, &z_B, &d_s);
                ser_time = omp_get_wtime() - start;
                ser_res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &z_B);
                res[1] += ser_res;
                tim[1] += ser_time;
            }
            printf("Method: ILS_SER, Block size: %d, Res: %.5f, Run time: %fs\n", size, res[1] / 10, tim[1] / 10);
            file << size << "," << res[1] / 10 << "," << tim[1] / 10 << ",\n";

            std::cout << "Block OMP:" << std::endl;
            index l = 2;
            for (index i = 0; i < 10; i++) {
                for (index n_proc = 40; n_proc >= 2; n_proc /= 2) {
                    free(z_B.x);
                    z_B.x = (scalar *) calloc(n, sizeof(scalar));
                    index iter = size == 32 ? 8 : 11;
                    start = omp_get_wtime();
                    z_B = *bsa.sils_block_search_omp(n_proc, iter, &bsa.R_A, &bsa.y_A, &z_B, &d_s);
                    omp_time = omp_get_wtime() - start;
                    omp_res = sils::find_residual<scalar, index, n>(&bsa.R_A, &bsa.y_A, &z_B);
                    res[l] += omp_res;
                    tim[l] += omp_time;
                    itr[l] += 10;
                    l++;
                }
                l = 2;
            }
            l = 2;
            for (index n_proc = 80; n_proc >= 2; n_proc /= 2) {
                file << size << "," << n_proc << ","
                     << res[l] / 10 << ","
                     << tim[l] / 10 << ","
                     << itr[l] / 10 << ",\n";

                printf("Block Size: %d, n_proc: %d, res :%f, num_iter: %f, Average time: %fs\n",
                       size, n_proc,
                       res[l] / 10,
                       itr[l] / 10,
                       tim[l] / 10);
                l++;
            }
            file << "Next,\n";

        }
        free(z_B.x);
    }
    file.close();
    printf("-------------------------------------------\n");
}