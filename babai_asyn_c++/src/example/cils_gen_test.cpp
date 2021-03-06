
#include "../source/cils.cpp"
#include "../source/cils_ils_search.cpp"
#include "../source/cils_block_search.cpp"
#include "../source/cils_babai_search.cpp"
#include "../source/cils_reduction.cpp"

template<typename scalar, typename index, index n>
void plot_run() {
    for (k = 1; k <= 3; k += 2)
        for (SNR = 15; SNR <= 35; SNR += 20) {
            printf("plot_res-------------------------------------------\n");
            std::cout << "Init, size: " << n << std::endl;
            std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
            vector<index> z_B(n, 0);

            scalar ser_qrd = 0, run_time;
            vector<scalar> bab_res(3, 0), bab_tim(3, 0), bab_ber(3, 0);
            vector<scalar> ser_res(3, 0), ser_tim(3, 0), ser_ber(3, 0);
            scalar omp_res[3][50] = {}, omp_ber[3][50] = {}, omp_tim[3][50] = {}, omp_itr[3][50] = {}, omp_qrd[3][50] = {}, omp_err[3][50] = {};
            cils::returnType<scalar, index> reT, qr_reT, qr_reT_omp;

            for (index i = 0; i < max_iter; i++) {
                run_time = omp_get_wtime();
                if (i % 20 == 0 && i != 0) {
                    for (index init = -1; init <= 1; init++) {
                        printf("++++++++++++++++++++++++++++++++++++++\n");
                        std::cout << "Block, size: " << block_size << std::endl;
                        std::cout << "Init, value: " << init << std::endl;
                        printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5f, Total Time: %.5fs\n",
                               bab_res[init + 1] / max_iter, bab_ber[init + 1] / max_iter, bab_tim[init + 1],
                               ser_qrd / max_iter,
                               (ser_qrd + bab_tim[init + 1]) / max_iter);
                        printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5f, Total Time: %.5fs\n",
                               block_size, ser_res[init + 1] / max_iter, ser_ber[init + 1] / max_iter,
                               ser_tim[init + 1],
                               ser_qrd / max_iter, (ser_qrd + ser_tim[init + 1]) / max_iter);
                        index l = 0;
                        for (index n_proc = min_proc; n_proc <= max_proc + 2 * min_proc; n_proc += min_proc) {
                            printf("Method: ILS_OMP, n_proc: %d, Res :%.5f, BER: %.5f, num_iter: %.5f, Time: %.5fs, Avg Time: %.5fs, "
                                   "Speed up: %.3f, QR Error: %.3f, QR Time: %.5fs, QR SpeedUp: %.3f, Total Time: %.5fs, Total SpeedUp: %.3f\n",
                                   n_proc > max_proc ? max_proc : n_proc, omp_res[init + 1][l] / max_iter,
                                   omp_ber[init + 1][l] / max_iter,
                                   omp_itr[init + 1][l] / max_iter,
                                   omp_tim[init + 1][l], omp_tim[init + 1][l] / max_iter,
                                   ser_tim[init + 1] / omp_tim[init + 1][l],
                                   omp_err[0][l] / max_iter, omp_qrd[0][l] / max_iter,
                                   ser_qrd / omp_qrd[0][l],
                                   (omp_qrd[0][l] + omp_tim[init + 1][l]) / max_iter,
                                   (ser_qrd + ser_tim[init + 1]) / (omp_qrd[0][l] + omp_tim[init + 1][l])
                            );
                            l++;
                        }
                        printf("++++++++++++++++++++++++++++++++++++++\n");
                    }
                }
                cils::cils<scalar, index, n> cils(k, SNR);
                if (i == 0) {
                    cils.init(is_read);
                    cils.init_y();
                    cils.cils_qr_decomposition_omp(0, 1, max_proc);
                    init_guess<scalar, index, n>(0, &z_B, &cils.x_R);
                    cils.cils_block_search_omp(max_proc, num_trials, 0, &d_s, &z_B);
                    continue;
                }

                cils.init(is_read);
                qr_reT = cils.cils_qr_decomposition_serial(0, 1);
                ser_qrd += qr_reT.run_time;
                cils.init_y();
                cils.init_res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &cils.x_t);
                cils.cils_back_solve(&cils.x_R);

                for (index init = -1; init <= 1; init++) {
                    init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                    reT = cils.cils_babai_search_serial(&z_B);
                    bab_res[init + 1] += cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
                    bab_ber[init + 1] += cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
                    bab_tim[init + 1] += reT.run_time;

                    init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                    reT = cils.cils_block_search_serial(&d_s, &z_B);
                    ser_res[init + 1] += cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
                    ser_ber[init + 1] += cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
                    ser_tim[init + 1] += reT.run_time;

                    index l = 0;
                    for (index n_proc = min_proc; n_proc <= max_proc + 2 * min_proc; n_proc += min_proc) {
//                if (init == -1) {
//                    qr_reT_omp = cils.cils_qr_decomposition_omp(0, 1, n_proc > max_proc ? max_proc : n_proc);
//                    omp_qrd[init + 1][l] += qr_reT_omp.run_time;
//                    omp_err[init + 1][l] += qr_reT_omp.num_iter;
//                }
                        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                        reT = cils.cils_block_search_omp(n_proc > max_proc ? max_proc : n_proc, num_trials, init, &d_s,
                                                         &z_B);
                        omp_res[init + 1][l] += cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
                        omp_ber[init + 1][l] += cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
                        omp_tim[init + 1][l] += reT.run_time;
                        omp_itr[init + 1][l] += reT.num_iter;
                        l++;
                    }
                }
                run_time = omp_get_wtime() - run_time;
                printf("%d-Time: %.5fs, ", i, run_time);
                cout.flush();
            }
            max_iter--;
            for (index init = -1; init <= 1; init++) {
                printf("++++++++++++++++++++++++++++++++++++++\n");
                std::cout << "Block, size: " << block_size << std::endl;
                std::cout << "Init, value: " << init << std::endl;
                printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Avg Solve Time: %.5fs, qr_time: %.5f, Total Time: %.5fs\n",
                       bab_res[init + 1] / max_iter, bab_ber[init + 1] / max_iter, bab_tim[init + 1] / max_iter,
                       ser_qrd / max_iter,
                       (ser_qrd + bab_tim[init + 1]) / max_iter);
                printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Avg Solve Time: %.5fs, qr_time: %.5f, Total Time: %.5fs\n",
                       block_size, ser_res[init + 1] / max_iter, ser_ber[init + 1] / max_iter,
                       ser_tim[init + 1] / max_iter,
                       ser_qrd / max_iter, (ser_qrd + ser_tim[init + 1]) / max_iter);
                index l = 0;
                for (index n_proc = min_proc; n_proc <= max_proc + 2 * min_proc; n_proc += min_proc) {
                    printf("Method: ILS_OMP, n_proc: %d, Res :%.5f, BER: %.5f, num_iter: %.5f, Time: %.5fs, Avg Time: %.5fs, "
                           "Speed up: %.3f, QR Error: %.3f, QR Time: %.5fs, QR SpeedUp: %.3f, Total Time: %.5fs, Total SpeedUp: %.3f\n",
                           n_proc > max_proc ? max_proc : n_proc, omp_res[init + 1][l] / max_iter,
                           omp_ber[init + 1][l] / max_iter,
                           omp_itr[init + 1][l] / max_iter,
                           omp_tim[init + 1][l], omp_tim[init + 1][l] / max_iter,
                           ser_tim[init + 1] / omp_tim[init + 1][l],
                           omp_err[0][l] / max_iter, omp_qrd[0][l] / max_iter,
                           ser_qrd / omp_qrd[0][l],
                           (omp_qrd[0][l] + omp_tim[init + 1][l]) / max_iter,
                           (ser_qrd + ser_tim[init + 1]) / (omp_qrd[0][l] + omp_tim[init + 1][l])
                    );
                    l++;
                }
                printf("++++++++++++++++++++++++++++++++++++++\n");
            }
            printf("End of current TASK.\n");
            printf("-------------------------------------------\n");
        }
}

template<typename scalar, typename index, index n>
long test_ils_search() {
    std::cout << "Init, size: " << n << std::endl;

    cils::cils<scalar, index, n> cils(k, SNR);
    index init = 0;
    scalar error = 0;
    cils::returnType<scalar, index> reT, qr_reT = {nullptr, 0, 0}, qr_reT_omp = {nullptr, 0, 0};
    for (index i = 0; i < max_iter; i++) {

        cils.init(is_read);

        if (!is_read) {
            qr_reT = cils.cils_qr_decomposition_serial(0, 1);
            cils.init_y();
            cils.init_res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &cils.x_t);
            cils.cils_back_solve(&cils.x_R);
        }
        printf("init_res: %.5f, sigma: %.5f, qr_error: %.1f\n", cils.init_res, cils.sigma, error);
        cout.flush();

        vector<index> z_B(n, 0);
        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);

        reT = cils.cils_babai_search_serial(&z_B);
        auto res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
        auto ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
        printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5f Total Time: %.5fs\n",
               res, ber, reT.run_time, qr_reT.run_time, qr_reT.run_time + reT.run_time);
        scalar bab_tim_constrained = reT.run_time;

//        if(res < 100) continue;

        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
        reT = cils.cils_block_search_serial(&d_s, &z_B);
        res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
        ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
        printf("\nMethod: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs, qr_time: %.5f Total Time: %.5fs\n",
               block_size, res, ber, reT.run_time, qr_reT.run_time, qr_reT.run_time + reT.run_time);
        scalar ils_tim_constrained = reT.run_time;
        cout.flush();

        for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
//            qr_reT_omp = cils.cils_qr_decomposition_omp(0, 1, n_proc > max_proc ? max_proc : n_proc);
//            cils.init_y();
//            cils.init_res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, &cils.x_t);
//            init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
//            reT = cils.cils_babai_search_omp(n_proc > max_proc ? max_proc : n_proc, num_trials, &z_B);
//            res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
//            ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
//            printf("Method: BAB_OMP, n_proc: %d, Res: %.5f, BER: %.5f, Num_iter: %d, "
//                   "Solve Time: %.5fs, Solve SpeedUp: %.3f, "
//                   "QR Error: %d, QR Time: %.5fs, QR SpeedUp: %.3f, "
//                   "Total Time: %.5fs, Total SpeedUp: %.3f\n",
//                   n_proc, res, ber, reT.num_iter, reT.run_time, (bab_tim_constrained / reT.run_time),
//                   qr_reT_omp.num_iter, qr_reT_omp.run_time, qr_reT.run_time / qr_reT_omp.run_time,
//                   reT.run_time + qr_reT_omp.run_time,
//                   (bab_tim_constrained + qr_reT.run_time) / (qr_reT_omp.run_time + reT.run_time));

            init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
            reT = cils.cils_block_search_omp(n_proc > max_proc ? max_proc : n_proc, num_trials, init, &d_s, &z_B);
            res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
            ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
            printf("Method: ILS_OMP, n_proc: %d, Res: %.5f, BER: %.5f, Num_iter: %d, "
                   "Solve Time: %.5fs, Solve SpeedUp: %.3f, "
                   "QR Error: %d, QR Time: %.5fs, QR SpeedUp: %.3f, "
                   "Total Time: %.5fs, Total SpeedUp: %.3f\n",
                   n_proc, res, ber, reT.num_iter, reT.run_time, (ils_tim_constrained / reT.run_time),
                   qr_reT_omp.num_iter, qr_reT_omp.run_time, qr_reT.run_time / qr_reT_omp.run_time,
                   reT.run_time + qr_reT_omp.run_time,
                   (ils_tim_constrained + qr_reT.run_time) / (qr_reT_omp.run_time + reT.run_time)
            );
        }
    }

}

template<typename scalar, typename index, index n>
void plot_res() {
    for (k = 1; k <= 3; k += 2)
        for (SNR = 15; SNR <= 35; SNR += 20) {
            printf("plot_res-------------------------------------------\n");
            std::cout << "Init, size: " << n << std::endl;
            std::cout << "Init, QAM: " << std::pow(4, k) << std::endl;
            std::cout << "Init, SNR: " << SNR << std::endl;

            //bool read_r, bool read_ra, bool read_xy
            cils::cils<scalar, index, n> cils(k, SNR);
            cils.init(is_read);
            auto reT = cils.cils_qr_decomposition_omp(0, 1, max_proc);
            cils.init_y();
            printf("init_res: %.5f, sigma: %.5f, qr time: %.5fs\n", cils.init_res, cils.sigma, reT.run_time);

            vector<index> z_B(n, 0);
            reT = cils.cils_babai_search_serial(&z_B);
            scalar res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
            scalar ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
            printf("Method: BAB_SER, Res: %.5f, BER: %.5f, Solve Time: %.5fs\n",
                   res, ber, reT.run_time);

            z_B.assign(n, 0);
            reT = cils.cils_block_search_serial(&d_s, &z_B);
            res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
            ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
            printf("Method: ILS_SER, Block size: %d, Res: %.5f, BER: %.5f, Solve Time: %.5fs\n",
                   block_size, res, ber, reT.run_time);

            for (index init = -1; init <= 1; init++) {
                cout << "init," << init << "\n";
                init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                reT = cils.cils_block_search_serial(&d_s, &z_B);
                res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
                ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);

                printf("Method: ILS_SER, Block size: %d, Res: %.5f, Ber: %.5f, Time: %.5fs\n",
                       block_size, res, ber, reT.run_time);
                for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
                    cout << d_s[d_s.size() - 1] << "," << n_proc << ",";
                    std::cout.flush();
                    for (index nswp = 1; nswp < max_iter; nswp++) {
                        init_guess<scalar, index, n>(init, &z_B, &cils.x_R);
                        reT = cils.cils_block_search_omp(n_proc > max_proc ? max_proc : n_proc, nswp, init, &d_s, &z_B);
                        res = cils::find_residual<scalar, index, n>(cils.R_A, cils.y_A, reT.x);
                        ber = cils::find_bit_error_rate<scalar, index, n>(reT.x, &cils.x_t, k);
                        printf("diff=%d, res=%.5f, ber=%.5f, ",
                               reT.num_iter > (N / block_size) ? (N / block_size) : reT.num_iter, res, ber);
                    }
                    cout << endl;
                }
            }

            printf("End of current TASK.\n");
            printf("-------------------------------------------\n");
        }
}
