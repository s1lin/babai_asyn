#include "../source/cils.cpp"

#include "../source/cils_block_search.cpp"
#include "../source/cils_init_point.cpp"
#include "../source/cils_partition_deficient.cpp"

#include <ctime>

template<typename scalar, typename index, index m, index n>
void init_point_test(int size, int rank) {

    vector<scalar> x_q(n, 0), x_tmp(n, 0), x_ser(n, 0), x_omp(n, 0), x_mpi(n, 0);
    auto *v_norm_qr = (double *) calloc(1, sizeof(double));

    scalar v_norm;
    cils::returnType<scalar, index> reT3;
    cils::cils<scalar, index, m, n> cils(qam, 35);
    //auto a = (double *)malloc(N * sizeof(double));
    cils.qam = qam;
    cils.upper = pow(2, qam) - 1;
    cils.u.fill(cils.upper);
    cout.flush();
    cils.init(rank);

    if (rank == 0) {
        //        cils.init_ud();
        scalar r = helper::find_residual<scalar, index>(m, n, cils.A.data(), cils.x_t.data(), cils.y_a.data());
        printf("[ INIT COMPLETE, RES:%8.5f, RES:%8.5f]\n", cils.init_res, r);

        time_t t0 = time(nullptr);
        struct tm *lt = localtime(&t0);
        char time_str[20];
        sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
                lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
                lt->tm_hour, lt->tm_min, lt->tm_sec
        );
        printf("====================[ TEST | INIT_POINT | %s ]==================================\n", time_str);

        cils::returnType<scalar, index> reT, reT2;
        //----------------------INIT POINT (SERIAL)--------------------------------//
        reT = cils.cils_grad_proj(x_q, 1e4);
        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_q.data(), x_tmp.data());
        v_norm_qr[0] = v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(),
                                                                     cils.y_a.data());
        scalar ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
        helper::display_vector<scalar, index>(n, x_tmp.data(), "x_q");
        printf("INI: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, v_norm, reT.run_time);

        //----------------------INIT POINT (OMP)--------------------------------//
        //STEP 1: init point by QRP
        reT = cils.cils_grad_proj_omp(x_omp, 1e4, 10);
        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_omp.data(), x_tmp.data());
        v_norm_qr[0] = v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(),
                                                                     cils.y_a.data());
        ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
        helper::display_vector<scalar, index>(n, x_tmp.data(), "x_q");
        printf("INI: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, v_norm, reT.run_time);
    }

}

template<typename scalar, typename index, index m, index n>
void block_optimal_test(int size, int rank) {
    vector<scalar> x_q(n, 0), x_tmp(n, 0), x_ser(n, 0), x_omp(n, 0), x_mpi(n, 0);
    auto *v_norm_qr = (double *) calloc(1, sizeof(double));

    scalar v_norm;
    cils::returnType<scalar, index> reT3;
    cils::cils<scalar, index, m, n> cils(qam, 35);
    //auto a = (double *)malloc(N * sizeof(double));
    cils.qam = qam;
    cils.upper = pow(2, qam) - 1;
    cils.u.fill(cils.upper);
    cout.flush();
    cils.init(rank);

    if (rank == 0) {
//        cils.init_ud();
        scalar r = helper::find_residual<scalar, index>(m, n, cils.A.data(), cils.x_t.data(), cils.y_a.data());
        printf("[ INIT COMPLETE, RES:%8.5f, RES:%8.5f]\n", cils.init_res, r);

        time_t t0 = time(nullptr);
        struct tm *lt = localtime(&t0);
        char time_str[20];
        sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
                lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
                lt->tm_hour, lt->tm_min, lt->tm_sec
        );
        printf("====================[ TEST | SIC_OPT | %s ]==================================\n", time_str);

        cils::returnType<scalar, index> reT, reT2;
        //----------------------INIT POINT (OPTIONAL)--------------------------------//
        //STEP 1: init point by QRP
        reT = cils.cils_grad_proj(x_q, 100);
        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_q.data(), x_tmp.data());
        v_norm_qr[0] = v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(),
                                                                     cils.y_a.data());
        scalar ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
        helper::display_vector<scalar, index>(n, x_tmp.data(), "x_q");
        printf("INI: ber: %8.5f, v_norm: %8.4f, time: %8.4f\n", ber, v_norm, reT.run_time);


        //----------------------OPTIMAL--------------------------------//
        //STEP 2: Optimal SERIAL SCP:
//        x_ser.assign(x_q.begin(), x_q.end());
//        reT2 = cils.cils_scp_block_optimal_serial(x_ser, v_norm_qr[0], 0);
//        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_ser.data(), x_tmp.data());
//        v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(), cils.y_a.data());
//
//        //Result Validation:
//        helper::display_vector<scalar, index>(n, x_ser.data(), "x_z");
//        helper::display_vector<scalar, index>(n, cils.x_t.data(), "x_t");
//        helper::display_vector<scalar, index>(n, x_tmp.data(), "x_p");
//        ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
//        printf("SER_BLOCK1: ber: %8.5f, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
//               ber, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);


        //STEP 3: SUBOptimal SERIAL SCP:
        x_ser.assign(x_q.begin(), x_q.end());
        reT2 = cils.cils_scp_block_suboptimal_serial(x_ser, v_norm_qr[0], 0);
        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_ser.data(), x_tmp.data());
        v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(), cils.y_a.data());

        //Result Validation:
        helper::display_vector<scalar, index>(n, x_ser.data(), "x_z");
        helper::display_vector<scalar, index>(n, cils.x_t.data(), "x_t");
        helper::display_vector<scalar, index>(n, x_tmp.data(), "x_p");
        ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
        printf("SER_BLOCK2: ber: %8.5f, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
               ber, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);
        //


        //STEP 3: Optimal OMP SCP:
        x_omp.assign(x_q.begin(), x_q.end());
        reT2 = cils.cils_scp_block_optimal_omp(x_omp, v_norm_qr[0], 5, false);
        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_omp.data(), x_tmp.data());
        v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(), cils.y_a.data());

        //Result Validation:
        helper::display_vector<scalar, index>(n, x_omp.data(), "x_z");
        helper::display_vector<scalar, index>(n, cils.x_t.data(), "x_t");
        helper::display_vector<scalar, index>(n, x_tmp.data(), "x_p");
        ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
        printf("OMP_BLOCK1: ber: %8.5f, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
               ber, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);


        //STEP 3: Optimal OMP SCP:
        x_omp.assign(x_q.begin(), x_q.end());
        reT2 = cils.cils_scp_block_suboptimal_omp(x_omp, v_norm_qr[0], 5, false);
        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_omp.data(), x_tmp.data());
        v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(), cils.y_a.data());

        //Result Validation:
        helper::display_vector<scalar, index>(n, x_omp.data(), "x_z");
        helper::display_vector<scalar, index>(n, cils.x_t.data(), "x_t");
        helper::display_vector<scalar, index>(n, x_tmp.data(), "x_p");
        ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
        printf("OMP_BLOCK2: ber: %8.5f, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
               ber, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);


        //----------------------BABAI--------------------------------//
        //STEP 4: Babai Serial SCP:
        x_ser.assign(x_q.begin(), x_q.end());
        reT2 = cils.cils_scp_block_babai_serial(x_ser, v_norm_qr[0], 0);
        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_ser.data(), x_tmp.data());
        v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(), cils.y_a.data());

        //Result Validation:
        helper::display_vector<scalar, index>(n, x_ser.data(), "x_z");
        helper::display_vector<scalar, index>(n, cils.x_t.data(), "x_t");
        helper::display_vector<scalar, index>(n, x_tmp.data(), "x_p");
        ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
        printf("SER_BABAI: ber: %8.5f, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
               ber, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);
//
//
//        //STEP 5: Babai OMP SCP:
//        x_omp.assign(x_q.begin(), x_q.end());
//        reT2 = cils.cils_scp_block_babai_omp(x_omp, v_norm_qr[0], 5, false);
//        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_omp.data(), x_tmp.data());
//        v_norm = helper::find_residual<scalar, index>(m, n, cils.A.data(), x_tmp.data(), cils.y_a.data());
//
//        //Result Validation:
//        helper::display_vector<scalar, index>(n, x_omp.data(), "x_z");
//        helper::display_vector<scalar, index>(n, cils.x_t.data(), "x_t");
//        helper::display_vector<scalar, index>(n, x_tmp.data(), "x_p");
//        ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
//        printf("OMP_BABAI: ber: %8.5f, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f\n",
//               ber, reT2.x[0], reT2.x[1], reT2.x[2], v_norm, reT2.run_time);


    }

//    //STEP 2: MPI-Block SCP:
//    MPI_Bcast(&x_mpi[0], n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    MPI_Bcast(&v_norm_qr[0], 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    MPI_Bcast(&cils.y_a[0], m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    MPI_Bcast(&cils.H[0], m * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    MPI_Barrier(MPI_COMM_WORLD);
//    reT3 = {{0, 0, 0}, 0, 0};
//    reT3 = cils.cils_scp_block_optimal_mpi(x_mpi, v_norm_qr, size, rank);
//
//    if (rank == 0) {
////        v_norm = reT3.info;
//        x_tmp.assign(n, 0);
//        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), x_mpi.data(), x_tmp.data());
//
//        //Result Validation:
//        helper::display_vector<scalar, index>(n, x_mpi.data(), "x_z");
//        helper::display_vector<scalar, index>(n, cils.x_t.data(), "x_t");
//        helper::display_vector<scalar, index>(n, x_tmp.data(), "x_p");
//        scalar ber = helper::find_bit_error_rate<scalar, index>(n, x_tmp.data(), cils.x_t.data(), cils.qam);
//        printf("MPI: ber: %8.5f, stopping: %1.1f, %1.1f, %1.1f, v_norm: %8.4f, time: %8.4f, rank:%d\n",
//               ber, reT3.x[0], reT3.x[1], reT3.x[2], v_norm, reT3.run_time, rank);
//    }
//    MPI_Barrier(MPI_COMM_WORLD);
}

template<typename scalar, typename index, index m, index n>
long plot_run(int size, int rank) {
    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | ILS | %s ]==================================\n", time_str);
//    for (SNR = 35; SNR >= 5; SNR -= 20) {
    index d_s_size = d_s.size();
    scalar res[4][200][2] = {}, ber[4][200][2] = {}, tim[5][200][2] = {}, spu[5][200][2] = {};//, t_spu[4][200][2] = {};
    //scalar stm[3][200][2] = {};
    //scalar all_time[7][1000][3][2] = {}; //Method, iter, init, qam

    index count = 0;
    scalar run_time;
    cils::returnType<scalar, index> reT;
    auto *v_norm_qr = (double *) calloc(1, sizeof(double));

    scalar r, t, b, iter, t2, r2, b2, iter2, prev_t_block, prev_t_babai;
    scalar b_ser_block, b_ser_babai, t_ser_block, t_ser_babai, t_init_pt, t_omp;
    index l = 0; //count for qam.

    vector<scalar> z_omp(n, 0), z_ser(n, 0), z_ini(n, 0), z_tmp(n, 0);
    cils::cils<scalar, index, m, n> cils(qam, SNR);

    for (index i = 1; i <= max_iter; i++) {
//            for (index k = 1; k <= 3; k += 2) {
        for (index k = 3; k >= 1; k -= 2) {

            count = k == 1 ? 0 : 1;
            run_time = omp_get_wtime();
            cils.qam = k;
            cils.upper = pow(2, k) - 1;
            cils.u.fill(cils.upper);
            cils.init(0);
            r = helper::find_residual<scalar, index>(m, n, cils.A.data(), cils.x_t.data(), cils.y_a.data());
            printf("[ INIT COMPLETE, RES:%8.5f, RES:%8.5f]\n", cils.init_res, r);

            for (index init = -1; init <= 2; init++) {
                l = 0;
                printf("[ TRIAL PHASE]\n");
                /*
                 * ------------------------------------------------------
                 * 1. INIT POINT (OPTIONAL) : a. SIC, b.QRP, c.GRAD, d.0
                 * ------------------------------------------------------
                 */
                z_ini.assign(n, 0);
                if (init == -1) {
                    reT = cils.cils_qrp_serial(z_ini);
                    cout << "1a. Method: INIT_QRP, ";
                } else if (init == 0) {
                    reT = cils.cils_sic_serial(z_ini);
                    cout << "1b. Method: INIT_SIC, ";
                } else if (init == 1) {
                    reT = cils.cils_grad_proj(z_ini, search_iter);
                    cout << "1c. Method: INIT_GRD, ";
                } else {
                    cout << "1d. Method: INIT_VT0, ";
                    cils.H.fill(0);
                    helper::eye<scalar, index>(n, cils.P.data());
                    for (index i1 = 0; i1 < m * n; i1++) {
                        cils.H[i1] = cils.A[i1];
                    }
                    reT = {{}, 0, 0};
                }

                helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ini.data(), z_tmp.data());
                b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), k);
                r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y_a.data());
                v_norm_qr[0] = r;
                t_init_pt = reT.run_time;
                res[init + 1][l][count] += r;
                ber[init + 1][l][count] += b;
                tim[init + 1][l][count] += t_init_pt;
                l++;
                printf("AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs\n",
                       res[init + 1][l][count] / i, ber[init + 1][l][count] / i, tim[init + 1][l][count] / i,
                       r, b, t_init_pt);


                /*
                 * -----------------------------------------------------------------
                 * 2. Block Optimal Serial-SCP
                 * -----------------------------------------------------------------
                 */
                z_ser.assign(z_ini.begin(), z_ini.end());
                z_tmp.assign(n, 0);
                index itr = 0;
                do {
                    reT = cils.cils_scp_block_suboptimal_serial(z_ser, v_norm_qr[0], itr > 0);
                    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ser.data(), z_tmp.data());
                    b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), cils.qam);
                    z_ser.assign(z_ini.begin(), z_ini.end());
                    itr++;
                } while (b > 0.3 && itr < 5);
                r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y_a.data());
                b_ser_block = b;
                t_ser_block = reT.run_time + t_init_pt;
                res[init + 1][l][count] += r;
                ber[init + 1][l][count] += b;
                tim[init + 1][l][count] += t_ser_block;
                l++;
                printf("2a. Method: SUBOP_SER, AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, "
                       "RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs\n",
                       res[init + 1][l][count] / i, ber[init + 1][l][count] / i, tim[init + 1][l][count] / i,
                       r, b, t_ser_block);

                /*
                 * -----------------------------------------------------------------
                 * 2b. Block subOptimal Serial-SCP
                 * -----------------------------------------------------------------
                 */
                z_ser.assign(z_ini.begin(), z_ini.end());
                z_tmp.assign(n, 0);
                itr = 0;
                do {
                    reT = cils.cils_scp_block_optimal_serial(z_ser, v_norm_qr[0], itr > 0);
                    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ser.data(), z_tmp.data());
                    b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), cils.qam);
                    z_ser.assign(z_ini.begin(), z_ini.end());
                    itr++;
                } while (b > 0.3 && itr < 5);
                r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y_a.data());
                printf("2b. Method: BLOPT_SER, AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, "
                       "RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs\n",
                       res[init + 1][l][count] / i, ber[init + 1][l][count] / i, tim[init + 1][l][count] / i,
                       r, b, reT.run_time + t_init_pt);

                /*
                 * -----------------------------------------------------------------
                 * 3. Block Babai Serial-SCP
                 * -----------------------------------------------------------------
                 */
                z_ser.assign(z_ini.begin(), z_ini.end());
                z_tmp.assign(n, 0);
                itr = 0;
                do {
                    reT = cils.cils_scp_block_babai_serial(z_ser, v_norm_qr[0], itr > 0);
                    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ser.data(), z_tmp.data());
                    b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), cils.qam);
                    z_ser.assign(z_ini.begin(), z_ini.end());
                    itr++;
                } while (b > 0.3 && itr < 5);
                r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y_a.data());
                b_ser_babai = b;
                t_ser_babai = reT.run_time + t_init_pt;
                res[init + 1][l][count] += r;
                ber[init + 1][l][count] += b;
                tim[init + 1][l][count] += t_ser_babai;
                l++;
                printf("3. Method: BABAI_SER, AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, "
                       "RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs\n",
                       res[init + 1][l][count] / i, ber[init + 1][l][count] / i, tim[init + 1][l][count] / i,
                       r, b, t_ser_babai);


                prev_t_babai = prev_t_block = INFINITY;
                for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {

                    /*
                     * -----------------------------------------------------------------
                     * 4. Block Optimal Parallel-SCP
                     * -----------------------------------------------------------------
                     */
                    index _ll = 0;
                    t = r = b = 0;
                    t2 = r2 = b2 = INFINITY;
                    cils::program_def::chunk = 1;
                    while (true) {
                        z_omp.assign(z_ini.begin(), z_ini.end());
                        reT = cils.cils_scp_block_suboptimal_omp(z_omp, v_norm_qr[0], n_proc, _ll > 0);
                        z_tmp.assign(n, 0);
                        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_omp.data(), z_tmp.data());
                        t = reT.run_time;
                        b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), k);
                        r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(),
                                                                 cils.y_a.data());

                        t2 = min(t, t2);
                        r2 = min(r, r2);
                        b2 = min(b, b2);
                        _ll++;

                        if (prev_t_block > t && b - b_ser_block < 0.1) break; //
                        if (_ll == 5) {
                            r = r2;
                            b = b2;
                            t = t2;
                            break; //
                        }
                    }
                    t_omp = t + t_init_pt;
                    res[init + 1][l][count] += r;
                    ber[init + 1][l][count] += b;
                    tim[init + 1][l][count] += t_omp;
                    spu[init + 1][l][count] += t_ser_block / t_omp;

                    printf("4. Method: BLOCK_OMP, N_PROC: %2d, AVG RES: %8.5f, AVG BER: %8.5f, "
                           "AVG TIME: %8.5fs, RES: %8.5f, BER: %8.5f, SER TIME: %8.5f, OMP TIME: %8.5fs, "
                           "SPEEDUP:%7.3f, AVG SPEEDUP: %7.3f.\n",
                           n_proc, res[init + 1][l][count] / i, ber[init + 1][l][count] / i,
                           tim[init + 1][l][count] / i, r, b, t_ser_block, t_omp,
                           t_ser_block / t_omp, spu[init + 1][l][count] / i);

                    l++;
                    prev_t_block = t;


                    /* -----------------------------------------------------------------
                     * 5. Block Babai Parallel-SCP
                     * -----------------------------------------------------------------
                     */

                    _ll = 0;
                    t = r = b = 0;
                    t2 = r2 = b2 = INFINITY;
                    cils::program_def::chunk = 1;
                    while (true) {
                        z_omp.assign(z_ini.begin(), z_ini.end());
                        reT = cils.cils_scp_block_babai_omp(z_omp, v_norm_qr[0], n_proc, _ll > 0);
                        z_tmp.assign(n, 0);
                        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_omp.data(), z_tmp.data());
                        t = reT.run_time;
                        b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), k);
                        r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(),
                                                                 cils.y_a.data());

                        t2 = min(t, t2);
                        r2 = min(r, r2);
                        b2 = min(b, b2);
                        _ll++;

                        if (prev_t_babai > t && b - b_ser_babai < 0.1) break; //
                        if (_ll == 5) {
                            r = r2;
                            b = b2;
                            t = t2;
                            break; //
                        }
                    }
                    t_omp = t + t_init_pt;
                    res[init + 1][l][count] += r;
                    ber[init + 1][l][count] += b;
                    tim[init + 1][l][count] += t_omp;
                    spu[init + 1][l][count] += t_ser_babai / t_omp;

                    printf("5. Method: BABAI_OMP, N_PROC: %2d, AVG RES: %8.5f, AVG BER: %8.5f, "
                           "AVG TIME: %8.5fs, RES: %8.5f, BER: %8.5f, SER TIME: %8.5f, OMP TIME: %8.5fs, "
                           "SPEEDUP:%7.3f, AVG SPEEDUP: %7.3f.\n",
                           n_proc, res[init + 1][l][count] / i, ber[init + 1][l][count] / i,
                           tim[init + 1][l][count] / i, r, b, t_ser_babai, t_omp,
                           t_ser_babai / t_omp, spu[init + 1][l][count] / i);

                    l++;
                    prev_t_babai = t;
                }
            }
            run_time = omp_get_wtime() - run_time;
            printf("++++++++++++++++++++++++++++++++++++++\n Trial %d, Elapsed Time: %.5fs. \n"
                   "++++++++++++++++++++++++++++++++++++++\n", i, run_time);
            cout.flush();
        }
        printf("\n---------------------\nITER:%d\n---------------------\n", i);
        if (i % plot_itr == 0) {//i % 50 == 0 &&
            PyObject *pName, *pModule, *pFunc;

            Py_Initialize();
            if (_import_array() < 0)
                PyErr_Print();
            npy_intp dim[3] = {4, 200, 2};
            npy_intp di4[3] = {5, 200, 2};

            scalar proc_nums[l - 2] = {};
            index ll = 0;

            for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
                proc_nums[ll] = n_proc;
                ll++;
            }
            npy_intp dpc[1] = {ll};

            PyObject *pRes = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, res);
            PyObject *pBer = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, ber);
            PyObject *pTim = PyArray_SimpleNewFromData(3, di4, NPY_DOUBLE, tim);
            PyObject *pSpu = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, spu);
            PyObject *pPrc = PyArray_SimpleNewFromData(1, dpc, NPY_DOUBLE, proc_nums);

            if (pRes == nullptr) printf("[ ERROR] pRes has a problem.\n");
            if (pBer == nullptr) printf("[ ERROR] pBer has a problem.\n");
            if (pTim == nullptr) printf("[ ERROR] pTim has a problem.\n");
            if (pPrc == nullptr) printf("[ ERROR] pPrc has a problem.\n");
            if (pSpu == nullptr) printf("[ ERROR] pSpu has a problem.\n");

            PyObject *sys_path = PySys_GetObject("path");
            PyList_Append(sys_path, PyUnicode_FromString(
                    "/home/shilei/CLionProjects/babai_asyn/babai_asyn_c++/src/example"));
            pName = PyUnicode_FromString("plot_helper");
            pModule = PyImport_Import(pName);

            if (pModule != nullptr) {
                pFunc = PyObject_GetAttrString(pModule, "plot_runtime_ud");
                if (pFunc && PyCallable_Check(pFunc)) {
                    PyObject *pArgs = PyTuple_New(14);
                    if (PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", n)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", SNR)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", qam)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", l)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 4, Py_BuildValue("i", i)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 5, pRes) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 6, pBer) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 7, pTim) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 8, pPrc) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 9, pSpu) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 10, Py_BuildValue("i", max_proc)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 11, Py_BuildValue("i", min_proc)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 12, Py_BuildValue("i", is_constrained)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 13, Py_BuildValue("i", m)) != 0) {
                        return false;
                    }

                    PyObject *pValue = PyObject_CallObject(pFunc, pArgs);

                } else {
                    if (PyErr_Occurred())
                        PyErr_Print();
                    fprintf(stderr, "Cannot find function qr\n");
                }
            } else {
                PyErr_Print();
                fprintf(stderr, "Failed to load file\n");

            }
        }
    }

    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");
//    }
    return 0;
}

template<typename scalar, typename index, index m, index n>
long plot_run_grad_omp(int size, int rank) {
    time_t t0 = time(nullptr);
    struct tm *lt = localtime(&t0);
    char time_str[20];
    sprintf(time_str, "%04d/%02d/%02d %02d:%02d:%02d",
            lt->tm_year + 1900, lt->tm_mon + 1, lt->tm_mday,
            lt->tm_hour, lt->tm_min, lt->tm_sec
    );
    printf("====================[ TEST | ILS | %s ]==================================\n", time_str);
    //    for (SNR = 35; SNR >= 5; SNR -= 20) {
    index d_s_size = d_s.size();
    scalar res[4][200][2] = {}, ber[4][200][2] = {}, tim[4][200][2] = {}, spu[4][200][2] = {};
    scalar tim_total[4][200][2] = {}, spu_total[4][200][2] = {};
    //scalar stm[3][200][2] = {};
    //scalar all_time[7][1000][3][2] = {}; //Method, iter, init, qam

    index count = 0;
    scalar run_time;
    cils::returnType<scalar, index> reT;
    auto *v_norm_qr = (double *) calloc(1, sizeof(double));

    scalar r, t, b, iter, t2, r2, b2, iter2, prev_t_block, prev_t_babai;
    scalar b_ser_block, b_ser_babai, t_ser_block, t_ser_babai, t_ser_block_total;
    scalar t_ser_babai_total, t_init_pt, t_init_pt_omp, t_omp, t_omp_total;
    index l = 0; //count for qam.

    vector<scalar> z_omp(n, 0), z_ser(n, 0), z_ini(n, 0), z_tmp(n, 0);
    cils::cils<scalar, index, m, n> cils(qam, SNR);

    for (index i = 1; i <= max_iter; i++) {
        //            for (index k = 1; k <= 3; k += 2) {
        for (index k = 3; k >= 1; k -= 2) {

            count = k == 1 ? 0 : 1;
            run_time = omp_get_wtime();
            cils.qam = k;
            cils.upper = pow(2, k) - 1;
            cils.u.fill(cils.upper);
            cils.init(0);
            r = helper::find_residual<scalar, index>(m, n, cils.A.data(), cils.x_t.data(), cils.y_a.data());
            printf("[ INIT COMPLETE, RES:%8.5f, RES:%8.5f]\n", cils.init_res, r);

            for (index init = -1; init <= 2; init++) {
                l = 0;
                printf("[ TRIAL PHASE]\n");
                /*
                 * ------------------------------------------------------
                 * 1. INIT POINT (OPTIONAL) : a. SIC, b.QRP, c.GRAD, d.0
                 * ------------------------------------------------------
                 */
                z_ini.assign(n, 0);
                if (init == -1) {
                    reT = cils.cils_qrp_serial(z_ini);
                    cout << "1a. Method: INIT_QRP, ";
                } else if (init == 0) {
                    reT = cils.cils_sic_serial(z_ini);
                    cout << "1b. Method: INIT_SIC, ";
                } else if (init == 1) {
                    reT = cils.cils_grad_proj(z_ini, search_iter);
                    cout << "1c. Method: INIT_GRD, ";
                } else {
                    cout << "1d. Method: INIT_VT0, ";
                    cils.H.fill(0);
                    helper::eye<scalar, index>(n, cils.P.data());
                    for (index i1 = 0; i1 < m * n; i1++) {
                        cils.H[i1] = cils.A[i1];
                    }
                    reT = {{}, 0, 0};
                }

                helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ini.data(), z_tmp.data());
                b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), k);
                r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y_a.data());
                v_norm_qr[0] = r;
                t_init_pt = reT.run_time;
                res[init + 1][l][count] += r;
                ber[init + 1][l][count] += b;
                tim[init + 1][l][count] += t_init_pt;
                l++;//0
                printf("AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs\n",
                       res[init + 1][l][count] / i, ber[init + 1][l][count] / i, tim[init + 1][l][count] / i,
                       r, b, t_init_pt);


                /*
                 * -----------------------------------------------------------------
                 * 2. Block Optimal Serial-SCP
                 * -----------------------------------------------------------------
                 */
                z_ser.assign(z_ini.begin(), z_ini.end());
                z_tmp.assign(n, 0);
                index itr = 0;
                do {
                    reT = cils.cils_scp_block_suboptimal_serial(z_ser, v_norm_qr[0], itr > 0);
                    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ser.data(), z_tmp.data());
                    b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), cils.qam);
                    z_ser.assign(z_ini.begin(), z_ini.end());
                    itr++;
                } while (b > 0.3 && itr < 5);
                r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y_a.data());
                b_ser_block = b;
                t_ser_block = reT.run_time;
                t_ser_block_total = reT.run_time + t_init_pt;

                res[init + 1][l][count] += r;
                ber[init + 1][l][count] += b;
                tim[init + 1][l][count] += t_ser_block;
                tim_total[init + 1][l][count] += t_ser_block_total;
                l++;//1
                printf("2a. Method: SUBOP_SER, AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, "
                       "RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs, TOTAL TIME: %.5fs\n",
                       res[init + 1][l][count] / i, ber[init + 1][l][count] / i, tim[init + 1][l][count] / i,
                       r, b, t_ser_block, t_ser_block_total);

                /*
                 * -----------------------------------------------------------------
                 * 3. Block Babai Serial-SCP
                 * -----------------------------------------------------------------
                 */
                z_ser.assign(z_ini.begin(), z_ini.end());
                z_tmp.assign(n, 0);
                itr = 0;
                do {
                    reT = cils.cils_scp_block_babai_serial(z_ser, v_norm_qr[0], itr > 0);
                    helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ser.data(), z_tmp.data());
                    b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), cils.qam);
                    z_ser.assign(z_ini.begin(), z_ini.end());
                    itr++;
                } while (b > 0.3 && itr < 5);
                r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y_a.data());
                b_ser_babai = b;
                t_ser_babai = reT.run_time;
                t_ser_babai_total = reT.run_time + t_init_pt;
                res[init + 1][l][count] += r;
                ber[init + 1][l][count] += b;
                tim[init + 1][l][count] += t_ser_babai;
                tim_total[init + 1][l][count] += t_ser_babai_total;
                l++;//2
                printf("3. Method: BABAI_SER, AVG RES: %.5f, AVG BER: %.5f, AVG TIME: %.5fs, "
                       "RES: %.5f, BER: %.5f, SOLVE TIME: %.5fs, TOTAL TIME: %.5fs\n",
                       res[init + 1][l][count] / i, ber[init + 1][l][count] / i, tim[init + 1][l][count] / i,
                       r, b, t_ser_babai, t_ser_babai_total);


                prev_t_babai = prev_t_block = INFINITY;
                for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
                    if (init == 1) {
                        /*
                         * -----------------------------------------------------------------
                         * 1d. INIT POINT GRAD
                         * -----------------------------------------------------------------
                         */

                        z_ini.assign(n, 0);
                        reT = cils.cils_grad_proj_omp(z_ini, search_iter, n_proc);
                        cout << "1d. Method: INIT_PGP, ";

                        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_ini.data(), z_tmp.data());
                        b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), k);
                        r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(), cils.y_a.data());
                        v_norm_qr[0] = r;
                        t_init_pt_omp = reT.run_time;
                        res[init + 1][l][count] += r;
                        ber[init + 1][l][count] += b;
                        tim[init + 1][l][count] += t_init_pt_omp;
                        spu[init + 1][l][count] += t_init_pt / t_init_pt_omp;
                        printf("N_PROC: %2d, AVG RES: %8.5f, AVG BER: %8.5f, "
                               "AVG TIME: %8.5fs, RES: %8.5f, BER: %8.5f, SER TIME: %8.5f, OMP TIME: %8.5fs, "
                               "SPEEDUP:%7.3f, AVG SPEEDUP: %7.3f.\n",
                               n_proc, res[init + 1][l][count] / i, ber[init + 1][l][count] / i,
                               tim[init + 1][l][count] / i, r, b, t_init_pt, t_init_pt_omp,
                               t_init_pt / t_init_pt_omp, spu[init + 1][l][count] / i);
                    } else{
                        t_init_pt_omp = t_init_pt;
                    }
                    l++;//3
                    /*
                     * -----------------------------------------------------------------
                     * 4. Block Optimal Parallel-SCP
                     * -----------------------------------------------------------------
                     */
                    index _ll = 0;
                    t = r = b = 0;
                    t2 = r2 = b2 = INFINITY;
                    cils::program_def::chunk = 1;
                    while (true) {
                        z_omp.assign(z_ini.begin(), z_ini.end());
                        reT = cils.cils_scp_block_suboptimal_omp(z_omp, v_norm_qr[0], n_proc, _ll > 0);
                        z_tmp.assign(n, 0);
                        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_omp.data(), z_tmp.data());
                        t = reT.run_time;
                        b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), k);
                        r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(),
                                                                 cils.y_a.data());

                        t2 = min(t, t2);
                        r2 = min(r, r2);
                        b2 = min(b, b2);
                        _ll++;

                        if (prev_t_block > t && b - b_ser_block < 0.1) break; //
                        if (_ll == 5) {
                            r = r2;
                            b = b2;
                            t = t2;
                            break; //
                        }
                    }
                    t_omp = t;
                    t_omp_total = t + t_init_pt_omp;
                    res[init + 1][l][count] += r;
                    ber[init + 1][l][count] += b;
                    tim[init + 1][l][count] += t_omp;
                    tim_total[init + 1][l][count] += t_omp_total;
                    spu[init + 1][l][count] += t_ser_block / t_omp;
                    spu_total[init + 1][l][count] += t_ser_block_total / t_omp_total;

                    printf("4. Method: BLOCK_OMP, N_PROC: %2d, AVG RES: %8.5f, AVG BER: %8.5f, "
                           "AVG TIME: %8.5fs, RES: %8.5f, BER: %8.5f, SER TIME: %8.5f, OMP TIME: %8.5fs, "
                           "SPEEDUP:%7.3f, AVG SPEEDUP: %7.3f, TOTAL TIME: %7.3f, TOTAL SPU: %7.3f.\n",
                           n_proc, res[init + 1][l][count] / i, ber[init + 1][l][count] / i,
                           tim[init + 1][l][count] / i, r, b, t_ser_block, t_omp,
                           t_ser_block / t_omp, spu[init + 1][l][count] / i, t_omp_total, t_ser_block_total / t_omp_total);

                    l++;//4
                    prev_t_block = t;


                    /* -----------------------------------------------------------------
                     * 5. Block Babai Parallel-SCP
                     * -----------------------------------------------------------------
                     */

                    _ll = 0;
                    t = r = b = 0;
                    t2 = r2 = b2 = INFINITY;
                    cils::program_def::chunk = 1;
                    while (true) {
                        z_omp.assign(z_ini.begin(), z_ini.end());
                        reT = cils.cils_scp_block_babai_omp(z_omp, v_norm_qr[0], n_proc, _ll > 0);
                        z_tmp.assign(n, 0);
                        helper::mtimes_Axy<scalar, index>(n, n, cils.P.data(), z_omp.data(), z_tmp.data());
                        t = reT.run_time;
                        b = helper::find_bit_error_rate<scalar, index>(n, z_tmp.data(), cils.x_t.data(), k);
                        r = helper::find_residual<scalar, index>(m, n, cils.A.data(), z_tmp.data(),
                                                                 cils.y_a.data());

                        t2 = min(t, t2);
                        r2 = min(r, r2);
                        b2 = min(b, b2);
                        _ll++;

                        if (prev_t_babai > t && b - b_ser_babai < 0.1) break; //
                        if (_ll == 5) {
                            r = r2;
                            b = b2;
                            t = t2;
                            break; //
                        }
                    }
                    t_omp = t;
                    t_omp_total = t + t_init_pt_omp;
                    res[init + 1][l][count] += r;
                    ber[init + 1][l][count] += b;
                    tim[init + 1][l][count] += t_omp;
                    spu[init + 1][l][count] += t_ser_babai / t_omp;
                    tim_total[init + 1][l][count] += t_omp_total;
                    spu_total[init + 1][l][count] += t_ser_babai_total / t_omp_total;

                    printf("5. Method: BABAI_OMP, N_PROC: %2d, AVG RES: %8.5f, AVG BER: %8.5f, "
                           "AVG TIME: %8.5fs, RES: %8.5f, BER: %8.5f, SER TIME: %8.5f, OMP TIME: %8.5fs, "
                           "SPEEDUP:%7.3f, AVG SPEEDUP: %7.3f, TOTAL TIME: %7.3f, TOTAL SPU: %7.3f.\n",
                           n_proc, res[init + 1][l][count] / i, ber[init + 1][l][count] / i,
                           tim[init + 1][l][count] / i, r, b, t_ser_babai, t_omp,
                           t_ser_babai / t_omp, spu[init + 1][l][count] / i, t_omp_total, t_ser_babai_total / t_omp_total);

                    l++;//5
                    prev_t_babai = t;
                }
            }
        }
        run_time = omp_get_wtime() - run_time;
        printf("++++++++++++++++++++++++++++++++++++++\n Trial %d, Elapsed Time: %.5fs. \n"
               "++++++++++++++++++++++++++++++++++++++\n", i, run_time);
        cout.flush();

        printf("\n---------------------\nITER:%d\n---------------------\n", i);
        if (i % plot_itr == 0) {//i % 50 == 0 &&
            PyObject *pName, *pModule, *pFunc;

            Py_Initialize();
            if (_import_array() < 0)
                PyErr_Print();
            npy_intp dim[3] = {4, 200, 2};
            /*
             * scalar res[1][200][2] = {}, ber[1][200][2] = {}, tim[1][200][2] = {}, spu[1][200][2] = {}, spu2[1][200][2] = {};
                scalar tim_total[1][200][2] = {}, spu_total[1][200][2] = {};
             */

            scalar proc_nums[l - 2] = {};
            index ll = 0;

            for (index n_proc = min_proc; n_proc <= max_proc; n_proc += min_proc) {
                proc_nums[ll] = n_proc;
                ll++;
            }
            npy_intp dpc[1] = {ll};

            PyObject *pRes = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, res);
            PyObject *pBer = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, ber);
            PyObject *pTim = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, tim);
            PyObject *pSpu = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, spu);
            PyObject *ptim_total = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, tim_total);
            PyObject *pspu_total = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, spu_total);
            PyObject *pPrc = PyArray_SimpleNewFromData(1, dpc, NPY_DOUBLE, proc_nums);

            if (pRes == nullptr) printf("[ ERROR] pRes has a problem.\n");
            if (pBer == nullptr) printf("[ ERROR] pBer has a problem.\n");
            if (pTim == nullptr) printf("[ ERROR] pTim has a problem.\n");
            if (pSpu == nullptr) printf("[ ERROR] pSpu has a problem.\n");
            if (ptim_total == nullptr) printf("[ ERROR] ptim_total has a problem.\n");
            if (pspu_total == nullptr) printf("[ ERROR] pspu_total has a problem.\n");
            if (pPrc == nullptr) printf("[ ERROR] pPrc has a problem.\n");

            PyObject *sys_path = PySys_GetObject("path");
            PyList_Append(sys_path, PyUnicode_FromString(
                    "/home/shilei/CLionProjects/babai_asyn/babai_asyn_c++/src/example"));
            pName = PyUnicode_FromString("plot_helper");
            pModule = PyImport_Import(pName);

            if (pModule != nullptr) {
                pFunc = PyObject_GetAttrString(pModule, "plot_runtime_ud_grad");
                if (pFunc && PyCallable_Check(pFunc)) {
                    PyObject *pArgs = PyTuple_New(16);
                    if (PyTuple_SetItem(pArgs, 0, Py_BuildValue("i", n)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 1, Py_BuildValue("i", SNR)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 2, Py_BuildValue("i", qam)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 3, Py_BuildValue("i", l)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 4, Py_BuildValue("i", i)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 5, pRes) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 6, pBer) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 7, pTim) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 8, pPrc) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 9, pSpu) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 10, ptim_total) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 11, pspu_total) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 12, Py_BuildValue("i", max_proc)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 13, Py_BuildValue("i", min_proc)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 14, Py_BuildValue("i", is_constrained)) != 0) {
                        return false;
                    }
                    if (PyTuple_SetItem(pArgs, 15, Py_BuildValue("i", m)) != 0) {
                        return false;
                    }

                    PyObject *pValue = PyObject_CallObject(pFunc, pArgs);

                } else {
                    if (PyErr_Occurred())
                        PyErr_Print();
                    fprintf(stderr, "Cannot find function qr\n");
                }
            } else {
                PyErr_Print();
                fprintf(stderr, "Failed to load file\n");

            }
        }
    }

    printf("End of current TASK.\n");
    printf("-------------------------------------------\n");
    //    }
    return 0;
}