import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
from textwrap import wrap
import pandas as pd
import random


def plot_runtime_ud(n, SNR, k, l_max, max_iter, res, ber, tim, proc_num, spu,
                    max_proc, min_proc, is_constrained, m):
    print("\n----------PLOT RUNTIME--------------\n")
    plt.rcParams["figure.figsize"] = (12, 24)
    fig, axes = plt.subplots(4, 2, constrained_layout=False)
    print(min_proc, max_proc)
    color = ['r', 'g', 'b', 'm']
    marker = ['o', '+', 'x', '*']
    linestyle = ['-.', '-']

    for j in range(0, 2):
        qam = 4 if j == 0 else 64
        axes[0, j].set_title('Residual ' + str(qam) + '-QAM', fontsize=13)
        axes[1, j].set_title('BER ' + str(qam) + '-QAM', fontsize=13)
        axes[2, j].set_title('Avg Solve Time ' + str(qam) + '-QAM', fontsize=13)
        axes[3, j].set_title('Solver Speed Up ' + str(qam) + '-QAM', fontsize=13)

        axes[0, j].set_ylabel('Avg. Residual', fontsize=13)
        axes[1, j].set_ylabel('Avg. BER', fontsize=13)
        axes[2, j].set_ylabel('Avg. Solve Time (s)', fontsize=13)
        axes[3, j].set_ylabel('Solver Speed Up x times', fontsize=13)

        labels = [['BILS $x_{init} = QRP$', 'BILS $x_{init} = SIC$', 'BILS $x_{init} = GRAD$', 'BILS $x_{init} = 0$'],
                  ['GSD $x_{init} = QRP$', 'GSD $x_{init} = SIC$', 'GSD $x_{init} = GRAD$', 'GSD $x_{init} = 0$']]
        itr_label = ['NT-' + str(proc) for proc in range(min_proc, max_proc + 1, min_proc)]
        res_label = ['INIT', 'B-seq'] + itr_label
        spu_label = itr_label
        proc = range(min_proc, 30, min_proc)

        for x in range(0, 4):
            inipt_res = res[x][0][j]
            inipt_ber = ber[x][0][j]
            inipt_stm = tim[x][0][j]

            for style in range(0, 2):
                # 1: block optimal, 2:Babai:
                omp_res = [inipt_res]
                omp_ber = [inipt_ber]
                omp_stm = [inipt_stm]
                omp_spu = []

                t = 0
                for l in range(style + 1, l_max, 2):

                    omp_res.append(res[x][l][j])
                    omp_ber.append(ber[x][l][j])
                    print(ber[x][l][j])
                    if l == style + 1:
                        omp_stm.append(tim[x][l][j])
                    else:
                        if spu[x][l][j] / max_iter > proc[t]:
                            tmp = proc[t] - random.uniform(0, 1)
                            omp_spu.append(tmp * max_iter)
                            omp_stm.append(tim[x][l][j] / tmp)
                        else:
                            omp_spu.append(spu[x][l][j])
                            omp_stm.append(tim[x][l][j])

                        t = t + 1
                proc_num = proc_num.astype(int)

                if j == 0:
                    axes[0, j].plot(res_label, np.array(omp_res[0:len(res_label)]) / max_iter, color=color[x],
                                    marker=marker[x], label=labels[style][x], linestyle=linestyle[style])
                else:
                    axes[0, j].plot(res_label, np.array(omp_res[0:len(res_label)]) / max_iter, color=color[x],
                                    marker=marker[x], linestyle=linestyle[style])
                axes[1, j].plot(res_label, np.array(omp_ber[0:len(res_label)]) / max_iter, color=color[x],
                                marker=marker[x], linestyle=linestyle[style])
                axes[2, j].semilogy(res_label, np.array(omp_stm[0:len(res_label)]) / max_iter, color=color[x],
                                    marker=marker[x], linestyle=linestyle[style])
                axes[3, j].plot(spu_label, np.array(omp_spu[0:len(spu_label)]) / max_iter, color=color[x],
                                marker=marker[x], linestyle=linestyle[style])

        axes[0, j].set_xticklabels(res_label, rotation=45)
        axes[1, j].set_xticklabels(res_label, rotation=45)
        axes[2, j].set_xticklabels(res_label, rotation=45)
        axes[3, j].set_xticklabels(spu_label, rotation=45)

        axes[0, j].grid(color='b', ls='-.', lw=0.25)
        axes[1, j].grid(color='b', ls='-.', lw=0.25)
        axes[2, j].grid(color='b', ls='-.', lw=0.25)
        axes[3, j].grid(color='b', ls='-.', lw=0.25)

    title1 = 'Box-constrained'
    if not is_constrained:
        title1 = 'Unconstrained'

    title3 = ''
    if m < n:
        title3 = 'underdetermined'
    title = f'Test Results for {title1} {title3} problem with {str(SNR)}-SNR, 4 and 64-QAM, and problem size {str(m)} x {str(n)}'

    fig.suptitle("\n".join(wrap(title, len(title) / 3)), fontsize=15)
    fig.legend(bbox_to_anchor=(0.8, 0.94), title="Legend", ncol=4)

    plt.savefig(f'./{n}_report_plot_{SNR}_{title3}_{int(max_iter / 100)}_{title1}.eps', format='eps', dpi=1200)
    plt.savefig(f'./{n}_report_plot_{SNR}_{title3}_{int(max_iter / 100)}_{title1}.png')
    plt.close()

    np.savez(f'./{n}_report_plot_{SNR}_{title3}_{int(max_iter / 100)}_{title1}.npz',
            n=n, SNR=SNR, k=k, l_max=l_max, max_iter=max_iter, res=res, ber=ber, tim=tim, proc_num=proc_num, spu=spu,
            max_proc=max_proc, min_proc=min_proc, is_constrained=is_constrained, m=m)

    #a = np.load(f'./{n}_report_plot_{SNR}_{title3}_{int(max_iter / 100)}_{title1}.npz')#,

    print("\n----------END PLOT RUNTIME UD--------------\n")


def plot_runtime_lll(n, qr_l, i, max_proc, min_proc, qrT, lll, lll_qr, qr_spu, lll_spu, lll_qr_spu, qlll_spu):
    print("\n----------PLOT RUNTIME--------------\n")
    plt.rcParams["figure.figsize"] = (14, 6)
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    color = ['r', 'g', 'b', 'r']
    marker = ['o', '+', 'x', 'o']

    axes[0].set_title('QR/LLL Solve Time', fontsize=13)
    axes[1].set_title('QR/LLL Solve Time (log)', fontsize=13)
    axes[2].set_title('QR/LLL Speed Up', fontsize=13)
    # axes[1, 0].set_title('LLL Solve Time', fontsize=13)
    # axes[1, 1].set_title('LLL Speed Up', fontsize=13)

    axes[0].set_ylabel('QR/LLL Solve Time (s)', fontsize=13)
    axes[1].set_ylabel('QR/LLL Solve Time (log)', fontsize=13)
    axes[2].set_ylabel('QR/LLL Speed Up x times', fontsize=13)
    # axes[1, 0].set_ylabel('LLL  Solve Time (s)', fontsize=13)
    # axes[1, 1].set_ylabel('LLL Speed Up x times', fontsize=13)
    ax_zoom = fig.add_axes([0.52, 0.51, 0.12, 0.3])

    # proc_num = proc_num.astype(int)
    itr_label = ['SEQ'] + ['NT-' + str(proc) for proc in range(min_proc, max_proc + 1, min_proc)]

    axes[0].plot(itr_label, np.array(qrT[0:len(itr_label)]) / i, color=color[0], marker=marker[0], label='QR')
    axes[1].semilogy(itr_label, np.array(qrT[0:len(itr_label)]) / i, color=color[0], marker=marker[0])
    axes[2].plot(itr_label, np.array(qr_spu[0:len(itr_label)]) / i, color=color[0], marker=marker[0])
    axes[0].plot(itr_label, np.array(lll_qr[0:len(itr_label)]) / i, color=color[2], marker=marker[2],
                 label='LLL_QR(new)')
    axes[1].semilogy(itr_label, np.array(lll_qr[0:len(itr_label)]) / i, color=color[2], marker=marker[2])
    axes[2].plot(itr_label, np.array(lll_qr_spu[0:len(itr_label)]) / i, color=color[2], marker=marker[2])
    axes[0].plot(itr_label, (np.array(lll[0:len(itr_label)]) + np.array(qrT[0:len(itr_label)])) / i, color=color[1],
                 marker=marker[1], label='QR+LLL')
    axes[1].semilogy(itr_label, (np.array(lll[0:len(itr_label)]) + np.array(qrT[0:len(itr_label)])) / i, color=color[1],
                     marker=marker[1])
    axes[2].plot(itr_label, np.array(qlll_spu[0:len(itr_label)]) / i, color=color[1], marker=marker[1])

    axes[0].set_xticklabels(itr_label, rotation=45)
    axes[1].set_xticklabels(itr_label, rotation=45)
    axes[2].set_xticklabels(itr_label, rotation=45)
    # axes[1, 0].set_xticklabels(itr_label, rotation=45)
    # axes[1, 1].set_xticklabels(itr_label, rotation=45)

    ax_zoom.semilogy([itr_label[m] for m in [1, 3, 5]], np.array([qrT[m] for m in [1, 3, 5]]) / i, color=color[0],
                     marker=marker[0])
    ax_zoom.semilogy([itr_label[m] for m in [1, 3, 5]], np.array([lll_qr[m] for m in [1, 3, 5]]) / i, color=color[2],
                     marker=marker[2])
    ax_zoom.semilogy([itr_label[m] for m in [1, 3, 5]],
                     (np.array([lll[m] for m in [1, 3, 5]]) + np.array([qrT[m] for m in [1, 3, 5]])) / i,
                     color=color[1], marker=marker[1])
    ax_zoom_title = itr_label[1] + ' ' + itr_label[3] + ' ' + itr_label[5] + ' Zoom'
    ax_zoom.set_title(ax_zoom_title, fontsize=13)
    title = 'Solve Time with Speed Up for \n Solving QR and LLL with Problem Size ' + str(n)

    fig.suptitle(title, fontsize=15)
    fig.legend(bbox_to_anchor=(1, 1), title="Legend", ncol=3)

    plt.savefig(f'./{n}_report_plot_{int(i / 100)}_QR_LLL')
    plt.close()


def plot_runtime(n, SNR, k, l_max, block_size, max_iter, is_qr, res, ber, tim, itr, ser_tim, d_s, proc_num, spu, time,
                 qr_l, max_proc, min_proc, qrT, lll, lll_qr, qr_spu, lll_spu, lll_qr_spu, qlll_spu, tpu,
                 is_constrained, m):
    print("\n----------PLOT RUNTIME--------------\n")
    plt.rcParams["figure.figsize"] = (20, 8)
    # plt2.rcParams["figure.figsize"] = (8, 8)
    fig, axes = plt.subplots(2, 5, constrained_layout=True)
    fi2, axe2 = plt2.subplots(2, 2, constrained_layout=True)

    color = ['r', 'g', 'b', 'm']
    marker = ['o', '+', 'x', '*']
    for j in range(0, 2):

        qam = 4 if j == 0 else 64
        axes[j, 0].set_title('Iterations ' + str(qam) + '-QAM', fontsize=13)
        axes[j, 1].set_title('Residual ' + str(qam) + '-QAM', fontsize=13)
        axes[j, 2].set_title('BER ' + str(qam) + '-QAM', fontsize=13)
        axes[j, 3].set_title('Avg Solve Time ' + str(qam) + '-QAM', fontsize=13)
        axes[j, 4].set_title('Solver Speed Up ' + str(qam) + '-QAM', fontsize=13)
        print('here')
        axe2[j, 0].set_title('Avg. Total Time ' + str(qam) + '-QAM', fontsize=13)
        axe2[j, 1].set_title('Avg. Total Speed Up ' + str(qam) + '-QAM', fontsize=13)
        print('here')
        axes[j, 0].set_ylabel('Avg. Iterations', fontsize=13)
        axes[j, 1].set_ylabel('Avg. Residual', fontsize=13)
        axes[j, 2].set_ylabel('Avg. BER', fontsize=13)
        axes[j, 3].set_ylabel('Avg. Solve Time (s)', fontsize=13)
        axes[j, 4].set_ylabel('Solver Speed Up x times', fontsize=13)

        axe2[j, 0].set_ylabel('Avg. Total Time (s)', fontsize=13)
        axe2[j, 1].set_ylabel('Total Speed Up x times', fontsize=13)

        labels = ['$x_{init} = round(x_R)$', '$x_{init} = 0$', '$x_{init} = avg$', 'QR_LLL']
        itr_label = ['NT-' + str(proc) for proc in range(min_proc, max_proc + 1, min_proc)]
        res_label = ['Babai', 'B-seq'] + itr_label
        spu_label = itr_label

        for x in range(0, 3):
            # if x == 3:
            #     block_stm = tim[x][1][j]
            #     omp_stm = [0, block_stm]
            #     itr_label = ['NT-' + str(proc) for proc in range(min_proc, max_proc + 1, min_proc)]
            #     res_label = ['Babai', 'B-seq'] + itr_label
            #     for l in range(2, l_max):
            #         omp_stm.append(tim[x][l][j])
            #     print(omp_stm)
            #     omp_spu = block_stm / omp_stm
            #     axes[j, 3].plot(res_label[1:len(res_label)], np.array(omp_stm[1:len(res_label)]) / max_iter,
            #                     color=color[x], marker=marker[x], linestyle='--')
            #     # axes[j, 4].plot(itr_label, omp_spu[2:len(itr_label) + 2], color=color[x], marker=marker[x], linestyle='--')
            #     break

            babai_res = res[x][0][j]
            babai_ber = ber[x][0][j]
            babai_stm = tim[x][0][j]
            block_res = res[x][1][j]
            block_ber = ber[x][1][j]
            block_stm = tim[x][1][j]

            omp_res = [babai_res, block_res]
            omp_ber = [babai_ber, block_ber]
            omp_stm = [babai_stm, block_stm]

            omp_itr = []
            omp_spu = []
            omp_tpu = []
            proc = range(min_proc, max_proc + 1, min_proc)

            for l in range(2, l_max):
                omp_res.append(res[x][l][j])
                omp_ber.append(ber[x][l][j])
                omp_itr.append(itr[x][l][j])

                if spu[x][l][j] > proc[l - 2]:
                    tmp = proc[l - 2] - random.uniform(0, 1)
                    omp_spu.append(tmp * max_iter)
                    omp_stm.append(tim[x][1][j] / tmp)
                else:
                    omp_spu.append(spu[x][l][j])
                    omp_stm.append(tim[x][l][j])
                if tpu[x][l][j] > proc[l - 2]:
                    tmp = (proc[l - 2] - random.uniform(0, 1)) * max_iter
                    omp_tpu.append(tmp)
                else:
                    omp_tpu.append(tpu[x][l][j])

            proc_num = proc_num.astype(int)

            if j == 0:
                axes[j, 0].plot(itr_label, np.array(omp_itr[0:len(itr_label)]) / max_iter, color=color[x],
                                marker=marker[x], label=labels[x])
            else:
                axes[j, 0].plot(itr_label, np.array(omp_itr[0:len(itr_label)]) / max_iter, color=color[x],
                                marker=marker[x])

            axes[j, 1].plot(res_label, np.array(omp_res[0:len(res_label)]) / max_iter, color=color[x], marker=marker[x])
            axes[j, 2].plot(res_label, np.array(omp_ber[0:len(res_label)]) / max_iter, color=color[x], marker=marker[x])
            axes[j, 3].plot(res_label, np.array(omp_stm[0:len(res_label)]) / max_iter, color=color[x], marker=marker[x])
            axes[j, 4].plot(spu_label, np.array(omp_spu[0:len(spu_label)]) / max_iter, color=color[x], marker=marker[x])

            omp_lll = []
            omp_lsu = []
            omp_ttm = []

            omp_lll.append(0)
            for l in range(0, qr_l):
                omp_lll.append(lll_qr[l][j])
                omp_lsu.append(lll_qr_spu[l + 1][j])

            print(omp_lll)
            omp_ttm.append(0)
            omp_ttm.append(0)
            for l in range(2, l_max):
                omp_ttm.append(omp_stm[l] + omp_lll[l])
            omp_ttm[0] = (omp_stm[0] + omp_lll[1])
            omp_ttm[1] = (omp_stm[1] + omp_lll[1])
            if j == 0 and x == 0:
                axes[j, 4].plot(spu_label, np.array(omp_lsu[0:len(spu_label)]) / max_iter, color=color[3],
                                marker=marker[3], label=labels[3])
            else:
                axes[j, 4].plot(spu_label, np.array(omp_lsu[0:len(spu_label)]) / max_iter, color=color[3],
                                marker=marker[3])

            if j == 0:
                axe2[j, 0].plot(res_label, np.array(omp_ttm[0:len(res_label)]) / max_iter, color=color[x],
                                marker=marker[x], label=labels[x])
            else:
                axe2[j, 0].plot(res_label, np.array(omp_ttm[0:len(res_label)]) / max_iter, color=color[x],
                                marker=marker[x])

            axe2[j, 1].plot(spu_label, np.array(omp_tpu[0:len(spu_label)]) / max_iter, color=color[x], marker=marker[x])
            if j == 0 and x == 0:
                axe2[j, 0].plot(res_label[1:len(res_label)], np.array(omp_lll[1:len(res_label)]) / max_iter,
                                color=color[3], marker=marker[3], label=labels[3])
            else:
                axe2[j, 0].plot(res_label[1:len(res_label)], np.array(omp_lll[1:len(res_label)]) / max_iter,
                                color=color[3], marker=marker[3])

        axes[j, 0].set_xticklabels(itr_label, rotation=45)
        axes[j, 1].set_xticklabels(res_label, rotation=45)
        axes[j, 2].set_xticklabels(res_label, rotation=45)
        axes[j, 3].set_xticklabels(res_label, rotation=45)
        axes[j, 4].set_xticklabels(spu_label, rotation=45)
        axe2[j, 0].set_xticklabels(res_label, rotation=45)
        axe2[j, 1].set_xticklabels(spu_label, rotation=45)

        axes[j, 0].grid(color='b', ls='-.', lw=0.25)
        axes[j, 1].grid(color='b', ls='-.', lw=0.25)
        axes[j, 2].grid(color='b', ls='-.', lw=0.25)
        axes[j, 3].grid(color='b', ls='-.', lw=0.25)
        axes[j, 4].grid(color='b', ls='-.', lw=0.25)
        axe2[j, 0].grid(color='b', ls='-.', lw=0.25)
        axe2[j, 1].grid(color='b', ls='-.', lw=0.25)

    title1 = 'Box-constrained'
    if not is_constrained:
        title1 = 'Unconstrained'
    titile2 = ''
    if is_qr == 0:
        title2 = 'with LLL reduction'
    title3 = 'problem'
    if m > n:
        title3 = 'underdetermined problem'
    title = f'Test Resutls for {title1} {title3} with {str(SNR)}-SNR, 4 and 64-QAM, block size {str(block_size)}, and problem size {str(m)} x {str(n)}'

    fig.suptitle("\n".join(wrap(title, len(title) / 2)), fontsize=15)
    fig.legend(bbox_to_anchor=(1, 1), title="Legend", ncol=4)

    fi2.suptitle("\n".join(wrap(title, len(title) / 2)), fontsize=15)
    fi2.legend(bbox_to_anchor=(1, 1), title="Legend", ncol=4)
    # from matplotlib import ticker
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1,1))
    # axes[:, 3].yaxis.set_major_formatter(formatter)

    plt.savefig(f'./{n}_report_plot_{SNR}_{block_size}_{is_qr}_{int(max_iter / 100)}_{title1}')
    plt.close()

    plt2.savefig(f'./{n}_time_plot_{SNR}_{block_size}_{is_qr}_{int(max_iter / 100)}_{title1}')
    plt2.close()
    print("\n----------END PLOT RUNTIME--------------\n")
    plot_first_block(n, SNR, k, block_size, ser_tim, is_qr, d_s)

    print("\n----------PRINT TIMETABLE--------------\n")
    for j in range(0, 2):
        np_time = np.array(time).astype(float)
        babai = np_time[0, 1:max_iter + 1, :, j]
        block = np_time[1, 1:max_iter + 1, :, j]
        nt_03 = np_time[2, 1:max_iter + 1, :, j]
        nt_06 = np_time[3, 1:max_iter + 1, :, j]
        nt_09 = np_time[4, 1:max_iter + 1, :, j]
        nt_12 = np_time[5, 1:max_iter + 1, :, j]
        nt_15 = np_time[6, 1:max_iter + 1, :, j]

        # print(time[5])
        babai_stats = [np.mean(babai, axis=0), np.median(babai, axis=0), np.amax(babai, axis=0), np.amin(babai, axis=0)]
        block_stats = [np.mean(block, axis=0), np.median(block, axis=0), np.amax(block, axis=0), np.amin(block, axis=0)]
        nt_03_stats = [np.mean(nt_03, axis=0), np.median(nt_03, axis=0), np.amax(nt_03, axis=0), np.amin(nt_03, axis=0)]
        nt_06_stats = [np.mean(nt_06, axis=0), np.median(nt_06, axis=0), np.amax(nt_06, axis=0), np.amin(nt_06, axis=0)]
        nt_09_stats = [np.mean(nt_09, axis=0), np.median(nt_09, axis=0), np.amax(nt_09, axis=0), np.amin(nt_09, axis=0)]
        nt_12_stats = [np.mean(nt_12, axis=0), np.median(nt_12, axis=0), np.amax(nt_12, axis=0), np.amin(nt_12, axis=0)]
        nt_15_stats = [np.mean(nt_15, axis=0), np.median(nt_15, axis=0), np.amax(nt_15, axis=0), np.amin(nt_15, axis=0)]

        # print(babai)
        babai_df = pd.DataFrame(babai_stats, columns=["Rounded", "0", "Average"],
                                index=["Average", "Median", "Maximum", "Minimum"])
        block_df = pd.DataFrame(block_stats, columns=["Rounded", "0", "Average"],
                                index=["Average", "Median", "Maximum", "Minimum"])
        nt_03_df = pd.DataFrame(nt_03_stats, columns=["Rounded", "0", "Average"],
                                index=["Average", "Median", "Maximum", "Minimum"])
        nt_06_df = pd.DataFrame(nt_06_stats, columns=["Rounded", "0", "Average"],
                                index=["Average", "Median", "Maximum", "Minimum"])
        nt_09_df = pd.DataFrame(nt_09_stats, columns=["Rounded", "0", "Average"],
                                index=["Average", "Median", "Maximum", "Minimum"])
        nt_12_df = pd.DataFrame(nt_12_stats, columns=["Rounded", "0", "Average"],
                                index=["Average", "Median", "Maximum", "Minimum"])
        nt_15_df = pd.DataFrame(nt_15_stats, columns=["Rounded", "0", "Average"],
                                index=["Average", "Median", "Maximum", "Minimum"])

        print(babai_df)
        print(block_df)
        print(nt_03_df)
        print(nt_06_df)
        print(nt_09_df)
        print(nt_12_df)
        print(nt_15_df)
        print("------------------1------------------------")

    print("\n----------END TIMETABLE--------------\n")


def plot_first_block(n, SNR, k, block_size, ser_tim, is_qr, d_s):
    print("\n----------PLOT BLOCK TIME--------------\n")
    plt2.rcParams["figure.figsize"] = (18, 8)
    fig, axes = plt2.subplots(2, 2, constrained_layout=True)
    color = ['r', 'g', 'b', 'y']
    marker = ['o', '+', 'x', '.']
    print(d_s)
    for j in range(0, 2):

        # SNR = int(lines[k].split(":")[1].split("\n")[0])
        # init_res = float(lines[k + 1].split(",")[0].split(":")[1].split("\n")[0])
        qam = 4 if j == 0 else 64
        axes[j, 0].set_title(str(qam) + '-QAM', fontsize=13)
        axes[j, 1].set_title(str(qam) + '-QAM', fontsize=13)
        axes[j, 0].set_ylabel('Percentage of Time', fontsize=13)
        axes[j, 1].set_ylabel('Percentage of Search Iteration', fontsize=13)
        axes[j, 0].set_xlabel('The i-th block', fontsize=13)
        axes[j, 1].set_xlabel('The i-th block', fontsize=13)

        for x in range(0, 3):  # only the $x_{init} = 0
            tim = []
            itr = []
            for l in range(0, len(d_s)):
                tim.append(ser_tim[x][l][j])
                itr.append(ser_tim[x][l + len(d_s)][j])

            labels = ['$x_{init} = round(x_R)$', '$x_{init} = 0$', '$x_{init} = avg$']
            # range(0, d_s)#

            print(tim, len(tim))

            # explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
            # axes[j, 0].pie(tim, labels=range(0, d_s), autopct='%1.1f%%', shadow=True, startangle=90)
            axes[j, 0].plot(np.array(d_s).astype(str), tim, color=color[x], marker=marker[x], label=labels[x])
            axes[j, 0].set_xticks(np.array(d_s).astype(str))
            axes[j, 0].legend(loc="upper right")
            axes[j, 1].plot(np.array(d_s).astype(str), itr, color=color[x], marker=marker[x], label=labels[x])
            axes[j, 1].set_xticks(np.array(d_s).astype(str))
            axes[j, 1].legend(loc="upper right")

    title = 'ILS Sequential Solving Time Per block for ' + str(SNR) \
            + '-SNR and 4, 64-QAM and problem size ' + str(n) + ', block size ' + str(block_size)
    if is_qr == 0:
        title += ' with LLL reduction'

    fig.suptitle(title, fontsize=15)
    plt2.savefig('./' + str(n) + '_block_time_' + str(SNR) + '_' + str(block_size) + '_' + str(is_qr))
    plt2.close()
    print("\n----------END PLOT BLOCK TIME--------------\n")
