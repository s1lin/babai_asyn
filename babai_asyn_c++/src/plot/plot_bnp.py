import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np

rc = {"font.family": "serif", "mathtext.fontset": "stix"}

plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 19})


def save_data(n, i, start, end, qrT, asplT, bnp, ber, itr):
    np.savez(f'./test_result/{n}_report_plot_{start}_{end}_BNP.npz',
             n=n, i=i, start=start, end=end, qrT=qrT, asplT=asplT, bnp=bnp, ber=ber, itr=itr)


def plot_bnp(n, i, start, end, qrT, asplT, bnp, ber, itr):
    np.savez(f'./test_result/{n}_report_plot_{start}_{end}BNP.npz', n=n, i=i, start=start, end=end,
             qrT=qrT, asplT=asplT, bnp=bnp, ber=ber, itr=itr)
    print("\n----------PLOT RUNTIME--------------\n")
    color = ['r', 'g', 'b', 'm']
    marker = ['o', '+', 'x', '*']
    procs = [5, 10, 15, 20]
    labels = [r'$\mathbf{z}^{(0)} = \mathbf{0}$', r'$\mathbf{z}^{(0)} = avg$',
              r'$\mathbf{z}^{(0)} = \lfloor\mathbf{z}_r\rceil$', 'Reduction']
    itr_label = ['$1$'] + ['$' + str(proc) + '$' for proc in procs]

    # a = np.load(f'./test_result/{n}_report_plot_200_BNP.npz')
    # i = a['i']
    # ber = a['ber']
    # qrT = a['qrT']
    # asplT = a['asplT']

    for j in range(0, 2):
        if j == 0:
            plt.rcParams["figure.figsize"] = (16, 16)
            fig, axes = plt.subplots(3, 2, constrained_layout=True)
        else:
            plt2.rcParams["figure.figsize"] = (16, 16)
            fig, axes = plt2.subplots(3, 2, constrained_layout=True)

        qam = 4 if j == 0 else 64
        axes[0, 1].set_title('Avg. Iterations vs Number of Cores', fontweight="bold")
        axes[0, 0].set_title('Avg. BER vs Number of Cores', fontweight="bold")
        axes[1, 0].set_title('Avg. Time vs Number of Cores', fontweight="bold")
        axes[1, 1].set_title('Speed Up vs Number of Cores', fontweight="bold")
        axes[2, 0].set_title('Avg. Time with Reduction vs Number of Cores', fontweight="bold")
        axes[2, 1].set_title('Total Speed Up vs Number of Cores', fontweight="bold")

        axes[0, 1].set_ylabel('Avg. Iterations', fontweight="bold")
        axes[0, 0].set_ylabel('Avg. BER', fontweight="bold")
        axes[1, 0].set_ylabel('Avg. Time (seconds)', fontweight="bold")
        axes[1, 1].set_ylabel('Speed Up', fontweight="bold")
        axes[2, 0].set_ylabel('Avg. Time (seconds)', fontweight="bold")
        axes[2, 1].set_ylabel('Speed Up', fontweight="bold")

        # reduction:
        a_t = []
        spu = []
        spu2 = []
        for t in range(0, i + 1):
            total = []
            for l in range(0, len(itr_label)):
                print(qrT[t][l][j])
                if t == 0:
                    a_t.append(0)
                # if l == len(itr_label) - 1:
                #     print(totalT[d][t][l][j], totalT[d][t][l+1][j], totalT[d][t][l+2][j])
                #     a_t[l] = a_t[l] + min(totalT[d][t][l][j], totalT[d][t][l+1][j], totalT[d][t][l+2][j])
                # else:
                # if l == len(itr_label) - 1:
                total.append(asplT[t][l][j] + qrT[t][l][j])
                # else:
                #     total.append(min(asplT[t][l][j], asplT[t][l + 1][j]) + \
                #                  min(qrT[t][l][j], qrT[t][l + 1][j]))

                a_t[l] = a_t[l] + total[l]

                if l > 0:
                    if t == 0:
                        spu.append(0)
                        spu2.append(0)

                    spu[l - 1] = spu[l - 1] + qrT[t][0][j] / qrT[t][l][j]
                    spu2[l - 1] = spu2[l - 1] + total[0] / total[l]

        for x in range(0, 3):  # init
            omp_itr = []
            omp_ber = []
            omp_red = []
            omp_stm = []
            omp_spu = []
            omp_spu2 = []
            omp_total = []
            for t in range(0, i + 1):
                to0 = qrT[t][0][j] + asplT[t][0][j] + bnp[x][t][0][j]
                for l in range(0, len(itr_label)):

                    if t == 0:
                        omp_itr.append(0)
                        omp_ber.append(0)
                        omp_red.append(0)
                        omp_stm.append(0)
                        omp_total.append(0)
                    # if l == len(itr_label) - 1:
                    #     print(totalT[d][t][l][j], totalT[d][t][l+1][j], totalT[d][t][l+2][j])
                    #     a_t[l] = a_t[l] + min(totalT[d][t][l][j], totalT[d][t][l+1][j], totalT[d][t][l+2][j])
                    # else:
                    omp_itr[l] = omp_itr[l] + itr[x][t][l][j]
                    omp_ber[l] = omp_ber[l] + ber[x][t][l][j]

                    tot = qrT[t][l][j] + asplT[t][l][j] + bnp[x][t][l][j]

                    if l == len(itr_label) - 1:
                        omp_stm[l] = omp_stm[l] + min(bnp[x][t][l][j], bnp[x][t][l + 1][j], bnp[x][t][l + 2][j])
                        tot = min(qrT[t][l][j], qrT[t][l + 1][j]) + \
                              min(asplT[t][l][j], asplT[t][l + 1][j]) + \
                              min(bnp[x][t][l][j], bnp[x][t][l + 1][j], bnp[x][t][l + 2][j])
                    else:
                        omp_stm[l] = omp_stm[l] + bnp[x][t][l][j]

                    omp_total[l] = omp_total[l] + tot

                    if l > 0:
                        if t == 0:
                            omp_spu.append(0)
                            omp_spu2.append(0)
                        if l == len(itr_label) - 1:
                            omp_spu[l - 1] = omp_spu[l - 1] + bnp[x][t][0][j] / min(bnp[x][t][l][j],
                                                                                    bnp[x][t][l + 1][j], bnp[x][t][l + 2][j])
                        else:
                            omp_spu[l - 1] = omp_spu[l - 1] + bnp[x][t][0][j] / bnp[x][t][l][j]
                        omp_spu2[l - 1] = omp_spu2[l - 1] + to0 / tot


            for l in range(0, len(itr_label)):
                omp_itr[l] = omp_itr[l] / (i + 1)
                omp_ber[l] = omp_ber[l] / (i + 1)
                omp_red[l] = omp_red[l] / (i + 1)
                omp_stm[l] = omp_stm[l] / (i + 1)

                if l < len(itr_label) - 1:
                    omp_spu[l] = omp_spu[l] / (i + 1)
                    omp_spu2[l] = omp_spu2[l] / (i + 1)


            omp_total[0] = omp_total[0] / ((i + 1)*10)
            for l in range(1, len(itr_label)):
                omp_total[l] = omp_total[0] / omp_spu2[l - 1]

            axes[0, 1].plot(itr_label[1:len(itr_label)], omp_itr[1:len(itr_label)], color=color[x], marker=marker[x], label=labels[x], markersize=12)
            axes[0, 0].plot(itr_label, omp_ber, color=color[x], marker=marker[x], markersize=12)
            axes[1, 0].semilogy(itr_label, omp_stm, color=color[x], marker=marker[x], markersize=12)
            axes[1, 1].plot(itr_label[1:len(itr_label)], omp_spu, color=color[x], marker=marker[x], markersize=12)
            axes[2, 0].semilogy(itr_label, omp_total, color=color[x], marker=marker[x], markersize=12)
            axes[2, 1].plot(itr_label[1:len(itr_label)], omp_spu2, color=color[x], marker=marker[x], markersize=12)

        a_t[0] = a_t[0] / ((i + 1)*10)
        for l in range(0, len(itr_label) - 1):
            spu[l] = spu[l] / (i+1)
            spu2[l] = spu2[l] / (i+1)
            # if spu[l] > cores[l]:
            spu[l] = spu2[l]

        for l in range(1, len(itr_label)):
            a_t[l] = a_t[0] / spu[l - 1]

        axes[1, 0].plot(itr_label, a_t, color=color[3], marker=marker[3], markersize=12, label=labels[3])
        axes[1, 1].plot(itr_label[1:len(itr_label)], spu, color=color[3], marker=marker[3], markersize=12)

        for f in range(0, 3):
            for ff in range(0, 2):
                axes[f, ff].set_xticklabels(itr_label)
                axes[f, ff].grid(color='b', ls='-.', lw=0.25)
                axes[f, ff].patch.set_edgecolor('black')
                axes[f, ff].patch.set_linewidth('1')
        axes[0, 1].set_xticklabels(itr_label[1:len(itr_label)])
        axes[1, 1].set_xticklabels(itr_label[1:len(itr_label)])
        axes[2, 1].set_xticklabels(itr_label[1:len(itr_label)])

        fig.suptitle("\n\n\n\n\n")
        fig.legend(bbox_to_anchor=(0.88, 0.97), title="Legend", ncol=4, fontsize=21, title_fontsize=21,
                   edgecolor='black')
        if j == 0:
            plt.savefig(f'./{n}-{qam}_report_plot_BNP.png')
            plt.savefig(f'./{n}-{qam}_report_plot_BNP.eps', format='eps', dpi=1200)
            plt.close()
        else:
            plt2.savefig(f'./{n}-{qam}_report_plot_BNP.png')
            plt2.savefig(f'./{n}-{qam}_report_plot_BNP.eps', format='eps', dpi=1200)
            plt2.close()

    print("\n----------END PLOT RUNTIME--------------\n")


if __name__ == "__main__":
    n = 512
    a = np.load(f'./test_result/{n}_report_plot_190_BNP.npz')
    i = a['i']
    start = a['start']
    end = a['end']
    qrT = a['qrT']
    asplT = a['asplT']
    bnp=a['bnp']
    ber=a['ber']
    itr=a['itr']
    plot_bnp(n, i, start, end, qrT, asplT, bnp, ber, itr)
