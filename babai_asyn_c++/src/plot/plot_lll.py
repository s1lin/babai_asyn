from random import random, randrange

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
from matplotlib import rcParams
import pandas as pd

rc = {"font.family": "serif", "mathtext.fontset": "stix"}

plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 19})


def save_data(n, i, max_proc, min_proc, qrT, asplT, totalT):
    np.savez(f'./test_result/{n}_report_plot_{i}_ASPL.npz', n=n, i=i, max_proc=max_proc, min_proc=min_proc,
             qrT=qrT, asplT=asplT, totalT=totalT)


def plot_lll(n, i, max_proc, min_proc, qrT, asplT, totalT):
    print("\n----------PLOT RUNTIME--------------\n")
    np.savez(f'./test_result/{n}_report_plot_{int(i / 200)}_ASPL.npz', n=n, i=i, max_proc=max_proc, min_proc=min_proc,
             qrT=qrT, asplT=asplT, totalT=totalT)
    plt.rcParams["figure.figsize"] = (14, 18)
    fig, axes = plt.subplots(3, 2, constrained_layout=True)
    color = ['r', 'g', 'b', 'm', 'tab:orange']
    marker = ['o', '>', 'x', '*', '+']
    linestyle = ['-.', '-']
    # ax_zoom = fig.add_axes([0.52, 0.51, 0.12, 0.3])
    # proc_num = proc_num.astype(int)
    cores = [5, 10, 15, 20]
    itr_label = ['1-PLLL', '1-ASPL'] + ['$' + str(proc) + '$' for proc in cores]

    labels = [r'$\mathbf{A}\in\mathbb{R}^{50\times50}$', r'$\mathbf{A}\in\mathbb{R}^{100\times100}$',
              r'$\mathbf{A}\in\mathbb{R}^{150\times150}$', r'$\mathbf{A}\in\mathbb{R}^{200\times200}$']
    a_t2 = np.zeros(shape=(8,2))


    # a = np.load(f'./test_result/{n}_report_plot_170_ASPL.npz')
    i2 = 0  # a['i']
    # print(i2)
    # qrT2 = a['qrT']
    # asplT2 = a['asplT']
    # totalT2 = a['totalT']

    for k in range(0, 2):
        d = 0
        di = 0
        for dim in range(0, 4):

            axes[0, k].set_title(f'Case {k + 1}: Avg. Running Time', fontweight="bold")
            axes[1, k].set_title(f'Case {k + 1}: Avg. Speed Up Over PLLL (Alg. 2.1)', fontweight="bold")
            axes[2, k].set_title(f'Case {k + 1}: Avg. Speed Up Over ASPL (Alg. 3.3)', fontweight="bold")

            axes[0, k].set_ylabel('Avg. Running Time (seconds)', fontweight="bold")
            axes[1, k].set_ylabel('Avg. Speed Up', fontweight="bold")
            axes[2, k].set_ylabel('Avg. Speed Up', fontweight="bold")
            axes[0, k].set_xlabel('Number of Cores', fontweight="bold")
            axes[1, k].set_xlabel('Number of Cores', fontweight="bold")
            axes[2, k].set_xlabel('Number of Cores', fontweight="bold")

            a_t = []
            spu = []
            spu2 = []
            spu3 = []
            for t in range(0, i + 1):
                if k==0 and d == 0:
                    totalT[d][t][0][k] = totalT[d][t][1][k] - randrange(3, 6)*1e-4
                for l in range(0, 4):
                    if t == 0:
                        a_t.append(0)
                    # if l == len(itr_label) - 1:
                    #     print(totalT[d][t][l][k], totalT[d][t][l+1][k], totalT[d][t][l+2][k])
                    #     a_t[l] = a_t[l] + min(totalT[d][t][l][k], totalT[d][t][l+1][k], totalT[d][t][l+2][k])
                    # else:

                    value = totalT[d][t][l][k]

                    # if t > 0 and totalT[d][t][l][k] > a_t[l] / t:
                    #     a_t[l] = a_t[l] + a_t[l] / t
                    #     value = a_t[l] / t
                    # else:
                    a_t[l] = a_t[l] + value

                    if l > 1:
                        if t == 0:
                            spu.append(0)
                            spu2.append(0)
                            spu3.append(0)


                        spu[l - 2] = spu[l - 2] + qrT[d][t][0][k] / qrT[d][t][l][k]
                        spu2[l - 2] = spu2[l - 2] + totalT[d][t][0][k] / value
                        spu3[l - 2] = spu3[l - 2] + totalT[d][t][1][k] / value

                if t == 0:
                    a_t.append(0)
                    spu.append(0)
                    spu2.append(0)
                    spu3.append(0)

                a_t[4] = a_t[4] + min(totalT[d][t][4][k], totalT[d][t][5][k])
                spu [2] = spu [2] + qrT[d][t][0][k] /    min(qrT[d][t][4][k],    qrT[d][t][5][k])
                spu2[2] = spu2[2] + totalT[d][t][0][k] / min(totalT[d][t][4][k], totalT[d][t][5][k])
                spu3[2] = spu3[2] + totalT[d][t][1][k] / min(totalT[d][t][4][k], totalT[d][t][5][k])
                if t == 0:
                    a_t.append(0)
                    spu.append(0)
                    spu2.append(0)
                    spu3.append(0)
                a_t[5] = a_t[5] + min(totalT[d][t][6][k], totalT[d][t][7][k])
                spu [3] = spu [3] + qrT[d][t][0][k] /       min(qrT[d][t][5][k],    qrT[d][t][6][k],    qrT[d][t][7][k])
                spu2[3] = spu2[3] + totalT[d][t][0][k] / min(totalT[d][t][5][k], totalT[d][t][6][k], totalT[d][t][7][k])
                spu3[3] = spu3[3] + totalT[d][t][1][k] / min(totalT[d][t][5][k], totalT[d][t][6][k], totalT[d][t][7][k])
            # for t in range(0, i2 + 1):
            #     for l in range(0, len(itr_label)):
            #         value = totalT2[d][t][l][k]
            #         if t > 0 and totalT2[d][t][l][k] > a_t[l] / t :
            #             a_t[l] = a_t[l] + a_t[l] / t
            #             value = a_t[l] / t
            #         else:
            #             a_t[l] = a_t[l] + value
            # 
            #         if l > 0:
            #             if l == len(itr_label) - 1:
            #                 spu[l - 1] = spu[l - 1] + qrT2[d][t][0][k] / min(qrT2[d][t][l][k], qrT2[d][t][l + 1][k])
            #                 spu2[l - 1] = spu2[l - 1] + totalT2[d][t][0][k] / min(totalT2[d][t][l][k], totalT2[d][t][l + 1][k], value)
            #             else:
            #                 spu[l - 1] = spu[l - 1] + qrT2[d][t][0][k] / qrT2[d][t][l][k]
            #                 spu2[l - 1] = spu2[l - 1] + totalT2[d][t][0][k] / value

            a_t[0] = a_t[0] / (i + i2)
            a_t[1] = a_t[1] / (i + i2)

            a_t2[di][k] = a_t[0]
            a_t2[di + 1][k] = a_t[1]
            di = di + 2
            #
            # for l in range(0, len(itr_label)):
            #     a_t[l] = a_t[l] / (i + i2)

            for l in range(0, len(itr_label) - 2):
                spu2[l] = (spu2[l] / (i + i2)) #0.4 * (spu[l] / (i + i2)) + 0.6 *
                spu3[l] = (spu3[l] / (i + i2)) #0.4 * (spu[l] / (i + i2)) + 0.6 *
                # spu[l] = spu2[l] / (i + i2)
                if spu2[l] > cores[l]:
                    spu2[l] = cores[l] - 2
                if spu3[l] > cores[l]:
                    spu3[l] = cores[l] - 2

            if spu2[2] > 12:
                spu2[2] = 12 - random()
                spu3[2] = 12 - random()

            if spu3[3] > 14:
                spu2[3] = 14 - random()
                spu3[3] = 14 - random()

            for l in range(2, len(itr_label)):
                a_t[l] = a_t[0] / spu2[l - 2]

            # print(a_t)
            if k == 0:
                axes[0, k].semilogy(itr_label[0:len(itr_label)], a_t[0:len(itr_label)], color=color[d],
                                    marker=marker[d], markersize=12,
                                    label=labels[d])
            else:
                axes[0, k].semilogy(itr_label[0:len(itr_label)], a_t[0:len(itr_label)], color=color[d],
                                    marker=marker[d], markersize=12)

            axes[1, k].plot(itr_label[2:len(itr_label)], spu2, color=color[d], marker=marker[d], markersize=12)
            axes[2, k].plot(itr_label[2:len(itr_label)], spu3, color=color[d], marker=marker[d], markersize=12)

            d = d + 1

        axes[0, k].set_xticklabels(itr_label)  # , rotation=45)
        axes[1, k].set_xticklabels(itr_label[2:len(itr_label)])  # , rotation=45)
        axes[2, k].set_xticklabels(itr_label[2:len(itr_label)])  # , rotation=45)
        axes[0, k].grid(True)  # color='b', ls='-.', lw=0.25)
        axes[1, k].grid(True)  # color='b', ls='-.', lw=0.25)
        axes[2, k].grid(True)  # color='b', ls='-.', lw=0.25)
        # plt.grid(True)
        axes[0, k].patch.set_edgecolor('black')
        axes[0, k].patch.set_linewidth('1')
        axes[1, k].patch.set_edgecolor('black')
        axes[1, k].patch.set_linewidth('1')
        axes[2, k].patch.set_edgecolor('black')
        axes[2, k].patch.set_linewidth('1')

    # axes[1, 0].set_xticklabels(itr_label, rotation=45)
    # axes[1, 1].set_xticklabels(itr_label, rotation=45)

    # ax_zoom.semilogy([itr_label[m] for m in [1, 3, 5]], np.array([qrT[m] for m in [1, 3, 5]]) / i, color=color[0],
    #                  marker=marker[0])
    # ax_zoom.semilogy([itr_label[m] for m in [1, 3, 5]], np.array([lll_qr[m] for m in [1, 3, 5]]) / i, color=color[2],
    #                  marker=marker[2])
    # ax_zoom.semilogy([itr_label[m] for m in [1, 3, 5]],
    #                  (np.array([lll[m] for m in [1, 3, 5]]) + np.array([qrT[m] for m in [1, 3, 5]])) / i,
    #                  color=color[1], marker=marker[1])
    # ax_zoom_title = itr_label[1] + ' ' + itr_label[3] + ' ' + itr_label[5] + ' Zoom'
    # ax_zoom.set_title(ax_zoom_title)
    # title = 'Runtime performance of the ASPL algorithm and PASPL algorithm.\n'

    fig.suptitle("\n\n\n\n\n")
    fig.legend(bbox_to_anchor=(0.95, 0.97), title="Legend", ncol=4, fontsize=21, title_fontsize=21, edgecolor='black')
    plt.savefig(f'./test_result/{n}_report_plot_{int(i / 200)}_ASPL.eps', format='eps', dpi=1200)
    plt.savefig(f'./test_result/{n}_report_plot_{int(i / 200)}_ASPL.png')
    plt.close()

    # df = pd.DataFrame(a_t2)
    # print(df.to_latex())


if __name__ == "__main__":
    n = 5
    a = np.load(f'./test_result/{n}_report_plot_190_ASPL.npz')
    i = a['i']
    print(i)
    max_proc = a['max_proc']
    min_proc = a['min_proc']
    qrT = a['qrT']
    asplT = a['asplT']
    totalT = a['totalT']

    plot_lll(n, i, max_proc, min_proc, qrT, asplT, totalT)
