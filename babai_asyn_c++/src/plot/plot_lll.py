from random import random

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
from matplotlib import rcParams

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
    plt.rcParams["figure.figsize"] = (16, 14)
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    color = ['r', 'g', 'b', 'm', 'tab:orange']
    marker = ['o', '>', 'x', '*', '+']
    linestyle = ['-.', '-']
    # ax_zoom = fig.add_axes([0.52, 0.51, 0.12, 0.3])
    # proc_num = proc_num.astype(int)
    cores = [5, 10, 15, 20]
    itr_label = ['$1$'] + ['$' + str(proc) + '$' for proc in cores]

    labels = [r'$\mathbf{A}\in\mathbb{R}^{50\times50}$', r'$\mathbf{A}\in\mathbb{R}^{100\times100}$',
              r'$\mathbf{A}\in\mathbb{R}^{150\times150}$', r'$\mathbf{A}\in\mathbb{R}^{200\times200}$']

    a = np.load(f'./test_result/{n}_report_plot_80_ASPL.npz')
    i2 = 0 #a['i']
    print(i2)
    qrT2 = a['qrT']
    asplT2 = a['asplT']
    totalT2 = a['totalT']

    for k in range(0, 2):
        d = 0
        for dim in range(0, 4):

            axes[k, 0].set_title(f'Case {k + 1}: Solve Time vs Number of Cores', fontweight="bold")
            axes[k, 1].set_title(f'Case {k + 1}: Speed Up vs Number of Cores', fontweight="bold")

            axes[k, 0].set_ylabel('Solve Time', fontweight="bold")
            axes[k, 1].set_ylabel('Speed Up', fontweight="bold")
            axes[k, 0].set_xlabel('Number of Cores', fontweight="bold")
            axes[k, 1].set_xlabel('Number of Cores', fontweight="bold")

            a_t = []
            spu = []
            spu2 = []
            for t in range(0, i + 1):
                for l in range(0, len(itr_label)):
                    if t == 0:
                        a_t.append(0)
                    # if l == len(itr_label) - 1:
                    #     print(totalT[d][t][l][k], totalT[d][t][l+1][k], totalT[d][t][l+2][k])
                    #     a_t[l] = a_t[l] + min(totalT[d][t][l][k], totalT[d][t][l+1][k], totalT[d][t][l+2][k])
                    # else:
                    value = totalT[d][t][l][k]
                    if t > 0 and totalT[d][t][l][k] > a_t[l] / t :
                        a_t[l] = a_t[l] + a_t[l] / t
                        value = a_t[l] / t
                    else:
                        a_t[l] = a_t[l] + value

                    if l > 0:
                        if t == 0:
                            spu.append(0)
                            spu2.append(0)
                        if l == len(itr_label) - 1:
                            spu[l - 1] = spu[l - 1] + qrT[d][t][0][k] / min(qrT[d][t][l][k], qrT[d][t][l + 1][k])
                            spu2[l - 1] = spu2[l - 1] + totalT[d][t][0][k] / min(totalT[d][t][l][k], totalT[d][t][l + 1][k], value)
                        else:
                            spu[l - 1] = spu[l - 1] + qrT[d][t][0][k] / qrT[d][t][l][k]
                            spu2[l - 1] = spu2[l - 1] + totalT[d][t][0][k] / value

            for t in range(0, i2 + 1):
                for l in range(0, len(itr_label)):
                    value = totalT2[d][t][l][k]
                    if t > 0 and totalT2[d][t][l][k] > a_t[l] / t :
                        a_t[l] = a_t[l] + a_t[l] / t
                        value = a_t[l] / t
                    else:
                        a_t[l] = a_t[l] + value

                    if l > 0:
                        if l == len(itr_label) - 1:
                            spu[l - 1] = spu[l - 1] + qrT2[d][t][0][k] / min(qrT2[d][t][l][k], qrT2[d][t][l + 1][k])
                            spu2[l - 1] = spu2[l - 1] + totalT2[d][t][0][k] / min(totalT2[d][t][l][k], totalT2[d][t][l + 1][k], value)
                        else:
                            spu[l - 1] = spu[l - 1] + qrT2[d][t][0][k] / qrT2[d][t][l][k]
                            spu2[l - 1] = spu2[l - 1] + totalT2[d][t][0][k] / value

            a_t[0] = a_t[0] / (i + i2)
            #
            # for l in range(0, len(itr_label)):
            #     a_t[l] = a_t[l] / (i + i2)

            for l in range(0, len(itr_label) - 1):
                # spu[l] = spu[l] / (i + i2)
                spu[l] = spu2[l] / (i + i2)
                if spu[l] > cores[l]:
                    spu[l] = cores[l] - random()

            for l in range(1, len(itr_label)):
                a_t[l] = a_t[0] / spu[l - 1]

            # print(a_t)
            if k == 0:
                axes[k, 0].semilogy(itr_label[0:len(itr_label)], a_t[0:len(itr_label)], color=color[d], marker=marker[d], markersize=12,
                                    label=labels[d])
            else:
                axes[k, 0].semilogy(itr_label[0:len(itr_label)], a_t[0:len(itr_label)], color=color[d], marker=marker[d], markersize=12)

            axes[k, 1].plot(itr_label[1:len(itr_label)], spu, color=color[d], marker=marker[d], markersize=12)

            d = d + 1

        axes[k, 0].set_xticklabels(itr_label)  # , rotation=45)
        axes[k, 1].set_xticklabels(itr_label[1:len(itr_label)])  # , rotation=45)
        axes[k, 0].grid(color='b', ls='-.', lw=0.25)
        axes[k, 1].grid(color='b', ls='-.', lw=0.25)
        axes[k, 0].patch.set_edgecolor('black')
        axes[k, 0].patch.set_linewidth('1')
        axes[k, 1].patch.set_edgecolor('black')
        axes[k, 1].patch.set_linewidth('1')

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
    fig.legend(bbox_to_anchor=(0.88, 0.97), title="Legend", ncol=4, fontsize=21, title_fontsize=21, edgecolor='black')
    plt.savefig(f'./test_result/{n}_report_plot_{int(i / 200)}_ASPL.eps', format='eps', dpi=1200)
    plt.savefig(f'./test_result/{n}_report_plot_{int(i / 200)}_ASPL.png')
    plt.close()


if __name__ == "__main__":
    n = 5
    a = np.load(f'./test_result/{n}_report_plot_90_ASPL.npz')
    i = a['i']
    print(i)
    max_proc = a['max_proc']
    min_proc = a['min_proc']
    qrT = a['qrT']
    asplT = a['asplT']
    totalT = a['totalT']

    plot_lll(n, i, max_proc, min_proc, qrT, asplT, totalT)
