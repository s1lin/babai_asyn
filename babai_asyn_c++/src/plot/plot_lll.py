import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
from textwrap import wrap
import pandas as pd
import random


def plot_lll(n, i, max_proc, min_proc, qrT, asplT, totalT):
    print("\n----------PLOT RUNTIME--------------\n")
    np.savez(f'./test_result/{n}_report_plot_{int(i / 200)}_ASPL.npz', n=n, i=i, max_proc=max_proc, min_proc=min_proc,
             qrT=qrT, asplT=asplT, totalT=totalT)
    plt.rcParams["figure.figsize"] = (18, 10)
    fig, axes = plt.subplots(2, 3, constrained_layout=True)
    color = ['r', 'g', 'b', 'm', 'tab:orange']
    marker = ['o', '+', 'x', '*', '>']
    linestyle = ['-.', '-']
    # ax_zoom = fig.add_axes([0.52, 0.51, 0.12, 0.3])
    # proc_num = proc_num.astype(int)
    itr_label = ['SEQ'] + ['NT-' + str(proc) for proc in range(min_proc, max_proc + 1, min_proc)]
    d = 0
    labels = [r'$\mathbf{A}\in\mathbb{R}^{40\times40}$', r'$\mathbf{A}\in\mathbb{R}^{80\times80}$',
              r'$\mathbf{A}\in\mathbb{R}^{120\times120}$',
              r'$\mathbf{A}\in\mathbb{R}^{160\times160}$', r'$\mathbf{A}\in\mathbb{R}^{200\times200}$']
    for dim in range(40, 41, 40):

        for k in range(0, 2):
            axes[k, 0].set_title(f'Case {k + 1}: Solve Time', fontsize=13)
            axes[k, 1].set_title(f'Case {k + 1}: Solve Time (log)', fontsize=13)
            axes[k, 2].set_title(f'Case {k + 1}: Speed Up', fontsize=13)

            axes[k, 0].set_ylabel('Solve Time (s)', fontsize=13)
            axes[k, 1].set_ylabel('Solve Time (log)', fontsize=13)
            axes[k, 2].set_ylabel('Speed Up x times', fontsize=13)

            a_t = []
            spu = []
            print(qrT)
            print(qrT[0][:])
            for t in range(0, i + 1):
                for l in range(0, len(itr_label)):
                    if t == 0:
                        a_t.append(0)
                    a_t[l] = a_t[l] + qrT[d][t][l][k]

                    if l > 0:
                        if t == 0:
                            spu.append(0)
                        spu[l - 1] = spu[l - 1] + qrT[d][t][0][k] / qrT[d][t][l][k]

            for l in range(0, len(itr_label)):
                a_t[l] = a_t[l] / i

            for l in range(0, len(itr_label) - 1):
                spu[l] = spu[l] / i

            # print(a_t)
            if k == 0:
                axes[k, 0].plot(itr_label, np.array(a_t), color=color[d], marker=marker[d], label=labels[d])
            else:
                axes[k, 0].plot(itr_label, np.array(a_t), color=color[d], marker=marker[d])

            axes[k, 1].semilogy(itr_label, np.array(a_t), color=color[d], marker=marker[d])
            axes[k, 2].plot(itr_label[1:len(itr_label)], spu, color=color[d], marker=marker[d])

            axes[k, 0].set_xticklabels(itr_label, rotation=45)
            axes[k, 1].set_xticklabels(itr_label, rotation=45)
            axes[k, 2].set_xticklabels(itr_label[1:len(itr_label)], rotation=45)

        d = d + 1
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
    # ax_zoom.set_title(ax_zoom_title, fontsize=13)
    title = 'Runtime performance of the ASPL algorithm and PASPL algorithm.\n'

    fig.suptitle(title, fontsize=15)
    fig.legend(bbox_to_anchor=(1, 1), title="Legend", ncol=5)

    plt.savefig(f'./test_result/{n}_report_plot_{int(i / 200)}_ASPL')
    plt.close()


if __name__ == "__main__":
    n = 5
    a = np.load(f'./test_result/{n}_report_plot_0_ASPL.npz')
    i = a['i']
    print(i)
    max_proc = a['max_proc']
    min_proc = a['min_proc']
    qrT = a['qrT']
    asplT = a['asplT']
    totalT = a['totalT']
    plot_lll(n, i, max_proc, min_proc, qrT, asplT, totalT)
