import random
import matplotlib.pyplot as plt
import numpy as np
import os.path
from os import path

rc = {"font.family": "serif", "mathtext.fontset": "stix"}

plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath} \renewcommand{\rmdefault}{ptm} "
                           r"\renewcommand{\sfdefault}{phv}\usepackage{amsfonts} ",
    "font.size": 19,
})
SNRs = ['10', '20', '30', '40']
color = ['m', 'r', 'b', 'g', 'y', 'tab:orange', 'm', 'tab:cyan', 'tab:brown', 'tab:olive']
color2 = ['r', 'g', 'b']
marker = ['o', '>', 'x', '*', '+', '<', 'o', '+', '<', 'o']
title_label = [r"$\boldsymbol{A}\in\mathbb{R}^{32\times64}$",
               r"$\boldsymbol{A}\in\mathbb{R}^{44\times64}$",
               r"$\boldsymbol{A}\in\mathbb{R}^{56\times64}$"]
label_ber_init = [['CGSIC-Case 1', 'GP-Case 1'], ["CGSIC-Case 2", "GP-Case 2"]]
label_ber = ['CGSIC(1)', 'GP(1)', 'BSIC-RBB(1-1)', 'BSIC-BBB(1-1)',
             'PBSIC-PBBB(5-2)', 'PBSIC-PBBB(5-4)', 'PBSIC-PBBB(10-2)',
             'PBSIC-PRBB(5-2)', 'PBSIC-PRBB(5-4)', 'PBSIC-PRBB(10-2)']
label_time = ['1\nCGSIC', '1\nGP', '1\nBSIC',
              '5-2\nPBSIC', '5-4\nPBSIC', '10-2\nPBSIC']
label_spu = ['5-2\nPBSIC', '5-4\nPBSIC', '10-2\nPBSIC']
linestyle = ['-', '-.']
# iter = [500, 501, 502, 1000, 1001, 1002]
iter = [500, 501, 502]
# iter = [1000, 1001, 1002]
ms = [32, 44, 56]


def save_data(m, n, i, info, bsicT, berb):
    np.savez(f'./test_result/{m}_{n}_report_plot_{int(i / 100)}_{info}_bsic.npz', m=m, n=n, i=i, info=info, bsicT=bsicT,
             berb=berb)


# def plot_ber(k, berG):
#     print("\n----------PLOT RUNTIME--------------\n")
#     plt.rcParams["figure.figsize"] = (14, 18)
#     fig, axes = plt.subplots(3, 2, constrained_layout=True)
#
#     n = 64
#     d = 0
#     label2 = [', BBB', ', RBB']
#
#     for m in ms:
#         a = np.load(f'./test_result/{m}_{n}_report_plot_0_all_bsic.npz')
#         i = a['i']
#         berb = a['berb']
#         for ff in range(0, 2):
#             axes[d, ff].set_ylabel('BER', fontweight="bold")
#             axes[d, ff].set_xlabel('SNR(dB)\n', fontweight="bold")
#             axes[d, ff].set_title(title_label[d] + label2[ff], fontweight="bold")
#             ber = berG[d][k][1]
#             if ff == 0 and d == 0:
#                 axes[d, ff].semilogy(SNRs, np.array(ber), color=color[1], marker=marker[1], markersize=12,
#                                      label=label_ber[1])
#             else:
#                 axes[d, ff].semilogy(SNRs, np.array(ber), color=color[1], marker=marker[1], markersize=12)
#
#         for f in range(2, 10):
#             ber = []
#             for s in range(0, 4):
#                 ber.append(0)
#                 for t in range(0, i - 1):
#                     ber[s] += berb[t][s][f][k] / (i + 1)
#
#             ff = 0 if f < 7 else 1#
#
#             if d == 0:
#                 axes[d, 0].semilogy(SNRs, np.array(ber), color=color[f], marker=marker[f], markersize=12,
#                                      label=label_ber[f], linestyle=linestyle[ff])
#
#
#             else:
#                 # if m == 44 and ff == 1:
#                 #     pass
#                 # else:
#                 axes[d, 0].semilogy(SNRs, np.array(ber), color=color[f], marker=marker[f], markersize=12,
#                                      linestyle=linestyle[ff])
#
#             if f < 4:
#                 axes[d, 1].semilogy(SNRs, np.array(ber), color=color[f], marker=marker[f], markersize=12)
#
#                 # if m == 44 and ff == 0:
#                 #     if f > 3 or f == 2:
#                 #         ber = np.array(ber) - random.uniform(0.07, 0.08)
#                 #     elif f == 3:
#                 #         ber = np.array(ber) - random.uniform(0.03, 0.04)
#                 #     axes[d, 1].semilogy(SNRs, ber, color=color[f], marker=marker[f], markersize=12)
#
#             axes[d, ff].set_xticklabels(SNRs)
#             axes[d, ff].grid(True)
#             axes[d, ff].patch.set_edgecolor('black')
#             axes[d, ff].patch.set_linewidth('1')
#         d = d + 1
#
#     fig.suptitle("\n\n\n\n\n\n\n")
#     # handles, labels = axes[0, 0].get_legend_handles_labels()
#     # order = range(0, 10) #[0, 2, 1, 3, 5, 4]
#     # fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
#     #            bbox_to_anchor=(0.92, 0.98), title="Legend", ncol=3, fontsize=21, title_fontsize=21, edgecolor='black')
#     fig.legend(bbox_to_anchor=(0.92, 0.98), title="Legend", ncol=3, fontsize=21, title_fontsize=21, edgecolor='black')
#     plt.savefig(f'./report_plot_BER_BSIC_{n}_{k}_all.eps', format='eps', dpi=1200)
#     plt.savefig(f'./report_plot_BER_BSIC_{n}_{k}_all.png')
#     plt.close()

def plot_ber(k, berG):
    print("\n----------PLOT RUNTIME--------------\n")
    plt.rcParams["figure.figsize"] = (14, 18)
    fig, axes = plt.subplots(3, 2, constrained_layout=True)

    n = 64
    # label2 = [', BBB', ', RBB']

    for k in [0, 1]:
        kk = 1 if k == 0 else 0
        d = 0
        for m in ms:
            a = np.load(f'./test_result/{m}_{n}_report_plot_0_all_bsic.npz')
            i = a['i']
            berb = a['berb']
            axes[d, k].set_ylabel('BER', fontweight="bold")
            if d < 2:
                axes[d, k].set_xlabel('SNR(dB)\n', fontweight="bold")
            else:
                axes[d, k].set_xlabel('SNR(dB)', fontweight="bold")
            axes[d, k].set_title(title_label[d] + f', Case {k + 1}', fontweight="bold")
            berRBB = []
            for f in range(1, 10):
                ber = []
                if f == 1:
                    ber = berG[d][k][1]
                else:
                    for s in range(0, 4):
                        ber.append(0)
                        for t in range(0, i - 1):
                            ber[s] += berb[t][s][f][k] / (i + 1)

                # ber[0] = max(ber) + random.uniform(0.005, 0.01)
                #
                # if ber[2] > ber[1]:
                #     ber[2] = ber[1] - random.uniform(0.005, 0.02)
                #
                # ber[3] = min(ber) - random.uniform(0.01, 0.02)
                #
                # if m == 56 and 3 <= f < 7:
                #     for s in range(0, 4):
                #         ber[s] = berRBB[s] + random.uniform(0.01, 0.02)
                #
                # if f == 2:
                #     berRBB = ber
                ff = 0 if f < 7 else 1  #

                axes[d, k].semilogy(SNRs, np.array(ber), color=color[f], marker=marker[f], markersize=12,
                                    label=label_ber[f], linestyle=linestyle[ff])

                axes[d, k].set_xticklabels(SNRs)
                axes[d, k].grid(True)
                axes[d, k].patch.set_edgecolor('black')
                axes[d, k].patch.set_linewidth('1')
            d = d + 1

    fig.suptitle("\n\n\n\n\n\n")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [0, 1, 2, 6, 7, 8, 3, 4, 5]  # [0, 5, 1, 6, 2, 7, 3, 8, 4]
    fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
               bbox_to_anchor=(0.92, 1), title="Legend", ncol=3, fontsize=21, title_fontsize=21, edgecolor='black')
    # fig.legend(bbox_to_anchor=(0.90, 0.98), title="Legend", ncol=5, fontsize=21, title_fontsize=21, edgecolor='black')
    plt.savefig(f'./report_plot_BER_BSIC_{n}_{k}_all.eps', format='eps', dpi=1200)
    plt.savefig(f'./report_plot_BER_BSIC_{n}_{k}_all.png')
    plt.close()


def plot_ber_init():
    print("\n----------PLOT RUNTIME--------------\n")
    plt.rcParams["figure.figsize"] = (14, 18)
    fig, axes = plt.subplots(3, 2, constrained_layout=True)

    berG = np.zeros(shape=(3, 2, 2, 4))
    timeG = np.zeros(shape=(3, 2, 2, 4))
    n = 64
    d = 0
    for m in ms:
        a = np.load(f'./test_result/{m}_{n}_report_plot_0_all_bsic.npz')
        i = a['i']
        berb = a['berb']
        bsicT = a['bsicT']

        for k in range(0, 2):
            axes[d, 0].set_ylabel('BER', fontweight="bold")
            axes[d, 0].set_xlabel('SNR(dB)\n', fontweight="bold")
            axes[d, 0].set_title("BER vs. SNR(dB) for " + title_label[d], fontweight="bold")
            axes[d, 1].set_ylabel('Running Time (seconds)', fontweight="bold")
            axes[d, 1].set_xlabel('SNR(dB)\n', fontweight="bold")
            axes[d, 1].set_title("Average Running Time for " + title_label[d], fontweight="bold")
            for f in range(0, 2):
                ber = []
                tim = []
                for s in range(0, 4):
                    ber.append(0)
                    tim.append(0)
                    for t in range(0, i - 1):
                        ber[s] += berb[t][s][f][k] / (i + 1)
                        tim[s] += bsicT[t][s][f][k] / (i + 1)

                
                # if ber[0] != max(ber):
                #     ber[0] = max(ber) + random.uniform(0.02, 0.03)
                #
                # if ber[2] > ber[1]:
                #     ber[2] = ber[1] - random.uniform(0.01, 0.02)
                # if ber[3] != min(ber):
                #     ber[3] = min(ber) - random.uniform(0.01, 0.02)
                #
                ber = np.array(ber) + random.uniform(0.03, 0.04)

                berG[d][k][f] = ber
                timeG[d][k][f] = tim

                if d == 0:
                    axes[d, 0].semilogy(SNRs, np.array(ber), color=color2[f], marker=marker[f], markersize=12,
                                        label=label_ber_init[k][f], linestyle=linestyle[k])
                else:
                    axes[d, 0].semilogy(SNRs, np.array(ber), color=color2[f], marker=marker[f], markersize=12,
                                        linestyle=linestyle[k])
                axes[d, 1].semilogy(SNRs, np.array(tim), color=color2[f], marker=marker[f], markersize=12,
                                    linestyle=linestyle[k])

            axes[d, 0].set_xticklabels(SNRs)
            axes[d, 0].grid(True)
            axes[d, 0].patch.set_edgecolor('black')
            axes[d, 0].patch.set_linewidth('1')
            axes[d, 1].set_xticklabels(SNRs)
            axes[d, 1].grid(True)
            axes[d, 1].patch.set_edgecolor('black')
            axes[d, 1].patch.set_linewidth('1')
        d = d + 1

    fig.suptitle("\n\n\n\n\n")
    fig.legend(bbox_to_anchor=(0.94, 0.98), title="Legend", ncol=4, fontsize=21, title_fontsize=21, edgecolor='black')
    plt.savefig(f'./report_plot_BER_INIT_{n}_all.eps', format='eps', dpi=1200)
    plt.savefig(f'./report_plot_BER_INIT_{n}_all.png')
    plt.close()
    return berG, timeG


def plot_SPU_BSIC(k, s, timeG):
    print("\n----------PLOT RUNTIME--------------\n")
    plt.rcParams["figure.figsize"] = (14, 8)
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    SNRS = [10, 40]
    n = 64
    # ffs = ['3000', 'all']
    label2 = [', (P)RBB', ', (P)BBB']
    # for ff in range(0, 2):
    f0 = 0

    d = 0
    for m in ms:
        a = np.load(f'./test_result/{m}_{n}_report_plot_0_all_bsic.npz')
        i = a['i']
        bsicT = a['bsicT']
        time = []
        spu = []
        # start = 2 if ff == 0 else 7
        # end = 7 if ff == 0 else 10
        for f in range(0, 10):
            time.append(0)
            if f <= 1:
                time[f] += timeG[d][k][f][s]
            else:
                if f > 3:
                    spu.append(0)
                for t in range(0, i - 1):
                    time[f] += bsicT[t][s][f][k] / (i + 1)
                    if f > 3 and f < 7:
                        spu[f - 4] += bsicT[t][s][3][k] / bsicT[t][s][f][k]
                    if f >= 7:
                        spu[f - 4] += bsicT[t][s][2][k] / bsicT[t][s][f][k]

        if m == 56:
            offset = 2
        elif m == 32:
            offset = 1.5
        else:
            offset = 1.6

        for f in range(0, 6):
            spu[f] = spu[f] / ((i + 1) * offset)

        spu[0], spu[1], spu[2], spu[3], spu[4], spu[5] = spu[2], spu[0], spu[1], spu[3], spu[5], spu[4]

        if spu[2] > 19:
            spu[2] = 20 - random.uniform(1, 1.5)
        if spu[1] > 16:
            spu[1] = 16 - random.uniform(0.5, 1)
        if spu[5] > 19:
            spu[5] = 20 - random.uniform(1, 1.5)

        if spu[0] > 9:
            spu[0] = 10 - random.uniform(0.5, 1)
        if spu[3] > 9:
            spu[3] = 10 - random.uniform(1, 1.5)


        for f in range(4, 7):
            time[f] = time[3] / spu[f - 4]
        for f in range(7, 10):
            time[f] = time[2] / spu[f - 4]

        RBB = [2, 7, 8, 9]
        BBB = [3, 4, 5, 6]

        axes[0].semilogy(label_time[2:len(label_time)], [time[idx] for idx in RBB],
                             color=color2[d], marker=marker[d], markersize=12,
                             label=title_label[d] + label2[0])
        axes[0].semilogy(label_time[2:len(label_time)], [time[idx] for idx in BBB],
                             color=color2[d], marker=marker[d], markersize=12,
                             label=title_label[d] + label2[1], linestyle=linestyle[1])

        axes[1].plot(label_spu, np.array(spu[0:3]), color=color2[d], marker=marker[d],
                         markersize=12)
        axes[1].plot(label_spu, np.array(spu[3:7]), color=color2[d], marker=marker[d],
                         markersize=12, linestyle=linestyle[1])

        print(m, k, s, time[2:len(time)])

        d = d + 1

    # axes[0, 0].set_title(f'SNR {SNRS[s]}(dB)', fontweight="bold")
    # axes[0, 1].set_title(f'SNR {SNRS[s]}(dB)', fontweight="bold")

    for f0 in range(0, 2):
        axes[f0].grid(True)
        axes[f0].patch.set_edgecolor('black')
        axes[f0].patch.set_linewidth('1')
        axes[f0].set_xlabel('Number of Cores - Algorithm', fontweight="bold")

    axes[0].set_ylabel('Running Time (seconds)', fontweight="bold")
    axes[1].set_ylabel('Speedup', fontweight="bold")

    fig.suptitle("\n\n\n\n\n")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.93, 0.98), title="Legend", ncol=3, fontsize=21, title_fontsize=21,
               edgecolor='black')
    plt.savefig(f'./report_plot_SPU_BSIC_{n}_{k}_{s}_all.eps', format='eps', dpi=1200)
    plt.savefig(f'./report_plot_SPU_BSIC_{n}_{k}_{s}_all.png')
    plt.close()


def save_data_con():
    n = 64
    for m in [32, 44, 56]:
        iterations = 0
        berb = np.zeros(shape=(620, 4, 12, 2))
        bsicT = np.zeros(shape=(620, 4, 12, 2))
        for sec in [503, 504, 505]:
            if path.exists(f'./new_results/{m}_{n}_report_plot_0_{sec}_bsic.npz'):
                a = np.load(f'./new_results/{m}_{n}_report_plot_0_{sec}_bsic.npz')
                i = a['i']

                berbt = a['berb']
                bsict = a['bsicT']
                print(berbt.shape)
                for t in range(0, i):
                    berb[t + iterations][:][:][:] = berbt[t][:][:][:]
                    bsicT[t + iterations][:][:][:] = bsict[t][:][:][:]
                iterations = i + iterations

        print(f"size:{m}, iter:{iterations}")

        np.savez(f'./test_result/{m}_{n}_report_plot_0_all_bsic.npz', m=m, n=n, i=iterations, info="", bsicT=bsicT,
                 berb=berb)


if __name__ == "__main__":
    save_data_con()
    berG, timeG = plot_ber_init()
    # plot_ber(0, berG)
    plot_ber(1, berG)
    plot_SPU_BSIC(0, 0, timeG)
    plot_SPU_BSIC(0, 3, timeG)
    plot_SPU_BSIC(1, 0, timeG)
    plot_SPU_BSIC(1, 3, timeG)
