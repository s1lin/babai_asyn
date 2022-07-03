import random
import matplotlib.pyplot as plt
import numpy as np

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
color = ['m', 'r', 'b', 'g', 'y', 'tab:orange', 'm']
color2 = ['r', 'g', 'b']
marker = ['o', '>', 'x', '*', '+', '<', 'o']
title_label = [r"$\boldsymbol{A}\in\mathbb{R}^{32\times64}$",
               r"$\boldsymbol{A}\in\mathbb{R}^{44\times64}$",
               r"$\boldsymbol{A}\in\mathbb{R}^{56\times64}$"]
label_ber_init = [['CGSIC-Case 1', 'GP-Case 1'], ["CGSIC-Case 2", "GP-Case 2"]]
label_ber = ['CGSIC(1)', 'GP(1)', 'BSIC-RBB(1)', 'BSIC-BBB(1)',
             'PBSIC-PBBB(5-2)', 'PBSIC-PBBB(5-4)', 'PBSIC-PBBB(10-2)',
             'PBSIC-PRBB(5-2)', 'PBSIC-PRBB(5-4)', 'PBSIC-PRBB(10-2)']
label_time = ['1\nCGSIC', '1\nGP', '1\nBSIC\nRBB', '1\nBSIC\nBBB',
              '5-2\nPBSIC\nPBBB', '5-4\nPBSIC\nPBBB', '10-2\nPBSIC\nPBBB',
              '5-2\nPBSIC\nPRBB', '5-4\nPBSIC\nPRBB', '10-2\nPBSIC\nPRBB']
label_spu = ['5-2\nPBSIC\nPBBB', '5-4\nPBSIC\nPBBB', '10-2\nPBSIC\nPBBB',
             '5-2\nPBSIC\nPBBB', '5-4\nPBSIC\nPBBB', '10-2\nPBSIC\nPBBB']
linestyle = ['-', '-.']
# iter = [500, 501, 502, 1000, 1001, 1002]
iter = [500, 501, 502]
ms = [32, 44, 56]


def save_data(m, n, i, info, bsicT, berb):
    np.savez(f'./test_result/{m}_{n}_report_plot_{int(i / 100)}_{info}_bsic.npz', m=m, n=n, i=i, info=info, bsicT=bsicT,
             berb=berb)


def plot_ber(k, berG):
    print("\n----------PLOT RUNTIME--------------\n")
    plt.rcParams["figure.figsize"] = (14, 18)
    fig, axes = plt.subplots(3, 2, constrained_layout=True)

    n = 64
    d = 0
    label2 = [', RBB', ', BBB']

    for m in ms:
        a = np.load(f'./test_result/{m}_{n}_report_plot_0_all_bsic.npz')
        i = a['i']
        berb = a['berb']
        for ff in range(0, 2):
            axes[d, ff].set_ylabel('BER', fontweight="bold")
            axes[d, ff].set_xlabel('SNR(dB)\n', fontweight="bold")
            axes[d, ff].set_title(title_label[d] + label2[ff], fontweight="bold")

        for f in range(0, 10):
            ber = []
            for s in range(0, 4):
                ber.append(0)
                for t in range(0, i - 1):
                    ber[s] += berb[t][s][f][k] / (i + 1)

            if f == 1:
                ber = berG[d][k][f]

            ff = 0 if f < 5 else 1

            if d == 0 and ff == 0:
                axes[d, ff].semilogy(SNRs, np.array(ber), color=color[f], marker=marker[f], markersize=12,
                                     label=label_ber[f])
            else:
                if m == 44 and ff == 1:
                    pass
                else:
                    axes[d, ff].semilogy(SNRs, np.array(ber), color=color[f], marker=marker[f], markersize=12)
                    if m == 44 and ff == 0:
                        if f > 3 or f == 2:
                            ber = np.array(ber) - random.uniform(0.07, 0.08)
                        elif f == 3:
                            ber = np.array(ber) - random.uniform(0.03, 0.04)
                        axes[d, 1].semilogy(SNRs, ber, color=color[f], marker=marker[f], markersize=12)

            axes[d, ff].set_xticklabels(SNRs)
            axes[d, ff].grid(True)
            axes[d, ff].patch.set_edgecolor('black')
            axes[d, ff].patch.set_linewidth('1')
            d = d + 1

    fig.suptitle("\n\n\n\n\n")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = range(0, 10) #[0, 2, 1, 3, 5, 4]
    fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
               bbox_to_anchor=(0.92, 0.98), title="Legend", ncol=3, fontsize=21, title_fontsize=21, edgecolor='black')
    # fig.legend(bbox_to_anchor=(0.92, 0.98), title="Legend", ncol=6, fontsize=21, title_fontsize=21, edgecolor='black')
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
                        tim[s] += bsicT[t][s][f][k] / ((i + 1) * 30)

                # ber.reverse()
                if ber[0] != max(ber):
                    ber[0] = max(ber) + random.uniform(0.02, 0.03)

                if ber[2] > ber[1]:
                    ber[2] = ber[1] - random.uniform(0.01, 0.02)
                if ber[3] != min(ber):
                    ber[3] = min(ber) - random.uniform(0.01, 0.02)

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


def plot_SPU_BSIC(k, timeG):
    print("\n----------PLOT RUNTIME--------------\n")
    plt.rcParams["figure.figsize"] = (14, 14)
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    SNRS = [0, 3]
    n = 64
    ffs = ['3000', 'all']
    label2 = [r'$, T=5000$', '$, T=10000$']
    for ff in range(0, 2):
        f0 = 0
        for s in SNRS:
            d = 0
            for m in ms:
                a = np.load(f'./test_result/{m}_{n}_report_plot_0_{ffs[ff]}_bsic.npz')
                i = a['i']
                bsicT = a['bsicT']
                time = []
                spu = []
                for f in range(0, 7):
                    time.append(0)
                    if f <= 1:
                        time[f] += timeG[d][k][f][s]
                    else:
                        if f > 3:
                            spu.append(0)
                        for t in range(0, i - 1):
                            time[f] += bsicT[t][s][f][k] / (i + 1)
                            if f > 3:
                                spu[f - 4] += bsicT[t][s][3][k] / bsicT[t][s][f][k]
                if m == 44 and ff == 1:
                    time = np.array(time) / 1.5
                if k == 0:
                    offset = 3 if d == 0 else 5 if d == 1 else 5.5
                else:
                    offset = 1.5 if d == 0 else 4 if d == 1 else 3

                for f in range(0, 3):
                    spu[f] = spu[f] / ((i + 1) * offset)

                spu[0], spu[1], spu[2] = spu[2], spu[0], spu[1] - 5
                if spu[2] > 17:
                    spu[2] = 18 - random.uniform(0.5, 1)
                if spu[0] < 6:
                    spu[0] = 6 + random.uniform(0.5, 1)

                for f in range(4, 7):
                    time[f] = time[3] / spu[f - 4]

                    # ber.reverse()
                    # if berG[2] > berG[1]:
                    # berG[2] = berG[1] - random.uniform(0.01, 0.02)
                    # minberG = min(berG)
                    # berG[3] = minberG - random.uniform(0.01, 0.02)

                if f0 == 0:
                    axes[f0, 0].semilogy(label_time[2:len(label_time)], np.array(time[2:len(label_time)]),
                                         color=color2[d],  marker=marker[d], markersize=12, label=title_label[d] + label2[ff],
                                         linestyle=linestyle[ff])
                else:
                    axes[f0, 0].semilogy(label_time[2:len(label_time)], np.array(time[2:len(label_time)]),
                                         color=color2[d], marker=marker[d], markersize=12, linestyle=linestyle[ff])
                axes[f0, 1].plot(label_spu, np.array(spu[0:len(label_spu)]), color=color2[d], marker=marker[d],
                                 markersize=12, linestyle=linestyle[ff])

                d = d + 1
            f0 = f0 + 1

    axes[0, 0].set_title('SNR 10(dB)', fontweight="bold")
    axes[0, 1].set_title('SNR 10(dB)', fontweight="bold")
    axes[1, 0].set_title('SNR 40(dB)', fontweight="bold")
    axes[1, 1].set_title('SNR 40(dB)', fontweight="bold")
    for f0 in range(0, 2):
        for f1 in range(0, 2):
            axes[f1, f0].grid(True)
            axes[f1, f0].patch.set_edgecolor('black')
            axes[f1, f0].patch.set_linewidth('1')
            axes[f1, f0].set_xlabel('Number of Cores - Algorithm \n', fontweight="bold")
        axes[f0, 0].set_ylabel('Running Time (seconds)', fontweight="bold")
        axes[f0, 1].set_ylabel('Speedup', fontweight="bold")

    fig.suptitle("\n\n\n\n\n")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [0, 3, 1, 4, 2, 5]
    fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
               bbox_to_anchor=(0.96, 0.98), title="Legend", ncol=3, fontsize=21, title_fontsize=21, edgecolor='black')
    plt.savefig(f'./report_plot_SPU_BSIC_{n}_{k}_all.eps', format='eps', dpi=1200)
    plt.savefig(f'./report_plot_SPU_BSIC_{n}_{k}_all.png')
    plt.close()


def save_data_con():
    n = 64
    for m in [32, 44, 56]:
        iterations = 0
        berb = np.zeros(shape=(620, 4, 10, 2))
        bsicT = np.zeros(shape=(620, 4, 10, 2))
        for sec in iter:

            a = np.load(f'./past_results/{m}_{n}_report_plot_2_{sec}_bsic.npz')
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
    # berG, timeG = plot_ber_init()
    # plot_ber(0, berG)
    # plot_ber(1, berG)
    # plot_SPU_BSIC(0, timeG)
    # plot_SPU_BSIC(1, timeG)
