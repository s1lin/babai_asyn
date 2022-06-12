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
color = ['r', 'g', 'b', 'm', 'tab:orange', 'y', 'r']
marker = ['o', '>', 'x', '*', '+', '<', 'o']
title_label = [r"$\boldsymbol{A}\in\mathbb{R}^{32\times64}$",
               r"$\boldsymbol{A}\in\mathbb{R}^{44\times64}$",
               r"$\boldsymbol{A}\in\mathbb{R}^{56\times64}$"]
label_ber = ['CGSIC(1)', 'GP(1)', 'BSIC-BB(1)', 'BSIC-BBB(1)', 'PBSIC-PBBB(5-2)', 'PBSIC-PBBB(10-2)', 'PBSIC-PBBB(5-4)']
label_time = ['1\nCGSIC', '1\nGP', '1\nBSIC\nBB', '1\nBSIC\nBBB', '5-2\nPBSIC\nPBBB', '10-2\nPBSIC\nPBBB', '5-4\nPBSIC\nPBBB']
label_spu = ['5-2\nPBSIC\nPBBB', '10-2\nPBSIC\nPBBB', '5-4\nPBSIC\nPBBB']
linestyle = ['-', '-.']


def save_data(m, n, i, info, bsicT, berb):
    np.savez(f'./test_result/{m}_{n}_report_plot_{int(i / 100)}_{info}_bsic.npz', m=m, n=n, i=i, info=info, bsicT=bsicT,
             berb=berb)


def plot_ber(iter):
    print("\n----------PLOT RUNTIME--------------\n")
    plt.rcParams["figure.figsize"] = (14, 18)
    fig, axes = plt.subplots(3, 2, constrained_layout=True)

    ms = [32, 44, 56]
    n = 64
    d = 0
    for m in ms:
        a = np.load(f'./past_results/{m}_{n}_report_plot_0_{iter}_bsic.npz')
        i = a['i']
        berb = a['berb']
        if iter == 10000:
            print(m, i)

        for k in range(0, 2):
            axes[d, k].set_ylabel('BER', fontweight="bold")
            axes[d, k].set_xlabel('SNR(dB)\n', fontweight="bold")
            axes[d, k].set_title(title_label[d] + f' for Case {k + 1}', fontweight="bold")
            for f in range(1, 7):
                ber = []
                for s in range(0, 4):
                    ber.append(0)
                    for t in range(0, i - 1):
                        ber[s] += berb[t][s][f][k] / (i + 1)

                # ber.reverse()

                # if berG[2] > berG[1]:
                # berG[2] = berG[1] - random.uniform(0.01, 0.02)
                # minberG = min(berG)
                # berG[3] = minberG - random.uniform(0.01, 0.02)

                if d == 0 and k == 0:
                    axes[d, k].semilogy(SNRs, np.array(ber), color=color[f], marker=marker[f], markersize=12, label=label_ber[f])
                else:
                    axes[d, k].semilogy(SNRs, np.array(ber), color=color[f], marker=marker[f], markersize=12)

            axes[d, k].set_xticklabels(SNRs)
            axes[d, k].grid(True)
            axes[d, k].patch.set_edgecolor('black')
            axes[d, k].patch.set_linewidth('1')
        d = d + 1

    fig.suptitle("\n\n\n\n\n")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [0, 3, 1, 4, 2, 5]
    fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
        bbox_to_anchor=(0.92, 0.98), title="Legend", ncol=3, fontsize=21, title_fontsize=21, edgecolor='black')
    plt.savefig(f'./report_plot_bsic_ber_{n}_{iter}.eps', format='eps', dpi=1200)
    plt.savefig(f'./report_plot_bsic_ber_{n}_{iter}.png')
    plt.close()


def plot_bsic_time(k, iter):
    print("\n----------PLOT RUNTIME--------------\n")
    plt.rcParams["figure.figsize"] = (14, 14)
    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    ms = [32, 44, 56]
    SNRS = [0, 3]
    n = 64
    f0 = 0
    for s in SNRS:
        d = 0
        for m in ms:
            a = np.load(f'./past_results/{m}_{n}_report_plot_0_{iter}_bsic.npz')
            i = a['i']
            bsicT = a['bsicT']
            time = []
            spu = []
            for f in range(0, 7):
                time.append(0)
                if f > 3:
                    spu.append(0)
                for t in range(0, i - 1):
                    time[f] += bsicT[t][s][f][k] / (i + 1)
                    if f > 3:
                        spu[f - 4] += bsicT[t][s][3][k] / bsicT[t][s][f][k]
            for f in range(0, 3):
                spu[f] = spu[f] / (i + 1)
                # ber.reverse()
                # if berG[2] > berG[1]:
                # berG[2] = berG[1] - random.uniform(0.01, 0.02)
                # minberG = min(berG)
                # berG[3] = minberG - random.uniform(0.01, 0.02)

            if f0 == 0:
                axes[f0, 0].semilogy(label_time[1:len(label_time)], np.array(time[1:len(label_time)]), color=color[d], marker=marker[d], markersize=12, label=title_label[d])
            else:
                axes[f0, 0].semilogy(label_time[1:len(label_time)], np.array(time[1:len(label_time)]), color=color[d], marker=marker[d], markersize=12)
            axes[f0, 1].semilogy(label_spu, np.array(spu[0:len(label_spu)]), color=color[d], marker=marker[d], markersize=12)

            d = d + 1
        f0 = f0 + 1

    axes[0, 0].set_title('SNR-10', fontweight="bold")
    axes[0, 1].set_title('SNR-10', fontweight="bold")
    axes[1, 0].set_title('SNR-40', fontweight="bold")
    axes[1, 1].set_title('SNR-40', fontweight="bold")
    for f0 in range(0, 2):
        for f1 in range(0, 2):
            axes[f1, f0].grid(True)
            axes[f1, f0].patch.set_edgecolor('black')
            axes[f1, f0].patch.set_linewidth('1')
            axes[f1, f0].set_xlabel('Number of Cores - Algorithm \n', fontweight="bold")
        axes[f0, 0].set_ylabel('Running Time (seconds)', fontweight="bold")
        axes[f0, 1].set_ylabel('Speedup', fontweight="bold")

    fig.suptitle("\n\n\n\n")
    fig.legend(bbox_to_anchor=(0.85, 0.97), title="Legend", ncol=4, fontsize=21, title_fontsize=21, edgecolor='black')
    plt.savefig(f'./report_plot_bsic_time_{n}_{k}_{iter}.eps', format='eps', dpi=1200)
    plt.savefig(f'./report_plot_bsic_time_{n}_{k}_{iter}.png')
    plt.close()


if __name__ == "__main__":
    plot_ber(3000)
    plot_ber(10000)
    plot_bsic_time(0, 3000)
    plot_bsic_time(0, 10000)
    plot_bsic_time(1, 3000)
    plot_bsic_time(1, 10000)
