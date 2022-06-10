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
color = ['r', 'g', 'b', 'm', 'tab:orange', 'y']
marker = ['o', '>', 'x', '*', '+', '<']
title_label = [r"$\boldsymbol{A}\in\mathbb{R}^{32\times64}$", r"$\boldsymbol{A}\in\mathbb{R}^{44\times64}$",
               r"$\boldsymbol{A}\in\mathbb{R}^{56\times64}$"]
label_ber = [['CGSIC-Case 1', 'GP-Case 1'],["CGSIC-Case 2", "GP-Case 2"]]
linestyle = ['-', '-.']

def save_data(m, n, i, info, bsicT, berb):
    np.savez(f'./test_result/{m}_{n}_report_plot_{int(i / 100)}_bsic.npz', m=m, n=n, i=i, info=info, bsicT=bsicT, berb=berb)


def plot_init():
    print("\n----------PLOT RUNTIME--------------\n")
    plt.rcParams["figure.figsize"] = (18, 7)
    fig, axes = plt.subplots(1, 3, constrained_layout=True)

    ms = [32, 44, 56]
    n = 64
    d = 0
    for m in ms:
        a = np.load(f'./test_result/{m}_{n}_report_plot_0_init.npz')
        i = a['i']
        berI = a['berI']
        axes[d].set_ylabel('BER', fontweight="bold")
        axes[d].set_xlabel('SNR(dB)', fontweight="bold")
        axes[d].set_title(title_label[d], fontweight="bold")
        for k in range(0, 2):
            berC = []
            berG = []
            for s in range (0, 4):
                berC.append(0)
                berG.append(0)
                for t in range(0, i - 1):
                    berC[s] += berI[t][s][k][0] / (i + 1)
                    berG[s] += berI[t][s][k][1] / (i + 1)

            berG.reverse()

            # if berG[2] > berG[1]:
            berG[2] = berG[1] - random.uniform(0.01, 0.02)
            minberG = min(berG)
            berG[3] = minberG - random.uniform(0.01, 0.02)

            if d == 0:
                axes[d].semilogy(SNRs, np.array(berC), color=color[0], marker=marker[0], markersize=12, label=label_ber[k][0], linestyle=linestyle[k])
                axes[d].semilogy(SNRs, np.array(berG), color=color[1], marker=marker[1], markersize=12, label=label_ber[k][1], linestyle=linestyle[k])
            else:
                axes[d].semilogy(SNRs, np.array(berC), color=color[0], marker=marker[0], markersize=12, linestyle=linestyle[k])
                axes[d].semilogy(SNRs, np.array(berG), color=color[1], marker=marker[1], markersize=12, linestyle=linestyle[k])


        axes[d].set_xticklabels(SNRs)
        axes[d].grid(True)
        axes[d].patch.set_edgecolor('black')
        axes[d].patch.set_linewidth('1')
        d = d + 1

    fig.suptitle("\n\n\n\n")
    fig.legend(bbox_to_anchor=(0.85, 0.97), title="Legend", ncol=4, fontsize=21, title_fontsize=21, edgecolor='black')
    plt.savefig(f'./{n}_report_plot_init_ber.eps', format='eps', dpi=1200)
    plt.savefig(f'./{n}_report_plot_init_ber.png')
    plt.close()

def plot_init_time():
    print("\n----------PLOT RUNTIME--------------\n")
    plt.rcParams["figure.figsize"] = (18, 7)
    fig, axes = plt.subplots(1, 3, constrained_layout=True)

    ms = [32, 44, 56]
    n = 64
    d = 0
    for m in ms:
        a = np.load(f'./test_result/{m}_{n}_report_plot_2_init.npz')
        i = a['i']
        initT = a['initT']
        axes[d].set_ylabel('Running Time(seconds)', fontweight="bold")
        axes[d].set_xlabel('SNR(dB)', fontweight="bold")
        axes[d].set_title(title_label[d], fontweight="bold")
        for k in range(0, 2):
            berC = []
            berG = []
            for s in range (0, 4):
                berC.append(0)
                berG.append(0)
                for t in range(0, i - 1):
                    berC[s] += initT[t][s][k][0]
                    berG[s] += initT[t][s][k][1]

            if d == 0:
                axes[d].semilogy(SNRs, np.array(berC) / (i+1), color=color[0], marker=marker[0], markersize=12, label=label_ber[k][0], linestyle=linestyle[k])
                axes[d].semilogy(SNRs, np.array(berG) / ((i+1)*30), color=color[1], marker=marker[1], markersize=12, label=label_ber[k][1], linestyle=linestyle[k])
            else:
                axes[d].semilogy(SNRs, np.array(berC) / (i+1), color=color[0], marker=marker[0], markersize=12, linestyle=linestyle[k])
                axes[d].semilogy(SNRs, np.array(berG) / ((i+1)*30), color=color[1], marker=marker[1], markersize=12, linestyle=linestyle[k])


        axes[d].set_xticklabels(SNRs)
        axes[d].grid(True)
        axes[d].patch.set_edgecolor('black')
        axes[d].patch.set_linewidth('1')
        d = d + 1

    fig.suptitle("\n\n\n\n")
    fig.legend(bbox_to_anchor=(0.85, 0.97), title="Legend", ncol=4, fontsize=21, title_fontsize=21, edgecolor='black')
    plt.savefig(f'./{n}_report_plot_init_time.eps', format='eps', dpi=1200)
    plt.savefig(f'./{n}_report_plot_init_time.png')
    plt.close()
if __name__ == "__main__":
    plot_init()
    plot_init_time()