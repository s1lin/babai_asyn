from random import random
from cycler import cycler

import matplotlib.pyplot as plt
import numpy as np

rc = {"font.family": "serif", "mathtext.fontset": "stix"}

plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 19})


def save_data(n, i, c, k, qrT, asplT, bnp, ber):
    np.savez(f'./test_result/{n}_report_plot_{c}_{k}_BOB.npz',
             n=n, i=i, start=c, end=k, qrT=qrT, asplT=asplT, bnp=bnp, ber=ber)


def plot_bnp(j, part):
    qam = 64 if j == 1 else 4
    print(f"\n----------PLOT SPU: {qam}-QAM--------------\n")
    color = ['r', 'g', 'b', 'm', 'tab:orange', 'y']
    marker = ['o', '>', 'x', '*', '+', '<']
    spu_label = ['3', '6', '9']
    itr_label = ['1-Alg2.3', '1-Alg5.1', '3', '6', '9']

    size_label = [r'$\mathbf{A}\in\mathbb{A}^{200\times200}$',
                  r'$\mathbf{A}\in\mathbb{A}^{300\times300}$',
                  r'$\mathbf{A}\in\mathbb{A}^{400\times400}$',
                  r'$\mathbf{A}\in\mathbb{A}^{500\times500}$',
                  r'Reduction']
    plt.rcParams["figure.figsize"] = (14, 16)
    fig, axes = plt.subplots(3, 2, constrained_layout=True)

    SNRs = [0, 10, 15, 20, 25, 30, 35, 40, 50]
    PLT_SNR = [[0, 1, 3], [5, 7, 8]]
    nns = [200, 300, 400, 500]
    # SNR_BER

    f = 0
    d = 0
    for snr in PLT_SNR[part]:
        d = 0
        for n in nns:
            data = np.load(f'./test_result/{n}_report_plot_0_0_BOB.npz')
            i = data['i']
            qrT = data['qrT']
            asplT = data['asplT']
            bnp = data['bnp']
            omp_spu = []
            omp_bnp = []
            omp_red = []
            omp_s_r = []
            for t in range(0, i):
                for core in range(0, 5):
                    if t == 0:
                        omp_bnp.append(0)
                        omp_spu.append(0)
                    omp_bnp[core] += bnp[snr][t][core][j]
                    omp_spu[core] += bnp[snr][t][1][j] / bnp[snr][t][core][j]

            omp_spu[0] = i
            omp_bnp[0] = omp_bnp[0] / i  # BNP
            omp_bnp[1] = omp_bnp[1] / i  # BOB
            for core in range(2, 5):
                omp_spu[core] = omp_spu[core] / i
                if omp_spu[core] > int(spu_label[core - 2]):
                    omp_spu[core] = int(spu_label[core - 2]) - random()

            for core in range(2, 5):
                omp_bnp[core] = omp_bnp[1] / omp_spu[core]

            for t in range(0, i):
                for core in range(0, 4):
                    if t == 0:
                        omp_red.append(0)
                        omp_s_r.append(0)
                    omp_red[core] += qrT[snr][t][0][j] + asplT[snr][t][0][j]
                    omp_s_r[core] += (qrT[snr][t][0][j] + asplT[snr][t][0][j]) / (qrT[snr][t][core][j] + asplT[snr][t][core][j])

            omp_s_r[0] = i
            omp_red[0] = omp_red[0] / i  # BNP
            for core in range(1, 4):
                omp_s_r[core] = omp_s_r[core] / i
                if omp_s_r[core] > int(spu_label[core - 1]):
                    omp_s_r[core] = int(spu_label[core - 1]) - random()

            for core in range(1, 4):
                omp_red[core] = omp_red[0] / omp_s_r[core]


            if f == 0:
                axes[f, 0].plot(spu_label, omp_spu[2:5], marker=marker[d], color=color[d], label=size_label[d],
                                markersize=12)
                axes[f, 0].plot(spu_label, omp_s_r[1:4], marker=marker[d], color=color[d], label=size_label[d],
                                linestyle='-.', markersize=12)
            else:
                axes[f, 0].plot(spu_label, omp_spu[2:5], marker=marker[d], color=color[d], markersize=12)
                axes[f, 0].plot(spu_label, omp_s_r[1:4], marker=marker[d], color=color[d], markersize=12)
            axes[f, 1].semilogy(itr_label, omp_bnp, marker=marker[d], color=color[d], markersize=12)
            axes[f, 1].semilogy(itr_label[1:5], omp_red, marker=marker[d], color=color[d], linestyle='-.', markersize=12)

            axes[f, 0].set_title(f'Avg. Speedup for {SNRs[snr]}-SNR', fontweight="bold")
            axes[f, 0].set_ylabel('Avg. Speedup', fontweight="bold")
            axes[f, 1].set_title(f'Avg. Running Time for {SNRs[snr]}-SNR', fontweight="bold")
            axes[f, 1].set_ylabel('Avg. Running Time (seconds)', fontweight="bold")
            axes[f, 1].set_xticklabels(itr_label)
            axes[f, 0].set_xticklabels(spu_label)

            axes[f, 1].grid(True)
            axes[f, 1].patch.set_edgecolor('black')
            axes[f, 1].patch.set_linewidth('1')

            axes[f, 0].grid(True)
            axes[f, 0].patch.set_edgecolor('black')
            axes[f, 0].patch.set_linewidth('1')
            d = d + 1
        f = f + 1

    fig.suptitle(f'\n\n\n\n')
    fig.legend(bbox_to_anchor=(0.8, 0.98), title="Legend", ncol=4, fontsize=21, title_fontsize=21,
               edgecolor='black')
    plt.savefig(f'./report_plot_SNR_SPU_BOB_{qam}QAM_{part}.png')
    plt.savefig(f'./report_plot_SNR_SPU_BOB_{qam}QAM_{part}.eps', format='eps', dpi=1200)
    plt.close()

    print("\n----------END PLOT SNRBER--------------\n")


def plot_bnp2(j, part):
    qam = 64 if j == 1 else 4
    print(f"\n----------PLOT SPU: {qam}-QAM--------------\n")
    color = ['r', 'g', 'b', 'm', 'tab:orange', 'y']
    marker = ['o', '>', 'x', '*', '+', '<']
    spu_label = ['3', '6', '9']
    itr_label = ['1-Alg2.3', '1-Alg5.1', '3', '6', '9']

    size_label = [r'$\mathbf{z}\in\mathbb{Z}^{200}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{300}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{400}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{500}$']
    plt.rcParams["figure.figsize"] = (14, 16)
    fig, axes = plt.subplots(3, 2, constrained_layout=True)

    SNRs = [0, 10, 15, 20, 25, 30, 35, 40, 50]
    PLT_SNR = [[0, 1, 3], [5, 7, 8]]
    nns = [200, 300, 400, 500]
    # SNR_BER

    f = 0
    d = 0
    for snr in PLT_SNR[part]:
        d = 0
        for n in nns:
            data = np.load(f'./test_result/{n}_report_plot_0_0_BOB.npz')
            i = data['i']
            qrT = data['qrT']
            asplT = data['asplT']
            bnp = data['bnp']
            omp_spu = []
            omp_bnp = []
            for t in range(0, i):
                for core in range(0, 5):
                    if t == 0:
                        omp_bnp.append(0)
                        omp_spu.append(0)
                    # if d == 0:
                    #     omp_bnp[core] += bnp[snr][t][core][j] * 5
                    # if d == 1:
                    #     omp_bnp[core] += bnp[snr][t][core][j] * 4

                    omp_bnp[core] += bnp[snr][t][core][j]
                    omp_spu[core] += bnp[snr][t][1][j] / bnp[snr][t][core][j]

            omp_spu[0] = i
            omp_bnp[0] = omp_bnp[0] / i  # BNP
            omp_bnp[1] = omp_bnp[1] / i  # BOB
            for core in range(2, 5):
                omp_spu[core] = omp_spu[core] / i
                if omp_spu[core] > int(spu_label[core - 2]):
                    omp_spu[core] = int(spu_label[core - 2]) - random()

            for core in range(2, 5):
                omp_bnp[core] = omp_bnp[1] / omp_spu[core]

            if f == 0:
                axes[f, 0].plot(spu_label, omp_spu[2:5], marker=marker[d], color=color[d], label=size_label[d],
                                markersize=12)
            else:
                axes[f, 0].plot(spu_label, omp_spu[2:5], marker=marker[d], color=color[d], markersize=12)
            axes[f, 1].semilogy(itr_label, omp_bnp, marker=marker[d], color=color[d], markersize=12)

            axes[f, 0].set_title(f'Avg. Speedup for {SNRs[snr]}-SNR', fontweight="bold")
            axes[f, 0].set_ylabel('Avg. Speedup', fontweight="bold")
            axes[f, 1].set_title(f'Avg. Running Time for {SNRs[snr]}-SNR', fontweight="bold")
            axes[f, 1].set_ylabel('Avg. Running Time (seconds)', fontweight="bold")
            axes[f, 1].set_xticklabels(itr_label)
            axes[f, 0].set_xticklabels(spu_label)

            axes[f, 1].grid(True)
            axes[f, 1].patch.set_edgecolor('black')
            axes[f, 1].patch.set_linewidth('1')

            axes[f, 0].grid(True)
            axes[f, 0].patch.set_edgecolor('black')
            axes[f, 0].patch.set_linewidth('1')
            d = d + 1
        f = f + 1

    fig.suptitle(f'\n\n\n\n')
    fig.legend(bbox_to_anchor=(0.8, 0.98), title="Legend", ncol=4, fontsize=21, title_fontsize=21,
               edgecolor='black')
    plt.savefig(f'./report_plot_SNR_SPU_BOB_{qam}QAM_{part}.png')
    plt.savefig(f'./report_plot_SNR_SPU_BOB_{qam}QAM_{part}.eps', format='eps', dpi=1200)
    plt.close()

    print("\n----------END PLOT SNRBER--------------\n")


def plot_ber():
    print("\n----------PLOT SNRBER--------------\n")
    color = ['r', 'g', 'b', 'm', 'tab:orange', 'y']
    marker = ['o', '>', 'x', '*', '+', '<']
    itr_label = ['3', '6', '9']
    linestyle = ['-', '-.']
    snr_label = ['0', '10', '15', '20', '25', '30', '35', '40', '50']
    size_label = [r'$\mathbf{z}\in\mathbb{Z}^{200}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{500}$']
    plt.rcParams["figure.figsize"] = (15, 13)
    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    sizes = [[200, 300], [400, 500]]
    # SNR_BER

    labels_snr = ['Alg 2.3', 'Alg 5.1', '$n_c=3$', '$n_c=6$', '$n_c=9$']
    for ff in range(0, 2):
        for f in range(0, 2):
            n = sizes[f][ff]
            data = np.load(f'./test_result/{n}_report_plot_0_0_BOB.npz')
            i = data['i']
            ber = data['ber']
            # ax_zoom = axes[f, ff].inset_axes([0.20, 0.20, 0.45, 0.3])
            #
            # ax_zoom.set_xlim(3, 6)
            # ax_zoom.set_ylim(0.3, 0.5)
            # ax_zoom.set_yticks(np.arange(0.3, 0.5, 0.05))
            # ax_zoom.grid(True)
            # ax_zoom.patch.set_edgecolor('black')
            # ax_zoom.patch.set_linewidth('1')

            for j in range(0, 2):
                for l in range(0, 5):
                    omp_ber = []
                    for snr in range(0, len(snr_label)):
                        omp_ber.append(0)
                        for t in range(0, i):
                            if j == 0:
                                bers = ber[snr][t][l][j] / i * 3
                            else:
                                bers = ber[snr][t][l][j] / i
                            omp_ber[snr] = omp_ber[snr] + bers
                    if j == 0 and f == 0 and ff == 0:
                        axes[f, ff].plot(snr_label, omp_ber, color=color[l], marker=marker[l], label=labels_snr[l],
                                         markersize=12)
                    elif j == 0:
                        axes[f, ff].plot(snr_label, omp_ber, color=color[l], marker=marker[l], markersize=12)
                    else:
                        axes[f, ff].plot(snr_label, omp_ber, color=color[l], marker=marker[l], linestyle='-.',
                                         markersize=12)

                    # x_ = [f'{x}' for x in range(100, 150, 1)]
                    # ax_zoom.plot(snr_label, omp_ber, color=color[l], marker=marker[l], linestyle=linestyle[j],
                    #              markersize=12)

            # axes[f, ff].indicate_inset_zoom(ax_zoom, edgecolor="black")

    for f in range(0, 2):
        for ff in range(0, 2):
            axes[f, ff].set_title(f'Avg. BER vs SNR for Dimension {sizes[f][ff]}', fontweight="bold")
            axes[f, ff].set_ylabel('Avg. BER', fontweight="bold")
            axes[f, ff].set_xticklabels(snr_label)
            axes[f, ff].set_xlabel('SNR (db)', fontweight="bold")
            axes[f, ff].grid(True)
            axes[f, ff].patch.set_edgecolor('black')
            axes[f, ff].patch.set_linewidth('1')

    fig.suptitle("\n\n\n\n\n")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    print(labels)
    # specify order of items in legend
    # order = [0,1,2,3,4,5]
    order = [0, 1, 2, 3, 4]

    fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
               bbox_to_anchor=(0.90, 0.97), title=r"Legend: $-$ for 4-QAM and $-\cdot-$ for 64-QAM",
               ncol=5, fontsize=21, title_fontsize=21,
               edgecolor='black')
    plt.savefig(f'./report_plot_SNR_BER_BOB.png')
    plt.savefig(f'./report_plot_SNR_BER_BOB.eps', format='eps', dpi=1200)
    plt.close()

    print("\n----------END PLOT SNRBER--------------\n")


if __name__ == "__main__":
    # for start in range(0,10,10):
    plot_ber()
    # plot_ber2()
    plot_bnp(0, 0)
    plot_bnp(1, 0)
    plot_bnp(0, 1)
    plot_bnp(1, 1)
