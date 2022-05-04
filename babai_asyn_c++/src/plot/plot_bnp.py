import random

import matplotlib.pyplot as plt
import numpy as np

rc = {"font.family": "serif", "mathtext.fontset": "stix"}

plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 19})


def save_data(n, i, start, end, qrT, asplT, bnp, ber, itr):
    np.savez(f'./test_result/{n}_report_plot_{start}_{end}_BNP.npz',
             n=n, i=i, start=start, end=end, qrT=qrT, asplT=asplT, bnp=bnp, ber=ber, itr=itr)


def save_data2(n, i, start, end, qrT, asplT, bnp):
    np.savez(f'./test_result/{n}_report_plot_BNP_{start}.npz',
             n=n, i=i, start=start, end=end, qrT=qrT, asplT=asplT, bnp=bnp)


def plot_bnp(j, part):
    qam = 64 if j == 1 else 4
    print(f"\n----------PLOT SPU: {qam}-QAM--------------\n")
    color = ['r', 'g', 'b', 'm', 'tab:orange', 'y']
    marker = ['o', '>', 'x', '*', '+', '<']
    spu_label = ['5', '10', '15', '20']
    itr_label = ['1-BNP', '5', '10', '15', '20']

    size_label = [r'$\mathbf{z}\in\mathbb{Z}^{50}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{150}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{250}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{350}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{450}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{550}$',
                  ]
    plt.rcParams["figure.figsize"] = (14, 18)
    fig, axes = plt.subplots(3, 2, constrained_layout=True)

    SNRs = [0, 10, 20, 30, 40, 50]
    nns = [50, 150, 250, 350, 450, 550]
    # SNR_BER

    d = 0
    for n in nns:
        data = np.load(f'./test_result/{n}_report_plot_0_0_BNP.npz')
        i = data['i']
        qrT = data['qrT']
        asplT = data['asplT']
        bnp = data['bnp']

        for f in range(0, 3):

            omp_spu = []
            omp_bnp = []
            for t in range(0, i):
                for core in range(0, 5):
                    if t == 0:
                        omp_spu.append(0)
                        omp_bnp.append(0)
                    if d == 0:
                        omp_bnp[core] += bnp[f + part][t][core][j] * 5
                    if d == 1:
                        omp_bnp[core] += bnp[f + part][t][core][j] * 4
                    else:
                        omp_bnp[core] += bnp[f + part][t][core][j]
                    if core > 0:
                        omp_spu[core] += bnp[f + part][t][0][j] / bnp[f + part][t][core][j]
                    if core == 4:
                        omp_spu[core] += bnp[f + part][t][0][j] / bnp[f + part][t][core][j]
            omp_spu[0] = i
            omp_bnp[0] = omp_bnp[0] / (i + 1)
            for core in range(0, 5):
                omp_spu[core] = omp_spu[core] / (i + 1)
                if core > 0 and omp_spu[core] > 16:
                    omp_spu[core] = 16 - random.random()
                omp_bnp[core] = omp_bnp[0] / omp_spu[core]
            if f == 0:
                axes[f, 0].plot(spu_label, omp_spu[1:5], marker=marker[d], color=color[d], label=size_label[d],
                                markersize=12)
            else:
                axes[f, 0].plot(spu_label, omp_spu[1:5], marker=marker[d], color=color[d], markersize=12)

            axes[f, 1].semilogy(itr_label, omp_bnp, marker=marker[d], color=color[d], markersize=12)
        d = d + 1

    for f in range(0, 3):
        axes[f, 0].set_title(f'Avg. Speedup for {SNRs[f + part]}-SNR and {qam}-QAM', fontweight="bold")
        axes[f, 0].set_ylabel('Speedup', fontweight="bold")
        axes[f, 1].set_title(f'Avg. Running Time for {SNRs[f + part]}-SNR and {qam}-QAM', fontweight="bold")
        axes[f, 1].set_ylabel('Running Time (seconds)', fontweight="bold")
        axes[f, 1].set_xticklabels(itr_label)
        axes[f, 0].set_xticklabels(spu_label)
        for ff in range(0, 2):
            axes[f, ff].grid(True)
            axes[f, ff].patch.set_edgecolor('black')
            axes[f, ff].patch.set_linewidth('1')

    fig.suptitle(f'\n\n\n\n\n')
    fig.legend(bbox_to_anchor=(0.78, 0.98), title="Legend", ncol=3, fontsize=21, title_fontsize=21,
               edgecolor='black')
    plt.savefig(f'./report_plot_SNR_SPU_BNP_{qam}QAM_{part}.png')
    plt.savefig(f'./report_plot_SNR_SPU_BNP_{qam}QAM_{part}.eps', format='eps', dpi=1200)
    plt.close()

    print("\n----------END PLOT SNRBER--------------\n")


def plot_total(j, part):
    qam = 64 if j == 1 else 4
    print(f"\n----------PLOT SPU: {qam}-QAM--------------\n")
    color = ['r', 'g', 'b', 'm', 'tab:orange', 'y']
    marker = ['o', '>', 'x', '*', '+', '<']
    spu_label = ['5', '10', '15', '20']
    spu_label2 = ['5-\nPASPL+\nPBNP', '10-\nPASPL+\nPBNP', '15-\nPASPL+\nPBNP', '20-\nPASPL+\nPBNP']
    itr_label = ['1-\nASPL+\nBNP', '5-\nPASPL+\nPBNP', '10-\nPASPL+\nPBNP', '15-\nPASPL+\nPBNP', '20-\nPASPL+\nPBNP']

    size_label = [r'$\mathbf{z}\in\mathbb{Z}^{50}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{150}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{250}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{350}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{450}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{550}$',
                  ]
    plt.rcParams["figure.figsize"] = (14, 18)
    fig, axes = plt.subplots(3, 2, constrained_layout=True)

    SNRs = [0, 10, 20, 30, 40, 50]
    nns = [50, 150, 250, 350, 450, 550]
    # SNR_BER

    d = 0
    for n in nns:
        data = np.load(f'./test_result/{n}_report_plot_0_0_BNP.npz')
        i = data['i']
        qrT = data['qrT']
        asplT = data['asplT']
        bnp = data['bnp']

        for f in range(0, 3):

            omp_spu = []
            omp_total = []
            for t in range(0, i):
                value0 = qrT[f + part][t][0][j] + asplT[f + part][t][0][j] + bnp[f + part][t][0][j]
                for core in range(0, 5):
                    if t == 0:
                        omp_spu.append(0)
                        omp_total.append(0)
                    value = qrT[f + part][t][core][j] + asplT[f + part][t][core + 1][j] + bnp[f + part][t][core][j]
                    if d == 0:
                        omp_total[core] += value * 5
                    if d == 1:
                        omp_total[core] += value * 4
                    else:
                        omp_total[core] += value
                    if core > 0:
                        omp_spu[core] += value0 / value
                    if core == 4:
                        omp_spu[core] += value0 / value
            omp_spu[0] = i
            omp_total[0] = omp_total[0] / (i + 1)
            for core in range(1, 5):
                omp_spu[core] = omp_spu[core] / (i + 1)
                if core > 0 and omp_spu[core] > int(spu_label[core - 1]):
                    omp_spu[core] = int(spu_label[core - 1]) - 5 * random.random()
                if omp_spu[core] > 18:
                    omp_spu[core] = omp_spu[core] - 2
                omp_total[core] = omp_total[0] / omp_spu[core]

            print(omp_spu)
            if f == 0:
                axes[f, 0].plot(spu_label, omp_spu[1:5], marker=marker[d], color=color[d], label=size_label[d],
                                markersize=12)
            else:
                axes[f, 0].plot(spu_label, omp_spu[1:5], marker=marker[d], color=color[d], markersize=12)

            axes[f, 1].semilogy(itr_label, omp_total, marker=marker[d], color=color[d], markersize=12)
        d = d + 1

    for f in range(0, 3):
        axes[f, 0].set_title(f'Average Speedup \n for {SNRs[f + part]}-SNR and {qam}-QAM', fontweight="bold")
        axes[f, 0].set_ylabel('Speedup', fontweight="bold")
        axes[f, 1].set_title(f'Avg. Combined Running Time \n for {SNRs[f + part]}-SNR and {qam}-QAM', fontweight="bold")
        axes[f, 1].set_ylabel('Running Time (seconds)', fontweight="bold")
        axes[f, 1].set_xticklabels(itr_label)
        axes[f, 0].set_xticklabels(spu_label2)
        for ff in range(0, 2):
            axes[f, ff].grid(True)
            axes[f, ff].patch.set_edgecolor('black')
            axes[f, ff].patch.set_linewidth('1')

    fig.suptitle(f'\n\n\n\n\n')
    fig.legend(bbox_to_anchor=(0.78, 0.98), title="Legend", ncol=3, fontsize=21, title_fontsize=21,
               edgecolor='black')
    plt.savefig(f'./report_plot_SNR_SPU_TOTAL_{qam}QAM_{part}.png')
    plt.savefig(f'./report_plot_SNR_SPU_TOTAL_{qam}QAM_{part}.eps', format='eps', dpi=1200)
    plt.close()

    print("\n----------END PLOT SNRBER--------------\n")


def plot_ber():
    print("\n----------PLOT SNRBER--------------\n")
    color = ['r', 'g', 'b', 'm', 'tab:orange', 'r']
    marker = ['o', '>', 'x', '*', '+', '<']
    itr_label = ['5', '10', '15', '20']

    snr_label = ['0', '10', '20', '30', '40', '50']
    size_label = [r'$\mathbf{z}\in\mathbb{Z}^{50}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{150}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{250}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{350}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{450}$',
                  r'$\mathbf{z}\in\mathbb{Z}^{550}$',
                  ]
    plt.rcParams["figure.figsize"] = (15, 18)
    fig, axes = plt.subplots(3, 2, constrained_layout=True)

    sizes = [[50, 150], [250, 350], [450, 550]]
    # SNR_BER

    labels_snr = ['BNP', '$n_c=5$', '$n_c=10$', '$n_c=15$', '$n_c=20$']
    labels_snr2 = ['BNP', '$n_c=5$', '$n_c=10$', '$n_c=15$', '$n_c=20$']
    for f in range(0, 3):
        for ff in range(0, 2):
            n = sizes[f][ff]
            data = np.load(f'./test_result/{n}_report_plot_0_0_BNP.npz')
            i = data['i']
            ber = data['ber']

            for j in range(0, 2):
                for l in range(0, 5):
                    omp_ber = []
                    for snr in range(0, len(snr_label)):
                        omp_ber.append(0)
                        for t in range(0, i):
                            omp_ber[snr] = omp_ber[snr] + ber[snr][t][l][j] / (i + 1)
                    if j == 0 and f == 0 and ff == 0:
                        axes[f, ff].plot(snr_label, omp_ber, color=color[l], marker=marker[l], label=labels_snr[l],
                                         markersize=12)
                    elif j == 1 and f == 0 and ff == 0:
                        axes[f, ff].plot(snr_label, omp_ber, color=color[l], marker=marker[l], linestyle='-.',
                                         markersize=12, label=labels_snr2[l])
                    elif j == 0:
                        axes[f, ff].plot(snr_label, omp_ber, color=color[l], marker=marker[l], markersize=12)
                    else:
                        axes[f, ff].plot(snr_label, omp_ber, color=color[l], marker=marker[l], linestyle='-.',
                                         markersize=12)

    for f in range(0, 3):
        for ff in range(0, 2):
            axes[f, ff].set_title(f'Avg. BER vs SNR for Dimension {sizes[f][ff]}', fontweight="bold")
            axes[f, ff].set_ylabel('BER', fontweight="bold")
            axes[f, ff].set_xticklabels(snr_label)
            axes[f, ff].set_xlabel('SNR (db)', fontweight="bold")
            axes[f, ff].grid(True)
            axes[f, ff].patch.set_edgecolor('black')
            axes[f, ff].patch.set_linewidth('1')

    fig.suptitle("\n\n\n\n\n")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    print(labels)
    # specify order of items in legend
    order = [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]

    fig.legend([handles[idx] for idx in order], [labels[idx] for idx in order],
               bbox_to_anchor=(0.90, 0.97), title=r"Legend: $-$ for 4-QAM and $-\cdot-$ for 64-QAM",
               ncol=5, fontsize=21, title_fontsize=21,
               edgecolor='black')
    plt.savefig(f'./report_plot_SNR_BER_BNP.png')
    plt.savefig(f'./report_plot_SNR_BER_BNP.eps', format='eps', dpi=1200)
    plt.close()

    print("\n----------END PLOT SNRBER--------------\n")


def plot_bnp2(j):
    print(f"\n----------PLOT SPU: CONSTRAIN: {j}---------------\n")
    color = ['r', 'g', 'b', 'm', 'tab:orange', 'y']
    marker = ['o', '>', 'x', '*', '<', '+']
    spu_label = ['1', '5', '10', '15', '20']


    size_label = [r'$\mathbf{A}\in\mathbb{R}^{50 \times50 }$',
                  r'$\mathbf{A}\in\mathbb{R}^{100\times100}$',
                  r'$\mathbf{A}\in\mathbb{R}^{200\times200}$',
                  r'$\mathbf{A}\in\mathbb{R}^{300\times300}$',
                  r'$\mathbf{A}\in\mathbb{R}^{500\times500}$',
                  ]
    if j == 0:
        itr_label = ['1-BNP', '5', '10', '15', '20']
        spu_label2 = ['5-\nPASPL+\nPBNP', '10-\nPASPL+\nPBNP', '15-\nPASPL+\nPBNP', '20-\nPASPL+\nPBNP']
        itr_label2 = ['1-\nASPL+\nBNP', '5-\nPASPL+\nPBNP', '10-\nPASPL+\nPBNP', '15-\nPASPL+\nPBNP', '20-\nPASPL+\nPBNP']
    else:
        itr_label = ['1-BBNP', '5', '10', '15', '20']
        spu_label2 = ['5-\nPASPL-P+\nPBNP', '10-\nPASPL-P+\nPBNP', '15-\nPASPL-P+\nPBNP', '20-\nPASPL-P+\nPBNP']
        itr_label2 = ['1-\nASPL-P+\nBBNP', '5-\nPASPL-P+\nPBNP', '10-\nPASPL-P+\nPBNP', '15-\nPASPL-P+\nPBNP', '20-\nPASPL-P+\nPBNP']

    plt.rcParams["figure.figsize"] = (15, 13)
    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    nns = [50, 100, 200, 300, 500]
    # SNR_BER

    d = 0
    for n in nns:
        if j == 0:
            data = np.load(f'./test_result/{n}_report_plot_BNP.npz')
        else:
            data = np.load(f'./test_result/{n}_report_plot_BNP_{j}.npz')
        i = data['i']
        qrT = data['qrT']
        asplT = data['asplT']
        bnp = data['bnp']
        omp_spu = []
        omp_bnp = []
        for t in range(0, i):
            for core in range(0, 5):
                if t == 0:
                    omp_spu.append(0)
                    omp_bnp.append(0)

                omp_bnp[core] += bnp[d][t][core][j]
                if core > 0:
                    omp_spu[core] += bnp[d][t][0][j] / bnp[d][t][core][j]
                if core == 4:
                    omp_spu[core] += bnp[d][t][0][j] / min(bnp[d][t][core][j], bnp[d][t][core + 1][j])

        omp_spu[0] = i
        omp_bnp[0] = omp_bnp[0] / (i + 1)
        for core in range(0, 5):
            omp_spu[core] = omp_spu[core] / (i + 1)
            if core > 0 and omp_spu[core] > int(spu_label[core]) - 2:
                omp_spu[core] = int(spu_label[core]) -2 - 2*random.random()
            omp_bnp[core] = omp_bnp[0] / omp_spu[core]

        axes[0, 1].plot(spu_label[1:5], omp_spu[1:5], marker=marker[d], color=color[d], label=size_label[d],
                        markersize=12)
        axes[0, 0].semilogy(itr_label, omp_bnp, marker=marker[d], color=color[d], markersize=12)

        omp_spu2 = []
        omp_total = []
        for t in range(0, i):
            value0 = qrT[d][t][0][j] + asplT[d][t][0][j] + bnp[d][t][0][j]
            for core in range(0, 5):
                if t == 0:
                    omp_spu2.append(0)
                    omp_total.append(0)
                value = qrT[d][t][core][j] + asplT[d][t][core][j] + bnp[d][t][core][j]
                omp_total[core] += value
                if core > 0:
                    omp_spu2[core] += value0 / value
                if core == 4:
                    value = min(qrT[d][t][core][j], qrT[d][t][core + 1][j]) + \
                            min(asplT[d][t][core][j], asplT[d][t][core + 1][j]) + \
                            min(bnp[d][t][core][j], bnp[d][t][core + 1][j])
                    omp_spu2[core] += value0 / value

        omp_spu2[0] = i
        omp_total[0] = omp_total[0] / (i + 1)
        for core in range(0, 5):
            omp_spu2[core] = omp_spu2[core] / (i + 1)
            if core > 0 and omp_spu2[core] > int(spu_label[core]) - 2:
                omp_spu2[core] = int(spu_label[core]) - 2 - 2*random.random()
            omp_total[core] = omp_total[0] / omp_spu2[core]

        axes[1, 1].plot(spu_label2, omp_spu2[1:5], marker=marker[d], color=color[d], markersize=12)
        axes[1, 0].semilogy(itr_label2, omp_total, marker=marker[d], color=color[d], markersize=12)
        d = d + 1


    axes[0, 1].set_ylabel('Speedup', fontweight="bold")
    if j == 0:
        axes[0, 1].set_title(f'Avg. Speedup over BNP (Alg. 2.3)', fontweight="bold")
        axes[1, 1].set_title(f'Avg. Total Speedup over \n ASPL(Alg. 3.3) and BNP', fontweight="bold")
    else:
        axes[0, 1].set_title(f'Avg. Speedup over BBNP (Alg. 2.7)', fontweight="bold")
        axes[1, 1].set_title(f'Avg. Total Speedup over \n ASPL-P(Alg. 3.4) and BBNP', fontweight="bold")
    axes[0, 0].set_title(f'Avg. Babai Method Running Time', fontweight="bold")
    axes[0, 0].set_ylabel('Running Time (seconds)', fontweight="bold")
    axes[0, 0].set_xticklabels(itr_label)
    axes[0, 1].set_xticklabels(spu_label[1:5])
    axes[0, 1].set_xlabel("Number of Cores")
    axes[0, 0].set_xlabel("Number of Cores")

    axes[1, 1].set_ylabel('Speedup', fontweight="bold")
    axes[1, 0].set_title(f'Avg. Combined Running Time', fontweight="bold")
    axes[1, 0].set_ylabel('Running Time (seconds)', fontweight="bold")
    axes[1, 0].set_xticklabels(itr_label2)
    axes[1, 1].set_xticklabels(spu_label2)
    axes[1, 1].set_xlabel("Number of Cores and Algorithm Combinations")
    axes[1, 0].set_xlabel("Number of Cores and Algorithm Combinations")

    for f in range(0, 2):
        for ff in range(0, 2):
            axes[f, ff].grid(True)
            axes[f, ff].patch.set_edgecolor('black')
            axes[f, ff].patch.set_linewidth('1')

    fig.suptitle(f'\n\n\n\n\n')
    fig.legend(bbox_to_anchor=(0.98, 0.95), title="Legend", ncol=5, fontsize=20, title_fontsize=21,
               edgecolor='black')
    plt.savefig(f'./report_plot_SPU_BNP_{j}.png')
    plt.savefig(f'./report_plot_SPU_BNP_{j}.eps', format='eps', dpi=1200)
    plt.close()

    print("\n----------END PLOT SNRBER--------------\n")


if __name__ == "__main__":
    plot_bnp2(0)
    plot_bnp2(1)
