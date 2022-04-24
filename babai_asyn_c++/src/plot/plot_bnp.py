import matplotlib.pyplot as plt
import numpy as np

rc = {"font.family": "serif", "mathtext.fontset": "stix"}

plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 19})


def save_data(n, i, start, end, qrT, asplT, bnp, ber, itr):
    np.savez(f'./test_result/{n}_report_plot_{start}_{end}_BNP.npz',
             n=n, i=i, start=start, end=end, qrT=qrT, asplT=asplT, bnp=bnp, ber=ber, itr=itr)


def plot_bnp(j):
    qam = 64 if j == 1 else 4
    print(f"\n----------PLOT SPU: {qam}-QAM--------------\n")
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
    plt.rcParams["figure.figsize"] = (14, 16)
    fig, axes = plt.subplots(3, 2, constrained_layout=True)

    sizes = [[50, 150], [250, 350], [450, 550]]
    SNRs = [[0, 10], [20, 30], [40, 50]]
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
            for ff in range(0, 2):
                omp_spu = []
                omp_bnp = []
                for t in range(0, i):
                    for core in range(0, 5):
                        if t == 0:
                            omp_spu.append(0)
                            omp_bnp.append(0)

                        omp_bnp[core] += bnp[d][t][core][j]
                        if core > 0:
                            omp_spu[core-1] += bnp[d][t][0][j] / bnp[d][t][core][j]
                        if core == 4:
                            omp_spu[core-1] += bnp[d][t][0][j] / min(bnp[d][t][core][j], bnp[d][t][core+1][j])

                for core in range(0, 5):
                    omp_spu[core] = omp_spu[core] / i
                if f == 0 and ff == 0:
                    axes[f, ff].semilogy(itr_label, omp_spu[0:4], color=color[d], marker=marker[d],
                                     label=size_label[d], markersize=12)
                else:
                    axes[f, ff].plot(itr_label, omp_spu[1:5], color=color[d], marker=marker[d], markersize=12)

        d = d + 1

    for f in range(0, 3):
        for ff in range(0, 2):
            axes[f, ff].set_title(f'Avg. Speedup vs Number of Cores for {SNRs[f][ff]}-SNR', fontweight="bold")
            axes[f, ff].set_ylabel('Avg. Speedup', fontweight="bold")
            axes[f, ff].set_xticklabels(itr_label)
            axes[f, ff].grid(color='b', ls='-.', lw=0.25)
            axes[f, ff].patch.set_edgecolor('black')
            axes[f, ff].patch.set_linewidth('1')


    fig.suptitle(f"{qam}-QAM\n\n\n\n\n")
    fig.legend(bbox_to_anchor=(0.88, 0.97), title="Legend", ncol=6, fontsize=21, title_fontsize=21,
               edgecolor='black')
    plt.savefig(f'./report_plot_SNR_SPU_BNP_{qam}QAM.png')
    plt.savefig(f'./report_plot_SNR_SPU_BNP_{qam}QAM.eps', format='eps', dpi=1200)
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
    plt.rcParams["figure.figsize"] = (14, 16)
    fig, axes = plt.subplots(3, 2, constrained_layout=True)

    sizes = [[50, 150], [250, 350], [450, 550]]
    SNRs = [[0, 10], [20, 30], [40, 50]]
    nns = [50, 150, 250, 350, 450, 550]
    # SNR_BER

    labels_snr = ['1-BNP', '$n_c=5$', '$n_c=10$', '$n_c=15$', '$n_c=20$']
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
                            omp_ber[snr] = omp_ber[snr] + ber[snr][t][l][j] / i
                    if j == 0 and f == 0 and ff == 0:
                        axes[f, ff].plot(snr_label, omp_ber, color=color[l], marker=marker[l], label=labels_snr[l],
                                         markersize=12)
                    elif j == 0:
                        axes[f, ff].plot(snr_label, omp_ber, color=color[l], marker=marker[l], markersize=12)
                    else:
                        axes[f, ff].plot(snr_label, omp_ber, color=color[l], marker=marker[l], linestyle='-.',

                                         markersize=12)

    for f in range(0, 3):
        for ff in range(0, 2):
            axes[f, ff].set_title(f'Avg. BER vs SNR for Dimension {sizes[f][ff]}', fontweight="bold")
            axes[f, ff].set_ylabel('Avg. BER', fontweight="bold")
            axes[f, ff].set_xticklabels(snr_label)
            axes[f, ff].set_xlabel('SNR', fontweight="bold")
            axes[f, ff].grid(color='b', ls='-.', lw=0.25)
            axes[f, ff].patch.set_edgecolor('black')
            axes[f, ff].patch.set_linewidth('1')

    fig.suptitle("\n\n\n\n\n")
    fig.legend(bbox_to_anchor=(0.88, 0.97), title="Legend", ncol=5, fontsize=21, title_fontsize=21,
               edgecolor='black')
    plt.savefig(f'./report_plot_SNR_BER_BNP.png')
    plt.savefig(f'./report_plot_SNR_BER_BNP.eps', format='eps', dpi=1200)
    plt.close()

    print("\n----------END PLOT SNRBER--------------\n")


if __name__ == "__main__":
    # for start in range(0,10,10):
    plot_ber()
    plot_bnp(0)
    plot_bnp(1)
