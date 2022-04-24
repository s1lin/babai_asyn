import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np

rc = {"font.family": "serif", "mathtext.fontset": "stix"}

plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({'font.size': 19})


def save_data(n, i, start, end, qrT, asplT, bnp, ber, itr):
    np.savez(f'./test_result/{n}_report_plot_{start}_{end}_BNP.npz',
             n=n, i=i, start=start, end=end, qrT=qrT, asplT=asplT, bnp=bnp, ber=ber, itr=itr)


def plot_bnp():
    print("\n----------PLOT SNRBER--------------\n")
    color = ['r', 'g', 'b', 'm', 'tab:orange', 'r']
    marker = ['o', '>', 'x', '*', '+', '<']
    procs = [5, 10, 15, 20]
    linestyle = ['-.', '-']

    itr_label = ['$1$-BNP'] + ['$' + str(proc) + '$' for proc in procs]
    snr_label = ['0', '10', '20', '30', '40', '50']
    plt.rcParams["figure.figsize"] = (14, 16)
    fig, axes = plt.subplots(3, 2, constrained_layout=True)

    plt2.rcParams["figure.figsize"] = (20, 16)
    fig2, axes2 = plt2.subplots(3, 2, constrained_layout=True)

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
    print(snr_label)
    for f in range(0, 3):
        for ff in range(0, 2):
            axes[f, ff].set_title(f'Avg. BER vs SNR for Dimension {sizes[f][ff]}', fontweight="bold")
            axes[f, ff].set_ylabel('Avg. BER', fontweight="bold")
            axes[f, ff].set_xticklabels(snr_label)
            axes[f, ff].set_xlabel('SNR', fontweight="bold")
            axes[f, ff].grid(color='b', ls='-.', lw=0.25)
            axes[f, ff].patch.set_edgecolor('black')
            axes[f, ff].patch.set_linewidth('1')

            # axes2[f, ff].set_title(f'Avg. Speedup vs Number of Cores for {SNRs[f][ff]}-SNR', fontweight="bold")
            # axes2[f, ff].set_ylabel('Avg. Speedup', fontweight="bold")
            # axes2[f, ff].set_xticklabels(itr_label)
            # axes2[f, ff].grid(color='b', ls='-.', lw=0.25)
            # axes2[f, ff].patch.set_edgecolor('black')
            # axes2[f, ff].patch.set_linewidth('1')

    fig.suptitle("\n\n\n\n\n")
    fig.legend(bbox_to_anchor=(0.88, 0.97), title="Legend", ncol=5, fontsize=21, title_fontsize=21,
               edgecolor='black')

    plt.savefig(f'./report_plot_SNR_BER_BNP_1.png')
    plt.savefig(f'./report_plot_SNR_BER_BNP_1.eps', format='eps', dpi=1200)
    plt.close()

    plt2.savefig(f'./report_plot_SNR_BER_BNP_2.png')
    plt2.savefig(f'./report_plot_SNR_BER_BNP_2.eps', format='eps', dpi=1200)
    plt2.close()

    print("\n----------END PLOT SNRBER--------------\n")


if __name__ == "__main__":
    # for start in range(0,10,10):
    plot_bnp()
