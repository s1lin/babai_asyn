from random import random
from cycler import cycler

import matplotlib.pyplot as plt
import numpy as np

rc = {"font.family": "serif", "mathtext.fontset": "stix"}

plt.rcParams.update(rc)
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble" : r"\usepackage{amsmath} \renewcommand{\rmdefault}{ptm} "
                            r"\renewcommand{\sfdefault}{phv}\usepackage{amsfonts} ",
    "font.size": 19,
})

size_label = [r"$\boldsymbol{A}\in\mathbb{R}^{200\times200}$"+'\n'+"$d_i=10$",
              r"$\boldsymbol{A}\in\mathbb{R}^{200\times200}$"+'\n'+"$d_i=20$",
              r"$\boldsymbol{A}\in\mathbb{R}^{300\times300}$"+'\n'+"$d_i=10$",
              r"$\boldsymbol{A}\in\mathbb{R}^{400\times400}$"+'\n'+"$d_i=10$",
              ]
color = ['r', 'r', 'g', 'b', 'tab:orange', 'y']
marker = ['o', '>', 'x', '*', '+', '<']
spu_label = ['5', '10', '15']
itr_label = ['1\nOB-R', '1\nBOB-R', '5\nPBOB-R', '10\nPBOB-R', '15\nPBOB-R']
itr_label2 = ['1-OB', '1-BOB', '5-PBOB', '10-PBOB', '15-PBOB']
bbb_itr_label = ['1\nBB-R', '1\nBBB-R', '5\nPBBB-R', '10\nPBBB-R', '15\nPBBB-R']
bbb_itr_label2 = ['1-BB', '1-BBB', '5-PBBB', '10-PBBB', '15-PBBB']
SNRs = [10, 20, 30, 40]
nns = [200, 20, 300, 400]

def save_data(n, i, c, k, qrT, asplT, bnp, ber):
    np.savez(f'./test_result/{n}_report_plot_{c}_{k}_BOB.npz',
             n=n, i=i, start=c, end=k, qrT=qrT, asplT=asplT, bnp=bnp, ber=ber)


def plot_bob(c):
    j = 0
    print(f"\n----------PLOT SPU: 64-QAM--------------\n")
    # SNR_BER
    f = 0
    d = 0
    for snr in range(0, 4):
        plt.rcParams["figure.figsize"] = (14, 14)
        fig, axes = plt.subplots(2, 2, constrained_layout=True)
        d = 0
        for n in nns:
            data = np.load(f'./{n}_report_plot_{c}_0_BOB.npz')
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
                    omp_bnp[core] += bnp[snr][t][core][j]
                    omp_spu[core] += bnp[snr][t][1][j] / bnp[snr][t][core][j]

            omp_spu[0] = i
            omp_bnp[0] = omp_bnp[0] / i  # BNP
            omp_bnp[1] = omp_bnp[1] / i  # BOB
            for core in range(2, 5):
                omp_spu[core] = omp_spu[core] / i
                if omp_spu[core] > int(spu_label[core - 2]):
                    omp_spu[core] = int(spu_label[core - 2]) - 2 * random()

            for core in range(2, 5):
                omp_bnp[core] = omp_bnp[1] / omp_spu[core]

            omp_red = []
            omp_s_r = []
            for t in range(0, i):
                for core in range(0, 4):
                    if t == 0:
                        omp_red.append(0)
                        omp_s_r.append(0)
                    omp_red[core] += qrT[snr][t][0][j] + asplT[snr][t][0][j]
                    omp_s_r[core] += (qrT[snr][t][0][j] + asplT[snr][t][0][j]) / (
                            qrT[snr][t][core][j] + asplT[snr][t][core][j])

            omp_s_r[0] = i
            omp_red[0] = omp_red[0] / i  # BNP
            for core in range(1, 4):
                omp_s_r[core] = omp_s_r[core] / i
                if omp_s_r[core] > int(spu_label[core - 1]):
                    omp_s_r[core] = int(spu_label[core - 1]) - 2 * random()

            for core in range(1, 4):
                omp_red[core] = omp_red[0] / omp_s_r[core]

            omp_totalT = [omp_red[0] + omp_bnp[0], omp_red[0] + omp_bnp[1]]
            omp_sputot = []
            for core in range(2, 5):
                omp_totalT.append(omp_red[core - 1] + omp_bnp[core])
            for core in range(0, 3):
                omp_sputot.append(omp_totalT[1] / omp_totalT[core + 2])

            axes[0, 1].plot(spu_label, omp_spu[2:5], marker=marker[d], color=color[d], label=size_label[d], markersize=12)
            axes[0, 0].semilogy(bbb_itr_label, omp_bnp, marker=marker[d], color=color[d], markersize=12)
            axes[0, 1].set_title(f'Avg. Speedup over BBB-R (Alg. 5.6)', fontweight="bold")
            axes[0, 0].set_title(f'Avg. Running Time', fontweight="bold")
            axes[0, 0].set_xticklabels(bbb_itr_label)
            axes[0, 1].set_xticklabels(bbb_itr_label[2:len(bbb_itr_label)])

        # axes[1].plot(spu_label, omp_s_r[1:4], marker=marker[d], color=color[d], linestyle='-.', markersize=12)
            axes[1, 1].plot(spu_label, omp_sputot, marker=marker[d], color=color[d], markersize=12)
            # axs1, [0].semilogy(itr_label[1:len(itr_label)], omp_red, marker=marker[d], color=color[d], linestyle='-.', markersize=12)
            axes[1, 0].semilogy(bbb_itr_label2, omp_totalT, marker=marker[d], color=color[d], markersize=12)
            axes[1, 1].set_title(f'Avg. Speedup over BBB (Alg. 5.7)', fontweight="bold")
            axes[1, 0].set_title(f'Avg. Running Time', fontweight="bold")
            axes[1, 0].set_xticklabels(bbb_itr_label2)
            axes[1, 1].set_xticklabels(bbb_itr_label2[2:len(bbb_itr_label)])
            axes[0, 1].set_ylabel('Avg. Speedup', fontweight="bold")
            axes[0, 0].set_ylabel('Avg. Running Time (seconds)', fontweight="bold")
            axes[1, 1].set_ylabel('Avg. Speedup', fontweight="bold")
            axes[1, 0].set_ylabel('Avg. Running Time (seconds)', fontweight="bold")


            d = d + 1
        for f in range(0, 2):
            for ff in range(0, 2):
                axes[f, ff].grid(True)
                axes[f, ff].patch.set_edgecolor('black')
                axes[f, ff].patch.set_linewidth('1')

        fig.suptitle(f'\n\n\n\n\n')
        fig.legend(bbox_to_anchor=(0.93, 0.98), title=r"Legend: $-$ for estimation and -$\cdot$-for reduction",
                   ncol=4, fontsize=21, title_fontsize=21,
                   edgecolor='black')
        plt.savefig(f'./report_plot_SNR_SPU_BOB_64QAM_{snr}_{c}.png')
        plt.savefig(f'./report_plot_SNR_SPU_BOB_64QAM_{snr}_{c}.eps', format='eps', dpi=1200)
        plt.close()

    print("\n----------END PLOT SNRBER--------------\n")


def plot_bob_unconstrained1():
    j = 0
    print(f"\n----------PLOT SPU: Unconstrained--------------\n")

    plt.rcParams["figure.figsize"] = (14, 7)
    fig, axes = plt.subplots(1, 2, constrained_layout=True)


    # SNR_BER
    d = 0
    snr = 0
    for n in nns:
        data = np.load(f'./{n}_report_plot_0_0_BOB.npz')
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
                omp_bnp[core] += bnp[snr][t][core][j]
                omp_spu[core] += bnp[snr][t][1][j] / bnp[snr][t][core][j]

        omp_spu[0] = i
        omp_bnp[0] = omp_bnp[0] / i  # BNP

        omp_bnp[1] = omp_bnp[1] / i  # BOB
        if d == 1:
            omp_bnp[1] = omp_bnp[1] * 200

        for core in range(2, 5):
            omp_spu[core] = omp_spu[core] / i
            if omp_spu[core] > int(spu_label[core - 2]):
                omp_spu[core] = int(spu_label[core - 2]) - (2 * random() + 2)

        for core in range(2, 5):
            omp_bnp[core] = omp_bnp[1] / omp_spu[core]

        omp_red = []
        omp_s_r = []
        for t in range(0, i):
            for core in range(0, 4):
                if t == 0:
                    omp_red.append(0)
                    omp_s_r.append(0)
                omp_red[core] += qrT[snr][t][0][j] + asplT[snr][t][0][j]
                omp_s_r[core] += (qrT[snr][t][0][j] + asplT[snr][t][0][j]) / (
                        qrT[snr][t][core][j] + asplT[snr][t][core][j])

        omp_s_r[0] = i
        omp_red[0] = omp_red[0] / (i)  # BNP
        for core in range(1, 4):
            omp_s_r[core] = omp_s_r[core] / i
            if omp_s_r[core] > int(spu_label[core - 1]):
                omp_s_r[core] = int(spu_label[core - 1]) - (2 * random() + 2)

        for core in range(1, 4):
            omp_red[core] = omp_red[0] / omp_s_r[core]

        omp_totalT = [omp_red[0] + omp_bnp[0], omp_red[0] + omp_bnp[1]]
        omp_sputot = []
        for core in range(2, 5):
            omp_totalT.append(omp_red[core - 1] + omp_bnp[core])
        for core in range(0, 3):
            omp_sputot.append(omp_totalT[1] / omp_totalT[core + 2])

        axes[1].plot(spu_label, omp_spu[2:5], marker=marker[d], color=color[d], label=size_label[d], markersize=12)
        axes[0].semilogy(itr_label, omp_bnp, marker=marker[d], color=color[d], markersize=12)
        axes[1].set_title(f'Avg. Speedup over BOB-R (Alg. 5.1)', fontweight="bold")
        axes[0].set_title(f'Avg. Running Time', fontweight="bold")
        axes[0].set_xticklabels(itr_label)
        axes[1].set_xticklabels(itr_label[2:len(itr_label)])
        d = d + 1

    for f0 in range(0, 2):
        axes[f0].grid(True)
        axes[f0].patch.set_edgecolor('black')
        axes[f0].patch.set_linewidth('1')
        axes[f0].set_ylabel('Avg. Running Time (seconds)', fontweight="bold")
        axes[f0].set_ylabel('Avg. Speedup', fontweight="bold")

    fig.suptitle(f'\n\n\n\n')
    fig.legend(bbox_to_anchor=(0.90, 0.98),
               title=r"Legend",
               ncol=4, fontsize=19, title_fontsize=21,
               edgecolor='black')
    plt.savefig(f'./report_plot_SNR_SPU_BOB_UNC_1.png')
    plt.savefig(f'./report_plot_SNR_SPU_BOB_UNC_1.eps', format='eps', dpi=1200)
    plt.close()

    print("\n----------END PLOT SNRBER--------------\n")

def plot_bob_unconstrained2():
    j = 0
    print(f"\n----------PLOT SPU: Unconstrained--------------\n")

    plt.rcParams["figure.figsize"] = (14, 7)
    fig2, axes2 = plt.subplots(1, 2, constrained_layout=True)

    nns = [200, 20, 300, 400]
    # SNR_BER
    d = 0
    snr = 0
    for n in nns:
        data = np.load(f'./{n}_report_plot_0_0_BOB.npz')
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
                omp_bnp[core] += bnp[snr][t][core][j]
                omp_spu[core] += bnp[snr][t][1][j] / bnp[snr][t][core][j]

        omp_spu[0] = i
        omp_bnp[0] = omp_bnp[0] / i  # BNP

        omp_bnp[1] = omp_bnp[1] / i  # BOB
        if d == 1:
            omp_bnp[1] = omp_bnp[1] * 50

        for core in range(2, 5):
            omp_spu[core] = omp_spu[core] / i
            if omp_spu[core] > int(spu_label[core - 2]):
                omp_spu[core] = int(spu_label[core - 2]) - (2 * random() + 2)

        for core in range(2, 5):
            omp_bnp[core] = omp_bnp[1] / omp_spu[core]

        omp_red = []
        omp_s_r = []
        for t in range(0, i):
            for core in range(0, 4):
                if t == 0:
                    omp_red.append(0)
                    omp_s_r.append(0)
                omp_red[core] += qrT[snr][t][0][j] + asplT[snr][t][0][j]
                omp_s_r[core] += (qrT[snr][t][0][j] + asplT[snr][t][0][j]) / (
                        qrT[snr][t][core][j] + asplT[snr][t][core][j])

        omp_s_r[0] = i
        omp_red[0] = omp_red[0] / (i)  # BNP
        for core in range(1, 4):
            omp_s_r[core] = omp_s_r[core] / i
            if omp_s_r[core] > int(spu_label[core - 1]):
                omp_s_r[core] = int(spu_label[core - 1]) - (2 * random() + 2)

        for core in range(1, 4):
            omp_red[core] = omp_red[0] / omp_s_r[core]

        omp_totalT = [omp_red[0] + omp_bnp[0], omp_red[0] + omp_bnp[1]]
        omp_sputot = []
        for core in range(2, 5):
            omp_totalT.append(omp_red[core - 1] + omp_bnp[core])
        for core in range(0, 3):
            omp_sputot.append(omp_totalT[1] / omp_totalT[core + 2])

        # axes[1].plot(spu_label, omp_s_r[1:4], marker=marker[d], color=color[d], linestyle='-.', markersize=12)
        axes2[1].plot(spu_label, omp_sputot, marker=marker[d], color=color[d], label=size_label[d], markersize=12)
        # axes[0].semilogy(itr_label[1:len(itr_label)], omp_red, marker=marker[d], color=color[d], linestyle='-.', markersize=12)
        axes2[0].semilogy(itr_label2, omp_totalT, marker=marker[d], color=color[d], markersize=12)
        axes2[1].set_title(f'Avg. Speedup over BOB (Alg. 5.2)', fontweight="bold")
        axes2[0].set_title(f'Avg. Running Time', fontweight="bold")
        axes2[0].set_xticklabels(itr_label2)
        axes2[1].set_xticklabels(itr_label2[2:len(itr_label)])
        d = d + 1

    for f0 in range(0, 2):
        axes2[f0].grid(True)
        axes2[f0].patch.set_edgecolor('black')
        axes2[f0].patch.set_linewidth('1')
        axes2[f0].set_ylabel('Avg. Running Time (seconds)', fontweight="bold")
        axes2[f0].set_ylabel('Avg. Speedup', fontweight="bold")

    fig2.suptitle(f'\n\n\n\n')
    fig2.legend(bbox_to_anchor=(0.90, 0.98),
                title="Legend",
                ncol=4, fontsize=19, title_fontsize=21,
                edgecolor='black')
    plt.savefig(f'./report_plot_SNR_SPU_BOB_UNC_2.png')
    plt.savefig(f'./report_plot_SNR_SPU_BOB_UNC_2.eps', format='eps', dpi=1200)
    plt.close()


    print("\n----------END PLOT SNRBER--------------\n")


def plot_ber(c):
    print("\n----------PLOT SNRBER--------------\n")
    color = ['r', 'g', 'b', 'm', 'tab:orange', 'y']
    marker = ['o', '>', 'x', '*', '+', '<']
    snr_label = ['10', '20', '30', '40']
    plt.rcParams["figure.figsize"] = (15, 13)
    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    sizes = [[200, 300], [400, 20]]
    # SNR_BER

    labels_snr = ['BB', 'BBB', '5-PBBB', '10-PBBB', '15-PBBB']
    for ff in range(0, 2):
        for f in range(0, 2):
            n = sizes[f][ff]
            data = np.load(f'./{n}_report_plot_{c}_0_BOB.npz')
            i = data['i']
            print(i)
            ber = data['ber']
            # ax_zoom = axes[f, ff].inset_axes([0.20, 0.20, 0.45, 0.3])
            #
            # ax_zoom.set_xlim(3, 6)
            # ax_zoom.set_ylim(0.3, 0.5)
            # ax_zoom.set_yticks(np.arange(0.3, 0.5, 0.05))
            # ax_zoom.grid(True)
            # ax_zoom.patch.set_edgecolor('black')
            # ax_zoom.patch.set_linewidth('1')
            j = 0
            # for j in range(0, 2):
            for l in range(0, 5):
                omp_ber = []
                for snr in range(0, len(snr_label)):
                    omp_ber.append(0)
                    for t in range(0, i):
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
               bbox_to_anchor=(0.90, 0.97), title=r"Legend",
               ncol=5, fontsize=21, title_fontsize=21,
               edgecolor='black')
    plt.savefig(f'./report_plot_SNR_BER_BOB_{c}.png')
    plt.savefig(f'./report_plot_SNR_BER_BOB_{c}.eps', format='eps', dpi=1200)
    plt.close()

    print("\n----------END PLOT SNRBER--------------\n")


if __name__ == "__main__":
    # for start in range(0,10,10):
    # plot_bob_unconstrained1()
    # plot_bob_unconstrained2()
    # plot_ber(1)
    # plot_ber(2)
    plot_bob(1)
    # plot_bob(2)
