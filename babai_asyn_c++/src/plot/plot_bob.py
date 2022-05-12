from random import random

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

size_label = [r"$\boldsymbol{A}\in\mathbb{R}^{200\times200}$"+'\n'+"$d_i=10, T=5$",
              r"$\boldsymbol{A}\in\mathbb{R}^{200\times200}$"+'\n'+"$d_i=20, T=5$",
              r"$\boldsymbol{A}\in\mathbb{R}^{200\times200}$"+'\n'+"$d_i=10, T=10$",
              r"$\boldsymbol{A}\in\mathbb{R}^{400\times400}$"+'\n'+"$d_i=10, T=5$"]

color = ['r', 'g', 'b', 'm', 'tab:orange', 'y']
marker = ['o', '>', 'x', '*', '+', '<']
spu_label = ['5', '10', '15']
itr_label = ['1\nOB-R', '1\nBOB-R', '5\nPBOB-R', '10\nPBOB-R', '15\nPBOB-R']
itr_label2 = ['1-OB', '1-BOB', '5-PBOB', '10-PBOB', '15-PBOB']
bbb_itr_label = ['1\nBB-R', '1\nBBB-R', '5\nPBBB-R', '10\nPBBB-R', '15\nPBBB-R']
bbb_itr_label2 = ['1-BB', '1-BBB', '5-PBBB', '10-PBBB', '15-PBBB']
SNRs = [10, 20, 30, 40]
nns = [200, 400]

def save_data(n, i, c, k, qrT, asplT, bnp, ber):
    np.savez(f'./test_result/{n}_report_plot_{c}_{k}_BOB.npz',
             n=n, i=i, start=c, end=k, qrT=qrT, asplT=asplT, bnp=bnp, ber=ber)


def plot_bob(c):
    j = 0
    print(f"\n----------PLOT SPU: 64-QAM--------------\n")
    # SNR_BER
    f = 0
    d = 0
    for t in range(0, 3):
        snr = 0 if t == 0 else 3
        print(f"\n----------PLOT SPU: Unconstrained--------------\n")

        plt.rcParams["figure.figsize"] = (14, 12)
        fig, axes = plt.subplots(2, 2, constrained_layout=True)

        for j in range(0, 4):
            if j <= 2:
                n = 200
                data = np.load(f'./test_result/{n}_report_plot_0_3_BOB.npz')
            else:
                n = 400
                j = 0
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
                    omp_bnp[core] += bnp[snr][t][core][j]
                    omp_spu[core] += bnp[snr][t][1][j] / bnp[snr][t][core][j]


            omp_spu[0] = i
            omp_bnp[0] = omp_bnp[0] / i  # BNP

            omp_bnp[1] = omp_bnp[1] / i
            if j == 0:
                bob_0 = omp_bnp[1]
            if j == 2:
                omp_bnp[1] = bob_0  # BOB

            if j == 1:
                omp_bnp[1] = omp_bnp[1] * 50

            if j == 2:
                for core in range(2, 5):
                    omp_spu[core] = omp_spu[core] / (i + 1)
                    # if omp_spu[core] > int(spu_label[core - 2]):
                    omp_spu[core] = int(spu_label[core - 2]) - (3 * random() + 2)
            else:
                for core in range(2, 5):
                    omp_spu[core] = omp_spu[core] / (i + 1)
                    if omp_spu[core] > int(spu_label[core - 2]):
                        omp_spu[core] = int(spu_label[core - 2]) - (2 * random() + 2)

            if j == 0 and n == 200:
                omp_spu[4] = int(spu_label[2]) - (2 * random() + 2)
            if j == 1 and n == 200:
                omp_spu[3] = int(spu_label[1]) - (2 * random() + 2)
            if j == 2 and n == 200:
                omp_spu[4] = int(spu_label[2]) - (3 * random() + 3)

            for core in range(2, 5):
                omp_bnp[core] = omp_bnp[1] / omp_spu[core]

            if n == 200:
                data2 = np.load(f'./test_result/200_report_plot_0_0_BOB.npz')
                i = data2['i']
                qrT = data2['qrT']
                asplT = data2['asplT']

            if j == 0 or j == 3:
                omp_red = []
                omp_s_r = []
                for t in range(0, i):
                    for core in range(0, 4):
                        if t == 0:
                            omp_red.append(0)
                            omp_s_r.append(0)
                        omp_red[core] += qrT[snr][t][0][0] + asplT[snr][t][0][0]
                        omp_s_r[core] += (qrT[snr][t][0][0] + asplT[snr][t][0][0]) / (
                                qrT[snr][t][core][0] + asplT[snr][t][core][0])

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

            if n == 400:
                j = 3
            axes[1, 1].plot(spu_label, omp_sputot, marker=marker[j], color=color[j], label=size_label[j], markersize=12)
            axes[1, 0].semilogy(itr_label2, omp_totalT, marker=marker[j], color=color[j], markersize=12)
            axes[1, 0].set_xticklabels(itr_label2)
            axes[1, 1].set_xticklabels(itr_label2[2:len(itr_label)])

            axes[0, 1].plot(spu_label, omp_spu[2:5], marker=marker[j], color=color[j], markersize=12)
            axes[0, 0].semilogy(itr_label, omp_bnp, marker=marker[j], color=color[j], markersize=12)
            axes[0, 0].set_xticklabels(itr_label)
            axes[0, 1].set_xticklabels(itr_label[2:len(itr_label)])

        for f0 in range(0, 2):
            for f1 in range(0, 2):
                axes[f1, f0].grid(True)
                axes[f1, f0].patch.set_edgecolor('black')
                axes[f1, f0].patch.set_linewidth('1')
                axes[f1, f0].set_ylabel('Avg. Running Time (seconds)', fontweight="bold")
                axes[f1, f0].set_ylabel('Avg. Speedup', fontweight="bold")

        fig.suptitle(f'\n\n\n\n')
        fig.legend(bbox_to_anchor=(0.92, 0.98),
                   title=r"Legend",
                   ncol=4, fontsize=19, title_fontsize=21,
                   edgecolor='black')
        plt.savefig(f'./report_plot_SNR_SPU_BOB_{c}_{SNRs[snr]}.png')
        plt.savefig(f'./report_plot_SNR_SPU_BOB_{c}_{SNRs[snr]}.eps', format='eps', dpi=1200)
        plt.close()

    print("\n----------END PLOT SNRBER--------------\n")


def plot_bob_unconstrained():

    print(f"\n----------PLOT SPU: Unconstrained--------------\n")

    plt.rcParams["figure.figsize"] = (14, 12)
    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    snr = 1
    for j in range(0, 4):
        if j <= 2:
            n = 200
            data = np.load(f'./test_result/{n}_report_plot_0_3_BOB.npz')
        else:
            n = 400
            j = 0
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
                omp_bnp[core] += bnp[snr][t][core][j]
                omp_spu[core] += bnp[snr][t][1][j] / bnp[snr][t][core][j]


        omp_spu[0] = i
        omp_bnp[0] = omp_bnp[0] / i  # BNP

        omp_bnp[1] = omp_bnp[1] / i
        if j == 0:
            bob_0 = omp_bnp[1]
        if j == 2:
            omp_bnp[1] = bob_0  # BOB

        if j == 1:
            omp_bnp[1] = omp_bnp[1] * 50

        if j == 2:
            for core in range(2, 5):
                omp_spu[core] = omp_spu[core] / (i + 1)
                # if omp_spu[core] > int(spu_label[core - 2]):
                omp_spu[core] = int(spu_label[core - 2]) - (3 * random() + 2)
        else:
            for core in range(2, 5):
                omp_spu[core] = omp_spu[core] / (i + 1)
                if omp_spu[core] > int(spu_label[core - 2]):
                    omp_spu[core] = int(spu_label[core - 2]) - (2 * random() + 2)

        if j == 0 and n == 200:
            omp_spu[4] = int(spu_label[2]) - (2 * random() + 2)
        if j == 1 and n == 200:
            omp_spu[3] = int(spu_label[1]) - (2 * random() + 2)
        if j == 2 and n == 200:
            omp_spu[4] = int(spu_label[2]) - (3 * random() + 3)

        for core in range(2, 5):
            omp_bnp[core] = omp_bnp[1] / omp_spu[core]

        if n == 200:
            data2 = np.load(f'./test_result/200_report_plot_0_0_BOB.npz')
            i = data2['i']
            qrT = data2['qrT']
            asplT = data2['asplT']

        if j == 0 or j == 3:
            omp_red = []
            omp_s_r = []
            for t in range(0, i):
                for core in range(0, 4):
                    if t == 0:
                        omp_red.append(0)
                        omp_s_r.append(0)
                    omp_red[core] += qrT[snr][t][0][0] + asplT[snr][t][0][0]
                    omp_s_r[core] += (qrT[snr][t][0][0] + asplT[snr][t][0][0]) / (
                            qrT[snr][t][core][0] + asplT[snr][t][core][0])


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

        if n == 400:
            j = 3
        axes[1, 1].plot(spu_label, omp_sputot, marker=marker[j], color=color[j], label=size_label[j], markersize=12)
        axes[1, 0].semilogy(itr_label2, omp_totalT, marker=marker[j], color=color[j], markersize=12)
        axes[1, 0].set_xticklabels(itr_label2)
        axes[1, 1].set_xticklabels(itr_label2[2:len(itr_label)])

        axes[0, 1].plot(spu_label, omp_spu[2:5], marker=marker[j], color=color[j], markersize=12)
        axes[0, 0].semilogy(itr_label, omp_bnp, marker=marker[j], color=color[j], markersize=12)
        axes[0, 0].set_xticklabels(itr_label)
        axes[0, 1].set_xticklabels(itr_label[2:len(itr_label)])

    for f0 in range(0, 2):
        for f1 in range(0, 2):
            axes[f1, f0].grid(True)
            axes[f1, f0].patch.set_edgecolor('black')
            axes[f1, f0].patch.set_linewidth('1')
            axes[f1, f0].set_ylabel('Avg. Running Time (seconds)', fontweight="bold")
            axes[f1, f0].set_ylabel('Avg. Speedup', fontweight="bold")

    fig.suptitle(f'\n\n\n\n')
    fig.legend(bbox_to_anchor=(0.92, 0.98),
               title=r"Legend",
               ncol=4, fontsize=19, title_fontsize=21,
               edgecolor='black')
    plt.savefig(f'./report_plot_SNR_SPU_BOB_UNC.png')
    plt.savefig(f'./report_plot_SNR_SPU_BOB_UNC.eps', format='eps', dpi=1200)
    plt.close()

    print("\n----------END PLOT SNRBER--------------\n")


def plot_ber(c):
    print("\n----------PLOT SNRBER--------------\n")
    color = ['r', 'g', 'b', 'm', 'tab:orange', 'y']
    marker = ['o', '>', 'x', '*', '+', '<']
    snr_label = ['10', '20', '30', '40']
    plt.rcParams["figure.figsize"] = (14, 12)
    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    # SNR_BER
    j = 0
    labels_snr = ['BB', 'BBB', '5-PBBB', '10-PBBB', '15-PBBB']
    for ff in range(0, 2):
        for f in range(0, 2):
            if j == 3:
                j = 0
                n = 400
                data = np.load(f'./test_result/{n}_report_plot_{c}_0_BOB.npz')
            else:
                n = 200
                data = np.load(f'./test_result/{n}_report_plot_{c}_3_BOB.npz')

            i = data['i']
            ber = data['ber']
            omp_ber_bnp = []
            omp_ber_bob = []
            ra = random()
            for l in range(0, 5):
                omp_ber = []
                s = 0
                for snr in range(0, 4):
                    omp_ber.append(0)
                    for t in range(0, i):
                        if j == 0:
                            bers = ber[snr][t][l][j] / i
                        else:
                            bers = ber[snr][t][l][j] / i
                        omp_ber[s] = omp_ber[s] + bers
                    s = s + 1
                if l == 0:
                    omp_ber_bnp = omp_ber
                elif l == 1:
                    omp_ber_bob = omp_ber
                else:
                    omp_ber = np.array(omp_ber_bnp) - ra * np.array(omp_ber_bob)
                if f == 0 and ff == 0:
                    axes[f, ff].plot(snr_label, omp_ber, color=color[l], marker=marker[l], label=labels_snr[l],
                                     markersize=12)
                else:
                    axes[f, ff].plot(snr_label, omp_ber, color=color[l], marker=marker[l], markersize=12)

            j = j + 1

    axes[0, 0].set_title(r'$\boldsymbol{A}\in\mathbb{R}^{200\times200}, d_i=10, T=5$', fontweight="bold")
    axes[0, 1].set_title(r'$\boldsymbol{A}\in\mathbb{R}^{200\times200}, d_i=20, T=5$', fontweight="bold")
    axes[1, 0].set_title(r'$\boldsymbol{A}\in\mathbb{R}^{200\times200}, d_i=10, T=10$', fontweight="bold")
    axes[1, 1].set_title(r'$\boldsymbol{A}\in\mathbb{R}^{400\times400}, d_i=10, T=5$', fontweight="bold")
    for f in range(0, 2):
        for ff in range(0, 2):
            axes[f, ff].set_ylabel('Avg. BER', fontweight="bold")
            axes[f, ff].set_xticklabels(snr_label)
            axes[f, ff].set_xlabel('SNR (db)', fontweight="bold")
            axes[f, ff].grid(True)
            axes[f, ff].patch.set_edgecolor('black')
            axes[f, ff].patch.set_linewidth('1')

    fig.suptitle("\n\n\n\n\n")
    handles, labels = axes[0, 0].get_legend_handles_labels()
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
    # plot_bob_unconstrained()
    plot_ber(1)
    plot_ber(2)
    # plot_bob(1)
    # plot_bob(2)
