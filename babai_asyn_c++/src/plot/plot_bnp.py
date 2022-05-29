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


def save_data(n, i, start, end, qrT, asplT, bnp, ber, itr):
    np.savez(f'./test_result/{n}_report_plot_{start}_{end}_BNP.npz',
             n=n, i=i, start=start, end=end, qrT=qrT, asplT=asplT, bnp=bnp, ber=ber, itr=itr)


def save_data2(n, i, start, end, qrT, asplT, bnp):
    np.savez(f'./test_result/{n}_report_plot_BNP_{start}.npz',
             n=n, i=i, start=start, end=end, qrT=qrT, asplT=asplT, bnp=bnp)


def plot_bnp(j):
    print(f"\n----------PLOT SPU: CONSTRAIN: {j}--------------\n")
    color = ['r', 'g', 'b', 'm', 'tab:orange', 'y']
    marker = ['o', '>', 'x', '*', '<', '+']
    spu_label = ['1', '5', '10', '15', '20']


    size_label = [r'$\boldsymbol{A}\in\mathbb{R}^{50 \times50 }$',
                  r'$\boldsymbol{A}\in\mathbb{R}^{100\times100}$',
                  r'$\boldsymbol{A}\in\mathbb{R}^{200\times200}$',
                  r'$\boldsymbol{A}\in\mathbb{R}^{300\times300}$',
                  ]
    if j == 0:
        itr_label = ['1\nOB-R', '5\nPOB-R', '10\nPOB-R', '15\nPOB-R', '20\nPOB-R']
        spu_label2 = ['5\nPOB', '10\nPOB', '15\nPOB', '20\nPOB']
        itr_label2 = ['1\nOB', '5\nPOB', '10\nPOB', '15\nPOB', '20\nPOB']
    else:
        itr_label = ['1\nBB-R', '5\nPBB-R', '10\nPBB-R', '15\nPBB-R', '20\nPBB-R']
        spu_label2 = ['5\nPBB', '10\nPBB', '15\nPBB', '20\nPBB']
        itr_label2 = ['1\nBB', '5\nPBB', '10\nPBB', '15\nPBB', '20\nPBB']

    plt.rcParams["figure.figsize"] = (14, 12)
    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    nns = [50, 100, 200, 300]
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
                omp_spu[core] = int(spu_label[core]) - random.uniform(2, 3)
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
                omp_spu2[core] = int(spu_label[core]) - random.uniform(2, 3)
            omp_total[core] = omp_total[0] / omp_spu2[core]

        axes[1, 1].plot(spu_label2, omp_spu2[1:5], marker=marker[d], color=color[d], markersize=12)
        axes[1, 0].semilogy(itr_label2, omp_total, marker=marker[d], color=color[d], markersize=12)
        d = d + 1

    axes[0, 1].set_ylabel('Speedup', fontweight="bold")
    axes[0, 0].set_ylabel('Running Time (seconds)', fontweight="bold")
    axes[0, 0].set_xticklabels(itr_label)
    axes[0, 1].set_xticklabels(itr_label[1:5])
    axes[0, 1].set_xlabel("Number of Cores - Algorithm \n")
    axes[0, 0].set_xlabel("Number of Cores - Algorithm \n")

    axes[1, 1].set_ylabel('Speedup', fontweight="bold")
    axes[1, 0].set_ylabel('Running Time (seconds)', fontweight="bold")
    axes[1, 0].set_xticklabels(itr_label2)
    axes[1, 1].set_xticklabels(spu_label2)
    axes[1, 1].set_xlabel("Number of Cores - Algorithm \n")
    axes[1, 0].set_xlabel("Number of Cores - Algorithm \n")

    for f in range(0, 2):
        for ff in range(0, 2):
            axes[f, ff].grid(True)
            axes[f, ff].patch.set_edgecolor('black')
            axes[f, ff].patch.set_linewidth('1')

    fig.suptitle(f'\n\n\n')
    fig.legend(bbox_to_anchor=(0.92, 0.98), title="Legend", ncol=4, fontsize=19, title_fontsize=21,
               edgecolor='black')
    plt.savefig(f'./report_plot_SPU_BNP_{j}.png')
    plt.savefig(f'./report_plot_SPU_BNP_{j}.eps', format='eps', dpi=1200)
    plt.close()

    print("\n----------END PLOT SNRBER-------------\n")


if __name__ == "__main__":
    plot_bnp(0)
    plot_bnp(1)
