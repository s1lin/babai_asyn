import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
from textwrap import wrap


def plot_runtime(n, SNR, k, l_max, res, ber, tim, itr):
    plt.rcParams["figure.figsize"] = (20, 8)
    fig, axes2 = plt.subplots(2, 5, constrained_layout=True)
    color = ['r', 'g', 'b', 'y']
    marker = ['o', '+', 'x', '.']

    for j in range(0, 2):

        # SNR = int(lines[k].split(":")[1].split("\n")[0])
        # init_res = float(lines[k + 1].split(",")[0].split(":")[1].split("\n")[0])
        qam = 0 if j == 4 else 64
        axes2[j, 0].set_title('Iterations ' + str(qam) + '-QAM', fontsize=13)
        axes2[j, 1].set_title('Residual ' + str(qam) + '-QAM', fontsize=13)
        axes2[j, 2].set_title('BER ' + str(qam) + '-QAM', fontsize=13)
        axes2[j, 3].set_title('Avg Solve Time ' + str(qam) + '-QAM', fontsize=13)
        axes2[j, 4].set_title('Solver Speed Up ' + str(qam) + '-QAM', fontsize=13)

        axes2[j, 0].set_ylabel('Avg. Iterations', fontsize=13)
        axes2[j, 1].set_ylabel('Avg. Residual', fontsize=13)
        axes2[j, 2].set_ylabel('Avg. BER', fontsize=13)
        axes2[j, 3].set_ylabel('Avg. Solve Time (s)', fontsize=13)
        axes2[j, 4].set_ylabel('Solver Speed Up x times', fontsize=13)

        for x in range(0, 3):

            babai_res = res[x][0][j]
            babai_ber = ber[x][0][j]
            babai_stm = tim[x][0][j]
            block_res = res[x][1][j]
            block_ber = ber[x][1][j]
            block_stm = tim[x][1][j]

            omp_res = [babai_res, block_res]
            omp_ber = [babai_ber, block_ber]
            omp_stm = [babai_stm, block_stm]

            omp_itr = []

            for l in range(2, l_max):
                omp_res.append(res[x][l][j])
                omp_ber.append(ber[x][l][j])
                omp_itr.append(itr[x][l][j])
                omp_stm.append(tim[x][l][j])

            omp_spu = block_stm / omp_stm

            labels = ['$x_{init} = round(x_R)$', '$x_{init} = 0$', '$x_{init} = avg$']
            itr_label = ['NT-4', 'NT-8', 'NT-12', 'NT-16', 'NT-20']
            res_label = ['Babai', 'B-seq', 'NT-4', 'NT-8', 'NT-12', 'NT-16', 'NT-20']
            spu_label = ['NT-4', 'NT-8', 'NT-12', 'NT-16', 'NT-20']

            axes2[j, 0].plot(itr_label, omp_itr[0:len(itr_label)], color=color[x], marker=marker[x], label=labels[x])
            axes2[j, 1].plot(res_label, omp_res[0:len(res_label)], color=color[x], marker=marker[x], label=labels[x])
            axes2[j, 2].plot(res_label, omp_ber[0:len(res_label)], color=color[x], marker=marker[x], label=labels[x])
            axes2[j, 3].plot(res_label, omp_stm[0:len(res_label)], color=color[x], marker=marker[x], label=labels[x])
            axes2[j, 4].plot(spu_label, omp_spu[2:len(spu_label) + 2], color=color[x], marker=marker[x], label=labels[x])

            axes2[j, 0].legend(loc="lower right")
            axes2[j, 1].legend(loc="center left")
            axes2[j, 2].legend(loc="upper right")
            axes2[j, 3].legend(loc="center right")
            axes2[j, 4].legend(loc="lower right")

    title = 'Residual Convergence and Bit Error Rate for ' + str(SNR) \
            + '-SNR and4, 64-QAM with different number of threads and block size 16'
    fig.suptitle("\n".join(wrap(title, 60)), fontsize=15)
    plt.savefig('/home/shilei/CLionProjects/babai_asyn/babai_asyn_c++/src/example/' + str(n) + '_run_plot_' + str(SNR))
    plt.show()
    plt.close()

