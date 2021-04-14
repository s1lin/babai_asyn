import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
from textwrap import wrap


def plot_runtime(n, SNR, k, l_max, block_size, max_iter, is_qr, res, ber, tim, itr, ser_tim, d_s, proc_num):
    print("\n----------PLOT RUNTIME--------------\n")
    plt.rcParams["figure.figsize"] = (20, 8)
    fig, axes2 = plt.subplots(2, 5, constrained_layout=True)
    color = ['r', 'g', 'b', 'y']
    marker = ['o', '+', 'x', '.']

    for j in range(0, 2):

        # SNR = int(lines[k].split(":")[1].split("\n")[0])
        # init_res = float(lines[k + 1].split(",")[0].split(":")[1].split("\n")[0])
        qam = 4 if j == 0 else 64
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
            print(omp_stm)
            omp_spu = block_stm / omp_stm

            proc_num = proc_num.astype(int)
            labels = ['$x_{init} = round(x_R)$', '$x_{init} = 0$', '$x_{init} = avg$']
            itr_label = ['NT-' + str(proc) for proc in proc_num]
            # itr_label[len(itr_label) - 1] = itr_label[len(itr_label) - 1] + '-3'
            # itr_label[len(itr_label) - 2] = itr_label[len(itr_label) - 2] + '-2'
            # itr_label[len(itr_label) - 3] = itr_label[len(itr_label) - 3] + '-1'
            print(itr_label)
            res_label = ['Babai', 'B-seq'] + itr_label
            print(res_label)
            spu_label = itr_label

            axes2[j, 0].plot(itr_label, omp_itr[0:len(itr_label)], color=color[x], marker=marker[x], label=labels[x])
            axes2[j, 1].plot(res_label, omp_res[0:len(res_label)], color=color[x], marker=marker[x], label=labels[x])
            axes2[j, 2].plot(res_label, omp_ber[0:len(res_label)], color=color[x], marker=marker[x], label=labels[x])
            axes2[j, 3].semilogy(res_label, omp_stm[0:len(res_label)], color=color[x], marker=marker[x], label=labels[x])
            axes2[j, 4].semilogy(spu_label, omp_spu[2:len(spu_label) + 2], color=color[x], marker=marker[x], label=labels[x])
            axes2[j, 0].set_xticklabels(itr_label, rotation=45)
            axes2[j, 1].set_xticklabels(res_label, rotation=45)
            axes2[j, 2].set_xticklabels(res_label, rotation=45)
            axes2[j, 3].set_xticklabels(res_label, rotation=45)
            axes2[j, 4].set_xticklabels(spu_label, rotation=45)

            axes2[j, 0].legend(loc="lower right")
            axes2[j, 1].legend(loc="center left")
            axes2[j, 2].legend(loc="upper right")
            axes2[j, 3].legend(loc="center right")
            axes2[j, 4].legend(loc="lower right")

    title = 'Residual Convergence and Bit Error Rate for ' + str(SNR) \
            + '-SNR and 4, 64-QAM with different number of threads and block size ' + str(block_size)
    if is_qr == 0:
        title += ' with LLL reduction'

    fig.suptitle("\n".join(wrap(title, 60)), fontsize=15)

    plt.savefig('./' + str(n) + '_report_plot_' + str(SNR) + '_' + str(block_size) + '_' + str(is_qr))
    plt.close()
    print("\n----------END PLOT RUNTIME--------------\n")
    plot_first_block(n, SNR, k, block_size, ser_tim, is_qr, d_s)


def plot_first_block(n, SNR, k, block_size, ser_tim, is_qr, d_s):
    print("\n----------PLOT BLOCK TIME--------------\n")
    plt2.rcParams["figure.figsize"] = (18, 8)
    fig, axes2 = plt2.subplots(2, 2, constrained_layout=True)
    color = ['r', 'g', 'b', 'y']
    marker = ['o', '+', 'x', '.']
    print(d_s)
    for j in range(0, 2):

        # SNR = int(lines[k].split(":")[1].split("\n")[0])
        # init_res = float(lines[k + 1].split(",")[0].split(":")[1].split("\n")[0])
        qam = 4 if j == 0 else 64
        axes2[j, 0].set_title(str(qam) + '-QAM', fontsize=13)
        axes2[j, 1].set_title(str(qam) + '-QAM', fontsize=13)
        axes2[j, 0].set_ylabel('Percentage of Time', fontsize=13)
        axes2[j, 1].set_ylabel('Percentage of Search Iteration', fontsize=13)
        axes2[j, 0].set_xlabel('The i-th block', fontsize=13)
        axes2[j, 1].set_xlabel('The i-th block', fontsize=13)

        for x in range(0, 3): #only the $x_{init} = 0
            tim = []
            itr = []
            for l in range(0, len(d_s)):
                tim.append(ser_tim[x][l][j])
                itr.append(ser_tim[x][l + len(d_s)][j])

            labels = ['$x_{init} = round(x_R)$', '$x_{init} = 0$', '$x_{init} = avg$']#range(0, d_s)#

            print(tim, len(tim))

            #explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')
            # axes2[j, 0].pie(tim, labels=range(0, d_s), autopct='%1.1f%%', shadow=True, startangle=90)
            axes2[j, 0].semilogy(d_s, tim, color=color[x], marker=marker[x], label=labels[x])
            axes2[j, 0].set_xticks(d_s)
            axes2[j, 0].legend(loc="upper right")
            axes2[j, 1].semilogy(d_s, itr, color=color[x], marker=marker[x], label=labels[x])
            axes2[j, 1].set_xticks(d_s)
            axes2[j, 1].legend(loc="upper right")

    title = 'ILS Sequential Solving Time Per block for ' + str(SNR) \
            + '-SNR and 4, 64-QAM and problem size ' + str(n) + ', block size ' + str(block_size)
    if is_qr == 0:
        title += ' with LLL reduction'
    # fig.suptitle("\n".join(wrap(title, 85)), fontsize=15)
    fig.suptitle(title, fontsize=15)
    plt2.savefig('./' + str(n) + '_block_time_' + str(SNR) + '_' + str(block_size) + '_' + str(is_qr))
    plt2.close()
    print("\n----------END PLOT BLOCK TIME--------------\n")