import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import numpy as np
from textwrap import wrap


def plot_residual(n, SNRs, f):
    for j in range(0, 2):
        k = 6
        SNR = SNRs[0]
        file = open(str(n) + '_' + str(f[j]) + '_' + str(SNR) + '_res.out', 'r')
        lines = file.readlines()
        plt.rcParams["figure.figsize"] = (13, 13)
        fig, axes = plt.subplots(3, 3, constrained_layout=True)
        print(lines[k])
        SNR = int(lines[k].split(":")[1].split("\n")[0])
        init_res = 0  # float(lines[k + 1].split(",")[0].split(":")[1].split("\n")[0])

        k = k + 4
        for init in range(-1, 2):
            line_str = lines[k].split(",")
            print(line_str)
            init_value = lines[k].split(",")[0].split("\n")[0]
            ser_res = float(lines[k + 1].split(",")[2].split(":")[1])
            ser_ber = float(lines[k + 1].split(",")[3].split(":")[1])
            k = k + 2
            print(line_str)
            color = ['r', 'g', 'b', 'y']
            marker = ['o', '+', 'x', '.']
            for m in range(0, 4):
                line_str = lines[k].split(",")
                print(line_str)
                num_thread = line_str[1]
                index = 2
                res = []
                ber = []
                diff = []
                while index < len(line_str) - 1:
                    diff.append(float(line_str[index].split("=")[1]))
                    res.append(float(line_str[index + 1].split("=")[1]))
                    ber.append(float(line_str[index + 2].split("=")[1]))
                    # print(float(line_str[index + 2].split("=")[1]))
                    index = index + 3
                if SNRs[0] == 35:
                    if init + 1 == 0:
                        axes[0, init + 1].semilogy(range(0, 10), np.array(res)[0:10], color=color[m],
                                                   marker=marker[m], label='num_thread = ' + str(num_thread))

                    else:
                        axes[0, init + 1].semilogy(range(0, 10), np.array(res)[0:10], color=color[m],
                                                   marker=marker[m])

                    axes[1, init + 1].plot(range(0, 10), np.array(ber)[0:10], color=color[m], marker=marker[m])
                    axes[2, init + 1].plot(range(0, 10), np.array(diff)[0:10], color=color[m], marker=marker[m])
                else:
                    if init + 1 == 0:
                        axes[0, init + 1].plot(range(0, 99), np.array(res)[0:99], color=color[m],
                                               label='num_thread = ' + str(num_thread))
                    else:
                        axes[0, init + 1].plot(range(0, 99), np.array(res)[0:99], color=color[m])

                    axes[1, init + 1].plot(range(1, 99), np.array(ber)[1:99], color=color[m])
                    axes[2, init + 1].plot(range(0, 99), np.array(diff)[0:99], color=color[m])
                    axes[1, init + 1].set_ylim(0.2, 0.6)

                k = k + 1
            # axes[2, init + 1].set_ylim(-99, 4200)
            if init + 1 == 0:
                axes[0, init + 1].axhline(y=init_res, xmin=0.0, xmax=1.0, color='c', linewidth=2.5,
                                          linestyle='dotted', label='True parameter')
                axes[0, init + 1].axhline(y=ser_res, xmin=0.0, xmax=1.0, color='m', linewidth=2.5, linestyle='dotted',
                                          label='Block Residual')
                axes[1, init + 1].axhline(y=ser_ber, xmin=0.0, xmax=1.0, color='k', linewidth=2.5, linestyle='dotted',
                                          label='Block BER')
            else:
                axes[0, init + 1].axhline(y=init_res, xmin=0.0, xmax=1.0, color='c', linewidth=2.5, linestyle='dotted')
                axes[0, init + 1].axhline(y=ser_res, xmin=0.0, xmax=1.0, color='m', linewidth=2.5, linestyle='dotted', )
                axes[1, init + 1].axhline(y=ser_ber, xmin=0.0, xmax=1.0, color='k', linewidth=2.5, linestyle='dotted', )

            # axes[0, init + 1].legend(loc="upper right")
            # axes[1, init + 1].legend(loc="center right")

            title_1 = 'Residual Convergence'
            title_2 = 'BER Convergence'
            title_3 = 'Difference Convergence'

            if init == 1:
                title_1 = title_1 + ' $x_{init} = avg$'
                title_2 = title_2 + ' $x_{init} = avg$'
                title_3 = title_3 + ' $x_{init} = avg$'
            elif init == -1:
                title_1 = title_1 + ' $x_{init} = round(x_R)$'
                title_2 = title_2 + ' $x_{init} = round(x_R)$'
                title_3 = title_3 + ' $x_{init} = round(x_R)$'
            elif init == 0:
                title_1 = title_1 + ' $x_{init} = 0$'
                title_2 = title_2 + ' $x_{init} = 0$'
                title_3 = title_3 + ' $x_{init} = 0$'

            axes[0, init + 1].set_title(title_1, fontsize=13)
            axes[1, init + 1].set_title(title_2, fontsize=13)
            axes[2, init + 1].set_title(title_3, fontsize=13)
            axes[2, init + 1].set_xlabel('Number of Iterations')
        title = 'Residual Convergence, Bit Error Rate, and differences for ' + str(SNRs[0]) + '-SNR and ' \
                + str(pow(4, f[j])) + '-QAM with different number of threads and block size 16'

        axes[0, 0].set_ylabel('Residual', fontsize=13)
        axes[1, 0].set_ylabel('Bit Error Rate', fontsize=13)
        axes[2, 0].set_ylabel('Difference of $z_{j+1}-z_{j}$', fontsize=13)
        fig.suptitle("\n".join(wrap(title, 60)), fontsize=15)
        fig.legend(loc='center right', title='Legend')
        fig.subplots_adjust(right=0.85)
        plt.savefig('./' + str(n) + '_res_plot_' + str(f[j]) + '_' + str(SNRs[0]))
        plt.close()
        k = k + 5


def plot_runtime(n, SNRs, f):
    plt2.rcParams["figure.figsize"] = (20, 8)
    fig, axes2 = plt2.subplots(2, 5, constrained_layout=True)
    color = ['r', 'g', 'b', 'y']
    marker = ['o', '+', 'x', '.']
    index = ['Iter', 'Time', 'Res', 'BER']
    # omp_tab = pd.DataFrame(np.random.randn(4, 7),
    #                        columns=['Babai', 'B-seq', 'NT-10', 'NT-20', 'NT-30', 'NT-40'])

    for j in range(0, 2):
        k = 5
        SNR = SNRs[0]
        file = open(str(n) + '_' + str(f[j]) + '_' + str(SNR) + '_plot.out', 'r')
        lines = file.readlines()
        print(lines[k].split(","))
        # SNR = int(lines[k].split(":")[1].split("\n")[0])
        # init_res = float(lines[k + 1].split(",")[0].split(":")[1].split("\n")[0])
        k = k + 9

        axes2[j, 0].set_title('Iterations ' + str(pow(4, f[j])) + '-QAM', fontsize=13)
        axes2[j, 1].set_title('Residual ' + str(pow(4, f[j])) + '-QAM', fontsize=13)
        axes2[j, 2].set_title('BER ' + str(pow(4, f[j])) + '-QAM', fontsize=13)
        axes2[j, 3].set_title('Avg Solve Time ' + str(pow(4, f[j])) + '-QAM', fontsize=13)
        axes2[j, 4].set_title('Solver Speed Up ' + str(pow(4, f[j])) + '-QAM', fontsize=13)

        axes2[j, 0].set_ylabel('Avg. Iterations', fontsize=13)
        axes2[j, 1].set_ylabel('Avg. Residual', fontsize=13)
        axes2[j, 2].set_ylabel('Avg. BER', fontsize=13)
        axes2[j, 3].set_ylabel('Avg. Solve Time (s)', fontsize=13)
        axes2[j, 4].set_ylabel('Solver Speed Up x times', fontsize=13)

        for x in range(0, 3):
            print(lines[k].split(","))
            init_value = int(lines[k].split(":")[1].split("\n")[0])

            k = k + 1
            print(lines[k].split(","))
            babai_res = float(lines[k].split(",")[1].split(":")[1])
            babai_ber = float(lines[k].split(",")[2].split(":")[1])
            babai_stm = float(lines[k].split(",")[3].split(":")[1].split("s")[0])
            babai_qrt = float(lines[k].split(",")[4].split(":")[1])
            babai_tim = float(lines[k].split(",")[5].split(":")[1].split("s")[0])

            k = k + 1
            print(lines[k].split(","))
            block_res = float(lines[k].split(",")[2].split(":")[1])
            block_ber = float(lines[k].split(",")[3].split(":")[1])
            block_stm = float(lines[k].split(",")[4].split(":")[1].split("s")[0])
            block_qrt = float(lines[k].split(",")[5].split(":")[1])
            block_tim = float(lines[k].split(",")[6].split(":")[1].split("s")[0])
            k = k + 1
            print(lines[k].split(","))
            omp_res = [babai_res, block_res]
            omp_ber = [babai_ber, block_ber]
            omp_stm = [babai_stm, block_stm]
            omp_qrt = [babai_qrt, block_qrt]
            omp_tim = [babai_tim, block_tim]
            omp_spu = []
            omp_qrs = []
            omp_tsu = []
            omp_itr = []
            for l in range(0, 4):
                print(lines[k].split(","))
                omp_res.append(float(lines[k].split(",")[2].split(":")[1]))
                omp_ber.append(float(lines[k].split(",")[3].split(":")[1]))
                omp_itr.append(float(lines[k].split(",")[4].split(":")[1]))

                omp_tim.append(float(lines[k].split(",")[11].split(":")[1].split("s")[0]))
                omp_qrt.append(float(lines[k].split(",")[9].split(":")[1].split("s")[0]))
                omp_stm.append(float(lines[k].split(",")[6].split(":")[1].split("s")[0]))

                omp_spu.append(float(lines[k].split(",")[7].split(":")[1]))
                omp_qrs.append(float(lines[k].split(",")[10].split(":")[1]))
                omp_tsu.append(float(lines[k].split(",")[12].split(":")[1].split("\n")[0]))
                k = k + 1

            labels = ['$x_{init} = round(x_R)$', '$x_{init} = 0$', '$x_{init} = avg$']

            axes2[j, 0].plot(['NT-10', 'NT-20', 'NT-30', 'NT-40'], omp_itr, color=color[x], marker=marker[x], label=labels[init_value + 1])
            axes2[j, 1].plot(['Babai', 'B-seq', 'NT-10', 'NT-20', 'NT-30', 'NT-40'], omp_res, color=color[x], marker=marker[x], label=labels[init_value + 1])
            axes2[j, 2].plot(['Babai', 'B-seq', 'NT-10', 'NT-20', 'NT-30', 'NT-40'], omp_ber, color=color[x], marker=marker[x], label=labels[init_value + 1])
            axes2[j, 3].plot(['Babai', 'B-seq', 'NT-10', 'NT-20', 'NT-30', 'NT-40'], omp_stm, color=color[x], marker=marker[x], label=labels[init_value + 1])
            axes2[j, 4].plot(['NT-10', 'NT-20', 'NT-30', 'NT-40'], omp_spu, color=color[x], marker=marker[x], label=labels[init_value + 1])

            if SNR == 35 and f[0] == 1:
                axes2[0, 2].set_ylim(-0.01, 0.01)

            axes2[j, 0].legend(loc="lower right")
            axes2[j, 1].legend(loc="upper left")
            axes2[j, 2].legend(loc="lower right")
            axes2[j, 3].legend(loc="center right")
            axes2[j, 4].legend(loc="upper right")

            k = k + 3

    title = 'Residual Convergence and Bit Error Rate for ' + str(SNRs[0]) + '-SNR and ' \
            + str(pow(4, f[0])) + '-QAM with different number of threads and block size 16'
    fig.suptitle("\n".join(wrap(title, 60)), fontsize=15)
    plt.savefig('./' + str(n) + '_run_plot_' + str(SNRs[0]) + '_' + str(f[0]))
    plt.close()


def plot_qrtime(n, SNRs, f):
    plt2.rcParams["figure.figsize"] = (12, 12)
    fig, axes2 = plt2.subplots(2, 5, constrained_layout=True)
    color = ['r', 'g', 'b', 'y']
    marker = ['o', '+', 'x', '.']
    index = ['Iter', 'Time', 'Res', 'BER']
    # omp_tab = pd.DataFrame(np.random.randn(4, 7),
    #                        columns=['Babai', 'B-seq', 'NT-10', 'NT-20', 'NT-30', 'NT-40'])

    for j in range(0, 1):
        k = 5
        SNR = SNRs[0]
        file = open(str(n) + '_' + str(f[j]) + '_' + str(SNR) + '_plot.out', 'r')
        lines = file.readlines()
        print(lines[k].split(","))
        # SNR = int(lines[k].split(":")[1].split("\n")[0])
        # init_res = float(lines[k + 1].split(",")[0].split(":")[1].split("\n")[0])
        k = k + 9

        axes2[j + 0, 0].set_title('Iterations ' + str(pow(4, f[j])) + '-QAM', fontsize=13)
        axes2[j + 0, 1].set_title('Residual ' + str(pow(4, f[j])) + '-QAM', fontsize=13)
        axes2[j + 0, 2].set_title('BER ' + str(pow(4, f[j])) + '-QAM', fontsize=13)
        axes2[j + 1, 0].set_title('Avg Solve Time ' + str(pow(4, f[j])) + '-QAM', fontsize=13)
        axes2[j + 1, 1].set_title('Total QR Time ' + str(pow(4, f[j])) + '-QAM', fontsize=13)
        axes2[j + 1, 2].set_title('Total Run Time ' + str(pow(4, f[j])) + '-QAM', fontsize=13)
        axes2[j + 2, 0].set_title('Solver Speed Up ' + str(pow(4, f[j])) + '-QAM', fontsize=13)
        axes2[j + 2, 1].set_title('QR Speed Up' + str(pow(4, f[j])) + '-QAM', fontsize=13)
        axes2[j + 2, 2].set_title('Total Speed Up' + str(pow(4, f[j])) + '-QAM', fontsize=13)

        axes2[j + 0, 0].set_ylabel('Avg. Iterations', fontsize=13)
        axes2[j + 0, 1].set_ylabel('Avg. Residual', fontsize=13)
        axes2[j + 0, 2].set_ylabel('Avg. BER', fontsize=13)
        axes2[j + 1, 0].set_ylabel('Avg. Solve Time (s)', fontsize=13)
        axes2[j + 1, 1].set_ylabel('Total QR Time (s)', fontsize=13)
        axes2[j + 1, 2].set_ylabel('Total Run Time (s)', fontsize=13)
        axes2[j + 2, 0].set_ylabel('Solver Speed Up x times', fontsize=13)
        axes2[j + 2, 1].set_ylabel('QR Speed Up x times', fontsize=13)
        axes2[j + 2, 2].set_ylabel('Total Speed Up x times', fontsize=13)

        axes2[j + 2, 0].set_xlabel('Methods', fontsize=13)
        axes2[j + 2, 1].set_xlabel('Methods', fontsize=13)
        axes2[j + 2, 2].set_xlabel('Methods', fontsize=13)

        for x in range(0, 3):
            print(lines[k].split(","))
            init_value = int(lines[k].split(":")[1].split("\n")[0])

            k = k + 1
            print(lines[k].split(","))
            babai_res = float(lines[k].split(",")[1].split(":")[1])
            babai_ber = float(lines[k].split(",")[2].split(":")[1])
            babai_stm = float(lines[k].split(",")[3].split(":")[1].split("s")[0])
            babai_qrt = float(lines[k].split(",")[4].split(":")[1])
            babai_tim = float(lines[k].split(",")[5].split(":")[1].split("s")[0])

            k = k + 1
            print(lines[k].split(","))
            block_res = float(lines[k].split(",")[2].split(":")[1])
            block_ber = float(lines[k].split(",")[3].split(":")[1])
            block_stm = float(lines[k].split(",")[4].split(":")[1].split("s")[0])
            block_qrt = float(lines[k].split(",")[5].split(":")[1])
            block_tim = float(lines[k].split(",")[6].split(":")[1].split("s")[0])
            k = k + 1
            print(lines[k].split(","))
            omp_res = [babai_res, block_res]
            omp_ber = [babai_ber, block_ber]
            omp_stm = [babai_stm, block_stm]
            omp_qrt = [babai_qrt, block_qrt]
            omp_tim = [babai_tim, block_tim]
            omp_spu = []
            omp_qrs = []
            omp_tsu = []
            omp_itr = []
            for l in range(0, 4):
                print(lines[k].split(","))
                omp_res.append(float(lines[k].split(",")[2].split(":")[1]))
                omp_ber.append(float(lines[k].split(",")[3].split(":")[1]))
                omp_itr.append(float(lines[k].split(",")[4].split(":")[1]))

                omp_tim.append(float(lines[k].split(",")[11].split(":")[1].split("s")[0]))
                omp_qrt.append(float(lines[k].split(",")[9].split(":")[1].split("s")[0]))
                omp_stm.append(float(lines[k].split(",")[6].split(":")[1].split("s")[0]))

                omp_spu.append(float(lines[k].split(",")[7].split(":")[1]))
                omp_qrs.append(float(lines[k].split(",")[10].split(":")[1]))
                omp_tsu.append(float(lines[k].split(",")[12].split(":")[1].split("\n")[0]))
                k = k + 1

            labels = ['$x_{init} = round(x_R)$', '$x_{init} = 0$', '$x_{init} = avg$']

            axes2[j + 0, 0].plot(['NT-10', 'NT-20', 'NT-30', 'NT-40'], omp_itr, color=color[x], marker=marker[x],
                                 label=labels[init_value + 1])
            axes2[j + 0, 1].plot(['Babai', 'B-seq', 'NT-10', 'NT-20', 'NT-30', 'NT-40'], omp_res, color=color[x],
                                 marker=marker[x], label=labels[init_value + 1])
            axes2[j + 0, 2].plot(['Babai', 'B-seq', 'NT-10', 'NT-20', 'NT-30', 'NT-40'], omp_ber, color=color[x],
                                 marker=marker[x], label=labels[init_value + 1])

            axes2[j + 1, 0].semilogy(['Babai', 'B-seq', 'NT-10', 'NT-20', 'NT-30', 'NT-40'], omp_stm, color=color[x],
                                     marker=marker[x], label=labels[init_value + 1])
            if init_value + 1 == 0:
                axes2[j + 1, 1].plot(['Babai', 'B-seq', 'NT-10', 'NT-20', 'NT-30', 'NT-40'], omp_qrt, color=color[x],
                                     marker=marker[x], label=labels[init_value + 1])
            axes2[j + 1, 2].semilogy(['Babai', 'B-seq', 'NT-10', 'NT-20', 'NT-30', 'NT-40'], omp_tim, color=color[x],
                                     marker=marker[x], label=labels[init_value + 1])

            axes2[j + 2, 0].semilogy(['NT-10', 'NT-20', 'NT-30', 'NT-40'], omp_spu, color=color[x], marker=marker[x],
                                     label=labels[init_value + 1])
            if init_value + 1 == 0:
                axes2[j + 2, 1].plot(['NT-10', 'NT-20', 'NT-30', 'NT-40'], omp_qrs, color=color[x], marker=marker[x],
                                     label=labels[init_value + 1])
            axes2[j + 2, 2].plot(['NT-10', 'NT-20', 'NT-30', 'NT-40'], omp_tsu, color=color[x], marker=marker[x],
                                 label=labels[init_value + 1])

            if SNR == 35 and f[0] == 1:
                axes2[0, 2].set_ylim(-0.01, 0.01)

            axes2[j + 0, 0].legend(loc="lower right")
            axes2[j + 0, 1].legend(loc="upper left")
            axes2[j + 0, 2].legend(loc="lower right")
            axes2[j + 1, 0].legend(loc="center right")
            # axes2[j + 1, 1].legend(loc="upper right")
            axes2[j + 1, 2].legend(loc="upper right")
            axes2[j + 2, 0].legend(loc="lower right")
            # axes2[j + 2, 1].legend(loc="upper right")
            axes2[j + 2, 2].legend(loc="lower right")

            k = k + 3

    title = 'Residual Convergence and Bit Error Rate for ' + str(SNRs[0]) + '-SNR and ' \
            + str(pow(4, f[0])) + '-QAM with different number of threads and block size 16'
    fig.suptitle("\n".join(wrap(title, 60)), fontsize=15)
    plt.savefig('./' + str(n) + '_run_plot_' + str(SNRs[0]) + '_' + str(f[0]))
    plt.close()


def plot_res(n):
    SNRs = [35]
    f = [1, 3]
    # file = open(str(n) + '_' + str(f) + '_res.out', 'r')
    # plot_residual(n, SNRs, f)
    plot_runtime(n, SNRs, f)


if __name__ == "__main__":
    plot_res(1024)
