from textwrap import wrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import pandas as pd


def plot_residual(n, f, file):
    lines = file.readlines()
    k = 6
    for j in range(1, 3):
        plt.rcParams["figure.figsize"] = (13, 7)
        fig, axes = plt.subplots(2, 3, constrained_layout=True)
        print(lines[k])
        SNR = int(lines[k].split(":")[1].split("\n")[0])
        init_res = float(lines[k + 1].split(":")[1].split("\n")[0])

        k = k + 4
        for init in range(-1, 2):
            line_str = lines[k].split(",")
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
                while index < len(line_str) - 1:
                    res.append(float(line_str[index].split("=")[1]))
                    ber.append(float(line_str[index + 1].split("=")[1]))
                    index = index + 2

                if init + 1 == 0:
                    axes[0, init + 1].plot(range(0, len(res), 5), np.array(res)[1:len(res):5], color=color[m],
                                               marker=marker[m], label='num_thread = ' + str(num_thread))

                else:
                    axes[0, init + 1].plot(range(0, len(res), 5), np.array(res)[1:len(res):5], color=color[m],
                                               marker=marker[m])
                    # axes[0, init + 1].axhline(y=init_res, xmin=0.0, xmax=1.0, color='r', linewidth=3,
                    #                           linestyle='dotted')
                    # axes[0, init + 1].axhline(y=ser_res, xmin=0.0, xmax=1.0, color='y', linewidth=3,
                    #                           linestyle='dotted')

                axes[1, init + 1].plot(range(0, len(ber), 5), np.array(ber)[1:len(ber):5], color=color[m],
                                       marker=marker[m])

                k = k + 1

            if init + 1 == 0:
                axes[0, init + 1].axhline(y=init_res, xmin=0.0, xmax=1.0, color='c', linewidth=3,
                                          linestyle='dotted',
                                          label='True parameter')
                axes[0, init + 1].axhline(y=ser_res, xmin=0.0, xmax=1.0, color='m', linewidth=3, linestyle='dotted',
                                          label='Block Residual')
                axes[1, init + 1].axhline(y=ser_ber, xmin=0.0, xmax=1.0, color='k', linewidth=3, linestyle='dotted',
                                          label='Block BER')
            else:
                axes[0, init + 1].axhline(y=init_res, xmin=0.0, xmax=1.0, color='c', linewidth=3, linestyle='dotted')
                axes[0, init + 1].axhline(y=ser_res, xmin=0.0, xmax=1.0, color='m', linewidth=3, linestyle='dotted', )
                axes[1, init + 1].axhline(y=ser_ber, xmin=0.0, xmax=1.0, color='k', linewidth=3, linestyle='dotted', )

            # axes[0, init + 1].legend(loc="upper right")
            # axes[1, init + 1].legend(loc="center right")

            title_1 = 'Residual Convergence'
            title_2 = 'BER'

            if init_value in '1':
                title_1 = title_1 + ' $x_{init} = avg$'
                title_2 = title_2 + ' $x_{init} = avg$'
            elif init_value in '-1':
                title_1 = title_1 + ' $x_{init} = round(x_R)$'
                title_2 = title_2 + ' $x_{init} = round(x_R)$'
            elif init_value in '0':
                title_1 = title_1 + ' $x_{init} = 0$'
                title_2 = title_2 + ' $x_{init} = 0$'

            axes[0, init + 1].set_title(title_1, fontsize=13)
            axes[1, init + 1].set_title(title_2, fontsize=13)
            axes[1, init + 1].set_xlabel('Number of Iterations')
        title = 'Residual Convergence and Bit Error Rate for ' + str(SNR) + '-SNR and ' \
                + str(pow(4, f)) + '-QAM with different number of threads and block size 16'

        axes[0, 0].set_ylabel('Residual')
        axes[1, 0].set_ylabel('Bit Error Rate')
        fig.suptitle("\n".join(wrap(title, 60)), fontsize=15)
        fig.legend(loc='center right', title='Legend')
        fig.subplots_adjust(right=0.85)
        plt.savefig('./' + str(n) + '_res_plot_' + str(f) + '_' + str(SNR))
        plt.close()
        k = k + 5


def plot_runtime(n, f, stop, file):
    lines = file.readlines()
    plt2.rcParams["figure.figsize"] = (15, 8)
    fig, axes2 = plt2.subplots(2, 4, constrained_layout=True)
    color = ['r', 'g', 'b', 'y']
    marker = ['o', '+', 'x', '.']
    index = ['Iter', 'Time', 'Res', 'BER']
    omp_tab = pd.DataFrame(np.random.randn(4, 6),
                           columns=['Babai', 'B-seq', 'NT-6', 'NT-12', 'NT-24', 'NT-48'])
    k = 6
    SNRs = [15, 35]
    for j in range(0, 2):
        SNR = int(lines[k].split(":")[1].split("\n")[0])
        init_res = float(lines[k + 1].split(":")[1].split("\n")[0])
        k = k + 6

        axes2[j, 0].set_title('Iterations ' + str(SNRs[j]) + '-SNR', fontsize=13)
        axes2[j, 1].set_title('Residual ' + str(SNRs[j]) + '-SNR', fontsize=13)
        axes2[j, 2].set_title('BER ' + str(SNRs[j]) + '-SNR', fontsize=13)
        axes2[j, 3].set_title('Running Time ' + str(SNRs[j]) + '-SNR', fontsize=13)

        axes2[j, 0].set_ylabel('Avg. Iterations')
        axes2[j, 1].set_ylabel('Avg. Residual')
        axes2[j, 2].set_ylabel('Avg. BER')
        axes2[j, 3].set_ylabel('Avg. Running Time')

        for x in range(0, 3):
            print(lines[k].split(","))
            init_value = int(lines[k].split(":")[1].split("\n")[0])

            k = k + 2
            print(lines[k].split(","))
            babai_res = float(lines[k].split(",")[2].split(":")[1])
            babai_ber = float(lines[k].split(",")[3].split(":")[1])
            babai_tim = float(lines[k].split(",")[4].split(":")[1].split("s")[0])
            k = k + 2
            print(lines[k].split(","))
            block_res = float(lines[k].split(",")[2].split(":")[1])
            block_ber = float(lines[k].split(",")[3].split(":")[1])
            block_tim = float(lines[k].split(",")[4].split(":")[1].split("s")[0])
            k = k + 6
            print(lines[k].split(","))
            omp_res = [init_res, babai_res, block_res]
            omp_ber = [babai_ber, block_ber]
            omp_tim = [babai_tim, block_tim]
            omp_tab.iloc[0, 0] = '/'
            omp_tab.iloc[0, 1] = '/'
            omp_tab.iloc[1, 0] = babai_tim
            omp_tab.iloc[1, 1] = block_tim
            omp_tab.iloc[2, 1] = babai_ber
            omp_tab.iloc[2, 1] = block_ber
            omp_itr = []
            for l in range(0, 4):
                print(lines[k].split(","))
                omp_res.append(float(lines[k].split(",")[2].split(":")[1]))
                omp_itr.append(float(lines[k].split(",")[4].split(":")[1]))
                omp_ber.append(float(lines[k].split(",")[3].split(":")[1]))
                omp_tim.append(float(lines[k].split(",")[5].split(":")[1].split("s")[0]))
                omp_tab.iloc[0, l + 2] = float(lines[k].split(",")[4].split(":")[1])
                omp_tab.iloc[2, l + 2] = float(lines[k].split(",")[3].split(":")[1])
                omp_tab.iloc[1, l + 2] = float(lines[k].split(",")[5].split(":")[1].split("s")[0])

                k = k + 1

            labels = ['$x_{init} = round(x_R)$', '$x_{init} = 0$', '$x_{init} = avg$']

            axes2[j, 0].plot(['NT-6', 'NT-12', 'NT-24', 'NT-48'], omp_itr,
                             color=color[x], marker=marker[x], label=labels[init_value + 1])
            axes2[j, 1].plot(['True', 'Babai', 'B-seq', 'NT-6', 'NT-12', 'NT-24', 'NT-48'], omp_res,
                             color=color[x], marker=marker[x], label=labels[init_value + 1])
            axes2[j, 2].plot(['Babai', 'B-seq', 'NT-6', 'NT-12', 'NT-24', 'NT-48'], omp_ber,
                             color=color[x], marker=marker[x], label=labels[init_value + 1])
            axes2[j, 3].semilogy(['Babai', 'B-seq', 'NT-6', 'NT-12', 'NT-24', 'NT-48'], omp_tim,
                                 color=color[x], marker=marker[x], label=labels[init_value + 1])

            axes2[j, 0].legend(loc="upper left")
            axes2[0, 1].legend(loc="center right")
            axes2[1, 1].legend(loc="upper right")
            axes2[j, 2].legend(loc="upper right")
            axes2[j, 3].legend(loc="upper right")

            k = k + 2

        # axes2[0].set_xticks([])
        # axes2[1].set_xticks([])
        # axes2[2].set_xticks([])
        # axes2[0].table(cellText=omp_tab.values, rowLabels=index, colLabels=omp_tab.columns,
        #                loc='bottom', colLoc='center', cellLoc='center')
        # axes2[1].table(cellText=omp_tab.values, rowLabels=index, colLabels=omp_tab.columns,
        #                loc='bottom', colLoc='center', cellLoc='center')
        # axes2[2].table(cellText=omp_tab.values, rowLabels=index, colLabels=omp_tab.columns,
        #                loc='bottom', colLoc='center', cellLoc='center')

        k = k + 3

    title = 'Residual Convergence and Bit Error Rate for 15-SNR, 35-SNR and ' \
            + str(pow(4, f)) + '-QAM with different number of threads and block size 16'
    fig.suptitle("\n".join(wrap(title, 60)), fontsize=15)
    plt.savefig('./' + str(n) + '_run_plot_' + str(f) + '_' + str(stop))
    plt.close()


def plot_res(n):
    stops = [-1, 0, 1, 5]
    for f in range(1, 4, 2):
        file = open(str(n) + '_' + str(f) + '_res.out', 'r')
        plot_residual(n, f, file)

        for stop in stops:
            file = open(str(n) + '_' + str(f) + '_plot_' + str(stop) + '.out', 'r')
            plot_runtime(n, f, stop, file)


if __name__ == "__main__":
    plot_res(4096)
