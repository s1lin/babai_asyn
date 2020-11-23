import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
from matplotlib import rcParams, rc, ticker


def plot_time(ser_time, mat_time, n_proc, omp_time, init_value):
    axes = plt.gca()
    axes.set_xlim([5, 215])
    #    axes.set_ylim([0.4, 2.7])
    left, width = .25, .75
    bottom, height = -.75, .75
    right = left + width
    top = bottom + height
    plt.axhline(y=ser_time, xmin=0.0, xmax=1.0, color='b', linestyle='dashed', label='Serial')
    plt.axhline(y=mat_time, xmin=0.0, xmax=1.0, color='g', linestyle='dotted', label='Matlab')
    plt.plot(n_proc, omp_time, color='r', marker='o', label='OpenMP')

    if init_value == -1:
        axes.text(right, top, 'Dimension: ' + str(n) + ', Init Guess: the round of real solution, Noise = 0.1',
                  horizontalalignment='right',
                  verticalalignment='bottom',
                  transform=axes.transAxes)
    else:
        axes.text(right, top, 'Dimension: ' + str(n) + ', Init Guess:' + str(init_value) + ', Noise = 0.1',
                  horizontalalignment='right',
                  verticalalignment='bottom',
                  transform=axes.transAxes)

    plt.xlabel('Number of Threads')
    plt.ylabel('Running Time')
    plt.title('Average Running Time For 10 Trials')

    plt.legend(loc='center left')
    plt.savefig('../figures/20201018/' + str(n) + '_tim_' + str(init_value))
    plt.close()


def plot_res(ser_res, mat_res, n_proc, omp_res, init_value, n):
    axes = plt.gca()
    axes.set_xlim([5, 215])
    #    axes.set_ylim([5, 7])
    left, width = .25, .75
    bottom, height = .20, .75
    right = left + width
    top = bottom + height
    plt.axhline(y=ser_res, xmin=0.0, xmax=1.0, color='b', linestyle='dashed', label='Serial')
    plt.axhline(y=mat_res, xmin=0.0, xmax=1.0, color='g', linestyle='dotted', label='Matlab')
    plt.plot(n_proc, omp_res, color='r', marker='o', label='OpenMP')

    if init_value == -1:
        axes.text(right, top, 'Dimension: ' + str(n) + ', Init Guess: the round of real solution, Noise = 0.1',
                  horizontalalignment='right',
                  verticalalignment='bottom',
                  transform=axes.transAxes)
    else:
        axes.text(right, top, 'Dimension: ' + str(n) + ', Init Guess:' + str(init_value) + ', Noise = 0.1',
                  horizontalalignment='right',
                  verticalalignment='bottom',
                  transform=axes.transAxes)

    plt.xlabel('Number of Threads')
    plt.ylabel('Residual')
    plt.title('Average Residual Time For 10 Trials')

    plt.legend(loc='upper left')
    plt.savefig('../figures/20201018/' + str(n) + '_res_' + str(init_value))
    plt.close()


def plot_res_time(n):
    file1 = open('cmake-build-debug/res_' + str(n) + '.csv', 'r')
    lines = file1.readlines()
    init_value = -2
    n_proc = np.arange(21)
    omp_res = np.arange(21, dtype=np.double)
    omp_time = np.arange(21, dtype=np.double)
    num_iter = np.arange(21, dtype=np.double)
    eig_res = ser_res = ser_time = eig_time = ser_time = mat_time = mat_res = 0

    index = i = 0
    for line in lines:
        line_str = line.split(",")
        print(line_str)
        if line == "Next,\n":
            plot_time(ser_time, mat_time, n_proc, omp_time, init_value)
            plot_res(ser_res, mat_res, n_proc, omp_res, init_value)
            index = i = 0
        else:
            init_value = int(line_str[0])
            if index == 0:
                ser_res = float(line_str[1])
                ser_time = float(line_str[2])
            elif index == 1:
                mat_res = float(line_str[1])
                mat_time = float(line_str[2])
            else:
                n_proc[i] = int(line_str[1])
                omp_res[i] = float(line_str[2])
                omp_time[i] = float(line_str[3])
                num_iter[i] = float(line_str[4])
                i = i + 1
            index = index + 1


def plot_res_conv(n):
    for f in range(1, 4, 2):
        print(f)
        file = open(str(n) + '_' + str(pow(4, f)) + '_15.out', 'r')
        lines = file.readlines()
        k = 0
        while k < len(lines):

            line = lines[k]
            if "SNR" in line:
                SNR = lines[k].split(":")[1].split("\n")[0]
            if "seconds" in line:
                k = k + 2
                while k < len(lines) and not "-----" in lines[k] \
                        and not "++++++" in lines[k] \
                        and not "it, si" in lines[k]:
                    line_str = lines[k].split(",")
                    init_value = line_str[0].split("\n")[0]
                    plt.rcParams["figure.figsize"] = (13, 4)

                    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
                    axes = plt.gca()

                    # plt.rcParams["font.sans-serif"] = "Comic Sans MS"
                    # plt.rcParams["font.family"] = "sans-serif"
                    # rc('font', **{'family': 'serif', 'serif':['Times']})
                    k = k + 2
                    line_str = lines[k].split(",")
                    end = len(line_str)
                    print(line_str[2:end])
                    line_str[end - 1] = line_str[end - 1].split("/n")[0]
                    ax1.semilogy(range(0, 11), np.array(line_str[2:end - 2]).astype(np.float), color='r', marker='o',
                                 label='num_thread = ' + line_str[1])
                    k = k + 1
                    line_str = lines[k].split(",")
                    line_str[end - 1] = line_str[end - 1].split("/n")[0]
                    print(line_str[2:end])
                    ax1.semilogy(range(0, 11), np.array(line_str[2:end - 2]).astype(np.float), color='g', marker='x',
                                 label='num_thread = ' + line_str[1])
                    k = k + 1
                    line_str = lines[k].split(",")
                    line_str[end - 1] = line_str[end - 1].split("/n")[0]
                    print(line_str[2:end])
                    ax1.semilogy(range(0, 11), np.array(line_str[2:end - 2]).astype(np.float), color='b', marker='+',
                                 label='num_thread = ' + line_str[1])
                    ax1.legend(loc="upper right")
                    ax1.set_xlabel('Number of Iterations')
                    ax1.set_ylabel('Residual')
                    ax1.set_title('Block Size ' + line_str[0], fontsize=10)

                    k = k + 3
                    line_str = lines[k].split(",")
                    end = len(line_str)
                    print(line_str[2:end])
                    line_str[end - 1] = line_str[end - 1].split("/n")[0]
                    ax2.semilogy(range(0, 11), np.array(line_str[2:end - 2]).astype(np.float), color='r', marker='o',
                                 label='num_thread = ' + line_str[1])
                    k = k + 1
                    line_str = lines[k].split(",")
                    line_str[end - 1] = line_str[end - 1].split("/n")[0]
                    print(line_str[2:end])
                    ax2.semilogy(range(0, 11), np.array(line_str[2:end - 2]).astype(np.float), color='g', marker='x',
                                 label='num_thread = ' + line_str[1])
                    k = k + 1
                    line_str = lines[k].split(",")
                    line_str[end - 1] = line_str[end - 1].split("/n")[0]
                    print(line_str[2:end])
                    ax2.semilogy(range(0, 11), np.array(line_str[2:end - 2]).astype(np.float), color='b', marker='+',
                                 label='num_thread = ' + line_str[1])
                    ax2.legend(loc="upper right")
                    ax2.set_xlabel('Number of Iterations')
                    # ax2.set_ylabel('Residual')
                    ax2.set_title('Block Size ' + line_str[0], fontsize=10)

                    k = k + 3
                    line_str = lines[k].split(",")
                    end = len(line_str)
                    print(line_str[2:end])
                    line_str[end - 1] = line_str[end - 1].split("/n")[0]
                    ax3.semilogy(range(0, 11), np.array(line_str[2:end - 2]).astype(np.float), color='r', marker='o',
                                 label='num_thread = ' + line_str[1])
                    k = k + 1
                    line_str = lines[k].split(",")
                    line_str[end - 1] = line_str[end - 1].split("/n")[0]
                    print(line_str[2:end])
                    ax3.semilogy(range(0, 11), np.array(line_str[2:end - 2]).astype(np.float), color='g', marker='x',
                                 label='num_thread = ' + line_str[1])
                    k = k + 1
                    line_str = lines[k].split(",")
                    line_str[end - 1] = line_str[end - 1].split("/n")[0]
                    print(line_str[2:end])
                    ax3.semilogy(range(0, 11), np.array(line_str[2:end - 2]).astype(np.float), color='b', marker='+',
                                 label='num_thread = ' + line_str[1])
                    ax3.legend(loc="upper right")
                    ax3.set_xlabel('Number of Iterations')
                    # ax3.set_ylabel('Residual')
                    ax3.set_title('Block Size ' + line_str[0], fontsize=10)
                    title = 'Residual Convergence for' + SNR + '-SNR and ' \
                            + str(pow(4, f)) + '-QAM with different block sizes'

                    if init_value in '1':
                        title = title + 'where each element in the initial point is the rounded mean'
                    elif init_value in '-1':
                        title = title + ' where the round of real solution as the initial point'
                    elif init_value in '0':
                        title = title + ' where each element in the initial point is 0'

                    fig.suptitle(title, fontsize=12)
                    plt.savefig(
                        './' + str(n) + '_res_' + init_value + "_" + SNR + "_" + line_str[0] + '_' + str(pow(4, f)))
                    plt.close()
                    k = k + 1

                if "+" not in lines[k]:
                    k = k + 8
                else:
                    k = k + 2

                plt2.rcParams["figure.figsize"] = (13, 9)
                fig, axes = plt2.subplots(2, 3)

                color = ['r', 'g', 'b']
                marker = ['o', '+', 'x']

                axes[0, 0].set_title('Block Size 8', fontsize=10)
                axes[0, 1].set_title('Block Size 16', fontsize=10)
                axes[0, 2].set_title('Block Size 32', fontsize=10)
                axes[1, 0].set_title('Block Size 8', fontsize=10)
                axes[1, 1].set_title('Block Size 16', fontsize=10)
                axes[1, 2].set_title('Block Size 32', fontsize=10)

                axes[1, 0].set_xlabel('Method')
                axes[1, 1].set_xlabel('Method')
                axes[1, 2].set_xlabel('Method')

                axes[0, 0].set_ylabel('Residual')
                axes[1, 0].set_ylabel('Running Time')

                for x in range(0, 3):
                    init_value = int(lines[k].split(":")[1].split("\n")[0])
                    print(lines[k].split(","))
                    for t in range(0, 3):
                        print("here:" + str(t))
                        k = k + 2
                        print(lines[k].split(","))
                        babai_res = float(lines[k].split(",")[1].split(":")[1])
                        babai_tim = float(lines[k].split(",")[2].split(":")[1].split("s")[0])
                        k = k + 2
                        print(lines[k].split(","))
                        block_res = float(lines[k].split(",")[2].split(":")[1])
                        block_tim = float(lines[k].split(",")[3].split(":")[1].split("s")[0])
                        k = k + 2
                        print(lines[k].split(","))
                        omp_res = [babai_res, block_res]
                        omp_tim = [babai_tim, block_tim]

                        for l in range(0, 4):
                            print(lines[k].split(","))
                            omp_res.append(float(lines[k].split(",")[2].split(":")[1]))
                            omp_tim.append(float(lines[k].split(",")[4].split(":")[1].split("s")[0]))
                            k = k + 1

                        if init_value == -1:
                            axes[0, t].plot(['Babai-1', 'BOB-1', 'BOB-3', 'BOB-12', 'BOB-48'], omp_res[0:5],
                                            color=color[x], marker=marker[x], label='$x_{init} = round(x_R)$')
                            axes[1, t].plot(['Babai-1', 'BOB-1', 'BOB-3', 'BOB-12', 'BOB-48'], omp_tim[0:5],
                                            color=color[x], marker=marker[x], label='$x_{init} = round(x_R)$')
                        elif init_value == 0:
                            axes[0, t].plot(['Babai-1', 'BOB-1', 'BOB-3', 'BOB-12', 'BOB-48'], omp_res[0:5],
                                            color=color[x], marker=marker[x], label='$x_{init} = 0$')
                            axes[1, t].plot(['Babai-1', 'BOB-1', 'BOB-3', 'BOB-12', 'BOB-48'], omp_tim[0:5],
                                            color=color[x], marker=marker[x], label='$x_{init} = 0$')
                        else:
                            axes[0, t].plot(['Babai-1', 'BOB-1', 'BOB-3', 'BOB-12', 'BOB-48'], omp_res[0:5],
                                            color=color[x], marker=marker[x], label='$x_{init} = avg$')
                            axes[1, t].plot(['Babai-1', 'BOB-1', 'BOB-3', 'BOB-12', 'BOB-48'], omp_tim[0:5],
                                            color=color[x], marker=marker[x], label='$x_{init} = avg$')

                        k = k + 2
                axes[0, 0].legend(loc="upper left")
                axes[0, 1].legend(loc="upper left")
                axes[0, 2].legend(loc="upper left")
                axes[1, 0].legend(loc="center left")
                axes[1, 1].legend(loc="center left")
                axes[1, 2].legend(loc="center left")
                title = 'Achieved Residual and Running Time for' + SNR + '-SNR and ' \
                        + str(pow(4, f)) + '-QAM with different block sizes and initial guesses'

                fig.suptitle(title, fontsize=12)

                plt2.savefig('./' + str(n) + '_res_' + SNR + "_tim_" + str(pow(4, f)))
                plt2.close()

            k = k + 1


if __name__ == "__main__":
    plot_res_conv(4096)
    # plot_res_conv(16384)

#
# for t in range(0, 3):
#     babai_res = float(lines[k].split(",")[1].split(":")[1])
#     babai_tim = float(lines[k].split(",")[2].split(":")[1].split("s")[0])
#     k = k + 2
#
#     block_res = float(lines[k].split(",")[2].split(":")[1])
#     block_tim = float(lines[k].split(",")[3].split(":")[1].split("s")[0])
#     k = k + 2
#
#     omp_res = []
#     omp_res.append(babai_res)
#     omp_res.append(block_res)
#     for l in range(0, 3):
#         omp_res.append(float(lines[k].split(",")[2].split(":")[1]))
#         k = k + 1
#
#     ax1.plot(label, omp_res, color=color[t], marker=marker[t])
#
#     k = k + 5
#
# k = k - 2
# init_value = int(lines[k].split(":")[1].split("\n")[0])
# k = k + 2
# for t in range(0, 3):
#     babai_res = float(lines[k].split(",")[1].split(":")[1])
#     babai_tim = float(lines[k].split(",")[2].split(":")[1].split("s")[0])
#     k = k + 2
#
#     block_res = float(lines[k].split(",")[2].split(":")[1])
#     block_tim = float(lines[k].split(",")[3].split(":")[1].split("s")[0])
#     k = k + 2
#
#     omp_res = []
#     omp_res.append(babai_res)
#     omp_res.append(block_res)
#     for l in range(0, 3):
#         omp_res.append(float(lines[k].split(",")[2].split(":")[1]))
#         k = k + 1
#
#     ax2.plot(label, omp_res, color=color[t], marker=marker[t])
#
#     k = k + 5
#
# k = k - 2
# init_value = int(lines[k].split(":")[1].split("\n")[0])
# k = k + 2
# for t in range(0, 3):
#     babai_res = float(lines[k].split(",")[1].split(":")[1])
#     babai_tim = float(lines[k].split(",")[2].split(":")[1].split("s")[0])
#     k = k + 2
#
#     block_res = float(lines[k].split(",")[2].split(":")[1])
#     block_tim = float(lines[k].split(",")[3].split(":")[1].split("s")[0])
#     k = k + 2
#
#     omp_res = []
#     omp_res.append(babai_res)
#     omp_res.append(block_res)
#     for l in range(0, 3):
#         omp_res.append(float(lines[k].split(",")[2].split(":")[1]))
#         k = k + 1
#
#     ax3.plot(label, omp_res, color=color[t], marker=marker[t])
#
#     k = k + 5
# k = k + 3
# plt.savefig('./' + str(n) + '_res_' + str(init_value) + "_" + SNR + "_res_" + str(pow(4, f)))
# plt.close()
