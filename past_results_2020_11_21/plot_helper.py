from textwrap import wrap

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import pandas as pd

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

                    k = k + 1
                    line_str = lines[k].split(",")
                    line_str[end - 1] = line_str[end - 1].split("/n")[0]
                    print(line_str[2:end])
                    ax1.semilogy(range(0, 11), np.array(line_str[2:end - 2]).astype(np.float), color='y', marker='.',
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

                    k = k + 1
                    line_str = lines[k].split(",")
                    line_str[end - 1] = line_str[end - 1].split("/n")[0]
                    print(line_str[2:end])
                    ax2.semilogy(range(0, 11), np.array(line_str[2:end - 2]).astype(np.float), color='y', marker='.',
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

                    k = k + 1
                    line_str = lines[k].split(",")
                    line_str[end - 1] = line_str[end - 1].split("/n")[0]
                    print(line_str[2:end])
                    ax3.semilogy(range(0, 11), np.array(line_str[2:end - 2]).astype(np.float), color='y', marker='.',
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
                fig, axes2 = plt2.subplots(2, 3)

                color = ['r', 'g', 'b', 'y']
                marker = ['o', '+', 'x', '.']

                axes2[0, 0].set_title('Block Size 8',  fontsize=11)
                axes2[0, 1].set_title('Block Size 16', fontsize=11)
                axes2[0, 2].set_title('Block Size 32', fontsize=11)

                axes2[0, 0].set_ylabel('Residual')
                axes2[1, 0].set_ylabel('Running Time')
                index = ['Iter', 'Time']
                omp_itr = pd.DataFrame(np.random.randn(2, 6),
                                       columns=['Babai-1', 'BOB-1', 'BOB-6', 'BOB-12', 'BOB-24', 'BOB-48'])
                # print(omp_itr.iloc[0, 0])
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
                        omp_itr.iloc[0, 0] = '/'
                        omp_itr.iloc[0, 1] = '/'
                        omp_itr.iloc[1, 0] = babai_tim
                        omp_itr.iloc[1, 1] = block_tim

                        for l in range(0, 4):
                            print(lines[k].split(","))
                            omp_res.append(float(lines[k].split(",")[2].split(":")[1]))
                            omp_itr.iloc[0, l + 2] = float(lines[k].split(",")[3].split(":")[1])
                            omp_itr.iloc[1, l + 2] = float(lines[k].split(",")[4].split(":")[1].split("s")[0])
                            omp_tim.append(float(lines[k].split(",")[4].split(":")[1].split("s")[0]))
                            k = k + 1

                        if init_value == -1:
                            axes2[0, t].plot(['Babai-1', 'BOB-1', 'BOB-6', 'BOB-12', 'BOB-24', 'BOB-48'], omp_res,
                                            color=color[x], marker=marker[x], label='$x_{init} = round(x_R)$')
                            axes2[1, t].plot(['Babai-1', 'BOB-1', 'BOB-6', 'BOB-12', 'BOB-24', 'BOB-48'], omp_tim,
                                            color=color[x], marker=marker[x], label='$x_{init} = round(x_R)$')
                        elif init_value == 0:
                            axes2[0, t].plot(['Babai-1', 'BOB-1', 'BOB-6', 'BOB-12', 'BOB-24', 'BOB-48'], omp_res,
                                            color=color[x], marker=marker[x], label='$x_{init} = 0$')
                            axes2[1, t].plot(['Babai-1', 'BOB-1', 'BOB-6', 'BOB-12', 'BOB-24', 'BOB-48'], omp_tim,
                                            color=color[x], marker=marker[x], label='$x_{init} = 0$')
                        else:
                            axes2[0, t].plot(['Babai-1', 'BOB-1', 'BOB-6', 'BOB-12', 'BOB-24', 'BOB-48'], omp_res,
                                            color=color[x], marker=marker[x], label='$x_{init} = avg$')
                            axes2[1, t].plot(['Babai-1', 'BOB-1', 'BOB-6', 'BOB-12', 'BOB-24', 'BOB-48'], omp_tim,
                                            color=color[x], marker=marker[x], label='$x_{init} = avg$')

                        axes2[1, t].set_xticks([])
                        axes2[1, t].table(cellText=omp_itr.values, rowLabels=index, colLabels=omp_itr.columns,
                                         loc='bottom', colLoc='center', cellLoc='center')

                        k = k + 2

                axes2[0, 0].legend(loc="upper left")
                axes2[0, 1].legend(loc="upper left")
                axes2[0, 2].legend(loc="upper left")
                axes2[1, 0].legend(loc="center left")
                axes2[1, 1].legend(loc="center left")
                axes2[1, 2].legend(loc="center left")
                title = 'Achieved Residual and Running Time for problem size' + str(n) + 'with ' + SNR + '-SNR and ' \
                        + str(pow(4, f)) + '-QAM with different block sizes and initial guesses without stopping ' \
                                           'Criteria'

                fig.suptitle("\n".join(wrap(title, 60)), fontsize=12)
                pd.set_option('display.max_columns', None)
                print(omp_itr)

                plt2.savefig('./' + str(n) + '_res_' + SNR + "_tim_" + str(pow(4, f)) + '_non_stop')
                plt2.close()
                k = k + 7

                plt3.rcParams["figure.figsize"] = (13, 9)
                fig, axes = plt3.subplots(2, 3)

                color = ['r', 'g', 'b', 'y']
                marker = ['o', '+', 'x', '.']

                axes[0, 0].set_title('Block Size 8', fontsize=11)
                axes[0, 1].set_title('Block Size 16', fontsize=11)
                axes[0, 2].set_title('Block Size 32', fontsize=11)

                axes[0, 0].set_ylabel('Residual')
                axes[1, 0].set_ylabel('Running Time')

                index = ['Iter', 'Time']
                omp_itr = pd.DataFrame(np.random.randn(2, 6),
                                       columns=['Babai-1', 'BOB-1', 'BOB-6', 'BOB-12', 'BOB-24', 'BOB-48'])
                # print(omp_itr.iloc[0, 0])
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
                        omp_itr.iloc[0, 0] = '/'
                        omp_itr.iloc[0, 1] = '/'
                        omp_itr.iloc[1, 0] = babai_tim
                        omp_itr.iloc[1, 1] = block_tim

                        for l in range(0, 4):
                            print(lines[k].split(","))
                            omp_res.append(float(lines[k].split(",")[2].split(":")[1]))
                            omp_itr.iloc[0, l + 2] = float(lines[k].split(",")[3].split(":")[1])
                            omp_itr.iloc[1, l + 2] = float(lines[k].split(",")[4].split(":")[1].split("s")[0])
                            omp_tim.append(float(lines[k].split(",")[4].split(":")[1].split("s")[0]))
                            k = k + 1

                        if init_value == -1:
                            axes[0, t].plot(['Babai-1', 'BOB-1', 'BOB-6', 'BOB-12', 'BOB-24', 'BOB-48'], omp_res,
                                            color=color[x], marker=marker[x], label='$x_{init} = round(x_R)$')
                            axes[1, t].plot(['Babai-1', 'BOB-1', 'BOB-6', 'BOB-12', 'BOB-24', 'BOB-48'], omp_tim,
                                            color=color[x], marker=marker[x], label='$x_{init} = round(x_R)$')
                        elif init_value == 0:
                            axes[0, t].plot(['Babai-1', 'BOB-1', 'BOB-6', 'BOB-12', 'BOB-24', 'BOB-48'], omp_res,
                                            color=color[x], marker=marker[x], label='$x_{init} = 0$')
                            axes[1, t].plot(['Babai-1', 'BOB-1', 'BOB-6', 'BOB-12', 'BOB-24', 'BOB-48'], omp_tim,
                                            color=color[x], marker=marker[x], label='$x_{init} = 0$')
                        else:
                            axes[0, t].plot(['Babai-1', 'BOB-1', 'BOB-6', 'BOB-12', 'BOB-24', 'BOB-48'], omp_res,
                                            color=color[x], marker=marker[x], label='$x_{init} = avg$')
                            axes[1, t].plot(['Babai-1', 'BOB-1', 'BOB-6', 'BOB-12', 'BOB-24', 'BOB-48'], omp_tim,
                                            color=color[x], marker=marker[x], label='$x_{init} = avg$')

                        axes[1, t].set_xticks([])
                        axes[1, t].table(cellText=omp_itr.values, rowLabels=index, colLabels=omp_itr.columns,
                                         loc='bottom', colLoc='center', cellLoc='center')

                        k = k + 2
                axes[0, 0].legend(loc="upper left")
                axes[0, 1].legend(loc="upper left")
                axes[0, 2].legend(loc="upper left")
                axes[1, 0].legend(loc="center left")
                axes[1, 1].legend(loc="center left")
                axes[1, 2].legend(loc="center left")
                title = 'Achieved Residual and Running Time for problem size' + str(n) + 'with ' + SNR + '-SNR and ' \
                        + str(pow(4, f)) + '-QAM with different block sizes and initial guesses with stopping ' \
                                           'Criteria'

                fig.suptitle("\n".join(wrap(title, 60)), fontsize=12)

                plt3.savefig('./' + str(n) + '_res_' + SNR + "_tim_" + str(pow(4, f)) + '_stopped')
                plt3.close()

            k = k + 1


if __name__ == "__main__":
    plot_res_conv(4096)
    plot_res_conv(16384)

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
