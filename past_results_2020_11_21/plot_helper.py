import numpy as np
import matplotlib.pyplot as plt


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
    for f in range(1,4,2):
        print(f)
        file = open(str(n) + '_' + str(pow(4,f)) +'_15.out', 'r')
        lines = file.readlines()

        for i in range(0, len(lines)):
            line = lines[i]
            if "SNR" in line:
                SNR = lines[i].split(":")[1].split("/n")[0]
            if "seconds" in line:
                i = i + 2
                while i < len(lines) and not "-----" in lines[i]:

                    line_str = lines[i].split(",")
                    init_value = line_str[0].split("/n")[0]

                    fig, (ax1, ax2, ax3) = plt.subplots(3,1)
                    axes = plt.gca()
                    plt.rcParams["figure.figsize"] = (10,10)

                    i = i + 2
                    line_str = lines[i].split(",")
                    end = len(line_str)
                    print(line_str[2:end])
                    line_str[end-1] = line_str[end-1].split("/n")[0]
                    ax1.semilogy(range(0, 13), np.array(line_str[2:end]).astype(np.float), color='r', marker='o',
                             label='num_thread = ' + line_str[1])
                    i = i + 1
                    line_str = lines[i].split(",")
                    line_str[end-1] = line_str[end-1].split("/n")[0]
                    print(line_str[2:end])
                    ax1.semilogy(range(0, 13), np.array(line_str[2:end]).astype(np.float), color='g', marker='x',
                             label='num_thread = ' + line_str[1])
                    i = i + 1
                    line_str = lines[i].split(",")
                    line_str[end-1] = line_str[end-1].split("/n")[0]
                    print(line_str[2:end])
                    ax1.semilogy(range(0, 13), np.array(line_str[2:end]).astype(np.float), color='b', marker='+',
                             label='num_thread = ' + line_str[1])
                    ax1.legend(loc="upper right")
                    ax1.set_ylabel('Residual $\log_{10}$')
                    ax1.set_title('Residual Convergence with Block Size ' + line_str[0])

                    i = i + 3
                    line_str = lines[i].split(",")
                    end = len(line_str)
                    print(line_str[2:end])
                    line_str[end-1] = line_str[end-1].split("/n")[0]
                    ax2.semilogy(range(0, 13), np.array(line_str[2:end]).astype(np.float), color='r', marker='o',
                                 label='num_thread = ' + line_str[1])
                    i = i + 1
                    line_str = lines[i].split(",")
                    line_str[end-1] = line_str[end-1].split("/n")[0]
                    print(line_str[2:end])
                    ax2.semilogy(range(0, 13), np.array(line_str[2:end]).astype(np.float), color='g', marker='x',
                                 label='num_thread = ' + line_str[1])
                    i = i + 1
                    line_str = lines[i].split(",")
                    line_str[end-1] = line_str[end-1].split("/n")[0]
                    print(line_str[2:end])
                    ax2.semilogy(range(0, 13), np.array(line_str[2:end]).astype(np.float), color='b', marker='+',
                                 label='num_thread = ' + line_str[1])
                    ax2.legend(loc="upper right")
                    ax2.set_ylabel('Residual $\log_{10}$')
                    ax2.set_title('Residual Convergence with Block Size ' + line_str[0])

                    i = i + 3
                    line_str = lines[i].split(",")
                    end = len(line_str)
                    print(line_str[2:end])
                    line_str[end-1] = line_str[end-1].split("/n")[0]
                    ax3.semilogy(range(0, 13), np.array(line_str[2:end]).astype(np.float), color='r', marker='o',
                                 label='num_thread = ' + line_str[1])
                    i = i + 1
                    line_str = lines[i].split(",")
                    line_str[end-1] = line_str[end-1].split("/n")[0]
                    print(line_str[2:end])
                    ax3.semilogy(range(0, 13), np.array(line_str[2:end]).astype(np.float), color='g', marker='x',
                                 label='num_thread = ' + line_str[1])
                    i = i + 1
                    line_str = lines[i].split(",")
                    line_str[end-1] = line_str[end-1].split("/n")[0]
                    print(line_str[2:end])
                    ax3.semilogy(range(0, 13), np.array(line_str[2:end]).astype(np.float), color='b', marker='+',
                                 label='num_thread = ' + line_str[1])
                    ax3.legend(loc="upper right")
                    ax3.set_xlabel('Number of Iterations')
                    ax3.set_ylabel('Residual $\log_{10}$')
                    ax3.set_title('Residual Convergence with Block Size ' + line_str[0])

                    if init_value in '-1':
                        fig.suptitle('Residual Convergence where the round of real solution as the initial point')
                    elif init_value in '0':
                        fig.suptitle('Residual Convergence where each element in the initial point is 0')
                    elif init_value in '1':
                        fig.suptitle('Residual Convergence where each element in the initial point is the rounded mean')

                    fig.suptitle('Residual Convergence with different block sizes')
                    plt.savefig('./' + str(n) + '_res_' + init_value + "_" + str(SNR) + "_" + line_str[0])
                    plt.close()
                    i = i + 1


if __name__ == "__main__":
    plot_res_conv(4096)
    plot_res_conv(16384)
