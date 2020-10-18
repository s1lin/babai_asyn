import numpy as np
import matplotlib.pyplot as plt


def plot_time(ser_time, mat_time, n_proc, omp_time, init_value):
    axes = plt.gca()
    axes.set_xlim([1, 90])
    axes.set_ylim([0, 3])
    left, width = .25, .75
    bottom, height = .20, .75
    right = left + width
    top = bottom + height
    plt.axhline(y=ser_time, xmin=0.0, xmax=1.0, color='b', linestyle='dashed', label='Serial')
    plt.axhline(y=mat_time, xmin=0.0, xmax=1.0, color='g', linestyle='dotted', label='Matlab')
    plt.plot(n_proc, omp_time, color='r', marker='o', label='OpenMP')

    axes.text(right, top, 'Dimension: ' + str(n) + ', init value:' + str(init_value) + ', noise = 0.1',
              horizontalalignment='right',
              verticalalignment='bottom',
              transform=axes.transAxes)

    plt.xlabel('Number of Threads')
    plt.ylabel('Running Time')
    plt.title('Average Running Time For 10 Iterations')

    plt.legend(loc='center right')
    plt.show()


def plot_res(ser_res, mat_res, n_proc, omp_res, init_value):
    axes = plt.gca()
    axes.set_xlim([1, 90])
    axes.set_ylim([0, 3])
    left, width = .25, .75
    bottom, height = .20, .75
    right = left + width
    top = bottom + height
    plt.axhline(y=ser_res, xmin=0.0, xmax=1.0, color='b', linestyle='dashed', label='Serial')
    plt.axhline(y=mat_res, xmin=0.0, xmax=1.0, color='g', linestyle='dotted', label='Matlab')
    plt.plot(n_proc, omp_res, color='r', marker='o', label='OpenMP')

    axes.text(right, top, 'Dimension: ' + str(n) + ', init value:' + str(init_value) + ', noise = 0.1',
              horizontalalignment='right',
              verticalalignment='bottom',
              transform=axes.transAxes)

    plt.xlabel('Number of Threads')
    plt.ylabel('Residual')
    plt.title('Average Residual Time For 10 Iterations')

    plt.legend(loc='center right')
    plt.show()


def plot_res_time(n):
    file1 = open('cmake-build-debug/res_16384.csv', 'r')
    lines = file1.readlines()
    init_value = -2
    n_proc = np.arange(21)
    omp_res = np.arange(21, dtype=np.double)
    omp_time = np.arange(21, dtype=np.double)
    num_iter = np.arange(21, dtype=np.double)
    eig_res = ser_res = ser_time = eig_time = ser_time = 0

    index = i = 0
    for line in lines:
        line_str = line.split(",")
        print(line_str)
        if line == "Next,\n":
            if n == 16384 and init_value == 0:
                mat_time = 2.4668
                mat_res = 12.9352
            if n == 16384 and init_value == 1:
                mat_time = 2.4668
                mat_res = 12.9352
            if n == 16384 and init_value == 0:
                mat_time = 2.4668
                mat_res = 12.9352

            plot_time(ser_time, mat_time, n_proc, omp_time, init_value)
            plot_res(ser_res, mat_res, n_proc, omp_res, init_value)

            index = i = 0
        else:
            init_value = int(line_str[0])
            if index == 0:
                eig_res = float(line_str[1])
                eig_time = float(line_str[2])
            elif index == 1:
                ser_res = float(line_str[1])
                ser_time = float(line_str[2])
            else:
                n_proc[i] = int(line_str[1])
                omp_res[i] = float(line_str[2])
                omp_time[i] = float(line_str[3])
                num_iter[i] = float(line_str[4])
            i = i + 1
            index = index + 1


if __name__ == "__main__":
    n = 16384
    plot_res_time(n)
