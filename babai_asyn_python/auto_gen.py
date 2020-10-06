import numpy as np
import multiprocessing as mp

x1 = []
def auto_gen(R, x, y, n):
    for k in range(n-1, -1, -1):
        x[k] = (y[k] - np.matmul(R[k, k:n], x[k:n])) / R[k, k]
    return x


def numerator(R, x, y, i, n):
    result = 0
    for j in range(n - 1, i - 1, -1):
        # print("%d %d %f" % (i, j, R[i, j]))
        result += R[i, j] * x[j]

    #print(round((y[i] - result)[0] / R[i, i]))
    #print((y[i] - result)[0] / R[i, i])

    x[i] = round((y[i] - result)[0] / R[i, i])
    return x


def collect_result(result):
    # global results
    global x1
    x1.append(result)


def auto_gen_async(R, x, y, nswp, n):

    pool = mp.Pool(mp.cpu_count())
    print(R)
    for iswp in range(0, nswp):
        for i in range(n, -1, -1):
            pool.apply_async(numerator, args=(R, x, y, i, n), callback=collect_result)
            # x[i] = result/R[i, i]
    pool.close()
    pool.join()

    return x1


if __name__ == "__main__":
    n = 10
    B = np.random.randn(n, n)
    q, R = np.linalg.qr(B)
    x = np.random.randint(-n, n, size=(n, 1))
    y = np.matmul(R, x) + 0.5 * np.random.randn(n, 1)

    #x1 = auto_gen_async(R, x, y, 5, n)
    #print("\nx1:")
    #print(x1[0])

    x0 = auto_gen(R, x, y, n)
    print("\nx0:")
    print(x0)