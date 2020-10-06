import numpy as np
import multiprocessing
from joblib import Parallel, delayed, Memory
import ctypes
from copy import deepcopy
from time import time, sleep
from multiprocessing import Pool, Process
from concurrent import futures


class babai_search_asyn:

    @staticmethod
    def auto_compare(n, n_jobs, nswp):
        # Init babai_search_asyn object, and results.
        bsa = babai_search_asyn(n)
        residual = np.zeros(3)
        # Run all the defs
        residual[0] = bsa.init_res
        raw_x0, residual[1], t1 = bsa.find_raw_x0()
        raw_x0p, j, residual[2], t2 = bsa.find_raw_x0_par(n_jobs, nswp)

        # print(np.transpose(raw_x0))
        # print(np.transpose(raw_x0p[:, j]))
        print(residual, t1, t2, j)
        # Compute the results
        # time = t1 - t2
        return residual, time, raw_x0

    # Constructor def
    def __init__(self, n):
        self.n = n
        self.z = np.ones(n)
        self.A = np.random.randn(n, n)
        self.q, self.R = np.linalg.qr(self.A)
        self.x0 = np.random.randint(-n, n, size=(n, 1))
        self.y = np.matmul(self.R, self.x0) + np.random.randn(n, 1)
        self.init_res = np.linalg.norm(self.y - np.matmul(self.R, self.x0))
        self.n_proc = multiprocessing.cpu_count()

    # Find raw_x0p with parallel pooling CPU ONLY for now.
    def find_raw_x0_par(self, n_proc, nswp):

        x_raw = np.zeros((self.n, self.n + 5), dtype=int)

        # for i in range(0, self.n + 5):
        x_raw[:, 0] = deepcopy(self.x0[:, 0])

        # shm = shared_memory.SharedMemory(create=True, size=x_raw.nbytes)
        x = np.ndarray(x_raw.shape, dtype=x_raw.dtype)
        # m = self.n + 5
        # shared_array_base = multiprocessing.Array(ctypes.c_double, self.n * m)
        # x = np.frombuffer(shared_array_base.get_obj())
        # x = np.reshape(x, (self.n, m))
        x[:] = x_raw[:]
        print(x)

        k = np.zeros(self.n + 1)
        b = deepcopy(self.y)
        j = 0

        raw_x0, tol, t = self.find_raw_x0()
        res = np.inf

        if n_proc == 0:
            n_proc = self.n_proc

        print('Creating pool with %d processes\n' % n_proc)

        # X = []
        # pool = Pool(n_proc)
        # start = time()
        # for j in range(0, nswp):
        #     for i in range(self.n - 1, -1, -1):
        #         X.append(pool.apply_async(func=self.deploy, args=(j, i, x)))
        #         #p = Process(name=str(j) + ' ' + str(i), target=self.deploy, args=(j, i, x))
        #         #p.start()
        #         #if j > 1:
        #             #res = np.linalg.norm(b - np.matmul(self.R, x[:, j].reshape(self.n, 1)))
        # for job in X:
        #     print(job.get())

        start = time()
        # with futures.ProcessPoolExecutor() as pool:
        #     for j in range(0, nswp):
        #         future_result = pool.submit(some_function_call, parameters1, parameters2)
        #         future_result.add_done_callback(total_error)

        # while res > tol:
        for j in range(0, nswp):
            Parallel(n_jobs=n_proc, require='sharedmem', verbose=1, mmap_mode='w+')(
                delayed(self.deploy)(i, j, x) for i in range(self.n - 1, -1, -1))
        #     j = j + 1
        #     if j > 2:
        #         res = np.linalg.norm(b - np.matmul(self.R, x[:, j].reshape(self.n, 1)))

        end = time()
        # p.wait()
        # pool.close()
        # pool.join()

        print(x)

        res = np.linalg.norm(b - np.matmul(self.R, x[:, nswp].reshape(self.n, 1)))
        print('time : %.9f seconds ' % (end - start))
        print(res, tol)
        # shm.close()
        # shm.unlink()

        return x, j, res, end - start

    def deploy(self, i, j, x):  # m, B, x, i, j, k, C, b, index):
        # print("---------------------------------------")
        # s = np.matmul(self.R[i, i + 1:self.n], x[i + 1:self.n, j])
        # x[i, j + 1] = np.round((self.y[i] - s) / self.R[i, i])

        # if j == 3:
        #     sleep(2)
        # print(x)
        # print('%d, %d'%(j, i))
        print(j)
        # print("%d->%d" %(x[i, j], x[i, j + 1]))
        return x

    # Find raw x0 in serial for loop.
    def find_raw_x0(self):

        raw_x0 = deepcopy(self.x0)
        start = time()
        print(self.x0)
        for k in range(self.n - 1, -1, -1):
            #s = (self.y[k] - np.matmul(self.R[k, k + 1:self.n], raw_x0[k + 1:self.n])) / self.R[k, k]
            print(raw_x0[k + 1:self.n])
            #raw_x0[k] = np.round(s)
        res = np.linalg.norm(self.y - np.matmul(self.R, raw_x0))
        end = time()
        return raw_x0, res, end - start


if __name__ == "__main__":
    babai_search_asyn.auto_compare(10, 0, 10)

# executor = Parallel(n_jobs=n_jobs)#, backend='multiprocessing')

# tasks = (delayed(self.deploy)(j, x) for j in range(0, nswp))
# out = executor(tasks)
# print(out)
# for j in range(0, nswp):
#    p = Process(target=self.deploy, args=(j, x))
#    p.start()

# while j <= nswp:
#     print('   task  index  i   j+1   k_next')

# print(out)
# for i in range(self.n - 1, -1, -1):
# task = get(getCurrentTask(), 'ID')
# index = self.n - i + 1
# k[self.n + 1, j] = i
# if i == k[self.n + 1, j]:
# s = self.deploy(m, B, x, i, j, k, C, b, index)
# s = np.matmul(self.R[i, i + 1:self.n], x[i + 1:self.n, j])
# x[i, j + 1] = np.round((b[i] - s) / self.R[i, i])
# k_next[i] = k[i] - 1
# print([0, i, i, x[i, j + 1], k_next[i]])
# print(i)
# else:
#     x[i, j + 1] = x[i, j]

# if j < self.n:
# k[self.n - j] = k_next[self.n - j + 1]
# k(m - j + 1) = 0
# j = j + 1
