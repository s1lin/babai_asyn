import numpy as np
# import matlab.engine


def qr(A, n):
    # print('Python function qr called')
    q, r = np.linalg.qr(np.reshape(A, (-1, n)))
    return q.ravel(), r.ravel()


# def test_matlab():
#     eng = matlab.engine.start_matlab()
#     a = matlab.double([1,4,9,16,25])
#     b = eng.sqrt(a)
#     print(b)
#     # A = eng.randn(256)
#
#
# if __name__ == "__main__":
#     test_matlab()
