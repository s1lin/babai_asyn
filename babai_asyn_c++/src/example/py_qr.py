import numpy as np


# import netCDF4 as nc


def qr(A, n):
    # print('Python function qr called')
    q, r = np.linalg.qr(np.reshape(A, (-1, n)))
    q = q.T
    r = r.T
    return q.ravel(), r.ravel()


def planerot(x):
    G = np.eye(2, 2)
    if x[1] != 0:
        r = np.linalg.norm(x)
        G[0, :] = x.T
        G[1, 0] = -x[1]
        G[1, 1] = x[0]
        G = G / r
        x = np.array([r, 0])
    return G, x


def sils_reduction_driver(A, v, x, n):
    A = np.reshape(A, (-1, n))
    # y = A @ x + v
    # y = np.reshape(y, (n, 1))
    # print(y.shape)
    # return sils_reduction(A, y)


#
# [R,Z,y] = sils_reduction(A,y) reduces the general standard integer
# least squares problem to an upper triangular one by the LLL-QRZ
# factorization Q'*A*Z = [R 0]. The orthogonal matrix Q
# is not produced.
#
# Inputs:
#    A - m-by-n real matrix with full column rank
#    y - m-dimensional real vector to be transformed to Q'*y
#
# Outputs:
#    R - n-by-n LLL-reduced upper triangular matrix
#    Z - n-by-n unimodular matrix, i.e., an integer matrix with
#    |det(Z)|=1
#    y - m-vector transformed from the input y by Q', i.e., y := Q'*y
#

# Subfunction: qrmcp

# Main Reference:
# X. Xie, X.-W. Chang, and M. Al Borno. Partial LLL Reduction,
# Proceedings of IEEE GLOBECOM 2011, 5 pages.

# Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
#          Xiaohu Xie, Tianyang Zhou
# Modified by Shilei Lin into Python
# Copyright (c) 2006-2021. Scientific Computing Lab, McGill University.
# October 2006. Last revision: June 2016
def sils_reduction(A, y):
    m, n = A.shape

    # QR factorization with minimum-column pivoting
    [R, piv, y] = qrmcp(A, y)
    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
    print(R, piv, y)
    # Obtain the permutation matrix Z
    Z = np.zeros(shape=(n, n))
    for j in range(0, n):
        Z[piv[j], j] = 1

    # ------------------------------------------------------------------
    # --------  Perfome the partial LLL reduction  ---------------------
    # ------------------------------------------------------------------

    k = 1

    while k < n:

        k1 = k - 1
        if abs(R[k1, k1]) < 1e-15:
            print(k1, R[k1, k1])
        zeta = round(R[k1, k] / R[k1, k1])
        alpha = R[k1, k] - zeta * R[k1, k1]

        if R[k1, k1] ** 2 > 2 * (alpha ** 2 + R[k, k] ** 2):
            if zeta != 0:
                # Perform a size reduction on R[k-1,k]
                R[k1, k] = alpha
                R[0:k - 1, k] = R[0:k - 1, k] - zeta * R[0:k - 1, k - 1]
                Z[:, k] = Z[:, k] - zeta * Z[:, k - 1]

                # Perform size reductions on R[1:k-2,k]
                for i in range(k - 2, 0, -1):
                    zeta = round(R[i, k] / R[i, i])
                    if zeta != 0:
                        R[0:i, k] = R[0:i, k] - zeta * R[0:i, i]
                        Z[:, k] = Z[:, k] - zeta * Z[:, i]

            # Permute columns k-1 and k of R and Z
            R[0:k, [k1, k]] = R[0:k, [k, k1]]
            Z[:, [k1, k]] = Z[:, [k, k1]]

            # Bring R back to an upper triangular matrix by a Givens rotation
            G, R[[k1, k], k1] = planerot(R[[k1, k], k1])
            #            R[[k1, k], k:n] = G @ R[[k1, k], k:n]

            # Apply the Givens rotation to y
            y_t = np.array([y[k1], y[k]])
            #            y_t = G @ y_t
            y[k1], y[k] = y_t[0], y_t[1]

            if k > 1:
                k = k - 1
        else:
            k = k + 1
    return R.ravel(), Z, y


# [R,piv,y] = qrmcp(A,y) computes the QR factorization of A with
#             minimum-column pivoting:
#             Q'BP = R (underdetermined A),
#             Q'BP = [R 0] (underdetermined A)
#             and computes Q'*y. The orthogonal matrix Q is not produced.
#
# Inputs:
#    A - m-by-n real matrix to be factorized
#    y - m-dimensional real vector to be transformed to Q'y
#
# Outputs:
#    R - m-by-n real upper trapezoidal matrix (m < n)
#        n-by-n real upper triangular matrix (m >= n)
#    piv - n-dimensional permutation vector representing P
#    y - m-vector transformed from the input y by Q, i.e., y := Q'*y

# Authors: Xiao-Wen Chang, www.cs.mcgill.ca/~chang
#          Xiaohu Xie, Tianyang Zhou
# Modified by Shilei Lin into Python
# Copyright (c) 2006-2021. Scientific Computing Lab, McGill University.
# October 2006 Last revision: June 2016
def qrmcp(A, y):
    m, n = A.shape
    colNormB = np.zeros((2, n))
    piv = np.array(range(0, n))

    for j in range(0, n):
        colNormB[0, j] = (np.linalg.norm(A[:, j])) ** 2

    n_dim = min(m - 1, n)
    for k in range(0, n_dim):
        # Find the column with minimum 2-norm in A(k:m,k:n)
        col = colNormB[0, k:n] - colNormB[1, k:n]
        print(col)
        min_index = np.argmin(col)
        min_value = col[min_index]

        col[min_index] = np.inf

        min_index_2 = np.argmin(col)
        min_value_2 = col[min_index_2]

        # if abs(min_value - min_value_2) < 1e-4:
        #     i = max(min_index, min_index_2)
        # else:
        i = min_index

        col[min_index] = min_value

        # i = np.argmin(colNormB[0, k:n] - colNormB[1, k:n])
        # print("%.100f, %.100f" % ((colNormB[0, k:n] - colNormB[1, k:n])[1], (colNormB[0, k:n] - colNormB[1, k:n])[17]))
        q = i + k
        print(i + 1)
        # Column interchange
        if q > k:
            print(piv.T + 1)
            piv[k], piv[q] = piv[q], piv[k]
            print(piv.T + 1)
            colNormB[:, [k, q]] = colNormB[:, [q, k]]
            A[:, [k, q]] = A[:, [q, k]]

        print(np.linalg.norm(A[k + 1:m, k]))
        if np.linalg.norm(A[k + 1:m, k]) > 0:  # A Householder transformation is needed
            v = np.array(A[k:m, k]).reshape(-1, 1)
            rho = np.linalg.norm(v)
            if v[0, 0] >= 0:
                rho = -rho
            v[0, 0] = v[0, 0] - rho
            tao = -1 / (rho * v[0, 0])
            A[k, k] = rho
            if m < n:
                A[k + 1:m, k] = 0

        #            A[k:m, k + 1:n] = A[k:m, k + 1:n] - tao * v * (v.T @ A[k:m, k + 1:n])

        # Update y by the Householder transformation
        #            y[k:m] = y[k:m, :] - tao * v * (v.T @ y[k:m])
        # print(y)

        # Update colnormB(2,k+1:n)
        colNormB[1, k + 1:n] = colNormB[1, k + 1:n] + A[k, k + 1:n] * A[k, k + 1:n]  # .* in Matlab!
        print(colNormB[1, k + 1:n])

    if m < n:
        R = A
    else:
        R = np.triu(A[0:n, 0:n])

    return R, piv, y

def matlabtest():
    import matlab.engine

    engine = matlab.engine.start_matlab()

# def test_matlab():
#     eng = matlab.engine.start_matlab()
#     a = matlab.double([1,4,9,16,25])
#     b = eng.sqrt(a)
#     print(b)
#     # A = eng.randn(256)
#
#
if __name__ == "__main__":
    matlabtest()
# fn = '/home/shilei/CLionProjects/babai_asyn/data/new32_35_3.nc'
# ds = nc.Dataset(fn)
# y = np.reshape(ds.variables['y_LLL'], (-1, 1)).astype(np.float32)
# A = np.reshape(ds.variables['A_A'], (-1, y.shape[0])).astype(np.float32)
# np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# print(A)


# R, Z, y = sils_reduction(A, y)
# print(A)
# print(y)
# print(R)
# print(np.get_include())
