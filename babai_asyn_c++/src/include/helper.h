/** \file
 * \brief Computation of integer least square problem
 * \author Shilei Lin
 * This file is part of CILS.
 *   CILS is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   CILS is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with CILS.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CILS_HELPER_H
#define CILS_HELPER_H

#endif //CILS_HELPER_H

#include <cmath>
#include <cstring>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <boost/numeric/ublas/vector_proxy.hpp>
#include <boost/numeric/ublas/triangular.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/math/tools/norms.hpp>

using namespace boost::numeric::ublas;


const double ZERO = 3.3121686421112381E-170;
using namespace std;
namespace helper {


    /**
     * Givens plane rotation
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param x : A 2-component column vector.
     * @param G : A 2-by-2 orthogonal matrix G so that y = G*x has y(2) = 0.
     */
    template<typename scalar, typename index>
    void planerot(scalar x[2], scalar G[4]) {
        if (x[1] != 0.0) {
            scalar absxk, r, scale, t;
            scale = ZERO;
            absxk = std::abs(x[0]);
            if (absxk > ZERO) {
                r = 1.0;
                scale = absxk;
            } else {
                t = absxk / ZERO;
                r = t * t;
            }
            absxk = std::abs(x[1]);
            if (absxk > scale) {
                t = scale / absxk;
                r = r * t * t + 1.0;
                scale = absxk;
            } else {
                t = absxk / scale;
                r += t * t;
            }
            r = scale * std::sqrt(r);
            scale = x[0] / r;
            G[0] = scale;
            G[2] = x[1] / r;
            G[1] = -x[1] / r;
            G[3] = scale;
            x[0] = r;
            x[1] = 0.0;
        } else {
            G[1] = 0.0;
            G[2] = 0.0;
            G[0] = 1.0;
            G[3] = 1.0;
        }
    }

//
//    /**
//     * The Euclidean norm of vector v. This norm is also called the 2-norm, vector magnitude, or Euclidean length.
//     * @tparam scalar : real number type
//     * @tparam index : integer type
//     * @param n : the size of the vector
//     * @param v : input vector
//     * @return
//     */
//    template<typename scalar, typename index>
//    scalar norm(const index n, const scalar *v) {
//        scalar y;
//        if (n == 0) {
//            return 0;
//        } else if (n == 1) {
//            return std::abs(v[0]);
//        }
//        scalar scale = ZERO;
//        for (index k = 0; k < n; k++) {
//            scalar absxk;
//            absxk = std::abs(v[k]);
//            if (absxk > scale) {
//                scalar t;
//                t = scale / absxk;
//                y = y * t * t + 1.0;
//                scale = absxk;
//            } else {
//                scalar t;
//                t = absxk / scale;
//                y += t * t;
//            }
//        }
//        y = scale * std::sqrt(y);
//        return y;
//    }
//
    /**
     * Find BER with given two vectors
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param n : integer scalar, size of the vector
     * @param x_b : input vector 1
     * @param x_t : input vector 2
     * @param k : log_4(qam)
     * @return
     */
    template<typename scalar, typename index>
    scalar find_bit_error_rate(b_vector &x_b, b_vector &x_t, const index k) {
        index error = 0, n = x_t.size();
        for (index i = 0; i < n; i++) {
            std::string binary_x_b, binary_x_t;
            switch (k) {
                case 1:
                    binary_x_b = std::bitset<1>((index) x_b[i]).to_string(); //to binary
                    binary_x_t = std::bitset<1>((index) x_t[i]).to_string();
                    break;
                case 2:
                    binary_x_b = std::bitset<2>((index) x_b[i]).to_string(); //to binary
                    binary_x_t = std::bitset<2>((index) x_t[i]).to_string();
                    break;
                default:
                    binary_x_b = std::bitset<3>((index) x_b[i]).to_string(); //to binary
                    binary_x_t = std::bitset<3>((index) x_t[i]).to_string();
                    break;
            }
//            cout << binary_x_b << "-" << binary_x_t << " ";
            for (index j = 0; j < k; j++) {
                if (binary_x_b[j] != binary_x_t[j])
                    error++;
            }
        }
        return (scalar) error / (n * k);
    }

    /**
     * Simple function for displaying a m-by-n matrix with name
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param A : matrix, in pointer
     * @param name: display name of the matrix
     */
    template<typename scalar, typename index>
    void display(b_matrix &A, const string &name) {
        cout << name << ": \n";
        for (index row = 0; row < A.size1(); row++) {
            for (index col = 0; col < A.size2(); col++) {
                printf("%8.4f ", A(row, col));
            }
            cout << "\n";
        }
        cout << endl;
    }

    /**
     * Simple function for displaying a m-by-n matrix with name
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param b : vector, in pointer
     * @param name: display name of the matrix
     */
    template<typename scalar, typename index>
    void display(scalar *b, const index size, const string &name) {
        cout << name << ": \n";
        for (index row = 0; row < size; row++) {
            printf("%8.4d ", b[row]);
        }
        cout << endl;
    }

    /**
     * Simple function for displaying the a vector with name
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param n : integer scalar, size of the vector
     * @param x : vector, in pointer
     * @param name: display name of the vector
     */
    template<typename scalar, typename index>
    void display(b_vector &x, const string &name) {
        cout << name << ": ";
        scalar sum = 0;
        for (index i = 0; i < x.size(); i++) {
            printf("%8.4f ", x[i]);
            sum += x[i];
        }
        printf("SUM = %8.5f\n", sum);
    }

    /**
     * Simple function for displaying the a vector with name
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param n : integer scalar, size of the vector
     * @param x : vector, in pointer
     * @param name: display name of the vector
     */
    template<typename scalar, typename index>
    void display(const si_vector &x, const string &name) {
        cout << name << ": ";
        scalar sum = 0;
        for (index i = 0; i < x.size(); i++) {
            printf("%8d ", x[i]);
            sum += x[i];
        }
        printf("SUM = %8.5f\n", sum);
    }

    /**
     * Simple function for displaying the a vector with name
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param n : integer scalar, size of the vector
     * @param x : vector, in pointer
     * @param name: display name of the vector
     */
    template<typename scalar, typename index>
    void display(const index *x, const index n, const string &name) {
        cout << name << ": ";
        scalar sum = 0;
        for (index i = 0; i < n; i++) {
            printf("%8d ", x[i]);
            sum += x[i];
        }
        printf("SUM = %8.5f\n", sum);
    }

    template<typename scalar, typename index>
    void inv(b_matrix &input, b_matrix &output) {
        //Clear variable.
        output.resize(input.size1(), input.size2());
        output.clear();

        typedef permutation_matrix<std::size_t> pmatrix;
        typedef boost::numeric::ublas::matrix<double> matrix;

        // create a working copy of the input
        matrix M(input.size1(), input.size2());
        for (int i = 0; i < input.size1() * input.size2(); i++) {
            M(i) = input[i];
        }

        // create a permutation matrix for the LU-factorization
        pmatrix pm(M.size1());
        // perform LU-factorization
        auto res = lu_factorize(M, pm);

        // create identity matrix of "inverse"
        matrix M_I(identity_matrix<typename matrix::value_type>(M.size1()));
        // backsubstitute to get the inverse
        lu_substitute(M, pm, M_I);
        output.resize(input.size1(), input.size2());
        output.clear();

        for (int i = 0; i < input.size1() * input.size2(); i++) {
            output[i] = M_I(i);
        }
    }
    
    template<typename scalar, typename index>
    void inv2(const b_matrix &x, b_matrix &y) {
        b_matrix b_x, ipiv, p;
        if ((x.size1() == 0) || (x.size2() == 0)) {
            int b_n;
            y.resize(x.size1(), x.size2());
            b_n = x.size1() * x.size2();
            for (int i{0}; i < b_n; i++) {
                y[i] = x[i];
            }
        } else {
            int b_n;
            int i;
            int i1;
            int ldap1;
            int n;
            int u1;
            int yk;
            n = x.size1();
            y.resize(x.size1(), x.size2());
            b_n = x.size1() * x.size2();
            for (i = 0; i < b_n; i++) {
                y[i] = 0.0;
            }
            b_x.resize(x.size1(), x.size2());
            b_n = x.size1() * x.size2();
            for (i = 0; i < b_n; i++) {
                b_x[i] = x[i];
            }
            b_n = x.size1();
            ipiv.resize(1, x.size1());
            ipiv[0] = 1;
            yk = 1;
            for (int k{2}; k <= b_n; k++) {
                yk++;
                ipiv[k - 1] = yk;
            }
            ldap1 = x.size1();
            b_n = x.size1() - 1;
            u1 = x.size1();
            if (b_n <= u1) {
                u1 = b_n;
            }
            for (int j{0}; j < u1; j++) {
                double smax;
                int jj;
                int jp1j;
                int mmj_tmp;
                mmj_tmp = n - j;
                yk = j * (n + 1);
                jj = j * (ldap1 + 1);
                jp1j = yk + 2;
                if (mmj_tmp < 1) {
                    b_n = -1;
                } else {
                    b_n = 0;
                    if (mmj_tmp > 1) {
                        smax = std::abs(b_x[jj]);
                        for (int k{2}; k <= mmj_tmp; k++) {
                            double s;
                            s = std::abs(b_x[(yk + k) - 1]);
                            if (s > smax) {
                                b_n = k - 1;
                                smax = s;
                            }
                        }
                    }
                }
                if (b_x[jj + b_n] != 0.0) {
                    if (b_n != 0) {
                        b_n += j;
                        ipiv[j] = b_n + 1;
                        for (int k{0}; k < n; k++) {
                            smax = b_x[j + k * n];
                            b_x[j + k * n] = b_x[b_n + k * n];
                            b_x[b_n + k * n] = smax;
                        }
                    }
                    i = jj + mmj_tmp;
                    for (int b_i{jp1j}; b_i <= i; b_i++) {
                        b_x[b_i - 1] = b_x[b_i - 1] / b_x[jj];
                    }
                }
                b_n = yk + n;
                yk = jj + ldap1;
                for (jp1j = 0; jp1j <= mmj_tmp - 2; jp1j++) {
                    smax = b_x[b_n + jp1j * n];
                    if (b_x[b_n + jp1j * n] != 0.0) {
                        i = yk + 2;
                        i1 = mmj_tmp + yk;
                        for (int b_i{i}; b_i <= i1; b_i++) {
                            b_x[b_i - 1] = b_x[b_i - 1] + b_x[((jj + b_i) - yk) - 1] * -smax;
                        }
                    }
                    yk += n;
                }
            }
            b_n = x.size1();
            p.resize(1, x.size1());
            p[0] = 1;
            yk = 1;
            for (int k{2}; k <= b_n; k++) {
                yk++;
                p[k - 1] = yk;
            }
            i = ipiv.size2();
            for (int k{0}; k < i; k++) {
                i1 = ipiv[k];
                if (i1 > k + 1) {
                    b_n = p[i1 - 1];
                    p[i1 - 1] = p[k];
                    p[k] = b_n;
                }
            }
            for (int k{0}; k < n; k++) {
                i = p[k];
                y[k + y.size1() * (i - 1)] = 1.0;
                for (int j{k + 1}; j <= n; j++) {
                    if (y[(j + y.size1() * (i - 1)) - 1] != 0.0) {
                        i1 = j + 1;
                        for (int b_i{i1}; b_i <= n; b_i++) {
                            y[(b_i + y.size1() * (i - 1)) - 1] =
                                    y[(b_i + y.size1() * (i - 1)) - 1] -
                                    y[(j + y.size1() * (i - 1)) - 1] *
                                    b_x[(b_i + b_x.size1() * (j - 1)) - 1];
                        }
                    }
                }
            }
            for (int j{0}; j < n; j++) {
                b_n = n * j - 1;
                for (int k{n}; k >= 1; k--) {
                    yk = n * (k - 1) - 1;
                    i = k + b_n;
                    if (y[i] != 0.0) {
                        y[i] = y[i] / b_x[k + yk];
                        for (int b_i{0}; b_i <= k - 2; b_i++) {
                            i1 = (b_i + b_n) + 1;
                            y[i1] = y[i1] - y[i] * b_x[(b_i + yk) + 1];
                        }
                    }
                }
            }
        }
    }
    /**
     * Determine whether all values of x are true by lambda expression.
     * @tparam index : integer type : integer required
     * @param x : Testing vector
     * @return true/false
     */
    template<typename index>
    bool if_all_x_true(const std::vector<bool> &x) {
        bool y = (!x.empty());
        if (!y) return y;
//        for(index k = 0; k < x.size(); k++){
//            if(!x[k]) {
//                y = false;
//                break;
//            }
//        }
        //If false, which means no false x, then return true.
        y = std::any_of(x.begin(), x.end(), [](const bool &e) { return !e; });
        return !y;
    }

    /**
     * Returns the same data as in a, but with no repetitions. b is in sorted order.
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param a : input vector to be processed
     * @param b : output vector to store the results
     */
    template<typename scalar, typename index>
    void unique_vector(b_vector &a, std::vector<scalar> &b) {
        index t, e, i, l, j, k, size_a_1 = a.size() + 1, size_a = a.size(), jj, p, r, q, r_j;
        std::vector<index> i_x(size_a, 0), j_x(size_a, 0);
        scalar absx;
        for (i = 1; i < size_a; i += 2) {
            if ((a[i - 1] <= a[i]) || std::isnan(a[i])) {
                i_x[i - 1] = i;
                i_x[i] = i + 1;
            } else {
                i_x[i - 1] = i + 1;
                i_x[i] = i;
            }
        }
        if ((size_a & 1) != 0) {
            i_x[size_a - 1] = size_a;
        }
        i = 2;
        while (i < size_a_1 - 1) {
            l = i << 1;
            j = 1;
            for (p = i + 1; p < size_a_1; p = r + i) {
                jj = j;
                q = p - 1;
                r = j + l;
                if (r > size_a_1) {
                    r = size_a_1;
                }
                k = 0;
                r_j = r - j;
                while (k + 1 <= r_j) {
                    absx = a[i_x[q] - 1];
                    t = i_x[jj - 1];
                    if ((a[t - 1] <= absx) || std::isnan(absx)) {
                        j_x[k] = t;
                        jj++;
                        if (jj == p) {
                            while (q + 1 < r) {
                                k++;
                                j_x[k] = i_x[q];
                                q++;
                            }
                        }
                    } else {
                        j_x[k] = i_x[q];
                        q++;
                        if (q + 1 == r) {
                            while (jj < p) {
                                k++;
                                j_x[k] = i_x[jj - 1];
                                jj++;
                            }
                        }
                    }
                    k++;
                }
                for (k = 0; k < r_j; k++) {
                    i_x[(j + k) - 1] = j_x[k];
                }
                j = r;
            }
            i = l;
        }
        b.resize(size_a);
        for (k = 0; k < size_a; k++) {
            b[k] = a[i_x[k] - 1];
        }
        k = 0;
        while ((k + 1 <= size_a) && std::isinf(b[k]) && (b[k] < 0.0)) {
            k++;
        }
        l = k;
        k = size_a;
        while ((k >= 1) && std::isnan(b[k - 1])) {
            k--;
        }
        p = size_a - k;
        bool flag = false;
        while ((!flag) && (k >= 1)) {
            if (std::isinf(b[k - 1]) && (b[k - 1] > 0.0)) {
                k--;
            } else {
                flag = true;
            }
        }
        i = (size_a - k) - p;
        jj = -1;
        if (l > 0) {
            jj = 0;
        }
        while (l + 1 <= k) {
            scalar x;
            x = b[l];
            index exitg2;
            do {
                exitg2 = 0;
                l++;
                if (l + 1 > k) {
                    exitg2 = 1;
                } else {
                    absx = std::abs(x / 2.0);
                    if ((!std::isinf(absx)) && (!std::isnan(absx))) {
                        if (absx <= ZERO) {
                            absx = ZERO;
                        } else {
                            frexp(absx, &e);
                            absx = std::ldexp(1.0, e - 53);
                        }
                    } else {
                        absx = NAN;
                    }
                    if ((!(std::abs(x - b[l]) < absx)) &&
                        ((!std::isinf(b[l])) || (!std::isinf(x)) ||
                         ((b[l] > 0.0) != (x > 0.0)))) {
                        exitg2 = 1;
                    }
                }
            } while (exitg2 == 0);
            jj++;
            b[jj] = x;
        }
        if (i > 0) {
            jj++;
            b[jj] = b[k];
        }
        l = k + i;
        for (j = 0; j < p; j++) {
            b[(jj + j) + 1] = b[l + j];
        }
        jj += p;
        if (1 > jj + 1) {
            t = 0;
        } else {
            t = jj + 1;
        }
        b.resize(t);
    }
//
//    /**
//     * Return the result of ||y-A*x||.
//     * @tparam scalar : real number type
//     * @tparam index : integer type
//     * @param m : integer scalar, size of the matrix
//     * @param n : integer scalar, size of the matrix
//     * @param A : matrix, m-by-n in pointer
//     * @param x : vector, n-by-1 in pointer
//     * @param y : vector, m-by-1 in pointer, storing result.
//     * @return residual : l2 norm
//     */
////    template<typename scalar, typename index>
////    inline scalar find_residual(const index m, const index n, const scalar *A, const scalar *x, const scalar *y) {
////        vector<scalar> Ax(m, 0);
////        mtimes_Axy(m, n, A, x, Ax.data());
////        for (index i = 0; i < m; i++) {
////            Ax[i] = y[i] - Ax[i];
////        }
////        return norm(m, Ax.data());
////    }
//
    /**
     * Return the result of ||y-Ax||_2 By Boost.
     * @tparam scalar : real number type
     * @tparam index : integer type
     * @param A : matrix, m-by-n in pointer
     * @param x : vector, n-by-1 in pointer
     * @param y : vector, m-by-1 in pointer, storing result.
     * @return residual : l2 norm
     */
    template<typename scalar, typename index>
    inline scalar find_residual(b_matrix &A, b_vector &x, b_vector &y) {
        b_vector Ax;
        prod(A, x, Ax);
        b_vector yAx(y.size());
        for (index i = 0; i < y.size(); i++){
            yAx[i] = y[i] - Ax[i];
        }
        return norm_2(yAx);
    }


    static double rt_hypotd_snf(double u0, double u1) {
        double a = std::abs(u0);
        double y = std::abs(u1);
        if (a < y) {
            a /= y;
            y *= std::sqrt(a * a + 1.0);
        } else if (a > y) {
            y /= a;
            y = a * std::sqrt(y * y + 1.0);
        } else if (!std::isnan(y)) {
            y = a * 1.4142135623730951;
        }
        return y;
    }

    namespace coder::internal {
        namespace blas {
            static void xgerc(int m, int n, double alpha1, int ix0, const double *y, double *A, int ia0, int lda) {
                if (alpha1 != 0.0) {
                    for (int j = 0; j < n; j++) {
                        if (y[j] != 0.0) {
                            double temp = y[j] * alpha1;
                            for (int i = ia0; i < m + ia0; i++) {
                                A[i - 1] = A[i - 1] + A[ix0 + i - ia0 - 1] * temp;
                            }
                        }
                        ia0 += lda;
                    }
                }
            }

            static double xnrm2(int n, const double *x, int ix0) {
                double y;
                y = 0.0;
                if (n >= 1) {
                    if (n == 1) {
                        y = std::abs(x[ix0 - 1]);
                    } else {
                        double scale = ZERO;
                        int kend = (ix0 + n) - 1;
                        for (int k = ix0; k <= kend; k++) {
                            double absxk = std::abs(x[k - 1]);
                            if (absxk > scale) {
                                double t = scale / absxk;
                                y = y * t * t + 1.0;
                                scale = absxk;
                            } else {
                                double t = absxk / scale;
                                y += t * t;
                            }
                        }
                        y = scale * std::sqrt(y);
                    }
                }
                return y;
            }

//
//
        } // namespace blas
        namespace lapack {
            static void xgeqp3(double *A, sd_vector &tau, si_vector &jpvt, int m, int n) {

                int i, knt, minmana;
                bool guard1;
                knt = m;
                minmana = n;
                if (knt < minmana) {
                    minmana = knt;
                }
                tau.resize(minmana);
                tau.clear();
                guard1 = false;

                auto work = new double[n]{};
                auto vn1 = new double[n]{};
                auto vn2 = new double[n]{};

                if ((m == 0) || (n == 0)) {
                    guard1 = true;
                } else {
                    knt = m;
                    minmana = n;
                    if (knt < minmana) {
                        minmana = knt;
                    }
                    if (minmana < 1) {
                        guard1 = true;
                    } else {
                        double smax;
                        int k, ma, minmn;
                        jpvt.resize(n);
                        jpvt.clear();
                        for (k = 0; k < n; k++) {
                            jpvt[k] = k + 1;
                        }
                        ma = m;
                        knt = m;
                        minmn = n;
                        if (knt < minmn) {
                            minmn = knt;
                        }


                        for (knt = 0; knt < n; knt++) {
                            smax = blas::xnrm2(m, A, knt * ma + 1);
                            vn1[knt] = smax;
                            vn2[knt] = smax;
                        }
                        for (int b_i = 0; b_i < minmn; b_i++) {
                            double s, temp2;
                            int pvt, ip1 = b_i + 2, ii_tmp = b_i * ma, ii = ii_tmp + b_i, nmi = n - b_i, mmi = m - b_i;
                            if (nmi < 1) {
                                minmana = -1;
                            } else {
                                minmana = 0;
                                if (nmi > 1) {
                                    smax = std::abs(vn1[b_i]);
                                    for (k = 2; k <= nmi; k++) {
                                        s = std::abs(vn1[(b_i + k) - 1]);
                                        if (s > smax) {
                                            minmana = k - 1;
                                            smax = s;
                                        }
                                    }
                                }
                            }
                            pvt = b_i + minmana;
                            if (pvt + 1 != b_i + 1) {
                                minmana = pvt * ma;
                                for (k = 0; k < m; k++) {
                                    knt = minmana + k;
                                    smax = A[knt];
                                    i = ii_tmp + k;
                                    A[knt] = A[i];
                                    A[i] = smax;
                                }
                                minmana = jpvt[pvt];
                                jpvt[pvt] = jpvt[b_i];
                                jpvt[b_i] = minmana;
                                vn1[pvt] = vn1[b_i];
                                vn2[pvt] = vn2[b_i];
                            }
                            if (b_i + 1 < m) {
                                temp2 = A[ii];
                                minmana = ii + 2;
                                tau[b_i] = 0.0;
                                if (mmi > 0) {
                                    smax = blas::xnrm2(mmi - 1, A, ii + 2);
                                    if (smax != 0.0) {
                                        s = rt_hypotd_snf(A[ii], smax);
                                        if (A[ii] >= 0.0) {
                                            s = -s;
                                        }
                                        if (std::abs(s) < 1.0020841800044864E-292) {
                                            knt = -1;
                                            i = ii + mmi;
                                            do {
                                                knt++;
                                                for (k = minmana; k <= i; k++) {
                                                    A[k - 1] = 9.9792015476736E+291 * A[k - 1];
                                                }
                                                s *= 9.9792015476736E+291;
                                                temp2 *= 9.9792015476736E+291;
                                            } while (std::abs(s) < 1.0020841800044864E-292);
                                            s = rt_hypotd_snf(temp2, blas::xnrm2(mmi - 1, A, ii + 2));
                                            if (temp2 >= 0.0) {
                                                s = -s;
                                            }
                                            tau[b_i] = (s - temp2) / s;
                                            smax = 1.0 / (temp2 - s);
                                            for (k = minmana; k <= i; k++) {
                                                A[k - 1] = smax * A[k - 1];
                                            }
                                            for (k = 0; k <= knt; k++) {
                                                s *= 1.0020841800044864E-292;
                                            }
                                            temp2 = s;
                                        } else {
                                            tau[b_i] = (s - A[ii]) / s;
                                            smax = 1.0 / (A[ii] - s);
                                            i = ii + mmi;
                                            for (k = minmana; k <= i; k++) {
                                                A[k - 1] = smax * A[k - 1];
                                            }
                                            temp2 = s;
                                        }
                                    }
                                }
                                A[ii] = temp2;
                            } else {
                                tau[b_i] = 0.0;
                            }
                            if (b_i + 1 < n) {
                                int ia;
                                temp2 = A[ii];
                                A[ii] = 1.0;
                                ii_tmp = (ii + ma) + 1;
                                if (tau[b_i] != 0.0) {
                                    bool exitg2;
                                    pvt = mmi;
                                    minmana = (ii + mmi) - 1;
                                    while ((pvt > 0) && (A[minmana] == 0.0)) {
                                        pvt--;
                                        minmana--;
                                    }
                                    knt = nmi - 1;
                                    exitg2 = false;
                                    while ((!exitg2) && (knt > 0)) {
                                        int exitg1;
                                        minmana = ii_tmp + (knt - 1) * ma;
                                        ia = minmana;
                                        do {
                                            exitg1 = 0;
                                            if (ia <= (minmana + pvt) - 1) {
                                                if (A[ia - 1] != 0.0) {
                                                    exitg1 = 1;
                                                } else {
                                                    ia++;
                                                }
                                            } else {
                                                knt--;
                                                exitg1 = 2;
                                            }
                                        } while (exitg1 == 0);
                                        if (exitg1 == 1) {
                                            exitg2 = true;
                                        }
                                    }
                                } else {
                                    pvt = 0;
                                    knt = 0;
                                }
                                if (pvt > 0) {
                                    if (knt != 0) {
                                        for (k = 0; k < knt; k++) {
                                            work[k] = 0.0;
                                        }
                                        k = 0;
                                        i = ii_tmp + ma * (knt - 1);
                                        for (nmi = ii_tmp; ma < 0 ? nmi >= i : nmi <= i; nmi += ma) {
                                            smax = 0.0;
                                            minmana = (nmi + pvt) - 1;
                                            for (ia = nmi; ia <= minmana; ia++) {
                                                smax += A[ia - 1] * A[(ii + ia) - nmi];
                                            }
                                            work[k] = work[k] + smax;
                                            k++;
                                        }
                                    }
                                    blas::xgerc(pvt, knt, -tau[b_i], ii + 1, work, A, ii_tmp, ma);
                                }
                                A[ii] = temp2;
                            }
                            for (knt = ip1; knt <= n; knt++) {
                                minmana = b_i + (knt - 1) * ma;
                                smax = vn1[knt - 1];
                                if (smax != 0.0) {
                                    s = std::abs(A[minmana]) / smax;
                                    s = 1.0 - s * s;
                                    if (s < 0.0) {
                                        s = 0.0;
                                    }
                                    temp2 = smax / vn2[knt - 1];
                                    temp2 = s * (temp2 * temp2);
                                    if (temp2 <= 1.4901161193847656E-8) {
                                        if (b_i + 1 < m) {
                                            smax = blas::xnrm2(mmi - 1, A, minmana + 2);
                                            vn1[knt - 1] = smax;
                                            vn2[knt - 1] = smax;
                                        } else {
                                            vn1[knt - 1] = 0.0;
                                            vn2[knt - 1] = 0.0;
                                        }
                                    } else {
                                        vn1[knt - 1] = smax * std::sqrt(s);
                                    }
                                }
                            }
                        }


                    }
                }
                if (guard1) {
                    jpvt.resize(n);
                    jpvt.clear();
                    for (i = 0; i < n; i++) {
                        jpvt[i] = i + 1;
                    }
                }

                delete[] work;
                delete[] vn1;
                delete[] vn2;
            }

//
//
            static void xorgqr(int m, int n, int k, double *A, int lda, const sd_vector &tau) {

                if (n >= 1) {
                    int b_i, b_k, c_i, i, ia, itau;
                    i = n - 1;
                    for (b_i = k; b_i <= i; b_i++) {
                        ia = b_i * lda;
                        b_k = m - 1;
                        for (c_i = 0; c_i <= b_k; c_i++) {
                            A[ia + c_i] = 0.0;
                        }
                        A[ia + b_i] = 1.0;
                    }
                    itau = k - 1;
                    b_i = n;
                    auto work = new double[b_i]{};
                    for (c_i = k; c_i >= 1; c_i--) {
                        int iaii = c_i + (c_i - 1) * lda;
                        if (c_i < n) {
                            int ic0;
                            int lastc;
                            int lastv;
                            A[iaii - 1] = 1.0;
                            ic0 = iaii + lda;
                            if (tau[itau] != 0.0) {
                                bool exitg2;
                                lastv = (m - c_i) + 1;
                                b_i = (iaii + m) - c_i;
                                while ((lastv > 0) && (A[b_i - 1] == 0.0)) {
                                    lastv--;
                                    b_i--;
                                }
                                lastc = n - c_i;
                                exitg2 = false;
                                while ((!exitg2) && (lastc > 0)) {
                                    int exitg1;
                                    b_i = ic0 + (lastc - 1) * lda;
                                    ia = b_i;
                                    do {
                                        exitg1 = 0;
                                        if (ia <= (b_i + lastv) - 1) {
                                            if (A[ia - 1] != 0.0) {
                                                exitg1 = 1;
                                            } else {
                                                ia++;
                                            }
                                        } else {
                                            lastc--;
                                            exitg1 = 2;
                                        }
                                    } while (exitg1 == 0);
                                    if (exitg1 == 1) {
                                        exitg2 = true;
                                    }
                                }
                            } else {
                                lastv = 0;
                                lastc = 0;
                            }
                            if (lastv > 0) {
                                if (lastc != 0) {
                                    for (b_i = 0; b_i < lastc; b_i++) {
                                        work[b_i] = 0.0;
                                    }
                                    b_i = 0;
                                    i = ic0 + lda * (lastc - 1);
                                    for (int iac{ic0}; lda < 0 ? iac >= i : iac <= i; iac += lda) {
                                        double c = 0.0;
                                        b_k = (iac + lastv) - 1;
                                        for (ia = iac; ia <= b_k; ia++) {
                                            c += A[ia - 1] * A[((iaii + ia) - iac) - 1];
                                        }
                                        work[b_i] = work[b_i] + c;
                                        b_i++;
                                    }
                                }
                                blas::xgerc(lastv, lastc, -tau[itau], iaii, work, A, ic0, lda);
                            }
                        }
                        if (c_i < m) {
                            b_i = iaii + 1;
                            i = (iaii + m) - c_i;
                            for (b_k = b_i; b_k <= i; b_k++) {
                                A[b_k - 1] = -tau[itau] * A[b_k - 1];
                            }
                        }
                        A[iaii - 1] = 1.0 - tau[itau];
                        for (b_i = 0; b_i <= c_i - 2; b_i++) {
                            A[(iaii - b_i) - 2] = 0.0;
                        }
                        itau--;
                    }
                    delete[] work;
                }
            }

        } // namespace lapack
    } // namespace coder


    /**
     * QR with Column Pivoting.
     * @param A_A 
     * @param Q_A 
     * @param R_A 
     * @param P_A 
     * @param _q 
     * @param eval 
     */
    void qrp(b_matrix &A_A, b_matrix &P_A, b_matrix &R_A, bool _q, bool eval) {
        b_matrix Q_A;
        int m = (int) A_A.size1(), n = (int) A_A.size2();

        auto A = new double[m * n]{};
        auto Q = new double[m * m]{};
        auto R = new double[m * n]{};
        auto P = new double[n * n]{};

        sd_vector tau;
        si_vector jpvt, jpvt1;

        int i, j, l;

        for (int col = 0; col < n; col++) {
            for (int row = 0; row < m; row++) {
                A[row + col * m] = A_A[row * n + col];
            }
        }

        if (m > n) {
            for (j = 0; j < n; j++) {
                for (l = 0; l < m; l++) {
                    Q[l + m * j] = A[l + m * j];
                }
            }
            i = n + 1;
            for (j = i; j < m + 1; j++) {
                for (l = 0; l < m; l++) {
                    Q[l + m * (j - 1)] = 0.0;
                }
            }
            coder::internal::lapack::xgeqp3(Q, tau, jpvt1, m, m);
            jpvt.resize(1, n);
            for (j = 0; j < n; j++) {
                jpvt[j] = jpvt1[j];
                for (l = 0; l <= j; l++) {
                    R[l + m * j] = Q[l + m * j];
                }
                i = j + 2;
                for (l = i; l < m + 1; l++) {
                    R[(l + m * j) - 1] = 0.0;
                }
            }
            coder::internal::lapack::xorgqr(m, m, n, Q, m, tau);
        } else {
            coder::internal::lapack::xgeqp3(A, tau, jpvt, m, n);
            for (j = 0; j < m; j++) {
                for (l = 0; l <= j; l++) {
                    R[l + m * j] = A[l + m * j];
                }
                i = j + 2;
                for (l = i; l < m + 1; l++) {
                    R[(l + m * j) - 1] = 0.0;
                }
            }
            i = m + 1;
            for (j = i; j <= n; j++) {
                for (l = 0; l < m; l++) {
                    R[l + m * (j - 1)] = A[l + m * (j - 1)];
                }
            }
            coder::internal::lapack::xorgqr(m, m, m, A, m, tau);
            if (_q || eval) {
                for (j = 0; j < m; j++) {
                    for (l = 0; l < m; l++) {
                        Q[l + m * j] = A[l + m * j];
                    }
                }
            }
        }

        for (j = 0; j < n; j++) {
            P[(jpvt[j] + n * j) - 1] = 1.0;
        }

        P_A.resize(n, n);
        for (int col = 0; col < n; col++) {
            for (int row = 0; row < n; row++) {
                P_A[row * P_A.size2() + col] = P[row + col * n];
            }
        }

        R_A.resize(m, n);
        for (int col = 0; col < m; col++) {
            for (int row = 0; row < n; row++) {
                R_A[row + col * n] = R[row * m + col];
            }
        }

        if (_q || eval) {
            Q_A.resize(m, m);
            for (int col = 0; col < m; col++) {
                for (int row = 0; row < m; row++) {
                    Q_A[row * Q_A.size2() + col] = Q[row + col * Q_A.size1()];
                }
            }
        }
        if (eval) {
            //Verify QRP i.e., A = QRP.
            b_matrix A_qr;
            prod(Q_A, R_A, A_qr);
            b_matrix A_pa, PAT;
            PAT = trans(P_A);
            prod(A_qr, PAT, A_pa);
            helper::display<double, int>(A_A, "A_A");
            helper::display<double, int>(A_pa, "A");
        }

        delete[] A;
        delete[] Q;
        delete[] R;
        delete[] P;
    }
}
