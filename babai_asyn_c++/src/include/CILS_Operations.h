//
// Created by shilei on 2/28/22.
//

#ifndef CILS_CILS_MATRIX_H
#define CILS_CILS_MATRIX_H

#endif //CILS_CILS_MATRIX_H

#include "CILS_Iterator.h"
#include <exception>
#include <iostream>

namespace cils {

    template<typename Integer, typename Scalar>
    CILS_Matrix <Integer, Scalar> trans(CILS_Matrix <Integer, Scalar> &A) {
        CILS_Matrix<Integer, Scalar> A_t(A.size2(), A.size1());
        for (int i = 0; i < A.size1(); i++) {
            int m = A.size2();
            for (int coffset = 0; coffset < m; coffset++) {
                A_t(coffset, i) = A(i, coffset);
            }
        }
        return A_t;
    }

    //Ax = b
    template<typename Integer, typename Scalar>
    void prod(CILS_Matrix <Integer, Scalar> &A, CILS_Vector <Integer, Scalar> &x, CILS_Vector <Integer, Scalar> &b) {
        Scalar sum;
        b.resize(A.size1(), false);
        for (Integer row = 0; row < A.size1(); row++) {
            sum = 0;
            for (Integer col = 0; col < A.size2(); col++) {
                sum += A(row, col) * x[col];
            }
            b[row] = sum;
        }
    }

    template<typename Integer, typename Scalar>
    void
    projection(CILS_Matrix <Integer, Scalar> &A, CILS_Vector <Integer, Scalar> &x, CILS_Vector <Integer, Scalar> &b,
               Integer lower, Integer upper) {
        Scalar sum;
        b.clear();
        for (Integer row = 0; row < A.size1(); row++) {
            sum = 0;
            for (Integer col = 0; col < A.size2(); col++) {
                sum += A(row, col) * x[col];
            }
            b[row] = std::max(std::min((Integer) sum, upper), lower);
        }
    }

    //QR = A
    template<typename Integer, typename Scalar>
    void prod(CILS_Matrix <Integer, Scalar> &Q, CILS_Matrix <Integer, Scalar> &R, CILS_Matrix <Integer, Scalar> &A) {
        Scalar sum = 0;
        if (Q.size2() != R.size1()) {
            std::cerr << " Q.size2() != R.size1() Size mismatch.";
            return;
        }
        A.resize(Q.size1(), R.size2());
        for (unsigned int j = 0; j < Q.size1(); j++) {
            for (unsigned int k = 0; k < Q.size2(); k++) {
                for (unsigned int i = 0; i < R.size2(); i++) {
                    A(j, i) += Q(j, k) * R(k, i);
                }
            }
        }
    }

    template<typename Integer, typename Scalar>
    Scalar norm_2(CILS_Vector <Integer, Scalar> &y) {
        Scalar sum = 0;
        std::for_each(y.begin(), y.end(), [&sum](auto const &i) {
            sum += i * i;
        });
        return sqrt(sum);
    }

    template<typename Integer, typename Scalar>
    Scalar inner_prod(CILS_Vector <Integer, Scalar> &x,
                      CILS_Vector <Integer, Scalar> &y) {
        Scalar sum = 0;
        for (unsigned int i; i < x.size(); i++) {
            sum += x[i] * y[i];
        }
        return sum;
    }
}