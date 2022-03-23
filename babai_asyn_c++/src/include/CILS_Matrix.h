//
// Created by shilei on 2/28/22.
//

#ifndef CILS_CILS_MATRIX_H
#define CILS_CILS_MATRIX_H

#endif //CILS_CILS_MATRIX_H


#include <exception>
#include <iostream>


namespace cils {

    template<typename Integer, typename Scalar>
    class CILS_Matrix {

    private:
        std::vector<Scalar> x;
        Integer s1, s2;

    public:

        Iterator <Integer, Scalar> begin() {
            return Iterator<Integer, Scalar>(&x[0]);
        }

        Iterator <Integer, Scalar> end() {
            return Iterator<Integer, Scalar>(&x[s1 * s2]);
        }

        CILS_Matrix() = default;

        CILS_Matrix(Integer size1, Integer size2) {
            this->s1 = size1;
            this->s2 = size2;
            this->x.resize(s1 * s2);
        }

        CILS_Matrix(CILS_Matrix &B) {
            this->s1 = B.s1;
            this->s2 = B.s2;
            this->x.resize(s1 * s2);
            std::copy(B.begin(), B.end(), this->begin());
        }

        ~CILS_Matrix() {
//            delete[] x;
        }

        Integer size1() {
            return s1;
        }

        Integer size2() {
            return s2;
        }

        void resize(Integer new_size1, Integer new_size2, bool keep = false) {
            this->s1 = new_size1;
            this->s2 = new_size2;
            this->x.resize(s1 * s2, keep);
        }

        void assign(Scalar value) {
            std::fill_n(this->begin(), s1 * s2, value);
        }


        void assign(CILS_Matrix &B) {
            std::copy(B.begin(), B.end(), this->begin());
        }


        void assign(CILS_Identity_Matrix <Integer, Scalar> &I) {
            std::copy(I.begin(), I.end(), this->begin());
        }


        Scalar &at_element(const Integer row, const Integer col) {
            return x[row + col * s2];
        }

        const Scalar &at_element(const Integer row, const Integer col) const {
            return x[row + col * s2];
        }

        void assign(Integer col, CILS_Vector <Integer, Scalar> &y) {
            for (unsigned int i = 0; i < s1; i++) {
                at_element(i, col) = y[i];
            }
        }

        void column(Integer col, CILS_Vector <Integer, Scalar> &y) {
            y.clear();
            for (unsigned int i = 0; i <= s1; i++) {
                y[i] = at_element(i, col);
            }
        }

        void clear() {
            std::fill_n(this->begin(), s1 * s2, 0);
        }

        Scalar *data() {
            return x;
        }

        Scalar &operator()(const Integer row, const Integer col) {
            return at_element(row, col);
        }

        const Scalar &operator()(const Integer row, const Integer col) const {
            return at_element(row, col);
        }


        Scalar &operator[](const Integer index) {
            return x[index];
        }

        const Scalar &operator[](const Integer index) const {
            return x.data()[index];
        }


    };

    template<typename Integer, typename Scalar>
    CILS_Matrix<Integer, Scalar> &
    operator+(const CILS_Matrix<Integer, Scalar> &A, const CILS_Matrix<Integer, Scalar> &B) {
        std::transform(A.begin(), A.end(), B.begin(), B.end(), std::plus<Scalar>());
    }

    template<typename Integer, typename Scalar>
    std::ostream &operator<<(std::ostream &os, CILS_Matrix<Integer, Scalar> &A) {
        printf("\n");
        for (Integer row = 0; row < A.size1(); row++) {
            for (Integer col = 0; col < A.size2(); col++) {
                printf("%8.4f ", A(row, col));
            }
            printf("\n");
        }
        printf("\n");
        return os;
    }
}