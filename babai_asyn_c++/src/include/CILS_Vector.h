//
// Created by shilei on 2/28/22.
//

#ifndef CILS_CILS_VECTOR_H
#define CILS_CILS_VECTOR_H

#endif //CILS_CILS_VECTOR_H

namespace cils {
    template<typename Integer, typename Scalar>
    class CILS_Vector {

    public:

        Scalar *x;
        Integer n;

        Iterator <Integer, Scalar> begin() {
            return Iterator<Integer, Scalar>(&x[0]);
        }

        Iterator <Integer, Scalar> end() {
            return Iterator<Integer, Scalar>(&x[n]);
        }

        CILS_Vector() = default;

        explicit CILS_Vector(Integer size) {
            this->n = size;
            this->x = new Scalar[size]();
        }

        CILS_Vector(CILS_Vector &y) {
            this->n = y.size();
            this->x = new Scalar[512]();
            std::copy(y.begin(), y.end(), this->begin());
        }

        CILS_Vector(Integer size, Integer value) {
            this->n = size;
            this->x = new Scalar[512]();
            std::fill_n(this->begin(), n, value);
        }

        Integer size() {
            return n;
        }

        ~CILS_Vector() {
//            delete[] x;
        }

        void resize(Integer new_size, bool keep = false) {
            if (keep) {
                this->n = new_size;
            } else {
                this->x = new Scalar[new_size]();
                this->n = new_size;
                std::fill_n(this->begin(), new_size, 0);
            }
        }

        void clear() {
            std::fill_n(this->begin(), n, 0);
        }

        void assign(Integer new_value) {
            std::fill_n(this->begin(), n, new_value);
        }

        void assign(CILS_Vector &y) {
            std::copy(y.begin(), y.end(), this->begin());
        }

        Scalar &operator[](const Integer i) {
            return x[i];
        }

        Scalar &operator[](const Integer i) const {
            return x[i];
        }

        Scalar operator+(const CILS_Vector &y) {
            std::transform(this->begin(), this->end(), y.begin(), y.begin(), std::plus<Scalar>());
        }

        CILS_Vector operator-(CILS_Vector &y) {
            CILS_Vector b(n);
            for (unsigned int i = 0; i < n; i++) {
                b[i] = x[i] - y[i];
            }
            return b;
        }

    };

    template<typename Integer, typename Scalar>
    std::ostream &operator<<(std::ostream &os, CILS_Vector<Integer, Scalar> &y) {
        printf("\n");
        for (Integer row = 0; row < y.size(); row++) {
            printf("%8.4f ", y[row]);
        }
        printf("\n");
        return os;
    }
}