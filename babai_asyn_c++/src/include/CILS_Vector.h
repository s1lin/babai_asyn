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

        std::vector<Scalar> x;
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
            this->x.resize(size);
            this->x.clear();
        }

        CILS_Vector(CILS_Vector &y) {
            this->n = y.size();
            this->x.resize(n);
            std::copy(y.begin(), y.end(), this->begin());
        }

        CILS_Vector(Integer size, Integer value) {
            this->n = size;
            this->x.resize(n);
            std::fill_n(this->begin(), n, value);
        }

        Integer size() {
            return n;
        }

        Integer size() const {
            return n;
        }

        void resize(Integer new_size, bool keep = false) {
            this->n = new_size;
            this->x.resize(new_size);
            if (!keep) {
                this->clear();
            }
        }

        void clear() {
            std::fill_n(this->begin(), n, 0);
        }

        void assign(Integer new_value) {
            std::fill_n(this->begin(), n, new_value);
        }

        void assign(CILS_Vector &y) {
            this->resize(y.size(), false);
            this->n = y.size();
            this->x.resize(n);
            for (int i = 0; i < n; i++){
                this->x[i] = y.x[i];
            }
        }

        Scalar &operator[](const Integer i) {
            return x[i];
        }

        Scalar const &operator[](const Integer i) const {
            return x[i];
        }

        Scalar const &operator()(const Integer ni, const Integer col, const Integer size_n) const {
            return x[ni * size_n - (ni * (ni + 1)) / 2 + col];
        }

        Scalar const &operator()(const Integer nj, const Integer col) const {
            return x[nj + col];
        }

        CILS_Vector &operator=(CILS_Vector const &y) {
            this->n = y.size();
            this->x.resize(y.size());
            for (int i = 0; i < y.size(); i++) {
                this->x[i] = y[i];
            }
            return *this;
        }

        CILS_Vector &operator-(CILS_Vector const &y) {
            for (int i = 0; i < y.size(); i++) {
                this->x[i] -= y[i];
            }
            return *this;
        }

        CILS_Vector &operator+(CILS_Vector const &y) {
            for (int i = 0; i < y.size(); i++) {
                this->x[i] += y[i];
            }
            return *this;
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