
#ifndef CILS_IDENTITY_MATRIX_H
#define CILS_IDENTITY_MATRIX_H

namespace cils {
    template<typename Integer, typename Scalar>
    class CILS_Identity_Matrix {
    public:
        Scalar *x;
        Integer s1, s2;

        Iterator <Integer, Scalar> begin() {
            return Iterator<Integer, Scalar>(&x[0]);
        }

        Iterator <Integer, Scalar> end() {
            return Iterator<Integer, Scalar>(&x[s1 * s2]);
        }


        CILS_Identity_Matrix() = default;

        CILS_Identity_Matrix(Integer size1, Integer size2) {
            size1 = s1;
            size2 = s2;
            this->x = new Scalar[s1 * s2]();
            for (unsigned int i = 0; i < s1; i++) {
                this(i, i) = 1;
            }
        }

        ~CILS_Identity_Matrix() {
            delete[] x;
        }

        Scalar operator[](const Integer i) {
            return x[i];
        }

        Scalar &operator()(const Integer row, const Integer col) const{
            return x[row * s1 + col];
        }

        Scalar &operator()(const Integer row, const Integer col) {
            return x[row * s1 + col];
        }

        Scalar &at_element (const Integer row, const Integer col) {
            return x[row * s1 + col];
        }

        void resize(Integer new_size1, Integer new_size2, bool keep = false) {
            if (keep) {
                //throw std::exception;
            } else {
//                delete[] x;
                this->x = new Scalar[new_size1 * new_size1]();
                this->s1 = new_size1;
                this->s2 = new_size2;
                for (unsigned int i = 0; i < s1; i++) {
                    at_element(i, i) = 1;
                }
            }
        }




    };

}

#endif //CILS_IDENTITY_MATRIX_H