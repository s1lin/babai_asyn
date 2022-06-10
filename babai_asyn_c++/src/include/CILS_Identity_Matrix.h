
#ifndef CILS_IDENTITY_MATRIX_H
#define CILS_IDENTITY_MATRIX_H

namespace cils {
    template<typename Integer, typename Scalar>
    class CILS_Identity_Matrix : public CILS_Matrix<Integer, Scalar> {
    public:

        CILS_Identity_Matrix() = default;

        CILS_Identity_Matrix(Integer size1, Integer size2) {
            size1 = this->s1;
            size2 = this->s2;
            this->x = new Scalar[size1 * size2]();
            for (unsigned int i = 0; i < size1; i++) {
                this(i, i) = 1;
            }
        }

        void reset() {
            this->clear();
            for (int i = 0; i < this->size1(); i++) {
                this->at_element(i, i) = 1;
            }
        }
    };

}

#endif //CILS_IDENTITY_MATRIX_H