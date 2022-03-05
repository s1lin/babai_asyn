//
// Created by shilei on 3/2/22.
//

#ifndef CILS_SOLVER_CILS_ITERATOR_H
#define CILS_SOLVER_CILS_ITERATOR_H

#include <iterator>

namespace cils {
    template<typename Integer, typename Scalar>
    struct Iterator {
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = Scalar;
        using pointer = Scalar *;
        using reference = Scalar &;

        Iterator(pointer ptr) : m_ptr(ptr) {}

        reference operator*() const { return *m_ptr; }

        pointer operator->() { return m_ptr; }

        Iterator &operator++() {
            m_ptr++;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        friend bool operator==(const Iterator &a, const Iterator &b) { return a.m_ptr == b.m_ptr; };

        friend bool operator!=(const Iterator &a, const Iterator &b) { return a.m_ptr != b.m_ptr; };

    private:

        pointer m_ptr;
    };
}

#endif //CILS_SOLVER_CILS_ITERATOR_H