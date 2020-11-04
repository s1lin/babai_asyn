#ifndef SILS_BABAI_ASYN_H
#define SILS_BABAI_ASYN_H

// Include Files
#include <cstddef>
#include <cstdlib>

// Function Declarations
namespace solver {
    template<typename scalar, typename index, bool use_eigen, bool is_read, bool is_write, index n>
    void sils_search(const scalar *R, const scalar *y, scalar *x, index size_R);
}
#endif

//
// File trailer for sils_babai_asyn.h
//
// [EOF]
//
