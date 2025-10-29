#ifndef MAGMA_SWAP_SCALAR_DP_HPP
#define MAGMA_SWAP_SCALAR_DP_HPP

#include <sycl/sycl.hpp>
#include "magma_internal.h"

// in-place swap through bitwise xor
/******************************************************************************/
template<typename T>
static __inline__ void
swap_internal(T* a, T* b)
{
    (*a) = (*a)^(*b);
    (*b) = (*b)^(*a);
    (*a) = (*a)^(*b);
}

/******************************************************************************/
static __inline__ void
magmablas_zswap_scalar_device(magmaDoubleComplex &a, magmaDoubleComplex &b) {
    swap_internal<long>( (long*)&MAGMA_Z_REAL(a), (long*)&MAGMA_Z_REAL(b) );
    swap_internal<long>( (long*)&MAGMA_Z_IMAG(a), (long*)&MAGMA_Z_IMAG(b) );
}

/******************************************************************************/
static __inline__ void
magmablas_cswap_scalar_device(magmaFloatComplex &a, magmaFloatComplex &b) {
    swap_internal<long>( (long*)&a, (long*)&b );
}

/******************************************************************************/
static __inline__ void
magmablas_dswap_scalar_device(double &a, double &b) {
    swap_internal<long>( (long*)&a, (long*)&b );
}

/******************************************************************************/
static __inline__ void
magmablas_sswap_scalar_device(float &a, float &b) {
    swap_internal<int>( (int*)&a, (int*)&b );
}

/******************************************************************************/
static __inline__ void
magmablas_iswap_scalar_device(int &a, int &b) {
    swap_internal<int>( (int*)&a, (int*)&b );
}

#endif    // MAGMA_SWAP_SCALAR_DP_HPP

