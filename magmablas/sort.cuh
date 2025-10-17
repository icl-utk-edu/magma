/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

*/

#ifndef SORT_CUH
#define SORT_CUH

#include "magma_internal.h"
#include "magma_templates.h"
#include "swap_scalar.cuh"

#define SORT_SM_MAX_THREADS (1024)
#define SORT_SM_MAX_LENGTH  (2 * SORT_SM_MAX_THREADS)

////////////////////////////////////////////////////////////////////////////////
//                      AUXILIARY FUNCTIONS
////////////////////////////////////////////////////////////////////////////////
template<typename T>
T magmablas_get_rmin() = delete;

template<>
float magmablas_get_rmin<float>() { return lapackf77_slamch("U"); }

template<>
double magmablas_get_rmin<double>() { return lapackf77_dlamch("U"); }

////////////////////////////////////////////////////////////////////////////////
template<typename T>
T magmablas_get_rmax() = delete;

template<>
float magmablas_get_rmax<float>() { return lapackf77_slamch("O"); }

template<>
double magmablas_get_rmax<double>() { return lapackf77_dlamch("O"); }

////////////////////////////////////////////////////////////////////////////////
template<typename T>
__device__ __inline__ void
magmablas_swap_scalar_device(T& a, T& b) = delete;

////////////////////////////////////////////////////////////////////////////////
// specialization for float
template<>
__device__ __inline__ void
magmablas_swap_scalar_device<float>(float& a, float& b)
{
    magmablas_sswap_scalar_device(a, b);
}

////////////////////////////////////////////////////////////////////////////////
// specialization for double
template<>
__device__ __inline__ void
magmablas_swap_scalar_device<double>(double& a, double& b)
{
    magmablas_dswap_scalar_device(a, b);
}

////////////////////////////////////////////////////////////////////////////////
magma_int_t next_pow2(magma_int_t n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    #ifdef MAGMA_ILP64
    n |= n >> 32;
    #endif
    n++;
    return (magma_int_t)n;
}
////////////////////////////////////////////////////////////////////////////////
//                END OF AUXILIARY FUNCTIONS
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// min/max reduction (with key) in shared memory
// sort: MagmaAscending or MagmaDescending
// n   : length of vector to be sorted
// i   : typically threadIdx.x (assuming 1D thread config)
// x   : vector in shared memory
// ind : index vector in shared memory
template<typename T>
__device__ static __noinline__ void
minmax_key_sm_device(magma_sort_t sort, const int n, const int i, T* x, int* ind)
{
    bool descending = (sort == MagmaDescending);
    bool ascending  = (sort == MagmaAscending );
    #pragma unroll
    for(int step = 1024; step > 0; step >>= 1) {
        if ( n > step ) {
            if ( i < step && i + step < n ) {
                if ( (descending && x[i] < x[i+step]) || (ascending && x[i] > x[i+step]) ) {
                    magmablas_iswap_scalar_device(ind[i], ind[i+step]);
                    magmablas_swap_scalar_device(x[i], x[i+step]);
                }
            }
             __syncthreads();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// sort (with index) in shared memory
// sort  : MagmaAscending or MagmaDescending
// n     : length of vector to be sorted
// tx    : typically threadIdx.x (assuming 1D thread config)
// sx    : vector in shared memory
// sindex: index vector in shared memory
template<typename T>
__device__ static __noinline__ void
sort_key_sm_device(magma_sort_t sort, const int n, const int tx, T* sx, int* sindex)
{
    for(int in = 0; in < n-1; in++) {
        minmax_key_sm_device<T>(sort, n-in, tx, sx+in, sindex+in);
    }
}

#endif // SORT_CUH
