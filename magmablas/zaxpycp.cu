/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

#define NB 64

/******************************************************************************/
// adds   x += r  --and--
// copies r = b
// each thread does one index, x[i] and r[i]
__global__ void
zaxpycp_kernel(
    int m,
    magmaDoubleComplex *r,
    magmaDoubleComplex *x,
    const magmaDoubleComplex *b)
{
    const int i = threadIdx.x + blockIdx.x*NB;
    if ( i < m ) {
        x[i] = MAGMA_Z_ADD( x[i], r[i] );
        r[i] = b[i];
    }
}


/***************************************************************************//**
    adds   x += r  --and--
    copies r = b
*******************************************************************************/
extern "C" void
magmablas_zaxpycp(
    magma_int_t m,
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr x,
    magmaDoubleComplex_const_ptr b,
    magma_queue_t queue )
{
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( m, NB ) );
    zaxpycp_kernel <<< grid, threads, 0, queue->cuda_stream() >>> ( m, r, x, b );
}
