/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds

*/
#include "magma_internal.h"

#define NB 64


/******************************************************************************/
// adds   x += r (including conversion to double)  --and--
// copies w = b
// each thread does one index, x[i] and w[i]
__global__ void
zcaxpycp_kernel(
    int m,
    magmaFloatComplex *r,
    magmaDoubleComplex *x,
    const magmaDoubleComplex *b,
    magmaDoubleComplex *w )
{
    const int i = threadIdx.x + blockIdx.x*NB;
    if ( i < m ) {
        x[i] = MAGMA_Z_ADD( x[i], MAGMA_Z_MAKE( MAGMA_Z_REAL( r[i] ),
                                                MAGMA_Z_IMAG( r[i] ) ) );
        w[i] = b[i];
    }
}


/***************************************************************************//**
    adds   x += r (including conversion to double)  --and--
    copies w = b
*******************************************************************************/
extern "C" void
magmablas_zcaxpycp(
    magma_int_t m,
    magmaFloatComplex_ptr r,
    magmaDoubleComplex_ptr x,
    magmaDoubleComplex_const_ptr b,
    magmaDoubleComplex_ptr w,
    magma_queue_t queue )
{
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( m, NB ) );
    zcaxpycp_kernel <<< grid, threads, 0, queue->cuda_stream() >>> ( m, r, x, b, w );
}
