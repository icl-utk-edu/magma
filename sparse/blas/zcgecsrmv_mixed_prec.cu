/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions mixed zc -> ds

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256


// CSR-SpMV kernel
__global__ void 
zcgecsrmv_mixed_prec_kernel( 
    int num_rows, 
    int num_cols, 
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * ddiagval,
    magmaFloatComplex * doffdiagval,
    magma_index_t * drowptr, 
    magma_index_t * dcolind,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy)
{
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    int j;

    if(row<num_rows){
        magmaDoubleComplex dot = ddiagval[ row ] * dx[ row ];
        int start = drowptr[ row ];
        int end = drowptr[ row+1 ];
        for( j=start; j<end; j++){
            magmaDoubleComplex val = 
            MAGMA_Z_MAKE( (double) MAGMA_C_REAL(doffdiagval[ j ]),
                          (double) MAGMA_C_IMAG(doffdiagval[ j ])  );
            dot += val * dx[ dcolind[j] ];
        }
        dy[ row ] =  dot *alpha + beta * dy[ row ];
    }
}


/**
    Purpose
    -------
    
    This routine computes y = alpha *  A *  x + beta * y on the GPU.
    A is a matrix in mixed precision, i.e. the diagonal values are stored in
    high precision, the offdiagonal values in low precision.
    The input format is a CSR (val, row, col) in FloatComplex storing all 
    offdiagonal elements and an array containing the diagonal values in 
    DoubleComplex.
    
    Arguments
    ---------
    
    @param[in]
    transA      magma_trans_t
                transposition parameter for A
                
    @param[in]
    m           magma_int_t
                number of rows in A

    @param[in]
    n           magma_int_t
                number of columns in A 

    @param[in]
    alpha       magmaDoubleComplex
                scalar multiplier

    @param[in]
    ddiagval    magmaDoubleComplex_ptr
                array containing diagonal values of A in DoubleComplex
                
    @param[in]
    doffdiagval magmaFloatComplex_ptr
                array containing offdiag values of A in CSR

    @param[in]
    drowptr     magmaIndex_ptr
                rowpointer of A in CSR

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in CSR

    @param[in]
    dx          magmaDoubleComplex_ptr
                input vector x

    @param[in]
    beta        magmaDoubleComplex
                scalar multiplier

    @param[out]
    dy          magmaDoubleComplex_ptr
                input/output vector y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zcgecsrmv_mixed_prec(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr ddiagval,
    magmaFloatComplex_ptr doffdiagval,
    magmaIndex_ptr drowptr,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue )
{
    dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_int_t threads = BLOCK_SIZE;
    zcgecsrmv_mixed_prec_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
        (m, n, alpha, ddiagval, doffdiagval, drowptr, dcolind, dx, beta, dy);

    return MAGMA_SUCCESS;
}


