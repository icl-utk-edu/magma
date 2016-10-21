/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512


// ELLPACK SpMV kernel
//Michael Garland
__global__ void 
zgeellmv_kernel( 
    int num_rows, 
    int num_cols,
    int num_cols_per_row,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row; n++ ) {
            int col = dcolind [ num_cols_per_row * row + n ];
            magmaDoubleComplex val = dval [ num_cols_per_row * row + n ];
            if ( val != 0)
                dot += val * dx[col ];
        }
        dy[ row ] = dot * alpha + beta * dy [ row ];
    }
}

// shifted ELLPACK SpMV kernel
//Michael Garland
__global__ void 
zgeellmv_kernel_shift( 
    int num_rows, 
    int num_cols,
    int num_cols_per_row,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex lambda, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    int offset,
    int blocksize,
    magma_index_t * addrows,
    magmaDoubleComplex * dy)
{
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row; n++ ) {
            int col = dcolind [ num_cols_per_row * row + n ];
            magmaDoubleComplex val = dval [ num_cols_per_row * row + n ];
            if ( val != 0)
                dot += val * dx[col ];
        }
        if ( row < blocksize )
            dy[ row ] = dot * alpha - lambda * dx[ offset+row ] + beta * dy [ row ];
        else
            dy[ row ] = dot * alpha - lambda * dx[ addrows[row-blocksize] ] + beta * dy [ row ];   
    }
}


/**
    Purpose
    -------
    
    This routine computes y = alpha *  A *  x + beta * y on the GPU.
    Input format is ELLPACK.
    
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
    nnz_per_row magma_int_t
                number of elements in the longest row 

    @param[in]
    alpha       magmaDoubleComplex
                scalar multiplier

    @param[in]
    dval        magmaDoubleComplex_ptr
                array containing values of A in ELLPACK

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in ELLPACK

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
magma_zgeellmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue )
{
    dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_int_t threads = BLOCK_SIZE;
    zgeellmv_kernel<<< grid, threads, 0, queue->cuda_stream() >>>
                  ( m, n, nnz_per_row, alpha, dval, dcolind, dx, beta, dy );


    return MAGMA_SUCCESS;
}



/**
    Purpose
    -------
    
    This routine computes y = alpha *( A - lambda I ) * x + beta * y on the GPU.
    Input format is ELLPACK.
    It is the shifted version of the ELLPACK SpMV.
    
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
    nnz_per_row magma_int_t
                number of elements in the longest row 
                
    @param[in]
    alpha       magmaDoubleComplex
                scalar multiplier
                
    @param[in]
    lambda      magmaDoubleComplex
                scalar multiplier

    @param[in]
    dval        magmaDoubleComplex_ptr
                array containing values of A in ELLPACK

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in ELLPACK

    @param[in]
    dx          magmaDoubleComplex_ptr
                input vector x

    @param[in]
    beta        magmaDoubleComplex
                scalar multiplier
                
    @param[in]
    offset      magma_int_t 
                in case not the main diagonal is scaled
                
    @param[in]
    blocksize   magma_int_t 
                in case of processing multiple vectors  
                
    @param[in]
    addrows     magmaIndex_ptr
                in case the matrixpowerskernel is used

    @param[out]
    dy          magmaDoubleComplex_ptr
                input/output vector y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zgeellmv_shift(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t nnz_per_row,
    magmaDoubleComplex alpha,
    magmaDoubleComplex lambda,
    magmaDoubleComplex_ptr dval,
    magmaIndex_ptr dcolind,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex beta,
    magma_int_t offset,
    magma_int_t blocksize,
    magmaIndex_ptr addrows,
    magmaDoubleComplex_ptr dy,
    magma_queue_t queue )
{
    dim3 grid( magma_ceildiv( m, BLOCK_SIZE ) );
    magma_int_t threads = BLOCK_SIZE;
    zgeellmv_kernel_shift<<< grid, threads, 0, queue->cuda_stream() >>>
                  ( m, n, nnz_per_row, alpha, lambda, dval, dcolind, dx, 
                                    beta, offset, blocksize, addrows, dy );


    return MAGMA_SUCCESS;
}
