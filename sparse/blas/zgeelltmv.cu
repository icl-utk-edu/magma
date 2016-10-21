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


// ELL SpMV kernel
//Michael Garland
template<bool betazero>
__global__ void 
zgeelltmv_kernel( 
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
    if (row < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row; n++ ) {
            int col = dcolind [ num_rows * n + row ];
            magmaDoubleComplex val = dval [ num_rows * n + row ];
            //if ( val != MAGMA_Z_ZERO )
                dot += val * dx[col ];
        }
        if (betazero) {
            dy[ row ] = dot * alpha;
        } else {
            dy[ row ] = dot * alpha + beta * dy [ row ];
        }
    }
}

// shifted ELL SpMV kernel
//Michael Garland
__global__ void 
zgeelltmv_kernel_shift( 
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
    if (row < num_rows ) {
        magmaDoubleComplex dot = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row; n++ ) {
            int col = dcolind [ num_rows * n + row ];
            magmaDoubleComplex val = dval [ num_rows * n + row ];
            if ( val != 0)
                dot += val * dx[col ];
        }
        if ( row < blocksize )
            dy[ row ] = dot * alpha - lambda 
                    * dx[ offset+row ] + beta * dy [ row ];
        else
            dy[ row ] = dot * alpha - lambda 
                    * dx[ addrows[row-blocksize] ] + beta * dy [ row ];            
    }
}


/**
    Purpose
    -------
    
    This routine computes y = alpha *  A^t *  x + beta * y on the GPU.
    Input format is ELL.
    
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
                array containing values of A in ELL

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in ELL

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

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zgeelltmv(
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
    if (beta == MAGMA_Z_ZERO) {
        zgeelltmv_kernel<true><<< grid, threads, 0, queue->cuda_stream() >>>
                  ( m, n, nnz_per_row, alpha, dval, dcolind, dx, beta, dy );
    } else {
        zgeelltmv_kernel<false><<< grid, threads, 0, queue->cuda_stream() >>>
                  ( m, n, nnz_per_row, alpha, dval, dcolind, dx, beta, dy );
    }


    return MAGMA_SUCCESS;
}


/**
    Purpose
    -------
    
    This routine computes y = alpha *( A - lambda I ) * x + beta * y on the GPU.
    Input format is ELL.
    
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
                array containing values of A in ELL

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in ELL

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
magma_zgeelltmv_shift(
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
    magmaDoubleComplex tmp_shift;
    //magma_zsetvector(1,&lambda,1,&tmp_shift,1); 
    tmp_shift = lambda;
    zgeelltmv_kernel_shift<<< grid, threads, 0, queue->cuda_stream() >>>
                  ( m, n, nnz_per_row, alpha, tmp_shift, dval, dcolind, dx, 
                            beta, offset, blocksize, addrows, dy );


    return MAGMA_SUCCESS;
}
