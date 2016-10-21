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


__global__ void 
zmgeellmv_kernel( 
    int num_rows, 
    int num_cols,
    int num_vecs,
    int num_cols_per_row,
    magmaDoubleComplex alpha, 
    magmaDoubleComplex * dval, 
    magma_index_t * dcolind,
    magmaDoubleComplex * dx,
    magmaDoubleComplex beta, 
    magmaDoubleComplex * dy)
{
int row = blockDim.x * blockIdx.x + threadIdx.x;

    extern __shared__ magmaDoubleComplex dot[];

    if(row < num_rows ){
        for( int i=0; i<num_vecs; i++)
                dot[ threadIdx.x + i*blockDim.x ] = MAGMA_Z_MAKE(0.0, 0.0);
        for ( int n = 0; n < num_cols_per_row; n++ ) {
            int col = dcolind [ num_cols_per_row * row + n ];
            magmaDoubleComplex val = dval [ num_cols_per_row * row + n ];
            if( val != 0){
                for( int i=0; i<num_vecs; i++)
                    dot[ threadIdx.x + i*blockDim.x ] += 
                                    val * dx[col + i * num_cols ];
            }
        }
        for( int i=0; i<num_vecs; i++)
                dy[ row + i*num_cols ] = dot[ threadIdx.x + i*blockDim.x ] 
                                * alpha + beta * dy [ row + i * num_cols ];
    }
}


/**
    Purpose
    -------
    
    This routine computes Y = alpha *  A *  X + beta * Y for X and Y sets of 
    num_vec vectors on the GPU. Input format is ELLPACK. 
    
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
    num_vecs    mama_int_t
                number of vectors
                
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
magma_zmgeellmv(
    magma_trans_t transA,
    magma_int_t m, magma_int_t n,
    magma_int_t num_vecs,
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
    unsigned int MEM_SIZE =  num_vecs* BLOCK_SIZE 
                            * sizeof( magmaDoubleComplex ); // num_vecs vectors 
    zmgeellmv_kernel<<< grid, threads, MEM_SIZE, queue->cuda_stream() >>>
        ( m, n, num_vecs, nnz_per_row, alpha, dval, dcolind, dx, beta, dy );


    return MAGMA_SUCCESS;
}
