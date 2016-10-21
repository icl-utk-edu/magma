/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "magmasparse_internal.h"

#define PRECISION_z
#define COMPLEX
#define BLOCKSIZE 256



//      does not yet work at this point!        //

__global__ void 
magma_zmisai_blockstruct_row_kernel(    
    magma_int_t n, 
    magma_int_t bs, 
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n+1 ){
        row[ i ] = i * bs;
    }
}// kernel 


__global__ void 
magma_zmisai_blockstruct_fill_l_kernel(    
    magma_int_t n, 
    magma_int_t bs, 
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val )
{
    int block = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
    int i = threadIdx.x;
    int j = threadIdx.y;
    int lrow = block * bs + i;
    int lcol = j + block * bs;
    int offset = block * bs*bs; 
    int loc = offset + lrow*bs +lcol;
    if( lrow < n ){
        if( lcol < n ){
            // val[loc] = MAGMA_Z_MAKE((double)(lrow+1),(double)(1+lcol));
            // col[loc] = lcol;
            if( lcol<=lrow ){
                val[loc] = MAGMA_Z_ONE;
                col[loc] = lcol;
            } else {
                val[loc] = MAGMA_Z_ZERO;
                col[loc] = lcol;
            } 
        } 
        // else {
        //         val[loc] = MAGMA_Z_ZERO;
        //         col[loc] = 0;
        // }
    }
}// kernel 

__global__ void 
magma_zmisai_blockstruct_fill_u_kernel(    
    magma_int_t n, 
    magma_int_t bs, 
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val )
{
    int block = blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x;
    int lrow = block * bs + threadIdx.x;
    int offset = block * bs*bs; 
    int j = threadIdx.y;
    int lcol = j + block * bs;
    int loc = offset + threadIdx.x*bs + threadIdx.y;
    if( lrow < n ){
        if( lcol < n ){
            if( lcol>=lrow ){
                val[loc] = MAGMA_Z_ONE;
                col[loc] = lcol;
            } else {
                val[loc] = MAGMA_Z_ZERO;
                col[loc] = lcol;
            } 
        } 
        else {
                val[loc] = MAGMA_Z_ZERO;
                col[loc] = 0;
        }
    }
}// kernel 


/**
    Purpose
    -------
    Generates a block-diagonal sparsity pattern with block-size bs on the GPU.

    Arguments
    ---------
    
    @param[in]
    n           magma_int_t
                Size of the matrix.
                
    @param[in]
    bs          magma_int_t
                Size of the diagonal blocks.
                
    @param[in]
    offs        magma_int_t
                Size of the first diagonal block.
                
    @param[in]
    uplotype    magma_uplo_t
                lower or upper triangular
                
    @param[in,out]
    A           magma_z_matrix*
                Generated sparsity pattern matrix.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmisai_blockstruct_gpu(
    magma_int_t n,
    magma_int_t bs,
    magma_int_t offs,
    magma_uplo_t uplotype,
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    offs = 0;
    
    A->val = NULL;
    A->col = NULL;
    A->row = NULL;
    A->rowidx = NULL;
    A->blockinfo = NULL;
    A->diag = NULL;
    A->dval = NULL;
    A->dcol = NULL;
    A->drow = NULL;
    A->drowidx = NULL;
    A->ddiag = NULL;
    A->num_rows = n;
    A->num_cols = n;
    A->nnz = n*max(bs,offs);
    A->memory_location = Magma_DEV;
    A->storage_type = Magma_CSR;
    printf(" allocate memory of size %lld and %lld\n", (long long) A->num_rows+1, (long long) A->nnz );
    magma_zmalloc( &A->dval, A->nnz );
    magma_index_malloc( &A->drow, A->num_rows+1 );
    magma_index_malloc( &A->dcol, A->nnz );
        
    int maxbs = 12; //max(offs, bs);
    
    int blocksize1 = BLOCKSIZE;
    int blocksize2 = 1;
    int blocksize3 = 1;
    int dimgrid1 = magma_ceildiv(n, BLOCKSIZE);
    int dimgrid2 = 1;
    int dimgrid3 = 1;
    
    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, blocksize3 );

    magma_zmisai_blockstruct_row_kernel<<< grid, block, 0, queue->cuda_stream() >>>
        ( A->num_rows, maxbs, A->drow, A->dcol, A->dval );
        
    blocksize1 = maxbs;
    blocksize2 = maxbs;
    dimgrid1 = min( int( sqrt( double( A->num_rows ))), 65535 );
    dimgrid2 = min(magma_ceildiv( A->num_rows, dimgrid1 ), 65535);
    dimgrid3 = magma_ceildiv( A->num_rows, dimgrid1*dimgrid2 );
    // dimgrid1 = n;
    // dimgrid2 = 1;
    // dimgrid3 = 1;
    
    
    dim3 grid2( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block2( blocksize1, blocksize2, 1 );
    
    // for now: no offset
    if( uplotype == MagmaLower ){printf("enter here\n");
        magma_zmisai_blockstruct_fill_l_kernel<<< grid2, block2, 0, queue->cuda_stream() >>>
            ( A->num_rows, maxbs, A->drow, A->dcol, A->dval );
    } else {
        magma_zmisai_blockstruct_fill_u_kernel<<< grid2, block2, 0, queue->cuda_stream() >>>
            ( A->num_rows, maxbs, A->drow, A->dcol, A->dval );
    }
    magma_z_mvisu(*A, queue );
    
    return info;
}
