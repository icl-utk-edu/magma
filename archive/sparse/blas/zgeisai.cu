/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "magmasparse_internal.h"
#include "shuffle.cuh"

#define PRECISION_z
#define COMPLEX
#define BLOCKSIZE 32
#define WARP_SIZE 32
#define WRP 32
#define WRQ 4


// initialize arrays with zero
__global__ void
magma_zgpumemzero_z(
    magmaDoubleComplex * d,
    int n,
    int dim_x,
    int dim_y )
{
    int i = blockIdx.y * gridDim.x + blockIdx.x;
    int idx = threadIdx.x;

    if( i >= n ){
       return;
    }
    if( idx >= dim_x ){
       return;
    }

    for( int j=0; j<dim_y; j++)
        d[ i*dim_x*dim_y + j*dim_y + idx ] = MAGMA_Z_MAKE( 0.0, 0.0 );
}

__global__ void
magma_zlocations_lower_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;
    if( i == 0 ){
        sizes[j] = count;
        rhs[ j*WARP_SIZE ] = MAGMA_Z_ONE;
    }

    if ( i<count ){
        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel


__global__ void
magma_zlocations_trunc_lower_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;

    // normal case
    if( count <= BLOCKSIZE ){ // normal case
        if( i == 0 ){
            sizes[j] = count;
            rhs[ j*WARP_SIZE ] = MAGMA_Z_ONE;
        }
        if ( i<count ){
            locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
        }
    }
    else {
        // truncate in this row to the blocksize,
        // take only the 32 elements close to the main diagonal into account
        count = BLOCKSIZE;
        if (i == 0) {
            sizes[j] = count;
            rhs[ j*WARP_SIZE ] = MAGMA_Z_ONE;
        }

        locations[ j*WARP_SIZE + i ] = col[ row[j+1]-BLOCKSIZE+i ];
    }
}// kernel



__global__ void
magma_zlocations_upper_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;
    if( i == 0 ){
        sizes[j] = count;
        rhs[ j*WARP_SIZE+count-1 ] = MAGMA_Z_ONE;
    }

    if ( i<count ){
        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel

__global__ void
magma_zlocations_trunc_upper_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;

    // normal case
    if( count <= BLOCKSIZE ){ // normal case
        if( i == 0 ){
            sizes[j] = count;
            rhs[ j*WARP_SIZE+count-1 ] = MAGMA_Z_ONE;
        }
        if ( i<count ){
            locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
        }
    }
    else {
        // truncate in this row to the blocksize,
        // take only the 32 elements close to the main diagonal into account
        count = BLOCKSIZE;
        if (i == 0) {
            sizes[j] = count;
            rhs[ j*WARP_SIZE+count-1 ] = MAGMA_Z_ONE;
        }

        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel

__global__ void
magma_zfilltrisystems_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs )
{
    int i = (blockDim.x * blockIdx.x + threadIdx.x);

    if ( i>=n ){
        return;
    }
    for( int j=0; j<sizes[ i ]; j++ ){// no need for first
        int k = row[ locations[ j+i*WARP_SIZE ] ];
        int l = i*WARP_SIZE;
        int idx = 0;
        while( k < row[ locations[ j+i*WARP_SIZE ]+1 ] && l < (i+1)*WARP_SIZE ){ // stop once this column is done
            if( locations[ l ] == col[k] ){ //match
                // int loc = i*WARP_SIZE*WARP_SIZE + j*WARP_SIZE + idx;
                trisystems[ i*WARP_SIZE*WARP_SIZE + j*WARP_SIZE + idx ]
                                                        = val[ k ];
                k++;
                l++;
                idx++;
            } else if( col[k] < locations[ l ] ){// need to check next element
                k++;
            } else { // element does not exist, i.e. l < LC.col[k]
                // printf("increment l\n");
                l++; // check next elment in the sparsity pattern
                idx++; // leave this element equal zero
            }
        }
    }
}// kernel


/**
    Purpose
    -------

    This routine prepares the batch of small triangular systems that
    need to be solved for computing the ISAI preconditioner.


    Arguments
    ---------

    @param[in]
    uplotype    magma_uplo_t
                input matrix

    @param[in]
    transtype   magma_trans_t
                input matrix

    @param[in]
    diagtype    magma_diag_t
                input matrix

    @param[in]
    L           magma_z_matrix
                triangular factor for which the ISAI matrix is computed.
                Col-Major CSR storage.

    @param[in]
    LC          magma_z_matrix
                sparsity pattern of the ISAI matrix.
                Col-Major CSR storage.

    @param[in,out]
    sizes       magma_index_t*
                array containing the sizes of the small triangular systems

    @param[in,out]
    locations   magma_index_t*
                array containing the locations in the respective column of L

    @param[in,out]
    trisystems  magmaDoubleComplex*
                batch of generated small triangular systems. All systems are
                embedded in uniform memory blocks of size BLOCKSIZE x BLOCKSIZE

    @param[in,out]
    rhs         magmaDoubleComplex*
                RHS of the small triangular systems

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zmprepare_batched_gpu(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_z_matrix L,
    magma_z_matrix LC,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs,
    magma_queue_t queue )
{
    int blocksize1 = WARP_SIZE;
    int blocksize2 = 1;
    int dimgrid1 = min( int( sqrt( double( LC.num_rows ))), 65535 );
    int dimgrid2 = min(magma_ceildiv( LC.num_rows, dimgrid1 ), 65535);
    int dimgrid3 = magma_ceildiv( LC.num_rows, dimgrid1*dimgrid2 );
    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );
    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );


    int blocksize21 = BLOCKSIZE;
    int blocksize22 = 1;

    int dimgrid21 = magma_ceildiv( LC.num_rows, blocksize21 );
    int dimgrid22 = 1;
    int dimgrid23 = 1;
    dim3 grid2( dimgrid21, dimgrid22, dimgrid23 );
    dim3 block2( blocksize21, blocksize22, 1 );

    magma_zgpumemzero_z<<< grid, block, 0, queue->cuda_stream() >>>(
        trisystems, LC.num_rows, WARP_SIZE, WARP_SIZE );

    magma_zgpumemzero_z<<< grid, block, 0, queue->cuda_stream() >>>(
        rhs, LC.num_rows, WARP_SIZE, 1);


   // magma_zprint_gpu( 32, 32, L.dval, 32, queue );

    // Runtime API
    // cudaFuncCachePreferShared: shared memory is 48 KB
    // cudaFuncCachePreferEqual: shared memory is 32 KB
    // cudaFuncCachePreferL1: shared memory is 16 KB
    // cudaFuncCachePreferNone: no preference
    //cudaFuncSetCacheConfig(cudaFuncCachePreferShared);


    if( uplotype == MagmaLower ){
        magma_zlocations_lower_kernel<<< grid, block, 0, queue->cuda_stream() >>>(
                        LC.num_rows,
                        LC.drow,
                        LC.dcol,
                        LC.dval,
                        sizes,
                        locations,
                        trisystems,
                        rhs );
    } else {
        magma_zlocations_upper_kernel<<< grid, block, 0, queue->cuda_stream() >>>(
                        LC.num_rows,
                        LC.drow,
                        LC.dcol,
                        LC.dval,
                        sizes,
                        locations,
                        trisystems,
                        rhs );
    }

    // magma_zprint_gpu( 32, 32, L.dval, 32, queue );


    magma_zfilltrisystems_kernel<<< grid2, block2, 0, queue->cuda_stream() >>>(
                        L.num_rows,
                        L.drow,
                        L.dcol,
                        L.dval,
                        sizes,
                        locations,
                        trisystems,
                        rhs );
    //magma_zprint_gpu( 32, 32, L.dval, 32, queue );

    return MAGMA_SUCCESS;
}


__global__ void
magma_zbackinsert_kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magmaDoubleComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int end = sizes[j];
    if( j >= n ){
        return;
    }

    if ( i>=end ){
        return;
    }

    val[row[j]+i] = rhs[j*WARP_SIZE+i];
}// kernel



/**
    Purpose
    -------
    Inserts the values into the preconditioner matrix

    Arguments
    ---------


    @param[in]
    uplotype    magma_uplo_t
                lower or upper triangular

    @param[in]
    transtype   magma_trans_t
                possibility for transposed matrix

    @param[in]
    diagtype    magma_diag_t
                unit diagonal or not

    @param[in,out]
    M           magma_z_matrix*
                SPAI preconditioner CSR col-major

    @param[out]
    sizes       magma_int_t*
                Number of Elements that are replaced.

    @param[out]
    locations   magma_int_t*
                Array indicating the locations.

    @param[out]
    trisystems  magmaDoubleComplex*
                trisystems

    @param[out]
    rhs         magmaDoubleComplex*
                right-hand sides

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmbackinsert_batched_gpu(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_z_matrix *M,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    int blocksize1 = WARP_SIZE;
    int blocksize2 = 1;
    int dimgrid1 = min( int( sqrt( double( M->num_rows ))), 65535 );
    int dimgrid2 = min(magma_ceildiv( M->num_rows, dimgrid1 ), 65535);
    int dimgrid3 = magma_ceildiv( M->num_rows, dimgrid1*dimgrid2 );

    dim3 grid( dimgrid1, dimgrid2, dimgrid3 );
    dim3 block( blocksize1, blocksize2, 1 );

    magma_zbackinsert_kernel<<< grid, block, 0, queue->cuda_stream() >>>(
            M->num_rows,
            M->drow,
            M->dcol,
            M->dval,
            sizes,
            rhs );

    return info;
}
