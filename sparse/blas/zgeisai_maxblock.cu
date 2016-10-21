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
#define BLOCKSIZE 32
#define WARP_SIZE 32
#define WRP 32
#define WRQ 4


__global__ void 
magma_zselect_insert_kernel(    
    magma_int_t n,
    magma_int_t p,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *rowMT,
    magma_index_t *colMT,
    magmaDoubleComplex *valMT,
    magma_index_t *selection,
    magma_index_t *sizes )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    
    magma_index_t select = selection[j];
    // return if no match for this thread block
    if( select != p ){
        return;    
    }
    magma_index_t count = sizes[j];
    
    if( i<count ){
        colMT[ rowMT[j]+i ] = col[ row[j]+i ];
        valMT[ rowMT[j]+i ] = val[ row[j]+i ];
    }
}// kernel 


__global__ void 
magma_zselect_rowptr_kernel(    
    magma_int_t n,
    magma_index_t *sizes,
    magma_index_t *rowMT )
{
    // unfortunately sequential...
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i == 0 ){
        magma_index_t count = 0;
        rowMT[0] = 0;
        magma_index_t j=0;
        for( j=0; j<n; j++ ){
                count = count+sizes[j];
                rowMT[j+1] = count;
        }
    }
}// kernel 


__global__ void 
magma_zselect_pattern_kernel(    
    magma_int_t n,
    magma_int_t p,
    magma_index_t *row,
    magma_index_t *selection,
    magma_index_t *sizes )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n ){
        magma_index_t diff = row[i+1] - row[i];
        if( diff <= WRP ){
             selection[ i ] = p;
             sizes[i] = diff;
        } 
    }
}// kernel 



/**
    Purpose
    -------
    This routine maximizes the pattern for the ISAI preconditioner. Precisely,
    it computes L, L^2, L^3, L^4, L^5 and then selects the columns of M_L 
    such that the nonzer-per-column are the lower max than the 
    implementation-specific limit (32).
    
    The input is the original matrix (row-major)
    The output is already col-major.

    Arguments
    ---------
    
    @param[in,out]
    L           magma_z_matrix
                Incomplete factor.
                
    @param[in,out]
    MT          magma_z_matrix*
                SPAI preconditioner structure, CSR col-major.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zgeisai_maxblock(
    magma_z_matrix L,
    magma_z_matrix *MT,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    int bs1 = 512;
    int bs2 = 1;
    int bs3 = 1;
    int gs1 = magma_ceildiv( L.num_rows, bs1 );
    int gs2 = 1;
    int gs3 = 1;
    dim3 block( bs1, bs2, bs3 );
    dim3 grid( gs1,gs2,gs3 );
    
    dim3 block0( 1, 1, 1 );
    dim3 grid0( 1, 1, 1 );
    
    
    int blocksize1 = WARP_SIZE;
    int blocksize2 = 1;
    int dimgrid1 = min( int( sqrt( double( L.num_rows ))), 65535 );
    int dimgrid2 = min(magma_ceildiv( L.num_rows, dimgrid1 ), 65535);
    int dimgrid3 = magma_ceildiv( L.num_rows, dimgrid1*dimgrid2 );
    dim3 block2( blocksize1, blocksize2, 1 );
    dim3 grid2( dimgrid1, dimgrid2, dimgrid3 );
    
    
    magma_z_matrix L2={Magma_CSR}, L3={Magma_CSR}, 
                   L4={Magma_CSR}, L5={Magma_CSR}, T={Magma_CSR};
                   
    magma_index_t *selections_d = NULL, *sizes_d = NULL;
    
    CHECK( magma_index_malloc( &selections_d, L.num_rows ) );
    CHECK( magma_index_malloc( &sizes_d, L.num_rows ) );
    
    magma_int_t nonzeros;
    // generate all pattern that may be considered
            
    // pattern L
    CHECK( magma_z_mtransfer( L, &T, Magma_DEV, Magma_DEV, queue ) );

    // pattern L^2
    CHECK( magma_z_spmm( MAGMA_Z_ONE, L, T, &L2, queue ) );
    // pattern L^3
    CHECK( magma_z_spmm( MAGMA_Z_ONE, T, L2, &L3, queue ) );
    // pattern L^4                           
    CHECK( magma_z_spmm( MAGMA_Z_ONE, T, L3, &L4, queue ) );
    // pattern L^5                           
     CHECK( magma_z_spmm( MAGMA_Z_ONE, T, L4, &L5, queue ) );

    // check for pattern L
    magma_zselect_pattern_kernel<<< grid, block, 0, queue->cuda_stream() >>>    
            ( L.num_rows, 1, L.drow, selections_d, sizes_d );
    // check for pattern L2
    magma_zselect_pattern_kernel<<< grid, block, 0, queue->cuda_stream() >>>    
            ( L.num_rows, 2, L2.drow, selections_d, sizes_d );
    // check for pattern L3
    magma_zselect_pattern_kernel<<< grid, block, 0, queue->cuda_stream() >>>    
            ( L.num_rows, 3, L3.drow, selections_d, sizes_d );
    // check for pattern L4
    magma_zselect_pattern_kernel<<< grid, block, 0, queue->cuda_stream() >>>    
            ( L.num_rows, 4, L4.drow, selections_d, sizes_d );
    // check for pattern L5
    magma_zselect_pattern_kernel<<< grid, block, 0, queue->cuda_stream() >>>    
            ( L.num_rows, 5, L5.drow, selections_d, sizes_d );

    //now allocate the roptr for MT
    CHECK( magma_index_malloc( &MT->drow, L.num_rows+1 ) );
    // global nonzero count + generate rowptr
    magma_zselect_rowptr_kernel<<< grid0, block0, 0, queue->cuda_stream() >>>    
            ( L.num_rows, sizes_d, MT->drow );
    cudaMemcpy( &nonzeros, MT->drow+L.num_rows, sizeof(magma_index_t), cudaMemcpyDeviceToHost);
    
    //now allocate the memory needed
    CHECK( magma_index_malloc( &MT->dcol, nonzeros ) );
    CHECK( magma_zmalloc( &MT->dval, nonzeros ) );
    
    // fill in some info
    MT->memory_location = Magma_DEV;
    MT->storage_type = Magma_CSR;
    MT->num_rows = L.num_rows;
    MT->num_cols = L.num_cols;
    MT->nnz = nonzeros;
    MT->true_nnz = nonzeros;
    MT->fill_mode = T.fill_mode;

    // now insert the data needed
    magma_zselect_insert_kernel<<< grid2, block2, 0, queue->cuda_stream() >>>    
            ( L.num_rows, 1, 
                L.drow, L.dcol, L.dval,
                MT->drow, MT->dcol, MT->dval,
                selections_d, sizes_d );    
            
    magma_zselect_insert_kernel<<< grid2, block2, 0, queue->cuda_stream() >>>    
            ( L.num_rows, 2, 
                L2.drow, L2.dcol, L2.dval,
                MT->drow, MT->dcol, MT->dval,
                selections_d, sizes_d );    
            
    magma_zselect_insert_kernel<<< grid2, block2, 0, queue->cuda_stream() >>>    
            ( L.num_rows, 3, 
                L3.drow, L3.dcol, L3.dval,
                MT->drow, MT->dcol, MT->dval,
                selections_d, sizes_d );   
            
    magma_zselect_insert_kernel<<< grid2, block2, 0, queue->cuda_stream() >>>    
            ( L.num_rows, 4, 
                L4.drow, L4.dcol, L4.dval,
                MT->drow, MT->dcol, MT->dval,
                selections_d, sizes_d );    
            
    magma_zselect_insert_kernel<<< grid2, block2, 0, queue->cuda_stream() >>>    
             ( L.num_rows, 5, 
                 L5.drow, L5.dcol, L5.dval,
                 MT->drow, MT->dcol, MT->dval,
                 selections_d, sizes_d );   
            
cleanup:
    magma_free( sizes_d );
    magma_free( selections_d );
    magma_zmfree( &T, queue );
    magma_zmfree( &L2, queue );
    magma_zmfree( &L3, queue );
    magma_zmfree( &L4, queue );
    magma_zmfree( &L5, queue );
    
    return info;
}
