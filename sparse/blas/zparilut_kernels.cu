/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "magmasparse_internal.h"

#define PRECISION_z


__global__ void 
magma_zparilut_L_kernel(   
    const magma_int_t num_rows, 
    const magma_index_t *A_row,  
    const magma_index_t *A_col,  
    const magmaDoubleComplex * __restrict__ A_val, 
    const magma_int_t L_nnz, 
    const magma_index_t *L_row, 
    const magma_index_t *L_rowidx, 
    const magma_index_t *L_col, 
    magmaDoubleComplex *L_val, 
    const magma_int_t U_nnz, 
    const magma_index_t *U_row, 
    const magma_index_t *U_rowidx, 
    const magma_index_t *U_col, 
    magmaDoubleComplex *U_val)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;

    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    int il, iu, jl, ju;
    
    if (k < L_nnz) {
        magmaDoubleComplex s, sp;
        int row = L_rowidx[k];
        int col = L_col[k];

        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        if (row == col) { // end check whether part of L
            L_val[k] = MAGMA_Z_ONE; // upper triangular has diagonal equal 1
        } else {
            s = zero;
            // check whether A contains element in this location
            for (int i = A_row[row]; i<A_row[row+1]; i++) {
                if (A_col[i] == col) {
                    s = A_val[i];
                    //break;
                }
            }
            //printf("k:%d row:%d val_A:%.2f\n", k, row, s);
            //now do the actual iteration
            il = L_row[row];
            iu = U_row[col];
            int endil = L_row[ row+1 ];
            int endiu = U_row[ col+1 ]; 
            
            do {
                sp = zero;
                jl = L_col[il];
                ju = U_col[iu];
    
                // avoid branching
                sp = ( jl == ju ) ? L_val[il] * U_val[iu] : sp;
                s  = ( jl == ju ) ? s-sp : s;
                il = ( jl <= ju ) ? il+1 : il;
                iu = ( jl >= ju ) ? iu+1 : iu;
            } while (il < endil && iu < endiu);
            // undo the last operation (it must be the last)
            s += sp;
            // write back to location e
            L_val[k] =  s / U_val[U_row[col+1]-1];
        }
    }
    
}// kernel 


__global__ void 
magma_zparilut_U_kernel(   
    const magma_int_t num_rows, 
    const magma_index_t *A_row,  
    const magma_index_t *A_col,  
    const magmaDoubleComplex * __restrict__ A_val, 
    const magma_int_t L_nnz, 
    const magma_index_t *L_row, 
    const magma_index_t *L_rowidx, 
    const magma_index_t *L_col, 
    magmaDoubleComplex *L_val, 
    const magma_int_t U_nnz, 
    const magma_index_t *U_row, 
    const magma_index_t *U_rowidx, 
    const magma_index_t *U_col, 
    magmaDoubleComplex *U_val)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    int il, iu, jl, ju;
    
    if (k < U_nnz) {
        magmaDoubleComplex s, sp;
        int row = U_col[k];
        int col = U_rowidx[k];

        s = zero;
        // check whether A contains element in this location
        for (int i = A_row[row]; i<A_row[row+1]; i++) {
            if (A_col[i] == col) {
                s = A_val[i];
                //break;
            }
        }
        //now do the actual iteration
        il = L_row[row];
        iu = U_row[col];
        int endil = L_row[ row+1 ];
        int endiu = U_row[ col+1 ]; 
        
        do {
            sp = zero;
            jl = L_col[il];
            ju = U_col[iu];

            // avoid branching
            sp = ( jl == ju ) ? L_val[il] * U_val[iu] : sp;
            s  = ( jl == ju ) ? s-sp : s;
            il = ( jl <= ju ) ? il+1 : il;
            iu = ( jl >= ju ) ? iu+1 : iu;
        } while (il < endil && iu < endiu);
        // undo the last operation (it must be the last)
        s += sp;
        // write back to location e
        U_val[k] =  s;
    } 
    
}// kernel 


/***************************************************************************//**
    
    Purpose
    -------
    This function does an ParILUT sweep. The difference to the ParILU sweep is
    that the nonzero pattern of A and the incomplete factors L and U can be 
    different. 
    The pattern determing which elements are iterated are hence the pattern 
    of L and U, not A. L has a unit diagonal.
    
    This is the GPU version of the asynchronous ParILUT sweep. 
    

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix*
                System matrix. The format is sorted CSR.

    @param[in,out]
    L           magma_z_matrix*
                Current approximation for the lower triangular factor
                The format is MAGMA_CSRCOO. This is sorted CSR plus the 
                rowindexes being stored.
                
    @param[in,out]
    U           magma_z_matrix*
                Current approximation for the lower triangular factor
                The format is MAGMA_CSRCOO. This is sorted CSR plus the 
                rowindexes being stored.
                              
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zparilut_sweep_gpu( 
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    int blocksize1 = 128;
    int blocksize2 = 1;

    int dimgrid11 = magma_ceildiv( L->nnz, blocksize1 );
    int dimgrid12 = 1;
    int dimgrid13 = 1;
    
    dim3 grid1( dimgrid11, dimgrid12, dimgrid13 );
    dim3 block1( blocksize1, blocksize2, 1 );
    
    int dimgrid21 = magma_ceildiv( U->nnz, blocksize1 );
    int dimgrid22 = 1;
    int dimgrid23 = 1;
    
    dim3 grid2( dimgrid21, dimgrid22, dimgrid23 );
    dim3 block2( blocksize1, blocksize2, 1 );

    // Runtime API
    // cudaFuncCachePreferShared: shared memory is 48 KB
    // cudaFuncCachePreferEqual: shared memory is 32 KB
    // cudaFuncCachePreferL1: shared memory is 16 KB
    // cudaFuncCachePreferNone: no preference
    //cudaFuncSetCacheConfig(cudaFuncCachePreferShared);

    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

    magma_zparilut_L_kernel<<< grid1, block1, 0, queue->cuda_stream() >>>
        (A->num_rows, A->drow, A->dcol, A->dval, 
         L->nnz, L->drow, L->drowidx, L->dcol, L->dval,
         U->nnz, U->drow, U->drowidx, U->dcol, U->dval);
        
    magma_zparilut_U_kernel<<< grid2, block2, 0, queue->cuda_stream() >>>
        (A->num_rows, A->drow, A->dcol, A->dval, 
         L->nnz, L->drow, L->drowidx, L->dcol, L->dval,
         U->nnz, U->drow, U->drowidx, U->dcol, U->dval);
         


    return MAGMA_SUCCESS;
}



__global__ void 
magma_zparilut_residuals_kernel(   
    const magma_int_t num_rows, 
    const magma_index_t *A_row,  
    const magma_index_t *A_col,  
    const magmaDoubleComplex * __restrict__ A_val, 
    const magma_index_t *L_row, 
    const magma_index_t *L_col, 
    const magmaDoubleComplex *L_val, 
    const magma_index_t *U_row, 
    const magma_index_t *U_col, 
    const magmaDoubleComplex *U_val,
    const magma_int_t R_nnz, 
    const magma_index_t *R_rowidx, 
    const magma_index_t *R_col, 
    magmaDoubleComplex *R_val)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex s, sp;
    int il, iu, jl, ju;
    
    if (k < R_nnz) {
        int row = R_rowidx[k];
        int col = R_col[k];
        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        s = zero;
        // check whether A contains element in this location
        for (int i = A_row[row]; i<A_row[row+1]; i++) {
            if (A_col[i] == col) {
                s = A_val[i];
                break;
            }
        }
        //now do the actual iteration
        il = L_row[row];
        iu = U_row[col];
        int endil = L_row[ row+1 ];
        int endiu = U_row[ col+1 ]; 
        
        do {
            sp = zero;
            jl = L_col[il];
            ju = U_col[iu];

            // avoid branching
            sp = ( jl == ju ) ? L_val[il] * U_val[iu] : sp;
            s  = ( jl == ju ) ? s-sp : s;
            il = ( jl <= ju ) ? il+1 : il;
            iu = ( jl >= ju ) ? iu+1 : iu;
        } while (il < endil && iu < endiu);
        // undo the last operation (it must be the last)
        s += sp;
        // write back to location e
        R_val[k] =  s;
    }
    
}// kernel 


/***************************************************************************//**
    Purpose
    -------
    This function computes the ILU residual in the locations included in the 
    sparsity pattern of R.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                System matrix. The format is sorted CSR.

    @param[in]
    L           magma_z_matrix
                Current approximation for the lower triangular factor
                The format is MAGMA_CSRCOO. This is sorted CSR plus the 
                rowindexes being stored.
                
    @param[in]
    U           magma_z_matrix
                Current approximation for the lower triangular factor
                The format is MAGMA_CSRCOO. This is sorted CSR plus the 
                rowindexes being stored.
                
    @param[in,out]
    R           magma_z_matrix*
                Sparsity pattern on which the ILU residual is computed. 
                R is in COO format. On output, R contains the ILU residual.
                              
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zparilut_residuals_gpu( 
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *R,
    magma_queue_t queue )
{
    int blocksize1 = 128;
    int blocksize2 = 1;

    int dimgrid11 = magma_ceildiv( R->nnz, blocksize1 );
    int dimgrid12 = 1;
    int dimgrid13 = 1;
    
    dim3 grid1( dimgrid11, dimgrid12, dimgrid13 );
    dim3 block1( blocksize1, blocksize2, 1 );
   
    // Runtime API
    // cudaFuncCachePreferShared: shared memory is 48 KB
    // cudaFuncCachePreferEqual: shared memory is 32 KB
    // cudaFuncCachePreferL1: shared memory is 16 KB
    // cudaFuncCachePreferNone: no preference
    //cudaFuncSetCacheConfig(cudaFuncCachePreferShared);

    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );

    magma_zparilut_residuals_kernel<<<grid1, block1, 0, queue->cuda_stream()>>>
        (A.num_rows, A.drow, A.dcol, A.dval, 
         L.drow, L.dcol, L.dval,
         U.drow, U.dcol, U.dval,
         R->nnz, R->drowidx, R->dcol, R->dval);
        

    return MAGMA_SUCCESS;
}
