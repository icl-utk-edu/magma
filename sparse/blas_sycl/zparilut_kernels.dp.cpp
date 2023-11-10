/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

/* Runtime API (this comment caused odd issues with hipify scripts) 
 *
 * cudaFuncCachePreferShared: shared memory is 48 KB
 * cudaFuncCachePreferEqual: shared memory is 32 KB
 * cudaFuncCachePreferL1: shared memory is 16 KB
 * cudaFuncCachePreferNone: no preference
 *
 * (spaces are added to prevent expansion from the script from messing up)
 * cudaFunc Set CacheConfig(cudaFuncCache PreferShared);
 */


#define PRECISION_z


void 
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
    magmaDoubleComplex *U_val,
    sycl::nd_item<3> item_ct1)
{
    int k = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);

    magmaDoubleComplex zero = MAGMA_Z_ZERO;
    int il, iu, jl, ju;
    
    if (k < L_nnz) {
        magmaDoubleComplex s, sp;
        int row = L_rowidx[k];
        int col = L_col[k];

        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        if (row == col) { // end check whether part of L
            /*
            DPCT1064:67: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
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


void 
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
    magmaDoubleComplex *U_val,
    sycl::nd_item<3> item_ct1)
{
    int k = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);

    magmaDoubleComplex zero = MAGMA_Z_ZERO;
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

    sycl::range<3> grid1(dimgrid13, dimgrid12, dimgrid11);
    sycl::range<3> block1(1, blocksize2, blocksize1);

    int dimgrid21 = magma_ceildiv( U->nnz, blocksize1 );
    int dimgrid22 = 1;
    int dimgrid23 = 1;

    sycl::range<3> grid2(dimgrid23, dimgrid22, dimgrid21);
    sycl::range<3> block2(1, blocksize2, blocksize1);

    // Runtime API (see top of file)
    /*
    DPCT1026:68: The call to cudaDeviceSetCacheConfig was removed because DPC++
    currently does not support setting cache config on devices.
    */

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto A_num_rows_ct0 = A->num_rows;
        auto A_drow_ct1 = A->drow;
        auto A_dcol_ct2 = A->dcol;
        auto A_dval_ct3 = A->dval;
        auto L_nnz_ct4 = L->nnz;
        auto L_drow_ct5 = L->drow;
        auto L_drowidx_ct6 = L->drowidx;
        auto L_dcol_ct7 = L->dcol;
        auto L_dval_ct8 = L->dval;
        auto U_nnz_ct9 = U->nnz;
        auto U_drow_ct10 = U->drow;
        auto U_drowidx_ct11 = U->drowidx;
        auto U_dcol_ct12 = U->dcol;
        auto U_dval_ct13 = U->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid1 * block1, block1),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zparilut_L_kernel(
                                 A_num_rows_ct0, A_drow_ct1, A_dcol_ct2,
                                 A_dval_ct3, L_nnz_ct4, L_drow_ct5,
                                 L_drowidx_ct6, L_dcol_ct7, L_dval_ct8,
                                 U_nnz_ct9, U_drow_ct10, U_drowidx_ct11,
                                 U_dcol_ct12, U_dval_ct13, item_ct1);
                         });
    });

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto A_num_rows_ct0 = A->num_rows;
        auto A_drow_ct1 = A->drow;
        auto A_dcol_ct2 = A->dcol;
        auto A_dval_ct3 = A->dval;
        auto L_nnz_ct4 = L->nnz;
        auto L_drow_ct5 = L->drow;
        auto L_drowidx_ct6 = L->drowidx;
        auto L_dcol_ct7 = L->dcol;
        auto L_dval_ct8 = L->dval;
        auto U_nnz_ct9 = U->nnz;
        auto U_drow_ct10 = U->drow;
        auto U_drowidx_ct11 = U->drowidx;
        auto U_dcol_ct12 = U->dcol;
        auto U_dval_ct13 = U->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid2 * block2, block2),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zparilut_U_kernel(
                                 A_num_rows_ct0, A_drow_ct1, A_dcol_ct2,
                                 A_dval_ct3, L_nnz_ct4, L_drow_ct5,
                                 L_drowidx_ct6, L_dcol_ct7, L_dval_ct8,
                                 U_nnz_ct9, U_drow_ct10, U_drowidx_ct11,
                                 U_dcol_ct12, U_dval_ct13, item_ct1);
                         });
    });

    return MAGMA_SUCCESS;
}



void 
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
    magmaDoubleComplex *R_val,
    sycl::nd_item<3> item_ct1)
{
    int k = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
            item_ct1.get_local_id(2);

    magmaDoubleComplex zero = MAGMA_Z_ZERO;
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

    sycl::range<3> grid1(dimgrid13, dimgrid12, dimgrid11);
    sycl::range<3> block1(1, blocksize2, blocksize1);

    // Runtime API (see top of file)

    /*
    DPCT1026:71: The call to cudaDeviceSetCacheConfig was removed because DPC++
    currently does not support setting cache config on devices.
    */

    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto R_nnz_ct10 = R->nnz;
        auto R_drowidx_ct11 = R->drowidx;
        auto R_dcol_ct12 = R->dcol;
        auto R_dval_ct13 = R->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid1 * block1, block1),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_zparilut_residuals_kernel(
                                 A.num_rows, A.drow, A.dcol, A.dval, L.drow,
                                 L.dcol, L.dval, U.drow, U.dcol, U.dval,
                                 R_nnz_ct10, R_drowidx_ct11, R_dcol_ct12,
                                 R_dval_ct13, item_ct1);
                         });
    });

    return MAGMA_SUCCESS;
}
