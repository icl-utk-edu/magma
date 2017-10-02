/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>

#define SWAP(a, b)  { tmp = a; a = b; b = tmp; }

/***************************************************************************//**
    Purpose
    -------
    This function does one synchronized ParILU sweep. Input and output are 
    different arrays.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix*
                System matrix.

    @param[in]
    L           magma_z_matrix*
                Current approximation for the lower triangular factor
                The format is sorted CSR.

    @param[in]
    U           magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is sorted CSC.
                
    @param[out]
    L_new       magma_z_matrix*
                Current approximation for the lower triangular factor
                The format is unsorted CSR.

    @param[out]
    U_new       magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is unsorted CSC.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/


extern "C" magma_int_t
magma_zparilut_sweep_sync(
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //printf("\n"); fflush(stdout);
    // parallel for using openmp
    
    // temporary vectors to swap the col/rowidx later
    // magma_index_t *tmpi;
    
    magma_z_matrix L_new={Magma_CSR}, U_new={Magma_CSR};
    magma_index_t *index_swap;
    magmaDoubleComplex *L_new_val = NULL, *U_new_val = NULL, *val_swap = NULL;
    
    // CHECK( magma_zmtransfer( *L, &L_new, Magma_CPU, Magma_CPU, queue ) );
    // CHECK( magma_zmtransfer( *U, &U_new, Magma_CPU, Magma_CPU, queue ) );
    CHECK( magma_zmalloc_cpu( &L_new_val, L->nnz ));
    CHECK( magma_zmalloc_cpu( &U_new_val, U->nnz ));
    
    #pragma omp parallel for
    for( magma_int_t e=0; e<U->nnz; e++){
        magma_int_t i,j,icol,jcol,iold;

        magma_index_t row = U->col[ e ];
        magma_index_t col = U->rowidx[ e ];
        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        if( row == col ){ 
            U_new_val[ e ] = MAGMA_Z_ONE; // upper triangular has diagonal equal 1
        } else {
            //printf("(%d,%d) ", row, col); fflush(stdout);
            magmaDoubleComplex A_e = MAGMA_Z_ZERO;
            // check whether A contains element in this location
            for( i = A->row[row]; i<A->row[row+1]; i++){
                if( A->col[i] == col ){
                    A_e = A->val[i];
                    break;
                }
            }

            //now do the actual iteration
            i = L->row[ row ];
            j = U->row[ col ];
            magma_int_t endi = L->row[ row+1 ];
            magma_int_t endj = U->row[ col+1 ];
            magmaDoubleComplex sum = MAGMA_Z_ZERO;
            magmaDoubleComplex lsum = MAGMA_Z_ZERO;
            do{
                lsum = MAGMA_Z_ZERO;
                iold = i;
                icol = L->col[i];
                jcol = U->col[j];
                if( icol == jcol ){
                    lsum = L->val[i] * U->val[j];
                    sum = sum + lsum;
                    i++;
                    j++;
                }
                else if( icol<jcol ){
                    i++;
                }
                else {
                    j++;
                }
            }while( i<endi && j<endj );
            sum = sum - lsum;

            // write back to location e
            U_new_val[ e ] =  ( A_e - sum ) / L->val[iold];
        }
    }// end omp parallel section
    
    
    #pragma omp parallel for
    for( magma_int_t e=0; e<L->nnz; e++){

        magma_int_t i,j,icol,jcol;

        magma_index_t row = L->rowidx[ e ];
        magma_index_t col = L->col[ e ];
        magmaDoubleComplex A_e = MAGMA_Z_ZERO;
        // check whether A contains element in this location
        for( i = A->row[row]; i<A->row[row+1]; i++){
            if( A->col[i] == col ){
                A_e = A->val[i];
                break;
            }
        }
        //now do the actual iteration
        i = L->row[ row ];
        j = U->row[ col ];
        magma_int_t endi = L->row[ row+1 ];
        magma_int_t endj = U->row[ col+1 ]; 
        magmaDoubleComplex sum = MAGMA_Z_ZERO;
        magmaDoubleComplex lsum = MAGMA_Z_ZERO;
        do{
            lsum = MAGMA_Z_ZERO;
            icol = L->col[i];
            jcol = U->col[j];
            
            if( icol == jcol ){
                lsum = L->val[i] * U_new_val[j];
                sum = sum + lsum;
                i++;
                j++;
            }
            else if( icol<jcol ){
                i++;
            }
            else {
                j++;
            }
        }while( i<endi && j<endj );
        sum = sum - lsum;
        // write back to location e
        L_new_val[ e ] =  ( A_e - sum );

    }// end omp parallel section


    



    
    val_swap = L_new_val;
    L_new_val = L->val;
    L->val = val_swap;
    
    val_swap = U_new_val;
    U_new_val = U->val;
    U->val = val_swap;
    
    magma_zmfree( &L_new, queue );
    magma_zmfree( &U_new, queue );
    magma_free_cpu( L_new_val );
    magma_free_cpu( U_new_val );
    
cleanup:
    return info;
}





/***************************************************************************//**
    Purpose
    -------
    This function scales the residuals of a lower triangular factor L with the 
    diagonal of U. The intention is to generate a good initial guess for 
    inserting the elements.

    Arguments
    ---------

    @param[in]
    L           magma_z_matrix
                Current approximation for the lower triangular factor
                The format is sorted CSR.

    @param[in]
    U           magma_z_matrix
                Current approximation for the upper triangular factor
                The format is sorted CSC.

    @param[in]
    hL          magma_z_matrix*
                Current approximation for the lower triangular factor
                The format is sorted CSR.

    @param[in]
    hU          magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is sorted CSC.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/


extern "C" magma_int_t
magma_zparilut_align_residuals(
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *Lnew,
    magma_z_matrix *Unew,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    #pragma omp parallel for
    for( magma_int_t row=0; row<L.num_rows; row++){
        magmaDoubleComplex Lscal = L.val[L.row[row+1]-1]; // last element in row
        for( magma_int_t el=Unew->row[row]; el<Unew->row[row+1]; el++){
            Unew->val[el] = Unew->val[el] / Lscal;           
        }
    }
    
cleanup:
    return info;
}



#endif  // _OPENMP