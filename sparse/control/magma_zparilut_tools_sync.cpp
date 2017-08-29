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
    for( magma_int_t e=0; e<L->nnz; e++){

        magma_int_t i,j,icol,jcol,jold;

        magma_index_t row = L->rowidx[ e ];
        magma_index_t col = L->col[ e ];
        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        if( col < row ){
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
                jold = j;
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
            L_new_val[ e ] =  ( A_e - sum ) / U->val[jold];
        } else if( row == col ){ // end check whether part of L
            L_new_val[ e ] = MAGMA_Z_ONE; // lower triangular has diagonal equal 1
        }
    }// end omp parallel section

   #pragma omp parallel for
    for( magma_int_t e=0; e<U->nnz; e++){
        {
            magma_int_t i,j,icol,jcol;

            magma_index_t row = U->col[ e ];
            magma_index_t col = U->rowidx[ e ];

            //printf("(%d,%d) ", row, col); fflush(stdout);
            magmaDoubleComplex A_e = MAGMA_Z_ZERO;
            // check whether A contains element in this location
            for( i = A->row[row]; i<A->row[row+1]; i++){
                if( A->col[i] == col ){
                    A_e = A->val[i];
                    i = A->row[row+1];
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
            U_new_val[ e ] =  ( A_e - sum );
        }
    }// end omp parallel section

    
    val_swap = L_new_val;
    L_new_val = L->val;
    L->val = val_swap;
    
    val_swap = U_new_val;
    U_new_val = U->val;
    U->val = val_swap;
    
    magma_zmfree( &L_new, queue );
    magma_zmfree( &U_new, queue );
    
cleanup:
    return info;
}





#endif  // _OPENMP