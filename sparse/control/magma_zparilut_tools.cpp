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
#endif

#define SWAP(a, b)  { tmp = a; a = b; b = tmp; }
#define SWAP_INT(a, b)  { tmpi = a; a = b; b = tmpi; }

#define AVOID_DUPLICATES
//#define NANCHECK

// this file is marked as deprecated, and will be removed in future.


/***************************************************************************//**
    Purpose
    -------
    Removes any element with absolute value smaller equal or larger equal
    thrs from the matrix and compacts the whole thing.

    Arguments
    ---------
    
    @param[in]
    order       magma_int_t
                order == 1: all elements smaller are discarded
                order == 0: all elements larger are discarded

    @param[in,out]
    A           magma_z_matrix*
                Matrix where elements are removed.


    @param[in]
    thrs        double*
                Threshold: all elements smaller are discarded

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_thrsrm(
    magma_int_t order,
    magma_z_matrix *A,
    double *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix B={Magma_CSR};
    B.num_rows = A->num_rows;
    B.num_cols = A->num_cols;
    B.storage_type = Magma_CSR;
    B.memory_location = Magma_CPU;
    
    CHECK( magma_index_malloc_cpu( &B.row, A->num_rows+1 ) );
    
    
    if( order == 1 ){
    // set col for values smaller threshold to -1
        #pragma omp parallel for
        for( magma_int_t row=0; row<A->num_rows; row++){
            magma_int_t rm = 0;
            magma_int_t el = 0;
            
            for( magma_int_t i=A->row[row]; i<A->row[row+1]; i++ ){
                if( MAGMA_Z_ABS(A->val[i]) <= *thrs ){
                    if( A->col[i]!=row ){
                        //printf("remove (%d %d) >> %.4e\n", row, A->col[i], A->val[i]);
                        // magma_int_t col = A->col[i];
                        A->col[i] = -1; // cheaper than val  
                        rm++;
                    } else {
                        ;
                    }
                } else {
                    el++;    
                }
            }
            B.row[row+1] = el;
        }
    } else {
        #pragma omp parallel for
        for( magma_int_t row=0; row<A->num_rows; row++){
            magma_int_t rm = 0;
            magma_int_t el = 0;
            
            for( magma_int_t i=A->row[row]; i<A->row[row+1]; i++ ){
                if( MAGMA_Z_ABS(A->val[i]) >= *thrs ){
                    if( A->col[i]!=row ){
                        //printf("remove (%d %d) >> %.4e\n", row, A->col[i], A->val[i]);
                        // magma_int_t col = A->col[i];
                        A->col[i] = -1; // cheaper than val  
                        rm++;
                    } else {
                        ;
                    }
                } else {
                    el++;    
                }
            }
            B.row[row+1] = el;
        }
    }
    
    // new row pointer
    B.row[ 0 ] = 0;
    CHECK( magma_zmatrix_createrowptr( B.num_rows, B.row, queue ) );
    B.nnz = B.row[ B.num_rows ];
    
    // allocate new arrays
    CHECK( magma_zmalloc_cpu( &B.val, B.nnz ) );
    CHECK( magma_index_malloc_cpu( &B.rowidx, B.nnz ) );
    CHECK( magma_index_malloc_cpu( &B.col, B.nnz ) );
    
    #pragma omp parallel for
    for( magma_int_t row=0; row<A->num_rows; row++){
        magma_index_t offset_old = A->row[row];
        magma_index_t offset_new = B.row[row];
        magma_index_t end_old = A->row[row+1];
        magma_int_t count = 0;
        for(magma_int_t i=offset_old; i<end_old; i++){
            if( A->col[i] > -1 ){ // copy this element
                B.col[ offset_new + count ] = A->col[i];
                B.val[ offset_new + count ] = A->val[i];
                B.rowidx[ offset_new + count ] = row;
                count++;

            }
        }
    }
    

    // finally, swap the matrices
    CHECK( magma_zmatrix_swap( &B, A, queue) );

    
cleanup:
    magma_zmfree( &B, queue );
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Removes any element with absolute value smaller equal or larger equal
    thrs from the matrix and compacts the whole thing.

    Arguments
    ---------
    
    @param[in]
    order       magma_int_t
                order == 1: all elements smaller are discarded
                order == 0: all elements larger are discarded

    @param[in,out]
    A           magma_z_matrix*
                Matrix where elements are removed.


    @param[in]
    thrs        double*
                Threshold: all elements smaller are discarded

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_thrsrm_U(
    magma_int_t order,
    magma_z_matrix L,
    magma_z_matrix *A,
    double *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix B={Magma_CSR};
    B.num_rows = A->num_rows;
    B.num_cols = A->num_cols;
    B.storage_type = Magma_CSR;
    B.memory_location = Magma_CPU;
    
    CHECK( magma_index_malloc_cpu( &B.row, A->num_rows+1 ) );
    
    
    if( order == 1 ){
    // set col for values smaller threshold to -1
        #pragma omp parallel for
        for( magma_int_t row=0; row<A->num_rows; row++){
            magma_int_t rm = 0;
            magma_int_t el = 0;
            
            for( magma_int_t i=A->row[row]; i<A->row[row+1]; i++ ){
                magmaDoubleComplex Lscal = L.val[L.row[A->col[i]+1]-1];
                if( MAGMA_Z_ABS(A->val[i]*Lscal) <= *thrs ){
                    if( A->col[i]!=row ){
                        //printf("remove (%d %d) >> %.4e\n", row, A->col[i], A->val[i]);
                        // magma_int_t col = A->col[i];
                        A->col[i] = -1; // cheaper than val  
                        rm++;
                    } else {
                        ;
                    }
                } else {
                    el++;    
                }
            }
            B.row[row+1] = el;
        }
    } else {
        #pragma omp parallel for
        for( magma_int_t row=0; row<A->num_rows; row++){
            magma_int_t rm = 0;
            magma_int_t el = 0;
            
            for( magma_int_t i=A->row[row]; i<A->row[row+1]; i++ ){
                magmaDoubleComplex Lscal = L.val[L.row[A->col[i]+1]-1];
                if( MAGMA_Z_ABS(A->val[i]*Lscal) >= *thrs ){
                    if( A->col[i]!=row ){
                        //printf("remove (%d %d) >> %.4e\n", row, A->col[i], A->val[i]);
                        // magma_int_t col = A->col[i];
                        A->col[i] = -1; // cheaper than val  
                        rm++;
                    } else {
                        ;
                    }
                } else {
                    el++;    
                }
            }
            B.row[row+1] = el;
        }
    }
    
    // new row pointer
    B.row[ 0 ] = 0;
    CHECK( magma_zmatrix_createrowptr( B.num_rows, B.row, queue ) );
    B.nnz = B.row[ B.num_rows ];
    
    // allocate new arrays
    CHECK( magma_zmalloc_cpu( &B.val, B.nnz ) );
    CHECK( magma_index_malloc_cpu( &B.rowidx, B.nnz ) );
    CHECK( magma_index_malloc_cpu( &B.col, B.nnz ) );
    
    #pragma omp parallel for
    for( magma_int_t row=0; row<A->num_rows; row++){
        magma_index_t offset_old = A->row[row];
        magma_index_t offset_new = B.row[row];
        magma_index_t end_old = A->row[row+1];
        magma_int_t count = 0;
        for(magma_int_t i=offset_old; i<end_old; i++){
            if( A->col[i] > -1 ){ // copy this element
                B.col[ offset_new + count ] = A->col[i];
                B.val[ offset_new + count ] = A->val[i];
                B.rowidx[ offset_new + count ] = row;
                count++;
            }
        }
    }
    // finally, swap the matrices
    CHECK( magma_zmatrix_swap( &B, A, queue) );
    
cleanup:
    magma_zmfree( &B, queue );
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Removes any element with absolute value smaller thrs from the matrix.
    It only uses the linked list and skips the ``removed'' elements

    Arguments
    ---------

    @param[in,out]
    A           magma_z_matrix*
                Matrix where elements are removed.


    @param[in]
    thrs        double*
                Threshold: all elements smaller are discarded

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_thrsrm_semilinked(
    magma_z_matrix *U,
    magma_z_matrix *US,
    double *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix B={Magma_CSR};
    B.num_rows = U->num_rows;
    B.num_cols = U->num_cols;
    B.storage_type = Magma_CSR;
    B.memory_location = Magma_CPU;
    
    CHECK( magma_index_malloc_cpu( &B.row, U->num_rows+1 ) );
    // set col for values smaller threshold to -1
    #pragma omp parallel for
    for( magma_int_t row=0; row<US->num_rows; row++){
        magma_int_t i = US->row[row];
        magma_int_t lasti=i;
        magma_int_t nexti=US->list[i];
        while( nexti!=0 ){
            if( MAGMA_Z_ABS( US->val[ i ] ) < *thrs ){
                    if( US->row[row] == i ){
                        //printf(" removed as first element in U rm: (%d,%d) at %d \n", row, US->col[ i ], i); fflush(stdout);
                            US->row[row] = nexti;
                            US->col[ i ] = -1;
                            US->val[ i ] = MAGMA_Z_ZERO;
                            lasti=i;
                            i = nexti;
                            nexti = US->list[nexti];
                    }
                    else{
                        //printf(" removed in linked list in U rm: (%d,%d) at %d\n", row, US->col[ i ], i); fflush(stdout);
                        US->list[lasti] = nexti;
                        US->col[ i ] = -1;
                        US->val[ i ] = MAGMA_Z_ZERO;
                        i = nexti;
                        nexti = US->list[nexti];
                    }
            } else {
                lasti = i;
                i = nexti;
                nexti = US->list[nexti];
            }
        }
    }
    /*
    printf("done\n");fflush(stdout);
    
    // get new rowpointer for U
    #pragma omp parallel for
    for( magma_int_t row=0; row<U->num_rows; row++){
        magma_int_t loc_count = 0;
        for( magma_int_t i=U->row[row]; i<U->row[row+1]; i++ ){
            if( U->col[i] > -1 ){
                loc_count++;    
            }
        }
        B.row[row+1] = loc_count;
    }
    
    // new row pointer
    B.row[ 0 ] = 0;
    CHECK( magma_zmatrix_createrowptr( B.num_rows, B.row, queue ) );
    B.nnz = B.row[ B.num_rows ];
    
    // allocate new arrays
    CHECK( magma_zmalloc_cpu( &B.val, U->nnz ) );
    CHECK( magma_index_malloc_cpu( &B.rowidx, U->nnz ) );
    CHECK( magma_index_malloc_cpu( &B.col, U->nnz ) );
    
    #pragma omp parallel for
    for( magma_int_t row=0; row<U->num_rows; row++){
        magma_index_t offset_old = U->row[row];
        magma_index_t offset_new = B.row[row];
        magma_index_t end_old = U->row[row+1];
        magma_int_t count = 0;
        for(magma_int_t i=offset_old; i<end_old; i++){
            if( U->col[i] > -1 ){ // copy this element
                B.col[ offset_new + count ] = U->col[i];
                B.val[ offset_new + count ] = U->val[i];
                B.rowidx[ offset_new + count ] = row;
                count++;
            }
        }
    }
    // finally, swap the matrices
    CHECK( magma_zmatrix_swap( &B, U, queue) );
        */
    // set the US pointer
    US->val = U->val;
    US->col = U->col;
    US->rowidx = U->rowidx;
    
  //  printf("done2\n");fflush(stdout);
    
cleanup:
    magma_zmfree( &B, queue );
    return info;
}



/***************************************************************************//**
    Purpose
    -------
    Removes a selected list of elements from the matrix.

    Arguments
    ---------

    @param[in]
    R           magma_z_matrix
                Matrix containing elements to be removed.
                
                
    @param[in,out]
    A           magma_z_matrix*
                Matrix where elements are removed.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_rmselected(
    magma_z_matrix R,
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix B={Magma_CSR};
    B.num_rows = A->num_rows;
    B.num_cols = A->num_cols;
    B.storage_type = Magma_CSR;
    B.memory_location = Magma_CPU;
    //printf("\n\n## R.nnz : %d\n", R.nnz);
    CHECK( magma_index_malloc_cpu( &B.row, A->num_rows+1 ) );
    // set col for values smaller threshold to -1
    #pragma omp parallel for
    for( magma_int_t el=0; el<R.nnz; el++){
        magma_int_t row = R.rowidx[el];
        magma_int_t col = R.col[el];
        //printf("candidate %d: (%d %d)...", el, row, col );
        for( magma_int_t i=A->row[row]; i<A->row[row+1]; i++ ){
            if( A->col[i] == col ){
                //printf("remove el %d (%d %d)\n", el, row, col );
                A->col[i] = -1;
                break;
            }
        }
    }
    
    // get new rowpointer for B
    #pragma omp parallel for
    for( magma_int_t row=0; row<A->num_rows; row++){
        magma_int_t loc_count = 0;
        for( magma_int_t i=A->row[row]; i<A->row[row+1]; i++ ){
            if( A->col[i] > -1 ){
                loc_count++;    
            }
        }
        B.row[row+1] = loc_count;
    }
    
    // new row pointer
    B.row[ 0 ] = 0;
    CHECK( magma_zmatrix_createrowptr( B.num_rows, B.row, queue ) );
    B.nnz = B.row[ B.num_rows ];
    
    // allocate new arrays
    CHECK( magma_zmalloc_cpu( &B.val, B.nnz ) );
    CHECK( magma_index_malloc_cpu( &B.rowidx, B.nnz ) );
    CHECK( magma_index_malloc_cpu( &B.col, B.nnz ) );
    
    #pragma omp parallel for
    for( magma_int_t row=0; row<A->num_rows; row++){
        magma_index_t offset_old = A->row[row];
        magma_index_t offset_new = B.row[row];
        magma_index_t end_old = A->row[row+1];
        magma_int_t count = 0;
        for(magma_int_t i=offset_old; i<end_old; i++){
            if( A->col[i] > -1 ){ // copy this element
                B.col[ offset_new + count ] = A->col[i];
                B.val[ offset_new + count ] = A->val[i];
                B.rowidx[ offset_new + count ] = row;
                count++;
            }
        }
    }
    // finally, swap the matrices
    CHECK( magma_zmatrix_swap( &B, A, queue) );
    
cleanup:
    magma_zmfree( &B, queue );
    return info;
}




/***************************************************************************//**
    Purpose
    -------
    This function takes a list of candidates with residuals, 
    and selects the largest in every row. The output matrix only contains these
    largest elements (respectively a zero element if there is no candidate for
    a certain row).

    Arguments
    ---------

    @param[in]
    order       magma_int_t
                order==1 -> largest
                order==0 -> smallest
                
    @param[in]
    A           magma_z_matrix*
                Matrix where elements are removed.
                
    @param[out]
    oneA        magma_z_matrix*
                Matrix where elements are removed.
                
                

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_selectoneperrow(
    magma_int_t order,
    magma_z_matrix *A,
    magma_z_matrix *oneA,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix B={Magma_CSR};
    double thrs = 1e-8;
    B.num_rows = A->num_rows;
    B.num_cols = A->num_cols;
    B.nnz = A->num_rows;
    B.storage_type = Magma_CSR;
    B.memory_location = Magma_CPU;
    oneA->num_rows = A->num_rows;
    oneA->num_cols = A->num_cols;
    oneA->nnz = A->num_rows;
    oneA->storage_type = Magma_CSR;
    oneA->memory_location = Magma_CPU;
    
    CHECK( magma_index_malloc_cpu( &B.row, A->num_rows+1 ) );
    CHECK( magma_index_malloc_cpu( &B.rowidx, A->num_rows) );
    CHECK( magma_index_malloc_cpu( &B.col, A->num_rows ) );
    CHECK( magma_zmalloc_cpu( &B.val, A->num_rows ) );
    if( order == 1 ){
        #pragma omp parallel for
        for( magma_int_t row=0; row<A->num_rows; row++){
            double max = 0.0;
            magma_int_t el = -1;
            for( magma_int_t i=A->row[row]; i<A->row[row+1]; i++ ){
                if( MAGMA_Z_ABS(A->val[i]) > max ){
                    el = i;
                    max = MAGMA_Z_ABS(A->val[i]);
                }
            }
            if( el > -1 ){
                B.col[row] = A->col[el];
                B.val[row] = A->val[el];
                B.rowidx[row] = row;
                B.row[row] = row;
            } else { 
                B.col[row] = -1;
                B.val[row] = MAGMA_Z_ZERO;
                B.rowidx[row] = row;
                B.row[row] = row;
            }
            
        }
        B.row[B.num_rows] = B.num_rows;
        CHECK( magma_zparilut_thrsrm( 1, &B, &thrs, queue ) );
    } else {
        #pragma omp parallel for
        for( magma_int_t row=0; row<A->num_rows; row++){
            double min = 1e18;
            magma_int_t el = -1;
            for( magma_int_t i=A->row[row]; i<A->row[row+1]; i++ ){
                if( MAGMA_Z_ABS(A->val[i]) < min && A->col[i]!=row ){
                        el = i;
                        min = MAGMA_Z_ABS(A->val[i]);
                }
            }
            if( el > -1 ){
                B.col[row] = A->col[el];
                B.val[row] = A->val[el];
                B.rowidx[row] = row;
                B.row[row] = row;
            } else { 
                B.col[row] = -1;
                B.val[row] = MAGMA_Z_ZERO;
                B.rowidx[row] = row;
                B.row[row] = row;
            }
            
        } 
        B.row[B.num_rows] = B.num_rows;
        CHECK( magma_zparilut_thrsrm( 1, &B, &thrs, queue ) );
    }
    
    // finally, swap the matrices
    // keep the copy!
   CHECK( magma_zmatrix_swap( &B, oneA, queue) );
    
cleanup:
    // magma_zmfree( &B, queue );
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This function takes a list of candidates with residuals, 
    and selects the largest in every row. The output matrix only contains these
    largest elements (respectively a zero element if there is no candidate for
    a certain row).

    Arguments
    ---------

    @param[in]
    order       magma_int_t
                order==1 -> largest
                order==0 -> smallest
                
    @param[in]
    A           magma_z_matrix*
                Matrix where elements are removed.
                
    @param[out]
    oneA        magma_z_matrix*
                Matrix where elements are removed.
                
                

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_selecttwoperrow(
    magma_int_t order,
    magma_z_matrix *A,
    magma_z_matrix *oneA,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix B={Magma_CSR};
    //double thrs = 1e-8;
    B.num_rows = A->num_rows;
    B.num_cols = A->num_cols;
    B.nnz = A->num_rows;
    B.storage_type = Magma_CSR;
    B.memory_location = Magma_CPU;
    oneA->num_rows = A->num_rows;
    oneA->num_cols = A->num_cols;
    oneA->nnz = A->num_rows;
    oneA->storage_type = Magma_CSR;
    oneA->memory_location = Magma_CPU;
    
    CHECK( magma_index_malloc_cpu( &B.row, A->num_rows+1 ) );
    CHECK( magma_index_malloc_cpu( &B.rowidx, A->num_rows*2) );
    CHECK( magma_index_malloc_cpu( &B.col, A->num_rows*2 ) );
    CHECK( magma_zmalloc_cpu( &B.val, A->num_rows*2 ) );
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<B.num_rows; i++){
            B.row[i] = 2*i;   
            for(magma_int_t j=0; j<2; j++){
                B.val[i*2+j] = MAGMA_Z_ZERO;
                B.col[i*2+j] = -1;
                B.rowidx[i*2+j] = i;
            }
                
    }
    
    if( order == 1 ){
        #pragma omp parallel for
        for( magma_int_t row=0; row<A->num_rows; row++){
            double max = 0.0;
            double max1 = 0.0;
            double max2 = 0.0;
            double tmp;
            magma_int_t tmpi;
            magma_int_t el1 = -1;
            magma_int_t el2 = -1;
            for( magma_int_t i=A->row[row]; i<A->row[row+1]; i++ ){
                if( MAGMA_Z_ABS(A->val[i]) > max ){
                    max2 = MAGMA_Z_ABS(A->val[i]);
                    el2 = i;
                    if( max2>max1){
                        SWAP(max1,max2);
                        SWAP_INT(el2,el1);
                    }
                    max = max2;
                }
            }
            if( el1 > -1 ){
                B.col[B.row[row]] = A->col[el1];
                B.val[B.row[row]] = A->val[el1];
                //printf("row:%d col1:%d val1:%f\n", row, B.col[row], B.val[row])
            } 
            if( el2 > -1 ){
                B.col[B.row[row]+1] = A->col[el2];
                B.val[B.row[row]+1] = A->val[el2];
            }
            
        }
        B.row[B.num_rows] = B.num_rows;
        // CHECK( magma_zparilut_thrsrm( 1, &B, &thrs, queue ) );
    } 
    
    // finally, swap the matrices
    // keep the copy!
   CHECK( magma_zmatrix_swap( &B, oneA, queue) );
    
cleanup:
    // magma_zmfree( &B, queue );
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This function takes a list of candidates with residuals, 
    and selects the largest in every row. The output matrix only contains these
    largest elements (respectively a zero element if there is no candidate for
    a certain row).

    Arguments
    ---------

    @param[in]
    L           magma_z_matrix
                Current lower triangular factor.
                
    @param[in]
    U           magma_z_matrix
                Current upper triangular factor.
                
    @param[in]
    A           magma_z_matrix*
                All residuals in L.
                
    @param[in]
    rtol        threshold rtol
    
    
    @param[out]
    oneA        magma_z_matrix*
                at most one per row, if larger thrs.
                
                

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_selectoneperrowthrs_lower(
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *A,
    double  rtol,
    magma_z_matrix *oneA,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix B={Magma_CSR};
    double thrs = 1e-8;
    B.num_rows = A->num_rows;
    B.num_cols = A->num_cols;
    B.nnz = A->num_rows;
    B.storage_type = Magma_CSR;
    B.memory_location = Magma_CPU;
    oneA->num_rows = A->num_rows;
    oneA->num_cols = A->num_cols;
    oneA->nnz = A->num_rows;
    oneA->storage_type = Magma_CSR;
    oneA->memory_location = Magma_CPU;
    
    CHECK( magma_index_malloc_cpu( &B.row, A->num_rows+1 ) );
    CHECK( magma_index_malloc_cpu( &B.rowidx, A->num_rows) );
    CHECK( magma_index_malloc_cpu( &B.col, A->num_rows ) );
    CHECK( magma_zmalloc_cpu( &B.val, A->num_rows ) );
 
    #pragma omp parallel for
    for( magma_int_t row=0; row<A->num_rows; row++){
        double diag_el = MAGMA_Z_ABS(U.val[U.row[row]]); 
        // last element in this row
        double max = 0.0;
        magma_int_t el = -1;
        for( magma_int_t i=A->row[row]; i<A->row[row+1]; i++ ){
            if( (MAGMA_Z_ABS(A->val[i]) > max) 
                && (MAGMA_Z_ABS(A->val[i]) > rtol / diag_el) ){
                el = i;
                max = MAGMA_Z_ABS(A->val[i]);
            }
        }
        if( el > -1 ){
            B.col[row] = A->col[el];
            B.val[row] = A->val[el];
            B.rowidx[row] = row;
            B.row[row] = row;
        } else { 
            B.col[row] = -1;
            B.val[row] = MAGMA_Z_ZERO;
            B.rowidx[row] = row;
            B.row[row] = row;
        }
        
    }
    B.row[B.num_rows] = B.num_rows;
    CHECK( magma_zparilut_thrsrm( 1, &B, &thrs, queue ) );
    
    // finally, swap the matrices
    // keep the copy!
   CHECK( magma_zmatrix_swap( &B, oneA, queue) );
    
cleanup:
    // magma_zmfree( &B, queue );
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This function takes a list of candidates with residuals, 
    and selects the largest in every row. The output matrix only contains these
    largest elements (respectively a zero element if there is no candidate for
    a certain row).

    Arguments
    ---------

    @param[in]
    L           magma_z_matrix
                Current lower triangular factor.
                
    @param[in]
    U           magma_z_matrix
                Current upper triangular factor.
                
    @param[in]
    A           magma_z_matrix*
                All residuals in L.
                
    @param[in]
    rtol        threshold rtol
    
    
    @param[out]
    oneA        magma_z_matrix*
                at most one per row, if larger thrs.
                
                

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_selectoneperrowthrs_upper(
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *A,
    double  rtol,
    magma_z_matrix *oneA,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix B={Magma_CSR};
    double thrs = 1e-8;
    B.num_rows = A->num_rows;
    B.num_cols = A->num_cols;
    B.nnz = A->num_rows;
    B.storage_type = Magma_CSR;
    B.memory_location = Magma_CPU;
    oneA->num_rows = A->num_rows;
    oneA->num_cols = A->num_cols;
    oneA->nnz = A->num_rows;
    oneA->storage_type = Magma_CSR;
    oneA->memory_location = Magma_CPU;
    
    CHECK( magma_index_malloc_cpu( &B.row, A->num_rows+1 ) );
    CHECK( magma_index_malloc_cpu( &B.rowidx, A->num_rows) );
    CHECK( magma_index_malloc_cpu( &B.col, A->num_rows ) );
    CHECK( magma_zmalloc_cpu( &B.val, A->num_rows ) );
 
    #pragma omp parallel for
    for( magma_int_t row=0; row<A->num_rows; row++){
        double diag_el = MAGMA_Z_ABS(U.val[U.row[row]]); 
        // first element in this row
        double max = 0.0;
        magma_int_t el = -1;
        for( magma_int_t i=A->row[row]; i<A->row[row+1]; i++ ){
            if( (MAGMA_Z_ABS(A->val[i]) > max) 
                && (MAGMA_Z_ABS(A->val[i]) > rtol / diag_el) ){
                el = i;
                max = MAGMA_Z_ABS(A->val[i]);
            }
        }
        if( el > -1 ){
            B.col[row] = A->col[el];
            B.val[row] = A->val[el];
            B.rowidx[row] = row;
            B.row[row] = row;
        } else { 
            B.col[row] = -1;
            B.val[row] = MAGMA_Z_ZERO;
            B.rowidx[row] = row;
            B.row[row] = row;
        }
    }
    B.row[B.num_rows] = B.num_rows;
    CHECK( magma_zparilut_thrsrm( 1, &B, &thrs, queue ) );
    
    // finally, swap the matrices
    // keep the copy!
   CHECK( magma_zmatrix_swap( &B, oneA, queue) );
cleanup:
    // magma_zmfree( &B, queue );
    return info;
}







/***************************************************************************//**
    Purpose
    -------
    This function takes a list of candidates with residuals, 
    and selects the largest in every row. The output matrix only contains these
    largest elements (respectively a zero element if there is no candidate for
    a certain row).

    Arguments
    ---------

    @param[in]
    order       magma_int_t
                order==0 lower triangular
                order==1 upper triangular
                
    @param[in]
    A           magma_z_matrix*
                Matrix where elements are removed.
                
    @param[out]
    oneA        magma_z_matrix*
                Matrix where elements are removed.
                
                

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_preselect(
    magma_int_t order,
    magma_z_matrix *A,
    magma_z_matrix *oneA,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    oneA->num_rows = A->num_rows;
    oneA->num_cols = A->num_cols;
    oneA->nnz = A->nnz - A->num_rows;
    oneA->storage_type = Magma_CSR;
    oneA->memory_location = Magma_CPU;
    
    CHECK( magma_zmalloc_cpu( &oneA->val, oneA->nnz ) );
    
    if( order == 1 ){ // don't copy the first
        #pragma omp parallel for
        for( magma_int_t row=0; row<A->num_rows; row++){
            for( magma_int_t i=A->row[row]+1; i<A->row[row+1]; i++ ){
                oneA->val[ i-row ] = A->val[i];
            }
        }
    } else { // don't copy the last
        #pragma omp parallel for
        for( magma_int_t row=0; row<A->num_rows; row++){
            for( magma_int_t i=A->row[row]; i<A->row[row+1]-1; i++ ){
                oneA->val[ i-row ] = A->val[i];
            }
        }            
    }
    
cleanup:
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This function takes a list of candidates with residuals, 
    and selects the largest in every row. The output matrix only contains these
    largest elements (respectively a zero element if there is no candidate for
    a certain row).

    Arguments
    ---------

    @param[in]
    L           magma_z_matrix*
                Matrix where elements are removed.
                
    @param[in]
    U           magma_z_matrix*
                Matrix where elements are removed.
                
    @param[out]
    oneL        magma_z_matrix*
                Matrix where elements are removed.
                
    @param[out]
    oneU        magma_z_matrix*
                Matrix where elements are removed.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_preselect_scale(
    magma_z_matrix *L,
    magma_z_matrix *oneL,
    magma_z_matrix *U,
    magma_z_matrix *oneU,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    oneL->num_rows = L->num_rows;
    oneL->num_cols = L->num_cols;
    oneL->nnz = L->nnz - L->num_rows;
    oneL->storage_type = Magma_CSR;
    oneL->memory_location = Magma_CPU;
    oneL->nnz = L->nnz-L->num_rows;
    
    oneU->num_rows = U->num_rows;
    oneU->num_cols = U->num_cols;
    oneU->nnz = L->nnz - U->num_rows;
    oneU->storage_type = Magma_CSR;
    oneU->memory_location = Magma_CPU;
    oneU->nnz = U->nnz-U->num_rows;
    
    CHECK( magma_zmalloc_cpu( &oneL->val, L->nnz-L->num_rows ) );
    CHECK( magma_zmalloc_cpu( &oneU->val, U->nnz-U->num_rows ) );
    
    { // don't copy the last
        #pragma omp parallel for
        for( magma_int_t row=0; row<L->num_rows; row++){
            for( magma_int_t i=L->row[row]; i<L->row[row+1]-1; i++ ){
                oneL->val[ i-row ] = L->val[i];
            }
        }     
        
    }
    { // don't copy the last
      // for U, we need to scale by the diagonal of L
      // unfortunatley, U is in CSC, so the factor is different for every element.
        #pragma omp parallel for
        for( magma_int_t row=0; row<U->num_rows; row++){
            for( magma_int_t i=U->row[row]; i<U->row[row+1]-1; i++ ){
                oneU->val[ i-row ] = U->val[i] * L->val[L->row[U->col[i]+1]-1];
            }
        }     
        
    }
    
cleanup:
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Transposes a matrix that already contains rowidx. The idea is to use a 
    linked list.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Matrix to transpose.
                
    @param[out]
    B           magma_z_matrix*
                Transposed matrix.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_transpose(
    magma_z_matrix A,
    magma_z_matrix *B,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_index_t *linked_list;
    magma_index_t *row_ptr;
    magma_index_t *last_rowel;
    
    magma_int_t el_per_block, num_threads=1;
    
    B->storage_type = A.storage_type;
    B->memory_location = A.memory_location;
    
    B->num_rows = A.num_rows;
    B->num_cols = A.num_cols;
    B->nnz      = A.nnz;
    
    CHECK( magma_index_malloc_cpu( &linked_list, A.nnz ));
    CHECK( magma_index_malloc_cpu( &row_ptr, A.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &last_rowel, A.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &B->row, A.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &B->rowidx, A.nnz ));
    CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));
    CHECK( magma_zmalloc_cpu( &B->val, A.nnz ) );
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        row_ptr[i] = -1;
    }
    #pragma omp parallel for
    for( magma_int_t i=0; i<A.num_rows+1; i++ ){
        B->row[i] = 0;
    }
    
    el_per_block = magma_ceildiv( A.num_rows, num_threads );

    #pragma omp parallel
    {
#ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
        for(magma_int_t i=0; i<A.nnz; i++ ){
            magma_index_t row = A.col[ i ];
            if( (row < (id+1)*el_per_block) && (row >=(id)*el_per_block)  ){
                if( row_ptr[row] == -1 ){
                    row_ptr[ row ] = i;
                    linked_list[ i ] = 0;
                    last_rowel[ row ] = i;
                } else {
                    linked_list[ last_rowel[ row ] ] = i;
                    linked_list[ i ] = 0;
                    last_rowel[ row ] = i;
                }
                B->row[row+1] = B->row[row+1] + 1;
            }
        }
    }
    
    // new rowptr
    B->row[0]=0;   
    magma_zmatrix_createrowptr( B->num_rows, B->row, queue );
    

    assert( B->row[B->num_rows] == A.nnz );
    
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t el = row_ptr[row];
        if( el>-1 ) {
            
            for( magma_int_t i=B->row[row]; i<B->row[row+1]; i++ ){
                // assert(A.col[el] == row);
                B->val[i] = A.val[el];
                B->col[i] = A.rowidx[el];
                B->rowidx[i] = row;
                el = linked_list[el];
            }
        }
    }
    
cleanup:
    magma_free_cpu( row_ptr );
    magma_free_cpu( last_rowel );
    magma_free_cpu( linked_list );
    return info;
}



/***************************************************************************//**
    Purpose
    -------
    This is a special routine with very limited scope. For a set of fill-in
    candidates in row-major format, it transposes the a submatrix, i.e. the
    submatrix consisting of the largest element in every column. 
    This function is only useful for delta<=1.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Matrix to transpose.
                
    @param[out]
    B           magma_z_matrix*
                Transposed matrix containing only largest elements in each col.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_transpose_select_one(
    magma_z_matrix A,
    magma_z_matrix *B,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //magma_index_t *linked_list;
    //magma_index_t *row_ptr;
    //magma_index_t *last_rowel;
    
    magma_int_t el_per_block, num_threads;
    //double thrs = 1e-6;
    
    B->storage_type = A.storage_type;
    B->memory_location = A.memory_location;
    
    B->num_rows = A.num_rows;
    B->num_cols = A.num_cols;
    B->nnz      = A.num_rows;
    
    //CHECK( magma_index_malloc_cpu( &linked_list, A.nnz ));
    //CHECK( magma_index_malloc_cpu( &row_ptr, A.num_rows ));
    //CHECK( magma_index_malloc_cpu( &last_rowel, A.num_rows ));
    CHECK( magma_index_malloc_cpu( &B->row, B->num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &B->rowidx, B->nnz ));
    CHECK( magma_index_malloc_cpu( &B->col, B->nnz ));
    CHECK( magma_zmalloc_cpu( &B->val, B->nnz ) );
    
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<B->num_rows; i++ ){
        B->val[i] = MAGMA_Z_ZERO;
    }
    #pragma omp parallel for
    for( magma_int_t i=0; i<B->num_rows; i++ ){
        B->row[i] = i;
        B->rowidx[i] = i;
        B->col[i] = -1;
    }
    B->row[B->num_rows] = B->nnz;
    
    el_per_block = magma_ceildiv( A.num_rows, num_threads );

    #pragma omp parallel
    {
#ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
        for(magma_int_t i=0; i<A.nnz; i++ ){
            magma_index_t row = A.col[ i ];
            if( (row < (id+1)*el_per_block) && (row >=(id)*el_per_block)  ){
                magmaDoubleComplex tmp_val;
                tmp_val = A.val[i];
                if( MAGMA_Z_ABS(tmp_val) > MAGMA_Z_ABS(B->val[row]) ){
                    B->val[row] = tmp_val;
                    B->col[ row ] = A.rowidx[i];
                } 
            }
        }
    }
    
    // no need if col==-1 is ignored
    // magma_zparilut_thrsrm( 1, B, &thrs, queue );
    
cleanup:
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    For the matrix U in CSR (row-major) this creates B containing
    a row-ptr to the columns and a linked list for the elements.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Matrix to transpose.
                
    @param[out]
    B           magma_z_matrix*
                Transposed matrix.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_create_collinkedlist(
    magma_z_matrix A,
    magma_z_matrix *B,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_index_t *last_rowel;
    
    magma_int_t el_per_block, num_threads;
    
    B->storage_type = A.storage_type;
    B->memory_location = A.memory_location;
    
    B->num_rows = A.num_rows;
    B->num_cols = A.num_cols;
    B->nnz      = A.nnz;
    
    CHECK( magma_index_malloc_cpu( &B->list, A.nnz ));
    CHECK( magma_index_malloc_cpu( &last_rowel, A.num_rows ));
    CHECK( magma_index_malloc_cpu( &B->row, A.num_rows+1 ));
    
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        B->row[i] = -1;
    }
    
    el_per_block = magma_ceildiv( A.num_rows, num_threads );

    #pragma omp parallel
    {
#ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
        for(magma_int_t i=0; i<A.nnz; i++ ){
            magma_index_t row = A.col[ i ];
            if( (row < (id+1)*el_per_block) && (row >=(id)*el_per_block)  ){
                if( B->row[row] == -1 ){
                    B->row[ row ] = i;
                    B->list[ i ] = 0;
                    last_rowel[ row ] = i;
                } else {
                    B->list[ last_rowel[ row ] ] = i;
                    B->list[ i ] = 0;
                    last_rowel[ row ] = i;
                }
            }
        }
    }
    B->val = A.val;
    B->col = A.col;
    B->rowidx = A.rowidx;
    
    
cleanup:
    magma_free_cpu( last_rowel );
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This function does an ParILU sweep.

    Arguments
    ---------

    @param[in,out]
    A          magma_z_matrix*
                Current ILU approximation
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in,out]
    L           magma_z_matrix*
                Current approximation for the lower triangular factor
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in,out]
    U           magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_sweep_semilinked(
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_z_matrix *US,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // parallel for using openmp
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
                    i = A->row[row+1];
                }
            }

            //now do the actual iteration
            i = L->row[ row ];
            j = US->row[ col ];
            magma_int_t endi = L->row[ row+1 ];
            magmaDoubleComplex sum = MAGMA_Z_ZERO;
            magmaDoubleComplex lsum = MAGMA_Z_ZERO;
            do{
                lsum = MAGMA_Z_ZERO;
                jold = j;
                icol = L->col[i];
                jcol = US->rowidx[j];
                if( icol == jcol ){
                    lsum = L->val[i] * US->val[j];
                    sum = sum + lsum;
                    i++;
                    j = US->list[j];
                }
                else if( icol<jcol ){
                    i++;
                }
                else {
                    j = US->list[j];
                }
            }while( i<endi && j!=0 );
            sum = sum - lsum;

            // write back to location e
            L->val[ e ] =  ( A_e - sum ) / US->val[jold];
        } else if( row == col ){ // end check whether part of L
            L->val[ e ] = MAGMA_Z_ONE; // lower triangular has diagonal equal 1
        }
    }// end omp parallel section
    
   #pragma omp parallel for
    for( magma_int_t e=0; e<US->nnz; e++){
        magma_index_t col = US->col[ e ];   
        if( col > -1 ) {
            magma_int_t i,j,icol,jcol;

            magma_index_t row = US->rowidx[ e ];


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
            j = US->row[ col ];
            magma_int_t endi = L->row[ row+1 ];
            magmaDoubleComplex sum = MAGMA_Z_ZERO;
            magmaDoubleComplex lsum = MAGMA_Z_ZERO;
            do{
                lsum = MAGMA_Z_ZERO;
                icol = L->col[i];
                jcol = US->rowidx[j];
                if( icol == jcol ){
                    lsum = L->val[i] * US->val[j];
                    sum = sum + lsum;
                    i++;
                    j = US->list[j];
                }
                else if( icol<jcol ){
                    i++;
                }
                else {
                    j = US->list[j];
                }
            }while( i<endi && j!=0 );
            sum = sum - lsum;

            // write back to location e
            US->val[ e ] =  ( A_e - sum );
        }
    }// end omp parallel section



    return info;
}



/***************************************************************************//**
    Purpose
    -------
    This function does an ParILU sweep.

    Arguments
    ---------

    @param[in,out]
    A          magma_z_matrix*
                Current ILU approximation
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in,out]
    L           magma_z_matrix*
                Current approximation for the lower triangular factor
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in,out]
    U           magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_sweep_list(
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //printf("\n"); fflush(stdout);
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<L->nnz; e++){
        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        if( L->list[e] > 0 ){
            magma_int_t i,j,icol,jcol,jold;

            magma_index_t row = L->rowidx[ e ];
            magma_index_t col = L->col[ e ];

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
            L->val[ e ] =  ( A_e - sum ) / U->val[jold];
        } else if( L->list[e]==0 ){ // end check whether part of L
            L->val[ e ] = MAGMA_Z_ONE; // lower triangular has diagonal equal 1
        }
    }// end omp parallel section

   #pragma omp parallel for
    for( magma_int_t e=0; e<U->nnz; e++){
        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        if( U->list[e] != -1 ){
            magma_int_t i,j,icol,jcol;

            magma_index_t row = U->rowidx[ e ];
            magma_index_t col = U->col[ e ];

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
            U->val[ e ] =  ( A_e - sum );
        }
    }// end omp parallel section



    return info;
}



/***************************************************************************//**
    Purpose
    -------
    This function computes the residuals.

    Arguments
    ---------


    @param[in,out]
    L           magma_z_matrix
                Current approximation for the lower triangular factor
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in,out]
    U           magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_residuals_semilinked(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix US,
    magma_z_matrix *L_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // printf("start\n"); fflush(stdout);
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<L_new->nnz; e++){
        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        {
            magma_int_t i,j,icol,jcol;

            magma_index_t row = L_new->rowidx[ e ];
            magma_index_t col = L_new->col[ e ];
            if( row != 0 || col != 0 ){
                // printf("(%d,%d) ", row, col); fflush(stdout);
                magmaDoubleComplex A_e = MAGMA_Z_ZERO;
                // check whether A contains element in this location
                for( i = A.row[row]; i<A.row[row+1]; i++){
                    if( A.col[i] == col ){
                        A_e = A.val[i];
                        i = A.row[row+1];
                    }
                }

                //now do the actual iteration
                i = L.row[ row ];
                j = US.row[ col ];
                magma_int_t endi = L.row[ row+1 ];
                magmaDoubleComplex sum = MAGMA_Z_ZERO;
                magmaDoubleComplex lsum = MAGMA_Z_ZERO;
                do{
                    lsum = MAGMA_Z_ZERO;
                    icol = L.col[i];
                    jcol = US.rowidx[j];
                    if( icol == jcol ){
                        lsum = L.val[i] * US.val[j];
                        sum = sum + lsum;
                        i++;
                        j=US.list[j];
                    }
                    else if( icol<jcol ){
                        i++;
                    }
                    else {
                        j=US.list[j];
                    }
                }while( i<endi && j!=0 );
                sum = sum - lsum;

                // write back to location e
                L_new->val[ e ] =  ( A_e - sum );
            } else {
                L_new->val[ e ] = MAGMA_Z_ZERO;
            }
        }
    }// end omp parallel section

    return info;
}

/***************************************************************************//**
    Purpose
    -------
    This function computes the residuals.

    Arguments
    ---------


    @param[in,out]
    L           magma_z_matrix
                Current approximation for the lower triangular factor
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in,out]
    U           magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_residuals_transpose(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *U_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // printf("start\n"); fflush(stdout);
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<U_new->nnz; e++){
        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        {
            magma_int_t i,j,icol,jcol;

            magma_index_t row = U_new->col[ e ];
            magma_index_t col = U_new->rowidx[ e ];
            if( row != 0 || col != 0 ){
                // printf("(%d,%d) ", row, col); fflush(stdout);
                magmaDoubleComplex A_e = MAGMA_Z_ZERO;
                // check whether A contains element in this location
                for( i = A.row[row]; i<A.row[row+1]; i++){
                    if( A.col[i] == col ){
                        A_e = A.val[i];
                        i = A.row[row+1];
                    }
                }

                //now do the actual iteration
                i = L.row[ row ];
                j = U.row[ col ];
                magma_int_t endi = L.row[ row+1 ];
                magma_int_t endj = U.row[ col+1 ];
                magmaDoubleComplex sum = MAGMA_Z_ZERO;
                magmaDoubleComplex lsum = MAGMA_Z_ZERO;
                do{
                    lsum = MAGMA_Z_ZERO;
                    icol = L.col[i];
                    jcol = U.col[j];
                    if( icol == jcol ){
                        lsum = L.val[i] * U.val[j];
                        sum = sum + lsum;
                        i++;
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
                U_new->val[ e ] =  ( A_e - sum );
            } else {
                U_new->val[ e ] = MAGMA_Z_ZERO;
            }
        }
    }// end omp parallel section

    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This function computes the residuals.

    Arguments
    ---------


    @param[in,out]
    L           magma_z_matrix
                Current approximation for the lower triangular factor
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in,out]
    U           magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_residuals_list(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *L_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // printf("start\n"); fflush(stdout);
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<L_new->nnz; e++){
        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        {
            magma_int_t i,j,icol,jcol;//,jold;

            magma_index_t row = L_new->rowidx[ e ];
            magma_index_t col = L_new->col[ e ];
            if( row != 0 || col != 0 ){
                // printf("(%d,%d) ", row, col); fflush(stdout);
                magmaDoubleComplex A_e = MAGMA_Z_ZERO;
                // check whether A contains element in this location
                for( i = A.row[row]; i<A.row[row+1]; i++){
                    if( A.col[i] == col ){
                        A_e = A.val[i];
                        i = A.row[row+1];
                    }
                }

                //now do the actual iteration
                i = L.row[ row ];
                j = U.row[ col ];
                magma_int_t endi = L.row[ row+1 ];
                magma_int_t endj = U.row[ col+1 ];
                magmaDoubleComplex sum = MAGMA_Z_ZERO;
                magmaDoubleComplex lsum = MAGMA_Z_ZERO;
                do{
                    lsum = MAGMA_Z_ZERO;
                    //jold = j;
                    icol = L.col[i];
                    jcol = U.col[j];
                    if( icol == jcol ){
                        lsum = L.val[i] * U.val[j];
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
                if( row>col ){
                    L_new->val[ e ] =  ( A_e - sum );
                } else {
                    L_new->val[ e ] =  ( A_e - sum );
                }
            } else {
                L_new->val[ e ] = MAGMA_Z_ZERO;
            }
        }
    }// end omp parallel section

    return info;
}



/***************************************************************************//**
    Purpose
    -------
    This function does an ParILU sweep.

    Arguments
    ---------

    @param[in,out]
    A          magma_z_matrix*
                Current ILU approximation
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in,out]
    L           magma_z_matrix*
                Current approximation for the lower triangular factor
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in,out]
    U           magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_sweep_linkedlist(
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //printf("\n"); fflush(stdout);
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<L->nnz; e++){
        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        if( L->list[e] > 0 ){
            magma_int_t i,j,icol,jcol,jold;

            magma_index_t row = L->rowidx[ e ];
            magma_index_t col = L->col[ e ];

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
                    i = L->list[i];
                    j = U->list[j];
                }
                else if( icol<jcol ){
                    i = L->list[i];
                }
                else {
                    j = U->list[j];
                }
            }while( i!=0 && j!=0 );
            sum = sum - lsum;

            // write back to location e
            L->val[ e ] =  ( A_e - sum ) / U->val[jold];
        } else if( L->list[e]==0 ){ // end check whether part of L
            L->val[ e ] = MAGMA_Z_ONE; // lower triangular has diagonal equal 1
        }
    }// end omp parallel section

   #pragma omp parallel for
    for( magma_int_t e=0; e<U->nnz; e++){
        // as we look at the lower triangular,
        // col<row, i->e-> disregard last element in row
        if( U->list[e] != -1 ){
            magma_int_t i,j,icol,jcol;

            magma_index_t row = U->rowidx[ e ];
            magma_index_t col = U->col[ e ];

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
            magmaDoubleComplex sum = MAGMA_Z_ZERO;
            magmaDoubleComplex lsum = MAGMA_Z_ZERO;
            do{
                lsum = MAGMA_Z_ZERO;
                icol = L->col[i];
                jcol = U->col[j];
                if( icol == jcol ){
                    lsum = L->val[i] * U->val[j];
                    sum = sum + lsum;
                    i = L->list[i];
                    j = U->list[j];
                }
                else if( icol<jcol ){
                    i = L->list[i];
                }
                else {
                    j = U->list[j];
                }
            }while( i!=0 && j!=0 );
            sum = sum - lsum;

            // write back to location e
            U->val[ e ] =  ( A_e - sum );
        }
    }// end omp parallel section



    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This function computes the residuals.

    Arguments
    ---------


    @param[in,out]
    L           magma_z_matrix
                Current approximation for the lower triangular factor
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in,out]
    U           magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_residuals_linkedlist(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *L_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // printf("start\n"); fflush(stdout);
    // parallel for using openmp
    #pragma omp parallel for
    for( magma_int_t e=0; e<L_new->nnz; e++){
        // as we look at the lower triangular,
        // col<row, i.e. disregard last element in row
        {
            magma_int_t i,j,icol,jcol,jold;

            magma_index_t row = L_new->rowidx[ e ];
            magma_index_t col = L_new->col[ e ];

            // printf("(%d,%d) ", row, col); fflush(stdout);
            magmaDoubleComplex A_e = MAGMA_Z_ZERO;
            // check whether A contains element in this location
            for( i = A.row[row]; i<A.row[row+1]; i++){
                if( A.col[i] == col ){
                    A_e = A.val[i];
                }
            }

            //now do the actual iteration
            i = L.row[ row ];
            j = U.row[ col ];
            magmaDoubleComplex sum = MAGMA_Z_ZERO;
            magmaDoubleComplex lsum = MAGMA_Z_ZERO;
            do{
                lsum = MAGMA_Z_ZERO;
                jold = j;
                icol = L.col[i];
                jcol = U.col[j];
                if( icol == jcol ){
                    lsum = L.val[i] * U.val[j];
                    sum = sum + lsum;
                    i = L.list[i];
                    j = U.list[j];
                }
                else if( icol<jcol ){
                    i = L.list[i];
                }
                else {
                    j = U.list[j];
                }
            }while( i!=0 && j!=0 );
            sum = sum - lsum;

            // write back to location e
            if( row>col ){
                L_new->val[ e ] =  ( A_e - sum ) / U.val[jold];
            } else {
                L_new->val[ e ] =  ( A_e - sum );
            }
        }
    }// end omp parallel section

    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This function creates a col-pointer and a linked list along the columns
    for a row-major CSR matrix

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in,out]
    AC          magma_z_matrix*
                The matrix A but with row-pointer being for col-major, same with
                linked list. The values, col and row indices are unchanged.
                The respective pointers point to the entities of A.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_colmajor(
    magma_z_matrix A,
    magma_z_matrix *AC,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t num_threads=1;
    magma_index_t *checkrow;
    magma_int_t el_per_block;
    AC->val=A.val;
    AC->col=A.rowidx;
    AC->rowidx=A.col;
    AC->row=NULL;
    AC->list=NULL;
    AC->memory_location = Magma_CPU;
    AC->storage_type = Magma_CSRLIST;
    AC->num_rows = A.num_rows;
    AC->num_cols = A.num_cols;
    AC->nnz = A.nnz;
    AC->true_nnz = A.true_nnz;

    CHECK( magma_index_malloc_cpu( &checkrow, A.true_nnz ));

    CHECK( magma_index_malloc_cpu( &AC->row, A.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &AC->list, A.true_nnz ));

#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif

    #pragma omp parallel for
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        checkrow[i] = -1;
    }
    #pragma omp parallel for
    for( magma_int_t i=0; i<A.true_nnz; i++ ){
        AC->list[i] = -1;
    }
     el_per_block = magma_ceildiv( A.num_rows, num_threads );

    #pragma omp parallel
    {
#ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
        for(magma_int_t i=0; i<A.true_nnz; i++ ){
            magma_index_t row = A.col[ i ];
            if( (row < (id+1)*el_per_block) && (row >=(id)*el_per_block)  ){
                // printf("thread %d handling row %d\n", id, row );
                //if( A.rowidx[ i ] < A.col[ i ]){
                //    printf("illegal element:(%d,%d)\n", A.rowidx[ i ], A.col[ i ] );
                //}
                if( checkrow[row] == -1 ){
                    // printf("thread %d write in row pointer at row[%d] = %d\n", id, row, i);
                    AC->row[ row ] = i;
                    AC->list[ i ] = 0;
                    checkrow[ row ] = i;
                } else {
                    // printf("thread %d list[%d] = %d\n", id, checkrow[ row ], i);
                    AC->list[ checkrow[ row ] ] = i;
                    AC->list[ i ] = 0;
                    checkrow[ row ] = i;
                }
            }
        }
    }

cleanup:
    magma_free_cpu( checkrow );
    return info;
}

/***************************************************************************//**
    Purpose
    -------
    This routine reorders the matrix (inplace) for easier access.

    Arguments
    ---------

    @param[in]
    LU          magma_z_matrix*
                Current ILU approximation.


    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_reorder(
    magma_z_matrix *LU,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magmaDoubleComplex *val=NULL;
    magma_index_t *col=NULL;
    magma_index_t *row=NULL;
    magma_index_t *rowidx=NULL;
    magma_index_t *list=NULL;
    magmaDoubleComplex *valt=NULL;
    magma_index_t *colt=NULL;
    magma_index_t *rowt=NULL;
    magma_index_t *rowidxt=NULL;
    magma_index_t *listt=NULL;

    magma_int_t nnz=0;


    CHECK( magma_zmalloc_cpu( &val, LU->true_nnz ));
    CHECK( magma_index_malloc_cpu( &rowidx, LU->true_nnz ));
    CHECK( magma_index_malloc_cpu( &col, LU->true_nnz ));
    CHECK( magma_index_malloc_cpu( &row, LU->num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &list, LU->true_nnz ));

    // do two sweeps to allow for parallel insertion
    row[ 0 ] = 0;
    #pragma omp parallel for
    for(magma_int_t rowc=0; rowc<LU->num_rows; rowc++){
        magma_index_t loc_nnz = 0;
        magma_int_t el = LU->row[ rowc ];
        do{
            loc_nnz++;
            el = LU->list[ el ];
        }while( el != 0 );
        row[ rowc+1 ] = loc_nnz;
    }

    // global count
    for( magma_int_t i = 0; i<LU->num_rows; i++ ){
        nnz = nnz + row[ i+1 ];
        row[ i+1 ] = nnz;
    }

    LU->nnz = nnz;
    // parallel insertion
    #pragma omp parallel for
    for(magma_int_t rowc=0; rowc<LU->num_rows; rowc++){
        magma_int_t el = LU->row[ rowc ];
        magma_int_t offset = row[ rowc ];
        magma_index_t loc_nnz = 0;
        do{
            magmaDoubleComplex valtt = LU->val[ el ];
            magma_int_t loc = offset+loc_nnz;
#ifdef NANCHECK
            if(magma_z_isnan_inf( valtt ) ){
                info = MAGMA_ERR_NAN;
                el = 0;
            } else
#endif
            {
                val[ loc ] = valtt;
                col[ loc ] = LU->col[ el ];
                rowidx[ loc ] = rowc;
                list[ loc ] = loc+1;
                loc_nnz++;
                el = LU->list[ el ];
            }
        }while( el != 0 );
        list[ offset+loc_nnz - 1 ] = 0;
    }

    listt = LU->list;
    rowt = LU->row;
    rowidxt = LU->rowidx;
    valt = LU->val;
    colt = LU->col;

    LU->list = list;
    LU->row = row;
    LU->rowidx = rowidx;
    LU->val = val;
    LU->col = col;

    list = listt;
    row = rowt;
    rowidx = rowidxt;
    val = valt;
    col = colt;

cleanup:
    magma_free_cpu( val );
    magma_free_cpu( col );
    magma_free_cpu( row );
    magma_free_cpu( rowidx );
    magma_free_cpu( list );
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This function creates a col-pointer and a linked list along the columns
    for a row-major CSR matrix

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                The format is unsorted CSR, the list array is used as linked
                list pointing to the respectively next entry.

    @param[in,out]
    AC          magma_z_matrix*
                The matrix A but with row-pointer being for col-major, same with
                linked list. The values, col and row indices are unchanged.
                The respective pointers point to the entities of A. Already allocated.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_colmajorup(
    magma_z_matrix A,
    magma_z_matrix *AC,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t num_threads=1;
    magma_index_t *checkrow;
    magma_int_t el_per_block;
    AC->nnz=A.nnz;
    AC->true_nnz=A.true_nnz;
    AC->val=A.val;
    AC->col=A.rowidx;
    AC->rowidx=A.col;

    CHECK( magma_index_malloc_cpu( &checkrow, A.num_rows ));

#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif

    #pragma omp parallel for
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        checkrow[i] = -1;
    }

    #pragma omp parallel for
    for( magma_int_t i=0; i<AC->true_nnz; i++ ){
        AC->list[ i ] = 0;
    }
     el_per_block = magma_ceildiv( A.num_rows, num_threads );

    #pragma omp parallel
    {
#ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
        for(magma_int_t i=0; i<A.nnz; i++ ){
            if( A.list[ i ]!= -1 ){
                magma_index_t row = A.col[ i ];
                if( (row < (id+1)*el_per_block) && (row >=(id)*el_per_block)  ){
                    if( checkrow[row] == -1 ){
                        AC->row[ row ] = i;
                        AC->list[ i ] = 0;
                        checkrow[ row ] = i;
                    } else {
                        AC->list[ checkrow[ row ] ] = i;
                        AC->list[ i ] = 0;
                        checkrow[ row ] = i;
                    }
                }
            }
        }
    }


cleanup:
    magma_free_cpu( checkrow );
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Inserts for the iterative dynamic ILU an new element in the (empty) place.

    Arguments
    ---------

    @param[in]
    num_rmL     magma_int_t
                Number of Elements that are replaced in L.

    @param[in]
    num_rmU     magma_int_t
                Number of Elements that are replaced in U.

    @param[in]
    rm_locL     magma_index_t*
                List containing the locations of the deleted elements.

    @param[in]
    rm_locU     magma_index_t*
                List containing the locations of the deleted elements.

    @param[in]
    L_new       magma_z_matrix
                Elements that will be inserted in L stored in COO format (unsorted).

    @param[in]
    U_new       magma_z_matrix
                Elements that will be inserted in U stored in COO format (unsorted).

    @param[in,out]
    L           magma_z_matrix*
                matrix where new elements are inserted.
                The format is unsorted CSR, list is used as linked
                list pointing to the respectively next entry.

    @param[in,out]
    U           magma_z_matrix*
                matrix where new elements are inserted. Row-major.
                The format is unsorted CSR, list is used as linked
                list pointing to the respectively next entry.

    @param[in,out]
    UR          magma_z_matrix*
                Same matrix as U, but column-major.
                The format is unsorted CSR, list is used as linked
                list pointing to the respectively next entry.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_insert(
    magma_int_t *num_rmL,
    magma_int_t *num_rmU,
    magma_index_t *rm_locL,
    magma_index_t *rm_locU,
    magma_z_matrix *L_new,
    magma_z_matrix *U_new,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_z_matrix *UR,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    // first part L
    #pragma omp parallel
    {
    #ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
    magma_int_t el = L_new->row[id];
    magma_int_t loc_lr = rm_locL[id];

    while( el>-1 ){
        magma_int_t loc = L->nnz + loc_lr;
        loc_lr++;
        magma_index_t new_row = L_new->rowidx[ el ];
        magma_index_t new_col = L_new->col[ el ];
        magma_index_t old_rowstart = L->row[ new_row ];
        //printf("%%candidate for L: (%d,%d) tid %d\n", new_row, new_col, id);
        if( new_col < L->col[ old_rowstart ] ){
            //printf("%%insert in L: (%d,%d) at location %d\n", new_row, new_col, loc);
            L->row[ new_row ] = loc;
            L->list[ loc ] = old_rowstart;
            L->rowidx[ loc ] = new_row;
            L->col[ loc ] = new_col;
            L->val[ loc ] = MAGMA_Z_ZERO;
        }
        else if( new_col == L->col[ old_rowstart ] ){
            ; //printf("%% tried to insert duplicate in L! case 1 tid %d location %d (%d,%d) = (%d,%d)\n", id, r, new_row,new_col,L->rowidx[ old_rowstart ], L->col[ old_rowstart ]); fflush(stdout);
        }
        else{
            magma_int_t j = old_rowstart;
            magma_int_t jn = L->list[j];
            // this will finish, as we consider the lower triangular
            // and we always have the diagonal!
            while( j!=0 ){
                if( L->col[jn]==new_col ){
                    // printf("%% tried to insert duplicate case 1 2 in L thread %d: (%d %d) \n", id, new_row, new_col); fflush(stdout);
                    j=0; //break;
                }else if( L->col[jn]>new_col ){
                    //printf("%%insert in L: (%d,%d) at location %d\n", new_row, new_col, loc);
                    L->list[j]=loc;
                    L->list[loc]=jn;
                    L->rowidx[ loc ] = new_row;
                    L->col[ loc ] = new_col;
                    L->val[ loc ] = MAGMA_Z_ZERO;
                    j=0; //break;
                } else{
                    j=jn;
                    jn=L->list[jn];
                }
            }
        }
        el = L_new->list[ el ];
    }
    }
    // second part U
    #pragma omp parallel
    {
    #ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
    magma_int_t el = U_new->row[id];
    magma_int_t loc_ur = rm_locU[id];

    while( el>-1 ){
        magma_int_t loc = U->nnz + loc_ur;
        loc_ur++;
        magma_index_t new_col = U_new->rowidx[ el ];    // we flip these entities to have the same
        magma_index_t new_row = U_new->col[ el ];    // situation like in the lower triangular case
        //printf("%%candidate for U: (%d,%d) tid %d\n", new_row, new_col, id);
        if( new_row < new_col ){
            printf("%% illegal candidate %5lld for U: (%d,%d)'\n", (long long)el, new_row, new_col);
        }
        //printf("%% candidate %d for U: (%d,%d)'\n", el, new_row, new_col);
        magma_index_t old_rowstart = U->row[ new_row ];
        //printf("%%candidate for U: tid %d %d < %d (%d,%d) going to %d+%d+%d+%d = %d\n", id, add_loc[id+1]-1, rm_locU[id+1], new_row, new_col, U->nnz, rm_locU[id], id, add_loc[id+1]-1, loc); fflush(stdout);
        if( new_col < U->col[ old_rowstart ] ){
            //  printf("%% insert in U as first element: (%d,%d)'\n", new_row, new_col);
            U->row[ new_row ] = loc;
            U->list[ loc ] = old_rowstart;
            U->rowidx[ loc ] = new_row;
            U->col[ loc ] = new_col;
            U->val[ loc ] = MAGMA_Z_ZERO;
        }
        else if( new_col == U->col[ old_rowstart ] ){
            ; //printf("%% tried to insert duplicate in U! case 1 single element (%d,%d) at %d \n", new_row, new_col, r);
        }
        else{
            magma_int_t j = old_rowstart;
            magma_int_t jn = U->list[j];
            while( j!=0 ){
                if( U->col[j]==new_col ){
                    // printf("%% tried to insert duplicate case 1 2 in U thread %d: (%d %d) \n", id, new_row, new_col);
                    j=0; //break;
                }else if( U->col[jn]>new_col ){
                    U->list[j]=loc;
                    U->list[loc]=jn;
                    U->rowidx[ loc ] = new_row;
                    U->col[ loc ] = new_col;
                    U->val[ loc ] = MAGMA_Z_ZERO;
                    j=0; //break;
                } else{
                    j=jn;
                    jn=U->list[jn];
                }
            }
        }
        el = U_new->list[el];
    }
    }

     return info;
}



/***************************************************************************//**
    Purpose
    -------
    This function identifies the candidates like they appear as ILU1 fill-in.
    In this version, the matrices are assumed unordered,
    the linked list is traversed to acces the entries of a row.

    Arguments
    ---------

    @param[in]
    L0          magma_z_matrix
                tril( ILU(0) ) pattern of original system matrix.
                
    @param[in]
    U0          magma_z_matrix
                triu( ILU(0) ) pattern of original system matrix.
                
    @param[in]
    L           magma_z_matrix
                Current lower triangular factor.

    @param[in]
    U           magma_z_matrix
                Current upper triangular factor.

    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates for L in COO format.

    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates for U in COO format.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_candidates(
    magma_z_matrix L0,
    magma_z_matrix U0,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *L_new,
    magma_z_matrix *U_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_index_t *insertedL;
    magma_index_t *insertedU;
    double thrs = 1e-8;
    
    magma_int_t orig = 1; // the pattern L0 and U0 is considered
    magma_int_t existing = 0; // existing elements are also considered
    magma_int_t ilufill = 1;
    
    magma_int_t num_threads;
    
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif
    
    // for now: also some part commented out. If it turns out
    // this being correct, I need to clean up the code.

    CHECK( magma_index_malloc_cpu( &L_new->row, L.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &U_new->row, U.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &insertedL, L.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &insertedU, U.num_rows+1 )); 
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<L.num_rows+1; i++ ){
        L_new->row[i] = 0;
        U_new->row[i] = 0;
        insertedL[i] = 0;
        insertedU[i] = 0;
    }
    L_new->num_rows = L.num_rows;
    L_new->num_cols = L.num_cols;
    L_new->storage_type = Magma_CSR;
    L_new->memory_location = Magma_CPU;
    
    U_new->num_rows = L.num_rows;
    U_new->num_cols = L.num_cols;
    U_new->storage_type = Magma_CSR;
    U_new->memory_location = Magma_CPU;
    
    // go over the original matrix - this is the only way to allow elements to come back...
    if( orig == 1 ){
       #pragma omp parallel for
        for( magma_index_t row=0; row<L0.num_rows; row++){
            magma_int_t numaddrowL = 0;
            magma_int_t ilu0 = L0.row[row];
            magma_int_t ilut = L.row[row];
            magma_int_t endilu0 = L0.row[ row+1 ];
            magma_int_t endilut = L.row[ row+1 ]; 
            magma_int_t ilu0col;
            magma_int_t ilutcol;
            do{
                ilu0col = L0.col[ ilu0 ];
                ilutcol = L.col[ ilut ];
                if( ilu0col == ilutcol ){
                    ilu0++;
                    ilut++;
                    if( existing==1 )
                        numaddrowL++;
                }
                else if( ilutcol<ilu0col ){
                    ilut++;
                    if( existing==1 )
                        numaddrowL++;
                }
                else {
                    // this element is missing in the current approximation
                    // mark it as candidate
                    numaddrowL++;
                    ilu0++;
                }
            }while( ilut<endilut && ilu0<endilu0 );
            // do the rest if existing
            if( ilu0<endilu0 ){
                do{
                    numaddrowL++;
                    ilu0++;
                }while( ilu0<endilu0 ); 
            }
            L_new->row[ row+1 ] = L_new->row[ row+1 ]+numaddrowL;
        }
        
        // same for U
       #pragma omp parallel for
        for( magma_index_t row=0; row<U0.num_rows; row++){
            magma_int_t numaddrowU = 0;
            magma_int_t ilu0 = U0.row[row];
            magma_int_t ilut = U.row[row];
            magma_int_t endilu0 = U0.row[ row+1 ];
            magma_int_t endilut = U.row[ row+1 ]; 
            magma_int_t ilu0col;
            magma_int_t ilutcol;
            do{
                ilu0col = U0.col[ ilu0 ];
                ilutcol = U.col[ ilut ];
                if( ilu0col == ilutcol ){
                    ilu0++;
                    ilut++;
                    if( existing==1 )
                        numaddrowU++;
                }
                else if( ilutcol<ilu0col ){
                    ilut++;
                    if( existing==1 )
                        numaddrowU++;
                }
                else {
                    // this element is missing in the current approximation
                    // mark it as candidate
                    numaddrowU++;
                    ilu0++;
                }
            }while( ilut<endilut && ilu0<endilu0 );
            if( ilu0<endilu0 ){
                do{
                    numaddrowU++;
                    ilu0++;
                }while( ilu0<endilu0 ); 
            }
            U_new->row[ row+1 ] = U_new->row[ row+1 ]+numaddrowU;
        }
    } // end original
    if( ilufill == 1 ){
        // how to determine candidates:
        // for each node i, look at any "intermediate" neighbor nodes numbered
        // less, and then see if this neighbor has another neighbor j numbered
        // more than the intermediate; if so, fill in is (i,j) if it is not
        // already nonzero
        #pragma omp parallel for
        for( magma_index_t row=0; row<L.num_rows; row++){
            magma_int_t numaddrowL = 0, numaddrowU = 0;
            // loop first element over row - only for elements smaller the diagonal
            for( magma_index_t el1=L.row[row]; el1<L.row[row+1]-1; el1++ ){
                magma_index_t col1 = L.col[ el1 ];
                // now check the upper triangular
                // second loop first element over row - only for elements larger the intermediate
                for( magma_index_t el2 = U.row[ col1 ]+1; el2 < U.row[ col1+1 ]; el2++ ){
                    magma_index_t col2 = U.col[ el2 ];
                    magma_index_t cand_row = row;
                    magma_index_t cand_col = col2;
                    // check whether this element already exists
                    // first case: part of L
                    if( cand_col < row ){
                        // check whether this element already exists in L
                        magma_int_t exist = 0;
                        if( existing == 0 ){
                            for(magma_index_t k=L.row[cand_row]; k<L.row[cand_row+1]; k++ ){
                                if( L.col[ k ] == cand_col ){
                                        exist = 1;
                                        break;
                                }
                            }
                        }
                        // if it does not exist, increase counter for this location
                        // use the entry one further down to allow for parallel insertion
                        if( exist == 0 ){
                            //printf("checked row: %d this element does not yet exist in L: (%d,%d)\n", cand_row, cand_col);
                            numaddrowL++;
                            //numaddL[ row+1 ]++;
                        }
                    } else {
                        // check whether this element already exists in U
                        magma_int_t exist = 0;
                        if( existing == 0 ){
                            for(magma_index_t k=U.row[cand_row]; k<U.row[cand_row+1]; k++ ){
                                if( U.col[ k ] == cand_col ){
                                        exist = 1;
                                        break;
                                }
                            }
                        }
                        // if it does not exist, increase counter for this location
                        // use the entry one further down to allow for parallel insertion
                        if( exist == 0 ){
                            //printf("checked row: %d this element does not yet exist in L: (%d,%d)\n", cand_row, cand_col);
                            numaddrowU++;
                            //numaddL[ row+1 ]++;
                        }
                    }
                }
    
            }
            U_new->row[ row+1 ] = U_new->row[ row+1 ]+numaddrowU;
            L_new->row[ row+1 ] = L_new->row[ row+1 ]+numaddrowL;
        }
    } // end ilu-fill
    //end = magma_sync_wtime( queue ); printf("llop 1.2 : %.2e\n", end-start);
    // #########################################################################

    // get the total candidate count
    L_new->nnz = 0;
    U_new->nnz = 0;
    L_new->row[ 0 ] = L_new->nnz;
    U_new->row[ 0 ] = U_new->nnz;

    #pragma omp parallel
    {
#ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
        if( id == 0 ){
            for( magma_int_t i = 0; i<L.num_rows; i++ ){
                L_new->nnz = L_new->nnz + L_new->row[ i+1 ];
                L_new->row[ i+1 ] = L_new->nnz;
            }
        }
        if( id == num_threads-1 ){
            for( magma_int_t i = 0; i<U.num_rows; i++ ){
                U_new->nnz = U_new->nnz + U_new->row[ i+1 ];
                U_new->row[ i+1 ] = U_new->nnz;
            }
        }
    }
    magma_zmalloc_cpu( &L_new->val, L_new->nnz );
    magma_index_malloc_cpu( &L_new->rowidx, L_new->nnz );
    magma_index_malloc_cpu( &L_new->col, L_new->nnz );
    
    magma_zmalloc_cpu( &U_new->val, U_new->nnz );
    magma_index_malloc_cpu( &U_new->rowidx, U_new->nnz );
    magma_index_malloc_cpu( &U_new->col, U_new->nnz );
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<L_new->nnz; i++ ){
        L_new->val[i] = MAGMA_Z_ZERO;
    }
    #pragma omp parallel for
    for( magma_int_t i=0; i<U_new->nnz; i++ ){
        U_new->val[i] = MAGMA_Z_ZERO;
    }
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<L_new->nnz; i++ ){
        L_new->col[i] = -1;
        L_new->rowidx[i] = -1;
    }
    #pragma omp parallel for
    for( magma_int_t i=0; i<U_new->nnz; i++ ){
        U_new->col[i] = -1;
        U_new->rowidx[i] = -1;
    }

    // #########################################################################

    
    if( orig == 1 ){
        
       #pragma omp parallel for
        for( magma_index_t row=0; row<L0.num_rows; row++){
            magma_int_t laddL = 0;
            magma_int_t offsetL = L_new->row[row];
            magma_int_t ilu0 = L0.row[row];
            magma_int_t ilut = L.row[row];
            magma_int_t endilu0 = L0.row[ row+1 ];
            magma_int_t endilut = L.row[ row+1 ]; 
            magma_int_t ilu0col;
            magma_int_t ilutcol;
            do{
                ilu0col = L0.col[ ilu0 ];
                ilutcol = L.col[ ilut ];
                if( ilu0col == ilutcol ){
                    ilu0++;
                    ilut++;
                    if( existing==1 ){
                        L_new->col[ offsetL + laddL ] = ilu0col;
                        L_new->rowidx[ offsetL + laddL ] = row;
                        L_new->val[ offsetL + laddL ] = MAGMA_Z_ONE;
                        laddL++;
                    }
                }
                else if( ilutcol<ilu0col ){
                    if( existing==1 ){
                        L_new->col[ offsetL + laddL ] = ilutcol;
                        L_new->rowidx[ offsetL + laddL ] = row;
                        L_new->val[ offsetL + laddL ] = MAGMA_Z_ONE;
                        laddL++;
                    }
                    ilut++;
                }
                else {
                    // this element is missing in the current approximation
                    // mark it as candidate
                    L_new->col[ offsetL + laddL ] = ilu0col;
                    L_new->rowidx[ offsetL + laddL ] = row;
                    L_new->val[ offsetL + laddL ] = MAGMA_Z_ONE + MAGMA_Z_ONE + MAGMA_Z_ONE;
                    laddL++;
                    ilu0++;
                }
            }while( ilut<endilut && ilu0<endilu0 );
            if( ilu0<endilu0 ){
                do{
                    ilu0col = L0.col[ ilu0 ];
                    L_new->col[ offsetL + laddL ] = ilu0col;
                    L_new->rowidx[ offsetL + laddL ] = row;
                    L_new->val[ offsetL + laddL ] = MAGMA_Z_ONE + MAGMA_Z_ONE + MAGMA_Z_ONE;
                    laddL++;
                    ilu0++;
                }while( ilu0<endilu0 ); 
            }
            insertedL[row] = laddL;
        }
        
        // same for U
       #pragma omp parallel for
        for( magma_index_t row=0; row<U0.num_rows; row++){
            magma_int_t laddU = 0;
            magma_int_t offsetU = U_new->row[row];
            magma_int_t ilu0 = U0.row[row];
            magma_int_t ilut = U.row[row];
            magma_int_t endilu0 = U0.row[ row+1 ];
            magma_int_t endilut = U.row[ row+1 ]; 
            magma_int_t ilu0col;
            magma_int_t ilutcol;
            do{
                ilu0col = U0.col[ ilu0 ];
                ilutcol = U.col[ ilut ];
                if( ilu0col == ilutcol ){
                    ilu0++;
                    ilut++;
                    if( existing==1 ){
                        U_new->col[ offsetU + laddU ] = ilu0col;
                        U_new->rowidx[ offsetU + laddU ] = row;
                        U_new->val[ offsetU + laddU ] = MAGMA_Z_ONE;
                        laddU++;
                    }
                }
                else if( ilutcol<ilu0col ){
                    if( existing==1 ){
                        U_new->col[ offsetU + laddU ] = ilutcol;
                        U_new->rowidx[ offsetU + laddU ] = row;
                        U_new->val[ offsetU + laddU ] = MAGMA_Z_ONE;
                        laddU++;
                    }
                    ilut++;
                }
                else {
                    // this element is missing in the current approximation
                    // mark it as candidate
                    U_new->col[ offsetU + laddU ] = ilu0col;
                    U_new->rowidx[ offsetU + laddU ] = row;
                    U_new->val[ offsetU + laddU ] = MAGMA_Z_ONE + MAGMA_Z_ONE + MAGMA_Z_ONE;
                    laddU++;
                    ilu0++;
                }
            }while( ilut<endilut && ilu0<endilu0 );
            if( ilu0<endilu0 ){
                do{
                    ilu0col = U0.col[ ilu0 ];
                    U_new->col[ offsetU + laddU ] = ilu0col;
                    U_new->rowidx[ offsetU + laddU ] = row;
                    U_new->val[ offsetU + laddU ] = MAGMA_Z_ONE + MAGMA_Z_ONE + MAGMA_Z_ONE;
                    laddU++;
                    ilu0++;
                }while( ilu0<endilu0 ); 
            }
            insertedU[row] = laddU;
        }
    } // end original
    
    if( ilufill==1 ){
        #pragma omp parallel for
        for( magma_index_t row=0; row<L.num_rows; row++){
            magma_int_t laddL = 0;
            magma_int_t laddU = 0;
            magma_int_t offsetL = L_new->row[row] + insertedL[row];
            magma_int_t offsetU = U_new->row[row] + insertedU[row];
            // loop first element over row - only for elements smaller the diagonal
            for( magma_index_t el1=L.row[row]; el1<L.row[row+1]-1; el1++ ){
                
                magma_index_t col1 = L.col[ el1 ];
                // now check the upper triangular
                // second loop first element over row - only for elements larger the intermediate
                for( magma_index_t el2 = U.row[ col1 ]+1; el2 < U.row[ col1+1 ]; el2++ ){
                    magma_index_t col2 = U.col[ el2 ];
                    magma_index_t cand_row = row;
                    magma_index_t cand_col = col2;
                    //$########### we now have the candidate cand_row cand_col
                    
                    
                    // check whether this element already exists
                    // first case: part of L
                    if( cand_col < row ){
                        magma_int_t exist = 0;
                        if( existing == 0 ){
                            for(magma_index_t k=L.row[cand_row]; k<L.row[cand_row+1]; k++ ){
                                if( L.col[ k ] == cand_col ){
                                        exist = 1;
                                        break;
                                }
                            }
                        }
    #ifdef AVOID_DUPLICATES
                        for( magma_int_t k=L_new->row[cand_row]; k<L_new->row[cand_row+1]; k++){
                            if( L_new->col[ k ] == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                break;
                            }
                        }
    #endif
                        // if it does not exist, increase counter for this location
                        // use the entry one further down to allow for parallel insertion
                        if( exist == 0 ){
                            //  printf("---------------->>>  candidate in L at (%d, %d)\n", cand_row, cand_col);
                            //add in the next location for this row
                            // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                            L_new->rowidx[ offsetL + laddL ] = cand_row;
                            L_new->col[ offsetL + laddL ] = cand_col;
                            L_new->val[ offsetL + laddL ] = MAGMA_Z_ONE;
                            // L_new->list[ numaddL[row] + laddL ] = -1;
                            // L_new->row[ numaddL[row] + laddL ] = -1;
                            laddL++;
                        }
                    } else {
                        // check whether this element already exists in U
                        magma_int_t exist = 0;
                        if( existing == 0 ){
                            for(magma_index_t k=U.row[cand_row]; k<U.row[cand_row+1]; k++ ){
                                if( U.col[ k ] == cand_col ){
                                        exist = 1;
                                        break;
                                }
                            }
                        }
    #ifdef AVOID_DUPLICATES
                        for( magma_int_t k=U_new->row[cand_row]; k<U_new->row[cand_row+1]; k++){
                            if( U_new->col[ k ] == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                break;
                            }
                        }
    #endif
                        // if it does not exist, increase counter for this location
                        // use the entry one further down to allow for parallel insertion
                        if( exist == 0 ){
                            //  printf("---------------->>>  candidate in U at (%d, %d) stored in %d\n", cand_row, cand_col, numaddU[row] + laddU);
                            //add in the next location for this row
                            // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                            U_new->rowidx[ offsetU + laddU ] = cand_row;
                            U_new->col[ offsetU + laddU ] = cand_col;
                            U_new->val[ offsetU + laddU ] = MAGMA_Z_ONE;
                            // U_new->list[ numaddU[row] + laddU ] = -1;
                            // U_new->row[ numaddU[row] + laddU ] = -1;
                            laddU++;
                            //if( cand_row > cand_col )
                             //   printf("inserted illegal candidate in case 4: (%d,%d) coming from (%d,%d)->(%d,%d)\n", cand_row, cand_col, row, col1, U.col[U.row[ cand_col ]], UR.col[ el2 ]);
                        }
                    }
                }
            }
            //insertedU[row] = insertedU[row] + laddU;
            //insertedL[row] = insertedL[row] + laddL;
        }
    } //end ilufill
    
#ifdef AVOID_DUPLICATES
        // #####################################################################
        
        CHECK( magma_zparilut_thrsrm( 1, L_new, &thrs, queue ) );
        CHECK( magma_zparilut_thrsrm( 1, U_new, &thrs, queue ) );

        // #####################################################################
#endif

cleanup:
    magma_free_cpu( insertedL );
    magma_free_cpu( insertedU );
    return info;
}



/***************************************************************************//**
    Purpose
    -------
    This function identifies the candidates like they appear as ILU1 fill-in.
    In this version, the matrices are assumed unordered,
    the linked list is traversed to acces the entries of a row.

    Arguments
    ---------

    @param[in]
    L0          magma_z_matrix
                tril( ILU(0) ) pattern of original system matrix.
                
    @param[in]
    U0          magma_z_matrix
                triu( ILU(0) ) pattern of original system matrix.
                
    @param[in]
    L           magma_z_matrix
                Current lower triangular factor.

    @param[in]
    U           magma_z_matrix
                Current upper triangular factor transposed.

    @param[in]
    UR          magma_z_matrix
                Current upper triangular factor - col-pointer and col-list.


    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates for L in COO format.

    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates for U in COO format.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_candidates_semilinked( // new
    magma_z_matrix L0, // CSR
    magma_z_matrix U0, // CSC
    magma_z_matrix L, // CSR 
    magma_z_matrix U, // CSC
    magma_z_matrix UT, // CSR
    magma_z_matrix *L_new, // CSR
    magma_z_matrix *U_new,  //CSC
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_index_t *insertedL;
    magma_index_t *insertedU;
    double thrs = 1e-8;
    
    magma_int_t orig = 1; // the pattern L0 and U0 is considered
    magma_int_t existing = 0; // existing elements are also considered
    magma_int_t ilufill = 1;
    
    magma_int_t num_threads;
    
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif
    
    // for now: also some part commented out. If it turns out
    // this being correct, I need to clean up the code.

    CHECK( magma_index_malloc_cpu( &L_new->row, L.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &U_new->row, U.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &insertedL, L.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &insertedU, U.num_rows+1 )); 
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<L.num_rows+1; i++ ){
        L_new->row[i] = 0;
        U_new->row[i] = 0;
        insertedL[i] = 0;
        insertedU[i] = 0;
    }
    L_new->num_rows = L.num_rows;
    L_new->num_cols = L.num_cols;
    L_new->storage_type = Magma_CSR;
    L_new->memory_location = Magma_CPU;
    
    U_new->num_rows = L.num_rows;
    U_new->num_cols = L.num_cols;
    U_new->storage_type = Magma_CSR;
    U_new->memory_location = Magma_CPU;
    
    // go over the original matrix - this is the only way to allow elements to come back...
    if( orig == 1 ){
       #pragma omp parallel for
        for( magma_index_t row=0; row<L0.num_rows; row++){
            magma_int_t numaddrowL = 0;
            magma_int_t ilu0 = L0.row[row];
            magma_int_t ilut = L.row[row];
            magma_int_t endilu0 = L0.row[ row+1 ];
            magma_int_t endilut = L.row[ row+1 ]; 
            magma_int_t ilu0col;
            magma_int_t ilutcol;
            do{
                ilu0col = L0.col[ ilu0 ];
                ilutcol = L.col[ ilut ];
                if( ilu0col == ilutcol ){
                    ilu0++;
                    ilut++;
                    if( existing==1 )
                        numaddrowL++;
                }
                else if( ilutcol<ilu0col ){
                    ilut++;
                    if( existing==1 )
                        numaddrowL++;
                }
                else {
                    // this element is missing in the current approximation
                    // mark it as candidate
                    numaddrowL++;
                    ilu0++;
                }
            }while( ilut<endilut && ilu0<endilu0 );
            L_new->row[ row+1 ] = L_new->row[ row+1 ]+numaddrowL;
        }
        
        // same for U
       #pragma omp parallel for
        for( magma_index_t row=0; row<U0.num_rows; row++){
            magma_int_t numaddrowU = 0;
            magma_int_t ilu0 = U0.row[row];
            magma_int_t ilut = U.row[row];
            magma_int_t endilu0 = U0.row[ row+1 ];
            magma_int_t endilut = U.row[ row+1 ]; 
            magma_int_t ilu0col;
            magma_int_t ilutcol;
            do{
                ilu0col = U0.col[ ilu0 ];
                ilutcol = U.col[ ilut ];
                if( ilu0col == ilutcol ){
                    ilu0++;
                    ilut++;
                    if( existing==1 )
                        numaddrowU++;
                }
                else if( ilutcol<ilu0col ){
                    ilut++;
                    if( existing==1 )
                        numaddrowU++;
                }
                else {
                    // this element is missing in the current approximation
                    // mark it as candidate
                    numaddrowU++;
                    ilu0++;
                }
            }while( ilut<endilut && ilu0<endilu0 );
            U_new->row[ row+1 ] = U_new->row[ row+1 ]+numaddrowU;
        }
    } // end original
    printf("originals done\n");
    if( ilufill == 1 ){
        // how to determine candidates:
        // go over the rows i in L, ignore diagonal
        // for every element in column k go over row k in U
        // is stored in CSC, so use the linked list
        // for every match in column j in U check whether this element already 
        // exists: i<k -> chick in L, i>j -> check in U
        // if not, it is a candidate
        // be aware: candidates for U are generated in CSC format!
        #pragma omp parallel for
        for( magma_index_t row=0; row<L.num_rows; row++){
            magma_int_t numaddrowL = 0, numaddrowU = 0;
            // loop first element over row - only for elements smaller the diagonal
            for( magma_index_t el1=L.row[row]; el1<L.row[row+1]-1; el1++ ){
                magma_index_t col1 = L.col[ el1 ];
                // now check the upper triangular
                // the upper triangular is stored in CSC!
                // use the linked list for access (UT)
                magma_index_t el2 = UT.row[ col1 ];
                do{
                    magma_index_t col2 = UT.rowidx[ el2 ];
                    magma_index_t cand_row = row;
                    magma_index_t cand_col = col2;
                    // check whether this element already exists
                    // first case: part of L
                    if( cand_col < cand_row ){
                        // check whether this element already exists in L
                        magma_int_t exist = 0;
                        if( existing == 0 ){
                            for(magma_index_t k=L.row[cand_row]; k<L.row[cand_row+1]; k++ ){
                                if( L.col[ k ] == cand_col ){
                                        exist = 1;
                                        break;
                                }
                            }
                        }
                        // if it does not exist, increase counter for this location
                        // use the entry one further down to allow for parallel insertion
                        if( exist == 0 ){
                            //printf("checked row: %d this element does not yet exist in L: (%d,%d)\n", cand_row, cand_col);
                            numaddrowL++;
                            //numaddL[ row+1 ]++;
                        }
                    } else {
                        // check whether this element already exists in U
                        // cand_row and cand_col are here flipped as we have U in transposed form (CSC)
                        magma_int_t exist = 0;
                        if( existing == 0 ){
                            for(magma_index_t k=U.row[cand_col]; k<U.row[cand_col+1]; k++ ){
                                if( U.col[ k ] == cand_row ){
                                        exist = 1;
                                        break;
                                }
                            }
                        }
                        // if it does not exist, increase counter for this location
                        // use the entry one further down to allow for parallel insertion
                        if( exist == 0 ){
                            //printf("checked row: %d this element does not yet exist in L: (%d,%d)\n", cand_row, cand_col);
                            numaddrowU++;
                            //numaddL[ row+1 ]++;
                        }
                    }
                    el2 = UT.list[el2]; 
                    printf("el2:%d\n", el2);
                }while( el2 > 0 );
    
            }
            U_new->row[ row+1 ] = U_new->row[ row+1 ]+numaddrowU;
            L_new->row[ row+1 ] = L_new->row[ row+1 ]+numaddrowL;
        }
    } // end ilu-fill
    //end = magma_sync_wtime( queue ); printf("llop 1.2 : %.2e\n", end-start);
    // #########################################################################
   printf("first sweep done\n");
    // get the total candidate count
    L_new->nnz = 0;
    U_new->nnz = 0;
    L_new->row[ 0 ] = L_new->nnz;
    U_new->row[ 0 ] = U_new->nnz;

    #pragma omp parallel
    {
#ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
        if( id == 0 ){
            for( magma_int_t i = 0; i<L.num_rows; i++ ){
                L_new->nnz = L_new->nnz + L_new->row[ i+1 ];
                L_new->row[ i+1 ] = L_new->nnz;
            }
        }
        if( id == num_threads-1 ){
            for( magma_int_t i = 0; i<U.num_rows; i++ ){
                U_new->nnz = U_new->nnz + U_new->row[ i+1 ];
                U_new->row[ i+1 ] = U_new->nnz;
            }
        }
    }
    
    magma_zmalloc_cpu( &L_new->val, L_new->nnz );
    magma_index_malloc_cpu( &L_new->rowidx, L_new->nnz );
    magma_index_malloc_cpu( &L_new->col, L_new->nnz );
    
    magma_zmalloc_cpu( &U_new->val, U_new->nnz );
    magma_index_malloc_cpu( &U_new->rowidx, U_new->nnz );
    magma_index_malloc_cpu( &U_new->col, U_new->nnz );
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<L_new->nnz; i++ ){
        L_new->val[i] = MAGMA_Z_ZERO;
    }
    #pragma omp parallel for
    for( magma_int_t i=0; i<U_new->nnz; i++ ){
        U_new->val[i] = MAGMA_Z_ZERO;
    }
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<L_new->nnz; i++ ){
        L_new->col[i] = -1;
        L_new->rowidx[i] = -1;
    }
    #pragma omp parallel for
    for( magma_int_t i=0; i<U_new->nnz; i++ ){
        U_new->col[i] = -1;
        U_new->rowidx[i] = -1;
    }

    // #########################################################################
printf("start second sweep\n");
    
    if( orig == 1 ){
        
       #pragma omp parallel for
        for( magma_index_t row=0; row<L0.num_rows; row++){
            magma_int_t laddL = 0;
            magma_int_t offsetL = L_new->row[row];
            magma_int_t ilu0 = L0.row[row];
            magma_int_t ilut = L.row[row];
            magma_int_t endilu0 = L0.row[ row+1 ];
            magma_int_t endilut = L.row[ row+1 ]; 
            magma_int_t ilu0col;
            magma_int_t ilutcol;
            do{
                ilu0col = L0.col[ ilu0 ];
                ilutcol = L.col[ ilut ];
                if( ilu0col == ilutcol ){
                    ilu0++;
                    ilut++;
                    if( existing==1 ){
                        L_new->col[ offsetL + laddL ] = ilu0col;
                        L_new->rowidx[ offsetL + laddL ] = row;
                        L_new->val[ offsetL + laddL ] = MAGMA_Z_ONE;
                        laddL++;
                    }
                }
                else if( ilutcol<ilu0col ){
                    if( existing==1 ){
                        L_new->col[ offsetL + laddL ] = ilutcol;
                        L_new->rowidx[ offsetL + laddL ] = row;
                        L_new->val[ offsetL + laddL ] = MAGMA_Z_ONE;
                        laddL++;
                    }
                    ilut++;
                }
                else {
                    // this element is missing in the current approximation
                    // mark it as candidate
                    L_new->col[ offsetL + laddL ] = ilu0col;
                    L_new->rowidx[ offsetL + laddL ] = row;
                    L_new->val[ offsetL + laddL ] = MAGMA_Z_ONE + MAGMA_Z_ONE + MAGMA_Z_ONE;
                    laddL++;
                    ilu0++;
                }
            }while( ilut<endilut && ilu0<endilu0 );
            insertedL[row] = laddL;
        }
        
        // same for U
       #pragma omp parallel for
        for( magma_index_t row=0; row<U0.num_rows; row++){
            magma_int_t laddU = 0;
            magma_int_t offsetU = U_new->row[row];
            magma_int_t ilu0 = U0.row[row];
            magma_int_t ilut = U.row[row];
            magma_int_t endilu0 = U0.row[ row+1 ];
            magma_int_t endilut = U.row[ row+1 ]; 
            magma_int_t ilu0col;
            magma_int_t ilutcol;
            do{
                ilu0col = U0.col[ ilu0 ];
                ilutcol = U.col[ ilut ];
                if( ilu0col == ilutcol ){
                    ilu0++;
                    ilut++;
                    if( existing==1 ){
                        U_new->col[ offsetU + laddU ] = ilu0col;
                        U_new->rowidx[ offsetU + laddU ] = row;
                        U_new->val[ offsetU + laddU ] = MAGMA_Z_ONE;
                        laddU++;
                    }
                }
                else if( ilutcol<ilu0col ){
                    if( existing==1 ){
                        U_new->col[ offsetU + laddU ] = ilutcol;
                        U_new->rowidx[ offsetU + laddU ] = row;
                        U_new->val[ offsetU + laddU ] = MAGMA_Z_ONE;
                        laddU++;
                    }
                    ilut++;
                }
                else {
                    // this element is missing in the current approximation
                    // mark it as candidate
                    U_new->col[ offsetU + laddU ] = ilu0col;
                    U_new->rowidx[ offsetU + laddU ] = row;
                    U_new->val[ offsetU + laddU ] = MAGMA_Z_ONE + MAGMA_Z_ONE + MAGMA_Z_ONE;
                    laddU++;
                    ilu0++;
                }
            }while( ilut<endilut && ilu0<endilu0 );
            insertedU[row] = laddU;
        }
    } // end original
    printf("originals done\n");
    if( ilufill==1 ){
        #pragma omp parallel for
        for( magma_index_t row=0; row<L.num_rows; row++){
            magma_int_t laddL = 0;
            magma_int_t laddU = 0;
            magma_int_t offsetL = L_new->row[row] + insertedL[row];
            magma_int_t offsetU = U_new->row[row] + insertedU[row];
            // loop first element over row - only for elements smaller the diagonal
            for( magma_index_t el1=L.row[row]; el1<L.row[row+1]-1; el1++ ){
                
                magma_index_t col1 = L.col[ el1 ];
                // now check the upper triangular
                // second loop first element over row - only for elements larger the intermediate
                magma_index_t el2 = UT.row[ col1 ];
                do{
                    magma_index_t col2 = UT.rowidx[ el2 ];
                    magma_index_t cand_row = row;
                    magma_index_t cand_col = col2;
                    //$########### we now have the candidate cand_row cand_col
                    
                    
                    // check whether this element already exists
                    // first case: part of L
                    if( cand_col < row ){
                        magma_int_t exist = 0;
                        if( existing == 0 ){
                            for(magma_index_t k=L.row[cand_row]; k<L.row[cand_row+1]; k++ ){
                                if( L.col[ k ] == cand_col ){
                                        exist = 1;
                                        break;
                                }
                            }
                        }
    #ifdef AVOID_DUPLICATES
                        for( magma_int_t k=L_new->row[cand_row]; k<L_new->row[cand_row+1]; k++){
                            if( L_new->col[ k ] == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                break;
                            }
                        }
    #endif
                        // if it does not exist, increase counter for this location
                        // use the entry one further down to allow for parallel insertion
                        if( exist == 0 ){
                            //  printf("---------------->>>  candidate in L at (%d, %d)\n", cand_row, cand_col);
                            //add in the next location for this row
                            // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                            L_new->rowidx[ offsetL + laddL ] = cand_row;
                            L_new->col[ offsetL + laddL ] = cand_col;
                            L_new->val[ offsetL + laddL ] = MAGMA_Z_ONE;
                            // L_new->list[ numaddL[row] + laddL ] = -1;
                            // L_new->row[ numaddL[row] + laddL ] = -1;
                            laddL++;
                        }
                    } else {
                        // check whether this element already exists in U
                        magma_int_t exist = 0;
                        if( existing == 0 ){
                            for(magma_index_t k=U.row[cand_col]; k<U.row[cand_col+1]; k++ ){
                                if( U.col[ k ] == cand_row ){
                                        exist = 1;
                                        break;
                                }
                            }
                        }
    #ifdef AVOID_DUPLICATES
                        for( magma_int_t k=U_new->row[cand_col]; k<U_new->row[cand_col+1]; k++){
                            if( U_new->col[ k ] == cand_row ){
                                // element included in LU and nonzero
                                exist = 1;
                                break;
                            }
                        }
    #endif
                        // if it does not exist, increase counter for this location
                        // use the entry one further down to allow for parallel insertion
                        if( exist == 0 ){
                            //  printf("---------------->>>  candidate in U at (%d, %d) stored in %d\n", cand_row, cand_col, numaddU[row] + laddU);
                            //add in the next location for this row
                            // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                            U_new->rowidx[ offsetU + laddU ] = cand_col;
                            U_new->col[ offsetU + laddU ] = cand_row;
                            U_new->val[ offsetU + laddU ] = MAGMA_Z_ONE;
                            // U_new->list[ numaddU[row] + laddU ] = -1;
                            // U_new->row[ numaddU[row] + laddU ] = -1;
                            laddU++;
                            //if( cand_row > cand_col )
                             //   printf("inserted illegal candidate in case 4: (%d,%d) coming from (%d,%d)->(%d,%d)\n", cand_row, cand_col, row, col1, U.col[U.row[ cand_col ]], UR.col[ el2 ]);
                        }
                    }
                    el2 = UT.list[el2]; 
                    printf("el2:%d\n", el2);
                }while( el2 > 0 );
            }
            //insertedU[row] = insertedU[row] + laddU;
            //insertedL[row] = insertedL[row] + laddL;
        }
    } //end ilufill
    printf("second sweep done\n");
#ifdef AVOID_DUPLICATES
        // #####################################################################
        
        CHECK( magma_zparilut_thrsrm( 1, L_new, &thrs, queue ) );
        CHECK( magma_zparilut_thrsrm( 1, U_new, &thrs, queue ) );

        // #####################################################################
#endif

cleanup:
    magma_free_cpu( insertedL );
    magma_free_cpu( insertedU );
    return info;
}





/***************************************************************************//**
    Purpose
    -------
    This routine removes matrix entries from the structure that are smaller
    than the threshold. It only counts the elements deleted, does not
    save the locations.


    Arguments
    ---------

    @param[out]
    thrs        magmaDoubleComplex*
                Thrshold for removing elements.

    @param[out]
    num_rm      magma_int_t*
                Number of Elements that have been removed.

    @param[in,out]
    LU          magma_z_matrix*
                Current ILU approximation where the identified smallest components
                are deleted.

    @param[in,out]
    LUC         magma_z_matrix*
                Corresponding col-list.

    @param[in,out]
    LU_new      magma_z_matrix*
                List of candidates in COO format.

    @param[out]
    rm_loc      magma_index_t*
                List containing the locations of the elements deleted.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_rm_thrs(
    double *thrs,
    magma_int_t *num_rm,
    magma_z_matrix *LU,
    magma_z_matrix *LU_new,
    magma_index_t *rm_loc,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t count_rm = 0;
    magma_int_t num_threads = -1;
    magma_int_t el_per_block;

    // never forget elements
    // magma_int_t offset = LU_new->diameter;
    // never forget last rm

    double bound = *thrs;

#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif
    el_per_block = magma_ceildiv( LU->num_rows, num_threads );

    #pragma omp parallel for
    for( magma_int_t i=0; i<num_threads; i++ ){
        rm_loc[i] = 0;
    }

    #pragma omp parallel
    {
#ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
        magma_int_t lbound = (id+1)*el_per_block;
        if( id == num_threads-1 ){
            lbound = LU->num_rows;
        }
        magma_int_t loc_rm_count = 0;
        for( magma_int_t r=(id)*el_per_block; r<lbound; r++ ){
            magma_int_t i = LU->row[r];
            magma_int_t lasti=i;
            magma_int_t nexti=LU->list[i];
            while( nexti!=0 ){
                if( MAGMA_Z_ABS( LU->val[ i ] ) < bound ){
                        loc_rm_count++;
                        if( LU->row[r] == i ){
                       //     printf(" removed as first element in U rm: (%d,%d) at %d count [%d] \n", r, LU->col[ i ], i, count_rm); fflush(stdout);
                                LU->row[r] = nexti;
                                lasti=i;
                                i = nexti;
                                nexti = LU->list[nexti];
                        }
                        else{
                          //  printf(" removed in linked list in U rm: (%d,%d) at %d count [%d] \n", r, LU->col[ i ], i, count_rm); fflush(stdout);
                            LU->list[lasti] = nexti;
                            i = nexti;
                            nexti = LU->list[nexti];
                        }
                }
                else{
                    lasti = i;
                    i = nexti;
                    nexti = LU->list[nexti];
                }
            }
        }
        rm_loc[ id ] = rm_loc[ id ] + loc_rm_count;
    }

    for(int j=0; j<num_threads; j++){
        count_rm = count_rm + rm_loc[j];
    }

    // never forget elements
    // LU_new->diameter = count_rm+LU_new->diameter;
    // not forget the last rm
    LU_new->diameter = count_rm;
    LU_new->nnz = LU_new->diameter;
    *num_rm = count_rm;

    return info;
}



/***************************************************************************//**
    Purpose
    -------
    This is a helper routine counting elements in a matrix in unordered
    Magma_CSRLIST format.

    Arguments
    ---------

    @param[in]
    L           magma_z_matrix*
                Matrix in Magm_CSRLIST format

    @param[out]
    num         magma_index_t*
                Number of elements counted.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_count(
    magma_z_matrix L,
    magma_int_t *num,
    magma_queue_t queue )
{
    magma_int_t info =0;

    (*num)=0;
    magma_int_t check = 1;
    if( 5 < L.col[L.list[L.row[5]]] &&  5 == L.rowidx[L.list[L.row[5]]] ){
        // printf("check based on: (%d,%d)\n", L.rowidx[L.list[L.row[5]]], L.col[L.list[L.row[5]]]);
        check = -1;
    }

    for( magma_int_t r=0; r < L.num_rows; r++ ) {
        magma_int_t i = L.row[r];
        magma_int_t nexti = L.list[i];
        do{
            if(check == 1 ){
                if( L.col[i] > r ){
                    // printf("error here: (%d,%d)\n",r, L.col[i]);
                    info = -1;
                    break;
                }
            } else if(check == -1 ){
                if( L.col[i] < r ){
                    // printf("error here: (%d,%d)\n",r, L.col[i]);
                    info = -1;
                    break;
                }
            }
            if( nexti != 0 && L.col[i] >  L.col[nexti] ){
                // printf("error here: %d(%d,%d) -> %d(%d,%d) \n",i,L.rowidx[i], L.col[i], nexti,L.rowidx[nexti], L.col[nexti] );
                info = -1;
                break;
            }

            (*num)++;
            i = nexti;
            nexti=L.list[nexti];
        } while (i != 0);
    }

    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Screens the new candidates for multiple elements in the same row.
    We allow for at most one new element per row.
    This changes the algorithm, but pays off in performance.


    Arguments
    ---------

    @param[in]
    num_rmL     magma_int_t
                Number of Elements that are replaced.

    @param[in]
    num_rmU     magma_int_t
                Number of Elements that are replaced.

    @param[in]
    rm_loc      magma_int_t*
                Number of Elements that are replaced by distinct threads.

    @param[in]
    L_new       magma_z_matrix*
                Elements that will be inserted stored in COO format (unsorted).

    @param[in]
    U_new       magma_z_matrix*
                Elements that will be inserted stored in COO format (unsorted).

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_select_candidates_L(
    magma_int_t *num_rm,
    magma_index_t *rm_loc,
    magma_z_matrix *L_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t num_threads=1;
    double element1 = 0;
    magma_int_t count = 0;
    double thrs1 = 1.00; //25;
    magma_int_t el_per_block;
    magma_int_t cand_per_block;
    double bound1;

    magma_index_t *bound=NULL;
    magma_index_t *firstelement=NULL, *lastelement=NULL;
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif

    el_per_block = magma_ceildiv( L_new->num_rows, num_threads );
    cand_per_block = magma_ceildiv( L_new->nnz, num_threads );

    CHECK( magma_index_malloc_cpu( &bound, (num_threads)*(num_threads) ));
    CHECK( magma_index_malloc_cpu( &firstelement, (num_threads)*(num_threads) ));
    CHECK( magma_index_malloc_cpu( &lastelement, (num_threads)*(num_threads) ));

    #pragma omp parallel for
    for( magma_int_t i=0; i<(num_threads)*(num_threads); i++ ){
        bound[i] = 0;
        firstelement[i] = -1;
        lastelement[i] = -1;
    }
    rm_loc[0] = 0;

    //start = magma_sync_wtime( queue );
    info = magma_zparilut_set_thrs_randomselect( (int)(*num_rm)*thrs1, L_new, 1, &element1, queue );
    count = 0;

    bound1 = element1;

    #pragma omp parallel
    {
#ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
        magma_index_t* first_loc;
        magma_index_t* last_loc;
        magma_index_t* count_loc;
        magma_index_malloc_cpu( &first_loc, (num_threads) );
        magma_index_malloc_cpu( &last_loc, (num_threads) );
        magma_index_malloc_cpu( &count_loc, (num_threads) );
        for(int z=0; z<num_threads; z++){
            first_loc[z] = -1;
            last_loc[z] = -1;
            count_loc[z] = 0;
        }
        magma_int_t lbound = id*cand_per_block;
        magma_int_t ubound = min( (id+1)*cand_per_block, L_new->nnz);
        for( magma_int_t i=lbound; i<ubound; i++ ){
            double val = MAGMA_Z_ABS(L_new->val[i]);

            if( val >= bound1 ){
                L_new->list[i] = -5;
                int tid = L_new->rowidx[i]/el_per_block;
                if( first_loc[tid] == -1 ){
                    first_loc[tid] = i;
                    last_loc[tid] = i;
                } else {
                   L_new->list[ last_loc[tid] ] = i;
                   last_loc[tid] = i;
                }
                count_loc[ tid ]++;
            }
        }
        for(int z=0; z<num_threads; z++){
            firstelement[z+(id*num_threads)] = first_loc[z];
            lastelement[z+(id*num_threads)] = last_loc[z];
            bound[ z+(id*num_threads) ] = count_loc[z];
        }
        magma_free_cpu( first_loc );
        magma_free_cpu( last_loc );
        magma_free_cpu( count_loc );
    }
    // count elements
    count = 0;
    #pragma omp parallel for
    for(int j=0; j<num_threads; j++){
        for(int z=1; z<num_threads; z++){
            bound[j] += bound[j+z*num_threads];
        }
    }


    for(int j=0; j<num_threads; j++){
            count = count + bound[j];
            rm_loc[j+1] = count;
            //printf("rm_loc[%d]:%d\n", j,rm_loc[j]);
    }
    *num_rm=count;
    //now combine the linked lists...
    #pragma omp parallel for
    for( magma_int_t i=0; i<num_threads; i++){
        for( magma_int_t j=num_threads-1; j>0; j--){
            if( ( firstelement[ i+(j*num_threads) ] > -1 ) &&
                ( lastelement[ i+((j-1)*num_threads) ] > -1 ) ){   // connect
                    L_new->list[ lastelement[ i+((j-1)*num_threads) ] ]
                        = firstelement[ i+(j*num_threads) ];
                //lastelement[ i+((j-1)*num_threads) ] = lastelement[ i+(j*num_threads) ];
            } else if( firstelement[ i+(j*num_threads) ] > -1 ) {
                firstelement[ i+((j-1)*num_threads) ] = firstelement[ i+(j*num_threads) ];
                lastelement[ i+((j-1)*num_threads) ] = lastelement[ i+(j*num_threads) ];
            }
        }
    }
    // use the rowpointer to start the linked list
    #pragma omp parallel
    for( magma_int_t i=0; i<num_threads; i++){
        L_new->row[i] = firstelement[i];
    }

cleanup:
    magma_free_cpu( bound );
    magma_free_cpu( firstelement );
    magma_free_cpu( lastelement );
    return info;
}

/***************************************************************************//**
    Purpose
    -------
    Screens the new candidates for multiple elements in the same row.
    We allow for at most one new element per row.
    This changes the algorithm, but pays off in performance.


    Arguments
    ---------

    @param[in]
    num_rmL     magma_int_t
                Number of Elements that are replaced.

    @param[in]
    num_rmU     magma_int_t
                Number of Elements that are replaced.

    @param[in]
    rm_loc      magma_int_t*
                Number of Elements that are replaced by distinct threads.

    @param[in]
    L_new       magma_z_matrix*
                Elements that will be inserted stored in COO format (unsorted).

    @param[in]
    U_new       magma_z_matrix*
                Elements that will be inserted stored in COO format (unsorted).

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_select_candidates_U(
    magma_int_t *num_rm,
    magma_index_t *rm_loc,
    magma_z_matrix *L_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t num_threads=1;
    double element1 = 0;
    magma_int_t count = 0;
    double thrs1 = 1.00; //25;
    magma_int_t el_per_block;
    magma_int_t cand_per_block;
    double bound1;

    magma_index_t *bound=NULL;
    magma_index_t *firstelement=NULL, *lastelement=NULL;
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif

    el_per_block = magma_ceildiv( L_new->num_rows, num_threads );
    cand_per_block = magma_ceildiv( L_new->nnz, num_threads );
    //start = magma_sync_wtime( queue );

    CHECK( magma_index_malloc_cpu( &bound, (num_threads)*(num_threads) ));
    CHECK( magma_index_malloc_cpu( &firstelement, (num_threads)*(num_threads) ));
    CHECK( magma_index_malloc_cpu( &lastelement, (num_threads)*(num_threads) ));

    #pragma omp parallel for
    for( magma_int_t i=0; i<(num_threads)*(num_threads); i++ ){
        bound[i] = 0;
        firstelement[i] = -1;
        lastelement[i] = -1;
    }
    rm_loc[0] = 0;


    info = magma_zparilut_set_thrs_randomselect( (int)(*num_rm)*thrs1, L_new, 1, &element1, queue );
    count = 0;

    bound1 = element1;

    #pragma omp parallel
    {
#ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
        magma_index_t* first_loc;
        magma_index_t* last_loc;
        magma_index_t* count_loc;
        magma_index_malloc_cpu( &first_loc, (num_threads) );
        magma_index_malloc_cpu( &last_loc, (num_threads) );
        magma_index_malloc_cpu( &count_loc, (num_threads) );

        for(int z=0; z<num_threads; z++){
            first_loc[z] = -1;
            last_loc[z] = -1;
            count_loc[z] = 0;
        }
        magma_int_t lbound = id*cand_per_block;
        magma_int_t ubound = min( (id+1)*cand_per_block, L_new->nnz);
        for( magma_int_t i=lbound; i<ubound; i++ ){
            double val = MAGMA_Z_ABS(L_new->val[i]);

            if( val >= bound1 ){
                L_new->list[i] = -5;
                int tid = L_new->col[i]/el_per_block;
                //if( tid == 4 && L_new->col[i] == 4 )
                 //   printf("element (%d,%d) at location %d going into block %d\n", L_new->rowidx[i], L_new->col[i], i, tid);
                if( first_loc[tid] == -1 ){
                    first_loc[tid] = i;
                    last_loc[tid] = i;
                } else {
                   L_new->list[ last_loc[tid] ] = i;
                   last_loc[tid] = i;
                }
                count_loc[ tid ]++;
            }
        }
        for(int z=0; z<num_threads; z++){
            firstelement[z+(id*num_threads)] = first_loc[z];
            lastelement[z+(id*num_threads)] = last_loc[z];
            bound[ z+(id*num_threads) ] = count_loc[z];
        }
        magma_free_cpu( first_loc );
        magma_free_cpu( last_loc );
        magma_free_cpu( count_loc );
    }
    // count elements
    count = 0;
    #pragma omp parallel for
    for(int j=0; j<num_threads; j++){
        for(int z=1; z<num_threads; z++){
            bound[j] += bound[j+z*num_threads];
        }
    }


    for(int j=0; j<num_threads; j++){
            count = count + bound[j];
            rm_loc[j+1] = count;
            //printf("rm_loc[%d]:%d\n", j,rm_loc[j]);
    }
    *num_rm=count;

    //now combine the linked lists...
    #pragma omp parallel for
    for( magma_int_t i=0; i<num_threads; i++){
        for( magma_int_t j=num_threads-1; j>0; j--){
            if( ( firstelement[ i+(j*num_threads) ] > -1 ) &&
                ( lastelement[ i+((j-1)*num_threads) ] > -1 ) ){   // connect
                    L_new->list[ lastelement[ i+((j-1)*num_threads) ] ]
                        = firstelement[ i+(j*num_threads) ];
                //lastelement[ i+((j-1)*num_threads) ] = lastelement[ i+(j*num_threads) ];
            } else if( firstelement[ i+(j*num_threads) ] > -1 ) {
                firstelement[ i+((j-1)*num_threads) ] = firstelement[ i+(j*num_threads) ];
                lastelement[ i+((j-1)*num_threads) ] = lastelement[ i+(j*num_threads) ];
            }
        }
    }
    // use the rowpointer to start the linked list
    #pragma omp parallel for
    for( magma_int_t i=0; i<num_threads; i++){
        L_new->row[i] = firstelement[i];
    }

cleanup:
    magma_free_cpu( bound );
    magma_free_cpu( firstelement );
    magma_free_cpu( lastelement );
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This routine approximates the threshold for removing num_rm elements.

    Arguments
    ---------

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.

    @param[in]
    LU          magma_z_matrix*
                Current ILU approximation.

    @param[in]
    order       magma_int_t
                Sort goal function: 0 = smallest, 1 = largest.

    @param[out]
    thrs        magmaDoubleComplex*
                Size of the num_rm-th smallest element.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_set_approx_thrs(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    double *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t sample_size =  8192;
    
    magmaDoubleComplex element;
    magmaDoubleComplex *val=NULL;
    const magma_int_t incy = 1;
    const magma_int_t incx = (int) (LU->nnz)/(sample_size);
    magma_int_t loc_nnz;
    double ratio;
    magma_int_t loc_num_rm;
    magma_int_t num_threads=1;
    magmaDoubleComplex *elements = NULL;
    magma_int_t lsize;
    magma_int_t lnum_rm;
    magmaDoubleComplex *lval;

#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif
     num_threads = 1;
    loc_nnz = (int) LU->nnz/incx;
    ratio = ((double)num_rm)/((double)LU->nnz);
    loc_num_rm = (int) ((double)ratio*(double)loc_nnz);
    CHECK( magma_zmalloc_cpu( &val, loc_nnz ));
    blasf77_zcopy(&loc_nnz, LU->val, &incx, val, &incy );
       
    lsize = loc_nnz/num_threads;
    lnum_rm = loc_num_rm/num_threads;

    lval = val;
    // compare with random select
    if( order == 0 ){
        magma_zselectrandom( lval, lsize, lnum_rm, queue );
        element = (lval[lnum_rm]);
    } else {
         magma_zselectrandom( lval, lsize, lsize-lnum_rm, queue );
        element = (lval[lsize-lnum_rm]);  
    }
               

    *thrs = MAGMA_Z_ABS(element);

cleanup:
    magma_free_cpu( val );
    magma_free_cpu( elements );
    return info;
}



/***************************************************************************//**
    Purpose
    -------
    This routine approximates the threshold for removing num_rm elements.

    Arguments
    ---------

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.

    @param[in]
    LU          magma_z_matrix*
                Current ILU approximation.

    @param[in]
    order       magma_int_t
                Sort goal function: 0 = smallest, 1 = largest.

    @param[out]
    thrs        double*
                Size of the num_rm-th smallest element.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_set_thrs_randomselect(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    double *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t size =  LU->nnz;
    const magma_int_t incx = 1;
    // copy as we may change the elements
    magmaDoubleComplex *val=NULL;
    CHECK( magma_zmalloc_cpu( &val, size ));
    assert( size > num_rm );
    blasf77_zcopy(&size, LU->val, &incx, val, &incx );
    if( order == 0 ){
        magma_zselectrandom( val, size, num_rm, queue );
        *thrs = MAGMA_Z_ABS(val[num_rm]);
    } else {
         magma_zselectrandom( val, size, size-num_rm, queue );
        *thrs = MAGMA_Z_ABS(val[size-num_rm]);  
    }

cleanup:
    magma_free_cpu( val );
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This routine approximates the threshold for removing num_rm elements.
    It takes into account the scaling with the diagonal.

    Arguments
    ---------

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.

    @param[in]
    L           magma_z_matrix*
                Current L approximation.
             
    @param[in]
    U           magma_z_matrix*
                Current U approximation.

    @param[in]
    order       magma_int_t
                Sort goal function: 0 = smallest, 1 = largest.

    @param[out]
    thrs        double*
                Size of the num_rm-th smallest element.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_set_thrs_L_scaled(
    magma_int_t num_rm,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_int_t order,
    double *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t size =  L->nnz;
    const magma_int_t incx = 1;
    // copy as we may change the elements
    magmaDoubleComplex *val=NULL;
    CHECK( magma_zmalloc_cpu( &val, size ));
    assert( size > num_rm );
    blasf77_zcopy(&size, L->val, &incx, val, &incx );
    if( order == 0 ){
        magma_zselectrandom( val, size, num_rm, queue );
        *thrs = MAGMA_Z_ABS(val[num_rm]);
    } else {
         magma_zselectrandom( val, size, size-num_rm, queue );
        *thrs = MAGMA_Z_ABS(val[size-num_rm]);  
    }

cleanup:
    magma_free_cpu( val );
    return info;
}




/***************************************************************************//**
    Purpose
    -------
    This routine approximates the threshold for removing num_rm elements.

    Arguments
    ---------

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.

    @param[in]
    LU          magma_z_matrix*
                Current ILU approximation.

    @param[in]
    order       magma_int_t
                Sort goal function: 0 = smallest, 1 = largest.

    @param[out]
    thrs        double*
                Size of the num_rm-th smallest element.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_set_thrs_randomselect_approx2(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    double *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t size =  LU->nnz;
    const magma_int_t incx = 1;
    magma_int_t incy = 1;
    // magma_int_t nz_copy = LU->nnz;
    // copy as we may change the elements
    magmaDoubleComplex *val=NULL;
    magma_int_t subset = num_rm *10;
    if(LU->nnz <=subset){
        CHECK( magma_zmalloc_cpu( &val, size ));
        assert( size > num_rm );
        blasf77_zcopy(&size, LU->val, &incx, val, &incx );
        if( order == 0 ){
            magma_zselectrandom( val, size, num_rm, queue );
            *thrs = MAGMA_Z_ABS(val[num_rm]);
        } else {
             magma_zselectrandom( val, size, size-num_rm, queue );
            *thrs = MAGMA_Z_ABS(val[size-num_rm]);  
        }
    } else {
        incy = LU->nnz/subset;
        size = subset;
        magma_int_t num_rm_loc = 10;
        assert( size > num_rm_loc );
        CHECK( magma_zmalloc_cpu( &val, size ));
        blasf77_zcopy(&size, LU->val, &incy, val, &incx );
        if( order == 0 ){
            magma_zselectrandom( val, size, num_rm_loc, queue );
            *thrs = MAGMA_Z_ABS(val[num_rm_loc]);
        } else {
             magma_zselectrandom( val, size, size-num_rm_loc, queue );
            *thrs = MAGMA_Z_ABS(val[size-num_rm_loc]);  
        }
    }

cleanup:
    magma_free_cpu( val );
    return info;
}

/***************************************************************************//**
    Purpose
    -------
    This routine approximates the threshold for removing num_rm elements.

    Arguments
    ---------

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.

    @param[in]
    LU          magma_z_matrix*
                Current ILU approximation.

    @param[in]
    order       magma_int_t
                Sort goal function: 0 = smallest, 1 = largest.

    @param[out]
    thrs        double*
                Size of the num_rm-th smallest element.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_set_thrs_randomselect_approx(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    double *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t size =  LU->nnz;
    const magma_int_t incx = 1;
    //magma_int_t incy = 1;
    //magma_int_t nz_copy = LU->nnz;
    magma_int_t num_threads = 1;
    magma_int_t el_per_block;
    //magma_int_t num_rm_loc;
    magmaDoubleComplex *dthrs;
    magmaDoubleComplex *val;
    

    if( LU->nnz <= 680){
       CHECK( magma_zparilut_set_thrs_randomselect(
           num_rm,
           LU,
           order,
           thrs,
           queue ) );
    } else {
        CHECK( magma_zmalloc_cpu( &val, size ));
        blasf77_zcopy(&size, LU->val, &incx, val, &incx );
        assert( size > num_rm );
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif
        num_threads = 272;
        CHECK( magma_zmalloc_cpu( &dthrs, num_threads ));
        
        
        el_per_block = magma_ceildiv( LU->nnz, num_threads );
        
        #pragma omp parallel for
        for( magma_int_t i=0; i<num_threads; i++ ){
            magma_int_t start = min(i*el_per_block, size);
            magma_int_t end = min((i+1)*el_per_block,LU->nnz);
            magma_int_t loc_nz = end-start;
            magma_int_t loc_rm = (int) (num_rm)/num_threads;
            if( i == num_threads-1){
                loc_rm = (int) (loc_nz * num_rm)/size;
            }
            if( loc_nz > loc_rm ){
            if( order == 0 ){
                magma_zselectrandom( val+start, loc_nz, loc_rm, queue );
                dthrs[i] = val[start+loc_rm];
            } else {
                 magma_zselectrandom( val+start, loc_nz, loc_nz-loc_rm, queue );
                dthrs[i] = val[start+loc_nz-loc_rm];  
            }
        }
            
        }
        
        // compute the median
        magma_zselectrandom( dthrs, num_threads, (num_threads+1)/2, queue);
        
        *thrs = MAGMA_Z_ABS(dthrs[(num_threads+1)/2]);
    }
cleanup:
    magma_free_cpu( val );
    magma_free_cpu( dthrs );
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    This routine approximates the threshold for removing num_rm elements.

    Arguments
    ---------

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.

    @param[in]
    LU          magma_z_matrix*
                Current ILU approximation.

    @param[in]
    order       magma_int_t
                Sort goal function: 0 = smallest, 1 = largest.

    @param[out]
    thrs        double*
                Size of the num_rm-th smallest element.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_set_thrs_randomselect_factors(
    magma_int_t num_rm,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_int_t order,
    double *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t size =  L->nnz+U->nnz;
    const magma_int_t incx = 1;
    // copy as we may change the elements
    magmaDoubleComplex *val=NULL;
    CHECK( magma_zmalloc_cpu( &val, size ));
    assert( size > num_rm );
    blasf77_zcopy(&L->nnz, L->val, &incx, val, &incx );
    blasf77_zcopy(&U->nnz, U->val, &incx, val+L->nnz, &incx );
    if( order == 0 ){
        magma_zselectrandom( val, size, num_rm, queue );
        *thrs = MAGMA_Z_ABS(val[num_rm]);
    } else {
         magma_zselectrandom( val, size, size-num_rm, queue );
        *thrs = MAGMA_Z_ABS(val[size-num_rm]);  
    }

cleanup:
    magma_free_cpu( val );
    return info;
}



/***************************************************************************//**
    Purpose
    -------
    This routine provides the exact threshold for removing num_rm elements.

    Arguments
    ---------

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.

    @param[in]
    LU          magma_z_matrix*
                Current ILU approximation.

    @param[in]
    order       magma_int_t
                Sort goal function: 0 = smallest, 1 = largest.

    @param[out]
    thrs        magmaDoubleComplex*
                Size of the num_rm-th smallest element.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_set_exact_thrs(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    magmaDoubleComplex *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magmaDoubleComplex element;
    magmaDoubleComplex *val=NULL;
    const magma_int_t incy = 1;
    const magma_int_t incx = 1;
    magma_int_t loc_nnz;
    double ratio;
    magma_int_t loc_num_rm;
    magma_int_t num_threads=1;
    magmaDoubleComplex *elements = NULL;

#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif
    // two options: either there are enough candidates such that we can use
    // a parallel first step of order-statistics, or not...
    
    ratio = ((double)num_rm)/((double)LU->nnz);
    loc_num_rm = (int) ((double)ratio*(double)loc_nnz);
    loc_num_rm = num_rm;
    CHECK( magma_zmalloc_cpu( &val, loc_nnz ));
    CHECK( magma_zmalloc_cpu( &elements, num_threads ));
    
    blasf77_zcopy(&loc_nnz, LU->val, &incx, val, &incy );
    // first step: every thread sorts a chunk of the array
    // extract the num_rm elements in this chunk
    // this only works if num_rm > loc_nnz / num_threads
    if( loc_nnz / num_threads > loc_num_rm ){
        #pragma omp parallel
        {
    #ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
            if(id<num_threads){
                magma_zorderstatistics(
                    val + id*loc_nnz/num_threads, loc_nnz/num_threads, loc_num_rm, order, &elements[id], queue );
            }
        }
        // Now copy the num_rm left-most elements of every chunk to the beginning of the array.
        for( magma_int_t i=1; i<num_threads; i++){
            blasf77_zcopy(&loc_num_rm, val+i*loc_nnz/num_threads, &incy, val+i*loc_num_rm, &incy );
        }
        // now we only look at the left num_threads*num_rm elements and use order stats
        magma_zorderstatistics(
                    val, num_threads*loc_num_rm, loc_num_rm, order, &element, queue );
    } else {
        magma_zorderstatistics(
                    val, loc_nnz, loc_num_rm, order, &element, queue );
    }

    *thrs = element;

cleanup:
    magma_free_cpu( val );
    magma_free_cpu( elements );
    return info;
}




/***************************************************************************//**
    Purpose
    -------
    This routine provides the exact threshold for removing num_rm elements.

    Arguments
    ---------

    @param[in]
    num_rm      magma_int_t
                Number of Elements that are replaced.

    @param[in]
    LU          magma_z_matrix*
                Current ILU approximation.

    @param[in]
    order       magma_int_t
                Sort goal function: 0 = smallest, 1 = largest.

    @param[out]
    thrs        magmaDoubleComplex*
                Size of the num_rm-th smallest element.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_set_approx_thrs_inc(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    magmaDoubleComplex *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magmaDoubleComplex element;
    magmaDoubleComplex *val=NULL;
    const magma_int_t incy = 1;
    const magma_int_t incx = (int) (LU->nnz)/(1024);
    magma_int_t loc_nnz;
    magma_int_t inc = 100;
    magma_int_t offset = 10;
    double ratio;
    magma_int_t loc_num_rm = num_rm;
    // magma_int_t num_threads=1;
    magmaDoubleComplex *elements = NULL;
    magma_int_t avg_count = 100;
    loc_nnz = (int) LU->nnz/incx;
    ratio = ((double)num_rm)/((double)LU->nnz);
    loc_num_rm = (int) ((double)ratio*(double)loc_nnz);
    
    
    CHECK( magma_zmalloc_cpu( &elements, avg_count ));
    
    CHECK( magma_zmalloc_cpu( &val, loc_nnz ));
    blasf77_zcopy(&loc_nnz, LU->val, &incx, val, &incy );
    for( magma_int_t i=1; i<avg_count; i++){
        magma_zorderstatistics_inc(
                    val + offset*i, loc_nnz - offset*i, loc_num_rm/inc, inc, order, &elements[i], queue );
    }
    
    
    element = MAGMA_Z_ZERO;
    for( int z=0; z < avg_count; z++){
        element = element+MAGMA_Z_MAKE(MAGMA_Z_ABS(elements[z]), 0.0);
    }
    element = element/MAGMA_Z_MAKE((double)avg_count, 0.0);
    
    *thrs = element;
cleanup:
    magma_free_cpu( val );
    magma_free_cpu( elements );
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
    
// cleanup:
    return info;
}

