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

#define AVOID_DUPLICATES
//#define NANCHECK

/***************************************************************************//**
    Purpose
    -------
    Generates a list of matrix entries being in either matrix. U = A \cup B

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Input matrix 1.

    @param[in,out]
    B           magma_z_matrix
                Input matrix 2.

    @param[out]
    U           magma_z_matrix*
                Not a real matrix, but the list of all matrix entries included 
                in either A or B. No duplicates.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zmatrix_cup(
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //printf("\n"); fflush(stdout);
    // parallel for using openmp
    assert( A.num_rows == B.num_rows );
    U->num_rows = A.num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_CPU;
    
    CHECK( magma_index_malloc_cpu( &U->row, U->num_rows+1 ) );
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t add = 0;
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        if( a<enda && b<endb ){
            do{
                acol = A.col[ a ];
                bcol = B.col[ b ];
                if( acol == bcol ){
                    add++;
                    a++;
                    b++;
                }
                else if( acol<bcol ){
                    add++;
                    a++;
                }
                else {
                    add++;
                    b++;
                }
            }while( a<enda && b<endb );
        }
        // now th rest - if existing
        if( a<enda ){
            do{
                add++;
                a++;
            }while( a<enda );            
        }
        if( b<endb ){
            do{
                add++;
                b++;
            }while( b<endb );            
        }
        U->row[ row+1 ] = add; 
    }
    
    // get the total element count
    U->row[ 0 ] = 0;
    CHECK( magma_zmatrix_createrowptr( U->num_rows, U->row, queue ) );
    U->nnz = U->row[ U->num_rows ];
        
    CHECK( magma_zmalloc_cpu( &U->val, U->nnz ) );
    CHECK( magma_index_malloc_cpu( &U->rowidx, U->nnz ) );
    CHECK( magma_index_malloc_cpu( &U->col, U->nnz ) );
    #pragma omp parallel for
    for( magma_int_t i=0; i<U->nnz; i++){
        U->val[i] = MAGMA_Z_ONE;
    }
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t add = 0;
        magma_int_t offset = U->row[row];
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        if( a<enda && b<endb ){
            do{
                acol = A.col[ a ];
                bcol = B.col[ b ];
                if( acol == bcol ){
                    U->col[ offset + add ] = acol;
                    U->rowidx[ offset + add ] = row;
                    U->val[ offset + add ] = A.val[ a ];
                    add++;
                    a++;
                    b++;
                }
                else if( acol<bcol ){
                    U->col[ offset + add ] = acol;
                    U->rowidx[ offset + add ] = row;
                    U->val[ offset + add ] = A.val[ a ];
                    add++;
                    a++;
                }
                else {
                    U->col[ offset + add ] = bcol;
                    U->rowidx[ offset + add ] = row;
                    U->val[ offset + add ] = B.val[ b ];
                    add++;
                    b++;
                }
            }while( a<enda && b<endb );
        }
        // now th rest - if existing
        if( a<enda ){
            do{
                acol = A.col[ a ];
                U->col[ offset + add ] = acol;
                U->rowidx[ offset + add ] = row;
                U->val[ offset + add ] = A.val[ a ];
                add++;
                a++;
            }while( a<enda );            
        }
        if( b<endb ){
            do{
                bcol = B.col[ b ];
                U->col[ offset + add ] = bcol;
                U->rowidx[ offset + add ] = row;
                U->val[ offset + add ] = B.val[ b ];
                add++;
                b++;
            }while( b<endb );            
        }
    }
cleanup:
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
magma_zparilut_thrsrm(
    magma_int_t order,
    magma_z_matrix *A,
    double *thrs,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_index_t nnz_count;
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
            #pragma unroll
            for( magma_int_t i=A->row[row]; i<A->row[row+1]; i++ ){
                if( MAGMA_Z_ABS(A->val[i]) <= *thrs ){
                    if( A->col[i]!=row ){
                        //printf("remove (%d %d) >> %.4e\n", row, A->col[i], A->val[i]);
                        magma_int_t col = A->col[i];
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
            #pragma unroll
            for( magma_int_t i=A->row[row]; i<A->row[row+1]; i++ ){
                if( MAGMA_Z_ABS(A->val[i]) >= *thrs ){
                    if( A->col[i]!=row ){
                        //printf("remove (%d %d) >> %.4e\n", row, A->col[i], A->val[i]);
                        magma_int_t col = A->col[i];
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
    
    magma_index_t nnz_count;
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
        printf("before:%d\n", A->nnz);
    
    magma_index_t nnz_count;
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
                printf("after:%d\n", A->nnz);
    
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
    
    magma_index_t nnz_count;
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
    
    magma_index_t nnz_count;
    oneA->num_rows = A->num_rows;
    oneA->num_cols = A->num_cols;
    oneA->nnz = A->nnz - A->num_rows;
    oneA->storage_type = Magma_CSR;
    oneA->memory_location = Magma_CPU;
    
    CHECK( magma_zmalloc_cpu( &oneA->val, A->nnz ) );
    
    if( order == 1 ){ // don't copy the last
        #pragma omp parallel for
        for( magma_int_t row=0; row<A->num_rows; row++){
            for( magma_int_t i=A->row[row]+1; i<A->row[row+1]; i++ ){
                oneA->val[ i-row ] = A->val[i];
            }
        }
    } else { // don't copy the fist
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
    Adds to a CSR matrix an array containing the rowindexes.

    Arguments
    ---------

    @param[in,out]
    A           magma_z_matrix*
                Matrix where rowindexes should be added.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zmatrix_addrowindex(
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // magma_free_cpu( &A->rowidx );
    
    CHECK( magma_index_malloc_cpu( &A->rowidx, A->nnz) );
    
    #pragma omp parallel for
    for( magma_int_t row=0; row<A->num_rows; row++){
        #pragma unroll
        for( magma_int_t i=A->row[row]; i<A->row[row+1]; i++ ){
            A->rowidx[i] = row;
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
    
    magma_int_t el_per_block, num_threads;
    
    B->storage_type = A.storage_type;
    B->memory_location = A.memory_location;
    
    B->num_rows = A.num_rows;
    B->num_cols = A.num_cols;
    B->nnz      = A.nnz;
    
    CHECK( magma_index_malloc_cpu( &linked_list, A.nnz ));
    CHECK( magma_index_malloc_cpu( &row_ptr, A.num_rows ));
    CHECK( magma_index_malloc_cpu( &last_rowel, A.num_rows ));
    CHECK( magma_index_malloc_cpu( &B->row, A.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &B->rowidx, A.nnz ));
    CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));
    CHECK( magma_zmalloc_cpu( &B->val, A.nnz ) );
    
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
    
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
        magma_int_t id = omp_get_thread_num();
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
            #pragma unroll
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
    This function generates a rowpointer out of a row-wise element count in 
    parallel.

    Arguments
    ---------

    @param[in]
    n           magma_indnt_t
                row-count.
                        
    @param[in,out]
    row         magma_index_t*
                Input: Vector of size n+1 containing the row-counts 
                        (offset by one).
                Output: Rowpointer.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zmatrix_createrowptr(
    magma_int_t n,
    magma_index_t *row,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_index_t *offset=NULL;//, *new_row;
    
    magma_int_t id, el_per_block, num_threads;
    magma_int_t loc_offset = 0;
    magma_int_t nnz = 0;
    
  //  CHECK( magma_index_malloc_cpu( &new_row, n+1 ));
  //  for( magma_int_t i = 0; i<n+1; i++ ){
  //      new_row[ i ] = row[ i ];
  //  }
    
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
    CHECK( magma_index_malloc_cpu( &offset, num_threads+1 ));
    el_per_block = magma_ceildiv( n, num_threads );
    
    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
        magma_int_t start = (id)*el_per_block;
        magma_int_t end = min((id+1)*el_per_block, n);
        
        magma_int_t loc_nz = 0;
        for(magma_int_t i=start; i<end; i++ ){
            loc_nz = loc_nz + row[i+1];
            row[i+1] = loc_nz;
        }
        offset[id+1] = loc_nz;
    }
    
    for( magma_int_t i=1; i<num_threads; i++ ){
        magma_int_t start = (i)*el_per_block;
        magma_int_t end = min((i+1)*el_per_block, n);
        loc_offset = loc_offset + offset[i];
        #pragma omp parallel for
        for(magma_int_t j=start; j<end; j++ ){
            row[j+1] = row[j+1]+loc_offset;        
        }
    }
    
    
    
   // for debugging
  // nnz = 0;
  // for( magma_int_t i = 0; i<n; i++ ){
  //     nnz=nnz + new_row[ i+1 ];
  //     new_row[ i+1 ] = nnz;
  // }
  // for( magma_int_t i = 0; i<n; i++ ){
  //     if( row[i] != new_row[i] )
  //         printf(" row[%d]: %d <<>> %d\n", i, row[i], new_row[i]);
  //     assert( row[i] == new_row[i] );
  // }
    
cleanup:
    magma_free_cpu( offset );
    //magma_free_cpu( new_row );
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
    
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        B->row[i] = -1;
    }
    
    el_per_block = magma_ceildiv( A.num_rows, num_threads );

    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
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
    Swaps two matrices. Useful if a loop modifies the name of a matrix.

    Arguments
    ---------

    @param[in,out]
    A           magma_z_matrix*
                Matrix to be swapped with B.
                
    @param[in,out]
    B           magma_z_matrix*
                Matrix to be swapped with A.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zmatrix_swap(
    magma_z_matrix *A,
    magma_z_matrix *B,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t tmp;
    magma_index_t *index_swap;
    magmaDoubleComplex *val_swap;
    
    assert( A->storage_type == B->storage_type );
    assert( A->memory_location == B->memory_location );
    
    SWAP( A->num_rows, B->num_rows );
    SWAP( A->num_cols, B->num_cols );
    SWAP( A->nnz, B->nnz );
    
    index_swap = A->row;
    A->row = B->row;
    B->row = index_swap;
    
    index_swap = A->rowidx;
    A->rowidx = B->rowidx;
    B->rowidx = index_swap;
    
    index_swap = A->col;
    A->col = B->col;
    B->col = index_swap;
    
    val_swap = A->val;
    A->val = B->val;
    B->val = val_swap;
    
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Generates a list of matrix entries being in both matrices. U = A \cap B

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Input matrix 1.

    @param[in,out]
    B           magma_z_matrix
                Input matrix 2.

    @param[out]
    U           magma_z_matrix*
                Not a real matrix, but the list of all matrix entries included 
                in both A and B.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zmatrix_cap(
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //printf("\n"); fflush(stdout);
    // parallel for using openmp
    assert( A.num_rows == B.num_rows );
    U->num_rows = A.num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_CPU;
    
    CHECK( magma_index_malloc_cpu( &U->row, A.num_rows+1 ) );
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t add = 0;
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        do{
            acol = A.col[ a ];
            bcol = B.col[ b ];
            if( acol == bcol ){
                add++;
                a++;
                b++;
            }
            else if( acol<bcol ){
                a++;
            }
            else {
                b++;
            }
        }while( a<enda && b<endb );
        U->row[ row+1 ] = add; 
    }
     
        // new row pointer
    U->row[ 0 ] = 0;
    CHECK( magma_zmatrix_createrowptr( U->num_rows, U->row, queue ) );
    U->nnz = U->row[ U->num_rows ];
    
    CHECK( magma_zmalloc_cpu( &U->val, U->nnz ) );
    CHECK( magma_index_malloc_cpu( &U->rowidx, U->nnz ) );
    CHECK( magma_index_malloc_cpu( &U->col, U->nnz ) );
    #pragma omp parallel for
    for( magma_int_t i=0; i<U->nnz; i++){
        U->val[i] = MAGMA_Z_ONE;
    }
    
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t add = 0;
        magma_int_t offset = U->row[row];
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        do{
            acol = A.col[ a ];
            bcol = B.col[ b ];
            if( acol == bcol ){
                U->col[ offset + add ] = acol;
                U->rowidx[ offset + add ] = row;
                add++;
                a++;
                b++;
            }
            else if( acol<bcol ){
                a++;
            }
            else {
                b++;
            }
        }while( a<enda && b<endb );
    }
cleanup:
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Generates a list of matrix entries being part of A but not of B. U = A \ B
    The values of A are preserved.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Element part of this.

    @param[in,out]
    B           magma_z_matrix
                Not part of this.

    @param[out]
    U           magma_z_matrix*
                Not a real matrix, but the list of all matrix entries included 
                in A not in B.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zmatrix_negcap(
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //printf("\n"); fflush(stdout);
    // parallel for using openmp
    assert( A.num_rows == B.num_rows );
    U->num_rows = A.num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_CPU;
    
    CHECK( magma_index_malloc_cpu( &U->row, A.num_rows+1 ) );
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t add = 0;
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        do{
            acol = A.col[ a ];
            bcol = B.col[ b ];
            if( acol == bcol ){
                a++;
                b++;
            }
            else if( acol<bcol ){
                add++;
                a++;
            }
            else {
                b++;
            }
        }while( a<enda && b<endb );
        // now th rest - if existing
        if( a<enda ){
            do{
                add++;
                a++;
            }while( a<enda );            
        }
        U->row[ row+1 ] = add; 
    }
        
    // new row pointer
    U->row[ 0 ] = 0;
    CHECK( magma_zmatrix_createrowptr( U->num_rows, U->row, queue ) );
    U->nnz = U->row[ U->num_rows ];
    
    CHECK( magma_zmalloc_cpu( &U->val, U->nnz ) );
    CHECK( magma_index_malloc_cpu( &U->rowidx, U->nnz ) );
    CHECK( magma_index_malloc_cpu( &U->col, U->nnz ) );
    
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t add = 0;
        magma_int_t offset = U->row[row];
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        do{
            acol = A.col[ a ];
            bcol = B.col[ b ];
            if( acol == bcol ){
                a++;
                b++;
            }
            else if( acol<bcol ){
                U->col[ offset + add ] = acol;
                U->rowidx[ offset + add ] = row;
                U->val[ offset + add ] = A.val[a];
                add++;
                a++;
            }
            else {
                b++;
            }
        }while( a<enda && b<endb );
        // now th rest - if existing
        if( a<enda ){
            do{
                acol = A.col[ a ];
                U->col[ offset + add ] = acol;
                U->rowidx[ offset + add ] = row;
                U->val[ offset + add ] = A.val[a];
                add++;
                a++;
            }while( a<enda );            
        }
    }

cleanup:
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Generates a list of matrix entries being part of tril(A) but not of B. 
    U = tril(A) \ B
    The values of A are preserved.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Element part of this.

    @param[in,out]
    B           magma_z_matrix
                Not part of this.

    @param[out]
    U           magma_z_matrix*
                Not a real matrix, but the list of all matrix entries included 
                in A not in B.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zmatrix_tril_negcap(
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //printf("\n"); fflush(stdout);
    // parallel for using openmp
    assert( A.num_rows == B.num_rows );
    U->num_rows = A.num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_CPU;
    
    CHECK( magma_index_malloc_cpu( &U->row, A.num_rows+1 ) );
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t add = 0;
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        if( a<enda && b<endb ){
            do{
                acol = A.col[ a ];
                bcol = B.col[ b ];
                if( acol > row ){
                    a = enda;
                    break;    
                }
                if( acol == bcol ){
                    a++;
                    b++;
                }
                else if( acol<bcol ){
                    add++;
                    a++;
                }
                else {
                    b++;
                }
            }while( a<enda && b<endb );
        }
        // now th rest - if existing
        if( a<enda ){
            do{
                acol = A.col[ a ];
                if( acol > row ){
                    a = enda;
                    break;    
                } else {
                    add++;
                    a++;
                }
            }while( a<enda );            
        }
        U->row[ row+1 ] = add; 
    }
        
    // new row pointer
    U->row[ 0 ] = 0;
    CHECK( magma_zmatrix_createrowptr( U->num_rows, U->row, queue ) );
    U->nnz = U->row[ U->num_rows ];
    
    CHECK( magma_zmalloc_cpu( &U->val, U->nnz ) );
    CHECK( magma_index_malloc_cpu( &U->rowidx, U->nnz ) );
    CHECK( magma_index_malloc_cpu( &U->col, U->nnz ) );
    
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t add = 0;
        magma_int_t offset = U->row[row];
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        if( a<enda && b<endb ){
            do{
                acol = A.col[ a ];
                bcol = B.col[ b ];
                if( acol > row ){
                    a = enda;
                    break;    
                }
                if( acol == bcol ){
                    a++;
                    b++;
                }
                else if( acol<bcol ){
                    U->col[ offset + add ] = acol;
                    U->rowidx[ offset + add ] = row;
                    U->val[ offset + add ] = A.val[a];
                    add++;
                    a++;
                }
                else {
                    b++;
                }
            }while( a<enda && b<endb );
        }
        // now th rest - if existing
        if( a<enda ){
            do{
                acol = A.col[ a ];
                if( acol > row ){
                    a = enda;
                    break;    
                } else {
                    U->col[ offset + add ] = acol;
                    U->rowidx[ offset + add ] = row;
                    U->val[ offset + add ] = A.val[a];
                    add++;
                    a++;
                }
            }while( a<enda );            
        }
    }
cleanup:
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Generates a list of matrix entries being part of tril(A). 
    U = tril(A)
    The values of A are preserved.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Element part of this.

    @param[in,out]
    B           magma_z_matrix
                Not part of this.

    @param[out]
    U           magma_z_matrix*
                Not a real matrix, but the list of all matrix entries included 
                in A not in B.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zmatrix_tril(
    magma_z_matrix A,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t nnz = 0;
    U->num_rows = A.num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_CPU;
    
    CHECK( magma_index_malloc_cpu( &U->row, A.num_rows+1 ) );
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t nz = 0;
        #pragma unroll
        for( magma_int_t i=A.row[row]; i<A.row[row+1]; i++ ){
            magma_index_t col = A.col[i];
            if( col <= row ){
                nz++;    
            } else {
                i=A.row[row+1];   
            }
        }
        U->row[row+1] = nz;
    }
    
    // new row pointer
    U->row[ 0 ] = 0;
    CHECK( magma_zmatrix_createrowptr( U->num_rows, U->row, queue ) );
    U->nnz = U->row[ U->num_rows ];
    
    // allocate memory
    CHECK( magma_zmalloc_cpu( &U->val, U->nnz ) );
    CHECK( magma_index_malloc_cpu( &U->col, U->nnz ) );
    
    // copy
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t nz = 0;
        magma_int_t offset = U->row[row];
        #pragma unroll
        for( magma_int_t i=A.row[row]; i<A.row[row+1]; i++ ){
            magma_index_t col = A.col[i];
            if( col <= row ){
                U->col[offset+nz] = col;
                U->val[offset+nz] = A.val[i];
                nz++;    
            } else {
                i=A.row[row+1];    
            }
        }
    }
    

cleanup:
    return info;
}



/***************************************************************************//**
    Purpose
    -------
    Generates a list of matrix entries being part of triu(A). 
    U = triu(A)
    The values of A are preserved.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Element part of this.

    @param[in,out]
    B           magma_z_matrix
                Not part of this.

    @param[out]
    U           magma_z_matrix*
                Not a real matrix, but the list of all matrix entries included 
                in A not in B.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zmatrix_triu(
    magma_z_matrix A,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t nnz = 0;
    U->num_rows = A.num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_CPU;
    
    CHECK( magma_index_malloc_cpu( &U->row, A.num_rows+1 ) );
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t nz = 0;
        #pragma unroll
        for( magma_int_t i=A.row[row]; i<A.row[row+1]; i++ ){
            magma_index_t col = A.col[i];
            if( col >= row ){
                nz++;    
            } else {
                ;    
            }
        }
        U->row[row+1] = nz;
    }
    
    // new row pointer
    U->row[ 0 ] = 0;
    CHECK( magma_zmatrix_createrowptr( U->num_rows, U->row, queue ) );
    U->nnz = U->row[ U->num_rows ];
    
    
    // allocate memory
    CHECK( magma_zmalloc_cpu( &U->val, U->nnz ) );
    CHECK( magma_index_malloc_cpu( &U->col, U->nnz ) );
    
    // copy
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t nz = 0;
        magma_int_t offset = U->row[row];
        #pragma unroll
        for( magma_int_t i=A.row[row]; i<A.row[row+1]; i++ ){
            magma_index_t col = A.col[i];
            if( col >= row ){
                U->col[offset+nz] = col;
                U->val[offset+nz] = A.val[i];
                nz++;    
            } else {
                ;    
            }
        }
    }
    
cleanup:
    return info;
}




/***************************************************************************//**
    Purpose
    -------
    Generates a list of matrix entries being part of triu(A) but not of B. 
    U = triu(A) \ B
    The values of A are preserved.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Element part of this.

    @param[in,out]
    B           magma_z_matrix
                Not part of this.

    @param[out]
    U           magma_z_matrix*
                Not a real matrix, but the list of all matrix entries included 
                in A not in B.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zmatrix_triu_negcap(
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //printf("\n"); fflush(stdout);
    // parallel for using openmp
    assert( A.num_rows == B.num_rows );
    U->num_rows = A.num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_CPU;
    
    CHECK( magma_index_malloc_cpu( &U->row, A.num_rows+1 ) );
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t add = 0;
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        if( a<enda && b<endb ){
            do{
                acol = A.col[ a ];
                bcol = B.col[ b ];
                if( acol == bcol ){
                    a++;
                    b++;
                }
                else if( acol<bcol ){
                    if( acol >= row ){
                        add++;
                        a++;
                    } else {
                        a++;
                    }
                }
                else {
                    b++;
                }
            }while( a<enda && b<endb );
        }
        // now th rest - if existing
        if( a<enda ){
            do{
                acol = A.col[ a ];
                if( acol >= row ){
                    add++;
                    a++;
                } else {
                    a++;
                }
            }while( a<enda );            
        }
        U->row[ row+1 ] = add; 
    }
        
    // new row pointer
    U->row[ 0 ] = 0;
    CHECK( magma_zmatrix_createrowptr( U->num_rows, U->row, queue ) );
    U->nnz = U->row[ U->num_rows ];
    
    CHECK( magma_zmalloc_cpu( &U->val, U->nnz ) );
    CHECK( magma_index_malloc_cpu( &U->rowidx, U->nnz ) );
    CHECK( magma_index_malloc_cpu( &U->col, U->nnz ) );
    
    #pragma omp parallel for
    for( magma_int_t row=0; row<A.num_rows; row++){
        magma_int_t add = 0;
        magma_int_t offset = U->row[row];
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        if( a<enda && b<endb ){
            do{
                acol = A.col[ a ];
                bcol = B.col[ b ];
                if( acol == bcol ){
                    a++;
                    b++;
                }
                else if( acol<bcol ){
                    if( acol >= row ){
                        U->col[ offset + add ] = acol;
                        U->rowidx[ offset + add ] = row;
                        U->val[ offset + add ] = A.val[a];
                        add++;
                        a++;
                    } else {
                        a++;
                    }
                }
                else {
                    b++;
                }
            }while( a<enda && b<endb );
        }
        // now th rest - if existing
        if( a<enda ){
            do{
                acol = A.col[ a ];
                if( acol >= row ){
                    U->col[ offset + add ] = acol;
                    U->rowidx[ offset + add ] = row;
                    U->val[ offset + add ] = A.val[a];
                    add++;
                    a++;
                } else {
                    a++;
                }
            }while( a<enda );            
        }
    }

cleanup:
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Computes the sum of the absolute values in this array / matrixlist.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Element list/matrix.

    @param[out]
    U           double*
                Sum of the absolute values.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparilut_elementsum(
    magma_z_matrix A,
    double *sum,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    double locsum = .0;
    
    #pragma omp parallel for reduction(+:locsum)
    for(magma_int_t i=0; i < A.nnz; i++ ){
        locsum = locsum + (MAGMA_Z_ABS(A.val[i]) * MAGMA_Z_ABS(A.val[i]));
    }
    
    *sum = sqrt( locsum );
    
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
magma_zparilut_sweep(
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
        } else if( row == col ){ // end check whether part of L
            L->val[ e ] = MAGMA_Z_ONE; // lower triangular has diagonal equal 1
        }
    }// end omp parallel section

   #pragma omp parallel for
    for( magma_int_t e=0; e<U->nnz; e++){
        {
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
            magma_int_t i,j,icol,jcol,jold;

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
                    jold = j;
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

    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }

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
        magma_int_t id = omp_get_thread_num();
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

    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }

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
        magma_int_t id = omp_get_thread_num();
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
    magma_int_t id = omp_get_thread_num();
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
    magma_int_t id = omp_get_thread_num();
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
magma_zparilut_candidates_linkedlist(
    magma_z_matrix L0,
    magma_z_matrix U0,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix UR,
    magma_z_matrix *L_new,
    magma_z_matrix *U_new,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_index_t *numaddL;
    magma_index_t *numaddU;
    magma_index_t *rowidxL, *rowidxU, *colL, *colU;
    magma_index_t *rowidxLt, *rowidxUt, *colLt, *colUt;

    magma_int_t unsym = 0;
    magma_int_t orig = 1;
    
    // for now: also some part commented out. If it turns out
    // this being correct, I need to clean up the code.

    CHECK( magma_index_malloc_cpu( &numaddL, L.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &numaddU, U.num_rows+1 ));

    CHECK( magma_index_malloc_cpu( &rowidxL, L_new->true_nnz ));
    CHECK( magma_index_malloc_cpu( &rowidxU, U_new->true_nnz ));
    CHECK( magma_index_malloc_cpu( &colL, L_new->true_nnz ));
    CHECK( magma_index_malloc_cpu( &colU, U_new->true_nnz ));

    #pragma omp parallel for
    for( magma_int_t i = 0; i<L.num_rows+1; i++ ){
        numaddL[i] = 0;
        numaddU[i] = 0;
    }
    
    
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
                }
                else if( ilutcol<ilu0col ){
                    ilutcol++;
                }
                else {
                    // this element is missing in the current approximation
                    // mark it as candidate
                    numaddrowL++;
                    ilu0col++;
                }
            }while( ilutcol<endilut && ilu0col<endilu0 );
            numaddL[ row+1 ] = numaddL[ row+1 ]+numaddrowL;
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
                }
                else if( ilutcol<ilu0col ){
                    ilutcol++;
                }
                else {
                    // this element is missing in the current approximation
                    // mark it as candidate
                    numaddrowU++;
                    ilu0col++;
                }
            }while( ilutcol<endilut && ilu0col<endilu0 );
            numaddU[ row+1 ] = numaddU[ row+1 ]+numaddrowU;
        }
    }

    // how to determine candidates:
    // for each node i, look at any "intermediate" neighbor nodes numbered
    // less, and then see if this neighbor has another neighbor j numbered
    // more than the intermediate; if so, fill in is (i,j) if it is not
    // already nonzero
// start = magma_sync_wtime( queue );
    // parallel loop
    /*
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // loop first element over row - only for elements smaller the diagonal
        //for( magma_index_t el1=L.row[row]; el1<L.row[row+1]; el1++ ){
        while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];

            // first check the lower triangular
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;


            // second loop first element over row - only for elements larger the intermediate
            while( L.list[ el2 ] != 0 ) {
            //for( el2=L.row[col1]; el2<L.row[col1+1]; el2++ ){
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row =  row;
                magma_index_t cand_col =  col2;
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkcol;
                magma_index_t checkel=L.row[cand_row];
                do{
                    checkcol = L.col[ checkel ];
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=0;
                    }else{
                        checkel = L.list[checkel];
                    }
                }while( checkel != 0 );
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numaddrowL++;
                    //numaddL[ row+1 ]++;
                }
                el2 = L.list[ el2 ];
            }
            el1 = L.list[ el1 ];
        }
        numaddU[ row+1 ] = numaddrowU;
        numaddL[ row+1 ] = numaddrowL;
    }
*/
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // loop first element over row - only for elements smaller the diagonal
        //for( magma_index_t el1=L.row[row]; el1<L.row[row+1]; el1++ ){
        while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];
            // now check the upper triangular
            magma_index_t start2 = UR.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            do {
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                // first case: part of L
                if( cand_col < row ){
                    // check whether this element already exists
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=L.row[cand_row];
                    do{
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = L.list[checkel];
                        }
                    }while( checkel != 0 );
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        //printf("checked row: %d this element does not yet exist in L: (%d,%d)\n", cand_row, cand_col);
                        numaddrowL++;
                        //numaddL[ row+1 ]++;
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=U.row[cand_col];
                    do{
                        checkcol = U.col[ checkel ];
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = U.list[checkel];
                        }
                    }while( checkel != 0 );
                    if( exist == 0 ){
                        numaddrowU++;
                        //numaddU[ row+1 ]++;
                    }
                }
                el2 = UR.list[ el2 ];
            }while( el2 != 0 );

            el1 = L.list[ el1 ];
        }
        numaddU[ row+1 ] = numaddU[ row+1 ]+numaddrowU;
        numaddL[ row+1 ] = numaddL[ row+1 ]+numaddrowL;
    }

    //#######################
    if( unsym == 1 ){
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = UR.row[ row ];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
                          // printf("row:%d el:%d\n", row, el1);
            // first check the lower triangular
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            while( L.list[ el2 ] != 0 ) {
            //for( el2=L.row[col1]; el2<L.row[col1+1]; el2++ ){
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row =  row;
                magma_index_t cand_col =  col2;
                //printf("check cand (%d,%d)\n",cand_row, cand_col);
                // check whether this element already exists
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=L.row[cand_row];
                    do{
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = L.list[checkel];
                        }
                    }while( checkel != 0 );
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        numaddrowL++;
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=U.row[cand_col];
                    do{
                        checkcol = U.col[ checkel ];
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = U.list[checkel];
                        }
                    }while( checkel != 0 );
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        numaddrowU++;
                    }
                }
                el2 = L.list[ el2 ];
            }
            el1 = UR.list[ el1 ];
        }while( el1 != 0 );

        numaddU[ row+1 ] = numaddU[ row+1 ]+numaddrowU;
        numaddL[ row+1 ] = numaddL[ row+1 ]+numaddrowL;
    }

    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = UR.row[ row ];
        magma_index_t el1 = start1;
        magma_int_t numaddrowL = 0, numaddrowU = 0;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
            // now check the upper triangular
            magma_index_t start2 = UR.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            do {
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
            // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkcol;
                magma_index_t checkel=U.row[cand_col];
                do{
                    checkcol = U.col[ checkel ];
                    if( checkcol == cand_row ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=0;
                    }else{
                        checkel = U.list[checkel];
                    }
                }while( checkel != 0 );
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if( exist == 0 ){
                    numaddrowU++;
                }

                el2 = UR.list[ el2 ];
            }while( el2 != 0 );

            el1 = UR.list[ el1 ];
        }while( el1 != 0 );

        numaddU[ row+1 ] = numaddU[ row+1 ]+numaddrowU;
        numaddL[ row+1 ] = numaddL[ row+1 ]+numaddrowL;
    } //loop over all rows
    } // unsym case

    //end = magma_sync_wtime( queue ); printf("llop 1.2 : %.2e\n", end-start);
    // #########################################################################

    // get the total candidate count
    L_new->nnz = 0;
    U_new->nnz = 0;
    numaddL[ 0 ] = L_new->nnz;
    numaddU[ 0 ] = U_new->nnz;

    #pragma omp parallel for
    for( magma_int_t i = 0; i<L.num_rows+1; i++ ){
            L_new->row[ i ] = 0;
            U_new->row[ i ] = 0;
    }

    L_new->nnz = 0;
    U_new->nnz = 0;
    numaddL[ 0 ] = L_new->nnz;
    numaddU[ 0 ] = U_new->nnz;
    for( magma_int_t i = 0; i<L.num_rows; i++ ){
        L_new->nnz=L_new->nnz + numaddL[ i+1 ];
        U_new->nnz=U_new->nnz + numaddU[ i+1 ];
        numaddL[ i+1 ] = L_new->nnz;
        numaddU[ i+1 ] = U_new->nnz;
    }

    if( L_new->nnz > L_new->true_nnz ){
        magma_free_cpu( L_new->val );
        magma_free_cpu( L_new->row );
        magma_free_cpu( L_new->rowidx );
        magma_free_cpu( L_new->col );
        magma_free_cpu( L_new->list );
        magma_zmalloc_cpu( &L_new->val, L_new->nnz*2 );
        magma_index_malloc_cpu( &L_new->rowidx, L_new->nnz*2 );
        magma_index_malloc_cpu( &L_new->col, L_new->nnz*2 );
        magma_index_malloc_cpu( &L_new->row, L_new->num_rows+1 );
        magma_index_malloc_cpu( &L_new->list, L_new->nnz*2 );
        L_new->true_nnz = L_new->nnz*2;
    }

    if( U_new->nnz > U_new->true_nnz ){
        magma_free_cpu( U_new->val );
        magma_free_cpu( U_new->row );
        magma_free_cpu( U_new->rowidx );
        magma_free_cpu( U_new->col );
        magma_free_cpu( U_new->list );
        magma_zmalloc_cpu( &U_new->val, U_new->nnz*2 );
        magma_index_malloc_cpu( &U_new->rowidx, U_new->nnz*2 );
        magma_index_malloc_cpu( &U_new->col, U_new->nnz*2 );
        magma_index_malloc_cpu( &U_new->row, U_new->num_rows+1 );
        magma_index_malloc_cpu( &U_new->list, U_new->nnz*2 );
        U_new->true_nnz = U_new->nnz*2;
    }
    // #########################################################################

    
    if( orig == 1 ){
        
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        
       #pragma omp parallel for
        for( magma_index_t row=0; row<L0.num_rows; row++){
            magma_int_t offsetL = numaddL[row]+L_new->row[row+1];
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
                }
                else if( ilutcol<ilu0col ){
                    ilutcol++;
                }
                else {
                    // this element is missing in the current approximation
                    // mark it as candidate
                    L_new->col[ offsetL + laddL ] = ilu0col;
                    L_new->rowidx[ offsetL + laddL ] = row;
                    laddL++;
                    ilu0col++;
                }
            }while( ilutcol<endilut && ilu0col<endilu0 );
            L_new->row[row+1] = L_new->row[row+1]+laddL;
        }
        
        // same for U
       #pragma omp parallel for
        for( magma_index_t row=0; row<U0.num_rows; row++){
            magma_int_t offsetU = numaddU[row]+U_new->row[row+1];
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
                }
                else if( ilutcol<ilu0col ){
                    ilutcol++;
                }
                else {
                    // this element is missing in the current approximation
                    // mark it as candidate
                    U_new->col[ offsetU + laddU ] = ilu0col;
                    U_new->rowidx[ offsetU + laddU ] = row;
                    laddU++;
                    ilu0col++;
                }
            }while( ilutcol<endilut && ilu0col<endilu0 );
            U_new->row[row+1] = U_new->row[row+1]+laddU;
        }
    }
    
    
    //start = magma_sync_wtime( queue );
    // insertion here
    // parallel loop
    /*
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = numaddL[row]+L_new->row[row+1];

        magma_index_t el1 = start1;
        // loop first element over row - only for elements smaller the diagonal
        //for( magma_index_t el1=L.row[row]; el1<L.row[row+1]; el1++ ){
        while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];

            // first check the lower triangular
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            while( L.list[ el2 ] != 0 ) {
            //for( el2=L.row[col1]; el2<L.row[col1+1]; el2++ ){
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkcol;
                magma_index_t checkel=L.row[cand_row];
                do{
                    checkcol = L.col[ checkel ];
                    if( checkcol == cand_col ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=0;
                    }else{
                        checkel = L.list[checkel];
                    }
                }while( checkel != 0 );
#ifdef AVOID_DUPLICATES
                    for( checkel=numaddL[cand_row]; checkel<offsetL+laddL; checkel++){
                        checkcol = L_new->col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=offsetL+laddL;
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
                    // L_new->list[ numaddL[row] + laddL ] =  -1;
                    // L_new->row[ numaddL[row] + laddL ] =  -1;
                    laddL++;
                }
                el2 = L.list[ el2 ];
            }
            el1 = L.list[ el1 ];
        }
        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
    }
*/
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = L.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = numaddL[row]+L_new->row[row+1];
        magma_int_t offsetU = numaddU[row]+U_new->row[row+1];
        magma_index_t el1 = start1;
        // loop first element over row - only for elements smaller the diagonal
        //for( magma_index_t el1=L.row[row]; el1<L.row[row+1]; el1++ ){
        while( L.list[el1] != 0 ) {
            magma_index_t col1 = L.col[ el1 ];

            // now check the upper triangular
            magma_index_t start2 = UR.row[ col1 ];
            magma_int_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            do {
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                // first case: part of L
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=L.row[cand_row];
                    do{
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = L.list[checkel];
                        }
                    }while( checkel != 0 );
#ifdef AVOID_DUPLICATES
                        for( checkel=numaddL[cand_row]; checkel<offsetL+laddL; checkel++){
                            checkcol = L_new->col[ checkel ];
                            if( checkcol == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetL+laddL;
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
                        // L_new->list[ numaddL[row] + laddL ] = -1;
                        // L_new->row[ numaddL[row] + laddL ] = -1;
                        laddL++;
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=U.row[cand_col];
                    do{
                        checkcol = U.col[ checkel ];
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = U.list[checkel];
                        }
                    }while( checkel != 0 );
#ifdef AVOID_DUPLICATES
                        for( checkel=numaddU[cand_row]; checkel<offsetU+laddU; checkel++){
                            checkcol = U_new->col[ checkel ];
                            if( checkcol == cand_row ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetU+laddU;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        //  printf("---------------->>>  candidate in U at (%d, %d) stored in %d\n", cand_row, cand_col, numaddU[row] + laddU);
                        //add in the next location for this row
                        // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        U_new->col[ offsetU + laddU ] = cand_col;
                        U_new->rowidx[ offsetU + laddU ] = cand_row;
                        // U_new->list[ numaddU[row] + laddU ] = -1;
                        // U_new->row[ numaddU[row] + laddU ] = -1;
                        laddU++;
                        //if( cand_row > cand_col )
                         //   printf("inserted illegal candidate in case 4: (%d,%d) coming from (%d,%d)->(%d,%d)\n", cand_row, cand_col, row, col1, U.col[U.row[ cand_col ]], UR.col[ el2 ]);
                    }
                }
                el2 = UR.list[ el2 ];
            }while( el2 != 0 );

            el1 = L.list[ el1 ];
        }
        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
    }


    //#######################
    if( unsym == 1 ){
    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = UR.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetL = numaddL[row]+L_new->row[row+1];
        magma_int_t offsetU = numaddU[row]+U_new->row[row+1];
        magma_index_t el1 = start1;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];

          // printf("row:%d el:%d\n", row, el1);
            // first check the lower triangular
            magma_index_t start2 = L.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            while( L.list[ el2 ] != 0 ) {
            //for( el2=L.row[col1]; el2<L.row[col1+1]; el2++ ){
                magma_index_t col2 = L.col[ el2 ];
                magma_index_t cand_row =  row;
                magma_index_t cand_col =  col2;
                // check whether this element already exists
                if( cand_col < row ){
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=L.row[cand_row];
                    do{
                        checkcol = L.col[ checkel ];
                        if( checkcol == cand_col ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = L.list[checkel];
                        }
                    }while( checkel != 0 );
#ifdef AVOID_DUPLICATES
                        for( checkel=numaddL[cand_row]; checkel<offsetL+laddL; checkel++){
                            checkcol = L_new->col[ checkel ];
                            if( checkcol == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetL+laddL;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        L_new->rowidx[ offsetL + laddL ] = cand_row;
                        L_new->col[ offsetL + laddL ] = cand_col;
                        // L_new->list[ numaddL[row] + laddL ] =  -1;
                        // L_new->row[ numaddL[row] + laddL ] =  -1;
                        laddL++;
                    }
                } else {
                    magma_int_t exist = 0;
                    magma_index_t checkcol;
                    magma_index_t checkel=U.row[cand_col];
                    do{
                        checkcol = U.col[ checkel ];
                        if( checkcol == cand_row ){
                            // element included in LU and nonzero
                            exist = 1;
                            checkel=0;
                        }else{
                            checkel = U.list[checkel];
                        }
                    }while( checkel != 0 );
#ifdef AVOID_DUPLICATES
                        for( checkel=numaddU[cand_row]; checkel<offsetU+laddU; checkel++){
                            checkcol = U_new->col[ checkel ];
                            if( checkcol == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                checkel=offsetU+laddU;
                            }
                        }
#endif
                    // if it does not exist, increase counter for this location
                    // use the entry one further down to allow for parallel insertion
                    if( exist == 0 ){
                        // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                        U_new->col[ offsetU + laddU ] = cand_col;
                        U_new->rowidx[ offsetU + laddU ] = cand_row;
                        // U_new->list[ numaddU[row] + laddU ] = -1;
                        // U_new->row[ numaddU[row] + laddU ] = -1;
                        laddU++;
                    }
                }
                el2 = L.list[ el2 ];
            }
            el1 = UR.list[ el1 ];
        }while( el1 != 0 );

        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
    }

    #pragma omp parallel for
    for( magma_index_t row=0; row<L.num_rows; row++){
        magma_index_t start1 = UR.row[ row ];
        magma_int_t laddL = 0;
        magma_int_t laddU = 0;
        magma_int_t offsetU = numaddU[row]+U_new->row[row+1];
        magma_index_t el1 = start1;
        // printf("start here\n");
        // loop first element over row - only for elements smaller the diagonal
        do{
            magma_index_t col1 = UR.col[ el1 ];
            // now check the upper triangular
            magma_index_t start2 = UR.row[ col1 ];
            magma_index_t el2 = start2;
            // second loop first element over row - only for elements larger the intermediate
            do{
                magma_index_t col2 = UR.col[ el2 ];
                magma_index_t cand_row = row;
                magma_index_t cand_col = col2;
                // check whether this element already exists
                magma_int_t exist = 0;
                magma_index_t checkcol;
                magma_index_t checkel=U.row[cand_col];
                do{
                    checkcol = U.col[ checkel ];
                    if( checkcol == cand_row ){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=0;
                    }else{
                        checkel = U.list[checkel];
                    }
                }while( checkel != 0 );
#ifdef AVOID_DUPLICATES
                for (checkel=numaddU[cand_row]; checkel<offsetU+laddU; checkel++) {
                    checkcol = U_new->col[ checkel ];
                    if (checkcol == cand_col){
                        // element included in LU and nonzero
                        exist = 1;
                        checkel=offsetU+laddU;
                    }
                }
#endif
                // if it does not exist, increase counter for this location
                // use the entry one further down to allow for parallel insertion
                if (exist == 0) {
                    // U_new->val[ numaddU[row] + laddU ] =  MAGMA_Z_MAKE(1e-14,0.0);
                    U_new->col[ offsetU + laddU ] = cand_col;
                    U_new->rowidx[ offsetU + laddU ] = cand_row;
                    // U_new->list[ numaddU[row] + laddU ] = -1;
                    // U_new->row[ numaddU[row] + laddU ] = -1;
                    laddU++;
                    //if( cand_row == 118 && cand_col == 163 )
                    //         printf("checked row insetion %d this element does not yet exist in U: (%d,%d) row starts with (%d,%d)\n", cand_row, cand_row, cand_col, U.col[ checkel ], cand_col );
                    //if( cand_row > cand_col )
                      //  printf("inserted illegal candidate in case 4: (%d,%d) coming from (%d,%d)->(%d,%d)\n", cand_row, cand_col, row, col1, U.col[U.row[ cand_col ]], UR.col[ el2 ]);
                }

                el2 = UR.list[ el2 ];
            } while (el2 != 0);

            el1 = UR.list[ el1 ];
        }while( el1 != 0 );

        L_new->row[row+1] = L_new->row[row+1]+laddL;
        U_new->row[row+1] = U_new->row[row+1]+laddU;
    } //loop over all rows
    } // unsym case

#ifdef AVOID_DUPLICATES
        // #####################################################################

        // get the total candidate count
        L_new->nnz = 0;
        U_new->nnz = 0;

        for( magma_int_t i = 0; i<L.num_rows; i++ ){
            L_new->nnz=L_new->nnz + L_new->row[ i+1 ];
            U_new->nnz=U_new->nnz + U_new->row[ i+1 ];
            L_new->row[ i+1 ] = L_new->nnz;
            U_new->row[ i+1 ] = U_new->nnz;
        }

        #pragma omp parallel for
        for( magma_int_t i = 0; i<L.num_rows; i++ ){
            for( magma_int_t j=L_new->row[ i ]; j<L_new->row[ i+1 ]; j++ ){
                magma_int_t k = j-L_new->row[ i ]+numaddL[ i ];
                //printf("j:%d, numaddL[ i+1 ]=%d\n", j, numaddL[ i+1 ]);
                colL[ j ]    =  L_new->col[ k ];
                rowidxL[ j ] =  L_new->rowidx[ k ];
            }
        }
        #pragma omp parallel for
        for( magma_int_t i = 0; i<L.num_rows; i++ ){
            for( magma_int_t j=U_new->row[ i ]; j<U_new->row[ i+1 ]; j++ ){
                magma_int_t k = j-U_new->row[ i ]+numaddU[ i ];
                colU[ j ]    =  U_new->col[ k ];
                rowidxU[ j ] =  U_new->rowidx[ k ];
            }
        }

        rowidxLt = L_new->rowidx;
        colLt = L_new->col;
        rowidxUt = U_new->rowidx;
        colUt = U_new->col;

        L_new->rowidx = rowidxL;
        L_new->col = colL;
        U_new->rowidx = rowidxU;
        U_new->col = colU;
        
        L_new->row[L_new->num_rows] = L_new->nnz;
        U_new->row[U_new->num_rows] = U_new->nnz;

        rowidxL = rowidxLt;
        colL = colLt;
        rowidxU = rowidxUt;
        colU = colUt;
        //end = magma_sync_wtime( queue ); printf("llop 1.2 : %.2e\n", end-start);
        // #####################################################################
#endif

cleanup:
    magma_free_cpu( numaddL );
    magma_free_cpu( numaddU );
    magma_free_cpu( rowidxL );
    magma_free_cpu( rowidxU );
    magma_free_cpu( colL );
    magma_free_cpu( colU );
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
    
    magma_int_t id, num_threads;
    
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
    
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
        magma_int_t id = omp_get_thread_num();
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

    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
    el_per_block = magma_ceildiv( LU->num_rows, num_threads );

    #pragma omp parallel for
    for( magma_int_t i=0; i<num_threads; i++ ){
        rm_loc[i] = 0;
    }

    #pragma omp parallel
    {
        magma_int_t id = omp_get_thread_num();
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
    double element1;
    magma_int_t count = 0;
    double thrs1 = 1.00; //25;
    magma_int_t el_per_block;
    magma_int_t cand_per_block;
    double bound1;

    magma_index_t *bound=NULL;
    magma_index_t *firstelement=NULL, *lastelement=NULL;
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }

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
        magma_int_t id = omp_get_thread_num();
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
    double element1;
    magma_int_t count = 0;
    double thrs1 = 1.00; //25;
    magma_int_t el_per_block;
    magma_int_t cand_per_block;
    double bound1;

    magma_index_t *bound=NULL;
    magma_index_t *firstelement=NULL, *lastelement=NULL;
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }

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
        magma_int_t id = omp_get_thread_num();
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

    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
     num_threads = 1;
/*
    //printf("largest element considered: %d\n", LU->nnz);
    if( LU->nnz < sample_size ){
        loc_nnz = LU->nnz;
        ratio = ((double)num_rm)/((double)LU->nnz);
        loc_num_rm = (int) ((double)ratio*(double)loc_nnz);
        CHECK( magma_zmalloc_cpu( &val, loc_nnz ));
        CHECK( magma_zmalloc_cpu( &elements, num_threads ));
        blasf77_zcopy(&loc_nnz, LU->val, &incy, val, &incy );
        {
            #pragma omp parallel
            {
                magma_int_t id = omp_get_thread_num();
                if(id<num_threads){
                   // magma_zorderstatistics(
                   //     val + id*loc_nnz/num_threads, loc_nnz/num_threads, loc_num_rm/num_threads, order, &elements[id], queue );
//printf("\n%% >>> trying random select instead <<<  %%\n");
    {
        magma_int_t lsize = loc_nnz/num_threads;
        magma_int_t lnum_rm = loc_num_rm/num_threads;
        magmaDoubleComplex *lval;
        lval = val + id*loc_nnz/num_threads;
                    // compare with random select
                    if( order == 0 ){
                        magma_zselectrandom( lval, lsize, lnum_rm, queue );
                        elements[id] = (lval[lnum_rm]);
                    } else {
                         magma_zselectrandom( lval, lsize, lsize-lnum_rm, queue );
                        elements[id] = (lval[lsize-lnum_rm]);  
                    }
    }
                }
            }
            element = MAGMA_Z_ZERO;
            for( int z=0; z < num_threads; z++){
                element = element+MAGMA_Z_MAKE(MAGMA_Z_ABS(elements[z]), 0.0);
            }
            element = element/MAGMA_Z_MAKE((double)num_threads, 0.0);
        }
    } 
    else
    */
    {
        loc_nnz = (int) LU->nnz/incx;
        ratio = ((double)num_rm)/((double)LU->nnz);
        loc_num_rm = (int) ((double)ratio*(double)loc_nnz);
     //   printf("loc_nnz:%d ratio:%d/%d = %.2e loc_num_rm:%d\n", loc_nnz, num_rm, LU->nnz, ratio, loc_num_rm);
        CHECK( magma_zmalloc_cpu( &val, loc_nnz ));
        blasf77_zcopy(&loc_nnz, LU->val, &incx, val, &incy );
        {
            /*
            if(num_threads > 1 ){
                CHECK( magma_zmalloc_cpu( &elements, num_threads ));
                #pragma omp parallel
                {
                    magma_int_t id = omp_get_thread_num();
                    if(id<num_threads){
                        //printf("\n%% >>> trying random select instead <<<  %%\n");
    {
        magma_int_t lsize = loc_nnz/num_threads;
        magma_int_t lnum_rm = loc_num_rm/num_threads;
        magmaDoubleComplex *lval;
        lval = val + id*loc_nnz/num_threads;
                    // compare with random select
                    if( order == 0 ){
                        magma_zselectrandom( lval, lsize, lnum_rm, queue );
                        elements[id] = (lval[lnum_rm]);
                    } else {
                         magma_zselectrandom( lval, lsize, lsize-lnum_rm, queue );
                        elements[id] = (lval[lsize-lnum_rm]);  
                    }
    }
                        
                    //    magma_zorderstatistics(
                      //      val + id*loc_nnz/num_threads, loc_nnz/num_threads, loc_num_rm/num_threads, order, &elements[id], queue );
                    }
                }
                element = MAGMA_Z_ZERO;
                for( int z=0; z < num_threads; z++){
                    element = element+MAGMA_Z_MAKE(MAGMA_Z_ABS(elements[z]), 0.0);
                }
                element = element/MAGMA_Z_MAKE((double)num_threads, 0.0);
            } 
            else 
            */
            {
               // magma_zorderstatistics(
               //     val, loc_nnz, loc_num_rm, order, &element, queue );
// printf("\n >>> trying random select instead <<<\n");
    {
        magma_int_t lsize = loc_nnz/num_threads;
        magma_int_t lnum_rm = loc_num_rm/num_threads;
        magmaDoubleComplex *lval;
        lval = val;
                    // compare with random select
                    if( order == 0 ){
                        magma_zselectrandom( lval, lsize, lnum_rm, queue );
                        element = (lval[lnum_rm]);
                    } else {
                         magma_zselectrandom( lval, lsize, lsize-lnum_rm, queue );
                        element = (lval[lsize-lnum_rm]);  
                    }
    }
               
            }
        }
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
    magma_int_t nz_copy = LU->nnz;
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
    magma_int_t incy = 1;
    magma_int_t nz_copy = LU->nnz;
    magma_int_t num_threads = 1;
    magma_int_t el_per_block;
    magma_int_t num_rm_loc;
    magmaDoubleComplex *dthrs=NULL;
    magmaDoubleComplex *val=NULL;
    

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
        #pragma omp parallel
        {
            num_threads = omp_get_max_threads();
        }
        num_threads = 680;
        CHECK( magma_zmalloc_cpu( &dthrs, num_threads ));
        
        
        el_per_block = magma_ceildiv( LU->nnz, num_threads );
        
        #pragma omp parallel for
        for( magma_int_t i=0; i<num_threads; i++ ){
            magma_int_t start = min((i)*el_per_block,LU->nnz);
            magma_int_t end = min((i+1)*el_per_block,LU->nnz);
            magma_int_t loc_nz = end-start;
            magma_int_t loc_rm = (int) (num_rm)/num_threads;
            if( i == num_threads-1){
                loc_rm = (int) (loc_nz * num_rm)/size;
            }
            if( loc_nz > loc_rm ){
                //printf(" problem %i: %d < %dL: start:%d end:%d\n", i, loc_nz, loc_rm, start, end );
                // assert( loc_nz > loc_rm );
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

    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
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
            magma_int_t id = omp_get_thread_num();
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
    magma_int_t num_threads=1;
    magmaDoubleComplex *elements = NULL;
    magma_int_t avg_count = 100;
    loc_nnz = (int) LU->nnz/incx;
    ratio = ((double)num_rm)/((double)LU->nnz);
    loc_num_rm = (int) ((double)ratio*(double)loc_nnz);
    
printf("start:%d", loc_nnz);
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
    // two options: either there are enough candidates such that we can use
    // a parallel first step of order-statistics, or not...
    
    
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
printf("done: %.4e\n", element);
cleanup:
    magma_free_cpu( val );
    magma_free_cpu( elements );
    return info;
}



#endif  // _OPENMP
