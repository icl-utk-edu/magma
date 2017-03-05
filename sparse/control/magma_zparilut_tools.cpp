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

#define AVOID_DUPLICATES
//#define NANCHECK

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
                    L_new->val[ e ] =  ( A_e - sum ) / U.val[jold];
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
                    //printf("%% tried to insert duplicate case 1 2 in L thread %d: (%d %d) \n", id, new_row, new_col); fflush(stdout);
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
                    //printf("%% tried to insert duplicate case 1 2 in U thread %d: (%d %d) \n", id, new_row, new_col);
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

    magma_int_t unsym = 1;
    
    // for now: also some part commented out. If it turns out
    // this being correct, I need to clean up the code.
    unsym = 0;

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
    }

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
        magma_index_malloc_cpu( &L_new->row, L_new->nnz*2 );
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
        magma_index_malloc_cpu( &U_new->row, U_new->nnz*2 );
        magma_index_malloc_cpu( &U_new->list, U_new->nnz*2 );
        U_new->true_nnz = U_new->nnz*2;
    }
    // #########################################################################

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
    }

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
    magmaDoubleComplex *thrs,
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

    double bound = MAGMA_Z_ABS(*thrs);

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
    magmaDoubleComplex element1;
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
    info = magma_zparilut_set_approx_thrs( (int)(*num_rm)*thrs1, L_new, 1, &element1, queue );
    count = 0;

    bound1 = MAGMA_Z_ABS(element1);

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
    magmaDoubleComplex element1;
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


    info = magma_zparilut_set_approx_thrs( (int)(*num_rm)*thrs1, L_new, 1, &element1, queue );
    count = 0;

    bound1 = MAGMA_Z_ABS(element1);

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
    magmaDoubleComplex *thrs,
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
    // num_threads = 1;

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
                    magma_zorderstatistics(
                        val + id*loc_nnz/num_threads, loc_nnz/num_threads, loc_num_rm/num_threads, order, &elements[id], queue );
                }
            }
            element = MAGMA_Z_ZERO;
            for( int z=0; z < num_threads; z++){
                element = element+MAGMA_Z_MAKE(MAGMA_Z_ABS(elements[z]), 0.0);
            }
            element = element/MAGMA_Z_MAKE((double)num_threads, 0.0);
        }
    } else {
        loc_nnz = (int) LU->nnz/incx;
        ratio = ((double)num_rm)/((double)LU->nnz);
        loc_num_rm = (int) ((double)ratio*(double)loc_nnz);
     //   printf("loc_nnz:%d ratio:%d/%d = %.2e loc_num_rm:%d\n", loc_nnz, num_rm, LU->nnz, ratio, loc_num_rm);
        CHECK( magma_zmalloc_cpu( &val, loc_nnz ));
        blasf77_zcopy(&loc_nnz, LU->val, &incx, val, &incy );
        {
            if(num_threads > 1 ){
                CHECK( magma_zmalloc_cpu( &elements, num_threads ));
                #pragma omp parallel
                {
                    magma_int_t id = omp_get_thread_num();
                    if(id<num_threads){
                        magma_zorderstatistics(
                            val + id*loc_nnz/num_threads, loc_nnz/num_threads, loc_num_rm/num_threads, order, &elements[id], queue );
                    }
                }
                element = MAGMA_Z_ZERO;
                for( int z=0; z < num_threads; z++){
                    element = element+MAGMA_Z_MAKE(MAGMA_Z_ABS(elements[z]), 0.0);
                }
                element = element/MAGMA_Z_MAKE((double)num_threads, 0.0);
            } else {
                magma_zorderstatistics(
                    val, loc_nnz, loc_num_rm, order, &element, queue );
            }
        }
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
magma_zparilut_set_thrs_randomselect(
    magma_int_t num_rm,
    magma_z_matrix *LU,
    magma_int_t order,
    magmaDoubleComplex *thrs,
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
        *thrs = val[num_rm];
    } else {
         magma_zselectrandom( val, size, size-num_rm, queue );
        *thrs = val[size-num_rm];  
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
