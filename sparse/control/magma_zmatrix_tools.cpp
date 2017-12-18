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

/***************************************************************************//**
    Purpose
    -------
    Generates a matrix  U = A \cup B. If both matrices have a nonzero value 
    in the same location, the value of A is used.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Input matrix 1.

    @param[in]
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
    magma_queue_t queue)
{
    magma_int_t info = 0;
    assert(A.num_rows == B.num_rows);
    U->num_rows = A.num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_CPU;
    
    CHECK(magma_index_malloc_cpu(&U->row, U->num_rows+1));
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
        magma_int_t add = 0;
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        if(a<enda && b<endb) {
            do{
                acol = A.col[ a ];
                bcol = B.col[ b ];
                
                if(acol == -1) { // stop in case acol = -1
                    a++;
                } 
                else if(bcol == -1) { // stop in case bcol = -1
                    b++;
                }
                else if(acol == bcol) {
                    add++;
                    a++;
                    b++;
                }
                else if(acol<bcol) {
                    add++;
                    a++;
                }
                else {
                    add++;
                    b++;
                }
            }while(a<enda && b<endb);
        }
        // now th rest - if existing
        if(a<enda) {
            do{
                add++;
                a++;
            }while(a<enda);            
        }
        if(b<endb) {
            do{
                add++;
                b++;
            }while(b<endb);            
        }
        U->row[ row+1 ] = add; 
    }
    
    // get the total element count
    U->row[ 0 ] = 0;
    CHECK(magma_zmatrix_createrowptr(U->num_rows, U->row, queue));
    U->nnz = U->row[ U->num_rows ];
        
    CHECK(magma_zmalloc_cpu(&U->val, U->nnz));
    CHECK(magma_index_malloc_cpu(&U->rowidx, U->nnz));
    CHECK(magma_index_malloc_cpu(&U->col, U->nnz));
    #pragma omp parallel for
    for (magma_int_t i=0; i<U->nnz; i++) {
        U->val[i] = MAGMA_Z_ONE;
    }
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
        magma_int_t add = 0;
        magma_int_t offset = U->row[row];
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        if(a<enda && b<endb) {
            do{
                acol = A.col[ a ];
                bcol = B.col[ b ];
                if(acol == -1) { // stop in case acol = -1
                    a++;
                } 
                else if(bcol == -1) { // stop in case bcol = -1
                    b++;
                }
                else if(acol == bcol) {
                    U->col[ offset + add ] = acol;
                    U->rowidx[ offset + add ] = row;
                    U->val[ offset + add ] = A.val[ a ];
                    add++;
                    a++;
                    b++;
                }
                else if(acol<bcol) {
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
            }while(a<enda && b<endb);
        }
        // now th rest - if existing
        if(a<enda) {
            do{
                acol = A.col[ a ];
                U->col[ offset + add ] = acol;
                U->rowidx[ offset + add ] = row;
                U->val[ offset + add ] = A.val[ a ];
                add++;
                a++;
            }while(a<enda);            
        }
        if(b<endb) {
            do{
                bcol = B.col[ b ];
                U->col[ offset + add ] = bcol;
                U->rowidx[ offset + add ] = row;
                U->val[ offset + add ] = B.val[ b ];
                add++;
                b++;
            }while(b<endb);            
        }
    }
cleanup:
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Generates a matrix with entries being in both matrices: U = A \cap B.
    The values in U are all ones.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Input matrix 1.

    @param[in]
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
    magma_queue_t queue)
{
    magma_int_t info = 0;
    //printf("\n"); fflush(stdout);
    // parallel for using openmp
    assert(A.num_rows == B.num_rows);
    U->num_rows = A.num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_CPU;
    
    CHECK(magma_index_malloc_cpu(&U->row, A.num_rows+1));
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
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
            if(acol == bcol) {
                add++;
                a++;
                b++;
            }
            else if(acol<bcol) {
                a++;
            }
            else {
                b++;
            }
        }while(a<enda && b<endb);
        U->row[ row+1 ] = add; 
    }
     
        // new row pointer
    U->row[ 0 ] = 0;
    CHECK(magma_zmatrix_createrowptr(U->num_rows, U->row, queue));
    U->nnz = U->row[ U->num_rows ];
    
    CHECK(magma_zmalloc_cpu(&U->val, U->nnz));
    CHECK(magma_index_malloc_cpu(&U->rowidx, U->nnz));
    CHECK(magma_index_malloc_cpu(&U->col, U->nnz));
    #pragma omp parallel for
    for (magma_int_t i=0; i<U->nnz; i++) {
        U->val[i] = MAGMA_Z_ONE;
    }
    
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
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
            if(acol == bcol) {
                U->col[ offset + add ] = acol;
                U->rowidx[ offset + add ] = row;
                add++;
                a++;
                b++;
            }
            else if(acol<bcol) {
                a++;
            }
            else {
                b++;
            }
        }while(a<enda && b<endb);
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
    magma_queue_t queue)
{
    magma_int_t info = 0;
    //printf("\n"); fflush(stdout);
    // parallel for using openmp
    assert(A.num_rows == B.num_rows);
    U->num_rows = A.num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_CPU;
    
    CHECK(magma_index_malloc_cpu(&U->row, A.num_rows+1));
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
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
            if(acol == bcol) {
                a++;
                b++;
            }
            else if(acol<bcol) {
                add++;
                a++;
            }
            else {
                b++;
            }
        }while(a<enda && b<endb);
        // now th rest - if existing
        if(a<enda) {
            do{
                add++;
                a++;
            }while(a<enda);            
        }
        U->row[ row+1 ] = add; 
    }
        
    // new row pointer
    U->row[ 0 ] = 0;
    CHECK(magma_zmatrix_createrowptr(U->num_rows, U->row, queue));
    U->nnz = U->row[ U->num_rows ];
    
    CHECK(magma_zmalloc_cpu(&U->val, U->nnz));
    CHECK(magma_index_malloc_cpu(&U->rowidx, U->nnz));
    CHECK(magma_index_malloc_cpu(&U->col, U->nnz));
    
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
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
            if(acol == bcol) {
                a++;
                b++;
            }
            else if(acol<bcol) {
                U->col[ offset + add ] = acol;
                U->rowidx[ offset + add ] = row;
                U->val[ offset + add ] = A.val[a];
                add++;
                a++;
            }
            else {
                b++;
            }
        }while(a<enda && b<endb);
        // now th rest - if existing
        if(a<enda) {
            do{
                acol = A.col[ a ];
                U->col[ offset + add ] = acol;
                U->rowidx[ offset + add ] = row;
                U->val[ offset + add ] = A.val[a];
                add++;
                a++;
            }while(a<enda);            
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
    magma_queue_t queue)
{
    magma_int_t info = 0;
    //printf("\n"); fflush(stdout);
    // parallel for using openmp
    assert(A.num_rows == B.num_rows);
    U->num_rows = A.num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_CPU;
    
    CHECK(magma_index_malloc_cpu(&U->row, A.num_rows+1));
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
        magma_int_t add = 0;
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        if(a<enda && b<endb) {
            do{
                acol = A.col[ a ];
                bcol = B.col[ b ];
                if(acol > row) {
                    a = enda;
                    break;    
                }
                if(acol == bcol) {
                    a++;
                    b++;
                }
                else if(acol<bcol) {
                    add++;
                    a++;
                }
                else {
                    b++;
                }
            }while(a<enda && b<endb);
        }
        // now th rest - if existing
        if(a<enda) {
            do{
                acol = A.col[ a ];
                if(acol > row) {
                    a = enda;
                    break;    
                } else {
                    add++;
                    a++;
                }
            }while(a<enda);            
        }
        U->row[ row+1 ] = add; 
    }
        
    // new row pointer
    U->row[ 0 ] = 0;
    CHECK(magma_zmatrix_createrowptr(U->num_rows, U->row, queue));
    U->nnz = U->row[ U->num_rows ];
    
    CHECK(magma_zmalloc_cpu(&U->val, U->nnz));
    CHECK(magma_index_malloc_cpu(&U->rowidx, U->nnz));
    CHECK(magma_index_malloc_cpu(&U->col, U->nnz));
    
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
        magma_int_t add = 0;
        magma_int_t offset = U->row[row];
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        if(a<enda && b<endb) {
            do{
                acol = A.col[ a ];
                bcol = B.col[ b ];
                if(acol > row) {
                    a = enda;
                    break;    
                }
                if(acol == bcol) {
                    a++;
                    b++;
                }
                else if(acol<bcol) {
                    U->col[ offset + add ] = acol;
                    U->rowidx[ offset + add ] = row;
                    U->val[ offset + add ] = A.val[a];
                    add++;
                    a++;
                }
                else {
                    b++;
                }
            }while(a<enda && b<endb);
        }
        // now th rest - if existing
        if(a<enda) {
            do{
                acol = A.col[ a ];
                if(acol > row) {
                    a = enda;
                    break;    
                } else {
                    U->col[ offset + add ] = acol;
                    U->rowidx[ offset + add ] = row;
                    U->val[ offset + add ] = A.val[a];
                    add++;
                    a++;
                }
            }while(a<enda);            
        }
    }
cleanup:
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Generates a matrix with entries being part of triu(A) but not of B. 
    U = triu(A) \ B
    The values of A are preserved.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Element part of this.

    @param[in]
    B           magma_z_matrix
                Not part of this.

    @param[out]
    U           magma_z_matrix*
                .

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
    magma_queue_t queue)
{
    magma_int_t info = 0;
    assert(A.num_rows == B.num_rows);
    U->num_rows = A.num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_CPU;
    
    CHECK(magma_index_malloc_cpu(&U->row, A.num_rows+1));
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
        magma_int_t add = 0;
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        if(a<enda && b<endb) {
            do{
                acol = A.col[ a ];
                bcol = B.col[ b ];
                if(acol == bcol) {
                    a++;
                    b++;
                }
                else if(acol<bcol) {
                    if(acol >= row) {
                        add++;
                        a++;
                    } else {
                        a++;
                    }
                }
                else {
                    b++;
                }
            }while(a<enda && b<endb);
        }
        // now th rest - if existing
        if(a<enda) {
            do{
                acol = A.col[ a ];
                if(acol >= row) {
                    add++;
                    a++;
                } else {
                    a++;
                }
            }while(a<enda);            
        }
        U->row[ row+1 ] = add; 
    }
        
    // new row pointer
    U->row[ 0 ] = 0;
    CHECK(magma_zmatrix_createrowptr(U->num_rows, U->row, queue));
    U->nnz = U->row[ U->num_rows ];
    
    CHECK(magma_zmalloc_cpu(&U->val, U->nnz));
    CHECK(magma_index_malloc_cpu(&U->rowidx, U->nnz));
    CHECK(magma_index_malloc_cpu(&U->col, U->nnz));
    
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
        magma_int_t add = 0;
        magma_int_t offset = U->row[row];
        magma_int_t a = A.row[row];
        magma_int_t b = B.row[row];
        magma_int_t enda = A.row[ row+1 ];
        magma_int_t endb = B.row[ row+1 ]; 
        magma_int_t acol;
        magma_int_t bcol;
        if(a<enda && b<endb) {
            do{
                acol = A.col[ a ];
                bcol = B.col[ b ];
                if(acol == bcol) {
                    a++;
                    b++;
                }
                else if(acol<bcol) {
                    if(acol >= row) {
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
            }while(a<enda && b<endb);
        }
        // now th rest - if existing
        if(a<enda) {
            do{
                acol = A.col[ a ];
                if(acol >= row) {
                    U->col[ offset + add ] = acol;
                    U->rowidx[ offset + add ] = row;
                    U->val[ offset + add ] = A.val[a];
                    add++;
                    a++;
                } else {
                    a++;
                }
            }while(a<enda);            
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
    magma_queue_t queue)
{
    magma_int_t info = 0;
    
    CHECK(magma_index_malloc_cpu(&A->rowidx, A->nnz));
    
    #pragma omp parallel for
    for (magma_int_t row=0; row<A->num_rows; row++) {
        
        for (magma_int_t i=A->row[row]; i<A->row[row+1]; i++) {
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
magma_zcsrcoo_transpose(
    magma_z_matrix A,
    magma_z_matrix *B,
    magma_queue_t queue)
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
    
    CHECK(magma_index_malloc_cpu(&linked_list, A.nnz));
    CHECK(magma_index_malloc_cpu(&row_ptr, A.num_rows+1));
    CHECK(magma_index_malloc_cpu(&last_rowel, A.num_rows+1));
    CHECK(magma_index_malloc_cpu(&B->row, A.num_rows+1));
    CHECK(magma_index_malloc_cpu(&B->rowidx, A.nnz));
    CHECK(magma_index_malloc_cpu(&B->col, A.nnz));
    CHECK(magma_zmalloc_cpu(&B->val, A.nnz));
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif
    
    #pragma omp parallel for
    for (magma_int_t i=0; i<A.num_rows; i++) {
        row_ptr[i] = -1;
    }
    #pragma omp parallel for
    for (magma_int_t i=0; i<A.num_rows+1; i++) {
        B->row[i] = 0;
    }
    
    el_per_block = magma_ceildiv(A.num_rows, num_threads);

    #pragma omp parallel
    {
#ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
        for (magma_int_t i=0; i<A.nnz; i++) {
            magma_index_t row = A.col[ i ];
            if((row < (id+1)*el_per_block) && (row >=(id)*el_per_block)) {
                if(row_ptr[row] == -1) {
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
    magma_zmatrix_createrowptr(B->num_rows, B->row, queue);
    

    assert(B->row[B->num_rows] == A.nnz);
    
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
        magma_int_t el = row_ptr[row];
        if(el>-1) {
            
            for (magma_int_t i=B->row[row]; i<B->row[row+1]; i++) {
                // assert(A.col[el] == row);
                B->val[i] = A.val[el];
                B->col[i] = A.rowidx[el];
                B->rowidx[i] = row;
                el = linked_list[el];
            }
        }
    }
    
cleanup:
    magma_free_cpu(row_ptr);
    magma_free_cpu(last_rowel);
    magma_free_cpu(linked_list);
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
    magma_queue_t queue)
{
    magma_int_t info = 0;
    magma_index_t *offset=NULL;
    
    magma_int_t el_per_block, num_threads;
    magma_int_t loc_offset = 0;
    
#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#else
    num_threads = 1;
#endif
    CHECK(magma_index_malloc_cpu(&offset, num_threads+1));
    el_per_block = magma_ceildiv(n, num_threads);
    
    #pragma omp parallel
    {
#ifdef _OPENMP
    magma_int_t id = omp_get_thread_num();
#else
    magma_int_t id = 0;
#endif
        magma_int_t start = (id)*el_per_block;
        magma_int_t end = min((id+1)*el_per_block, n);
        
        magma_int_t loc_nz = 0;
        for (magma_int_t i=start; i<end; i++) {
            loc_nz = loc_nz + row[i+1];
            row[i+1] = loc_nz;
        }
        offset[id+1] = loc_nz;
    }
    
    for (magma_int_t i=1; i<num_threads; i++) {
        magma_int_t start = (i)*el_per_block;
        magma_int_t end = min((i+1)*el_per_block, n);
        loc_offset = loc_offset + offset[i];
        #pragma omp parallel for
        for (magma_int_t j=start; j<end; j++) {
            row[j+1] = row[j+1]+loc_offset;        
        }
    }
    
cleanup:
    magma_free_cpu(offset);
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
    magma_queue_t queue)
{
    magma_int_t info = 0;
    magma_int_t tmp;
    magma_index_t *index_swap;
    magmaDoubleComplex *val_swap;
    
    assert(A->storage_type == B->storage_type);
    assert(A->memory_location == B->memory_location);
    
    SWAP(A->num_rows, B->num_rows);
    SWAP(A->num_cols, B->num_cols);
    SWAP(A->nnz, B->nnz);
    
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
    Extracts the lower triangular of a matrix: L = tril(A).
    The values of A are preserved.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Element part of this.

    @param[out]
    L           magma_z_matrix*
                Lower triangular part of A.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zmatrix_tril(
    magma_z_matrix A,
    magma_z_matrix *L,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    
    L->num_rows = A.num_rows;
    L->num_cols = A.num_cols;
    L->storage_type = Magma_CSR;
    L->memory_location = Magma_CPU;
    
    CHECK(magma_index_malloc_cpu(&L->row, A.num_rows+1));
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
        magma_int_t nz = 0;
        
        for (magma_int_t i=A.row[row]; i<A.row[row+1]; i++) {
            magma_index_t col = A.col[i];
            if(col <= row) {
                nz++;    
            } else {
                i=A.row[row+1];   
            }
        }
        L->row[row+1] = nz;
    }
    
    // new row pointer
    L->row[ 0 ] = 0;
    CHECK(magma_zmatrix_createrowptr(L->num_rows, L->row, queue));
    L->nnz = L->row[ L->num_rows ];
    
    // allocate memory
    CHECK(magma_zmalloc_cpu(&L->val, L->nnz));
    CHECK(magma_index_malloc_cpu(&L->col, L->nnz));
    
    // copy
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
        magma_int_t nz = 0;
        magma_int_t offset = L->row[row];
        
        for (magma_int_t i=A.row[row]; i<A.row[row+1]; i++) {
            magma_index_t col = A.col[i];
            if(col <= row) {
                L->col[offset+nz] = col;
                L->val[offset+nz] = A.val[i];
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
    Extracts the lower triangular of a matrix: U = triu(A).
    The values of A are preserved.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Element part of this.

    @param[out]
    U           magma_z_matrix*
                Lower triangular part of A.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zmatrix_triu(
    magma_z_matrix A,
    magma_z_matrix *U,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    
    U->num_rows = A.num_rows;
    U->num_cols = A.num_cols;
    U->storage_type = Magma_CSR;
    U->memory_location = Magma_CPU;
    
    CHECK(magma_index_malloc_cpu(&U->row, A.num_rows+1));
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
        magma_int_t nz = 0;
        
        for (magma_int_t i=A.row[row]; i<A.row[row+1]; i++) {
            magma_index_t col = A.col[i];
            if(col >= row) {
                nz++;    
            } else {
                ;    
            }
        }
        U->row[row+1] = nz;
    }
    
    // new row pointer
    U->row[ 0 ] = 0;
    CHECK(magma_zmatrix_createrowptr(U->num_rows, U->row, queue));
    U->nnz = U->row[ U->num_rows ];
    
    
    // allocate memory
    CHECK(magma_zmalloc_cpu(&U->val, U->nnz));
    CHECK(magma_index_malloc_cpu(&U->col, U->nnz));
    
    // copy
    #pragma omp parallel for
    for (magma_int_t row=0; row<A.num_rows; row++) {
        magma_int_t nz = 0;
        magma_int_t offset = U->row[row];
        
        for (magma_int_t i=A.row[row]; i<A.row[row+1]; i++) {
            magma_index_t col = A.col[i];
            if(col >= row) {
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
    Computes the sum of the absolute values in a matrix.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                Element list/matrix.

    @param[out]
    sum         double*
                Sum of the absolute values.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zmatrix_abssum(
    magma_z_matrix A,
    double *sum,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    
    double locsum = .0;
    
    #pragma omp parallel for reduction(+:locsum)
    for (magma_int_t i=0; i < A.nnz; i++) {
        locsum = locsum + (MAGMA_Z_ABS(A.val[i]) * MAGMA_Z_ABS(A.val[i]));
    }
    
    *sum = sqrt(locsum);
    
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    SOrts the elements in a CSR matrix for increasing column index.

    Arguments
    ---------

    @param[in,out]
    A           magma_z_matrix*
                CSR matrix, sorted on output.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zcsr_sort(
    magma_z_matrix *A,
    magma_queue_t queue)
{
    magma_int_t info = 0;
    
    if (A->memory_location == Magma_CPU && A->storage_type == Magma_CSR){
        #pragma omp parallel  
        for (int row=0; row<A->num_rows; row++) {
            magma_zindexsort(&A->col[A->row[row]], 0, 
                A->row[row+1]-A->row[row]-1, queue);
        }
    } else {
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
    
    return info;
}
