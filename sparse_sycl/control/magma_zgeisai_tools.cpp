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

#define WARP_SIZE 32


/***************************************************************************//**
    Purpose
    -------
    Takes a sparse matrix and generates
        * an array containing the sizes of the different systems
        * an array containing the indices with the locations in the sparse
          matrix where the data comes from and goes back to
        * an array containing all the sparse triangular systems
            - padded with zeros to size 32x32
        * an array containing the RHS

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

    @param[in]
    L           magma_z_matrix
                Matrix in CSR format

    @param[in]
    LC          magma_z_matrix
                same matrix, also CSR, but col-major

    @param[in,out]
    sizes       magma_int_t*
                Number of Elements that are replaced.

    @param[in,out]
    locations   magma_int_t*
                Array indicating the locations.

    @param[in,out]
    trisystems  magmaDoubleComplex*
                trisystems

    @param[in,out]
    rhs         magmaDoubleComplex*
                right-hand sides

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmprepare_batched(
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
    magma_int_t info = 0;

    magma_int_t warpsize = WARP_SIZE;

    #pragma omp parallel for
    for( magma_int_t i=0; i<L.num_rows*warpsize*warpsize; i++ ){
        trisystems[i] = MAGMA_Z_ZERO;
    }
    #pragma omp parallel for
    for( magma_int_t i=0; i<L.num_rows*warpsize; i++ ){
        rhs[i] = MAGMA_Z_ZERO;
        locations[i] = 0;
    }

    if( uplotype == MagmaLower ){
        // fill sizes and rhs and first column of trisystems
        #pragma omp parallel for
        for( magma_int_t i=0; i<L.num_rows; i++ ){
            magma_int_t size = 0;
            for( magma_int_t j=LC.row[i]; j<LC.row[i+1]; j++ ){
                locations[ i*warpsize + size ] = LC.col[j];
                //trisystems[ i*warpsize*warpsize + size ] = LC.val[j];
                size++;
            }
            sizes[ i ] = size;
            rhs[ i*warpsize ] = MAGMA_Z_ONE;
        }
    } else {
        // fill sizes and rhs and first column of trisystems
        #pragma omp parallel for
        for( magma_int_t i=0; i<L.num_rows; i++ ){
            magma_int_t size = 0;
            for( magma_int_t j=LC.row[i]; j<LC.row[i+1]; j++ ){
                locations[ i*warpsize + size ] = LC.col[j];
                //trisystems[ i*warpsize*warpsize + size ] = LC.val[j];
                size++;
            }
            sizes[ i ] = size;
            rhs[ i*(warpsize)+sizes[i]-1 ] = MAGMA_Z_ONE;
        }
    }

    // fill rest of trisystems
    #pragma omp parallel for
    for( magma_int_t i=0; i<L.num_rows; i++ ){
        for( magma_int_t j=0; j<sizes[ i ]; j++ ){// no need for first
            magma_int_t k = L.row[ locations[ j+i*warpsize ] ];
            magma_int_t l = i*warpsize;
            magma_int_t idx = 0;
            while( k < L.row[ locations[ j+i*warpsize ]+1 ] && l < (i+1)*warpsize ){ // stop once this column is done
                // printf("k:%d<%d l:%d<%d\n",k,L.row[ locations[ j ]+1 ],l,(i+1)*warpsize );

                if( locations[ l ] == L.col[k] ){ //match
                    // printf("match: %d = %d insert %.2f at %d\n",locations[ l ], L.col[k], L.val[ k ], loc);
                    trisystems[ i*warpsize*warpsize + j*warpsize + idx ]
                                                            = L.val[ k ];
                    k++;
                    l++;
                    idx++;
                } else if( L.col[k] < locations[ l ] ){// need to check next element
                    // printf("increment k\n");
                    k++;
                } else { // element does not exist, i.e. l < L.col[k]
                    // printf("increment l\n");
                    l++; // check next elment in the sparsity pattern
                    idx++; // leave this element equal zero
                }
            }
        }
    }

    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Does all triangular solves

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

    @param[in]
    L           magma_z_matrix
                Matrix in CSR format

    @param[in]
    LC          magma_z_matrix
                same matrix, also CSR, but col-major

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
magma_zmtrisolve_batched(
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
    magma_int_t info = 0;

    magma_int_t warpsize = WARP_SIZE;
    magma_int_t ione     = 1;

    #pragma omp parallel for
    for(magma_int_t i=0; i<L.num_rows; i++){
        blasf77_ztrsv( lapack_uplo_const(uplotype),
                        lapack_trans_const(transtype),
                        lapack_diag_const(diagtype),
                           (magma_int_t*)&sizes[i],
                           &trisystems[i*warpsize*warpsize], &warpsize,
                           &rhs[i*warpsize], &ione );
    }

    return info;
}


/***************************************************************************//**
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
magma_zmbackinsert_batched(
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

    magma_int_t warpsize = WARP_SIZE;

    #pragma omp parallel for
    for(magma_int_t i=0; i<M->num_rows; i++){
        for(magma_int_t j=0; j<sizes[i]; j++){
            M->val[M->row[i]+j] = rhs[i*warpsize+j];
        }
    }

    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Checks for a matrix whether the batched ISAI works for a given
    thread-block size

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                system matrix

    @param[in]
    batchsize   magma_int_t
                Size of the batch (GPU thread block).

    @param[out]
    maxsize     magma_int_t*
                maximum A(:,i) and A(i,:).

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmiluspai_sizecheck(
    magma_z_matrix A,
    magma_index_t batchsize,
    magma_index_t *maxsize,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_z_matrix AT={Magma_CSR};

    CHECK( magma_zmtranspose( A, &AT, queue ) );
    *maxsize = 0;
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        if( A.row[i+1] - A.row[i] > *maxsize ){
            *maxsize = A.row[i+1] - A.row[i];
        }
    }
    for( magma_int_t i=0; i<AT.num_rows; i++ ){
        if( AT.row[i+1] - AT.row[i] > *maxsize ){
            *maxsize = AT.row[i+1] - AT.row[i];
        }
    }

    if( *maxsize > batchsize ){
        info = -(*maxsize - batchsize);
    }

    magma_zmfree( &AT, queue );

cleanup:
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    Generates a block-diagonal sparsity pattern with block-size bs

    Arguments
    ---------

    @param[in]
    n           magma_int_t
                Size of the matrix.

    @param[in]
    bs          magma_int_t
                Size of the diagonal blocks.

    @param[in]
    offs        magma_int_t
                Size of the first diagonal block.

    @param[in]
    uplotype    magma_uplo_t
                lower or upper triangular

    @param[in,out]
    A           magma_z_matrix*
                Generated sparsity pattern matrix.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmisai_blockstruct(
    magma_int_t n,
    magma_int_t bs,
    magma_int_t offs,
    magma_uplo_t uplotype,
    magma_z_matrix *A,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t i, k, j;

    A->val = NULL;
    A->col = NULL;
    A->row = NULL;
    A->rowidx = NULL;
    A->blockinfo = NULL;
    A->diag = NULL;
    A->dval = NULL;
    A->dcol = NULL;
    A->drow = NULL;
    A->drowidx = NULL;
    A->ddiag = NULL;
    A->num_rows = n;
    A->num_cols = n;
    A->nnz = n*max(bs,offs);
    A->memory_location = Magma_CPU;
    A->storage_type = Magma_CSR;

    CHECK( magma_zmalloc_cpu( &A->val, A->nnz ));
    CHECK( magma_index_malloc_cpu( &A->row, A->num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &A->col, A->nnz ));

    // default: every row has bs elements
    #pragma omp parallel for
    for( i=0; i<offs; i++ ){
        A->row[i] = offs * i;
    }
    // default: every other row has bs elements
    #pragma omp parallel for
    for( i=offs; i<n+1; i++ ){
        A->row[i] = bs * (i-offs)+offs*offs;
    }

    if( uplotype == MagmaLower ){
        // make the first block different
        for( i=0; i<offs; i+=offs ){
            for( k=i; k<min(A->num_rows,i+offs); k++ ){
                int c = i;
                for( j=A->row[k]; j<A->row[k+1]; j++ ){
                    if( c<n ){
                        A->col[j] = c;
                        if( c <= k){
                            A->val[j] = MAGMA_Z_ONE;
                        } else {
                            A->val[j] = MAGMA_Z_ZERO;
                        }
                        c++;
                    } else {
                        A->col[j] = 0;
                        A->val[j] = MAGMA_Z_ZERO;
                        c++;
                    }
                }
            }
        }
        // now the rest
        for( i=offs; i<n; i+=bs ){
            for( k=i; k<min(A->num_rows,i+bs); k++ ){
                int c = i;
                for( j=A->row[k]; j<A->row[k+1]; j++ ){
                    if( c<n ){
                        A->col[j] = c;
                        if( c <= k){
                            A->val[j] = MAGMA_Z_ONE;
                        } else {
                            A->val[j] = MAGMA_Z_ZERO;
                        }
                        c++;
                    } else {
                        A->col[j] = 0;
                        A->val[j] = MAGMA_Z_ZERO;
                        c++;
                    }
                }
            }
        }
    } else if( uplotype == MagmaUpper ){
        // make the first block different
        for( i=0; i<offs; i+=offs ){
            for( k=i; k<min(A->num_rows,i+offs); k++ ){
                int c = i;
                for( j=A->row[k]; j<A->row[k+1]; j++ ){
                    if( c<n ){
                        A->col[j] = c;
                        if( c >= k){
                            A->val[j] = MAGMA_Z_ONE;
                        } else {
                            A->val[j] = MAGMA_Z_ZERO;
                        }
                        c++;
                    } else {
                        A->col[j] = 0;
                        A->val[j] = MAGMA_Z_ZERO;
                        c++;
                    }
                }
            }
        }
        // now the rest
        for( i=offs; i<n; i+=bs ){
            for( k=i; k<min(A->num_rows,i+bs); k++ ){
                int c = i;
                for( j=A->row[k]; j<A->row[k+1]; j++ ){
                    if( c<n ){
                        A->col[j] = c;
                        if( c >= k){
                            A->val[j] = MAGMA_Z_ONE;
                        } else {
                            A->val[j] = MAGMA_Z_ZERO;
                        }
                        c++;
                    } else {
                        A->col[j] = 0;
                        A->val[j] = MAGMA_Z_ZERO;
                        c++;
                    }
                }
            }
        }
    }

    CHECK( magma_zmcsrcompressor( A, queue ) );

cleanup:
    return info;
}
