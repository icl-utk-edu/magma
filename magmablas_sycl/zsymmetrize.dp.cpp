/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Mark Gates
*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define NB 64

/*
    Matrix is m x m, and is divided into block rows, each NB x m.
    Each block has NB threads.
    Each thread copies one row, iterating across all columns below diagonal.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (i >= m) are disabled.
*/
void
zsymmetrize_lower( int m, magmaDoubleComplex *dA, int ldda ,
                   sycl::nd_item<3> item_ct1)
{
    // dA iterates across row i and dAT iterates down column i.
    int i = item_ct1.get_group(2) * NB + item_ct1.get_local_id(2);
    magmaDoubleComplex *dAT = dA;
    if ( i < m ) {
        dA  += i;
        dAT += i*ldda;
        magmaDoubleComplex *dAend = dA + i*ldda;  // end at diagonal dA(i,i)
        while( dA < dAend ) {
            *dAT = MAGMA_Z_CONJ(*dA);  // upper := lower
            dA  += ldda;
            dAT += 1;
        }
        /*
        DPCT1064:1332: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        *dA = MAGMA_Z_MAKE(MAGMA_Z_REAL(*dA), 0); // make diagonal real
    }
}


// only difference with _lower version is direction dA=dAT instead of dAT=dA.
void
zsymmetrize_upper( int m, magmaDoubleComplex *dA, int ldda ,
                   sycl::nd_item<3> item_ct1)
{
    // dA iterates across row i and dAT iterates down column i.
    int i = item_ct1.get_group(2) * NB + item_ct1.get_local_id(2);
    magmaDoubleComplex *dAT = dA;
    if ( i < m ) {
        dA  += i;
        dAT += i*ldda;
        magmaDoubleComplex *dAend = dA + i*ldda;  // end at diagonal dA(i,i)
        while( dA < dAend ) {
            *dA = MAGMA_Z_CONJ(*dAT);  // lower := upper
            dA  += ldda;
            dAT += 1;
        }
        /*
        DPCT1064:1333: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        *dA = MAGMA_Z_MAKE(MAGMA_Z_REAL(*dA), 0); // make diagonal real
    }
}


/***************************************************************************//**
    Purpose
    -------
    
    ZSYMMETRIZE copies lower triangle to upper triangle, or vice-versa,
    to make dA a general representation of a symmetric matrix.
    In Complex, it sets the diagonal to be Real.
    
    Arguments
    ---------
    
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA that is valid on input.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix dA.  M >= 0.
    
    @param[in,out]
    dA      COMPLEX_16 array, dimension (LDDA,N)
            The m by m matrix dA.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_symmetrize
*******************************************************************************/
extern "C" void
magmablas_zsymmetrize(
    magma_uplo_t uplo, magma_int_t m,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( ldda < max(1,m) )
        info = -4;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if ( m == 0 )
        return;

    sycl::range<3> threads(1, 1, NB);
    sycl::range<3> grid(1, 1, magma_ceildiv(m, NB));

    if ( uplo == MagmaUpper ) {
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zsymmetrize_upper(m, dA, ldda, item_ct1);
                           });
    }
    else {
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zsymmetrize_lower(m, dA, ldda, item_ct1);
                           });
    }
}
