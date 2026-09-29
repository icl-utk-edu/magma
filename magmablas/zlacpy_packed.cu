/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

#define dA(i_, j_)   dA[ (j_) * ldda  + (i_)]
#define dAP(i_, j_) dAP[N*(j_) - (j_)*((j_)+1)/2 + (i_)]

#define BLK_X 32
#define BLK_Y  4

////////////////////////////////////////////////////////////////////////////////
static __device__
void zlacpy_full2packed_device(
    magma_uplo_t uplo, int m, int n,
    magma_uplo_t uplo_packed, int npacked,
    magmaDoubleComplex *dA, int Ai, int Aj, int ldda,
    magmaDoubleComplex *dAP, int APi, int APj)
{
    int gtx = blockIdx.x * BLK_X + threadIdx.x;
    int gty = blockIdx.y * BLK_Y + threadIdx.y;

    if( gtx >= m || gty >= n ) return;

    bool active = ( uplo == MagmaFull                ) ||
                  ( uplo == MagmaLower && gty <= gtx ) ||
                  ( uplo == MagmaUpper && gtx <= gty ) ;

    if( active == true ) {
        if(uplo_packed == MagmaLower)
            dAP_LOWER(gtx+APi, gty+APj) = dA(gtx+Ai, gty+Aj);
        else
            dAP_UPPER(gtx+APi, gty+APj) = dA(gtx+Ai, gty+Aj);
    }
}

////////////////////////////////////////////////////////////////////////////////
__global__
void zlacpy_full2packed_kernel_batched(
    magma_uplo_t uplo, int m, int n,
    magma_uplo_t uplo_packed, int npacked,
    magmaDoubleComplex_ptr dA_array[], int Ai, int Aj, int ldda,
    magmaDoubleComplex_ptr dAP_array[], int APi, int APj )
{
    int batchid = blockIdx.z;
    zlacpy_full2packed_device(
            uplo, m, n, uplo_packed, npacked,
            dA_array[batchid], Ai, Aj, ldda,
            dAP_array[batchid], APi, APj);
}

/***************************************************************************//**
    Purpose
    -------
    ZLACPY_FULL2PACKED copies either the lower or the upper triangular part of
    a matrix dA stored in column-major format to another matrix dAP stored in packed
    format.

    This is the batch version of the routine

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA to be copied to dAP.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
            Otherwise:  All of each matrix dA

    @param[in]
    m       INTEGER
            The number of rows of the matrix dA to be copied  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix dA to be copied  N >= 0.

    @param[in]
    uplo_packed    magma_uplo_t
            Specifies which part of dAP is stored in packed format.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part

    @param[in]
    npacked       INTEGER
            The size of the packed matrix AP.

    @param[in]
    dA_array     array of pointers, dimension(batchCount)
            Each is a COMPLEX_16 array, dimension (LDDA,N)
            The N-by-N matrix dA.
            If UPLO = MagmaUpper, only the upper triangle part is copied to dAP;
            if UPLO = MagmaLower, only the lower triangle part is copied to dAP.

    @param[in]
    Ai      INTEGER
            The row offset of A.

    @param[in]
    Aj      INTEGER
            The column offset of A.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,M).

    @param[out]
    dAP_array    array of pointers, dimension(batchCount)
            Each is a COMPLEX_16 array, dimension N*(N+1)/2
            On exit, dAP = stores copied part of dA as a linear array (packed format)

    @param[in]
    APi     INTEGER
            The row offset of AP.

    @param[in]
    APj     INTEGER
            The column offset of AP.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @param[in]
    batchCount    INTEGER
            The number of matrices to operate on.

    @ingroup magma_lacpy
*******************************************************************************/
extern "C" void
magmablas_zlacpy_full2packed_batched(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magma_uplo_t uplo_packed, magma_int_t npacked,
    magmaDoubleComplex_ptr dA_array[], magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex_ptr dAP_array[], magma_int_t APi, magma_int_t APj,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper && uplo != MagmaFull)
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    if ( uplo_packed != MagmaLower && uplo_packed != MagmaUpper )
        info = -4;
    if ( npacked < 0 )
        info = -5;
    else if ( Ai < 0 )
        info = -7;
    else if ( Aj < 0 )
        info = -8;
    else if ( ldda < max(1,n))
        info = -9;
    else if ( APi < 0 )
        info = -11;
    else if ( APj < 0 )
        info = -12;
    else if ( batchCount < 0 )
        info = -13;

    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    if ( m == 0 || n == 0 || npacked == 0 || batchCount == 0 ) {
        return;
    }

    dim3 threads( BLK_X, BLK_Y );
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid( magma_ceildiv(m, BLK_X), magma_ceildiv(n, BLK_Y), ibatch);

        zlacpy_full2packed_kernel_batched<<< grid, threads, 0, queue->cuda_stream() >>>
        (uplo, m, n, uplo_packed, npacked, dA_array+i, Ai, Aj, ldda, dAP_array+i, APi, APj);
    }
}
