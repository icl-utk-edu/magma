/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/
#include <cuda_runtime.h>

#include "magma_internal.h"
#include "batched_kernel_param.h"

#define A(i_, j_)  (dA + (i_) + (j_)*ldda)

#define POTRF_NATIVE_RECNB       (128)
#define POTRF_NATIVE_PANEL_SQ    (1024)
#define POTRF_NATIVE_PANEL_RECT  (2048)
#define POTRF_NATIVE_TRSM_SWITCH (7000)

/******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_recpanel_square_native(
    magma_uplo_t uplo, magma_int_t n, magmaDoubleComplex* dA, magma_int_t ldda,
    magma_int_t *dinfo, magma_int_t gbstep, magma_queue_t queue)
{
    magma_int_t recnb = POTRF_NATIVE_RECNB;
    magma_int_t info = 0;
    double d_alpha = -1.0, d_beta = 1.0;

    if(n <= recnb){
        magma_zpotf2_native(uplo, n, A(0, 0), ldda, gbstep, dinfo, queue);
        return info; 
    }

    magma_int_t n1 = n/2;
    magma_int_t n2 = n - n1;

    // factorize A(0, 0)
    magma_zpotrf_recpanel_square_native(uplo, n1, A(0, 0), ldda, dinfo, gbstep, queue);

    // trsm
    magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                 n2, n1, MAGMA_Z_ONE, 
                 A(0, 0), ldda, 
                 A(n1, 0), ldda, queue );

    // herk
    magma_zherk( uplo, MagmaNoTrans, 
                 n2, n1, 
                 d_alpha, A(n1, 0),  ldda, 
                 d_beta,  A(n1, n1), ldda, queue );

    // factorize A(n1, n1) 
    magma_zpotrf_recpanel_square_native(uplo, n2, A(n1, n1), ldda, dinfo, gbstep+n1, queue);

    return info;
}


/******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_panel_rectangle_native(
    magma_uplo_t uplo, magma_int_t n, magma_int_t nb,   
    magmaDoubleComplex* dA,    magma_int_t ldda,
    magma_int_t *dinfo, magma_int_t gbstep, magma_queue_t queue)
{
    magma_int_t info = 0;
    info = magma_zpotf2_native(uplo, min(n, nb), A(0, 0), ldda, gbstep, dinfo, queue);
    
    if(info == 0){
        if ( (n-nb) > 0) { 
            if( n-nb > POTRF_NATIVE_TRSM_SWITCH ){
                magmablas_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                         n-nb, nb, MAGMA_Z_ONE, 
                         A(0, 0) , ldda, 
                         A(nb, 0), ldda, 
                         queue );
            }
            else{
                magma_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                         n-nb, nb, MAGMA_Z_ONE, 
                         A(0, 0) , ldda, 
                         A(nb, 0), ldda, 
                         queue );
            }
        }
    }
    return info;
}


/******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_recpanel_rectangle_native(
    magma_uplo_t uplo, magma_int_t m, magma_int_t n,    
    magmaDoubleComplex* dA, magma_int_t ldda,
    magma_int_t *dinfo, magma_int_t gbstep, magma_queue_t queue)
{
    magma_int_t info = 0;
    magma_int_t recnb = POTRF_NATIVE_RECNB;
    // Quick return if possible
    if (m == 0 || n == 0) {
        return info;
    }
    
    if (m < n) {
        printf("error m < n %lld < %lld\n", (long long) m, (long long) n );
        info = -101;
        magma_xerbla( __func__, -(info) );
        return info;
    }

    magmaDoubleComplex alpha = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex beta  = MAGMA_Z_ONE;
    double d_alpha = -1.0, d_beta = 1.0;
    
    if (n <= recnb) {
        info = magma_zpotrf_panel_rectangle_native(uplo, m, n, A(0,0), ldda, dinfo, gbstep, queue);
    }
    else {
        // split A over two [A1 A2]
        magma_int_t n1 = n/2;
        magma_int_t n2 = n-n1;
        // panel on A1
        info = magma_zpotrf_recpanel_rectangle_native(uplo, m, n1, A(0, 0), ldda, dinfo, gbstep, queue);
        if (info != 0) {
            return info;
        }
        // update A2
        magma_zherk( uplo, MagmaNoTrans, 
                     n2, n1, 
                     d_alpha, A(n1, 0),  ldda, 
                     d_beta,  A(n1, n1), ldda, queue );
        magma_zgemm( MagmaNoTrans, MagmaConjTrans, 
                     m-n1-n2, n2, n1,
                     alpha, A(n, 0) , ldda, 
                            A(n1, 0), ldda, 
                     beta,  A(n, n1), ldda, queue );
        // panel on A2
        info = magma_zpotrf_recpanel_rectangle_native(uplo, m-n1, n2, A(n1, n1), ldda, dinfo, gbstep, queue);
    }
    return info;
}


/***************************************************************************//**
    Purpose
    -------
    ZPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix A. This version does not require work
    space on the GPU passed as input. GPU memory is allocated in the
    routine.

    The factorization has the form
        A = U**H * U,  if uplo = MagmaUpper, or
        A = L  * L**H, if uplo = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    This routine performs all the computation on the GPU (no CPU involved).

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of A is stored;
      -     = MagmaLower:  Lower triangle of A is stored.
      
      Only uplo = MagmaLower is supported

    @param[in]
    n       INTEGER
            The order of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (LDA,N)
            On entry, the Hermitian matrix A.  If uplo = MagmaUpper, the leading
            N-by-N upper triangular part of A contains the upper
            triangular part of the matrix A, and the strictly lower
            triangular part of A is not referenced.  If uplo = MagmaLower, the
            leading N-by-N lower triangular part of A contains the lower
            triangular part of the matrix A, and the strictly upper
            triangular part of A is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization A = U**H * U or A = L * L**H.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,N).

    @param[out]
    dinfo    INTEGER on the GPU memory
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @ingroup magma_potrf_native
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_native(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex *dA, magma_int_t ldda,
    magma_int_t *dinfo, magma_queue_t queue)
{
    magma_int_t arginfo = 0; 
    
    if ( uplo != MagmaUpper && uplo != MagmaLower) {
        arginfo = -1;
    } else if (n < 0) {
        arginfo = -2;
    } else if (ldda < max(1,n)) {
        arginfo = -4;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return (arginfo);
    }
        
    // Quick return if possible
    if (n == 0) {
        return (arginfo);
    }

    double d_alpha = -1.0;
    double d_beta  = 1.0;
    magma_int_t j, k, ib, use_stream;
    magma_int_t nb;

    // TODO: more robust tuning
    magma_int_t panel_sq = ( n >= 20000 ); 
    nb = panel_sq ? POTRF_NATIVE_PANEL_SQ : POTRF_NATIVE_PANEL_RECT; 
    
    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable\n");
    }
    else {
        for (j = 0; j < n; j += nb) {
            ib = min(nb, n-j);
            // panel
            if( panel_sq )
                magma_zpotrf_recpanel_square_native(uplo, ib, A(j, j), ldda, dinfo, j, queue);
            else
                magma_zpotrf_recpanel_rectangle_native(uplo, n-j, ib, A(j, j), ldda, dinfo, j, queue);
            // end of panel
            
            if ( (n-j-ib) > 0) {
                if ( panel_sq )
                    magmablas_ztrsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit, 
                                 n-j-ib, ib, MAGMA_Z_ONE, 
                                 A(j,    j), ldda, 
                                 A(j+ib, j), ldda, queue );
    
                magma_zherk( uplo, MagmaNoTrans, n-j-ib, ib, 
                             d_alpha, A(j+ib, j), ldda, 
                             d_beta,  A(j+ib, j+ib), ldda, queue );
            }
        }
    }
    magma_queue_sync(queue);
    
    return arginfo;
}
