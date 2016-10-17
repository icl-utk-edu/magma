/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

/***************************************************************************//**
    @deprecated
    
    Purpose
    -------
    ZLAQPS computes a step of QR factorization with column pivoting
    of a complex M-by-N matrix A by using Blas-3.  It tries to factorize
    NB columns from A starting from the row OFFSET+1, and updates all
    of the matrix with Blas-3 xGEMM.

    In some cases, due to catastrophic cancellations, it cannot
    factorize NB columns.  Hence, the actual number of factorized
    columns is returned in KB.

    Block A(1:OFFSET,1:N) is accordingly pivoted, but not factorized.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A. M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. N >= 0

    @param[in]
    offset  INTEGER
            The number of rows of A that have been factorized in
            previous steps.

    @param[in]
    nb      INTEGER
            The number of columns to factorize.

    @param[out]
    kb      INTEGER
            The number of columns actually factorized.

    @param[in,out]
    dA      COMPLEX_16 array, dimension (LDDA,N), on the GPU.
            On entry, the M-by-N matrix A.
            On exit, block A(OFFSET+1:M,1:KB) is the triangular
            factor obtained and block A(1:OFFSET,1:N) has been
            accordingly pivoted, but no factorized.
            The rest of the matrix, block A(OFFSET+1:M,KB+1:N) has
            been updated.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A. LDDA >= max(1,M).

    @param[in,out]
    jpvt    INTEGER array, dimension (N)
            JPVT(I) = K <==> Column K of the full matrix A has been
            permuted into position I in AP.

    @param[out]
    tau     COMPLEX_16 array, dimension (KB)
            The scalar factors of the elementary reflectors.

    @param[in,out]
    vn1     DOUBLE PRECISION array, dimension (N)
            The vector with the partial column norms.

    @param[in,out]
    vn2     DOUBLE PRECISION array, dimension (N)
            The vector with the exact column norms.

    @param[in,out]
    dauxv   COMPLEX_16 array, dimension (NB), on the GPU
            Auxiliary vector.

    @param[in,out]
    dF      COMPLEX_16 array, dimension (LDDF,NB), on the GPU
            Matrix F' = L*Y'*A.

    @param[in]
    lddf    INTEGER
            The leading dimension of the array F. LDDF >= max(1,N).

    @ingroup magma_laqps
*******************************************************************************/
extern "C" magma_int_t
magma_zlaqps_gpu(
    magma_int_t m, magma_int_t n, magma_int_t offset,
    magma_int_t nb, magma_int_t *kb,
    magmaDoubleComplex_ptr dA,  magma_int_t ldda,
    magma_int_t *jpvt, magmaDoubleComplex *tau,
    double *vn1, double *vn2,
    magmaDoubleComplex_ptr dauxv,
    magmaDoubleComplex_ptr dF,  magma_int_t lddf)
{
#define  dA(i, j) (dA  + (i) + (j)*(ldda))
#define  dF(i, j) (dF  + (i) + (j)*(lddf))

    magmaDoubleComplex c_zero    = MAGMA_Z_MAKE( 0.,0.);
    magmaDoubleComplex c_one     = MAGMA_Z_MAKE( 1.,0.);
    magmaDoubleComplex c_neg_one = MAGMA_Z_MAKE(-1.,0.);
    magma_int_t ione = 1;
    
    magma_int_t i__1, i__2;
    magmaDoubleComplex z__1;
    
    magma_int_t k, rk;
    magmaDoubleComplex_ptr dAks;
    magmaDoubleComplex tauk = MAGMA_Z_ZERO;
    magma_int_t pvt;
    double tol3z;
    magma_int_t itemp;

    double lsticc;
    magmaDouble_ptr dlsticcs;
    magma_dmalloc( &dlsticcs, 1+256*(n+255)/256 );

    tol3z = magma_dsqrt( lapackf77_dlamch("Epsilon"));

    lsticc = 0;
    k = 0;
    magma_zmalloc( &dAks, nb );

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    while( k < nb && lsticc == 0 ) {
        rk = offset + k;
        
        /* Determine ith pivot column and swap if necessary */
        // subtract 1 from Fortran/CUBLAS idamax; pvt, k are 0-based.
        pvt = k + magma_idamax( n-k, &vn1[k], ione, queue ) - 1;
        
        if (pvt != k) {
            /* F gets swapped so F must be sent at the end to GPU   */
            i__1 = k;
            magmablas_zswap( m, dA(0, pvt), ione, dA(0, k), ione, queue );

            magmablas_zswap( i__1, dF(pvt, 0), lddf, dF(k, 0), lddf, queue );
            itemp     = jpvt[pvt];
            jpvt[pvt] = jpvt[k];
            jpvt[k]   = itemp;
            magma_dswap( 2, &vn1[pvt], n+offset, &vn1[k], n+offset, queue );
        }

        /* Apply previous Householder reflectors to column K:
           A(RK:M,K) := A(RK:M,K) - A(RK:M,1:K-1)*F(K,1:K-1)'.
           Optimization: multiply with beta=0; wait for vector and subtract */
        if (k > 0) {
            //#define RIGHT_UPDATE
            #ifdef RIGHT_UPDATE
                i__1 = m - offset - nb;
                i__2 = k;
                magma_zgemv( MagmaNoTrans, i__1, i__2,
                             c_neg_one, A(offset+nb, 0), lda,
                                        F(k,         0), ldf,
                             c_one,     A(offset+nb, k), ione, queue );
            #else
                i__1 = m - rk;
                i__2 = k;
                magma_zgemv( MagmaNoTrans, i__1, i__2,
                             c_neg_one, dA(rk, 0), ldda,
                                        dF(k,  0), lddf,
                             c_one,     dA(rk, k), ione, queue );
            #endif
        }
        
        /*  Generate elementary reflector H(k). */
        magma_zlarfg_gpu( m-rk, dA(rk, k), dA(rk + 1, k), &tau[k], &vn1[k], &dAks[k], queue );

        /* needed to avoid the race condition */
        if (k == 0) magma_zsetvector(  1,    &c_one,        1, dA(rk, k), 1, queue );
        else        magma_zcopymatrix( 1, 1, dA(offset, 0), 1, dA(rk, k), 1, queue );

        /* Compute Kth column of F:
           Compute  F(K+1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K) on the GPU */
        if (k < n-1 || k > 0) magma_zgetvector( 1, &tau[k], 1, &tauk, 1, queue );
        if (k < n-1) {
            i__1 = m - rk;
            i__2 = n - k - 1;

            /* Multiply on GPU */
            magma_zgemv( MagmaConjTrans, m-rk, n-k-1,
                         tauk,   dA( rk,  k+1 ), ldda,
                                 dA( rk,  k   ), 1,
                         c_zero, dF( k+1, k   ), 1, queue );
        }
        
        /* Incremental updating of F:
           F(1:N,K) := F(1:N,K)                        - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'*A(RK:M,K).
           F(1:N,K) := tau(K)*A(RK:M,K+1:N)'*A(RK:M,K) - tau(K)*F(1:N,1:K-1)*A(RK:M,1:K-1)'*A(RK:M,K)
                    := tau(K)(A(RK:M,K+1:N)' - F(1:N,1:K-1)*A(RK:M,1:K-1)') A(RK:M,K)
           so, F is (updated A)*V */
        if (k > 0) {
            z__1 = MAGMA_Z_NEGATE( tauk );
            #ifdef RIGHT_UPDATE
                i__1 = m - offset - nb;
                i__2 = k;
                magma_zgemv( MagmaConjTrans, i__1, i__2,
                             z__1,   dA(offset+nb, 0), lda,
                                     dA(offset+nb, k), ione,
                             c_zero, dauxv, ione, queue );
                
                i__1 = k;
                magma_zgemv( MagmaNoTrans, n-k-1, i__1,
                             c_one, F(k+1,0), ldf,
                                    dauxv,     ione,
                             c_one, F(k+1,k), ione, queue );
            #else
                i__1 = m - rk;
                i__2 = k;
                magma_zgemv( MagmaConjTrans, i__1, i__2,
                             z__1,   dA(rk, 0), ldda,
                                     dA(rk, k), ione,
                             c_zero, dauxv, ione, queue );
                
                /* I think we only need stricly lower-triangular part :) */
                magma_zgemv( MagmaNoTrans, n-k-1, i__2,
                             c_one, dF(k+1,0), lddf,
                                    dauxv,     ione,
                             c_one, dF(k+1,k), ione, queue );
            #endif
        }
        
        /* Optimization: On the last iteration start sending F back to the GPU */
        
        /* Update the current row of A:
           A(RK,K+1:N) := A(RK,K+1:N) - A(RK,1:K)*F(K+1:N,1:K)'.               */
        if (k < n-1) {
            i__1 = n - k - 1;
            i__2 = k + 1;
            #ifdef RIGHT_UPDATE
                /* right-looking update of rows,                     */
                magma_zgemm( MagmaNoTrans, MagmaConjTrans, nb-k, i__1, ione,
                             c_neg_one, dA(rk,  k  ), ldda,
                                        dF(k+1, k  ), lddf,
                             c_one,     dA(rk,  k+1), ldda, queue );
            #else
                /* left-looking update of rows,                     *
                 * since F=A'v with original A, so no right-looking */
                magma_zgemm( MagmaNoTrans, MagmaConjTrans, ione, i__1, i__2,
                             c_neg_one, dA(rk, 0  ), ldda,
                                        dF(k+1,0  ), lddf,
                             c_one,     dA(rk, k+1), ldda, queue );
            #endif
        }
        
        /* Update partial column norms. */
        if (rk < min(m, n+offset)-1 ) {
            magmablas_dznrm2_row_check_adjust( n-k-1, tol3z, &vn1[k+1], &vn2[k+1], 
                                               dA(rk,k+1), ldda, dlsticcs, queue );

            magma_dgetvector( 1, &dlsticcs[0], 1, &lsticc, 1, queue );
        }
        
        ++k;
    }
    magma_zcopymatrix( 1, k, dAks, 1, dA(offset, 0), ldda+1, queue );

    // leave k as the last column done
    --k;
    *kb = k + 1;
    rk = offset + *kb - 1;

    /* Apply the block reflector to the rest of the matrix:
       A(OFFSET+KB+1:M,KB+1:N) := A(OFFSET+KB+1:M,KB+1:N) - A(OFFSET+KB+1:M,1:KB)*F(KB+1:N,1:KB)'  */
    if (*kb < min(n, m - offset)) {
        i__1 = m - rk - 1;
        i__2 = n - *kb;
        
        magma_zgemm( MagmaNoTrans, MagmaConjTrans, i__1, i__2, *kb,
                     c_neg_one, dA(rk+1, 0  ), ldda,
                                dF(*kb,  0  ), lddf,
                     c_one,     dA(rk+1, *kb), ldda, queue );
    }
    /* Recomputation of difficult columns. */
    if ( lsticc > 0 ) {
        // printf( " -- recompute dnorms --\n" );
        magmablas_dznrm2_check( m-rk-1, n-*kb, dA(rk+1,*kb), ldda,
                                &vn1[*kb], dlsticcs, queue );
        magma_dcopymatrix( n-*kb, 1, &vn1[*kb], *kb, &vn2[*kb], *kb, queue );
    }
    magma_free( dAks );
    magma_free( dlsticcs );

    magma_queue_destroy( queue );

    return MAGMA_SUCCESS;
} /* magma_zlaqps */
