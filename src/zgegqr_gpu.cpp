/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @precisions normal z -> s d c

*/
#include "magma_internal.h"

#define COMPLEX

// === Define what BLAS to use ============================================
#undef  magma_ztrsm
#define magma_ztrsm magmablas_ztrsm
// === End defining what BLAS to use ======================================

/***************************************************************************//**
    Purpose
    -------
    ZGEGQR orthogonalizes the N vectors given by a complex M-by-N matrix A:
           
        A = Q * R.

    On exit, if successful, the orthogonal vectors Q overwrite A
    and R is given in work (on the CPU memory).
    The routine is designed for tall-and-skinny matrices: M >> N, N <= 128.
    
    This version uses normal equations and SVD in an iterative process that
    makes the computation numerically accurate.
    
    Arguments
    ---------
    @param[in]
    ikind   INTEGER
            Several versions are implemented indiceted by the ikind value:
            1:  This version uses normal equations and SVD in an iterative process
                that makes the computation numerically accurate.
            2:  This version uses a standard LAPACK-based orthogonalization through
                MAGMA's QR panel factorization (magma_zgeqr2x3_gpu) and magma_zungqr
            3:  Modified Gram-Schmidt (MGS)
            4.  Cholesky QR [ Note: this method uses the normal equations which
                                    squares the condition number of A, therefore
                                    ||I - Q'Q|| < O(eps cond(A)^2)               ]

    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  m >= n >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A. 128 >= n >= 0.

    @param[in,out]
    dA      COMPLEX_16 array on the GPU, dimension (ldda,n)
            On entry, the m-by-n matrix A.
            On exit, the m-by-n matrix Q with orthogonal columns.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,m).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param
    dwork   (GPU workspace) COMPLEX_16 array, dimension:
            n^2                    for ikind = 1
            3 n^2 + min(m, n) + 2  for ikind = 2
            0 (not used)           for ikind = 3
            n^2                    for ikind = 4

    @param[out]
    work    (CPU workspace) COMPLEX_16 array, dimension 3 n^2.
            On exit, work(1:n^2) holds the rectangular matrix R.
            Preferably, for higher performance, work should be in pinned memory.
 
    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  for ikind = 1 and 4, the normal equations were not
                  positive definite, so the factorization could not be
                  completed, and the solution has not been computed.
                  For ikind = 3, the space is not linearly independent.
                  For all these cases the rank (< n) of the space is returned.

    @ingroup magma_gegqr
*******************************************************************************/
extern "C" magma_int_t
magma_zgegqr_gpu(
    magma_int_t ikind, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA,   magma_int_t ldda,
    magmaDoubleComplex_ptr dwork, magmaDoubleComplex *work,
    magma_int_t *info )
{
    #define work(i_,j_) (work + (i_) + (j_)*n)
    #define dA(i_,j_)   (dA   + (i_) + (j_)*ldda)
    
    magma_int_t i = 0, j, k, n2 = n*n;
    magma_int_t ione = 1;
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    double cn;

    /* check arguments */
    *info = 0;
    if (ikind < 1 || ikind > 4) {
        *info = -1;
    } else if (m < 0 || m < n) {
        *info = -2;
    } else if (n < 0 || n > 128) {
        *info = -3;
    } else if (ldda < max(1,m)) {
        *info = -5;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_queue_t queue;
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queue );

    if (ikind == 1) {
        // === Iterative, based on SVD =========================================
        magmaDoubleComplex *U, *VT, *vt, *R, *G, *hwork, *tau;
        double *S;

        R    = work;             // Size n * n
        G    = R    + n*n;       // Size n * n
        VT   = G    + n*n;       // Size n * n
        
        magma_zmalloc_cpu( &hwork, 32 + 2*n*n + 2*n );
        if ( hwork == NULL ) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        
        magma_int_t lwork=n*n+32; // First part f hwork; used as workspace in svd
        
        U    = hwork + n*n + 32;   // Size n*n
        S    = (double*)(U + n*n); // Size n
        tau  = U + n*n + n;        // Size n
        
        #ifdef COMPLEX
        double *rwork;
        magma_dmalloc_cpu( &rwork, 5*n );
        if ( rwork == NULL ) {
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        #endif
        
        double eps = lapackf77_dlamch("Epsilon");
        do {
            i++;
            
            magma_zgemm( MagmaConjTrans, MagmaNoTrans, n, n, m, c_one,
                         dA, ldda, dA, ldda, c_zero, dwork, n, queue );
            magma_zgetmatrix( n, n, dwork, n, G, n, queue );
            
            lapackf77_zgesvd( "n", "a", &n, &n, G, &n, S, U, &n, VT, &n,
                              hwork, &lwork,
                              #ifdef COMPLEX
                              rwork,
                              #endif
                              info );
            
            for (k=0; k < n; k++) {
                S[k] = magma_dsqrt( S[k] );
                
                if (S[k] < eps) {
                    *info = k;
                    return *info;
                }
            }
            
            for (k=0; k < n; k++) {
                vt = VT + k*n;
                for (j=0; j < n; j++)
                    vt[j] = vt[j] * S[j];
            }
            lapackf77_zgeqrf( &n, &n, VT, &n, tau, hwork, &lwork, info );
            
            if (i == 1)
                blasf77_zcopy( &n2, VT, &ione, R, &ione );
            else
                blasf77_ztrmm( "l", "u", "n", "n", &n, &n, &c_one, VT, &n, R, &n );
            
            magma_zsetmatrix( n, n, VT, n, dwork, n, queue );
            magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                         m, n, c_one, dwork, n, dA, ldda, queue );

            cn = S[0]/S[n-1];
        } while (cn > 10.f && i<5);
        
        magma_free_cpu( hwork );
        #ifdef COMPLEX
        magma_free_cpu( rwork );
        #endif
        // ================== end of ikind == 1 ================================
    }
    else if (ikind == 2) {
        // ================== LAPACK based      ================================
        magma_int_t min_mn = min(m, n);
        magma_int_t nb = n;

        magmaDoubleComplex_ptr dtau = dwork + 2*n*n;
        magmaDoubleComplex_ptr dT   = dwork;
        magmaDoubleComplex_ptr ddA  = dwork + n*n;
        magmaDoubleComplex *tau  = work+n*n;

        magmablas_zlaset( MagmaFull, n, n, c_zero, c_zero, dT, n, queue );
        magma_zgeqr2x3_gpu( m, n, dA, ldda, dtau, dT, ddA,
                            (double*)(dwork + min_mn + 2*n*n), info );
        magma_zgetmatrix( min_mn, 1, dtau, min_mn, tau, min_mn, queue );
        magma_zgetmatrix( n, n, ddA, n, work, n, queue );
        magma_zungqr_gpu( m, n, n, dA, ldda, tau, dT, nb, info );
        // ================== end of ikind == 2 ================================
    }
    else if (ikind == 3) {
        // ================== MGS               ================================
        double eps = lapackf77_dlamch("Epsilon");
        for (j = 0; j < n; j++) {
            for (i = 0; i < j; i++) {
                *work(i, j) = magma_zdotc( m, dA(0,i), 1, dA(0,j), 1, queue );
                magma_zaxpy( m, -(*work(i,j)),  dA(0,i), 1, dA(0,j), 1, queue );
            }
            for (i = j; i < n; i++) {
                *work(i, j) = MAGMA_Z_ZERO;
            }
            //*work(j,j) = MAGMA_Z_MAKE( magma_dznrm2( m, dA(0,j), 1), 0., queue );
            *work(j,j) = magma_zdotc( m, dA(0,j), 1, dA(0,j), 1, queue );
            *work(j,j) = MAGMA_Z_MAKE( sqrt(MAGMA_Z_REAL( *work(j,j) )), 0. );
            if (MAGMA_Z_ABS(*work(j,j)) < eps) {
                *info = j;
                break;
            }
            magma_zscal( m, MAGMA_D_ONE/ *work(j,j), dA(0,j), 1, queue );
        }
        // ================== end of ikind == 3 ================================
    }
    else if (ikind == 4) {
        // ================== Cholesky QR       ================================
        magma_zgemm( MagmaConjTrans, MagmaNoTrans, n, n, m, c_one,
                     dA, ldda, dA, ldda, c_zero, dwork, n, queue );
        magma_zgetmatrix( n, n, dwork, n, work, n, queue );
        lapackf77_zpotrf( "u", &n, work, &n, info );
        magma_zsetmatrix( n, n, work, n, dwork, n, queue );
        magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                     m, n, c_one, dwork, n, dA, ldda, queue );
        // ================== end of ikind == 4 ================================
    }
             
    magma_queue_destroy( queue );

    return *info;
} /* magma_zgegqr_gpu */
