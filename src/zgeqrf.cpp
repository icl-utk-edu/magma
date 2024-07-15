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

/***************************************************************************//**
    Purpose
    -------
    ZGEQRF computes a QR factorization of a COMPLEX_16 M-by-N matrix A:
    A = Q * R. This version does not require work space on the GPU
    passed as input. GPU memory is allocated in the routine.

    This uses 2 queues to overlap communication and computation.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    A       COMPLEX_16 array, dimension (LDA,N)
            On entry, the M-by-N matrix A.
            On exit, the elements on and above the diagonal of the array
            contain the min(M,N)-by-N upper trapezoidal matrix R (R is
            upper triangular if m >= n); the elements below the diagonal,
            with the array TAU, represent the orthogonal matrix Q as a
            product of min(m,n) elementary reflectors (see Further
            Details).
    \n
            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    tau     COMPLEX_16 array, dimension (min(M,N))
            The scalar factors of the elementary reflectors (see Further
            Details).

    @param[out]
    work    (workspace) COMPLEX_16 array, dimension (MAX(1,LWORK))
            On exit, if INFO = 0, WORK[0] returns the optimal LWORK.
    \n
            Higher performance is achieved if WORK is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.  LWORK >= N*NB,
            where NB can be obtained through magma_get_zgeqrf_nb( M, N ).
    \n
            If LWORK = -1, then a workspace query is assumed; the routine
            only calculates the optimal size of the WORK array, returns
            this value as the first entry of the WORK array, and no error
            message related to LWORK is issued.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

        Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

        H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_geqrf
*******************************************************************************/
extern "C" magma_int_t
magma_zgeqrf(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A,    magma_int_t lda,
    magmaDoubleComplex *tau,
    magmaDoubleComplex *work, magma_int_t lwork,
    magma_int_t *info )
{
    #define  A(i_,j_)  (A + (i_) + (j_)*lda)
    
    #ifdef MAGMA_HAVE_OPENCL
    #define dA(i_,j_)  dA,    ((i_) + (j_)*ldda + dA_offset)
    #define dT(i_,j_)  dT,    ((i_) + (j_)*nb   + dT_offset)
    #define dwork(i_)  dwork, ((i_)             + dwork_offset)
    #else
    #define dA(i_,j_) (dA    + (i_) + (j_)*ldda)
    #define dT(i_,j_) (dT    + (i_) + (j_)*nb)
    #define dwork(i_) (dwork + (i_))
    #endif
    
    /* Constants */
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;
    
    /* Local variables */
    magmaDoubleComplex* work_local = NULL;
    magmaDoubleComplex_ptr dA, dT, dwork;
    magma_int_t i, ib, min_mn, ldda, lddwork, old_i, old_ib;
    
    /* Function Body */
    *info = 0;
    magma_int_t nb = magma_get_zgeqrf_nb( m, n );
    
    magma_int_t lwkopt = n*nb;
    work[0] = magma_zmake_lwork( lwkopt );
    bool lquery = (lwork == -1);
    if (m < 0) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (lda < max(1,m)) {
        *info = -4;
    } else if (lwork < max(1, lwkopt) && ! lquery) {
        *info = -7;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }
    else if (lquery)
        return *info;
    
    min_mn = min( m, n );
    if (min_mn == 0) {
        work[0] = c_one;
        return *info;
    }
    
    if (nb <= 1 || 4*nb >= min(m,n) ) {
        /* Use CPU code. */
        lapackf77_zgeqrf( &m, &n, A, &lda, tau, work, &lwork, info );
        return *info;
    }
    
    // largest N for larfb is n-nb (trailing matrix lacks 1st panel)
    lddwork = magma_roundup( n, 32 ) - nb;
    ldda    = magma_roundup( m, 32 );
    
    magma_int_t ngpu = magma_num_gpus();
    if ( ngpu > 1 ) {
        /* call multiple-GPU interface  */
        return magma_zgeqrf_m( ngpu, m, n, A, lda, tau, work, lwork, info );
    }
    
    // allocate space for dA, dwork, and dT
    if (MAGMA_SUCCESS != magma_zmalloc( &dA, n*ldda + nb*lddwork + nb*nb )) {
        /* alloc failed so call non-GPU-resident version */
        return magma_zgeqrf_ooc( m, n, A, lda, tau, work, lwork, info );
    }
    
    // Need at least 2*nb*nb to store T and upper triangle of V simultaneously.
    // For better LAPACK compatability, which needs N*NB,
    // allow lwork < 2*NB*NB and allocate here if needed.
    if (lwork < 2*nb*nb) {
        if (MAGMA_SUCCESS != magma_zmalloc_cpu( &work_local, 2*nb*nb )) {
            magma_free( dA );
            *info = MAGMA_ERR_HOST_ALLOC;
            return *info;
        }
        work = work_local;
    }
    
    dwork = dA + n*ldda;
    dT    = dA + n*ldda + nb*lddwork;
    
    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    
    if ( (nb > 1) && (nb < min_mn) ) {
        /* Use blocked code initially.
           Asynchronously send the matrix to the GPU except the first panel. */
        magma_zsetmatrix_async( m, n-nb,
                                 A(0,nb), lda,
                                dA(0,nb), ldda, queues[0] );
        
        old_i = 0;
        old_ib = nb;
        for (i = 0; i < min_mn-nb; i += nb) {
            ib = min( min_mn-i, nb );
            if (i > 0) {
                /* get i-th panel from device */
                magma_queue_sync( queues[1] );
                magma_zgetmatrix_async( m-i, ib,
                                        dA(i,i), ldda,
                                         A(i,i), lda, queues[0] );
                
                /* Apply H' to A(i:m,i+2*ib:n) from the left */
                magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                  m-old_i, n-old_i-2*old_ib, old_ib,
                                  dA(old_i, old_i),          ldda, dT(0,0),  nb,
                                  dA(old_i, old_i+2*old_ib), ldda, dwork(0), lddwork, queues[1] );
                
                magma_zgetmatrix_async( i, ib,
                                        dA(0,i), ldda,
                                         A(0,i), lda, queues[1] );
                magma_queue_sync( queues[0] );
            }
            
            magma_int_t rows = m-i;
            lapackf77_zgeqrf( &rows, &ib, A(i,i), &lda, tau+i, work, &lwork, info );
            
            /* Form the triangular factor of the block reflector
               H = H(i) H(i+1) . . . H(i+ib-1) */
            lapackf77_zlarft( MagmaForwardStr, MagmaColumnwiseStr,
                              &rows, &ib, A(i,i), &lda, tau+i, work, &ib );
            
            magma_zpanel_to_q( MagmaUpper, ib, A(i,i), lda, work+ib*ib );
            
            /* put i-th V matrix onto device */
            magma_zsetmatrix_async( rows, ib, A(i,i), lda, dA(i,i), ldda, queues[0] );
            
            /* put T matrix onto device */
            magma_queue_sync( queues[1] );
            magma_zsetmatrix_async( ib, ib, work, ib, dT(0,0), nb, queues[0] );
            magma_queue_sync( queues[0] );
            
            if (i + ib < n) {
                if (i+ib < min_mn-nb) {
                    /* Apply H' to A(i:m,i+ib:i+2*ib) from the left (look-ahead) */
                    magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, ib, ib,
                                      dA(i, i   ), ldda, dT(0,0),  nb,
                                      dA(i, i+ib), ldda, dwork(0), lddwork, queues[1] );
                    magma_zq_to_panel( MagmaUpper, ib, A(i,i), lda, work+ib*ib );
                }
                else {
                    /* After last panel, update whole trailing matrix. */
                    /* Apply H' to A(i:m,i+ib:n) from the left */
                    magma_zlarfb_gpu( MagmaLeft, MagmaConjTrans, MagmaForward, MagmaColumnwise,
                                      rows, n-i-ib, ib,
                                      dA(i, i   ), ldda, dT(0,0),  nb,
                                      dA(i, i+ib), ldda, dwork(0), lddwork, queues[1] );
                    magma_zq_to_panel( MagmaUpper, ib, A(i,i), lda, work+ib*ib );
                }
                
                old_i  = i;
                old_ib = ib;
            }
        }
    } else {
        i = 0;
    }
    
    /* Use unblocked code to factor the last or only block. */
    if (i < min_mn) {
        ib = n-i;
        if (i != 0) {
            magma_zgetmatrix( m, ib, dA(0,i), ldda, A(0,i), lda, queues[1] );
        }
        magma_int_t rows = m-i;
        lapackf77_zgeqrf( &rows, &ib, A(i,i), &lda, tau+i, work, &lwork, info );
    }
    
    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    
    work[0] = magma_zmake_lwork( lwkopt );  // before free( work_local )
    
    magma_free( dA );
    magma_free_cpu( work_local );  // if allocated
    
    return *info;
} /* magma_zgeqrf */
