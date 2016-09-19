/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @author Mark Gates
       
       @precisions normal z -> s d c
*/
#include "magma_internal.h"


/***************************************************************************//**
    Purpose
    -------
    ZGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.  This version does not
    require work space on the GPU passed as input. GPU memory is allocated
    in the routine.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    It uses 2 queues to overlap communication and computation.

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
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.
    \n
            Higher performance is achieved if A is in pinned memory, e.g.
            allocated using magma_malloc_pinned.

    @param[in]
    lda     INTEGER
            The leading dimension of the array A.  LDA >= max(1,M).

    @param[out]
    ipiv    INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @ingroup magma_getrf
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrf(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv,
    magma_int_t *info)
{
    #ifdef HAVE_clBLAS
    #define  dA(i_, j_)     dA, ((i_)*nb  + (j_)*nb*ldda + dA_offset)
    #define dAT(i_, j_)    dAT, ((i_)*nb*lddat + (j_)*nb + dAT_offset)
    #define dwork(i_)    dwork, (i_)
    #else
    #define  dA(i_, j_) (   dA + (i_)*nb  + (j_)*nb*ldda)
    #define dAT(i_, j_) (  dAT + (i_)*nb*lddat + (j_)*nb)
    #define dwork(i_)   (dwork + (i_))
    #endif
    
    // Constants
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    
    // Local variables
    magmaDoubleComplex *work;
    magmaDoubleComplex_ptr dA, dAT, dwork;
    magma_int_t iinfo, nb;

    /* Check arguments */
    *info = 0;
    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (lda < max(1,m))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* Function Body */
    nb = magma_get_zgetrf_nb( m, n );

    if ( (nb <= 1) || (nb >= min(m,n)) ) {
        /* Use CPU code. */
        lapackf77_zgetrf( &m, &n, A, &lda, ipiv, info );
    }
    else {
        /* Use hybrid blocked code. */
        magma_int_t maxm, maxn, ldda, lddat, maxdim;
        magma_int_t i, j, rows, cols, s = min(m, n)/nb;
        
        maxm = magma_roundup( m, 32 );
        maxn = magma_roundup( n, 32 );
        maxdim = max( maxm, maxn );
        
        lddat = maxn;
        ldda  = maxm;
        
        /* set number of GPUs */
        magma_int_t ngpu = magma_num_gpus();
        if ( ngpu > 1 ) {
            /* call multi-GPU non-GPU-resident interface  */
            magma_zgetrf_m( ngpu, m, n, A, lda, ipiv, info );
            return *info;
        }
        
        magma_queue_t queues[2] = { NULL, NULL };
        magma_device_t cdev;
        magma_getdevice( &cdev );
        magma_queue_create( cdev, &queues[0] );
        magma_queue_create( cdev, &queues[1] );
        
        /* check the memory requirement */
        size_t mem_size = magma_mem_size( queues[0] );
        mem_size /= sizeof(magmaDoubleComplex);

        magma_int_t h = 1+(2+ngpu);
        magma_int_t ngpu2 = ngpu;
        magma_int_t NB = (magma_int_t)(0.8*mem_size/maxm - h*nb);
        const char* ngr_nb_char = getenv("MAGMA_NGR_NB");
        if ( ngr_nb_char != NULL )
            NB = max( nb, min( NB, atoi(ngr_nb_char) ) );

        if ( ngpu > magma_ceildiv( NB, nb )) {
            ngpu2 = magma_ceildiv( NB, nb );
            h = 1+(2+ngpu2);
            NB = (magma_int_t)(0.8*mem_size/maxm - h*nb);
        }
        if ( ngpu2*NB < n ) {
            /* require too much memory, so call non-GPU-resident version */
            magma_zgetrf_m( ngpu, m, n, A, lda, ipiv, info );
            return *info;
        }

        work = A;
        if (maxdim*maxdim < 2*maxm*maxn) {
            // if close to square, allocate square matrix and transpose in-place
            // dwork is nb*maxm for panel, and maxdim*maxdim for A
            if (MAGMA_SUCCESS != magma_zmalloc( &dwork, nb*maxm + maxdim*maxdim )) {
                /* alloc failed so call non-GPU-resident version */
                magma_zgetrf_m( ngpu, m, n, A, lda, ipiv, info );
                return *info;
            }
            dA = dwork + nb*maxm;
            
            ldda = lddat = maxdim;
            magma_zsetmatrix( m, n, A, lda, dA(0,0), ldda, queues[0] );
            
            dAT = dA;
            magmablas_ztranspose_inplace( maxdim, dAT(0,0), lddat, queues[0] );
        }
        else {
            // if very rectangular, allocate dA and dAT and transpose out-of-place
            // dwork is nb*maxm for panel, and maxm*maxn for A
            if (MAGMA_SUCCESS != magma_zmalloc( &dwork, (nb + maxn)*maxm )) {
                /* alloc failed so call non-GPU-resident version */
                magma_zgetrf_m( ngpu, m, n, A, lda, ipiv, info );
                return *info;
            }
            dA = dwork + nb*maxm;
            
            magma_zsetmatrix( m, n, A, lda, dA(0,0), ldda, queues[0] );
            
            if (MAGMA_SUCCESS != magma_zmalloc( &dAT, maxm*maxn )) {
                /* alloc failed so call non-GPU-resident version */
                magma_free( dwork );
                magma_zgetrf_m( ngpu, m, n, A, lda, ipiv, info );
                return *info;
            }
            
            magmablas_ztranspose( m, n, dA(0,0), ldda, dAT(0,0), lddat, queues[0] );
        }
        
        lapackf77_zgetrf( &m, &nb, work, &lda, ipiv, &iinfo );

        for( j = 0; j < s; j++ ) {
            // get j-th panel from device
            cols = maxm - j*nb;
            
            if (j > 0) {
                magmablas_ztranspose( nb, cols, dAT(j,j), lddat, dwork(0), cols, queues[0] );
                magma_queue_sync( queues[0] );
                
                magma_zgetmatrix_async( m-j*nb, nb, dwork(0), cols, work, lda, queues[1] );
                
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n - (j+1)*nb, nb,
                             c_one, dAT(j-1,j-1), lddat,
                                    dAT(j-1,j+1), lddat, queues[0] );
                magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(j+1)*nb, m-j*nb, nb,
                             c_neg_one, dAT(j-1,j+1), lddat,
                                        dAT(j,  j-1), lddat,
                             c_one,     dAT(j,  j+1), lddat, queues[0] );
                
                // do the cpu part
                rows = m - j*nb;
                magma_queue_sync( queues[1] );
                lapackf77_zgetrf( &rows, &nb, work, &lda, ipiv+j*nb, &iinfo );
            }
            if (*info == 0 && iinfo > 0)
                *info = iinfo + j*nb;

            // put j-th panel onto device
            magma_zsetmatrix_async( m-j*nb, nb, work, lda, dwork(0), cols, queues[1] );
            
            for( i=j*nb; i < j*nb + nb; ++i ) {
                ipiv[i] += j*nb;
            }
            magmablas_zlaswp( n, dAT(0,0), lddat, j*nb + 1, j*nb + nb, ipiv, 1, queues[0] );

            magma_queue_sync( queues[1] );
            
            magmablas_ztranspose( cols, nb, dwork(0), cols, dAT(j,j), lddat, queues[0] );

            // do the small non-parallel computations (next panel update)
            if (s > (j+1)) {
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             nb, nb,
                             c_one, dAT(j, j  ), lddat,
                                    dAT(j, j+1), lddat, queues[0] );
                magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                             nb, m-(j+1)*nb, nb,
                             c_neg_one, dAT(j,   j+1), lddat,
                                        dAT(j+1, j  ), lddat,
                             c_one,     dAT(j+1, j+1), lddat, queues[0] );
            }
            else {
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             n-s*nb, nb,
                             c_one, dAT(j, j  ), lddat,
                                    dAT(j, j+1), lddat, queues[0] );
                magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                             n-(j+1)*nb, m-(j+1)*nb, nb,
                             c_neg_one, dAT(j,   j+1), lddat,
                                        dAT(j+1, j  ), lddat,
                             c_one,     dAT(j+1, j+1), lddat, queues[0] );
            }
        }
        
        magma_int_t nb0 = min( m - s*nb, n - s*nb );
        if ( nb0 > 0 ) {
            rows = m - s*nb;
            cols = maxm - s*nb;
            
            magmablas_ztranspose( nb0, rows, dAT(s,s), lddat, dwork(0), cols, queues[0] );
            magma_zgetmatrix_async( rows, nb0, dwork(0), cols, work, lda, queues[0] );
            magma_queue_sync( queues[0] );
            
            // do the cpu part
            lapackf77_zgetrf( &rows, &nb0, work, &lda, ipiv+s*nb, &iinfo );
            if (*info == 0 && iinfo > 0)
                *info = iinfo + s*nb;
            
            for( i=s*nb; i < s*nb + nb0; ++i ) {
                ipiv[i] += s*nb;
            }
            magmablas_zlaswp( n, dAT(0,0), lddat, s*nb + 1, s*nb + nb0, ipiv, 1, queues[0] );
            
            // put j-th panel onto device
            magma_zsetmatrix_async( rows, nb0, work, lda, dwork(0), cols, queues[0] );
            magmablas_ztranspose( rows, nb0, dwork(0), cols, dAT(s,s), lddat, queues[0] );
    
            magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                         n-s*nb-nb0, nb0,
                         c_one, dAT(s, s),     lddat,
                                dAT(s, s)+nb0, lddat, queues[0] );
        }
        
        // undo transpose
        if (maxdim*maxdim < 2*maxm*maxn) {
            magmablas_ztranspose_inplace( maxdim, dAT(0,0), lddat, queues[0] );
            magma_zgetmatrix( m, n, dAT(0,0), lddat, A, lda, queues[0] );
        }
        else {
            magmablas_ztranspose( n, m, dAT(0,0), lddat, dA(0,0), ldda, queues[0] );
            magma_zgetmatrix( m, n, dA(0,0), ldda, A, lda, queues[0] );
            magma_free( dAT );
        }
        magma_free( dwork );
 
        magma_queue_destroy( queues[0] );
        magma_queue_destroy( queues[1] );
    }
    
    return *info;
} /* magma_zgetrf */

#undef dAT
