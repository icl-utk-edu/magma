/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Tingxing Dong

   @precisions normal z -> s d c
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"

#define dA(i, j)  (dA + (i) + (j)*ldda)
#define PARSWAP

magma_int_t
magma_zgetf2_native_blocked(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *dipiv, magma_int_t *dinfo,
    magma_int_t gbstep, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    if (m < 0) {
        arginfo = -1;
    } else if (n < 0 ) {
        arginfo = -2;
    } else if (ldda < max(1,m)) {
        arginfo = -4;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magma_int_t nb = BATF2_NB;

    magma_int_t min_mn = min(m, n);
    magma_int_t gbj, j, step, ib;

    for( j=0; j < min_mn; j += nb){
        ib = min(nb, min_mn-j);
        for (step=0; step < ib; step++) {
            gbj = j+step;
            // find the max of the column gbj
            arginfo = magma_izamax_native( m-gbj, dA, 1, gbj, ldda, dipiv, dinfo, gbstep, queue);

            if (arginfo != 0 ) return arginfo;
            // Apply the interchange to columns 1:N. swap the whole row
            magma_zswap_native(n, dA, ldda, gbj, dipiv, queue);
            if (arginfo != 0 ) return arginfo;
            // Compute elements J+1:M of J-th column.
            if (gbj < m) {
                arginfo = magma_zscal_zgeru_native( m-gbj, ib-step, gbj, dA, ldda, dinfo, gbstep, queue );
                if (arginfo != 0 ) return arginfo;
            }
        }

        if ( (n-j-ib) > 0) {
            magma_zgetf2trsm_2d_native( ib, n-j-ib, 
                                        dA(j,j   ), ldda, 
                                        dA(j,j+ib), ldda, queue);

            magma_zgemm( MagmaNoTrans, MagmaNoTrans, 
                         m-(j+ib), n-(j+ib), ib, 
                         c_neg_one, dA(ib+j, j   ), ldda, 
                                    dA(j   , ib+j), ldda, 
                         c_one,     dA(ib+j, ib+j), ldda, 
                                    queue);
        }
    }

    return 0;
}

magma_int_t
magma_zgetf2_native_recursive(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *dipiv, magma_int_t *dipivinfo, 
    magma_int_t *dinfo, magma_int_t gbstep, 
    magma_queue_t queue, magma_queue_t update_queue)
{
    magma_int_t arginfo = 0;
    if (m < 0 || m > ZGETF2_FUSED_MAX_M) {
        arginfo = -1;
    } else if (n < 0 ) {
        arginfo = -2;
    } else if (ldda < max(1,m)) {
        arginfo = -4;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    magma_event_t events[2]; 
    magma_event_create( &events[0] );
    magma_event_create( &events[1] );

    magma_int_t nb;
    magma_int_t sm_count = magma_getdevice_multiprocessor_count();
    if     (sm_count >= 32){nb = 32;}
    else if(sm_count >= 16){nb = 16;}
    else if(sm_count >=  8){nb =  8;}
    else if(sm_count >=  4){nb =  4;}
    else if(sm_count >=  2){nb =  2;}
    else                   {nb =  1;}

    if( n <= nb){
        magma_int_t* update_flags = dipivinfo; 
        // wait for all kernels in the update queue to end before calling the panel kernel
        magma_event_record( events[0], update_queue );
        magma_queue_wait_event( queue, events[0] );
        magma_zgetf2_native_fused( m, n, dA(0,0), ldda, dipiv, gbstep, update_flags, dinfo, queue );
        magma_event_record( events[1], queue ); 
        magma_queue_wait_event( update_queue, events[1] );
        return 0;
    }
    else{
        magma_int_t n1 = n / 2;
        magma_int_t n2 = n - n1;

        // lu on A1
        magma_zgetf2_native_recursive(m, n1, dA(0,0), ldda, dipiv, dipivinfo, dinfo, gbstep, queue, update_queue);

        // swap left
        #ifdef PARSWAP
        setup_pivinfo( dipivinfo, dipiv, m, n1, queue);  // setup pivinfo
        magma_zlaswp_rowparallel_native( n2, dA(0,n1), ldda, dA(0,n1), ldda, 0, n1, dipivinfo, queue);
        #else
        magma_zlaswp_rowserial_native(n2, dA(0,n1), ldda, 0, n1, dipiv, queue);
        #endif

        // update (trsm + gemm)
        magma_zgetf2trsm_2d_native( n1, n2, 
                                    dA(0,0), ldda, 
                                    dA(0,n1), ldda, queue);

        magma_zgemm( MagmaNoTrans, MagmaNoTrans, 
                     m-n1, n2, n1, 
                     MAGMA_Z_NEG_ONE, dA(n1,  0), ldda, 
                                      dA(0 , n1), ldda, 
                     MAGMA_Z_ONE,     dA(n1, n1), ldda, queue );

        // lu on A2 
        magma_zgetf2_native_recursive(m-n1, n2, dA(n1,n1), ldda, dipiv+n1, dipivinfo, dinfo, gbstep, queue, update_queue );

        // swap right: if PARSWAP is set, we need to call setup_pivinfo
        #ifdef PARSWAP
        setup_pivinfo( dipivinfo, dipiv+n1, m-n1, n2, queue);  // setup pivinfo
        #endif

        adjust_ipiv( dipiv+n1, n2, n1, queue);

        #ifdef PARSWAP
        magma_zlaswp_rowparallel_native(n1, dA(n1,0), ldda, dA(n1,0), ldda, n1, n, dipivinfo, queue);
        #else
        magma_zlaswp_rowserial_native(n1, dA(0,0), ldda, n1, n, dipiv, queue);
        #endif
    }

    magma_event_destroy( events[0] );
    magma_event_destroy( events[1] );

    return 0;
}
/***************************************************************************//**
    Purpose
    -------
    ZGETF2 computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a GPU-only routine. The host CPU is not used. 

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of each matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix A.  N >= 0.

    @param[in,out]
    dA      A COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of A.  LDDA >= max(1,M).

    @param[out]
    dipiv   An INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    dipivinfo  An INTEGER array, for internal use.

    @param[out]
    dinfo         INTEGER, stored on the GPU
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    gbstep  INTEGER
            internal use.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @param[in]
    update_queue   magma_queue_t
                   Internal use.

    This is an internal routine.

    @ingroup magma_getf2_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetf2_native(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t *dipiv, magma_int_t* dipivinfo, 
    magma_int_t *dinfo, magma_int_t gbstep, 
    magma_queue_t queue, magma_queue_t update_queue)
{
    magma_int_t arch = magma_getdevice_arch();
    if(m > ZGETF2_FUSED_MAX_M || arch < 300){
      magma_zgetf2_native_blocked(m, n, dA, ldda, dipiv, dinfo, gbstep, queue);
    }
    else{
      magma_zgetf2_native_recursive(m, n, dA, ldda, dipiv, dipivinfo, dinfo, gbstep, queue, update_queue);
    }    
    return 0;
}
