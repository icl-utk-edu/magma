/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       

*/
#include <cuda.h>    // for CUDA_VERSION
#include <cuda_runtime.h>
#include "magma_internal.h"
//#include "nvToolsExt.h"
//#define MAGMA_PRINTF printf
#define MAGMA_PRINTF(...)
/***************************************************************************//**
    Purpose
    -------
    XSHGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges. It uses mixed precision 
    FP32/FP16-w/o TensorCores factorization techniques.


    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.
    
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      REAL array on the GPU, dimension (LDDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

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

    @param[in]
    enable_tc  MAGMA_MP_TYPE_T
               internal and expert API uses. enable/disable tensor cores

    @param[in]
    mp_algo_type  MAGMA_MP_TYPE_T
               internal and expert API uses.

    More details can be found in 
    Azzam Haidar, Stanimire Tomov, Jack Dongarra, and Nicholas J. Higham. 2018. 
    Harnessing GPU tensor cores for fast FP16 arithmetic to speed up mixed-precision 
    iterative refinement solvers. In Proceedings of the International Conference for 
    High Performance Computing, Networking, Storage, and Analysis (SC '18). 
    IEEE Press, Piscataway, NJ, USA, Article 47, 11 pages.
    
    @ingroup magma_getrf
*******************************************************************************/
extern "C" magma_int_t
magma_xshgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info,
    magma_mp_type_t enable_tc,
    magma_mp_type_t mp_algo_type )
{
#if CUDA_VERSION >= 9000
    #ifdef HAVE_clBLAS
    #define  dA(i_, j_) dA,  (dA_offset  + (i_)       + (j_)*ldda)
    #define dAT(i_, j_) dAT, (dAT_offset + (i_)*lddat + (j_))
    #define dAP(i_, j_) dAP, (             (i_)          + (j_)*maxm)
    #else
    #define  dA(i_, j_) (dA  + (i_)       + (j_)*ldda)
    #define dAT(i_, j_) (dAT + (i_)*lddat + (j_))
    #define dAP(i_, j_) (dAP + (i_)       + (j_)*maxm)
    #endif

    float c_one     = MAGMA_S_ONE;
    float c_neg_one = MAGMA_S_NEG_ONE;

    magma_int_t iinfo, nb, jb, nextj, nextjb;
    magma_int_t maxm, maxn, minmn, maxnb;
    magma_int_t i, j, rows, lddat, ldwork;
    magmaFloat_ptr dAT=NULL, dAP=NULL, work=NULL;
    
    cublasMath_t mode;
    cublasStatus_t cuerr;
    cublasGemmAlgo_t ALGO = CUBLAS_GEMM_DFALT;
    magmaHalf *dApanel_hp=NULL, *dAtrsm1_hp=NULL, *dAtrsm2_hp=NULL, *dwork_hp=NULL;

    /* Check arguments */
    *info = 0;
    if (m < 0)
        *info = -1;
    else if (n < 0)
        *info = -2;
    else if (ldda < max(1,m))
        *info = -4;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* Function Body */
    minmn = min( m, n );
    maxnb = 512;
    nb    = magma_get_xgetrf_nb( m, n, n, enable_tc, mp_algo_type );

    magma_queue_t queues[2] = { NULL };
    magma_event_t event[2] = { NULL };
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    magma_event_create( &event[0] );
    magma_event_create( &event[1] );

    // can be replaced by the magma cross over
    if (nb <= 1 || nb >= min(m,n)) {
        /* Use CPU code. */
        if ( MAGMA_SUCCESS != magma_smalloc_cpu( &work, m*n )) {
            *info = MAGMA_ERR_HOST_ALLOC;
            goto cleanup;
        }
        magma_sgetmatrix( m, n, dA(0,0), ldda, work, m, queues[0] );
        lapackf77_sgetrf( &m, &n, work, &m, ipiv, info );
        magma_ssetmatrix( m, n, work, m, dA(0,0), ldda, queues[0] );
        magma_free_cpu( work );  work=NULL;
    }
    else {
        /* Use hybrid blocked code. */
        maxm = magma_roundup( m, 32 );
        maxn = magma_roundup( n, 32 );

        if (MAGMA_SUCCESS != magma_smalloc( &dAP, maxnb*maxm )) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            goto cleanup;
        }

        // square matrices can be done in place;
        // rectangular requires copy to transpose
        if ( m == n ) {
            dAT = dA;
            lddat = ldda;
            magmablas_stranspose_inplace( m, dAT(0,0), lddat, queues[0] );
        }
        else {
            lddat = maxn;  // N-by-M
            if (MAGMA_SUCCESS != magma_smalloc( &dAT, lddat*maxm )) {
                *info = MAGMA_ERR_DEVICE_ALLOC;
                goto cleanup;
            }
            magmablas_stranspose( m, n, dA(0,0), ldda, dAT(0,0), lddat, queues[0] );
        }
        magma_queue_sync( queues[0] );  // finish transpose

        ldwork = maxm;
        if (MAGMA_SUCCESS != magma_smalloc_pinned( &work, ldwork*maxnb )) {
            *info = MAGMA_ERR_HOST_ALLOC;
            goto cleanup;
        }

        //---------------------------------------------------
        if( mp_algo_type == Magma_MP_GEMEX_I16_O32_C32 )
        {
        magma_malloc( (void**) &dwork_hp,  maxnb*(maxm+maxn+maxnb)*sizeof(magmaHalf) );
        dApanel_hp = dwork_hp;
        dAtrsm1_hp = dApanel_hp + maxm*maxnb;
        dAtrsm2_hp = dAtrsm1_hp + maxnb*maxnb;
        }
        //---------------------------------------------------

        for( j=0; j < minmn; j += jb ) {
            jb = min(nb, minmn-j);
            rows = m - j;
            if(j==0)
            {
                // transpose the panel and send it to CPU
                magmablas_stranspose( jb, m-j, dAT(j,j), lddat, dAP(0,0), maxm, queues[1] );
                magma_queue_sync( queues[1] );  // wait for transpose
                magma_sgetmatrix_async( m-j, jb, dAP(0,0), maxm, work, ldwork, queues[0] );
            }

            // do the cpu part
            magma_queue_sync( queues[0] );  // wait to get work
            lapackf77_sgetrf( &rows, &jb, work, &ldwork, ipiv+j, &iinfo );
            if ( *info == 0 && iinfo > 0 ){
                *info = iinfo + j;
                printf("error sgetrf inside xshgetrf voici info %d\n", (int)*info);
                goto cleanup;
            }

            magma_ssetmatrix_async( m-j, jb, work, ldwork, dAP, maxm, queues[0] );
            for( i=j; i < j + jb; ++i ) {
                ipiv[i] += j;
            }
            magmablas_slaswp( n, dAT(0,0), lddat, j + 1, j + jb, ipiv, 1, queues[1] );

            magma_queue_sync( queues[0] );
            magmablas_stranspose( m-j, jb, dAP(0,0), maxm, dAT(j,j), lddat, queues[1] );
            if( mp_algo_type == Magma_MP_GEMEX_I16_O32_C32 )
            {
                magma_event_record( event[0], queues[1] );
            }
            nextj  = j+jb;
            nb     = magma_get_xgetrf_nb( minmn-nextj, minmn-nextj, jb, enable_tc, mp_algo_type );
            nextjb = min(nb, minmn-nextj);
            
            magma_strsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                    nextjb, jb,
                    c_one, dAT(j, j    ), lddat,
                    dAT(j, nextj), lddat, queues[1] );

            if( (nextjb) > 0 && (m-nextj) > 0 )
            {
                if( enable_tc ==  Magma_MP_ENABLE_TC_MATH )
                {
                    ALGO  = CUBLAS_GEMM_DFALT_TENSOR_OP;
                    cuerr = cublasSetMathMode(queues[1]->cublas_handle(), CUBLAS_TENSOR_OP_MATH);
                }
                if( mp_algo_type == Magma_MP_GEMEX_I16_O32_C32 )
                {
                    magmablas_convert_sp2hp(nextjb, jb,
                            dAT(j, nextj), lddat, 
                            dAtrsm1_hp, maxnb, queues[1]);
                    magma_queue_wait_event(queues[0], event[0]);
                    magmablas_convert_sp2hp(jb, m-nextj,
                            dAT(nextj, j), lddat, 
                            dApanel_hp, jb, queues[0]);
                    magma_queue_sync( queues[0] );
                    cublasGemmEx( queues[1]->cublas_handle(), 
                            cublas_trans_const( MagmaNoTrans ), cublas_trans_const( MagmaNoTrans ),
                            int(nextjb), int(m-nextj), int(jb),
                            &c_neg_one, dAtrsm1_hp,        CUDA_R_16F, int(maxnb),
                            dApanel_hp,        CUDA_R_16F, int(jb),
                            &c_one,     dAT(nextj, nextj), CUDA_R_32F, int(lddat),
                            CUDA_R_32F, ALGO);
                }
                else if( mp_algo_type == Magma_MP_GEMEX_I32_O32_C32 ) {
                    cublasGemmEx( queues[1]->cublas_handle(), 
                            cublas_trans_const( MagmaNoTrans ), cublas_trans_const( MagmaNoTrans ),
                            int(nextjb), int(m-nextj), int(jb),
                            &c_neg_one, dAT(j,     nextj), CUDA_R_32F, int(lddat),
                            dAT(nextj, j    ), CUDA_R_32F, int(lddat),
                            &c_one,     dAT(nextj, nextj), CUDA_R_32F, int(lddat),
                            CUDA_R_32F, ALGO);
                }
                else if( mp_algo_type == Magma_MP_SGEMM ) {
                    magma_sgemm( MagmaNoTrans, MagmaNoTrans,
                            nextjb, m-nextj, jb,
                            c_neg_one, dAT(j,     nextj), lddat,
                            dAT(nextj, j    ), lddat,
                            c_one,     dAT(nextj, nextj), lddat, queues[1] );
                }
                if( enable_tc ==  Magma_MP_ENABLE_TC_MATH )
                {
                    ALGO  = CUBLAS_GEMM_DFALT;
                    cuerr = cublasSetMathMode(queues[1]->cublas_handle(), CUBLAS_DEFAULT_MATH);
                }
            }
            magmablas_stranspose( nextjb, m-nextj, dAT(nextj, nextj), lddat, dAP(0,0), maxm, queues[1] );
            magma_queue_sync( queues[1] );
            magma_sgetmatrix_async( m-nextj, nextjb, dAP(0,0), maxm, work, ldwork, queues[0] );
            
            magma_strsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                    n-(nextj+nextjb), jb,
                    c_one, dAT(j, j           ), lddat,
                    dAT(j, nextj+nextjb), lddat, queues[1] );
            if( (n-(nextj+nextjb)) > 0 && (m-nextj) > 0 )
            {
                if( enable_tc ==  Magma_MP_ENABLE_TC_MATH )
                {
                    ALGO  = CUBLAS_GEMM_DFALT_TENSOR_OP;
                    cuerr = cublasSetMathMode(queues[1]->cublas_handle(), CUBLAS_TENSOR_OP_MATH);
                }
                if( mp_algo_type == Magma_MP_GEMEX_I16_O32_C32 )
                {
                    magmablas_convert_sp2hp(n-(nextj+nextjb), jb,
                            dAT(j, nextj+nextjb), lddat, 
                            dAtrsm2_hp, maxn, queues[1]);

                    cublasGemmEx( queues[1]->cublas_handle(), 
                            cublas_trans_const( MagmaNoTrans ), cublas_trans_const( MagmaNoTrans ),
                            int(maxn-(nextj+nextjb)), int(m-nextj), int(jb),
                            &c_neg_one, dAtrsm2_hp,               CUDA_R_16F, int(maxm),
                            dApanel_hp,               CUDA_R_16F, int(jb),
                            &c_one,     dAT(nextj, nextj+nextjb), CUDA_R_32F, int(lddat),
                            CUDA_R_32F, ALGO);
                }
                else if( mp_algo_type == Magma_MP_GEMEX_I32_O32_C32 ) {
                    cublasGemmEx( queues[1]->cublas_handle(), 
                            cublas_trans_const( MagmaNoTrans ), cublas_trans_const( MagmaNoTrans ),
                            int(maxn-(nextj+nextjb)), int(m-nextj), int(jb),
                            &c_neg_one, dAT(j,     nextj+nextjb), CUDA_R_32F, int(lddat),
                            dAT(nextj, j           ), CUDA_R_32F, int(lddat),
                            &c_one,     dAT(nextj, nextj+nextjb), CUDA_R_32F, int(lddat),
                            CUDA_R_32F, ALGO);
                }
                else if( mp_algo_type == Magma_MP_SGEMM ) {
                    magma_sgemm( MagmaNoTrans, MagmaNoTrans,
                            maxn-(nextj+nextjb), m-nextj, jb,
                            c_neg_one, dAT(j,     nextj+nextjb), lddat,
                            dAT(nextj, j           ), lddat,
                            c_one,     dAT(nextj, nextj+nextjb), lddat, queues[1] );
                }
                if( enable_tc ==  Magma_MP_ENABLE_TC_MATH )
                {
                    ALGO  = CUBLAS_GEMM_DFALT;
                    cuerr = cublasSetMathMode(queues[1]->cublas_handle(), CUBLAS_DEFAULT_MATH);
                }
            }
        }

        
        // undo transpose
        if ( m == n ) {
            magmablas_stranspose_inplace( m, dAT(0,0), lddat, queues[1] );
        }
        else {
            magmablas_stranspose( n, m, dAT(0,0), lddat, dA(0,0), ldda, queues[1] );
        }
    }

cleanup:
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );
    magma_event_destroy( event[0] );
    magma_event_destroy( event[1] );
    
    magma_free( dAP );
    if (m != n) {
        magma_free( dAT );
    }
    magma_free_pinned( work );
    if( mp_algo_type == Magma_MP_GEMEX_I16_O32_C32 )
    {
        magma_free( dwork_hp );
    }

    MAGMA_UNUSED( cuerr );
    MAGMA_UNUSED( mode  );
    return *info;
#else
    return MAGMA_ERR_NOT_SUPPORTED;
#endif
} /* magma_xshgetrf_gpu */

/***************************************************************************//**
    Purpose
    -------
    HTGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges. It uses mixed precision 
    FP32/FP16-TensorCores factorization techniques.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.
    
    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      REAL array on the GPU, dimension (LDDA,N).
            On entry, the M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

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
magma_htgetrf_gpu(
    magma_int_t m, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *ipiv,
    magma_int_t *info )
{
    magma_xshgetrf_gpu(m, n, dA, ldda, ipiv, info, 
            Magma_MP_ENABLE_TC_MATH, Magma_MP_GEMEX_I16_O32_C32);
    return *info;    
}
