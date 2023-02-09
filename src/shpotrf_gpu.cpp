/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah

*/

#include "magma_internal.h"

// rocBLAS has a very slow ssyrk
// switch to magmablas_ssyrk (which internally calls rocBLAS SGEMM)
#if defined(MAGMA_HAVE_HIP)
#define magma_ssyrk    magmablas_ssyrk
#endif

// this flag enables the fp16-accelerated sgemm, which exists in cublasGemmEx,
// but not in hipblas (as of rocm-5.2)
#define CUDA_USE_FAST_SGEMM
static magma_int_t
magma_sgemm_fp16(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t m, magma_int_t n, magma_int_t k,
    float alpha, float* dA, magma_int_t ldda, magmaHalf* dhA, magma_int_t lddha,
                 float* dB, magma_int_t lddb, magmaHalf* dhB, magma_int_t lddhb,
    float beta,  float* dC, magma_int_t lddc,
    magma_queue_t queue )
{
    #if (defined(MAGMA_HAVE_CUDA) && CUDA_VERSION >= 7500)
        #ifdef CUDA_USE_FAST_SGEMM
        cublasGemmEx( queue->cublas_handle(),
                      cublas_trans_const( transA ), cublas_trans_const( transB ),
                      (int)m, (int)n, (int)k,
                      (const void*) &alpha, (const void*) dA, CUDA_R_32F, (int)ldda,
                                            (const void*) dB, CUDA_R_32F, (int)lddb,
                      (const void*) &beta,  (      void*) dC, CUDA_R_32F, (int)lddc,
                      CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
        #else
        cublasGemmEx( queue->cublas_handle(),
            cublas_trans_const( transA ), cublas_trans_const( transB ),
            (int)m, (int)n, (int)k,
            (const void*) &alpha, (const void*) dhA, CUDA_R_16F, (int)lddha,
                                  (const void*) dhB, CUDA_R_16F, (int)lddhb,
            (const void*) &beta,  (      void*) dC,  CUDA_R_32F, (int)lddc,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP );
        #endif
        return 0;
    #elif defined(MAGMA_HAVE_HIP)
    hipblasGemmEx( queue->hipblas_handle(),
		           hipblas_trans_const( transA ), hipblas_trans_const( transB ),
		           int(m), int(n), int(k),
		           (void*)&alpha, (void*)dhA, HIPBLAS_R_16F, (int)lddha,
                                  (void*)dhB, HIPBLAS_R_16F, (int)lddhb,
		           (void*)&beta,  (void*)dC,  HIPBLAS_R_32F, (int)lddc,
		           HIPBLAS_R_32F, HIPBLAS_GEMM_DEFAULT);
    return 0;
    #else
    return MAGMA_ERR_NOT_SUPPORTED;
    #endif
}

/***************************************************************************//**
    Purpose
    -------
    SPOTRF computes the Cholesky factorization of a real symmetric
    positive definite matrix dA.

    The factorization has the form
        dA = U**H * U,   if UPLO = MagmaUpper, or
        dA = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    This factorization uses half precision during the trailing matrix
    updates, thus its accuracy is not up to FP32 precision. It is used in
    mixed precision solvers exploiting half precision.

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.

    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.

    @param[in,out]
    dA      REAL array on the GPU, dimension (LDDA,N)
            On entry, the symmetric matrix dA.  If UPLO = MagmaUpper, the leading
            N-by-N upper triangular part of dA contains the upper
            triangular part of the matrix dA, and the strictly lower
            triangular part of dA is not referenced.  If UPLO = MagmaLower, the
            leading N-by-N lower triangular part of dA contains the lower
            triangular part of the matrix dA, and the strictly upper
            triangular part of dA is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @param[in]
    mode    magma_mode_t
      -     = MagmaNative:  Factorize dA using GPU only mode (only uplo=MagmaLower is available);
      -     = MagmaHybrid:  Factorize dA using Hybrid (CPU/GPU) mode.

    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_shpotrf_LL_expert_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t nb, magma_int_t recnb,
    magma_int_t *info, magma_mode_t mode )
{
    #define  dA(i_, j_) (dA  + (i_) + (j_)*ldda)
    #define dhW(i_, j_) (dhW + (i_) + (j_)*lddha)

    /* Constants */
    const float c_one     = MAGMA_S_ONE;
    const float c_neg_one = MAGMA_S_NEG_ONE;
    const float d_one     =  1.0;
    const float d_neg_one = -1.0;

    /* Local variables */
    magma_int_t j, jb;
    magma_int_t *dinfo=NULL;
    float *work=NULL;
    magmaHalf *dhW=NULL;

    *info = 0;
    if (uplo != MagmaLower) {
        printf("only uplo = MagmaLower is supported\n");
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_event_t events[2];
    // half precision workspace
    magmaHalf* dhA     = NULL;
    magmaHalf* dhB     = NULL;
    magma_int_t lddha  = magma_roundup(n, 128);
    #if defined(MAGMA_HAVE_HIP) || ( defined(MAGMA_HAVE_CUDA) && !defined(CUDA_USE_FAST_SGEMM) )
    magma_int_t lhwork = lddha * n;
    if( MAGMA_SUCCESS != magma_malloc( (void**)&dhW, lhwork*sizeof(magmaHalf)) ) {
        *info = MAGMA_ERR_HOST_ALLOC;
        goto cleanup;
    }
    #endif
    dhA = dhW;
    dhB = dhW;
    //#endif

    if (mode == MagmaHybrid) {
        if ( MAGMA_SUCCESS != magma_smalloc_pinned( &work, nb*nb ) ) {
            *info = MAGMA_ERR_HOST_ALLOC;
            goto cleanup;
        }
    }
    else {
        if (MAGMA_SUCCESS != magma_imalloc( &dinfo, 1 ) ) {
            *info = MAGMA_ERR_DEVICE_ALLOC;
            goto cleanup;
        }
    }

    magma_queue_t queues[2];
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    magma_queue_create( cdev, &queues[1] );
    magma_event_create(&events[0]);
    magma_event_create(&events[1]);
    if (mode == MagmaNative)
        magma_setvector( 1, sizeof(magma_int_t), info, 1, dinfo, 1, queues[0]);

    // Compute the Cholesky factorization A = L*L'.
    for (j=0; j < n; j += nb) {
        // apply all previous updates to diagonal block,
        // then transfer it to CPU (if hybrid)
        jb = min( nb, n-j );
        magma_ssyrk( MagmaLower, MagmaNoTrans, jb, j,
                     d_neg_one, dA(j, 0), ldda,
                     d_one,     dA(j, j), ldda, queues[0] );

        if (mode == MagmaHybrid) {
            magma_sgetmatrix_async( jb, jb,
                                    dA(j, j), ldda,
                                    work,     jb, queues[0] );
        }
        else {
            magma_spotrf_rectile_native(MagmaLower, jb, recnb,
                                        dA(j, j), ldda, j,
                                        dinfo, info, queues[0] );
            if(*info != 0) {
                return *info;
                break;
            }
        }

        // apply all previous updates to block column below diagonal block
        if (j+jb < n) {
            magma_queue_wait_event(queues[1], events[0]);
            dhA = dhW(j+jb, 0);
            dhB = dhW(j,    0);
            magma_sgemm_fp16( MagmaNoTrans, MagmaConjTrans,
                         n-j-jb, jb, j,
                         c_neg_one, dA(j+jb, 0), ldda, dhA, lddha,
                                    dA(j,    0), ldda, dhB, lddha,
                         c_one,     dA(j+jb, j), ldda, queues[1] );
            magma_event_record(events[1], queues[1]);
        }

        // simultaneous with above sgemm, transfer diagonal block,
        // factor it on CPU, and test for positive definiteness
        // Azzam: The above section can be moved here the code will look cleaner.
        if (mode == MagmaHybrid) {
            magma_queue_sync( queues[0] );
            lapackf77_spotrf( MagmaLowerStr, &jb, work, &jb, info );
            magma_ssetmatrix_async( jb, jb,
                                    work,     jb,
                                    dA(j, j), ldda, queues[0] );
            if (*info != 0) {
                *info = *info + j;
                break;
            }
        }

        // apply diagonal block to block column below diagonal
        if (j+jb < n) {
            magma_queue_wait_event(queues[0], events[1]);
            magma_strsm( MagmaRight, MagmaLower, MagmaConjTrans, MagmaNonUnit,
                         n-j-jb, jb,
                         c_one, dA(j,    j), ldda,
                                dA(j+jb, j), ldda, queues[0] );

            #if defined(MAGMA_HAVE_HIP) || ( defined(MAGMA_HAVE_CUDA) && !defined(CUDA_USE_FAST_SGEMM) )
            magma_int_t s2h_info;
            magmablas_slag2h(n-j, jb, dA(j, j), ldda, dhW(j, j), lddha, &s2h_info, queues[0]);
            #endif

            magma_event_record(events[0], queues[0]);
        }
    }

    if (mode == MagmaNative)
        magma_getvector( 1, sizeof(magma_int_t), dinfo, 1, info, 1, queues[0]);

cleanup:
    magma_queue_sync( queues[0] );
    magma_queue_sync( queues[1] );
    magma_event_destroy( events[0] );
    magma_event_destroy( events[1] );
    magma_queue_destroy( queues[0] );
    magma_queue_destroy( queues[1] );

    if (mode == MagmaHybrid) {
        if(work)  magma_free_pinned( work );
    }
    else {
        if(dinfo) magma_free( dinfo );
    }

    if(dhW) magma_free(dhW);

    return *info;
} /* magma_shpotrf_LL_expert_gpu */

/***************************************************************************//**
    magma_shpotrf_LL_expert_gpu with mode = MagmaHybrid.
    Computation is hybrid, part on CPU (panels), part on GPU (matrix updates).
    @see magma_shpotrf_LL_expert_gpu
    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_shpotrf_gpu(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magma_mode_t mode = MagmaHybrid;
    magma_int_t nb = 512;
    magma_int_t recnb = 128;
    magma_shpotrf_LL_expert_gpu(uplo, n, dA, ldda, nb, recnb, info, mode);
    return *info;
}

/***************************************************************************//**
    magma_shpotrf_LL_expert_gpu with mode = MagmaNative.
    Computation is done only on the GPU, not on the CPU.
    @see magma_shpotrf_LL_expert_gpu
    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_shpotrf_native(
    magma_uplo_t uplo, magma_int_t n,
    magmaFloat_ptr dA, magma_int_t ldda,
    magma_int_t *info )
{
    magma_mode_t mode = MagmaNative;
    magma_int_t nb = 512;
    magma_int_t recnb = 128;
    magma_shpotrf_LL_expert_gpu(uplo, n, dA, ldda, nb, recnb, info, mode);
    return *info;
}
