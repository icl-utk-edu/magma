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

/******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_lg_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magma_int_t *info_array,  magma_int_t batchCount, magma_queue_t queue)
{
#define dAarray(i,j)  dA_array, i, j   

    magma_int_t arginfo = 0;
    magma_int_t j, k, ib, use_stream;
    double d_one = 1.0, d_neg_one = -1.0;
    magmaDoubleComplex c_neg_one = MAGMA_Z_MAKE(-1.0, 0);
    magmaDoubleComplex c_one     = MAGMA_Z_MAKE( 1.0, 0);
    magma_device_t cdev;
    magma_getdevice( &cdev );

    // queues for streamed herk
    magma_int_t create_stream, streamid;
    const magma_int_t nbstreams=4;
    magma_queue_t queues[nbstreams];

    // aux array for streamed herk
    magmaDoubleComplex** cpuAarray = NULL;
    magma_malloc_cpu((void**) &cpuAarray, batchCount*sizeof(magmaDoubleComplex*));
    if(cpuAarray == NULL) goto fin;
    magma_getvector( batchCount, sizeof(magmaDoubleComplex*), dA_array, 1, cpuAarray, 1, queue);

    if ( n > 2048 ) {
        #ifndef MAGMA_NOWARNING
        printf("=========================================================================================\n"
               "   WARNING batched routines are designed for small sizes. It might be better to use the\n"
               "   Native/Hybrid classical routines if you want good performance.\n"
               "=========================================================================================\n");
        #endif
    }

    magma_int_t nb, recnb;
    magma_get_zpotrf_batched_nbparam(n, &nb, &recnb);

    // queues for streamed herk
    create_stream = magma_zrecommend_cublas_gemm_stream(MagmaNoTrans, MagmaConjTrans, n-nb, n-nb, nb);
    if(create_stream){
        for (k=0; k < nbstreams; k++) {
            magma_queue_create( cdev, &queues[k] );
        }
    }

    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable\n");
        goto fin;
    }
    else {
        for (j = 0; j < n; j += nb) {
            ib = min(nb, n-j);

            //  panel factorization

            arginfo = magma_zpotrf_recpanel_batched(
                                uplo, n-j, ib, recnb,
                                dAarray(j, j), ldda,
                                info_array, j, batchCount, queue);
            if (arginfo != 0 ) goto fin;

            // update
            if ( (n-j-ib) > 0) {
                use_stream = magma_zrecommend_cublas_gemm_stream(MagmaNoTrans, MagmaConjTrans, n-j-ib, n-j-ib, ib);
                if (use_stream){ 
                    // use streamed herk
                    magma_queue_sync(queue); 
                    for (k=0; k < batchCount; k++){
                        streamid = k%nbstreams;                                       
                        magma_zherk( MagmaLower, MagmaNoTrans, n-j-ib, ib, 
                            d_neg_one, (const magmaDoubleComplex*) cpuAarray[k] + j+ib+j*ldda     , ldda, 
                            d_one,                                 cpuAarray[k] + j+ib+(j+ib)*ldda, ldda, queues[streamid] );
                    }
                    for (magma_int_t s=0; s < nbstreams; s++)
                        magma_queue_sync(queues[s]);
                }
                else{
                    magmablas_zherk_batched_core( uplo, MagmaNoTrans, n-j-ib, ib,
                                          c_neg_one, dAarray(j+ib, j), ldda,
                                                     dAarray(j+ib, j), ldda,  
                                          c_one,     dAarray(j+ib, j+ib), ldda, 
                                          batchCount, queue );
                }
            } 
        }
    }
    if(create_stream){
        for (k=0; k < nbstreams; k++) {
            magma_queue_destroy( queues[k] );
        }
    }

fin:
    magma_queue_sync(queue);
    magma_free_cpu( cpuAarray );
    return arginfo;

#undef dAarray
}


/***************************************************************************//**
    Purpose
    -------
    ZPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix dA.

    The factorization has the form
        dA = U**H * U,   if UPLO = MagmaUpper, or
        dA = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.
    This is the fixed size batched version of the operation. 

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.
            Only MagmaLower is supported.

    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.

    @param[in,out]
    dA_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N)
             On entry, each pointer is a Hermitian matrix dA.  
             If UPLO = MagmaUpper, the leading
             N-by-N upper triangular part of dA contains the upper
             triangular part of the matrix dA, and the strictly lower
             triangular part of dA is not referenced.  If UPLO = MagmaLower, the
             leading N-by-N lower triangular part of dA contains the lower
             triangular part of the matrix dA, and the strictly upper
             triangular part of dA is not referenced.
    \n
             On exit, if corresponding entry in info_array = 0, 
             each pointer is the factor U or L from the Cholesky
             factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER
            The leading dimension of each array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    info_array    Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_potrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_batched(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magma_int_t *info_array,  magma_int_t batchCount, 
    magma_queue_t queue)
{
    cudaMemset(info_array, 0, batchCount*sizeof(magma_int_t));
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
        return arginfo;
    }

    // Quick return if possible
    if (n == 0) {
        return arginfo;
    }
    

    magma_int_t crossover = magma_get_zpotrf_batched_crossover();

    if (n > crossover){   
        arginfo = magma_zpotrf_lg_batched(uplo, n, dA_array, ldda, info_array, batchCount, queue);
    }
    else{
        #if defined(VERSION20)
            arginfo = magma_zpotrf_lpout_batched(uplo, n, dA_array, 0, 0, ldda, 0, info_array, batchCount, queue);
        #elif defined(VERSION33)
            arginfo = magma_zpotrf_v33_batched(uplo, n, dA_array, ldda, info_array, batchCount, queue);
        #elif defined(VERSION31)
            arginfo = magma_zpotrf_lpin_batched(uplo, n, dA_array, ldda, 0, info_array, batchCount, queue);
        #else
            printf("ERROR NO VERSION CHOSEN\n");
        #endif
    }
    magma_queue_sync(queue);

    return arginfo;
}
