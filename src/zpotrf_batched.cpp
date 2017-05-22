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
    magma_int_t arginfo = 0;

#define A(i_, j_)  (A + (i_) + (j_)*ldda)   
    double d_alpha = -1.0;
    double d_beta  = 1.0;

    if ( n > 2048 ) {
        #ifndef MAGMA_NOWARNING
        printf("=========================================================================================\n"
               "   WARNING batched routines are designed for small sizes. It might be better to use the\n"
               "   Native/Hybrid classical routines if you want good performance.\n"
               "=========================================================================================\n");
        #endif
    }

    magma_int_t j, k, ib, use_stream;
    magma_int_t nb, recnb;
    magma_get_zpotrf_batched_nbparam(n, &nb, &recnb);

    magmaDoubleComplex **dA_displ   = NULL;
    magmaDoubleComplex **dW0_displ  = NULL;
    magmaDoubleComplex **dW1_displ  = NULL;
    magmaDoubleComplex **dW2_displ  = NULL;
    magmaDoubleComplex **dW3_displ  = NULL;
    magmaDoubleComplex **dW4_displ  = NULL;
    magmaDoubleComplex **dinvA_array = NULL;
    magmaDoubleComplex **dwork_array = NULL;

    magma_malloc((void**)&dA_displ,   batchCount * sizeof(*dA_displ));
    magma_malloc((void**)&dW0_displ,  batchCount * sizeof(*dW0_displ));
    magma_malloc((void**)&dW1_displ,  batchCount * sizeof(*dW1_displ));
    magma_malloc((void**)&dW2_displ,  batchCount * sizeof(*dW2_displ));
    magma_malloc((void**)&dW3_displ,  batchCount * sizeof(*dW3_displ));
    magma_malloc((void**)&dW4_displ,  batchCount * sizeof(*dW4_displ));
    magma_malloc((void**)&dinvA_array, batchCount * sizeof(*dinvA_array));
    magma_malloc((void**)&dwork_array,    batchCount * sizeof(*dwork_array));

    magma_int_t invA_msize = magma_roundup( n, ZTRTRI_BATCHED_NB )*ZTRTRI_BATCHED_NB;
    magma_int_t dwork_msize = n*nb;
    magmaDoubleComplex* dinvA      = NULL;
    magmaDoubleComplex* dwork      = NULL; // dinvA and dwork are workspace in ztrsm
    magmaDoubleComplex **cpuAarray = NULL;
    magma_zmalloc( &dinvA, invA_msize * batchCount);
    magma_zmalloc( &dwork, dwork_msize * batchCount );
    magma_malloc_cpu((void**) &cpuAarray, batchCount*sizeof(magmaDoubleComplex*));
    /* check allocation */
    if ( dA_displ  == NULL || dW0_displ == NULL || dW1_displ   == NULL || dW2_displ   == NULL || 
         dW3_displ == NULL || dW4_displ == NULL || dinvA_array == NULL || dwork_array == NULL || 
         dinvA     == NULL || dwork     == NULL || cpuAarray   == NULL ) {
        magma_free(dA_displ);
        magma_free(dW0_displ);
        magma_free(dW1_displ);
        magma_free(dW2_displ);
        magma_free(dW3_displ);
        magma_free(dW4_displ);
        magma_free(dinvA_array);
        magma_free(dwork_array);
        magma_free( dinvA );
        magma_free( dwork );
        magma_free_cpu(cpuAarray);
        magma_int_t info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(info) );
        return info;
    }
    magmablas_zlaset( MagmaFull, invA_msize, batchCount, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dinvA, invA_msize, queue );
    magmablas_zlaset( MagmaFull, dwork_msize, batchCount, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dwork, dwork_msize, queue );
    magma_zset_pointer( dwork_array, dwork, 1, 0, 0, dwork_msize, batchCount, queue );
    magma_zset_pointer( dinvA_array, dinvA, ZTRTRI_BATCHED_NB, 0, 0, invA_msize, batchCount, queue );


    magma_int_t streamid;
    const magma_int_t nbstreams=10;
    magma_queue_t queues[nbstreams];
    for (k=0; k < nbstreams; k++) {
        magma_device_t cdev;
        magma_getdevice( &cdev );
        magma_queue_create( cdev, &queues[k] );
    }
    magma_getvector( batchCount, sizeof(magmaDoubleComplex*), dA_array, 1, cpuAarray, 1, queue);

    if (uplo == MagmaUpper) {
        printf("Upper side is unavailable\n");
        goto fin;
    }
    else {
        for (j = 0; j < n; j += nb) {
            ib = min(nb, n-j);
#if 1
            //===============================================
            //  panel factorization
            //===============================================
            magma_zdisplace_pointers(dA_displ, dA_array, ldda, j, j, batchCount, queue);
            magma_zset_pointer( dwork_array, dwork, 1, 0, 0, dwork_msize, batchCount, queue );
            magma_zset_pointer( dinvA_array, dinvA, ZTRTRI_BATCHED_NB, 0, 0, invA_msize, batchCount, queue );


            if (recnb == nb)
            {
                arginfo = magma_zpotrf_panel_batched(
                                   uplo, n-j, ib,
                                   dA_displ, ldda,
                                   dwork_array, dwork_msize,
                                   dinvA_array, invA_msize,
                                   dW0_displ, dW1_displ, dW2_displ,
                                   dW3_displ, dW4_displ,
                                   info_array, j, batchCount, queue);
            }
            else {
                //arginfo = magma_zpotrf_rectile_batched(
                arginfo = magma_zpotrf_recpanel_batched(
                                   uplo, n-j, ib, recnb,
                                   dA_displ, ldda,
                                   dwork_array, dwork_msize,
                                   dinvA_array, invA_msize,
                                   dW0_displ, dW1_displ, dW2_displ,
                                   dW3_displ, dW4_displ, 
                                   info_array, j, batchCount, queue);
            }
            if (arginfo != 0 ) goto fin;
            //===============================================
            // end of panel
            //===============================================
#endif            
#if 1
            //real_Double_t gpu_time;
            //gpu_time = magma_sync_wtime(queue);
            if ( (n-j-ib) > 0) {
                use_stream = magma_zrecommend_cublas_gemm_stream(MagmaNoTrans, MagmaConjTrans, n-j-ib, n-j-ib, ib);
                if (use_stream)
                { 
                    //-------------------------------------------
                    //          USE STREAM  HERK
                    //-------------------------------------------
                    // since it use different queue I need to wait the panel.
                    /* you must know the matrix layout inorder to do it */  
                    magma_queue_sync(queue); 
                    for (k=0; k < batchCount; k++)
                    {
                        streamid = k%nbstreams;                                       
                        // call herk, class zherk must call cpu pointer 
                        magma_zherk( MagmaLower, MagmaNoTrans, n-j-ib, ib, 
                            d_alpha, 
                            (const magmaDoubleComplex*) cpuAarray[k] + j+ib+j*ldda, ldda, 
                            d_beta,
                            cpuAarray[k] + j+ib+(j+ib)*ldda, ldda, queues[streamid] );
                    }
                    // need to synchronise to be sure that panel do not start before
                    // finishing the update at least of the next panel
                    // if queue is NULL, no need to sync
                    if (queue != NULL) {
                        for (magma_int_t s=0; s < nbstreams; s++)
                            magma_queue_sync(queues[s]);
                    }
                }
                else
                {
                    //-------------------------------------------
                    //          USE BATCHED GEMM(which is a HERK in fact, since it only access the lower part)
                    //-------------------------------------------
                    magma_zdisplace_pointers(dA_displ, dA_array, ldda, j+ib, j, batchCount, queue);
                    magma_zdisplace_pointers(dW1_displ, dA_array, ldda, j+ib, j+ib, batchCount, queue);
                    magmablas_zherk_batched( uplo, MagmaNoTrans, n-j-ib, ib,
                                          d_alpha, dA_displ, ldda, 
                                          d_beta,  dW1_displ, ldda, 
                                          batchCount, queue );
                }
            } 
            //gpu_time = magma_sync_wtime(queue) - gpu_time;
            //real_Double_t flops = (n-j-ib) * (n-j-ib) * ib / 1e9 * batchCount;
            //real_Double_t gpu_perf = flops / gpu_time;
            //printf("Rows= %lld, Colum=%lld, herk time = %7.2fms, Gflops= %7.2f\n",
            //       (long long)(n-j-ib), (long long) ib, gpu_time*1000, gpu_perf);
#endif
        }
    }

fin:
    magma_queue_sync(queue);
    for (k=0; k < nbstreams; k++) {
        magma_queue_destroy( queues[k] );
    }

    magma_free(dA_displ);
    magma_free(dW0_displ);
    magma_free(dW1_displ);
    magma_free(dW2_displ);
    magma_free(dW3_displ);
    magma_free(dW4_displ);
    magma_free(dinvA_array);
    magma_free(dwork_array);
    magma_free( dinvA );
    magma_free( dwork );
    magma_free_cpu(cpuAarray);

    return arginfo;
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

    if (n > crossover )
    {   
        // The memory allocation/deallocation inside this routine takes about 3-4ms 
        arginfo = magma_zpotrf_lg_batched(uplo, n, dA_array, ldda, info_array, batchCount, queue);
    }
    else
    {
        #if defined(VERSION20)
            arginfo = magma_zpotrf_lpout_batched(uplo, n, dA_array, ldda, 0, info_array, batchCount, queue);
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
