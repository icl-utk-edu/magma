/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Azzam Haidar
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/
#define PRECISION_z

#include "magma_internal.h"
#include "batched_kernel_param.h"
/******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_lg_vbatched(
    magma_uplo_t uplo, magma_int_t* n, magma_int_t max_n, 
    magmaDoubleComplex **dA_array, magma_int_t* ldda,
    magma_int_t *info_array,  magma_int_t batchCount, magma_queue_t queue)
{
    double d_alpha = -1.0;
    double d_beta  = 1.0;
    
    magma_int_t arginfo = 0;
    magma_int_t j, k;
    magma_int_t nb = POTRF_NB;
    
    // aux integer vector 
    magma_int_t *njvec=NULL; 
    magma_int_t *ibvec=NULL;
    magma_int_t *dinvA_lvec=NULL;
    magma_int_t *dwork_lvec=NULL;
    magma_int_t *dw_aux=NULL;
    magma_int_t *jibvec=NULL;  
    magma_int_t *dinvA_batch_offset = NULL;
    magma_int_t *dwork_batch_offset = NULL;
    magma_int_t *ncpu = NULL;
    magma_int_t *lda = NULL; 
    magma_imalloc( &njvec,  batchCount );
    magma_imalloc( &ibvec, batchCount );
    magma_imalloc( &dinvA_lvec, batchCount );
    magma_imalloc( &dwork_lvec, batchCount );
    magma_imalloc( &dw_aux, batchCount );
    magma_imalloc( &jibvec, batchCount );
    magma_imalloc( &dinvA_batch_offset, batchCount );
    magma_imalloc( &dwork_batch_offset, batchCount );
    magma_imalloc_cpu( &lda, batchCount );
    magma_imalloc_cpu( &ncpu,  batchCount ); 
   
    magmaDoubleComplex **dA_displ    = NULL;
    magmaDoubleComplex **dW0_displ   = NULL;
    magmaDoubleComplex **dW1_displ   = NULL;
    magmaDoubleComplex **dW2_displ   = NULL;
    magmaDoubleComplex **dW3_displ   = NULL;
    magmaDoubleComplex **dW4_displ   = NULL;
    magmaDoubleComplex **dinvA_array = NULL;
    magmaDoubleComplex **dwork_array = NULL;
    magmaDoubleComplex **hA_array   = NULL;

    magma_malloc((void**)&dA_displ,   batchCount * sizeof(*dA_displ));
    magma_malloc((void**)&dW0_displ,  batchCount * sizeof(*dW0_displ));
    magma_malloc((void**)&dW1_displ,  batchCount * sizeof(*dW1_displ));
    magma_malloc((void**)&dW2_displ,  batchCount * sizeof(*dW2_displ));
    magma_malloc((void**)&dW3_displ,  batchCount * sizeof(*dW3_displ));
    magma_malloc((void**)&dW4_displ,  batchCount * sizeof(*dW4_displ));
    magma_malloc((void**)&dinvA_array, batchCount * sizeof(*dinvA_array));
    magma_malloc((void**)&dwork_array,    batchCount * sizeof(*dwork_array));
    magma_malloc_cpu((void**) &hA_array, batchCount*sizeof(magmaDoubleComplex*));

    // check allocation
    if ( dA_displ   == NULL || dW0_displ == NULL || dW1_displ   == NULL || dW2_displ   == NULL || 
         dW3_displ  == NULL || dW4_displ == NULL || dinvA_array == NULL || dwork_array == NULL || 
         hA_array  == NULL || njvec     == NULL || ibvec       == NULL || dinvA_lvec  == NULL || 
         dwork_lvec == NULL || ncpu      == NULL || lda         == NULL || jibvec      == NULL || 
         dwork_batch_offset == NULL || dinvA_batch_offset == NULL) {
        magma_free(dA_displ);
        magma_free(dW0_displ);
        magma_free(dW1_displ);
        magma_free(dW2_displ);
        magma_free(dW3_displ);
        magma_free(dW4_displ);
        magma_free(dinvA_array);
        magma_free(dwork_array);
        magma_free(njvec);
        magma_free(ibvec);
        magma_free(dinvA_lvec);
        magma_free(dwork_lvec);
        magma_free(dw_aux);
        magma_free(jibvec);
        magma_free(dwork_batch_offset);
        magma_free(dinvA_batch_offset);
        
        magma_free_cpu(hA_array);
        magma_free_cpu(ncpu);
        magma_free_cpu(lda);
        magma_int_t info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    // compute the total size for invA = sum of magma_roundup( n[], ZTRTRI_BATCHED_NB )*ZTRTRI_BATCHED_NB; 
    magma_ivec_roundup( batchCount, n, ZTRTRI_BATCHED_NB, dinvA_lvec, queue); 
    magma_ivec_mulc( batchCount, dinvA_lvec, ZTRTRI_BATCHED_NB, dinvA_lvec, queue);
    magma_int_t total_invA_msize = magma_isum_reduce( batchCount, dinvA_lvec, dw_aux, batchCount, queue);
                  
    // compute the size of dwork = sum of n[] * nb
    magma_int_t total_dwork_msize = nb * magma_isum_reduce(batchCount, n, dw_aux, batchCount, queue); 
    magma_ivec_mulc( batchCount, n, nb, dwork_lvec, queue);
    
    // dinvA and dwork are workspace in ztrsm
    magmaDoubleComplex* dinvA      = NULL;
    magmaDoubleComplex* dwork      = NULL; 
    magma_zmalloc( &dinvA, total_invA_msize);
    magma_zmalloc( &dwork, total_dwork_msize );
    
    // check allocation of dinvA and dwork
    if(dinvA == NULL || dwork == NULL)
    {
        magma_free( dinvA );
        magma_free( dwork );
        magma_int_t info = MAGMA_ERR_DEVICE_ALLOC;
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    // dimensions to pass to zlaset
    magma_int_t dinvA_m, dinvA_n, dwork_m, dwork_n; 
    dinvA_m = total_invA_msize  / ZTRTRI_BATCHED_NB; // full division, no remainder
    dwork_m = total_dwork_msize / nb;             // full division, no remainder
    dinvA_n = ZTRTRI_BATCHED_NB; 
    dwork_n = nb; 
    magmablas_zlaset( MagmaFull, dinvA_m, dinvA_n, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dinvA, dinvA_m, queue );
    magmablas_zlaset( MagmaFull, dwork_m, dwork_n, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dwork, dwork_m, queue );
    
    // To init dwork_array and dinvA_array, we need to perform prefix sums on dwork_lvec and dinvA_lvec
    magma_prefix_sum_outofplace_w(dinvA_lvec, dinvA_batch_offset, batchCount, dw_aux, batchCount, queue);
    magma_prefix_sum_outofplace_w(dwork_lvec, dwork_batch_offset, batchCount, dw_aux, batchCount, queue);
    
    // since coulmn displacement is zero, we can pass a vector with dummy values as lda 
    magma_zset_pointer_var_cc( dwork_array, dwork, dw_aux, 0, 0, dwork_batch_offset, batchCount, queue);
    magma_zset_pointer_var_cc( dinvA_array, dinvA, dw_aux, 0, 0, dinvA_batch_offset, batchCount, queue);

    magma_int_t streamid;
    const magma_int_t nbstreams=32;
    magma_queue_t queues[nbstreams];
    magma_device_t cdev;
    bool cpu_arrays_copied = false; 
    
    magma_getdevice( &cdev );
    for(k=0; k<nbstreams; k++){
        magma_queue_create( cdev, &queues[k] );
    }
    
    
    for(j = 0; j < max_n; j+=nb) {
        magma_int_t ib = min(nb, max_n-j);
        //compute ibvec[] = min(nb, n[]-j);
        magma_ivec_addc( batchCount, n, -j, njvec, queue);
        magma_ivec_minc( batchCount, njvec, nb, ibvec, queue);
        //panel
        magma_zdisplace_pointers_var_cc(dA_displ, dA_array, ldda, j, j, batchCount, queue);
        arginfo = magma_zpotrf_panel_vbatched(
                           uplo, njvec, max_n-j, 
                           ibvec, nb,  
                           dA_displ, ldda,
                           dwork_array, dwork_lvec,
                           dinvA_array, dinvA_lvec,
                           dW0_displ, dW1_displ, dW2_displ, 
                           dW3_displ, dW4_displ, 
                           info_array, 0, batchCount, queue);

        if(arginfo != 0 ) goto fin;
        // end of panel
        // compute n[]-j-ibvec[]: subtract ibvec from njvec
        magma_ivec_add( batchCount, 1, njvec, -1, ibvec, njvec, queue);
        magma_int_t njvec_max = max_n-j-ib;
        if( njvec_max > 0 ){
            magma_int_t use_streamed_herk = magma_zrecommend_cublas_gemm_stream(MagmaNoTrans, MagmaTrans, njvec_max, njvec_max, nb);
            if( use_streamed_herk ){ 
                if(cpu_arrays_copied == false) {
                    magma_getvector( batchCount, sizeof(magmaDoubleComplex*), dA_array, 1, hA_array, 1, queue);
                    magma_getvector( batchCount, sizeof(magma_int_t), n, 1, ncpu, 1, queue);
                    magma_getvector( batchCount, sizeof(magma_int_t), ldda, 1, lda, 1, queue);
                    cpu_arrays_copied = true; 
                }
                // USE STREAM  HERK
                for(k=0; k<batchCount; k++){
                    streamid = k%nbstreams;                                       
                    magma_int_t my_ib = min(nb, ncpu[k]-j);
                    if(ncpu[k]-j-my_ib > 0 && my_ib > 0){
                        magma_zherk( MagmaLower, MagmaNoTrans, ncpu[k]-j-my_ib, my_ib, 
                            d_alpha, 
                            (const magmaDoubleComplex*) hA_array[k] + j+my_ib+j*lda[k], lda[k], 
                            d_beta,
                            hA_array[k] + j+my_ib+(j+my_ib)*lda[k], lda[k], queues[streamid] );
                    }
                }
                if (queue != NULL) {
                    for (magma_int_t s=0; s < nbstreams; s++)
                        magma_queue_sync( queues[s] );
                }
            }
            else{
                // use magmablas
                // displacements here are correct unless at the last step (where herk is not actually performed) 
                magma_zdisplace_pointers_var_cc(dA_displ, dA_array, ldda, j+ib, j, batchCount, queue);
                magma_zdisplace_pointers_var_cc(dW1_displ, dA_array, ldda, j+ib, j+ib, batchCount, queue);
                magmablas_zherk_vbatched_max_nocheck( uplo, MagmaNoTrans, njvec, ibvec, 
                                         d_alpha, dA_displ, ldda,
                                         d_beta, dW1_displ, ldda,  
                                         batchCount, njvec_max, nb, queue );
            }
        } 
    }

fin:
    magma_queue_sync(queue);
    for(k=0; k<nbstreams; k++){
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
    magma_free(njvec);
    magma_free(ibvec);
    magma_free(dinvA_lvec);
    magma_free(dwork_lvec);
    magma_free(dw_aux);
    magma_free(jibvec);
    magma_free(dinvA_batch_offset);
    magma_free(dwork_batch_offset);
    magma_free_cpu(ncpu);
    magma_free_cpu(hA_array);

    return arginfo;
}

/******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_vbatched_max_nocheck(
    magma_uplo_t uplo, magma_int_t *n, 
    magmaDoubleComplex **dA_array, magma_int_t *ldda,
    magma_int_t *info_array,  magma_int_t batchCount, 
    magma_int_t max_n, magma_queue_t queue)
{   
    magma_int_t arginfo = 0;
    magma_int_t crossover = magma_get_zpotrf_vbatched_crossover();   
    if( max_n > crossover ) {   
        arginfo = magma_zpotrf_lg_vbatched(uplo, n, max_n, dA_array, ldda, info_array,  batchCount, queue);
    }
    else{
        arginfo = magma_zpotrf_lpout_vbatched(uplo, n, max_n, dA_array, ldda, 0, info_array, batchCount, queue);
    }
    magma_queue_sync(queue);
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
    This is the variable size batched version of the operation. 

    Arguments
    ---------
    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.
            Only MagmaLower is supported.

    @param[in]
    n       INTEGER array, dimension(batchCount + 1).
            Each element N specifies the order of each matrix A.  N >= 0.

    @param[in,out]
    dA_array      Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array A on the GPU, dimension (LDDA,N)
             On entry, each pointer is a Hermitian matrix A.  
             If UPLO = MagmaUpper, the leading
             N-by-N upper triangular part of A contains the upper
             triangular part of the matrix A, and the strictly lower
             triangular part of A is not referenced.  If UPLO = MagmaLower, the
             leading N-by-N lower triangular part of A contains the lower
             triangular part of the matrix A, and the strictly upper
             triangular part of A is not referenced.
    \n
             On exit, if corresponding entry in info_array = 0, 
             each pointer is the factor U or L from the Cholesky
             factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER array, dimension(batchCount + 1).
            Each element LDDA specifies the leading dimension of each array A.  LDDA >= max(1,N).
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
magma_zpotrf_vbatched(
    magma_uplo_t uplo, magma_int_t *n, 
    magmaDoubleComplex **dA_array, magma_int_t *ldda,
    magma_int_t *info_array,  magma_int_t batchCount, 
    magma_queue_t queue)
{   
    magma_int_t max_n, arginfo = 0;
    
    arginfo =  magma_potrf_vbatched_checker( uplo, n, ldda, batchCount, queue );   
    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // compute the max. dimensions
    magma_imax_size_1(n, batchCount, queue); 
    magma_getvector(1, sizeof(magma_int_t), &n[batchCount], 1, &max_n, 1, queue);
    
    arginfo = magma_zpotrf_vbatched_max_nocheck(uplo, n, dA_array, ldda, info_array,  batchCount, max_n, queue);
    
    return arginfo;
}
