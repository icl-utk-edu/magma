/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Adrien Remy

   @precisions normal z -> s d c
 */
#include <cuda_runtime.h>

#include "magma_internal.h"
#include "batched_kernel_param.h"

/***************************************************************************//**
    Purpose
    -------
    ZGETRF computes an LU factorization of a general M-by-N matrix A without pivoting

    The factorization has the form
        A = L * U
    where L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a batched version that factors batchCount M-by-N matrices in parallel.
    dA, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of each matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix A.  N >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_nopiv_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrf_nopiv_batched(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex **dA_array, 
        magma_int_t ldda,
        magma_int_t *info_array, 
        magma_int_t batchCount, magma_queue_t queue)
{
#define A(i_, j_)  (A + (i_) + (j_)*ldda)   
    magma_int_t min_mn = min(m, n);
    cudaMemset(info_array, 0, batchCount*sizeof(magma_int_t));

    /* Check arguments */
    magma_int_t arginfo = 0;
    if (m < 0)
        arginfo = -1;
    else if (n < 0)
        arginfo = -2;
    else if (ldda < max(1,m))
        arginfo = -4;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        if (min_mn == 0 ) return arginfo;

    if ( m >  2048 || n > 2048 ) {
        #ifndef MAGMA_NOWARNING
        printf("=========================================================================================\n"
               "   WARNING batched routines are designed for small sizes. It might be better to use the\n"
               "   Native/Hybrid classical routines if you want good performance.\n"
               "=========================================================================================\n");
        #endif
    }


    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magma_int_t nb, recnb, ib, i, k, pm, use_stream;
    magma_get_zgetrf_batched_nbparam(n, &nb, &recnb);

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
    magma_malloc((void**)&dwork_array, batchCount * sizeof(*dwork_array));

    magma_int_t invA_msize = magma_roundup( n, ZTRTRI_BATCHED_NB )*ZTRTRI_BATCHED_NB;
    magma_int_t dwork_msize = max(m,n)*nb;
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
    magma_zset_pointer( dwork_array, dwork, n, 0, 0, dwork_msize, batchCount, queue );
    magma_zset_pointer( dinvA_array, dinvA, ZTRTRI_BATCHED_NB, 0, 0, invA_msize, batchCount, queue );

    magma_int_t streamid;
    const magma_int_t nbstreams=10;
    magma_queue_t queues[nbstreams];
    for (i=0; i < nbstreams; i++) {
        magma_device_t cdev;
        magma_getdevice( &cdev );
        magma_queue_create( cdev, &queues[i] );
    }
    magma_getvector( batchCount, sizeof(magmaDoubleComplex*), dA_array, 1, cpuAarray, 1, queue);


    for (i = 0; i < min_mn; i += nb) 
    {
        ib = min(nb, min_mn-i);
        pm = m-i;
        magma_zdisplace_pointers(dA_displ, dA_array, ldda, i, i, batchCount, queue);
        magma_zset_pointer( dwork_array, dwork, nb, 0, 0, dwork_msize, batchCount, queue );
#if 0
        /* buggy: TODO */
        arginfo = magma_zgetrf_panel_nopiv_batched(
                pm, ib,
                dA_displ, ldda,
                dwork_array, nb, 
                dinvA_array, invA_msize, 
                dW0_displ, dW1_displ, dW2_displ, 
                dW3_displ, dW4_displ,
                info_array, i,
                batchCount, queue); 
 
#else
        arginfo = magma_zgetrf_recpanel_nopiv_batched(
                pm, ib, 32,
                dA_displ, ldda,
                dwork_array, nb, 
                dinvA_array, invA_msize, 
                dW0_displ, dW1_displ, dW2_displ, 
                dW3_displ, dW4_displ,
                info_array, i,
                batchCount, queue);   
#endif

        if (arginfo != 0 ) goto fin;

#define RUN_ALL
#ifdef RUN_ALL

        if ( (i + ib) < n)
        {
            // swap right side and trsm     
            //magma_zdisplace_pointers(dA_displ, dA_array, ldda, i, i+ib, batchCount);
            magma_zset_pointer( dwork_array, dwork, nb, 0, 0, dwork_msize, batchCount, queue ); // I don't think it is needed Azzam

            magma_zdisplace_pointers(dA_displ, dA_array, ldda, i, i, batchCount, queue);
            magma_zdisplace_pointers(dW0_displ, dA_array, ldda, i, i+ib, batchCount, queue);
            magmablas_ztrsm_work_batched( MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit, 1,
                    ib, n-i-ib,
                    MAGMA_Z_ONE,
                    dA_displ,    ldda, // dA
                    dW0_displ,   ldda, // dB
                    dwork_array,  nb, // dX
                    dinvA_array,  invA_msize, 
                    dW1_displ,   dW2_displ, 
                    dW3_displ,   dW4_displ,
                    1, batchCount, queue );

            if ( (i + ib) < m)
            {    
                // if gemm size is > 160 use a streamed classical cublas gemm since it is faster
                // the batched is faster only when M=N <= 160 for K40c
                //-------------------------------------------
                //          USE STREAM  GEMM
                //-------------------------------------------
                use_stream = magma_zrecommend_cublas_gemm_stream(MagmaNoTrans, MagmaNoTrans, m-i-ib, n-i-ib, ib);

                if (use_stream)
                { 
                    //printf("caling streamed dgemm %lld %lld %lld\n", (long long)(m-i-ib), (long long)(n-i-ib), (long long) ib );

                    // since it use different queue I need to wait the TRSM and swap.
                    magma_queue_sync(queue); 
                    for (k=0; k < batchCount; k++)
                    {
                        streamid = k%nbstreams;                                       
                        magma_zgemm( MagmaNoTrans, MagmaNoTrans, 
                                m-i-ib, n-i-ib, ib,
                                c_neg_one, cpuAarray[k] + (i+ib)+i*ldda, ldda, 
                                           cpuAarray[k] + i+(i+ib)*ldda, ldda,
                                c_one,     cpuAarray[k] + (i+ib)+(i+ib)*ldda, ldda, queues[streamid] );
                    }
                    // need to synchronise to be sure that zgetf2 do not start before
                    // finishing the update at least of the next panel
                    // if queue is NULL, no need to sync
                    if ( queue != NULL ) {
                         for (magma_int_t s=0; s < nbstreams; s++)
                             magma_queue_sync(queues[s]);
                     }
                }
                //-------------------------------------------
                //          USE BATCHED GEMM
                //-------------------------------------------
                else
                {
                    magma_zdisplace_pointers(dA_displ, dA_array,  ldda, i+ib,    i, batchCount, queue);
                    magma_zdisplace_pointers(dW1_displ, dA_array, ldda,    i, i+ib, batchCount, queue);
                    magma_zdisplace_pointers(dW2_displ, dA_array, ldda, i+ib, i+ib, batchCount, queue);
                    //printf("caling batched dgemm %lld %lld %lld\n", (long long)(m-i-ib), (long long)(n-i-ib), (long long) ib );
                    magma_zgemm_batched( MagmaNoTrans, MagmaNoTrans, m-i-ib, n-i-ib, ib, 
                                         c_neg_one, dA_displ,  ldda, 
                                                    dW1_displ, ldda, 
                                         c_one,     dW2_displ, ldda, 
                                         batchCount, queue );
                } // end of batched/streamed gemm
            } // end of if ( (i + ib) < m) 
        } // end of if ( (i + ib) < n)
#endif
    }// end of for

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
