
/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Ahmad Ahmad

   @precisions normal z -> s d c
 */


#ifndef MAGMABLAS_ZPOTF2_DEVICES_Z_H
#define MAGMABLAS_ZPOTF2_DEVICES_Z_H
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

//extern __shared__ magmaDoubleComplex shared_data[];

/******************************************************************************/
static inline void zpotf2_sminout_anywidth_device(const int m, const int n, magmaDoubleComplex *A, const int lda, int* info,
                                                  sycl::nd_item<3> item_ct1)
{
    const int tx = item_ct1.get_local_id(2);
    magmaDoubleComplex factor;
    int linfo = 0;

    #pragma unroll
    for (int iter=0; iter < n; iter++)
    {
        //sqrt(diag) and zdscal
        #ifdef ENABLE_COND1
        if ( tx >= iter && tx < m )
        {
        #endif
            double xreal = MAGMA_Z_REAL(A[iter + iter * lda]);
            linfo = ( linfo == 0 && (xreal <= MAGMA_D_ZERO) ) ? (iter+1) : linfo;
            xreal = sycl::sqrt(xreal);
            factor = MAGMA_Z_MAKE(1.0 / xreal, 0.0);
#ifdef ENABLE_COND1
        }
        #endif
        /*
        DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // must sync to make sure that A[iter + iter * lda]
                            // is read by all threads before modifying it
#ifdef ENABLE_COND1
        if ( tx >= iter && tx < m )
        {
        #endif
            A[ tx + iter * lda ] *= factor; // or use the next line and remove the sync above
            //A[ tx + iter * lda ]  = tx == iter ? MAGMA_Z_MAKE(xreal, 0.0) : A[ tx + iter * lda ] * factor;
        #ifdef ENABLE_COND1
        }
        #endif
        /*
        DPCT1065:2: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // zlacgv: TODO, zherk
        #ifdef ENABLE_COND1
        if ( tx > iter && tx < m )
        {
        #endif
            #pragma unroll
            for (int j=iter+1; j < n; j++)
            {
                A [tx + j * lda] -= A[tx + iter * lda]  *  MAGMA_Z_CONJ(A[iter * lda + j]);
            }
        #ifdef ENABLE_COND1
        }
        #endif
        /*
        DPCT1065:3: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }// end of iter
    // ENABLE_COND1 must be disabled, which the default config., so that the right info is returned
    if(tx == 0) *info = linfo;
    /*
    DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
}


/******************************************************************************/
static inline void zpotf2_sminout_fixsize_device(const int m, magmaDoubleComplex *A, const int lda, int* info,
                                                 sycl::nd_item<3> item_ct1)
{
    const int tx = item_ct1.get_local_id(2);
    magmaDoubleComplex factor;
    int linfo = 0;

    #pragma unroll
    for (int iter=0; iter < POTF2_NB; iter++)
    {
        //sqrt(diag) and zdscal
        #ifdef ENABLE_COND2
        if ( tx >= iter && tx < m )
        {
        #endif
            double xreal = MAGMA_Z_REAL(A[iter + iter * lda]);
            linfo = ( linfo == 0 && (xreal <= MAGMA_D_ZERO || xreal != xreal )) ? (iter+1) : linfo;
            xreal = sycl::sqrt(xreal);
            factor = MAGMA_Z_MAKE(1.0 / xreal, 0.0);
#ifdef ENABLE_COND2
        }
        #endif
        /*
        DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
#ifdef ENABLE_COND2
        if ( tx >= iter && tx < m )
        {
        #endif
            A[ tx + iter * lda ] *= factor;

            //A[ tx + iter * lda ]  = tx == iter ? MAGMA_Z_MAKE(xreal, 0.0) : A[ tx + iter * lda ] * factor;
            //if (tx < POTF2_NB) row[ tx ] = MAGMA_Z_CONJ( A[ tx + iter * lda ] );
            //if (tx < POTF2_NB) A[ iter + tx * lda ] = MAGMA_Z_CONJ( A[ tx + iter * lda ] );
        #ifdef ENABLE_COND2
        }
        #endif

        /*
        DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // zherk
        #ifdef ENABLE_COND2
        if ( tx > iter && tx < m )
        {
        #endif
            #pragma unroll
            for (int j=iter+1; j < POTF2_NB; j++)
            {
                A [tx + j * lda] -= A[tx + iter * lda]  *  MAGMA_Z_CONJ(A[iter * lda + j]);
                //A [tx + j * lda] -= A[tx + iter * lda]  *  row[j];
                //A [tx + j * lda] -= A[tx + iter * lda]  *  A[iter +lda * j];
            }
        #ifdef ENABLE_COND2
        }
        #endif
        /*
        DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }// end of iter
    // ENABLE_COND1 must be disabled, which the default config., so that the right info is returned
    if(tx == 0) *info = linfo;
    /*
    DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
}


/******************************************************************************/
static inline void zgemm_v20_1_fixsize_device(int m, int k,
        const magmaDoubleComplex* __restrict__ A0, const int lda,
        magmaDoubleComplex *sC, magmaDoubleComplex  *sB,
        sycl::nd_item<3> item_ct1)
{
    const int tx = item_ct1.get_local_id(2);
    magmaDoubleComplex rC[POTF2_NB];
    magmaDoubleComplex rA[POTF2_NB];
    magmaDoubleComplex rp[POTF2_NB];

    // prefetch next block.
    #ifdef ENABLE_COND4
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            rp[i] = A0[tx + i * lda];
            /*
            DPCT1064:10: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rC[i] = MAGMA_Z_ZERO;
        }
    #ifdef ENABLE_COND4
    }
    #endif

    /*
    DPCT1065:8: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // accumulate 
    #pragma unroll
    for (int iter=0; iter < k; iter += POTF2_NB)
    {
        #ifdef ENABLE_COND4
        if (tx < m) 
        {
        #endif
            // rp to rA
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                rA[i] = rp[i];
            }
        #ifdef ENABLE_COND4
        }
        #endif

        // rA to sB
        if (tx < POTF2_NB)
        {
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                sB[tx + i * POTF2_NB] = MAGMA_Z_CONJ(rp[i]);
            }
        }

        /*
        DPCT1065:11: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // prefetch next block. Azzam
        #ifdef ENABLE_COND4
        if (tx < m )
        {
        #endif
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                rp[i] = A0[tx + (i+(iter+POTF2_NB)) * lda];
            }
        #ifdef ENABLE_COND4
        }
        #endif
        //__syncthreads();

        // multiply current block
        #ifdef ENABLE_COND4
        if (tx < m)
        {
        #endif
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                #pragma unroll
                for (int col=0; col < POTF2_NB; col++)
                {
                    // A0 is multiplied by POTF2_NB times
                    rC[col] +=  rA[i] * sB[col + i * POTF2_NB];
                }
            }
        #ifdef ENABLE_COND4
        }
        #endif
        /*
        DPCT1065:12: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }//end of accumulation

    // finalyzing gemm.
    #ifdef ENABLE_COND4
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            sC[tx + i *m] = rp[i] - rC[i];
        }
    #ifdef ENABLE_COND4
    }
    #endif
    /*
    DPCT1065:9: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
}


/******************************************************************************/
static inline void zgemm_v20_1_anywidth_device(int m, int n, int k,
        const magmaDoubleComplex* __restrict__ A0, int lda,
        magmaDoubleComplex *sC, magmaDoubleComplex  *sB,
        sycl::nd_item<3> item_ct1)
{
    const int tx = item_ct1.get_local_id(2);
    magmaDoubleComplex rC[POTF2_NB];
    magmaDoubleComplex rA[POTF2_NB];
    magmaDoubleComplex rp[POTF2_NB];

    const int bound_A = lda*(k+n-1)+m-1;

    // prefetch next block.
    #ifdef ENABLE_COND5
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            rp[i] = A0[min(bound_A, (int)(tx + i * lda))];
            /*
            DPCT1064:15: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            rC[i] = MAGMA_Z_ZERO;
        }
    #ifdef ENABLE_COND5
    }
    #endif

    /*
    DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // accumulate
    #pragma unroll
    for (int iter=0; iter < k; iter += POTF2_NB)
    {
        #ifdef ENABLE_COND5
        if (tx < m)
        {
        #endif
            // rp to rA
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                rA[i] = rp[i];
            }
        #ifdef ENABLE_COND5
        }
        #endif

        // rA to sB
        if (tx < POTF2_NB)
        {
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                sB[tx + i * POTF2_NB] = MAGMA_Z_CONJ(rp[i]);
            }
        }

        /*
        DPCT1065:16: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // prefetch next block. Azzam
        #ifdef ENABLE_COND5
        if (tx < m )
        {      
        #endif
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                rp[i] = A0[min(
                    bound_A,
                    (int)(tx + (i + (iter + POTF2_NB)) *
                                   lda))]; // min(bound,xxx) is to avoid reading
                                           // out of bound
            }
        #ifdef ENABLE_COND5
        }
        #endif
        //__syncthreads();

        // multiply current block
        #ifdef ENABLE_COND5
        if (tx < m) 
        {
        #endif
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                #pragma unroll
                for (int col=0; col < POTF2_NB; col++)
                {
                    // A0 is multiplied by POTF2_NB times
                    rC[col] +=  rA[i] * sB[col + i * POTF2_NB];
                }
            }
        #ifdef ENABLE_COND5
        }
        #endif
        /*
        DPCT1065:17: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }//end of accumulation

    // finalyzing gemm.
    #ifdef ENABLE_COND5
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            sC[tx + i *m] = rp[i] - rC[i];
        }
    #ifdef ENABLE_COND5
    }
    #endif
    /*
    DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
}


/******************************************************************************/
static inline void zpotf2_smlpout_fixwidth_device(const int m,
        magmaDoubleComplex *A0, magmaDoubleComplex *A, int lda,
        const int localstep, const int gbstep,
        magma_int_t *info, sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto shared_data = (magmaDoubleComplex *)dpct_local;

    // checkinfo to avoid computation of the singular matrix
    #ifndef BATCH_DISABLE_CHECKING
    if (*info != 0 ) return;
    #endif

    const int orginfo = (*info);
    int panel_info = 0, newinfo = 0;
    const int tx = item_ct1.get_local_id(2);
    magmaDoubleComplex *sdata_A =
        shared_data + item_ct1.get_local_id(1) * (m + POTF2_NB) * POTF2_NB;
    magmaDoubleComplex *sdata_B = sdata_A + m * POTF2_NB;


    #if 1
    zgemm_v20_1_fixsize_device(m, localstep, A0, lda, sdata_A, sdata_B,
                               item_ct1);
#else
    zgemm_v20_1_anywidth_device(m, POTF2_NB, localstep, 
                       A0, lda, sdata_A, sdata_B);
    #endif

    // panel fact. in shared memory
    zpotf2_sminout_fixsize_device(m, sdata_A, m, &panel_info, item_ct1);
    //----------------------------------------------------
    // Check for not SPD generating info
    #ifndef BATCH_DISABLE_CHECKING
    if(tx == 0) {
        newinfo = ( orginfo == 0 && panel_info != 0 ) ? panel_info + localstep + gbstep : orginfo;
        (*info) = newinfo;
    }
    /*
    DPCT1065:18: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#endif
    //----------------------------------------------------

    //copy sdata_A to A
    #ifdef ENABLE_COND6
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            #ifdef BATCH_DISABLE_CLEANUP
            A[tx + i * lda] = sdata_A[tx + i * m];
            #else
            if (tx >= i) A[tx + i * lda] = sdata_A[tx + i * m];
            #endif
        }
    #ifdef ENABLE_COND6
    }
    __syncthreads();
    #endif
}


/******************************************************************************/
static inline void zpotf2_smlpout_anywidth_device(const int m, const int n,
        magmaDoubleComplex *A0, magmaDoubleComplex *A, int lda,
        const int localstep, const int gbstep,
        magma_int_t *info, sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto shared_data = (magmaDoubleComplex *)dpct_local;
    // checkinfo to avoid computation of the singular matrix
    #ifndef BATCH_DISABLE_CHECKING
    if (*info != 0 ) return;
    #endif

    const int orginfo = (*info);
    int panel_info = 0, newinfo = 0;
    const int tx = item_ct1.get_local_id(2);
    magmaDoubleComplex *sdata_A =
        shared_data + item_ct1.get_local_id(1) * (m + POTF2_NB) * POTF2_NB;
    magmaDoubleComplex *sdata_B = sdata_A + m * POTF2_NB;

    #if 0
    zgemm_v20_1_fixsize_device(m, localstep, 
                       A0, lda, sdata_A, sdata_B);
    zpotf2_sminout_fixsize_device(m, sdata_A, m);
    #else
    zgemm_v20_1_anywidth_device(m, n, localstep, A0, lda, sdata_A, sdata_B,
                                item_ct1);
    #endif

    zpotf2_sminout_anywidth_device(m, n, sdata_A, m, &panel_info, item_ct1);
    //----------------------------------------------------
    // Check for not SPD generating info
    #ifndef BATCH_DISABLE_CHECKING
    if(tx == 0) {
        newinfo = ( orginfo == 0 && panel_info != 0 ) ? panel_info + localstep + gbstep : orginfo;
        (*info) = newinfo;
    }
    /*
    DPCT1065:19: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
#endif
    //----------------------------------------------------


    //copy sdata_A to A
    #ifdef ENABLE_COND6
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < n; i++)
        {
            #ifdef BATCH_DISABLE_CLEANUP
            A[tx + i * lda] = sdata_A[tx + i * m];
            #else
            if (tx >= i) A[tx + i * lda] = sdata_A[tx + i * m];
            #endif
        }
    #ifdef ENABLE_COND6
    }
    __syncthreads();
    #endif
}

#endif // MAGMABLAS_ZPOTF2_DEVICES_Z_H
