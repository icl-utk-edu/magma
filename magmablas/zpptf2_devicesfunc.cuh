/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Ahmad Ahmad
   @author Natalie Beams

   @precisions normal z -> s d c
 */


#ifndef MAGMABLAS_ZPPTF2_DEVICES_Z_H
#define MAGMABLAS_ZPPTF2_DEVICES_Z_H

// formula for lower part access
#define PACKED(i_, j_, N_) (N_*j_ - j_*(j_+1)/2 + i_)
/******************************************************************************/
// See corresponding device function in zpotf2_devicesfunc.cuh
static inline __device__ void zgemm_packed_v20_1_fixsize_device(int m, int k,
        const magmaDoubleComplex* __restrict__ A0, int A0_row_start, const int lda,
        magmaDoubleComplex *sC, magmaDoubleComplex  *sB)
{
    const int tx = threadIdx.x;
    magmaDoubleComplex rC[POTF2_NB];
    magmaDoubleComplex rA[POTF2_NB];
    magmaDoubleComplex rp[POTF2_NB];

    // prefetch next block.
    #pragma unroll
    for (int i=0; i < POTF2_NB; i++)
    {
        rp[i] = A0[(i > tx + A0_row_start) ? PACKED(i, (tx + A0_row_start), lda) : PACKED((tx + A0_row_start), i, lda)];
        rC[i] = MAGMA_Z_ZERO;
    }

    __syncthreads();



    // accumulate
    #pragma unroll
    for (int iter=0; iter < k; iter += POTF2_NB)
    {
        // rp to rA
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            rA[i] = rp[i];
        }

        // rA to sB
        if (tx < POTF2_NB)
        {
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                sB[tx + i * POTF2_NB] = MAGMA_Z_CONJ(rp[i]);
            }
        }

        __syncthreads();

        // prefetch next block. Azzam
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            rp[i] = A0[(i+iter+POTF2_NB > tx + A0_row_start) ?
		       PACKED((i+iter+POTF2_NB), (tx + A0_row_start), lda) :
                       PACKED((tx + A0_row_start), (i+iter+POTF2_NB), lda)];
        }
        //__syncthreads();

        // multiply current block
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
    __syncthreads();
    }//end of accumulation

    // finalizing gemm.
    #pragma unroll
    for (int i=0; i < POTF2_NB; i++)
    {
        sC[tx + i *m] = rp[i] - rC[i];
    }
    __syncthreads();
}


/******************************************************************************/
static inline __device__ void zgemm_packed_v20_1_anywidth_device(int m, int n, int k,
        const magmaDoubleComplex* __restrict__ A0, int A0_row_start, int lda,
        magmaDoubleComplex *sC, magmaDoubleComplex  *sB)
{
    const int tx = threadIdx.x;
    magmaDoubleComplex rC[POTF2_NB];
    magmaDoubleComplex rA[POTF2_NB];
    magmaDoubleComplex rp[POTF2_NB];

    // k+n is the total number of columns in the matrix
    const int bound_A = (k+n)*(k+n+1)/2 - 1;

    // prefetch next block.
    #pragma unroll
    for (int i=0; i < POTF2_NB; i++)
    {
        rp[i] = A0[min(bound_A, (i > tx + A0_row_start) ? PACKED(i, (tx + A0_row_start), lda) :
			                                  PACKED((tx + A0_row_start), i, lda))];
        rC[i] = MAGMA_Z_ZERO;
    }

    __syncthreads();



    // accumulate
    #pragma unroll
    for (int iter=0; iter < k; iter += POTF2_NB)
    {
        // rp to rA
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            rA[i] = rp[i];
        }

        // rA to sB
        if (tx < POTF2_NB)
        {
            #pragma unroll
            for (int i=0; i < POTF2_NB; i++)
            {
                sB[tx + i * POTF2_NB] = MAGMA_Z_CONJ(rp[i]);
            }
        }

        __syncthreads();

        // prefetch next block. Azzam
        #pragma unroll
        for (int i=0; i < POTF2_NB; i++)
        {
            rp[i] = A0[min(bound_A, (i+iter+POTF2_NB > tx + A0_row_start) ?
	    	                   PACKED((i+iter+POTF2_NB), (tx + A0_row_start), lda) :
                                   PACKED((tx + A0_row_start), (i+iter+POTF2_NB), lda))];
        }
        //__syncthreads();

        // multiply current block
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
    __syncthreads();
    }//end of accumulation

    // finalizing gemm.
    #pragma unroll
    for (int i=0; i < POTF2_NB; i++)
    {
        sC[tx + i *m] = rp[i] - rC[i];
    }
    __syncthreads();
}


/******************************************************************************/
static inline __device__ void zpptf2_smlpout_fixwidth_device(const int m,
        magmaDoubleComplex *A0, int A0_row_start, magmaDoubleComplex *A, int lda,
        const int localstep, const int gbstep,
        magma_int_t *info)
{
    extern __shared__ magmaDoubleComplex shared_data[];

    // checkinfo to avoid computation of the singular matrix
    #ifndef BATCH_DISABLE_CHECKING
    if (*info != 0 ) return;
    #endif

    const int orginfo = (*info);
    int panel_info = 0, newinfo = 0;
    const int tx = threadIdx.x;
    magmaDoubleComplex *sdata_A = shared_data + threadIdx.y * (m+POTF2_NB)*POTF2_NB;
    magmaDoubleComplex *sdata_B = sdata_A + m * POTF2_NB;

    zgemm_packed_v20_1_fixsize_device(m, localstep,
                       A0, A0_row_start, lda, sdata_A, sdata_B);

    // panel fact. in shared memory
    zpotf2_sminout_fixsize_device(m, sdata_A, m, &panel_info);
    //----------------------------------------------------
    // Check for not SPD generating info
    #ifndef BATCH_DISABLE_CHECKING
    if(tx == 0) {
        newinfo = ( orginfo == 0 && panel_info != 0 ) ? panel_info + localstep + gbstep : orginfo;
        (*info) = newinfo;
    }
    __syncthreads();
    #endif
    //----------------------------------------------------

    //copy sdata_A to A
    #pragma unroll
    for (int i=0; i < POTF2_NB; i++)
    {
        #ifdef BATCH_DISABLE_CLEANUP
        A[PACKED(tx, i, m)] = sdata_A[tx + i * m];
        #else
        if (tx >= i) A[PACKED(tx, i, m)] = sdata_A[tx + i * m];
        #endif
    }
}


/******************************************************************************/
static inline __device__ void zpptf2_smlpout_anywidth_device(const int m, const int n,
        magmaDoubleComplex *A0, int A0_row_start, magmaDoubleComplex *A, int lda,
        const int localstep, const int gbstep,
        magma_int_t *info)
{
    extern __shared__ magmaDoubleComplex shared_data[];
    // checkinfo to avoid computation of the singular matrix
    #ifndef BATCH_DISABLE_CHECKING
    if (*info != 0 ) return;
    #endif

    const int orginfo = (*info);
    int panel_info = 0, newinfo = 0;
    const int tx = threadIdx.x;
    magmaDoubleComplex *sdata_A = shared_data + threadIdx.y * (m+POTF2_NB)*POTF2_NB;
    magmaDoubleComplex *sdata_B = sdata_A + m * POTF2_NB;

    zgemm_packed_v20_1_anywidth_device(m, n, localstep,
                       A0, A0_row_start, lda, sdata_A, sdata_B);

    zpotf2_sminout_anywidth_device(m, n, sdata_A, m, &panel_info);
    //----------------------------------------------------
    // Check for not SPD generating info
    #ifndef BATCH_DISABLE_CHECKING
    if(tx == 0) {
        newinfo = ( orginfo == 0 && panel_info != 0 ) ? panel_info + localstep + gbstep : orginfo;
        (*info) = newinfo;
    }
    __syncthreads();
    #endif
    //----------------------------------------------------


    //copy sdata_A to A
    #pragma unroll
    for (int i=0; i < n; i++)
    {
        #ifdef BATCH_DISABLE_CLEANUP
        A[PACKED(tx, i, m] = sdata_A[tx + i * m];
        #else
        if (tx >= i) A[PACKED(tx, i, m)] = sdata_A[tx + i * m];
        #endif
    }
}

#endif // MAGMABLAS_ZPPTF2_DEVICES_Z_H
