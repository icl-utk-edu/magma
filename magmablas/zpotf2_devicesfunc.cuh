
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


//extern __shared__ magmaDoubleComplex shared_data[];

/******************************************************************************/
static inline __device__ void zpotf2_sminout_anywidth_device(const int m, const int n, magmaDoubleComplex *A, const int lda, int* info)
{
    const int tx = threadIdx.x;
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
            xreal = sqrt(xreal);
            factor = MAGMA_Z_MAKE(1.0/xreal, 0.0);
        #ifdef ENABLE_COND1
        }
        #endif
        __syncthreads(); // must sync to make sure that A[iter + iter * lda] is read by all threads before modifying it
        #ifdef ENABLE_COND1
        if ( tx >= iter && tx < m )
        {
        #endif
            A[ tx + iter * lda ] *= factor; // or use the next line and remove the sync above
            //A[ tx + iter * lda ]  = tx == iter ? MAGMA_Z_MAKE(xreal, 0.0) : A[ tx + iter * lda ] * factor;
        #ifdef ENABLE_COND1
        }
        #endif
        __syncthreads();


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
        __syncthreads();
    }// end of iter
    // ENABLE_COND1 must be disabled, which the default config., so that the right info is returned
    if(tx == 0) *info = linfo;
    __syncthreads();
}


/******************************************************************************/
template<int NB>
static inline __device__ void zpotf2_sminout_fixsize_device(const int m, magmaDoubleComplex *A, const int lda, int* info)
{
    const int tx = threadIdx.x;
    magmaDoubleComplex factor;
    int linfo = 0;

    #pragma unroll
    for (int iter=0; iter < NB; iter++)
    {
        //sqrt(diag) and zdscal
        #ifdef ENABLE_COND2
        if ( tx >= iter && tx < m )
        {
        #endif
            double xreal = MAGMA_Z_REAL(A[iter + iter * lda]);
            linfo = ( linfo == 0 && (xreal <= MAGMA_D_ZERO || xreal != xreal )) ? (iter+1) : linfo;
            xreal = sqrt(xreal);
            factor = MAGMA_Z_MAKE(1.0/xreal, 0.0);
        #ifdef ENABLE_COND2
        }
        #endif
        __syncthreads();
        #ifdef ENABLE_COND2
        if ( tx >= iter && tx < m )
        {
        #endif
            A[ tx + iter * lda ] *= factor;

            //A[ tx + iter * lda ]  = tx == iter ? MAGMA_Z_MAKE(xreal, 0.0) : A[ tx + iter * lda ] * factor;
            //if (tx < NB) row[ tx ] = MAGMA_Z_CONJ( A[ tx + iter * lda ] );
            //if (tx < NB) A[ iter + tx * lda ] = MAGMA_Z_CONJ( A[ tx + iter * lda ] );
        #ifdef ENABLE_COND2
        }
        #endif

        __syncthreads();


        // zherk
        #ifdef ENABLE_COND2
        if ( tx > iter && tx < m )
        {
        #endif
            #pragma unroll
            for (int j=iter+1; j < NB; j++)
            {
                A [tx + j * lda] -= A[tx + iter * lda]  *  MAGMA_Z_CONJ(A[iter * lda + j]);
                //A [tx + j * lda] -= A[tx + iter * lda]  *  row[j];
                //A [tx + j * lda] -= A[tx + iter * lda]  *  A[iter +lda * j];
            }
        #ifdef ENABLE_COND2
        }
        #endif
        __syncthreads();
    }// end of iter
    // ENABLE_COND1 must be disabled, which the default config., so that the right info is returned
    if(tx == 0) *info = linfo;
    __syncthreads();
}


/******************************************************************************/
template<int NB>
static inline __device__ void zgemm_v20_1_fixsize_device(int m, int k,
        const magmaDoubleComplex* __restrict__ A0, const int lda,
        magmaDoubleComplex *sC, magmaDoubleComplex  *sB)
{
    const int tx = threadIdx.x;
    magmaDoubleComplex rC[NB];
    magmaDoubleComplex rA[NB];
    magmaDoubleComplex rp[NB];

    // prefetch next block.
    #ifdef ENABLE_COND4
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < NB; i++)
        {
            rp[i] = A0[tx + i * lda];
            rC[i] = MAGMA_Z_ZERO;
        }
    #ifdef ENABLE_COND4
    }
    #endif

    __syncthreads();



    // accumulate
    #pragma unroll
    for (int iter=0; iter < k; iter += NB)
    {
        #ifdef ENABLE_COND4
        if (tx < m)
        {
        #endif
            // rp to rA
            #pragma unroll
            for (int i=0; i < NB; i++)
            {
                rA[i] = rp[i];
            }
        #ifdef ENABLE_COND4
        }
        #endif

        // rA to sB
        if (tx < NB)
        {
            #pragma unroll
            for (int i=0; i < NB; i++)
            {
                sB[tx + i * NB] = MAGMA_Z_CONJ(rp[i]);
            }
        }

        __syncthreads();

        // prefetch next block. Azzam
        #ifdef ENABLE_COND4
        if (tx < m )
        {
        #endif
            #pragma unroll
            for (int i=0; i < NB; i++)
            {
                rp[i] = A0[tx + (i+(iter+NB)) * lda];
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
            for (int i=0; i < NB; i++)
            {
                #pragma unroll
                for (int col=0; col < NB; col++)
                {
                    // A0 is multiplied by NB times
                    rC[col] +=  rA[i] * sB[col + i * NB];
                }
            }
        #ifdef ENABLE_COND4
        }
        #endif
        __syncthreads();
    }//end of accumulation

    // finalyzing gemm.
    #ifdef ENABLE_COND4
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < NB; i++)
        {
            sC[tx + i *m] = rp[i] - rC[i];
        }
    #ifdef ENABLE_COND4
    }
    #endif
    __syncthreads();
}


/******************************************************************************/
template<int NB>
static inline __device__ void zgemm_v20_1_anywidth_device(int m, int n, int k,
        const magmaDoubleComplex* __restrict__ A0, int lda,
        magmaDoubleComplex *sC, magmaDoubleComplex  *sB)
{
    const int tx = threadIdx.x;
    magmaDoubleComplex rC[NB];
    magmaDoubleComplex rA[NB];
    magmaDoubleComplex rp[NB];

    const int bound_A = lda*(k+n-1)+m-1;

    // prefetch next block.
    #ifdef ENABLE_COND5
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < NB; i++)
        {
            rp[i] = A0[min(bound_A, tx + i * lda)];
            rC[i] = MAGMA_Z_ZERO;
        }
    #ifdef ENABLE_COND5
    }
    #endif

    __syncthreads();



    // accumulate
    #pragma unroll
    for (int iter=0; iter < k; iter += NB)
    {
        #ifdef ENABLE_COND5
        if (tx < m)
        {
        #endif
            // rp to rA
            #pragma unroll
            for (int i=0; i < NB; i++)
            {
                rA[i] = rp[i];
            }
        #ifdef ENABLE_COND5
        }
        #endif

        // rA to sB
        if (tx < NB)
        {
            #pragma unroll
            for (int i=0; i < NB; i++)
            {
                sB[tx + i * NB] = MAGMA_Z_CONJ(rp[i]);
            }
        }

        __syncthreads();

        // prefetch next block. Azzam
        #ifdef ENABLE_COND5
        if (tx < m )
        {
        #endif
            #pragma unroll
            for (int i=0; i < NB; i++)
            {
                rp[i] = A0[min(bound_A, tx + (i+(iter+NB)) * lda)]; // min(bound,xxx) is to avoid reading out of bound
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
            for (int i=0; i < NB; i++)
            {
                #pragma unroll
                for (int col=0; col < NB; col++)
                {
                    // A0 is multiplied by NB times
                    rC[col] +=  rA[i] * sB[col + i * NB];
                }
            }
        #ifdef ENABLE_COND5
        }
        #endif
        __syncthreads();
    }//end of accumulation

    // finalyzing gemm.
    #ifdef ENABLE_COND5
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < NB; i++)
        {
            sC[tx + i *m] = rp[i] - rC[i];
        }
    #ifdef ENABLE_COND5
    }
    #endif
    __syncthreads();
}


/******************************************************************************/
template<int NB>
static inline __device__ void zpotf2_smlpout_fixwidth_device(const int m,
        magmaDoubleComplex *A0, magmaDoubleComplex *A, int lda,
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
    magmaDoubleComplex *sdata_A = shared_data + threadIdx.y * (m+NB)*NB;
    magmaDoubleComplex *sdata_B = sdata_A + m * NB;


    #if 1
    zgemm_v20_1_fixsize_device<NB>(m, localstep,
                       A0, lda, sdata_A, sdata_B);
    #else
    zgemm_v20_1_anywidth_device<NB>(m, NB, localstep,
                       A0, lda, sdata_A, sdata_B);
    #endif

    // panel fact. in shared memory
    zpotf2_sminout_fixsize_device<NB>(m, sdata_A, m, &panel_info);
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
    #ifdef ENABLE_COND6
    if (tx < m)
    {
    #endif
        #pragma unroll
        for (int i=0; i < NB; i++)
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
template<int NB>
static inline __device__ void zpotf2_smlpout_anywidth_device(const int m, const int n,
        magmaDoubleComplex *A0, magmaDoubleComplex *A, int lda,
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
    magmaDoubleComplex *sdata_A = shared_data + threadIdx.y * (m+NB)*NB;
    magmaDoubleComplex *sdata_B = sdata_A + m * NB;

    #if 0
    zgemm_v20_1_fixsize_device<NB>(m, localstep,
                       A0, lda, sdata_A, sdata_B);
    zpotf2_sminout_fixsize_device<NB>(m, sdata_A, m);
    #else
    zgemm_v20_1_anywidth_device<NB>(m, n, localstep,
                       A0, lda, sdata_A, sdata_B);
    #endif

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
