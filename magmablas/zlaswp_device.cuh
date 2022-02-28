/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Azzam Haidar
       @author Tingxing Dong
       @author Ahmad Abdelfattah
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"

#define SWP_WIDTH      4
#define BLK_SIZE       256
#define ZLASWP_COL_NTH 32
// SWP_WIDTH is number of threads in a block


/******************************************************************************/
static __device__
void zlaswp_rowparallel_devfunc(
                              int n, int width, int height,
                              magmaDoubleComplex *dA, int lda,
                              magmaDoubleComplex *dout, int ldo,
                              magma_int_t* pivinfo)
{
    extern __shared__ magmaDoubleComplex shared_data[];

    //int height = k2- k1;
    //int height = blockDim.x;
    const int tid = threadIdx.x;
    const int bx  = blockIdx.x;
    dA   += SWP_WIDTH * bx * lda;
    dout += SWP_WIDTH * bx * ldo;
    magmaDoubleComplex *sdata = shared_data;

    //if (bx == gridDim.x -1)
    //{
    //    width = n - bx * SWP_WIDTH;
    //}
    const int nblocks = magma_ceildiv(n, SWP_WIDTH);
    if(bx >= nblocks) return;
    width = (bx < nblocks-1) ? SWP_WIDTH : n - (nblocks-1)*SWP_WIDTH;

    #if 0
    __syncthreads();
    if(blockIdx.z == 3 && tid == 0) {
        printf("(%d, %d, %d) - (%d, %d, %d): n = %d, width = %d\n",
                blockIdx.x, blockIdx.y, blockIdx.z,
                threadIdx.x, threadIdx.y, threadIdx.z,
                n, width);
    }
    __syncthreads();
    #endif

    if (tid < height)
    {
        int mynewroworig = pivinfo[tid]-1; //-1 to get the index in C
        int itsreplacement = pivinfo[mynewroworig] -1; //-1 to get the index in C
        //printf("%d: mynewroworig = %d, itsreplacement = %d\n", tid, mynewroworig, itsreplacement);
        #pragma unroll
        for (int i=0; i < width; i++)
        {
            sdata[ tid + i * height ]    = dA[ mynewroworig + i * lda ];
            dA[ mynewroworig + i * lda ] = dA[ itsreplacement + i * lda ];
        }
    }
    __syncthreads();

    if (tid < height)
    {
        // copy back the upper swapped portion of A to dout
        #pragma unroll
        for (int i=0; i < width; i++)
        {
            dout[tid + i * ldo] = sdata[tid + i * height];
        }
    }
}
