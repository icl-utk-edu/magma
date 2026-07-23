/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Tingxing Dong
       @author Azzam Haidar
       @author Natalie Beams

*/
#ifndef MAGMABLAS_PACKED_GEMV_TEMPLATE_H
#define MAGMABLAS_PACKED_GEMV_TEMPLATE_H

#include "gemm_template_device_defs.cuh"// use make_FloatingPoint

// formula for lower part access
#define PACKED(i_, j_, N_) (N_*(j_) - (j_)*(j_+1)/2 + i_)
/******************************************************************************/
// op<trans>( x ) returns x or conj(x).
template< const magma_trans_t conjugate, typename T >
__host__ __device__ static inline
T op( T& x )
{
    if (conjugate == MagmaConjTrans) {
        return conj(x);
    } else {
        return x;
    }
}


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE> 
static __device__ void
gemvn_packed_template_device(
    int m, int n, T alpha,
    const T * __restrict__ A, int roffA, int coffA, int lda,
    const T * __restrict__ x, int incx, T beta,
    T       * __restrict__ y, int incy)
{
    if (m <= 0 || n <= 0) return;

    int num_threads = blockDim.x * blockDim.y * blockDim.z;
    
    if (DIM_X * DIM_Y != num_threads) return; // need to launch exactly the same number of threads as template parameters indicate

    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    // threads are all configurated locally
    int tx = thread_id % DIM_X;
    int ty = thread_id / DIM_X;

    int ind = blockIdx.x*TILE_SIZE + tx;

    __shared__ T sdata[DIM_X * DIM_Y];


    int st = blockIdx.x * TILE_SIZE;

    int ed = min(st+TILE_SIZE, magma_roundup(m,DIM_X));
    
    int iters = (ed-st)/DIM_X;

    for (int i=0; i < iters; i++)
    {   
        T res = make_FloatingPoint(0.0, 0.0);
        
        if ((ind < m) && (roffA + ind < lda))
        {
            for (int col=ty; col < n; col += DIM_Y)
            {  
                res += A[(ind + roffA >= col + coffA) ? PACKED(ind + roffA, col + coffA, lda) : PACKED(col + coffA, ind + roffA, lda)] * x[col*incx];
            }
        }

        if (DIM_X >= num_threads) // indicated 1D threads configuration. Shared memory is not needed, reduction is done naturally
        {
            if (ty == 0 && ind < m)
            {
                y[ind*incy] = alpha*res + beta*y[ind*incy];
            }
        }
        else 
        {
            sdata[ty + tx * DIM_Y] = res;

            __syncthreads(); 

            if ( DIM_Y > 16)
            { 
                magma_sum_reduce< DIM_Y >( ty, sdata + tx * DIM_Y);
            }
            else
            {
                if (ty == 0 && ind < m)
                {
                    for (int i=1; i < DIM_Y; i++)
                    {
                        sdata[tx * DIM_Y] += sdata[i + tx * DIM_Y]; 
                    }
                }
            }

            if (ty == 0 && ind < m)
            {
                y[ind*incy] = alpha*sdata[tx * DIM_Y] + beta*y[ind*incy];
            }

            __syncthreads();
        }

        ind += DIM_X;
    }
}


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE,  magma_trans_t trans> 
static __device__ void
gemvc_packed_template_device(
    int m, int n, T alpha,
    const T * __restrict__ A, int roffA, int coffA, int lda,
    const T * __restrict__ x, int incx, T beta,
    T       * __restrict__ y, int incy)
{
    if (m <= 0 || n <= 0) return;

    int num_threads = blockDim.x * blockDim.y * blockDim.z;
    
    if (DIM_X * DIM_Y != num_threads) return; // need to launch exactly the same number of threads as template parameters indicate

    int thread_id = threadIdx.x + threadIdx.y * blockDim.x;

    // threads are all configurated locally
    int tx = thread_id % DIM_X;
    int ty = thread_id / DIM_X;

    __shared__ T sdata[DIM_X * DIM_Y];

    T res;
    int mfull = (m / DIM_X) * DIM_X;
    
    int start = blockIdx.x * TILE_SIZE + ty;
    int iters;

    //#define usefixedcondition
    #ifdef usefixedcondition
        /*fixed condition*/
        iters = TILE_SIZE / DIM_Y;
    #else
        /* flexible condition based on global n (has drops when size is roughly bigger than TILE_SIZE)*/
        //int iters = magma_ceildiv(min(n,TILE_SIZE), DIM_Y);

        /* flexible condition based on my local nloc=ed-st*/
        int st = blockIdx.x * TILE_SIZE;
        //int ed = magma_ceildiv( min(n, st + TILE_SIZE), DIM_Y ) * DIM_Y; 
        int ed = min(st+TILE_SIZE, magma_roundup(n,DIM_Y));
        iters = (ed-st)/DIM_Y;
    #endif

    int tx_roff = (tx < m) ? tx : 0;	 

    for (int i = 0; i < iters; i++)// at 2Gflops/ overhead
    //for (int col=start; col < (blockIdx.x+1)*TILE_SIZE; col += DIM_Y)// at least 3Gflop/s overhead
    {
        int col = start + i * DIM_Y;

        res = make_FloatingPoint(0.0, 0.0);

        // partial sums
        if (col < n)
        {    
            for (int i=0; i < mfull; i += DIM_X) {
                res += op<trans>(A[(tx_roff + i + roffA >= col + coffA) ? PACKED(tx_roff + i + roffA, col + coffA, lda) : PACKED(col + coffA, tx_roff + i + roffA, lda)]) * x[(tx_roff + i)*incx];
            }
            if ( tx + mfull < m ) {
                res += op<trans>(A[(tx_roff + mfull + roffA >= col + coffA) ? PACKED(tx_roff + mfull + roffA, col + coffA, lda) : PACKED(col + coffA, tx_roff + mfull + roffA, lda)]) * x[(tx_roff + mfull)*incx];
            }
        }
        sdata[tx + ty * DIM_X] = res;

        // tree reduction of partial sums,
        // from DIM_X sums to ... 128 to 64 to 32 ... to 1 sum in sdata[0]
        if ( DIM_X > 16)
        { 
            magma_sum_reduce< DIM_X >( tx, sdata + ty * DIM_X);
        }
        else
        {
            __syncthreads();

            if (tx == 0 && col < n)
            {
                for (int i=1; i < m && i < DIM_X; i++)
                {
                    sdata[0 + ty * DIM_X] += sdata[i + ty * DIM_X];
                }
            }
            __syncthreads();
        }

        if ( tx == 0 && col < n) {
            y[col*incy] = alpha*sdata[0 + ty * DIM_X] + beta*y[col*incy];
        }

        __syncthreads();
    }
}


#endif // MAGMABLAS_PACKED_GEMV_TEMPLATE_H
