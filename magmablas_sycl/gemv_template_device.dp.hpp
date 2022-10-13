/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Tingxing Dong
       @author Azzam Haidar

*/
#ifndef MAGMABLAS_GEMV_TEMPLATE_H
#define MAGMABLAS_GEMV_TEMPLATE_H

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "gemm_template_device_defs.dp.hpp" // use make_FloatingPoint

/******************************************************************************/
// op<trans>( x ) returns x or conj(x).
template< const magma_trans_t conjugate, typename T >
static inline
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
static void
gemvn_template_device(
    int m, int n, T alpha,
    const T * __restrict__ A, int lda,
    const T * __restrict__ x, int incx, T beta,
    T       * __restrict__ y, int incy, sycl::nd_item<3> item_ct1, T *sdata)
{
    if (m <= 0 || n <= 0) return;

    int num_threads = item_ct1.get_local_range(2) *
                      item_ct1.get_local_range(1) * item_ct1.get_local_range(0);

    if (DIM_X * DIM_Y != num_threads) return; // need to launch exactly the same number of threads as template parameters indicate

    int thread_id = item_ct1.get_local_id(2) +
                    item_ct1.get_local_id(1) * item_ct1.get_local_range(2);

    // threads are all configurated locally
    int tx = thread_id % DIM_X;
    int ty = thread_id / DIM_X;

    int ind = item_ct1.get_group(2) * TILE_SIZE + tx;

    int st = item_ct1.get_group(2) * TILE_SIZE;

    int ed = min(st+TILE_SIZE, magma_roundup(m,DIM_X));
    
    int iters = (ed-st)/DIM_X;

    for (int i=0; i < iters; i++)
    {   
        if (ind < m ) A += ind;

        T res = make_FloatingPoint(0.0, 0.0);
        
        if (ind < m )
        {
            for (int col=ty; col < n; col += DIM_Y)
            {       
                res += A[col*lda] * x[col*incx];
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

            /*
            DPCT1065:174: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            if ( DIM_Y > 16)
            { 
                magma_sum_reduce< DIM_Y >( ty, sdata + tx * DIM_Y, item_ct1);
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

            /*
            DPCT1065:175: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }

        if ( ind < m) A -= ind;

        ind += DIM_X;
    }
}


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE,  magma_trans_t trans> 
static void
gemvc_template_device(
    int m, int n, T alpha,
    const T * __restrict__ A, int lda,
    const T * __restrict__ x, int incx, T beta,
    T       * __restrict__ y, int incy, sycl::nd_item<3> item_ct1, T *sdata)
{
    if (m <= 0 || n <= 0) return;

    int num_threads = item_ct1.get_local_range(2) *
                      item_ct1.get_local_range(1) * item_ct1.get_local_range(0);

    if (DIM_X * DIM_Y != num_threads) return; // need to launch exactly the same number of threads as template parameters indicate

    int thread_id = item_ct1.get_local_id(2) +
                    item_ct1.get_local_id(1) * item_ct1.get_local_range(2);

    // threads are all configurated locally
    int tx = thread_id % DIM_X;
    int ty = thread_id / DIM_X;

    T res;
    int mfull = (m / DIM_X) * DIM_X;

    int start = item_ct1.get_group(2) * TILE_SIZE + ty;
    int iters;

    //#define usefixedcondition
    #ifdef usefixedcondition
        /*fixed condition*/
        iters = TILE_SIZE / DIM_Y;
    #else
        /* flexible condition based on global n (has drops when size is roughly bigger than TILE_SIZE)*/
        //int iters = magma_ceildiv(min(n,TILE_SIZE), DIM_Y);

        /* flexible condition based on my local nloc=ed-st*/
        int st = item_ct1.get_group(2) * TILE_SIZE;
        //int ed = magma_ceildiv( min(n, st + TILE_SIZE), DIM_Y ) * DIM_Y; 
        int ed = min(st+TILE_SIZE, magma_roundup(n,DIM_Y));
        iters = (ed-st)/DIM_Y;
    #endif


    if (tx < m) A += tx;
    
    for (int i = 0; i < iters; i++)// at 2Gflops/ overhead
    //for (int col=start; col < (blockIdx.x+1)*TILE_SIZE; col += DIM_Y)// at least 3Gflop/s overhead
    {
        int col = start + i * DIM_Y;

        if ( col < n) A += col*lda;

        res = make_FloatingPoint(0.0, 0.0);

        // partial sums
        if (col < n)
        {    
            for (int i=0; i < mfull; i += DIM_X) {
                res += op<trans>(A[i]) * x[(tx + i)*incx];
            }
            if ( tx + mfull < m ) {
                res += op<trans>(A[mfull]) * x[(tx + mfull)*incx];
            }
        }
        sdata[tx + ty * DIM_X] = res;

        // tree reduction of partial sums,
        // from DIM_X sums to ... 128 to 64 to 32 ... to 1 sum in sdata[0]
        if ( DIM_X > 16)
        { 
            magma_sum_reduce< DIM_X >( tx, sdata + ty * DIM_X, item_ct1);
        }
        else
        {
            /*
            DPCT1065:177: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            if (tx == 0 && col < n)
            {
                for (int i=1; i < m && i < DIM_X; i++)
                {
                    sdata[0 + ty * DIM_X] += sdata[i + ty * DIM_X];
                }
            }
            /*
            DPCT1065:178: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }

        if ( tx == 0 && col < n) {
            y[col*incy] = alpha*sdata[0 + ty * DIM_X] + beta*y[col*incy];
        }

        /*
        DPCT1065:176: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if ( col < n)  A -= col * lda;
    }
}


#endif // MAGMABLAS_GEMV_TEMPLATE_H
