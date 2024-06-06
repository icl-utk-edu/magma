/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar

       @precisions normal z -> s d c
*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

#define PRECISION_z

#define SLDA(N)              ( (N==15||N==23||N==31)? N : (N+1) )

#ifdef MAGMA_HAVE_CUDA
#define MAX_THREADS          (128)
#else
#define MAX_THREADS          (256)
#endif
#define BATCH_GEMV_NTCOL(N)  (max(1,MAX_THREADS/N))

////////////////////////////////////////////////////////////////////////////////
template<typename T, int N>
void
zgemvn_batched_smallsq_kernel(
        const T alpha,
        T const * const * dA_array, const T* dA, int ldda, int strideA,
        T const * const * dx_array, const T* dx, int incx, int stridex,
        const T beta,
        T**       dy_array,       T* dy, int incy, int stridey,
        const int batchCount, sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto zdata = (T *)dpct_local;

    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int bx = item_ct1.get_group(2);

    const int batchid = bx * item_ct1.get_local_range(1) + ty;

    const T* A;
    const T* x;
    T* y;
    if(batchid < batchCount) {
      A = (dA_array == NULL) ? dA + batchid * strideA : dA_array[batchid];
      x = (dx_array == NULL) ? dx + batchid * stridex : dx_array[batchid];
      y = (dy_array == NULL) ? dy + batchid * stridey : dy_array[batchid];
    }
    T rA[N] = {MAGMA_Z_ZERO};

    // shared memory
    T* sx = (T*)(zdata);
    sx += ty * N;

    // read x in shmem
    if(batchid < batchCount) { 
      sx[tx] = x[tx * incx];
    }
    /*
    DPCT1065:120: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    T ry;
    if(batchid < batchCount) {
      ry = (beta == MAGMA_Z_ZERO)
               ? MAGMA_Z_ZERO
               : beta * y[tx * incy];
#pragma unroll
      for(int j = 0; j < N; j++) {
          rA[j] = A[j * ldda + tx];
      }

      T rTmp = MAGMA_Z_ZERO;
#pragma unroll
      for(int j = 0; j < N; j++) {
          rTmp += rA[j] * sx[j];
      }

      rTmp *= alpha;
      ry   += rTmp;
      y[tx * incy] = ry;
    }

}

////////////////////////////////////////////////////////////////////////////////
template<typename T, int N>
void
zgemvc_batched_smallsq_kernel(
        const magma_trans_t transA, const T alpha,
        T const * const * dA_array, const T* dA, int ldda, int strideA,
        T const * const * dx_array, const T* dx, int incx, int stridex,
        const T beta,
        T**       dy_array,       T* dy, int incy, int stridey,
        const int batchCount, sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto zdata = (T *)dpct_local;

    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int bx = item_ct1.get_group(2);
    const int nty = item_ct1.get_local_range(1);
    const int slda = SLDA(N);

    const int batchid = bx * item_ct1.get_local_range(1) + ty;

    const T* A;
    const T* x;
    T* y;
    if(batchid < batchCount) { 
      A = (dA_array == NULL) ? dA + batchid * strideA : dA_array[batchid];
      x = (dx_array == NULL) ? dx + batchid * stridex : dx_array[batchid];
      y = (dy_array == NULL) ? dy + batchid * stridey : dy_array[batchid];
    }
    T rA[N] = {MAGMA_Z_ZERO};

    // shared memory
    T* sA = (T*)(zdata);
    T* sx = sA + nty * slda * N;
    sA += ty * slda * N;
    sx += ty * N;

    T ry;
    if(batchid < batchCount) { 
      ry = (beta == MAGMA_Z_ZERO)
               ? MAGMA_Z_ZERO
               : beta * y[tx * incy];

      // read x in shmem
      sx[tx] = x[tx * incx];

      #pragma unroll
      for(int j = 0; j < N; j++) {
          rA[j] = A[j * ldda + tx];
      }

      // transpose
      #pragma unroll
      for(int j = 0; j < N; j++) {
          #if defined(PRECISION_z) || defined(PRECISION_c)
          sA[tx * slda + j] = (transA == MagmaConjTrans) ? MAGMA_Z_CONJ(rA[j]) : rA[j];
          #else
          sA[tx * slda + j] = rA[j];
          #endif
      }
    }
    /*
    DPCT1065:121: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if(batchid < batchCount) {
#pragma unroll
      for(int j = 0; j < N; j++) {
           rA[j] = sA[j * slda + tx];
      }

      T rTmp = MAGMA_Z_ZERO;
#pragma unroll
      for(int j = 0; j < N; j++) {
          rTmp += rA[j] * sx[j];
      }

      rTmp *= alpha;
      ry   += rTmp;
      y[tx * incy] = ry;
    }
}

////////////////////////////////////////////////////////////////////////////////
template <int N>
static int zgemv_batched_smallsq_kernel_driver(
    magma_trans_t transA, const magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, const magmaDoubleComplex *dA,
    magma_int_t ldda, magma_int_t strideA, magmaDoubleComplex const * const * dx_array,
    const magmaDoubleComplex *dx, magma_int_t incx, magma_int_t stridex,
    const magmaDoubleComplex beta, magmaDoubleComplex **dy_array,
    magmaDoubleComplex *dy, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t ntcol  = BATCH_GEMV_NTCOL(N);
    magma_int_t shmem  = N * sizeof(magmaDoubleComplex);
    if( !(transA == MagmaNoTrans) ) {
        const int slda = SLDA(N);
        shmem += slda * N * sizeof(magmaDoubleComplex);
    }
    shmem *= ntcol;

    // get max. dynamic shared memory on the GPU
    int nthreads_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::max_work_group_size>();
    int shmem_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::local_mem_size>();

    magma_int_t total_threads = N * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        return -100;
    }

    const int nblocks = magma_ceildiv(batchCount, ntcol);
    sycl::range<3> grid(1, 1, nblocks);
    sycl::range<3> threads(1, ntcol, N);
    if( transA == MagmaNoTrans ) {
       try {
    	   ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
           sycl::local_accessor<uint8_t, 1>
                       dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);
             cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                         zgemvn_batched_smallsq_kernel<magmaDoubleComplex, N>(
                            alpha, dA_array, dA, ldda, strideA, 
                              dx_array, dx, incx, stridex,
                            beta,  dy_array, dy, incy, stridey, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                          });
	   });
       } catch (sycl::exception const &exc) {
         return -100;
       }
    }
    else {
       try {
          ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
           sycl::local_accessor<uint8_t, 1>
                       dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);
             cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                         zgemvc_batched_smallsq_kernel<magmaDoubleComplex, N>(
                           transA, 
                           alpha, dA_array, dA, ldda, strideA, 
                              dx_array, dx, incx, stridex,
                            beta,  dy_array, dy, incy, stridey, batchCount,
                            item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                         });
	  });
       } catch (sycl::exception const &exc) {
         return -100;
       }
    }

    return 0;
}

////////////////////////////////////////////////////////////////////////////////
static magma_int_t
zgemv_batched_smallsq_core(
    magma_trans_t transA, magma_int_t n,
    const magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, const magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t strideA,
    magmaDoubleComplex const * const * dx_array, const magmaDoubleComplex* dx, magma_int_t incx, magma_int_t stridex,
    const magmaDoubleComplex beta,
    magmaDoubleComplex**       dy_array,       magmaDoubleComplex* dy, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    switch(n){
        case  1: info = zgemv_batched_smallsq_kernel_driver< 1>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  2: info = zgemv_batched_smallsq_kernel_driver< 2>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  3: info = zgemv_batched_smallsq_kernel_driver< 3>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  4: info = zgemv_batched_smallsq_kernel_driver< 4>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  5: info = zgemv_batched_smallsq_kernel_driver< 5>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  6: info = zgemv_batched_smallsq_kernel_driver< 6>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  7: info = zgemv_batched_smallsq_kernel_driver< 7>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  8: info = zgemv_batched_smallsq_kernel_driver< 8>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case  9: info = zgemv_batched_smallsq_kernel_driver< 9>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 10: info = zgemv_batched_smallsq_kernel_driver<10>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 11: info = zgemv_batched_smallsq_kernel_driver<11>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 12: info = zgemv_batched_smallsq_kernel_driver<12>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 13: info = zgemv_batched_smallsq_kernel_driver<13>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 14: info = zgemv_batched_smallsq_kernel_driver<14>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 15: info = zgemv_batched_smallsq_kernel_driver<15>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 16: info = zgemv_batched_smallsq_kernel_driver<16>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 17: info = zgemv_batched_smallsq_kernel_driver<17>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 18: info = zgemv_batched_smallsq_kernel_driver<18>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 19: info = zgemv_batched_smallsq_kernel_driver<19>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 20: info = zgemv_batched_smallsq_kernel_driver<20>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 21: info = zgemv_batched_smallsq_kernel_driver<21>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 22: info = zgemv_batched_smallsq_kernel_driver<22>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 23: info = zgemv_batched_smallsq_kernel_driver<23>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 24: info = zgemv_batched_smallsq_kernel_driver<24>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 25: info = zgemv_batched_smallsq_kernel_driver<25>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 26: info = zgemv_batched_smallsq_kernel_driver<26>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 27: info = zgemv_batched_smallsq_kernel_driver<27>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 28: info = zgemv_batched_smallsq_kernel_driver<28>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 29: info = zgemv_batched_smallsq_kernel_driver<29>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 30: info = zgemv_batched_smallsq_kernel_driver<30>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 31: info = zgemv_batched_smallsq_kernel_driver<31>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        case 32: info = zgemv_batched_smallsq_kernel_driver<32>(transA, alpha, dA_array, dA, ldda, strideA, dx_array, dx, incx, stridex, beta, dy_array, dy, incy, stridey, batchCount, queue); break;
        default:;
    }

    return info;
}

/******************************************************************************/
extern "C" magma_int_t
magmablas_zgemv_batched_smallsq(
    magma_trans_t transA, magma_int_t n,
    const magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ldda,
    magmaDoubleComplex const * const * dx_array, magma_int_t incx,
    const magmaDoubleComplex beta,
    magmaDoubleComplex**       dy_array, magma_int_t incy,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < n )
        info = -5;
    else if ( incx <= 0 )
        info = -7;
    else if ( incy <= 0 )
        info = -10;

    if (info != 0) {
        return info;
    }

    info = zgemv_batched_smallsq_core(
            transA, n,
            alpha, dA_array, NULL, ldda, 0,
                   dx_array, NULL, incx, 0,
            beta,  dy_array, NULL, incy, 0,
            batchCount, queue );

    return info;
}

/******************************************************************************/
extern "C" magma_int_t
magmablas_zgemv_batched_strided_smallsq(
    magma_trans_t transA, magma_int_t n,
    const magmaDoubleComplex alpha,
    const magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t strideA,
    const magmaDoubleComplex* dx, magma_int_t incx, magma_int_t stridex,
    const magmaDoubleComplex beta,
    magmaDoubleComplex* dy, magma_int_t incy, magma_int_t stridey,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( n < 0 )
        info = -2;
    else if ( ldda < n )
        info = -5;
    else if ( incx <= 0 )
        info = -7;
    else if ( incy <= 0 )
        info = -10;

    if (info != 0) {
        return info;
    }

    info = zgemv_batched_smallsq_core(
            transA, n,
            alpha, NULL, dA, ldda, strideA,
                   NULL, dx, incx, stridex,
            beta,  NULL, dy, incy, stridey,
            batchCount, queue );

    return info;
}
