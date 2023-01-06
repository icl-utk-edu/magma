/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Tingxing Dong

       @precisions normal z -> s d c
*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

#define BLOCK_SIZE 256


#define dA(a_1,a_2) (dA  + (a_1) + (a_2)*(local_lda))

#include "zlarfg_devicesfunc.dp.hpp"

/******************************************************************************/
static 
void zlarfx_device(
    int m, int n,  magmaDoubleComplex *v, magmaDoubleComplex *tau,
    magmaDoubleComplex *dc, magma_int_t ldc, magmaDoubleComplex* sum,
    sycl::nd_item<3> item_ct1)
{
    if (n <= 0) return;
    /*
    DPCT1064:370: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    if (MAGMA_Z_EQUAL(*tau, MAGMA_Z_ZERO)) return; // check singularity

    const int tx = item_ct1.get_local_id(2);

    magmaDoubleComplex lsum;

    for (int k=0; k < n; k++)
    {
        /* perform  w := v' * C  */
        if (tx < BLOCK_SIZE)
        {
            if (tx == 0)
                lsum = dc[0+ldc*k]; //since V[0] should be one
            else
                /*
                DPCT1064:373: Migrated make_cuDoubleComplex call is used in a
                macro definition and is not valid for all macro uses. Adjust the
                code.
                */
                lsum = MAGMA_Z_ZERO;
            for (int j = tx+1; j < m; j += BLOCK_SIZE) {
                lsum += MAGMA_Z_MUL( MAGMA_Z_CONJ( v[j] ), dc[j+ldc*k] );
            }

            sum[tx] = lsum;
        }

        magma_sum_reduce<BLOCK_SIZE>(tx, sum, item_ct1);
        /*
        DPCT1065:371: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        magmaDoubleComplex z__1 = - MAGMA_Z_CONJ(*tau) * sum[0];
        /*  C := C - v * w  */
        if (tx < BLOCK_SIZE)
        {
            for (int j = tx+1; j < m; j += BLOCK_SIZE)
                dc[j+ldc*k] += z__1 * v[j];
        }
        if (tx == 0) dc[0+ldc*k] += z__1;

        /*
        DPCT1065:372: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
}


/******************************************************************************/
static 
void zgeqr2_device( magma_int_t m, magma_int_t n,
                               magmaDoubleComplex* dA, magma_int_t lda,
                               magmaDoubleComplex *dtau,
                               magmaDoubleComplex *dv,
                               magmaDoubleComplex *sum,
                               double *swork,
                               magmaDoubleComplex *scale,
                               double *sscale, sycl::nd_item<3> item_ct1)
{
    //lapack zlarfg, compute the norm, scale and generate the householder vector
    zlarfg_device(m, dv, &(dv[1]), 1, dtau, swork, sscale, scale, item_ct1);

    /*
    DPCT1065:374: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    //update the trailing matix with the householder
    zlarfx_device(m, n, dv, dtau, dA, lda, sum, item_ct1);

    /*
    DPCT1065:375: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
}


/******************************************************************************/


/******************************************************************************/

void zgeqr2_sm_kernel_batched(
        int m, int n,
        magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t lda,
        magmaDoubleComplex **dtau_array, magma_int_t taui ,
        sycl::nd_item<3> item_ct1, uint8_t *dpct_local,
        magmaDoubleComplex *scale, magmaDoubleComplex *sum, double *swork,
        double *sscale)
{
    auto shared_data = (magmaDoubleComplex *)dpct_local;

    magmaDoubleComplex *dA = dA_array[item_ct1.get_group(0)];
    magmaDoubleComplex *dtau = dtau_array[item_ct1.get_group(0)];

    dA   += Aj * lda + Ai;
    dtau += taui;

    magmaDoubleComplex *sdata = (magmaDoubleComplex*)shared_data;

    const int tx = item_ct1.get_local_id(2);

    //load data from global to shared memory
    for (int s=0; s < n; s++)
    {
        for (int j = tx; j < m; j += BLOCK_SIZE)
        {
            sdata[j + s * m] = dA[j + s * lda];
        }
    }

    /*
    DPCT1065:376: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (int s=0; s < min(m,n); s++)
    {
        zgeqr2_device(m - s, n - (s + 1), &(sdata[s + (s + 1) * m]), m,
                      dtau + s, &(sdata[s + s * m]), sum, swork, scale, sscale,
                      item_ct1);
    } // end of s

    //copy back to global memory
    for (int s=0; s < n; s++)
    {
        for (int j = tx; j < m; j += BLOCK_SIZE)
        {
            dA[j + s * lda] = sdata[j + s * m];
        }
    }
}


/******************************************************************************/

void zgeqr2_column_sm_kernel_batched(
        int m, int n,
        magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t lda,
        magmaDoubleComplex **dtau_array, magma_int_t taui ,
        sycl::nd_item<3> item_ct1, uint8_t *dpct_local,
        magmaDoubleComplex *scale, magmaDoubleComplex *sum, double *swork,
        double *sscale)
{
    auto shared_data = (magmaDoubleComplex *)dpct_local;

    magmaDoubleComplex *dA = dA_array[item_ct1.get_group(0)];
    magmaDoubleComplex *dtau = dtau_array[item_ct1.get_group(0)];
    dA   += Aj * lda + Ai;
    dtau += taui;

    magmaDoubleComplex *sdata = (magmaDoubleComplex*)shared_data;

    const int tx = item_ct1.get_local_id(2);

    for (int s=0; s < min(m,n); s++)
    {
        //load one vector in shared memory: sdata
        for (int j = tx; j < m-s; j += BLOCK_SIZE)
        {
            sdata[j] = dA[s + j + s * lda];
        }

        /*
        DPCT1065:377: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        //sdata is written
        zgeqr2_device(m - s, n - (s + 1), &(dA[s + (s + 1) * lda]), lda,
                      dtau + s, sdata, sum, swork, scale, sscale, item_ct1);

        for (int j = tx; j < m-s; j += BLOCK_SIZE)
        {
            dA[s + j + s * lda] = sdata[j];
        }

        /*
        DPCT1065:378: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
}


/******************************************************************************/

void zgeqr2_kernel_batched(
        int m, int n,
        magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t lda,
        magmaDoubleComplex **dtau_array, magma_int_t taui ,
        sycl::nd_item<3> item_ct1, magmaDoubleComplex *scale,
        magmaDoubleComplex *sum, double *swork, double *sscale)
{
    magmaDoubleComplex *dA = dA_array[item_ct1.get_group(0)];
    magmaDoubleComplex *dtau = dtau_array[item_ct1.get_group(0)];
    dA   += Aj * lda + Ai;
    dtau += taui;

    for (int s=0; s < min(m,n); s++)
    {
        zgeqr2_device(m - s, n - (s + 1), &(dA[s + (s + 1) * lda]), lda,
                      dtau + s, &(dA[s + s * lda]), sum, swork, scale, sscale,
                      item_ct1);
    }
}


/******************************************************************************/
extern "C" magma_int_t
magma_zgeqr2_fused_batched(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
        magmaDoubleComplex **dtau_array, magma_int_t taui,
        magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    /* Check arguments */
    magma_int_t arginfo = 0;

    if (m < 0)
        arginfo = -1;
    else if (n < 0 || n > 32)
        arginfo = -2;
    else if (ldda < max(1,m))
        arginfo = -4;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // try the register version
    arginfo = magma_zgeqr2_fused_reg_batched(
                m, n, dA_array, Ai, Aj, ldda,
                dtau_array, taui, info_array, 0, batchCount, queue );
    if ( arginfo == 0 ) return arginfo;

    // register version did not launch kernel
    // try shared memory version
    magma_int_t nthreads = magma_get_zgeqr2_fused_sm_batched_nthreads(m, n);
    arginfo = magma_zgeqr2_fused_sm_batched(
                m, n, dA_array, Ai, Aj, ldda,
                dtau_array, taui, info_array, nthreads, 0, batchCount, queue );

    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    ZGEQR2 computes a QR factorization of a complex m by n matrix A:
    A = Q * R.

    This version implements the right-looking QR with non-blocking.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N)
             On entry, the M-by-N matrix A.
             On exit, the elements on and above the diagonal of the array
             contain the min(M,N)-by-N upper trapezoidal matrix R (R is
             upper triangular if m >= n); the elements below the diagonal,
             with the array TAU, represent the orthogonal matrix Q as a
             product of min(m,n) elementary reflectors (see Further
             Details).

    @param[in]
    ldda     INTEGER
             The leading dimension of the array dA.  LDDA >= max(1,M).
             To benefit from coalescent memory accesses LDDA must be
             divisible by 16.

    @param[out]
    dtau_array Array of pointers, dimension (batchCount).
             Each is a COMPLEX_16 array, dimension (min(M,N))
             The scalar factors of the elementary reflectors (see Further
             Details).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    Further Details
    ---------------
    The matrix Q is represented as a product of elementary reflectors

       Q = H(1) H(2) . . . H(k), where k = min(m,n).

    Each H(i) has the form

       H(i) = I - tau * v * v'

    where tau is a complex scalar, and v is a complex vector with
    v(1:i-1) = 0 and v(i) = 1; v(i+1:m) is stored on exit in A(i+1:m,i),
    and tau in TAU(i).

    @ingroup magma_geqr2_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgeqr2_batched(magma_int_t m, magma_int_t n,
                     magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
                     magmaDoubleComplex **dtau_array, magma_int_t taui,
                     magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue)
{
    /* Check arguments */
    magma_int_t arginfo = 0;
    magma_device_t device;
    magma_getdevice( &device );
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

    magma_int_t k = min(m,n);

    // first, try the fused geqr2
    arginfo = magma_zgeqr2_fused_batched(m, n, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, batchCount, queue);
    if ( arginfo == 0 ) return arginfo;

    // reaching this point means that the fused routine does not support
    // the size of the input panel, proceed with more generic code

    // static shared memory requirement, valid for:
    // zgeqr2_sm_kernel_batched, zgeqr2_column_sm_kernel_batched, zgeqr2_kernel_batched
    magma_int_t static_shmem = (BLOCK_SIZE + 1 ) * ( sizeof(magmaDoubleComplex) + sizeof(double) );

    // dynamic shared memory
    magma_int_t dynamic_shmem_sm_kernel        = sizeof(magmaDoubleComplex) * m * k;
    magma_int_t dynamic_shmem_column_sm_kernel = sizeof(magmaDoubleComplex) * m;

    // total shared memory
    magma_int_t total_shmem_sm_kernel        = static_shmem + dynamic_shmem_sm_kernel;
    magma_int_t total_shmem_column_sm_kernel = static_shmem + dynamic_shmem_column_sm_kernel;

    // max. dynamic shared memory allowed per thread-block
    int shmem_max = 0;
//    #if CUDA_VERSION >= 9000
//    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
//    if ( total_shmem_sm_kernel <= shmem_max) {
        /*
        DPCT1007:379: Migration of cudaFuncSetAttribute is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
//        cudaFuncSetAttribute(zgeqr2_sm_kernel_batched,
//                             cudaFuncAttributeMaxDynamicSharedMemorySize,
//                             dynamic_shmem_sm_kernel);
//    }

//    if ( total_shmem_column_sm_kernel <= shmem_max) {
        /*
        DPCT1007:380: Migration of cudaFuncSetAttribute is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
//        cudaFuncSetAttribute(zgeqr2_column_sm_kernel_batched,
//                             cudaFuncAttributeMaxDynamicSharedMemorySize,
//                             dynamic_shmem_column_sm_kernel);
//    }
//    #else
//    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
//    #endif    // CUDA_VERSION >= 9000


    magma_int_t max_batchCount = queue->get_maxBatch();
    sycl::range<3> threads(1, 1, BLOCK_SIZE);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, 1);

        if ( total_shmem_sm_kernel <= shmem_max ) {
            //load panel in shared memory and factorize it and copy back to gloabl memory
            //intend for small panel to avoid overfill of shared memory.
            //this kernel is composed of device routine and thus clean
            /*
            DPCT1049:381: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    /*
                    DPCT1083:1666: The size of local memory in the migrated code
                    may be different from the original code. Check that the
                    allocated memory size in the migrated code is correct.
                    */
                    sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        dpct_local_acc_ct1(
                            sycl::range<1>(sizeof(magmaDoubleComplex) *
                                           (m * k)),
                            cgh);
                    sycl::accessor<magmaDoubleComplex, 0,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        scale_acc_ct1(cgh);
                    sycl::accessor<magmaDoubleComplex, 1,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sum_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);
                    sycl::accessor<double, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        swork_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);
                    sycl::accessor<double, 0, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sscale_acc_ct1(cgh);

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqr2_sm_kernel_batched(
                                             m, k, dA_array + i, Ai, Aj, ldda,
                                             dtau_array + i, taui, item_ct1,
                                             dpct_local_acc_ct1.get_pointer(),
                                             scale_acc_ct1.get_pointer(),
                                             sum_acc_ct1.get_pointer(),
                                             swork_acc_ct1.get_pointer(),
                                             sscale_acc_ct1.get_pointer());
                                     });
                });
        }
        else if ( total_shmem_column_sm_kernel <= shmem_max ) {
            //load one column vector in shared memory and householder it and used it to update trailing matrix which is global memory
            /*
            DPCT1049:382: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    /*
                    DPCT1083:1667: The size of local memory in the migrated code
                    may be different from the original code. Check that the
                    allocated memory size in the migrated code is correct.
                    */
                    sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        dpct_local_acc_ct1(
                            sycl::range<1>(sizeof(magmaDoubleComplex) * (m)),
                            cgh);
                    sycl::accessor<magmaDoubleComplex, 0,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        scale_acc_ct1(cgh);
                    sycl::accessor<magmaDoubleComplex, 1,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sum_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);
                    sycl::accessor<double, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        swork_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);
                    sycl::accessor<double, 0, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sscale_acc_ct1(cgh);

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqr2_column_sm_kernel_batched(
                                             m, k, dA_array + i, Ai, Aj, ldda,
                                             dtau_array + i, taui, item_ct1,
                                             dpct_local_acc_ct1.get_pointer(),
                                             scale_acc_ct1.get_pointer(),
                                             sum_acc_ct1.get_pointer(),
                                             swork_acc_ct1.get_pointer(),
                                             sscale_acc_ct1.get_pointer());
                                     });
                });
        }
        else {
            //not use dynamic shared memory at all
            /*
            DPCT1049:383: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<magmaDoubleComplex, 0,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        scale_acc_ct1(cgh);
                    sycl::accessor<magmaDoubleComplex, 1,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sum_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);
                    sycl::accessor<double, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        swork_acc_ct1(sycl::range<1>(BLOCK_SIZE), cgh);
                    sycl::accessor<double, 0, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sscale_acc_ct1(cgh);

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqr2_kernel_batched(
                                             m, k, dA_array + i, Ai, Aj, ldda,
                                             dtau_array + i, taui, item_ct1,
                                             scale_acc_ct1.get_pointer(),
                                             sum_acc_ct1.get_pointer(),
                                             swork_acc_ct1.get_pointer(),
                                             sscale_acc_ct1.get_pointer());
                                     });
                });
        }
    }

    return arginfo;
}
