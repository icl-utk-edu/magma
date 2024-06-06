/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
       @author Azzam Haidar

       @precisions normal z -> s d c
*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"
#include "zgeqr2_batched_fused.dp.hpp"
#include "batched_kernel_param.h"

#define PRECISION_z

////////////////////////////////////////////////////////////////////////////////

void
zgeqr2_fused_sm_kernel_batched(
    int M, int N,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t *info_array, magma_int_t batchCount, sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto zdata = (magmaDoubleComplex *)dpct_local;
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int ntx = item_ct1.get_local_range(2);
    const int nty = item_ct1.get_local_range(1);
    const int batchid = item_ct1.get_group(2) * nty + ty;
    if(batchid >= batchCount) return;

    const int slda  = SLDA(M);
    magmaDoubleComplex* dA   = dA_array[batchid] + Aj * ldda + Ai;
    magmaDoubleComplex* dtau = dtau_array[batchid] + taui;
    magma_int_t* info = &info_array[batchid];

    // shared memory pointers
    magmaDoubleComplex* sA    = (magmaDoubleComplex*)(zdata);
    magmaDoubleComplex* sY    = sA   + (nty * slda * N);
    magmaDoubleComplex* stau  = sY   + (nty * N);
    magmaDoubleComplex* sTmp  = stau + nty * N;
    sA    += ty * slda * N;
    sY    += ty * N;
    stau  += ty * N;
    sTmp  += ty * ntx;
    double* snorm = (double*) (sTmp); // must be set after offsetting w.r.t. ty

    magmaDoubleComplex alpha, tau, tmp, scale = MAGMA_Z_ZERO;
    double norm = MAGMA_D_ZERO, norm_no_alpha = MAGMA_D_ZERO, beta;

    if( tx == 0 ){
        (*info) = 0;
    }

    // init tau
    if(tx < N) {
        stau[tx] = MAGMA_Z_ZERO;
    }

    // read
    for(int j = 0; j < N; j++){
        for(int i = tx; i < M; i+=ntx) {
            sA(i,j) = dA[ j * ldda + i ];
        }
    }
    /*
    DPCT1065:422: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for(int j = 0; j < N; j++){
        alpha = sA(j,j);

        zgeqr2_compute_norm(M-j-1, &sA(j+1,j), snorm, tx, ntx, item_ct1);
        // there is a sync at the end of zgeqr2_compute_norm

        norm_no_alpha = snorm[0];
        norm = norm_no_alpha + MAGMA_Z_REAL(alpha) * MAGMA_Z_REAL(alpha) + MAGMA_Z_IMAG(alpha) * MAGMA_Z_IMAG(alpha);
        norm = sycl::sqrt(norm);
        bool zero_nrm = (norm_no_alpha == 0) && (MAGMA_Z_IMAG(alpha) == 0);
        tau   = MAGMA_Z_ZERO;
        scale = MAGMA_Z_ONE;
        if(!zero_nrm) {
            beta = -sycl::copysign(norm, real(alpha));
            scale = MAGMA_Z_DIV( MAGMA_Z_ONE,  alpha - MAGMA_Z_MAKE(beta, 0));
            tau = MAGMA_Z_MAKE( (beta - real(alpha)) / beta, -imag(alpha) / beta );
        }

        if(tx == 0) {
            stau[j] = tau;
            sA(j, j) = MAGMA_Z_ONE;
        }

        // scale the current column below the diagonal
        for(int i = (tx+j+1); i < M; i+=ntx) {
            sA(i,j) *= scale;
        }
        /*
        DPCT1065:425: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // copy the first portion of the column into tmp
        // since M > N and ntx >= N, this portion must
        // have the diagonal
        alpha = (zero_nrm) ? alpha : MAGMA_Z_MAKE(beta, MAGMA_D_ZERO);
        tmp   = (tx ==  j) ? alpha : sA(tx, j);

        // write the column into global memory
        dA[j * ldda + tx] = tmp;
        for(int i = tx+ntx; i < M; i+=ntx) {
            dA[ j * ldda + i ] = sA(i, j);
        }

        // now compute (I - tau * v * v') A
        // first: y = tau * v' * A (row vector)
        zgeqr2_compute_vtA_device(M, N, j, sA, slda, sY, tau, sTmp, tx, ntx,
                                  item_ct1);
        /*
        DPCT1065:426: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // now compute: A = A - v * y
        for(int jj = j+1; jj < N; jj++){
            for(int i = tx+j; i < M; i+=ntx) {
                sA(i,jj) -= sA(i,j) * sY[jj];
            }
        }
        /*
        DPCT1065:427: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    // write tau and the last column
    if(tx < N) {
        dtau[tx] = stau[tx];
    }
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t magma_zgeqr2_fused_sm_batched(
    magma_int_t m, magma_int_t n, magmaDoubleComplex **dA_array, magma_int_t Ai,
    magma_int_t Aj, magma_int_t ldda, magmaDoubleComplex **dtau_array,
    magma_int_t taui, magma_int_t *info_array, magma_int_t nthreads,
    magma_int_t check_launch_only, magma_int_t batchCount,
    magma_queue_t queue) try {
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

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return arginfo;

    // disable this kernel for n > 8
    if( m < n || n > 8) return -100;

    nthreads = min(nthreads, m);

    const magma_int_t ntcol = 1;
    magma_int_t shmem = ( SLDA(m) * n * sizeof(magmaDoubleComplex) );
    shmem            += ( n        * sizeof(magmaDoubleComplex) );  // sY
    shmem            += ( n        * sizeof(magmaDoubleComplex) );  // stau
    shmem            += ( nthreads * sizeof(magmaDoubleComplex) );  // used for snorm and for computing v' * A
    shmem            *= ntcol;
    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    sycl::range<3> grid(1, 1, gridx);
    sycl::range<3> threads(1, ntcol, nthreads);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, shmem_max = 0;
    nthreads_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::max_work_group_size>();
    shmem_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::local_mem_size>();

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        // printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;
        return arginfo;
    }

    if( check_launch_only == 1 ) return arginfo;

    void *kernel_args[] = {&m, &n, &dA_array, &Ai, &Aj, &ldda, &dtau_array, &taui, &info_array, &batchCount};
    ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto M_ct0 = *(int *)kernel_args[0];
                    auto N_ct1 = *(int *)kernel_args[1];
                    auto dA_array_ct2 = *(magmaDoubleComplex ***)kernel_args[2];
                    auto Ai_ct3 = *(magma_int_t *)kernel_args[3];
                    auto Aj_ct4 = *(magma_int_t *)kernel_args[4];
                    auto ldda_ct5 = *(magma_int_t *)kernel_args[5];
                    auto dtau_array_ct6 =
                        *(magmaDoubleComplex ***)kernel_args[6];
                    auto taui_ct7 = *(magma_int_t *)kernel_args[7];
                    auto info_array_ct8 = *(magma_int_t **)kernel_args[8];
                    auto batchCount_ct9 = *(magma_int_t *)kernel_args[9];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqr2_fused_sm_kernel_batched(
                                             M_ct0, N_ct1, dA_array_ct2, Ai_ct3,
                                             Aj_ct4, ldda_ct5, dtau_array_ct6,
                                             taui_ct7, info_array_ct8,
                                             batchCount_ct9, item_ct1,
                                             dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                                     });
                });
    /*
    DPCT1000:433: Error handling if-stmt was detected but could not be
    rewritten.
    */
//    if (e != 0) {
        // printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        /*
        DPCT1001:432: The statement could not be removed.
        */
//        arginfo = -100;
//    }

    return arginfo;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
