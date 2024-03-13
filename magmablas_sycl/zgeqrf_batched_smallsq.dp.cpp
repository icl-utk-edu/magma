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
#include "sync.dp.hpp"
#include "batched_kernel_param.h"

#ifdef MAGMA_HAVE_HIP
#define block_sync    __syncthreads
#else
#define block_sync    magmablas_syncwarp
#endif


#define SLDA(N)    ( (N==15||N==23||N==31)? (N+2) : (N+1) )
template<int N>
#ifdef MAGMA_HAVE_HIP
__launch_bounds__(64) // one warp
#endif

void
zgeqrf_batched_sq1d_reg_kernel(
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t *info_array, magma_int_t batchCount, sycl::nd_item<3> item_ct1,
    uint8_t *dpct_local)
{
    auto zdata = (magmaDoubleComplex *)dpct_local;
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int batchid =
        item_ct1.get_group(2) * item_ct1.get_local_range(1) + ty;
    if(batchid >= batchCount) return;
    if(tx >= N) return;

    const int slda  = SLDA(N);
    magmaDoubleComplex* dA   = dA_array[batchid] + Aj * ldda + Ai;
    magmaDoubleComplex* dtau = dtau_array[batchid] + taui;
    magma_int_t* info = &info_array[batchid];
    // shared memory pointers
    magmaDoubleComplex* sA = (magmaDoubleComplex*)(zdata + ty * slda * N);
    double *sdw = (double *)(zdata + item_ct1.get_local_range(1) * slda * N);
    sdw += ty * N;

    /*
    DPCT1064:445: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex rA[N] = {MAGMA_Z_ZERO};
    magmaDoubleComplex alpha, tau, tmp, zsum, scale = MAGMA_Z_ZERO;
    double sum = MAGMA_D_ZERO, norm = MAGMA_D_ZERO, beta;

    if( tx == 0 ){
        (*info) = 0;
    }

    // init tau
    /*
    DPCT1064:446: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    dtau[tx] = MAGMA_Z_ZERO;
    // read
    #pragma unroll
    for(int i = 0; i < N; i++){
        rA[i] = dA[ i * ldda + tx ];
    }

    #pragma unroll
    for(int i = 0; i < N-1; i++){
        sA[ i * slda + tx] = rA[i];
        sdw[tx] = ( MAGMA_Z_REAL(rA[i]) * MAGMA_Z_REAL(rA[i]) + MAGMA_Z_IMAG(rA[i]) * MAGMA_Z_IMAG(rA[i]) );
        block_sync(item_ct1);
        alpha = sA[i * slda + i];
        sum = MAGMA_D_ZERO;
        #pragma unroll
        for(int j = i; j < N; j++){
            sum += sdw[j];
        }
        norm = sycl::sqrt(sum);
        beta = -sycl::copysign(norm, real(alpha));
        scale = MAGMA_Z_DIV( MAGMA_Z_ONE,  alpha - MAGMA_Z_MAKE(beta, 0));
        tau = MAGMA_Z_MAKE( (beta - real(alpha)) / beta, -imag(alpha) / beta );

        if(tx == i){
            dtau[i] = tau;
        }

        /*
        DPCT1064:447: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        tmp = (tx == i) ? MAGMA_Z_MAKE(beta, MAGMA_D_ZERO) : rA[i] * scale;

        if(tx >= i){
            rA[i] = tmp;
        }

        dA[ i * ldda + tx ] = rA[i];
        /*
        DPCT1064:448: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        rA[i] = (tx == i) ? MAGMA_Z_ONE : rA[i];
        /*
        DPCT1064:449: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        rA[i] = (tx < i) ? MAGMA_Z_ZERO : rA[i];
        tmp = MAGMA_Z_CONJ( rA[i] ) * MAGMA_Z_CONJ( tau );

        block_sync(item_ct1);
#pragma unroll
        for(int j = i+1; j < N; j++){
            sA[j * slda + tx] = rA[j] * tmp;
        }
        block_sync(item_ct1);

        /*
        DPCT1064:450: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        zsum = MAGMA_Z_ZERO;
#pragma unroll
        for(int j = i; j < N; j++){
            zsum += sA[tx * slda + j];
        }
        sA[tx * slda + N] = zsum;
        block_sync(item_ct1);

#pragma unroll
        for(int j = i+1; j < N; j++){
            rA[j] -= rA[i] * sA[j * slda + N];
        }
        block_sync(item_ct1);
    }
    // write the last column
    dA[ (N-1) * ldda + tx ] = rA[N-1];
}

/***************************************************************************//**
    Purpose
    -------
    ZGEQRF computes a QR factorization of a complex M-by-N matrix A:
    A = Q * R.

    This is a batched version of the routine, and works only for small
    square matrices of size up to 32.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The size of the matrix A.  N >= 0.

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

    @ingroup magma_geqrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgeqrf_batched_smallsq(magma_int_t n, magmaDoubleComplex **dA_array,
                             magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
                             magmaDoubleComplex **dtau_array, magma_int_t taui,
                             magma_int_t *info_array, magma_int_t batchCount,
                             magma_queue_t queue) try {
    magma_int_t arginfo = 0;
    magma_int_t m = n;
    if( (m < 0) || ( m > 32 ) ){
        arginfo = -1;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( m == 0 || n == 0) return 0;

    #ifdef MAGMA_HAVE_HIP
    const magma_int_t ntcol = max(1, 64/n);
    #else
    const magma_int_t ntcol = magma_get_zgeqrf_batched_ntcol(m, n);
    #endif

    magma_int_t shmem = ( SLDA(m) * m * sizeof(magmaDoubleComplex) );
    shmem            += ( m * sizeof(double) );
    shmem            *= ntcol;
    magma_int_t nth   = magma_ceilpow2(m);
    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    sycl::range<3> grid(1, 1, gridx);
    sycl::range<3> threads(1, ntcol, nth);

    void *kernel_args[] = {&dA_array, &Ai, &Aj, &ldda, &dtau_array, &taui, &info_array, &batchCount};

    int e = 0;
    // TODO: error handling (sycl exceptions --> MAGMA errors)
    switch(m){
        /*
        DPCT1049:453: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 1: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<1>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:454: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 2: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<2>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:455: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 3: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<3>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:456: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 4: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<4>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:457: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 5: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<5>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:458: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 6: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<6>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:459: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 7: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<7>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:460: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 8: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<8>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:461: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 9: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<9>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:462: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 10: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<10>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:463: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 11: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<11>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:464: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 12: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<12>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:465: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 13: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<13>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:466: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 14: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<14>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:467: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 15: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<15>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:468: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 16: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<16>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:469: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 17: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<17>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:470: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 18: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<18>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:471: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 19: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<19>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:472: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 20: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<20>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:473: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 21: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<21>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:474: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 22: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<22>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:475: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 23: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<23>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:476: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 24: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<24>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:477: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 25: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<25>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:478: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 26: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<26>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:479: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 27: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<27>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:480: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 28: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<28>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:481: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 29: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<29>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:482: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 30: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<30>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:483: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 31: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<31>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        /*
        DPCT1049:484: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 32: ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto dA_array_ct0 = *(magmaDoubleComplex ***)kernel_args[0];
                    auto Ai_ct1 = *(magma_int_t *)kernel_args[1];
                    auto Aj_ct2 = *(magma_int_t *)kernel_args[2];
                    auto ldda_ct3 = *(magma_int_t *)kernel_args[3];
                    auto dtau_array_ct4 =
                        *(magmaDoubleComplex ***)kernel_args[4];
                    auto taui_ct5 = *(magma_int_t *)kernel_args[5];
                    auto info_array_ct6 = *(magma_int_t **)kernel_args[6];
                    auto batchCount_ct7 = *(magma_int_t *)kernel_args[7];

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zgeqrf_batched_sq1d_reg_kernel<32>(
                                             dA_array_ct0, Ai_ct1, Aj_ct2,
                                             ldda_ct3, dtau_array_ct4, taui_ct5,
                                             info_array_ct6, batchCount_ct7,
                                             item_ct1,
                                             dpct_local_acc_ct1.get_pointer());
                                     });
                });
            break;
        default: arginfo = -100;
    }

    /*
    DPCT1000:452: Error handling if-stmt was detected but could not be
    rewritten.
    */
    // TODO
    if (e != 0) {
        /*
        DPCT1001:451: The statement could not be removed.
        */
        arginfo = -100;
    }

    return arginfo;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
