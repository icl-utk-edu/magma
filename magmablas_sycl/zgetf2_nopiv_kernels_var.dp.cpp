/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include <sycl/sycl.hpp>
#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"
#include "shuffle.dp.hpp"

#define PRECISION_z
#include "zgetf2_devicefunc.dp.hpp"
#include "zgetf2_nopiv_devicefunc.dp.hpp"

/******************************************************************************/

void zscal_zgeru_nopiv_tinypivots_kernel_vbatched(
        int max_m, int max_n,
        magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
        double *dtol_array, double eps, magma_int_t *info_array, magma_int_t batchCount,
        const sycl::nd_item<3> &item_ct1)
{
    const int batchid = item_ct1.get_local_id(2) +
                        item_ct1.get_group(2) * item_ct1.get_local_range(2);
    if(batchid >= batchCount) return;

    int my_M    = (int)M[batchid];
    int my_N    = (int)N[batchid];
    int my_ldda = (int)ldda[batchid];
    double tol  = (dtol_array ? dtol_array[batchid] : eps);

    if( my_M <= Ai || my_N <= Aj ) return;

    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *info = &info_array[batchid];

    magmaDoubleComplex rA = dA[0];
    double val = sycl::fabs(MAGMA_Z_REAL(rA)) + sycl::fabs(MAGMA_Z_IMAG(rA));

    // If the tolerance is zero this does nothing
    if(val < tol)
    {
        int sign = (MAGMA_Z_REAL( rA ) < 0 ? -1 : 1);
        rA = MAGMA_Z_MAKE(sign * tol, 0);
        dA[0] = rA;
        *info++;
    }
}

/******************************************************************************/
extern "C"
magma_int_t magma_zscal_zgeru_nopiv_vbatched(
        magma_int_t max_M, magma_int_t max_N,
        magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        double *dtol_array, double eps, magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
        magma_int_t batchCount, magma_queue_t queue)
{
    const int tbx = 256;
    sycl::range<3> threads(1, 1, tbx);

    magma_int_t max_batchCount = queue->get_maxBatch();

    // First check the pivots and replace the tiny ones by the operation's tolerance
    if(dtol_array != NULL || eps != 0) {
        for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
            magma_int_t ibatch = min(max_batchCount, batchCount-i);
            sycl::range<3> grid(1, 1, magma_ceildiv(ibatch, tbx));

            {
                ((sycl::queue *)(queue->sycl_stream()))
                    ->submit([&](sycl::handler &cgh) {
                        magma_int_t *M_i_ct2 = M + i;
                        magma_int_t *N_i_ct3 = N + i;
                        magmaDoubleComplex **dA_array_i_ct4 = dA_array + i;
                        magma_int_t *ldda_i_ct7 = ldda + i;
                        double *dtol_array_dtol_array_i_NULL_ct8 =
                            (dtol_array ? dtol_array + i : NULL);
                        magma_int_t *info_array_i_ct10 = info_array + i;

                        cgh.parallel_for(
                            sycl::nd_range<3>(grid * threads, threads),
                            [=](sycl::nd_item<3> item_ct1) {
                                zscal_zgeru_nopiv_tinypivots_kernel_vbatched(
                                    max_M, max_N, M_i_ct2, N_i_ct3,
                                    dA_array_i_ct4, Ai, Aj, ldda_i_ct7,
                                    dtol_array_dtol_array_i_NULL_ct8, eps,
                                    info_array_i_ct10, ibatch, item_ct1);
                            });
                    });
            }
        }
    }
    // Now call the regular scal and geru routine
    // TODO: the current implementation takes the info array but does not alter it 
    // so it's safe to use this function. If that changes in the future, then this 
    // function should be replaced with one that does not alter the info array 
    return magma_zscal_zgeru_vbatched(
        max_M, max_N, M, N, dA_array, Ai, Aj, ldda, info_array, step, gbstep,
        batchCount, queue
    );
}

/******************************************************************************/
#define dA(i,j)              dA[(j) * my_ldda + (i)]
#define sA(i,j)              sA[(j) * my_M + (i)]
void
zgetf2_nopiv_fused_sm_kernel_vbatched(
        int max_M, int max_N, int max_minMN, int max_MxN,
        magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex** dA_array, int Ai, int Aj, magma_int_t* ldda,
        double* dtol_array, double eps, magma_int_t *info_array,  int gbstep, int batchCount ,
        const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local)
{
    auto zdata = (magmaDoubleComplex *)dpct_local;
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int ntx = item_ct1.get_local_range(2);
    const int batchid =
        (item_ct1.get_group(2) * item_ct1.get_local_range(1)) + ty;
    if(batchid >= batchCount) return;

    // read data of assigned problem
    int my_M         = (int)M[batchid];
    int my_N         = (int)N[batchid];
    int my_ldda      = (int)ldda[batchid];
    double tol       = (dtol_array ? dtol_array[batchid] : eps);
    int my_minmn     = min(my_M, my_N);
    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t* info      = &info_array[batchid];

    // check offsets
    if( my_M <= Ai || my_N <= Aj ) return;
    my_M     -= Ai;
    my_N     -= Aj;
    my_M      = min(my_M, max_M);
    my_N      = min(my_N, max_N);
    my_minmn  = min(my_M, my_N);

    magmaDoubleComplex *sA = (magmaDoubleComplex*)(zdata);
    magmaDoubleComplex reg  = MAGMA_Z_ZERO;
    magmaDoubleComplex rTmp = MAGMA_Z_ZERO;

    int linfo = (gbstep == 0) ? 0 : *info;
    double rx_abs_max = MAGMA_D_ZERO;

    // read
    for(int j = 0; j < my_N; j++){
        for(int i = tx; i < my_M; i+=ntx) {
            sA(i,j) = dA(i,j);
        }
    }
    /*
    DPCT1065:4: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for(int j = 0; j < my_minmn; j++){
        rx_abs_max = sycl::fabs(MAGMA_Z_REAL(sA(j, j))) +
                     sycl::fabs(MAGMA_Z_IMAG(sA(j, j)));
        /*
        DPCT1118:1: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:5: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // If a non-zero tolerance is specified, replace the small diagonal elements 
        // and increment the info to indicate the number of replacements 
        if(rx_abs_max < tol)
        {
            int sign = (MAGMA_Z_REAL( sA(j,j) ) < 0 ? -1 : 1);
            sA(j,j) = MAGMA_Z_MAKE(sign * tol, 0);
            rx_abs_max = tol;
            linfo++;
            /*
            DPCT1118:3: SYCL group functions and algorithms must be encountered
            in converged control flow. You may need to adjust the code.
            */
            /*
            DPCT1065:7: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }

        // If the tolerance is zero, the above condition is never satisfied, so the info
        // will be the first singularity 
        linfo = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (gbstep+j+1) : linfo;

        reg = (rx_abs_max == MAGMA_D_ZERO) ? MAGMA_Z_ONE : MAGMA_Z_DIV( MAGMA_Z_ONE, sA(j,j) );
        for(int i = (tx+j+1); i < my_M; i+=ntx) {
            rTmp    = reg * sA(i,j);
            sA(i,j) = rTmp;
            for(int jj = j+1; jj < my_N; jj++) {
                sA(i,jj) -= rTmp * sA(j,jj);
            }
        }
        /*
        DPCT1118:2: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:6: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    if(tx == 0){
        (*info) = (magma_int_t)( linfo );
    }

    // write A
    for(int j = 0; j < my_N; j++) {
        for(int i = tx; i < my_M; i+=ntx) {
            dA(i,j) = sA(i,j);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zgetf2_nopiv_fused_sm_vbatched(
    magma_int_t max_M, magma_int_t max_N, magma_int_t max_minMN, magma_int_t max_MxN,
    magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    double* dtol_array, double eps, magma_int_t* info_array, magma_int_t gbstep,
    magma_int_t nthreads, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_device_t device;
    magma_getdevice( &device );

    nthreads = nthreads <= 0 ? (max_M/2) : nthreads;
    nthreads = magma_roundup(nthreads, 32);
    nthreads = min(nthreads, 1024);

    // in a variable-size setting, setting ntcol > 1 may lead to
    // kernel deadlocks due to different thread-groups calling
    // syncthreads at different points
    const magma_int_t ntcol = 1;
    int         shmem = ( max_MxN   * sizeof(magmaDoubleComplex) );
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
        //printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;
        return arginfo;
    }

    if( check_launch_only == 1 ) return arginfo;

    try {
          ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        zgetf2_nopiv_fused_sm_kernel_vbatched(
			    max_M, max_N, max_minMN, max_MxN, m, n, dA_array, Ai, Aj,
			    ldda, dtol_array, eps, info_array, gbstep, batchCount, item_ct1,
                            dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            })
            .wait();
      }
      catch (sycl::exception const &exc) {
            arginfo = -100;
      }

    return arginfo;
}


/******************************************************************************/
#define SLDA(n)              ( (((n)+1)%4) == 0 ? (n) : (n+1) )
#define ibatch    (0)
template<int max_N>
void
zgetf2_nopiv_fused_kernel_vbatched(
        int max_M,
        magma_int_t* M, magma_int_t* N,
        magmaDoubleComplex** dA_array, int Ai, int Aj, magma_int_t* ldda,
        double* dtol_array, double eps, 
        magma_int_t* info_array, int batchCount,
        const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local)
{
    auto data = (magmaDoubleComplex *)dpct_local;
    const int tx = item_ct1.get_local_id(2);
    const int batchid = item_ct1.get_group(2) * item_ct1.get_local_range(1) +
                        item_ct1.get_local_id(1);
    if(batchid >= batchCount)return;

    // read data of assigned problem
    int my_M         = (int)M[batchid];
    int my_N         = (int)N[batchid];
    int my_ldda      = (int)ldda[batchid];
    double tol       = (dtol_array ? dtol_array[batchid] : eps);
    int my_minmn     = (int)min(my_M, my_N);
    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;

    // check offsets
    if( my_M <= Ai || my_N <= Aj ) return;
    // (my_M, my_N) based on (M,N) and offsets (Ai,Aj)
    my_M     -= Ai;
    my_N     -= Aj;

    // now compare (my_M,my_N) with max_M, max_N
    my_M = min(my_M, max_M);
    my_N = min(my_N, max_N);
    my_minmn  = min(my_M, my_N);

    int rowid, gbstep = Aj;
    int linfo   = (gbstep == 0) ? 0 : info_array[batchid];
    const int slda = SLDA(max_M);
    magmaDoubleComplex  rA[max_N] = {MAGMA_Z_ZERO};

    // init sA into identity
    magmaDoubleComplex* sA = (magmaDoubleComplex*)data;
    #pragma unroll
    for(int j = 0; j < max_N; j++) {
        sA[j * slda + tx] = (j == tx) ? MAGMA_Z_ONE : MAGMA_Z_ZERO;
    }
    /*
    DPCT1065:10: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // read A into sm then mv to reg
    if(tx < my_M) {
        for(int j = 0; j < my_N; j++) {
            sA[j * slda + tx] = dA[j * my_ldda + tx];
        }
    }
    /*
    DPCT1065:11: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

#pragma unroll
    for(int j = 0; j < max_N; j++){
        rA[j] = sA[ j * slda + tx ];
    }
    /*
    DPCT1065:12: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    zgetf2_nopiv_fused_device<max_N>(
             max_M, my_minmn, rA,
             tol,
             sA, linfo, gbstep, rowid,
	     item_ct1);

    /*
    DPCT1065:13: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // write to shared
    #pragma unroll
    for(int j = 0; j < max_N; j++){
        sA[ j * slda + rowid ] = rA[j];
    }
    /*
    DPCT1065:14: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if(tx == 0){
        info_array[batchid] = (magma_int_t)( linfo );
    }

    // write to global
    if(tx < my_M) {
        for(int j = 0; j < my_N; j++) {
            dA[j * my_ldda + tx] = sA[j * slda + tx];
        }
    }
}

/******************************************************************************/
template<int max_N>
static magma_int_t
magma_zgetf2_nopiv_fused_kernel_driver_vbatched(
    magma_int_t max_M,
    magma_int_t* M, magma_int_t* N,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    double* dtol_array, double eps, magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_device_t device;
    magma_getdevice( &device );

    // this kernel works only if m <= n for every matrix
    // this is only for short-wide sizes that fit in shared memory
    // should not affect performance for other shapes
    max_M = max(max_M, max_N);

    int ntcol = 1;
    int shmem = 0, shmem_1 = 0, shmem_2 = 0;
    shmem_1 += max_N * sizeof(magmaDoubleComplex);
    shmem_1 += max_M * sizeof(double);

    shmem_2 += SLDA(max_M) * max_N * sizeof(magmaDoubleComplex);

    shmem  = max(shmem_1, shmem_2);
    shmem *= ntcol;

    sycl::range<3> grid(1, 1, magma_ceildiv(batchCount, ntcol));
    sycl::range<3> threads(1, ntcol, max_M);
    
    int nthreads_max, nthreads = max_M * ntcol, shmem_max = 0;
    nthreads_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::max_work_group_size>();
    shmem_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::local_mem_size>();

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;
        return arginfo;
    }

    try {
          ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
        		zgetf2_nopiv_fused_kernel_vbatched<max_N>(
   			    max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array,
			    batchCount, item_ct1,
                            dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            })
            .wait();
      }
      catch (sycl::exception const &exc) {
          arginfo = -100;
      }

    return arginfo;
}

/******************************************************************************/
extern "C" magma_int_t
magma_zgetf2_nopiv_fused_vbatched(
    magma_int_t max_M, magma_int_t max_N,
    magma_int_t max_minMN, magma_int_t max_MxN,
    magma_int_t* M, magma_int_t* N,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    double* dtol_array, double eps, magma_int_t *info_array, magma_int_t batchCount,
    magma_queue_t queue)
{
    //printf("max_M = %d, max_N = %d\n", max_M, max_N);

    magma_int_t info = 0;
    if(max_M < 0 ) {
        info = -1;
    }
    else if(max_N < 0){
        info = -2;
    }

    if(info < 0) return info;


    info = -1;
    switch(max_N) {
        case  1: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched< 1>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case  2: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched< 2>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case  3: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched< 3>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case  4: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched< 4>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case  5: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched< 5>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case  6: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched< 6>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case  7: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched< 7>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case  8: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched< 8>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case  9: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched< 9>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 10: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<10>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 11: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<11>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 12: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<12>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 13: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<13>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 14: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<14>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 15: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<15>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 16: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<16>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 17: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<17>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 18: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<18>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 19: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<19>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 20: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<20>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 21: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<21>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 22: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<22>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 23: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<23>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 24: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<24>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 25: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<25>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 26: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<26>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 27: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<27>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 28: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<28>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 29: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<29>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 30: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<30>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 31: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<31>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        case 32: info = magma_zgetf2_nopiv_fused_kernel_driver_vbatched<32>(max_M, M, N, dA_array, Ai, Aj, ldda, dtol_array, eps, info_array, batchCount, queue); break;
        default: ;
    }

    if( info != 0 ) {
        // try sm version
        magma_int_t sm_nthreads = max(32, max_M / 2);
        sm_nthreads = magma_roundup(sm_nthreads, 32);
        info = magma_zgetf2_nopiv_fused_sm_vbatched(
                    max_M, max_N, max_minMN, max_MxN,
                    M, N, dA_array, Ai, Aj, ldda,
                    dtol_array, eps,
                    info_array, Aj, sm_nthreads, 0, batchCount, queue );
    }

    return info;
}
