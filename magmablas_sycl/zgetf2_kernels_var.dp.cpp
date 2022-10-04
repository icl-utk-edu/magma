/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"
#include "shuffle.dp.hpp"

#define PRECISION_z
#include "zgetf2_devicefunc.dp.hpp"

/******************************************************************************/
void
izamax_kernel_vbatched(
        int length, magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
        magma_int_t** ipiv_array, int ipiv_i,
        magma_int_t *info_array, int step, int gbstep ,
        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto sdata = (double *)dpct_local;

    const int batchid = item_ct1.get_group(2);

    // compute the actual length
    int my_M    = (int)M[batchid];
    int my_N    = (int)N[batchid];
    int my_ldda = (int)ldda[batchid];
    // check if offsets produce out-of-bound pointers
    if( my_M <= Ai || my_N <= Aj ) return;

    // compute the length of the vector for each matrix
    my_M -= Ai;
    my_M  = min(my_M, length);

    magmaDoubleComplex *dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *ipiv = ipiv_array[batchid] + ipiv_i;
    int tx = item_ct1.get_local_id(2);

    double *shared_x = sdata;
    int *shared_idx = (int*)(shared_x + zamax);

    izamax_devfunc(my_M, dA, 1, shared_x, shared_idx, item_ct1);

    if (tx == 0) {
        *ipiv = shared_idx[0] + step + 1; // Fortran Indexing & adjust pivot
        if (shared_x[0] == MAGMA_D_ZERO) {
            info_array[batchid] = shared_idx[0] + step + gbstep + 1;
        }
    }
}

/******************************************************************************/
extern "C" magma_int_t
magma_izamax_vbatched(
        magma_int_t length, magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magma_int_t** ipiv_array, magma_int_t ipiv_i,
        magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
        magma_int_t batchCount, magma_queue_t queue)
{
    sycl::range<3> grid(1, 1, batchCount);
    sycl::range<3> threads(1, 1, zamax);

    /*
    DPCT1049:636: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->cuda_stream()))->submit([&](sycl::handler &cgh) {
        /*
        DPCT1083:1668: The size of local memory in the migrated code may be
        different from the original code. Check that the allocated memory size
        in the migrated code is correct.
        */
        sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            dpct_local_acc_ct1(
                sycl::range<1>(zamax * (sizeof(double) + sizeof(int))), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             izamax_kernel_vbatched(
                                 length, M, N, dA_array, Ai, Aj, ldda,
                                 ipiv_array, ipiv_i, info_array, step, gbstep,
                                 item_ct1, dpct_local_acc_ct1.get_pointer());
                         });
    });

    return 0;
}

/******************************************************************************/

void zswap_kernel_vbatched(
        int max_n, magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t* ldda,
        magma_int_t** ipiv_array, int piv_adjustment, sycl::nd_item<3> item_ct1,
        int *jp)
{
    const int batchid = item_ct1.get_group(2);
    const int my_ldda = (int)ldda[batchid];
    int my_M          = (int)M[batchid];
    int my_N          = (int)N[batchid];
    int my_minmn      = min(my_M, my_N);

    // check if offsets produce out-of-bound pointers
    if( my_M <= Ai || my_N <= Aj || my_minmn <= Ai ) return;

    my_N -= Aj; // this is the maximum possible width
    my_N = min(my_N, max_n);

    // read the pivot entry at Ai
    magma_int_t *ipiv = ipiv_array[batchid] + Ai;

    if (item_ct1.get_local_id(2) == 0) {
        *jp = ipiv[0] - 1; // roll-back Fortran indexing
        // magma_izamax_vbatched adjusts the pivot, so roll it back
        // because Ai and Aj are offsets that already take care of that
        *jp -= piv_adjustment;
    }
    /*
    DPCT1065:637: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (*jp == 0) return; // no swapping required

    magmaDoubleComplex *dA  = dA_array[batchid] + Aj * my_ldda + Ai;
    magmaDoubleComplex *dA1 = dA;
    magmaDoubleComplex *dA2 = dA + *jp;

    zswap_device_v2(my_N, dA1, my_ldda, dA2, my_ldda, item_ct1);
}

/******************************************************************************/
extern "C" magma_int_t
magma_zswap_vbatched(
        magma_int_t max_n, magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t** ipiv_array, magma_int_t piv_adjustment,
        magma_int_t batchCount, magma_queue_t queue)
{
    sycl::range<3> grid(1, 1, batchCount);
    sycl::range<3> threads(1, 1, zamax);

    /*
    DPCT1049:638: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->cuda_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<int, 0, sycl::access_mode::read_write,
                       sycl::access::target::local>
            jp_acc_ct1(cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             zswap_kernel_vbatched(max_n, M, N, dA_array, Ai,
                                                   Aj, ldda, ipiv_array,
                                                   piv_adjustment, item_ct1,
                                                   jp_acc_ct1.get_pointer());
                         });
    });

    return 0;
}

/******************************************************************************/

void zscal_zgeru_1d_generic_kernel_vbatched(
        int max_m, int max_n,
        magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
        magma_int_t *info_array, int step, int gbstep,
        sycl::nd_item<3> item_ct1)
{
    const int batchid = item_ct1.get_group(0);
    int my_M    = (int)M[batchid];
    int my_N    = (int)N[batchid];
    int my_ldda = (int)ldda[batchid];

    if( my_M <= Ai || my_N <= Aj ) return;
    my_M -= Ai; // this is the largest possible m per matrix
    my_N -= Aj; // this is the largest possible n per matrix

    my_M = min(my_M, max_m);
    my_N = min(my_N, max_n);

    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t *info = &info_array[batchid];
    zscal_zgeru_generic_device(my_M, my_N, dA, my_ldda, info, step, gbstep,
                               item_ct1);
}


/******************************************************************************/
extern "C"
magma_int_t magma_zscal_zgeru_vbatched(
        magma_int_t max_M, magma_int_t max_N,
        magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
        magma_int_t batchCount, magma_queue_t queue)
{
    /*
    Specialized kernel which merged zscal and zgeru the two kernels
    1) zscale the first column vector A(1:M-1,0) with 1/A(0,0);
    2) Performe a zgeru Operation for trailing matrix of A(1:M-1,1:N-1) += alpha*x*y**T, where
       alpha := -1.0; x := A(1:M-1,0) and y:= A(0,1:N-1);
    */

    magma_int_t max_batchCount = queue->get_maxBatch();
    const int tbx = 256;
    sycl::range<3> threads(1, 1, tbx);

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        sycl::range<3> grid(ibatch, 1, magma_ceildiv(max_M, tbx));

        /*
        DPCT1049:639: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zscal_zgeru_1d_generic_kernel_vbatched(
                                   max_M, max_N, M + i, N + i, dA_array + i, Ai,
                                   Aj, ldda + i, info_array + i, step, gbstep,
                                   item_ct1);
                           });
    }
    return 0;
}

/******************************************************************************/
#define dA(i,j)              dA[(j) * my_ldda + (i)]
#define sA(i,j)              sA[(j) * my_M + (i)]

void
zgetf2_fused_sm_kernel_vbatched(
        int max_M, int max_N, int max_minMN, int max_MxN,
        magma_int_t *M, magma_int_t *N,
        magmaDoubleComplex** dA_array, int Ai, int Aj, magma_int_t* ldda,
        magma_int_t** dipiv_array, int ipiv_i,
        magma_int_t *info,  int gbstep, int batchCount ,
        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
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
    int my_minmn     = min(my_M, my_N);
    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t* dipiv     = dipiv_array[batchid] + ipiv_i;

    // check offsets
    if( my_M <= Ai || my_N <= Aj || my_minmn <= ipiv_i ) return;
    my_M     -= Ai;
    my_N     -= Aj;
    my_M      = min(my_M, max_M);
    my_N      = min(my_N, max_N);
    my_minmn  = min(my_M, my_N);

    magmaDoubleComplex *sA = (magmaDoubleComplex*)(zdata);
    double *dsx = (double *)(sA + item_ct1.get_local_range(1) * max_MxN);
    int *isx = (int *)(dsx + item_ct1.get_local_range(1) * max_M);
    int *sipiv = (int *)(isx + item_ct1.get_local_range(1) * max_M);
    dsx   += ty * max_M;
    isx   += ty * max_M;
    sipiv += ty * max_minMN;

    magmaDoubleComplex reg  = MAGMA_Z_ZERO;
    magmaDoubleComplex rTmp = MAGMA_Z_ZERO;

    int max_id;
    int linfo = (gbstep == 0) ? 0 : *info;
    double rx_abs_max = MAGMA_D_ZERO;

    // init sipiv
    for(int i = tx; i < my_minmn; i+=ntx) {
        sipiv[i] = 0;
    }

    // read
    for(int j = 0; j < my_N; j++){
        for(int i = tx; i < my_M; i+=ntx) {
            sA(i,j) = dA(i,j);
        }
    }
    /*
    DPCT1065:640: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for(int j = 0; j < my_minmn; j++){
        // izamax and find pivot
        for(int i = j+tx; i < my_M; i+=ntx) {
            dsx[i] = sycl::fabs(x()(sA(i, j))) + sycl::fabs(y()(sA(i, j)));
            isx[ i ] = i-j;
        }
        /*
        DPCT1065:641: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        magma_getidmax_n(my_M - j, tx, dsx + j, isx + j, item_ct1);
        // the above devfunc has syncthreads at the end
        rx_abs_max = dsx[j];
        max_id     = j + isx[j];
        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (gbstep+j+1) : linfo;
        if( tx == 0 ) sipiv[ j ] = max_id;
        /*
        DPCT1065:642: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // swap
        if(max_id != j) {
            for(int i = tx; i < my_N; i+=ntx) {
                reg          = sA(j     ,i);
                sA(j,i)      = sA(max_id,i);
                sA(max_id,i) = reg;
            }
        }
        /*
        DPCT1065:643: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if( linfo == 0 ) {
            /*
            DPCT1064:645: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            reg = MAGMA_Z_DIV(MAGMA_Z_ONE, sA(j, j));
            for(int i = (tx+j+1); i < my_M; i+=ntx) {
                rTmp    = reg * sA(i,j);
                sA(i,j) = rTmp;
                for(int jj = j+1; jj < my_N; jj++) {
                    sA(i,jj) -= rTmp * sA(j,jj);
                }
            }
        }
        /*
        DPCT1065:644: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    if(tx == 0){
        (*info) = (magma_int_t)( linfo );
    }

    // write pivot
    for(int i = tx; i < my_minmn; i+=ntx) {
        dipiv[i] = (magma_int_t)(sipiv[i] + 1);
    }

    // write A
    for(int j = 0; j < my_N; j++) {
        for(int i = tx; i < my_M; i+=ntx) {
            dA(i,j) = sA(i,j);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t magma_zgetf2_fused_sm_vbatched(
    magma_int_t max_M, magma_int_t max_N, magma_int_t max_minMN,
    magma_int_t max_MxN, magma_int_t *m, magma_int_t *n,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj,
    magma_int_t *ldda, magma_int_t **dipiv_array, magma_int_t ipiv_i,
    magma_int_t *info_array, magma_int_t gbstep, magma_int_t nthreads,
    magma_int_t check_launch_only, magma_int_t batchCount,
    magma_queue_t queue) try {
    magma_int_t arginfo = 0;
    magma_device_t device;
    magma_getdevice( &device );

    nthreads = nthreads <= 0 ? (max_M/2) : nthreads;
    #ifdef MAGMA_HAVE_CUDA
    nthreads = magma_roundup(nthreads, 32);
    #else
    nthreads = magma_roundup(nthreads, 64);
    #endif
    nthreads = min(nthreads, 1024);

    // in a variable-size setting, setting ntcol > 1 may lead to
    // kernel deadlocks due to different thread-groups calling
    // syncthreads at different points
    const magma_int_t ntcol = 1;
    int         shmem = ( max_MxN   * sizeof(magmaDoubleComplex) );
    shmem            += ( max_M     * sizeof(double) );
    shmem            += ( max_M     * sizeof(int) );
    shmem            += ( max_minMN * sizeof(int) );
    shmem            *= ntcol;
    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    sycl::range<3> grid(1, 1, gridx);
    sycl::range<3> threads(1, ntcol, nthreads);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, shmem_max = 0;
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        /*
        DPCT1007:648: Migration of cudaFuncSetAttribute is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        cudaFuncSetAttribute(zgetf2_fused_sm_kernel_vbatched,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;
        return arginfo;
    }

    if( check_launch_only == 1 ) return arginfo;

    void *kernel_args[] = {&max_M, &max_N, &max_minMN, &max_MxN, &m, &n, &dA_array, &Ai, &Aj, &ldda, &dipiv_array, &ipiv_i, &info_array, &gbstep, &batchCount};
    /*
    DPCT1049:649: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    int e = ((sycl::queue *)(queue->cuda_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                    auto max_M_ct0 = *(int *)kernel_args[0];
                    auto max_N_ct1 = *(int *)kernel_args[1];
                    auto max_minMN_ct2 = *(int *)kernel_args[2];
                    auto max_MxN_ct3 = *(int *)kernel_args[3];
                    auto M_ct4 = *(magma_int_t **)kernel_args[4];
                    auto N_ct5 = *(magma_int_t **)kernel_args[5];
                    auto dA_array_ct6 = *(magmaDoubleComplex ***)kernel_args[6];
                    auto Ai_ct7 = *(int *)kernel_args[7];
                    auto Aj_ct8 = *(int *)kernel_args[8];
                    auto ldda_ct9 = *(magma_int_t **)kernel_args[9];
                    auto dipiv_array_ct10 = *(magma_int_t ***)kernel_args[10];
                    auto ipiv_i_ct11 = *(int *)kernel_args[11];
                    auto info_ct12 = *(magma_int_t **)kernel_args[12];
                    auto gbstep_ct13 = *(int *)kernel_args[13];
                    auto batchCount_ct14 = *(int *)kernel_args[14];

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgetf2_fused_sm_kernel_vbatched(
                                max_M_ct0, max_N_ct1, max_minMN_ct2,
                                max_MxN_ct3, M_ct4, N_ct5, dA_array_ct6, Ai_ct7,
                                Aj_ct8, ldda_ct9, dipiv_array_ct10, ipiv_i_ct11,
                                info_ct12, gbstep_ct13, batchCount_ct14,
                                item_ct1, dpct_local_acc_ct1.get_pointer());
                        });
                });
    /*
    DPCT1000:647: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        /*
        DPCT1001:646: The statement could not be removed.
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

/******************************************************************************/
#define SLDA(n)              ( (((n)+1)%4) == 0 ? (n) : (n+1) )
#define ibatch    (0)
template<int max_N>
void
zgetf2_fused_kernel_vbatched(
        int max_M,
        magma_int_t* M, magma_int_t* N,
        magmaDoubleComplex** dA_array, int Ai, int Aj, magma_int_t* ldda,
        magma_int_t** dipiv_array, int ipiv_i,
        magma_int_t* info_array, int batchCount, sycl::nd_item<3> item_ct1,
        uint8_t *dpct_local)
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
    int my_minmn     = (int)min(my_M, my_N);
    magmaDoubleComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;
    magma_int_t* dipiv     = dipiv_array[batchid] + ipiv_i;

    // check offsets
    if( my_M <= Ai || my_N <= Aj || my_minmn <= ipiv_i ) return;
    // (my_M, my_N) based on (M,N) and offsets (Ai,Aj)
    my_M     -= Ai;
    my_N     -= Aj;

    // now compare (my_M,my_N) with max_M, max_N
    my_M = min(my_M, max_M);
    my_N = min(my_N, max_N);
    my_minmn  = min(my_M, my_N);

    int rowid, gbstep = Aj;
    int orginfo = (gbstep == 0) ? 0 : info_array[batchid];
    int linfo   = 0;
    const int slda = SLDA(max_M);
    /*
    DPCT1064:654: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex rA[max_N] = {MAGMA_Z_ZERO};

    // init sA into identity
    magmaDoubleComplex* sA = (magmaDoubleComplex*)data;
    #pragma unroll
    for(int j = 0; j < max_N; j++) {
        /*
        DPCT1064:655: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        sA[j * slda + tx] = MAGMA_Z_ZERO;
    }

    // read A into sm then mv to reg
    if(tx < my_M) {
        for(int j = 0; j < my_N; j++) {
            sA[j * slda + tx] = dA[j * my_ldda + tx];
        }
    }
    /*
    DPCT1065:650: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

#pragma unroll
    for(int j = 0; j < max_N; j++){
        rA[j] = sA[ j * slda + tx ];
    }
    /*
    DPCT1065:651: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    zgetf2_fused_device<max_N>(max_M, my_minmn, rA, dipiv, sA, linfo, gbstep,
                               rowid, item_ct1);

    /*
    DPCT1065:652: Consider replacing sycl::nd_item::barrier() with
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
    DPCT1065:653: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // ignore any info beyond minmn
    // (meaning singularity is encountered at the padded matrix)
    linfo = (linfo >= my_minmn) ? 0 : linfo;
    linfo = (orginfo == 0) ? linfo : orginfo;

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
template <int max_N>
static magma_int_t magma_zgetf2_fused_kernel_driver_vbatched(
    magma_int_t max_M, magma_int_t *M, magma_int_t *N,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj,
    magma_int_t *ldda, magma_int_t **dipiv_array, magma_int_t ipiv_i,
    magma_int_t *info_array, magma_int_t batchCount, magma_queue_t queue) try {
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
    shmem_1 += max_M * sizeof(int);    // not magma_int_t
    shmem_1 += max_N * sizeof(int);    // not magma_int_t

    shmem_2 += SLDA(max_M) * max_N * sizeof(magmaDoubleComplex);

    //printf("max_M = %d, max_N = %d\n", max_M, max_N);
    //printf("shmem-1 = %d, shmem-2 = %d\n", shmem_1, shmem_2);

    shmem  = max(shmem_1, shmem_2);
    shmem *= ntcol;

    sycl::range<3> grid(1, 1, magma_ceildiv(batchCount, ntcol));
    sycl::range<3> threads(1, ntcol, max_M);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, nthreads = max_M * ntcol, shmem_max = 0;
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        /*
        DPCT1007:659: Migration of cudaFuncSetAttribute is not supported by the
        Intel(R) DPC++ Compatibility Tool.
        */
        cudaFuncSetAttribute(zgetf2_fused_kernel_vbatched<max_N>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;
        return arginfo;
    }

    void *kernel_args[] = {&max_M, &M, &N, &dA_array, &Ai, &Aj, &ldda, &dipiv_array, &ipiv_i, &info_array, &batchCount};
    /*
    DPCT1007:658: Migration of cudaLaunchKernel is not supported by the Intel(R)
    DPC++ Compatibility Tool.
    */
    int e = cudaLaunchKernel((void *)zgetf2_fused_kernel_vbatched<max_N>, grid,
                             threads, kernel_args, shmem, queue->cuda_stream());
    /*
    DPCT1000:657: Error handling if-stmt was detected but could not be
    rewritten.
    */
    if (e != 0) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        /*
        DPCT1001:656: The statement could not be removed.
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

/******************************************************************************/
extern "C" magma_int_t
magma_zgetf2_fused_vbatched(
    magma_int_t max_M, magma_int_t max_N,
    magma_int_t max_minMN, magma_int_t max_MxN,
    magma_int_t* M, magma_int_t* N,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magma_int_t **dipiv_array, magma_int_t ipiv_i,
    magma_int_t *info_array, magma_int_t batchCount,
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


    info = -1; // init a negative value
    switch(max_N) {
        #if 1
        case  1: info = magma_zgetf2_fused_kernel_driver_vbatched< 1>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  2: info = magma_zgetf2_fused_kernel_driver_vbatched< 2>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  3: info = magma_zgetf2_fused_kernel_driver_vbatched< 3>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  4: info = magma_zgetf2_fused_kernel_driver_vbatched< 4>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  5: info = magma_zgetf2_fused_kernel_driver_vbatched< 5>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  6: info = magma_zgetf2_fused_kernel_driver_vbatched< 6>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  7: info = magma_zgetf2_fused_kernel_driver_vbatched< 7>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  8: info = magma_zgetf2_fused_kernel_driver_vbatched< 8>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case  9: info = magma_zgetf2_fused_kernel_driver_vbatched< 9>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 10: info = magma_zgetf2_fused_kernel_driver_vbatched<10>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 11: info = magma_zgetf2_fused_kernel_driver_vbatched<11>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 12: info = magma_zgetf2_fused_kernel_driver_vbatched<12>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 13: info = magma_zgetf2_fused_kernel_driver_vbatched<13>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 14: info = magma_zgetf2_fused_kernel_driver_vbatched<14>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 15: info = magma_zgetf2_fused_kernel_driver_vbatched<15>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 16: info = magma_zgetf2_fused_kernel_driver_vbatched<16>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 17: info = magma_zgetf2_fused_kernel_driver_vbatched<17>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 18: info = magma_zgetf2_fused_kernel_driver_vbatched<18>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 19: info = magma_zgetf2_fused_kernel_driver_vbatched<19>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 20: info = magma_zgetf2_fused_kernel_driver_vbatched<20>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 21: info = magma_zgetf2_fused_kernel_driver_vbatched<21>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 22: info = magma_zgetf2_fused_kernel_driver_vbatched<22>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 23: info = magma_zgetf2_fused_kernel_driver_vbatched<23>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 24: info = magma_zgetf2_fused_kernel_driver_vbatched<24>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 25: info = magma_zgetf2_fused_kernel_driver_vbatched<25>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 26: info = magma_zgetf2_fused_kernel_driver_vbatched<26>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 27: info = magma_zgetf2_fused_kernel_driver_vbatched<27>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 28: info = magma_zgetf2_fused_kernel_driver_vbatched<28>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 29: info = magma_zgetf2_fused_kernel_driver_vbatched<29>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 30: info = magma_zgetf2_fused_kernel_driver_vbatched<30>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 31: info = magma_zgetf2_fused_kernel_driver_vbatched<31>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        case 32: info = magma_zgetf2_fused_kernel_driver_vbatched<32>(max_M, M, N, dA_array, Ai, Aj, ldda, dipiv_array, ipiv_i, info_array, batchCount, queue); break;
        #endif
        default: ;
    }

    //printf("info from reg = %d\n", info);

    if( info != 0 ) {
        // try sm version
        magma_int_t sm_nthreads = max(32, max_M / 2);
        sm_nthreads = magma_roundup(sm_nthreads, 32);
        info = magma_zgetf2_fused_sm_vbatched(
                    max_M, max_N, max_minMN, max_MxN,
                    M, N, dA_array, Ai, Aj, ldda,
                    dipiv_array, ipiv_i,
                    info_array, Aj, sm_nthreads, 0, batchCount, queue );
    }

    return info;
}
