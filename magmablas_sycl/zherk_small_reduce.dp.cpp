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

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "atomics.dp.hpp"
#include "batched_kernel_param.h"

#define PRECISION_z
#define SLDA(N)    ( (N==15||N==23||N==31)? N : (N+1) )

//-----------------------------------------------------------------------------
void
zherk_small_reduce_scale_beta_kernel(magma_uplo_t uplo, int N, magmaDoubleComplex beta, magmaDoubleComplex* dC, int lddc,
                                     sycl::nd_item<3> item_ct1)
{
    const int gtx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    const int gty = item_ct1.get_group(1) * item_ct1.get_local_range(1) +
                    item_ct1.get_local_id(1);

    magma_int_t lo = (uplo == MagmaLower) ? gty : gtx;
    magma_int_t hi = (uplo == MagmaLower) ? gtx : gty;
    
    if(gtx < N && gty < N && lo <= hi) {
        // ignore the imaginary part of C for complex precisions, as per the definition of HERK
        magmaDoubleComplex rC = dC[gty * lddc + gtx];
        #if defined(PRECISION_z) || defined(PRECISION_c)
        /*
        DPCT1064:1032: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        if (gtx == gty) rC = MAGMA_Z_MAKE(MAGMA_Z_REAL(rC), MAGMA_D_ZERO);
#endif
        dC[gty * lddc + gtx] = beta * rC;
    }
}

//-----------------------------------------------------------------------------
template<int N>
void
zherk_small_reduce_kernel(
        magma_uplo_t uplo, magma_trans_t trans, int k, 
        const magmaDoubleComplex alpha, magmaDoubleComplex *dA, const int ldda, 
        magmaDoubleComplex *dC, const int lddc, const int nthread_blocks,
        sycl::nd_item<3> item_ct1, uint8_t *dpct_local)
{
    auto zdata = (magmaDoubleComplex *)dpct_local;

    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int tz = item_ct1.get_local_id(0);
    const int bx = item_ct1.get_group(2) * item_ct1.get_local_range(0) + tz;
    const int slda = SLDA(N);
    magmaDoubleComplex rTmp = MAGMA_Z_ZERO;

    magmaDoubleComplex* sA = (magmaDoubleComplex*)(zdata);
    sA += tz * slda * N;

    // make sure only nthread_blocks blocks are used
    const int max_nblocks = magma_ceildiv(k, N);
    const int nblocks = min( nthread_blocks, max_nblocks );
    if(bx >= nblocks) return;

    // determine your share of k
    const int segment = magma_roundup(k, nblocks) / nblocks; 
    const int myk = min(segment, k - bx * segment);

    // advance dA
    dA += ( trans == MagmaNoTrans ) ? bx * segment * ldda : bx * segment;

    // main loop
    int kk = 0;
    for(kk = 0; kk < myk-N; kk += N) {
        // read A
        sA[ty * slda + tx] = dA[ty * ldda + tx];
        /*
        DPCT1065:1034: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // multiply A x A^T or A^T x A
        if(trans == MagmaNoTrans) {
            #pragma unroll
            for(int j = 0; j < N; j++){
                rTmp += sA[j * slda + tx] * MAGMA_Z_CONJ( sA[j * slda + ty] );
            }
        }
        else {
            #pragma unroll
            for(int j = 0; j < N; j++){
                rTmp += MAGMA_Z_CONJ(sA[tx * slda + j]) * sA[ty * slda + j];
            }
        }
        /*
        DPCT1065:1035: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // advance A
        dA += ( trans == MagmaNoTrans ) ? N * ldda : N;
    }
    
    // txy is used for last partial block
    const int txy = (trans == MagmaNoTrans) ? ty : tx;
    if(txy < myk-kk) {
        sA[ty * slda + tx] = dA[ty * ldda + tx];
    }
    else {
        sA[ty * slda + tx] = MAGMA_Z_ZERO;
    }
    /*
    DPCT1065:1033: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
    // multiply A x A^T or A^T x A
    if(trans == MagmaNoTrans) {
        #pragma unroll
        for(int j = 0; j < N; j++){
            rTmp += sA[j * slda + tx] * MAGMA_Z_CONJ( sA[j * slda + ty] );
        }
    }
    else {
        #pragma unroll
        for(int j = 0; j < N; j++){
            rTmp += MAGMA_Z_CONJ(sA[tx * slda + j]) * sA[ty * slda + j];
        }
    }

    // write through atomics
    magma_int_t tlo = (uplo == MagmaLower) ? ty : tx;
    magma_int_t thi = (uplo == MagmaLower) ? tx : ty;
    if(tlo <= thi)
        magmablas_zatomic_add(dC + ty*lddc + tx, alpha * rTmp);
}

/***************************************************************************//**
    Purpose
    -------
    ZHERK performs one of the Hermitian rank k operations

    C := alpha*A*A**H + beta*C,

    or

    C := alpha*A**H*A + beta*C,

    where alpha and beta are real scalars, C is an n by n Hermitian
    matrix and A is an n by k matrix in the first case and a k by n
    matrix in the second case.

    This is a special routine that supports n up to 32 only. It assumes that 
    k is very large so that the computation of the small matrix C is distributed 
    across many thread blocks. The number of thread blocks can be defined by the 
    user through the interface. However, the kernel can work with a maximum of 
    ceil(k / n) thread blocks. Extra thread blocks, if any, are ignored by the kernel. 
    Reduction across thread blocks is performed using atomics. 

    Parameters
    ----------

    @param[in]
    uplo    magma_uplo_t.
           On entry, uplo specifies whether the upper or lower
           triangular part of the array C is to be referenced as
           follows:

           uplo = MagmaUpper Only the upper triangular part of C
           is to be referenced.

           uplo = MagmaLower Only the lower triangular part of C
           is to be referenced.

    @param[in]
    trans   magma_trans_t.
            On entry, trans specifies the operation to be performed as
            follows:

            trans = MagmaNoTrans,   C := alpha*A*A**H + beta*C.

            trans = MagmaConjTrans, C := alpha*A**H*A + beta*C.

    @param[in]
    n       INTEGER.
            On entry,  specifies the order of the matrix C. N must be
            at least zero, and at most 32.

    @param[in]
    k       INTEGER.
            On entry with trans = MagmaNoTrans, k specifies the number
            of columns of the matrix A, and on entry with
            trans = MagmaConjTrans, k specifies the number of rows of the
            matrix A. K must be at least zero.

    @param[in]
    alpha   DOUBLE PRECISION
            On entry, ALPHA specifies the scalar alpha.
    
    @param[in]
    dA       A COMPLEX_16 array DIMENSION ( ldda, ka ), where ka is
             k  when  trans = MagmaNoTrans,  and is  n  otherwise.
             Before entry with  trans = MagmaNoTrans,  the leading  n by k
             part of the array A must contain the matrix A, otherwise
             the leading  k by n  part of the array A must contain  the
             matrix A.
    
    @param[in]
    ldda    INTEGER.
            On entry, ldda specifies the first dimension of A as declared
            in the calling (sub) program. When  trans = MagmaNoTrans then
            ldda must be at least  max( 1, n ), otherwise  ldda must be at
            least  max( 1, k ).
    
    @param[in]
    beta    DOUBLE PRECISION.
            On entry,  BETA  specifies the scalar  beta.  When  BETA  is
            supplied as zero then C need not be set on input.

    @param[in,out]
    dC       A COMPLEX_16 array of DIMENSION ( lddc, n ).
             Before entry with uplo = MagmaUpper, the leading n by n
             upper triangular part of the array C must contain the upper
             triangular part of the Hermitian matrix and the strictly
             lower triangular part of C is not referenced. On exit, the
             upper triangular part of the array C is overwritten by the
             upper triangular part of the updated matrix.
             Before entry with uplo = MagmaLower, the leading n by n
             lower triangular part of the array C must contain the lower
             triangular part of the Hermitian matrix and the strictly
             upper triangular part of C is not referenced. On exit, the
             lower triangular part of the array C is overwritten by the
             lower triangular part of the updated matrix.
             Note that the imaginary parts of the diagonal elements need
             not be set, they are assumed to be zero, and on exit they
             are set to zero.

    @param[in]
    lddc    INTEGER.
            On entry, lddc specifies the first dimension of C as declared
            in  the  calling  (sub)  program.   lddc  must  be  at  least
            max( 1, n ).
    
    @param[in]
    nthread_blocks  INTEGER
                    The number of thread blocks used to update C.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_herk
*******************************************************************************/
extern "C" void 
magmablas_zherk_small_reduce( 
    magma_uplo_t uplo, magma_trans_t trans, 
    magma_int_t n, magma_int_t k, 
    double alpha, magmaDoubleComplex* dA, magma_int_t ldda,
    double beta,  magmaDoubleComplex* dC, magma_int_t lddc, 
    magma_int_t nthread_blocks, magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -1;
    #if defined(PRECISION_c) || defined(PRECISION_z)
    else if ( trans != MagmaNoTrans && trans != MagmaConjTrans )
    #else
    else if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
    #endif
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( k < 0 )
        info = -4;
    else if ( trans == MagmaNoTrans ? ldda < n : ldda < k )
        info = -7;
    else if ( lddc < n )
        info = -10;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;   // info
    }

    /*
    DPCT1064:1037: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex z_alpha = MAGMA_Z_MAKE(alpha, 0.0);
    magmaDoubleComplex z_beta = MAGMA_Z_MAKE(beta, 0.0);

    // This routine supports output matrix size up to 32x32 only
    assert(n <= 32);

    // first, scale by beta
    sycl::range<3> scale_block(1, 16, 16);
    sycl::range<3> scale_grid(1, magma_ceildiv(n, scale_block[1]),
                              magma_ceildiv(n, scale_block[2]));
    /*
    DPCT1049:1036: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(scale_grid * scale_block, scale_block),
                       [=](sycl::nd_item<3> item_ct1) {
                           zherk_small_reduce_scale_beta_kernel(
                               uplo, n, z_beta, dC, lddc, item_ct1);
                       });

    // second, alpha A x A^T or alpha A^T x A
    magma_int_t slda = SLDA(n);
    /*
    DPCT1083:1039: The size of local memory in the migrated code may be
    different from the original code. Check that the allocated memory size in
    the migrated code is correct.
    */
    magma_int_t shmem = slda * n * sizeof(magmaDoubleComplex);

    // check num threads and shmem
    assert(n * n <= MAX_NTHREADS);
    assert(shmem <= (47 * 1024)); // 47 KB max per thread block

    sycl::range<3> grid(1, 1, nthread_blocks);
    sycl::range<3> threads(1, n, n);

    switch(n){
        /*
        DPCT1049:1038: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 1: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<1>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1040: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 2: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<2>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1041: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 3: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<3>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1042: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 4: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<4>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1043: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 5: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<5>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1044: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 6: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<6>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1045: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 7: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<7>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1046: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 8: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<8>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1047: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 9: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<9>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1048: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 10: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<10>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1049: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 11: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<11>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1050: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 12: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<12>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1051: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 13: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<13>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1052: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 14: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<14>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1053: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 15: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<15>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1054: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 16: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<16>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1055: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 17: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<17>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1056: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 18: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<18>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1057: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 19: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<19>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1058: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 20: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<20>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1059: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 21: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<21>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1060: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 22: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<22>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1061: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 23: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<23>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1062: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 24: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<24>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1063: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 25: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<25>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1064: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 26: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<26>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1065: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 27: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<27>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1066: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 28: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<28>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1067: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 29: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<29>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1068: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 30: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<30>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1069: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 31: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<31>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        /*
        DPCT1049:1070: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        case 32: ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<uint8_t, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    dpct_local_acc_ct1(sycl::range<1>(shmem), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zherk_small_reduce_kernel<32>(
                                         uplo, trans, k, z_alpha, dA, ldda, dC,
                                         lddc, nthread_blocks, item_ct1,
                                         dpct_local_acc_ct1.get_pointer());
                                 });
            });
            break;
        default: {printf("N = %lld is not supported\n", (long long)n);}
    }
}


