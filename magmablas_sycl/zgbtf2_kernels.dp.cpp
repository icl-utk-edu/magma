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
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#include "batched_kernel_param.h"
#include "magma_templates.h"
#include "shuffle.dp.hpp"
#include "zgetf2_devicefunc.dp.hpp"

#define PRECISION_z

#define A(i, j)  (A + (i) + (j)*lda)   // A(i, j) means at i row, j column

#define GBTF2_JU_FILLIN_MAX_THREADS (64)
#define GBTF2_SWAP_MAX_THREADS      (128)
#define GBTF2_SCAL_GER_MAX_THREADS  (64)

/******************************************************************************/
// This kernel must be called before pivot adjustment and before updating ju

void
zgbtf2_adjust_ju_fillin_kernel_batched(
    int n, int kl, int ku,
    magmaDoubleComplex** dAB_array, int lddab,
    magma_int_t** dipiv_array, int* ju_array, int gbstep, int batchCount,
    const sycl::nd_item<3> &item_ct1)
{
    const int gtx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2); // global thread x-index
    const int batchid = item_ct1.get_group(0);

    //ju = max(ju, min(j+ku+jp, n-1));
    magma_int_t* ipiv = dipiv_array[batchid];
    magmaDoubleComplex *dAB = dAB_array[batchid];

    int jp   = (int)(ipiv[gbstep]) - 1;    // undo fortran indexing
    int ju1  = (gbstep == 0) ? 0 : ju_array[batchid];
    int ju2  = max(ju1, min(gbstep+ku+jp, n-1));

    if(gtx < kl) {
        for(int j = ju1+1; j <= ju2; j++) {
            dAB[j*lddab + gtx] = MAGMA_Z_ZERO;
        }
    }
}

/******************************************************************************/
// auxiliary routine that sets the necessary fill-in elements based on the new pivot
// must be called before pivot adjustment and before updating ju
extern "C"
void magma_zgbtrf_set_fillin(
        magma_int_t n, magma_int_t kl, magma_int_t ku,
        magmaDoubleComplex** dAB_array, magma_int_t lddab,
        magma_int_t** dipiv_array, int* ju_array, magma_int_t gbstep,
        magma_int_t batchCount, magma_queue_t queue)
{
    // if kl = 0, use at least one thread to set ju
    const int nthreads = min(kl+1, GBTF2_JU_FILLIN_MAX_THREADS);
    const int nblocks  = magma_ceildiv(kl, nthreads);
    sycl::range<3> threads(1, 1, nthreads);
    sycl::range<3> grid(batchCount, 1, nblocks);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           zgbtf2_adjust_ju_fillin_kernel_batched(
                               n, kl, ku, dAB_array, lddab, dipiv_array,
                               ju_array, gbstep, batchCount, item_ct1);
                       });
}

/******************************************************************************/

void zgbtf2_swap_kernel_batched(
        magmaDoubleComplex **dAB_array, magma_int_t ai, magma_int_t aj, magma_int_t lddab,
        magma_int_t** dipiv_array, int ipiv_offset,
        int* ju_array, magma_int_t gbstep, const sycl::nd_item<3> &item_ct1)
{
    const int tx = item_ct1.get_local_id(2);
    const int ntx = item_ct1.get_local_range(2);
    const int batchid = item_ct1.get_group(2);
    magmaDoubleComplex *dAB = dAB_array[batchid] + aj * lddab + ai;
    magma_int_t *ipiv = dipiv_array[batchid] + ipiv_offset;

    int ju = ju_array[batchid];
    int jp = (int)ipiv[0] - 1;
    int swap_len = ju - gbstep + 1;

    if( !(jp == 0) ) {
        magmaDoubleComplex tmp;
        //magmaDoubleComplex *sR1 = &sAB(kv   ,j);
        //magmaDoubleComplex *sR2 = &sAB(kv+jp,j);
        magmaDoubleComplex *dR1 = dAB;      // 1st row with the diagonal
        magmaDoubleComplex *dR2 = dAB + jp; // 2nd row with the pivot
        for(int i = tx; i < swap_len; i+=ntx) {
            tmp                = dR1[i * (lddab-1)];
            dR1[i * (lddab-1)] = dR2[i * (lddab-1)];
            dR2[i * (lddab-1)] = tmp;
        }
    }
}

/******************************************************************************/
extern "C" magma_int_t
magma_zgbtf2_zswap_batched(
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex **dAB_array, magma_int_t ai, magma_int_t aj, magma_int_t lddab,
    magma_int_t** dipiv_array, magma_int_t ipiv_offset,
    int* ju_array, magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)
{
    const int nthreads = min(kl+ku+1, GBTF2_SWAP_MAX_THREADS);
    sycl::range<3> threads(1, 1, nthreads);
    sycl::range<3> grid(1, 1, batchCount);

    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           zgbtf2_swap_kernel_batched(
                               dAB_array, ai, aj, lddab, dipiv_array,
                               ipiv_offset, ju_array, gbstep, item_ct1);
                       });

    return 0;
}


/******************************************************************************/

void zgbtf2_scal_ger_kernel_batched(
    int m, int n, int kl, int ku,
    magmaDoubleComplex **dAB_array, int ai, int aj, int lddab,
    int* ju_array, int gbstep, magma_int_t *info_array,
    const sycl::nd_item<3> &item_ct1)
{
    const int gtx = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
    const int batchid = item_ct1.get_group(0);
    int ju            = ju_array[batchid];
    int swap_length   = ju - gbstep + 1;
    int km            = 1 + min( kl, m-gbstep ); // diagonal + subdiagonal(s)

    if( info_array[batchid] != 0 ) return;

    magmaDoubleComplex* dAB = dAB_array[batchid] + aj * lddab + ai;
    magmaDoubleComplex  rA  = MAGMA_Z_ZERO, reg = MAGMA_Z_ZERO;

    if( gtx > 0 && gtx < km ) {
        reg = MAGMA_Z_DIV(MAGMA_Z_ONE, dAB[0]);
        rA  = dAB[ gtx ];
        rA *= reg;
        dAB[ gtx ] = rA;

        for(int i = 1; i < swap_length; i++)
            dAB[i * (lddab-1) + gtx] -= rA * dAB[i * (lddab-1) + 0];
    }
}


/******************************************************************************/
extern "C"
magma_int_t
magma_zgbtf2_scal_ger_batched(
    magma_int_t m, magma_int_t n, magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex **dAB_array, magma_int_t ai, magma_int_t aj, magma_int_t lddab,
    int* ju_array, magma_int_t gbstep, magma_int_t *info_array,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t km = 1 + min( kl, m-gbstep ); // diagonal + subdiagonal(s)
    magma_int_t nthreads = GBTF2_SCAL_GER_MAX_THREADS;
    magma_int_t nblocks  = magma_ceildiv(km, nthreads);

    sycl::range<3> threads(1, 1, GBTF2_SCAL_GER_MAX_THREADS);

    magma_int_t max_batchCount = queue->get_maxBatch();
    for(magma_int_t s = 0; s < batchCount; s+=max_batchCount) {
        magma_int_t ibatch = min(batchCount-s, max_batchCount);
        sycl::range<3> grid(ibatch, 1, nblocks);

        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zgbtf2_scal_ger_kernel_batched(
                                   m, n, kl, ku, dAB_array + s, ai, aj, lddab,
                                   ju_array + s, gbstep, info_array + s,
                                   item_ct1);
                           });
    }
    return 0;
}

/******************************************************************************/
void zgbtf2_native_kernel(
    int m, int n, int nb, int kl, int ku,
    magmaDoubleComplex *dA, int ldda, magma_int_t *ipiv,
    int* ju, int gbstep, magma_int_t *dinfo, const sycl::nd_item<3> &item_ct1,
    sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst, sycl::memory_scope::device, sycl::access::address_space::global_space> &sync_ct1,
    uint8_t *dpct_local)
{
#define dA(i,j) dA[(j)*ldda + (i)]
    auto zdata = (magmaDoubleComplex *)dpct_local;
    const int tx = item_ct1.get_local_id(2);
    const int ntx = item_ct1.get_local_range(2);
    const int bx = item_ct1.get_group(2);
    const int jc     = bx + gbstep;
    const int kv     = kl + ku;
    const int mband  = kv + 1 + kl;

    int linfo    = (gbstep == 0) ?  0 : *dinfo;
    int local_ju = (gbstep == 0) ? -1 : *ju;
    int jp = 0;
    double rx_abs_max = 0;
    magmaDoubleComplex tmp = MAGMA_Z_ZERO, reg = MAGMA_Z_ZERO;

    // setup shared memory
    magmaDoubleComplex* sA = zdata;
    double*             sX = (double*)(sA + mband);
    int*                sI = (int*)(sX + kl+1);

    // init sA to zero & sX to [0, 1, 2, ...]
    for(int i = tx; i < mband; i+=ntx)
        sA[i] = MAGMA_Z_ZERO;
    for(int i = tx; i < (kl+1); i+=ntx)
        sX[i] = i;
    /*
    DPCT1065:157: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // determine column start/end
    int col_start = 0, col_end = 0;
    if( jc <= local_ju ) {
        // columns affected by previous factorization steps
        col_start  = 0;
        col_end    = mband-1;
    }
    else {
        // new columns
        col_start  = kl + max(ku-jc,0);
        col_end    = kl + ku + min(kl, n-1-jc);
    }

    // read columns
    for(int i = col_start+tx; i <= col_end; i+=ntx) {
        sA[i] = dA(i, jc);
    }
    /*
    DPCT1065:158: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // main loop
    for(int j = 0; j < nb; j++) {
        int gbj = j + gbstep;
        int km  = 1 + min( kl, m-gbj ); // diagonal and subdiagonal(s)
        // find pivot
        if(bx == j) {
            if(km >= 128) {
                for(int i = tx; i < km; i+=ntx) {
                    sX[i] = sycl::fabs(MAGMA_Z_REAL(sA[kv + i])) +
                            sycl::fabs(MAGMA_Z_IMAG(sA[kv + i]));
                    sI[i] = i;
                }
                /*
                DPCT1065:161: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                magma_getidmax_n(km, tx, sX, sI, item_ct1);
                jp         = sI[0];
                rx_abs_max = sX[0];
            }
            else{
                for(int i = tx; i < km; i+=ntx) {
                    sX[i] = sycl::fabs(MAGMA_Z_REAL(sA[kv + i])) +
                            sycl::fabs(MAGMA_Z_IMAG(sA[kv + i]));
                }
                /*
                DPCT1065:162: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                rx_abs_max = sX[0];
                jp = 0;
                for(int i = 1; i < km; i++) {
                    if( sX[i] > rx_abs_max ) {
                        rx_abs_max = sX[i];
                        jp         = i;
                    }
                }
            }

            linfo    = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (gbj+1) : linfo;
            local_ju = max(local_ju, min(gbj+ku+jp, n-1));

            if(tx == 0) {
                ipiv[gbj] = jp + gbj + 1;  // +1 for fortran indexing
                *dinfo    = (magma_int_t)linfo;
                *ju       = local_ju;
            }
        }
        dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);

        // read information written by j-th block
        if(bx >= j) {
            jp       = ipiv[gbj] - gbj - 1;
            linfo    = (int)(*dinfo);
            local_ju = *ju;
        }
        //local_ju = max(local_ju, min(gbstep+j+ku+jp, n-1));
        //swap_length = local_ju - (j+gbstep) + 1;
        /*
        DPCT1065:159: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // swap
        if(jc >= (j+gbstep) && jc <= local_ju && tx == 0) {
            if(jp != 0) {
                int j1 = (kv +  0) - (bx-j);
                int j2 = (kv + jp) - (bx-j);
                tmp    = sA[j1];
                sA[j1] = sA[j2];
                sA[j2] = tmp;
            }
        }
        /*
        DPCT1065:160: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // scal & write to global memory
        if(bx == j) {
            reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sA[kv] );
            for(int i = tx; i < (km-1); i+=ntx) {
                sA[kv+1+i] *= reg;
            }
            /*
            DPCT1065:163: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            for(int i = tx; i < mband; i+=ntx) {
                dA(i,jc) = sA[i];
            }
        }
        dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);

        // ger
        if(jc > gbj && jc <= local_ju) {
            int j1 = (kv + 0) - (bx-j);
            for(int i = tx; i < km-1; i+=ntx) {
                sA[j1+1+i] -= sA[j1] * dA(kv+1+i,gbj);
            }
            /*
            DPCT1065:164: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
    }

    // write columns [nb : ju]
    if(jc >= gbstep && jc <= local_ju) {
        for(int i = tx; i < mband; i+=ntx) {
            dA(i,jc) = sA[i];
        }
    }
#undef dA
}

/******************************************************************************/
extern "C" magma_int_t
magma_zgbtf2_native_work(magma_int_t m, magma_int_t n, magma_int_t kl,
                         magma_int_t ku, magmaDoubleComplex *dA,
                         magma_int_t ldda, magma_int_t *ipiv, magma_int_t *info,
                         void *device_work, magma_int_t *lwork,
                         magma_queue_t queue)
{
    magma_int_t kv    = kl + ku;
    magma_int_t mband = kv + 1 + kl;

    *info  = 0;
    if( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( kl < 0 )
        *info = -3;
    else if ( ku < 0 )
        *info = -4;
    else if ( ldda < mband )
        *info = -6;

    // calculate workspace required
    magma_int_t lwork_required = 0;
    lwork_required += 1 * sizeof(magma_int_t); // ju
    lwork_required += 1 * sizeof(magma_int_t); // dinfo

    if(*lwork < 0) {
       // query assumed
       *lwork = lwork_required;
       return *info;
    }

    if(*lwork < lwork_required) {
        *info = -11;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_int_t sm_count = magma_getdevice_multiprocessor_count();
    magma_int_t nb       = max(8, sm_count - (kv + 1));
    magma_int_t nthreads = magma_roundup(kv+1,32);
    int nthreads_max = queue->sycl_stream()->get_device()
	    .get_info<sycl::info::device::max_work_group_size>();
    if ( nthreads > nthreads_max) {
        *info = -100;
        return *info;
    }

    // device pointers
    int *ju    = (int *)device_work;
    magma_int_t *dinfo = (magma_int_t*) ju + 1;

    magma_int_t shmem = 0;
    shmem += mband  * sizeof(magmaDoubleComplex);
    shmem += (kl+1) * sizeof(double);
    shmem += (kl+1) * sizeof(int);

    sycl::range<3> threads(1, 1, nthreads);
    for(magma_int_t gbstep = 0; gbstep < n; gbstep += nb) {
        magma_int_t ib      = min(nb, n-gbstep);
        magma_int_t nblocks = min(ib+kv+1, n-gbstep);
        sycl::range<3> grid(1, 1, nblocks);
        dpct::global_memory<unsigned int, 0> d_sync_ct1(0);
        unsigned *sync_ct1 = d_sync_ct1.get_ptr(dpct::get_default_queue());
        dpct::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();
        try {
  	  ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shmem), cgh);
                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        auto atm_sync_ct1 = sycl::atomic_ref<
                            unsigned int, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                            sync_ct1[0]);
                        zgbtf2_native_kernel(m, n, nb, kl,
                                             ku, dA, ldda, ipiv,
                                             ju, gbstep, dinfo,
                                             item_ct1, atm_sync_ct1,
                                             dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            }).wait();
	}  
        catch (sycl::exception const &exc) {
            *info = -100;
        }
    }

    magma_igetvector_async( 1, dinfo, 1, info, 1, queue );

    return *info;
}

/******************************************************************************/
extern "C"
magma_int_t
magma_zgbtf2_native(
    magma_int_t m, magma_int_t n, magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t* ipiv,
    magma_int_t* info, magma_queue_t queue)
{
    magma_int_t kv    = kl + ku;
    magma_int_t mband = kv + 1 + kl;

    *info  = 0;
    if( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( kl < 0 )
        *info = -3;
    else if ( ku < 0 )
        *info = -4;
    else if ( ldda < mband )
        *info = -6;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // query workspace
    magma_int_t lwork[1] = {-1};
    magma_zgbtf2_native_work(m, n, kl, ku, NULL, ldda, NULL, info, NULL, lwork, queue);

    void* device_work = NULL;
    magma_malloc(&device_work, lwork[0]);

    magma_zgbtf2_native_work(m, n, kl, ku, dA, ldda, ipiv, info, device_work, lwork, queue);

    magma_free(device_work);
    return *info;
}

/******************************************************************************/
// kernel for gbtf2 using cooperative groups and 1D cyclic dist. of columns
// among thread-blocks
void zgbtf2_native_kernel_v2(
    int m, int n, int nb, int NB, int kl, int ku,
    magmaDoubleComplex *dA, int ldda, magma_int_t *ipiv,
    int* ju, int gbstep, magma_int_t *dinfo, const sycl::nd_item<3> &item_ct1,
    sycl::atomic_ref<unsigned int, sycl::memory_order::seq_cst, sycl::memory_scope::device, sycl::access::address_space::global_space> &sync_ct1,
    uint8_t *dpct_local)
{
#define dA(i, j) dA[(j)*ldda + (i)]
#define sA(i, j) sA[(j)*slda + (i)]
    auto zdata = (magmaDoubleComplex *)dpct_local;
    const int nb1 = nb + 1;
    const int tx = item_ct1.get_local_id(2);
    const int ntx = item_ct1.get_local_range(2);
    const int bx = item_ct1.get_group(2);
    const int nbx = item_ct1.get_group_range(2);
    const int kv     = kl + ku;
    const int mband  = kv + 1 + kl;
    const int slda   = mband;

    int linfo    = (gbstep == 0) ?  0 : *dinfo;
    int local_ju = (gbstep == 0) ? -1 : *ju;
    int jp = 0;
    double rx_abs_max = 0;
    magmaDoubleComplex tmp = MAGMA_Z_ZERO, reg = MAGMA_Z_ZERO;

    // setup shared memory
    magmaDoubleComplex* sA = zdata;
    double*             sX = (double*)( sA + slda * nb1 );
    int*                sI = (int*)(sX + kl+1);

    // init sA to zero & sX to [0, 1, 2, ...]
    for(int i = tx; i < slda*nb1; i+=ntx)
        sA[i] = MAGMA_Z_ZERO;
    /*
    DPCT1065:167: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // read columns -- nb1 cols/TB
    const int total_columns    = min(n-gbstep, nbx * nb1);
    const int total_factorize  = min(total_columns, NB); //min(nbx * nb, total_columns);
    const int my_total_columns = (total_columns / nbx) + ((bx < (total_columns % nbx)) ? 1 : 0);
    //const int my_last_column   = (my_total_columns-1) * nbx + bx;

    int col_start = 0, col_end = 0, lj = 0, glj = 0;
    for(int j = bx; j < total_columns; j += nbx) {
        // determine column start/end
        int jc = j + gbstep;
        col_start = (jc <= local_ju) ? 0       : kl + max(ku-jc,0);
        col_end   = (jc <= local_ju) ? mband-1 : kl + ku + min(kl, n-1-jc);

        // read columns
        for(int i = col_start+tx; i <= col_end; i+=ntx) {
            sA(i,lj) = dA(i, jc);
        }
        lj++;
    }
    /*
    DPCT1065:168: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for(int j = 0; j < total_factorize; j++) {
        int gbj     = j + gbstep;
        int km      = 1 + min( kl, m-gbj );
        int pivoter = j%nbx;

        // find pivot
        if( bx ==  pivoter) {
            lj = j / nbx;
            if(km >= 128) {
                for(int i = tx; i < km; i+=ntx) {
                    sX[i] = sycl::fabs(MAGMA_Z_REAL(sA(kv + i, lj))) +
                            sycl::fabs(MAGMA_Z_IMAG(sA(kv + i, lj)));
                    sI[i] = i;
                }
                /*
                DPCT1065:174: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                magma_getidmax_n(km, tx, sX, sI, item_ct1);
                jp         = sI[0];
                rx_abs_max = sX[0];
            }
            else{
                for(int i = tx; i < km; i+=ntx) {
                    sX[i] = sycl::fabs(MAGMA_Z_REAL(sA(kv + i, lj))) +
                            sycl::fabs(MAGMA_Z_IMAG(sA(kv + i, lj)));
                }
                /*
                DPCT1065:175: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                rx_abs_max = sX[0];
                jp = 0;
                for(int i = 1; i < km; i++) {
                    if( sX[i] > rx_abs_max ) {
                        rx_abs_max = sX[i];
                        jp         = i;
                    }
                }
            }
            linfo    = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (gbj+1) : linfo;
            local_ju = max(local_ju, min(gbj+ku+jp, n-1));
            /*
            DPCT1065:171: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // swap current col. only, and write pivot info
            if(tx == 0) {
                ipiv[gbj] = jp + gbj + 1;  // +1 for fortran indexing
                *dinfo    = (magma_int_t)linfo;
                *ju       = local_ju;

                if(jp != 0) {
                    int j1 = (kv +  0);
                    int j2 = (kv + jp);
                    tmp    = sA(j1,lj);
                    sA(j1,lj) = sA(j2,lj);
                    sA(j2,lj) = tmp;
                }
            }
            /*
            DPCT1065:172: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // scal current column and write it to global mem.
            /*
            DPCT1064:185: Migrated cuCdiv call is used in a macro/template
            definition and may not be valid for all macro/template uses. Adjust
            the code.
            */
            reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sA(kv,lj) );
            for(int i = tx; i < (km-1); i+=ntx) {
                sA(kv+1+i,lj) *= reg;
            }
            /*
            DPCT1065:173: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            for(int i = tx; i < mband; i+=ntx) {
                dA(i,gbj) = sA(i,lj);
            }
        }
        dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);

        // other blocks read the information
        if( !(bx == pivoter) ) {
            jp       = ipiv[gbj] - gbj - 1;
            linfo    = (int)(*dinfo);
            local_ju = *ju;
        }
        /*
        DPCT1065:169: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // determine your next update column
        //lj  = (bx <= pivoter) ? (gbj / nb1) + 1 : (gbj / nb1);
        lj  = (bx <= pivoter) ? (j / nbx) + 1 : (j / nbx);
        lj  = min(lj, my_total_columns-1);
        glj = min(gbstep + (lj * nbx) + bx, n-1);

        // swap
        if(tx == 0 && glj > gbj && glj <= local_ju) {
            if(jp != 0) {
                int j1 = (kv +  0) - (glj-gbj);
                int j2 = (kv + jp) - (glj-gbj);
                tmp       = sA(j1,lj);
                sA(j1,lj) = sA(j2,lj);
                sA(j2,lj) = tmp;
            }
        }
        /*
        DPCT1065:170: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // ger
        if(glj > gbj && glj <= local_ju) {
            int j1 = (kv + 0) - (glj-gbj);
            for(int i = tx; i < km-1; i+=ntx) {
                sA(j1+1+i,lj) -= sA(j1,lj) * dA(kv+1+i,gbj);
            }
            /*
            DPCT1065:176: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
    } // end of main loop


    int jj = gbstep + (nb * nbx + bx);

    if( jj < n ) {
        for(int i = tx; i < mband; i+=ntx) {
            dA(i,jj) = sA(i,nb);
        }
    }
#undef dA
#undef sA
}

/******************************************************************************/
extern "C" magma_int_t
magma_zgbtf2_native_v2_work(magma_int_t m, magma_int_t n, magma_int_t kl,
                            magma_int_t ku, magmaDoubleComplex *dA,
                            magma_int_t ldda, magma_int_t *ipiv,
                            magma_int_t *info, void *device_work,
                            magma_int_t *lwork, magma_queue_t queue)
{
    magma_int_t kv    = kl + ku;
    magma_int_t mband = kv + 1 + kl;

    *info  = 0;
    if( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( kl < 0 )
        *info = -3;
    else if ( ku < 0 )
        *info = -4;
    else if ( ldda < mband )
        *info = -6;

    // calculate workspace required
    magma_int_t lwork_required = 0;
    lwork_required += 1 * sizeof(magma_int_t); // ju
    lwork_required += 1 * sizeof(magma_int_t); // dinfo

    if(*lwork < 0) {
       // query assumed
       *lwork = lwork_required;
       return *info;
    }

    if(*lwork < lwork_required) {
        *info = -11;
    }

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_int_t nb  = 64;
    magma_int_t nb1 = nb+1;
    magma_int_t NB  = nb * (kv + 1);

    magma_int_t nthreads = magma_roundup(kv+1,32);
    int nthreads_max = queue->sycl_stream()->get_device()
	    .get_info<sycl::info::device::max_work_group_size>();
    if ( nthreads > nthreads_max) {
        *info = -100;
        return *info;
    }
    magma_int_t slda     = mband;

    // device pointers
    int *ju    = (int*)device_work;
    magma_int_t *dinfo = (magma_int_t*) ju + 1;

    magma_int_t shmem = 0;
    shmem += slda  * nb1 * sizeof(magmaDoubleComplex);
    shmem += (kl+1) * sizeof(double);
    shmem += (kl+1) * sizeof(int);

    sycl::range<3> threads(1, 1, nthreads);

    for(magma_int_t gbstep = 0; gbstep < n; gbstep += NB) {
        magma_int_t ib      = min(NB, n-gbstep);
        magma_int_t nblocks = min(ib, kv+1);
        magma_int_t nb      = max(1, ib / nblocks);
        sycl::range<3> grid(1, 1, nblocks);

        dpct::global_memory<unsigned int, 0> d_sync_ct1(0);
        unsigned *sync_ct1 = d_sync_ct1.get_ptr(dpct::get_default_queue());
        dpct::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();
        try { 
	  ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(shmem), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * threads, threads),
                    [=](sycl::nd_item<3> item_ct1) {
                        auto atm_sync_ct1 = sycl::atomic_ref<
                            unsigned int, sycl::memory_order::seq_cst,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
                            sync_ct1[0]);
                        zgbtf2_native_kernel_v2(
                            m, n, nb, NB, kl, ku,
                            dA, ldda, ipiv, ju, gbstep,
                            dinfo, item_ct1, atm_sync_ct1,
                            dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                    });
            })
            .wait();
      }
      catch (sycl::exception const &exc) {
            *info = -100;
      }
    }

    magma_igetvector_async( 1, dinfo, 1, info, 1, queue );

    return *info;
}

/******************************************************************************/
extern "C"
magma_int_t
magma_zgbtf2_native_v2(
    magma_int_t m, magma_int_t n, magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex* dA, magma_int_t ldda, magma_int_t* ipiv,
    magma_int_t* info, magma_queue_t queue)
{
    magma_int_t kv    = kl + ku;
    magma_int_t mband = kv + 1 + kl;

    *info  = 0;
    if( m < 0 )
        *info = -1;
    else if ( n < 0 )
        *info = -2;
    else if ( kl < 0 )
        *info = -3;
    else if ( ku < 0 )
        *info = -4;
    else if ( ldda < mband )
        *info = -6;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    // query workspace
    magma_int_t lwork[1] = {-1};
    magma_zgbtf2_native_v2_work(m, n, kl, ku, NULL, ldda, NULL, info, NULL, lwork, queue);

    void* device_work = NULL;
    magma_malloc(&device_work, lwork[0]);

    magma_zgbtf2_native_v2_work(m, n, kl, ku, dA, ldda, ipiv, info, device_work, lwork, queue);

    magma_free(device_work);
    return *info;
}

