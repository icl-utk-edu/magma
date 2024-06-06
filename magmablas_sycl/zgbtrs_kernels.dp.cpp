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

#define PRECISION_z

#define GBTRS_SWAP_THREADS (128)

#define GBTRS_GERU_THREADS_X (32)
#define GBTRS_GERU_THREADS_Y (4)

#define GBTRS_UPPER_THREADS (128)

#ifdef PRECISION_z
#define GBTRS_LOWER_NB      (4)
#define GBTRS_UPPER_NB      (4)
#elif defined(PRECISION_s)
#define GBTRS_LOWER_NB      (16)
#define GBTRS_UPPER_NB      (16)
#else // d, c
#define GBTRS_LOWER_NB      (8)
#define GBTRS_UPPER_NB      (8)
#endif

#define GBTRS_LOWER_NRHS    (4)
#define GBTRS_UPPER_NRHS    (4)

////////////////////////////////////////////////////////////////////////////////


void zgbtrs_swap_kernel_batched(
        int n,
        magmaDoubleComplex** dA_array, int ldda,
        magma_int_t** dipiv_array, int j, const sycl::nd_item<3> &item_ct1)
{
    const int ntx = item_ct1.get_local_range(2);
    const int tx = item_ct1.get_local_id(2);
    const int batchid = item_ct1.get_group(2);

    magmaDoubleComplex* dA    = dA_array[batchid];
    magma_int_t*        dipiv = dipiv_array[batchid];

    int jp = dipiv[j] - 1; // undo fortran indexing
    if( j != jp ) {
        for(int i = tx; i < n; i+=ntx) {
            magmaDoubleComplex tmp = dA[i * ldda +  j];
            dA[i * ldda +  j]      = dA[i * ldda + jp];
            dA[i * ldda + jp]      = tmp;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////


void zgeru_kernel_batched(
        int m, int n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex** dX_array, int xi, int xj, int lddx, int incx,
        magmaDoubleComplex** dY_array, int yi, int yj, int lddy, int incy,
        magmaDoubleComplex** dA_array, int ai, int aj, int ldda ,
        const sycl::nd_item<3> &item_ct1)
{
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int ntx = item_ct1.get_local_range(2);
    const int nty = item_ct1.get_local_range(1);
    const int gtx = item_ct1.get_group(2) * ntx + tx;
    const int batchid = item_ct1.get_group(0);

    magmaDoubleComplex* dX    = dX_array[batchid] + xj * lddx + xi;
    magmaDoubleComplex* dY    = dY_array[batchid] + yj * lddy + yi;
    magmaDoubleComplex* dA    = dA_array[batchid] + aj * ldda + ai;

    if(gtx < m) {
        for(int j = ty; j < n; j += nty) {
            dA[j * ldda + gtx] += alpha * dX[gtx * incx] * dY[j * incy];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////


void zgbtrs_upper_columnwise_kernel_batched(
        int n, int kl, int ku, int nrhs, int j,
        magmaDoubleComplex** dA_array, int ldda,
        magmaDoubleComplex** dB_array, int lddb,
        const sycl::nd_item<3> &item_ct1)
{
#define dA(i,j) dA[(j)*ldda + (i)]
#define dB(i,j) dB[(j)*lddb + (i)]

    const int kv      = kl + ku;
    const int tx = item_ct1.get_local_id(2);
    const int ntx = item_ct1.get_local_range(2);
    const int batchid = item_ct1.get_group(2);
    //const int je      = (n-1) - j;

    magmaDoubleComplex* dA = dA_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];

    // advance dA/dB based on j
    dA += j * ldda + kv;
    dB += j;

    const int nupdates = min(kv, j);
    magmaDoubleComplex s;
    for(int rhs = 0; rhs < nrhs; rhs++) {
        s = dB(0,rhs) * MAGMA_Z_DIV(MAGMA_Z_ONE, dA(0,0));
        /*
        DPCT1065:120: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if(tx == 0) dB(0,rhs) = s;
        for(int i = tx; i < nupdates ; i+= ntx) {
            dB(-i-1,rhs) -= s * dA(-i-1,0);
        }
    }

#undef dA
#undef dB
}

////////////////////////////////////////////////////////////////////////////////
template <int MAX_THREADS, int NB>
void zgbtrs_lower_blocked_kernel_batched(
    int n, int kl, int ku, int nrhs, int nrhs_nb, magmaDoubleComplex **dA_array,
    int ldda, magma_int_t **dipiv_array, magmaDoubleComplex **dB_array,
    int lddb, const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local)
{
#define dB(i, j)  dB[(j)*lddb + (i)]
#define sB(i, j)  sB[(j)*sldb + (i)]

    auto zdata = (magmaDoubleComplex *)dpct_local;
    const int kv      = kl + ku;
    const int tx = item_ct1.get_local_id(2);
    const int ntx = item_ct1.get_local_range(2);
    const int bx = item_ct1.get_group(2);
    const int by = item_ct1.get_group(1);
    const int batchid = bx;
    const int my_rhs  = min(nrhs_nb, nrhs - by * nrhs_nb);
    const int sldb    = (NB+kl);

    magmaDoubleComplex* dA = dA_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];
    magma_int_t* dipiv     = dipiv_array[batchid];

    magmaDoubleComplex rA[NB] = {MAGMA_Z_ZERO};
    magmaDoubleComplex* sB    = (magmaDoubleComplex*)zdata;
    int* sipiv                = (int*)( sB + nrhs_nb * sldb );

    // advance dA and dB
    dA += kv+1;
    dB += by * nrhs_nb * lddb;

    int b_elements_1        = min(NB, n);
    magmaDoubleComplex ztmp = MAGMA_Z_ZERO;

    for(int itx = tx; itx < b_elements_1; itx+=ntx) {
        for(int jb = 0; jb < my_rhs; jb++) {
            sB(itx, jb) = dB(itx, jb);
        }
    }

    for(int j = 0; j < n/*n1*/; j+=NB) {
        int nb = min(NB, n-j);
        // read A
        if(nb == NB) {
            #pragma unroll
            for(int ja = 0; ja < NB; ja++) {
                rA[ja] = dA[ja * ldda + tx];
            }
        }
        else{
            #pragma unroll
            for(int ja = 0; ja < NB; ja++) {
                rA[ja] = (ja < nb) ? dA[ja * ldda + tx] : MAGMA_Z_ZERO;
            }
        }


        // read pivot info
        for(int ip = tx; ip < nb; ip+=ntx) {
            sipiv[ip] = (int)( dipiv[ip] );
        }

        // read extra B elements to have a total of (nb + kl) elements
        int b_elements_2 = min(nb+kl-b_elements_1, n-j-b_elements_1);

        for(int itx = tx; itx < b_elements_2; itx+=ntx) {
            for(int jb = 0; jb < my_rhs; jb++) {
                sB(itx+b_elements_1, jb) = dB(itx+b_elements_1, jb);
            }
        }
        /*
        DPCT1065:122: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // swap & rank-1 update
        #pragma unroll
        for(int ja = 0; ja < NB; ja++) {
            // swap: note that the swap only affects the segment we read from B
            // since we always read extra KL elements
            int jp = sipiv[ja] - j - 1;
            if(ja < nb && jp != ja) {
                for(int jb = tx; jb < my_rhs; jb+=ntx) {
                    magmaDoubleComplex ztmp = sB(jp, jb);
                    sB(jp, jb)              = sB(ja, jb);
                    sB(ja, jb)              = ztmp;
                }
            }
            /*
            DPCT1065:124: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // apply
            for(int jb = 0; jb < my_rhs; jb++) {
                sB(tx+ja+1, jb) -= rA[ja] * sB(ja,jb);
            }
            /*
            DPCT1065:125: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

        } // end of swap & rank-1 updates

        // write part of B that is finished and shift the the rest up

        for(int itx = tx; itx < nb; itx+=ntx) {
            for(int jb = 0; jb < my_rhs; jb++) {
                dB(itx, jb) = sB(itx, jb);
            }
        }
        /*
        DPCT1065:123: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // shift up
        int shift_size = b_elements_1 + b_elements_2 - nb;
        #if 0
        for(int itx = tx; itx < shift_size; itx+=ntx) {
            for(int jb = 0; jb < my_rhs; jb++) {
                sB(itx, jb) = sB(itx+nb, jb);
            }
        }
        __syncthreads();
        #else
        for(int is = 0; is < shift_size; is += ntx) {
            int active_threads = min(shift_size-is, ntx);
            for(int jb = 0; jb < my_rhs; jb++) {
                if(tx < active_threads) {
                    ztmp = sB(tx+nb, jb);
                }
                /*
                DPCT1065:126: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                if(tx < active_threads) {
                    sB(tx, jb) = ztmp;
                }
                /*
                DPCT1065:127: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            }
        }
        #endif

        b_elements_1 = shift_size; /*b_elements_2*/;
        dA    += nb * ldda;
        dB    += nb;
        dipiv += nb;
    }
}

////////////////////////////////////////////////////////////////////////////////
template <int MAX_THREADS, int NB>
void zgbtrs_upper_blocked_kernel_batched(
    int n, int kl, int ku, int nrhs, int nrhs_nb, magmaDoubleComplex **dA_array,
    int ldda, magmaDoubleComplex **dB_array, int lddb,
    const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local)
{
#define dA(i, j)  dA[(j)*ldda + (i)]
#define dB(i, j)  dB[(j)*lddb + (i)]
#define sB(i, j)  sB[(j)*sldb + (i)]
#define sBr(i, j) sBr[(j)*sldb + (i)]

    auto zdata = (magmaDoubleComplex *)dpct_local;
    const int kv      = kl + ku;
    const int kb      = NB + kv;
    const int tx = item_ct1.get_local_id(2);
    const int ntx = item_ct1.get_local_range(2);
    const int rtx     = ntx-1-tx;  // reverse tx
    const int bx = item_ct1.get_group(2);
    const int by = item_ct1.get_group(1);
    const int batchid = bx;
    const int my_rhs  = min(nrhs_nb, nrhs - by * nrhs_nb);
    const int sldb    = kb;

    magmaDoubleComplex* dA = dA_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];

    magmaDoubleComplex rA[NB] = {MAGMA_Z_ZERO};
    magmaDoubleComplex* sB    = (magmaDoubleComplex*)zdata;
    magmaDoubleComplex* stmp  = sB + nrhs_nb * sldb;
    magmaDoubleComplex  ztmp  = MAGMA_Z_ZERO;

    // advance dA, dB, sB
    dA += (n-1) * ldda + kv;             // backwards
    dB += (by * nrhs_nb * lddb) + (n-1); // backwards

    for(int itx = tx; itx < kb; itx+=ntx) {sB[itx] = MAGMA_Z_ZERO;}
    /*
    DPCT1065:129: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    magmaDoubleComplex* sBr = sB + (kb-1);
    // we need (NB+kv) elements in one sweep
    int b_elements_1     = min(NB, n);
    for(int itx = rtx; itx < b_elements_1; itx+=ntx) {
        for(int jb = 0; jb < my_rhs; jb++) {
            sBr(-itx, jb) = dB(-itx, jb);
        }
    }
    /*
    DPCT1065:130: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for(int fj = 0; fj < n; fj+=NB) {
        int nb = min(NB, n-fj);
        //int j  = (n-1) - fj;

        // read A
        if(nb == NB) {
            #pragma unroll
            for(int ja = 0; ja < NB; ja++) {
                rA[NB-1-ja] = dA(-rtx, -ja);
            }
        }
        else{
            #pragma unroll
            for(int ja = 0; ja < NB; ja++) {
                rA[NB-1-ja] = (ja < nb) ? dA(-rtx, -ja) : MAGMA_Z_ZERO;
            }
        }

        // read extra B elements to have a total of (nb + kl) elements
        int b_elements_2 = min(kb-b_elements_1, n-fj-b_elements_1);
        for(int itx = rtx; itx < b_elements_2; itx+=ntx) {
            for(int jb = 0; jb < my_rhs; jb++) {
                sBr(-(b_elements_1+itx), jb) = dB(-(b_elements_1+itx), jb);
            }
        }
        /*
        DPCT1065:131: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // apply block of A (divide + rank-1 updates)
        #pragma unroll
        for(int ja = NB-1; ja >= 0; ja--) {
            int jj = (NB-1) - ja;
            if(rtx == 0) {
                stmp[0] = MAGMA_Z_DIV(MAGMA_Z_ONE, rA[ja]);
            }
            /*
            DPCT1065:134: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            for(int jb = tx; jb < my_rhs; jb+=ntx) {
                sB(kb-1-jj, jb) *= stmp[0];
            }
            /*
            DPCT1065:135: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // rank-1 update
            ztmp = (rtx == 0) ? MAGMA_Z_ZERO : rA[ja];
            for(int jb = 0; jb < my_rhs; jb++) {
                sBr(-jj-rtx, jb) -= ztmp * sB(kb-1-jj,jb);
            }
            /*
            DPCT1065:136: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

        } // end of swap & rank-1 updates

        // write part of B that is finished and shift the the rest down

        for(int itx = rtx; itx < nb; itx+=ntx) {
            for(int jb = 0; jb < my_rhs; jb++) {
                dB(-itx, jb) = sBr(-itx, jb);
            }
        }
        /*
        DPCT1065:132: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // shift down
        int shift_size = b_elements_1 + b_elements_2 - nb;
        #if 0
        for(int itx = rtx; itx < shift_size; itx+=ntx) {
            for(int jb = 0; jb < my_rhs; jb++) {
                sBr(-itx, jb) = sBr(-itx-nb, jb);
            }
        }
        #elif 1
        for(int is = 0; is < shift_size; is += ntx) {
            int active_threads = min(ntx, shift_size-is);
            //printf("shift-size = %d, active threads = %d\n", shift_size, active_threads);
            for(int jb = 0; jb < my_rhs; jb++) {
                if(rtx < active_threads) {
                    ztmp = sBr(-rtx-is-nb, jb);
                }
                /*
                DPCT1065:137: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                if(rtx < active_threads) {
                    sBr(-rtx-is, jb) = ztmp;
                }
                /*
                DPCT1065:138: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();
            }
        }
        #else
        if(tx == 0) {
            for(int is = 0; is < shift_size; is++) {
                for(int jb = 0; jb < my_rhs; jb++) {
                    sBr(-is,jb) = sBr(-is-nb,jb);
                }
            }
        }
        #endif
        /*
        DPCT1065:133: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        b_elements_1 = shift_size; /*b_elements_2*/;
        dA    -= nb * ldda;
        dB    -= nb;
    }
}

////////////////////////////////////////////////////////////////////////////////
extern "C"
void magmablas_zgbtrs_swap_batched(
        magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t ldda,
        magma_int_t** dipiv_array, magma_int_t j,
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t nthreads = min(n, GBTRS_SWAP_THREADS);
    magma_int_t nblocks  = batchCount;

    sycl::range<3> grid(1, 1, nblocks);
    sycl::range<3> threads(1, 1, nthreads);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           zgbtrs_swap_kernel_batched(n, dA_array, ldda,
                                                      dipiv_array, j, item_ct1);
                       });
}

////////////////////////////////////////////////////////////////////////////////
extern "C"
void magmablas_zgeru_batched_core(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex** dX_array, magma_int_t xi, magma_int_t xj, magma_int_t lddx, magma_int_t incx,
        magmaDoubleComplex** dY_array, magma_int_t yi, magma_int_t yj, magma_int_t lddy, magma_int_t incy,
        magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
        magma_int_t batchCount, magma_queue_t queue )
{
    if(m == 0 || n == 0 || batchCount == 0) return;

    magma_int_t ntx     = min(m, GBTRS_GERU_THREADS_X);
    magma_int_t nty     = min(n, GBTRS_GERU_THREADS_Y);
    magma_int_t nblocks = magma_ceildiv(m, GBTRS_GERU_THREADS_X);

    sycl::range<3> threads(1, nty, ntx);

    magma_int_t max_batchCount = queue->get_maxBatch();
    for(magma_int_t ib = 0; ib < batchCount; ib += max_batchCount){
        magma_int_t ibatch = min(max_batchCount, batchCount - ib);
        sycl::range<3> grid(ibatch, 1, nblocks);

        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zgeru_kernel_batched(
                                   m, n, alpha, dX_array + ib, xi, xj, lddx,
                                   incx, dY_array + ib, yi, yj, lddy, incy,
                                   dA_array + ib, ai, aj, ldda, item_ct1);
                           });
    }
}

////////////////////////////////////////////////////////////////////////////////
extern "C"
void magmablas_zgbtrs_upper_columnwise_batched(
        magma_int_t n, magma_int_t kl, magma_int_t ku,
        magma_int_t nrhs, magma_int_t j,
        magmaDoubleComplex** dA_array, magma_int_t ldda,
        magmaDoubleComplex** dB_array, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t kv       = kl + ku;
    magma_int_t nthreads = min(GBTRS_UPPER_THREADS, kv+1);
    magma_int_t nblocks  = batchCount;

    sycl::range<3> grid(1, 1, nblocks);
    sycl::range<3> threads(1, 1, nthreads);
    ((sycl::queue *)(queue->sycl_stream()))
        ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                       [=](sycl::nd_item<3> item_ct1) {
                           zgbtrs_upper_columnwise_kernel_batched(
                               n, kl, ku, nrhs, j, dA_array, ldda, dB_array,
                               lddb, item_ct1);
                       });
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t magmablas_zgbtrs_lower_blocked_batched(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda, magma_int_t **dipiv_array,
    magmaDoubleComplex **dB_array, magma_int_t lddb, magma_int_t batchCount,
    magma_queue_t queue)
{
    magma_int_t nb         = GBTRS_LOWER_NB;
    magma_int_t nrhs_nb    = GBTRS_LOWER_NRHS;
    magma_int_t nthreads   = kl;
    magma_int_t nthreads32 = magma_roundup(nthreads, 32);
    magma_int_t nblocks_x  = batchCount;
    magma_int_t nblocks_y  = magma_ceildiv(nrhs, nrhs_nb);
    magma_int_t sldb       = (nb + kl);


    magma_int_t shmem = 0;
    shmem += sldb * nrhs_nb * sizeof(magmaDoubleComplex);
    shmem += nb * sizeof(int);

    sycl::range<3> threads(1, 1, nthreads);
    sycl::range<3> grid(1, nblocks_y, nblocks_x);

    magma_int_t arginfo = 0;
    int nthreads_max, shmem_max = 0;
    nthreads_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::max_work_group_size>();
    shmem_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::local_mem_size>();
    if (nthreads > nthreads_max || shmem > shmem_max ) {
        arginfo = -100;
        return arginfo;
    }

    switch( nthreads32 ) {
        case 32:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<32,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 64:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<64,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
	case 96:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<96,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 128:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<128,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 160:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<160,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 192:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<192,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 224:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<224,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 256:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<256,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 288:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<288,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 320:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<320,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 352:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<352,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 384:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<384,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 416:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<416,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 448:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<448,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 480:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<480,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 512:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<512,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 544:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<544,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 576:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<576,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 608:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<608,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 640:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<640,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 672:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<672,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 704:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<704,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 736:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<736,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 768:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<768,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 800:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<800,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 832:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<832,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 864:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<864,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 896:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<896,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 928:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<928,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 960:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<960,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 992:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<992,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 1024:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_lower_blocked_kernel_batched<1024,
                                                                GBTRS_LOWER_NB>(
                                n, kl, ku, nrhs, nrhs_nb,
                                dA_array, ldda, dipiv_array,
                                dB_array, lddb, item_ct1,
                                dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        default: arginfo = -100;
    }
    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t magmablas_zgbtrs_upper_blocked_batched(
    magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
    magmaDoubleComplex **dA_array, magma_int_t ldda,
    magmaDoubleComplex **dB_array, magma_int_t lddb, magma_int_t batchCount,
    magma_queue_t queue)
{
    magma_int_t kv         = kl + ku;
    magma_int_t nb         = GBTRS_UPPER_NB;
    magma_int_t nrhs_nb    = GBTRS_UPPER_NRHS;
    magma_int_t nthreads   = kv + 1;
    magma_int_t nthreads32 = magma_roundup(nthreads, 32);
    magma_int_t nblocks_x  = batchCount;
    magma_int_t nblocks_y  = magma_ceildiv(nrhs, nrhs_nb);
    magma_int_t sldb       = (nb + kv);


    magma_int_t shmem = 0;
    shmem += sldb * nrhs_nb * sizeof(magmaDoubleComplex);  // sB
    shmem += 1 * sizeof(magmaDoubleComplex);  // stmp

    sycl::range<3> threads(1, 1, nthreads);
    sycl::range<3> grid(1, nblocks_y, nblocks_x);

    magma_int_t arginfo = 0;
    int nthreads_max, shmem_max = 0;
    nthreads_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::max_work_group_size>();
    shmem_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::local_mem_size>();
    if (nthreads > nthreads_max || shmem > shmem_max ) {
        arginfo = -100;
        return arginfo;
    }

    switch( nthreads32 ) {
        case 32:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<32,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 64:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<64,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 96:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<96,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 128:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<128,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 160:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<160,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 192:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<192,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 224:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<224,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 256:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<256,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 288:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<288,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 320:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<320,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 352:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<352,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 384:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<384,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 416:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<416,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 448:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<448,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 480:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<480,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 512:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<512,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 544:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<544,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 576:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<576,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 608:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<608,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 640:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<640,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 672:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<672,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 704:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<704,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 736:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<736,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 768:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<768,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 800:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<800,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 832:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<832,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 864:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<864,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 896:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<896,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 928:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<928,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 960:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<960,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 992:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<992,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        case 1024:
            ((sycl::queue *)(queue->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                        sycl::range<1>(shmem), cgh);

                    cgh.parallel_for(
                        sycl::nd_range<3>(grid * threads, threads),
                        [=](sycl::nd_item<3> item_ct1) {
                            zgbtrs_upper_blocked_kernel_batched<1024,
                                                                GBTRS_UPPER_NB>(
                                n, kl, ku, nrhs, nrhs_nb, dA_array, ldda,
				dB_array, lddb, item_ct1, dpct_local_acc_ct1.get_multi_ptr<sycl::access::decorated::no>().get());
                        });
                });
        break;
        default: arginfo = -100;
    }
    return arginfo;
}
