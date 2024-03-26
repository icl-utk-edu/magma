/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
       @author Stan Tomov

       @precisions normal z -> s d c
*/

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"
#include "zgbtf2_devicefunc.dp.hpp"

// use this so magmasubs will replace with relevant precision, so we can comment out
// the switch case that causes compilation failure
#define PRECISION_z


#ifdef MAGMA_HAVE_HIP
#define NTCOL(M)        (max(1,64/(M)))
#else
#define NTCOL(M)        (max(1,64/(M)))
#endif

#define SLDAB(MBAND)    ((MBAND)+1)

////////////////////////////////////////////////////////////////////////////////
template <int NTX>
void zgbtrf_batched_sliding_window_loopout_kernel_sm(
    int m, int nb, int n, int kl, int ku, magmaDoubleComplex **dAB_array,
    int ABi, int ABj, int lddab, magma_int_t **ipiv_array, int *ju_array,
    magma_int_t *info_array, int batchCount, const sycl::nd_item<3> &item_ct1,
    uint8_t *dpct_local)
{
#define sAB(i,j)        sAB[(j)*sldab + (i)]
#define dAB(i,j)        dAB[(j)*lddab + (i)]

    auto zdata = (magmaDoubleComplex *)dpct_local;
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int ntx = item_ct1.get_local_range(2);
    const int batchid =
        item_ct1.get_group(2) * item_ct1.get_local_range(1) + ty;
    if(batchid >= batchCount) return;

    const int minmn   = min(m,nb);
    const int kv      = kl + ku;
    const int mband   = (kl + 1 + kv);
    const int sldab   = SLDAB(mband);
    const int sldab_1 = sldab-1;

    // the kernel assumes a panel of mband x nb,
    // and accounts for mband x (kv+1) extra space "worst case" for pivoting
    const int nn      = nb + kv + 1;

    magmaDoubleComplex* dAB = dAB_array[batchid];
    magma_int_t* ipiv = ipiv_array[batchid];
    int linfo = (ABj == 0) ? 0 : info_array[batchid];

    // shared memory pointers
    magmaDoubleComplex *sAB = (magmaDoubleComplex*)(zdata);
    double *dsx = (double *)(sAB + item_ct1.get_local_range(1) * sldab * nn);
    int *sipiv = (int *)(dsx + item_ct1.get_local_range(1) * (kl + 1));
    sAB   += ty * nn * sldab;
    dsx   += ty * (kl+1);
    sipiv += ty * minmn;

    // pointers for trailing matrices
    magmaDoubleComplex *sAB_trail = sAB;
    int last_column_read = 0;

    int ju = (ABj == 0) ? 0 : ju_array[batchid];

    // init sAB
    for(int i = tx; i < nn*sldab; i+=ntx) {
        sAB[i] = MAGMA_Z_ZERO;
    }
    /*
    DPCT1065:192: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // read columns ABj to ju, and account for offsets
    int jtmp = (ABj == 0) ? nb-1 : max(ju, ABj+nb-1);
    int juu  = (ju  == 0) ? -1 : ju;

    if( ABj > 0 ) {
        read_sAB_updated_columns(mband, n, ABj, ju, kl, ku, dAB, lddab, sAB, sldab, ntx, tx);
        sAB_trail += sldab * (ju-ABj+1);
    }

    if( ABj+nb-1 > ju ) {
        read_sAB_new_columns(mband, n, juu+1, ABj+nb-1, kl, ku, dAB, lddab, sAB_trail, sldab, ntx, tx);
        sAB_trail += sldab * ((ABj+nb-1) - (juu+1) + 1);
    }
    /*
    DPCT1065:193: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // advance trailing ptrs
    //sAB_trail = sAB + sldab * (jtmp-ABj+1);
    last_column_read = jtmp;

    for(int j = 0; j < minmn; j++) {
        // izamax
        int km = 1 + min( kl, m-j ); // diagonal and subdiagonal(s)
        if(tx < km) {
            dsx[tx] = sycl::fabs(MAGMA_Z_REAL(sAB(kv + tx, j))) +
                      sycl::fabs(MAGMA_Z_IMAG(sAB(kv + tx, j)));
        }
        /*
        DPCT1065:194: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        double rx_abs_max = dsx[0];
        int    jp       = 0;
        for(int i = 1; i < km; i++) {
            if( dsx[i] > rx_abs_max ) {
                rx_abs_max = dsx[i];
                jp         = i;
            }
        }

        //ju = max(ju, min(j+ku+jp, n-1));
        ju  = max(ju, min(ABj+j+ku+jp, n-1));
        int swap_len = ju - (j+ABj) + 1;
        if(tx == 0) {
            sipiv[j] = ABj + jp + j + 1;    // +1 for fortran indexing (adjustment included w.r.t ABj)
        }

        if(ju > last_column_read) {
            // read up to ju into shared memory
            int jstart = min(last_column_read + 1, n-1);
            int jend   = ju;
            read_sAB_new_columns(mband, n, jstart, jend, kl, ku, dAB, lddab, sAB_trail, sldab, ntx, tx);
            /*
            DPCT1065:199: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            last_column_read = ju;
            sAB_trail += sldab * (jend - jstart + 1);
        }

        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (ABj+jp+j+1) : linfo;
        /*
        DPCT1065:195: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier(); // wait for the trailing matrix read

        // swap
        if( !(jp == 0) ) {
            magmaDoubleComplex tmp;
            magmaDoubleComplex *sR1 = &sAB(kv   ,j);
            magmaDoubleComplex *sR2 = &sAB(kv+jp,j);
            for(int i = tx; i < swap_len; i+=ntx) {
                tmp              = sR1[i * sldab_1];
                sR1[i * sldab_1] = sR2[i * sldab_1];
                sR2[i * sldab_1] = tmp;
            }
        }
        /*
        DPCT1065:196: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // scal
        magmaDoubleComplex reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sAB(kv,j) );
        for(int i = tx; i < (km-1); i+=ntx) {
            sAB(kv+1+i, j) *= reg;
        }
        /*
        DPCT1065:197: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // ger
        reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ZERO : MAGMA_Z_ONE;
        magmaDoubleComplex *sU = &sAB(kv,j);
        magmaDoubleComplex *sV = &sAB(kv+1,j);
        if( tx < (km-1) ) {
            for(int jj = 1; jj < swap_len; jj++) {
                sV[jj * sldab_1 + tx] -= sV[tx] * sU[jj * sldab_1 + 0] * reg;
            }
        }
        /*
        DPCT1065:198: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    // write info & ju
    if(tx == 0) {
        info_array[batchid] = linfo;
        ju_array[batchid]   = ju;
    }

    // write pivot
    ipiv += ABj;
    for(int i = tx; i < minmn; i+=ntx) {
        ipiv[i] = (magma_int_t)sipiv[i];
    }

    write_sAB_columns(mband, n, ABj, last_column_read, kl, ku, sAB, sldab, dAB, lddab, ntx, tx);

#undef sAB
#undef dAB
}

////////////////////////////////////////////////////////////////////////////////
template <int NTX>
void zgbtrf_batched_sliding_window_loopin_kernel_sm(
    int m, int nb, int n, int kl, int ku, magmaDoubleComplex **dAB_array,
    int lddab, magma_int_t **ipiv_array, magma_int_t *info_array,
    int batchCount, const sycl::nd_item<3> &item_ct1, uint8_t *dpct_local)
{
#define sAB(i,j)        sAB[(j)*sldab + (i)]
#define dAB(i,j)        dAB[(j)*lddab + (i)]

    auto zdata = (magmaDoubleComplex *)dpct_local;
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int ntx = item_ct1.get_local_range(2);
    const int batchid =
        item_ct1.get_group(2) * item_ct1.get_local_range(1) + ty;
    if(batchid >= batchCount) return;

    const int minmn   = min(m,n);
    const int kv      = kl + ku;
    const int mband   = (kl + 1 + kv);
    const int sldab   = SLDAB(mband);
    const int sldab_1 = sldab-1;

    // the kernel assumes a panel of mband x nb,
    // and accounts for mband x (kv+1) extra space "worst case" for pivoting
    const int nn      = nb + kv + 1;

    magmaDoubleComplex* dAB = dAB_array[batchid];
    magma_int_t* ipiv = ipiv_array[batchid];
    int linfo = 0;

    // shared memory pointers
    magmaDoubleComplex *sAB = (magmaDoubleComplex*)(zdata);
    double *dsx = (double *)(sAB + item_ct1.get_local_range(1) * sldab * nn);
    int *sipiv = (int *)(dsx + item_ct1.get_local_range(1) * (kl + 1));
    sAB   += ty * nn * sldab;
    dsx   += ty * (kl+1);
    sipiv += ty * minmn;

    magmaDoubleComplex *sABtmp = sAB;
    int last_column_read = 0;
    int cached_columns   = 0;   // number of columns cached from previous iteration

    int ju = -1;

    // init sAB
    for(int i = tx; i < nn*sldab; i+=ntx) {
        sAB[i] = MAGMA_Z_ZERO;
    }
    /*
    DPCT1065:201: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for(int gbj = 0; gbj < minmn; gbj+=nb) {
        int ib = min(nb, n-gbj);
        int j1 = ju+1;
        int j2 = ju+ib-cached_columns;
        sABtmp = &sAB(0,cached_columns);

        // read at least ib columns
        if(cached_columns < ib) {
            read_sAB_new_columns(mband, n, j1, j2, kl, ku, dAB, lddab, sABtmp, sldab, ntx, tx);
        }
        /*
        DPCT1065:202: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        last_column_read = (cached_columns < ib) ? j2 : last_column_read;
        cached_columns   = max(cached_columns, ib);
        sABtmp           = &sAB(0,cached_columns);

        // factorization loop
        for(int j = 0; j < ib; j++) {
            // izamax
            int km = 1 + min( kl, m-j ); // diagonal and subdiagonal(s)
            if(tx < km) {
                dsx[tx] = sycl::fabs(MAGMA_Z_REAL(sAB(kv + tx, j))) +
                          sycl::fabs(MAGMA_Z_IMAG(sAB(kv + tx, j)));
            }
            /*
            DPCT1065:204: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            double rx_abs_max = dsx[0];
            int    jp         = 0;
            for(int i = 1; i < km; i++) {
                if( dsx[i] > rx_abs_max ) {
                    rx_abs_max = dsx[i];
                    jp         = i;
                }
            }

            ju  = max(ju, min(gbj+j+ku+jp, n-1));
            int swap_len = ju - (j+gbj) + 1;
            if(tx == 0) {
                sipiv[j] = gbj + jp + j + 1;    // +1 for fortran indexing (adjustment included w.r.t gbj)
            }

            if(ju > last_column_read) {
                // read up to ju into shared memory
                int jstart = min(last_column_read + 1, n-1);
                int jend   = ju;
                read_sAB_new_columns(mband, n, jstart, jend, kl, ku, dAB, lddab, sABtmp, sldab, ntx, tx);
                /*
                DPCT1065:209: Consider replacing sycl::nd_item::barrier() with
                sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

                last_column_read = ju;
                sABtmp         += (jend - jstart + 1) * sldab;
                cached_columns += (jend - jstart + 1);
            }

            linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (gbj+jp+j+1) : linfo;
            /*
            DPCT1065:205: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier(); // wait for the trailing matrix read

            // swap
            if( !(jp == 0) ) {
                magmaDoubleComplex tmp;
                magmaDoubleComplex *sR1 = &sAB(kv   ,j);
                magmaDoubleComplex *sR2 = &sAB(kv+jp,j);
                for(int i = tx; i < swap_len; i+=ntx) {
                    tmp              = sR1[i * sldab_1];
                    sR1[i * sldab_1] = sR2[i * sldab_1];
                    sR2[i * sldab_1] = tmp;
                }
            }
            /*
            DPCT1065:206: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // scal
            magmaDoubleComplex reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sAB(kv,j) );
            for(int i = tx; i < (km-1); i+=ntx) {
                sAB(kv+1+i, j) *= reg;
            }
            /*
            DPCT1065:207: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // ger
            reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ZERO : MAGMA_Z_ONE;
            magmaDoubleComplex *sU = &sAB(kv,j);
            magmaDoubleComplex *sV = &sAB(kv+1,j);
            if( tx < (km-1) ) {
                for(int jj = 1; jj < swap_len; jj++) {
                    sV[jj * sldab_1 + tx] -= sV[tx] * sU[jj * sldab_1 + 0] * reg;
                }
            }
            /*
            DPCT1065:208: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
        // end of factorization loop

        // write ib columns
        write_sAB_columns(mband, n, gbj, gbj+ib-1, kl, ku, sAB, sldab, dAB, lddab, ntx, tx);

        cached_columns -= ib;

        // write pivot
        for(int i = tx; i < ib; i+=ntx) {
            ipiv[gbj+i] = (magma_int_t)sipiv[i];
        }
        /*
        DPCT1065:203: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // shift the remaining columns to the left
        // then zero-out the rest of the columns
        {
            const int tpg    = min(ntx, mband);
            const int groups = max(1, ntx / mband);
            const int active = min(ntx, groups * tpg);
            const int tx_    = tx % mband;
            const int ty_    = tx / mband;

            magmaDoubleComplex tmp = MAGMA_Z_ZERO;
            for(int j = 0; j < cached_columns; j+=groups) {
                for(int i=0; i < mband; i+=tpg) {
                    int src_j = ib+j+ty_;
                    if(tx < active && src_j < nn) {
                        tmp = sAB(i+tx_, ib+j+ty_);
                    }
                    /*
                    DPCT1065:211: Consider replacing sycl::nd_item::barrier()
                    with
                    sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                    for better performance if there is no access to global
                    memory.
                    */
                    item_ct1.barrier();

                    if(tx < active && src_j < nn) {
                        sAB(i+tx_, j+ty_) = tmp;
                    }
                    /*
                    DPCT1065:212: Consider replacing sycl::nd_item::barrier()
                    with
                    sycl::nd_item::barrier(sycl::access::fence_space::local_space)
                    for better performance if there is no access to global
                    memory.
                    */
                    item_ct1.barrier();
                }
            }

            // zero out the rest of the columns
            if(tx < active) {
                for(int j = ty_+cached_columns; j < nn; j+=groups) {
                    for(int i=tx_; i < mband; i+=tpg) {
                        sAB(i,j) = MAGMA_Z_ZERO;
                    }
                }
            }
            /*
            DPCT1065:210: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
        // end of the shift

    }
    // end of the main loop over min_mn in steps of nb

    // write info
    if(tx == 0) {
        info_array[batchid] = linfo;
    }

#undef sAB
#undef dAB
}

////////////////////////////////////////////////////////////////////////////////
template <int NTX>
static magma_int_t magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver(
    magma_int_t m, magma_int_t nb, magma_int_t n, magma_int_t kl,
    magma_int_t ku, magmaDoubleComplex **dAB_array, magma_int_t abi,
    magma_int_t abj, magma_int_t lddab, magma_int_t **ipiv_array,
    magma_int_t *info_array, magma_int_t nthreads, magma_int_t ntcol,
    int *ju_array, magma_int_t batchCount, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;

    magma_int_t kv      = kl + ku;
    magma_int_t mband   = kv + 1 + kl;
    magma_int_t sldab   = SLDAB(mband);

    if( m == 0 || nb == 0 || n == 0) return 0;

    nthreads = max( nthreads, (kl + 1) );
    ntcol    = max(1, ntcol);

    // calculate maximum width based on worst case
    magma_int_t nn = nb + kv + 1;

    magma_int_t shmem  = 0;
    shmem += sldab * nn * sizeof(magmaDoubleComplex); // sAB
    shmem += (kl + 1)  * sizeof(double); // dsx
    shmem += min(m,nb)  * sizeof(magma_int_t); // pivot
    shmem *= ntcol;

    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    sycl::range<3> threads(1, ntcol, nthreads);
    sycl::range<3> grid(1, 1, gridx);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, shmem_max;
    nthreads_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::max_work_group_size>();
    shmem_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::local_mem_size>();

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads (%lld) or too much shared memory (%f KB)\n",
        //        __func__, (long long)total_threads, (double)shmem/1024. );
        arginfo = -100;
        return arginfo;
    }

    try {
      ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(shmem), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
                             zgbtrf_batched_sliding_window_loopout_kernel_sm<NTX>(
                                 m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array,
				 ju_array, info_array, batchCount, item_ct1,
                                 dpct_local_acc_ct1.get_pointer());
                         });
      });
    }
    catch (sycl::exception const &exc) {
        //printf("error in %s : failed to launch kernel\n", __func__);
        arginfo = -100;
    }

    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
template <int NTX>
static magma_int_t magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver(
    magma_int_t m, magma_int_t nb, magma_int_t n, magma_int_t kl,
    magma_int_t ku, magmaDoubleComplex **dAB_array, magma_int_t lddab,
    magma_int_t **ipiv_array, magma_int_t *info_array, magma_int_t nthreads,
    magma_int_t ntcol, magma_int_t batchCount, magma_queue_t queue)
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;

    magma_int_t kv      = kl + ku;
    magma_int_t mband   = kv + 1 + kl;
    magma_int_t sldab   = SLDAB(mband);

    if( m == 0 || nb == 0 || n == 0) return 0;

    nthreads = max( nthreads, (kl + 1) );
    ntcol    = max(1, ntcol);

    // calculate maximum width based on worst case
    magma_int_t nn = nb + kv + 1;

    magma_int_t shmem  = 0;
    shmem += sldab * nn * sizeof(magmaDoubleComplex); // sAB
    shmem += (kl + 1)  * sizeof(double); // dsx
    shmem += min(m,nb)  * sizeof(magma_int_t); // pivot
    shmem *= ntcol;

    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    sycl::range<3> threads(1, ntcol, nthreads);
    sycl::range<3> grid(1, 1, gridx);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, shmem_max;
    nthreads_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::max_work_group_size>();
    shmem_max = queue->sycl_stream()->get_device().get_info<sycl::info::device::local_mem_size>();

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads (%lld) or too much shared memory (%f KB)\n",
        //        __func__, (long long)total_threads, (double)shmem/1024. );
        arginfo = -100;
        return arginfo;
    }

    try {
      ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(shmem), cgh);

        cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                         [=](sycl::nd_item<3> item_ct1) {
			     zgbtrf_batched_sliding_window_loopin_kernel_sm<NTX>(
                                 m, nb, n, kl, ku, dAB_array, lddab, ipiv_array,
				 info_array, batchCount, item_ct1,
                                 dpct_local_acc_ct1.get_pointer());
                         });
      });
    }
    catch (sycl::exception const &exc) {
        //printf("error in %s : failed to launch kernel\n", __func__);
        arginfo = -100;
    }

    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
static magma_int_t
magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_instantiator(
    magma_int_t m,  magma_int_t nb, magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t abi, magma_int_t abj, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t nthreads, magma_int_t ntcol, int* ju_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t nthreads32 = magma_roundup(nthreads, 32);
    switch(nthreads32) {
        case   32: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver<  32>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case   64: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver<  64>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case   96: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver<  96>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  128: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 128>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  160: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 160>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  192: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 192>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  224: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 224>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  256: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 256>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  288: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 288>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  320: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 320>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  352: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 352>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  384: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 384>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  416: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 416>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  448: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 448>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  480: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 480>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  512: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 512>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  544: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 544>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  576: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 576>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  608: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 608>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  640: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 640>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  672: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 672>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  704: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 704>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  736: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 736>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  768: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 768>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  800: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 800>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  832: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 832>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  864: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 864>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  896: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 896>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  928: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 928>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  960: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 960>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  992: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver< 992>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case 1024: arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_driver<1024>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        default: arginfo = -100;
    }
    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    ZGBTRF computes an LU factorization of a COMPLEX m-by-n band matrix A
    using partial pivoting with row interchanges.

    This is the batched version of the algorithm, which performs the factorization
    on a batch of matrices with the same size and lower/upper bandwidths.

    This routine has shared memory requirements that may exceed the capacity of
    the GPU. In such a case, the routine exits immediately, returning a negative
    error code.

    Arguments
    ---------
    @param[in]
    M     INTEGER
          The number of rows of the matrix A.  M >= 0.

    @param[in]
    N     INTEGER
          The number of columns of the matrix A.  N >= 0.

    @param[in]
    KL    INTEGER
          The number of subdiagonals within the band of A.  KL >= 0.

    @param[in]
    KU    INTEGER
          The number of superdiagonals within the band of A.  KU >= 0.

    @param[in,out]
    dAB_array    Array of pointers, dimension (batchCount).
          Each is a COMPLEX_16 array, dimension (LDDAB,N)
          On entry, the matrix AB in band storage, in rows KL+1 to
          2*KL+KU+1; rows 1 to KL of the array need not be set.
          The j-th column of A is stored in the j-th column of the
          array AB as follows:
          AB(kl+ku+1+i-j,j) = A(i,j) for max(1,j-ku)<=i<=min(m,j+kl)

          On exit, details of the factorization: U is stored as an
          upper triangular band matrix with KL+KU superdiagonals in
          rows 1 to KL+KU+1, and the multipliers used during the
          factorization are stored in rows KL+KU+2 to 2*KL+KU+1.
          See below for further details.

    @param[in]
    LDDAB INTEGER
          The leading dimension of the array AB.  LDAB >= 2*KL+KU+1.

    @param[out]
    dIPIV_array    Array of pointers, dimension (batchCount).
          Each is an INTEGER array, dimension (min(M,N))
          The pivot indices; for 1 <= i <= min(M,N), row i of the
          matrix was interchanged with row IPIV(i).

    @param[out]
    dINFO_array    INTEGER array, dimension (batchCount)
          Each is the INFO output for a given matrix
          = 0: successful exit
          < 0: if INFO = -i, the i-th argument had an illegal value
          > 0: if INFO = +i, U(i,i) is exactly zero. The factorization
               has been completed, but the factor U is exactly
               singular, and division by zero will occur if it is used
               to solve a system of equations.

    @param[in,out]
    device_work  Workspace, allocated on device memory by the user

    @param[in,out]
    lwork        INTEGER pointer
                 The size of the workspace (device_work) in bytes
                 - lwork[0] < 0: a workspace query is assumed, the routine
                   calculates the required amount of workspace and returns
                   it in lwork. The workspace is not referenced, and no
                   computation is performed.
                -  lwork[0] >= 0: the routine assumes that the user has provided
                   a workspace with the size in lwork.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

  Further Details
  ===============

  The band storage scheme is illustrated by the following example, when
  M = N = 6, KL = 2, KU = 1:

  On entry:                       On exit:

      *    *    *    +    +    +       *    *    *   u14  u25  u36
      *    *    +    +    +    +       *    *   u13  u24  u35  u46
      *   a12  a23  a34  a45  a56      *   u12  u23  u34  u45  u56
     a11  a22  a33  a44  a55  a66     u11  u22  u33  u44  u55  u66
     a21  a32  a43  a54  a65   *      m21  m32  m43  m54  m65   *
     a31  a42  a53  a64   *    *      m31  m42  m53  m64   *    *

  Array elements marked * are not used by the routine; elements marked
  + need not be set on entry, but are required by the routine to store
  elements of U because of fill-in resulting from the row interchanges.


    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgbtrf_batched_sliding_window_loopout(
    magma_int_t m,  magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    void* device_work, magma_int_t *lwork,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t kv      = kl + ku;

    if( m < 0 )
        arginfo = -1;
    else if ( n < 0 )
        arginfo = -2;
    else if ( kl < 0 )
        arginfo = -3;
    else if ( ku < 0 )
        arginfo = -4;
    else if ( lddab < (kl+kv+1) )
        arginfo = -6;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( m == 0 || n == 0 || batchCount == 0) return 0;

    magma_int_t nb       = 8;
    magma_int_t nthreads = kl+1;
    magma_int_t ntcol    = 1;
    magma_get_zgbtrf_batched_params(m, n, kl, ku, &nb, &nthreads);

    // calculate required workspace
    magma_int_t lwork_bytes = 0;
    lwork_bytes += batchCount * sizeof(int); // no need for magma_int_t here

    if( *lwork < 0) {
        *lwork = lwork_bytes;
        arginfo = 0;
        return arginfo;
    }

    if( *lwork < lwork_bytes ) {
        arginfo = -13;
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // ju_array holds (per problem) the index of the last column affected
    // by the previous factorization stage
    int* ju_array = (int*)device_work;

    for(int j = 0; j < n; j += nb) {
        magma_int_t ib = min(nb, n-j);
        arginfo = magma_zgbtrf_batched_sliding_window_loopout_sm_kernel_instantiator(
                    m, ib, n, kl, ku,
                    dAB_array, 0, j, lddab,
                    ipiv_array, info_array,
                    nthreads, ntcol, ju_array, batchCount, queue );

        if( arginfo != 0) {
            break;
        }
    }
    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    ZGBTRF computes an LU factorization of a COMPLEX m-by-n band matrix A
    using partial pivoting with row interchanges.

    This is the batched version of the algorithm, which performs the factorization
    on a batch of matrices with the same size and lower/upper bandwidths.

    This routine has shared memory requirements that may exceed the capacity of
    the GPU. In such a case, the routine exits immediately, returning a negative
    error code.

    Arguments
    ---------
    @param[in]
    M     INTEGER
          The number of rows of the matrix A.  M >= 0.

    @param[in]
    N     INTEGER
          The number of columns of the matrix A.  N >= 0.

    @param[in]
    KL    INTEGER
          The number of subdiagonals within the band of A.  KL >= 0.

    @param[in]
    KU    INTEGER
          The number of superdiagonals within the band of A.  KU >= 0.

    @param[in,out]
    dAB_array    Array of pointers, dimension (batchCount).
          Each is a COMPLEX_16 array, dimension (LDDAB,N)
          On entry, the matrix AB in band storage, in rows KL+1 to
          2*KL+KU+1; rows 1 to KL of the array need not be set.
          The j-th column of A is stored in the j-th column of the
          array AB as follows:
          AB(kl+ku+1+i-j,j) = A(i,j) for max(1,j-ku)<=i<=min(m,j+kl)

          On exit, details of the factorization: U is stored as an
          upper triangular band matrix with KL+KU superdiagonals in
          rows 1 to KL+KU+1, and the multipliers used during the
          factorization are stored in rows KL+KU+2 to 2*KL+KU+1.
          See below for further details.

    @param[in]
    LDDAB INTEGER
          The leading dimension of the array AB.  LDAB >= 2*KL+KU+1.

    @param[out]
    dIPIV_array    Array of pointers, dimension (batchCount).
          Each is an INTEGER array, dimension (min(M,N))
          The pivot indices; for 1 <= i <= min(M,N), row i of the
          matrix was interchanged with row IPIV(i).

    @param[out]
    dINFO_array    INTEGER array, dimension (batchCount)
          Each is the INFO output for a given matrix
          = 0: successful exit
          < 0: if INFO = -i, the i-th argument had an illegal value
          > 0: if INFO = +i, U(i,i) is exactly zero. The factorization
               has been completed, but the factor U is exactly
               singular, and division by zero will occur if it is used
               to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

  Further Details
  ===============

  The band storage scheme is illustrated by the following example, when
  M = N = 6, KL = 2, KU = 1:

  On entry:                       On exit:

      *    *    *    +    +    +       *    *    *   u14  u25  u36
      *    *    +    +    +    +       *    *   u13  u24  u35  u46
      *   a12  a23  a34  a45  a56      *   u12  u23  u34  u45  u56
     a11  a22  a33  a44  a55  a66     u11  u22  u33  u44  u55  u66
     a21  a32  a43  a54  a65   *      m21  m32  m43  m54  m65   *
     a31  a42  a53  a64   *    *      m31  m42  m53  m64   *    *

  Array elements marked * are not used by the routine; elements marked
  + need not be set on entry, but are required by the routine to store
  elements of U because of fill-in resulting from the row interchanges.


    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgbtrf_batched_sliding_window_loopin(
    magma_int_t m,  magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo  = 0;
    magma_int_t kv = kl + ku;

    if( m < 0 )
        arginfo = -1;
    else if ( n < 0 )
        arginfo = -2;
    else if ( kl < 0 )
        arginfo = -3;
    else if ( ku < 0 )
        arginfo = -4;
    else if ( lddab < (kl+kv+1) )
        arginfo = -6;
    else if (batchCount < 0)
        arginfo = -9;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( m == 0 || n == 0 || batchCount == 0) return 0;

    magma_int_t nb       = 8;
    magma_int_t nthreads = kl+1;
    magma_int_t ntcol    = 1;

    magma_get_zgbtrf_batched_params(m, n, kl, ku, &nb, &nthreads);
    magma_int_t nthreads32 = magma_roundup(nthreads, 32);
    switch(nthreads32) {
        case   32: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver<  32>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case   64: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver<  64>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case   96: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver<  96>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  128: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 128>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  160: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 160>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  192: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 192>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  224: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 224>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  256: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 256>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  288: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 288>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  320: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 320>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  352: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 352>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  384: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 384>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  416: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 416>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  448: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 448>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  480: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 480>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  512: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 512>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  544: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 544>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  576: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 576>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  608: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 608>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  640: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 640>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  672: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 672>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  704: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 704>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  736: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 736>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  768: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 768>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  800: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 800>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  832: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 832>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  864: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 864>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  896: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 896>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  928: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 928>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  960: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 960>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  992: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver< 992>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case 1024: arginfo = magma_zgbtrf_batched_sliding_window_loopin_sm_kernel_driver<1024>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        default: arginfo = -100;
    }

    return arginfo;
}
