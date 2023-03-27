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

#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

// use this so magmasubs will replace with relevant precision, so we can comment out
// the switch case that causes compilation failure
#define PRECISION_z

#define SLIDING_WINDOW_V1    // multiple calls, redundant memory traffic
//#define SLIDING_WINDOW_V2    // single call, optimal memory traffic


#ifdef MAGMA_HAVE_HIP
#define NTCOL(M)        (max(1,64/(M)))
#else
#define NTCOL(M)        (max(1,64/(M)))
#endif

#define SLDAB(MBAND)    ((MBAND)+1)
#define sAB(i,j)        sAB[(j)*sldab + (i)]
#define dAB(i,j)        dAB[(j)*lddab + (i)]

////////////////////////////////////////////////////////////////////////////////
// read from column jstart to column jend (inclusive) from dAB to sAB
// jstart and jend are global column indices with respect to dAB
__device__ __inline__ void
read_sAB_updated_columns(
    int mband, int n, int jstart, int jend, int kl, int ku,
    magmaDoubleComplex *dAB, int lddab,
    magmaDoubleComplex *sAB, int sldab,
    int ntx, int tx)
{
    const int tpg    = min(ntx, mband);
    const int groups = max(1, ntx / mband);
    const int active = max(ntx, groups * mband);
    const int tx_    = tx % mband;
    const int ty_    = tx / mband;

    if(tx < active) {
        for(int j = jstart + ty_; j <= jend; j += groups) {
            int col_start = 0;       //kl + max(ku-j,0);
            int col_end   = mband-1; //kl + ku + min(kl, n-1-j);
            for(int i = tx_+col_start; i <= col_end; i+=tpg) {
                sAB(i,j-jstart) = dAB(i,j);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
// read from column jstart to column jend (inclusive) from dAB to sAB
// jstart and jend are global column indices with respect to dAB
__device__ __inline__ void
read_sAB_new_columns(
    int mband, int n, int jstart, int jend, int kl, int ku,
    magmaDoubleComplex *dAB, int lddab,
    magmaDoubleComplex *sAB, int sldab,
    int ntx, int tx)
{
    const int tpg    = min(ntx, mband);
    const int groups = max(1, ntx / mband);
    const int active = max(ntx, groups * mband);
    const int tx_    = tx % mband;
    const int ty_    = tx / mband;

    if(tx < active) {
        for(int j = jstart + ty_; j <= jend; j += groups) {
            int col_start = kl + max(ku-j,0);
            int col_end   = kl + ku + min(kl, n-1-j);
            for(int i = tx_+col_start; i <= col_end; i+=tpg) {
                sAB(i,j-jstart) = dAB(i,j);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
__device__ __inline__ void
write_sAB_columns(
    int mband, int n, int jstart, int jend, int kl, int ku,
    magmaDoubleComplex *sAB, int sldab,
    magmaDoubleComplex *dAB, int lddab,
    int ntx, int tx)
{
    const int tpg    = min(ntx, mband);
    const int groups = max(1, ntx / mband);
    const int active = max(ntx, groups * mband);
    const int tx_    = tx % mband;
    const int ty_    = tx / mband;

    if(tx < active) {
        for(int j = jstart + ty_; j <= jend; j += groups) {
            for(int i = tx_; i < mband; i+=tpg) {
                dAB(i,j) = sAB(i,j-jstart);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
template<int NTX>
__global__ __launch_bounds__(NTX)
void
zgbtrf_batched_sliding_window_kernel_sm(
    magma_int_t m, magma_int_t nb, magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, int ABi, int ABj, int lddab,
    magma_int_t** ipiv_array, int* ju_array,
    magma_int_t *info_array, int batchCount)
{
    extern __shared__ magmaDoubleComplex zdata[];
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int ntx = blockDim.x;
    const int batchid = blockIdx.x * blockDim.y + ty;
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
    double* dsx             = (double*)(sAB + blockDim.y * sldab * nn);
    int* sipiv              = (int*)(dsx + blockDim.y * (kl+1));
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
    __syncthreads();

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
    __syncthreads();

    // advance trailing ptrs
    //sAB_trail = sAB + sldab * (jtmp-ABj+1);
    last_column_read = jtmp;

    for(int j = 0; j < minmn; j++) {
        // izamax
        int km = 1 + min( kl, m-j ); // diagonal and subdiagonal(s)
        if(tx < km) {
            dsx[ tx ] = fabs(MAGMA_Z_REAL( sAB(kv+tx,j) )) + fabs(MAGMA_Z_IMAG( sAB(kv+tx,j) ));
        }
        __syncthreads();

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
            __syncthreads();

            last_column_read = ju;
            sAB_trail += sldab * (jend - jstart + 1);
        }

        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (ABj+jp+j+1) : linfo;
        __syncthreads(); // wait for the trailing matrix read

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
        __syncthreads();

        // scal
        magmaDoubleComplex reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sAB(kv,j) );
        for(int i = tx; i < (km-1); i+=ntx) {
            sAB(kv+1+i, j) *= reg;
        }
        __syncthreads();

        // ger
        reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ZERO : MAGMA_Z_ONE;
        magmaDoubleComplex *sU = &sAB(kv,j);
        magmaDoubleComplex *sV = &sAB(kv+1,j);
        if( tx < (km-1) ) {
            for(int jj = 1; jj < swap_len; jj++) {
                sV[jj * sldab_1 + tx] -= sV[tx] * sU[jj * sldab_1 + 0] * reg;
            }
        }
        __syncthreads();
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
}

////////////////////////////////////////////////////////////////////////////////
template<int NTX>
__global__ __launch_bounds__(NTX)
void
zgbtrf_batched_sliding_window_kernel_sm_v2(
    magma_int_t m, magma_int_t nb, magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, int lddab,
    magma_int_t** ipiv_array, magma_int_t *info_array,
    int batchCount)
{
    extern __shared__ magmaDoubleComplex zdata[];
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int ntx = blockDim.x;
    const int batchid = blockIdx.x * blockDim.y + ty;
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
    int linfo = 0;

    // shared memory pointers
    magmaDoubleComplex *sAB = (magmaDoubleComplex*)(zdata);
    double* dsx             = (double*)(sAB + blockDim.y * sldab * nn);
    int* sipiv              = (int*)(dsx + blockDim.y * (kl+1));
    sAB   += ty * nn * sldab;
    dsx   += ty * (kl+1);
    sipiv += ty * minmn;

    magmaDoubleComplex *sABtmp = &sAB;
    int last_column_read = 0;
    int cached_columns   = 0;   // number of columns cached from previous iteration

    int ju = -1;

    // init sAB
    for(int i = tx; i < nn*sldab; i+=ntx) {
        sAB[i] = MAGMA_Z_ZERO;
    }
    __syncthreads();

    for(int gbj = 0; gbj < minmn; gbj+=nb) {
        int ib = min(nb, n-gbj);
        int j1 = ju+1;
        int j2 = ju+ib-cached_columns;
        sABtmp = &sAB(0,cached_columns);

        // read at least ib columns
        if(cached_columns < ib) {
            read_sAB_new_columns(mband, n, j1, j2, kl, ku, dAB, lddab, sABtmp, sldab, ntx, tx);
        }
        __syncthreads();

        last_column_read = (cached_columns < ib) ? j2 : last_column_read;
        cached_columns   = max(cached_columns, ib);
        sABtmp           = &sAB(0,cached_columns);

        // factorization loop
        for(int j = 0; j < ib; j++) {
            // izamax
            int km = 1 + min( kl, m-j ); // diagonal and subdiagonal(s)
            if(tx < km) {
                dsx[ tx ] = fabs(MAGMA_Z_REAL( sAB(kv+tx,j) )) + fabs(MAGMA_Z_IMAG( sAB(kv+tx,j) ));
            }
            __syncthreads();

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
                __syncthreads();

                last_column_read = ju;
                sABtmp         += (jend - jstart + 1) * sldab;
                cached_columns += (jend - jstart + 1);
            }

            linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (gbj+jp+j+1) : linfo;
            __syncthreads(); // wait for the trailing matrix read

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
            __syncthreads();

            // scal
            magmaDoubleComplex reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sAB(kv,j) );
            for(int i = tx; i < (km-1); i+=ntx) {
                sAB(kv+1+i, j) *= reg;
            }
            __syncthreads();

            // ger
            reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ZERO : MAGMA_Z_ONE;
            magmaDoubleComplex *sU = &sAB(kv,j);
            magmaDoubleComplex *sV = &sAB(kv+1,j);
            if( tx < (km-1) ) {
                for(int jj = 1; jj < swap_len; jj++) {
                    sV[jj * sldab_1 + tx] -= sV[tx] * sU[jj * sldab_1 + 0] * reg;
                }
            }
            __syncthreads();
        }
        // end of factorization loop

        // write ib columns
        write_sAB_columns(mband, n, gbj, gbj+ib-1, kl, ku, sAB, sldab, dAB, lddab, ntx, tx);

        cached_columns -= ib;

        // write pivot
        for(int i = tx; i < minmn; i+=ntx) {
            ipiv[gbj+i] = (magma_int_t)sipiv[i];
        }
        __syncthreads();

        // shift the remaining columns to the left
        {
            const int tpg    = min(ntx, mband);
            const int groups = max(1, ntx / mband);
            const int active = max(ntx, groups * mband);
            const int tx_    = tx % mband;
            const int ty_    = tx / mband;

            magmaDoubleComplex tmp = MAGMA_Z_ZERO;
            for(int j = 0; j < cached_columns; j+=groups) {
                for(int i=0; i < mband; i+=tpg) {
                    if(tx < active) {
                        tmp = sAB(i+tx, ib+j+ty_)
                    }
                    __syncthreads();

                    if(tx < active) {
                        sAB(i+tx, j+ty) = tmp;
                    }
                    __syncthreads();
                }
            }
        }
        // end of the shift

    }
    // end of the main loop over min_mn in steps of nb

    // write info
    if(tx == 0) {
        info_array[batchid] = linfo;
    }
}

////////////////////////////////////////////////////////////////////////////////
template<int NTX>
static magma_int_t
magma_zgbtrf_batched_sliding_window_sm_kernel_driver(
    magma_int_t m,  magma_int_t nb, magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t abi, magma_int_t abj, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t nthreads, magma_int_t ntcol, int* ju_array,
    magma_int_t batchCount, magma_queue_t queue )
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
    dim3 threads(nthreads, ntcol, 1);
    dim3 grid(gridx, 1, 1);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, shmem_max;
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zgbtrf_batched_sliding_window_kernel_sm<NTX>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads (%lld) or too much shared memory (%f KB)\n",
        //        __func__, (long long)total_threads, (double)shmem/1024. );
        arginfo = -100;
        return arginfo;
    }

    void *kernel_args[] = {&m, &nb, &n, &kl, &ku, &dAB_array, &abi, &abj, &lddab, &ipiv_array, &ju_array, &info_array, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)zgbtrf_batched_sliding_window_kernel_sm<NTX>, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
template<int NTX>
static magma_int_t
magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2(
    magma_int_t m,  magma_int_t nb, magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t nthreads, magma_int_t ntcol,
    magma_int_t batchCount, magma_queue_t queue )
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
    dim3 threads(nthreads, ntcol, 1);
    dim3 grid(gridx, 1, 1);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, shmem_max;
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zgbtrf_batched_sliding_window_kernel_sm_v2<NTX>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads (%lld) or too much shared memory (%f KB)\n",
        //        __func__, (long long)total_threads, (double)shmem/1024. );
        arginfo = -100;
        return arginfo;
    }

    void *kernel_args[] = {&m, &nb, &n, &kl, &ku, &dAB_array, &lddab, &ipiv_array, &info_array, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)zgbtrf_batched_sliding_window_kernel_sm_v2<NTX>, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
static magma_int_t
magma_zgbtrf_batched_sliding_window_sm_kernel_instantiator(
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
        case   32: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver<  32>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case   64: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver<  64>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case   96: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver<  96>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  128: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 128>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  160: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 160>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  192: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 192>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  224: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 224>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  256: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 256>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  288: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 288>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  320: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 320>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  352: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 352>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  384: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 384>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  416: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 416>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  448: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 448>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  480: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 480>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  512: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 512>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  544: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 544>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  576: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 576>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  608: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 608>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  640: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 640>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  672: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 672>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  704: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 704>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  736: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 736>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  768: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 768>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  800: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 800>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  832: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 832>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  864: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 864>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  896: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 896>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  928: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 928>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  960: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 960>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  992: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver< 992>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case 1024: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver<1024>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        default: arginfo = -100;
    }
    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zgbtrf_batched_sliding_window_work(
    magma_int_t m,  magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t nb, magma_int_t nthreads,
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
    magma_int_t ntcol = 1;

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
        arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_instantiator(
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
magma_zgbtrf_batched_sliding_window(
    magma_int_t m,  magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo  = 0;
    magma_int_t nb       = 32;
    magma_int_t nthreads = kl+1;

    magma_get_zgbtrf_batched_params(m, n, kl, ku, &nb, &nthreads);

    // query workspace
    magma_int_t lwork[1] = {-1};
    magma_zgbtrf_batched_sliding_window_work(
        m, n, kl, ku,
        NULL, lddab, NULL, NULL,
        nb, nthreads, NULL, lwork,
        batchCount, queue );

    void* device_work = NULL;
    magma_malloc((void**)&device_work, lwork[0]);

    arginfo = magma_zgbtrf_batched_sliding_window_work(
                m, n, kl, ku,
                dAB_array, lddab, ipiv_array, info_array,
                nb, nthreads,
                device_work, lwork,
                batchCount, queue );

    magma_free( device_work );
    return arginfo;
}


extern "C" magma_int_t
magma_zgbtrf_batched_sliding_window_v2(
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

    magma_int_t nb       = 32;
    magma_int_t nthreads = kl+1;
    magma_int_t ntcol    = 1;

    magma_get_zgbtrf_batched_params(m, n, kl, ku, &nb, &nthreads);
    magma_int_t nthreads32 = magma_roundup(nthreads, 32);

    switch(nthreads32) {
        case   32: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2<  32>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case   64: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2<  64>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case   96: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2<  96>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  128: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 128>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  160: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 160>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  192: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 192>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  224: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 224>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  256: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 256>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  288: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 288>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  320: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 320>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  352: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 352>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  384: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 384>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  416: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 416>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  448: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 448>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  480: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 480>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  512: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 512>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  544: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 544>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  576: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 576>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  608: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 608>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  640: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 640>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  672: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 672>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  704: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 704>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  736: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 736>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  768: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 768>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  800: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 800>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  832: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 832>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  864: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 864>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  896: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 896>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  928: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 928>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  960: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 960>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case  992: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2< 992>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        case 1024: arginfo = magma_zgbtrf_batched_sliding_window_sm_kernel_driver_v2<1024>(m, nb, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, nthreads, ntcol, batchCount, queue ); break;
        default: arginfo = -100;
    }

    return arginfo;
}
