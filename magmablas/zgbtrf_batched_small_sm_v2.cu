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

#ifdef MAGMA_HAVE_HIP
#define NTCOL(M)        (max(1,64/(M)))
#else
#define NTCOL(M)        (max(1,64/(M)))
#endif

#define SLDAB(MBAND)    ((MBAND)+1)
#define sAB(i,j)        sAB[(j)*sldab + (i)]
#define dAB(i,j)        dAB[(j)*lddab + (i)]

//#define DBG

////////////////////////////////////////////////////////////////////////////////
template<typename T>
__device__ void print_memory(
                const char* msg,
                int m, int n, T* sA, int lda,
                int tx, int ty, int tz,
                int bx, int by, int bz)
{
#if defined(PRECISION_d) && defined(DBG)
    __syncthreads();
    if(threadIdx.x == tx && threadIdx.y == ty && threadIdx.z == tz &&
       blockIdx.x  == bx && blockIdx.y  == by && blockIdx.z  == bz) {
        printf("%s = [ \n", msg);
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                printf("%8.4f  ", (double)(sA[j*lda+i]));
            }
            printf("\n");
        }
        printf("]; \n");
    }
    __syncthreads();
#endif
}

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
    #ifdef DBG
    __syncthreads();
    if(tx == 0 && blockIdx.x == 7)printf("reading columns %d to %d\n", jstart, jend);
    __syncthreads();
    #endif

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
    #ifdef DBG
    __syncthreads();
    if(tx == 0 && blockIdx.x == 7)printf("reading columns %d to %d\n", jstart, jend);
    __syncthreads();
    #endif

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
write_sAB(
    int mband, int n, int kl, int ku,
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
        for(int j = ty_; j < n; j += groups) {
            for(int i = tx_; i < mband; i+=tpg) {
                dAB(i,j) = sAB(i,j);
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
zgbtrf_batched_kernel_small_sm_v2(
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

    magmaDoubleComplex* dAB = dAB_array[batchid]; //+ ABj * lddab + ABi;
    magma_int_t* ipiv = ipiv_array[batchid];
    int linfo = 0;

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
    //======================================================
    //int jtmp = (ABj == 0) ? nb-1 : max(ju, ABj+nb-1);
    //if( ABj == 0 ) {
    //    read_sAB_new_columns(mband, n, ABj, jtmp, kl, ku, dAB, lddab, sAB, sldab, ntx, tx);
    //}
    //else{
    //    read_sAB_updated_columns(mband, n, ABj, jtmp, kl, ku, dAB, lddab, sAB, sldab, ntx, tx);
    //}
    __syncthreads();

    // advance trailing ptrs
    //sAB_trail = sAB + sldab * (jtmp-ABj+1);
    last_column_read = jtmp;

    #ifdef DBG
    __syncthreads();
    print_memory<magmaDoubleComplex>
    ("read", mband, nn, sAB, sldab, 0, 0, 0, 7, 0, 0);
    __syncthreads();
    #endif

    //int ju = 0;
    //for(int jj = 0; jj < ABj; jj++) {
    //    int jp_ = ipiv[jj] - 1; // reverse fortran indexing
    //    ju = max(ju, min(ku+jp_, n-1));
    //}

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

        #ifdef DBG
        __syncthreads();
        if(tx == 0 && ty == 0 && batchid == 7) {
            printf("j = %d, pivot = %f at %d, sipiv[%d] = %d\n", j, rx_abs_max, jp, j, sipiv[j]);
            printf("ju = %d, swap_length = %d \n", ju, swap_len);
            printf("ju = %d, last_column_read = %d\n", ju, last_column_read);
        }
        __syncthreads();
        #endif
        if(ju > last_column_read) {
            // read up to ju into shared memory
            int jstart = min(last_column_read + 1, n-1);
            int jend   = ju;
            //jstart = min(jstart+ABj, n-1);
            //jend   = min(jend + ABj, n-1);
            read_sAB_new_columns(mband, n, jstart, jend, kl, ku, dAB, lddab, sAB_trail, sldab, ntx, tx);
            __syncthreads();

            last_column_read = ju;
            sAB_trail += sldab * (jend - jstart + 1);
        }

        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (ABj+jp+j+1) : linfo;
        __syncthreads(); // wait for the trailing matrix read


        print_memory<magmaDoubleComplex>
        ("after reading", mband, nn, sAB, sldab, 0, 0, 0, 7, 0, 0);


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

        #ifdef DBG
        print_memory<magmaDoubleComplex>
        ("swap", mband, nn, sAB, sldab, 0, 0, 0, 7, 0, 0);
        #endif

        // scal
        magmaDoubleComplex reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sAB(kv,j) );
        for(int i = tx; i < (km-1); i+=ntx) {
            sAB(kv+1+i, j) *= reg;
        }
        __syncthreads();

        #ifdef DBG
        print_memory<magmaDoubleComplex>
        ("scal", mband, nn, sAB, sldab, 0, 0, 0, 7, 0, 0);
        #endif

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

        #ifdef DBG
        print_memory<magmaDoubleComplex>
        ("ger", mband, nn, sAB, sldab, 0, 0, 0, 7, 0, 0);
        #endif

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

    #ifdef DBG
    __syncthreads();
    if(tx == 0 && ty == 0 && batchid == 7) {
        printf("sipiv: ");
        for(int ss = 0; ss < minmn; ss++) {printf("%-d ", ipiv[ss]);} printf("\n");
    }
    __syncthreads();
    #endif

    //write_sAB(mband, last_column_read+1, kl, ku, sAB, sldab, (dAB + ABj * lddab + ABi), lddab, ntx, tx);
    write_sAB_columns(mband, n, ABj, last_column_read, kl, ku, sAB, sldab, dAB, lddab, ntx, tx);

    #ifdef DBG
    __syncthreads();
    if(tx == 0 && ty == 0 && batchid == 7) {
        printf("dipiv: ");
        for(int ss = 0; ss < minmn; ss++) {printf("%-d ", ipiv[ss]);} printf("\n");
    }
    __syncthreads();
    #endif

}

/***************************************************************************//**
    Purpose
    -------
    zgbtrf_batched computes the LU factorization of a square N-by-N matrix A
    using partial pivoting with row interchanges.
    This routine can deal only with square matrices of size up to 32

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a batched version that factors batchCount M-by-N matrices in parallel.
    dAB, ipiv, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    n       INTEGER
            The size of each matrix A.  N >= 0.

    @param[in,out]
    dAB_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDAB,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    lddab    INTEGER
            The leading dimension of each array A.  LDDAB >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
template<int NTX>
static magma_int_t
magma_zgbtrf_batched_small_sm_v2_kernel_driver(
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
    shmem += (kl + 1)  * sizeof(double);        // dsx
    shmem += min(m,nb)  * sizeof(magma_int_t);   // pivot
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
        cudaFuncSetAttribute(zgbtrf_batched_kernel_small_sm_v2<NTX>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        printf("error: kernel %s requires too many threads (%lld) or too much shared memory (%f KB)\n",
                __func__, (long long)total_threads, (double)shmem/1024. );
        arginfo = -100;
        return arginfo;
    }

    void *kernel_args[] = {&m, &nb, &n, &kl, &ku, &dAB_array, &abi, &abj, &lddab, &ipiv_array, &ju_array, &info_array, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)zgbtrf_batched_kernel_small_sm_v2<NTX>, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
static magma_int_t
magma_zgbtrf_batched_small_sm_v2_kernel_instantiator(
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
        case   32: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver<  32>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case   64: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver<  64>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case   96: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver<  96>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  128: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 128>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  160: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 160>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  192: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 192>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  224: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 224>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  256: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 256>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  288: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 288>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  320: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 320>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  352: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 352>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  384: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 384>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  416: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 416>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  448: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 448>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  480: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 480>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  512: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 512>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  544: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 544>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  576: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 576>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  608: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 608>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  640: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 640>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  672: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 672>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  704: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 704>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  736: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 736>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  768: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 768>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  800: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 800>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  832: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 832>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  864: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 864>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  896: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 896>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  928: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 928>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  960: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 960>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case  992: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver< 992>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        case 1024: arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_driver<1024>(m, nb, n, kl, ku, dAB_array, abi, abj, lddab, ipiv_array, info_array, nthreads, ntcol, ju_array, batchCount, queue ); break;
        default: arginfo = -100;
    }
    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zgbtrf_batched_small_sm_v2_work(
    magma_int_t m,  magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t nb, magma_int_t nthreads, magma_int_t ntcol,
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

    #ifdef DBG
    magmaDoubleComplex *hh[8];
    magma_getvector(8, sizeof(magmaDoubleComplex*), dAB_array, 1, hh, 1, queue);
    magmaDoubleComplex *htmp=hh[7];
    #endif

    //zgbtrf_batched_init(n, kl, ku, dAB_array, lddab, batchCount, queue );

    for(int j = 0; j < n; j += nb) {
        magma_int_t ib = min(nb, n-j);

        #ifdef DBG
        printf("calling kernel with j = %d -- ib == %d\n", j, ib);
        printf("--------------------------------------\n");
        #endif

        arginfo = magma_zgbtrf_batched_small_sm_v2_kernel_instantiator(
                    m, ib, n, kl, ku,
                    dAB_array, 0, j, lddab,
                    ipiv_array, info_array,
                    nthreads, ntcol, ju_array, batchCount, queue );

        if( arginfo != 0) {
            break;
        }

        #ifdef DBG
        magma_queue_sync( queue );
        printf("output:\n");
        magma_zprint_gpu(kv+kl+1, n, htmp, lddab, queue);
        magma_queue_sync( queue );
        #endif

    }
    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
extern "C" magma_int_t
magma_zgbtrf_batched_small_sm_v2(
    magma_int_t m,  magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t nthreads, magma_int_t ntcol,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t nb      = 32;

    // query workspace
    magma_int_t lwork[1] = {-1};
    magma_zgbtrf_batched_small_sm_v2_work(
        m, n, kl, ku,
        NULL, lddab, NULL, NULL,
        nb, nthreads, ntcol, NULL, lwork,
        batchCount, queue );

    void* device_work = NULL;
    magma_malloc((void**)&device_work, lwork[0]);

    arginfo = magma_zgbtrf_batched_small_sm_v2_work(
                m, n, kl, ku,
                dAB_array, lddab, ipiv_array, info_array,
                nb, nthreads, ntcol,
                device_work, lwork,
                batchCount, queue );

    magma_free( device_work );
    return arginfo;
}
