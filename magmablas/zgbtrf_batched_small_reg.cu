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
__device__ __inline__ void
read_sAB(
    int mband, int n, int kl, int ku,
    magmaDoubleComplex *dAB, int lddab,
    magmaDoubleComplex *sAB, int sldab,
    int ntx, int tx)
{
#if 0
    const int tpg    = min(ntx, mband);
    const int groups = max(1, ntx / mband);
    const int active = max(ntx, groups * mband);
    const int tx_    = tx % mband;
    const int ty_    = tx / mband;

    if(tx < active) {
        for(int j = ty_; j < n; j += groups) {
            int col_start = kl + max(ku-j,0);
            int col_end   = kl + ku + min(kl, n-1-j);
            for(int i = tx_+col_start; i <= col_end; i+=tpg) {
                sAB(i,j) = dAB(i,j);
            }
        }
    }
#else
    for(int j = 0; j < n; j++) {
        int col_start = kl + max(ku-j,0);
        int col_end   = kl + ku + min(kl, n-1-j);
        for(int i = tx + col_start; i <= col_end; i+=ntx) {
            sAB(i,j) = dAB(i,j);
        }
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////
__device__ __inline__ void
write_sAB(
    int mband, int n, int kl, int ku,
    magmaDoubleComplex *sAB, int sldab,
    magmaDoubleComplex *dAB, int lddab,
    int ntx, int tx)
{
#if 1
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
#else
    // write AB
    for(int j = 0; j < n; j++) {
        for(int i = tx; i < mband; i+=ntx) {
            dAB(i,j) = sAB(i,j);
        }
    }
#endif
}

////////////////////////////////////////////////////////////////////////////////
__global__ void
zgbtrf_batched_kernel_small_sm(
    magma_int_t m, magma_int_t n,
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

    const int minmn   = min(m,n);
    const int kv      = kl + ku;
    const int mband   = (kl + 1 + kv);
    const int sldab   = SLDAB(mband);
    const int sldab_1 = sldab-1;

    magmaDoubleComplex* dAB = dAB_array[batchid];
    int linfo = 0;

    // shared memory pointers
    magmaDoubleComplex *sAB = (magmaDoubleComplex*)(zdata);
    double* dsx             = (double*)(sAB + blockDim.y * n * sldab);
    int* sipiv              = (int*)(dsx + blockDim.y * (kl+1));
    sAB   += ty * n * sldab;
    dsx   += ty * (kl+1);
    sipiv += ty * minmn;

    // init sAB
    for(int i = tx; i < n*sldab; i+=ntx) {
        sAB[i] = MAGMA_Z_ZERO;
    }
    __syncthreads();

    // read
    read_sAB(mband, n, kl, ku, dAB, lddab, sAB, sldab, ntx, tx);
    __syncthreads();

    print_memory<magmaDoubleComplex>
    ("read", mband, n, sAB, sldab, 0, 0, 0, 0, 0, 0);

    int ju = 0;
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

        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (j+1) : linfo;

        if(tx == 0) {
            sipiv[j] = jp + j + 1;    // +1 for fortran indexing
        }

        ju = max(ju, min(j+ku+jp, n-1));
        int swap_len = ju - j + 1;

        __syncthreads();
        #ifdef DBG
        if(tx == 0 && ty == 0) {
            printf("pivot = %f at %d, sipiv[%d] = %d\n", rx_abs_max, jp, j, sipiv[j]);
            printf("ju = %d, swap_length = %d\n", ju, swap_len);
        }
        #endif
        __syncthreads();

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

        //print_memory<magmaDoubleComplex>
        //("swap", mband, n, sAB, sldab, 0, 0, 0, 0, 0, 0);


        // scal
        magmaDoubleComplex reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sAB(kv,j) );
        for(int i = tx; i < (km-1); i+=ntx) {
            sAB(kv+1+i, j) *= reg;
        }
        __syncthreads();

        //print_memory<magmaDoubleComplex>
        //("scal", mband, n, sAB, sldab, 0, 0, 0, 0, 0, 0);

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

        //print_memory<magmaDoubleComplex>
        //("ger", mband, n, sAB, sldab, 0, 0, 0, 0, 0, 0);

    }


    // write info
    if(tx == 0) info_array[batchid] = linfo;

    // write pivot
    magma_int_t* ipiv = ipiv_array[batchid];
    for(int i = tx; i < minmn; i+=ntx) {
        ipiv[i] = (magma_int_t)sipiv[i];
    }

    #ifdef DBG
    __syncthreads();
    if(tx == 0 && ty == 0) {
        for(int ss = 0; ss < minmn; ss++) {printf("-%d ", ipiv[ss]);} printf("\n");
    }
    __syncthreads();
    #endif

    write_sAB(mband, n, kl, ku, sAB, sldab, dAB, lddab, ntx, tx);

    #ifdef DBG
    __syncthreads();
    if(tx == 0 && ty == 0) {
        for(int ss = 0; ss < minmn; ss++) {printf("-%d ", ipiv[ss]);} printf("\n");
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
extern "C" magma_int_t
magma_zgbtrf_batched_small_reg(
    magma_int_t m,  magma_int_t n,
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

    if( m == 0 || n == 0 ) return 0;

    nthreads = max( nthreads, (kl + 1) );
    ntcol    = max(1, ntcol);

    magma_int_t shmem  = 0;
    shmem += sldab * n * sizeof(magmaDoubleComplex); // sAB
    shmem += (kl + 1)  * sizeof(double);        // dsx
    shmem += min(m,n)  * sizeof(magma_int_t);   // pivot
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
        cudaFuncSetAttribute(zgbtrf_batched_kernel_small_sm, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
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

    void *kernel_args[] = {&m, &n, &kl, &ku, &dAB_array, &lddab, &ipiv_array, &info_array, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)zgbtrf_batched_kernel_small_sm, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}
