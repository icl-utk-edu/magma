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

void
zgbtrf_batched_kernel_small_sm(
    magma_int_t m, magma_int_t n,
    magma_int_t kl, magma_int_t kl,
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
    for(int j = 0; j < n; j++) {
        int col_start = kl + max(ku-j,0);
        int col_end   = kl + ku + min(kl, n-1-j);
        for(int i = tx + col_start; i <= col_end; i+=ntx) {
            sAB(i,j) = dAB(i,j);
        }
    }
    __syncthreads();

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

        // swap
        if( !(jp == 0) ) {
            magmaDouobleComplex tmp;
            magmaDoubleComplex *sR1 = &sAB(kv   ,j);
            magmaDoubleComplex *sR2 = &sAB(kv+jp,j);
            for(int i = tx; i < swap_len; i+=ntx) {
                tmp              = sR1[i * sldab_1];
                sR1[i * sldab_1] = sR2[i * sldab_1];
                sR1[i * sldab_1] = tmp;
            }
        }
        __syncthreads();

        // scal
        magmaDoubleComplex reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_ZONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sAB(kv,j) );
        for(int i = tx; i < (km-1); i+=ntx) {
            sAB(kv+1+i, j) *= reg;
        }
        __syncthreads();

        // ger
        reg = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ZERO : MAGMA_Z_ONE;
        magmaDoubleComplex *sV = &sAB(kv,j);
        if( tx > 0 && tx < (km-1) ) {
            for(int jj = 1; jj < swap_len; jj++) {
                sV[jj * (sldab-1) + tx] -= sV[tx] * sAB[jj * (sldab-1) + 0] * reg;
            }
        }
        __syncthreads();
    }

    // write info
    if(tx == 0) info_array[batchid] = linfo;

    // write pivot
    magma_int_t* ipiv = ipiv_array[batchid];
    for(int i = tx; i < minmn; i+=ntx) {
        ipiv[i] = sipiv[i];
    }

    // write AB
    for(int j = 0; j < n; j++) {
        for(int i = tx; i <= mband; i+=ntx) {
            dAB(i,j) = sAB(i,j);
        }
    }
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
magma_zgbtrf_batched_small(
    magma_int_t m,  magma_int_t n,
    magma_int_t kl, magma_int_t ku,
    magmaDoubleComplex** dAB_array, magma_int_t lddab,
    magma_int_t** ipiv_array, magma_int_t* info_array,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t kv      = kl + ku;
    magma_int_t mband   = kv + 1 + kl;

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

    magma_int_t nthreads = kl + 1;
    magma_int_t sldab    = SLDAB(mband);
    magma_int_t ntcol    = 1;   //NTCOL(nthreads);

    magma_int_t shmem  = 0;
    shmem += sldab * n * sizeof(magmaDoubleComplex); // sAB
    shmem += (kl + 1)  * sizeof(double);        // dsx
    shmem += min(m,n)  * sizeof(magma_int_t);   // pivot
    shmem *= ntcol;

    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 threads(nthreads, ntcol, 1);
    dim3 grid(gridx, 1, 1);

    zgbtrf_batched_kernel_small_sm<<<grid, threads, shmem, queue->cuda_stream()>>>
    (m, n, kl, ku, dAB_array, lddab, ipiv_array, info_array, batchCount);

    return arginfo;
}
