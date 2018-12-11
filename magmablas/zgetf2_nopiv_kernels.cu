/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Azzam Haidar
       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "magma_templates.h"
#include "sync.cuh"
#include "shuffle.cuh"
#include "batched_kernel_param.h"

// This kernel uses registers for matrix storage, shared mem. for communication.
// It also uses lazy swap.
extern __shared__ magmaDoubleComplex zdata[];

template<int N>
__device__ void
zgetf2_nopiv_device(int m, magmaDoubleComplex* dA, int ldda, magma_int_t *info, const int tx, magmaDoubleComplex* sx, int gbstep)
{
    magmaDoubleComplex rA[N] = {MAGMA_Z_ZERO};
    magmaDoubleComplex reg = MAGMA_Z_ZERO; 
    
    int linfo = 0;
    double abs;
    // check from previous calls if the panel factorization failed previously
    // this is necessary to report the correct info value 
    if(gbstep > 0 && *info != 0) return;

    // read 
    #pragma unroll
    for(int i = 0; i < N; i++){
        rA[i] = dA[ i * ldda + tx ];
    }
        
    #pragma unroll
    for(int i = 0; i < N; i++){
        if(tx == i){
            #pragma unroll
            for(int j = 0; j < N; j++)
                sx[j] = rA[j];
        }
        __syncthreads();

        abs = fabs(MAGMA_Z_REAL( sx[i] )) + fabs(MAGMA_Z_IMAG( sx[i] ));
        linfo = ( abs == MAGMA_D_ZERO && linfo == 0) ? (gbstep+i+1) : linfo;
        //linfo = ( abs  == MAGMA_D_ZERO ) ? min(linfo,gbstep+i+1):0;
        reg   = (linfo == 0 ) ? MAGMA_Z_DIV(MAGMA_Z_ONE, sx[i] ) : MAGMA_Z_ONE;

        // scal and ger
        if( tx > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < N; j++){
                rA[j] -= rA[i] * sx[j];
            }
        }
        __syncthreads();
    }

    if(tx == 0){
        (*info) = (magma_int_t)( linfo );
    }

    // write
    #pragma unroll
    for(int i = 0; i < N; i++){
        dA[ i * ldda + tx ] = rA[i];
    }
}

/******************************************************************************/
extern __shared__ magmaDoubleComplex zdata[];
template<int N, int NPOW2>
__global__ void
zgetf2_nopiv_batched_kernel( int m, magmaDoubleComplex** dA_array, int ai, int aj, int ldda, 
                             magma_int_t* info_array, int gbstep, int batchCount)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int batchid = blockIdx.x * blockDim.y + ty;
    if(batchid >= batchCount)return;

    magmaDoubleComplex* dA = dA_array[batchid] + aj * ldda + ai;
    magma_int_t* info = &info_array[batchid];
    magmaDoubleComplex* sx = (magmaDoubleComplex*)zdata;
    sx += ty * NPOW2;

    zgetf2_nopiv_device<N>(m, dA, ldda, info, tx, sx, gbstep);
}
/***************************************************************************//**
    Purpose
    -------
    zgetf2_nopiv computes the non-pivoting LU factorization of an M-by-N matrix A.
    This routine can deal with matrices of limited widths, so it is for internal use.

    The factorization has the form
       A = L * U
    where L is lower triangular with unit diagonal elements (lower
    trapezoidal if m > n), and U is upper triangular (upper
    trapezoidal if m < n).

    This is a batched version that factors batchCount M-by-N matrices in parallel.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows the matrix A.  N >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ai      INTEGER
            Row offset for dA_array.

    @param[in]
    aj      INTEGER
            Column offset for dA_array.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

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
    gbstep      INTEGER
                Internal use.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t 
magma_zgetf2_nopiv_internal_batched( 
    magma_int_t m, magma_int_t n, 
    magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, 
    magma_int_t* info_array, magma_int_t gbstep, 
    magma_int_t batchCount, magma_queue_t queue )
{
    #define dAarray(i,j) dA_array, i, j

    magma_int_t arginfo = 0;
    if (m < 0) {
        arginfo = -1;
    } else if (n < 0 || n > 32 || (m > 512 && n > 16) ) {
        arginfo = -2;
    } else if (ai < 0) {
        arginfo = -4;
    } else if (aj < 0) {
        arginfo = -5;
    } else if (ldda < max(1,m)) {
        arginfo = -6;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // Quick return if possible
    if (m == 0 || n == 0) {
        return arginfo;
    }

    magma_int_t m1 = (m > MAX_NTHREADS) ? MAX_NTHREADS : m;
    magma_int_t m2 = m - m1;

    const magma_int_t ntcol = (m1 > 32) ? 1 : (2 * (32/m1));
    magma_int_t shmem = ntcol * magma_ceilpow2(n) * sizeof(magmaDoubleComplex);
    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 threads(m1, ntcol, 1);
    dim3 grid(gridx, 1, 1);
    switch(n){
        case  1: zgetf2_nopiv_batched_kernel< 1, magma_ceilpow2( 1)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case  2: zgetf2_nopiv_batched_kernel< 2, magma_ceilpow2( 2)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case  3: zgetf2_nopiv_batched_kernel< 3, magma_ceilpow2( 3)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case  4: zgetf2_nopiv_batched_kernel< 4, magma_ceilpow2( 4)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case  5: zgetf2_nopiv_batched_kernel< 5, magma_ceilpow2( 5)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case  6: zgetf2_nopiv_batched_kernel< 6, magma_ceilpow2( 6)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case  7: zgetf2_nopiv_batched_kernel< 7, magma_ceilpow2( 7)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case  8: zgetf2_nopiv_batched_kernel< 8, magma_ceilpow2( 8)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case  9: zgetf2_nopiv_batched_kernel< 9, magma_ceilpow2( 9)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 10: zgetf2_nopiv_batched_kernel<10, magma_ceilpow2(10)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 11: zgetf2_nopiv_batched_kernel<11, magma_ceilpow2(11)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 12: zgetf2_nopiv_batched_kernel<12, magma_ceilpow2(12)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 13: zgetf2_nopiv_batched_kernel<13, magma_ceilpow2(13)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 14: zgetf2_nopiv_batched_kernel<14, magma_ceilpow2(14)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 15: zgetf2_nopiv_batched_kernel<15, magma_ceilpow2(15)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 16: zgetf2_nopiv_batched_kernel<16, magma_ceilpow2(16)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 17: zgetf2_nopiv_batched_kernel<17, magma_ceilpow2(17)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 18: zgetf2_nopiv_batched_kernel<18, magma_ceilpow2(18)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 19: zgetf2_nopiv_batched_kernel<19, magma_ceilpow2(19)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 20: zgetf2_nopiv_batched_kernel<20, magma_ceilpow2(20)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 21: zgetf2_nopiv_batched_kernel<21, magma_ceilpow2(21)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 22: zgetf2_nopiv_batched_kernel<22, magma_ceilpow2(22)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 23: zgetf2_nopiv_batched_kernel<23, magma_ceilpow2(23)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 24: zgetf2_nopiv_batched_kernel<24, magma_ceilpow2(24)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 25: zgetf2_nopiv_batched_kernel<25, magma_ceilpow2(25)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 26: zgetf2_nopiv_batched_kernel<26, magma_ceilpow2(26)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 27: zgetf2_nopiv_batched_kernel<27, magma_ceilpow2(27)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 28: zgetf2_nopiv_batched_kernel<28, magma_ceilpow2(28)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 29: zgetf2_nopiv_batched_kernel<29, magma_ceilpow2(29)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 30: zgetf2_nopiv_batched_kernel<30, magma_ceilpow2(30)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 31: zgetf2_nopiv_batched_kernel<31, magma_ceilpow2(31)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        case 32: zgetf2_nopiv_batched_kernel<32, magma_ceilpow2(32)><<<grid, threads, shmem, queue->cuda_stream()>>>(m1, dA_array, ai, aj, ldda, info_array, gbstep, batchCount); break;
        default: printf("error: panel width %lld is not supported\n", (long long) n);
    }

    if(m2 > 0){
        magmablas_ztrsm_recursive_batched( 
            MagmaRight, MagmaUpper, MagmaNoTrans, MagmaNonUnit, 
            m2, n, MAGMA_Z_ONE, 
            dAarray(ai   ,aj), ldda, 
            dAarray(ai+m1,aj), ldda, batchCount, queue );
    }

    #undef dAarray
    return arginfo;
}
