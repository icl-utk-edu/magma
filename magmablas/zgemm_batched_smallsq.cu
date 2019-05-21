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

#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

#define SLDA(N)    ( (N==15||N==23||N==31)? N : (N+1) )

extern __shared__ magmaDoubleComplex zdata[];
template<int N>
__global__ void
zgemm_batched_smallsq_kernel(
        const magma_trans_t transA, magma_trans_t transB, 
        const magmaDoubleComplex alpha, magmaDoubleComplex const * const * dA_array, int ai, int aj, int ldda, 
                                        magmaDoubleComplex const * const * dB_array, int bi, int bj, int lddb, 
        const magmaDoubleComplex beta,  magmaDoubleComplex**               dC_array, int ci, int cj, int lddc, 
        const int batchCount)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int bx = blockIdx.x;
    
    const int batchid = bx * blockDim.z + tz;
    if(batchid >= batchCount) return;
    
    const magmaDoubleComplex* __restrict__ dA = dA_array[batchid] + aj * ldda + ai;
    const magmaDoubleComplex* __restrict__ dB = dB_array[batchid] + bj * lddb + bi;
          magmaDoubleComplex* __restrict__ dC = dC_array[batchid] + cj * lddc + ci;
    
    magmaDoubleComplex rC = MAGMA_Z_ZERO; 
    magmaDoubleComplex rTmp = MAGMA_Z_ZERO; 
    
    const int slda = SLDA(N);
    const int sldb = SLDA(N);
    magmaDoubleComplex* sA = (magmaDoubleComplex*)(zdata);
    magmaDoubleComplex* sB = (magmaDoubleComplex*)(zdata + blockDim.z * slda * N);
    
    sA += tz * slda * N;
    sB += tz * sldb * N;
    
    // read A & B 
    if(transA == MagmaNoTrans){
        sA[ty * slda + tx] = dA[ty * ldda + tx];
    }
    else{
        sA[tx * slda + ty] = (transA == MagmaTrans) ? dA[ty * ldda + tx] : MAGMA_Z_CONJ( dA[ty * ldda + tx] );
    }

    if(transB == MagmaNoTrans){
        sB[ty * sldb + tx] = dB[ty * lddb + tx];
    }
    else{
        sB[tx * sldb + ty] = (transB == MagmaTrans) ? dB[ty * lddb + tx] : MAGMA_Z_CONJ( dB[ty * lddb + tx] );
    }
    __syncthreads(); 

    if(beta != MAGMA_Z_ZERO){
        rC = beta * dC[ty * lddc + tx];
    }

    // multiply
    rTmp = MAGMA_Z_ZERO;
    #pragma unroll
    for(int j = 0; j < N; j++){
        rTmp += sA[j * slda + tx] * sB[ty * sldb + j]; 
    }
    rC += alpha * rTmp;

    // write from rC
    dC[ty * lddc + tx] = rC;
}


extern "C" void 
magmablas_zgemm_batched_smallsq(
    magma_trans_t transA, magma_trans_t transB, 
    magma_int_t m, magma_int_t n, magma_int_t k, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex const * const * dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda, 
    magmaDoubleComplex const * const * dB_array, magma_int_t bi, magma_int_t bj, magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex **dC_array, magma_int_t ci, magma_int_t cj, magma_int_t lddc, 
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t info = 0;
    if      ( transA != MagmaNoTrans && transA != MagmaTrans && transA != MagmaConjTrans )
        info = -1;
    else if ( transB != MagmaNoTrans && transB != MagmaTrans && transB != MagmaConjTrans )
        info = -2;
    else if ( m < 0 )
        info = -3;
    else if ( n < 0 )
        info = -4;
    else if ( k < 0 )
        info = -5;
    else if ( transA == MagmaNoTrans ? ldda < m : ldda < k )
        info = -8;
    else if ( transB == MagmaNoTrans ? lddb < k : lddb < n )
        info = -10;
    else if ( lddc < m )
        info = -13;
    
    if( !(m == n  && n == k) ){
        printf("Only square sizes are supported\n");
        info = -1;
    }

    if( m > 32){
        printf("Only square sizes of up to 32 are supported\n");
        info = -1;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
    }

    if ( m <= 0 || n <= 0 || k <= 0 ) return;
    
    magma_int_t ntcol  = magma_get_zgemm_batched_ntcol( m );
    magma_int_t shmem  = ( SLDA(m)*m + SLDA(n)*n ) * sizeof(magmaDoubleComplex);
                shmem *= ntcol;

    const int nblocks = magma_ceildiv(batchCount, ntcol);
    dim3 grid(nblocks, 1, 1);
    dim3 threads(m, m, ntcol);

    switch(m){
        case  1: zgemm_batched_smallsq_kernel< 1><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  2: zgemm_batched_smallsq_kernel< 2><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  3: zgemm_batched_smallsq_kernel< 3><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  4: zgemm_batched_smallsq_kernel< 4><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  5: zgemm_batched_smallsq_kernel< 5><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  6: zgemm_batched_smallsq_kernel< 6><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  7: zgemm_batched_smallsq_kernel< 7><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  8: zgemm_batched_smallsq_kernel< 8><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case  9: zgemm_batched_smallsq_kernel< 9><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 10: zgemm_batched_smallsq_kernel<10><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 11: zgemm_batched_smallsq_kernel<11><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 12: zgemm_batched_smallsq_kernel<12><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 13: zgemm_batched_smallsq_kernel<13><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 14: zgemm_batched_smallsq_kernel<14><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 15: zgemm_batched_smallsq_kernel<15><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 16: zgemm_batched_smallsq_kernel<16><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 17: zgemm_batched_smallsq_kernel<17><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 18: zgemm_batched_smallsq_kernel<18><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 19: zgemm_batched_smallsq_kernel<19><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 20: zgemm_batched_smallsq_kernel<20><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 21: zgemm_batched_smallsq_kernel<21><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 22: zgemm_batched_smallsq_kernel<22><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 23: zgemm_batched_smallsq_kernel<23><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 24: zgemm_batched_smallsq_kernel<24><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 25: zgemm_batched_smallsq_kernel<25><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 26: zgemm_batched_smallsq_kernel<26><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 27: zgemm_batched_smallsq_kernel<27><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 28: zgemm_batched_smallsq_kernel<28><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 29: zgemm_batched_smallsq_kernel<29><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 30: zgemm_batched_smallsq_kernel<30><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 31: zgemm_batched_smallsq_kernel<31><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        case 32: zgemm_batched_smallsq_kernel<32><<<grid, threads, shmem, queue->cuda_stream()>>>(transA, transB, alpha, dA_array, ai, aj, ldda, dB_array, bi, bj, lddb, beta,  dC_array, ci, cj, lddc, batchCount); break;
        default:;
    }
}
