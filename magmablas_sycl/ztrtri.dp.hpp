/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar
       
       Definitions used in ztrtri_diag.cu ztrtri_lower.cu ztrtri_upper.cu
*/

#ifndef ZTRTRI_H
#define ZTRTRI_H

#define PRECISION_z 

// define 0 for large initializations
#define Z0 MAGMA_Z_ZERO

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "batched_kernel_param.h"
#if   defined(TRTRI_BATCHED)
#define IB    (ZTRTRI_BATCHED_BLOCK_SIZE)
#define NB    (ZTRTRI_BATCHED_NB)
#elif defined(TRTRI_NONBATCHED)
#define IB    (16)
#define NB    (128)
#else
#error "One of {TRTRI_BATCHED, TRTRI_NONBATCHED} must be defined."
#endif

/*
 * zaxpy16 computes c += alpha*b, where b and c are 16-element vectors.
 */
static void
zaxpy16(
    magmaDoubleComplex alpha,
    const magmaDoubleComplex * __restrict__ b,
    magmaDoubleComplex       * __restrict__ c )
{
    c[0]  += alpha * b[0];
    c[1]  += alpha * b[1];
    c[2]  += alpha * b[2];
    c[3]  += alpha * b[3];
    c[4]  += alpha * b[4];
    c[5]  += alpha * b[5];
    c[6]  += alpha * b[6];
    c[7]  += alpha * b[7];
    c[8]  += alpha * b[8];
    c[9]  += alpha * b[9];
    c[10] += alpha * b[10];
    c[11] += alpha * b[11];
    c[12] += alpha * b[12];
    c[13] += alpha * b[13];
    c[14] += alpha * b[14];
    c[15] += alpha * b[15];
}


// unused -- but nearly identical code throughout ztrtri_lower & upper.cu
static void
zgemm_kernel_16(
    magmaDoubleComplex *A, int lda,
    magmaDoubleComplex *B, int ldb,
    magmaDoubleComplex *C, int ldc,
    magmaDoubleComplex alpha, int jb, int tx, int ty, sycl::nd_item<3> item_ct1,
    sycl::local_accessor<magmaDoubleComplex, 2> sB)
{
    const magmaDoubleComplex *Blast = B + jb;

    // compute NT x 16 block of C
    // each thread computes one 1x16 row, C(id,0:15)
    magmaDoubleComplex rC[16] = {Z0, Z0, Z0, Z0, Z0, Z0, Z0, Z0, Z0, Z0, Z0, Z0, Z0, Z0, Z0, Z0};
    magmaDoubleComplex rA[4];

    do {
        // load 16 x 16 block of B using NX x NY threads
        #pragma unroll
        for (int i = 0; i < 16; i += item_ct1.get_local_range(2)) {
#pragma unroll
            for (int j = 0; j < 16; j += item_ct1.get_local_range(1)) {
                sB[tx + i][ty + j] = B[i + j*ldb];
            }
        }
        /*
        DPCT1065:0: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // load NT x 16 block of A; each thread initially loads 1x4 row,
        // then continues loading more elements as axpys are done.
        rA[0] = A[0*lda];
        rA[1] = A[1*lda];
        rA[2] = A[2*lda];
        rA[3] = A[3*lda];

        // axpy:  C(id,:) += A(id,k) * B(k,:) for k={0,0}, ..., 15
        zaxpy16( rA[0], &sB[ 0][0], rC );  rA[0] = A[ 4*lda];
        zaxpy16( rA[1], &sB[ 1][0], rC );  rA[1] = A[ 5*lda];
        zaxpy16( rA[2], &sB[ 2][0], rC );  rA[2] = A[ 6*lda];
        zaxpy16( rA[3], &sB[ 3][0], rC );  rA[3] = A[ 7*lda];
        
        zaxpy16( rA[0], &sB[ 4][0], rC );  rA[0] = A[ 8*lda];
        zaxpy16( rA[1], &sB[ 5][0], rC );  rA[1] = A[ 9*lda];
        zaxpy16( rA[2], &sB[ 6][0], rC );  rA[2] = A[10*lda];
        zaxpy16( rA[3], &sB[ 7][0], rC );  rA[3] = A[11*lda];

        zaxpy16( rA[0], &sB[ 8][0], rC );  rA[0] = A[12*lda];
        zaxpy16( rA[1], &sB[ 9][0], rC );  rA[1] = A[13*lda];
        zaxpy16( rA[2], &sB[10][0], rC );  rA[2] = A[14*lda];
        zaxpy16( rA[3], &sB[11][0], rC );  rA[3] = A[15*lda];

        zaxpy16( rA[0], &sB[12][0], rC );
        zaxpy16( rA[1], &sB[13][0], rC );
        zaxpy16( rA[2], &sB[14][0], rC );
        zaxpy16( rA[3], &sB[15][0], rC );

        // move to next block of A and B
        A += 16*lda;
        B += 16;
        /*
        DPCT1065:1: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    } while( B < Blast );

    // write NT x 16 result; each thread writes one 16x1 row, C(id,0:15)
    for( int i = 0; i < 16; i++ ) {
        C[0] = alpha*rC[i];
        C += ldc;
    }
}


SYCL_EXTERNAL void
ztrtri_diag_lower_kernel(
    magma_diag_t diag, int n, const magmaDoubleComplex *A, int lda, magmaDoubleComplex *d_invA, sycl::nd_item<3> item_ct1, magmaDoubleComplex *sB);

SYCL_EXTERNAL void
triple_zgemm16_part1_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm16_part2_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm32_part1_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm32_part2_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm64_part1_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm64_part2_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part1_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part2_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part3_lower_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages, sycl::nd_item<3> item_ct1);

    
SYCL_EXTERNAL void
ztrtri_diag_upper_kernel(
    magma_diag_t diag, int n, const magmaDoubleComplex *A, int lda, magmaDoubleComplex *d_invA,
    sycl::nd_item<3> item_ct1, magmaDoubleComplex *sB);

SYCL_EXTERNAL void
triple_zgemm16_part1_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm16_part2_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm32_part1_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm32_part2_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm64_part1_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm64_part2_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part1_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part2_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages,
    sycl::nd_item<3> item_ct1,
    sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part3_upper_kernel(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_invA, int jb, int npages,
    sycl::nd_item<3> item_ct1);







SYCL_EXTERNAL void
ztrtri_diag_lower_kernel_batched(
    magma_diag_t diag, int n, magmaDoubleComplex const * const * dA_array, int lda, magmaDoubleComplex **dinvA_array, sycl::nd_item<3> item_ct1, magmaDoubleComplex *sB);

SYCL_EXTERNAL void
triple_zgemm16_part1_lower_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm16_part2_lower_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm32_part1_lower_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm32_part2_lower_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm64_part1_lower_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm64_part2_lower_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part1_lower_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part2_lower_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part3_lower_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1);

    
SYCL_EXTERNAL void
ztrtri_diag_upper_kernel_batched(
    magma_diag_t diag, int n, magmaDoubleComplex const * const * dA_array, int lda, magmaDoubleComplex **dinvA_array, sycl::nd_item<3> item_ct1, magmaDoubleComplex *sB);


SYCL_EXTERNAL void
triple_zgemm16_part1_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm16_part2_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm32_part1_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm32_part2_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm64_part1_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm64_part2_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part1_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part2_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part3_upper_kernel_batched(
    int n, magmaDoubleComplex const * const * Ain_array, int lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1);



/* vbatched kernels */
////////////////////////////////////////////////////////////////////////////////////////////////////
SYCL_EXTERNAL void
ztrtri_diag_lower_kernel_vbatched(
    magma_diag_t diag, magma_int_t* n, magmaDoubleComplex const * const * dA_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, sycl::nd_item<3> item_ct1, magmaDoubleComplex *xB);

SYCL_EXTERNAL void
triple_zgemm16_part1_lower_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm16_part2_lower_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm32_part1_lower_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm32_part2_lower_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm64_part1_lower_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm64_part2_lower_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part1_lower_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part2_lower_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part3_lower_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1);


SYCL_EXTERNAL void
ztrtri_diag_upper_kernel_vbatched(
    magma_diag_t diag, magma_int_t* n, magmaDoubleComplex const * const * dA_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, sycl::nd_item<3> item_ct1, magmaDoubleComplex *sB);

SYCL_EXTERNAL void
triple_zgemm16_part1_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm16_part2_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm32_part1_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm32_part2_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm64_part1_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm64_part2_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part1_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part2_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1, sycl::local_accessor<magmaDoubleComplex, 2> sB);

SYCL_EXTERNAL void
triple_zgemm_above64_part3_upper_kernel_vbatched(
    magma_int_t* n, magmaDoubleComplex const * const * Ain_array, magma_int_t* lda, magmaDoubleComplex **dinvA_array, int jb, int npages, sycl::nd_item<3> item_ct1);

#endif // ZTRTRI_H
