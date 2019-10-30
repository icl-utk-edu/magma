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
       
       This file implements lower case, and is called by ztrtri_kernel.cu.
       It's convenient to have separate files for lower & upper, to diff the sources.
*/

#include "magma_internal.h"

/*
    This inverts the diagonal IB by IB inner blocks of A,
    and stores the results in d_dinvA.
    Each thread block with IB threads does one inner block.
    Each thread deals with one row of the inner block.
*/
static __device__ void
ztrtri_diag_lower_device(
    magma_diag_t diag, int n, const magmaDoubleComplex *A, int lda, magmaDoubleComplex *d_dinvA)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int blk_ind = bx*IB;
    int ind = blk_ind + tx;

    A += blk_ind + blk_ind*lda;  // A(blk_ind, blk_ind)

    // TODO sB should be [IB][IB+1] to avoid bank conflicts, right?
    __shared__ magmaDoubleComplex sB[IB*IB];
    magmaDoubleComplex y_tx;

    // load lower triangle of inner block of A; zero upper triangle & outside matrix
    #pragma unroll
    for( int j=0; j < IB; j++ ) {
        if (tx >= j && ind < n) {
            sB[tx + j*IB] = A[tx + j*lda];
        }
        else {
            sB[tx + j*IB] = MAGMA_Z_ZERO;
        }
    }
    __syncthreads();
    
    // invert the diagonal
    if (diag == MagmaUnit) {
        sB[tx + tx*IB] = MAGMA_Z_ONE;
    }
    else {
        if ( sB[tx + tx*IB] == MAGMA_Z_ZERO ) {  // singular or outside matrix
            sB[tx + tx*IB] = MAGMA_Z_ONE;
        }
        else {
            sB[tx + tx*IB] = MAGMA_Z_ONE / sB[tx + tx*IB];
        }
    }
    
    // compute elements j+1:IB-1 of j-th column.
    for( int j=IB-2; j >= 0; j-- ) {
        if ( tx > j ) {
            // trmv:  y = sB(j+1:IB-1, j+1:IB-1) * sB(j+1:IB-1, j)
            // each thread sums one element, y[tx]
            y_tx = MAGMA_Z_ZERO;
            #pragma unroll
            for( int k=j+1; k < IB; k++ )
                y_tx += sB[tx + k*IB] * sB[k + j*IB];
    
            // scal:  sB(j+1:IB-1, j) = -sB(j,j) * y
            sB[tx + j*IB] = -sB[j + j*IB] * y_tx;
        }
        __syncthreads();
    }
    
    // go to the (bx / ib_per_NB) outer NB*NB block,
    // then  the (bx % ib_per_NB) inner IB*IB block inside that.
    int ib_per_NB = NB/IB;
    d_dinvA += (bx / ib_per_NB)*NB*NB
             + (bx % ib_per_NB)*(NB*IB + IB);
    
    // write result
    #pragma unroll
    for( int j=0; j < IB; j++ ) {
        d_dinvA[tx + j*NB] = sB[tx + j*IB];
    }
}


/*
    Let A be an NB*NB lower triangular matrix, and B its inverse.
    Then the block decomposition
    
        [ A11   0  ] * [ B11   0  ] = [ I 0 ]
        [ A21  A22 ]   [ B21  B22 ]   [ 0 I ]
    
    yields
    
        A11*B11 = I            ==>  B11 =  A11^{-1},
        A22*B22 = I            ==>  B22 =  A22^{-1},
        A21*B11 + A22*B21 = 0  ==>  B21 = -A22^{-1}*A21*B11 = -B22*A21*B11.
    
    ztrtri_diag_kernel inverts A11 and A22.
    triple_zgemm16 routines multiply:
    part 1:  B21 =  A21 * B11,
    part 2:  B21 = -B22 * B21.
    
    At this level, inner block is jb=16, with one 4x4 thread block per inner block.
    Each submatrix Aij and Bij is jb x jb.
    The submatrix dimension is multiplied by 2 at each level,
    so the next level is jb*2 = 32.
    A "page" is the next bigger block, here jb*2=32,
                   [ B11   0  ]
    which contains [ B21  B22 ].
    Outer blocks are NB x NB.
    
    A21 may have < jb rows, but is guaranteed to have jb cols since A22 is on
    the right. This makes a single check easy to do.
    
    B is stored in workspace that is a full multiple of NB x NB; no checks needed.
    
    We split this into part1 & part2 to synchronize all blocks and make sure
    that writes to B12 are observed by all blocks.
*/

/*
 * B21 = A21 * B11
 */
static __device__ void
triple_zgemm16_part1_lower_device(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    // emulate 3D grid: NX * (NY*npages)
    const int by   = blockIdx.y / npages;
    const int page = blockIdx.y % npages;
    const int tx   = threadIdx.x;
    const int ty   = threadIdx.y;
    const int ibx  = blockIdx.x * (blockDim.x*blockDim.y);
    const int iby  = by*16;
    const int id   = tx + ty*blockDim.x;
    const int ind  = page*jb*2 + jb + ibx + id;
    __shared__ magmaDoubleComplex sB[16][17];

    // go to the (page / pages_per_NB) outer NB*NB block,
    // then  the (page % pages_per_NB) inner (jb*2)*(jb*2) page inside that.
    int pages_per_NB = NB/(jb*2);
    d_dinvA += (page / pages_per_NB)*NB*NB
             + (page % pages_per_NB)*(jb*2*NB + jb*2);

    //--------------------------part one---------------------------//
    {
        // B21 = A21 * B11
        const magmaDoubleComplex *A, *B;
        magmaDoubleComplex *C;
        int ldb = NB;
        int ldc = NB;

        // in gemm notation: C = A*B
        A = Ain + page*jb*2*lda + page*jb*2 + jb;  // A21
        B = d_dinvA;                               // B11
        C = d_dinvA + jb;                          // B21

        A += ibx + id;
        B += tx + (iby + ty)*ldb;
        C += ibx + id + iby*ldc;

        const magmaDoubleComplex *Blast = B + jb;

        // compute NT x 16 block of C
        // each thread computes one 1x16 row, C(id,0:15)
        magmaDoubleComplex rC[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        magmaDoubleComplex rA[4]  = {0, 0, 0, 0};

        do {
            // TODO this won't coalesce, will it? unless NX=32 (or maybe 16 with doubles, or 8 with double-complex)
            // load 16 x 16 block of B using NX x 4 threads
            #pragma unroll
            for( int i=0; i < 16; i += 4 ) {  // += blockDim.x
                #pragma unroll
                for( int j=0; j < 16; j += 4 ) {  // += blockDim.y
                    sB[tx + i][ty + j] = B[i + j*ldb];
                }
            }
            __syncthreads();

            if ( ind < n ) {
                // load NT x 16 block of A; each thread initially loads 1x4 row,
                // then continues loading more elements as axpys are done.
                rA[0] = A[0*lda];
                rA[1] = A[1*lda];
                rA[2] = A[2*lda];
                rA[3] = A[3*lda];
                
                // axpy:  C(id,:) += A(id,k) * B(k,:) for k=0, ..., 15
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
            }

            // move to next block of A and B
            A += 16*lda;
            B += 16;
            __syncthreads();
        } while( B < Blast );

        // TODO instead of writing result, copy it to sB and do part 2.
        // Would only work for jb=16, because only then does rC fit into sB.
        // If sB were [NT][16+], then rC would fit into sB.
        
        // write NT x 16 result; each thread writes one 16x1 row, C(id,0:15)
        for( int i = 0; i < 16; i++ ) {
            C[0] = rC[i];
            C += ldc;
        }
    }
}


/*
 * B21 = -B22 * B21
 */
static __device__ void
triple_zgemm16_part2_lower_device(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    const int by   = blockIdx.y / npages;
    const int page = blockIdx.y % npages;
    const int tx   = threadIdx.x;
    const int ty   = threadIdx.y;
    const int ibx  = blockIdx.x * (blockDim.x*blockDim.y);
    const int iby  = by*16;
    const int id   = tx + ty*blockDim.x;
    const int ind  = page*jb*2 + jb + ibx + id;
    __shared__ magmaDoubleComplex sB[16][17];
    
    // go to the (page / pages_per_NB) outer NB*NB block,
    // then  the (page % pages_per_NB) inner (jb*2)*(jb*2) page inside that.
    int pages_per_NB = NB/(jb*2);
    d_dinvA += (page / pages_per_NB)*NB*NB
             + (page % pages_per_NB)*(jb*2*NB + jb*2);

    //--------------------------part two---------------------------//
    {
        // B21 = -B22 * B21
        const magmaDoubleComplex *A, *B;
        magmaDoubleComplex *C;
        int lda = NB;  // shadows lda argument
        int ldb = NB;
        int ldc = NB;

        // in gemm notation: C = A*B
        A = d_dinvA + jb*NB + jb;  // B22
        C = d_dinvA + jb;          // B21
        B = C;                     // B21, okay to overwrite

        A += ibx + id;
        B += tx + (iby + ty)*ldb;
        C += ibx + id + iby*ldc;
        
        // TODO factor this out:
        // gemm16<NX, NY> computes NT x 16 block of C:
        // C(1:nt, 1:16) = A(1:nt, 1:jb) * B(1:jb, 1:16)
        // where NT = NX * NY.
        // part 1: gemm16<4,4>( /*NT, 16,*/ jb,  1, A21, lda, B11, NB, /*0*/, B21, NB, n, ind, tx, ty );
        // part 2: gemm16<4,4>( /*NT, 16,*/ jb, -1, B22, NB,  B21, NB, /*0*/, B21, NB, n, ind, tx, ty );  // okay for C to overwrite B

        const magmaDoubleComplex *Blast = B + jb;

        // compute NT x 16 block of C
        // each thread computes one 1x16 row, C(id,0:15)
        magmaDoubleComplex rC[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        magmaDoubleComplex rA[4]  = {0, 0, 0, 0};

        do {
            // load 16 x 16 block of B using NX x 4 threads
            #pragma unroll
            for( int i=0; i < 16; i += 4 ) {  // += blockDim.x
                #pragma unroll
                for( int j=0; j < 16; j += 4 ) {  // += blockDim.y
                    sB[tx + i][ty + j] = B[i + j*ldb];
                }
            }
            __syncthreads();
            
            if ( ind < n ) {
                // load NT x 16 block of A; each thread initially loads 1x4 row,
                // then continues loading more elements as axpys are done.
                rA[0] = A[0*lda];
                rA[1] = A[1*lda];
                rA[2] = A[2*lda];
                rA[3] = A[3*lda];

                // axpy:  C(id,:) += A(id,k) * B(k,:) for k=0, ..., 15
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
            }

            // move to next block of A and B
            A += 16*lda;
            B += 16;
            __syncthreads();
        } while( B < Blast );

        // write NT x 16 result; each thread writes one 16x1 row, C(id,0:15)
        for( int i = 0; i < 16; i++ ) {
            C[0] = -rC[i];
            C += ldc;
        }
    }
}


/*
 * B21 = A21 * B11
 */
static __device__ void
triple_zgemm32_part1_lower_device(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    const int by   = blockIdx.y / npages;
    const int page = blockIdx.y % npages;
    const int tx   = threadIdx.x;
    const int ty   = threadIdx.y;
    const int ibx  = blockIdx.x * (blockDim.x*blockDim.y);
    const int iby  = by*16;
    const int id   = tx + ty*blockDim.x;
    const int ind  = page*jb*2 + jb + ibx + id;
    __shared__ magmaDoubleComplex sB[16][17];
    
    // go to the (page / pages_per_NB) outer NB*NB block,
    // then  the (page % pages_per_NB) inner (jb*2)*(jb*2) page inside that.
    int pages_per_NB = NB/(jb*2);
    d_dinvA += (page / pages_per_NB)*NB*NB
             + (page % pages_per_NB)*(jb*2*NB + jb*2);

    //--------------------------part one---------------------------//
    {
        // B21 = A21 * B11
        const magmaDoubleComplex *A, *B;
        magmaDoubleComplex *C;
        int ldb = NB;
        int ldc = NB;

        // in gemm notation: C = A*B
        A = Ain + page*jb*2*lda + page*jb*2 + jb;  // A21
        B = d_dinvA;                               // B11
        C = d_dinvA + jb;                          // B21

        A += ibx + id;
        B += tx + (iby + ty)*ldb;
        C += ibx + id + iby*ldc;

        const magmaDoubleComplex *Blast = B + jb;

        // compute NT x 16 block of C
        // each thread computes one 1x16 row, C(id,0:15)
        magmaDoubleComplex rC[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        magmaDoubleComplex rA[4]  = {0, 0, 0, 0};
        
        do {
            // load 16 x 16 block of B using NX x 4 threads
            #pragma unroll
            for( int i=0; i < 16; i += 8 ) {  // += blockDim.x
                #pragma unroll
                for( int j=0; j < 16; j += 4 ) {  // += blockDim.y
                    sB[tx + i][ty + j] = B[i + j*ldb];
                }
            }
            __syncthreads();

            if ( ind < n ) {
                // load NT x 16 block of A; each thread initially loads 1x4 row,
                // then continues loading more elements as axpys are done.
                rA[0] = A[0*lda];
                rA[1] = A[1*lda];
                rA[2] = A[2*lda];
                rA[3] = A[3*lda];
                
                // axpy:  C(id,:) += A(id,k) * B(k,:) for k=0, ..., 15
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
            }
            
            // move to next block of A and B
            A += 16*lda;
            B += 16;
            __syncthreads();
        } while( B < Blast );

        // write NT x 16 result; each thread writes one 16x1 row, C(id,0:15)
        for( int i = 0; i < 16; i++ ) {
            C[0] = rC[i];
            C += ldc;
        }
    }
}


/*
 * B21 = -B22 * B21
 */
static __device__ void
triple_zgemm32_part2_lower_device(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    const int by   = blockIdx.y / npages;
    const int page = blockIdx.y % npages;
    const int tx   = threadIdx.x;
    const int ty   = threadIdx.y;
    const int ibx  = blockIdx.x * (blockDim.x*blockDim.y);
    const int iby  = by*16;
    const int id   = tx + ty*blockDim.x;
    const int ind  = page*jb*2 + jb + ibx + id;
    __shared__ magmaDoubleComplex sB[16][17];
    
    // go to the (page / pages_per_NB) outer NB*NB block,
    // then  the (page % pages_per_NB) inner (jb*2)*(jb*2) page inside that.
    int pages_per_NB = NB/(jb*2);
    d_dinvA += (page / pages_per_NB)*NB*NB
             + (page % pages_per_NB)*(jb*2*NB + jb*2);

    //--------------------------part two---------------------------//
    {
        // B21 = -B22 * B21
        const magmaDoubleComplex *A, *B;
        magmaDoubleComplex *C;
        int lda = NB;
        int ldb = NB;
        int ldc = NB;

        // in gemm notation: C = A*B
        A = d_dinvA + jb*NB + jb;  // B22
        C = d_dinvA + jb;          // B21
        B = C;                     // B21, okay to overwrite

        A += ibx + id;
        B += tx + (iby + ty)*ldb;
        C += ibx + id + iby*ldc;

        const magmaDoubleComplex *Blast = B + jb;

        // compute NT x 16 block of C
        // each thread computes one 1x16 row, C(id,0:15)
        magmaDoubleComplex rC[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        magmaDoubleComplex rA[4]  = {0, 0, 0, 0};

        do {
            // load 16 x 16 block of B using NX x 4 threads
            #pragma unroll
            for( int i=0; i < 16; i += 8 ) {  // += blockDim.x
                #pragma unroll
                for( int j=0; j < 16; j += 4 ) {  // += blockDim.y
                    sB[tx + i][ty + j] = B[i + j*ldb];
                }
            }
            __syncthreads();

            if ( ind < n ) {
                // load NT x 16 block of A; each thread initially loads 1x4 row,
                // then continues loading more elements as axpys are done.
                rA[0] = A[0*lda];
                rA[1] = A[1*lda];
                rA[2] = A[2*lda];
                rA[3] = A[3*lda];
                
                // axpy:  C(id,:) += A(id,k) * B(k,:) for k=0, ..., 15
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
            }

            // move to next block of A and B
            A += 16*lda;
            B += 16;
            __syncthreads();
        } while( B < Blast );

        // write NT x 16 result; each thread writes one 16x1 row, C(id,0:15)
        for( int i = 0; i < 16; i++ ) {
            C[0] = -rC[i];
            C += ldc;
        }
    }
}


/*
 * B21 = A21 * B11
 */
static __device__ void
triple_zgemm64_part1_lower_device(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    const int by   = blockIdx.y / npages;
    const int page = blockIdx.y % npages;
    const int tx   = threadIdx.x;
    const int ty   = threadIdx.y;
    const int ibx  = blockIdx.x*64;
    const int iby  = by*16;
    const int id   = tx + ty*16;
    const int ind  = page*jb*2 + jb + ibx + id;
    __shared__ magmaDoubleComplex sB[16][17];
    
    // go to the (page / pages_per_NB) outer NB*NB block,
    // then  the (page % pages_per_NB) inner (jb*2)*(jb*2) page inside that.
    int pages_per_NB = NB/(jb*2);
    d_dinvA += (page / pages_per_NB)*NB*NB
             + (page % pages_per_NB)*(jb*2*NB + jb*2);

    //--------------------------part one---------------------------//
    {
        // B21 = A21 * B11
        const magmaDoubleComplex *A, *B;
        magmaDoubleComplex *C;
        int ldb = NB;
        int ldc = NB;

        // in gemm notation: C = A*B
        A = Ain + page*jb*2*lda + page*jb*2 + jb;  // A21
        B = d_dinvA;                               // B11
        C = d_dinvA + jb;                          // B21

        A += ibx + id;
        B += tx + (iby + ty)*ldb;
        C += ibx + id + iby*ldc;

        const magmaDoubleComplex *Blast = B + jb;

        // compute NT x 16 block of C
        // each thread computes one 1x16 row, C(id,0:15)
        magmaDoubleComplex rC[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        magmaDoubleComplex rA[4]  = {0, 0, 0, 0};

        do {
            // load 16 x 16 block of B using NX x 4 threads
            #pragma unroll
            for( int i=0; i < 16; i += 16 ) {  // += blockDim.x
                #pragma unroll
                for( int j=0; j < 16; j += 4 ) {  // += blockDim.y
                    sB[tx + i][ty + j] = B[i + j*ldb];
                }
            }
            __syncthreads();

            if ( ind < n ) {
                // load NT x 16 block of A; each thread initially loads 1x4 row,
                // then continues loading more elements as axpys are done.
                rA[0] = A[0*lda];
                rA[1] = A[1*lda];
                rA[2] = A[2*lda];
                rA[3] = A[3*lda];
                
                // axpy:  C(id,:) += A(id,k) * B(k,:) for k=0, ..., 15
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
            }

            // move to next block of A and B
            A += 16*lda;
            B += 16;
            __syncthreads();
        } while( B < Blast );

        // write NT x 16 result; each thread writes one 16x1 row, C(id,0:15)
        for( int i = 0; i < 16; i++ ) {
            C[0] = rC[i];
            C += ldc;
        }
    }
}


/*
 * B21 = -B22 * B21
 */
static __device__ void
triple_zgemm64_part2_lower_device(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    const int by   = blockIdx.y / npages;
    const int page = blockIdx.y % npages;
    const int tx   = threadIdx.x;
    const int ty   = threadIdx.y;
    const int ibx  = blockIdx.x*64;
    const int iby  = by*16;
    const int id   = tx + ty*16;
    const int ind  = page*jb*2 + jb + ibx + id;
    __shared__ magmaDoubleComplex sB[16][17];

    // go to the (page / pages_per_NB) outer NB*NB block,
    // then  the (page % pages_per_NB) inner (jb*2)*(jb*2) page inside that.
    int pages_per_NB = NB/(jb*2);
    d_dinvA += (page / pages_per_NB)*NB*NB
             + (page % pages_per_NB)*(jb*2*NB + jb*2);

    //--------------------------part two---------------------------//
    {
        // B21 = -B22 * B21
        const magmaDoubleComplex *A, *B;
        magmaDoubleComplex *C;
        int lda = NB;
        int ldb = NB;
        int ldc = NB;

        // in gemm notation: C = A*B
        A = d_dinvA + jb*NB + jb;  // B22
        C = d_dinvA + jb;          // B21
        B = C;                     // B21, okay to overwrite

        A += ibx + id;
        B += tx + (iby + ty)*ldb;
        C += ibx + id + iby*ldc;

        const magmaDoubleComplex *Blast = B + jb;

        // compute NT x 16 block of C
        // each thread computes one 1x16 row, C(id,0:15)
        magmaDoubleComplex rC[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        magmaDoubleComplex rA[4]  = {0, 0, 0, 0};

        do {
            // load 16 x 16 block of B using NX x 4 threads
            #pragma unroll
            for( int i=0; i < 16; i += 16 ) {  // += blockDim.x
                #pragma unroll
                for( int j=0; j < 16; j += 4 ) {  // += blockDim.y
                    sB[tx + i][ty + j] = B[i + j*ldb];
                }
            }
            __syncthreads();

            if ( ind < n ) {
                // load NT x 16 block of A; each thread initially loads 1x4 row,
                // then continues loading more elements as axpys are done.
                rA[0] = A[0*lda];
                rA[1] = A[1*lda];
                rA[2] = A[2*lda];
                rA[3] = A[3*lda];
                
                // axpy:  C(id,:) += A(id,k) * B(k,:) for k=0, ..., 15
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
            }

            // move to next block of A and B
            A += 16*lda;
            B += 16;
            __syncthreads();
        } while( B < Blast );

        // write NT x 16 result; each thread writes one 16x1 row, C(id,0:15)
        for( int i = 0; i < 16; i++ ) {
            C[0] = -rC[i];
            C += ldc;
        }
    }
}


/*
 * B21 = A21 * B11
 */
static __device__ void
triple_zgemm_above64_part1_lower_device(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    const int by   = blockIdx.y / npages;
    const int page = blockIdx.y % npages;
    const int tx   = threadIdx.x;
    const int ty   = threadIdx.y;
    const int ibx  = blockIdx.x*64;
    const int iby  = by*16;
    const int id   = tx + ty*16;
    const int ind  = page*jb*2 + jb + ibx + id;
    __shared__ magmaDoubleComplex sB[16][17];
    
    // go to the (page / pages_per_NB) outer NB*NB block,
    // then  the (page % pages_per_NB) inner (jb*2)*(jb*2) page inside that.
    int pages_per_NB = NB/(jb*2);
    d_dinvA += (page / pages_per_NB)*NB*NB
             + (page % pages_per_NB)*(jb*2*NB + jb*2);

    //--------------------------part one---------------------------//
    {
        // B21 = A21 * B11
        const magmaDoubleComplex *A, *B;
        magmaDoubleComplex *C;
        int ldb = NB;
        int ldc = NB;

        // For jb > 64, we process B21 as gridDim.x sections of 64 rows each, with gridDim.x > 1.
        // Each section needs all of the B matrix, so C cannot overwrite B.
        // Therefore, store B21 temporarily in the previously unused B12 matrix
        // (i.e., above diagonal), then in part 3, zero out B12.
        //
        // Kernels with jb <= 64 don't have this problem, because only the
        // NT x 16 section of C that overwrites the same section of B depends
        // on that section of B.
        //
        // in gemm notation: C = A*B
        A = Ain + page*jb*2*lda + page*jb*2 + jb;  // A21
        B = d_dinvA;                               // B11
        C = d_dinvA + jb*NB;                       // B21; write to B12 temp location

        A += ibx + id;
        B += tx + (iby + ty)*ldb;
        C += ibx + id + iby*ldc;

        const magmaDoubleComplex *Blast = B + jb;

        // compute NT x 16 block of C
        // each thread computes one 1x16 row, C(id,0:15)
        magmaDoubleComplex rC[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        magmaDoubleComplex rA[4]  = {0, 0, 0, 0};

        do {
            // load 16 x 16 block of B using NX x 4 threads
            #pragma unroll
            for( int i=0; i < 16; i += 16 ) {  // += blockDim.x
                #pragma unroll
                for( int j=0; j < 16; j += 4 ) {  // += blockDim.y
                    sB[tx + i][ty + j] = B[i + j*ldb];
                }
            }
            __syncthreads();

            if ( ind < n ) {
                // load NT x 16 block of A; each thread initially loads 1x4 row,
                // then continues loading more elements as axpys are done.
                rA[0] = A[0*lda];
                rA[1] = A[1*lda];
                rA[2] = A[2*lda];
                rA[3] = A[3*lda];
                
                // axpy:  C(id,:) += A(id,k) * B(k,:) for k=0, ..., 15
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
            }

            // move to next block of A and B
            A += 16*lda;
            B += 16;
            __syncthreads();
        } while( B < Blast );

        // write NT x 16 result; each thread writes one 16x1 row, C(id,0:15)
        for( int i = 0; i < 16; i++ ) {
            C[0] = rC[i];
            C += ldc;
        }
    }
}


/*
 * B21 = -B22 * B21
 */
static __device__ void
triple_zgemm_above64_part2_lower_device(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    const int by   = blockIdx.y / npages;
    const int page = blockIdx.y % npages;
    const int tx   = threadIdx.x;
    const int ty   = threadIdx.y;
    const int ibx  = blockIdx.x*64;
    const int iby  = by*16;
    const int id   = tx + ty*16;
    const int ind  = page*jb*2 + jb + ibx + id;
    __shared__ magmaDoubleComplex sB[16][17];

    // go to the (page / pages_per_NB) outer NB*NB block,
    // then  the (page % pages_per_NB) inner (jb*2)*(jb*2) page inside that.
    int pages_per_NB = NB/(jb*2);
    d_dinvA += (page / pages_per_NB)*NB*NB
             + (page % pages_per_NB)*(jb*2*NB + jb*2);

    //--------------------------part two---------------------------//
    {
        // B21 = -B22 * B21
        const magmaDoubleComplex *A, *B;
        magmaDoubleComplex *C;
        int lda = NB;
        int ldb = NB;
        int ldc = NB;

        // in gemm notation: C = A*B
        A = d_dinvA + jb*NB + jb;  // B22
        B = d_dinvA + jb*NB;       // B21, read from B12 temp location
        C = d_dinvA + jb;          // B21

        A += ibx + id;
        B += tx + (iby + ty)*ldb;
        C += ibx + id + iby*ldc;

        const magmaDoubleComplex *Blast = B + jb;

        // compute NT x 16 block of C
        // each thread computes one 1x16 row, C(id,0:15)
        magmaDoubleComplex rC[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        magmaDoubleComplex rA[4]  = {0, 0, 0, 0};

        do {
            // load 16 x 16 block of B using NX x 4 threads
            #pragma unroll
            for( int i=0; i < 16; i += 16 ) {  // += blockDim.x
                #pragma unroll
                for( int j=0; j < 16; j += 4 ) {  // += blockDim.y
                    sB[tx + i][ty + j] = B[i + j*ldb];
                }
            }
            __syncthreads();

            if ( ind < n ) {
                // load NT x 16 block of A; each thread initially loads 1x4 row,
                // then continues loading more elements as axpys are done.
                rA[0] = A[0*lda];
                rA[1] = A[1*lda];
                rA[2] = A[2*lda];
                rA[3] = A[3*lda];
                
                // axpy:  C(id,:) += A(id,k) * B(k,:) for k=0, ..., 15
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
            }

            // move to next block of A and B
            A += 16*lda;
            B += 16;
            __syncthreads();
        } while( B < Blast );

        // write NT x 16 result; each thread writes one 16x1 row, C(id,0:15)
        for( int i = 0; i < 16; i++ ) {
            C[0] = -rC[i];
            C += ldc;
        }
    }
}


/*
 * zero out B12 temp location
 */
static __device__ void
triple_zgemm_above64_part3_lower_device(
    int n, const magmaDoubleComplex *Ain, int lda, magmaDoubleComplex *d_dinvA, int jb, int npages)
{
    const int by   = blockIdx.y / npages;
    const int page = blockIdx.y % npages;
    const int tx   = threadIdx.x;
    const int ty   = threadIdx.y;
    const int ibx  = blockIdx.x*64;
    const int iby  = by*16;
    const int id   = tx + ty*16;

    // go to the (page / pages_per_NB) outer NB*NB block,
    // then  the (page % pages_per_NB) inner (jb*2)*(jb*2) page inside that.
    int pages_per_NB = NB/(jb*2);
    d_dinvA += (page / pages_per_NB)*NB*NB
             + (page % pages_per_NB)*(jb*2*NB + jb*2);

    //--------------------------part three---------------------------//
    {
        // zero out B12 temp location
        magmaDoubleComplex *B12;
        int ldb = NB;

        B12 = d_dinvA + jb*NB;
        B12 += ibx + id + iby*ldb;
        
        #pragma unroll
        for( int i = 0; i < 16; i++ ) {
            B12[i*ldb] = MAGMA_Z_ZERO;
        }
    }
}
