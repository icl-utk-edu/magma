/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Mark Gates
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "commonblas_z.h"

#define NB_X         64
#define NB_Y          4
#define bank_shift   33
#define quarter_NB_X 16
#define half_NB_X    32


/***************************************************************************//**
    Lower case, compute block multiply, work = A*x, for any size n:
    
           [ (A11*x1)   (A21^H*x2)          (A31^H*x3)                 ]   [ A11  A21^H  A31^H ]   [ x1 ]
    work = [   ---      (A21*x1 + A22*x2)   (A32^H*x3)                 ] = [ A21  A22    A32^H ] * [ x2 ]
           [   ---        ---               (A31*x1 + A32*x2 + A33*x3) ]   [ A31  A32    A33   ]   [ x3 ]
    
    Uses a 64x4 thread block.
    For     diagonal tiles, covers a 64x64 tile using three 32x32 tiles (plus one gets transposed).
    For off-diagonal tiles, covers a 64x64 tile using four  64x16 tiles.
    In both cases, each thread multiplies 4 elements.
    
    For rows past the bottom of the matrix, the A pointer is adjusted to be the
    last valid row of A, which multiple threads will read.
    Extra rows are ignored when saving results to work.
    Columns past the right edge are explicitly ignored when loading.
    x values past the bottom are set to zero, thus, extra columns are zeroed
    when multiplying.
    
    Previously:
           [ (A11*x1)       ---                                          ]
    work = [ (A21^H*x2)   (A21*x1 + A22*x2)     ---                      ]
           [ (A31^H*x3)   (A32^H*x3)          (A31*x1 + A32*x2 + A33*x3) ]
    which doesn't work as well because that has dimension blocks*NB by blocks,
    where blocks*NB >= n, and it can be that blocks*NB > lda, so it won't fit in
    lda*blocks space. This is why it used to need lwork = lda*(blocks + 1).
*******************************************************************************/
void
zhemv_kernel_L_mgpu(
    int n,
    magmaDoubleComplex const * __restrict__ A, int lda,
    magmaDoubleComplex const * __restrict__ x, int incx,
    magmaDoubleComplex       * __restrict__ work,
    int my_gpu_id,
    int ngpu,
    int block_offset , sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
    magmaDoubleComplex *sx_blk, magmaDoubleComplex *sx_jj)
{
#if (DPCT_COMPATIBILITY_TEMP >= 200) || defined(MAGMA_HAVE_HIP)

    // treats sA as 16x64 block
    #define sA16(i_, j_) (sA[(i_)][(j_)])  // i.e., sA[ (i_)*(NB_X+3) + (j_) ]
    
    // treats sA as 32x32 block
    #define sA32(i_, j_) (sA[0][(i_) + bank_shift*(j_)])
    
    // 64x4 thread block
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);
    const int blk = item_ct1.get_group(2);
    const int blk_ind = NB_X * blk;
    const int td  = NB_X * ty + tx;

    // GPUs are renumbered so that GPU 0 starts with block 0, GPU 1 starts with block 1, etc.
    if ( blk < my_gpu_id ) {
        return;
    }

    // 32x8 thread block
    const int tx2 = td % half_NB_X;
    const int ty2 = td / half_NB_X;

    // If this blk has fewer than NB_X rows, partial is the number of valid rows,
    // so tx = 0, ..., partial-1 are valid rows, and tx >= partial are invalid.
    // Else, partial == 0.
    const int partial =
        (blk == item_ct1.get_group_range(2) - 1 ? ((n + block_offset) % NB_X)
                                                : 0);

    magmaDoubleComplex psum, psum_t;
    /*
    DPCT1064:928: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    magmaDoubleComplex total = MAGMA_Z_ZERO;

    // sA is used as a 32x32 block, sA32(i,j),
    // and as a 16x64 block, sA16(i,j), in different parts of the code.
    // sA must be at least half_NB_X*bank_shift = 32x33 = 1056;
    // quarter_NB_X*(NB_X + 2) = 16*(64 + 2) = 1056
    // TODO +3 used in zhemv (single GPU); why?
    // for x[ blk ]
    // for x[ jj ], which cycles over all blocks left of diag

    magmaDoubleComplex rA[4];
    magmaDoubleComplex psums_t[4];

    // --------------------
    // load 64x1 block x(blk_ind + 0:63) into sx_blk
    x += (blk_ind + tx)*incx;  // x is x(blk_ind + tx)
    if ( ty == 0 ) {
        // GPUs are renumbered so that GPU 0 has block 0, which is partial of offset.
        if ( (partial && tx >= partial) ||
             (blk == 0  /*&& my_gpu_id == 0*/  && tx < block_offset) ) {
            sx_blk[tx] = MAGMA_Z_ZERO;
        }
        else {
            sx_blk[tx] = x[0];
        }
    }

    // --------------------
    // move to block row
    work += blk*lda;     // work is work(0, blk)

    A += blk_ind;        // A is A(blk_ind, 0)
    A += ty2*lda + tx2;  // A is A(blk_ind + tx2, ty2)
    if ( blk % ngpu == my_gpu_id ) {
        // this GPU owns this diagonal block, so
        // move to 32x32 diag block
        A += (blk/ngpu)*NB_X*lda;  // A is A(blk_ind + tx2, blk_ind + ty2)

        // load 32x32 diag block A(blk_ind + 0:31, blk_ind + 0:31) into sA,
        // as four 32x8 sections one after another:
        // columns 0:7, then 8:15, then 16:23, then 24:31
        if ( partial ) {
            if ( tx2 >= partial ) {
                A = A - tx2 + (partial - 1);  // A is A(blk_ind + partial-1, blk_ind + ty2), the bottom-most valid row
            }
            #pragma unroll
            for (int j=0; j < half_NB_X; j += 8) {
                if ( ty2+j < partial ) {
                    sA32(tx2, ty2 + j) = A[j*lda];
                }
                else {
                    /*
                    DPCT1064:945: Migrated make_cuDoubleComplex call is used in
                    a macro definition and is not valid for all macro uses.
                    Adjust the code.
                    */
                    sA32(tx2, ty2 + j) = MAGMA_Z_ZERO;
                }
            }
            if ( tx2 >= partial ) {
                A = A + tx2 - (partial - 1);  // A is A(blk_ind + tx2, blk_ind + ty2)
            }
        }
        else {
            #pragma unroll
            for (int j=0; j < half_NB_X; j += 8) {
                sA32(tx2, ty2 + j) = A[j*lda];
            }
        }
        /*
        DPCT1065:929: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // symmetrize 32x32 diag block, copying lower to upper triangle,
        // as four 32x8 sections in parallel:
        // columns 0,4,8,12,16,20,24,28; then 1,5,...,29; then 2,6,...,30, then 3,7,...,31
        #pragma unroll
        for (int j=ty2*4; j < ty2*4 + 4; j++) {
            if ( j < tx2 ) {
                sA32(j, tx2) = MAGMA_Z_CONJ( sA32(tx2, j) );
            }
        }
        /*
        DPCT1065:930: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // multiply 32x32 diag block * x
        // each thread does partial row sA(tx2, ty2*4 : ty2*4 + 3)
        /*
        DPCT1064:946: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        psum = MAGMA_Z_ZERO;
#pragma unroll
        for (int j=0; j < 4; j++) {
            psum += sA32(tx2, ty2*4 + j) * sx_blk[ty2*4 + j];
        }
        /*
        DPCT1065:931: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // store partial row sums
        sA32(ty2, tx2) = psum;
        /*
        DPCT1065:932: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // sum up partial row sums, so thread (tx2,0) has total for row (blk_ind + tx2)
        if ( ty2 == 0 ) {
            total = sA32(0, tx2) + sA32(1, tx2)
                  + sA32(2, tx2) + sA32(3, tx2)
                  + sA32(4, tx2) + sA32(5, tx2)
                  + sA32(6, tx2) + sA32(7, tx2);
        }
        /*
        DPCT1065:933: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // --------------------
        // move to next 32x32 diag block, then repeat steps from first diag block
        A += half_NB_X + half_NB_X*lda;  // A is A(blk_ind + NB/2 + tx2, blk_ind + NB/2 + ty2)

        // load 32x32 diag block A[block + 0:31, block + 0:31] into sA
        if ( partial ) {
            if ( tx2 + half_NB_X >= partial ) {
                A = A - (tx2 + half_NB_X) + (partial - 1);
            }
            #pragma unroll
            for (int j=0; j < half_NB_X; j += 8) {
                if ( ty2+j + half_NB_X < partial ) {
                    sA32(tx2, ty2 + j) = A[j*lda];
                }
                else {
                    sA32(tx2, ty2 + j) = MAGMA_Z_ZERO;
                }
            }
            if ( tx2 + half_NB_X >= partial ) {
                A = A + (tx2 + half_NB_X) - (partial - 1);
            }
        }
        else {
            #pragma unroll
            for (int j=0; j < half_NB_X; j += 8) {
                sA32(tx2, ty2 + j) = A[j*lda];
            }
        }
        /*
        DPCT1065:934: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // symmetrize 32x32 diag block, copying lower to upper triangle
        #pragma unroll
        for (int j=ty2*4; j < ty2*4 + 4; j++) {
            if ( j < tx2 ) {
                sA32(j, tx2) = MAGMA_Z_CONJ( sA32(tx2, j) );
            }
        }
        /*
        DPCT1065:935: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // multiply 32x32 diag block * x
        /*
        DPCT1064:947: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        psum = MAGMA_Z_ZERO;
#pragma unroll
        for (int j=0; j < 4; j++) {
            psum += sA32(tx2, ty2*4 + j) * sx_blk[half_NB_X + ty2*4 + j];
        }
        /*
        DPCT1065:936: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // store partial row sums
        sA32(ty2, tx2) = psum;
        /*
        DPCT1065:937: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // sum up partial row sums, so thread (tx2,1) has total for row (blk_ind + NB/2 + tx2)
        if ( ty2 == 1 ) {
            total = sA32(0, tx2) + sA32(1, tx2)
                  + sA32(2, tx2) + sA32(3, tx2)
                  + sA32(4, tx2) + sA32(5, tx2)
                  + sA32(6, tx2) + sA32(7, tx2);
        }
        /*
        DPCT1065:938: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // --------------------
        // move to off-diag 32x32 block
        A -= half_NB_X*lda;  // A is A(blk_ind + NB/2 + tx2, blk_ind + ty2)

        // load 32x32 block of A into sA,
        // as four 32x8 sections one after another:
        // columns 0:7, then 8:15, then 16:23, then 24:31
        if ( partial ) {
            if ( tx2 + half_NB_X >= partial ) {
                A = A - (tx2 + half_NB_X) + (partial - 1);
            }
            #pragma unroll
            for (int j=0; j < half_NB_X; j += 8) {
                if ( ty2+j < partial ) {
                    sA32(tx2, ty2 + j) = A[j*lda];
                }
                else {
                    /*
                    DPCT1064:948: Migrated make_cuDoubleComplex call is used in
                    a macro definition and is not valid for all macro uses.
                    Adjust the code.
                    */
                    sA32(tx2, ty2 + j) = MAGMA_Z_ZERO;
                }
            }
            if ( tx2 + half_NB_X >= partial ) {
                A = A + (tx2 + half_NB_X) - (partial - 1);
            }
        }
        else {
            #pragma unroll
            for (int j=0; j < half_NB_X; j += 8) {
                sA32(tx2, ty2 + j) = A[j*lda];
            }
        }
        /*
        DPCT1065:939: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // multiply 32x32 block (below diag)
        psum = MAGMA_Z_ZERO;
        #pragma unroll
        for (int j=0; j < 4; j++) {
            psum += sA32(tx2, ty2 + j*8) * sx_blk[j*8 + ty2];
        }
        //__syncthreads();  // no sync needed here

        // multiply transposed 32x32 block (above diag)
        /*
        DPCT1064:949: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        psum_t = MAGMA_Z_ZERO;
#pragma unroll
        for (int j=0; j < 4; j++) {
            psum_t += MAGMA_Z_CONJ( sA32(ty2*4 + j, tx2) ) * sx_blk[half_NB_X + ty2*4 + j];
        }
        /*
        DPCT1065:940: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // store partial sums for non-transposed 32x32 block
        sA32(ty2, tx2) = psum;
        /*
        DPCT1065:941: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // sum up partial row sums, so thread (tx2,1) has total for row (blk_ind + NB/2 + tx2)
        if ( ty2 == 1 ) {
            total = total
                  + sA32(0, tx2) + sA32(1, tx2)
                  + sA32(2, tx2) + sA32(3, tx2)
                  + sA32(4, tx2) + sA32(5, tx2)
                  + sA32(6, tx2) + sA32(7, tx2);
        }
        /*
        DPCT1065:942: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // store partial sums for transposed 32x32 block
        sA32(ty2, tx2) = psum_t;
        /*
        DPCT1065:943: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // sum up partial row sums, so thread (tx2,0) has total for row (blk_ind + tx2)
        if ( ty2 == 0 ) {
            total = total
                  + sA32(0, tx2) + sA32(1, tx2)
                  + sA32(2, tx2) + sA32(3, tx2)
                  + sA32(4, tx2) + sA32(5, tx2)
                  + sA32(6, tx2) + sA32(7, tx2);
        }
        /*
        DPCT1065:944: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // --------------------
        // move to leftmost 64x64 block in block row, and
        // switch thread offset from (tx2,ty2) 32x8 block to (tx,ty) 64x4 block
        A -= half_NB_X;            // A is A(blk_ind + tx2, blk_ind + ty2)
        A -= (blk/ngpu)*NB_X*lda;  // A is A(blk_ind + tx2,           ty2)
    }

    // finish switching thread offset
    A -= ty2*lda + tx2;  // A is A(blk_ind, 0)
    A += 4*ty*lda + tx;  // A is A(blk_ind + tx, 4*ty)

    if ( partial && tx >= partial ) {
        A = A - tx + (partial - 1);  // A is A(blk_ind + partial-1, 4*ty), the bottom-most valid row
    }
    
    x -= blk_ind*incx;  // x is x(tx)

    // 16x16 thread block
    const int tx4 = td % quarter_NB_X;
    const int ty4 = td / quarter_NB_X;

    // cycle over blocks jj left of diagonal, in block row blk
    for (int jj=my_gpu_id; jj < blk; jj += ngpu) {
        // load 64x1 block x(jj_ind + 0:63) into sx_jj
        // since this block is left of diagonal, x must have all NB rows
        // only the first block column (jj=0, on GPU 0) deals with offset
        if ( ty == 0 ) {
            if ( jj == 0 && tx < block_offset ) {
                /*
                DPCT1064:953: Migrated make_cuDoubleComplex call is used in a
                macro definition and is not valid for all macro uses. Adjust the
                code.
                */
                sx_jj[tx] = MAGMA_Z_ZERO;
            }
            else {
                sx_jj[tx] = x[jj*NB_X*incx];
            }
        }
        /*
        DPCT1065:950: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        for (int k=0; k < 4; k++) {
            // load 64x16 block of A into rA, 4 elements per thread,
            // as four 64x4 sections in parallel:
            // columns 0,4,8,12; then 1,5,9,13; then 2,6,10,14; then 3,7,11,15
            // since this block is left of diagonal, it has all NB columns,
            // and block of x must have all NB rows.
            #pragma unroll
            for (int j=0; j < 4; j++) {
                rA[j] = A[j*lda];
            }

            // 1) multiply 64x16 block A_{blk,jj} * x_jj
            //    each thread does partial row rA(tx + 16*k, ty*4 + 16*k : ty*4 + 3 + 16*k)
            // 2) multiply transposed 16x64 block A_{blk,jj}^H * x_blk,
            //    storing each product Aji*xi to sA(j,i)
            #pragma unroll
            for (int j=0; j < 4; j++) {
                total += rA[j] * sx_jj[quarter_NB_X*k + ty*4 + j];  // y_blk = A_{blk,jj}   * x_jj
                sA16(ty*4 + j, tx) = MAGMA_Z_CONJ( rA[j] ) * sx_blk[tx];  // y_jj  = A_{blk,jj}^H * x_blk
            }
            /*
            DPCT1065:954: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // do partial row sums for transposed 16x64 result
            // use 16x16 thread grid (tx4, ty4) instead of 64x4 (tx, ty)
            // sum sixteen 16x4 sections in parallel:
            // columns 0,4,8,...,60; then 1,5,...,61; then 2,6,...,62; then 3,7,...,63
            /*
            DPCT1064:956: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            psum_t = MAGMA_Z_ZERO;
#pragma unroll
            for (int j=0; j < 4; j++) {
                psum_t += sA16(tx4, ty4*4 + j);
            }
            /*
            DPCT1065:955: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // store partial row sums of transposed result, y_jj (locally)
            psums_t[k] = psum_t;

            // move right to next 64x16 block
            A += lda * quarter_NB_X;  // A is A(blk_ind + tx#, jj*NB_x + (k+1)*NB_X/4 + 4*ty), # tx or partial
        }
        // already at next 64x64 block
        // A is A(blk_ind + tx#, (jj+1)*NB_x + 4*ty), # tx or partial

        // store partial row sums of transposed result, y_jj
        #pragma unroll
        for (int k=0; k < 4; k++) {
            sA16(tx4, ty4 + quarter_NB_X*k) = psums_t[k];
        }
        /*
        DPCT1065:951: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // sum up partial row sums of transposed result, y_jj, and store final total to workspace
        // thread (tx4,ty4) where ty4 < 4 sums row tx4 + ty4*16
        // since this is the transposed block above the diagonal, it must have all NB rows
        if ( ty4 < 4 ) {
            int ty4_nb4 = ty4*quarter_NB_X;
            psum_t = sA16(tx4,  0 + ty4_nb4) + sA16(tx4,  1 + ty4_nb4)
                   + sA16(tx4,  2 + ty4_nb4) + sA16(tx4,  3 + ty4_nb4)
                   + sA16(tx4,  4 + ty4_nb4) + sA16(tx4,  5 + ty4_nb4)
                   + sA16(tx4,  6 + ty4_nb4) + sA16(tx4,  7 + ty4_nb4)
                   + sA16(tx4,  8 + ty4_nb4) + sA16(tx4,  9 + ty4_nb4)
                   + sA16(tx4, 10 + ty4_nb4) + sA16(tx4, 11 + ty4_nb4)
                   + sA16(tx4, 12 + ty4_nb4) + sA16(tx4, 13 + ty4_nb4)
                   + sA16(tx4, 14 + ty4_nb4) + sA16(tx4, 15 + ty4_nb4);
            work[jj*NB_X + tx4 + ty4_nb4] = psum_t;  // store at work( jj*NB_X + tx4 + ty4*16, blk )
        }
        /*
        DPCT1065:952: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    // store row sums
    sA16(ty, tx) = total;
    /*
    DPCT1065:927: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // sum up final total, y_blk, for row tx
    if ( ty == 0 && (partial == 0 || tx < partial) ) {
        total = sA16(0, tx)
              + sA16(1, tx)
              + sA16(2, tx)
              + sA16(3, tx);
        work[blk*NB_X + tx] = total;  // store at work( blk*NB_X + tx, blk )
    }
#endif  /* (__CUDA_ARCH__ >= 200) || defined(MAGMA_HAVE_HIP) */
}
// end zhemv_kernel_L_mgpu


/***************************************************************************//**
    Lower case, sum up partial results per GPU.
    Each block sums one block row; each thread sums one row.
    
    On input (for 3 blocks):
           [ (A11*x1)   (A21^H*x2)          (A31^H*x3)                 ]
    work = [   ---      (A21*x1 + A22*x2)   (A32^H*x3)                 ]
           [   ---        ---               (A31*x1 + A32*x2 + A33*x3) ]
    
    On output:
              [ (A11*x1) + (A21^H*x2) + (A31^H*x3) ]
    y = alpha*[ (A21*x1 + A22*x2)     + (A32^H*x3) ]
              [ (A21*x1 + A22*x2 + A33*x3)         ]
    Note beta*y is not included here; see magmablas_zhemv_mgpu_sync.
    
    The above workspace is distributed over multiple GPUs as diagrammed for 5 blocks:
    
                  [ * x x x x ]  blk=0  * data for non-transposed row   w_blk = A_{blk,1:blk} * x_{1:blk}
    work[gpu=0] = [   *       ]  blk=1  x data for     transposed block w_jj  = A_{blk,jj}^H  * x_{blk}
                  [     * x x ]  blk=2  blanks are not set
                  [       *   ]  blk=3
                  [         * ]  blk=4
    
                  [           ]  blk=0  (blank)
    work[gpu=1] = [   * x x x ]  blk=1
                  [     *     ]  blk=2
                  [       * x ]  blk=3
                  [         * ]  blk=4
    
    On output, rows across are summed up.
    Entries left of the diagonal blocks are not accessed.
    Blank rows, where a GPU has no data to contribute, are explicitly set to zero in y.
    
                  [ * + x + x + x ]
    y[gpu=0]    = [ *             ]
                  [ * + x         ]
                  [ *             ]
    
                  [ 0             ]  (explicitly set to 0)
    y[gpu=1]    = [ * + x + x     ]
                  [ *             ]
                  [ *             ]
*******************************************************************************/
void
zhemv_kernel_L_mgpu_sum(
    int n,
    magmaDoubleComplex alpha,
    int lda,
    magmaDoubleComplex       * __restrict__ y, int incy,
    magmaDoubleComplex const * __restrict__ work,
    int my_gpu_id,
    int ngpu,
    int block_offset, sycl::nd_item<3> item_ct1)
{
    int tx = item_ct1.get_local_id(2);
    int blk = item_ct1.get_group(2);
    int blk_ind = blk * NB_X;
    int ind     = blk_ind + tx;
    int blocks = item_ct1.get_group_range(2);

    // Don't write outside [block_offset, ..., n+block_offset)
    if ( ind >= block_offset && ind < n+block_offset ) {
        /*
        DPCT1064:957: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        magmaDoubleComplex Ax = MAGMA_Z_ZERO;
        // GPUs are renumbered so that GPU 0 starts with block 0,
        // GPU 1 starts with block 1, etc.,
        // therefore only blk >= my_gpu_id have non-zero data.
        if ( blk >= my_gpu_id ) {
            work += ind;
            // if this GPU owns block-column blk, all blocks j=[blk, ..., blocks) contain data;
            // else only block j=blk contains data.
            int last = blocks-1;
            if ( blk % ngpu != my_gpu_id ) {
                last = blk;
            }
            for (int j = blk; j <= last; ++j) {
                Ax += work[j*lda];
            }
        }
        y[ind * incy] = alpha*Ax;  // see magmablas_zhemv_sync for beta*y
    }
}
// end zhemv_kernel_L_mgpu_sum


/***************************************************************************//**
    Purpose
    -------
    magmablas_zhemv_mgpu performs the matrix-vector operation:

        y := alpha*A*x + beta*y,

    where alpha and beta are scalars, x and y are n element vectors and
    A is an n by n Hermitian matrix.

    Arguments
    ----------
    @param[in]
    uplo    magma_uplo_t.
            On entry, UPLO specifies whether the upper or lower
            triangular part of the array A is to be referenced as
            follows:
      -     = MagmaUpper:  Only the upper triangular part of A is to be referenced. **Not currently supported.**
      -     = MagmaLower:  Only the lower triangular part of A is to be referenced.

    @param[in]
    n       INTEGER.
            On entry, N specifies the order of the matrix A.
            N must be at least zero.

    @param[in]
    alpha   COMPLEX_16.
            On entry, ALPHA specifies the scalar alpha.

    @param[in]
    d_lA    Array of pointers, dimension (ngpu), to block-column distributed
            matrix A, with block size nb.
            d_lA[dev] is a COMPLEX_16 array on GPU dev, of
            dimension (LDDA, nlocal), where
    \n
                     { floor(n/nb/ngpu)*nb + nb    if dev <  floor(n/nb) % ngpu,
            nlocal = { floor(n/nb/ngpu)*nb + n%nb  if dev == floor(n/nb) % ngpu,
                     { floor(n/nb/ngpu)*nb         otherwise.
    \n
            Before entry with  UPLO = MagmaUpper, the leading n by n
            upper triangular part of the array A must contain the upper
            triangular part of the Hermitian matrix and the strictly
            lower triangular part of A is not referenced.
            Before entry with UPLO = MagmaLower, the leading n by n
            lower triangular part of the array A must contain the lower
            triangular part of the Hermitian matrix and the strictly
            upper triangular part of A is not referenced.
            Note that the imaginary parts of the diagonal elements need
            not be set and are assumed to be zero.
            
    @param[in]
    offset  INTEGER.
            Row & column offset to start of matrix A within the distributed d_lA
            structure. Note that N is the size of this multiply, excluding the
            offset, so the size of the original parent matrix is N+offset.
            Also, x and y do not have an offset.
    
    @param[in]
    ldda    INTEGER.
            On entry, LDDA specifies the first dimension of A as declared
            in the calling (sub) program. LDDA must be at least
            max( 1, n + offset ).
            It is recommended that ldda is multiple of 16. Otherwise
            performance would be deteriorated as the memory accesses
            would not be fully coalescent.

    @param[in]
    x       COMPLEX_16 array **on the CPU** (not the GPU), of dimension at least
            ( 1 + ( n - 1 )*abs( INCX ) ).
            Before entry, the incremented array X must contain the n
            element vector x.

    @param[in]
    incx    INTEGER.
            On entry, INCX specifies the increment for the elements of
            X. INCX must not be zero.

    @param[in]
    beta    COMPLEX_16.
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[in,out]
    y       COMPLEX_16 array **on the CPU** (not the GPU), of dimension at least
            ( 1 + ( n - 1 )*abs( INCY ) ).
            Before entry, the incremented array Y must contain the n
            element vector y. On exit, Y is overwritten by the updated
            vector y.

    @param[in]
    incy    INTEGER.
            On entry, INCY specifies the increment for the elements of
            Y. INCY must not be zero.

    @param
    hwork   (workspace) COMPLEX_16 array on the CPU, of dimension (lhwork).
    
    @param[in]
    lhwork  INTEGER.
            The dimension of the array hwork. lhwork >= ngpu*nb.
    
    @param
    dwork   (workspaces) Array of pointers, dimension (ngpu), to workspace on each GPU.
            dwork[dev] is a COMPLEX_16 array on GPU dev, of dimension (ldwork).
    
    @param[in]
    ldwork  INTEGER.
            The dimension of each array dwork[dev].
            ldwork >= ldda*( ceil((n + offset % nb) / nb) + 1 ).
    
    @param[in]
    ngpu    INTEGER.
            The number of GPUs to use.
    
    @param[in]
    nb      INTEGER.
            The block size used for distributing d_lA. Must be 64.
    
    @param[in]
    queues  magma_queue_t array of dimension (ngpu).
            queues[dev] is an execution queue on GPU dev.
    
    @ingroup magma_hemv
*******************************************************************************/
extern "C"
magma_int_t
magmablas_zhemv_mgpu(
    magma_uplo_t uplo,
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_const_ptr const d_lA[], magma_int_t ldda,
    magma_int_t offset,
    magmaDoubleComplex const *x,       magma_int_t incx,
    magmaDoubleComplex beta,                                         // unused, see magmablas_zhemv_mgpu_sync
    magmaDoubleComplex       *y,       magma_int_t incy,             // unused
    magmaDoubleComplex       *hwork,   magma_int_t lhwork,
    magmaDoubleComplex_ptr    dwork[], magma_int_t ldwork,
    magma_int_t ngpu,
    magma_int_t nb,
    magma_queue_t queues[] )
{
    magma_int_t arch = magma_getdevice_arch();
    #ifndef MAGMA_HAVE_HIP
    if ( arch < 200  ) {
        // --------------------
        // no CUDA ARCH 1.x version
        fprintf( stderr, "%s not supported on CUDA arch 1.x\n", __func__ );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
    #endif    

    // --------------------
    // CUDA ARCH 2.x (Fermi) version
    bool upper = (uplo == MagmaUpper);
    
    magma_int_t offset_block_id = offset / NB_X;
    magma_int_t offset_gpu_id   = offset_block_id % ngpu;
    magma_int_t block_offset    = offset % NB_X;
    
    magma_int_t blocks = magma_ceildiv( n + block_offset, NB_X );
    magma_int_t ldwmin = ldda*(blocks + 1);
    magma_int_t lhwmin = n*ngpu;
    
    /*
     * Test the input parameters.
     */
    magma_int_t info = 0;
    if ( (! upper) && (uplo != MagmaLower) ) {
        info = -1;
    } else if ( n < 0 ) {
        info = -2;
    } else if ( ldda < max(1,n+offset) ) {
        info = -5;
    } else if ( offset < 0 ) {
        info = -6;
    } else if ( incx == 0 ) {
        info = -8;
    } else if ( incy == 0 ) {
        info = -11;
    } else if ( lhwork < lhwmin ) {
        info = -13;
    } else if ( ldwork < ldwmin ) {
        info = -15;
    } else if ( ngpu < 1 ) {
        info = -16;
    } else if ( nb != NB_X ) {
        info = -17;
    }
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    /*
     * Quick return if possible.
     */
    if ( n == 0 )
        return info;
    
    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    magma_int_t dev;
    for (dev=0; dev < ngpu; dev++) {
        magma_setdevice( dev );
        
        // blocks before the offset block
        magma_int_t num_blocks_skipped = offset_block_id / ngpu;
        if ( dev < offset_gpu_id ) {
            num_blocks_skipped += 1;
        }
        
        // shift dA to first block >= offset block that is owned by this GPU
        magmaDoubleComplex const *dA_dev    = d_lA[dev] + offset_block_id*NB_X + num_blocks_skipped*NB_X*ldda;
        
        // first column of dwork is to broadcast x to all GPUs.
        // remaining blocks number of columns is for partial sums from
        // each block, as in single GPU version.
        magmaDoubleComplex       *dx_dev    = dwork[dev];
        magmaDoubleComplex       *dwork_dev = dwork[dev] + ldda;
        
        // renumber GPUs starting from the offset block
        magma_int_t new_gpu_id = (dev + ngpu - offset_gpu_id) % ngpu;

        sycl::range<3> grid(1, 1, blocks);

        // copy x to each GPU
        magma_zsetvector_async( n, x, incx, dx_dev + block_offset, 1, queues[dev] );
        
        // perform work = A*x, partial row sums
        sycl::range<3> threads(1, NB_Y, NB_X);

        // perform w = sum( work ), larger partial row sums
        sycl::range<3> threads_sum(1, 1, NB_X);

        if ( upper ) {
            /*
            DPCT1049:958: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queues[dev]->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<magmaDoubleComplex, 2,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sA_acc_ct1(sycl::range<2>(16 /*quarter_NB_X*/,
                                                  67 /*NB_X + 3*/),
                                   cgh);
                    sycl::accessor<magmaDoubleComplex, 1,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sx_blk_acc_ct1(sycl::range<1>(64 /*NB_X*/), cgh);
                    sycl::accessor<magmaDoubleComplex, 1,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sx_jj_acc_ct1(sycl::range<1>(64 /*NB_X*/), cgh);

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zhemv_kernel_U_mgpu(
                                             n, dA_dev, ldda, dx_dev, 1,
                                             dwork_dev, new_gpu_id, ngpu,
                                             block_offset, item_ct1, sA_acc_ct1,
                                             sx_blk_acc_ct1.get_pointer(),
                                             sx_jj_acc_ct1.get_pointer());
                                     });
                });

            /*
            DPCT1049:959: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queues[dev]->sycl_stream()))
                ->parallel_for(
                    sycl::nd_range<3>(grid * threads_sum, threads_sum),
                    [=](sycl::nd_item<3> item_ct1) {
                        zhemv_kernel_U_mgpu_sum(n, alpha, ldda, dx_dev, 1,
                                                dwork_dev, new_gpu_id, ngpu,
                                                block_offset, item_ct1);
                    });
        }
        else {
            /*
            DPCT1049:960: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queues[dev]->sycl_stream()))
                ->submit([&](sycl::handler &cgh) {
                    sycl::accessor<magmaDoubleComplex, 2,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sA_acc_ct1(sycl::range<2>(16 /*quarter_NB_X*/,
                                                  66 /*NB_X + 2*/),
                                   cgh);
                    sycl::accessor<magmaDoubleComplex, 1,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sx_blk_acc_ct1(sycl::range<1>(64 /*NB_X*/), cgh);
                    sycl::accessor<magmaDoubleComplex, 1,
                                   sycl::access_mode::read_write,
                                   sycl::access::target::local>
                        sx_jj_acc_ct1(sycl::range<1>(64 /*NB_X*/), cgh);

                    cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                     [=](sycl::nd_item<3> item_ct1) {
                                         zhemv_kernel_L_mgpu(
                                             n, dA_dev, ldda, dx_dev, 1,
                                             dwork_dev, new_gpu_id, ngpu,
                                             block_offset, item_ct1, sA_acc_ct1,
                                             sx_blk_acc_ct1.get_pointer(),
                                             sx_jj_acc_ct1.get_pointer());
                                     });
                });

            /*
            DPCT1049:961: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
            ((sycl::queue *)(queues[dev]->sycl_stream()))
                ->parallel_for(
                    sycl::nd_range<3>(grid * threads_sum, threads_sum),
                    [=](sycl::nd_item<3> item_ct1) {
                        zhemv_kernel_L_mgpu_sum(n, alpha, ldda, dx_dev, 1,
                                                dwork_dev, new_gpu_id, ngpu,
                                                block_offset, item_ct1);
                    });
        }
    }
    
    // 2nd loop in case hwork is not pinned, causing this to be sync instead of async.
    for (dev=0; dev < ngpu; dev++) {
        // copy w to CPU
        magma_setdevice( dev );
        magmaDoubleComplex       *dx_dev    = dwork[dev];
        magma_zgetvector_async( n, dx_dev + block_offset, 1, &hwork[dev*n], 1, queues[dev] );
    }
    
    // see magmablas_zhemv_mgpu_sync for final row sums
    
    magma_setdevice( orig_dev );
    return info;
}


/***************************************************************************//**
    Synchronizes and acculumates final zhemv result.
    For convenience, the parameters are identical to magmablas_zhemv_mgpu
    (though some are unused here).
    
    @see magmablas_zhemv_mgpu
    @ingroup magma_hemv
*******************************************************************************/
extern "C" magma_int_t
magmablas_zhemv_mgpu_sync(
    magma_uplo_t uplo,                                               // unused, see magmablas_zhemv_mgpu
    magma_int_t n,
    magmaDoubleComplex alpha,                                        // unused
    magmaDoubleComplex_const_ptr const d_lA[], magma_int_t ldda,
    magma_int_t offset,                                              // unused
    magmaDoubleComplex const *x,       magma_int_t incx,             // unused
    magmaDoubleComplex beta,
    magmaDoubleComplex       *y,       magma_int_t incy,             // unused
    magmaDoubleComplex       *hwork,   magma_int_t lhwork,
    magmaDoubleComplex_ptr    dwork[], magma_int_t ldwork,           // unused
    magma_int_t ngpu,
    magma_int_t nb,                                                  // unused
    magma_queue_t queues[] )
{
    /*
    DPCT1064:962: Migrated make_cuDoubleComplex call is used in a macro
    definition and is not valid for all macro uses. Adjust the code.
    */
    const magmaDoubleComplex c_one = MAGMA_Z_ONE;
    const magma_int_t ione = 1;
    
    magma_device_t dev;
    
    magma_int_t lhwmin  = n*ngpu;
    
    /*
     * Test the input parameters.
     */
    magma_int_t info = 0;
    //if ( (! upper) && (uplo != MagmaLower) ) {  // unused
    //    info = -1;
    //} else
    if ( n < 0 ) {
        info = -2;
    } else if ( ldda < max(1,n+offset) ) {
        info = -5;
    } else if ( offset < 0 ) {
        info = -6;
    } else if ( incx == 0 ) {
        info = -8;
    } else if ( incy == 0 ) {
        info = -11;
    } else if ( lhwork < lhwmin ) {
        info = -13;
    //} else if ( ldwork < ldwmin ) {  // unused
    //    info = -15;
    } else if ( ngpu < 1 ) {
        info = -16;
    } else if ( nb != NB_X ) {
        info = -17;
    }
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    /*
     * Quick return if possible.
     */
    if ( n == 0 )
        return info;
    
    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    // scale y = beta*y
    blasf77_zscal( &n, &beta, y, &incy );
    
    // sum reduce, y += sum( hwork )
    for (dev=0; dev < ngpu; ++dev) {
        magma_setdevice( dev );
        magma_queue_sync( queues[dev] );
        blasf77_zaxpy( &n, &c_one, &hwork[dev*n], &ione, y, &ione );
    }
    
    magma_setdevice( orig_dev );
    return info;
}
