/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       zsymv_upper.cu is nearly identical to zhemv_upper.cu, just change names and drop MAGMA_Z_CONJ.
       
       zhemv_kernel_U (upper) in zhemv_upper.cu is very similar to
       zhemv_kernel_L (lower) in zhemv.cu; diff the two files to compare.
       
       @precisions normal z -> s d c
       
       @author Mark Gates
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "commonblas_z.h"

#define PRECISION_z

#define NB_X         64
#define NB_Y          4
#define bank_shift   33
#define quarter_NB_X 16
#define half_NB_X    32


/***************************************************************************//**
    Upper case, compute block multiply, work = A*x, for any size n:
    
           [ (A11*x1 + A12*x2 + A13*x3)     ---                 ---    ]   [ A11    A12    A13 ]   [ x1 ]
    work = [ (A12^H*x1)                   (A22*x2 + A23*x3)     ---    ] = [ A12^H  A22    A23 ] * [ x2 ]
           [ (A13^H*x1)                   (A23^H*x2)          (A33*x3) ]   [ A13^H  A23^H  A33 ]   [ x3 ]
    
    The order is different from the lower case, because
    the upper case processes a block row from the diagonal to the right, whereas
    the lower case processes a block row from the diagonal to the left.
    
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
*******************************************************************************/
SYCL_EXTERNAL void zhemv_kernel_U_mgpu(
    int n, magmaDoubleComplex const *__restrict__ A, int lda,
    magmaDoubleComplex const *__restrict__ x, int incx,
    magmaDoubleComplex *__restrict__ work, int my_gpu_id, int ngpu,
    int block_offset, sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write,
                   sycl::access::target::local>
        sA,
    magmaDoubleComplex *sx_blk, magmaDoubleComplex *sx_jj)
{
#if defined(PRECISION_s) || defined(PRECISION_d) || defined(PRECISION_c) ||    \
    (DPCT_COMPATIBILITY_TEMP >= 200) || defined(MAGMA_HAVE_HIP)

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

    // 32x8 thread block
    const int tx2 = td % half_NB_X;
    const int ty2 = td / half_NB_X;

    // If this blk has fewer than NB_X rows, partial is the number of valid rows,
    // so tx = 0, ..., partial-1 are valid rows, and tx >= partial are invalid.
    // Else, partial == 0.
    int partial =
        (blk == item_ct1.get_group_range(2) - 1 ? ((n + block_offset) % NB_X)
                                                : 0);

    magmaDoubleComplex psum, psum_t;
    magmaDoubleComplex total = MAGMA_Z_ZERO;

    // sA is used as a 32x32 block, sA32(i,j),
    // and as a 16x64 block, sA16(i,j), in different parts of the code.
    // sA must be at least half_NB_X*bank_shift = 32x33 = 1056;
    // quarter_NB_X*(NB_X + 2) = 16*(64 + 2) = 1056
     /* Why +3? seems it only needs +2. Does +3 reduce bank conflicts? */
      // for x[ blk ]
      // for x[ jj ], which cycles over all blocks right of diag

    magmaDoubleComplex rA[4];
    magmaDoubleComplex psums_t[4];

    // --------------------
    // load 64x1 block x(blk_ind + 0:63) into sx_blk
    x += (blk_ind + tx)*incx;  // x is x(blk_ind + tx)
    if ( ty == 0 ) {
        if ( (partial && tx >= partial) ||
             (blk == 0 && tx < block_offset) ) {
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
        A += (blk/ngpu)*NB_X*lda;    // A is A(blk_ind + tx2, blk_ind + ty2)
        
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
                    DPCT1064:980: Migrated make_cuDoubleComplex call is used in
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
        DPCT1065:964: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // symmetrize 32x32 diag block, copying upper to lower triangle,
        // as four 32x8 sections in parallel:
        // columns 0,4,8,12,16,20,24,28; then 1,5,...,29; then 2,6,...,30, then 3,7,...,31
        #pragma unroll
        for (int j=ty2*4; j < ty2*4 + 4; j++) {
            if ( j > tx2 ) {
                sA32(j, tx2) = MAGMA_Z_CONJ( sA32(tx2, j) );
            }
        }
        /*
        DPCT1065:965: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // multiply 32x32 diag block * x
        // each thread does partial row sA(tx2, ty2*4 : ty2*4 + 3)
        /*
        DPCT1064:981: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        psum = MAGMA_Z_ZERO;
#pragma unroll
        for (int j=0; j < 4; j++) {
            psum += sA32(tx2, ty2*4 + j) * sx_blk[ty2*4 + j];
        }
        /*
        DPCT1065:966: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // store partial row sums
        sA32(ty2, tx2) = psum;
        /*
        DPCT1065:967: Consider replacing sycl::nd_item::barrier() with
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
        DPCT1065:968: Consider replacing sycl::nd_item::barrier() with
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
                    /*
                    DPCT1064:982: Migrated make_cuDoubleComplex call is used in
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
        DPCT1065:969: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // symmetrize 32x32 diag block, copying upper to lower triangle
        #pragma unroll
        for (int j=ty2*4; j < ty2*4 + 4; j++) {
            if ( j > tx2 ) {
                sA32(j, tx2) = MAGMA_Z_CONJ( sA32(tx2, j) );
            }
        }
        /*
        DPCT1065:970: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // multiply 32x32 diag block * x
        /*
        DPCT1064:983: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        psum = MAGMA_Z_ZERO;
#pragma unroll
        for (int j=0; j < 4; j++) {
            psum += sA32(tx2, ty2*4 + j) * sx_blk[half_NB_X + ty2*4 + j];
        }
        /*
        DPCT1065:971: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // store partial row sums
        sA32(ty2, tx2) = psum;
        /*
        DPCT1065:972: Consider replacing sycl::nd_item::barrier() with
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
        DPCT1065:973: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // --------------------
        // move to off-diag 32x32 block
        A -= half_NB_X;  // A is A(blk_ind + tx2, blk_ind + NB/2 + ty2)
        
        // load 32x32 block of A into sA,
        // as four 32x8 sections one after another:
        // columns 0:7, then 8:15, then 16:23, then 24:31
        if ( partial ) {
            if ( tx2 >= partial ) {
                A = A - (tx2) + (partial - 1);
            }
            #pragma unroll
            for (int j=0; j < half_NB_X; j += 8) {
                if ( ty2+j + half_NB_X < partial ) {
                    sA32(tx2, ty2 + j) = A[j*lda];
                }
                else {
                    /*
                    DPCT1064:984: Migrated make_cuDoubleComplex call is used in
                    a macro definition and is not valid for all macro uses.
                    Adjust the code.
                    */
                    sA32(tx2, ty2 + j) = MAGMA_Z_ZERO;
                }
            }
            if ( tx2 >= partial ) {
                A = A + (tx2) - (partial - 1);
            }
        }
        else {
            #pragma unroll
            for (int j=0; j < half_NB_X; j += 8) {
                sA32(tx2, ty2 + j) = A[j*lda];
            }
        }
        /*
        DPCT1065:974: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // multiply 32x32 block (below diag)
        /*
        DPCT1064:985: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        psum = MAGMA_Z_ZERO;
#pragma unroll
        for (int j=0; j < 4; j++) {
            psum += MAGMA_Z_CONJ( sA32(ty2 + j*8, tx2) ) * sx_blk[j*8 + ty2];
        }
        //__syncthreads();  // no sync needed here
        
        // multiply transposed 32x32 block (above diag)
        /*
        DPCT1064:986: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        psum_t = MAGMA_Z_ZERO;
#pragma unroll
        for (int j=0; j < 4; j++) {
            psum_t += sA32(tx2, ty2*4 + j) * sx_blk[half_NB_X + ty2*4 + j];
        }
        /*
        DPCT1065:975: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // store partial sums for non-transposed 32x32 block
        sA32(ty2, tx2) = psum;
        /*
        DPCT1065:976: Consider replacing sycl::nd_item::barrier() with
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
        DPCT1065:977: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // store partial sums for transposed 32x32 block
        sA32(ty2, tx2) = psum_t;
        /*
        DPCT1065:978: Consider replacing sycl::nd_item::barrier() with
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
        DPCT1065:979: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // --------------------
        // move to next 64x64 block right of diag in block row, and
        // switch thread offset from (tx2,ty2) 32x8 block to (tx,ty) 64x4 block
        A -= half_NB_X*lda;        // A is A(blk_ind + tx2, blk_ind + ty2)
        A -= (blk/ngpu)*NB_X*lda;  // A is A(blk_ind + tx2,           ty2)
    }
    
    // finish switching thread offset
    A -= ty2*lda + tx2;  // A is A(blk_ind,      0)
    A += 4*ty*lda + tx;  // A is A(blk_ind + tx, 4*ty)
    int next = blk + (my_gpu_id + ngpu - 1 - blk % ngpu) % ngpu + 1;
    A += (next/ngpu)*NB_X*lda;  // A is A(blk_ind + tx, next*NB_X + 4*ty)
    
    // Unlike lower case, don't adjust A here for partial # of rows.
    // Since block is right of diagonal, it must have all NB rows,
    // but can have < NB columns, dealt with when loading below.
    
    x -= blk_ind*incx;  // x is x(tx)
    
    // 16x16 thread block
    const int tx4 = td % quarter_NB_X;
    const int ty4 = td / quarter_NB_X;
    
    // cycle over blocks jj right of diagonal, in block row blk
    for (int jj = next; jj < item_ct1.get_group_range(2); jj += ngpu) {
        partial =
            (jj == item_ct1.get_group_range(2) - 1 ? ((n + block_offset) % NB_X)
                                                   : 0);

        // load 64x1 block x(jj_ind + 0:63) into sx_jj
        // block is right of diagonal, so don't need to worry about offset here
        if ( ty == 0 ) {
            if ( partial == 0 || tx < partial ) {
                sx_jj[tx] = x[jj*NB_X*incx];
            }
            else {
                /*
                DPCT1064:990: Migrated make_cuDoubleComplex call is used in a
                macro definition and is not valid for all macro uses. Adjust the
                code.
                */
                sx_jj[tx] = MAGMA_Z_ZERO;
            }
        }
        /*
        DPCT1065:987: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        for (int k=0; k < 4; k++) {
            // load 64x16 block of A into rA, 4 elements per thread,
            // as four 64x4 sections in parallel:
            // columns 0,4,8,12; then 1,5,9,13; then 2,6,10,14; then 3,7,11,15
            if ( partial ) {
                #pragma unroll
                for (int j=0; j < 4; j++) {
                    if ( 4*ty + j + k*quarter_NB_X < partial ) {
                        rA[j] = A[j*lda];
                    }
                    else {
                        /*
                        DPCT1064:993: Migrated make_cuDoubleComplex call is used
                        in a macro definition and is not valid for all macro
                        uses. Adjust the code.
                        */
                        rA[j] = MAGMA_Z_ZERO;
                    }
                }
            }
            else {
                #pragma unroll
                for (int j=0; j < 4; j++) {
                    rA[j] = A[j*lda];
                }
            }
            
            // 1) multiply 64x16 block A_{blk,jj} * x_jj
            //    each thread does partial row rA(tx + 16*k, ty*4 + 16*k : ty*4 + 3 + 16*k)
            // 2) multiply 16x64 block A_{blk,jj} * x_blk,
            //    storing each product Aji*xi to sA(j,i)
            #pragma unroll
            for (int j=0; j < 4; j++) {
                total += rA[j] * sx_jj[quarter_NB_X*k + ty*4 + j];  // y_blk = A_{blk,jj}   * x_jj
                sA16(ty*4 + j, tx) = MAGMA_Z_CONJ( rA[j] ) * sx_blk[tx];  // y_jj  = A_{blk,jj}^H * x_blk
            }
            /*
            DPCT1065:991: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // do partial row sums for transposed 16x64 result
            // use 16x16 thread grid (tx4, ty4) instead of 64x4 (tx, ty)
            // sum sixteen 16x4 sections in parallel:
            // columns 0,4,8,...,60; then 1,5,...,61; then 2,6,...,62; then 3,7,...,63
            /*
            DPCT1064:994: Migrated make_cuDoubleComplex call is used in a macro
            definition and is not valid for all macro uses. Adjust the code.
            */
            psum_t = MAGMA_Z_ZERO;
#pragma unroll
            for (int j=0; j < 4; j++) {
                psum_t += sA16(tx4, ty4*4 + j);
            }
            /*
            DPCT1065:992: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // store partial row sums of transposed result, y_jj (locally)
            psums_t[k] = psum_t;
    
            // move right to next 64x16 block
            A += lda * quarter_NB_X;  // A is A(blk_ind + tx, jj*NB_X + (k+1)*NB_X/4 + 4*ty)
        }
        // already at next 64x64 block
        // A is A(blk_ind + tx, (jj+1)*NB_x + 4*ty)
    
        // store partial row sums of transposed result, y_jj
        #pragma unroll
        for (int k=0; k < 4; k++) {
            sA16(tx4, ty4 + quarter_NB_X*k) = psums_t[k];
        }
        /*
        DPCT1065:988: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // sum up partial row sums of transposed result, y_jj, and store final total to workspace
        // thread (tx4,ty4) where ty4 < 4 sums row tx4 + ty4*16
        if ( ty4 < 4 && (partial == 0 || tx4 + ty4*quarter_NB_X < partial) ) {
            int ty4_nb4 = ty4*quarter_NB_X;
            psum_t = sA16(tx4,  0 + ty4_nb4) + sA16(tx4,  1 + ty4_nb4)
                   + sA16(tx4,  2 + ty4_nb4) + sA16(tx4,  3 + ty4_nb4)
                   + sA16(tx4,  4 + ty4_nb4) + sA16(tx4,  5 + ty4_nb4)
                   + sA16(tx4,  6 + ty4_nb4) + sA16(tx4,  7 + ty4_nb4)
                   + sA16(tx4,  8 + ty4_nb4) + sA16(tx4,  9 + ty4_nb4)
                   + sA16(tx4, 10 + ty4_nb4) + sA16(tx4, 11 + ty4_nb4)
                   + sA16(tx4, 12 + ty4_nb4) + sA16(tx4, 13 + ty4_nb4)
                   + sA16(tx4, 14 + ty4_nb4) + sA16(tx4, 15 + ty4_nb4);
            work[jj*NB_X + tx4 + ty4_nb4] = psum_t;  //MAGMA_Z_MAKE( tx4, blk );  // store at work( jj*NB_X + tx4 + ty4*16, blk )
        }
        /*
        DPCT1065:989: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }
    
    // store row sums
    sA16(ty, tx) = total;
    /*
    DPCT1065:963: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    partial =
        (blk == item_ct1.get_group_range(2) - 1 ? ((n + block_offset) % NB_X)
                                                : 0);

    // sum up final total, y_blk, for row tx
    if ( ty == 0 && (partial == 0 || tx < partial) ) {
        total = sA16(0, tx)
              + sA16(1, tx)
              + sA16(2, tx)
              + sA16(3, tx);
        work[blk*NB_X + tx] = total;  //MAGMA_Z_MAKE( tx, blk );  // store at work( blk*NB_X + tx, blk )
    }
#endif  /* PRECISION_[sdc] || (__CUDA_ARCH__ >= 200) || defined(MAGMA_HAVE_HIP) */
}
// end zhemv_kernel_U_mgpu


/***************************************************************************//**
    Upper case, sum up partial results per GPU.
    Each block sums one block row; each thread sums one row.
    
    On input (for 3 blocks):
           [ (A11*x1 + A12*x2 + A13*x3)     ---                 ---    ]
    work = [ (A12^H*x1)                   (A22*x2 + A23*x3)     ---    ]
           [ (A13^H*x1)                   (A23^H*x2)          (A33*x3) ]
    
    On output:
              [ (A11*x1 + A12*x2 + A13*x3)         ]
    y = alpha*[ (A12^H*x1) + (A22*x2 + A23*x3)     ]
              [ (A13^H*x1) + (A23^H*x2) + (A33*x3) ]
    Note beta*y is not included here; see magmablas_zhemv_mgpu_sync.
    
    The above workspace is distributed over multiple GPUs as diagrammed for 5 blocks:
    
                  [ *         ]  blk=0  * data for non-transposed row   w_blk = A_{blk,1:nblock} * x_{blk:nblock}
    work[gpu=0] = [   *       ]  blk=1  x data for     transposed block w_jj  = A_{blk,jj}^H     * x_{blk}
                  [ x x *     ]  blk=2  blanks are not set
                  [       *   ]  blk=3
                  [ x x x x * ]  blk=4
    
                  [ *         ]  blk=0
    work[gpu=1] = [ x *       ]  blk=1
                  [     *     ]  blk=2
                  [ x x x *   ]  blk=3
                  [         * ]  blk=4
    
    On output, rows across are summed up.
    Entries right of the diagonal blocks are not accessed.
    There are no blank lines; work has been set to 0 if a GPU has no data to contribute.
    
                  [ *                 ]
    y[gpu=0]    = [ *                 ]
                  [ x + x + *         ]
                  [ *                 ]
                  [ x + x + x + x + * ]
    
                  [ *                 ]
    y[gpu=1]    = [ x + *             ]
                  [ *                 ]
                  [ x + x + x + *     ]
                  [ *                 ]
*******************************************************************************/
SYCL_EXTERNAL void
zhemv_kernel_U_mgpu_sum(int n, magmaDoubleComplex alpha, int lda,
                        magmaDoubleComplex *__restrict__ y, int incy,
                        magmaDoubleComplex const *__restrict__ work,
                        int my_gpu_id, int ngpu, int block_offset,
                        sycl::nd_item<3> item_ct1)
{
    int tx = item_ct1.get_local_id(2);
    int blk = item_ct1.get_group(2);
    int blk_ind = blk * NB_X;
    int ind     = blk_ind + tx;

    // Don't write outside [block_offset, ..., n+block_offset)
    if ( ind >= block_offset && ind < n+block_offset ) {
        /*
        DPCT1064:995: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        magmaDoubleComplex Ax = MAGMA_Z_ZERO;
        work += ind;
        // if this GPU owns block-column blk, all blocks j=[0, ..., blk] contain data;
        // else only block j=blk contains data.
        int first = 0;
        if ( blk % ngpu != my_gpu_id ) {
            first = blk;
        }
        for (int j = first; j <= blk; ++j) {
            Ax += work[j*lda];
        }
        y[ind * incy] = alpha*Ax;  // see magmablas_zhemv_sync for beta*y
    }
}
// end zhemv_kernel_L_mgpu_sum
