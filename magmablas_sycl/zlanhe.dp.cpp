/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @precisions normal z -> s d c

*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"

#define inf_bs 32
#define max_bs 64

#define PRECISION_z
#define COMPLEX


// =============================================================================
// inf-norm

/******************************************************************************/
/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:n-1, for || A ||_inf,
 * where n is any size and A is stored lower.
 * Has ceil( n / inf_bs ) blocks of (inf_bs x 4) threads each (inf_bs=32).
 * z precision uses > 16 KB shared memory, so requires Fermi (arch >= 200). */
void
zlanhe_inf_kernel_lower(
    int n,
    const magmaDoubleComplex * __restrict__ A, int lda,
    double * __restrict__ dwork,
    int n_full_block, int n_mod_bs , sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write, sycl::access::target::local> la)
{
#if (defined(PRECISION_s) || defined(PRECISION_d) || defined(PRECISION_c) ||   \
     DPCT_COMPATIBILITY_TEMP >= 200) ||                                        \
    defined(MAGMA_HAVE_HIP)
    int tx = item_ct1.get_local_id(2);
    int ty = item_ct1.get_local_id(1);

    int diag = item_ct1.get_group(2) * inf_bs;
    int ind = item_ct1.get_group(2) * inf_bs + tx;

    double res = 0.;

    if (item_ct1.get_group(2) < n_full_block) {
        // ------------------------------
        // All full block rows
        A += ind;
        A += ty * lda;
        
        // ----------
        // loop over all blocks left of the diagonal block
        for (int i=0; i < diag; i += inf_bs ) {
            // 32x4 threads cooperatively load 32x32 block
            #pragma unroll 8
            for (int j=0; j < inf_bs; j += 4) {
                la[tx][ty+j] = A[j*lda];
            }
            A += lda*inf_bs;
            /*
            DPCT1065:1110: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // compute 4 partial sums of each row, i.e.,
            // for ty=0:  res = sum( la[tx, 0: 7] )
            // for ty=1:  res = sum( la[tx, 8:15] )
            // for ty=2:  res = sum( la[tx,16:23] )
            // for ty=3:  res = sum( la[tx,24:31] )
            #pragma unroll 8             
            for (int j=ty*8; j < ty*8 + 8; j++) {
                res += MAGMA_Z_ABS( la[tx][j] );
            }
            /*
            DPCT1065:1111: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
        
        // ----------
        // load diagonal block
        #pragma unroll 8
        for (int j=0; j < inf_bs; j += 4) {
            la[tx][ty+j] = A[j*lda];
        }
        /*
        DPCT1065:1106: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // copy lower triangle to upper triangle, and
        // make diagonal real (zero imaginary part)
        #pragma unroll 8
        for (int i=ty*8; i < ty*8 + 8; i++) {
            if ( i < tx ) {
                la[i][tx] = la[tx][i];
            }
            #ifdef COMPLEX
            else if ( i == tx ) {
                la[i][i] = MAGMA_Z_MAKE(x()(la[i][i]), 0);
            }
            #endif
        }
        /*
        DPCT1065:1107: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // partial row sums
        #pragma unroll 8
        for (int j=ty*8; j < ty*8 + 8; j++) {
            res += MAGMA_Z_ABS( la[tx][j] );
        }
        /*
        DPCT1065:1108: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // ----------
        // loop over all 32x32 blocks below diagonal block
        A += inf_bs;
        for (int i=diag + inf_bs; i < n - n_mod_bs; i += inf_bs ) {
            // load block (transposed)
            #pragma unroll 8
            for (int j=0; j < inf_bs; j += 4) {
                la[ty+j][tx] = A[j*lda];
            }
            A += inf_bs;
            /*
            DPCT1065:1112: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // partial row sums
            #pragma unroll 8
            for (int j=ty*8; j < ty*8 + 8; j++) {
                res += MAGMA_Z_ABS( la[tx][j] );
            }
            /*
            DPCT1065:1113: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
        
        // ----------
        // last partial block, which is (n_mod_bs by inf_bs)
        if ( n_mod_bs > 0 ) {
            // load block (transposed), with zeros for rows outside matrix
            #pragma unroll 8
            for (int j=0; j < inf_bs; j += 4) {
                if ( tx < n_mod_bs ) {
                    la[ty+j][tx] = A[j*lda];
                }
                else {
                    la[ty+j][tx] = MAGMA_Z_ZERO;
                }
            }
            /*
            DPCT1065:1114: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // partial row sums
            #pragma unroll 8
            for (int j=ty*8; j < ty*8 + 8; j++) {
                res += MAGMA_Z_ABS( la[tx][j] );
            }
            /*
            DPCT1065:1115: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
        
        // ----------
        // 32x4 threads store partial sums into shared memory
        /*
        DPCT1064:1116: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        la[tx][ty] = MAGMA_Z_MAKE(res, 0.);
        /*
        DPCT1065:1109: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // first column of 32x1 threads computes final sum of each row
        if ( ty == 0 ) {
            res = res
                + MAGMA_Z_REAL( la[tx][1] )
                + MAGMA_Z_REAL( la[tx][2] )
                + MAGMA_Z_REAL( la[tx][3] );
            dwork[ind] = res;
        }
    }
    else {
        // ------------------------------
        // Last, partial block row
        // Threads past end of matrix (i.e., ind >= n) are redundantly assigned
        // the last row (n-1). At the end, those results are ignored -- only
        // results for ind < n are saved into dwork.
        if ( tx < n_mod_bs ) {
            A += ind;
        }
        else {
            A += (item_ct1.get_group(2) * inf_bs + n_mod_bs -
                  1); // redundantly do last row
        }
        A += ty * lda;
        
        // ----------
        // loop over all blocks left of the diagonal block
        // each is (n_mod_bs by inf_bs)
        for (int i=0; i < diag; i += inf_bs ) {
            // load block
            #pragma unroll 8
            for (int j=0; j < inf_bs; j += 4) {
                la[tx][ty+j] = A[j*lda];
            }
            A += lda*inf_bs;
            /*
            DPCT1065:1119: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // partial row sums
            #pragma unroll 8
            for (int j=0; j < 8; j++) {
                res += MAGMA_Z_ABS( la[tx][j+ty*8] );
            }
            /*
            DPCT1065:1120: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
        
        // ----------
        // partial diagonal block
        if ( ty == 0 && tx < n_mod_bs ) {
            // sum rows left of diagonal
            for (int j=0; j < tx; j++) {
                res += MAGMA_Z_ABS( *A );
                A += lda;
            }
            // sum diagonal (ignoring imaginary part)
            res += MAGMA_D_ABS( MAGMA_Z_REAL( *A ));
            A += 1;
            // sum column below diagonal
            for (int j=tx+1; j < n_mod_bs; j++) {
                res += MAGMA_Z_ABS( *A );
                A += 1;
            }
        }
        /*
        DPCT1065:1117: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // ----------
        // 32x4 threads store partial sums into shared memory
        la[tx][ty]= MAGMA_Z_MAKE( res, 0. );
        /*
        DPCT1065:1118: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // first column of 32x1 threads computes final sum of each row
        // rows outside matrix are ignored
        if ( ty == 0 && tx < n_mod_bs ) {
            res = res
                + MAGMA_Z_REAL( la[tx][1] )
                + MAGMA_Z_REAL( la[tx][2] )
                + MAGMA_Z_REAL( la[tx][3] );
            dwork[ind] = res;
        }
    }
#endif /* (PRECISION_s || PRECISION_d || PRECISION_c || __CUDA_ARCH__ >= 200 || defined(MAGMA_HAVE_HIP)) */
}


/******************************************************************************/
/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:n-1, for || A ||_inf,
 * where n is any size and A is stored upper.
 * Has ceil( n / inf_bs ) blocks of (inf_bs x 4) threads each (inf_bs=32).
 * z precision uses > 16 KB shared memory, so requires Fermi (arch >= 200).
 * The upper implementation is similar to lower, but processes blocks
 * in the transposed order:
 * lower goes from left over to diagonal, then down to bottom;
 * upper goes from top  down to diagonal, then over to right.
 * Differences are noted with # in comments. */
void
zlanhe_inf_kernel_upper(
    int n,
    const magmaDoubleComplex * __restrict__ A, int lda,
    double * __restrict__ dwork,
    int n_full_block, int n_mod_bs , sycl::nd_item<3> item_ct1,
    sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write, sycl::access::target::local> la)
{
#if (defined(PRECISION_s) || defined(PRECISION_d) || defined(PRECISION_c) ||   \
     DPCT_COMPATIBILITY_TEMP >= 200) ||                                        \
    defined(MAGMA_HAVE_HIP)
    int tx = item_ct1.get_local_id(2);
    int ty = item_ct1.get_local_id(1);

    int diag = item_ct1.get_group(2) * inf_bs;
    int ind = item_ct1.get_group(2) * inf_bs + tx;

    double res = 0.;

    if (item_ct1.get_group(2) < n_full_block) {
        // ------------------------------
        // All full block #columns
        A += item_ct1.get_group(2) * inf_bs * lda + tx; //#
        A += ty * lda;
        
        // ----------
        // loop over all blocks #above the diagonal block
        for (int i=0; i < diag; i += inf_bs ) {
            // 32x4 threads cooperatively load 32x32 block (#transposed)
            #pragma unroll 8
            for (int j=0; j < inf_bs; j += 4) {
                la[ty+j][tx] = A[j*lda];               //#
            }
            A += inf_bs;                               //#
            /*
            DPCT1065:1125: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // compute 4 partial sums of each row, i.e.,
            // for ty=0:  res = sum( la[tx, 0: 7] )
            // for ty=1:  res = sum( la[tx, 8:15] )
            // for ty=2:  res = sum( la[tx,16:23] )
            // for ty=3:  res = sum( la[tx,24:31] )
            #pragma unroll 8             
            for (int j=ty*8; j < ty*8 + 8; j++) {
                res += MAGMA_Z_ABS( la[tx][j] );
            }
            /*
            DPCT1065:1126: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
        
        // ----------
        // load diagonal block
        #pragma unroll 8
        for (int j=0; j < inf_bs; j += 4) {
            la[tx][ty+j] = A[j*lda];
        }
        /*
        DPCT1065:1121: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // copy #upper triangle to #lower triangle, and
        // make diagonal real (zero imaginary part)
        #pragma unroll 8
        for (int i=ty*8; i < ty*8 + 8; i++) {
            if ( i > tx ) {                            //#
                la[i][tx] = la[tx][i];
            }
            #ifdef COMPLEX
            else if ( i == tx ) {
                /*
                DPCT1064:1127: Migrated make_cuDoubleComplex call is used in a
                macro definition and is not valid for all macro uses. Adjust the
                code.
                */
                la[i][i] = MAGMA_Z_MAKE(x()(la[i][i]), 0);
            }
            #endif
        }
        /*
        DPCT1065:1122: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // partial row sums
        #pragma unroll 8
        for (int j=ty*8; j < ty*8 + 8; j++) {
            res += MAGMA_Z_ABS( la[tx][j] );
        }
        /*
        DPCT1065:1123: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // ----------
        // loop over all 32x32 blocks #right of diagonal block
        A += inf_bs*lda;                               //#
        for (int i=diag + inf_bs; i < n - n_mod_bs; i += inf_bs ) {
            // load block (#non-transposed)
            #pragma unroll 8
            for (int j=0; j < inf_bs; j += 4) {
                la[tx][ty+j] = A[j*lda];               //#
            }
            A += inf_bs*lda;                           //#
            /*
            DPCT1065:1128: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // partial row sums
            #pragma unroll 8
            for (int j=ty*8; j < ty*8 + 8; j++) {
                res += MAGMA_Z_ABS( la[tx][j] );
            }
            /*
            DPCT1065:1129: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
        
        // ----------
        // last partial block, which is #(inf_bs by n_mod_bs)
        if ( n_mod_bs > 0 ) {
            // load block (#non-transposed), with zeros for #cols outside matrix
            #pragma unroll 8
            for (int j=0; j < inf_bs; j += 4) {
                if ( ty+j < n_mod_bs ) {               //#
                    la[tx][ty+j] = A[j*lda];           //#
                }
                else {
                    /*
                    DPCT1064:1132: Migrated make_cuDoubleComplex call is used in
                    a macro definition and is not valid for all macro uses.
                    Adjust the code.
                    */
                    la[tx][ty + j] = MAGMA_Z_ZERO; //#
                }
            }
            /*
            DPCT1065:1130: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // partial row sums
            #pragma unroll 8
            for (int j=ty*8; j < ty*8 + 8; j++) {
                res += MAGMA_Z_ABS( la[tx][j] );
            }
            /*
            DPCT1065:1131: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
        
        // ----------
        // 32x4 threads store partial sums into shared memory
        /*
        DPCT1064:1133: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        la[tx][ty] = MAGMA_Z_MAKE(res, 0.);
        /*
        DPCT1065:1124: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // first column of 32x1 threads computes final sum of each row
        if ( ty == 0 ) {
            res = res
                + MAGMA_Z_REAL( la[tx][1] )
                + MAGMA_Z_REAL( la[tx][2] )
                + MAGMA_Z_REAL( la[tx][3] );
            dwork[ind] = res;
        }
    }
    else {
        // ------------------------------
        // Last, partial block #column
        // Instead of assigning threads ind >= n to the last row (n-1), as in Lower,
        // Upper simply adjusts loop bounds to avoid loading columns outside the matrix.
        // Again, at the end, those results are ignored -- only
        // results for ind < n are saved into dwork.
        A += item_ct1.get_group(2) * inf_bs * lda + tx; //#
        A += ty * lda;
        
        // ----------
        // loop over all blocks #above the diagonal block
        // each is #(inf_bs by n_mod_bs)
        for (int i=0; i < diag; i += inf_bs ) {
            // load block (#transposed), #ignoring columns outside matrix
            #pragma unroll 8
            for (int j=0; j < inf_bs; j += 4) {
                if ( ty+j < n_mod_bs ) {
                    la[ty+j][tx] = A[j*lda];
                }
            }
            A += inf_bs;                               //#
            /*
            DPCT1065:1136: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();

            // partial row sums
            #pragma unroll 8
            for (int j=0; j < 8; j++) {
                res += MAGMA_Z_ABS( la[tx][j+ty*8] );
            }
            /*
            DPCT1065:1137: Consider replacing sycl::nd_item::barrier() with
            sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
            better performance if there is no access to global memory.
            */
            item_ct1.barrier();
        }
        
        // ----------
        // partial diagonal block
        if ( ty == 0 && tx < n_mod_bs ) {
            // #transpose pointer within diagonal block
            // #i.e., from A = A(tx,ty), transpose to A = A(ty,tx).
            A = A - tx - ty*lda + tx*lda + ty;
            
            // sum #column above diagonal
            for (int j=0; j < tx; j++) {
                res += MAGMA_Z_ABS( *A );
                A += 1;                                //#
            }
            // sum diagonal (ignoring imaginary part)
            res += MAGMA_D_ABS( MAGMA_Z_REAL( *A ));
            A += lda;                                  //#
            // sum #row right of diagonal
            for (int j=tx+1; j < n_mod_bs; j++) {
                res += MAGMA_Z_ABS( *A );
                A += lda;                              //#
            }
        }
        /*
        DPCT1065:1134: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // ----------
        // 32x4 threads store partial sums into shared memory
        /*
        DPCT1064:1138: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        la[tx][ty] = MAGMA_Z_MAKE(res, 0.);
        /*
        DPCT1065:1135: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // first column of 32x1 threads computes final sum of each row
        // rows outside matrix are ignored
        if ( ty == 0 && tx < n_mod_bs ) {
            res = res
                + MAGMA_Z_REAL( la[tx][1] )
                + MAGMA_Z_REAL( la[tx][2] )
                + MAGMA_Z_REAL( la[tx][3] );
            dwork[ind] = res;
        }
    }
#endif /* (PRECISION_s || PRECISION_d || PRECISION_c || __CUDA_ARCH__ >= 200 || defined(MAGMA_HAVE_HIP)) */
}


/******************************************************************************/
/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:n-1, for || A ||_inf */
extern "C" void
zlanhe_inf(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_const_ptr A, magma_int_t lda,
    magmaDouble_ptr dwork,
    magma_queue_t queue )
{
    sycl::range<3> threads(1, 4, inf_bs);
    sycl::range<3> grid(1, 1, magma_ceildiv(n, inf_bs));

    magma_int_t n_full_block = (n - n % inf_bs) / inf_bs;
    magma_int_t n_mod_bs = n % inf_bs;
    if ( uplo == MagmaLower) {
        /*
        DPCT1049:1139: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 2,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    la_acc_ct1(sycl::range<2>(32 /*inf_bs*/, 33 /*inf_bs+1*/),
                               cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zlanhe_inf_kernel_lower(
                                         n, A, lda, dwork, n_full_block,
                                         n_mod_bs, item_ct1, la_acc_ct1);
                                 });
            });
    }
    else {
        /*
        DPCT1049:1140: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 2,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    la_acc_ct1(sycl::range<2>(32 /*inf_bs*/, 33 /*inf_bs+1*/),
                               cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zlanhe_inf_kernel_upper(
                                         n, A, lda, dwork, n_full_block,
                                         n_mod_bs, item_ct1, la_acc_ct1);
                                 });
            });
    }
}


// =============================================================================
// max-norm

/******************************************************************************/
/* Computes dwork[i] = max( abs( A(i,0:i) )), i=0:n-1, for ||A||_max, where A is stored lower */
void
zlanhe_max_kernel_lower(
    int n,
    const magmaDoubleComplex * __restrict__ A, int lda,
    double * __restrict__ dwork , sycl::nd_item<3> item_ct1)
{
    int ind = item_ct1.get_group(2) * max_bs + item_ct1.get_local_id(2);
    double res = 0;

    if (ind < n) {
        A += ind;
        for (int j=0; j < ind; ++j) {
            res = max_nan( res, MAGMA_Z_ABS( *A ));
            A += lda;
        }
        // diagonal element (ignoring imaginary part)
        res = max_nan( res, MAGMA_D_ABS( MAGMA_Z_REAL( *A )));
        dwork[ind] = res;
    }
}


/******************************************************************************/
/* Computes dwork[i] = max( abs( A(i,0:i) )), i=0:n-1, for ||A||_max, where A is stored upper. */
void
zlanhe_max_kernel_upper(
    int n,
    const magmaDoubleComplex * __restrict__ A, int lda,
    double * __restrict__ dwork , sycl::nd_item<3> item_ct1)
{
    int ind = item_ct1.get_group(2) * max_bs + item_ct1.get_local_id(2);
    double res = 0;

    if (ind < n) {
        A += ind;
        A += (n-1)*lda;
        for (int j=n-1; j > ind; j--) {
            res = max_nan( res, MAGMA_Z_ABS( *A ));
            A -= lda;
        }
        // diagonal element (ignoring imaginary part)
        res = max_nan( res, MAGMA_D_ABS( MAGMA_Z_REAL( *A )));
        dwork[ind] = res;
    }
}


/******************************************************************************/
/* Computes dwork[i] = max( abs( A(i,:) )), i=0:n-1, for ||A||_max */
extern "C" void
zlanhe_max(
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_const_ptr A, magma_int_t lda,
    magmaDouble_ptr dwork,
    magma_queue_t queue )
{
    sycl::range<3> threads(1, 1, max_bs);
    sycl::range<3> grid(1, 1, magma_ceildiv(n, max_bs));

    if ( uplo == MagmaLower ) {
        /*
        DPCT1049:1141: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zlanhe_max_kernel_lower(n, A, lda, dwork,
                                                       item_ct1);
                           });
    }
    else {
        /*
        DPCT1049:1142: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->cuda_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zlanhe_max_kernel_upper(n, A, lda, dwork,
                                                       item_ct1);
                           });
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZLANHE returns the value of the one norm, or the Frobenius norm, or
    the infinity norm, or the element of largest absolute value of a
    complex Hermitian matrix A.
    
       ZLANHE = ( max(abs(A(i,j))), NORM = MagmaMaxNorm
                (
                ( norm1(A),         NORM = MagmaOneNorm
                (
                ( normI(A),         NORM = MagmaInfNorm
                (
                ( normF(A),         NORM = MagmaFrobeniusNorm ** not yet supported
    
    where norm1 denotes the one norm of a matrix (maximum column sum),
    normI denotes the infinity norm of a matrix (maximum row sum) and
    normF denotes the Frobenius norm of a matrix (square root of sum of squares).
    Note that max(abs(A(i,j))) is not a consistent matrix norm.
    
    On error, returns ZLANHE < 0: if ZLANHE = -i, the i-th argument had an illegal value.
    
    Arguments:
    ----------
    @param[in]
    norm    magma_norm_t
            Specifies the value to be returned in ZLANHE as described above.
    
    @param[in]
    uplo    magma_uplo_t
            Specifies whether the upper or lower triangular part of the
            Hermitian matrix A is to be referenced.
      -     = MagmaUpper: Upper triangular part of A is referenced
      -     = MagmaLower: Lower triangular part of A is referenced
    
    @param[in]
    n       INTEGER
            The order of the matrix A. N >= 0. When N = 0, ZLANHE is
            set to zero.
    
    @param[in]
    dA      COMPLEX*16 array on the GPU, dimension (LDDA,N)
            The Hermitian matrix A. If UPLO = MagmaUpper, the leading n by n
            upper triangular part of A contains the upper triangular part
            of the matrix A, and the strictly lower triangular part of A
            is not referenced. If UPLO = MagmaLower, the leading n by n lower
            triangular part of A contains the lower triangular part of
            the matrix A, and the strictly upper triangular part of A is
            not referenced. Note that the imaginary parts of the diagonal
            elements need not be set and are assumed to be zero.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array A. LDDA >= max(N,1).
    
    @param
    dwork   (workspace) DOUBLE PRECISION array on the GPU, dimension (MAX(1,LWORK)),
            where LWORK >= N.
            NOTE: this is different than LAPACK, where WORK is required
            only for norm1 and normI. Here max-norm also requires WORK.
    
    @param[in]
    lwork   INTEGER
            The dimension of the array DWORK. LWORK >= max( 1, N ).
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_lanhe
*******************************************************************************/
extern "C" double
magmablas_zlanhe(
    magma_norm_t norm, magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    // 1-norm == inf-norm since A is Hermitian
    bool inf_norm = (norm == MagmaInfNorm || norm == MagmaOneNorm);
    bool max_norm = (norm == MagmaMaxNorm);
    
    // inf_norm Double-Complex requires > 16 KB shared data (arch >= 200)
    #if defined(PRECISION_z)
    const bool inf_implemented = (magma_getdevice_arch() >= 200);
    #else
    const bool inf_implemented = true;
    #endif
    
    if ( ! (max_norm || (inf_norm && inf_implemented)) )
        info = -1;
    else if ( uplo != MagmaUpper && uplo != MagmaLower )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < n )
        info = -5;
    else if ( lwork < n )
        info = -7;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    /* Quick return */
    if ( n == 0 )
        return 0;
        
    double res = 0;
    if ( inf_norm ) {
        zlanhe_inf( uplo, n, dA, ldda, dwork, queue );
    }
    else {
        zlanhe_max( uplo, n, dA, ldda, dwork, queue );
    }
    /*
    DPCT1049:1143: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->cuda_stream()))->submit([&](sycl::handler &cgh) {
        sycl::accessor<double, 1, sycl::access_mode::read_write,
                       sycl::access::target::local>
            smax_acc_ct1(sycl::range<1>(512), cgh);

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 512),
                                           sycl::range<3>(1, 1, 512)),
                         [=](sycl::nd_item<3> item_ct1) {
                             magma_max_nan_kernel(
                                 n, dwork, item_ct1,
                                 (double *)smax_acc_ct1.get_pointer());
                         });
    });
    magma_dgetvector( 1, &dwork[0], 1, &res, 1, queue );
    
    return res;
}
