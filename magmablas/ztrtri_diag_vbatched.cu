/*
    -- MAGMA (version 1.4) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

       @author Peng Du
       @author Tingxing Dong
       @author Mark Gates
       @author Azzam Haidar
       
       File named ztrtri_diag.cu to avoid name conflict with src/ztrtri.o
       in the library. The actual kernels are in ztrtri_lower.cu and ztrtri_upper.cu
*/

#include "magma_internal.h"

#define    TRTRI_BATCHED
#include "ztrtri.cuh"

/***************************************************************************//**
    Purpose
    -------
    ztrtri_diag inverts the NB x NB diagonal blocks of A.

    Arguments
    ----------
    @param[in]
    uplo    magma_uplo_t.
            On entry, uplo specifies whether the matrix A is an upper or
            lower triangular matrix as follows:
      -     = MagmaUpper:  A is an upper triangular matrix.
      -     = MagmaLower:  A is a  lower triangular matrix.

    @param[in]
    diag    magma_diag_t.
            On entry, diag specifies whether or not A is unit triangular
            as follows:
      -     = MagmaUnit:     A is assumed to be unit triangular.
      -     = MagmaNonUnit:  A is not assumed to be unit triangular.

    @param[in]
    nmax    INTEGER.
            maximum value of n.

    @param[in]
    n       INTEGER array, dimension(batchCount)
            On entry, each entry specifies the order of the corresponding matrix A. N >= 0.

    @param[in]
    dA_array      COMPLEX_16 array of dimension ( ldda, n )
            The triangular matrix A.
    \n
            If UPLO = 'U', the leading N-by-N upper triangular part of A
            contains the upper triangular matrix, and the strictly lower
            triangular part of A is not referenced.
    \n
            If UPLO = 'L', the leading N-by-N lower triangular part of A
            contains the lower triangular matrix, and the strictly upper
            triangular part of A is not referenced.
    \n
            If DIAG = 'U', the diagonal elements of A are also not referenced
            and are assumed to be 1.

    @param[in]
    ldda    INTEGER.
            The leading dimension of the array A.  LDDA >= max(1,N).

    @param[out]
    dinvA_array COMPLEX_16 array of dimension (NB, ceil(n/NB)*NB),
            where NB = 128.
            On exit, contains inverses of the NB-by-NB diagonal blocks of A.

    @param[in]
    resetozero INTEGER
               If not zero, each array dinvA will be reset to all zeros 
    
    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_trtri_batched
*******************************************************************************/
extern "C" void
magmablas_ztrtri_diag_vbatched(
    magma_uplo_t uplo, magma_diag_t diag, magma_int_t nmax, magma_int_t *n,
    magmaDoubleComplex const * const *dA_array, magma_int_t *ldda,
    magmaDoubleComplex **dinvA_array, 
    magma_int_t resetozero, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if (uplo != MagmaLower && uplo != MagmaUpper)
        info = -1;
    else if (diag != MagmaNonUnit && diag != MagmaUnit)
        info = -2;
    else if (nmax < 0)
        info = -3;
    //else if (ldda < n)
    //    info = -5;

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info
    }
    
    // allocate temp buffers for dimensions
    magma_int_t *mm, *nn;
    magma_imalloc( &mm, batchCount );
    magma_imalloc( &nn, batchCount );
    
    int nblocks = magma_ceildiv( nmax, IB );

    if ( resetozero ) {
        // roundup dimensions in 'n' and write it to 'mm' : magma_roundup( n, NB ) 
        magma_ivec_roundup( batchCount, n, NB, mm, queue);
        // set vector 'nn' to NB
        magma_ivec_setc( batchCount, nn, NB, queue);
        magma_int_t max_m = magma_roundup( nmax, NB );
        magma_int_t max_n = NB;
        //magmablas_zlaset_batched (MagmaFull, magma_roundup( n, NB ), NB, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dinvA_array, magma_roundup( n, NB ), batchCount, queue);
        magmablas_zlaset_vbatched(MagmaFull, max_m, max_n, mm, nn, MAGMA_Z_ZERO, MAGMA_Z_ZERO, dinvA_array, mm, batchCount, queue);
    }
    // if someone want to use cudamemset he need to set the whole vectors 
    // of initial size otherwise it is a bug and thus need to have dinvA_length 
    // in input parameter and has been tested and was slower.
    //was not the largest size computed by the high API getrf_batched then it is bug and need to use magmablas_zlaset_batched


    if ( uplo == MagmaLower ) {
        // invert diagonal IB x IB inner blocks
        dim3 diaggrid( nblocks, 1, batchCount );  // emulate 3D grid
        ztrtri_diag_lower_kernel_vbatched<<< diaggrid, IB, 0, queue->cuda_stream() >>>( diag, n, dA_array, ldda, dinvA_array );

        // build up NB x NB blocks (assuming IB=16 here):
        // use   16 x 16  blocks to build  32 x 32  blocks,  1 x (1 x npages) grid,  4 x 4 threads;
        // then  32 x 32  blocks to build  64 x 64  blocks,  1 x (2 x npages) grid,  8 x 4 threads;
        // then  64 x 64  blocks to build 128 x 128 blocks,  1 x (4 x npages) grid, 16 x 4 threads;
        // then 128 x 128 blocks to build 256 x 256 blocks,  2 x (8 x npages) grid, 16 x 4 threads.
        for( int jb=IB; jb < NB; jb *= 2 ) {
            int kb = jb*2;
            int npages = magma_ceildiv( nmax, kb );
            dim3 threads( (jb <= 32 ? jb/4 : 16), 4 );
            dim3 grid( jb/(threads.x*threads.y), npages*(jb/16), batchCount );  // emulate 3D grid: NX * (NY*npages), for CUDA ARCH 1.x
            
            //printf( "n %d, jb %d, grid %d x %d (%d x %d)\n", n, jb, grid.x, grid.y, grid.y / npages, npages );
            switch (jb) {
                case 16:
                    triple_zgemm16_part1_lower_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_zgemm16_part2_lower_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
                case 32:
                    triple_zgemm32_part1_lower_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_zgemm32_part2_lower_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
                case 64:
                    triple_zgemm64_part1_lower_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_zgemm64_part2_lower_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
                default:
                    triple_zgemm_above64_part1_lower_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_zgemm_above64_part2_lower_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_zgemm_above64_part3_lower_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
            }
            if ( kb >= nmax ) break;
        }
    }
    else {
        dim3 diaggrid( nblocks, 1, batchCount );  // emulate 3D grid
        ztrtri_diag_upper_kernel_vbatched<<< diaggrid, IB, 0, queue->cuda_stream() >>>( diag, n, dA_array, ldda, dinvA_array );

        // update the inverse up to the size of IB
        for( int jb=IB; jb < NB; jb*=2 ) {
            int kb = jb*2;
            int npages = magma_ceildiv( nmax, kb );
            dim3 threads( (jb <= 32 ? jb/4 : 16), 4 );
            dim3 grid( jb/(threads.x*threads.y), npages*(jb/16), batchCount );  // emulate 3D grid: NX * (NY*npages), for CUDA ARCH 1.x
            
            switch (jb) {
                case 16:
                    triple_zgemm16_part1_upper_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_zgemm16_part2_upper_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
                case 32:
                    triple_zgemm32_part1_upper_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_zgemm32_part2_upper_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
                case 64:
                    triple_zgemm64_part1_upper_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_zgemm64_part2_upper_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
                default:
                    triple_zgemm_above64_part1_upper_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_zgemm_above64_part2_upper_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    triple_zgemm_above64_part3_upper_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>( n, dA_array, ldda, dinvA_array, jb, npages );
                    break;
            }
            if ( kb >= nmax ) break;
        }
    }
    
    // free allocated buffers
    magma_free(mm);
    magma_free(nn);
}
