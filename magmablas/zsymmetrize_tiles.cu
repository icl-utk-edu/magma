/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Mark Gates
*/
#include "magma_internal.h"

#define NB 64

/*
    Symmetrizes ntile tiles at a time, e.g., all diagonal tiles of a matrix.
    Grid is ceil(m/NB) x ntile.
    Each tile is m x m, and is divided into block rows, each NB x m.
    Each block has NB threads.
    Each thread copies one row, iterating across all columns below diagonal.
    The bottom block of rows may be partially outside the matrix;
    if so, rows outside the matrix (i >= m) are disabled.
*/
__global__ void
zsymmetrize_tiles_lower( int m, magmaDoubleComplex *dA, int ldda, int mstride, int nstride )
{
    // shift dA to tile's top-left corner
    dA += blockIdx.y*(mstride + nstride*ldda);
    
    // dA iterates across row i and dAT iterates down column i.
    int i = blockIdx.x*NB + threadIdx.x;
    magmaDoubleComplex *dAT = dA;
    if ( i < m ) {
        dA  += i;
        dAT += i*ldda;
        magmaDoubleComplex *dAend = dA + i*ldda;  // end at diagonal dA(i,i)
        while( dA < dAend ) {
            *dAT = MAGMA_Z_CONJ(*dA);  // upper := lower
            dA  += ldda;
            dAT += 1;
        }
        *dA = MAGMA_Z_MAKE( MAGMA_Z_REAL(*dA), 0 );  // make diagonal real
    }
}


// only difference with _lower version is direction dA=dAT instead of dAT=dA.
__global__ void
zsymmetrize_tiles_upper( int m, magmaDoubleComplex *dA, int ldda, int mstride, int nstride )
{
    // shift dA to tile's top-left corner
    dA += blockIdx.y*(mstride + nstride*ldda);
    
    // dA iterates across row i and dAT iterates down column i.
    int i = blockIdx.x*NB + threadIdx.x;
    magmaDoubleComplex *dAT = dA;
    if ( i < m ) {
        dA  += i;
        dAT += i*ldda;
        magmaDoubleComplex *dAend = dA + i*ldda;  // end at diagonal dA(i,i)
        while( dA < dAend ) {
            *dA  = MAGMA_Z_CONJ(*dAT);  // lower := upper
            dA  += ldda;
            dAT += 1;
        }
        *dA = MAGMA_Z_MAKE( MAGMA_Z_REAL(*dA), 0 );  // make diagonal real
    }
}


/***************************************************************************//**
    Purpose
    -------
    
    ZSYMMETRIZE_TILES copies lower triangle to upper triangle, or vice-versa,
    to make some blocks of dA into general representations of a symmetric block.
    In Complex, it sets the diagonal to be Real.

    This processes NTILE blocks, typically the diagonal blocks.
    Each block is offset by mstride rows and nstride columns from the previous block.
    
    Arguments
    ---------
    
    @param[in]
    uplo    magma_uplo_t
            Specifies the part of the matrix dA that is valid on input.
      -     = MagmaUpper:      Upper triangular part
      -     = MagmaLower:      Lower triangular part
    
    @param[in]
    m       INTEGER
            The number of rows & columns of each square block of dA.  M >= 0.
    
    @param[in,out]
    dA      COMPLEX_16 array, dimension (LDDA,N)
            The matrix dA. N = m + nstride*(ntile-1).
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= max(1, m + mstride*(ntile-1)).
    
    @param[in]
    ntile   INTEGER
            Number of blocks to symmetrize. ntile >= 0.
    
    @param[in]
    mstride INTEGER
            Row offset from start of one block to start of next block. mstride >= 0.
            Either (mstride >= m) or (nstride >= m), to prevent m-by-m tiles
            from overlapping.
    
    @param[in]
    nstride INTEGER
            Column offset from start of one block to start of next block. nstride >= 0.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_symmetrize_batched
*******************************************************************************/
extern "C" void
magmablas_zsymmetrize_tiles(
    magma_uplo_t uplo, magma_int_t m,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_int_t ntile, magma_int_t mstride, magma_int_t nstride,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( uplo != MagmaLower && uplo != MagmaUpper )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( ldda < max(1,m + mstride*(ntile-1)) )
        info = -5;
    else if ( ntile < 0 )
        info = -6;
    else if ( mstride < 0 )
        info = -7;
    else if ( nstride < 0 )
        info = -8;
    else if ( mstride < m && nstride < m )  // only one must be >= m.
        info = -7;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;
    }
    
    if ( m == 0 || ntile == 0 )
        return;
    
    dim3 threads( NB, 1 );
    dim3 grid( magma_ceildiv( m, NB ), ntile );
    
    //printf( "m %d, grid %d x %d, threads %d\n", m, grid.x, grid.y, threads.x );
    if ( uplo == MagmaUpper ) {
        zsymmetrize_tiles_upper
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, dA, ldda, mstride, nstride );
    }
    else {
        zsymmetrize_tiles_lower
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m, dA, ldda, mstride, nstride );
    }
}
