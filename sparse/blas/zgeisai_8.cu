/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#include "magmasparse_internal.h"
#include "shuffle.cuh"

#define PRECISION_z
#define COMPLEX
#define BLOCKSIZE 8
#define WARP_SIZE 8
#define WRP 8
#define WRQ 4


#include <cuda.h>  // for CUDA_VERSION

#if (CUDA_VERSION >= 7000)

__device__
void ztrsv_lower_8kernel_general(magmaDoubleComplex *dA, magmaDoubleComplex *dB, int *sizes)
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;

    magmaDoubleComplex rB[ 2 ];
    magmaDoubleComplex rA[ 2 ];

    int n;
    int k;
    int N = sizes[j];

    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;


    // Read B to regs.
    #pragma unroll
    for (n = 0; n < 2; n++)
        rB[n] = dB[n*WARP_SIZE+idn];


    // Triangular solve in regs.
    #pragma unroll
    for (k = 0; k < N; k++)
    {
        #pragma unroll
        for (n = 0; n < 2; n++)
            rA[n] = dA[k*WARP_SIZE+n*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB[k/WARP_SIZE] /= rA[k/WARP_SIZE];

        magmaDoubleComplex top = magmablas_zshfl(rB[k/WARP_SIZE], k%WARP_SIZE);

        #pragma unroll
        for (n = 0; n < 2; n++)
            if (n*WARP_SIZE+idn > k)
                rB[n] -= (top*rA[n]);
    }
    // Drop B to dev mem.
    #pragma unroll
    for (n = 0; n < 2; n++)
        if (n*WARP_SIZE+idn < N)
            dB[n*WARP_SIZE+idn] = rB[n];

#endif
}


__device__
void ztrsv_upper_8kernel_general(magmaDoubleComplex *dA, magmaDoubleComplex *dB, int *sizes)
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;

    magmaDoubleComplex rB[ 2 ];
    magmaDoubleComplex rA[ 2 ];

    int n;
    int N = sizes[j];

    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;


    // Read B to regs.
    #pragma unroll
    for (n = 0; n < 2; n++)
        rB[n] = dB[n*WARP_SIZE+idn];


    // Triangular solve in regs.
    #pragma unroll
    for (int k = N-1; k > -1; k--)
    {
        #pragma unroll
        for (n = 0; n < 2; n++)
            rA[n] = dA[k*WARP_SIZE+n*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB[k/WARP_SIZE] /= rA[k/WARP_SIZE];

        magmaDoubleComplex top = magmablas_zshfl(rB[k/WARP_SIZE], k%WARP_SIZE);

        #pragma unroll
        for (n = 0; n < 2; n++)
            if (n*WARP_SIZE+idn < k)
                rB[n] -= (top*rA[n]);
    }
    // Drop B to dev mem.
    #pragma unroll
    for (n = 0; n < 2; n++)
        if (n*WARP_SIZE+idn < N)
            dB[n*WARP_SIZE+idn] = rB[n];

#endif
}



__device__
void ztrsv_lower_8kernel_1(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 1; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex top = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ztrsv_lower_8kernel_2(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 2; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex top = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ztrsv_lower_8kernel_3(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 3; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex top = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ztrsv_lower_8kernel_4(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 4; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex top = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ztrsv_lower_8kernel_5(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 5; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex top = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ztrsv_lower_8kernel_6(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 6; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex top = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ztrsv_lower_8kernel_7(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 7; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex top = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ztrsv_lower_8kernel_8(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 0; k < 8; k++)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex top = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn > k)
            rB -= (top*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}

__global__
void ztrsv_lower_8kernel_switch(magmaDoubleComplex *dA, magmaDoubleComplex *dB, int *sizes, int num_rows )
{
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    if (j < num_rows) {
        int N = sizes[j];
        switch( N ) {
            case  1:
                ztrsv_lower_8kernel_1( dA, dB ); break;
            case  2:
                ztrsv_lower_8kernel_2( dA, dB ); break;
            case  3:
                ztrsv_lower_8kernel_3( dA, dB ); break;
            case  4:
                ztrsv_lower_8kernel_4( dA, dB ); break;
            case  5:
                ztrsv_lower_8kernel_5( dA, dB ); break;
            case  6:
                ztrsv_lower_8kernel_6( dA, dB ); break;
            case  7:
                ztrsv_lower_8kernel_7( dA, dB ); break;
            case  8:
                ztrsv_lower_8kernel_8( dA, dB ); break;
            default:
                ztrsv_lower_8kernel_general( dA, dB, sizes ); break;
        }
    }
}
__device__
void ztrsv_upper_8kernel_1(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 1-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex bottom = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ztrsv_upper_8kernel_2(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 2-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex bottom = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ztrsv_upper_8kernel_3(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 3-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex bottom = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ztrsv_upper_8kernel_4(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 4-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex bottom = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ztrsv_upper_8kernel_5(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 5-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex bottom = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ztrsv_upper_8kernel_6(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 6-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex bottom = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ztrsv_upper_8kernel_7(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 7-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex bottom = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}



__device__
void ztrsv_upper_8kernel_8(magmaDoubleComplex *dA, magmaDoubleComplex *dB )
{
#if (defined( REAL ) && ( __CUDA_ARCH__ >= 300 ))
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int idn = threadIdx.x;
    magmaDoubleComplex rB;
    magmaDoubleComplex rA;
    dA += (j)*WARP_SIZE*WARP_SIZE;
    dB += (j)*WARP_SIZE;

    // Read B to regs.
    rB = dB[idn];
    // Triangular solve in regs.
    #pragma unroll
    for (int k = 8-1; k >-1; k--)
    {
        rA = dA[k*WARP_SIZE+idn];
        if (k%WARP_SIZE == idn)
            rB /= rA;
        magmaDoubleComplex bottom = magmablas_zshfl(rB, k%WARP_SIZE);
        if ( idn < k)
            rB -= (bottom*rA);
    }
    // Drop B to dev mem.
    dB[idn] = rB;
#endif
}

__global__
void ztrsv_upper_8kernel_switch(magmaDoubleComplex *dA, magmaDoubleComplex *dB, int *sizes, int num_rows )
{
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    if (j < num_rows) {
        int N = sizes[j];
        switch( N ) {
            case  1:
                ztrsv_upper_8kernel_1( dA, dB ); break;
            case  2:
                ztrsv_upper_8kernel_2( dA, dB ); break;
            case  3:
                ztrsv_upper_8kernel_3( dA, dB ); break;
            case  4:
                ztrsv_upper_8kernel_4( dA, dB ); break;
            case  5:
                ztrsv_upper_8kernel_5( dA, dB ); break;
            case  6:
                ztrsv_upper_8kernel_6( dA, dB ); break;
            case  7:
                ztrsv_upper_8kernel_7( dA, dB ); break;
            case  8:
                ztrsv_upper_8kernel_8( dA, dB ); break;
            default:
                ztrsv_upper_8kernel_general( dA, dB, sizes ); break;
        }
    }
}


// initialize arrays with zero
__global__ void
magma_zgpumemzero_8kernel(
    magmaDoubleComplex * d,
    int n,
    int dim_x,
    int dim_y )
{
    int i = blockIdx.y * gridDim.x + blockIdx.x;
    int idx = threadIdx.x;

    if( i >= n ){
       return;
    }
    if( idx >= dim_x ){
       return;
    }

    for( int j=0; j<dim_y; j++)
        d[ i*dim_x*dim_y + j*dim_y + idx ] = MAGMA_Z_MAKE( 0.0, 0.0 );
}

__global__ void
magma_zlocations_lower_8kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;
    if( i == 0 ){
        sizes[j] = count;
        rhs[ j*WARP_SIZE ] = MAGMA_Z_ONE;
    }

    if ( i<count ){
        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel


__global__ void
magma_zlocations_trunc_lower_8kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;

    // normal case
    if( count <= BLOCKSIZE ){ // normal case
        if( i == 0 ){
            sizes[j] = count;
            rhs[ j*WARP_SIZE ] = MAGMA_Z_ONE;
        }
        if ( i<count ){
            locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
        }
    }
    else {
        // truncate in this row to the blocksize,
        // take only the 8 elements close to the main diagonal into account
        count = BLOCKSIZE;
        if (i == 0) {
            sizes[j] = count;
            rhs[ j*WARP_SIZE ] = MAGMA_Z_ONE;
        }

        locations[ j*WARP_SIZE + i ] = col[ row[j+1]-BLOCKSIZE+i ];
    }
}// kernel



__global__ void
magma_zlocations_upper_8kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;
    if( i == 0 ){
        sizes[j] = count;
        rhs[ j*WARP_SIZE+count-1 ] = MAGMA_Z_ONE;
    }

    if ( i<count ){
        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel

__global__ void
magma_zlocations_trunc_upper_8kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;

    if( j >= n ){
        return;
    }
    int start = row[j];
    int end = row[j+1];
    int count = end-start;

    // normal case
    if( count <= BLOCKSIZE ){ // normal case
        if( i == 0 ){
            sizes[j] = count;
            rhs[ j*WARP_SIZE+count-1 ] = MAGMA_Z_ONE;
        }
        if ( i<count ){
            locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
        }
    }
    else {
        // truncate in this row to the blocksize,
        // take only the 8 elements close to the main diagonal into account
        count = BLOCKSIZE;
        if (i == 0) {
            sizes[j] = count;
            rhs[ j*WARP_SIZE+count-1 ] = MAGMA_Z_ONE;
        }

        locations[ j*WARP_SIZE + i ] = col[ row[j]+i ];
    }
}// kernel

__global__ void
magma_zfilltrisystems_8kernel(
    magma_int_t offset,
    magma_int_t limit,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs )
{
    int i = (blockDim.x * blockIdx.x + threadIdx.x)+offset;
    int ii = (blockDim.x * blockIdx.x + threadIdx.x);

    if ( ii>=limit ){
        return;
    }
    //if ( i<offset ){
    //    return;
    //}

    for( int j=0; j<sizes[ i ]; j++ ){// no need for first
        int k = row[ locations[ j+i*WARP_SIZE ] ];
        int l = i*WARP_SIZE;
        int idx = 0;
        while( k < row[ locations[ j+i*WARP_SIZE ]+1 ] && l < (i+1)*WARP_SIZE ){ // stop once this column is done
            if( locations[ l ] == col[k] ){ //match
                // int loc = i*WARP_SIZE*WARP_SIZE + j*WARP_SIZE + idx;
                trisystems[ ii*WARP_SIZE*WARP_SIZE + j*WARP_SIZE + idx ]
                                                        = val[ k ];
                k++;
                l++;
                idx++;
            } else if( col[k] < locations[ l ] ){// need to check next element
                k++;
            } else { // element does not exist, i.e. l < LC.col[k]
                // printf("increment l\n");
                l++; // check next elment in the sparsity pattern
                idx++; // leave this element equal zero
            }
        }
    }
}// kernel


__global__ void
magma_zbackinsert_8kernel(
    magma_int_t n,
    magma_index_t *row,
    magma_index_t *col,
    magmaDoubleComplex *val,
    magma_index_t *sizes,
    magmaDoubleComplex *rhs )
{
    int i = threadIdx.x;
    int j = blockIdx.y * gridDim.x + blockIdx.x;
    int end = sizes[j];
    if( j >= n ){
        return;
    }

    if ( i>=end ){
        return;
    }

    val[row[j]+i] = rhs[j*WARP_SIZE+i];
}// kernel



#endif

/**
    Purpose
    -------
    This routine is designet to combine all kernels into one.

    Arguments
    ---------


    @param[in]
    uplotype    magma_uplo_t
                lower or upper triangular

    @param[in]
    transtype   magma_trans_t
                possibility for transposed matrix

    @param[in]
    diagtype    magma_diag_t
                unit diagonal or not

    @param[in]
    L           magma_z_matrix
                triangular factor for which the ISAI matrix is computed.
                Col-Major CSR storage.

    @param[in,out]
    M           magma_z_matrix*
                SPAI preconditioner CSR col-major

    @param[out]
    sizes       magma_int_t*
                Number of Elements that are replaced.

    @param[out]
    locations   magma_int_t*
                Array indicating the locations.

    @param[out]
    trisystems  magmaDoubleComplex*
                trisystems

    @param[out]
    rhs         magmaDoubleComplex*
                right-hand sides

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zisaigenerator_8_gpu(
    magma_uplo_t uplotype,
    magma_trans_t transtype,
    magma_diag_t diagtype,
    magma_z_matrix L,
    magma_z_matrix *M,
    magma_index_t *sizes,
    magma_index_t *locations,
    magmaDoubleComplex *trisystems,
    magmaDoubleComplex *rhs,
    magma_queue_t queue )
{
    magma_int_t info = 0;

#if (CUDA_VERSION >= 7000)
    magma_int_t arch = magma_getdevice_arch();

    cudaDeviceSetCacheConfig( cudaFuncCachePreferL1 );


    // routine 1
    int r1bs1 = WARP_SIZE;
    int r1bs2 = 1;
    int r1dg1 = min( int( sqrt( double( M->num_rows ))), 65535 );
    int r1dg2 = min(magma_ceildiv( M->num_rows, r1dg1 ), 65535);
    int r1dg3 = magma_ceildiv( M->num_rows, r1dg1*r1dg2 );

    dim3 r1block( r1bs1, r1bs2, 1 );
    dim3 r1grid( r1dg1, r1dg2, r1dg3 );

    int r2bs1 = WARP_SIZE;
    int r2bs2 = 1;
    int r2dg1 = magma_ceildiv( L.num_rows, r2bs1 );
    int r2dg2 = 1;
    int r2dg3 = 1;
    dim3 r2block( r2bs1, r2bs2, 1 );
    dim3 r2grid( r2dg1, r2dg2, r2dg3 );

    int r3bs1 = WARP_SIZE;
    int r3bs2 = 1;
    int r3dg1 = magma_ceildiv( 32000, r2bs1 );
    int r3dg2 = 1;
    int r3dg3 = 1;
    dim3 r3block( r3bs1, r3bs2, 1 );
    dim3 r3grid( r3dg1, r3dg2, r3dg3 );

    int recursive = magma_ceildiv( M->num_rows, 32000 );

    if (arch >= 300) {
        magma_zgpumemzero_8kernel<<< r1grid, r1block, 0, queue->cuda_stream() >>>(
                rhs, L.num_rows, WARP_SIZE, 1);

        if (uplotype == MagmaLower) {
            magma_zlocations_lower_8kernel<<< r1grid, r1block, 0, queue->cuda_stream() >>>(
                            M->num_rows,
                            M->drow,
                            M->dcol,
                            M->dval,
                            sizes,
                            locations,
                            trisystems,
                            rhs );
        }
        else {
            magma_zlocations_upper_8kernel<<< r1grid, r1block, 0, queue->cuda_stream() >>>(
                            M->num_rows,
                            M->drow,
                            M->dcol,
                            M->dval,
                            sizes,
                            locations,
                            trisystems,
                            rhs );
        }

        // chunk it recursively into batches of 800
        for( int z=0; z<recursive; z++ ){
            int limit = min(32000, L.num_rows-32000*z);

            magma_zgpumemzero_8kernel<<< r1grid, r1block, 0, queue->cuda_stream() >>>(
                trisystems, limit, WARP_SIZE, WARP_SIZE );

            magma_zfilltrisystems_8kernel<<< r3grid, r3block, 0, queue->cuda_stream() >>>(
                                32000*z,
                                limit,
                                L.drow,
                                L.dcol,
                                L.dval,
                                sizes,
                                locations,
                                trisystems,
                                rhs );

            // routine 2
            if (uplotype == MagmaLower) {
                ztrsv_lower_8kernel_switch<<< r1grid, r1block, 0, queue->cuda_stream() >>>(
                        trisystems,
                        rhs+32000*8*z,
                        sizes+32000*z,
                        limit );
            }
            else {
                ztrsv_upper_8kernel_switch<<< r1grid, r1block, 0, queue->cuda_stream() >>>(
                        trisystems,
                        rhs+32000*8*z,
                        sizes+32000*z,
                        limit );
            }
        }

        // routine 3
        magma_zbackinsert_8kernel<<< r1grid, r1block, 0, queue->cuda_stream() >>>(
                M->num_rows,
                M->drow,
                M->dcol,
                M->dval,
                sizes,
                rhs );
    }
    else {
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
#else
    // CUDA < 7000
    printf( "%% error: ISAI preconditioner requires CUDA > 7.0.\n" );
    info = MAGMA_ERR_NOT_SUPPORTED;
#endif

    return info;
}
