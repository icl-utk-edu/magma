// This is a simple standalone example. See README.txt

#include <stdio.h>
#include <stdlib.h>

#include "cublas_v2.h"
#include "magma_v2.h"      // also includes cublas_v2.h
#include "magma_lapack.h"  // if you need BLAS & LAPACK


// ------------------------------------------------------------
// Replace with your code to initialize the A matrix.
// This simply initializes it to random values.
// Note that A is stored column-wise, not row-wise.
//
// m   - number of rows,    m >= 0.
// n   - number of columns, n >= 0.
// A   - m-by-n array of size lda*n.
// lda - leading dimension of A, lda >= m.
//
// When lda > m, rows (m, ..., lda-1) below the bottom of the matrix are ignored.
// This is helpful for working with sub-matrices, and for aligning the top
// of columns to memory boundaries (or avoiding such alignment).
// Significantly better memory performance is achieved by having the outer loop
// over columns (j), and the inner loop over rows (i), than the reverse.
void zfill_matrix(
    magma_int_t m, magma_int_t n, magmaDoubleComplex *A, magma_int_t lda )
{
    #define A(i_, j_) A[ (i_) + (j_)*lda ]
    
    magma_int_t i, j;
    for (j=0; j < n; ++j) {
        for (i=0; i < m; ++i) {
            A(i,j) = MAGMA_Z_MAKE( rand() / ((double) RAND_MAX),    // real part
                                   rand() / ((double) RAND_MAX) );  // imag part
        }
    }
    
    #undef A
}


// ------------------------------------------------------------
// Replace with your code to initialize the X rhs.
void zfill_rhs(
    magma_int_t m, magma_int_t nrhs, magmaDoubleComplex *X, magma_int_t ldx )
{
    zfill_matrix( m, nrhs, X, ldx );
}


// ------------------------------------------------------------
// Replace with your code to initialize the dA matrix on the GPU device.
// This simply leverages the CPU version above to initialize it to random values,
// and copies the matrix to the GPU.
void zfill_matrix_gpu(
    magma_int_t m, magma_int_t n, magmaDoubleComplex *dA, magma_int_t ldda,
    magma_queue_t queue )
{
    magmaDoubleComplex *A;
    magma_int_t lda = ldda;
    magma_zmalloc_cpu( &A, m*lda );
    if (A == NULL) {
        fprintf( stderr, "malloc failed\n" );
        return;
    }
    zfill_matrix( m, n, A, lda );
    magma_zsetmatrix( m, n, A, lda, dA, ldda, queue );
    magma_free_cpu( A );
}


// ------------------------------------------------------------
// Replace with your code to initialize the dX rhs on the GPU device.
void zfill_rhs_gpu(
    magma_int_t m, magma_int_t nrhs, magmaDoubleComplex *dX, magma_int_t lddx,
    magma_queue_t queue )
{
    zfill_matrix_gpu( m, nrhs, dX, lddx, queue );
}


// ------------------------------------------------------------
// Solve A * X = B, where A and X are stored in CPU host memory.
// Internally, MAGMA transfers data to the GPU device
// and uses a hybrid CPU + GPU algorithm.
void cpu_interface( magma_int_t n, magma_int_t nrhs )
{
    magmaDoubleComplex *A=NULL, *X=NULL;
    magma_int_t *ipiv=NULL;
    magma_int_t lda  = n;
    magma_int_t ldx  = lda;
    magma_int_t info = 0;
    
    // magma_*malloc_cpu routines for CPU memory are type-safe and align to memory boundaries,
    // but you can use malloc or new if you prefer.
    magma_zmalloc_cpu( &A, lda*n );
    magma_zmalloc_cpu( &X, ldx*nrhs );
    magma_imalloc_cpu( &ipiv, n );
    if (A == NULL || X == NULL || ipiv == NULL) {
        fprintf( stderr, "malloc failed\n" );
        goto cleanup;
    }
    
    // Replace these with your code to initialize A and X
    zfill_matrix( n, n, A, lda );
    zfill_rhs( n, nrhs, X, ldx );
    
    magma_zgesv( n, 1, A, lda, ipiv, X, ldx, &info );
    if (info != 0) {
        fprintf( stderr, "magma_zgesv failed with info=%d\n", info );
    }
    
    // TODO: use result in X
    
cleanup:
    magma_free_cpu( A );
    magma_free_cpu( X );
    magma_free_cpu( ipiv );
}


// ------------------------------------------------------------
// Solve dA * dX = dB, where dA and dX are stored in GPU device memory.
// Internally, MAGMA uses a hybrid CPU + GPU algorithm.
void gpu_interface( magma_int_t n, magma_int_t nrhs )
{
    magmaDoubleComplex *dA=NULL, *dX=NULL;
    magma_int_t *ipiv=NULL;
    magma_int_t ldda = magma_roundup( n, 32 );  // round up to multiple of 32 for best GPU performance
    magma_int_t lddx = ldda;
    magma_int_t info = 0;
    magma_queue_t queue=NULL;
    
    // magma_*malloc routines for GPU memory are type-safe,
    // but you can use cudaMalloc if you prefer.
    magma_zmalloc( &dA, ldda*n );
    magma_zmalloc( &dX, lddx*nrhs );
    magma_imalloc_cpu( &ipiv, n );  // ipiv always on CPU
    if (dA == NULL || dX == NULL || ipiv == NULL) {
        fprintf( stderr, "malloc failed\n" );
        goto cleanup;
    }
    
    magma_int_t dev = 0;
    magma_queue_create( dev, &queue );
    
    // Replace these with your code to initialize A and X
    zfill_matrix_gpu( n, n, dA, ldda, queue );
    zfill_rhs_gpu( n, nrhs, dX, lddx, queue );
    
    magma_zgesv_gpu( n, 1, dA, ldda, ipiv, dX, lddx, &info );
    if (info != 0) {
        fprintf( stderr, "magma_zgesv_gpu failed with info=%d\n", info );
    }
    
    // TODO: use result in dX
    
cleanup:
    magma_queue_destroy( queue );
    magma_free( dA );
    magma_free( dX );
    magma_free_cpu( ipiv );
}


// ------------------------------------------------------------
int main( int argc, char** argv )
{
    magma_init();
    
    magma_int_t n = 1000;
    magma_int_t nrhs = 1;
    
    printf( "using MAGMA CPU interface\n" );
    cpu_interface( n, nrhs );

    printf( "using MAGMA GPU interface\n" );
    gpu_interface( n, nrhs );
    
    magma_finalize();
    return 0;
}
