/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Mark Gates
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"


// Initialize matrix to random.
// Having ensures the same ISEED is always used,
// so we can re-generate the identical matrix.
void init_matrix(
    magma_opts &opts,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda )
{
    magma_int_t iseed_save[4];
    for (magma_int_t i = 0; i < 4; ++i) {
        iseed_save[i] = opts.iseed[i];
    }

    magma_generate_matrix( opts, m, n, A, lda );

    // restore iseed
    for (magma_int_t i = 0; i < 4; ++i) {
        opts.iseed[i] = iseed_save[i];
    }
}


// On input, A and ipiv is LU factorization of A. On output, A is overwritten.
// Requires m == n.
// Uses init_matrix() to re-generate original A as needed.
// Generates random RHS b and solves Ax=b.
// Returns residual, |Ax - b| / (n |A| |x|).
double get_residual(
    magma_opts &opts,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex *A, magma_int_t lda,
    magma_int_t *ipiv )
{
    if ( m != n ) {
        printf( "\nERROR: residual check defined only for square matrices\n" );
        return -1;
    }
    
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magma_int_t ione = 1;
    
    // this seed should be DIFFERENT than used in init_matrix
    // (else x is column of A, so residual can be exactly zero)
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t info = 0;
    magmaDoubleComplex *x, *b;
    
    // initialize RHS
    TESTING_CHECK( magma_zmalloc_cpu( &x, n ));
    TESTING_CHECK( magma_zmalloc_cpu( &b, n ));
    lapackf77_zlarnv( &ione, ISEED, &n, b );
    blasf77_zcopy( &n, b, &ione, x, &ione );
    
    // solve Ax = b
    lapackf77_zgetrs( "Notrans", &n, &ione, A, &lda, ipiv, x, &n, &info );
    if (info != 0) {
        printf("lapackf77_zgetrs returned error %lld: %s.\n",
               (long long) info, magma_strerror( info ));
    }
    
    // reset to original A
    init_matrix( opts, m, n, A, lda );
    
    // compute r = Ax - b, saved in b
    blasf77_zgemv( "Notrans", &m, &n, &c_one, A, &lda, x, &ione, &c_neg_one, b, &ione );
    
    // compute residual |Ax - b| / (n*|A|*|x|)
    double norm_x, norm_A, norm_r, work[1];
    norm_A = lapackf77_zlange( "F", &m, &n, A, &lda, work );
    norm_r = lapackf77_zlange( "F", &n, &ione, b, &n, work );
    norm_x = lapackf77_zlange( "F", &n, &ione, x, &n, work );
    
    //printf( "r=\n" ); magma_zprint( 1, n, b, 1 );
    
    magma_free_cpu( x );
    magma_free_cpu( b );
    
    //printf( "r=%.2e, A=%.2e, x=%.2e, n=%lld\n", norm_r, norm_A, norm_x, (long long) n );
    return norm_r / (n * norm_A * norm_x);
}


// On input, LU and ipiv is LU factorization of A. On output, LU is overwritten.
// Works for any m, n.
// Uses init_matrix() to re-generate original A as needed.
// Returns error in factorization, |PA - LU| / (n |A|)
// This allocates 3 more matrices to store A, L, and U.
double get_LU_error(
    magma_opts &opts,
    magma_int_t M, magma_int_t N,
    magmaDoubleComplex *LU, magma_int_t lda,
    magma_int_t *ipiv)
{
    magma_int_t min_mn = min(M,N);
    magma_int_t ione   = 1;
    magma_int_t i, j;
    magmaDoubleComplex alpha = MAGMA_Z_ONE;
    magmaDoubleComplex beta  = MAGMA_Z_ZERO;
    magmaDoubleComplex *A, *L, *U;
    double work[1], matnorm, residual;
    
    TESTING_CHECK( magma_zmalloc_cpu( &A, lda*N    ));
    TESTING_CHECK( magma_zmalloc_cpu( &L, M*min_mn ));
    TESTING_CHECK( magma_zmalloc_cpu( &U, min_mn*N ));
    memset( L, 0, M*min_mn*sizeof(magmaDoubleComplex) );
    memset( U, 0, min_mn*N*sizeof(magmaDoubleComplex) );

    // set to original A
    init_matrix( opts, M, N, A, lda );
    lapackf77_zlaswp( &N, A, &lda, &ione, &min_mn, ipiv, &ione);
    
    // copy LU to L and U, and set diagonal to 1
    lapackf77_zlacpy( MagmaLowerStr, &M, &min_mn, LU, &lda, L, &M      );
    lapackf77_zlacpy( MagmaUpperStr, &min_mn, &N, LU, &lda, U, &min_mn );
    for (j=0; j < min_mn; j++)
        L[j+j*M] = MAGMA_Z_MAKE( 1., 0. );
    
    matnorm = lapackf77_zlange("f", &M, &N, A, &lda, work);

    blasf77_zgemm("N", "N", &M, &N, &min_mn,
                  &alpha, L, &M, U, &min_mn, &beta, LU, &lda);

    for( j = 0; j < N; j++ ) {
        for( i = 0; i < M; i++ ) {
            LU[i+j*lda] = MAGMA_Z_SUB( LU[i+j*lda], A[i+j*lda] );
        }
    }
    residual = lapackf77_zlange("f", &M, &N, LU, &lda, work);

    magma_free_cpu( A );
    magma_free_cpu( L );
    magma_free_cpu( U );

    return residual / (matnorm * N);
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgetrf
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, gpu_perf, gpu_time, cpu_perf=0, cpu_time=0;
    double          error;
    magmaDoubleComplex *h_A;
    magmaDoubleComplex_ptr d_A;
    magma_int_t     *ipiv;
    magma_int_t M, N, n2, lda, ldda, info, min_mn;
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol = opts.tolerance * lapackf77_dlamch("E");
    
    printf("%% version %lld\n", (long long) opts.version );
    if ( opts.check == 2 ) {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |Ax-b|/(N*|A|*|x|)\n");
    }
    else {
        printf("%%   M     N   CPU Gflop/s (sec)   GPU Gflop/s (sec)   |PA-LU|/(N*|A|)\n");
    }
    printf("%%========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            min_mn = min(M, N);
            lda    = M;
            n2     = lda*N;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            gflops = FLOPS_ZGETRF( M, N ) / 1e9;
            
            TESTING_CHECK( magma_imalloc_cpu( &ipiv, min_mn ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,  n2     ));
            TESTING_CHECK( magma_zmalloc( &d_A,  ldda*N ));
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                init_matrix( opts, M, N, h_A, lda );
                
                cpu_time = magma_wtime();
                lapackf77_zgetrf( &M, &N, h_A, &lda, ipiv, &info );
                cpu_time = magma_wtime() - cpu_time;
                cpu_perf = gflops / cpu_time;
                if (info != 0) {
                    printf("lapackf77_zgetrf returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
            }
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            init_matrix( opts, M, N, h_A, lda );
            if ( opts.version == 2 ) {
                // no pivoting versions, so set ipiv to identity
                for (magma_int_t i=0; i < min_mn; ++i ) {
                    ipiv[i] = i+1;
                }
            }
            magma_zsetmatrix( M, N, h_A, lda, d_A, ldda, opts.queue );
            
            gpu_time = magma_wtime();
            if ( opts.version == 1 ) {
                magma_zgetrf_gpu( M, N, d_A, ldda, ipiv, &info);
            }
            else if ( opts.version == 2 ) {
                magma_zgetrf_nopiv_gpu( M, N, d_A, ldda, &info);
            }
            else if ( opts.version == 3 ) {
                magma_zgetrf_native( M, N, d_A, ldda, ipiv, &info);
            }
            gpu_time = magma_wtime() - gpu_time;
            gpu_perf = gflops / gpu_time;
            if (info != 0) {
                printf("magma_zgetrf_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* =====================================================================
               Check the factorization
               =================================================================== */
            if ( opts.lapack ) {
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)",
                       (long long) M, (long long) N, cpu_perf, cpu_time, gpu_perf, gpu_time );
            }
            else {
                printf("%5lld %5lld     ---   (  ---  )   %7.2f (%7.2f)",
                       (long long) M, (long long) N, gpu_perf, gpu_time );
            }
            if ( opts.check == 2 ) {
                magma_zgetmatrix( M, N, d_A, ldda, h_A, lda, opts.queue );
                error = get_residual( opts, M, N, h_A, lda, ipiv );
                printf("   %8.2e   %s\n", error, (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else if ( opts.check ) {
                magma_zgetmatrix( M, N, d_A, ldda, h_A, lda, opts.queue );
                error = get_LU_error( opts, M, N, h_A, lda, ipiv );
                printf("   %8.2e   %s\n", error, (error < tol ? "ok" : "failed"));
                status += ! (error < tol);
            }
            else {
                printf("     ---  \n");
            }
            
            magma_free_cpu( ipiv );
            magma_free_cpu( h_A );
            magma_free( d_A );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
