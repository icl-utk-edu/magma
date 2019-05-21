/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
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

#define     h_A(i_, j_) (h_A     + (i_) + (j_)*lda)
#define h_dinvA(i_, j_) (h_dinvA + (i_) + (j_)*nb)


/* ////////////////////////////////////////////////////////////////////////////
   -- like axpy for matrices: B += alpha*A.
*/
void zgeadd(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    const magmaDoubleComplex *A, magma_int_t lda,
    magmaDoubleComplex       *B, magma_int_t ldb )
{
    #define A(i_, j_) (A + (i_) + (j_)*lda)
    #define B(i_, j_) (B + (i_) + (j_)*ldb)
    
    const magma_int_t ione = 1;
    magma_int_t j;
    
    for( j=0; j < n; ++j ) {
        blasf77_zaxpy( &m, &alpha, A(0,j), &ione, B(0,j), &ione );
    }
}


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing ztrtri
*/
int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time=0;  //, cpu_perf=0, cpu_time=0;
    double          magma_error, norm_invA, work[1];
    magma_int_t i, j, N, lda, ldda, info;
    magma_int_t jb, nb, nblock, size_inv;
    magma_int_t *ipiv;

    magmaDoubleComplex *h_A, *h_dinvA;
    magmaDoubleComplex_ptr d_A, d_dinvA;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    int status = 0;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    opts.lapack |= opts.check;  // check (-c) implies lapack (-l)
    
    double tol = opts.tolerance * lapackf77_dlamch("E");
    const char *uplo_ = lapack_uplo_const(opts.uplo);

    // this is the NB hard coded into ztrtri_diag.
    nb = 128;
    
    printf("%% uplo = %s, diag = %s\n",
           lapack_uplo_const(opts.uplo), lapack_diag_const(opts.diag) );
    printf("%%   N  MAGMA Gflop/s (ms)   MAGMA error\n");
    printf("%%======================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda = N;
            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            nblock = magma_ceildiv( N, nb );
            gflops = nblock * FLOPS_ZTRTRI( nb ) / 1e9;
            
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,    lda*N ));
            TESTING_CHECK( magma_imalloc_cpu( &ipiv,   N     ));
            
            size_inv = nblock*nb*nb;
            TESTING_CHECK( magma_zmalloc( &d_A,    ldda*N ));
            TESTING_CHECK( magma_zmalloc( &d_dinvA, size_inv ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_dinvA, size_inv ));
            
            /* Initialize the matrices */
            /* Factor A into LU to get well-conditioned triangular matrix.
             * Copy L to U, since L seems okay when used with non-unit diagonal
             * (i.e., from U), while U fails when used with unit diagonal. */
            //sizeA = lda*N;
            magma_generate_matrix( opts, N, N, h_A, lda );
            lapackf77_zgetrf( &N, &N, h_A, &lda, ipiv, &info );
            for( j = 0; j < N; ++j ) {
                for( i = 0; i < j; ++i ) {
                    *h_A(i,j) = *h_A(j,i);
                }
            }
            
            /* =====================================================================
               Performs operation using MAGMABLAS
               =================================================================== */
            magma_zsetmatrix( N, N, h_A, lda, d_A, ldda, opts.queue );
            
            magma_time = magma_sync_wtime( opts.queue );
            magmablas_ztrtri_diag( opts.uplo, opts.diag, N, d_A, ldda, d_dinvA, opts.queue );
            magma_time = magma_sync_wtime( opts.queue ) - magma_time;
            magma_perf = gflops / magma_time;
            
            magma_zgetvector( size_inv, d_dinvA, 1, h_dinvA, 1, opts.queue );
            
            if ( opts.verbose ) {
                printf( "A%lld=", (long long) N );
                magma_zprint( N, N, h_A, lda );
                printf( "d_dinvA%lld=", (long long) N );
                magma_zprint( min(N+4, nb), min(N+4, nblock*nb), h_dinvA, nb );
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                //cpu_time = magma_wtime();
                lapackf77_ztrtri(
                    lapack_uplo_const(opts.uplo), lapack_diag_const(opts.diag),
                    &N, h_A, &lda, &info );
                //cpu_time = magma_wtime() - cpu_time;
                //cpu_perf = gflops / cpu_time;
            }
            
            /* =====================================================================
               Check the result
               =================================================================== */
            if ( opts.check ) {
                // |invA - invA_magma| / |invA|, accumulated over all diagonal blocks
                magma_error = 0;
                norm_invA   = 0;
                for( i=0; i < N; i += nb ) {
                    jb = min( nb, N-i );
                    zgeadd( jb, jb, c_neg_one, h_A(i, i), lda, h_dinvA(0, i), nb );
                    magma_error = max( magma_error, lapackf77_zlantr( "M", uplo_, MagmaNonUnitStr, &jb, &jb, h_dinvA(0, i), &nb,  work ));
                    norm_invA   = max( norm_invA,   lapackf77_zlantr( "M", uplo_, MagmaNonUnitStr, &jb, &jb, h_A(i, i),     &lda, work ));
                }
                magma_error /= norm_invA;
                
                // CPU is doing N-by-N inverse, while GPU is doing (N/NB) NB-by-NB inverses.
                // So don't compare performance.
                printf("%5lld   %7.2f (%7.2f)   %8.2e   %s\n",
                        (long long) N,
                        magma_perf,  1000.*magma_time,
                        //cpu_perf,    1000.*cpu_time,
                        magma_error,
                        (magma_error < tol ? "ok" : "failed"));
                status += ! (magma_error < tol);
            }
            else {
                printf("%5lld   %7.2f (%7.2f)      ---\n",
                        (long long) N,
                        magma_perf,  1000.*magma_time );
            }
            
            magma_free_cpu( h_A     );
            magma_free_cpu( ipiv    );
            
            magma_free( d_A     );
            magma_free( d_dinvA );
            magma_free_cpu( h_dinvA );
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
