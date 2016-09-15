/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @precisions mixed zc -> ds
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <assert.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zlat2c and clat2z
*/
int main( int argc, char** argv )
{
    #define  A(i_,j_) ( A + (i_) + (j_)*lda)
    #define SA(i_,j_) (SA + (i_) + (j_)*lda)
    
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    real_Double_t   gbytes, gpu_perf, gpu_time, cpu_perf, cpu_time;
    double error, work[1];
    float serror, swork[1];
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaFloatComplex  s_neg_one = MAGMA_C_NEG_ONE;
    magma_int_t ione = 1;
    magma_int_t n, lda, ldda, size, info;
    magma_int_t ISEED[4] = {0,0,0,1};
    int status = 0;
    magmaFloatComplex   *SA, *SR;
    magmaDoubleComplex   *A,  *R;
    magmaFloatComplex_ptr  dSA;
    magmaDoubleComplex_ptr dA;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    magma_uplo_t uplo[] = { MagmaLower, MagmaUpper };
    
    printf("%% func   uplo     N     CPU GB/s (ms)       GPU GB/s (ms)     ||R||_F\n");
    printf("%%====================================================================\n");
    for( int iuplo = 0; iuplo < 2; ++iuplo ) {
      for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            n = opts.nsize[itest];
            lda  = n;
            ldda = magma_roundup( n, opts.align );  // multiple of 32 by default
            // 0.5*(n+1)*n double-complex loads and 0.5*(n+1)*n single-complex stores (and vice-versa for clat2z)
            gbytes = (real_Double_t) 0.5*(n+1)*n * (sizeof(magmaDoubleComplex) + sizeof(magmaFloatComplex)) / 1e9;
            size = ldda*n;  // ldda >= lda
            
            TESTING_CHECK( magma_cmalloc_cpu( &SA, size ));
            TESTING_CHECK( magma_zmalloc_cpu( &A, size ));
            TESTING_CHECK( magma_cmalloc_cpu( &SR, size ));
            TESTING_CHECK( magma_zmalloc_cpu( &R, size ));
            
            TESTING_CHECK( magma_cmalloc( &dSA, size ));
            TESTING_CHECK( magma_zmalloc( &dA, size ));
            
            lapackf77_zlarnv( &ione, ISEED, &size,  A );
            lapackf77_clarnv( &ione, ISEED, &size, SA );
            
            magma_zsetmatrix( n, n, A,  lda, dA,  ldda, opts.queue );
            magma_csetmatrix( n, n, SA, lda, dSA, ldda, opts.queue );
            
            /* =====================================================================
               Performs operation using LAPACK zlat2c
               =================================================================== */
            info = 0;
            cpu_time = magma_wtime();
            lapackf77_zlat2c( lapack_uplo_const(uplo[iuplo]), &n, A, &lda, SA, &lda, &info );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            if (info != 0) {
                printf("lapackf77_zlat2c returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* ====================================================================
               Performs operation using MAGMA zlat2c
               =================================================================== */
            gpu_time = magma_sync_wtime( opts.queue );
            magmablas_zlat2c( uplo[iuplo], n, dA, ldda, dSA, ldda, opts.queue, &info );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            if (info != 0) {
                printf("magmablas_zlat2c returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            magma_cgetmatrix( n, n, dSA, ldda, SR, lda, opts.queue );
            
            if ( opts.verbose ) {
                printf( "A=  " );  magma_zprint( n, n, A,  lda );
                printf( "SA= " );  magma_cprint( n, n, SA, lda );
                printf( "dA= " );  magma_zprint_gpu( n, n, dA,  ldda, opts.queue );
                printf( "dSA=" );  magma_cprint_gpu( n, n, dSA, ldda, opts.queue );
            }
            
            /* =====================================================================
               compute error |SA_magma - SA_lapack|
               should be zero if both are IEEE compliant
               =================================================================== */
            blasf77_caxpy( &size, &s_neg_one, SA, &ione, SR, &ione );
            serror = lapackf77_clange( "Fro", &n, &n, SR, &lda, swork );
            
            printf( "zlat2c %5s %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                    lapack_uplo_const(uplo[iuplo]), (long long) n,
                    cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                    serror, (serror == 0 ? "ok" : "failed") );
            status += ! (serror == 0);
            
            /* =====================================================================
               Reset matrices
               =================================================================== */
            lapackf77_zlarnv( &ione, ISEED, &size,  A );
            lapackf77_clarnv( &ione, ISEED, &size, SA );
            
            magma_zsetmatrix( n, n, A,  lda, dA,  ldda, opts.queue );
            magma_csetmatrix( n, n, SA, lda, dSA, ldda, opts.queue );
            
            /* =====================================================================
               Performs operation using LAPACK clat2z
               LAPACK doesn't implement clat2z; use our own simple implementation.
               =================================================================== */
            cpu_time = magma_wtime();
            if ( uplo[iuplo] == MagmaLower ) {
                for( int j=0; j < n; ++j ) {
                    for( int i=j; i < n; ++i ) {
                        *A(i,j) = MAGMA_Z_MAKE( real(*SA(i,j)), imag(*SA(i,j)) );
                    }
                }
            }
            else { // upper
                for( int j=0; j < n; ++j ) {
                    for( int i=0; i <= j; ++i ) {
                        *A(i,j) = MAGMA_Z_MAKE( real(*SA(i,j)), imag(*SA(i,j)) );
                    }
                }
            }
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gbytes / cpu_time;
            if (info != 0) {
                printf("lapackf77_clat2z returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            /* ====================================================================
               Performs operation using MAGMA clat2z
               =================================================================== */
            magma_csetmatrix( n, n, SA, lda, dSA, ldda, opts.queue );
            
            gpu_time = magma_sync_wtime( opts.queue );
            magmablas_clat2z( uplo[iuplo], n, dSA, ldda, dA, ldda, opts.queue, &info );
            gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            gpu_perf = gbytes / gpu_time;
            if (info != 0) {
                printf("magmablas_clat2z returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            magma_zgetmatrix( n, n, dA, ldda, R, lda, opts.queue );
            
            if ( opts.verbose ) {
                printf( "A=  " );  magma_zprint( n, n, A,  lda );
                printf( "SA= " );  magma_cprint( n, n, SA, lda );
                printf( "dA= " );  magma_zprint_gpu( n, n, dA,  ldda, opts.queue );
                printf( "dSA=" );  magma_cprint_gpu( n, n, dSA, ldda, opts.queue );
            }
            
            /* =====================================================================
               compute error |A_magma - A_lapack|
               should be zero if both are IEEE compliant
               =================================================================== */
            blasf77_zaxpy( &size, &c_neg_one, A, &ione, R, &ione );
            error = lapackf77_zlange( "Fro", &n, &n, R, &lda, work );
            
            printf( "clat2z %5s %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)   %8.2e   %s\n",
                    lapack_uplo_const(uplo[iuplo]), (long long) n,
                    cpu_perf, cpu_time*1000., gpu_perf, gpu_time*1000.,
                    error, (error == 0 ? "ok" : "failed") );
            status += ! (error == 0);
            
            magma_free_cpu( SA );
            magma_free_cpu( A );
            magma_free_cpu( SR );
            magma_free_cpu( R );
            
            magma_free( dSA );
            magma_free( dA );
            printf( "\n" );
            fflush( stdout );
        }
        if ( opts.niter > 1 ) {
            printf( "\n" );
        }
      }
      printf( "\n" );
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
