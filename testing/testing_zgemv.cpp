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
#include "magma_operators.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgemv
*/
int main(int argc, char **argv)
{
    #ifdef HAVE_clBLAS
    #define dA(i_, j_)  dA, ((i_) + (j_)*ldda)
    #define dX(i_)      dX, ((i_))
    #define dY(i_)      dY, ((i_))
    #else                   
    #define dA(i_, j_) (dA + (i_) + (j_)*ldda)
    #define dX(i_)     (dX + (i_))
    #define dY(i_)     (dY + (i_))
    #endif
    
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gflops, magma_perf, magma_time, dev_perf, dev_time, cpu_perf, cpu_time;
    double          magma_error, dev_error, work[1];
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t M, N, Xm, Ym, lda, ldda, sizeA, sizeX, sizeY;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE(  1.5, -2.3 );
    magmaDoubleComplex beta  = MAGMA_Z_MAKE( -0.6,  0.8 );
    magmaDoubleComplex *A, *X, *Y, *Ydev, *Ymagma;
    magmaDoubleComplex_ptr dA, dX, dY;
    int status = 0;
    
    // used only with CUDA
    MAGMA_UNUSED( magma_perf );
    MAGMA_UNUSED( magma_time );
    MAGMA_UNUSED( magma_error );
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    // Allow 3*eps; complex needs 2*sqrt(2) factor; see Higham, 2002, sec. 3.6.
    double eps = lapackf77_dlamch("E");
    double tol = 3*eps;

    printf("%% trans = %s\n", lapack_trans_const(opts.transA) );
    #ifdef HAVE_CUBLAS
        printf("%%   M     N   MAGMA Gflop/s (ms)  %s Gflop/s (ms)   CPU Gflop/s (ms)  MAGMA error  %s error\n",
                g_platform_str, g_platform_str );
    #else
        printf("%%   M     N   %s Gflop/s (ms)   CPU Gflop/s (ms)  %s error\n",
                g_platform_str, g_platform_str );
    #endif
    printf("%%==================================================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            M = opts.msize[itest];
            N = opts.nsize[itest];
            lda    = M;
            ldda   = magma_roundup( M, opts.align );  // multiple of 32 by default
            gflops = FLOPS_ZGEMV( M, N ) / 1e9;

            if ( opts.transA == MagmaNoTrans ) {
                Xm = N;
                Ym = M;
            } else {
                Xm = M;
                Ym = N;
            }

            sizeA = lda*N;
            sizeX = incx*Xm;
            sizeY = incy*Ym;
            
            TESTING_CHECK( magma_zmalloc_cpu( &A,       sizeA ));
            TESTING_CHECK( magma_zmalloc_cpu( &X,       sizeX ));
            TESTING_CHECK( magma_zmalloc_cpu( &Y,       sizeY ));
            TESTING_CHECK( magma_zmalloc_cpu( &Ydev,    sizeY ));
            TESTING_CHECK( magma_zmalloc_cpu( &Ymagma,  sizeY ));
            
            TESTING_CHECK( magma_zmalloc( &dA, ldda*N ));
            TESTING_CHECK( magma_zmalloc( &dX, sizeX  ));
            TESTING_CHECK( magma_zmalloc( &dY, sizeY  ));
            
            /* Initialize the matrix */
            lapackf77_zlarnv( &ione, ISEED, &sizeA, A );
            lapackf77_zlarnv( &ione, ISEED, &sizeX, X );
            lapackf77_zlarnv( &ione, ISEED, &sizeY, Y );
            
            // for error checks
            double Anorm = lapackf77_zlange( "F", &M, &N, A, &lda, work );
            double Xnorm = lapackf77_zlange( "F", &Xm, &ione, X, &Xm, work );
            double Ynorm = lapackf77_zlange( "F", &Ym, &ione, Y, &Ym, work );
            
            /* =====================================================================
               Performs operation using cuBLAS / clBLAS
               =================================================================== */
            magma_zsetmatrix( M, N, A, lda, dA(0,0), ldda, opts.queue );
            magma_zsetvector( Xm, X, incx, dX(0), incx, opts.queue );
            magma_zsetvector( Ym, Y, incy, dY(0), incy, opts.queue );
            
            magma_flush_cache( opts.cache );
            dev_time = magma_sync_wtime( opts.queue );
            magma_zgemv( opts.transA, M, N,
                         alpha, dA(0,0), ldda,
                                dX(0),   incx,
                         beta,  dY(0),   incy, opts.queue );
            dev_time = magma_sync_wtime( opts.queue ) - dev_time;
            dev_perf = gflops / dev_time;
            
            magma_zgetvector( Ym, dY(0), incy, Ydev, incy, opts.queue );
            
            /* =====================================================================
               Performs operation using MAGMABLAS (currently only with CUDA)
               =================================================================== */
            #ifdef HAVE_CUBLAS
                magma_zsetvector( Ym, Y, incy, dY(0), incy, opts.queue );
                
                magma_flush_cache( opts.cache );
                magma_time = magma_sync_wtime( opts.queue );
                magmablas_zgemv( opts.transA, M, N,
                                 alpha, dA(0,0), ldda,
                                        dX(0),   incx,
                                 beta,  dY(0),   incy, opts.queue );
                magma_time = magma_sync_wtime( opts.queue ) - magma_time;
                magma_perf = gflops / magma_time;
                
                magma_zgetvector( Ym, dY(0), incy, Ymagma, incy, opts.queue );
            #endif
            
            /* =====================================================================
               Performs operation using CPU BLAS
               =================================================================== */
            magma_flush_cache( opts.cache );
            cpu_time = magma_wtime();
            blasf77_zgemv( lapack_trans_const(opts.transA), &M, &N,
                           &alpha, A, &lda,
                                   X, &incx,
                           &beta,  Y, &incy );
            cpu_time = magma_wtime() - cpu_time;
            cpu_perf = gflops / cpu_time;
            
            /* =====================================================================
               Check the result
               =================================================================== */
            // See testing_zgemm for formula. Here K = Xm.
            blasf77_zaxpy( &Ym, &c_neg_one, Y, &incy, Ydev, &incy );
            dev_error = lapackf77_zlange( "F", &Ym, &ione, Ydev, &Ym, work )
                            / (sqrt(double(Xm+2))*fabs(alpha)*Anorm*Xnorm + 2*fabs(beta)*Ynorm);
            
            // Really tall or wide (e.g., 200000 x 10) matrices need looser bound.
            // TODO: investigate why.
            tol = (M < 20000 && N < 20000 ? 3*eps : opts.tolerance*eps);
            
            #ifdef HAVE_CUBLAS
                blasf77_zaxpy( &Ym, &c_neg_one, Y, &incy, Ymagma, &incy );
                magma_error = lapackf77_zlange( "F", &Ym, &ione, Ymagma, &Ym, work )
                            / (sqrt(double(N+2))*fabs(alpha)*Anorm*Xnorm + 2*fabs(beta)*Ynorm);
                
                bool okay = (magma_error < tol) && (dev_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)    %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e     %8.2e   %s\n",
                       (long long) M, (long long) N,
                       magma_perf,  1000.*magma_time,
                       dev_perf,    1000.*dev_time,
                       cpu_perf,    1000.*cpu_time,
                       magma_error, dev_error,
                       (okay ? "ok" : "failed"));
            #else
                bool okay = (dev_error < tol);
                status += ! okay;
                printf("%5lld %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)    %8.2e   %s\n",
                       (long long) M, (long long) N,
                       dev_perf,    1000.*dev_time,
                       cpu_perf,    1000.*cpu_time,
                       dev_error,
                       (okay ? "ok" : "failed"));
            #endif
            
            magma_free_cpu( A );
            magma_free_cpu( X );
            magma_free_cpu( Y );
            magma_free_cpu( Ydev    );
            magma_free_cpu( Ymagma  );
            
            magma_free( dA );
            magma_free( dX );
            magma_free( dY );
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
