/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Azzam Haidar
       @author Stan Tomov
       @author Mark Gates

       @precisions normal z -> s d c

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zheevd_gpu
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    /* Constants */
    const double d_zero = 0;
    const magma_int_t izero = 0;
    const magma_int_t ione  = 1;
    
    /* Local variables */
    real_Double_t   gpu_time, cpu_time;
    magmaDoubleComplex *h_A, *h_R, *h_Z, *h_work, aux_work[1], unused[1];
    magmaDoubleComplex_ptr d_R, d_Z;
    #ifdef COMPLEX
    double *rwork, aux_rwork[1];
    magma_int_t lrwork;
    #endif
    double *w1, *w2, result[4]={0, 0, 0, 0}, eps, abstol, runused[1];
    magma_int_t *iwork, *isuppz, *ifail, aux_iwork[1];
    magma_int_t N, Nfound, info, lwork, liwork, lda, ldda;
    eps = lapackf77_dlamch( "E" );
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    // checking NoVec requires LAPACK
    opts.lapack |= (opts.check && opts.jobz == MagmaNoVec);
    
    double tol    = opts.tolerance * lapackf77_dlamch("E");
    double tolulp = opts.tolerance * lapackf77_dlamch("P");

    #ifdef REAL
    if (opts.version == 3 || opts.version == 4) {
        printf("%% magma_zheevr and magma_zheevx not available for real precisions (single, double).\n");
        status = -1;
        return status;
    }
    #endif

    if (opts.version > 4) {
        fprintf( stderr, "%% error: no version %lld, only 1-4.\n", (long long) opts.version );
        status = -1;
        return status;
    }

    const char *versions[] = {
        "dummy",
        "zheevd_gpu",
        "zheevdx_gpu",
        "zheevr_gpu (Complex only)",
        "zheevx_gpu (Complex only)"
    };

    printf("%% jobz = %s, uplo = %s, version = %lld (%s)\n",
           lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
           (long long)opts.version, versions[opts.version] );

    printf("%%   N   CPU Time (sec)   GPU Time (sec)   |S-S_magma|   |A-USU^H|   |I-U^H U|\n");
    printf("%%============================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda  = N;
            ldda = magma_roundup( N, opts.align );  // multiple of 32 by default
            abstol = 0;  // auto, in zheevr
            
            magma_range_t range;
            magma_int_t il, iu;
            double vl, vu;
            opts.get_range( N, &range, &vl, &vu, &il, &iu );

            // query for workspace sizes
            if ( opts.version == 1 || opts.version == 2 ) {
                magma_zheevd_gpu( opts.jobz, opts.uplo,
                                  N, NULL, ldda, NULL,  // A, w
                                  NULL, lda,            // host A
                                  aux_work,  -1,
                                  #ifdef COMPLEX
                                  aux_rwork, -1,
                                  #endif
                                  aux_iwork, -1,
                                  &info );
            }
            else if ( opts.version == 3 ) {
                #ifdef COMPLEX
                magma_zheevr_gpu( opts.jobz, range, opts.uplo,
                                  N, NULL, ldda,     // A
                                  vl, vu, il, iu, abstol,
                                  &Nfound, NULL,         // w
                                  NULL, ldda, NULL,  // Z, isuppz
                                  NULL, lda,         // host A
                                  NULL, lda,         // host Z
                                  aux_work,  -1,
                                  #ifdef COMPLEX
                                  aux_rwork, -1,
                                  #endif
                                  aux_iwork, -1,
                                  &info );
                #endif
            }
            else if ( opts.version == 4 ) {
                #ifdef COMPLEX
                magma_zheevx_gpu( opts.jobz, range, opts.uplo,
                                  N, NULL, ldda,     // A
                                  vl, vu, il, iu, abstol,
                                  &Nfound, NULL,         // w
                                  NULL, ldda,        // Z
                                  NULL, lda,         // host A
                                  NULL, lda,         // host Z
                                  aux_work,  -1,
                                  #ifdef COMPLEX
                                  aux_rwork,
                                  #endif
                                  aux_iwork,
                                  NULL,              // ifail
                                  &info );
                // zheevx doesn't query rwork, iwork; set them for consistency
                aux_rwork[0] = double(7*N);
                aux_iwork[0] = double(5*N);
                #endif
            }
            lwork  = (magma_int_t) MAGMA_Z_REAL( aux_work[0] );
            #ifdef COMPLEX
            lrwork = (magma_int_t) aux_rwork[0];
            #endif
            liwork = aux_iwork[0];
            
            /* Allocate host memory for the matrix */
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,    N*lda  ));
            TESTING_CHECK( magma_dmalloc_cpu( &w1,     N      ));
            TESTING_CHECK( magma_dmalloc_cpu( &w2,     N      ));
            #ifdef COMPLEX
            TESTING_CHECK( magma_dmalloc_cpu( &rwork,  lrwork ));
            #endif
            TESTING_CHECK( magma_imalloc_cpu( &iwork,  liwork ));
            
            TESTING_CHECK( magma_zmalloc_pinned( &h_R,    N*lda  ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_work, lwork  ));
            
            TESTING_CHECK( magma_zmalloc( &d_R,    N*ldda ));
            
            if (opts.version == 3) {
                TESTING_CHECK( magma_zmalloc( &d_Z,    N*ldda     ));
                TESTING_CHECK( magma_zmalloc_cpu( &h_Z,    N*lda      ));
                TESTING_CHECK( magma_imalloc_cpu( &isuppz, 2*max(1,N) ));
            }
            if (opts.version == 4) {
                TESTING_CHECK( magma_zmalloc( &d_Z,    N*ldda     ));
                TESTING_CHECK( magma_zmalloc_cpu( &h_Z,    N*lda      ));
                TESTING_CHECK( magma_imalloc_cpu( &ifail,  N          ));
            }
            
            /* Clear eigenvalues, for |S-S_magma| check when fraction < 1. */
            lapackf77_dlaset( "Full", &N, &ione, &d_zero, &d_zero, w1, &N );
            lapackf77_dlaset( "Full", &N, &ione, &d_zero, &d_zero, w2, &N );
            
            /* Initialize the matrix */
            magma_generate_matrix( opts, N, N, h_A, lda );
            magma_zsetmatrix( N, N, h_A, lda, d_R, ldda, opts.queue );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if ( opts.version == 1 ) {
                magma_zheevd_gpu( opts.jobz, opts.uplo,
                                  N, d_R, ldda, w1,
                                  h_R, lda,
                                  h_work, lwork,
                                  #ifdef COMPLEX
                                  rwork, lrwork,
                                  #endif
                                  iwork, liwork,
                                  &info );
            }
            else if ( opts.version == 2 ) {  // version 2: zheevdx computes selected eigenvalues/vectors
                magma_zheevdx_gpu( opts.jobz, range, opts.uplo,
                                   N, d_R, ldda,
                                   vl, vu, il, iu,
                                   &Nfound, w1,
                                   h_R, lda,
                                   h_work, lwork,
                                   #ifdef COMPLEX
                                   rwork, lrwork,
                                   #endif
                                   iwork, liwork,
                                   &info );
                //printf( "il %lld, iu %lld, Nfound %lld\n", (long long) il, (long long) iu, (long long) Nfound );
            }
            else if ( opts.version == 3 ) {  // version 3: MRRR, computes selected eigenvalues/vectors
                // only complex version available
                #ifdef COMPLEX
                magma_zheevr_gpu( opts.jobz, range, opts.uplo,
                                  N, d_R, ldda,
                                  vl, vu, il, iu, abstol,
                                  &Nfound, w1,
                                  d_Z, ldda, isuppz,
                                  h_R, lda,  // host A
                                  h_Z, lda,  // host Z
                                  h_work, lwork,
                                  #ifdef COMPLEX
                                  rwork, lrwork,
                                  #endif
                                  iwork, liwork,
                                  &info );
                magmablas_zlacpy( MagmaFull, N, N, d_Z, ldda, d_R, ldda, opts.queue );
                #endif
            }
            else if ( opts.version == 4 ) {  // version 3: zheevx (QR iteration), computes selected eigenvalues/vectors
                // only complex version available
                #ifdef COMPLEX
                magma_zheevx_gpu( opts.jobz, range, opts.uplo,
                                  N, d_R, ldda,
                                  vl, vu, il, iu, abstol,
                                  &Nfound, w1,
                                  d_Z, ldda,
                                  h_R, lda,
                                  h_Z, lda,
                                  h_work, lwork,
                                  #ifdef COMPLEX
                                  rwork, /*lrwork,*/
                                  #endif
                                  iwork, /*liwork,*/
                                  ifail,
                                  &info );
                magmablas_zlacpy( MagmaFull, N, N, d_Z, ldda, d_R, ldda, opts.queue );
                #endif
            }
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_zheevd_gpu returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            bool okay = true;
            if ( opts.check && opts.jobz != MagmaNoVec ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zcds]drvst routine.
                   A is factored as A = U S U^H and the following 3 tests computed:
                   (1)    | A - U S U^H | / ( |A| N )
                   (2)    | I - U^H U   | / ( N )
                   (3)    | S(with U) - S(w/o U) | / | S |    // currently disabled, but compares to LAPACK
                   =================================================================== */
                magma_zgetmatrix( N, N, d_R, ldda, h_R, lda, opts.queue );
                
                magmaDoubleComplex *work;
                TESTING_CHECK( magma_zmalloc_cpu( &work, 2*N*N ));
                
                // e is unused since kband=0; tau is unused since itype=1
                lapackf77_zhet21( &ione, lapack_uplo_const(opts.uplo), &N, &izero,
                                  h_A, &lda,
                                  w1, runused,
                                  h_R, &lda,
                                  h_R, &lda,
                                  unused, work,
                                  #ifdef COMPLEX
                                  rwork,
                                  #endif
                                  &result[0] );
                result[0] *= eps;
                result[1] *= eps;
                
                magma_free_cpu( work );  work=NULL;
                
                // Disable third eigenvalue check that calls routine again --
                // it obscures whether error occurs in first call above or in this call.
                // But see comparison to LAPACK below.
                //
                //magma_zsetmatrix( N, N, h_A, lda, d_R, ldda, opts.queue );
                //magma_zheevd_gpu( MagmaNoVec, opts.uplo,
                //                  N, d_R, ldda, w2,
                //                  h_R, lda,
                //                  h_work, lwork,
                //                  #ifdef COMPLEX
                //                  rwork, lrwork,
                //                  #endif
                //                  iwork, liwork,
                //                  &info);
                //if (info != 0) {
                //    printf("magma_zheevd_gpu returned error %lld: %s.\n",
                //           (long long) info, magma_strerror( info ));
                //}
                //
                //double maxw=0, diff=0;
                //for( int j=0; j < N; j++ ) {
                //    maxw = max(maxw, fabs(w1[j]));
                //    maxw = max(maxw, fabs(w2[j]));
                //    diff = max(diff, fabs(w1[j] - w2[j]));
                //}
                //result[2] = diff / (N*maxw);
            }
            
            /* =====================================================================
               Performs operation using LAPACK
               =================================================================== */
            if ( opts.lapack ) {
                cpu_time = magma_wtime();
                if ( opts.version == 1 || opts.version == 2 ) {
                    lapackf77_zheevd( lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
                                      &N, h_A, &lda, w2,
                                      h_work, &lwork,
                                      #ifdef COMPLEX
                                      rwork, &lrwork,
                                      #endif
                                      iwork, &liwork,
                                      &info );
                }
                else if ( opts.version == 3 ) {
                    lapackf77_zheevr( lapack_vec_const(opts.jobz),
                                      lapack_range_const(range),
                                      lapack_uplo_const(opts.uplo),
                                      &N, h_A, &lda,
                                      &vl, &vu, &il, &iu, &abstol,
                                      &Nfound, w2,
                                      h_Z, &lda, isuppz,
                                      h_work, &lwork,
                                      #ifdef COMPLEX
                                      rwork, &lrwork,
                                      #endif
                                      iwork, &liwork,
                                      &info );
                    lapackf77_zlacpy( "Full", &N, &N, h_Z, &lda, h_A, &lda );
                }
                else if ( opts.version == 4 ) {
                    lapackf77_zheevx( lapack_vec_const(opts.jobz),
                                      lapack_range_const(range),
                                      lapack_uplo_const(opts.uplo),
                                      &N, h_A, &lda,
                                      &vl, &vu, &il, &iu, &abstol,
                                      &Nfound, w2,
                                      h_Z, &lda,
                                      h_work, &lwork,
                                      #ifdef COMPLEX
                                      rwork,
                                      #endif
                                      iwork,
                                      ifail,
                                      &info );
                    lapackf77_zlacpy( "Full", &N, &N, h_Z, &lda, h_A, &lda );
                }
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0) {
                    printf("lapackf77_zheevd returned error %lld: %s.\n",
                           (long long) info, magma_strerror( info ));
                }
                
                // compare eigenvalues
                double maxw=0, diff=0;
                for( int j=0; j < N; j++ ) {
                    maxw = max(maxw, fabs(w1[j]));
                    maxw = max(maxw, fabs(w2[j]));
                    diff = max(diff, fabs(w1[j] - w2[j]));
                }
                result[3] = diff / (N*maxw);
                
                okay = okay && (result[3] < tolulp);
                printf("%5lld   %9.4f        %9.4f         %8.2e  ",
                       (long long) N, cpu_time, gpu_time, result[3] );
            }
            else {
                printf("%5lld      ---           %9.4f           ---     ",
                       (long long) N, gpu_time);
            }
            
            // print error checks
            if ( opts.check && opts.jobz != MagmaNoVec ) {
                okay = okay && (result[0] < tol) && (result[1] < tol);
                printf("    %8.2e    %8.2e", result[0], result[1] );
            }
            else {
                printf("      ---         ---   ");
            }
            printf("   %s\n", (okay ? "ok" : "failed"));
            status += ! okay;

            magma_free_cpu( h_A    );
            magma_free_cpu( w1     );
            magma_free_cpu( w2     );
            #ifdef COMPLEX
            magma_free_cpu( rwork  );
            #endif
            magma_free_cpu( iwork  );
            
            magma_free_pinned( h_R    );
            magma_free_pinned( h_work );
            
            magma_free( d_R );
            
            if ( opts.version == 3 ) {
                magma_free( d_Z    );
                magma_free_cpu( h_Z    );
                magma_free_cpu( isuppz );
            }
            if ( opts.version == 4 ) {
                magma_free( d_Z    );
                magma_free_cpu( h_Z    );
                magma_free_cpu( ifail  );
            }
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
