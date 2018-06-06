/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Raffaele Solca
       @author Azzam Haidar
       @author Mark Gates

       @precisions normal z -> c d s

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#define COMPLEX

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhegvd
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    /* Constants */
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const double d_one     =  1.;
    const double d_neg_one = -1.;
    magma_int_t ione = 1;
    
    /* Local variables */
    real_Double_t   gpu_time, cpu_time;
    magmaDoubleComplex *h_A, *h_R, *h_B, *h_S, *h_work;
    double *w1, *w2;
    double Anorm, result[4] = {0, 0, 0, 0};
    magma_int_t *iwork;
    magma_int_t N, n2, info, nb, lwork, liwork, lda;
    #ifdef COMPLEX
    double *rwork;
    magma_int_t lrwork;
    #endif
    int status = 0;

    magma_opts opts;
    opts.matrix = "rand_dominant";  // default
    opts.parse_opts( argc, argv );

    double tol    = opts.tolerance * lapackf77_dlamch("E");
    double tolulp = opts.tolerance * lapackf77_dlamch("P");
    
    // checking NoVec requires LAPACK
    opts.lapack |= (opts.check && opts.jobz == MagmaNoVec);
    
    // pass ngpu = -1 to test multi-GPU code using 1 gpu
    magma_int_t abs_ngpu = abs( opts.ngpu );
    
    printf("%% itype = %lld, jobz = %s, uplo = %s, ngpu %lld\n",
           (long long) opts.itype, lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
           (long long) abs_ngpu );

    if (opts.version == 1) {
        printf("%%   N   CPU Time (sec)   GPU Time (sec)   |D-D_magma|   |AZ-BZD|   |I-ZZ^H B|\n");
    }
    else if ( opts.version == 2) {
        printf("%%   N   CPU Time (sec)   GPU Time (sec)   |D-D_magma|   |ABZ-ZD|   |I-ZZ^H B|\n");
    }
    else if ( opts.version == 3) {
        printf("%%   N   CPU Time (sec)   GPU Time (sec)   |D-D_magma|   |BAZ-ZD|   |B-ZZ^H|\n");
    }
    printf("%%===========================================================================\n");
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda    = N;
            n2     = lda*N;
            nb     = magma_get_zhetrd_nb(N);
            #ifdef COMPLEX
                lwork  = max( N + N*nb, 2*N + N*N );
                lrwork = 1 + 5*N +2*N*N;
            #else
                lwork  = max( 2*N + N*nb, 1 + 6*N + 2*N*N );
            #endif
            liwork = 3 + 5*N;

            TESTING_CHECK( magma_zmalloc_cpu( &h_A,    n2     ));
            TESTING_CHECK( magma_zmalloc_cpu( &h_B,    n2     ));
            TESTING_CHECK( magma_dmalloc_cpu( &w1,     N      ));
            TESTING_CHECK( magma_dmalloc_cpu( &w2,     N      ));
            #ifdef COMPLEX
            TESTING_CHECK( magma_dmalloc_cpu( &rwork,  lrwork ));
            #endif
            TESTING_CHECK( magma_imalloc_cpu( &iwork,  liwork ));
            
            TESTING_CHECK( magma_zmalloc_pinned( &h_R,    n2     ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_S,    n2     ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_work, lwork  ));
            
            /* Initialize the matrix */
            magma_generate_matrix( opts, N, N, h_A, lda );
            magma_generate_matrix( opts, N, N, h_B, lda );
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_B, &lda, h_S, &lda );
            
            /* ====================================================================
               Performs operation using MAGMA
               =================================================================== */
            gpu_time = magma_wtime();
            if (opts.ngpu == 1) {
                magma_zhegvd( opts.itype, opts.jobz, opts.uplo,
                              N, h_R, lda, h_S, lda, w1,
                              h_work, lwork,
                              #ifdef COMPLEX
                              rwork, lrwork,
                              #endif
                              iwork, liwork,
                              &info );
            }
            else {
                magma_zhegvd_m( abs_ngpu, opts.itype, opts.jobz, opts.uplo,
                                N, h_R, lda, h_S, lda, w1,
                                h_work, lwork,
                                #ifdef COMPLEX
                                rwork, lrwork,
                                #endif
                                iwork, liwork,
                                &info );
            }
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_zhegvd returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }
            
            bool okay = true;
            if ( opts.check && opts.jobz != MagmaNoVec ) {
                /* =====================================================================
                   Check the results following the LAPACK's [zc]hegvd routine.
                   A x = lambda B x is solved
                   and the following 3 tests computed:
                   (1)    | A Z - B Z D | / ( |A| |Z| N )   (itype = 1)
                          | A B Z - Z D | / ( |A| |Z| N )   (itype = 2)
                          | B A Z - Z D | / ( |A| |Z| N )   (itype = 3)
                   (2)    | I - V V^H B | / ( N )           (itype = 1,2)
                          | B - V V^H   | / ( |B| N )       (itype = 3)
                   (3)    | D(with V) - D(w/o V) | / | D |
                   =================================================================== */
                //magmaDoubleComplex *tau;
                
                #ifdef REAL
                double *rwork = h_work + N*N;
                #endif

                if ( opts.itype == 1 || opts.itype == 2 ) {
                    lapackf77_zlaset( "A", &N, &N, &c_zero, &c_one, h_S, &lda);
                    blasf77_zgemm("N", "C", &N, &N, &N, &c_one, h_R, &lda, h_R, &lda, &c_zero, h_work, &N);
                    blasf77_zhemm("R", lapack_uplo_const(opts.uplo), &N, &N, &c_neg_one, h_B, &lda, h_work, &N, &c_one, h_S, &lda);
                    result[1] = lapackf77_zlange("1", &N, &N, h_S, &lda, rwork) / N;
                }
                else if ( opts.itype == 3 ) {
                    lapackf77_zlacpy( MagmaFullStr, &N, &N, h_B, &lda, h_S, &lda);
                    blasf77_zherk(lapack_uplo_const(opts.uplo), "N", &N, &N, &d_neg_one, h_R, &lda, &d_one, h_S, &lda);
                    Anorm     = safe_lapackf77_zlanhe("1", lapack_uplo_const(opts.uplo), &N, h_B, &lda, rwork);
                    result[1] = safe_lapackf77_zlanhe("1", lapack_uplo_const(opts.uplo), &N, h_S, &lda, rwork)
                              / (N*Anorm);
                }
                
                result[0] = 1.;
                result[0] /= safe_lapackf77_zlanhe("1", lapack_uplo_const(opts.uplo), &N, h_A, &lda, rwork);
                result[0] /= lapackf77_zlange("1", &N, &N, h_R, &lda, rwork);
                
                if ( opts.itype == 1 ) {
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &N, &c_one, h_A, &lda, h_R, &lda, &c_zero, h_work, &N);
                    for (int i=0; i < N; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &N, &c_neg_one, h_B, &lda, h_R, &lda, &c_one, h_work, &N);
                    result[0] *= lapackf77_zlange("1", &N, &N, h_work, &lda, rwork)/N;
                }
                else if ( opts.itype == 2 ) {
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &N, &c_one, h_B, &lda, h_R, &lda, &c_zero, h_work, &N);
                    for (int i=0; i < N; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &N, &c_one, h_A, &lda, h_work, &N, &c_neg_one, h_R, &lda);
                    result[0] *= lapackf77_zlange("1", &N, &N, h_R, &lda, rwork)/N;
                }
                else if ( opts.itype == 3 ) {
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &N, &c_one, h_A, &lda, h_R, &lda, &c_zero, h_work, &N);
                    for (int i=0; i < N; ++i)
                        blasf77_zdscal(&N, &w1[i], &h_R[i*N], &ione);
                    blasf77_zhemm("L", lapack_uplo_const(opts.uplo), &N, &N, &c_one, h_B, &lda, h_work, &N, &c_neg_one, h_R, &lda);
                    result[0] *= lapackf77_zlange("1", &N, &N, h_R, &lda, rwork)/N;
                }
                
                /*
                assert( lwork >= 2*N*N );
                lapackf77_zhet21( &ione, lapack_uplo_const(opts.uplo), &N, &izero,
                                  h_A, &lda,
                                  w1, w1,
                                  h_R, &lda,
                                  h_R, &lda,
                                  tau, h_work, rwork, &result[0] );
                */
                
                // Disable eigenvalue check which calls routine again --
                // it obscures whether error occurs in first call above or in this call.
                // But see comparison to LAPACK below.
                //
                //lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
                //lapackf77_zlacpy( MagmaFullStr, &N, &N, h_B, &lda, h_S, &lda );
                //
                //magma_zhegvd( opts.itype, MagmaNoVec, opts.uplo,
                //              N, h_R, lda, h_S, lda, w2,
                //              h_work, lwork,
                //              #ifdef COMPLEX
                //              rwork, lrwork,
                //              #endif
                //              iwork, liwork,
                //              &info );
                //if (info != 0) {
                //    printf("magma_zhegvd returned error %lld: %s.\n",
                //           (long long) info, magma_strerror( info ));
                //}
                //
                //double maxw=0, diff=0;
                //for (int j=0; j < N; j++) {
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
                lapackf77_zhegvd( &opts.itype, lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
                                  &N, h_A, &lda, h_B, &lda, w2,
                                  h_work, &lwork,
                                  #ifdef COMPLEX
                                  rwork, &lrwork,
                                  #endif
                                  iwork, &liwork,
                                  &info );
                cpu_time = magma_wtime() - cpu_time;
                if (info != 0) {
                    printf("lapackf77_zhegvd returned error %lld: %s.\n",
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
                printf("%5lld   %9.4f        %9.4f        %8.2e   ",
                       (long long) N, cpu_time, gpu_time, result[3] );
            }
            else {
                printf("%5lld      ---           %9.4f          ---      ",
                       (long long) N, gpu_time);
            }
            
            // print error checks
            if ( opts.check && opts.jobz != MagmaNoVec ) {
                okay = okay && (result[0] < tol) && (result[1] < tol);
                printf("   %8.2e   %8.2e", result[0], result[1] );
            }
            else {
                printf("     ---        ---   ");
            }
            printf("   %s\n", (okay ? "ok" : "failed"));
            status += ! okay;
            
            magma_free_cpu( h_A    );
            magma_free_cpu( h_B    );
            magma_free_cpu( w1     );
            magma_free_cpu( w2     );
            #ifdef COMPLEX
            magma_free_cpu( rwork  );
            #endif
            magma_free_cpu( iwork  );
            
            magma_free_pinned( h_R    );
            magma_free_pinned( h_S    );
            magma_free_pinned( h_work );
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
