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

#include "../control/magma_threadsetting.h"  // internal header

#define COMPLEX

static magma_int_t check_orthogonality(magma_int_t M, magma_int_t N, magmaDoubleComplex *Q, magma_int_t LDQ, double eps);
static magma_int_t check_reduction(magma_uplo_t uplo, magma_int_t N, magma_int_t bw, magmaDoubleComplex *A, double *D, magma_int_t LDA, magmaDoubleComplex *Q, double eps );
static magma_int_t check_solution(magma_int_t N, magma_int_t Nfound, double *E1, double *E2, double tolulp);

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zhegvdx
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t gpu_time;

    magmaDoubleComplex *h_A, *h_R, *h_work, unused[1];;

    #ifdef COMPLEX
    double *rwork;
    magma_int_t lrwork;
    #endif

    double *w1, *w2, result[4]={0, 0, 0, 0}, runused[1];
    magma_int_t *iwork;
    magma_int_t N, Nfound, n2, info, lda, lwork, liwork;
    magma_int_t info_ortho     = 0;
    magma_int_t info_solution  = 0;
    magma_int_t info_reduction = 0;
    magma_int_t ione = 1;
    magma_int_t izero = 0;
    int status = 0;

    magma_opts opts;
    opts.parse_opts( argc, argv );

    double tol    = opts.tolerance * lapackf77_dlamch("E");
    //double tolulp = opts.tolerance * lapackf77_dlamch("P");

    // pass ngpu = -1 to test multi-GPU code using 1 gpu
    magma_int_t abs_ngpu = abs( opts.ngpu );

    printf("%% jobz = %s, uplo = %s, ngpu %lld\n",
           lapack_vec_const(opts.jobz), lapack_uplo_const(opts.uplo),
           (long long) abs_ngpu);

    printf("%%   N     M  GPU Time (sec)   ||I-Q^H Q||/N   ||A-QDQ^H||/(||A||N)   |D-D_magma|/(|D| * N)\n");
    printf("%%=========================================================================================\n");
    magma_int_t threads = magma_get_parallel_numthreads();
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        for( int iter = 0; iter < opts.niter; ++iter ) {
            N = opts.nsize[itest];
            lda = N;
            n2  = lda*N;

            magma_range_t range;
            magma_int_t il, iu;
            double vl, vu;
            opts.get_range( N, &range, &vl, &vu, &il, &iu );

            magma_zheevdx_getworksize(N, threads, (opts.jobz == MagmaVec),
                                     &lwork,
                                     #ifdef COMPLEX
                                     &lrwork,
                                     #endif
                                     &liwork);

            if( opts.jobz == MagmaNoVec ) {
                // For LAPACK test using zheevx.
                #ifdef COMPLEX
                lrwork = 7*N;
                #endif
                liwork = 5*N;
            }

            /* Allocate host memory for the matrix */
            TESTING_CHECK( magma_zmalloc_cpu( &h_A,   n2 ));
            TESTING_CHECK( magma_dmalloc_cpu( &w1,    N ));
            TESTING_CHECK( magma_dmalloc_cpu( &w2,    N ));
            TESTING_CHECK( magma_imalloc_cpu( &iwork, liwork ));

            TESTING_CHECK( magma_zmalloc_pinned( &h_R,    n2    ));
            TESTING_CHECK( magma_zmalloc_pinned( &h_work, lwork ));
            #ifdef COMPLEX
            TESTING_CHECK( magma_dmalloc_pinned( &rwork, lrwork ));
            #endif

            /* Initialize the matrix */
            magma_generate_matrix( opts, N, N, h_A, lda );

            if (opts.warmup) {
                // ==================================================================
                // Warmup using MAGMA
                // ==================================================================
                lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
                if (opts.ngpu == 1) {
                    //printf("calling zheevdx_2stage 1 GPU\n");
                    magma_zheevdx_2stage( opts.jobz, range, opts.uplo, N,
                                          h_R, lda,
                                          vl, vu, il, iu,
                                          &Nfound, w1,
                                          h_work, lwork,
                                          #ifdef COMPLEX
                                          rwork, lrwork,
                                          #endif
                                          iwork, liwork,
                                          &info );
                } else {
                    //printf("calling zheevdx_2stage_m %lld GPU\n", (long long) opts.ngpu);
                    magma_zheevdx_2stage_m( abs_ngpu, opts.jobz, range, opts.uplo, N,
                                            h_R, lda,
                                            vl, vu, il, iu,
                                            &Nfound, w1,
                                            h_work, lwork,
                                            #ifdef COMPLEX
                                            rwork, lrwork,
                                            #endif
                                            iwork, liwork,
                                            &info );
                }
            }

            // ===================================================================
            // Performs operation using MAGMA
            // ===================================================================
            lapackf77_zlacpy( MagmaFullStr, &N, &N, h_A, &lda, h_R, &lda );
            gpu_time = magma_wtime();
            if (opts.ngpu == 1) {
                //printf("calling zheevdx_2stage 1 GPU\n");
                magma_zheevdx_2stage( opts.jobz, range, opts.uplo, N,
                                      h_R, lda,
                                      vl, vu, il, iu,
                                      &Nfound, w1,
                                      h_work, lwork,
                                      #ifdef COMPLEX
                                      rwork, lrwork,
                                      #endif
                                      iwork, liwork,
                                      &info );
            } else {
                //printf("calling zheevdx_2stage_m %lld GPU\n", (long long) opts.ngpu);
                magma_zheevdx_2stage_m( abs_ngpu, opts.jobz, range, opts.uplo, N,
                                        h_R, lda,
                                        vl, vu, il, iu,
                                        &Nfound, w1,
                                        h_work, lwork,
                                        #ifdef COMPLEX
                                        rwork, lrwork,
                                        #endif
                                        iwork, liwork,
                                        &info );
            }
            gpu_time = magma_wtime() - gpu_time;
            if (info != 0) {
                printf("magma_zheevdx_2stage returned error %lld: %s.\n",
                       (long long) info, magma_strerror( info ));
            }

            printf("%5lld %5lld  %7.2f      ",
                   (long long) N, (long long) Nfound, gpu_time );

            if ( opts.check ) {
                info_solution  = 0;
                info_ortho     = 0;
                info_reduction = 0;
                //double eps   = lapackf77_dlamch("E")*lapackf77_dlamch("B");
                double eps    = lapackf77_dlamch("E");
                double abstol = 2 * lapackf77_dlamch("S");

                magmaDoubleComplex* h_Z;
                magma_int_t* ifail;
                TESTING_CHECK( magma_zmalloc_cpu( &h_Z,    N*lda      ));
                TESTING_CHECK( magma_imalloc_cpu( &ifail,  N          ));

                /* Check the orthogonality, reduction and the eigen solutions */
                if (opts.jobz == MagmaVec) {
                    // Disable the following old tests and use LAPACK routines.
                    //info_ortho = check_orthogonality(N, N, h_R, lda, eps);
                    //info_reduction = check_reduction(opts.uplo, N, 1, h_A, w1, lda, h_R, eps);

                    /* =====================================================================
                       Check the results following the LAPACK's [zcds]drvst routine.
                       A is factored as A = U S U^H and the following 3 tests computed:
                       (1)    | A - U S U^H | / ( |A| N ) if all eigenvectors were computed
                              | U^H A U - S | / ( |A| Nfound ) otherwise
                       (2)    | I - U^H U   | / ( N )
                       (3)    | S(with U) - S(w/o U) | / | S |    // currently disabled, but compares to LAPACK
                       =================================================================== */
                    magmaDoubleComplex* work;
                    TESTING_CHECK( magma_zmalloc_cpu( &work, 2*N*N ));

                    // e is unused since kband=0; tau is unused since itype=1
                    if( Nfound == N ) {
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
                    } else {
                        lapackf77_zhet22( &ione, lapack_uplo_const(opts.uplo), &N, &Nfound, &izero,
                                          h_A, &lda,
                                          w1, runused,
                                          h_R, &lda,
                                          h_R, &lda,
                                          unused, work,
                                          #ifdef COMPLEX
                                          rwork,
                                          #endif
                                          &result[0] );
                    }
                    result[0] *= eps;
                    result[1] *= eps;
                    info_reduction = result[0] >= tol;
                    info_ortho     = result[1] >= tol;
                    printf("       %8.2e      %8.2e", result[0], result[1] );
                    magma_free_cpu( work );  work=NULL;
                } else {
                    printf("         ---                ---  ");
                }
                lapackf77_zheevx( "N",
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

                info_solution = check_solution(N, Nfound, w2, w1, eps);

                magma_free_cpu(h_Z);
                magma_free_cpu(ifail);

                bool okay = (info_solution == 0) && (info_ortho == 0) && (info_reduction == 0);
                status += ! okay;
                printf("  %s", (okay ? "ok" : "failed"));
            }
            printf("\n");

            magma_free_cpu( h_A   );
            magma_free_cpu( w1    );
            magma_free_cpu( w2    );
            magma_free_cpu( iwork );

            magma_free_pinned( h_R    );
            magma_free_pinned( h_work );
            #ifdef COMPLEX
            magma_free_pinned( rwork  );
            #endif
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



/*-------------------------------------------------------------------
 * Check the orthogonality of Q
 */
static magma_int_t check_orthogonality(magma_int_t M, magma_int_t N, magmaDoubleComplex *Q, magma_int_t LDQ, double eps)
{
    double d_one     =  1.0;
    double d_neg_one = -1.0;
    magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    double  normQ, result;
    magma_int_t     info_ortho;
    magma_int_t     minMN = min(M, N);
    double *work;
    TESTING_CHECK( magma_dmalloc_cpu( &work, minMN ));

    /* Build the idendity matrix */
    magmaDoubleComplex *Id;
    TESTING_CHECK( magma_zmalloc_cpu( &Id, minMN*minMN ));
    lapackf77_zlaset("A", &minMN, &minMN, &c_zero, &c_one, Id, &minMN);

    /* Perform Id - Q^H Q */
    if (M >= N)
        blasf77_zherk("U", "C", &N, &M, &d_one, Q, &LDQ, &d_neg_one, Id, &N);
    else
        blasf77_zherk("U", "N", &M, &N, &d_one, Q, &LDQ, &d_neg_one, Id, &M);

    normQ = safe_lapackf77_zlanhe("I", "U", &minMN, Id, &minMN, work);

    result = normQ / (minMN * eps);
    printf( "      %8.2e", normQ / minMN );

    // TODO: use opts.tolerance instead of hard coding 60
    if (std::isnan(result) || std::isinf(result) || (result > 60.0)) {
        info_ortho = 1;
    }
    else {
        info_ortho = 0;
    }
    magma_free_cpu( work );
    magma_free_cpu( Id   );

    return info_ortho;
}


/*------------------------------------------------------------
 *  Check the reduction
 */
static magma_int_t check_reduction(magma_uplo_t uplo, magma_int_t N, magma_int_t bw, magmaDoubleComplex *A, double *D, magma_int_t LDA, magmaDoubleComplex *Q, double eps )
{
    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex *TEMP, *Residual;
    double *work;
    double Anorm, Rnorm, result;
    magma_int_t info_reduction;
    magma_int_t i;
    magma_int_t ione=1;

    TESTING_CHECK( magma_zmalloc_cpu( &TEMP,     N*N ));
    TESTING_CHECK( magma_zmalloc_cpu( &Residual, N*N ));
    TESTING_CHECK( magma_dmalloc_cpu( &work,     N ));

    /* Compute TEMP =  Q * LAMBDA */
    lapackf77_zlacpy("A", &N, &N, Q, &LDA, TEMP, &N);
    for (i = 0; i < N; i++) {
        blasf77_zdscal(&N, &D[i], &(TEMP[i*N]), &ione);
    }
    /* Compute Residual = A - Q * LAMBDA * Q^H */
    /* A is Hermitian but both upper and lower
     * are assumed valable here for checking
     * otherwise it need to be symetrized before
     * checking.
     */
    lapackf77_zlacpy("A", &N, &N, A, &LDA, Residual, &N);
    blasf77_zgemm("N", "C", &N, &N, &N, &c_neg_one, TEMP, &N, Q, &LDA, &c_one, Residual,     &N);

    // since A has been generated by larnv and we did not symmetrize,
    // so only the uplo portion of A should be equal to Q*LAMBDA*Q^H
    // for that Rnorm use zlanhe instead of zlange
    Rnorm = safe_lapackf77_zlanhe("1", lapack_uplo_const(uplo), &N, Residual, &N, work);
    Anorm = safe_lapackf77_zlanhe("1", lapack_uplo_const(uplo), &N, A,        &LDA, work);

    result = Rnorm / ( Anorm * N * eps);
    printf("           %8.2e",  Rnorm / ( Anorm * N));

    // TODO: use opts.tolerance instead of hard coding 60
    if (std::isnan(result) || std::isinf(result) || (result > 60.0)) {
        info_reduction = 1;
    }
    else {
        info_reduction = 0;
    }

    magma_free_cpu( TEMP     );
    magma_free_cpu( Residual );
    magma_free_cpu( work     );

    return info_reduction;
}


/*------------------------------------------------------------
 *  Check the eigenvalues
 */
static magma_int_t check_solution(magma_int_t N, magma_int_t Nfound, double *E1, double *E2, double tolulp)
{
    magma_int_t   info_solution, i;
    double unfl   = lapackf77_dlamch("Safe minimum");
    double resid;
    double maxtmp;
    double maxdif = fabs( fabs(E1[0]) - fabs(E2[0]) );
    double maxeig = max( fabs(E1[0]), fabs(E2[0]) );
    for (i = 1; i < Nfound; i++) {
        resid   = fabs(fabs(E1[i])-fabs(E2[i]));
        maxtmp  = max(fabs(E1[i]), fabs(E2[i]));

        /* Update */
        maxeig = max(maxtmp, maxeig);
        maxdif  = max(resid,  maxdif );
    }
    maxtmp = maxdif / max(unfl, max(maxeig, maxdif));

    printf("              %8.2e", maxdif / (max(maxeig, maxdif)) );

    if (std::isnan(maxtmp) || std::isinf(maxtmp) || (maxtmp >= tolulp * N)) {
        info_solution = 1;
    }
    else {
        info_solution = 0;
    }
    return info_solution;
}
