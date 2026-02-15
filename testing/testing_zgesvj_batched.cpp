/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Mark Gates
       @author Ahmad Abdelfattah

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <string>
#include <algorithm>  // find

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#if defined(_OPENMP)
#include <omp.h>
#include "../control/magma_threadsetting.h"  // internal header
#endif

#define PRECISION_z
#define COMPLEX

// -----------------------------------------------------------------------------
// define which cpu routine to use: gesvd is much faster than gesvj (unblocked)
#define CPU_GESVD
//#define CPU_GESVJ

// -----------------------------------------------------------------------------
// returns true if vec contains value.
static bool string_contains( const char* func, const char* name )
{
    std::string routine(func);
    return ( routine.find(name) != std::string::npos);
}

// -----------------------------------------------------------------------------
// batch wrapper over check_zgesvd, with the
// support of gesvj (by transposing v)
static void check_zgesvj_batched(
    const char* func,
    magma_int_t check,
    magma_vec_t jobu,
    magma_vec_t jobv,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex** hA_array, magma_int_t lda,
    double** hS_array,
    magmaDoubleComplex** hU_array, magma_int_t ldu,
    magmaDoubleComplex** hV_array, magma_int_t ldv,
    magma_int_t* hinfo_array, magma_int_t batchCount,
    double result[4], magma_int_t &status )
{
    magma_int_t vm   = n;
    magma_int_t vn   = min(m, n);
    magma_int_t ldvt = min(m, n); // vt is always min(m, n) x n

    magmaDoubleComplex* tmpV = new magmaDoubleComplex[ldvt * vm];
    const double nan = MAGMA_D_NAN;
    double tmp[5] = { nan, nan, nan, nan, nan };

    for(magma_int_t s = 0; s < batchCount; s++) {
        if (hinfo_array[s] != 0) {
            printf( "%s returned error %lld for problem %lld: %s.\n",
                func, (long long) hinfo_array[s], (long long) s, magma_strerror( hinfo_array[s] ));
            status += 1;
        }

        if( string_contains(func, "gesvd") ) {
            // v is already transposed, just copy
            lapackf77_zlacpy( MagmaFullStr, &vn, &vm, hV_array[s], &ldv, tmpV, &ldvt );
        }
        else {
            // v needs to be transposed
            for(magma_int_t ir = 0; ir < vm; ir++) {
                for(magma_int_t ic = 0; ic < vn; ic++) {
                    tmpV[ir * ldvt + ic] = MAGMA_Z_CONJ( hV_array[s][ic * ldv + ir] );
                }
            }
        }

        check_zgesvd( check, jobu, jobv, m, n, hA_array[s], lda, hS_array[s],
                      hU_array[s], ldu,
                      tmpV, ldvt,
                      tmp );

        for(magma_int_t itmp = 0; itmp < 4; itmp++) {
            result[ itmp ] = (s == 0 || tmp[ itmp ] < 0) ? tmp[ itmp ] : magma_max_nan( result[ itmp ], tmp[ itmp ]);
        }
    }
    delete[] tmpV;
}

/* ////////////////////////////////////////////////////////////////////////////
   -- Testing zgesvj batched (one-sided Jacobi SVD)
*/
int main( int argc, char** argv)
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    // Constants
    const magma_int_t ione     = 1;
    const double d_neg_one = -1;
    const double nan = MAGMA_D_NAN;

    // Local variables
    real_Double_t   gpu_time=0, cpu_time=0;
    magmaDoubleComplex *hA, *hR, *hU, *hV, *hwork;
    magmaDoubleComplex *dA, *dU, *dV;
    magmaDoubleComplex **hA_array, **hR_array, **hU_array, **hV_array, **hwork_array;
    magmaDoubleComplex **dA_array, **dU_array, **dV_array;
    double *hS, *hSref, *dS, work[1];
    double **hS_array, **hSref_array, **dS_array;
    magma_int_t M, N, NN, lda, ldda, ldu, ldv, ldvt, lddu, lddv, min_mn, lwork;
    magma_int_t um, un, vm, vn;
    magma_int_t *hinfo_array, *dinfo_array;
    magma_int_t status = 0;
    #ifdef COMPLEX
    magma_int_t lrwork = 0;
    double *hrwork;
    double **hrwork_array;
    #endif

    #ifdef CPU_GESVD
    const magma_int_t ineg_one = -1;
    magma_int_t info;
    magmaDoubleComplex dummy[1], unused[1];
    double runused[1];
    #else
    char ju[1], jv[1];
    #endif

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );
    magma_int_t batchCount = opts.batchcount;

    double tol = opts.tolerance * lapackf77_dlamch("E");

    #ifdef CPU_GESVD
    const char *cpu_func = "lapackf77_zgesvd";
    #else
    const char *cpu_func = "lapackf77_zgesvj";
    #endif

    // print which CPU routine is used
    if(opts.lapack) {
        printf("%%lapack test is using %s\n",
        string_contains(cpu_func, "gesvd") ? "gesvd" : "gesvj" );
    }

    // force opts.check to one if it is not zero
    if( opts.check ) opts.check = 1;

    // Allocate pointer arrays outside the main loop (cpu)
    TESTING_CHECK( magma_malloc_cpu( (void**) &hA_array, batchCount * sizeof(magmaDoubleComplex*) ));
    TESTING_CHECK( magma_malloc_cpu( (void**) &hR_array, batchCount * sizeof(magmaDoubleComplex*) ));
    TESTING_CHECK( magma_malloc_cpu( (void**) &hU_array, batchCount * sizeof(magmaDoubleComplex*) ));
    TESTING_CHECK( magma_malloc_cpu( (void**) &hV_array, batchCount * sizeof(magmaDoubleComplex*) ));
    TESTING_CHECK( magma_malloc_cpu( (void**) &hS_array, batchCount * sizeof(double*) ));
    TESTING_CHECK( magma_malloc_cpu( (void**) &hSref_array, batchCount * sizeof(double*) ));
    TESTING_CHECK( magma_imalloc_cpu( &hinfo_array, batchCount ));
    if(opts.lapack) {
        TESTING_CHECK( magma_malloc_cpu( (void**) &hwork_array, batchCount * sizeof(magmaDoubleComplex*) ));
        #ifdef COMPLEX
        TESTING_CHECK( magma_malloc_cpu( (void**) &hrwork_array, batchCount * sizeof(double*) ));
        #endif
    }

    // Allocate pointer arrays outside the main loop (gpu)
    TESTING_CHECK( magma_malloc( (void**) &dA_array, batchCount * sizeof(magmaDoubleComplex*) ));
    TESTING_CHECK( magma_malloc( (void**) &dS_array, batchCount * sizeof(double*) ));
    TESTING_CHECK( magma_malloc( (void**) &dU_array, batchCount * sizeof(magmaDoubleComplex*) ));
    TESTING_CHECK( magma_malloc( (void**) &dV_array, batchCount * sizeof(magmaDoubleComplex*) ));
    TESTING_CHECK( magma_imalloc( &dinfo_array, batchCount ));

    printf( "%% jobu,v      BatchCount    M     N   CPU time (ms )   GPU time (ms )   |S-Sref|   |A-USV^H|  |I-UU^H|/M  |I-VV^H|/N   S sorted\n" );
    printf( "%%====================================================================================================================================\n" );
    // main loop
    for( int itest = 0; itest < opts.ntest; ++itest ) {
      for( auto jobu = opts.jobu.begin(); jobu != opts.jobu.end(); ++jobu ) {
      for( auto jobv = opts.jobv.begin(); jobv != opts.jobv.end(); ++jobv ) {
        // magma_zgesvj_batched accepts MagmaVec and MagmaSomeVec (both are equivalent)
        // GESVD accepts only 's' = some vectors
        magma_vec_t jobu_cpu = (*jobu == MagmaVec) ? MagmaSomeVec : (*jobu);
        magma_vec_t jobv_cpu = (*jobv == MagmaVec) ? MagmaSomeVec : (*jobv);

        if ( *jobu == MagmaOverwriteVec || *jobu == MagmaAllVec ||
             *jobv == MagmaOverwriteVec || *jobv == MagmaAllVec ) {
            printf( "skipping invalid combination jobu=%c, jobvt=%c\n", lapacke_vec_const(*jobu), lapacke_vec_const(*jobv));
            printf( "allowed options are 'n' = MagmaNoVec, 'v' = MagmaVec, or 's' = MagmaSomeVec \n" );
            continue;
        }

        for( int iter = 0; iter < opts.niter; ++iter ) {
            M      = opts.msize[itest];
            N      = opts.nsize[itest];
            min_mn = min(M, N);
            NN     = batchCount * N;

            um  = M;
            un  = min_mn;
            vm  = N;
            vn  = min_mn;

            lda  = M;
            ldu  = um;
            ldv  = vm;
            ldvt = vn;

            ldda = magma_roundup( lda, opts.align );  // multiple of 32 by default
            lddu = magma_roundup( ldu, opts.align );  // multiple of 32 by default
            lddv = magma_roundup( ldv, opts.align );  // multiple of 32 by default

            /* =====================================================================
               query for workspace size -- LAPACK
               =================================================================== */
            if(opts.lapack) {
                magma_int_t query_lapack = 0;
                #ifdef CPU_GESVD
                lapackf77_zgesvd( lapack_vec_const(jobu_cpu), lapack_vec_const(jobv_cpu), &M, &N,
                                  unused, &lda, runused,
                                  unused, &ldu,
                                  unused, &ldvt,
                                  dummy, &ineg_one,
                                  #ifdef COMPLEX
                                  runused,
                                  #endif
                                  &info );
                assert( info == 0 );
                query_lapack = (magma_int_t) MAGMA_Z_REAL( dummy[0] );
                #ifdef COMPLEX
                lrwork = 5 * min_mn;
                #endif
                #else
                //--------------------------------------------------------------------------
                //jacobi -- skip query
                ju[0] = { ( *jobu == MagmaNoVec ? 'N' : 'U') };
                jv[0] = { ( *jobv == MagmaNoVec ? 'N' : 'V') };
                query_lapack = max(6, M+N); // as per MKL documentation
                #ifdef COMPLEX
                lrwork = max(6, M+N); // as per MKL documentation
                #endif
                #endif // CPU_GESVD
                lwork = query_lapack;
            }

            /* =====================================================================
               Allocate memory
               =================================================================== */
            // cpu
            TESTING_CHECK( magma_zmalloc_cpu( &hA,    batchCount*lda*N  ));
            TESTING_CHECK( magma_dmalloc_cpu( &hS,    batchCount*min_mn ));
            TESTING_CHECK( magma_dmalloc_cpu( &hSref, batchCount*min_mn ));
            TESTING_CHECK( magma_zmalloc_cpu( &hU,    batchCount*ldu*un ));
            TESTING_CHECK( magma_zmalloc_cpu( &hV,    batchCount*ldv*vn ));
            TESTING_CHECK( magma_zmalloc_cpu( &hR,    batchCount*lda*N ));
            if(opts.lapack) {
                TESTING_CHECK( magma_zmalloc_cpu( &hwork, batchCount*lwork ));
                #ifdef COMPLEX
                TESTING_CHECK( magma_dmalloc_cpu( &hrwork, batchCount*lrwork ));
                #endif
            }

            // gpu
            TESTING_CHECK( magma_zmalloc( &dA, batchCount*ldda*N ));
            TESTING_CHECK( magma_dmalloc( &dS, batchCount*min_mn ));
            TESTING_CHECK( magma_zmalloc( &dU, batchCount*lddu*un ));
            TESTING_CHECK( magma_zmalloc( &dV, batchCount*lddv*vn ));

            /* Initialize pointer arrays */
            for(magma_int_t s = 0; s < batchCount; s++) {
                hA_array[s] = hA + s*lda*N;
                hR_array[s] = hR + s*lda*N;
                hS_array[s] = hS + s*min_mn;
                hU_array[s] = hU + s*ldu*un;
                hV_array[s] = hV + s*ldv*vn;
                hSref_array[s]  = hSref  + s*min_mn;
                if(opts.lapack) {
                    hwork_array[s]  = hwork  + s*lwork;
                    #ifdef COMPLEX
                    hrwork_array[s] = hrwork + s*lrwork;
                    #endif
                }
            }
            magma_zset_pointer( dA_array, dA, ldda, 0, 0, ldda*N,  batchCount, opts.queue );
            magma_zset_pointer( dU_array, dU, lddu, 0, 0, lddu*un, batchCount, opts.queue );
            magma_zset_pointer( dV_array, dV, lddv, 0, 0, lddv*vn, batchCount, opts.queue );
            magma_dset_pointer( dS_array, dS,    1, 0, 0, min_mn,  batchCount, opts.queue );

            /* Initialize the matrix */
            for(magma_int_t s = 0; s < batchCount; s++) {
                magma_generate_matrix( opts, M, N, hA_array[s], lda, hSref_array[s] );
            }
            // copy hA to hR
            lapackf77_zlacpy( MagmaFullStr, &M, &NN, hA, &lda, hR, &lda );

            magma_flush_cache( opts.cache );
            /* ====================================================================
                Performs operation using MAGMA
                =================================================================== */
            magma_zsetmatrix( M, NN, hA, lda, dA, ldda, opts.queue );
            if(opts.version == 1) {
                // top-level interface w/ internal workspace
                gpu_time = magma_sync_wtime( opts.queue );
                magma_zgesvj_batched(
                    *jobu, *jobv, M, N,
                    dA_array, ldda, dS_array,
                    dU_array, lddu,
                    dV_array, lddv,
                    dinfo_array, batchCount, opts.queue );
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;
            }
            else if(opts.version == 2) {
                // expert interface w/ user-managed workspace
                void* device_work = NULL;
                int64_t device_lwork[1] = {-1};

                // query workspace
                magma_zgesvj_expert_batched(
                    *jobu, *jobv, M, N,
                    NULL, ldda, NULL, NULL, lddu, NULL, lddv, NULL,
                    NULL, device_lwork,
                    batchCount, opts.queue );

                // allocate workspace
                if(device_lwork[0] > 0) {
                    magma_malloc(&device_work, device_lwork[0]);
                }

                // actual run
                gpu_time = magma_sync_wtime( opts.queue );
                magma_zgesvj_expert_batched(
                    *jobu, *jobv, M, N,
                    dA_array, ldda, dS_array,
                    dU_array, lddu, dV_array, lddv,
                    dinfo_array, device_work, device_lwork,
                    batchCount, opts.queue );
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

                // free workspace
                if(device_work != NULL) magma_free( device_work );
            }
            else if( opts.version == 3) {
                // expert interface w/ user-managed workspace
                // performs a QR factorization first
                void* device_work = NULL;
                int64_t device_lwork[1] = {-1};

                // query qworkspace
                magma_zgesvj_qr_expert_batched(
                    *jobu, *jobv, M, N,
                    NULL, ldda,  NULL,
                    NULL, lddu,  NULL, lddv,
                    dinfo_array, NULL, device_lwork,
                    batchCount, opts.queue );

                // allocate qworkspace
                if(device_lwork[0] > 0) {
                    magma_malloc(&device_work, device_lwork[0]);
                }

                // actual run
                gpu_time = magma_sync_wtime( opts.queue );
                magma_zgesvj_qr_expert_batched(
                    *jobu, *jobv, M, N,
                    dA_array, ldda, dS_array,
                    dU_array, lddu, dV_array, lddv,
                    dinfo_array, device_work, device_lwork,
                    batchCount, opts.queue );
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

                // free qworkspace
                if(device_work != NULL) magma_free( device_work );
            }
            else if( opts.version == 4 ) {
                // expert interface w/ user-managed workspace
                // blocked code only, with several controls exposed to the user

                // if nb is not set by the user, call magma_get_zgesvj_batched_nb
                magma_int_t gesvj_nb = (opts.nb <= 0) ? magma_get_zgesvj_batched_nb( M, N ) : opts.nb;
                magma_int_t gesvj_max_sweeps = 100;
                double eps                   = lapackf77_dlamch("E");
                double heevj_tol             = pow(10,floor(0.3 * log10(1/eps)));  // starting tolerance (as multiples of eps) for heevj, empirically decided
                double heevj_tol_min         = 30; // minimum tolerance (as multiples of eps) for heevj
                double heevj_tol_scal        = 10; // heevj_tol is divided by heevj_tol_scal after every Jacobi sweep (to disable, set heevj_tol to desired tolerance and set heevj_tol_scal to 1)
                magma_int_t heevj_max_sweeps = 1;  // partial or full eigensolver (a full solver impacts performance)

                void* device_work = NULL;
                int64_t device_lwork[1] = {-1};

                // query qworkspace
                magma_zgesvj_blocked_expert_batched(
                    *jobu, *jobv, M, N,
                    NULL, ldda, NULL, NULL, lddu, NULL, lddv, NULL,
                    gesvj_nb, gesvj_max_sweeps,
                    heevj_max_sweeps, heevj_tol, heevj_tol_min, heevj_tol_scal,
                    NULL, device_lwork, batchCount, opts.queue);

                // allocate qworkspace
                if(device_lwork[0] > 0) {
                    magma_malloc(&device_work, device_lwork[0]);
                }

                // actual run
                gpu_time = magma_sync_wtime( opts.queue );
                magma_zgesvj_blocked_expert_batched(
                    *jobu, *jobv, M, N,
                    dA_array, ldda, dS_array,
                    dU_array, lddu, dV_array, lddv, dinfo_array,
                    gesvj_nb, gesvj_max_sweeps,
                    heevj_max_sweeps, heevj_tol, heevj_tol_min, heevj_tol_scal,
                    device_work, device_lwork,
                    batchCount, opts.queue );
                gpu_time = magma_sync_wtime( opts.queue ) - gpu_time;

                // free qworkspace
                if(device_work != NULL) magma_free( device_work );
            }

            magma_zgetmatrix( M, min_mn*batchCount, dU, lddu, hU, ldu, opts.queue );
            magma_zgetmatrix( N, batchCount*min_mn, dV, lddv, hV, ldv, opts.queue );
            magma_dgetvector( batchCount*min_mn, dS, 1, hS, 1, opts.queue );
            magma_getvector( batchCount, sizeof(magma_int_t), dinfo_array, 1, hinfo_array, 1, opts.queue);

            const char *func = "magma_zgesvj";
            // force check to fail if gesvd returns info error
            double result[5]        = { nan, nan, nan, nan, nan };
            check_zgesvj_batched(func, opts.check, *jobu, *jobv,
                                 M, N,
                                 hA_array, lda, hS_array,
                                 hU_array, ldu,
                                 hV_array, ldv,
                                 hinfo_array, batchCount,
                                 result, status);

            if ( opts.lapack ) {
                /* =====================================================================
                   Performs operation using LAPACK
                   =================================================================== */
                lapackf77_zlacpy( MagmaFullStr, &M, &NN, hA, &lda, hR, &lda );
                magma_flush_cache( opts.cache );
                cpu_time = magma_wtime();
                #if defined(_OPENMP)
                magma_int_t nthreads = magma_get_lapack_numthreads();
                magma_set_lapack_numthreads(1);
                magma_set_omp_numthreads(nthreads);
                #pragma omp parallel for schedule(dynamic)
                #endif
                for(magma_int_t s = 0; s < batchCount; s++) {
                    #ifdef CPU_GESVD
                    // gesvd returns vt instead of v
                    lapackf77_zgesvd( lapack_vec_const(jobu_cpu), lapack_vec_const(jobv_cpu), &M, &N,
                                      hR_array[s], &lda, hSref_array[s],
                                      hU_array[s], &ldu,
                                      hV_array[s], &ldvt,
                                      hwork_array[s], &lwork,
                                      #ifdef COMPLEX
                                      hrwork_array[s],
                                      #endif
                                      &hinfo_array[s]);
                    #else
                    lapackf77_zgesvj( "G", ju, jv, &M, &N,
                                      hR_array[s], &lda, hSref_array[s], &N,
                                      hV_array[s], &ldv,
                                      hwork_array[s], &lwork,
                                      #ifdef COMPLEX
                                      hrwork_array[s], &lrwork,
                                      #endif
                                      &hinfo_array[s]);
                    #endif
                }
                #if defined(_OPENMP)
                    magma_set_lapack_numthreads(nthreads);
                #endif
                cpu_time = magma_wtime() - cpu_time;

                /* =====================================================================
                   Check MAGMA's singular values compared to LAPACK
                   =================================================================== */
                if ( opts.magma ) {
                    double S_err;
                    result[4] = MAGMA_D_ZERO;
                    for(magma_int_t s = 0; s < batchCount; s++) {
                        blasf77_daxpy( &min_mn, &d_neg_one, hS_array[s], &ione, hSref_array[s], &ione );
                        S_err  = lapackf77_dlange( "f", &min_mn, &ione, hSref_array[s], &min_mn, work );
                        S_err /= ( min_mn * lapackf77_dlange( "f", &min_mn, &ione, hS_array[s],    &min_mn, work ) );
                        result[4]  = magma_max_nan( result[4], S_err);
                    }
                }
                printf( "   %c%c     %10lld    %5lld %5lld   %9.4f        %9.4f     ",
                        lapacke_vec_const(*jobu), lapacke_vec_const(*jobv),
                        (long long) batchCount, (long long) M, (long long) N,
                        cpu_time*1000., gpu_time*1000. );
            }
            else {
                result[4] = -1;  // indicates S - Sref not checked
                printf( "   %c%c     %10lld    %5lld %5lld      ---           %9.4f     ",
                        lapacke_vec_const(*jobu), lapacke_vec_const(*jobv),
                        (long long) batchCount, (long long) M, (long long) N,
                        gpu_time*1000. );
            }

            /* =====================================================================
               Print error checks
               =================================================================== */
            bool okay   = true;
            bool sorted = true;
            if ( opts.magma ) {
                if ( result[4] < 0. ) { printf(  "     ---   " ); } else { printf(  "   %8.2e", result[4]); }  // S - Sref
                if ( result[0] < 0. ) { printf( "      ---   " ); } else { printf( "    %8.2e", result[0]); }  // A - USV'
                if ( result[1] < 0. ) { printf( "      ---   " ); } else { printf( "    %8.2e", result[1]); }  // I - UU'
                if ( result[2] < 0. ) { printf( "      ---   " ); } else { printf( "    %8.2e", result[2]); }  // I - VV'
                okay = okay && (result[0] < tol) && (result[1] < tol)
                            && (result[2] < tol) && (result[3] == 0.)
                            && (result[4] < tol);
                sorted = sorted && (result[3] == 0.);
            }
            status += ! okay;
            printf( "   %-3s   %-6s", (sorted ? "yes" : "no"), (okay ? "ok" : "failed") );
            printf( "\n" );

            magma_free( dA );
            magma_free( dS );
            magma_free( dU );
            magma_free( dV );

            magma_free_cpu( hA );
            magma_free_cpu( hU );
            magma_free_cpu( hV );
            magma_free_cpu( hS );
            magma_free_cpu( hSref );
            magma_free_cpu( hR    );
            if(opts.lapack) {
                magma_free_cpu( hwork );
                #ifdef COMPLEX
                magma_free_cpu( hrwork );
                #endif
            }

            fflush( stdout );
        } // iter
        if ( opts.niter > 1 /*|| opts.svd_work.size() > 1*/ ) {
            printf( "\n" );
        }
      }}  // jobu, jobv
      if ( opts.jobu.size() > 1 || opts.jobv.size() > 1 ) {
          printf( "%%----------\n" );
      }
    }

    magma_free( dA_array );
    magma_free( dS_array );
    magma_free( dU_array );
    magma_free( dV_array );
    magma_free( dinfo_array );

    magma_free_cpu( hA_array );
    magma_free_cpu( hR_array );
    magma_free_cpu( hU_array );
    magma_free_cpu( hV_array );
    magma_free_cpu( hS_array );
    magma_free_cpu( hSref_array );
    magma_free_cpu( hinfo_array );
    if(opts.lapack) {
        magma_free_cpu( hwork_array );
        #ifdef COMPLEX
        magma_free_cpu( hrwork_array );
        #endif
    }

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
