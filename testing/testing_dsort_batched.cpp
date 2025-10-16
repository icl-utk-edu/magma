/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal d -> s
*/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>

// includes, project
#include "magma_v2.h"
#include "magma_lapack.h"
#include "magma_operators.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing sort_batched (double)
*/
int main(int argc, char **argv)
{
    #define  X(i_, j_)  ( X + (i_) + (j_)*lda)
    #define  Y(i_, j_)  ( Y + (i_) + (j_)*lda)

    #define dX(i_, j_)  (dX + (i_) + (j_)*ldda)
    #define dY(i_, j_)  (dY + (i_) + (j_)*ldda)

    TESTING_CHECK( magma_init() );
    magma_print_environment();

    real_Double_t   gbytes, dev_bw, dev_time, cpu_bw, cpu_time;
    magma_int_t ione     = 1;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t N, size_x, size_y, size_i;
    magma_int_t incx = 1;
    magma_int_t incy = 1;
    double *hX, *hY, *hYmagma;
    double *dX, *dY;
    double **dX_array, **dY_array;
    magma_int_t *dI;     // index
    magma_int_t **dI_array;
    int status = 0;

    magma_opts opts( MagmaOptsBatched );
    opts.parse_opts( argc, argv );

    magma_int_t batchCount = opts.batchcount;

    TESTING_CHECK( magma_malloc( (void**) &dX_array, batchCount * sizeof(double*) ));
    TESTING_CHECK( magma_malloc( (void**) &dY_array, batchCount * sizeof(double*) ));
    TESTING_CHECK( magma_malloc( (void**) &dI_array, batchCount * sizeof(magma_int_t*) ));

    std::vector<magma_sort_t> sort_direction = {MagmaAscending, MagmaDescending};

    printf("%% Sort   BatchCount       N       CPU GB/s (ms)   MAGMA GB/s (ms)  sorted?\n");
    printf("%%=========================================================================\n");
    for(auto isort = sort_direction.begin(); isort != sort_direction.end(); isort++) {
        for( int itest = 0; itest < opts.ntest; ++itest ) {
            for( int iter = 0; iter < opts.niter; ++iter ) {
                N = opts.msize[itest];
                gbytes = batchCount * N * sizeof(double) / 1e9;
                size_x = batchCount * N * incx;
                size_y = batchCount * N * incy;
                size_i = batchCount * N;

                TESTING_CHECK( magma_dmalloc_cpu( &hX,      size_x ));
                TESTING_CHECK( magma_dmalloc_cpu( &hY,      size_y ));
                TESTING_CHECK( magma_dmalloc_cpu( &hYmagma, size_y ));

                TESTING_CHECK( magma_dmalloc( &dX, size_x ));
                TESTING_CHECK( magma_dmalloc( &dY, size_y ));
                TESTING_CHECK( magma_imalloc( &dI, size_i ));

                /* Initialize the matrix */
                lapackf77_dlarnv( &ione, ISEED, &size_x, hX );

                // assign pointers
                magma_dset_pointer( dX_array, dX, N, 0, 0, N, batchCount, opts.queue );
                magma_dset_pointer( dY_array, dY, N, 0, 0, N, batchCount, opts.queue );
                magma_iset_pointer( dI_array, dI, N, 0, 0, N, batchCount, opts.queue );

                /* =====================================================================
                   Performs operation using MAGMA
                   =================================================================== */
                magma_dsetvector( batchCount * N, hX, incx, dX, incx, opts.queue );

                magma_flush_cache( opts.cache );
                dev_time = magma_sync_wtime( opts.queue );
                magmablas_dsort_batched(*isort, N, dX_array, incx, dY_array, incy, dI_array, batchCount, opts.queue);
                dev_time = magma_sync_wtime( opts.queue ) - dev_time;
                dev_bw = gbytes / dev_time;

                magma_dgetvector( batchCount * N, dY, incy, hYmagma, incy, opts.queue );

                /* =====================================================================
                   Performs operation using CPU BLAS
                   =================================================================== */
                // copy hX to hY
                memcpy(hY, hX, batchCount * N * sizeof(double));
                std::vector<double> Yvec(hY, hY + batchCount * N);

                magma_flush_cache( opts.cache );
                cpu_time = magma_wtime();
                if(*isort == MagmaAscending) {
                    #pragma omp parallel for schedule(dynamic)
                    for(magma_int_t s = 0; s < batchCount; s++) {
                        std::sort( Yvec.begin() + s * N, Yvec.begin() + (s+1)*N );
                    }
                }
                else {
                    #pragma omp parallel for schedule(dynamic)
                    for(magma_int_t s = 0; s < batchCount; s++) {
                        std::sort( Yvec.begin() + s * N, Yvec.begin() + (s+1)*N, std::greater<double>() );
                    }
                }

                cpu_time = magma_wtime() - cpu_time;
                cpu_bw = gbytes / cpu_time;


                /* =====================================================================
                   Check the result
                   =================================================================== */
                bool okay = true;
                for(magma_int_t i = 0; i < batchCount * N; i++) {
                    if( ! (hYmagma[i] == Yvec[i]) ) {
                        okay = false;
                        break;
                    }
                }

                status += ! okay;
                printf("%6s   ", (*isort == MagmaAscending) ? "A" : "D");
                printf("%10lld   %5lld   %7.2f (%7.2f)   %7.2f (%7.2f)  %s\n",
                       (long long) batchCount, (long long) N,
                       cpu_bw,    1000.*cpu_time,
                       dev_bw,    1000.*dev_time,
                       (okay ? "ok" : "failed"));

                magma_free_cpu( hX );
                magma_free_cpu( hY );
                magma_free_cpu( hYmagma );

                magma_free( dX );
                magma_free( dY );
                magma_free( dI );
                fflush( stdout );
            }

            if ( opts.niter > 1 ) {
                printf( "\n" );
            }
        }
        printf("\n");
    }

    magma_free( dX_array );
    magma_free( dY_array );
    magma_free( dI_array );

    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    return status;
}
