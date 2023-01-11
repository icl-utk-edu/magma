/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Mark Gates
       
       These tests ensure that the MAGMA wrappers around CUBLAS calls are
       correct, for example,
       magma_ztrmm(...) and cublasZtrmm(...) produce /exactly/ the same results.
       It tests all combinations of options (trans, uplo, ...) for each size.
*/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// make sure that asserts are enabled
#undef NDEBUG
#include <assert.h>

// includes, project
#include "flops.h"
#include "magma_v2.h"
#include "magma_lapack.h"
#include "testings.h"

#ifdef MAGMA_HAVE_SYCL
#include "../control/magma_internal.h" // to get sycl queue from magma queue struct
#endif

#define A(i,j)  &A[  (i) + (j)*ld ]
#define dA(i,j) &dA[ (i) + (j)*ld ]
#define dB(i,j) &dB[ (i) + (j)*ld ]
#define C2(i,j) &C2[ (i) + (j)*ld ]
#define LU(i,j) &LU[ (i) + (j)*ld ]

int main( int argc, char** argv )
{
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    
    real_Double_t   gflops, t1, t2;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magma_int_t ione = 1;
    magma_trans_t trans[] = { MagmaNoTrans, MagmaConjTrans, MagmaTrans };
    magma_uplo_t  uplo [] = { MagmaLower, MagmaUpper };
    magma_diag_t  diag [] = { MagmaUnit, MagmaNonUnit };
    magma_side_t  side [] = { MagmaLeft, MagmaRight };
    
    magmaDoubleComplex  *A,  *B,  *C,   *C2, *LU;
    magmaDoubleComplex_ptr dA, dB, dC1, dC2;
    magmaDoubleComplex alpha = MAGMA_Z_MAKE( 0.5, 0.1 );
    magmaDoubleComplex beta  = MAGMA_Z_MAKE( 0.7, 0.2 );
    double dalpha = 0.6;
    double dbeta  = 0.8;
    double work[1], error, total_error;
    magma_int_t ISEED[4] = {0,0,0,1};
    magma_int_t m, n, k, size, maxn, ld, info;
    magma_int_t *piv;
    
    magma_opts opts;
    opts.parse_opts( argc, argv );
    
    #if defined(MAGMA_HAVE_CUDA) || defined(MAGMA_HAVE_SYCL)
    printf( "Compares magma wrapper function to cublas function; all diffs should be exactly 0.\n\n" );
    
    total_error = 0.;
    for( int itest = 0; itest < opts.ntest; ++itest ) {
        m = opts.msize[itest];
        n = opts.nsize[itest];
        k = opts.ksize[itest];
        printf("%%========================================================================\n");
        printf( "m=%lld, n=%lld, k=%lld\n", (long long) m, (long long) n, (long long) k );
        
        // allocate matrices
        // over-allocate so they can be any combination of {m,n,k} x {m,n,k}.
        maxn = max( max( m, n ), k );
        ld = max( 1, maxn );
        size = ld*maxn;
        TESTING_CHECK( magma_imalloc_cpu( &piv, maxn ));
        
        TESTING_CHECK( magma_zmalloc_pinned( &A,   size ));
        TESTING_CHECK( magma_zmalloc_pinned( &B,   size ));
        TESTING_CHECK( magma_zmalloc_pinned( &C,   size ));
        TESTING_CHECK( magma_zmalloc_pinned( &C2,  size ));
        TESTING_CHECK( magma_zmalloc_pinned( &LU,  size ));
        
        TESTING_CHECK( magma_zmalloc( &dA,  size ));
        TESTING_CHECK( magma_zmalloc( &dB,  size ));
        TESTING_CHECK( magma_zmalloc( &dC1, size ));
        TESTING_CHECK( magma_zmalloc( &dC2, size ));
        
        // initialize matrices
        size = maxn*maxn;
        lapackf77_zlarnv( &ione, ISEED, &size, A  );
        lapackf77_zlarnv( &ione, ISEED, &size, B  );
        lapackf77_zlarnv( &ione, ISEED, &size, C  );
        
        printf( "%%========= Level 1 BLAS ==========\n" );
        
        // ----- test ZSWAP
        // swap columns 2 and 3 of dA, then copy to C2 and compare with A
        if ( n >= 3 ) {
            magma_zsetmatrix( m, n, A, ld, dA, ld, opts.queue );
            magma_zsetmatrix( m, n, A, ld, dB, ld, opts.queue );
            magma_zswap( m, dA(0,1), 1, dA(0,2), 1, opts.queue );
            magma_zswap( m, dB(0,1), 1, dB(0,2), 1, opts.queue );
            #ifdef MAGMA_HAVE_CUDA
            // check results, storing diff between magma and cuda calls in C2
            cublasZaxpy( opts.handle, int(ld*n), &c_neg_one, dA, 1, dB, 1 );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::axpy(
                *opts.handle, int(ld * n),
                c_neg_one,
                (magmaDoubleComplex *)dA, 1, (magmaDoubleComplex *)dB, 1);
            #endif
            magma_zgetmatrix( m, n, dB, ld, C2, ld, opts.queue );
            error = lapackf77_zlange( "F", &m, &k, C2, &ld, work );
            total_error += error;
            printf( "zswap             diff %.2g\n", error );
        }
        else {
            printf( "zswap skipped for n < 3\n" );
        }
        
        // ----- test IZAMAX
        // get argmax of column of A
        magma_zsetmatrix( m, k, A, ld, dA, ld, opts.queue );
        error = 0;
        for( int j = 0; j < k; ++j ) {
            magma_int_t i1 = magma_izamax( m, dA(0,j), 1, opts.queue );
            int i2;  // not magma_int_t
            #ifdef MAGMA_HAVE_CUDA
            cublasIzamax( opts.handle, int(m), dA(0,j), 1, &i2 );
            #elif defined(MAGMA_HAVE_SYCL)
	    int64_t *res_temp_ptr_ct1 =
            sycl::malloc_shared<int64_t>(1, *opts.queue->sycl_stream());
            oneapi::mkl::blas::column_major::iamax(*opts.handle, int(m),
                                             (magmaDoubleComplex *)dA(0, j),
                                             1, res_temp_ptr_ct1)
            .wait();
            int res_temp_host_ct2 = (int)*res_temp_ptr_ct1;
            dpct::dpct_memcpy(&i2, &res_temp_host_ct2, sizeof(int));
            sycl::free(res_temp_ptr_ct1, *opts.queue->sycl_stream());
            #endif
            // todo need sync here?
            assert( i1 == i2 );
            error += abs( i1 - i2 );
        }
        total_error += error;
        gflops = (double)m * k / 1e9;
        printf( "izamax            diff %.2g\n", error );
        printf( "\n" );
        
        printf( "%%========= Level 2 BLAS ==========\n" );
        
        // ----- test ZGEMV
        // c = alpha*A*b + beta*c,  with A m*n; b,c m or n-vectors
        // try no-trans/trans
        for( int ia = 0; ia < 3; ++ia ) {
            magma_zsetmatrix( m, n, A,  ld, dA,  ld, opts.queue );
            magma_zsetvector( maxn, B, 1, dB,  1, opts.queue );
            magma_zsetvector( maxn, C, 1, dC1, 1, opts.queue );
            magma_zsetvector( maxn, C, 1, dC2, 1, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_zgemv( trans[ia], m, n, alpha, dA, ld, dB, 1, beta, dC1, 1, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            #ifdef MAGMA_HAVE_CUDA
            cublasZgemv( opts.handle, trans_const(trans[ia]),
                         int(m), int(n), &alpha, dA, int(ld), dB, 1, &beta, dC2, 1 );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::gemv(
                *opts.handle, syclblas_trans_const(trans[ia]), int(m), int(n),
                alpha,
                (magmaDoubleComplex *)dA, int(ld), (magmaDoubleComplex *)dB,
                1, beta,
                (magmaDoubleComplex *)dC2, 1);
            #endif
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            size = (trans[ia] == MagmaNoTrans ? m : n);
            #ifdef MAGMA_HAVE_CUDA
            cublasZaxpy( opts.handle, int(size), &c_neg_one, dC1, 1, dC2, 1 );
            #elif defined(MAGMA_HAVE_SYCL)
            oneapi::mkl::blas::column_major::axpy(
                *opts.handle, int(size),
                c_neg_one,
                (magmaDoubleComplex *)dC1, 1, (magmaDoubleComplex *)dC2, 1);
            #endif
            magma_zgetvector( size, dC2, 1, C2, 1, opts.queue );
            error = lapackf77_zlange( "F", &size, &ione, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_ZGEMV( m, n ) / 1e9;
            printf( "zgemv( %c )        diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_trans_const(trans[ia]), error, gflops/t1, gflops/t2 );
        }
        printf( "\n" );
        
        // ----- test ZHEMV
        // c = alpha*A*b + beta*c,  with A m*m symmetric; b,c m-vectors
        // try upper/lower
        for( int iu = 0; iu < 2; ++iu ) {
            magma_zsetmatrix( m, m, A, ld, dA, ld, opts.queue );
            magma_zsetvector( m, B, 1, dB,  1, opts.queue );
            magma_zsetvector( m, C, 1, dC1, 1, opts.queue );
            magma_zsetvector( m, C, 1, dC2, 1, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_zhemv( uplo[iu], m, alpha, dA, ld, dB, 1, beta, dC1, 1, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            #ifdef MAGMA_HAVE_CUDA
            cublasZhemv( opts.handle, uplo_const(uplo[iu]),
                         int(m), &alpha, dA, int(ld), dB, 1, &beta, dC2, 1 );
            #elif defined(MAGMA_HAVE_SYCL)
            oneapi::mkl::blas::column_major::hemv(
                *opts.handle, syclblas_uplo_const(uplo[iu]), int(m),
                alpha,
                (magmaDoubleComplex *)dA, int(ld), (magmaDoubleComplex *)dB,
                1, beta,
                (magmaDoubleComplex *)dC2, 1);
            #endif
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            #ifdef MAGMA_HAVE_CUDA
            cublasZaxpy( opts.handle, int(m), &c_neg_one, dC1, 1, dC2, 1 );
            #elif defined(MAGMA_HAVE_SYCL)
            oneapi::mkl::blas::column_major::axpy(
                *opts.handle, int(m),
		c_neg_one,
                (magmaDoubleComplex *)dC1, 1, (magmaDoubleComplex *)dC2, 1);
            #endif
            magma_zgetvector( m, dC2, 1, C2, 1, opts.queue );
            error = lapackf77_zlange( "F", &m, &ione, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_ZHEMV( m ) / 1e9;
            printf( "zhemv( %c )        diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_uplo_const(uplo[iu]), error, gflops/t1, gflops/t2 );
        }
        printf( "\n" );
        
        // ----- test ZTRSV
        // solve A*c = c,  with A m*m triangular; c m-vector
        // try upper/lower, no-trans/trans, unit/non-unit diag
        // Factor A into LU to get well-conditioned triangles, else solve yields garbage.
        // Still can give garbage if solves aren't consistent with LU factors,
        // e.g., using unit diag for U, so copy lower triangle to upper triangle.
        // Also used for trsm later.
        lapackf77_zlacpy( "Full", &maxn, &maxn, A, &ld, LU, &ld );
        lapackf77_zgetrf( &maxn, &maxn, LU, &ld, piv, &info );
        for( int j = 0; j < maxn; ++j ) {
            for( int i = 0; i < j; ++i ) {
                *LU(i,j) = *LU(j,i);
            }
        }
        for( int iu = 0; iu < 2; ++iu ) {
        for( int it = 0; it < 3; ++it ) {
        for( int id = 0; id < 2; ++id ) {
            magma_zsetmatrix( m, m, LU, ld, dA, ld, opts.queue );
            magma_zsetvector( m, C, 1, dC1, 1, opts.queue );
            magma_zsetvector( m, C, 1, dC2, 1, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_ztrsv( uplo[iu], trans[it], diag[id], m, dA, ld, dC1, 1, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            #ifdef MAGMA_HAVE_CUDA
	    cublasZtrsv( opts.handle, uplo_const(uplo[iu]), cublas_trans_const(trans[it]),
                         diag_const(diag[id]), int(m), dA, int(ld), dC2, 1 );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::trsv(
                *opts.handle, syclblas_uplo_const(uplo[iu]),
                syclblas_trans_const(trans[it]), syclblas_diag_const(diag[id]),
                int(m), (magmaDoubleComplex *)dA, int(ld),
                (magmaDoubleComplex *)dC2, 1);
            #endif
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            #ifdef MAGMA_HAVE_CUDA
            cublasZaxpy( opts.handle, int(m), &c_neg_one, dC1, 1, dC2, 1 );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::axpy(
                *opts.handle, int(m),
                c_neg_one,
                (magmaDoubleComplex *)dC1, 1, (magmaDoubleComplex *)dC2, 1);
            #endif
            magma_zgetvector( m, dC2, 1, C2, 1, opts.queue );
            error = lapackf77_zlange( "F", &m, &ione, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_ZTRSM( MagmaLeft, m, 1 ) / 1e9;
            printf( "ztrsv( %c, %c, %c )  diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_uplo_const(uplo[iu]), lapacke_trans_const(trans[it]), lapacke_diag_const(diag[id]),
                    error, gflops/t1, gflops/t2 );
        }}}
        printf( "\n" );
        
        printf( "%%========= Level 3 BLAS ==========\n" );
        
        // ----- test ZGEMM
        // C = alpha*A*B + beta*C,  with A m*k or k*m; B k*n or n*k; C m*n
        // try combinations of no-trans/trans
        for( int ia = 0; ia < 3; ++ia ) {
        for( int ib = 0; ib < 3; ++ib ) {
            bool nta = (trans[ia] == MagmaNoTrans);
            bool ntb = (trans[ib] == MagmaNoTrans);
            magma_zsetmatrix( (nta ? m : k), (nta ? m : k), A, ld, dA,  ld, opts.queue );
            magma_zsetmatrix( (ntb ? k : n), (ntb ? n : k), B, ld, dB,  ld, opts.queue );
            magma_zsetmatrix( m, n, C, ld, dC1, ld, opts.queue );
            magma_zsetmatrix( m, n, C, ld, dC2, ld, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_zgemm( trans[ia], trans[ib], m, n, k, alpha, dA, ld, dB, ld, beta, dC1, ld, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            #ifdef MAGMA_HAVE_CUDA
            cublasZgemm( opts.handle, trans_const(trans[ia]), cublas_trans_const(trans[ib]),
                         int(m), int(n), int(k), &alpha, dA, int(ld), dB, int(ld), &beta, dC2, int(ld) );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::gemm(
                *opts.handle, syclblas_trans_const(trans[ia]),
                syclblas_trans_const(trans[ib]), int(m), int(n), int(k),
                alpha,
                (magmaDoubleComplex *)dA, int(ld), (magmaDoubleComplex *)dB,
                int(ld), beta,
                (magmaDoubleComplex *)dC2, int(ld));
            #endif
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            #ifdef MAGMA_HAVE_CUDA
            cublasZaxpy( opts.handle, int(ld*n), &c_neg_one, dC1, 1, dC2, 1 );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::axpy(
                *opts.handle, int(ld * n),
                c_neg_one,
                (magmaDoubleComplex *)dC1, 1, (magmaDoubleComplex *)dC2, 1);
            #endif
            magma_zgetmatrix( m, n, dC2, ld, C2, ld, opts.queue );
            error = lapackf77_zlange( "F", &m, &n, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_ZGEMM( m, n, k ) / 1e9;
            printf( "zgemm( %c, %c )     diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_trans_const(trans[ia]), lapacke_trans_const(trans[ib]),
                    error, gflops/t1, gflops/t2 );
        }}
        printf( "\n" );
        
        // ----- test ZHEMM
        // C = alpha*A*B + beta*C  (left)  with A m*m symmetric; B,C m*n; or
        // C = alpha*B*A + beta*C  (right) with A n*n symmetric; B,C m*n
        // try left/right, upper/lower
        for( int is = 0; is < 2; ++is ) {
        for( int iu = 0; iu < 2; ++iu ) {
            magma_zsetmatrix( m, m, A, ld, dA,  ld, opts.queue );
            magma_zsetmatrix( m, n, B, ld, dB,  ld, opts.queue );
            magma_zsetmatrix( m, n, C, ld, dC1, ld, opts.queue );
            magma_zsetmatrix( m, n, C, ld, dC2, ld, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_zhemm( side[is], uplo[iu], m, n, alpha, dA, ld, dB, ld, beta, dC1, ld, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            #ifdef MAGMA_HAVE_CUDA
            cublasZhemm( opts.handle, side_const(side[is]), cublas_uplo_const(uplo[iu]),
                         int(m), int(n), &alpha, dA, int(ld), dB, int(ld), &beta, dC2, int(ld) );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::hemm(
                *opts.handle, syclblas_side_const(side[is]),
                syclblas_uplo_const(uplo[iu]), int(m), int(n),
                alpha,
                (magmaDoubleComplex *)dA, int(ld), (magmaDoubleComplex *)dB,
                int(ld), beta,
                (magmaDoubleComplex *)dC2, int(ld));
            #endif
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            #ifdef MAGMA_HAVE_CUDA
            cublasZaxpy( opts.handle, int(ld*n), &c_neg_one, dC1, 1, dC2, 1 );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::axpy(
                *opts.handle, int(ld * n),
                c_neg_one,
                (magmaDoubleComplex *)dC1, 1, (magmaDoubleComplex *)dC2, 1);
            #endif
            magma_zgetmatrix( m, n, dC2, ld, C2, ld, opts.queue );
            error = lapackf77_zlange( "F", &m, &n, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_ZHEMM( side[is], m, n ) / 1e9;
            printf( "zhemm( %c, %c )     diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_side_const(side[is]), lapacke_uplo_const(uplo[iu]),
                    error, gflops/t1, gflops/t2 );
        }}
        printf( "\n" );
        
        // ----- test ZHERK
        // C = alpha*A*A^H + beta*C  (no-trans) with A m*k and C m*m symmetric; or
        // C = alpha*A^H*A + beta*C  (trans)    with A k*m and C m*m symmetric
        // try upper/lower, no-trans/trans
        for( int iu = 0; iu < 2; ++iu ) {
        for( int it = 0; it < 3; ++it ) {
            magma_zsetmatrix( n, k, A, ld, dA,  ld, opts.queue );
            magma_zsetmatrix( n, n, C, ld, dC1, ld, opts.queue );
            magma_zsetmatrix( n, n, C, ld, dC2, ld, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_zherk( uplo[iu], trans[it], n, k, dalpha, dA, ld, dbeta, dC1, ld, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            #ifdef MAGMA_HAVE_CUDA
            cublasZherk( opts.handle, uplo_const(uplo[iu]), cublas_trans_const(trans[it]),
                         int(n), int(k), &dalpha, dA, int(ld), &dbeta, dC2, int(ld) );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::herk(
                *opts.handle, syclblas_uplo_const(uplo[iu]),
                syclblas_trans_const(trans[it]), int(n), int(k), dalpha,
                (magmaDoubleComplex *)dA, int(ld), dbeta,
                (magmaDoubleComplex *)dC2, int(ld));
            #endif
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            #ifdef MAGMA_HAVE_CUDA
            cublasZaxpy( opts.handle, int(ld*n), &c_neg_one, dC1, 1, dC2, 1 );
            #elif defined(MAGMA_HAVE_SYCL)
            oneapi::mkl::blas::column_major::axpy(
                *opts.handle, int(ld * n),
                c_neg_one,
                (magmaDoubleComplex *)dC1, 1, (magmaDoubleComplex *)dC2, 1);
            #endif
            magma_zgetmatrix( n, n, dC2, ld, C2, ld, opts.queue );
            error = lapackf77_zlange( "F", &n, &n, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_ZHERK( k, n ) / 1e9;
            printf( "zherk( %c, %c )     diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_uplo_const(uplo[iu]), lapacke_trans_const(trans[it]),
                    error, gflops/t1, gflops/t2 );
        }}
        printf( "\n" );
        
        // ----- test ZHER2K
        // C = alpha*A*B^H + ^alpha*B*A^H + beta*C  (no-trans) with A,B n*k; C n*n symmetric; or
        // C = alpha*A^H*B + ^alpha*B^H*A + beta*C  (trans)    with A,B k*n; C n*n symmetric
        // try upper/lower, no-trans/trans
        for( int iu = 0; iu < 2; ++iu ) {
        for( int it = 0; it < 3; ++it ) {
            bool nt = (trans[it] == MagmaNoTrans);
            magma_zsetmatrix( (nt ? n : k), (nt ? n : k), A, ld, dA,  ld, opts.queue );
            magma_zsetmatrix( n, n, C, ld, dC1, ld, opts.queue );
            magma_zsetmatrix( n, n, C, ld, dC2, ld, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_zher2k( uplo[iu], trans[it], n, k, alpha, dA, ld, dB, ld, dbeta, dC1, ld, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            #ifdef MAGMA_HAVE_CUDA
            cublasZher2k( opts.handle, uplo_const(uplo[iu]), cublas_trans_const(trans[it]),
                          int(n), int(k), &alpha, dA, int(ld), dB, int(ld), &dbeta, dC2, int(ld) );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::her2k(
                *opts.handle, syclblas_uplo_const(uplo[iu]),
                syclblas_trans_const(trans[it]), int(n), int(k),
                alpha,
                (magmaDoubleComplex *)dA, int(ld), (magmaDoubleComplex *)dB,
                int(ld), dbeta, (magmaDoubleComplex *)dC2, int(ld));
            #endif
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            #ifdef MAGMA_HAVE_CUDA
            cublasZaxpy( opts.handle, int(ld*n), &c_neg_one, dC1, 1, dC2, 1 );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::axpy(
                *opts.handle, int(ld * n),
                c_neg_one,
                (magmaDoubleComplex *)dC1, 1, (magmaDoubleComplex *)dC2, 1);
            #endif
            magma_zgetmatrix( n, n, dC2, ld, C2, ld, opts.queue );
            error = lapackf77_zlange( "F", &n, &n, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_ZHER2K( k, n ) / 1e9;
            printf( "zher2k( %c, %c )    diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_uplo_const(uplo[iu]), lapacke_trans_const(trans[it]),
                    error, gflops/t1, gflops/t2 );
        }}
        printf( "\n" );
        
        // ----- test ZTRMM
        // C = alpha*A*C  (left)  with A m*m triangular; C m*n; or
        // C = alpha*C*A  (right) with A n*n triangular; C m*n
        // try left/right, upper/lower, no-trans/trans, unit/non-unit
        for( int is = 0; is < 2; ++is ) {
        for( int iu = 0; iu < 2; ++iu ) {
        for( int it = 0; it < 3; ++it ) {
        for( int id = 0; id < 2; ++id ) {
            bool left = (side[is] == MagmaLeft);
            magma_zsetmatrix( (left ? m : n), (left ? m : n), A, ld, dA,  ld, opts.queue );
            magma_zsetmatrix( m, n, C, ld, dC1, ld, opts.queue );
            magma_zsetmatrix( m, n, C, ld, dC2, ld, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_ztrmm( side[is], uplo[iu], trans[it], diag[id], m, n, alpha, dA, ld, dC1, ld, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            // note cublas does trmm out-of-place (i.e., adds output matrix C),
            // but allows C=B to do in-place.
            t2 = magma_sync_wtime( opts.queue );
            #ifdef MAGMA_HAVE_CUDA
            cublasZtrmm( opts.handle, side_const(side[is]), cublas_uplo_const(uplo[iu]),
                         trans_const(trans[it]), cublas_diag_const(diag[id]),
                         int(m), int(n), &alpha, dA, int(ld), dC2, int(ld), dC2, int(ld) );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::trmm(
                *opts.handle, syclblas_side_const(side[is]),
                syclblas_uplo_const(uplo[iu]), syclblas_trans_const(trans[it]),
                syclblas_diag_const(diag[id]), int(m), int(n),
                alpha,
                (magmaDoubleComplex *)dA, int(ld),
                (magmaDoubleComplex *)dC2, int(ld));
            #endif
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            #ifdef MAGMA_HAVE_CUDA
            cublasZaxpy( opts.handle, int(ld*n), &c_neg_one, dC1, 1, dC2, 1 );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::axpy(
                *opts.handle, int(ld * n),
                c_neg_one,
                (magmaDoubleComplex *)dC1, 1, (magmaDoubleComplex *)dC2, 1);
            #endif
            magma_zgetmatrix( m, n, dC2, ld, C2, ld, opts.queue );
            error = lapackf77_zlange( "F", &n, &n, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_ZTRMM( side[is], m, n ) / 1e9;
            printf( "ztrmm( %c, %c )     diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_uplo_const(uplo[iu]), lapacke_trans_const(trans[it]),
                    error, gflops/t1, gflops/t2 );
        }}}}
        printf( "\n" );
        
        // ----- test ZTRSM
        // solve A*X = alpha*B  (left)  with A m*m triangular; B m*n; or
        // solve X*A = alpha*B  (right) with A n*n triangular; B m*n
        // try left/right, upper/lower, no-trans/trans, unit/non-unit
        for( int is = 0; is < 2; ++is ) {
        for( int iu = 0; iu < 2; ++iu ) {
        for( int it = 0; it < 3; ++it ) {
        for( int id = 0; id < 2; ++id ) {
            bool left = (side[is] == MagmaLeft);
            magma_zsetmatrix( (left ? m : n), (left ? m : n), LU, ld, dA,  ld, opts.queue );
            magma_zsetmatrix( m, n, C, ld, dC1, ld, opts.queue );
            magma_zsetmatrix( m, n, C, ld, dC2, ld, opts.queue );
            
            t1 = magma_sync_wtime( opts.queue );
            magma_ztrsm( side[is], uplo[iu], trans[it], diag[id], m, n, alpha, dA, ld, dC1, ld, opts.queue );
            t1 = magma_sync_wtime( opts.queue ) - t1;
            
            t2 = magma_sync_wtime( opts.queue );
            #ifdef MAGMA_HAVE_CUDA
            cublasZtrsm( opts.handle, side_const(side[is]), cublas_uplo_const(uplo[iu]),
                         trans_const(trans[it]), cublas_diag_const(diag[id]),
                         int(m), int(n), &alpha, dA, int(ld), dC2, int(ld) );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::trsm(
                *opts.handle, syclblas_side_const(side[is]),
                syclblas_uplo_const(uplo[iu]), syclblas_trans_const(trans[it]),
                syclblas_diag_const(diag[id]), int(m), int(n),
                alpha,
                (magmaDoubleComplex *)dA, int(ld),
                (magmaDoubleComplex *)dC2, int(ld));
            #endif
            t2 = magma_sync_wtime( opts.queue ) - t2;
            
            // check results, storing diff between magma and cuda call in C2
            #ifdef MAGMA_HAVE_CUDA
            cublasZaxpy( opts.handle, int(ld*n), &c_neg_one, dC1, 1, dC2, 1 );
            #elif defined(MAGMA_HAVE_SYCL)
	    oneapi::mkl::blas::column_major::axpy(
                *opts.handle, int(ld * n),
                c_neg_one,
                (magmaDoubleComplex *)dC1, 1, (magmaDoubleComplex *)dC2, 1);
            #endif
            magma_zgetmatrix( m, n, dC2, ld, C2, ld, opts.queue );
            error = lapackf77_zlange( "F", &n, &n, C2, &ld, work );
            total_error += error;
            gflops = FLOPS_ZTRSM( side[is], m, n ) / 1e9;
            printf( "ztrsm( %c, %c )     diff %.2g,  Gflop/s %7.2f, %7.2f\n",
                    lapacke_uplo_const(uplo[iu]), lapacke_trans_const(trans[it]),
                    error, gflops/t1, gflops/t2 );
        }}}}
        printf( "\n" );
        
        // cleanup
        magma_free_cpu( piv );
        magma_free_pinned( A   );
        magma_free_pinned( B   );
        magma_free_pinned( C   );
        magma_free_pinned( C2  );
        magma_free_pinned( LU  );
        magma_free( dA  );
        magma_free( dB  );
        magma_free( dC1 );
        magma_free( dC2 );
        fflush( stdout );
    }
   
    #else
    
    printf("Not checking for exact error==0.0, since functions may not be direct wrappers on HIP\n");

    #endif
 
    if ( total_error != 0. ) {
        printf( "total error %.2g -- ought to be 0 -- some test failed (see above).\n",
                total_error );
    }
    else {
        printf( "all tests passed\n" );
    }
    
    opts.cleanup();
    TESTING_CHECK( magma_finalize() );
    
    int status = (total_error != 0.);
    return status;
}
