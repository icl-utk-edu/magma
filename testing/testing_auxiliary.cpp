/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/
#include <stdlib.h>
#include <stdio.h>

// tests internal routines: magma_{set,get}_lapack_numthreads, magma_get_parallel_numthreads
// so include magma_internal.h instead of magma_v2.h
#include "../control/magma_internal.h"  // internal header


/******************************************************************************/
// warn( condition ) is like assert, but doesn't abort. Also counts number of failures.
magma_int_t gFailures = 0;

void warn_helper( int cond, const char* str, const char* file, int line )
{
    if ( ! cond ) {
        printf( "*** testing_auxiliary error: %s:%d: assertion %s failed\n", file, line, str );
        gFailures += 1;
    }
}

#define warn(x) warn_helper( (x), #x, __FILE__, __LINE__ )


/******************************************************************************/
void test_num_gpus()
{
    printf( "%%=====================================================================\n%s\n", __func__ );
    
    magma_int_t ngpu;
    magma_device_t devices[ MagmaMaxGPUs ];
    magma_getdevices( devices, MagmaMaxGPUs, &ngpu );
    magma_int_t maxgpu = min( ngpu, MagmaMaxGPUs );
    
    printf( "$MAGMA_NUM_GPUS     ngpu     expect\n" );
    printf( "%%==================================\n" );
    
#ifndef _MSC_VER // not Windows
    
    unsetenv("MAGMA_NUM_GPUS");
    ngpu = magma_num_gpus();
    printf( "%-18s  %7lld  %6lld\n\n", "not set", (long long) ngpu, (long long) 1 );
    warn( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_GPUS"), (long long) ngpu, (long long) 1 );
    warn( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "-1", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_GPUS"), (long long) ngpu, (long long) 1 );
    warn( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "2junk", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_GPUS"), (long long) ngpu, (long long) 1 );
    warn( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "0", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_GPUS"), (long long) ngpu, (long long) 1 );
    warn( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "1", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_GPUS"), (long long) ngpu, (long long) 1 );
    warn( ngpu == 1 );
    
    setenv("MAGMA_NUM_GPUS", "2", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_GPUS"), (long long) ngpu, (long long) 2 );
    warn( ngpu == min(  2, maxgpu ) );
    
    setenv("MAGMA_NUM_GPUS", "4", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_GPUS"), (long long) ngpu, (long long) 4 );
    warn( ngpu == min(  4, maxgpu ) );
    
    setenv("MAGMA_NUM_GPUS", "8", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_GPUS"), (long long) ngpu, (long long) 8 );
    warn( ngpu == min(  8, maxgpu ) );
    
    setenv("MAGMA_NUM_GPUS", "16", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_GPUS"), (long long) ngpu, (long long) 16 );
    warn( ngpu == min( 16, maxgpu ) );
    
    setenv("MAGMA_NUM_GPUS", "1000", 1 );
    ngpu = magma_num_gpus();
    printf( "%-18s  %7lld  %6lld (maxgpu)\n\n", getenv("MAGMA_NUM_GPUS"), (long long) ngpu, (long long) maxgpu );
    warn( ngpu == min( 1000, maxgpu ) );
    
#endif // not Windows
}


/******************************************************************************/
void test_num_threads()
{
    printf( "%%=====================================================================\n%s\n", __func__ );
    
    // test that getting & setting numthreads works
    magma_int_t p_nthread_orig = magma_get_parallel_numthreads();
    magma_int_t l_nthread_orig = magma_get_lapack_numthreads();
    printf( "get;      parallel_numthread=%2lld, lapack_numthread=%2lld\n",
            (long long) p_nthread_orig, (long long) l_nthread_orig );
    
    magma_set_lapack_numthreads( 4 );
    magma_int_t p_nthread = magma_get_parallel_numthreads();
    magma_int_t l_nthread = magma_get_lapack_numthreads();
    printf( "set( 4);  parallel_numthread=%2lld, lapack_numthread=%2lld (expect  4)\n",
            (long long) p_nthread, (long long) l_nthread );
    warn( p_nthread == p_nthread_orig );
    warn( l_nthread == 4 );
    
    magma_set_lapack_numthreads( 1 );
    p_nthread = magma_get_parallel_numthreads();
    l_nthread = magma_get_lapack_numthreads();
    printf( "set( 1);  parallel_numthread=%2lld, lapack_numthread=%2lld (expect  1)\n",
            (long long) p_nthread, (long long) l_nthread );
    warn( p_nthread == p_nthread_orig );
    warn( l_nthread == 1 );
    
    magma_set_lapack_numthreads( 8 );
    p_nthread = magma_get_parallel_numthreads();
    l_nthread = magma_get_lapack_numthreads();
    printf( "set( 8);  parallel_numthread=%2lld, lapack_numthread=%2lld (expect  8)\n",
            (long long) p_nthread, (long long) l_nthread );
    warn( p_nthread == p_nthread_orig );
    warn( l_nthread == 8 );
    
    magma_set_lapack_numthreads( l_nthread_orig );
    p_nthread = magma_get_parallel_numthreads();
    l_nthread = magma_get_lapack_numthreads();
    printf( "set(%2lld);  parallel_numthread=%2lld, lapack_numthread=%2lld (expect %2lld)\n",
            (long long) l_nthread_orig, (long long) p_nthread, (long long) l_nthread, (long long) l_nthread_orig );
    warn( p_nthread == p_nthread_orig );
    warn( l_nthread == l_nthread_orig );
    
#ifndef _MSC_VER // not Windows
    // test that parsing MAGMA_NUM_THREADS works
    
    // TODO need some way to get ncores. This is circular: assume with huge
    // NUM_THREADS that the routine gives the ncores. The user can verify.
    setenv("MAGMA_NUM_THREADS", "10000", 1 );
    magma_int_t ncores = magma_get_parallel_numthreads();
    
    magma_int_t omp_threads = ncores;
    const char* omp_str = getenv("OMP_NUM_THREADS");
    if ( omp_str != NULL ) {
        omp_threads = atoi( omp_str );
    }
    
    printf( "\nusing ncores=%lld, omp_num_threads=%lld\n\n", (long long) ncores, (long long) omp_threads );
    
    printf( "$MAGMA_NUM_THREADS  nthread  expect\n" );
    printf( "%%==================================\n" );
    
    unsetenv("MAGMA_NUM_THREADS");
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7lld  %6lld (omp_threads)\n\n", "not set", (long long) p_nthread, (long long) omp_threads );
    warn( p_nthread == omp_threads );
    
    setenv("MAGMA_NUM_THREADS", "", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_THREADS"), (long long) p_nthread, (long long) 1 );
    warn( p_nthread == 1 );
    
    setenv("MAGMA_NUM_THREADS", "-1", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_THREADS"), (long long) p_nthread, (long long) 1 );
    warn( p_nthread == 1 );
    
    setenv("MAGMA_NUM_THREADS", "2junk", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_THREADS"), (long long) p_nthread, (long long) 1 );
    warn( p_nthread == 1 );
    
    setenv("MAGMA_NUM_THREADS", "0", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_THREADS"), (long long) p_nthread, (long long) 1 );
    warn( p_nthread == 1 );
    
    setenv("MAGMA_NUM_THREADS", "1", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_THREADS"), (long long) p_nthread, (long long) 1 );
    warn( p_nthread == 1 );
    
    setenv("MAGMA_NUM_THREADS", "2", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_THREADS"), (long long) p_nthread, (long long) 2 );
    warn( p_nthread == min(  2, ncores ) );
    
    setenv("MAGMA_NUM_THREADS", "4", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_THREADS"), (long long) p_nthread, (long long) 4 );
    warn( p_nthread == min(  4, ncores ) );
    
    setenv("MAGMA_NUM_THREADS", "8", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_THREADS"), (long long) p_nthread, (long long) 8 );
    warn( p_nthread == min(  8, ncores ) );
    
    setenv("MAGMA_NUM_THREADS", "16", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7lld  %6lld\n\n", getenv("MAGMA_NUM_THREADS"), (long long) p_nthread, (long long) 16 );
    warn( p_nthread == min( 16, ncores ) );
    
    setenv("MAGMA_NUM_THREADS", "1000", 1 );
    p_nthread = magma_get_parallel_numthreads();
    printf( "%-18s  %7lld  %6lld (ncores)\n\n", getenv("MAGMA_NUM_THREADS"), (long long) p_nthread, (long long) ncores );
    warn( p_nthread == min( 1000, ncores ) );
#endif // not Windows
}


/******************************************************************************/
void test_xerbla()
{
    printf( "%%=====================================================================\n%s\n", __func__ );
    magma_int_t info;
    info = -MAGMA_ERR_DEVICE_ALLOC;  magma_xerbla( __func__, -(info) );
    info = -MAGMA_ERR_HOST_ALLOC;    magma_xerbla( __func__, -(info) );
    info = -MAGMA_ERR;               magma_xerbla( __func__, -(info) );
    info =  3;                       magma_xerbla( __func__, -(info) );
    info =  2;                       magma_xerbla( __func__, -(info) );
    info =  1;                       magma_xerbla( __func__, -(info) );
    info =  0;                       magma_xerbla( __func__, -(info) );
    info = -1;                       magma_xerbla( __func__, -(info) );
    info = -2;                       magma_xerbla( __func__, -(info) );
    info = -3;                       magma_xerbla( __func__, -(info) );
    info = MAGMA_ERR;                magma_xerbla( __func__, -(info) );
    info = MAGMA_ERR_HOST_ALLOC;     magma_xerbla( __func__, -(info) );
    info = MAGMA_ERR_DEVICE_ALLOC;   magma_xerbla( __func__, -(info) );
}


/******************************************************************************/
void test_indices()
{
    printf( "%%=====================================================================\n%s\n", __func__ );
    // example given in magma_indices_1D_bcyclic() docs
    magma_int_t dj0=0, dj1=0;
    
    // -----
    magma_indices_1D_bcyclic( 10, 3, 0, 13, 69, &dj0, &dj1 );
    warn( dj0 == 10 );
    warn( dj1 == 29 );

    magma_indices_1D_bcyclic( 10, 3, 1, 13, 69, &dj0, &dj1 );
    warn( dj0 ==  3 );
    warn( dj1 == 20 );

    magma_indices_1D_bcyclic( 10, 3, 2, 13, 69, &dj0, &dj1 );
    warn( dj0 ==  0 );
    warn( dj1 == 20 );
    
    // -----
    magma_indices_1D_bcyclic( 10, 3, 0, 13, 70, &dj0, &dj1 );
    warn( dj0 == 10 );
    warn( dj1 == 30 );

    magma_indices_1D_bcyclic( 10, 3, 1, 13, 70, &dj0, &dj1 );
    warn( dj0 ==  3 );
    warn( dj1 == 20 );

    magma_indices_1D_bcyclic( 10, 3, 2, 13, 70, &dj0, &dj1 );
    warn( dj0 ==  0 );
    warn( dj1 == 20 );
    
    // -----
    magma_indices_1D_bcyclic( 10, 3, 0, 13, 71, &dj0, &dj1 );
    warn( dj0 == 10 );
    warn( dj1 == 30 );

    magma_indices_1D_bcyclic( 10, 3, 1, 13, 71, &dj0, &dj1 );
    warn( dj0 ==  3 );
    warn( dj1 == 21 );

    magma_indices_1D_bcyclic( 10, 3, 2, 13, 71, &dj0, &dj1 );
    warn( dj0 ==  0 );
    warn( dj1 == 20 );
    
    // -----
    magma_indices_1D_bcyclic( 10, 3, 0, 13, 72, &dj0, &dj1 );
    warn( dj0 == 10 );
    warn( dj1 == 30 );

    magma_indices_1D_bcyclic( 10, 3, 1, 13, 72, &dj0, &dj1 );
    warn( dj0 ==  3 );
    warn( dj1 == 22 );

    magma_indices_1D_bcyclic( 10, 3, 2, 13, 72, &dj0, &dj1 );
    warn( dj0 ==  0 );
    warn( dj1 == 20 );
}


/******************************************************************************/
int main( int argc, char** argv )
{
    magma_init();
    
    test_num_gpus();
    test_num_threads();
    test_xerbla();
    test_indices();
    
    if ( gFailures > 0 ) {
        printf( "\n*** %lld tests failed.\n", (long long) gFailures );
    }
    else {
        printf( "\nAll tests passed.\n" );
    }
    
    magma_finalize();
    return 0;
}
