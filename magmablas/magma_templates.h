/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Mark Gates
*/
#ifndef MAGMA_TEMPLATES_H
#define MAGMA_TEMPLATES_H


/******************************************************************************/
template< int n, typename T, typename ID >
__device__ void 
magma_getidmax( /*int n,*/ int i, T* x, ID* ind )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { if ( x[i] < x[i+1024] || (x[i] == x[i+1024] && ind[i+1024] < ind[i]) ) { ind[i] = ind[i+1024]; x[i] = x[i+1024]; } }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { if ( x[i] < x[i+ 512] || (x[i] == x[i+ 512] && ind[i+ 512] < ind[i]) ) { ind[i] = ind[i+ 512]; x[i] = x[i+ 512]; } }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { if ( x[i] < x[i+ 256] || (x[i] == x[i+ 256] && ind[i+ 256] < ind[i]) ) { ind[i] = ind[i+ 256]; x[i] = x[i+ 256]; } }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { if ( x[i] < x[i+ 128] || (x[i] == x[i+ 128] && ind[i+ 128] < ind[i]) ) { ind[i] = ind[i+ 128]; x[i] = x[i+ 128]; } }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { if ( x[i] < x[i+  64] || (x[i] == x[i+  64] && ind[i+  64] < ind[i]) ) { ind[i] = ind[i+  64]; x[i] = x[i+  64]; } }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { if ( x[i] < x[i+  32] || (x[i] == x[i+  32] && ind[i+  32] < ind[i]) ) { ind[i] = ind[i+  32]; x[i] = x[i+  32]; } }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads                                                                       
    // because of implicit warp level synchronization.                                                                          
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { if ( x[i] < x[i+  16] || (x[i] == x[i+  16] && ind[i+  16] < ind[i]) ) { ind[i] = ind[i+  16]; x[i] = x[i+  16]; } }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { if ( x[i] < x[i+   8] || (x[i] == x[i+   8] && ind[i+   8] < ind[i]) ) { ind[i] = ind[i+   8]; x[i] = x[i+   8]; } }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { if ( x[i] < x[i+   4] || (x[i] == x[i+   4] && ind[i+   4] < ind[i]) ) { ind[i] = ind[i+   4]; x[i] = x[i+   4]; } }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { if ( x[i] < x[i+   2] || (x[i] == x[i+   2] && ind[i+   2] < ind[i]) ) { ind[i] = ind[i+   2]; x[i] = x[i+   2]; } }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { if ( x[i] < x[i+   1] || (x[i] == x[i+   1] && ind[i+   1] < ind[i]) ) { ind[i] = ind[i+   1]; x[i] = x[i+   1]; } }  __syncthreads(); }
}
// end magma_getidmax


/***************************************************************************//**
    Same as magma_getidmax(),
    but takes n as runtime argument instead of compile-time template parameter.
    @ingroup magma_kernel
*******************************************************************************/
template< typename T, typename ID >
__device__ void 
magma_getidmax_n( int n, int i, T* x, ID* ind )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { if ( x[i] < x[i+1024] || (x[i] == x[i+1024] && ind[i+1024] < ind[i]) ) { ind[i] = ind[i+1024]; x[i] = x[i+1024]; } }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { if ( x[i] < x[i+ 512] || (x[i] == x[i+ 512] && ind[i+ 512] < ind[i]) ) { ind[i] = ind[i+ 512]; x[i] = x[i+ 512]; } }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { if ( x[i] < x[i+ 256] || (x[i] == x[i+ 256] && ind[i+ 256] < ind[i]) ) { ind[i] = ind[i+ 256]; x[i] = x[i+ 256]; } }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { if ( x[i] < x[i+ 128] || (x[i] == x[i+ 128] && ind[i+ 128] < ind[i]) ) { ind[i] = ind[i+ 128]; x[i] = x[i+ 128]; } }  __syncthreads(); } 
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { if ( x[i] < x[i+  64] || (x[i] == x[i+  64] && ind[i+  64] < ind[i]) ) { ind[i] = ind[i+  64]; x[i] = x[i+  64]; } }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { if ( x[i] < x[i+  32] || (x[i] == x[i+  32] && ind[i+  32] < ind[i]) ) { ind[i] = ind[i+  32]; x[i] = x[i+  32]; } }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads                                              
    // because of implicit warp level synchronization.                                                 
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { if ( x[i] < x[i+  16] || (x[i] == x[i+  16] && ind[i+  16] < ind[i]) ) { ind[i] = ind[i+  16]; x[i] = x[i+  16]; } }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { if ( x[i] < x[i+   8] || (x[i] == x[i+   8] && ind[i+   8] < ind[i]) ) { ind[i] = ind[i+   8]; x[i] = x[i+   8]; } }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { if ( x[i] < x[i+   4] || (x[i] == x[i+   4] && ind[i+   4] < ind[i]) ) { ind[i] = ind[i+   4]; x[i] = x[i+   4]; } }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { if ( x[i] < x[i+   2] || (x[i] == x[i+   2] && ind[i+   2] < ind[i]) ) { ind[i] = ind[i+   2]; x[i] = x[i+   2]; } }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { if ( x[i] < x[i+   1] || (x[i] == x[i+   1] && ind[i+   1] < ind[i]) ) { ind[i] = ind[i+   1]; x[i] = x[i+   1]; } }  __syncthreads(); }
}
// end magma_getidmax_n



/***************************************************************************//**
    Does max reduction of n-element array x, leaving total in x[0].
    Contents of x are destroyed in the process.
    With k threads, can reduce array up to 2*k in size.
    Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
    Having n as template parameter allows compiler to evaluate some conditions at compile time.
    Calls __syncthreads before & after reduction.
    @ingroup magma_kernel
*******************************************************************************/
template< int n, typename T >
__device__ void
magma_max_reduce( /*int n,*/ int i, T* x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] = max( x[i], x[i+1024] ); }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] = max( x[i], x[i+ 512] ); }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] = max( x[i], x[i+ 256] ); }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] = max( x[i], x[i+ 128] ); }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] = max( x[i], x[i+  64] ); }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] = max( x[i], x[i+  32] ); }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] = max( x[i], x[i+  16] ); }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] = max( x[i], x[i+   8] ); }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] = max( x[i], x[i+   4] ); }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] = max( x[i], x[i+   2] ); }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] = max( x[i], x[i+   1] ); }  __syncthreads(); }
}
// end max_reduce


/***************************************************************************//**
    Same as magma_max_reduce(),
    but takes n as runtime argument instead of compile-time template parameter.
    @ingroup magma_kernel
*******************************************************************************/
template< typename T >
__device__ void
magma_max_reduce_n( int n, int i, T* x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] = max( x[i], x[i+1024] ); }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] = max( x[i], x[i+ 512] ); }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] = max( x[i], x[i+ 256] ); }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] = max( x[i], x[i+ 128] ); }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] = max( x[i], x[i+  64] ); }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] = max( x[i], x[i+  32] ); }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] = max( x[i], x[i+  16] ); }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] = max( x[i], x[i+   8] ); }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] = max( x[i], x[i+   4] ); }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] = max( x[i], x[i+   2] ); }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] = max( x[i], x[i+   1] ); }  __syncthreads(); }
}
// end max_reduce_n


/***************************************************************************//**
    max that propogates nan consistently:
    max_nan( 1,   nan ) = nan
    max_nan( nan, 1   ) = nan

    For x=nan, y=1:
    nan < y is false, yields x (nan)

    For x=1, y=nan:
    x < nan    is false, would yield x, but
    isnan(nan) is true, yields y (nan)
    @ingroup magma_kernel
*******************************************************************************/
template< typename T >
__host__ __device__
inline T max_nan( T x, T y )
{
    return (isnan(y) || (x) < (y) ? (y) : (x));
}


/***************************************************************************//**
    Same as magma_max_reduce(), but propogates nan values.

    Does max reduction of n-element array x, leaving total in x[0].
    Contents of x are destroyed in the process.
    With k threads, can reduce array up to 2*k in size.
    Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
    Having n as template parameter allows compiler to evaluate some conditions at compile time.
    Calls __syncthreads before & after reduction.
    @ingroup magma_kernel
*******************************************************************************/
template< int n, typename T >
__device__ void
magma_max_nan_reduce( /*int n,*/ int i, T* x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] = max_nan( x[i], x[i+1024] ); }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] = max_nan( x[i], x[i+ 512] ); }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] = max_nan( x[i], x[i+ 256] ); }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] = max_nan( x[i], x[i+ 128] ); }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] = max_nan( x[i], x[i+  64] ); }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] = max_nan( x[i], x[i+  32] ); }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] = max_nan( x[i], x[i+  16] ); }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] = max_nan( x[i], x[i+   8] ); }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] = max_nan( x[i], x[i+   4] ); }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] = max_nan( x[i], x[i+   2] ); }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] = max_nan( x[i], x[i+   1] ); }  __syncthreads(); }
}
// end max_nan_reduce


/***************************************************************************//**
    Same as magma_max_nan_reduce(),
    but takes n as runtime argument instead of compile-time template parameter.
    @ingroup magma_kernel
*******************************************************************************/
template< typename T >
__device__ void
magma_max_nan_reduce_n( int n, int i, T* x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] = max_nan( x[i], x[i+1024] ); }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] = max_nan( x[i], x[i+ 512] ); }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] = max_nan( x[i], x[i+ 256] ); }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] = max_nan( x[i], x[i+ 128] ); }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] = max_nan( x[i], x[i+  64] ); }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] = max_nan( x[i], x[i+  32] ); }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] = max_nan( x[i], x[i+  16] ); }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] = max_nan( x[i], x[i+   8] ); }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] = max_nan( x[i], x[i+   4] ); }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] = max_nan( x[i], x[i+   2] ); }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] = max_nan( x[i], x[i+   1] ); }  __syncthreads(); }
}
// end max_nan_reduce


/***************************************************************************//**
    max reduction, for arbitrary size vector. Leaves max(x) in x[0].
    Uses only one thread block of 512 threads, so is not efficient for really large vectors.
    @ingroup magma_kernel
*******************************************************************************/
template< typename T >
__global__ void
magma_max_nan_kernel( int n, T* x )
{
    __shared__ T smax[ 512 ];
    int tx = threadIdx.x;
    
    smax[tx] = 0;
    for( int i=tx; i < n; i += 512 ) {
        smax[tx] = max_nan( smax[tx], x[i] );
    }
    magma_max_nan_reduce< 512 >( tx, smax );
    if ( tx == 0 ) {
        x[0] = smax[0];
    }
}


/***************************************************************************//**
    Does sum reduction of n-element array x, leaving total in x[0].
    Contents of x are destroyed in the process.
    With k threads, can reduce array up to 2*k in size.
    Assumes number of threads <= 1024 (which is max number of threads up to CUDA capability 3.0)
    Having n as template parameter allows compiler to evaluate some conditions at compile time.
    Calls __syncthreads before & after reduction.
    @ingroup magma_kernel
*******************************************************************************/
template< int n, typename T >
__device__ void
magma_sum_reduce( /*int n,*/ int i, T* x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}
// end sum_reduce


/***************************************************************************//**
    sum reduction, for arbitrary size vector. Leaves sum(x) in x[0].
    Uses only one thread block of 512 threads, so is not efficient for really large vectors.
    @ingroup magma_kernel
*******************************************************************************/
template< typename T >
__global__ void
magma_sum_reduce_kernel( int n, T* x )
{
    __shared__ T sum[ 512 ];
    int tx = threadIdx.x;
    
    sum[tx] = 0;
    for( int i=tx; i < n; i += 512 ) {
        sum[tx] += x[i];
    }
    magma_sum_reduce< 512 >( tx, sum );
    if ( tx == 0 ) {
        x[0] = sum[0];
    }
}


/***************************************************************************//**
    Same as magma_sum_reduce(),
    but takes n as runtime argument instead of compile-time template parameter.
    @ingroup magma_kernel
*******************************************************************************/
template< typename T >
__device__ void
magma_sum_reduce_n( int n, int i, T* x )
{
    __syncthreads();
    if ( n > 1024 ) { if ( i < 1024 && i + 1024 < n ) { x[i] += x[i+1024]; }  __syncthreads(); }
    if ( n >  512 ) { if ( i <  512 && i +  512 < n ) { x[i] += x[i+ 512]; }  __syncthreads(); }
    if ( n >  256 ) { if ( i <  256 && i +  256 < n ) { x[i] += x[i+ 256]; }  __syncthreads(); }
    if ( n >  128 ) { if ( i <  128 && i +  128 < n ) { x[i] += x[i+ 128]; }  __syncthreads(); }
    if ( n >   64 ) { if ( i <   64 && i +   64 < n ) { x[i] += x[i+  64]; }  __syncthreads(); }
    if ( n >   32 ) { if ( i <   32 && i +   32 < n ) { x[i] += x[i+  32]; }  __syncthreads(); }
    // probably don't need __syncthreads for < 16 threads
    // because of implicit warp level synchronization.
    if ( n >   16 ) { if ( i <   16 && i +   16 < n ) { x[i] += x[i+  16]; }  __syncthreads(); }
    if ( n >    8 ) { if ( i <    8 && i +    8 < n ) { x[i] += x[i+   8]; }  __syncthreads(); }
    if ( n >    4 ) { if ( i <    4 && i +    4 < n ) { x[i] += x[i+   4]; }  __syncthreads(); }
    if ( n >    2 ) { if ( i <    2 && i +    2 < n ) { x[i] += x[i+   2]; }  __syncthreads(); }
    if ( n >    1 ) { if ( i <    1 && i +    1 < n ) { x[i] += x[i+   1]; }  __syncthreads(); }
}
// end sum_reduce_n


/***************************************************************************//**
    Does sum reduction of each column of M x N array x,
    leaving totals in x[0][j] = sum( x[0:m-1][j] ), for 0 <= j < n.
    Contents of x are destroyed in the process.
    Calls __syncthreads before & after reduction.
    @ingroup magma_kernel
*******************************************************************************/
template< int m, int n, typename T >
__device__ void
magma_sum_reduce_2d( int i, int j, T x[m][n] )
{
    __syncthreads();
    if ( m > 1024 ) { if ( i < 1024 && i + 1024 < m ) { x[i][j] += x[i+1024][j]; }  __syncthreads(); }
    if ( m >  512 ) { if ( i <  512 && i +  512 < m ) { x[i][j] += x[i+ 512][j]; }  __syncthreads(); }
    if ( m >  256 ) { if ( i <  256 && i +  256 < m ) { x[i][j] += x[i+ 256][j]; }  __syncthreads(); }
    if ( m >  128 ) { if ( i <  128 && i +  128 < m ) { x[i][j] += x[i+ 128][j]; }  __syncthreads(); }
    if ( m >   64 ) { if ( i <   64 && i +   64 < m ) { x[i][j] += x[i+  64][j]; }  __syncthreads(); }
    if ( m >   32 ) { if ( i <   32 && i +   32 < m ) { x[i][j] += x[i+  32][j]; }  __syncthreads(); }
    if ( m >   16 ) { if ( i <   16 && i +   16 < m ) { x[i][j] += x[i+  16][j]; }  __syncthreads(); }
    if ( m >    8 ) { if ( i <    8 && i +    8 < m ) { x[i][j] += x[i+   8][j]; }  __syncthreads(); }
    if ( m >    4 ) { if ( i <    4 && i +    4 < m ) { x[i][j] += x[i+   4][j]; }  __syncthreads(); }
    if ( m >    2 ) { if ( i <    2 && i +    2 < m ) { x[i][j] += x[i+   2][j]; }  __syncthreads(); }
    if ( m >    1 ) { if ( i <    1 && i +    1 < m ) { x[i][j] += x[i+   1][j]; }  __syncthreads(); }
}
// end sum_reduce_2d


/***************************************************************************//**
    Does sum reduction of each "column" of M0 x M1 x M2 array x,
    leaving totals in x[0][j][k] = sum( x[0:m0-1][j][k] ), for 0 <= j < m1, 0 <= k < m2.
    Contents of x are destroyed in the process.
    Calls __syncthreads before & after reduction.
    @ingroup magma_kernel
*******************************************************************************/
template< int m0, int m1, int m2, typename T >
__device__ void
magma_sum_reduce_3d( int i, int j, int k, T x[m0][m1][m2] )
{
    __syncthreads();
    if ( m0 > 1024 ) { if ( i < 1024 && i + 1024 < m0 ) { x[i][j][k] += x[i+1024][j][k]; }  __syncthreads(); }
    if ( m0 >  512 ) { if ( i <  512 && i +  512 < m0 ) { x[i][j][k] += x[i+ 512][j][k]; }  __syncthreads(); }
    if ( m0 >  256 ) { if ( i <  256 && i +  256 < m0 ) { x[i][j][k] += x[i+ 256][j][k]; }  __syncthreads(); }
    if ( m0 >  128 ) { if ( i <  128 && i +  128 < m0 ) { x[i][j][k] += x[i+ 128][j][k]; }  __syncthreads(); }
    if ( m0 >   64 ) { if ( i <   64 && i +   64 < m0 ) { x[i][j][k] += x[i+  64][j][k]; }  __syncthreads(); }
    if ( m0 >   32 ) { if ( i <   32 && i +   32 < m0 ) { x[i][j][k] += x[i+  32][j][k]; }  __syncthreads(); }
    if ( m0 >   16 ) { if ( i <   16 && i +   16 < m0 ) { x[i][j][k] += x[i+  16][j][k]; }  __syncthreads(); }
    if ( m0 >    8 ) { if ( i <    8 && i +    8 < m0 ) { x[i][j][k] += x[i+   8][j][k]; }  __syncthreads(); }
    if ( m0 >    4 ) { if ( i <    4 && i +    4 < m0 ) { x[i][j][k] += x[i+   4][j][k]; }  __syncthreads(); }
    if ( m0 >    2 ) { if ( i <    2 && i +    2 < m0 ) { x[i][j][k] += x[i+   2][j][k]; }  __syncthreads(); }
    if ( m0 >    1 ) { if ( i <    1 && i +    1 < m0 ) { x[i][j][k] += x[i+   1][j][k]; }  __syncthreads(); }
}
// end sum_reduce_3d

#endif // MAGMA_TEMPLATES_H
