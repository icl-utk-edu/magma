/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"

#define BLOCK_SIZE 256

#define PRECISION_z


// initialize arrays with zero
__global__ void
magma_zgpumemzero(  
    magmaDoubleComplex * d, 
    int n, 
    int k )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        for (int j = 0; j < k; j++)
            d[ i+j*n ] = MAGMA_Z_MAKE( 0.0, 0.0 );
    }
}

// dot product
__global__ void
magma_zdot_kernel( 
    int Gs,
    int n, 
    magmaDoubleComplex * v,
    magmaDoubleComplex * r,
    magmaDoubleComplex * vtmp)
{
    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;

    temp[ Idx ] = ( i < n ) ? v[ i ] * r[ i ] : MAGMA_Z_MAKE( 0.0, 0.0);
    __syncthreads();
    if ( Idx < 128 ){
        temp[ Idx ] += temp[ Idx + 128 ];
    }
    __syncthreads();
    if ( Idx < 64 ){
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            temp[ Idx ] += temp[ Idx + 32 ]; __syncthreads();
            temp[ Idx ] += temp[ Idx + 16 ]; __syncthreads();
            temp[ Idx ] += temp[ Idx + 8  ]; __syncthreads();
            temp[ Idx ] += temp[ Idx + 4  ]; __syncthreads();
            temp[ Idx ] += temp[ Idx + 2  ]; __syncthreads();
            temp[ Idx ] += temp[ Idx + 1  ]; __syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    if ( Idx == 0 ){
            vtmp[ blockIdx.x ] = temp[ 0 ];
    }
}

// dot product for multiple vectors
__global__ void
magma_zblockdot_kernel( 
    int Gs,
    int n, 
    int k,
    magmaDoubleComplex * v,
    magmaDoubleComplex * r,
    magmaDoubleComplex * vtmp)
{
    extern __shared__ magmaDoubleComplex temp[]; 
    int Idx = threadIdx.x;   
    int i   = blockIdx.x * blockDim.x + Idx;
    int j;

    // k vectors v(i)
    if (i<n){
        for( j=0; j<k; j++)
            temp[Idx+j*blockDim.x] = v[i+j*n] * r[i];
    }
    else {
        for( j=0; j<k; j++)
            temp[Idx+j*blockDim.x] = MAGMA_Z_MAKE( 0.0, 0.0);
    }
    __syncthreads();
    if ( Idx < 128 ){
        for( j=0; j<k; j++){
            temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 128 ];
        }
    }
    __syncthreads();
    if ( Idx < 64 ){
        for( j=0; j<k; j++){
            temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 64 ];
        }
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<k; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 32 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 16 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 8 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 4 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 2 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 1 ];
                __syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<k; j++){
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 32 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 16 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 8 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 4 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 2 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 1 ];
            }
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            for( j=0; j<k; j++){
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 32 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 16 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 8 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 4 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 2 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 1 ];
            }
        }
    #endif
    if ( Idx == 0 ){
        for( j=0; j<k; j++){
            vtmp[ blockIdx.x+j*n ] = temp[ j*blockDim.x ];
        }
    }
}

// block reduction for multiple vectors
__global__ void
magma_zblockreduce_kernel( 
    int Gs,
    int n, 
    int k,
    magmaDoubleComplex * vtmp,
    magmaDoubleComplex * vtmp2 )
{
    extern __shared__ magmaDoubleComplex temp[];    
    int Idx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + Idx;  
    int j;
    for( j=0; j<k; j++){
        temp[ Idx+j*blockDim.x ] =  ( i < n ) ? vtmp[ i+j*n ] 
                                        : MAGMA_Z_MAKE( 0.0, 0.0);
    }
    __syncthreads();
    if ( Idx < 128 ){
        for( j=0; j<k; j++){
            temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 128 ];
        }
    }
    __syncthreads();
    if ( Idx < 64 ){
        for( j=0; j<k; j++){
            temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 64 ];
        }
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<k; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 32 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 16 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 8 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 4 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 2 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*blockDim.x ] += temp[ Idx+j*blockDim.x + 1 ];
                __syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<k; j++){
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 32 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 16 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 8 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 4 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 2 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 1 ];
            }
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            for( j=0; j<k; j++){
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 32 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 16 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 8 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 4 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 2 ];
                temp2[ Idx+j*blockDim.x ] += temp2[ Idx+j*blockDim.x + 1 ];
            }
        }
    #endif
    if ( Idx == 0 ){
        for( j=0; j<k; j++){
            vtmp2[ blockIdx.x+j*n ] = temp[ j*blockDim.x ];
        }
    }
}

// accelerated reduction for one vector
__global__ void
magma_zreduce_kernel_fast( int Gs,
                           int n, 
                           magmaDoubleComplex * vtmp,
                           magmaDoubleComplex * vtmp2 )
{
    extern __shared__ magmaDoubleComplex temp[];    
    int Idx = threadIdx.x;
    int blockSize = 128;
    int gridSize = blockSize  * 2 * gridDim.x; 
    temp[Idx] = MAGMA_Z_MAKE( 0.0, 0.0);
    int i = blockIdx.x * ( blockSize * 2 ) + Idx;   
    while (i < Gs ) {
        temp[ Idx  ] += vtmp[ i ]; 
        temp[ Idx  ] += ( i + blockSize < Gs ) ? vtmp[ i + blockSize ] 
                                                : MAGMA_Z_MAKE( 0.0, 0.0); 
        i += gridSize;
    }
    __syncthreads();
    if ( Idx < 64 ){
        temp[ Idx ] += temp[ Idx + 64 ];
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            temp[ Idx ] += temp[ Idx + 32 ]; __syncthreads();
            temp[ Idx ] += temp[ Idx + 16 ]; __syncthreads();
            temp[ Idx ] += temp[ Idx + 8  ]; __syncthreads();
            temp[ Idx ] += temp[ Idx + 4  ]; __syncthreads();
            temp[ Idx ] += temp[ Idx + 2  ]; __syncthreads();
            temp[ Idx ] += temp[ Idx + 1  ]; __syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            temp2[ Idx ] += temp2[ Idx + 32 ];
            temp2[ Idx ] += temp2[ Idx + 16 ];
            temp2[ Idx ] += temp2[ Idx + 8 ];
            temp2[ Idx ] += temp2[ Idx + 4 ];
            temp2[ Idx ] += temp2[ Idx + 2 ];
            temp2[ Idx ] += temp2[ Idx + 1 ];
        }
    #endif
    if ( Idx == 0 ){
        vtmp2[ blockIdx.x ] = temp[ 0 ];
    }
}

// accelerated block reduction for multiple vectors
__global__ void
magma_zblockreduce_kernel_fast( 
    int Gs,
    int n, 
    int k,
    magmaDoubleComplex * vtmp,
    magmaDoubleComplex * vtmp2 )
{
    extern __shared__ magmaDoubleComplex temp[];    
    int Idx = threadIdx.x;
    int blockSize = 128;
    int gridSize = blockSize  * 2 * gridDim.x; 
    int j;

    for( j=0; j<k; j++){
        int i = blockIdx.x * ( blockSize * 2 ) + Idx;   
        temp[Idx+j*(blockSize)] = MAGMA_Z_MAKE( 0.0, 0.0);
        while (i < Gs ) {
            temp[ Idx+j*(blockSize)  ] += vtmp[ i+j*n ]; 
            temp[ Idx+j*(blockSize)  ] += 
                ( i + (blockSize) < Gs ) ? vtmp[ i+j*n + (blockSize) ] 
                                                : MAGMA_Z_MAKE( 0.0, 0.0); 
            i += gridSize;
        }
    }
    __syncthreads();
    if ( Idx < 64 ){
        for( j=0; j<k; j++){
            temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 64 ];
        }
    }
    __syncthreads();
    #if defined(PRECISION_z) || defined(PRECISION_c)
        if( Idx < 32 ){
            for( j=0; j<k; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 32 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 16 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 8 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 4 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 2 ];
                __syncthreads();
            for( j=0; j<k; j++)
                temp[ Idx+j*(blockSize) ] += temp[ Idx+j*(blockSize) + 1 ];
                __syncthreads();
        }
    #endif
    #if defined(PRECISION_d)
        if( Idx < 32 ){
            volatile double *temp2 = temp;
            for( j=0; j<k; j++){
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 32 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 16 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 8 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 4 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 2 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 1 ];
            }
        }
    #endif
    #if defined(PRECISION_s)
        if( Idx < 32 ){
            volatile float *temp2 = temp;
            for( j=0; j<k; j++){
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 32 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 16 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 8 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 4 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 2 ];
                temp2[ Idx+j*(blockSize) ] += temp2[ Idx+j*(blockSize) + 1 ];
            }
        }
    #endif
    if ( Idx == 0 ){
        for( j=0; j<k; j++){
            vtmp2[ blockIdx.x+j*n ] = temp[ j*(blockSize) ];
        }
    }
}

/**
    Purpose
    -------

    Computes the scalar product of a set of vectors v_i such that

    skp = ( <v_0,r>, <v_1,r>, .. )

    Returns the vector skp.

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i and r

    @param[in]
    k           int
                # vectors v_i

    @param[in]
    v           magmaDoubleComplex_ptr 
                v = (v_0 .. v_i.. v_k)

    @param[in]
    r           magmaDoubleComplex_ptr 
                r

    @param[in]
    d1          magmaDoubleComplex_ptr 
                workspace

    @param[in]
    d2          magmaDoubleComplex_ptr 
                workspace

    @param[out]
    skp         magmaDoubleComplex_ptr 
                vector[k] of scalar products (<v_i,r>...)

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zmdotc(
    magma_int_t n, 
    magma_int_t k, 
    magmaDoubleComplex_ptr v, 
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    int local_block_size=256;
    dim3 Bs( local_block_size );
    dim3 Gs( magma_ceildiv( n, local_block_size ) );
    dim3 Gs_next;
    int Ms =  (k)* (local_block_size) * sizeof( magmaDoubleComplex ); // k vecs 
    magmaDoubleComplex_ptr aux1 = d1, aux2 = d2;
    int b = 1;        

    if (k>1) {
        magma_zblockdot_kernel<<< Gs, Bs, Ms, queue->cuda_stream() >>>( Gs.x, n, k, v, r, d1 );
    }
    else {
        magma_zdot_kernel<<< Gs, Bs, Ms, queue->cuda_stream() >>>( Gs.x, n, v, r, d1 );
    }
/*
    // not necessary to zero GPU mem
    magma_zgpumemzero<<< Gs, Bs, 0, queue->cuda_stream >>>( d1, n*k,1 );
    magma_zgpumemzero<<< Gs, Bs, 0, queue->cuda_stream >>>( d2, n*k,1 );
    //magmablas_zlaset( MagmaFull, n, k, d1, n, UNKNOWN );
    //magmablas_zlaset( MagmaFull, n, k, d2, n, UNKNOWN );
    while( Gs.x > 1 ) {
        Gs_next.x = magma_ceildiv( Gs.x, Bs.x );
        magma_zblockreduce_kernel<<< Gs_next.x, Bs.x, Ms, queue->cuda_stream >>> 
                                        ( Gs.x, n, k, aux1, aux2 );
        Gs.x = Gs_next.x;
        b = 1 - b;
        if ( b ) { aux1 = d1; aux2 = d2; }
        else   { aux2 = d1; aux1 = d2; }
    }
    for( int j=0; j<k; j++) {
            magma_zcopyvector( 1, aux1+j*n, 1, skp+j, 1, UNKNOWN );
    }
*/
   
    if ( k>1) {
        while( Gs.x > 1 ) {
            Gs_next.x = magma_ceildiv( Gs.x, Bs.x );
            if ( Gs_next.x == 1 ) Gs_next.x = 2;
            magma_zblockreduce_kernel_fast<<< Gs_next.x/2, Bs.x/2, Ms/2, queue->cuda_stream() >>> 
                        ( Gs.x, n, k, aux1, aux2 );
            Gs_next.x = Gs_next.x /2;
            Gs.x = Gs_next.x;
            b = 1 - b;
            if ( b ) { aux1 = d1; aux2 = d2; }
            else   { aux2 = d1; aux1 = d2; }
        }
    }
    else {
        while( Gs.x > 1 ) {
            Gs_next.x = magma_ceildiv( Gs.x, Bs.x );
            if ( Gs_next.x == 1 ) Gs_next.x = 2;
            magma_zreduce_kernel_fast<<< Gs_next.x/2, Bs.x/2, Ms/2, queue->cuda_stream() >>> 
                        ( Gs.x, n, aux1, aux2 );
            Gs_next.x = Gs_next.x /2;
            Gs.x = Gs_next.x;
            b = 1 - b;
            if ( b ) { aux1 = d1; aux2 = d2; }
            else   { aux2 = d1; aux1 = d2; }
        }
    }

    magma_zcopyvector_async( k, aux1, n, skp, 1, queue );

    return MAGMA_SUCCESS;
}

/**
    Purpose
    -------

    This is an extension of the merged dot product above by chunking
    the set of vectors v_i such that the data always fits into cache.
    It is equivalent to a matrix vecor product Vr where V
    contains few rows and many columns. The computation is the same:

    skp = ( <v_0,r>, <v_1,r>, .. )

    Returns the vector skp.

    Arguments
    ---------

    @param[in]
    n           int
                length of v_i and r

    @param[in]
    k           int
                # vectors v_i

    @param[in]
    v           magmaDoubleComplex_ptr 
                v = (v_0 .. v_i.. v_k)

    @param[in]
    r           magmaDoubleComplex_ptr 
                r

    @param[in]
    d1          magmaDoubleComplex_ptr 
                workspace

    @param[in]
    d2          magmaDoubleComplex_ptr 
                workspace

    @param[out]
    skp         magmaDoubleComplex_ptr 
                vector[k] of scalar products (<v_i,r>...)

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zgemvmdot(
    magma_int_t n, 
    magma_int_t k, 
    magmaDoubleComplex_ptr v, 
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr d1,
    magmaDoubleComplex_ptr d2,
    magmaDoubleComplex_ptr skp,
    magma_queue_t queue )
{
    int rows_left = k;
    int offset = 0;
    int chunk_size = 4;
    // process in chunks of 10 - has to be adapted to hardware and precision
    while( rows_left > (chunk_size) ) {
        magma_zmdotc( n, chunk_size, v+offset*n, r, d1, d2, skp+offset, queue );
        offset = offset + chunk_size;
        rows_left = rows_left-chunk_size;
    }
    // process rest
    magma_zmdotc( n, rows_left, v+offset*n, r, d1, d2, skp+offset, queue ); 


    return MAGMA_SUCCESS;
}
