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

#define BLOCK_SIZE 512

#define PRECISION_z


// These routines merge multiple kernels from tfqmr into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_ztfqmr_1_kernel(  
    int num_rows, 
    int num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex sigma,
    magmaDoubleComplex *v, 
    magmaDoubleComplex *Au,
    magmaDoubleComplex *u_m,
    magmaDoubleComplex *pu_m,
    magmaDoubleComplex *u_mp1,
    magmaDoubleComplex *w, 
    magmaDoubleComplex *d,
    magmaDoubleComplex *Ad )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            u_mp1[ i+j*num_rows ] = u_m[ i+j*num_rows ] - alpha * v[ i+j*num_rows ];
            w[ i+j*num_rows ] = w[ i+j*num_rows ] - alpha * Au[ i+j*num_rows ];
            d[ i+j*num_rows ] = pu_m[ i+j*num_rows ] + sigma * d[ i+j*num_rows ];
            Ad[ i+j*num_rows ] = Au[ i+j*num_rows ] + sigma * Ad[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    u_mp1 = u_mp1 - alpha*v;
    w = w - alpha*Au;
    d = pu_m + sigma*d;
    Ad = Au + sigma*Ad;
    
    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    alpha       magmaDoubleComplex
                scalar
                
    @param[in]
    sigma       magmaDoubleComplex
                scalar
                
    @param[in]
    v           magmaDoubleComplex_ptr 
                vector
                
    @param[in]
    Au          magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    u_m         magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    pu_m         magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    u_mp1       magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    w           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    d           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    Ad          magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_ztfqmr_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex sigma,
    magmaDoubleComplex_ptr v, 
    magmaDoubleComplex_ptr Au,
    magmaDoubleComplex_ptr u_m,
    magmaDoubleComplex_ptr pu_m,
    magmaDoubleComplex_ptr u_mp1,
    magmaDoubleComplex_ptr w, 
    magmaDoubleComplex_ptr d,
    magmaDoubleComplex_ptr Ad,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_ztfqmr_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, sigma,
                     v, Au, u_m, pu_m, u_mp1, w, d, Ad );

    return MAGMA_SUCCESS;
}


__global__ void
magma_ztfqmr_2_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex eta,
    magmaDoubleComplex_ptr d,
    magmaDoubleComplex_ptr Ad,
    magmaDoubleComplex_ptr x, 
    magmaDoubleComplex_ptr r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            x[ i+j*num_rows ] = x[ i+j*num_rows ] + eta * d[ i+j*num_rows ];
            r[ i+j*num_rows ] = r[ i+j*num_rows ] - eta * Ad[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    x = x + eta * d
    r = r - eta * Ad

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    eta         magmaDoubleComplex
                scalar
                
    @param[in]
    d           magmaDoubleComplex_ptr 
                vector
                
    @param[in]
    Ad          magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    x           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    r           magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_ztfqmr_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex eta,
    magmaDoubleComplex_ptr d,
    magmaDoubleComplex_ptr Ad,
    magmaDoubleComplex_ptr x, 
    magmaDoubleComplex_ptr r, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_ztfqmr_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, eta, d, Ad, x, r );

    return MAGMA_SUCCESS;
}


__global__ void
magma_ztfqmr_3_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex beta,
    magmaDoubleComplex *w,
    magmaDoubleComplex *u_m,
    magmaDoubleComplex *u_mp1 )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            u_mp1[ i+j*num_rows ] = w[ i+j*num_rows ] + beta * u_m[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    u_mp1 = w + beta*u_mp1

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    beta        magmaDoubleComplex
                scalar
                
    @param[in]
    w           magmaDoubleComplex_ptr 
                vector
                
    @param[in]
    u_m         magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    u_mp1       magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_ztfqmr_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr w,
    magmaDoubleComplex_ptr u_m,
    magmaDoubleComplex_ptr u_mp1, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_ztfqmr_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, w, u_m, u_mp1 );

    return MAGMA_SUCCESS;
}


__global__ void
magma_ztfqmr_4_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex beta,
    magmaDoubleComplex *Au_new,
    magmaDoubleComplex *v,
    magmaDoubleComplex *Au )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaDoubleComplex tmp = Au_new[ i+j*num_rows ];
                v[ i+j*num_rows ] = tmp + beta * Au[ i+j*num_rows ] 
                                    + beta * beta * v[ i+j*num_rows ];
                Au[ i+j*num_rows ] = tmp; 
        }
    }
}

/**
    Purpose
    -------

    Merges multiple operations into one kernel:

    v = Au_new + beta*(Au+beta*v);
    Au = Au_new

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    beta        magmaDoubleComplex
                scalar
                
    @param[in]
    Au_new      magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    v           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    Au          magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_ztfqmr_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr Au_new,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr Au, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_ztfqmr_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, Au_new, v, Au );

    return MAGMA_SUCCESS;
}


__global__ void
magma_ztfqmr_5_kernel(  
    int num_rows,                   
    int num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex sigma,
    magmaDoubleComplex *v, 
    magmaDoubleComplex *Au,
    magmaDoubleComplex *u_mp1,
    magmaDoubleComplex *w, 
    magmaDoubleComplex *d,
    magmaDoubleComplex *Ad )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            w[ i+j*num_rows ] = w[ i+j*num_rows ] - alpha * Au[ i+j*num_rows ];
            d[ i+j*num_rows ] = u_mp1[ i+j*num_rows ] + sigma * d[ i+j*num_rows ];
            Ad[ i+j*num_rows ] = Au[ i+j*num_rows ] + sigma * Ad[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    w = w - alpha*Au;
    d = pu_m + sigma*d;
    Ad = Au + sigma*Ad;
    
    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    alpha       magmaDoubleComplex
                scalar
                
    @param[in]
    sigma       magmaDoubleComplex
                scalar
                
    @param[in]
    v           magmaDoubleComplex_ptr 
                vector
                
    @param[in]
    Au          magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    u_mp1       magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    w           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    d           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    Ad          magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_ztfqmr_5(  
    magma_int_t num_rows,               
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex sigma,
    magmaDoubleComplex_ptr v, 
    magmaDoubleComplex_ptr Au,
    magmaDoubleComplex_ptr u_mp1,
    magmaDoubleComplex_ptr w, 
    magmaDoubleComplex_ptr d,
    magmaDoubleComplex_ptr Ad,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_ztfqmr_5_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, sigma,
                     v, Au, u_mp1, w, d, Ad );

    return MAGMA_SUCCESS;
}
