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


// These routines merge multiple kernels from bicgstab into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_zbicgstab_1_kernel(  
    int num_rows, 
    int num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex omega,
    magmaDoubleComplex *r, 
    magmaDoubleComplex *v,
    magmaDoubleComplex *p )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            p[ i+j*num_rows ] = r[ i+j*num_rows ] + 
                beta * ( p[ i+j*num_rows ] - omega * v[ i+j*num_rows ] );
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    p = r + beta * ( p - omega * v )
    
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
    omega       magmaDoubleComplex
                scalar
                
    @param[in]
    r           magmaDoubleComplex_ptr 
                vector
                
    @param[in]
    v           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    p           magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zbicgstab_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex omega,
    magmaDoubleComplex_ptr r, 
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr p,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_zbicgstab_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, omega,
                     r, v, p );

    return MAGMA_SUCCESS;
}


__global__ void
magma_zbicgstab_2_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr s )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            s[ i+j*num_rows ] = r[ i+j*num_rows ] - alpha * v[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    s = r - alpha v

    Arguments
    ---------

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
    r           magmaDoubleComplex_ptr 
                vector
                
    @param[in]
    v           magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    s           magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zbicgstab_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr v,
    magmaDoubleComplex_ptr s, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_zbicgstab_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, r, v, s );

    return MAGMA_SUCCESS;
}


__global__ void
magma_zbicgstab_3_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex alpha,
    magmaDoubleComplex omega,
    magmaDoubleComplex *p,
    magmaDoubleComplex *s,
    magmaDoubleComplex *t,
    magmaDoubleComplex *x,
    magmaDoubleComplex *r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaDoubleComplex tmp = s[ i+j*num_rows ];
            x[ i+j*num_rows ] = x[ i+j*num_rows ] 
                        + alpha * p[ i+j*num_rows ] + omega * tmp;
            r[ i+j*num_rows ] = tmp - omega * t[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    x = x + alpha * p + omega * s
    r = s - omega * t

    Arguments
    ---------

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
    omega       magmaDoubleComplex
                scalar
                
    @param[in]
    p           magmaDoubleComplex_ptr 
                vector
                    
    @param[in]
    s           magmaDoubleComplex_ptr 
                vector
                    
    @param[in]
    t           magmaDoubleComplex_ptr 
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
magma_zbicgstab_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex omega,
    magmaDoubleComplex_ptr p,
    magmaDoubleComplex_ptr s,
    magmaDoubleComplex_ptr t,
    magmaDoubleComplex_ptr x,
    magmaDoubleComplex_ptr r,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_zbicgstab_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, omega, p, s, t, x, r );

    return MAGMA_SUCCESS;
}


__global__ void
magma_zbicgstab_4_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex alpha,
    magmaDoubleComplex omega,
    magmaDoubleComplex *y,
    magmaDoubleComplex *z,
    magmaDoubleComplex *s,
    magmaDoubleComplex *t,
    magmaDoubleComplex *x,
    magmaDoubleComplex *r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            x[ i+j*num_rows ] = x[ i+j*num_rows ] 
                        + alpha * y[ i+j*num_rows ] + omega * z[ i+j*num_rows ];
            r[ i+j*num_rows ] = s[ i+j*num_rows ] - omega * t[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    x = x + alpha * y + omega * z
    r = s - omega * t

    Arguments
    ---------

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
    omega       magmaDoubleComplex
                scalar
                
    @param[in]
    y           magmaDoubleComplex_ptr 
                vector
                
    @param[in]
    z           magmaDoubleComplex_ptr 
                vector
                    
    @param[in]
    s           magmaDoubleComplex_ptr 
                vector
                    
    @param[in]
    t           magmaDoubleComplex_ptr 
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
magma_zbicgstab_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex omega,
    magmaDoubleComplex_ptr y,
    magmaDoubleComplex_ptr z,
    magmaDoubleComplex_ptr s,
    magmaDoubleComplex_ptr t,
    magmaDoubleComplex_ptr x,
    magmaDoubleComplex_ptr r,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_zbicgstab_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, omega, y, z, s, t, x, r );

    return MAGMA_SUCCESS;
}
