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


// These routines merge multiple kernels from zcgs into one.

/* -------------------------------------------------------------------------- */

__global__ void
magma_zcgs_1_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex beta,
    magmaDoubleComplex *r,
    magmaDoubleComplex *q,
    magmaDoubleComplex *u,
    magmaDoubleComplex *p )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaDoubleComplex tmp;
            tmp =  r[ i+j*num_rows ] + beta * q[ i+j*num_rows ];
            p[ i+j*num_rows ] = tmp + beta * q[ i+j*num_rows ] 
                                + beta * beta * p[ i+j*num_rows ];
            u[ i+j*num_rows ] = tmp;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    u = r + beta q
    p = u + beta*(q + beta*p)

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
    r           magmaDoubleComplex_ptr 
                vector

    @param[in]
    q           magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    u           magmaDoubleComplex_ptr 
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
magma_zcgs_1(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr q, 
    magmaDoubleComplex_ptr u,
    magmaDoubleComplex_ptr p,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_zcgs_1_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, beta, r, q, u, p );

    return MAGMA_SUCCESS;
}


__global__ void
magma_zcgs_2_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex *r,
    magmaDoubleComplex *u,
    magmaDoubleComplex *p )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaDoubleComplex tmp;
            tmp = r[ i+j*num_rows ];
            u[ i+j*num_rows ] = tmp;
            p[ i+j*num_rows ] = tmp;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    u = r
    p = r

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                dimension m
                
    @param[in]
    num_cols    magma_int_t
                dimension n
                
    @param[in]
    r           magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    u           magmaDoubleComplex_ptr 
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
magma_zcgs_2(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex_ptr r,
    magmaDoubleComplex_ptr u,
    magmaDoubleComplex_ptr p, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_zcgs_2_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, r, u, p);

    return MAGMA_SUCCESS;
}


__global__ void
magma_zcgs_3_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex alpha,
    magmaDoubleComplex *v_hat,
    magmaDoubleComplex *u,
    magmaDoubleComplex *q,
    magmaDoubleComplex *t )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            magmaDoubleComplex uloc,  tmp;
            uloc = u[ i+j*num_rows ];
            tmp = uloc - alpha * v_hat[ i+j*num_rows ];
            t[ i+j*num_rows ] = tmp + uloc;
            q[ i+j*num_rows ] = tmp;
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    q = u - alpha v_hat
    t = u + q

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
    v_hat       magmaDoubleComplex_ptr 
                vector
    
    @param[in]
    u           magmaDoubleComplex_ptr 
                vector

    @param[in,out]
    q           magmaDoubleComplex_ptr 
                vector
                
    @param[in,out]
    t           magmaDoubleComplex_ptr 
                vector

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" 
magma_int_t
magma_zcgs_3(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr v_hat,
    magmaDoubleComplex_ptr u, 
    magmaDoubleComplex_ptr q,
    magmaDoubleComplex_ptr t, 
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_zcgs_3_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, v_hat, u, q, t );

    return MAGMA_SUCCESS;
}


__global__ void
magma_zcgs_4_kernel(  
    int num_rows,
    int num_cols,
    magmaDoubleComplex alpha,
    magmaDoubleComplex *u_hat,
    magmaDoubleComplex *t,
    magmaDoubleComplex *x,
    magmaDoubleComplex *r )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i<num_rows ) {
        for( int j=0; j<num_cols; j++ ){
            x[ i+j*num_rows ] = x[ i+j*num_rows ] 
                                + alpha * u_hat[ i+j*num_rows ];
            r[ i+j*num_rows ] = r[ i+j*num_rows ] 
                                - alpha * t[ i+j*num_rows ];
        }
    }
}

/**
    Purpose
    -------

    Mergels multiple operations into one kernel:

    x = x + alpha u_hat
    r = r -alpha*A u_hat = r -alpha*t

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
    u_hat       magmaDoubleComplex_ptr 
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
magma_zcgs_4(  
    magma_int_t num_rows, 
    magma_int_t num_cols, 
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr u_hat,
    magmaDoubleComplex_ptr t,
    magmaDoubleComplex_ptr x, 
    magmaDoubleComplex_ptr r,
    magma_queue_t queue )
{
    dim3 Bs( BLOCK_SIZE );
    dim3 Gs( magma_ceildiv( num_rows, BLOCK_SIZE ) );
    magma_zcgs_4_kernel<<< Gs, Bs, 0, queue->cuda_stream() >>>( num_rows, num_cols, alpha, u_hat, t, x, r );

    return MAGMA_SUCCESS;
}
