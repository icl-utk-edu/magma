/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma_internal.h"
#include "commonblas_z.h"

// 512 is maximum number of threads for CUDA capability 1.x
#define BLOCK_SIZE 512

#define COMPLEX


/******************************************************************************/
__global__
void magma_zlarfgx_gpu_kernel( int n, magmaDoubleComplex* dx0, magmaDoubleComplex* dx,
                               magmaDoubleComplex *dtau, double *dxnorm,
                               magmaDoubleComplex *dA, int it)
{
    const int i = threadIdx.x;
    const int j = i + BLOCK_SIZE * blockIdx.x;
    __shared__ magmaDoubleComplex scale;
    __shared__ double xnorm;
  
    magmaDoubleComplex dxi;

    if ( j < n-1 )
        dxi = dx[j];
  
    if ( i == 0 ) {
        xnorm = *dxnorm;
        #ifdef REAL
        double alpha = *dx0;
        double alphai = MAGMA_Z_ZERO;
        if ( (xnorm == 0 && alphai == MAGMA_Z_ZERO ) || n == 1 )
        #else
        magmaDoubleComplex alpha = *dx0;
        double alphar =  MAGMA_Z_REAL(alpha), alphai = MAGMA_Z_IMAG(alpha);
        if ( (xnorm == 0 && alphai == MAGMA_Z_ZERO ) || n == 0 )
        #endif
        {
            *dtau = MAGMA_Z_ZERO;
            *dA   = *dx0;
        }
        else {
            #ifdef REAL
                // no need to compute the norm as it is passed as input
                double beta  = xnorm; // sqrt( alpha*alpha + xnorm*xnorm );
                beta  = -copysign( beta, alpha );
     
                // todo: deal with badly scaled vectors (see lapack's larfg)
                if (j == 0) {
                    *dtau = (beta - alpha) / beta;
                    //*dx0  = 1.; //cannot be done here because raise condition all threadblock need to read it for alpha
                    *dA   = beta;
                }
    
                scale = 1. / (alpha - beta);
            #else
                // no need to compute the norm as it is passed as input
                double beta  = xnorm; // sqrt( alphar*alphar + alphai*alphai + xnorm*xnorm );
                beta  = -copysign( beta, alphar );
    
                // todo: deal with badly scaled vectors (see lapack's larfg)
                if (j == 0) {
                    *dtau = MAGMA_Z_MAKE((beta - alphar)/beta, -alphai/beta);
                    //*dx0  = MAGMA_Z_MAKE(  1., 0.); //cannot be done here because raise condition all threadblock need to read it for alpha
                    *dA   = MAGMA_Z_MAKE(beta, 0.);
                }
    
                alpha = MAGMA_Z_MAKE( MAGMA_Z_REAL(alpha) - beta, MAGMA_Z_IMAG(alpha));
                scale = MAGMA_Z_DIV( MAGMA_Z_ONE, alpha);
            #endif
        }
    }

    // scale x
    __syncthreads();
    if ( xnorm != 0 && j < n-1)
        dx[j] = MAGMA_Z_MUL(dxi, scale);

    if (j < it) {
        *( dA-it+j) = *(dx0-it+j);
        *(dx0-it+j) = MAGMA_Z_MAKE(0., 0.);
    }
}


/***************************************************************************//**
    Generates Householder elementary reflector H = I - tau v v^T to reduce
        H [ dx0 ] = [ beta ]
          [ dx  ]   [ 0    ]
    with beta = ¬±norm( [dx0, dx] ) = ¬±dxnorm[0].
    Stores v over dx; first element of v is 1 and is not stored.
    Stores beta over dx0.
    Stores tau.
    
    The difference with LAPACK's zlarfg is that the norm of dx, and hance beta,
    are computed outside the routine and passed to it in dxnorm (array on the GPU).
*******************************************************************************/
extern "C" void
magma_zlarfgx_gpu(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dA, magma_int_t iter, 
    magma_queue_t queue )
{
    dim3 blocks( magma_ceildiv( n, BLOCK_SIZE ) );
    dim3 threads( BLOCK_SIZE );
 
    magma_zlarfgx_gpu_kernel
        <<< blocks, threads, 0, queue->cuda_stream() >>>
        ( n, dx0, dx, dtau, dxnorm, dA, iter);
}


/***************************************************************************//**
    Generates Householder elementary reflector H = I - tau v v^T to reduce
        H [ dx0 ] = [ beta ]
          [ dx  ]   [ 0    ]
    with beta = ¬±norm( [dx0, dx] ) = ¬±dxnorm[0].
    Stores v over dx; first element of v is 1 and is not stored.
    Stores beta over dx0.
    Stores tau.
    
    The difference with LAPACK's zlarfg is that the norm of dx, and hance beta,
    are computed outside the routine and passed to it in dxnorm (array on the GPU).
*******************************************************************************/
extern "C" void
magma_zlarfgtx_gpu(
    magma_int_t n,
    magmaDoubleComplex_ptr dx0,
    magmaDoubleComplex_ptr dx,
    magmaDoubleComplex_ptr dtau,
    magmaDouble_ptr        dxnorm,
    magmaDoubleComplex_ptr dA, magma_int_t iter,
    magmaDoubleComplex_ptr V,  magma_int_t ldv,
    magmaDoubleComplex_ptr T,  magma_int_t ldt,
    magmaDoubleComplex_ptr dwork,
    magma_queue_t queue )
{
    /*  Generate the elementary reflector H(iter)  */
    magma_zlarfgx_gpu(n, dx0, dx, dtau, dxnorm, dA, iter, queue);
    
    if (iter == 0) {
        magmaDoubleComplex tt = MAGMA_Z_ONE;
        magmablas_zlacpy( MagmaFull, 1, 1, dtau, 1, T+iter+iter*ldt, 1, queue );
        magma_zsetmatrix( 1, 1, &tt, 1, dx0, 1, queue );
    }
    else {
        /* Compute the iter-th column of T */
        magma_zgemv_kernel3
            <<< iter, BLOCK_SIZE, 0, queue->cuda_stream() >>>
            ( n, V, ldv, dx0, dwork, dtau );
        
        magma_ztrmv_kernel2
            <<< iter, iter,       0, queue->cuda_stream() >>>
            ( T, ldt, dwork, T+iter*ldt, dtau );
    }
}
