/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/
#include "magmasparse_internal.h"


/**
    Purpose
    -------

    This is an interface to any custom sparse matrix vector product.
    It should compute y = alpha*FUNCTION(x) + beta*y
    The vectors are located on the device, the scalars on the CPU.


    Arguments
    ---------

    @param[in]
    m           magma_int_t
                number of rows
                
    @param[in]
    n           magma_int_t
                number of columns
                
    @param[in]
    alpha       magmaDoubleComplex
                scalar alpha
                
    @param[in]
    x           magmaDoubleComplex *
                input vector x
                
    @param[in]
    beta        magmaDoubleComplex
                scalar beta
    @param[out]
    y           magmaDoubleComplex *
                output vector y
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zcustomspmv(
    magma_int_t m,
    magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex beta,
    magmaDoubleComplex *x,
    magmaDoubleComplex *y,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // vector access via x.dval, y.dval
    // sizes are x.num_rows, x.num_cols
    
    magma_zge3pt( m, n, alpha, beta, x, y, queue );
    
    return info;
}
