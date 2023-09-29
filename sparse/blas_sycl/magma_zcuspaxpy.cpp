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

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )

/**
    Purpose
    -------

    This is an interface to the cuSPARSE routine csrgeam computing the sum
    of two sparse matrices stored in csr format:

        C = alpha * A + beta * B


    Arguments
    ---------

    @param[in]
    alpha       magmaDoubleComplex*
                scalar

    @param[in]
    A           magma_z_matrix
                input matrix

    @param[in]
    beta        magmaDoubleComplex*
                scalar

    @param[in]
    B           magma_z_matrix
                input matrix

    @param[out]
    AB          magma_z_matrix*
                output matrix AB = alpha * A + beta * B

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/

extern "C" magma_int_t
magma_zcuspaxpy(
    magmaDoubleComplex *alpha, magma_z_matrix A,
    magmaDoubleComplex *beta, magma_z_matrix B,
    magma_z_matrix *AB,
    magma_queue_t queue )
{
   magma_unsupported_sparse(magma_zcuspaxpy);
}
