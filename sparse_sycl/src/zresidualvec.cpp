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

#define  r(i_)  (r->dval)+i_*dofs
#define  b(i_)  (b.dval)+i_*dofs

/**
    Purpose
    -------

    Computes the residual r = b-Ax for a solution approximation x.
    It returns both, the actual residual and the residual vector

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in]
    b           magma_z_matrix
                RHS b

    @param[in]
    x           magma_z_matrix
                solution approximation

    @param[in,out]
    r           magma_z_matrix*
                residual vector 
                
    @param[out]
    res         magmaDoubleComplex*
                return residual 

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zresidualvec(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix x,
    magma_z_matrix *r, double *res,
    magma_queue_t queue )
{
    magma_int_t info =0;

    // some useful variables
    magmaDoubleComplex zero = MAGMA_Z_ZERO, one = MAGMA_Z_ONE,
                                            mone = MAGMA_Z_NEG_ONE;
    magma_int_t dofs = A.num_rows;
    
    if ( A.num_rows == b.num_rows ) {
        CHECK( magma_z_spmv( mone, A, x, zero, *r, queue ));      // r = A x
        magma_zaxpy( dofs, one, b.dval, 1, r->dval, 1, queue );          // r = r - b
        *res =  magma_dznrm2( dofs, r->dval, 1, queue );            // res = ||r||
        //               /magma_dznrm2( dofs, b.dval, 1, queue );               /||b||
        //printf( "relative residual: %e\n", *res );
    } else if ((b.num_rows*b.num_cols)%A.num_rows== 0 ) {
        magma_int_t num_vecs = b.num_rows*b.num_cols/A.num_rows;

        CHECK( magma_z_spmv( mone, A, x, zero, *r, queue ));           // r = A x

        for( magma_int_t i=0; i<num_vecs; i++) {
            magma_zaxpy( dofs, one, b(i), 1, r(i), 1, queue );   // r = r - b
            res[i] =  magma_dznrm2( dofs, r(i), 1, queue );        // res = ||r||
        }
        //               /magma_dznrm2( dofs, b.dval, 1, queue );               /||b||
        //printf( "relative residual: %e\n", *res );
    } else {
        printf("%%error: dimensions do not match.\n");
        info = MAGMA_ERR_NOT_SUPPORTED;
    }

cleanup:
    return info;
}
