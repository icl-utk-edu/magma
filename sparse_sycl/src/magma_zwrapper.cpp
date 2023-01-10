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


/**
    Purpose
    -------

    This is the interface to MAGMA-sparse functionalities. The zopts structure
    contains all information about which functionality is requested, where it is
    computed etc.

    Arguments
    ---------
    
    @param[in]
    zopts       magma_zopts
                Structure containing all information which node-level operation 
                is requested.

    @param[in]
    A           magma_z_matrix
                Sparse matrix A in CSR format.
                
    @param[in]
    x           magma_z_matrix*
                Output vector x.

    @param[in]
    b           magma_z_matrix
                Input vector b.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zwrapper(
    magma_zopts *zopts,
    magma_z_matrix A, 
    magma_z_matrix *x,
    magma_z_matrix b,
    magma_queue_t queue ){
 
    magma_int_t info = 0;
    
    switch( zopts->operation ) {
        case Magma_SOLVE:
            {
                CHECK( magma_zsolverinfo_init( &zopts->solver_par, &zopts->precond_par, queue ) );
                CHECK( magma_zeigensolverinfo_init( &zopts->solver_par, queue ) );
                CHECK( magma_z_precondsetup( A, b, &zopts->solver_par, &zopts->precond_par, queue ) );
                CHECK( magma_z_solver( A, b, x, zopts, queue ) );
                CHECK( magma_zsolverinfo( &zopts->solver_par, &zopts->precond_par, queue ) );
                CHECK( magma_zsolverinfo_free( &zopts->solver_par, &zopts->precond_par, queue ) );
                break;
            }
        case Magma_GENERATEPREC:
            {
                CHECK( magma_z_precondsetup( A, b, &zopts->solver_par, &zopts->precond_par, queue ) );   
                break;
            }
        case Magma_PRECONDLEFT:
            {
                magma_trans_t trans = MagmaNoTrans;
                CHECK( magma_z_applyprecond_left( trans, A, b, x, &zopts->precond_par, queue ) );
                break;
            }
        case Magma_PRECONDRIGHT:
            {
                magma_trans_t trans = MagmaNoTrans;
                CHECK( magma_z_applyprecond_right( trans, A, b, x, &zopts->precond_par, queue ) );
                break;
            }
        case Magma_SPMV:break;
        default:
            printf("error: no MAGMA-spare operation specified.\n"); break;
    }

cleanup:
    return info; 
}
