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

    This is an interface that allows to use any iterative solver on the linear
    system Ax = b. All linear algebra objects are expected to be on the device,
    the linear algebra objects are MAGMA-sparse specific structures 
    (dense matrix b, dense matrix x, sparse/dense matrix A).
    The additional parameter zopts contains information about the solver
    and the preconditioner.
    * the type of solver
    * the relative / absolute stopping criterion
    * the maximum number of iterations
    * the preconditioner type
    * ...
    Please see magmasparse_types.h for details about the fields and
    magma_zutil_sparse.cpp for the possible options.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                sparse matrix A

    @param[in]
    b           magma_z_matrix
                input vector b

    @param[in]
    x           magma_z_matrix*
                output vector x

    @param[in]
    zopts     magma_zopts
              options for solver and preconditioner
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_solver(
    magma_z_matrix A, magma_z_matrix b,
    magma_z_matrix *x, magma_zopts *zopts,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // make sure RHS is a dense matrix
    if ( b.storage_type != Magma_DENSE ) {
        printf( "error: sparse RHS not yet supported.\n" );
        return MAGMA_ERR_NOT_SUPPORTED;
    }
    if( b.num_cols == 1 ){
        switch( zopts->solver_par.solver ) {
            case  Magma_BICG:
                    CHECK( magma_zbicg( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PBICG:
                    CHECK( magma_zpbicg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_BICGSTAB:
                    CHECK( magma_zbicgstab( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_BICGSTABMERGE: 
                    CHECK( magma_zbicgstab_merge( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PBICGSTAB:
                    CHECK( magma_zpbicgstab( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PBICGSTABMERGE:
                    CHECK( magma_zpbicgstab_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_CG:
                    CHECK( magma_zcg_res( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_CGMERGE:
                    CHECK( magma_zcg_merge( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PCG:
                    CHECK( magma_zpcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PCGMERGE:
                    CHECK( magma_zpcg_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_CGS:
                    CHECK( magma_zcgs( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_CGSMERGE:
                    CHECK( magma_zcgs_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_PCGS:
                    CHECK( magma_zpcgs( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_PCGSMERGE:
                    CHECK( magma_zpcgs_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_QMR:
                    CHECK( magma_zqmr( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_QMRMERGE:
                    CHECK( magma_zqmr_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_PQMR:
                    CHECK( magma_zpqmr( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_PQMRMERGE:
                    CHECK( magma_zpqmr_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_TFQMR:
                    CHECK( magma_ztfqmr( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_TFQMRMERGE:
                    CHECK( magma_ztfqmr_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_PTFQMR:
                    CHECK( magma_zptfqmr( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_PTFQMRMERGE:
                    CHECK( magma_zptfqmr_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_GMRES:
                    CHECK( magma_zfgmres( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PGMRES:
                    CHECK( magma_zfgmres( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_IDR:
                    CHECK( magma_zidr( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_IDRMERGE:
                    CHECK( magma_zidr_merge( A, b, x, &zopts->solver_par, queue )); break;
                    //CHECK( magma_zidr_strms( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_PIDR:
                    CHECK( magma_zpidr( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PIDRMERGE:
                    CHECK( magma_zpidr_merge( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
                    //CHECK( magma_zpidr_strms( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_LOBPCG:
                    CHECK( magma_zlobpcg( A, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_ITERREF:
                    CHECK( magma_ziterref( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_JACOBI:
                    CHECK( magma_zjacobi( A, b, x, &zopts->solver_par, queue )); break;
            case  Magma_BAITER:
                    CHECK( magma_zbaiter( A, b, x, &zopts->solver_par, &zopts->precond_par, queue ) ); break;
            case  Magma_BAITERO:
                    CHECK( magma_zbaiter_overlap( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_BOMBARD:
                    CHECK( magma_zbombard( A, b, x, &zopts->solver_par, queue ) ); break;
            case  Magma_BOMBARDMERGE:
                    CHECK( magma_zbombard_merge( A, b, x, &zopts->solver_par, queue ) ); break;
            // case  Magma_PARDISO:
            //         CHECK( magma_zpardiso( A, b, x, &zopts->solver_par, queue ) ); break;
            default:
                    printf("error: solver class not supported.\n"); break;
        }
    }
    else {
        switch( zopts->solver_par.solver ) {
            case  Magma_CG:
                    CHECK( magma_zbpcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_PCG:
                    CHECK( magma_zbpcg( A, b, x, &zopts->solver_par, &zopts->precond_par, queue )); break;
            case  Magma_LOBPCG:
                    CHECK( magma_zlobpcg( A, &zopts->solver_par, &zopts->precond_par, queue )); break;
            default:
                    printf("error: only 1 RHS supported for this solver class.\n"); break;
        }
    }
cleanup:
    return info; 
}
