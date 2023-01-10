/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Hartwig Anzt

       @precisions normal z -> s d c
*/

#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex Hermitian N-by-N positive definite matrix A.
    This is a GPU implementation of the Jacobi method allowing for
    domain overlap.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in]
    b           magma_z_matrix
                RHS b

    @param[in,out]
    x           magma_z_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_z_solver_par*
                solver parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgesv
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobidomainoverlap(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    
    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    
    magma_int_t dofs = A.num_rows*b.num_cols;
    //double nom0 = 0.0;
    // generate the domain overlap
    magma_int_t num_ind = 0;
    magma_index_t *indices={0}, *hindices={0};
    
    magma_z_matrix r={Magma_CSR}, d={Magma_CSR};
    magma_z_matrix hA={Magma_CSR};
    
    // prepare solver feedback
    solver_par->solver = Magma_JACOBI;

    real_Double_t tempo1, tempo2;
    double residual;
    CHECK( magma_zresidual( A, b, *x, &residual, queue ));
    solver_par->init_res = residual;
    solver_par->res_vec = NULL;
    solver_par->timing = NULL;



    CHECK( magma_zvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_z_spmv( c_one, A, *x, c_zero, r, queue ));          // r = A x
    magma_zaxpy( dofs, c_neg_one, b.dval, 1, r.dval, 1, queue );           // r = r - b
    //nom0 = magma_dznrm2( dofs, r.dval, 1, queue );                      // den = || r ||

    // Jacobi setup
    CHECK( magma_zjacobisetup_diagscal( A, &d, queue ));
    magma_z_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = solver_par->maxiter;
    

    CHECK( magma_index_malloc_cpu( &hindices, A.num_rows*10 ));
    CHECK( magma_index_malloc( &indices, A.num_rows*10 ));

    CHECK( magma_zmtransfer( A, &hA, Magma_DEV, Magma_CPU, queue ));
    CHECK( magma_zdomainoverlap( hA.num_rows, &num_ind, hA.row, hA.col, hindices, queue ));
    /*num_ind = A.num_rows*5;
    for(magma_int_t i=0; i<num_ind; i++){
        hindices[i] = i%A.num_rows;
    }
*/
   
    CHECK( magma_index_malloc( &indices, num_ind ));
    magma_index_setvector( num_ind, hindices, 1, indices, 1, queue );
    
    
    
    tempo1 = magma_sync_wtime( queue );

    // Jacobi iterator
    CHECK( magma_zjacobispmvupdateselect(jacobiiter_par.maxiter, num_ind, indices,
                                                        A, r, b, d, r, x, queue ));
    
    //magma_zjacobispmvupdate(jacobiiter_par.maxiter, A, r, b, d, x, queue );

    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    CHECK(  magma_zresidualvec( A, b, *x, &r, &residual, queue));
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if ( solver_par->init_res > solver_par->final_res )
        info = MAGMA_SUCCESS;
    else
        info = MAGMA_DIVERGENCE;
    
    
cleanup:
    magma_free_cpu( hindices );
    magma_zmfree( &r, queue );
    magma_zmfree( &d, queue );
    magma_zmfree(&hA, queue );
    magma_free( indices );
    
    solver_par->info = info;
    return info;
}   /* magma_zjacobi */
