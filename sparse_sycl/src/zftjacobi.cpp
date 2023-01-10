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

    Iterates the solution approximation according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    This routine takes the system matrix A and the RHS b as input.
    This is the fault-tolerant version of Jacobi according to ScalLA'15.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix M = D^(-1) * (L+U)
                
    @param[in]
    b           magma_z_matrix
                input RHS b
                
    @param[in,out]
    x           magma_z_matrix*
                iteration vector x

    @param[in,out]
    solver_par  magma_z_solver_par*
                solver parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zftjacobi(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // some useful variables
    real_Double_t tempo1, tempo2, runtime=0;
    double residual;
    
    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magma_int_t dofs = A.num_cols;
    magma_int_t k = solver_par->verbose;
    
    magma_z_matrix xkm2 = {Magma_CSR};
    magma_z_matrix xkm1 = {Magma_CSR};
    magma_z_matrix xk = {Magma_CSR};
    magma_z_matrix z = {Magma_CSR};
    magma_z_matrix c = {Magma_CSR};
    magma_int_t *flag_t = NULL;
    magma_int_t *flag_fp = NULL;
    
    double delta = 0.9;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;

    magma_z_matrix r={Magma_CSR}, d={Magma_CSR}, ACSR={Magma_CSR};
    
    CHECK( magma_zmconvert(A, &ACSR, A.storage_type, Magma_CSR, queue ) );

    // prepare solver feedback
    solver_par->solver = Magma_JACOBI;
    solver_par->info = MAGMA_SUCCESS;

    // solver setup
    CHECK( magma_zvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK(  magma_zresidualvec( ACSR, b, *x, &r, &residual, queue));
    solver_par->init_res = residual;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t) residual;
    }
    //nom0 = residual;

    // Jacobi setup
    CHECK( magma_zjacobisetup_diagscal( ACSR, &d, queue ));
    magma_z_solver_par jacobiiter_par;
    if ( solver_par->verbose > 0 ) {
        jacobiiter_par.maxiter = solver_par->verbose;
    }
    else {
        jacobiiter_par.maxiter = solver_par->maxiter;
    }
    k = jacobiiter_par.maxiter;
    

    
    CHECK( magma_zvinit( &xkm2, Magma_DEV, b.num_rows, 1, c_zero, queue ));
    CHECK( magma_zvinit( &xkm1, Magma_DEV, b.num_rows, 1, c_zero, queue ));
    CHECK( magma_zvinit( &xk, Magma_DEV, b.num_rows, 1, c_zero, queue ));
    CHECK( magma_zvinit( &z, Magma_DEV, b.num_rows, 1, c_zero, queue ));
    CHECK( magma_zvinit( &c, Magma_DEV, b.num_rows, 1, c_zero, queue ));

    CHECK( magma_imalloc( &flag_t, b.num_rows ));
    CHECK( magma_imalloc( &flag_fp, b.num_rows ));
    
    if ( solver_par->verbose != 0 ) {
        // k iterations for startup
        tempo1 = magma_sync_wtime( queue );
        CHECK( magma_zjacobispmvupdate(k, ACSR, r, b, d, x, queue ));
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2 - tempo1;
        solver_par->spmv_count=solver_par->spmv_count+k;
        solver_par->numiter=solver_par->numiter+k;
        // save in xkm2
        magma_zcopyvector( dofs, x->dval, 1, xkm2.dval, 1, queue ); 
        // two times k iterations for computing the contraction constants
        tempo1 = magma_sync_wtime( queue );
        CHECK( magma_zjacobispmvupdate(k, ACSR, r, b, d, x, queue ));
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2 - tempo1;
        solver_par->spmv_count=solver_par->spmv_count+k;
        solver_par->numiter=solver_par->numiter+k;
        // save in xkm1
        tempo1 = magma_sync_wtime( queue );
        magma_zcopyvector( dofs, x->dval, 1, xkm1.dval, 1, queue );
        CHECK( magma_zjacobispmvupdate(k, ACSR, r, b, d, x, queue ));
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2 - tempo1;
        solver_par->spmv_count=solver_par->spmv_count+k;
        solver_par->numiter=solver_par->numiter+k;
        // compute contraction constants
        magma_zcopyvector( dofs, x->dval, 1, xk.dval, 1, queue );
        magma_zftjacobicontractions( xkm2, xkm1, xk, &z, &c, queue );
    }
    
    // Jacobi iterator
    do {
        tempo1 = magma_sync_wtime( queue );
        solver_par->numiter = solver_par->numiter+jacobiiter_par.maxiter;
        CHECK( magma_zjacobispmvupdate(jacobiiter_par.maxiter, ACSR, r, b, d, x, queue ));
        if( solver_par->verbose != 0 ){
            magma_zftjacobiupdatecheck( delta, &xk, x, &z, c, flag_t, flag_fp, queue ); 
        }
        solver_par->spmv_count=solver_par->spmv_count+k;
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2 - tempo1;
        if ( solver_par->verbose > 0 ) {
            CHECK(  magma_zresidualvec( ACSR, b, *x, &r, &residual, queue));
            solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                = (real_Double_t) residual;
            solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                = (real_Double_t) runtime;
        }
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );

    solver_par->runtime = (real_Double_t) runtime;
    CHECK(  magma_zresidualvec( A, b, *x, &r, &residual, queue));
    solver_par->final_res = residual;

    if ( solver_par->init_res > solver_par->final_res )
        info = MAGMA_SUCCESS;
    else
        info = MAGMA_DIVERGENCE;
    
cleanup:
    magma_zmfree( &r, queue );
    magma_zmfree( &d, queue );
    magma_zmfree( &ACSR, queue );
    magma_zmfree( &xkm2, queue );
    magma_zmfree( &xkm1, queue ); 
    magma_zmfree( &xk, queue );
    magma_zmfree( &z, queue ); 
    magma_zmfree( &c, queue ); 
    magma_free( flag_t );
    magma_free( flag_fp );
    
    solver_par->info = info;
    return info;
}   /* magma_zftjacobi */
