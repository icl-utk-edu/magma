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
       A * x = b
    via the block asynchronous iteration method on GPU.

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
    precond_par magma_z_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgesv
    ********************************************************************/

extern "C" magma_int_t
magma_zbaiter(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
        
    // prepare solver feedback
    solver_par->solver = Magma_BAITER;
    
    // some useful variables 
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;

    // initial residual
    real_Double_t tempo1, tempo2, runtime=0;
    double residual;
    magma_int_t localiter = precond_par->maxiter;
    
    magma_z_matrix Ah={Magma_CSR}, ACSR={Magma_CSR}, dA={Magma_CSR}, D={Magma_CSR}, 
                    R={Magma_CSR}, D_d={Magma_CSR}, R_d={Magma_CSR}, r={Magma_CSR};

    CHECK( magma_zmtransfer( A, &Ah, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_zmconvert( Ah, &ACSR, Ah.storage_type, Magma_CSR, queue ));

    CHECK( magma_zmtransfer( ACSR, &dA, Magma_CPU, Magma_DEV, queue ));
    
    CHECK( magma_zvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK(  magma_zresidualvec( dA, b, *x, &r, &residual, queue));
    solver_par->init_res = residual;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t) residual;
    }
    // setup
    CHECK( magma_zcsrsplit( 0, 256, ACSR, &D, &R, queue ));
    CHECK( magma_zmtransfer( D, &D_d, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_zmtransfer( R, &R_d, Magma_CPU, Magma_DEV, queue ));
    
    magma_int_t iterinc;
    if( solver_par->verbose == 0 ){
        iterinc = solver_par->maxiter;
    }
    else{
        iterinc = solver_par->verbose;
    }
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    // block-asynchronous iteration iterator
    do
    {
        tempo1 = magma_sync_wtime( queue );
        solver_par->numiter+= iterinc;
        for( int z=0; z<iterinc; z++){
            CHECK( magma_zbajac_csr( localiter, D_d, R_d, b, x, queue ));
        }
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2-tempo1;
        if ( solver_par->verbose > 0 ) {
        CHECK(  magma_zresidualvec( dA, b, *x, &r, &residual, queue));
            solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                = (real_Double_t) residual;
            solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                = (real_Double_t) runtime;
        }
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );

    solver_par->runtime = runtime;
    CHECK(  magma_zresidual( dA, b, *x, &residual, queue));
    solver_par->final_res = residual;
    solver_par->numiter = solver_par->maxiter;

    if ( solver_par->init_res > solver_par->final_res ){
        info = MAGMA_SUCCESS;
    }
    else {
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_zmfree(&r, queue );
    magma_zmfree(&D, queue );
    magma_zmfree(&R, queue );
    magma_zmfree(&D_d, queue );
    magma_zmfree(&R_d, queue );
    magma_zmfree(&dA, queue );
    magma_zmfree(&ACSR, queue );
    magma_zmfree(&Ah, queue );

    solver_par->info = info;
    return info;
}   /* magma_zbaiter */
