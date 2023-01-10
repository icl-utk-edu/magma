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

#define  q(i)     (q.dval + (i)*dofs)

/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a general matrix.
    This is a GPU implementation of the Biconjugate Gradient Stabilized method.
    The difference to magma_zbicgstab is that we use specifically designed
    kernels merging multiple operations into one kernel.

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
magma_zbicgstab_merge3(
    magma_z_matrix A, magma_z_matrix b,
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // prepare solver feedback
    solver_par->solver = Magma_BICGSTABMERGE;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    
    // solver variables
    magmaDoubleComplex alpha, beta, omega, rho_old, rho_new, *skp_h={0};
    double nom, nom0, betanom, nomb;

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE;
    
    magma_int_t dofs = A.num_rows;

    // workspace
    magma_z_matrix q={Magma_CSR}, r={Magma_CSR}, rr={Magma_CSR}, p={Magma_CSR}, v={Magma_CSR}, s={Magma_CSR}, t={Magma_CSR};
    magmaDoubleComplex *d1=NULL, *d2=NULL, *skp=NULL;
    d1 = NULL;
    d2 = NULL;
    skp = NULL;
    CHECK( magma_zmalloc( &d1, dofs*(2) ));
    CHECK( magma_zmalloc( &d2, dofs*(2) ));
    // array for the parameters
    CHECK( magma_zmalloc( &skp, 8 ));
    // skp = [alpha|beta|omega|rho_old|rho|nom|tmp1|tmp2]
    CHECK( magma_zvinit( &q, Magma_DEV, dofs*6, 1, c_zero, queue ));

    // q = rr|r|p|v|s|t
    rr.memory_location = Magma_DEV; rr.dval = NULL; rr.num_rows = rr.nnz = dofs; rr.num_cols = 1; rr.storage_type = Magma_DENSE;
    r.memory_location = Magma_DEV; r.dval = NULL; r.num_rows = r.nnz = dofs; r.num_cols = 1; r.storage_type = Magma_DENSE;
    p.memory_location = Magma_DEV; p.dval = NULL; p.num_rows = p.nnz = dofs; p.num_cols = 1; p.storage_type = Magma_DENSE;
    v.memory_location = Magma_DEV; v.dval = NULL; v.num_rows = v.nnz = dofs; v.num_cols = 1; v.storage_type = Magma_DENSE;
    s.memory_location = Magma_DEV; s.dval = NULL; s.num_rows = s.nnz = dofs; s.num_cols = 1; s.storage_type = Magma_DENSE;
    t.memory_location = Magma_DEV; t.dval = NULL; t.num_rows = t.nnz = dofs; t.num_cols = 1; t.storage_type = Magma_DENSE;

    rr.dval = q(0);
    r.dval = q(1);
    p.dval = q(2);
    v.dval = q(3);
    s.dval = q(4);
    t.dval = q(5);

    // solver setup
    CHECK(  magma_zresidualvec( A, b, *x, &r, &nom0, queue));
    magma_zcopy( dofs, r.dval, 1, q(0), 1, queue );                            // rr = r
    magma_zcopy( dofs, r.dval, 1, q(1), 1, queue );                            // q = r
    betanom = nom0;
    nom = nom0*nom0;
    rho_new = magma_zdotc( dofs, r.dval, 1, r.dval, 1, queue );             // rho=<rr,r>
    rho_old = omega = alpha = MAGMA_Z_MAKE( 1.0, 0. );
    beta = rho_new;
    solver_par->init_res = nom0;
    // array on host for the parameters
    CHECK( magma_zmalloc_cpu( &skp_h, 8 ));
    
    nomb = magma_dznrm2( dofs, b.dval, 1, queue );
    if ( nomb == 0.0 ){
        nomb=1.0;
    }       
    solver_par->final_res = solver_par->init_res;
    solver_par->iter_res = solver_par->init_res;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = nom0;
        solver_par->timing[0] = 0.0;
    }
    
    skp_h[0]=alpha;
    skp_h[1]=beta;
    skp_h[2]=omega;
    skp_h[3]=rho_old;
    skp_h[4]=rho_new;
    skp_h[5]=MAGMA_Z_MAKE(nom, 0.0);
    magma_zsetvector( 8, skp_h, 1, skp, 1, queue );
    CHECK( magma_z_spmv( c_one, A, r, c_zero, v, queue ));             // z = A r
    nomb = magma_dznrm2( dofs, b.dval, 1, queue );
    if( nom0 < solver_par->atol ||
        nom0/nomb < solver_par->rtol ){
        info = MAGMA_SUCCESS;
        goto cleanup;
    }

    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );

    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    // start iteration
    do
    {
        solver_par->numiter++;

        // computes p=r+beta*(p-omega*v)
        CHECK( magma_zbicgmerge1( dofs, skp, v.dval, r.dval, p.dval, queue ));

        CHECK( magma_z_spmv( c_one, A, p, c_zero, v, queue ));         // v = Ap
        solver_par->spmv_count++;
        CHECK( magma_zmdotc( dofs, 1, q.dval, v.dval, d1, d2, skp, queue ));
        CHECK( magma_zbicgmerge4(  1, skp, queue ));
        CHECK( magma_zbicgmerge2( dofs, skp, r.dval, v.dval, s.dval, queue )); // s=r-alpha*v

        CHECK( magma_z_spmv( c_one, A, s, c_zero, t, queue ));         // t=As
        solver_par->spmv_count++;
        CHECK( magma_zmdotc( dofs, 2, q.dval+4*dofs, t.dval, d1, d2, skp+6, queue ));
        CHECK( magma_zbicgmerge4(  2, skp, queue ));

        CHECK( magma_zbicgmerge_xrbeta( dofs, d1, d2, q.dval, r.dval, p.dval,
                                                    s.dval, t.dval, x->dval, skp, queue ));

        // check stopping criterion
        magma_zgetvector_async( 1 , skp+5, 1, skp_h+5, 1, queue );
        betanom = sqrt(MAGMA_Z_REAL(skp_h[5]));

        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) betanom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        
        if (  betanom  < solver_par->atol || 
              betanom/nomb < solver_par->rtol ) {
            break;
        }
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );
    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    CHECK(  magma_zresidualvec( A, b, *x, &r, &residual, queue));
    solver_par->iter_res = betanom;
    solver_par->final_res = residual;

    if ( solver_par->numiter < solver_par->maxiter ) {
        info = MAGMA_SUCCESS;
    } else if ( solver_par->init_res > solver_par->final_res ) {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) betanom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_SLOW_CONVERGENCE;
        if( solver_par->iter_res < solver_par->atol ||
            solver_par->iter_res/solver_par->init_res < solver_par->rtol ){
            info = MAGMA_SUCCESS;
        }
    }
    else {
        if ( solver_par->verbose > 0 ) {
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) betanom;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_zmfree(&q, queue );  // frees all vectors
    magma_free(d1);
    magma_free(d2);
    magma_free( skp );
    magma_free_cpu( skp_h );

    solver_par->info = info;
    return info;
}   /* zbicgstab_merge */
