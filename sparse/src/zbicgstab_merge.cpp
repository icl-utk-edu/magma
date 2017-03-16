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

    Solves a system of linear equations
       A * X = B
    where A is a general matrix.
    This is a GPU implementation of the Biconjugate Gradient Stabilized method.

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
magma_zbicgstab_merge(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // prepare solver feedback
    solver_par->solver = Magma_BICGSTAB;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;

    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    
    magma_int_t dofs = A.num_rows * b.num_cols;

    // workspace
    magma_z_matrix r={Magma_CSR}, rr={Magma_CSR}, p={Magma_CSR}, v={Magma_CSR}, 
    s={Magma_CSR}, t={Magma_CSR}, d1={Magma_CSR}, d2={Magma_CSR};
    CHECK( magma_zvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &rr,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &s, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &t, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &d1, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &d2, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

    
    // solver variables
    magmaDoubleComplex alpha, beta, omega, rho_old, rho_new;
    double betanom, nom0, r0, res, nomb;
    res=0;
    //double nom;
    //double den;

    // solver setup
    CHECK(  magma_zresidualvec( A, b, *x, &r, &nom0, queue));
    magma_zcopy( dofs, r.dval, 1, rr.dval, 1, queue );                  // rr = r
    betanom = nom0;
    //nom = nom0*nom0;
    rho_new = magma_zdotc( dofs, r.dval, 1, r.dval, 1, queue );             // rho=<rr,r>
    rho_old = omega = alpha = MAGMA_Z_MAKE( 1.0, 0. );
    solver_par->init_res = nom0;

    CHECK( magma_z_spmv( c_one, A, r, c_zero, v, queue ));              // z = A r
    //den = MAGMA_Z_REAL( magma_zdotc( dofs, v.dval, 1, r.dval, 1), queue ); // den = z' * r

    nomb = magma_dznrm2( dofs, b.dval, 1, queue );
    if ( nomb == 0.0 ){
        nomb=1.0;
    }       
    if ( (r0 = nomb * solver_par->rtol) < ATOLERANCE ){
        r0 = ATOLERANCE;
    }
    
    solver_par->final_res = solver_par->init_res;
    solver_par->iter_res = solver_par->init_res;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = nom0;
        solver_par->timing[0] = 0.0;
    }
    if ( nomb < r0 ) {
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
        rho_old = rho_new;                                    // rho_old=rho

        rho_new = magma_zdotc( dofs, rr.dval, 1, r.dval, 1, queue );  // rho=<rr,r>
        beta = rho_new/rho_old * alpha/omega;   // beta=rho/rho_old *alpha/omega
        if( magma_z_isnan_inf( beta ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
        
        // p = r + beta * ( p - omega * v )
        magma_zbicgstab_1(  
        r.num_rows, 
        r.num_cols, 
        beta,
        omega,
        r.dval, 
        v.dval,
        p.dval,
        queue );

        CHECK( magma_z_spmv( c_one, A, p, c_zero, v, queue ));      // v = Ap
        solver_par->spmv_count++;
        //alpha = rho_new / tmpval;
        alpha = rho_new /magma_zdotc( dofs, rr.dval, 1, v.dval, 1, queue );
        if( magma_z_isnan_inf( alpha ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
        // s = r - alpha v
        magma_zbicgstab_2(  
        r.num_rows, 
        r.num_cols, 
        alpha,
        r.dval,
        v.dval,
        s.dval, 
        queue );

        CHECK( magma_z_spmv( c_one, A, s, c_zero, t, queue ));       // t=As
        solver_par->spmv_count++;
        omega = magma_zdotc( dofs, t.dval, 1, s.dval, 1, queue )   // omega = <s,t>/<t,t>
                   / magma_zdotc( dofs, t.dval, 1, t.dval, 1, queue );
                        
        // x = x + alpha * p + omega * s
        // r = s - omega * t
        magma_zbicgstab_3(  
        r.num_rows, 
        r.num_cols, 
        alpha,
        omega,
        p.dval,
        s.dval,
        t.dval,
        x->dval,
        r.dval,
        queue );

        res = betanom = magma_dznrm2( dofs, r.dval, 1, queue );

        //nom = betanom*betanom;

        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }

        if ( res/nomb <= solver_par->rtol || res <= solver_par->atol ){
            break;
        }
    }
    while ( solver_par->numiter+1 <= solver_par->maxiter );
    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    CHECK(  magma_zresidualvec( A, b, *x, &r, &residual, queue));
    solver_par->iter_res = res;
    solver_par->final_res = residual;

    if ( solver_par->numiter < solver_par->maxiter && info == MAGMA_SUCCESS ) {
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
        if( solver_par->iter_res < solver_par->rtol*nomb ||
            solver_par->iter_res < solver_par->atol ) {
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
    magma_zmfree(&r, queue );
    magma_zmfree(&rr, queue );
    magma_zmfree(&p, queue );
    magma_zmfree(&v, queue );
    magma_zmfree(&s, queue );
    magma_zmfree(&t, queue );
    magma_zmfree(&d1, queue );
    magma_zmfree(&d2, queue );

    solver_par->info = info;
    return info;
}   /* magma_zbicgstab_merge */
