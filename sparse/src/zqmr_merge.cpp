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
    where A is a complex general matrix A.
    This is a GPU implementation of the Quasi-Minimal Residual method (QMR).

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
magma_zqmr_merge(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // prepare solver feedback
    solver_par->solver = Magma_QMRMERGE;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    
    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE;
    // solver variables
    double nom0, r0, res=0, nomb;
    magmaDoubleComplex rho = c_one, rho1 = c_one, eta = -c_one , pds = c_one, 
                        thet = c_one, thet1 = c_one, epsilon = c_one, 
                        beta = c_one, delta = c_one, pde = c_one, rde = c_one,
                        gamm = c_one, gamm1 = c_one, psi = c_one;
    
    magma_int_t dofs = A.num_rows* b.num_cols;

    // need to transpose the matrix
    magma_z_matrix AT={Magma_CSR};
    
    // GPU workspace
    magma_z_matrix r={Magma_CSR}, r_tld={Magma_CSR},
                    v={Magma_CSR}, w={Magma_CSR}, wt={Magma_CSR},
                    d={Magma_CSR}, s={Magma_CSR}, z={Magma_CSR}, q={Magma_CSR}, 
                    p={Magma_CSR}, pt={Magma_CSR}, y={Magma_CSR};
    CHECK( magma_zvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &r_tld, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &w, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &wt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &d, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &s, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &z, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &q, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &pt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &y, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

    
    // solver setup
    CHECK(  magma_zresidualvec( A, b, *x, &r, &nom0, queue));
    solver_par->init_res = nom0;
    magma_zcopy( dofs, r.dval, 1, r_tld.dval, 1, queue );   
    magma_zcopy( dofs, r.dval, 1, y.dval, 1, queue );   
    magma_zcopy( dofs, r.dval, 1, v.dval, 1, queue );  
    magma_zcopy( dofs, r.dval, 1, wt.dval, 1, queue );   
    magma_zcopy( dofs, r.dval, 1, z.dval, 1, queue );  
    
    // transpose the matrix
    magma_zmtransposeconjugate( A, &AT, queue );
    
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
        solver_par->res_vec[0] = (real_Double_t)nom0;
        solver_par->timing[0] = 0.0;
    }
    if ( nom0 < r0 ) {
        info = MAGMA_SUCCESS;
        goto cleanup;
    }

    psi = magma_zsqrt( magma_zdotc( dofs, z.dval, 1, z.dval, 1, queue ));
    rho = magma_zsqrt( magma_zdotc( dofs, y.dval, 1, y.dval, 1, queue ));
    
        // v = y / rho
        // y = y / rho
        // w = wt / psi
        // z = z / psi
    magma_zqmr_1(  
    r.num_rows, 
    r.num_cols, 
    rho,
    psi,
    y.dval, 
    z.dval,
    v.dval,
    w.dval,
    queue );
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
        if( magma_z_isnan_inf( rho ) || magma_z_isnan_inf( psi ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
 
            // delta = z' * y;
        delta = magma_zdotc( dofs, z.dval, 1, y.dval, 1, queue );
        
        if( magma_z_isnan_inf( delta ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
        
        // no precond: yt = y, zt = z
        //magma_zcopy( dofs, y.dval, 1, yt.dval, 1 );
        //magma_zcopy( dofs, z.dval, 1, zt.dval, 1 );
        
        if( solver_par->numiter == 1 ){
                // p = y;
                // q = z;
            magma_zcopy( dofs, y.dval, 1, p.dval, 1, queue );
            magma_zcopy( dofs, z.dval, 1, q.dval, 1, queue );
        }
        else{
            pde = psi * delta / epsilon;
            rde = rho * MAGMA_Z_CONJ(delta/epsilon);
            
                // p = y - pde * p
                // q = z - rde * q
            magma_zqmr_2(  
            r.num_rows, 
            r.num_cols, 
            pde,
            rde,
            y.dval,
            z.dval,
            p.dval, 
            q.dval, 
            queue );
        }
        if( magma_z_isnan_inf( rho ) || magma_z_isnan_inf( psi ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
        
        CHECK( magma_z_spmv( c_one, A, p, c_zero, pt, queue ));
        solver_par->spmv_count++;
            // epsilon = q' * pt;
        epsilon = magma_zdotc( dofs, q.dval, 1, pt.dval, 1, queue );
        beta = epsilon / delta;

        if( magma_z_isnan_inf( epsilon ) || magma_z_isnan_inf( beta ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
            // v = pt - beta * v
            // y = v
        magma_zqmr_3(  
        r.num_rows, 
        r.num_cols, 
        beta,
        pt.dval,
        v.dval,
        y.dval,
        queue );
        
        
        rho1 = rho;      
            // rho = norm(y);
        rho = magma_zsqrt( magma_zdotc( dofs, y.dval, 1, y.dval, 1, queue ));
        
            // wt = A' * q - beta' * w;
        CHECK( magma_z_spmv( c_one, AT, q, c_zero, wt, queue ));
        solver_par->spmv_count++;
        magma_zaxpy( dofs, - MAGMA_Z_CONJ( beta ), w.dval, 1, wt.dval, 1, queue );  
        
                    // no precond: z = wt
        magma_zcopy( dofs, wt.dval, 1, z.dval, 1, queue );
        


        thet1 = thet;        
        thet = rho / (gamm * MAGMA_Z_MAKE( MAGMA_Z_ABS(beta), 0.0 ));
        gamm1 = gamm;        
        
        gamm = c_one / magma_zsqrt(c_one + thet*thet);        
        eta = - eta * rho1 * gamm * gamm / (beta * gamm1 * gamm1);        

        if( magma_z_isnan_inf( thet ) || magma_z_isnan_inf( gamm ) || magma_z_isnan_inf( eta ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
        
        if( solver_par->numiter == 1 ){
                // d = eta * p + pds * d;
                // s = eta * pt + pds * d;
                // x = x + d;
                // r = r - s;
            magma_zqmr_4(  
            r.num_rows, 
            r.num_cols, 
            eta,
            p.dval,
            pt.dval,
            d.dval, 
            s.dval, 
            x->dval, 
            r.dval, 
            queue );
        }
        else{
            pds = (thet1 * gamm) * (thet1 * gamm);
            
                // d = eta * p + pds * d;
                // s = eta * pt + pds * d;
                // x = x + d;
                // r = r - s;
            magma_zqmr_5(  
            r.num_rows, 
            r.num_cols, 
            eta,
            pds,
            p.dval,
            pt.dval,
            d.dval, 
            s.dval, 
            x->dval, 
            r.dval, 
            queue );
        }
            // psi = norm(z);
        psi = magma_zsqrt( magma_zdotc( dofs, z.dval, 1, z.dval, 1, queue ) );
        
        res = magma_dznrm2( dofs, r.dval, 1, queue );
        
        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        
        // v = y / rho
        // y = y / rho
        // w = wt / psi
        // z = z / psi
        magma_zqmr_1(  
        r.num_rows, 
        r.num_cols, 
        rho,
        psi,
        y.dval, 
        z.dval,
        v.dval,
        w.dval,
        queue );

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
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
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
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
                solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) res;
                solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
            }
        }
        info = MAGMA_DIVERGENCE;
    }
    
cleanup:
    magma_zmfree(&r, queue );
    magma_zmfree(&r_tld, queue );
    magma_zmfree(&v,  queue );
    magma_zmfree(&w,  queue );
    magma_zmfree(&wt, queue );
    magma_zmfree(&d,  queue );
    magma_zmfree(&s,  queue );
    magma_zmfree(&z,  queue );
    magma_zmfree(&q,  queue );
    magma_zmfree(&p,  queue );
    magma_zmfree(&pt, queue );
    magma_zmfree(&y,  queue );
    magma_zmfree(&AT, queue );
    
    solver_par->info = info;
    return info;
}   /* magma_zqmr_merge */
