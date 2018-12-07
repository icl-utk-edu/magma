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

#define PRECISION_z

/*
#define  q(i)     (q.dval + (i)*dofs)
#define  z(i)     (z.dval + (i)*dofs)
#define  H(i,j)  H[(i)   + (j)*(1+ldh)]
#define HH(i,j) HH[(i)   + (j)*ldh]
#define dH(i,j) dH[(i)   + (j)*(1+ldh)]
*/

// simulate 2-D arrays at the cost of some arithmetic
#define V(i) (V.dval+(i)*dofs)
#define W(i) (W.dval+(i)*dofs)
//#define Vv(i) (&V.dval[(i)*n])
//#define Wv(i) (&W.dval[(i)*n])
#define H(i,j) (H[(j)*m1+(i)])
#define ABS(x)   ((x)<0 ? (-(x)) : (x))


#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


static void
GeneratePlaneRotation(magmaDoubleComplex dx, magmaDoubleComplex dy, magmaDoubleComplex *cs, magmaDoubleComplex *sn)
{
#if defined(PRECISION_s) | defined(PRECISION_d)
    if (dy == MAGMA_Z_ZERO) {
        *cs = MAGMA_Z_ONE;
        *sn = MAGMA_Z_ZERO;
    } else if (MAGMA_Z_ABS((dy)) > MAGMA_Z_ABS((dx))) {
        magmaDoubleComplex temp = dx / dy;
        *sn = MAGMA_Z_ONE / magma_zsqrt( ( MAGMA_Z_ONE + temp*temp));
        *cs = temp * (*sn);
    } else {
        magmaDoubleComplex temp = dy / dx;
        *cs = MAGMA_Z_ONE / magma_zsqrt( ( MAGMA_Z_ONE + temp*temp ));
        *sn = temp * (*cs);
    }
#else   
    // below the code Joss Knight from MathWorks provided me with - this works. 
    // No idea why the above code fails for complex - maybe rounding.
    real_Double_t rho = sqrt(MAGMA_Z_REAL(MAGMA_Z_CONJ(dx)*dx + MAGMA_Z_CONJ(dy)*dy));
    *cs = dx / rho;
    *sn = dy / rho;
#endif
}

static void ApplyPlaneRotation(magmaDoubleComplex *dx, magmaDoubleComplex *dy, magmaDoubleComplex cs, magmaDoubleComplex sn)
{
#if defined(PRECISION_s) | defined(PRECISION_d)
      magmaDoubleComplex temp = (*dx);
      *dx =  cs * (*dx) + sn * (*dy);
      *dy = -sn * temp + cs * (*dy);
#else  
    // below the code Joss Knight from MathWorks provided me with - this works. 
    // No idea why the above code fails for complex - maybe rounding.
    magmaDoubleComplex temp  =  MAGMA_Z_CONJ(cs) * (*dx) +  MAGMA_Z_CONJ(sn) * (*dy);
    *dy = -(sn) * (*dx) + cs * (*dy);
    *dx = temp;
#endif
}



/**
    Purpose
    -------

    Solves a system of linear equations
       A * X = B
    where A is a complex sparse matrix stored in the GPU memory.
    X and B are complex vectors stored on the GPU memory.
    This is a GPU implementation of the right-preconditioned flexible GMRES.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                descriptor for matrix A

    @param[in]
    b           magma_z_matrix
                RHS b vector

    @param[in,out]
    x           magma_z_matrix*
                solution approximation

    @param[in,out]
    solver_par  magma_z_solver_par*
                solver parameters

    @param[in]
    precond_par magma_z_preconditioner*
                preconditioner
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgesv
    ********************************************************************/

extern "C" magma_int_t
magma_zfgmres(
    magma_z_matrix A, magma_z_matrix b, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_z_preconditioner *precond_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    magma_int_t dofs = A.num_rows;

    // prepare solver feedback
    solver_par->solver = Magma_PGMRES;
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    
    //Chronometry
    real_Double_t tempo1, tempo2;

    magma_int_t dim = solver_par->restart;
    magma_int_t m1 = dim+1; // used inside H macro
    magma_int_t i, j, k;
    magmaDoubleComplex beta;
    
    double rel_resid, resid0=1, r0=0.0, betanom = 0.0, nom, nomb;
    
    magma_z_matrix v_t={Magma_CSR}, w_t={Magma_CSR}, t={Magma_CSR}, t2={Magma_CSR}, V={Magma_CSR}, W={Magma_CSR};
    v_t.memory_location = Magma_DEV;
    v_t.num_rows = dofs;
    v_t.num_cols = 1;
    v_t.dval = NULL;
    v_t.storage_type = Magma_DENSE;

    w_t.memory_location = Magma_DEV;
    w_t.num_rows = dofs;
    w_t.num_cols = 1;
    w_t.dval = NULL;
    w_t.storage_type = Magma_DENSE;
    
    magmaDoubleComplex temp;
    
    magmaDoubleComplex *H={0}, *s={0}, *cs={0}, *sn={0};

    CHECK( magma_zvinit( &t, Magma_DEV, dofs, 1, MAGMA_Z_ZERO, queue ));
    CHECK( magma_zvinit( &t2, Magma_DEV, dofs, 1, MAGMA_Z_ZERO, queue ));
    
    CHECK( magma_zmalloc_pinned( &H, (dim+1)*dim ));
    CHECK( magma_zmalloc_pinned( &s,  dim+1 ));
    CHECK( magma_zmalloc_pinned( &cs, dim ));
    CHECK( magma_zmalloc_pinned( &sn, dim ));
    
    
    CHECK( magma_zvinit( &V, Magma_DEV, dofs*(dim+1), 1, MAGMA_Z_ZERO, queue ));
    CHECK( magma_zvinit( &W, Magma_DEV, dofs*dim, 1, MAGMA_Z_ZERO, queue ));
    
    CHECK(  magma_zresidual( A, b, *x, &nom, queue));
    nomb = magma_dznrm2( dofs, b.dval, 1, queue );

    solver_par->init_res = nom;
    
    if ( (r0 = nomb * solver_par->rtol) < ATOLERANCE ){
        r0 = ATOLERANCE;
    }
    
    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    

    tempo1 = magma_sync_wtime( queue );
    do
    {
        // compute initial residual and its norm
        // A.mult(n, 1, x, n, V(0), n);                        // V(0) = A*x
        CHECK( magma_z_spmv( MAGMA_Z_ONE, A, *x, MAGMA_Z_ZERO, t, queue ));
        solver_par->numiter++;
        solver_par->spmv_count++;
        magma_zcopy( dofs, t.dval, 1, V(0), 1, queue );
        
        temp = MAGMA_Z_MAKE(-1.0, 0.0);
        magma_zaxpy( dofs,temp, b.dval, 1, V(0), 1, queue );           // V(0) = V(0) - b
        beta = MAGMA_Z_MAKE( magma_dznrm2( dofs, V(0), 1, queue ), 0.0 ); // beta = norm(V(0))
        if( magma_z_isnan_inf( beta ) ){
            info = MAGMA_DIVERGENCE;
            break;
        }
        
        if (solver_par->numiter == 1){
            solver_par->init_res = MAGMA_Z_REAL( beta );
            resid0 = MAGMA_Z_REAL( beta );
        
            if ( (r0 = nomb * solver_par->rtol) < ATOLERANCE ){
                r0 = ATOLERANCE;
            }
            if ( resid0 < r0 ) {
                solver_par->final_res = solver_par->init_res;
                solver_par->iter_res = solver_par->init_res;
                info = MAGMA_SUCCESS;
                goto cleanup;
            }
        }
        tempo2 = magma_sync_wtime( queue );
        if ( solver_par->verbose > 0 ) {
            solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) betanom; 
            solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                        = (real_Double_t) tempo2-tempo1;
        }

        
        temp = -1.0/beta;
        magma_zscal( dofs, temp, V(0), 1, queue );                 // V(0) = -V(0)/beta

        // save very first residual norm
        if (solver_par->numiter == 0)
            solver_par->init_res = MAGMA_Z_REAL( beta );

        for (i = 1; i < dim+1; i++)
            s[i] = MAGMA_Z_ZERO;
        s[0] = beta;

        i = -1;
        do {
            i++;
            
            // M.apply(n, 1, V(i), n, W(i), n);
            v_t.dval = V(i);
            CHECK( magma_z_applyprecond_left( MagmaNoTrans, A, v_t, &t, precond_par, queue ));
            CHECK( magma_z_applyprecond_right( MagmaNoTrans, A, t, &t2, precond_par, queue ));
            magma_zcopy( dofs, t2.dval, 1, W(i), 1, queue );

            // A.mult(n, 1, W(i), n, V(i+1), n);
            w_t.dval = W(i);
            CHECK( magma_z_spmv( MAGMA_Z_ONE, A, w_t, MAGMA_Z_ZERO, t, queue ));
            solver_par->numiter++;
            solver_par->spmv_count++;
            magma_zcopy( dofs, t.dval, 1, V(i+1), 1, queue );
            
            for (k = 0; k <= i; k++)
            {
                H(k, i) = magma_zdotc( dofs, V(k), 1, V(i+1), 1, queue );
                temp = -H(k,i);
                // V(i+1) -= H(k, i) * V(k);
                magma_zaxpy( dofs,-H(k,i), V(k), 1, V(i+1), 1, queue );
            }

            H(i+1, i) = MAGMA_Z_MAKE( magma_dznrm2( dofs, V(i+1), 1, queue), 0. ); // H(i+1,i) = ||r||
            temp = 1.0 / H(i+1, i);
            // V(i+1) = V(i+1) / H(i+1, i)
            magma_zscal( dofs, temp, V(i+1), 1, queue );    //  (to be fused)
    
            for (k = 0; k < i; k++)
                ApplyPlaneRotation(&H(k,i), &H(k+1,i), cs[k], sn[k]);
          
            GeneratePlaneRotation(H(i,i), H(i+1,i), &cs[i], &sn[i]);
            ApplyPlaneRotation(&H(i,i), &H(i+1,i), cs[i], sn[i]);
            ApplyPlaneRotation(&s[i], &s[i+1], cs[i], sn[i]);
            
            betanom = MAGMA_Z_ABS( s[i+1] );
            rel_resid = betanom / nomb;
            if ( solver_par->verbose > 0 ) {
                tempo2 = magma_sync_wtime( queue );
                if ( (solver_par->numiter)%solver_par->verbose==0 ) {
                    solver_par->res_vec[(solver_par->numiter)/solver_par->verbose]
                            = (real_Double_t) betanom;
                    solver_par->timing[(solver_par->numiter)/solver_par->verbose]
                            = (real_Double_t) tempo2-tempo1;
                }
            }
            if (rel_resid <= solver_par->rtol || betanom <= solver_par->atol ){
                info = MAGMA_SUCCESS;
                break;
            }
        }
        while (i+1 < dim && solver_par->numiter+1 <= solver_par->maxiter);

        // solve upper triangular system in place
        for (j = i; j >= 0; j--)
        {
            s[j] /= H(j,j);
            for (k = j-1; k >= 0; k--)
                s[k] -= H(k,j) * s[j];
        }

        // update the solution
        for (j = 0; j <= i; j++)
        {
            // x = x + s[j] * W(j)
            magma_zaxpy( dofs, s[j], W(j), 1, x->dval, 1, queue );
        }
    }
    while (rel_resid > solver_par->rtol
                && solver_par->numiter+1 <= solver_par->maxiter);

    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    CHECK( magma_zresidual( A, b, *x, &residual, queue ));
    solver_par->iter_res = betanom;
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
    // free pinned memory
    magma_free_pinned(s);
    magma_free_pinned(cs);
    magma_free_pinned(sn);
    magma_free_pinned(H);

    //free DEV memory
    magma_zmfree( &V, queue);
    magma_zmfree( &W, queue);
    magma_zmfree( &t, queue);
    magma_zmfree( &t2, queue);

    solver_par->info = info;
    return info;
} /* magma_zfgmres */
