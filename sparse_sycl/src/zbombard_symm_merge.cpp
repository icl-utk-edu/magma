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
    This is a GPU implementation of the iterative bombardment suggested in
    Barrett et al.
    ''Algorithmic bombardment for the iterative solution of linear systems: 
      A poly-iterative approach''
    using BiCGSTAB, CGS and MQR. At this point, it only works for symmetric A.

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
magma_zbombard_merge(
    magma_z_matrix A, magma_z_matrix b, 
    magma_z_matrix *x, magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
        // queue variables
    const magma_queue_t squeue = 0;    // synchronous kernel queues
    const magma_int_t nqueues = 3;     // number of queues
    magma_queue_t queues[nqueues];    
    magma_int_t q1flag = 0;
    
        // set asynchronous kernel queues
    //printf("%% Kernel queues: (orig, queue) = (%p, %p)\n", (void *)orig_queue, (void *)queue);
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &queues[0] );
    if ( queue != squeue ) {
        queues[1] = queue;
        q1flag = 0;
    } else {
        magma_device_t cdev;
        magma_getdevice( &cdev );
        magma_queue_create( cdev, &(queues[1]) );
        queue = queues[1];
        q1flag = 1;
    }
    magma_device_t cdev;
    magma_getdevice( &cdev );
    magma_queue_create( cdev, &(queues[2]) );
    for (int i = 0; i < nqueues; ++i ) {
        ; //printf("Kernel queue #%d = %p\n", i, (void *)queues[i]);
    }

    
    // 1=QMR, 2=CGS, 3+BiCGSTAB
    magma_int_t flag = 0;
    
    int mdot = 1;
    
    // prepare solver feedback
    solver_par->solver = Magma_BOMBARD;
    solver_par->numiter = 0;
    
    // constants
    const magmaDoubleComplex c_zero    = MAGMA_Z_ZERO;
    const magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    
    // solver variables
    double nom0, r0, res, Q_res, C_res, B_res, nomb;
    
    //QMR
    magmaDoubleComplex Q_rho = c_one, Q_rho1 = c_one, Q_eta = -c_one , Q_pds = c_one, 
                        Q_thet = c_one, Q_thet1 = c_one, Q_epsilon = c_one, 
                        Q_beta = c_one, Q_delta = c_one, Q_pde = c_one, Q_rde = c_one,
                        Q_gamm = c_one, Q_gamm1 = c_one, Q_psi = c_one;
    //CGS
    magmaDoubleComplex C_rho, C_rho_l = c_one, C_alpha, C_beta;
    //BiCGSTAB
    magmaDoubleComplex B_alpha, B_beta, B_omega, B_rho_old, B_rho_new;
    
    magma_int_t dofs = A.num_rows* b.num_cols;

    // need to transpose the matrix
    
    // GPU workspace
    // overall initial residual
    magma_z_matrix r_tld={Magma_CSR};
    
    // QMR
    magma_z_matrix AT = {Magma_CSR}, Q_r={Magma_CSR},  Q_x={Magma_CSR},
                    Q_v={Magma_CSR}, Q_w={Magma_CSR}, Q_wt={Magma_CSR},
                    Q_d={Magma_CSR}, Q_s={Magma_CSR}, Q_z={Magma_CSR}, Q_q={Magma_CSR}, 
                    Q_p={Magma_CSR}, Q_pt={Magma_CSR}, Q_y={Magma_CSR};
    // CGS
    magma_z_matrix C_r={Magma_CSR}, C_rt={Magma_CSR}, C_x={Magma_CSR},
                    C_p={Magma_CSR}, C_q={Magma_CSR}, C_u={Magma_CSR}, C_v={Magma_CSR},  C_t={Magma_CSR},
                    C_p_hat={Magma_CSR}, C_q_hat={Magma_CSR}, C_u_hat={Magma_CSR}, C_v_hat={Magma_CSR};
    //BiCGSTAB
    magma_z_matrix B_r={Magma_CSR}, B_x={Magma_CSR}, B_p={Magma_CSR}, B_v={Magma_CSR}, 
                    B_s={Magma_CSR}, B_t={Magma_CSR};
                    
    // multi-vector for block-SpMV 
    magma_z_matrix SpMV_in_1={Magma_CSR}, SpMV_out_1={Magma_CSR}, 
                    SpMV_in_2={Magma_CSR}, SpMV_out_2={Magma_CSR}; 
                    
    // workspace for zmzdotc
    magma_z_matrix d1={Magma_CSR},  d2={Magma_CSR}, skp={Magma_CSR};

                    
    CHECK( magma_zvinit( &r_tld, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    
    CHECK( magma_zvinit( &SpMV_in_1,  Magma_DEV, A.num_rows, b.num_cols * 3, c_zero, queue ));
    CHECK( magma_zvinit( &SpMV_out_1, Magma_DEV, A.num_rows, b.num_cols * 3, c_zero, queue ));
    CHECK( magma_zvinit( &SpMV_in_2,  Magma_DEV, A.num_rows, b.num_cols * 3, c_zero, queue ));
    CHECK( magma_zvinit( &SpMV_out_2, Magma_DEV, A.num_rows, b.num_cols * 3, c_zero, queue ));
    
    CHECK( magma_zvinit( &d1, Magma_DEV, A.num_rows, b.num_cols * 4, c_zero, queue ));
    CHECK( magma_zvinit( &d2, Magma_DEV, A.num_rows, b.num_cols * 4, c_zero, queue ));
    CHECK( magma_zvinit( &skp, Magma_CPU, 4, 1, c_zero, queue ));
    
    // QMR
    CHECK( magma_zvinit( &Q_r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_w, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_wt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_d, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_s, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_z, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_q, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_pt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_y, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &Q_x, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    // QMR
    CHECK( magma_zvinit( &C_r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_rt,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_x,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_p_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_q, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_q_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_u, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_u_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_v_hat, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &C_t, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    // BiCGSTAB
    CHECK( magma_zvinit( &B_r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &B_x,Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &B_p, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &B_v, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &B_s, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK( magma_zvinit( &B_t, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));

    // solver setup
    CHECK(  magma_zresidualvec( A, b, *x, &r_tld, &nom0, queue));
    solver_par->init_res = nom0;
    res = nom0;
    
    // QMR
    magma_zcopy( dofs, r_tld.dval, 1, Q_r.dval, 1, queue );   
    magma_zcopy( dofs, r_tld.dval, 1, Q_y.dval, 1, queue );   
    magma_zcopy( dofs, r_tld.dval, 1, Q_v.dval, 1, queue );  
    magma_zcopy( dofs, r_tld.dval, 1, Q_wt.dval, 1, queue );   
    magma_zcopy( dofs, r_tld.dval, 1, Q_z.dval, 1, queue ); 
    magma_zcopy( dofs, x->dval, 1, Q_x.dval, 1, queue ); 
    // transpose the matrix
    magma_zmtransposeconjugate( A, &AT, queue );
    
    // CGS
    magma_zcopy( dofs, r_tld.dval, 1, C_r.dval, 1, queue );   
    magma_zcopy( dofs, x->dval, 1, C_x.dval, 1, queue ); 
    
    // BiCGSTAB
    magma_zcopy( dofs, r_tld.dval, 1, B_r.dval, 1, queue );   
    magma_zcopy( dofs, x->dval, 1, B_x.dval, 1, queue ); 
    CHECK( magma_z_spmv( c_one, A, B_r, c_zero, B_v, queue ));     
    magma_zcopy( dofs, B_v.dval, 1, SpMV_out_1.dval+2*dofs, 1, queue );
    
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

    Q_psi = magma_zsqrt( magma_zdotc( dofs, Q_z.dval, 1, Q_z.dval, 1), queue );
    Q_rho = magma_zsqrt( magma_zdotc( dofs, Q_y.dval, 1, Q_y.dval, 1), queue );
    
    // BiCGSTAB
    B_rho_new = magma_zdotc( dofs, B_r.dval, 1, B_r.dval, 1, queue );            
    B_rho_old = B_omega = B_alpha = MAGMA_Z_MAKE( 1.0, 0. );
    
        // v = y / rho
        // y = y / rho
        // w = wt / psi
        // z = z / psi
    magma_zqmr_1(  
    b.num_rows, 
    b.num_cols, 
    Q_rho,
    Q_psi,
    Q_y.dval, 
    Q_z.dval,
    Q_v.dval,
    Q_w.dval,
    queue );
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    tempo1 = magma_sync_wtime( queue );
    
    solver_par->numiter = 0;
    // start iteration
    do
    {
        solver_par->numiter++;
        
        if(mdot == 0){   
                //QMR: delta = z' * y;
            Q_delta = magma_zdotc( dofs, Q_z.dval, 1, Q_y.dval, 1, queue );
            
                //CGS: rho = r' * r_tld
            C_rho = magma_zdotc( dofs, C_r.dval, 1, r_tld.dval, 1, queue );
            
                // BiCGSTAB
            B_rho_old = B_rho_new;    
            B_rho_new = magma_zdotc( dofs, r_tld.dval, 1, B_r.dval, 1, queue );  // rho=<rr,r>
            B_beta = B_rho_new/B_rho_old * B_alpha/B_omega;   // beta=rho/rho_old *alpha/omega
        }
        else {
            B_rho_old = B_rho_new; 
            magma_zmdotc4(
            b.num_rows,  
            Q_z.dval, 
            Q_y.dval,
            r_tld.dval, 
            C_r.dval,
            r_tld.dval, 
            B_r.dval,
            r_tld.dval, 
            B_r.dval,
            d1.dval,
            d2.dval,
            skp.val,
            queue );
            
            Q_delta = skp.val[0];
            C_rho = skp.val[1];
            B_rho_new = skp.val[2];
            B_beta = B_rho_new/B_rho_old * B_alpha/B_omega;
        }
        
        if( solver_par->numiter == 1 ){
                //QMR: p = y;
                //QMR: q = z;
            magma_zcopy( dofs, Q_y.dval, 1, SpMV_in_1.dval, 1, queue );
            magma_zcopy( dofs, Q_z.dval, 1, SpMV_in_2.dval, 1, queue );
            
                //QMR: u = r;
                //QMR: p = r;
            magma_zcgs_2(  
            b.num_rows, 
            b.num_cols, 
            C_r.dval,
            C_u.dval,
            SpMV_in_1.dval+dofs,
            queue );
        }
        else{
            Q_pde = Q_psi * Q_delta / Q_epsilon;
            Q_rde = Q_rho * MAGMA_Z_CNJG(Q_delta/Q_epsilon);
            
            C_beta = C_rho / C_rho_l;  
            
                //QMR p = y - pde * p
                //QMR q = z - rde * q
            magma_zqmr_2(  
            b.num_rows, 
            b.num_cols, 
            Q_pde,
            Q_rde,
            Q_y.dval,
            Q_z.dval,
            SpMV_in_1.dval, 
            SpMV_in_2.dval, 
            queue );
            
                  //CGS: u = r + beta*q;
                  //CGS: p = u + beta*( q + beta*p );
            magma_zcgs_1(  
            b.num_rows, 
            b.num_cols, 
            C_beta,
            C_r.dval,
            C_q.dval, 
            C_u.dval,
            SpMV_in_1.dval+dofs,
            queue );
        }
            // BiCGSTAB: p = r + beta * ( p - omega * v )
        magma_zbicgstab_1(  
        b.num_rows, 
        b.num_cols, 
        B_beta,
        B_omega,
        B_r.dval, 
        SpMV_out_1.dval+2*dofs,
        SpMV_in_1.dval+2*dofs,
        queue );
        /*
        //QMR
        CHECK( magma_z_spmv( c_one, A, Q_p, c_zero, Q_pt, queue ));
        //CGS
        CHECK( magma_z_spmv( c_one, A, C_p, c_zero, C_v_hat, queue ));
        // BiCGSTAB
        CHECK( magma_z_spmv( c_one, A, B_p, c_zero, B_v, queue )); 
        */
        
        // gather everything for a block-SpMV
        //magma_zcopy( dofs, Q_p.dval, 1, SpMV_in_1.dval          , 1, queue );
        //magma_zcopy( dofs, C_p.dval, 1, SpMV_in_1.dval+dofs     , 1, queue );
        //magma_zcopy( dofs, B_p.dval, 1, SpMV_in_1.dval+2*dofs   , 1, queue );
        
        // block SpMV
        CHECK( magma_z_spmv( c_one, A, SpMV_in_1, c_zero, SpMV_out_1, queue ));
        
        // scatter results
        //magma_zcopy( dofs, SpMV_out_1.dval          , 1, Q_pt.dval, 1, queue );
        //magma_zcopy( dofs, SpMV_out_1.dval+dofs     , 1, C_v_hat.dval, 1, queue );
        //magma_zcopy( dofs, SpMV_out_1.dval+2*dofs   , 1, B_v.dval, 1, queue );

        if( mdot == 0 ) {      
                //QMR: epsilon = q' * pt;
            Q_epsilon = magma_zdotc( dofs, SpMV_in_2.dval, 1, SpMV_out_1.dval, 1, queue );
            Q_beta = Q_epsilon / Q_delta;
                //CGS: alpha = r_tld' * v_hat
            C_alpha = C_rho / magma_zdotc( dofs, r_tld.dval, 1, SpMV_out_1.dval+dofs, 1, queue );
            C_rho_l = C_rho;
                //BiCGSTAB
            B_alpha = B_rho_new / magma_zdotc( dofs, r_tld.dval, 1, SpMV_out_1.dval+2*dofs, 1, queue );
        }else{
            magma_zmdotc4(
            b.num_rows,  
            SpMV_in_2.dval, 
            SpMV_out_1.dval,
            r_tld.dval, 
            SpMV_out_1.dval+dofs,
            r_tld.dval, 
            SpMV_out_1.dval+2*dofs,
            r_tld.dval, 
            SpMV_out_1.dval+2*dofs,
            d1.dval,
            d2.dval,
            skp.val,
            queue );
            
            Q_epsilon = skp.val[0];
            Q_beta = Q_epsilon / Q_delta;
            C_alpha = C_rho / skp.val[1];
            C_rho_l = C_rho;
            B_alpha = B_rho_new / skp.val[2];
        }
        
            //QMR: v = pt - beta * v
            //QMR: y = v
        magma_zqmr_3(  
        b.num_rows, 
        b.num_cols, 
        Q_beta,
        SpMV_out_1.dval,
        Q_v.dval,
        Q_y.dval,
        queue );
        
            //CGS: q = u - alpha v_hat
            //CGS: t = u + q
        magma_zcgs_3(  
        b.num_rows, 
        b.num_cols, 
        C_alpha,
        SpMV_out_1.dval+dofs,
        C_u.dval, 
        C_q.dval,
        SpMV_in_2.dval+dofs, 
        queue );
        
            // BiCGSTAB: s = r - alpha v
        magma_zbicgstab_2(  
        b.num_rows, 
        b.num_cols, 
        B_alpha,
        B_r.dval,
        SpMV_out_1.dval+2*dofs,
        SpMV_in_2.dval+2*dofs, 
        queue );
            


            // gather everything for a block-SpMV
        //magma_zcopy( dofs, Q_q.dval, 1, SpMV_in_2.dval          , 1, queue );
        //magma_zcopy( dofs, C_t.dval, 1, SpMV_in_2.dval+dofs     , 1, queue );
        //magma_zcopy( dofs, B_s.dval, 1, SpMV_in_2.dval+2*dofs   , 1, queue );

            // block SpMV
        CHECK( magma_z_spmv( c_one, A, SpMV_in_2, c_zero, SpMV_out_2, queue ));

            // scatter results
        //magma_zcopy( dofs, SpMV_out_2.dval          , 1, Q_wt.dval, 1, queue );
        //magma_zcopy( dofs, SpMV_out_2.dval+dofs     , 1, C_rt.dval, 1, queue );
        //magma_zcopy( dofs, SpMV_out_2.dval+2*dofs   , 1, B_t.dval, 1, queue );        
                
        Q_rho1 = Q_rho;     
        if( mdot == 0 ) {
                //QMR rho = norm(y);
            Q_rho = magma_zsqrt( magma_zdotc( dofs, Q_y.dval, 1, Q_y.dval, 1), queue );
            

            // BiCGSTAB
            B_omega = magma_zdotc( dofs, SpMV_out_2.dval+2*dofs, 1, SpMV_in_2.dval+2*dofs, 1 )   // omega = <s,t>/<t,t>
                       / magma_zdotc( dofs, SpMV_out_2.dval+2*dofs, 1, SpMV_out_2.dval+2*dofs, 1, queue );
        }else{
            magma_zmdotc4(
            b.num_rows,  
            Q_y.dval, 
            Q_y.dval,
            Q_y.dval, 
            Q_y.dval,
            SpMV_out_2.dval+2*dofs, 
            SpMV_in_2.dval+2*dofs,
            SpMV_out_2.dval+2*dofs, 
            SpMV_out_2.dval+2*dofs,
            d1.dval,
            d2.dval,
            skp.val,
            queue );
            
            Q_rho = magma_zsqrt(skp.val[0]);
            B_omega = skp.val[2]/skp.val[3];
        }               
                   
        
        // QMR
        Q_thet1 = Q_thet;        
        Q_thet = Q_rho / (Q_gamm * MAGMA_Z_MAKE( MAGMA_Z_ABS(Q_beta), 0.0 ));
        Q_gamm1 = Q_gamm;        
        
        Q_gamm = c_one / magma_zsqrt(c_one + Q_thet*Q_thet);        
        Q_eta = - Q_eta * Q_rho1 * Q_gamm * Q_gamm / (Q_beta * Q_gamm1 * Q_gamm1);        
        
        if ( solver_par->numiter == 1 ) {
                //QMR: d = eta * p + pds * d;
                //QMR: s = eta * pt + pds * d;
                //QMR: x = x + d;
                //QMR: r = r - s;
            magma_zqmr_4(  
            b.num_rows, 
            b.num_cols, 
            Q_eta,
            SpMV_in_1.dval,
            SpMV_out_1.dval,
            Q_d.dval, 
            Q_s.dval, 
            Q_x.dval, 
            Q_r.dval, 
            queue );
        }
        else {
            Q_pds = (Q_thet1 * Q_gamm) * (Q_thet1 * Q_gamm);
            
                //QMR: d = eta * p + pds * d;
                //QMR: s = eta * pt + pds * d;
                //QMR: x = x + d;
                //QMR: r = r - s;
            magma_zqmr_5(  
            b.num_rows, 
            b.num_cols, 
            Q_eta,
            Q_pds,
            SpMV_in_1.dval,
            SpMV_out_1.dval,
            Q_d.dval, 
            Q_s.dval, 
            Q_x.dval, 
            Q_r.dval, 
            queue );
        }
        
        
        // CGS: r = r -alpha*A u_hat
        // CGS: x = x + alpha u_hat
        magma_zcgs_4(  
        b.num_rows, 
        b.num_cols, 
        C_alpha,
        SpMV_in_2.dval+dofs,
        SpMV_out_2.dval+dofs,
        C_x.dval, 
        C_r.dval,
        queue );
        
            // BiCGSTAB: x = x + alpha * p + omega * s
            // BiCGSTAB: r = s - omega * t
        magma_zbicgstab_3(  
        b.num_rows, 
        b.num_cols, 
        B_alpha,
        B_omega,
        SpMV_in_1.dval+2*dofs,
        SpMV_in_2.dval+2*dofs,
        SpMV_out_2.dval+2*dofs,
        B_x.dval,
        B_r.dval,
        queue );
        
        if( mdot == 0 ){
            Q_res = magma_dznrm2( dofs, Q_r.dval, 1, queue );
            C_res = magma_dznrm2( dofs, C_r.dval, 1, queue );
            B_res = magma_dznrm2( dofs, B_r.dval, 1, queue );
                //QMR: psi = norm(z);
            Q_psi = magma_zsqrt( magma_zdotc( dofs, Q_z.dval, 1, Q_z.dval, 1), queue );
        }
        else {
            magma_zmdotc4(
            b.num_rows,  
            Q_r.dval, 
            Q_r.dval,
            C_r.dval, 
            C_r.dval,
            B_r.dval, 
            B_r.dval,
            Q_z.dval, 
            Q_z.dval,
            d1.dval,
            d2.dval,
            skp.val,
            queue );
        
            Q_res = MAGMA_Z_ABS(magma_zsqrt(skp.val[0]));
            C_res = MAGMA_Z_ABS(magma_zsqrt(skp.val[1]));
            B_res = MAGMA_Z_ABS(magma_zsqrt(skp.val[2]));
                //QMR: psi = norm(z);
            Q_psi = magma_zsqrt(skp.val[3]);
        }
        
        
            //QMR: v = y / rho
            //QMR: y = y / rho
            //QMR: w = wt / psi
            //QMR: z = z / psi    
            //QMR: wt = A' * q - beta' * w
            //QMR: no precond: z = wt
        magma_zqmr_6(  
        b.num_rows, 
        b.num_cols, 
        Q_beta,
        Q_rho,
        Q_psi,
        Q_y.dval, 
        Q_z.dval,
        Q_v.dval,
        Q_w.dval,
        SpMV_out_2.dval,
        queue );
        
          // printf(" %e   %e   %e\n", Q_res, C_res, B_res);
        if( Q_res < res ){
            res = Q_res;
            flag = 1;
        }
        if( C_res < res ){
            res = C_res;
            flag = 2;
        }
        if( B_res < res ){
            res = B_res;
            flag = 3;
        }

        if ( solver_par->verbose > 0 ) {
            tempo2 = magma_sync_wtime( queue );
            if ( (solver_par->numiter)%solver_par->verbose == c_zero ) {
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
        
    // copy back the best solver
    switch ( flag ) {
        case 1:
            printf("%% QMR fastest solver.\n");
            magma_zcopy( dofs, Q_x.dval, 1, x->dval, 1, queue ); 
            break;
       case 2:
            printf("%% CGS fastest solver.\n");
            magma_zcopy( dofs, C_x.dval, 1, x->dval, 1, queue ); 
            break;
       case 3:
            printf("%% BiCGSTAB fastest solver.\n");
            magma_zcopy( dofs, B_x.dval, 1, x->dval, 1, queue ); 
            break;
    }


    
    tempo2 = magma_sync_wtime( queue );
    solver_par->runtime = (real_Double_t) tempo2-tempo1;
    double residual;
    CHECK(  magma_zresidualvec( A, b, *x, &r_tld, &residual, queue));
    solver_par->iter_res = res;
    solver_par->final_res = residual;

    if ( solver_par->numiter < solver_par->maxiter ) {
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
    
    magma_zmfree(&r_tld, queue );
    
    magma_zmfree(&SpMV_in_1,  queue );
    magma_zmfree(&SpMV_out_1, queue );
    magma_zmfree(&SpMV_in_2,  queue );
    magma_zmfree(&SpMV_out_2, queue );
    
    magma_zmfree(&d1, queue );
    magma_zmfree(&d2,  queue );
    magma_zmfree(&skp, queue );
    
    magma_zmfree(&AT,  queue );
    
    // QMR
    magma_zmfree(&Q_r, queue );
    magma_zmfree(&Q_v,  queue );
    magma_zmfree(&Q_w,  queue );
    magma_zmfree(&Q_wt, queue );
    magma_zmfree(&Q_d,  queue );
    magma_zmfree(&Q_s,  queue );
    magma_zmfree(&Q_z,  queue );
    magma_zmfree(&Q_q,  queue );
    magma_zmfree(&Q_p,  queue );
    magma_zmfree(&Q_pt, queue );
    magma_zmfree(&Q_y,  queue );
    magma_zmfree(&Q_x,  queue );
    // CGS
    magma_zmfree(&C_r, queue );
    magma_zmfree(&C_rt, queue );
    magma_zmfree(&C_x, queue );
    magma_zmfree(&C_p, queue );
    magma_zmfree(&C_q, queue );
    magma_zmfree(&C_u, queue );
    magma_zmfree(&C_v, queue );
    magma_zmfree(&C_t, queue );
    magma_zmfree(&C_p_hat, queue );
    magma_zmfree(&C_q_hat, queue );
    magma_zmfree(&C_u_hat, queue );
    magma_zmfree(&C_v_hat, queue );
    // BiCGSTAB
    magma_zmfree(&B_r, queue );
    magma_zmfree(&B_x, queue );
    magma_zmfree(&B_p, queue );
    magma_zmfree(&B_v, queue );
    magma_zmfree(&B_s, queue );
    magma_zmfree(&B_t, queue );
    
    
    
    // destroy asynchronous queues
    magma_queue_destroy( queues[0] );
    if ( q1flag == 1 ) {
        magma_queue_destroy( queues[1] );
    }
    magma_queue_destroy( queues[2] );

    solver_par->info = info;
    return info;
}   /* magma_zbombard_merge */
