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
    This is a GPU implementation of the Jacobi method.

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
magma_zjacobi(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    
    // some useful variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    
    real_Double_t tempo1, tempo2, runtime=0;
    double residual;
    
    //double nom0 = 0.0;

    magma_z_matrix r={Magma_CSR}, d={Magma_CSR}, ACSR={Magma_CSR};

    CHECK( magma_zmconvert(A, &ACSR, A.storage_type, Magma_CSR, queue ) );
    
    // prepare solver feedback
    solver_par->solver = Magma_JACOBI;
    solver_par->info = MAGMA_SUCCESS;

    // Jacobi setup
    CHECK( magma_zjacobisetup_diagscal( ACSR, &d, queue ));
    
    // solver setup
    
    CHECK( magma_zvinit( &r, Magma_DEV, A.num_rows, b.num_cols, c_zero, queue ));
    CHECK(  magma_zresidualvec( ACSR, b, *x, &r, &residual, queue));
    solver_par->init_res = residual;
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t) residual;
    }
    //nom0 = residual;
    // set the initial guess to D^{-1}b
    CHECK( magma_zmfree( x, queue ) );
    CHECK( magma_zmtransfer(d, x, Magma_DEV, Magma_DEV, queue ) );
    CHECK(  magma_zresidualvec( ACSR, b, *x, &r, &residual, queue));
    if ( solver_par->verbose > 0 ) {
        solver_par->res_vec[0] = (real_Double_t) residual;
    }

    magma_z_solver_par jacobiiter_par;
    if ( solver_par->verbose > 0 ) {
        jacobiiter_par.maxiter = solver_par->verbose;
    }
    else {
        jacobiiter_par.maxiter = solver_par->maxiter;
    }

    solver_par->numiter = 0;
    solver_par->spmv_count = 0;
    // Jacobi iterator
    do
    {
        tempo1 = magma_sync_wtime( queue );
        solver_par->numiter = solver_par->numiter+jacobiiter_par.maxiter;
        //CHECK( magma_zjacobiiter_sys( A, b, d, r, x, &jacobiiter_par, queue ) );
        CHECK( magma_zjacobispmvupdate(jacobiiter_par.maxiter, ACSR, r, b, d, x, queue ));
        solver_par->spmv_count = solver_par->spmv_count+jacobiiter_par.maxiter;
        tempo2 = magma_sync_wtime( queue );
        runtime += tempo2 - tempo1;
        //CHECK( magma_zjacobispmvupdate_bw(jacobiiter_par.maxiter, A, r, b, d, x, queue ));
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

    solver_par->info = info;
    return info;
}   /* magma_zjacobi */


/**
    Purpose
    -------

    Prepares the Matrix M for the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    It returns the preconditioner Matrix M and a vector d
    containing the diagonal elements.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in]
    M           magma_z_matrix*
                M = D^(-1) * (L+U)

    @param[in,out]
    d           magma_z_matrix*
                vector with diagonal elements of A
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobisetup_matrix(
    magma_z_matrix A,
    magma_z_matrix *M, magma_z_matrix *d,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t i;

    magma_z_matrix A_h1={Magma_CSR}, A_h2={Magma_CSR}, B={Magma_CSR}, C={Magma_CSR};
    magma_z_matrix diag={Magma_CSR};
    CHECK( magma_zvinit( &diag, Magma_CPU, A.num_rows, 1, MAGMA_Z_ZERO, queue ));

    if ( A.storage_type != Magma_CSR) {
        CHECK( magma_zmtransfer( A, &A_h1, A.memory_location, Magma_CPU, queue ));
        CHECK( magma_zmconvert( A_h1, &B, A_h1.storage_type, Magma_CSR, queue ));
    }
    else {
        CHECK( magma_zmtransfer( A, &B, A.memory_location, Magma_CPU, queue ));
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ) {
        magma_int_t start = (B.drow[rowindex]);
        magma_int_t end = (B.drow[rowindex+1]);
        for( i=start; i<end; i++ ) {
            if ( B.dcol[i]==rowindex ) {
                diag.val[rowindex] = B.val[i];
                if ( MAGMA_Z_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n",
                                                                int(rowindex));
            }
        }
        for( i=start; i<end; i++ ) {
            B.val[i] = B.val[i] / diag.val[rowindex];
            if ( B.dcol[i]==rowindex ) {
                B.val[i] = MAGMA_Z_MAKE( 0., 0. );
            }
        }
    }
    CHECK( magma_z_csr_compressor(&B.val, &B.drow, &B.dcol,
                           &C.val, &C.drow, &C.dcol, &B.num_rows, queue ));
    C.num_rows = B.num_rows;
    C.num_cols = B.num_cols;
    C.memory_location = B.memory_location;
    C.nnz = C.drow[B.num_rows];
    C.storage_type = B.storage_type;
    C.memory_location = B.memory_location;
    if ( A.storage_type != Magma_CSR) {
        CHECK( magma_zmconvert( C, &A_h2, Magma_CSR, A_h1.storage_type, queue ));
        CHECK( magma_zmtransfer( A_h2, M, Magma_CPU, A.memory_location, queue ));
    }
    else {
        CHECK( magma_zmtransfer( C, M, Magma_CPU, A.memory_location, queue ));
    }
    CHECK( magma_zmtransfer( diag, d, Magma_CPU, A.memory_location, queue ));

    if ( A.storage_type != Magma_CSR) {
        magma_zmfree( &A_h1, queue );
        magma_zmfree( &A_h2, queue );
    }
    
cleanup:
    magma_zmfree( &B, queue );
    magma_zmfree( &C, queue );

    magma_zmfree( &diag, queue );
 
    return info;
}


/**
    Purpose
    -------


    It returns a vector d
    containing the inverse diagonal elements.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in,out]
    d           magma_z_matrix*
                vector with diagonal elements
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobisetup_diagscal(
    magma_z_matrix A, magma_z_matrix *d,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t i;

    magma_z_matrix A_h1={Magma_CSR}, B={Magma_CSR};
    magma_z_matrix diag={Magma_CSR};
    CHECK( magma_zvinit( &diag, Magma_CPU, A.num_rows, 1, MAGMA_Z_ZERO, queue ));

    if ( A.storage_type != Magma_CSR || A.memory_location != Magma_CPU ) {
        CHECK( magma_zmtransfer( A, &A_h1, A.memory_location, Magma_CPU, queue ));
        CHECK( magma_zmconvert( A_h1, &B, A_h1.storage_type, Magma_CSR, queue ));
        
        for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ) {
            magma_int_t start = (B.drow[rowindex]);
            magma_int_t end = (B.drow[rowindex+1]);
            for( i=start; i<end; i++ ) {
                if ( B.dcol[i]==rowindex ) {
                    diag.val[rowindex] = 1.0/B.val[i];
                    break;
                }
            }
            if ( diag.val[rowindex] == MAGMA_Z_ZERO ){
                printf(" error: zero diagonal element in row %d!\n",
                                                            int(rowindex));
                
                if ( A.storage_type != Magma_CSR) {
                    magma_zmfree( &A_h1, queue );
                }
                magma_zmfree( &B, queue );
                magma_zmfree( &diag, queue );
                info = MAGMA_ERR_BADPRECOND;
                goto cleanup;
            }
        }
    }
    else{
        for( magma_int_t rowindex=0; rowindex<A.num_rows; rowindex++ ) {
            magma_int_t start = (A.drow[rowindex]);
            magma_int_t end = (A.drow[rowindex+1]);
            for( i=start; i<end; i++ ) {
                if ( A.dcol[i]==rowindex ) {
                    diag.val[rowindex] = 1.0/A.val[i];
                    break;
                }
            }
            if ( diag.val[rowindex] == MAGMA_Z_ZERO ){
                printf(" error: zero diagonal element in row %d!\n",
                                                            int(rowindex));
                
                if ( A.storage_type != Magma_CSR) {
                    magma_zmfree( &A_h1, queue );
                }
                magma_zmfree( &B, queue );
                magma_zmfree( &diag, queue );
                info = MAGMA_ERR_BADPRECOND;
                goto cleanup;
            }
        }
    }
    CHECK( magma_zmtransfer( diag, d, Magma_CPU, Magma_DEV, queue ));
    
cleanup:
    magma_zmfree( &A_h1, queue );
    magma_zmfree( &B, queue );
    magma_zmfree( &diag, queue );
 
    return info;
}



/**
    Purpose
    -------

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Returns the vector c

    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                RHS b

    @param[in]
    d           magma_z_matrix
                vector with diagonal entries

    @param[in]
    c           magma_z_matrix*
                c = D^(-1) * b
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobisetup_vector(
    magma_z_matrix b, magma_z_matrix d,
    magma_z_matrix *c,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix diag={Magma_CSR}, c_t={Magma_CSR}, b_h={Magma_CSR}, tmp={Magma_CSR};
    
    if ( b.memory_location == Magma_CPU ) {
        CHECK( magma_zvinit( &c_t, Magma_CPU, b.num_rows, b.num_cols, MAGMA_Z_ZERO, queue ));

        CHECK( magma_zmtransfer( b, &b_h, b.memory_location, Magma_CPU, queue ));
        CHECK( magma_zmtransfer( d, &diag, b.memory_location, Magma_CPU, queue ));

        for( magma_int_t rowindex=0; rowindex<b.num_rows; rowindex++ ) {
            c_t.val[rowindex] = b_h.val[rowindex] / diag.val[rowindex];
        }
        CHECK( magma_zmtransfer( c_t, c, Magma_CPU, b.memory_location, queue ));
    }
    else if ( b.memory_location == Magma_DEV ) {
        // fill vector
        CHECK( magma_zvinit( &tmp, Magma_DEV, b.num_rows, b.num_cols, MAGMA_Z_ZERO, queue ));
        CHECK( magma_zjacobisetup_vector_gpu(
                    b.num_rows, b, d, *c, &tmp, queue ));
        goto cleanup;
    }

cleanup:
    magma_zmfree( &tmp, queue );
    magma_zmfree( &diag, queue );
    magma_zmfree( &c_t, queue );
    magma_zmfree( &b_h, queue );
    
    return info;
}


/**
    Purpose
    -------

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in]
    b           magma_z_matrix
                RHS b

    @param[in]
    M           magma_z_matrix*
                M = D^(-1) * (L+U)

    @param[in]
    c           magma_z_matrix*
                c = D^(-1) * b
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobisetup(
    magma_z_matrix A, magma_z_matrix b,
    magma_z_matrix *M, magma_z_matrix *c,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t i;

    magma_z_matrix A_h1={Magma_CSR}, A_h2={Magma_CSR}, B={Magma_CSR}, C={Magma_CSR};
    magma_z_matrix diag={Magma_CSR}, c_t={Magma_CSR}, b_h={Magma_CSR};
    CHECK( magma_zvinit( &c_t, Magma_CPU, A.num_rows, b.num_cols, MAGMA_Z_ZERO, queue ));
    CHECK( magma_zvinit( &diag, Magma_CPU, A.num_rows, b.num_cols, MAGMA_Z_ZERO, queue ));
    CHECK( magma_zmtransfer( b, &b_h, A.memory_location, Magma_CPU, queue ));

    if ( A.storage_type != Magma_CSR ) {
        CHECK( magma_zmtransfer( A, &A_h1, A.memory_location, Magma_CPU, queue ));
        CHECK( magma_zmconvert( A_h1, &B, A_h1.storage_type, Magma_CSR, queue ));
    }
    else {
        CHECK( magma_zmtransfer( A, &B, A.memory_location, Magma_CPU, queue ));
    }
    for( magma_int_t rowindex=0; rowindex<B.num_rows; rowindex++ ) {
        magma_int_t start = (B.drow[rowindex]);
        magma_int_t end = (B.drow[rowindex+1]);
        for( i=start; i<end; i++ ) {
            if ( B.dcol[i]==rowindex ) {
                diag.val[rowindex] = B.val[i];
                if ( MAGMA_Z_REAL( diag.val[rowindex]) == 0 )
                    printf(" error: zero diagonal element in row %d!\n",
                                                               int(rowindex));
            }
        }
        for( i=start; i<end; i++ ) {
            B.val[i] = B.val[i] / diag.val[rowindex];
            if ( B.dcol[i]==rowindex ) {
                B.val[i] = MAGMA_Z_MAKE( 0., 0. );
            }
        }
        c_t.val[rowindex] = b_h.val[rowindex] / diag.val[rowindex];
    }

    CHECK( magma_z_csr_compressor(&B.val, &B.drow, &B.dcol,
                           &C.val, &C.drow, &C.dcol, &B.num_rows, queue ));

    C.num_rows = B.num_rows;
    C.num_cols = B.num_cols;
    C.memory_location = B.memory_location;
    C.nnz = C.drow[B.num_rows];
    C.storage_type = B.storage_type;
    C.memory_location = B.memory_location;
    if ( A.storage_type != Magma_CSR) {
        A_h2.alignment = A.alignment;
        A_h2.blocksize = A.blocksize;
        CHECK( magma_zmconvert( C, &A_h2, Magma_CSR, A_h1.storage_type, queue ));
        CHECK( magma_zmtransfer( A_h2, M, Magma_CPU, A.memory_location, queue ));
    }
    else {
        CHECK( magma_zmtransfer( C, M, Magma_CPU, A.memory_location, queue ));
    }
    CHECK( magma_zmtransfer( c_t, c, Magma_CPU, A.memory_location, queue ));

    if ( A.storage_type != Magma_CSR) {
        magma_zmfree( &A_h1, queue );
        magma_zmfree( &A_h2, queue );
    }
    
cleanup:
    magma_zmfree( &B, queue );
    magma_zmfree( &C, queue );
    magma_zmfree( &diag, queue );
    magma_zmfree( &c_t, queue );
    magma_zmfree( &b_h, queue );

    return info;
}



/**
    Purpose
    -------

    Iterates the solution approximation according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    This routine takes the iteration matrix M as input.

    Arguments
    ---------

    @param[in]
    M           magma_z_matrix
                input matrix M = D^(-1) * (L+U)

    @param[in]
    c           magma_z_matrix
                c = D^(-1) * b

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
magma_zjacobiiter(
    magma_z_matrix M, magma_z_matrix c, magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    
    magma_int_t dofs = M.num_rows*x->num_cols;
    magma_z_matrix t={Magma_CSR}, swap={Magma_CSR};
    CHECK( magma_zvinit( &t, Magma_DEV, M.num_rows, x->num_cols, c_zero, queue ));


    for( magma_int_t i=0; i<solver_par->maxiter; i++ ) {
        CHECK( magma_z_spmv( c_neg_one, M, *x, c_zero, t, queue ));        // t = - M * x
        magma_zaxpy( dofs, c_one , c.dval, 1 , t.dval, 1, queue );        // t = t + c

        // swap so that x again contains solution, and y is ready to be used
        swap = *x;
        *x = t;
        t = swap;
    }

cleanup:
    magma_zmfree( &t, queue );

    solver_par->info = info;
    return info;
}   /* magma_zjacobiiter */



/**
    Purpose
    -------

    Iterates the solution approximation according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Arguments
    ---------

    @param[in]
    M           magma_z_matrix
                input matrix M = D^(-1) * (L+U)

    @param[in,out]
    x           magma_z_matrix*
                iteration vector x

    @param[in,out]
    solver_par  magma_z_solver_par*
                solver parameters

    @param[in]
    precond     magma_z_precond_par*
                precond parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobiiter_precond(
    magma_z_matrix M, 
    magma_z_matrix *x,
    magma_z_solver_par *solver_par, 
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    
    magma_int_t dofs = M.num_rows;
    magma_int_t num_vecs = x->num_rows / dofs;
    magma_z_matrix swap={Magma_CSR};

    for( magma_int_t i=0; i<solver_par->maxiter; i++ ) {
        CHECK( magma_z_spmv( c_neg_one, M, *x, c_zero, precond->work2, queue )); // t = - M * x

        magma_zaxpy( num_vecs*dofs, c_one ,
                precond->work1.dval, 1 , precond->work2.dval, 1, queue ); // t = t + c

        // swap so that x again contains solution, and y is ready to be used
        swap = *x;
        *x = precond->work2;
        precond->work2 = swap;
    }
    
cleanup:
    solver_par->info = info;
    return info;
}   /* magma_zjacobiiter */



    /**
    Purpose
    -------

    Iterates the solution approximation according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    This routine takes the system matrix A and the RHS b as input.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix M = D^(-1) * (L+U)
                
    @param[in]
    b           magma_z_matrix
                input RHS b
                
    @param[in]
    d           magma_z_matrix
                input matrix diagonal elements diag(A)
                
    @param[in]
    t           magma_z_matrix
                temporary vector


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
magma_zjacobiiter_sys(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix d,
    magma_z_matrix t,
    magma_z_matrix *x,
    magma_z_solver_par *solver_par,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // local variables
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO, c_one = MAGMA_Z_ONE;

    for( magma_int_t i=0; i<solver_par->maxiter; i++ ) {
        CHECK( magma_z_spmv( c_one, A, *x, c_zero, t, queue ));        // t =  A * x
        CHECK( magma_zjacobiupdate( t, b, d, x, queue ));
        // swap so that x again contains solution, and y is ready to be used
        //swap = *x;
        //*x = t;
        //t = swap;
        //magma_zcopy( dofs, t.dval, 1 , x->dval, 1 );               // x = t
    }
    
cleanup:
    solver_par->info = info;
    return info;
}   /* magma_zjacobiiter_sys */
