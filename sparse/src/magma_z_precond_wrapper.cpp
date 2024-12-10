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

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is chosen. It approximates x for A x = y.

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

    @param[in,out]
    precond     magma_z_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_precond(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // set up precond parameters as solver parameters
    magma_z_solver_par psolver_par;
    psolver_par.rtol = precond->rtol;
    psolver_par.maxiter = precond->maxiter;
    psolver_par.restart = precond->restart;
    psolver_par.verbose = 0;
    magma_z_preconditioner pprecond;
    pprecond.solver = Magma_NONE;
    pprecond.maxiter = 3;

    switch( precond->solver ) {
        case  Magma_CG:
                CHECK( magma_zcg_res( A, b, x, &psolver_par, queue )); break;
        case  Magma_BICGSTAB:
                CHECK( magma_zbicgstab( A, b, x, &psolver_par, queue )); break;
        case  Magma_GMRES:
                CHECK( magma_zfgmres( A, b, x, &psolver_par, &pprecond, queue )); break;
        case  Magma_JACOBI:
                CHECK( magma_zjacobi( A, b, x, &psolver_par, queue )); break;
        case  Magma_IDR:
                CHECK( magma_zidr( A, b, x, &psolver_par, queue )); break;
        case  Magma_CGS:
                CHECK( magma_zcgs( A, b, x, &psolver_par, queue )); break;
        case  Magma_QMR:
                CHECK( magma_zqmr( A, b, x, &psolver_par, queue )); break;
        case  Magma_TFQMR:
                CHECK( magma_ztfqmr( A, b, x, &psolver_par, queue )); break;
        default:
                CHECK( magma_zcg_res( A, b, x, &psolver_par, queue )); break;
    }
cleanup:
    return info;
}



/**
    Purpose
    -------

    For a given input matrix M and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is preprocessed.
    E.g. for Jacobi: the scaling-vetor, for ILU the factorization.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                sparse matrix M

    @param[in]
    b           magma_z_matrix
                input vector y
    
    @param[in]
    solver      magma_z_solver_par
                solver structure using the preconditioner
                
    @param[in,out]
    precond     magma_z_preconditioner
                preconditioner
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_precondsetup(
    magma_z_matrix A, magma_z_matrix b,
    magma_z_solver_par *solver,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    // magma_zprecondfree( precond, queue );
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    
    tempo1 = magma_sync_wtime( queue );
    
    if( A.num_rows != A.num_cols ){
        printf("%% warning: non-square matrix.\n");
        printf("%% Fallback: no preconditioner.\n");
        precond->solver = Magma_NONE;
    } 
    
    if ( precond->solver == Magma_JACOBI ) {
        info = magma_zjacobisetup_diagscal( A, &(precond->d), queue );
    }
    else if ( precond->solver == Magma_PASTIX ) {
        //info = magma_zpastixsetup( A, b, precond, queue );
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
    // ILU and related
    else if ( precond->solver == Magma_ILU ) {
        info = magma_zcumilusetup( A, precond, queue );
    }
    
    // symmetric: Cholesky variant
    else if ( precond->solver == Magma_ICC ) {
        info = magma_zcumiccsetup( A, precond, queue );
    }
    // none case
    else if ( precond->solver == Magma_NONE ) {
        info = MAGMA_SUCCESS;
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
    if( 
        ( solver->solver == Magma_PQMR  || 
          solver->solver == Magma_PQMRMERGE  || 
          solver->solver == Magma_PBICG ||
          solver->solver == Magma_LSQR ) &&
        ( precond->solver == Magma_ILU      || 
            precond->solver == Magma_ICC ) ) { // also prepare the transpose
        info = magma_zcumilusetup_transpose( A, precond, queue );
    }
    
    tempo2 = magma_sync_wtime( queue );
    precond->setuptime = tempo2-tempo1;
    
    return info;
}



/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective preconditioner
    is applied.
    E.g. for Jacobi: the scaling-vetor, for ILU the triangular solves.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                sparse matrix A

    @param[in]
    b           magma_z_matrix
                input vector b

    @param[in,out]
    x           magma_z_matrix*
                output vector x

    @param[in]
    precond     magma_z_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_applyprecond(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix tmp={Magma_CSR};

    if ( precond->solver == Magma_JACOBI ) {
        CHECK( magma_zjacobi_diagscal( b.num_rows, precond->d, b, x, queue ));
    }
    else if ( precond->solver == Magma_PASTIX ) {
        //CHECK( magma_zapplypastix( b, x, precond, queue ));
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
    else if ( precond->solver == Magma_ILU ) {
        CHECK( magma_zvinit( &tmp, Magma_DEV, b.num_rows, b.num_cols, MAGMA_Z_ZERO, queue ));
    }
    else if ( precond->solver == Magma_ICC ) {
        CHECK( magma_zvinit( &tmp, Magma_DEV, b.num_rows, b.num_cols, MAGMA_Z_ZERO, queue ));
    }
    else if ( precond->solver == Magma_NONE ) {
        magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
    }
    else {
        printf( "error: preconditioner type not yet supported.\n" );
        info = MAGMA_ERR_NOT_SUPPORTED;
    }
cleanup:
    magma_zmfree( &tmp, queue );
    //magmablasSetKernelStream( orig_queue );
    return info;
}


/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective left preconditioner
    is applied.
    E.g. for Jacobi: the scaling-vetor, for ILU the left triangular solve.

    Arguments
    ---------

    @param[in]
    trans       magma_trans_t
                mode of the preconditioner: MagmaTrans or MagmaNoTrans
                
    @param[in]
    A           magma_z_matrix
                sparse matrix A

    @param[in]
    b           magma_z_matrix
                input vector b

    @param[in,out]
    x           magma_z_matrix*
                output vector x

    @param[in]
    precond     magma_z_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_applyprecond_left(
    magma_trans_t trans,
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
        //Chronometry
    real_Double_t tempo1, tempo2;
    
    tempo1 = magma_sync_wtime( queue );

    magma_zopts zopts;
    zopts.solver_par.solver = precond->trisolver;
    zopts.solver_par.maxiter = precond->maxiter;
    zopts.solver_par.verbose = 0;
    zopts.solver_par.version = 0;
    zopts.solver_par.restart = 50;
    zopts.solver_par.atol = 1e-16;
    zopts.solver_par.rtol = 1e-10;
    
    if( trans == MagmaNoTrans ) {
        if ( precond->solver == Magma_JACOBI ) {
            CHECK( magma_zjacobi_diagscal( b.num_rows, precond->d, b, x, queue ));
        }
        else if ( precond->solver == Magma_ILU &&
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ){
            CHECK( magma_zapplycumilu_l( b, x, precond, queue ));
        }
        else if (  precond->solver == Magma_ILU &&
                  ( precond->trisolver == Magma_SYNCFREESOLVE ) ){
            CHECK( magma_zapplycumilu_l( b, x, precond, queue ));
        }
        else if (  precond->solver == Magma_ICC &&
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ){
            CHECK( magma_zapplycumicc_l( b, x, precond, queue ));
        }
        else if (  precond->solver == Magma_ILU ) {
            magma_z_solver( precond->L, b, x, &zopts, queue );
        }
        else if ( precond->solver == Magma_NONE ) {
            magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
        }
        else if ( precond->solver == Magma_FUNCTION ) {
            CHECK( magma_zapplycustomprecond_l( b, x, precond, queue ));
        }
        else {
            printf( "error: preconditioner type not yet supported.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED; 
        }
    } else if ( trans == MagmaTrans ){
        if ( precond->solver == Magma_JACOBI ) {
            CHECK( magma_zjacobi_diagscal( b.num_rows, precond->d, b, x, queue ));
        }
        else if (  precond->solver == Magma_ILU && 
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ){
            CHECK( magma_zapplycumilu_l_transpose( b, x, precond, queue ));
        }
        else if (  precond->solver == Magma_ICC &&
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ){
            CHECK( magma_zapplycumicc_l( b, x, precond, queue ));
        }
        else if (  precond->solver == Magma_ILU ) {
            magma_z_solver( precond->L, b, x, &zopts, queue );
        }
        else if ( precond->solver == Magma_NONE ) {
            magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
        }
        else if ( precond->solver == Magma_FUNCTION ) {
            CHECK( magma_zapplycustomprecond_l( b, x, precond, queue ));
        }
        else {
            printf( "error: preconditioner type not yet supported.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED; 
        }
    } else {
        printf( "error: preconditioner type not yet supported.\n" );
        info = MAGMA_ERR_NOT_SUPPORTED; 
    }
    
    tempo2 = magma_sync_wtime( queue );
    precond->runtime += tempo2-tempo1;
    
cleanup:
    return info;
}


/**
    Purpose
    -------

    For a given input matrix A and vectors x, y and the
    preconditioner parameters, the respective right-preconditioner
    is applied.
    E.g. for Jacobi: the scaling-vetor, for ILU the right triangular solve.

    Arguments
    ---------
    
    @param[in]
    trans       magma_trans_t
                mode of the preconditioner: MagmaTrans or MagmaNoTrans

    @param[in]
    A           magma_z_matrix
                sparse matrix A

    @param[in]
    b           magma_z_matrix
                input vector b

    @param[in,out]
    x           magma_z_matrix*
                output vector x

    @param[in]
    precond     magma_z_preconditioner
                preconditioner

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_applyprecond_right(
    magma_trans_t trans,
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
        //Chronometry
    real_Double_t tempo1, tempo2;
    
    tempo1 = magma_sync_wtime( queue );
    
    magma_zopts zopts;
    zopts.solver_par.solver = precond->trisolver;
    zopts.solver_par.maxiter = precond->maxiter;
    zopts.solver_par.verbose = 0;
    zopts.solver_par.version = 0;
    zopts.solver_par.restart = 50;
    zopts.solver_par.atol = 1e-16;
    zopts.solver_par.rtol = 1e-10;
    
    if( trans == MagmaNoTrans ) {
        if ( precond->solver == Magma_JACOBI ) {
            magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );    // x = b
        }
        else if (  precond->solver == Magma_ILU && 
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ) {
            CHECK( magma_zapplycumilu_r( b, x, precond, queue ));
        }
        else if (  precond->solver == Magma_ILU && 
                  ( precond->trisolver == Magma_SYNCFREESOLVE ) ){
            CHECK( magma_zapplycumilu_r( b, x, precond, queue ));
        }
        else if (precond->solver == Magma_ILUT) {
            printf( "error: preconditioner requires OpenMP.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED;
            //magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );
        }
        else if (  precond->solver == Magma_ICC &&
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ){
            CHECK( magma_zapplycumicc_r( b, x, precond, queue ));
        }
        else if (  precond->solver == Magma_ILU ) { 
            magma_z_solver( precond->U, b, x, &zopts, queue );
        }
        else if ( precond->solver == Magma_NONE ) {
            magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
        }
        else if ( precond->solver == Magma_FUNCTION ) {
            CHECK( magma_zapplycustomprecond_r( b, x, precond, queue ));
        }
        else {
            printf( "error: preconditioner type not yet supported.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED;
        }
    } else if ( trans == MagmaTrans ){
        if ( precond->solver == Magma_JACOBI ) {
            magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );    // x = b
        }
        else if (  precond->solver == Magma_ILU && 
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ){
            CHECK( magma_zapplycumilu_r_transpose( b, x, precond, queue ));
        }
        else if (  precond->solver == Magma_ICC && 
                  ( precond->trisolver == Magma_CUSOLVE ||
                    precond->trisolver == 0 ) ){
            CHECK( magma_zapplycumicc_r( b, x, precond, queue ));
        }
        else if (  precond->solver == Magma_ILU ) {
            magma_z_solver( precond->U, b, x, &zopts, queue );
        }
        else if ( precond->solver == Magma_NONE ) {
            magma_zcopy( b.num_rows*b.num_cols, b.dval, 1, x->dval, 1, queue );      //  x = b
        }
        else if ( precond->solver == Magma_FUNCTION ) {
            CHECK( magma_zapplycustomprecond_r( b, x, precond, queue ));
        }
        else {
            printf( "error: preconditioner type not yet supported.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED;
        }
    }
    
    tempo2 = magma_sync_wtime( queue );
    precond->runtime += tempo2-tempo1;
        
cleanup:
    return info;
}
