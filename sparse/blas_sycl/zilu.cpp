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
#include "magma_trisolve.h"

#define PRECISION_z

/**
    Purpose
    -------

    Prepares the ILU preconditioner via the cuSPARSE.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/

extern "C" magma_int_t
magma_zcumilusetup(
    magma_z_matrix A,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
  printf("error: magma_zcumilusetup is not supported with SYCL backend.\n");
  return magma_unsupported_sparse(magma_z_cumilusetup);

}



/**
    Purpose
    -------

    Prepares the ILU transpose preconditioner via the cuSPARSE.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/

extern "C" magma_int_t
magma_zcumilusetup_transpose(
    magma_z_matrix A,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
  printf("error: magma_zcumilusetup_transpose is not supported with SYCL backend.\n");
  return magma_unsupported_sparse(magma_z_cumilusetup_transpose);
}



/**
    Purpose
    -------

    Prepares the ILU triangular solves via cuSPARSE using an ILU factorization
    matrix stored either in precond->M or on the device as
    precond->L and precond->U.

    Arguments
    ---------

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/

extern "C" magma_int_t
magma_zcumilugeneratesolverinfo(
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
  printf("error: magma_zcumilugeneratesolverinfo is not supported with SYCL backend.\n");
  return magma_unsupported_sparse(magma_zcumilugeneratesolverinfo);
}


/**
    Purpose
    -------

    Performs the left triangular solves using the ILU preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                RHS

    @param[in,out]
    x           magma_z_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/

extern "C" magma_int_t
magma_zapplycumilu_l(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
  printf("error: magma_zapplycumilu_l is not supported with SYCL backend.\n");
  return magma_unsupported_sparse(magma_zapplycumilu_l);
}



/**
    Purpose
    -------

    Performs the left triangular solves using the transpose ILU preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                RHS

    @param[in,out]
    x           magma_z_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/
   
extern "C" magma_int_t
magma_zapplycumilu_l_transpose(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
  printf("error: magma_zapplycumilu_l_transpose is not supported with SYCL backend.\n");
  return magma_unsupported_sparse(magma_zapplycumilu_l_transpose);
}


/**
    Purpose
    -------

    Performs the right triangular solves using the ILU preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                RHS

    @param[in,out]
    x           magma_z_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/

extern "C" magma_int_t
magma_zapplycumilu_r(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
  printf("error: magma_zapplycumilu_r is not supported with SYCL backend.\n");
  return magma_unsupported_sparse(magma_zapplycumilu_r);
}


/**
    Purpose
    -------

    Performs the right triangular solves using the transpose ILU preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                RHS

    @param[in,out]
    x           magma_z_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/

extern "C" magma_int_t
magma_zapplycumilu_r_transpose(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
  printf("error: magma_zapplycumilu_r_transpose is not supported with SYCL backend.\n");
  return magma_unsupported_sparse(magma_zapplycumilu_r_transpose);
}


/**
    Purpose
    -------

    Prepares the IC preconditioner via cuSPARSE.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zhepr
*******************************************************************************/

extern "C" magma_int_t
magma_zcumiccsetup(
    magma_z_matrix A,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
  printf("error: magma_zcumiccsetup is not supported with SYCL backend.\n");
  return magma_unsupported_sparse(magma_z_cumiccsetup);
}

/**
    Purpose
    -------

    Prepares the IC preconditioner solverinfo via cuSPARSE for a triangular
    matrix present on the device in precond->M.

    Arguments
    ---------
    
    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zhepr
    ********************************************************************/

extern "C" magma_int_t
magma_zcumicgeneratesolverinfo(
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
  printf("error: magma_zcumicgeneratesolverinfo is not supported with SYCL backend.\n");
  return magma_unsupported_sparse(magma_zcumicgeneratesolverinfo);
}



/**
    Purpose
    -------

    Performs the left triangular solves using the ICC preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                RHS

    @param[in,out]
    x           magma_z_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zhepr
    ********************************************************************/

extern "C" magma_int_t
magma_zapplycumicc_l(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
  printf("error: magma_zapplycumicc_l is not supported with SYCL backend.\n");
  return magma_unsupported_sparse(magma_zapplycumicc_l);
}


/**
    Purpose
    -------

    Performs the right triangular solves using the ICC preconditioner.

    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                RHS

    @param[in,out]
    x           magma_z_matrix*
                vector to precondition

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zhepr
    ********************************************************************/

extern "C" magma_int_t
magma_zapplycumicc_r(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
  printf("error: magma_zapplycumicc_r is not supported with SYCL backend.\n");
  return magma_unsupported_sparse(magma_zapplycumicc_r);
}



/**
    Purpose
    -------

    Performs the left triangular solves using the IC preconditioner via Jacobi.

    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                RHS

    @param[out]
    x           magma_z_matrix*
                vector to precondition

    @param[in]
    precond     magma_z_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

   @ingroup magmasparse_zgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_zapplyiteric_l(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t dofs = precond->L.num_rows;
    magma_z_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = precond->maxiter;

    // compute c = D^{-1}b and copy c as initial guess to x
    CHECK( magma_zjacobisetup_vector_gpu( dofs, b, precond->d,
                                                precond->work1, x, queue ));
    // Jacobi iterator
    CHECK( magma_zjacobiiter_precond( precond->L, x, &jacobiiter_par, precond , queue ));

cleanup:
    return info;
}


/**
    Purpose
    -------

    Performs the right triangular solves using the IC preconditioner via Jacobi.

    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                RHS

    @param[out]
    x           magma_z_matrix*
                vector to precondition

    @param[in]
    precond     magma_z_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_zapplyiteric_r(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t dofs = precond->U.num_rows;
    magma_z_solver_par jacobiiter_par;
    jacobiiter_par.maxiter = precond->maxiter;

    // compute c = D^{-1}b and copy c as initial guess to x
    CHECK( magma_zjacobisetup_vector_gpu( dofs, b, precond->d,
                                                precond->work1, x, queue ));

    // Jacobi iterator
    CHECK( magma_zjacobiiter_precond( precond->U, x, &jacobiiter_par, precond , queue ));
    
cleanup:
    return info;
}

