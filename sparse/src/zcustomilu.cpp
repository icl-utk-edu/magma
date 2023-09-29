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


#include "../blas/magma_trisolve.h"

#define COMPLEX

/* For hipSPARSE, they use a separate complex type than for hipBLAS */
#ifdef MAGMA_HAVE_HIP
  #define hipblasDoubleComplex hipDoubleComplex
#endif

/**
    Purpose
    -------

    Reads in an Incomplete LU preconditioner.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A
                
    @param[in]
    b           magma_z_matrix
                input RHS b

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_zcustomilusetup(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
#ifndef MAGMA_HAVE_SYCL
    magma_int_t info = 0;
    
    magma_z_matrix hA={Magma_CSR};
    char preconditionermatrix[255];
    
    // first L
    snprintf( preconditionermatrix, sizeof(preconditionermatrix),
                "/Users/hanzt0114cl306/work/matrices/matrices/ILUT_L.mtx" );
    
    CHECK( magma_z_csr_mtx( &hA, preconditionermatrix , queue) );
    CHECK( magma_zmtransfer( hA, &precond->L, Magma_CPU, Magma_DEV , queue ));
    // extract the diagonal of L into precond->d
    CHECK( magma_zjacobisetup_diagscal( precond->L, &precond->d, queue ));
    CHECK( magma_zvinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, queue ));

    magma_zmfree( &hA, queue );
    
    // now U
    snprintf( preconditionermatrix, sizeof(preconditionermatrix),
                "/Users/hanzt0114cl306/work/matrices/matrices/ILUT_U.mtx" );

    CHECK( magma_z_csr_mtx( &hA, preconditionermatrix , queue) );
    CHECK( magma_zmtransfer( hA, &precond->U, Magma_CPU, Magma_DEV , queue ));
    // extract the diagonal of U into precond->d2
    CHECK( magma_zjacobisetup_diagscal( precond->U, &precond->d2, queue ));
    CHECK( magma_zvinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, queue ));

    CHECK(magma_ztrisolve_analysis(precond->L, &precond->cuinfoL, false, true, false, queue));
    CHECK(magma_ztrisolve_analysis(precond->U, &precond->cuinfoU, true, false, false, queue));
        
cleanup:        
    magma_zmfree( &hA, queue );
    
    return info;
#else
    magma_unsupported_sparse(magma_zcustomicsetup);
#endif    
}
    
