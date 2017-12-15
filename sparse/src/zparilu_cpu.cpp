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
#ifdef _OPENMP
#include <omp.h>
#endif

#define PRECISION_z


/***************************************************************************//**
    Purpose
    -------

    Prepares the iterative threshold Incomplete LU preconditioner. The strategy
    is interleaving a parallel fixed-point iteration that approximates an
    incomplete factorization for a given nonzero pattern with a procedure that
    adaptively changes the pattern. Much of this new algorithm has fine-grained
    parallelism, and we show that it can efficiently exploit the compute power
    of shared memory architectures.

    This is the routine used in the publication by Anzt, Chow, Dongarra:
    ''ParILUT - A new parallel threshold ILU factorization''
    submitted to SIAM SISC in 2017.

    This function requires OpenMP, and is only available if OpenMP is activated.
    
    The parameter list is:
    
    precond.sweeps : number of ParILUT steps
    precond.atol   : absolute fill ratio (1.0 keeps nnz constant)
    precond.rtol   : how many candidates are added to the sparsity pattern
                        * 1.0 one per row
                        * < 1.0 a fraction of those
                        * > 1.0 all candidates

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
*******************************************************************************/
extern "C"
magma_int_t
magma_zparilu_cpu(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_ERR_NOT_SUPPORTED;
    
#ifdef _OPENMP
    info = 0;

    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrL=NULL;
    cusparseMatDescr_t descrU=NULL;

    magma_z_matrix hAh={Magma_CSR}, hA={Magma_CSR}, hL={Magma_CSR}, 
    hU={Magma_CSR}, hAL={Magma_CSR}, hAU={Magma_CSR}, 
    hAUt={Magma_CSR}, hUT={Magma_CSR}, hAtmp={Magma_CSR}, hACOO={Magma_CSR};

    // copy original matrix as COO to device
    CHECK(magma_zmtransfer(A, &hAh, A.memory_location, Magma_CPU, queue));
    CHECK(magma_zmconvert(hAh, &hA, hAh.storage_type, Magma_CSR, queue));
    CHECK(magma_zmconvert(hA, &hACOO, Magma_CSR, Magma_CSRCOO, queue));
    magma_zmfree(&hAh, queue);


    // in case using fill-in
    CHECK(magma_zsymbilu(&hA, precond->levels, &hAL, &hAUt,  queue));
    magma_zmfree(&hAL, queue);
    magma_zmfree(&hAUt, queue);
    magma_zmatrix_tril( hA, &hAL, queue );
    magma_zmtranspose(hA, &hAh, queue );
    hAU.diagorder_type = Magma_UNITY;
    magma_zmatrix_tril( hAh, &hAU, queue );
    
    // This is the actual ParILU kernel. 
    // It can be called directly if
    // - the system matrix hACOO is available in COO format on the CPU 
    // - hAL is the lower triangular in CSR on the CPU
    // - hAU is the upper triangular in CSC on the CPU
    // The kernel is located in sparse/control/zparilut_tools_sync.cu.
    //
    for (int i=0; i<precond->sweeps; i++) {
        CHECK(magma_zparilu_sweep_sync(hACOO, &hAL, &hAU, queue));
    }
    
    CHECK(magma_z_cucsrtranspose(hAU, &hUT, queue));

    magma_zmfree(&hAU, queue);
    CHECK(magma_zmlumerge(hAL, hUT, &hAtmp, queue));


    CHECK(magma_zmtransfer(hAtmp, &precond->M, Magma_CPU, Magma_DEV, queue));

    hAL.diagorder_type = Magma_UNITY;
    CHECK(magma_zmconvert(hAtmp, &hAL, Magma_CSR, Magma_CSRL, queue));
    hAL.storage_type = Magma_CSR;
    CHECK(magma_zmconvert(hAtmp, &hAU, Magma_CSR, Magma_CSRU, queue));
    hAU.storage_type = Magma_CSR;

    magma_zmfree(&hAtmp, queue);

    // for cusparse uncomment this
    CHECK(magma_zmtransfer(hAL, &precond->L, Magma_CPU, Magma_DEV, queue));
    CHECK(magma_zmtransfer(hAU, &precond->U, Magma_CPU, Magma_DEV, queue));
    
    /*
    //-- for ba-solve uncomment this

    if(RL.nnz != 0 )
        CHECK(magma_zmtransfer(RL, &precond->L, Magma_CPU, Magma_DEV, queue));
    else {
        precond->L.nnz = 0;
        precond->L.val = NULL;
        precond->L.col = NULL;
        precond->L.row = NULL;
        precond->L.blockinfo = NULL;
    }

    if(RU.nnz != 0 )
        CHECK(magma_zmtransfer(RU, &precond->U, Magma_CPU, Magma_DEV, queue));
    else {
        precond->U.nnz = 0;
        precond->L.val = NULL;
        precond->L.col = NULL;
        precond->L.row = NULL;
        precond->L.blockinfo = NULL;
    }
    //-- for ba-solve uncomment this
    */

        // extract the diagonal of L into precond->d
    CHECK(magma_zjacobisetup_diagscal(precond->L, &precond->d, queue));
    CHECK(magma_zvinit(&precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, 
        queue));
    
    // extract the diagonal of U into precond->d2
    CHECK(magma_zjacobisetup_diagscal(precond->U, &precond->d2, queue));
    CHECK(magma_zvinit(&precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, 
        queue));

    magma_zmfree(&hAL, queue);
    magma_zmfree(&hAU, queue);
    
    /*
    //-- for ba-solve uncomment this
    // magma_zmfree(&DL, queue);
    // magma_zmfree(&RL, queue);
    // magma_zmfree(&DU, queue);
    // magma_zmfree(&RU, queue);
    //-- for ba-solve uncomment this
    */

    // CUSPARSE context for cuSPARSE triangular solves//
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle ));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrL ));
    CHECK_CUSPARSE(cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_TRIANGULAR ));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER ));
    CHECK_CUSPARSE(cusparseCreateSolveAnalysisInfo(&precond->cuinfoL ));
    CHECK_CUSPARSE(cusparseZcsrsv_analysis(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->L.num_rows,
        precond->L.nnz, descrL,
        precond->L.val, precond->L.row, precond->L.col, precond->cuinfoL ));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrU ));
    CHECK_CUSPARSE(cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_TRIANGULAR ));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER ));
    CHECK_CUSPARSE(cusparseCreateSolveAnalysisInfo(&precond->cuinfoU ));
    CHECK_CUSPARSE(cusparseZcsrsv_analysis(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->U.num_rows,
        precond->U.nnz, descrU,
        precond->U.val, precond->U.row, precond->U.col, precond->cuinfoU ));
    
    
cleanup:
    cusparseDestroy(cusparseHandle );
    cusparseDestroyMatDescr(descrL );
    cusparseDestroyMatDescr(descrU );
    cusparseHandle=NULL;
    descrL=NULL;
    descrU=NULL;
    magma_zmfree(&hAh, queue);
    magma_zmfree(&hA, queue);
    magma_zmfree(&hL, queue);
    magma_zmfree(&hU, queue);
    magma_zmfree(&hAL, queue);
    magma_zmfree(&hAU, queue);
    magma_zmfree(&hAUt, queue);
    magma_zmfree(&hUT, queue);
    magma_zmfree(&hAtmp, queue);
    magma_zmfree(&hACOO, queue);
    /*
    //-- for ba-solve uncomment this
    // magma_zmfree(&DL, queue);
    // magma_zmfree(&RL, queue);
    // magma_zmfree(&DU, queue);
    // magma_zmfree(&RU, queue);
    //-- for ba-solve uncomment this
    */

#endif
    return info;
}
