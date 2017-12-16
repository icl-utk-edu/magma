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


/***************************************************************************//**
    Purpose
    -------

    Generates an ILU(0) preconditer via fixed-point iterations. 
    For reference, see:
    E. Chow and A. Patel: "Fine-grained Parallel Incomplete LU Factorization", 
    SIAM Journal on Scientific Computing, 37, C169-C193 (2015). 
    
    This is the GPU implementation of the ParILU

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
magma_zparilu_gpu(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue)
{
    magma_int_t info = 0;

    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrL=NULL;
    cusparseMatDescr_t descrU=NULL;

    magma_z_matrix hAT={Magma_CSR}, hA={Magma_CSR}, hAL={Magma_CSR}, 
    hAU={Magma_CSR}, hAUT={Magma_CSR}, hAtmp={Magma_CSR}, hACOO={Magma_CSR},
    dAL={Magma_CSR}, dAU={Magma_CSR}, dAUT={Magma_CSR}, dACOO={Magma_CSR};

    // copy original matrix as COO to device
    if (A.memory_location != Magma_CPU || A.storage_type != Magma_CSR) {
        CHECK(magma_zmtransfer(A, &hAT, A.memory_location, Magma_CPU, queue));
        CHECK(magma_zmconvert(hAT, &hA, hAT.storage_type, Magma_CSR, queue));
        magma_zmfree(&hAT, queue);
    } else {
        CHECK(magma_zmtransfer(A, &hA, A.memory_location, Magma_CPU, queue));
    }

    // in case using fill-in
    if (precond->levels > 0) {
        CHECK(magma_zsymbilu(&hA, precond->levels, &hAL, &hAUT,  queue));
        magma_zmfree(&hAL, queue);
        magma_zmfree(&hAUT, queue);
    }
    CHECK(magma_zmconvert(hA, &hACOO, hA.storage_type, Magma_CSRCOO, queue));
    
    //get L
    magma_zmatrix_tril(hA, &hAL, queue);
    // we need 1 on the main diagonal of L
    #pragma omp parallel for
    for (int k=0; k < hAL.num_rows; k++) {
        hAL.val[hAL.row[k+1]-1] = MAGMA_Z_ONE;
    }
    
    // get U
    magma_zmtranspose(hA, &hAT, queue);
    magma_zmatrix_tril(hAT, &hAU, queue);
    magma_zmfree(&hAT, queue);
    
    CHECK(magma_zmtransfer(hAL, &dAL, Magma_CPU, Magma_DEV, queue));
    CHECK(magma_zmtransfer(hAU, &dAU, Magma_CPU, Magma_DEV, queue));
    CHECK(magma_zmtransfer(hACOO, &dACOO, Magma_CPU, Magma_DEV, queue));
    
    // This is the actual ParILU kernel. 
    // It can be called directly if
    // - the system matrix hACOO is available in COO format on the CPU 
    // - hAL is the lower triangular in CSR on the CPU
    // - hAU is the upper triangular in CSC on the CPU (U transpose in CSR)
    // The kernel is located in sparse/blas/zparilu_kernels.cu
    //
    for (int i=0; i<precond->sweeps; i++) {
        CHECK(magma_zparilu_csr(dACOO, dAL, dAU, queue));
    }
    CHECK(magma_z_cucsrtranspose(dAU, &dAUT, queue));

    CHECK(magma_zmtransfer(dAL, &precond->L, Magma_DEV, Magma_DEV, queue));
    CHECK(magma_zmtransfer(dAUT, &precond->U, Magma_DEV, Magma_DEV, queue));
    
    // For Jacobi-type triangular solves
    // extract the diagonal of L into precond->d
    CHECK(magma_zjacobisetup_diagscal(precond->L, &precond->d, queue));
    CHECK(magma_zvinit(&precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, 
        queue));
    
    // For Jacobi-type triangular solves
    // extract the diagonal of U into precond->d2
    CHECK(magma_zjacobisetup_diagscal(precond->U, &precond->d2, queue));
    CHECK(magma_zvinit(&precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, 
        queue));

    magma_zmfree(&hAL, queue);
    magma_zmfree(&hAU, queue);
    
    // CUSPARSE context for cuSPARSE triangular solves//
    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrL));
    CHECK_CUSPARSE(cusparseSetMatType(descrL, CUSPARSE_MATRIX_TYPE_TRIANGULAR));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descrL, CUSPARSE_DIAG_TYPE_NON_UNIT));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrL, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descrL, CUSPARSE_FILL_MODE_LOWER));
    CHECK_CUSPARSE(cusparseCreateSolveAnalysisInfo(&precond->cuinfoL));
    CHECK_CUSPARSE(cusparseZcsrsv_analysis(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->L.num_rows,
        precond->L.nnz, descrL,
        precond->L.val, precond->L.row, precond->L.col, precond->cuinfoL));
    CHECK_CUSPARSE(cusparseCreateMatDescr(&descrU));
    CHECK_CUSPARSE(cusparseSetMatType(descrU, CUSPARSE_MATRIX_TYPE_TRIANGULAR));
    CHECK_CUSPARSE(cusparseSetMatDiagType(descrU, CUSPARSE_DIAG_TYPE_NON_UNIT));
    CHECK_CUSPARSE(cusparseSetMatIndexBase(descrU, CUSPARSE_INDEX_BASE_ZERO));
    CHECK_CUSPARSE(cusparseSetMatFillMode(descrU, CUSPARSE_FILL_MODE_UPPER));
    CHECK_CUSPARSE(cusparseCreateSolveAnalysisInfo(&precond->cuinfoU));
    CHECK_CUSPARSE(cusparseZcsrsv_analysis(cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->U.num_rows,
        precond->U.nnz, descrU,
        precond->U.val, precond->U.row, precond->U.col, precond->cuinfoU));
    
    
cleanup:
    cusparseDestroy(cusparseHandle);
    cusparseDestroyMatDescr(descrL);
    cusparseDestroyMatDescr(descrU);
    cusparseHandle=NULL;
    descrL=NULL;
    descrU=NULL;
    magma_zmfree(&dAL, queue);
    magma_zmfree(&dAU, queue);
    magma_zmfree(&dAUT, queue);
    magma_zmfree(&dACOO, queue);
    magma_zmfree(&hAT, queue);
    magma_zmfree(&hA, queue);
    magma_zmfree(&hAL, queue);
    magma_zmfree(&hAU, queue);
    magma_zmfree(&hAUT, queue);
    magma_zmfree(&hAtmp, queue);
    magma_zmfree(&hACOO, queue);

    
    return info;
}

