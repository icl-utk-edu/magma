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

    Generates an IC(0) preconditer via fixed-point iterations. 
    For reference, see:
    E. Chow and A. Patel: "Fine-grained Parallel Incomplete LU Factorization", 
    SIAM Journal on Scientific Computing, 37, C169-C193 (2015). 
    
    This is the GPU implementation of the ParIC

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
magma_zparic_gpu(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue)
{
    magma_int_t info = MAGMA_ERR_NOT_SUPPORTED;
    
#ifdef _OPENMP
    info = 0;

    magma_z_matrix hAT={Magma_CSR}, hA={Magma_CSR}, hAL={Magma_CSR}, 
    hAU={Magma_CSR}, hAUT={Magma_CSR}, hAtmp={Magma_CSR}, hACOO={Magma_CSR},
    dAL={Magma_CSR}, dACOO={Magma_CSR};

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
    
    CHECK(magma_zmconvert(hAL, &hACOO, hA.storage_type, Magma_CSRCOO, queue));
    CHECK(magma_zmtransfer(hAL, &dAL, Magma_CPU, Magma_DEV, queue));
    CHECK(magma_zmtransfer(hACOO, &dACOO, Magma_CPU, Magma_DEV, queue));
    
    // This is the actual ParIC kernel. 
    // It can be called directly if
    // - the system matrix hALCOO is available in COO format on the GPU 
    // - hAL is the lower triangular in CSR on the GPU
    // The kernel is located in sparse/blas/zparic_kernels.cu
    //
    for (int i=0; i<precond->sweeps; i++) {
        CHECK(magma_zparic_csr(dACOO, dAL, queue));
    }
    

    CHECK(magma_zmtransfer(dAL, &precond->L, Magma_DEV, Magma_DEV, queue));
    CHECK(magma_z_cucsrtranspose(precond->L, &precond->U, queue));
    CHECK(magma_zmtransfer(precond->L, &precond->M, Magma_DEV, Magma_DEV, queue));
    
    if (precond->trisolver == 0 || precond->trisolver == Magma_CUSOLVE) {
        CHECK(magma_zcumicgeneratesolverinfo(precond, queue));
    } else {
        //prepare for iterative solves

        // extract the diagonal of L into precond->d
        CHECK(magma_zjacobisetup_diagscal(precond->L, &precond->d, queue));
        CHECK(magma_zvinit(&precond->work1, Magma_DEV, hA.num_rows, 1, 
            MAGMA_Z_ZERO, queue));

        // extract the diagonal of U into precond->d2
        CHECK(magma_zjacobisetup_diagscal(precond->U, &precond->d2, queue));
        CHECK(magma_zvinit(&precond->work2, Magma_DEV, hA.num_rows, 1, 
            MAGMA_Z_ZERO, queue));
    }

cleanup:
    magma_zmfree(&hAT, queue);
    magma_zmfree(&hA, queue);
    magma_zmfree(&hAL, queue);
    magma_zmfree(&dAL, queue);
    magma_zmfree(&hAU, queue);
    magma_zmfree(&hAUT, queue);
    magma_zmfree(&hAtmp, queue);
    magma_zmfree(&hACOO, queue);
    magma_zmfree(&dACOO, queue);

#endif
    return info;
}
