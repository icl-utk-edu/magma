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

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define PRECISION_z


/**
    Purpose
    -------

    Prepares the iterative threshold Incomplete Cholesky preconditioner.
    
    This function requires OpenMP, and is only available if OpenMP is activated. 

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
magma_zparictsetup(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

#ifdef _OPENMP

    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrL=NULL;
    cusparseMatDescr_t descrU=NULL;
    magma_index_t *rm_loc = NULL; 
    magma_index_t *rm_locT = NULL; 
    magma_int_t tri;
    
    magma_z_matrix hA={Magma_CSR}, LU={Magma_CSR}, LU_new={Magma_CSR}, 
                   LUCSR={Magma_CSR}, L={Magma_CSR};
    
    magma_int_t num_rm, num_rm_gl;
    magmaDoubleComplex thrs = MAGMA_Z_ZERO;
    
    omp_lock_t rowlock[A.num_rows];
    for (magma_int_t i=0; i<A.num_rows; i++){
        omp_init_lock(&(rowlock[i]));
    }
    magma_set_omp_numthreads( 4 );
    
    CHECK( magma_index_malloc_cpu( &rm_loc, A.nnz ) );  
    num_rm_gl = 0.025*A.nnz;
    tri = 0;
    
    CHECK( magma_zmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
    magma_zmconvert( hA, &L, Magma_CSR, Magma_CSRL, queue );
    magma_zmconvert( L, &LU, Magma_CSR, Magma_CSRLIST, queue );
    
    magma_zmalloc_cpu( &LU_new.val, LU.nnz*5 );
    magma_index_malloc_cpu( &LU_new.rowidx, LU.nnz*5 );
    magma_index_malloc_cpu( &LU_new.col, LU.nnz*5 );
    LU_new.num_rows = LU.num_rows;
    LU_new.num_cols = LU.num_cols;
    LU_new.storage_type = Magma_COO;
    LU_new.memory_location = Magma_CPU;

    magma_zmdynamicic_sweep( hA, &LU, queue );
    magma_zmdynamicic_sweep( hA, &LU, queue );
    magma_zmdynamicic_sweep( hA, &LU, queue );
    magma_zmdynamicic_sweep( hA, &LU, queue );

    for( magma_int_t iters =0; iters<precond->sweeps; iters++ ) {
        num_rm = num_rm_gl;
        info = magma_zmdynamicilu_set_thrs( num_rm, &LU, &thrs, queue );
        
        // workaround to avoid breakdown
        if( info !=0 ){
            printf("%% error: breakdown in iteration :%d. fallback.\n\n", iters+1);
            info = 0;
            break;
        }
        magma_zmfree( &LUCSR, queue );
        magma_zmconvert( LU, &LUCSR, Magma_CSRLIST, Magma_CSR, queue );  
        // end workaround
        
        magma_zmdynamicilu_rm_thrs( &thrs, &num_rm, &LU, &LU_new, rm_loc, rowlock, queue );
        magma_zmdynamicic_sweep( hA, &LU, queue );
        magma_zmdynamicic_candidates( LU, &LU_new, queue );
        magma_zmdynamicic_residuals( hA, LU, &LU_new, queue );
        magma_zmdynamicic_insert( tri, num_rm, rm_loc, &LU_new, &LU, rowlock, queue );
        magma_zmdynamicic_sweep( hA, &LU, queue );
    }

    // for CUSPARSE
    CHECK( magma_zmtransfer( LUCSR, &precond->M, Magma_CPU, Magma_DEV , queue ));
    
        // copy the matrix to precond->L and (transposed) to precond->U
    CHECK( magma_zmtransfer(precond->M, &(precond->L), Magma_DEV, Magma_DEV, queue ));
    CHECK( magma_zmtranspose( precond->L, &(precond->U), queue ));
    
    // extract the diagonal of L into precond->d
    CHECK( magma_zjacobisetup_diagscal( precond->L, &precond->d, queue ));
    CHECK( magma_zvinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, queue ));

    // extract the diagonal of U into precond->d2
    CHECK( magma_zjacobisetup_diagscal( precond->U, &precond->d2, queue ));
    CHECK( magma_zvinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, queue ));


    // CUSPARSE context //
    CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrL ));
    CHECK_CUSPARSE( cusparseSetMatType( descrL, CUSPARSE_MATRIX_TYPE_TRIANGULAR ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrL, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrL, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE( cusparseSetMatFillMode( descrL, CUSPARSE_FILL_MODE_LOWER ));
    CHECK_CUSPARSE( cusparseCreateSolveAnalysisInfo( &precond->cuinfoL ));
    CHECK_CUSPARSE( cusparseZcsrsv_analysis( cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->M.num_rows,
        precond->M.nnz, descrL,
        precond->M.val, precond->M.row, precond->M.col, precond->cuinfoL ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrU ));
    CHECK_CUSPARSE( cusparseSetMatType( descrU, CUSPARSE_MATRIX_TYPE_TRIANGULAR ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrU, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrU, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE( cusparseSetMatFillMode( descrU, CUSPARSE_FILL_MODE_LOWER ));
    CHECK_CUSPARSE( cusparseCreateSolveAnalysisInfo( &precond->cuinfoU ));
    CHECK_CUSPARSE( cusparseZcsrsv_analysis( cusparseHandle,
        CUSPARSE_OPERATION_TRANSPOSE, precond->M.num_rows,
        precond->M.nnz, descrU,
        precond->M.val, precond->M.row, precond->M.col, precond->cuinfoU ));

    
    cleanup:
        
    for (magma_int_t i=0; i<A.num_rows; i++){
        omp_destroy_lock(&(rowlock[i]));
    }
    magma_free_cpu( rm_loc );
    magma_free_cpu( rm_locT );
    cusparseDestroy( cusparseHandle );
    cusparseDestroyMatDescr( descrL );
    cusparseDestroyMatDescr( descrU );
    cusparseHandle=NULL;
    descrL=NULL;
    descrU=NULL;    
    magma_zmfree( &hA, queue );
    magma_zmfree( &LUCSR, queue );
    magma_zmfree( &LU_new, queue );
    magma_zmfree( &L, queue );
    magma_zmfree( &LU, queue );
#endif
    return info;
}
