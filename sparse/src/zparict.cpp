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
    real_Double_t start, end;
#ifdef _OPENMP

    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrL=NULL;
    cusparseMatDescr_t descrU=NULL;
    magma_index_t *rm_loc = NULL; 
    magma_index_t *rm_locT = NULL; 
    magma_int_t tri;
    
    magma_z_matrix hA={Magma_CSR}, LU={Magma_CSR}, LUT={Magma_CSR}, LU_new={Magma_CSR}, 
                   LUCSR={Magma_CSR}, L={Magma_CSR}, U={Magma_CSR},
                   hAL={Magma_CSR}, hAUt={Magma_CSR};
                   
    magma_int_t num_rm, num_rm_gl;
    magmaDoubleComplex thrs = MAGMA_Z_ZERO;
    
    magma_int_t approx = 0;

    omp_lock_t rowlock[1];

    CHECK( magma_index_malloc_cpu( &rm_loc, A.nnz ) ); 
    CHECK( magma_index_malloc_cpu( &rm_locT, A.nnz ) ); 
    num_rm_gl = precond->rtol*A.nnz;
    num_rm = num_rm_gl;
    tri = 0;

    CHECK( magma_zmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
    
    
        // in case using fill-in
    CHECK( magma_zsymbilu( &hA, precond->levels, &hAL, &hAUt , queue ));
    // need only lower triangular
    magma_zmfree(&hAUt, queue );
    magma_zmfree(&hAL, queue );
    
    magma_zmconvert( hA, &L, Magma_CSR, Magma_CSRL, queue );
    magma_zmconvert( L, &LU, Magma_CSR, Magma_CSRLIST, queue );
    magma_zmconvert( LU, &LUCSR, Magma_CSRLIST, Magma_CSR, queue ); 
    magma_zmconvert( hA, &U, Magma_CSR, Magma_CSRU, queue );
    magma_zmconvert( U, &LUT, Magma_CSR, Magma_CSRLIST, queue );
    magma_index_malloc_cpu( &L.list, L.nnz );
    
    for( magma_int_t t=0; t<LUT.nnz; t++ ){
           LUT.val[t] = MAGMA_Z_ONE;
    }

    magma_zmalloc_cpu( &LU_new.val, LU.nnz*20 );
    magma_index_malloc_cpu( &LU_new.rowidx, LU.nnz*20 );
    magma_index_malloc_cpu( &LU_new.col, LU.nnz*20 );
    magma_index_malloc_cpu( &LU_new.row, LU.nnz*20 );
    magma_index_malloc_cpu( &LU_new.list, LU.nnz*20 );
    LU_new.num_rows = LU.num_rows;
    LU_new.num_cols = LU.num_cols;
    LU_new.storage_type = Magma_COO;
    LU_new.memory_location = Magma_CPU;
    
    for( magma_int_t z=0; z<num_rm; z++ ){
        rm_loc[z] = LU.nnz+z;
        rm_locT[z] = LU.nnz+z;
        LU.val[ LU.nnz+z ] = MAGMA_Z_ZERO;
        LU.list[ LU.nnz+z ] = -1;
        LUT.val[ LU.nnz+z ] = MAGMA_Z_ZERO;
        LUT.list[ LU.nnz+z ] = -1;
    }

    magma_zparict_sweep( &L, &LU, queue );
    magma_zparict_sweep( &L, &LU, queue );
    magma_zparict_sweep( &L, &LU, queue );
    magma_zparict_sweep( &L, &LU, queue );
    

    //info = magma_zparilut_set_approx_thrs( num_rm, &LU, 0, &thrs, queue );
    start = magma_sync_wtime( queue );
    
    
    for (magma_int_t iters = 0; iters < precond->sweeps; iters++) {
        // first: candidates
        printf("\n%%candidates..."); fflush(stdout);
        magma_zmilu0_candidates( L, LU, LUT, &LU_new, queue );
        
        // then residuals
        printf("residuals..."); fflush(stdout);
        magma_zparict_residuals( A, LU, &LU_new, queue );
        magma_zmeliminate_duplicates( num_rm, &LU_new, queue );
        LU.nnz = LU.nnz+num_rm;
        LUT.nnz = LUT.nnz+num_rm;
        //then insert the largest residuals
        printf("insert..."); fflush(stdout);
        magma_zparilut_insert_LU( num_rm, rm_loc, rm_locT, &LU_new, &LU, &LUT, queue );
        
        //now do a sweep
        printf("sweep..."); fflush(stdout);
        magma_zparict_sweep( &L, &LU, queue );
        magma_zparict_sweep( &L, &LU, queue );
        printf("threshold..."); fflush(stdout);
        info = magma_zparilut_set_thrs( num_rm, &LU, 0, &thrs, queue );
        if (info != 0) {
            printf("%% error: breakdown in iteration :%d. fallback.\n\n", iters+1);
            info = 0;
            break;
        }
        
        // and remove
        printf("remove "); fflush(stdout);
        printf("%d elements...", num_rm);
        magma_zparilut_rm_thrs( &thrs, &num_rm, &LU, &LU_new, rm_loc, rowlock, queue );
        magma_zparilut_rm_thrs_U( &thrs, &num_rm, &LUT, &LU_new, rm_locT, rowlock, queue );
        
        printf("reorder..."); fflush(stdout);
        info = magma_zparilut_reorder( &LU, queue );
        info = magma_zparilut_reorder( &LUT, queue );
        if (info != 0) {
            printf("%% error: breakdown in iteration :%d. fallback.\n\n", iters+1);
            info = 0;
            break;
        }
        
        LU.nnz = LU.nnz-num_rm;
        LUT.nnz = LUT.nnz-num_rm;   
        
        //now do a sweep
        printf("sweep..."); fflush(stdout);
        magma_zparict_sweep( &L, &LU, queue );
        magma_zparict_sweep( &L, &LU, queue );
        magma_zparilut_copy( LU, &LUCSR, queue );
        magma_zdiagcheck_cpu( LUCSR, queue );
        for (magma_int_t z = 0; z < num_rm; z++) {
            rm_loc[z] = LU.nnz+z;
            rm_locT[z] = LU.nnz+z;
            LU.val[ LU.nnz+z ] = MAGMA_Z_ZERO;
            LU.list[ LU.nnz+z ] = -1;
            LUT.val[ LU.nnz+z ] = MAGMA_Z_ZERO;
            LUT.list[ LU.nnz+z ] = -1;
        }
        printf("done.\n"); fflush(stdout);
    }
    /*
    
    for( magma_int_t iters =0; iters<precond->sweeps; iters++ ) {
        num_rm = num_rm_gl;
        magma_zparilut_set_approx_thrs( num_rm, &LU, 0, &thrs, queue );

        magma_zparilut_rm_thrs( &thrs, &num_rm, &LU, &LU_new, rm_loc, rowlock, queue );
        if( approx == 0 ){
            magma_zparilut_rm_thrs_U( &thrs, &num_rm, &LUT, &LU_new, rm_locT, rowlock, queue );
        }
        magma_zparilut_zero( &L, queue );
        magma_zparict_sweep( &L, &LU, queue );
        
        if( approx == 0){
            magma_zmilu0_candidates( L, LU, LUT, &LU_new, queue );
        } else {
            magma_zparict_candidates( LU, &LU_new, queue );  
        }
         
        magma_zparict_residuals( L, LU, &LU_new, queue );
        
        if( approx == 0){
            magma_zmeliminate_duplicates( num_rm, &LU_new, queue );
            magma_zparilut_insert_LU( num_rm, rm_loc, rm_locT, &LU_new, &LU, &LUT, queue );
        } else {
            magma_zparict_insert( tri, num_rm, rm_loc, &LU_new, &LU, rowlock, queue );  
        }
        
        magma_zparilut_reorder( &LUT, queue );
        info = magma_zparilut_reorder( &LU, queue );
        // workaround to avoid breakdown
        if( info !=0 ){
            printf("%% error: breakdown in iteration :%d. fallback.\n\n", iters+1);
            info = 0;
            break;
        }
        magma_zparilut_copy( LU, & LUCSR, queue );
        // end workaround
        
        magma_zparict_sweep( &L, &LU, queue );
        printf("%% removed elements:%d\n", num_rm);
    }
    end = magma_sync_wtime( queue ); printf("%% >> preconditioner generation: %.4e\n", end-start);
    
    */
    printf("done2.\n"); fflush(stdout);
    magma_zdiagcheck_cpu( LUCSR, queue );
    // for CUSPARSE
    CHECK( magma_zmtransfer( LUCSR, &precond->M, Magma_CPU, Magma_DEV , queue ));
    printf("done3.\n"); fflush(stdout);
    magma_zdiagcheck( precond->M, queue );
        // copy the matrix to precond->L and (transposed) to precond->U
    CHECK( magma_zmtransfer(precond->M, &(precond->L), Magma_DEV, Magma_DEV, queue ));
    CHECK( magma_zmtranspose( precond->L, &(precond->U), queue ));
    magma_zdiagcheck( precond->L, queue );
    magma_zdiagcheck( precond->U, queue );
    printf("done34.\n"); fflush(stdout);
    // extract the diagonal of L into precond->d
    CHECK( magma_zjacobisetup_diagscal( precond->L, &precond->d, queue ));
    CHECK( magma_zvinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, queue ));
printf("done35.\n"); fflush(stdout);
    // extract the diagonal of U into precond->d2
    CHECK( magma_zjacobisetup_diagscal( precond->U, &precond->d2, queue ));
    CHECK( magma_zvinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, queue ));

printf("done4.\n"); fflush(stdout);
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
printf("done5.\n"); fflush(stdout);
    
    cleanup:
        
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
    magma_zmfree( &U, queue );
    magma_zmfree( &LU, queue );
    magma_zmfree( &LUT, queue );
#endif
    return info;
}
    
