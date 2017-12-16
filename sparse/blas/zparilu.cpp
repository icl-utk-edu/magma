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



// This file is deprecated and will be removed in future.
// The ParILU/ParIC functionality is provided by 
// src/zparilu_gpu.cpp and src/zparic_gpu.cpp
// 

/**
    Purpose
    -------

    Prepares the ILU preconditioner via the iterative ILU iteration.

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
magma_zparilusetup(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrL=NULL;
    cusparseMatDescr_t descrU=NULL;

    magma_z_matrix hAh={Magma_CSR}, hA={Magma_CSR}, hL={Magma_CSR}, hU={Magma_CSR},
    hAcopy={Magma_CSR}, hAL={Magma_CSR}, hAU={Magma_CSR}, hAUt={Magma_CSR},
    hUT={Magma_CSR}, hAtmp={Magma_CSR}, hACSRCOO={Magma_CSR}, dAinitguess={Magma_CSR},
    dL={Magma_CSR}, dU={Magma_CSR};

    // copy original matrix as CSRCOO to device
    CHECK( magma_zmtransfer(A, &hAh, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_zmconvert( hAh, &hA, hAh.storage_type, Magma_CSR , queue ));
    magma_zmfree(&hAh, queue );

    CHECK( magma_zmtransfer( hA, &hAcopy, Magma_CPU, Magma_CPU , queue ));

    // in case using fill-in
    CHECK( magma_zsymbilu( &hAcopy, precond->levels, &hAL, &hAUt,  queue ));
    // add a unit diagonal to L for the algorithm
    CHECK( magma_zmLdiagadd( &hAL , queue ));
    // transpose U for the algorithm
    CHECK( magma_z_cucsrtranspose(  hAUt, &hAU , queue ));
    magma_zmfree( &hAUt , queue );

    // ---------------- initial guess ------------------- //
    CHECK( magma_zmconvert( hAcopy, &hACSRCOO, Magma_CSR, Magma_CSRCOO , queue ));
    CHECK( magma_zmtransfer( hACSRCOO, &dAinitguess, Magma_CPU, Magma_DEV , queue ));
    magma_zmfree(&hACSRCOO, queue );
    magma_zmfree(&hAcopy, queue );

    // transfer the factor L and U
    CHECK( magma_zmtransfer( hAL, &dL, Magma_CPU, Magma_DEV , queue ));
    CHECK( magma_zmtransfer( hAU, &dU, Magma_CPU, Magma_DEV , queue ));
    magma_zmfree(&hAL, queue );
    magma_zmfree(&hAU, queue );

    for(int i=0; i<precond->sweeps; i++){
        CHECK( magma_zparilu_csr( dAinitguess, dL, dU , queue ));
    }

    CHECK( magma_zmtransfer( dL, &hL, Magma_DEV, Magma_CPU , queue ));
    CHECK( magma_zmtransfer( dU, &hU, Magma_DEV, Magma_CPU , queue ));
    CHECK( magma_z_cucsrtranspose(  hU, &hUT , queue ));

    magma_zmfree(&dL, queue );
    magma_zmfree(&dU, queue );
    magma_zmfree(&hU, queue );
    CHECK( magma_zmlumerge( hL, hUT, &hAtmp, queue ));

    magma_zmfree(&hL, queue );
    magma_zmfree(&hUT, queue );

    CHECK( magma_zmtransfer( hAtmp, &precond->M, Magma_CPU, Magma_DEV , queue ));

    hAL.diagorder_type = Magma_UNITY;
    CHECK( magma_zmconvert(hAtmp, &hAL, Magma_CSR, Magma_CSRL, queue ));
    hAL.storage_type = Magma_CSR;
    CHECK( magma_zmconvert(hAtmp, &hAU, Magma_CSR, Magma_CSRU, queue ));
    hAU.storage_type = Magma_CSR;

    magma_zmfree(&hAtmp, queue );

    // CHECK( magma_zcsrsplit( 0, 256, hAL, &DL, &RL , queue ));
    // CHECK( magma_zcsrsplit( 0, 256, hAU, &DU, &RU , queue ));
    // 
    // CHECK( magma_zmtransfer( DL, &precond->LD, Magma_CPU, Magma_DEV , queue ));
    // CHECK( magma_zmtransfer( DU, &precond->UD, Magma_CPU, Magma_DEV , queue ));

    // for cusparse uncomment this
    CHECK( magma_zmtransfer( hAL, &precond->L, Magma_CPU, Magma_DEV , queue ));
    CHECK( magma_zmtransfer( hAU, &precond->U, Magma_CPU, Magma_DEV , queue ));
    
/*

    //-- for ba-solve uncomment this

    if( RL.nnz != 0 )
        CHECK( magma_zmtransfer( RL, &precond->L, Magma_CPU, Magma_DEV , queue ));
    else {
        precond->L.nnz = 0;
        precond->L.val = NULL;
        precond->L.col = NULL;
        precond->L.row = NULL;
        precond->L.blockinfo = NULL;
    }

    if( RU.nnz != 0 )
        CHECK( magma_zmtransfer( RU, &precond->U, Magma_CPU, Magma_DEV , queue ));
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
    CHECK( magma_zjacobisetup_diagscal( precond->L, &precond->d, queue ));
    CHECK( magma_zvinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, queue ));
    
    // extract the diagonal of U into precond->d2
    CHECK( magma_zjacobisetup_diagscal( precond->U, &precond->d2, queue ));
    CHECK( magma_zvinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, queue ));

    magma_zmfree(&hAL, queue );
    magma_zmfree(&hAU, queue );
    // magma_zmfree(&DL, queue );
    // magma_zmfree(&RL, queue );
    // magma_zmfree(&DU, queue );
    // magma_zmfree(&RU, queue );

    // CUSPARSE context //
    CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrL ));
    CHECK_CUSPARSE( cusparseSetMatType( descrL, CUSPARSE_MATRIX_TYPE_TRIANGULAR ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrL, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrL, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE( cusparseSetMatFillMode( descrL, CUSPARSE_FILL_MODE_LOWER ));
    CHECK_CUSPARSE( cusparseCreateSolveAnalysisInfo( &precond->cuinfoL ));
    CHECK_CUSPARSE( cusparseZcsrsv_analysis( cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->L.num_rows,
        precond->L.nnz, descrL,
        precond->L.val, precond->L.row, precond->L.col, precond->cuinfoL ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrU ));
    CHECK_CUSPARSE( cusparseSetMatType( descrU, CUSPARSE_MATRIX_TYPE_TRIANGULAR ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrU, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrU, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE( cusparseSetMatFillMode( descrU, CUSPARSE_FILL_MODE_UPPER ));
    CHECK_CUSPARSE( cusparseCreateSolveAnalysisInfo( &precond->cuinfoU ));
    CHECK_CUSPARSE( cusparseZcsrsv_analysis( cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->U.num_rows,
        precond->U.nnz, descrU,
        precond->U.val, precond->U.row, precond->U.col, precond->cuinfoU ));
    
    
cleanup:
    cusparseDestroy( cusparseHandle );
    cusparseDestroyMatDescr( descrL );
    cusparseDestroyMatDescr( descrU );
    cusparseHandle=NULL;
    descrL=NULL;
    descrU=NULL;
    magma_zmfree( &hAh, queue );
    magma_zmfree( &hA, queue );
    magma_zmfree( &hL, queue );
    magma_zmfree( &hU, queue );
    magma_zmfree( &hAcopy, queue );
    magma_zmfree( &hAL, queue );
    magma_zmfree( &hAU, queue );
    magma_zmfree( &hAUt, queue );
    magma_zmfree( &hUT, queue );
    magma_zmfree( &hAtmp, queue );
    magma_zmfree( &hACSRCOO, queue );
    magma_zmfree( &dAinitguess, queue );
    magma_zmfree( &dL, queue );
    magma_zmfree( &dU, queue );
    // magma_zmfree( &DL, queue );
    // magma_zmfree( &DU, queue );
    // magma_zmfree( &RL, queue );
    // magma_zmfree( &RU, queue );

    return info;
}



/**
    Purpose
    -------

    Updates an existing preconditioner via additional iterative ILU sweeps for
    previous factorization initial guess (PFIG).
    See  Anzt et al., Parallel Computing, 2015.

    Arguments
    ---------
    
    @param[in]
    A           magma_z_matrix
                input matrix A, current target system

    @param[in]
    precond     magma_z_preconditioner*
                preconditioner parameters

    @param[in]
    updates     magma_int_t 
                number of updates
    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_zhepr
    ********************************************************************/
extern "C"
magma_int_t
magma_zpariluupdate(
    magma_z_matrix A,
    magma_z_preconditioner *precond,
    magma_int_t updates,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_z_matrix hALt={Magma_CSR};
    magma_z_matrix d_h={Magma_CSR};
    
    magma_z_matrix hL={Magma_CSR}, hU={Magma_CSR},
    hAcopy={Magma_CSR}, hAL={Magma_CSR}, hAU={Magma_CSR}, hAUt={Magma_CSR},
    hUT={Magma_CSR}, hAtmp={Magma_CSR},
    dL={Magma_CSR}, dU={Magma_CSR};

    if ( updates > 0 ){
        CHECK( magma_zmtransfer( precond->M, &hAcopy, Magma_DEV, Magma_CPU , queue ));
        // in case using fill-in
        CHECK( magma_zsymbilu( &hAcopy, precond->levels, &hAL, &hAUt,  queue ));
        // add a unit diagonal to L for the algorithm
        CHECK( magma_zmLdiagadd( &hAL , queue ));
        // transpose U for the algorithm
        CHECK( magma_z_cucsrtranspose(  hAUt, &hAU , queue ));
        // transfer the factor L and U
        CHECK( magma_zmtransfer( hAL, &dL, Magma_CPU, Magma_DEV , queue ));
        CHECK( magma_zmtransfer( hAU, &dU, Magma_CPU, Magma_DEV , queue ));
        magma_zmfree(&hAL, queue );
        magma_zmfree(&hAU, queue );
        magma_zmfree(&hAUt, queue );
        magma_zmfree(&precond->M, queue );
        magma_zmfree(&hAcopy, queue );
        
        // copy original matrix as CSRCOO to device
        for(int i=0; i<updates; i++){
            CHECK( magma_zparilu_csr( A, dL, dU, queue ));
        }
        CHECK( magma_zmtransfer( dL, &hL, Magma_DEV, Magma_CPU , queue ));
        CHECK( magma_zmtransfer( dU, &hU, Magma_DEV, Magma_CPU , queue ));
        CHECK( magma_z_cucsrtranspose(  hU, &hUT , queue ));
        magma_zmfree(&dL, queue );
        magma_zmfree(&dU, queue );
        magma_zmfree(&hU, queue );
        CHECK( magma_zmlumerge( hL, hUT, &hAtmp, queue ));
        // for CUSPARSE
        CHECK( magma_zmtransfer( hAtmp, &precond->M, Magma_CPU, Magma_DEV , queue ));
        
        magma_zmfree(&hL, queue );
        magma_zmfree(&hUT, queue );
        hAL.diagorder_type = Magma_UNITY;
        CHECK( magma_zmconvert(hAtmp, &hAL, Magma_CSR, Magma_CSRL, queue ));
        hAL.storage_type = Magma_CSR;
        CHECK( magma_zmconvert(hAtmp, &hAU, Magma_CSR, Magma_CSRU, queue ));
        hAU.storage_type = Magma_CSR;
        
        magma_zmfree(&hAtmp, queue );
        CHECK( magma_zmtransfer( hAL, &precond->L, Magma_CPU, Magma_DEV , queue ));
        CHECK( magma_zmtransfer( hAU, &precond->U, Magma_CPU, Magma_DEV , queue ));
        magma_zmfree(&hAL, queue );
        magma_zmfree(&hAU, queue );
    
        magma_zmfree( &precond->d , queue );
        magma_zmfree( &precond->d2 , queue );
        
        CHECK( magma_zjacobisetup_diagscal( precond->L, &precond->d, queue ));
        CHECK( magma_zjacobisetup_diagscal( precond->U, &precond->d2, queue ));
    }

cleanup:
    magma_zmfree(&d_h, queue );
    magma_zmfree(&hALt, queue );
    
    return info;
}


/**
    Purpose
    -------

    Prepares the IC preconditioner via the iterative IC iteration.

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

    @ingroup magmasparse_zhepr
    ********************************************************************/
extern "C"
magma_int_t
magma_zparicsetup(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrL=NULL;
    cusparseMatDescr_t descrU=NULL;

    magma_z_matrix hAh={Magma_CSR}, hA={Magma_CSR}, hAtmp={Magma_CSR},
    hAL={Magma_CSR}, hAUt={Magma_CSR}, hALt={Magma_CSR}, hM={Magma_CSR},
    hACSRCOO={Magma_CSR}, dAinitguess={Magma_CSR}, dL={Magma_CSR};
    magma_z_matrix d_h={Magma_CSR};


    // copy original matrix as CSRCOO to device
    CHECK( magma_zmtransfer(A, &hAh, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_zmconvert( hAh, &hA, hAh.storage_type, Magma_CSR , queue ));
    magma_zmfree(&hAh, queue );

    // in case using fill-in
    CHECK( magma_zsymbilu( &hA, precond->levels, &hAL, &hAUt , queue ));

    // need only lower triangular
    magma_zmfree(&hAUt, queue );
    magma_zmfree(&hAL, queue );
    CHECK( magma_zmconvert( hA, &hAtmp, Magma_CSR, Magma_CSRL , queue ));
    magma_zmfree(&hA, queue );

    // ---------------- initial guess ------------------- //
    CHECK( magma_zmconvert( hAtmp, &hACSRCOO, Magma_CSR, Magma_CSRCOO , queue ));
    //int blocksize = 1;
    //magma_zmreorder( hACSRCOO, n, blocksize, blocksize, blocksize, &hAinitguess , queue );
    CHECK( magma_zmtransfer( hACSRCOO, &dAinitguess, Magma_CPU, Magma_DEV , queue ));
    magma_zmfree(&hACSRCOO, queue );
    CHECK( magma_zmtransfer( hAtmp, &dL, Magma_CPU, Magma_DEV , queue ));
    magma_zmfree(&hAtmp, queue );

    for(int i=0; i<precond->sweeps; i++){
        CHECK( magma_zparic_csr( dAinitguess, dL , queue ));
    }
    CHECK( magma_zmtransfer( dL, &hAL, Magma_DEV, Magma_CPU , queue ));
    magma_zmfree(&dL, queue );
    magma_zmfree(&dAinitguess, queue );


    // for CUSPARSE
    CHECK( magma_zmtransfer( hAL, &precond->M, Magma_CPU, Magma_DEV , queue ));

    // Jacobi setup
    CHECK( magma_zjacobisetup_matrix( precond->M, &precond->L, &precond->d , queue ));

    // for Jacobi, we also need U
    CHECK( magma_z_cucsrtranspose(   hAL, &hALt , queue ));
    CHECK( magma_zjacobisetup_matrix( hALt, &hM, &d_h , queue ));

    CHECK( magma_zmtransfer( hM, &precond->U, Magma_CPU, Magma_DEV , queue ));

    magma_zmfree(&hM, queue );

    magma_zmfree(&d_h, queue );


        // copy the matrix to precond->L and (transposed) to precond->U
    CHECK( magma_zmtransfer(precond->M, &(precond->L), Magma_DEV, Magma_DEV, queue ));
    CHECK( magma_zmtranspose( precond->L, &(precond->U), queue ));

    // extract the diagonal of L into precond->d
    CHECK( magma_zjacobisetup_diagscal( precond->L, &precond->d, queue ));
    CHECK( magma_zvinit( &precond->work1, Magma_DEV, hAL.num_rows, 1, MAGMA_Z_ZERO, queue ));

    // extract the diagonal of U into precond->d2
    CHECK( magma_zjacobisetup_diagscal( precond->U, &precond->d2, queue ));
    CHECK( magma_zvinit( &precond->work2, Magma_DEV, hAL.num_rows, 1, MAGMA_Z_ZERO, queue ));


    magma_zmfree(&hAL, queue );
    magma_zmfree(&hALt, queue );


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
    cusparseDestroy( cusparseHandle );
    cusparseDestroyMatDescr( descrL );
    cusparseDestroyMatDescr( descrU );
    cusparseHandle=NULL;
    descrL=NULL;
    descrU=NULL;    
    magma_zmfree( &hAh, queue );
    magma_zmfree( &hA, queue );
    magma_zmfree( &hAtmp, queue );
    magma_zmfree( &hAL, queue );
    magma_zmfree( &hAUt, queue );
    magma_zmfree( &hALt, queue );
    magma_zmfree( &hM, queue );
    magma_zmfree( &hACSRCOO, queue );
    magma_zmfree( &dAinitguess, queue );
    magma_zmfree( &dL, queue );
    magma_zmfree( &d_h, queue );
    
    return info;
}


/**
    Purpose
    -------

    Updates an existing preconditioner via additional iterative IC sweeps for
    previous factorization initial guess (PFIG).
    See  Anzt et al., Parallel Computing, 2015.

    Arguments
    ---------
    
    @param[in]
    A           magma_z_matrix
                input matrix A, current target system

    @param[in]
    precond     magma_z_preconditioner*
                preconditioner parameters

    @param[in]
    updates     magma_int_t 
                number of updates
    
    @param[in]
    queue       magma_queue_t
                Queue to execute in.
                
    @ingroup magmasparse_zhepr
    ********************************************************************/
extern "C"
magma_int_t
magma_zparicupdate(
    magma_z_matrix A,
    magma_z_preconditioner *precond,
    magma_int_t updates,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_z_matrix hALt={Magma_CSR};
    magma_z_matrix d_h={Magma_CSR};
        
    if( updates > 0 ){
        // copy original matrix as CSRCOO to device
        for(int i=0; i<updates; i++){
            CHECK( magma_zparic_csr( A, precond->M , queue ));
        }
        //magma_zmtransfer( precond->M, &hALt, Magma_DEV, Magma_CPU , queue );
        magma_zmfree(&precond->L, queue );
        magma_zmfree(&precond->U, queue );
        magma_zmfree( &precond->d , queue );
        magma_zmfree( &precond->d2 , queue );
        
        // copy the matrix to precond->L and (transposed) to precond->U
        CHECK( magma_zmtransfer(precond->M, &(precond->L), Magma_DEV, Magma_DEV, queue ));
        CHECK( magma_zmtranspose( precond->L, &(precond->U), queue ));

        CHECK( magma_zjacobisetup_diagscal( precond->L, &precond->d, queue ));
        CHECK( magma_zjacobisetup_diagscal( precond->U, &precond->d2, queue ));
    }
    
cleanup:
    magma_zmfree(&d_h, queue );
    magma_zmfree(&hALt, queue );
    
    return info;
}

