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
#include <cuda.h>  // for CUDA_VERSION

#include "magma_trisolve.h"

#define PRECISION_z

/* For hipSPARSE, they use a separate complex type than for hipBLAS */
#if defined(MAGMA_HAVE_HIP)
  #ifdef PRECISION_z
    #define hipblasDoubleComplex hipDoubleComplex
  #elif defined(PRECISION_c)
    #define hipblasComplex hipComplex
  #endif
#endif

#if CUDA_VERSION >= 12000
   #define cusparseCreateCsrsm2Info(info)
   #define cusparseDestroyCsrsm2Info(info)
#endif

// todo: make it spacific
#if CUDA_VERSION >= 11000 || defined(MAGMA_HAVE_HIP)
#define cusparseCreateSolveAnalysisInfo(info) cusparseCreateCsrsm2Info(info) 
#else
#define cusparseCreateSolveAnalysisInfo(info)                                                   \
        CHECK_CUSPARSE( cusparseCreateSolveAnalysisInfo( info ))
#endif

#if CUDA_VERSION >= 11000 || defined(MAGMA_HAVE_HIP)
#define cusparseDestroySolveAnalysisInfo(info) cusparseDestroyCsrsm2Info(info)
#endif

// todo: check the info and linfo if we have to give it back; free memory? 
#if CUDA_VERSION >= 11000
#define cusparseZcsrsm_analysis(handle, op, rows, nnz, descrA, dval, drow, dcol, info )         \
    {                                                                                           \
        magmaDoubleComplex alpha = MAGMA_Z_ONE;                                                 \
        cuDoubleComplex *B;                                                                     \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_GENERAL );                             \
        cusparseZcsrsm2_bufferSizeExt(handle, 0, op, CUSPARSE_OPERATION_NON_TRANSPOSE,          \
                                      rows, 1, nnz, (const cuDoubleComplex *)&alpha,            \
                                      descrA, dval, drow, dcol,                                 \
                                      B, rows, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, &bufsize); \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        cusparseZcsrsm2_analysis(handle, 0, op, CUSPARSE_OPERATION_NON_TRANSPOSE,               \
                                 rows, 1, nnz, (const cuDoubleComplex *)&alpha,                 \
                                 descrA, dval, drow, dcol,                                      \
                                 B, rows, info, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buf);           \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }

#elif defined(MAGMA_HAVE_HIP)
#define cusparseZcsrsm_analysis(handle, op, rows, nnz, descrA, dval, drow, dcol, info )         \
    {                                                                                           \
        magmaDoubleComplex alpha = MAGMA_Z_ONE;                                                 \
        hipDoubleComplex *B;                                                                     \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        hipsparseZcsrsm2_bufferSizeExt(handle, 0, op, HIPSPARSE_OPERATION_NON_TRANSPOSE,         \
                                      rows, 1, nnz, (const hipDoubleComplex *)&alpha,           \
                                      descrA, (const hipDoubleComplex *)dval, (const int *)drow, (const int *)dcol,  \
                                      (const hipDoubleComplex *)B, rows, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, &bufsize); \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        hipsparseZcsrsm2_analysis(handle, 0, op, HIPSPARSE_OPERATION_NON_TRANSPOSE,               \
                                 rows, 1, nnz, (const hipDoubleComplex *)&alpha,                 \
                                 descrA, (const hipDoubleComplex *)dval, drow, dcol,            \
                                 B, rows, info, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);           \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }



#endif

#if CUDA_VERSION >= 11000
#define cusparseZcsr2csc(handle, cols, rows, nnz, dval, drow, dcol, prdval, prdcol, prdrow,     \
                         action, base)                                                          \
    {                                                                                           \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        cusparseCsr2cscEx2_bufferSize(handle, cols, rows, nnz, dval, drow, dcol, prdval,        \
                                      prdcol, prdrow, CUDA_C_64F, action, base,                 \
                                      CUSPARSE_CSR2CSC_ALG1, &bufsize);                         \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        cusparseCsr2cscEx2(handle, cols, rows, nnz, dval, drow, dcol, prdval, prdcol, prdrow,   \
                           CUDA_C_64F, action, base, CUSPARSE_CSR2CSC_ALG1, buf);               \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }
#endif

// todo: info is passed from analysis; to change info with this linfo & remove linfo from here
#if CUDA_VERSION >= 11000
#define cusparseZcsric0(handle, op, rows, nnz, descrA, dval, drow, dcol, info )                 \
    {                                                                                           \
        int bufsize;                                                                            \
        void *buf;                                                                              \
        csric02Info_t linfo;                                                                    \
        cusparseCreateCsric02Info(&linfo);                                                      \
        cusparseZcsric02_bufferSize(handle, rows, nnz, descrA, dval, drow, dcol,linfo,&bufsize);\
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        cusparseZcsric02_analysis(handle, rows, nnz, descrA, dval, drow, dcol, linfo,           \
                                  CUSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                         \
        int numerical_zero;                                                                     \
        if (CUSPARSE_STATUS_ZERO_PIVOT ==                                                       \
            cusparseXcsric02_zeroPivot( handle, linfo, &numerical_zero ))                       \
            printf("A(%d,%d) is missing\n", numerical_zero, numerical_zero);                    \
        cusparseZcsric02(handle, rows, nnz, descrA, dval, drow, dcol, linfo,                    \
                         CUSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                                  \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }
#elif defined(MAGMA_HAVE_HIP)
#define cusparseZcsric0(handle, op, rows, nnz, descrA, dval, drow, dcol, info )                 \
    {                                                                                           \
        int bufsize;                                                                            \
        void *buf;                                                                              \
        csric02Info_t linfo;                                                                    \
        hipsparseCreateCsric02Info(&linfo);                                                      \
        hipsparseZcsric02_bufferSize(handle, rows, nnz, descrA, (hipDoubleComplex*)dval, drow, dcol,linfo,&bufsize);\
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        hipsparseZcsric02_analysis(handle, rows, nnz, descrA, (hipDoubleComplex*)dval, drow, dcol, linfo,           \
                                  HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                         \
        int numerical_zero;                                                                     \
        if (HIPSPARSE_STATUS_ZERO_PIVOT ==                                                       \
            hipsparseXcsric02_zeroPivot( handle, linfo, &numerical_zero ))                       \
            printf("A(%d,%d) is missing\n", numerical_zero, numerical_zero);                    \
        hipsparseZcsric02(handle, rows, nnz, descrA, (hipDoubleComplex*)dval, drow, dcol, linfo,                    \
                         HIPSPARSE_SOLVE_POLICY_NO_LEVEL, buf);                                  \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    } 
#else
#define cusparseZcsric0(handle, op, rows, nnz, descrA, dval, drow, dcol, info )                 \
    CHECK_CUSPARSE( cusparseZcsric0(handle, op, rows, descrA, dval, drow, dcol, info ))
#endif

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
    magma_int_t info = 0;
    
    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrA=NULL;
#if CUDA_VERSION >= 7000 || defined(MAGMA_HAVE_HIP)
    csrilu02Info_t info_M=NULL;
    void *pBuffer = NULL;
#endif
    
    // magma_zprint_matrix(A, queue );
    // copy matrix into preconditioner parameter
    magma_z_matrix hA={Magma_CSR}, hACSR={Magma_CSR};
    magma_z_matrix hL={Magma_CSR}, hU={Magma_CSR};
    CHECK( magma_zmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_zmconvert( hA, &hACSR, hA.storage_type, Magma_CSR, queue ));

    // in case using fill-in
    if( precond->levels > 0 ){
        magma_z_matrix hAL={Magma_CSR}, hAUt={Magma_CSR};
        CHECK( magma_zsymbilu( &hACSR, precond->levels, &hAL, &hAUt,  queue ));
        magma_zmfree(&hAL, queue);
        magma_zmfree(&hAUt, queue);
    }

    CHECK( magma_zmtransfer(hACSR, &(precond->M), Magma_CPU, Magma_DEV, queue ));

    magma_zmfree( &hA, queue );
    magma_zmfree( &hACSR, queue );

    // CUSPARSE context //
    CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
    CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrA ));
    CHECK_CUSPARSE( cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_GENERAL ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrA, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrA, CUSPARSE_INDEX_BASE_ZERO ));
    cusparseCreateSolveAnalysisInfo( &(precond->cuinfoILU) );

    // use kernel to manually check for zeros n the diagonal
    CHECK( magma_zdiagcheck( precond->M, queue ) );
    
#if CUDA_VERSION >= 7000 
    // this version has the bug fixed where a zero on the diagonal causes a crash
    CHECK_CUSPARSE( cusparseCreateCsrilu02Info(&info_M) );
    int buffersize;
    int structural_zero;
    int numerical_zero;
    
    CHECK_CUSPARSE(
    cusparseZcsrilu02_bufferSize( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M,
                         &buffersize ) );
    
    CHECK( magma_malloc((void**)&pBuffer, buffersize) );

    CHECK_CUSPARSE( cusparseZcsrilu02_analysis( cusparseHandle,
            precond->M.num_rows, precond->M.nnz, descrA,
            precond->M.dval, precond->M.drow, precond->M.dcol,
            info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer ));
    
    CHECK_CUSPARSE( cusparseXcsrilu02_zeroPivot( cusparseHandle, info_M, &numerical_zero ) );
    CHECK_CUSPARSE( cusparseXcsrilu02_zeroPivot( cusparseHandle, info_M, &structural_zero ) );
    
    CHECK_CUSPARSE(
    cusparseZcsrilu02( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer) );

#elif defined(MAGMA_HAVE_HIP)

    // this version has the bug fixed where a zero on the diagonal causes a crash
    CHECK_CUSPARSE( hipsparseCreateCsrilu02Info(&info_M) );
    int buffersize;
    int structural_zero;
    int numerical_zero;
    
    CHECK_CUSPARSE(
    hipsparseZcsrilu02_bufferSize( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         (hipDoubleComplex*)precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M,
                         &buffersize ) );
    
    CHECK( magma_malloc((void**)&pBuffer, buffersize) );

    CHECK_CUSPARSE( hipsparseZcsrilu02_analysis( cusparseHandle,
            precond->M.num_rows, precond->M.nnz, descrA,
            (hipDoubleComplex*)precond->M.dval, precond->M.drow, precond->M.dcol,
            info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer ));
    
    CHECK_CUSPARSE( hipsparseXcsrilu02_zeroPivot( cusparseHandle, info_M, &numerical_zero ) );
    CHECK_CUSPARSE( hipsparseXcsrilu02_zeroPivot( cusparseHandle, info_M, &structural_zero ) );
    
    CHECK_CUSPARSE(
    hipsparseZcsrilu02( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         (hipDoubleComplex*)precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M, HIPSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer) );

#else
    // this version contains the bug but is needed for backward compability
    cusparseZcsrsm_analysis( cusparseHandle,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             precond->M.num_rows, precond->M.nnz, descrA,
                             precond->M.dval, precond->M.drow, precond->M.dcol,
                             precond->cuinfoILU );
    CHECK_CUSPARSE( cusparseZcsrilu0( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                      precond->M.num_rows, descrA,
                      precond->M.dval,
                      precond->M.drow,
                      precond->M.dcol,
                      precond->cuinfoILU ));
#endif

    CHECK( magma_zmtransfer( precond->M, &hA, Magma_DEV, Magma_CPU, queue ));

    hL.diagorder_type = Magma_UNITY;
    CHECK( magma_zmconvert( hA, &hL , Magma_CSR, Magma_CSRL, queue ));
    hU.diagorder_type = Magma_VALUE;
    CHECK( magma_zmconvert( hA, &hU , Magma_CSR, Magma_CSRU, queue ));
    CHECK( magma_zmtransfer( hL, &(precond->L), Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_zmtransfer( hU, &(precond->U), Magma_CPU, Magma_DEV, queue ));
    
    // malloc aux space for sync-free sptrsv 
    CHECK( magma_index_malloc( &(precond->L_dgraphindegree), precond->M.num_rows ));
    CHECK( magma_index_malloc( &(precond->L_dgraphindegree_bak), precond->M.num_rows ));
    CHECK( magma_index_malloc( &(precond->U_dgraphindegree), precond->M.num_rows ));
    CHECK( magma_index_malloc( &(precond->U_dgraphindegree_bak), precond->M.num_rows ));

    if( precond->trisolver == Magma_CUSOLVE || precond->trisolver == 0 ){
        CHECK(magma_ztrisolve_analysis(precond->L, &precond->cuinfoL, false, false, false, queue));
        CHECK(magma_ztrisolve_analysis(precond->U, &precond->cuinfoU, true, false, false, queue));
    } else if( precond->trisolver == Magma_SYNCFREESOLVE ){
            magma_zmfree(&hL, queue );
            magma_zmfree(&hU, queue );
            magma_zmtransfer( precond->L, &hL, Magma_DEV, Magma_DEV, queue );
            // conversion using CUSPARSE
            #ifdef MAGMA_HAVE_HIP
            hipsparseZcsr2csc(cusparseHandle, hL.num_cols, 
                             hL.num_rows, hL.nnz,
                             (hipDoubleComplex*)hL.dval, hL.drow, hL.dcol, 
                             (hipDoubleComplex*)precond->L.dval, precond->L.dcol, precond->L.drow,
                             CUSPARSE_ACTION_NUMERIC,
                             CUSPARSE_INDEX_BASE_ZERO);
            #else
            cusparseZcsr2csc(cusparseHandle, hL.num_cols, 
                             hL.num_rows, hL.nnz,
                             hL.dval, hL.drow, hL.dcol, 
                             precond->L.dval, precond->L.dcol, precond->L.drow,
                             CUSPARSE_ACTION_NUMERIC,
                             CUSPARSE_INDEX_BASE_ZERO);

            #endif

            magma_zmtransfer( precond->U, &hU, Magma_DEV, Magma_DEV, queue );
            // conversion using CUSPARSE

            #ifdef MAGMA_HAVE_HIP
            hipsparseZcsr2csc(cusparseHandle, hU.num_cols, 
                             hU.num_rows, hU.nnz,
                             (hipDoubleComplex*)hU.dval, hU.drow, hU.dcol, 
                             (hipDoubleComplex*)precond->U.dval, precond->U.dcol, precond->U.drow,
                             CUSPARSE_ACTION_NUMERIC,
                             CUSPARSE_INDEX_BASE_ZERO);
            #else
            cusparseZcsr2csc(cusparseHandle, hU.num_cols, 
                             hU.num_rows, hU.nnz,
                             hU.dval, hU.drow, hU.dcol, 
                             precond->U.dval, precond->U.dcol, precond->U.drow,
                             CUSPARSE_ACTION_NUMERIC,
                             CUSPARSE_INDEX_BASE_ZERO);
            #endif

            // set this to be CSC
            precond->U.storage_type = Magma_CSC;
            precond->L.storage_type = Magma_CSC;
            
            // analysis sparsity structures of L and U
            magma_zgecscsyncfreetrsm_analysis(precond->L.num_rows, 
                precond->L.nnz, precond->L.dval, 
                precond->L.drow, precond->L.dcol, 
                precond->L_dgraphindegree, precond->L_dgraphindegree_bak, 
                queue);
            magma_zgecscsyncfreetrsm_analysis(precond->U.num_rows, 
                precond->U.nnz, precond->U.dval, 
                precond->U.drow, precond->U.dcol, 
                precond->U_dgraphindegree, precond->U_dgraphindegree_bak, 
                queue);

            magma_zmfree(&hL, queue );
            magma_zmfree(&hU, queue );
    } else {
        //prepare for iterative solves
        
        // extract the diagonal of L into precond->d
        CHECK( magma_zjacobisetup_diagscal( precond->L, &precond->d, queue ));
        // precond->d.memory_location = Magma_DEV;
        CHECK( magma_zvinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, queue ));
        
        // extract the diagonal of U into precond->d2
        CHECK( magma_zjacobisetup_diagscal( precond->U, &precond->d2, queue ));
        // precond->d2.memory_location = Magma_DEV;
        CHECK( magma_zvinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, queue ));
    }

    
cleanup:
#if CUDA_VERSION >= 7000 || defined(MAGMA_HAVE_HIP)
    magma_free( pBuffer );
    cusparseDestroyCsrilu02Info( info_M );
#endif
    cusparseDestroySolveAnalysisInfo( precond->cuinfoILU );
    cusparseDestroyMatDescr( descrA );
    cusparseDestroy( cusparseHandle );
    magma_zmfree( &hA, queue );
    magma_zmfree( &hACSR, queue );
    magma_zmfree(&hA, queue );
    magma_zmfree(&hL, queue );
    magma_zmfree(&hU, queue );

    return info;
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
    magma_int_t info = 0;
    magma_z_matrix Ah1={Magma_CSR}, Ah2={Magma_CSR};

    // transpose the matrix
    magma_zmtransfer( precond->L, &Ah1, Magma_DEV, Magma_CPU, queue );
    magma_zmconvert( Ah1, &Ah2, A.storage_type, Magma_CSR, queue );
    magma_zmfree(&Ah1, queue );
    magma_zmtransposeconjugate( Ah2, &Ah1, queue );
    magma_zmfree(&Ah2, queue );
    Ah2.blocksize = A.blocksize;
    Ah2.alignment = A.alignment;
    magma_zmconvert( Ah1, &Ah2, Magma_CSR, A.storage_type, queue );
    magma_zmfree(&Ah1, queue );
    magma_zmtransfer( Ah2, &(precond->LT), Magma_CPU, Magma_DEV, queue );
    magma_zmfree(&Ah2, queue );
    
    magma_zmtransfer( precond->U, &Ah1, Magma_DEV, Magma_CPU, queue );
    magma_zmconvert( Ah1, &Ah2, A.storage_type, Magma_CSR, queue );
    magma_zmfree(&Ah1, queue );
    magma_zmtransposeconjugate( Ah2, &Ah1, queue );
    magma_zmfree(&Ah2, queue );
    Ah2.blocksize = A.blocksize;
    Ah2.alignment = A.alignment;
    magma_zmconvert( Ah1, &Ah2, Magma_CSR, A.storage_type, queue );
    magma_zmfree(&Ah1, queue );
    magma_zmtransfer( Ah2, &(precond->UT), Magma_CPU, Magma_DEV, queue );
    magma_zmfree(&Ah2, queue );
   
    CHECK(magma_ztrisolve_analysis(precond->LT, &precond->cuinfoLT, true, false, false, queue));
    CHECK(magma_ztrisolve_analysis(precond->UT, &precond->cuinfoUT, false, false, false, queue));

cleanup:
    magma_zmfree(&Ah1, queue );
    magma_zmfree(&Ah2, queue );

    return info;
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
    magma_int_t info = 0;
    
    magma_z_matrix hA={Magma_CSR}, hL={Magma_CSR}, hU={Magma_CSR};
    
    if (precond->L.memory_location != Magma_DEV ){
        CHECK( magma_zmtransfer( precond->M, &hA,
        precond->M.memory_location, Magma_CPU, queue ));

        hL.diagorder_type = Magma_UNITY;
        CHECK( magma_zmconvert( hA, &hL , Magma_CSR, Magma_CSRL, queue ));
        hU.diagorder_type = Magma_VALUE;
        CHECK( magma_zmconvert( hA, &hU , Magma_CSR, Magma_CSRU, queue ));
        CHECK( magma_zmtransfer( hL, &(precond->L), Magma_CPU, Magma_DEV, queue ));
        CHECK( magma_zmtransfer( hU, &(precond->U), Magma_CPU, Magma_DEV, queue ));
        
        magma_zmfree(&hA, queue );
        magma_zmfree(&hL, queue );
        magma_zmfree(&hU, queue );
    }
    
    CHECK(magma_ztrisolve_analysis(precond->L, &precond->cuinfoL, false, false, false, queue));
    CHECK(magma_ztrisolve_analysis(precond->U, &precond->cuinfoU, true, false, false, queue));
    
    if( precond->trisolver != 0 && precond->trisolver != Magma_CUSOLVE ){
        //prepare for iterative solves

        // extract the diagonal of L into precond->d
        CHECK( magma_zjacobisetup_diagscal( precond->L, &precond->d, queue ));
        CHECK( magma_zvinit( &precond->work1, Magma_DEV, precond->U.num_rows, 1, MAGMA_Z_ZERO, queue ));
        
        // extract the diagonal of U into precond->d2
        CHECK( magma_zjacobisetup_diagscal( precond->U, &precond->d2, queue ));
        CHECK( magma_zvinit( &precond->work2, Magma_DEV, precond->U.num_rows, 1, MAGMA_Z_ZERO, queue ));
    }
    
cleanup:     
    return info;
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
    magma_int_t info = 0;
        
    magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0);

    // CUSPARSE context //
    if( precond->trisolver == Magma_CUSOLVE || precond->trisolver == 0 ){
        CHECK(magma_ztrisolve(precond->L, precond->cuinfoL, false, false, false, b, *x, queue));
    } else if( precond->trisolver == Magma_SYNCFREESOLVE ){
        magma_zgecscsyncfreetrsm_solve( precond->L.num_rows,
            precond->L.nnz, 
            precond->L.dval, precond->L.drow, precond->L.dcol, 
            precond->L_dgraphindegree, precond->L_dgraphindegree_bak, 
            x->dval, b.dval, 0, //MAGMA_CSC_SYNCFREE_SUBSTITUTION_FORWARD
            1, // rhs
            queue );
    }
       

cleanup:
    return info;
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
    magma_int_t info = 0;
    
    magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0);

    // CUSPARSE context //
    CHECK(magma_ztrisolve(precond->LT, precond->cuinfoLT, true, false, false, b, *x, queue));
    
    

cleanup:
    return info;
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
    magma_int_t info = 0;
    
    magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0);

    // CUSPARSE context //
    if( precond->trisolver == Magma_CUSOLVE || precond->trisolver == 0 ){
        CHECK(magma_ztrisolve(precond->U, precond->cuinfoU, true, false, false, b, *x, queue));
    } else if( precond->trisolver == Magma_SYNCFREESOLVE ){
        magma_zgecscsyncfreetrsm_solve( precond->U.num_rows,
            precond->U.nnz,
            precond->U.dval, precond->U.drow, precond->U.dcol, 
            precond->U_dgraphindegree, precond->U_dgraphindegree_bak, 
            x->dval, b.dval, 1, //MAGMA_CSC_SYNCFREE_SUBSTITUTION_BACKWARD
            1, // rhs
            queue );
    }
    
    

cleanup:
    return info; 
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
    magma_int_t info = 0;
    
    magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0);

    CHECK(magma_ztrisolve(precond->UT, precond->cuinfoUT, false, false, false, b, *x, queue));
    
cleanup:
    return info; 
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
    magma_int_t info = 0;
    
    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrA=NULL;
    cusparseMatDescr_t descrL=NULL;
    cusparseMatDescr_t descrU=NULL;
#if CUDA_VERSION >= 7000
    csric02Info_t info_M=NULL;
    void *pBuffer = NULL;
#endif
    
    magma_z_matrix hA={Magma_CSR}, hACSR={Magma_CSR}, U={Magma_CSR};
    CHECK( magma_zmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
    U.diagorder_type = Magma_VALUE;
    CHECK( magma_zmconvert( hA, &hACSR, hA.storage_type, Magma_CSR, queue ));

    // in case using fill-in
    if( precond->levels > 0 ){
            magma_z_matrix hAL={Magma_CSR}, hAUt={Magma_CSR};
            CHECK( magma_zsymbilu( &hACSR, precond->levels, &hAL, &hAUt,  queue ));
            magma_zmfree(&hAL, queue);
            magma_zmfree(&hAUt, queue);
    }

    CHECK( magma_zmconvert( hACSR, &U, Magma_CSR, Magma_CSRL, queue ));
    magma_zmfree( &hACSR, queue );
    CHECK( magma_zmtransfer(U, &(precond->M), Magma_CPU, Magma_DEV, queue ));

    // CUSPARSE context //
    CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
    CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrA ));
    cusparseCreateSolveAnalysisInfo( &(precond->cuinfoILU) );
    // use kernel to manually check for zeros n the diagonal
    CHECK( magma_zdiagcheck( precond->M, queue ) );
    
#if CUDA_VERSION >= 12000
    // this version has the bug fixed where a zero on the diagonal causes a crash
    CHECK_CUSPARSE( cusparseCreateCsric02Info(&info_M) );
    CHECK_CUSPARSE( cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_GENERAL ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrA, CUSPARSE_INDEX_BASE_ZERO ));
    int buffersize;
    int structural_zero;
    int numerical_zero;
    
    CHECK_CUSPARSE(
    cusparseZcsric02_bufferSize( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M,
                         &buffersize ) );
    
    CHECK( magma_malloc((void**)&pBuffer, buffersize) );

    CHECK_CUSPARSE( cusparseZcsric02_analysis( cusparseHandle,
            precond->M.num_rows, precond->M.nnz, descrA,
            precond->M.dval, precond->M.drow, precond->M.dcol,
            info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer ));
    CHECK_CUSPARSE( cusparseXcsric02_zeroPivot( cusparseHandle, info_M, &numerical_zero ) );
    CHECK_CUSPARSE( cusparseXcsric02_zeroPivot( cusparseHandle, info_M, &structural_zero ) );

    CHECK_CUSPARSE(
    cusparseZcsric02( cusparseHandle,
                         precond->M.num_rows, precond->M.nnz, descrA,
                         precond->M.dval, precond->M.drow, precond->M.dcol,
                         info_M, CUSPARSE_SOLVE_POLICY_NO_LEVEL, pBuffer) );    

#else
    // this version contains the bug but is needed for backward compability
    CHECK_CUSPARSE( cusparseSetMatType( descrA, CUSPARSE_MATRIX_TYPE_SYMMETRIC ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrA, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrA, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE( cusparseSetMatFillMode( descrA, CUSPARSE_FILL_MODE_LOWER ));
    
    // todo: Zcsric0 needs different analysis (cusparseZcsric02_analysis)
    cusparseZcsrsm_analysis( cusparseHandle,
                             CUSPARSE_OPERATION_NON_TRANSPOSE,
                             precond->M.num_rows, precond->M.nnz, descrA,
                             precond->M.dval, precond->M.drow, precond->M.dcol,
                             precond->cuinfoILU );
    cusparseZcsric0( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     precond->M.num_rows, precond->M.nnz, descrA,
                     precond->M.dval,
                     precond->M.drow,
                     precond->M.dcol,
                     precond->cuinfoILU );
#endif

    CHECK( magma_zmtransfer( precond->M, &precond->L, 
        Magma_DEV, Magma_DEV, queue ));
    CHECK( magma_zmtranspose(precond->M, &precond->U, queue ));

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



/*
    // to enable also the block-asynchronous iteration for the triangular solves
    CHECK( magma_zmtransfer( precond->M, &hA, Magma_DEV, Magma_CPU, queue ));
    hA.storage_type = Magma_CSR;

    magma_z_matrix hD, hR, hAt

    CHECK( magma_zcsrsplit( 256, hA, &hD, &hR, queue ));

    CHECK( magma_zmtransfer( hD, &precond->LD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_zmtransfer( hR, &precond->L, Magma_CPU, Magma_DEV, queue ));

    magma_zmfree(&hD, queue );
    magma_zmfree(&hR, queue );

    CHECK( magma_z_cucsrtranspose(   hA, &hAt, queue ));

    CHECK( magma_zcsrsplit( 256, hAt, &hD, &hR, queue ));

    CHECK( magma_zmtransfer( hD, &precond->UD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_zmtransfer( hR, &precond->U, Magma_CPU, Magma_DEV, queue ));
    
    magma_zmfree(&hD, queue );
    magma_zmfree(&hR, queue );
    magma_zmfree(&hA, queue );
    magma_zmfree(&hAt, queue );
*/

cleanup:
#if CUDA_VERSION >= 7000
    magma_free( pBuffer );
    cusparseDestroyCsric02Info( info_M );
#endif
    cusparseDestroySolveAnalysisInfo( precond->cuinfoILU );
    cusparseDestroyMatDescr( descrL );
    cusparseDestroyMatDescr( descrU );
    cusparseDestroyMatDescr( descrA );
    cusparseDestroy( cusparseHandle );
    magma_zmfree(&U, queue );
    magma_zmfree(&hA, queue );

    return info;
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
    magma_int_t info = 0;
    
    CHECK(magma_ztrisolve_analysis(precond->M, &precond->cuinfoL, false, false, false, queue));
    CHECK(magma_ztrisolve_analysis(precond->M, &precond->cuinfoU, false, false, true, queue));
    

/*
    // to enable also the block-asynchronous iteration for the triangular solves
    CHECK( magma_zmtransfer( precond->M, &hA, Magma_DEV, Magma_CPU, queue ));
    hA.storage_type = Magma_CSR;

    CHECK( magma_zcsrsplit( 256, hA, &hD, &hR, queue ));

    CHECK( magma_zmtransfer( hD, &precond->LD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_zmtransfer( hR, &precond->L, Magma_CPU, Magma_DEV, queue ));

    magma_zmfree(&hD, queue );
    magma_zmfree(&hR, queue );

    CHECK( magma_z_cucsrtranspose(   hA, &hAt, queue ));

    CHECK( magma_zcsrsplit( 256, hAt, &hD, &hR, queue ));

    CHECK( magma_zmtransfer( hD, &precond->UD, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_zmtransfer( hR, &precond->U, Magma_CPU, Magma_DEV, queue ));
    
    magma_zmfree(&hD, queue );
    magma_zmfree(&hR, queue );
    magma_zmfree(&hA, queue );
    magma_zmfree(&hAt, queue );
*/

cleanup:
    return info;
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
    magma_int_t info = 0;
    
    magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0);

    CHECK(magma_ztrisolve(precond->M, precond->cuinfoL, false, false, false, b, *x, queue));

cleanup:
    return info; 
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
    magma_int_t info = 0;
    
    magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0);

    CHECK(magma_ztrisolve(precond->M, precond->cuinfoU, false, false, true, b, *x, queue));
    
    

cleanup:
    return info; 
}
