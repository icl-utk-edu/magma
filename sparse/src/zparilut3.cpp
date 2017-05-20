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
    submitted to SIAM SISC in 2016.

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
*******************************************************************************/
extern "C"
magma_int_t
magma_zparilut3setup(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
#ifdef _OPENMP

    real_Double_t start, end;
    real_Double_t t_rm=0.0, t_add=0.0, t_res=0.0, t_sweep1=0.0, t_sweep2=0.0, t_cand=0.0,
                    t_transpose1=0.0, t_transpose2=0.0, t_selectrm=0.0,
                    t_selectadd=0.0, t_nrm=0.0, t_total = 0.0, accum=0.0;
                    
    char filenameL[sizeof "LT_rm20_step10.m"];
    char filenameU[sizeof "UT_rm20_step10.m"];

                    
    double sum, sumL, sumU;

    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrL=NULL;
    cusparseMatDescr_t descrU=NULL;
    magma_z_matrix hA={Magma_CSR}, A0={Magma_CSR}, hAT={Magma_CSR}, hL={Magma_CSR}, hU={Magma_CSR},
                    oneL={Magma_CSR}, oneU={Magma_CSR},
                    L={Magma_CSR}, U={Magma_CSR}, L_new={Magma_CSR}, U_new={Magma_CSR}, UT={Magma_CSR};
    magma_z_matrix L0={Magma_CSR}, U0={Magma_CSR};  
    magma_int_t num_rmL, num_rmU;
    double thrsL = 0.0;
    double thrsU = 0.0;

    magma_int_t num_threads, timing = 1; // print timing
    magma_int_t L0nnz, U0nnz;

    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }


    CHECK( magma_zmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_zmtransfer( A, &A0, A.memory_location, Magma_CPU, queue ));

        // in case using fill-in
    if( precond->levels > 0 ){
        CHECK( magma_zsymbilu( &hA, precond->levels, &hL, &hU , queue ));
    }
    magma_zmfree(&hU, queue );
    magma_zmfree(&hL, queue );
    L.diagorder_type = Magma_UNITY;
    magma_zmatrix_tril( hA, &L, queue );
    magma_zmtranspose(hA, &hAT, queue );
    magma_zmatrix_tril( hA, &UT, queue );
    magma_zmtranspose(UT, &U, queue );
    magma_zmfree(&UT, queue );
    
    CHECK( magma_zmtranspose( U, &UT, queue) );
    L.rowidx = NULL;
    UT.rowidx = NULL;
    magma_zmatrix_addrowindex( &L, queue ); 
    magma_zmatrix_addrowindex( &UT, queue ); 
    //CHECK( magma_zparilut_sweep( &A0, &L, &UT, queue ) );
    //CHECK( magma_zparilut_sweep( &A0, &L, &UT, queue ) );
    //CHECK( magma_zparilut_sweep( &A0, &L, &UT, queue ) );
    //CHECK( magma_zparilut_sweep( &A0, &L, &UT, queue ) );
    //CHECK( magma_zparilut_sweep( &A0, &L, &UT, queue ) );
    L0nnz=L.nnz;
    U0nnz=U.nnz;
        
    // need only lower triangular
    magma_zmfree(&U, queue );
    CHECK( magma_zmtranspose( UT, &U, queue) );
    CHECK( magma_zmtransfer( L, &L0, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_zmtransfer( U, &U0, A.memory_location, Magma_CPU, queue ));
    magma_zmatrix_addrowindex( &U, queue );
    magma_zmfree(&UT, queue );
    magma_free_cpu( UT.row ); UT.row = NULL;
    magma_free_cpu( UT.list ); UT.list = NULL;
    CHECK( magma_zparilut_create_collinkedlist( U, &UT, queue) );
   

    if (timing == 1) {
        printf("performance_%d = [\n%%iter L.nnz U.nnz    ILU-Norm     candidat  resid     ILU-norm  selectad  add       transp1   sweep1    selectrm  remove    sweep2    transp2   total       accum\n", (int) num_threads);
    }

    //##########################################################################

    for( magma_int_t iters =0; iters<precond->sweeps; iters++ ) {
    t_rm=0.0; t_add=0.0; t_res=0.0; t_sweep1=0.0; t_sweep2=0.0; t_cand=0.0;
                        t_transpose1=0.0; t_transpose2=0.0; t_selectrm=0.0;
                        t_selectadd=0.0; t_nrm=0.0; t_total = 0.0;

        // try something crazy...
        // LU = L * U
        // Z = A cup LU
        // residuals for all
        // this gives us the ILU norm
        // then Z1 = Z \ L
        // then Z2 = Z1 \ U
        // use Z2 as candidate list
     //   printf("start with SpMM LU = L*U\n");
        
           //      CHECK( magma_zmtransfer( A, &Z, A.memory_location, Magma_CPU, queue ));
           //      CHECK( magma_zsymbilu( &Z, 1, &hL, &hU , queue ));
           //      magma_zmfree(&hU, queue );
           //      magma_zmfree(&hL, queue );
     //   printf("Z = LU cup A0\n");
     
        num_rmL = max( (L_new.nnz-L0nnz*(1+precond->atol*(iters+1)/precond->sweeps)), 0 );
        num_rmU = max( (U_new.nnz-U0nnz*(1+precond->atol*(iters+1)/precond->sweeps)), 0 );
        
        start = magma_sync_wtime( queue );
        magma_zparilut_candidates( L0, U0, L, U, &hL, &hU, queue );
        end = magma_sync_wtime( queue ); t_cand=+end-start;
        start = magma_sync_wtime( queue );
        
        end = magma_sync_wtime( queue ); t_transpose1+=end-start;
        start = magma_sync_wtime( queue );
        magma_zparilut_residuals_semilinked( hA, L, UT, &hL, queue );
        magma_zparilut_residuals_semilinked( hA, L, UT, &hU, queue );
        end = magma_sync_wtime( queue ); t_res=+end-start;
        start = magma_sync_wtime( queue );
        magma_zparilut_elementsum( hL, &sumL, queue );
        magma_zparilut_elementsum( hU, &sumU, queue );
        sum = sumL + sumU;
        end = magma_sync_wtime( queue ); t_nrm+=end-start;
        start = magma_sync_wtime( queue );
        
        
        // alternative: select one per row ad check with the value larger the rtol*abs(diag)
        magma_zparilut_selectoneperrowthrs_lower( L, U, &hL, precond->rtol, &oneL, queue );
        magma_zparilut_selectoneperrowthrs_upper( U, U, &hU, precond->rtol, &oneU, queue );        
        CHECK( magma_zmatrix_swap( &oneL, &hL, queue) );
        CHECK( magma_zmatrix_swap( &oneU, &hU, queue) );
        magma_zmfree( &oneL, queue );
        magma_zmfree( &oneU, queue );
        // end alternative
        /*
        magma_zparilut_selectoneperrow( 1, &hL, &oneL, queue );
        magma_zparilut_selectonepercol( 1, &hU, &oneU, queue );
        CHECK( magma_zmatrix_swap( &oneL, &hL, queue) );
        CHECK( magma_zmatrix_swap( &oneU, &hU, queue) );
        magma_zmfree( &oneL, queue );
        magma_zmfree( &oneU, queue );
        num_rmL = max(hL.nnz * ( precond->rtol ),0);
        num_rmU = max(hU.nnz * ( precond->rtol ),0);
        // num_rmL = max(hL.nnz * ( precond->rtol-0.15*iters ),0);
        // num_rmU = max(hU.nnz * ( precond->rtol-0.15*iters ),0);
        //printf("hL:%d  hU:%d\n", num_rmL, num_rmU);
        //#pragma omp parallel
        {
          //  magma_int_t id = omp_get_thread_num();
            //if( id == 0 ){
                if( num_rmL>0 ){
                    magma_zparilut_set_thrs_randomselect( num_rmL, &hL, 1, &thrsL, queue );
                } else {
                    thrsL = 0.0;
                }
            //} 
            //if( id == num_threads-1 ){
                if( num_rmU>0 ){
                    magma_zparilut_set_thrs_randomselect( num_rmU, &hU, 1, &thrsU, queue );
                } else {
                    thrsU = 0.0;
                }
            //}
        }
        magma_zparilut_thrsrm( 1, &hL, &thrsL, queue );
        magma_zparilut_thrsrm( 1, &hU, &thrsU, queue );
        
        */
        end = magma_sync_wtime( queue ); t_selectadd+=end-start;

        start = magma_sync_wtime( queue );
        CHECK( magma_zmatrix_cup(  L, hL, &L_new, queue ) );
        CHECK( magma_zmatrix_cup(  U, hU, &U_new, queue ) );
        end = magma_sync_wtime( queue ); t_add=+end-start;
        magma_zmfree( &hL, queue );
        magma_zmfree( &hU, queue );
       
        // using linked list
        start = magma_sync_wtime( queue );
        magma_free_cpu( UT.row ); UT.row = NULL;
        magma_free_cpu( UT.list ); UT.list = NULL;
        CHECK( magma_zparilut_create_collinkedlist( U_new, &UT, queue) );
        end = magma_sync_wtime( queue ); t_transpose2+=end-start;
        start = magma_sync_wtime( queue );
        CHECK( magma_zparilut_sweep_semilinked( &A0, &L_new, &UT, queue ) );
        end = magma_sync_wtime( queue ); t_sweep1+=end-start;
        num_rmL = max( (L_new.nnz-L0nnz*(1+precond->atol*(iters+1)/precond->sweeps)), 0 );
        num_rmU = max( (U_new.nnz-U0nnz*(1+precond->atol*(iters+1)/precond->sweeps)), 0 );
        start = magma_sync_wtime( queue );
        // pre-select: ignore the diagonal entries
        magma_zparilut_preselect( 0, &L_new, &oneL, queue );
        magma_zparilut_preselect( 1, &U_new, &oneU, queue );
        
        //#pragma omp parallel
        {
          //  magma_int_t id = omp_get_thread_num();
            //if( id == 0 ){
                if( num_rmL>0 ){
                    magma_zparilut_set_thrs_randomselect( num_rmL, &oneL, 0, &thrsL, queue );
                } else {
                    thrsL = 0.0;
                }
            //} 
            //if( id == num_threads-1 ){
                if( num_rmU>0 ){
                    magma_zparilut_set_thrs_randomselect( num_rmU, &oneU, 0, &thrsU, queue );
                } else {
                    thrsU = 0.0;
                }
            //}
        }
        // magma_zparilut_set_thrs_randomselect( num_rmL, &L_new, 0, &thrsL, queue );
        // magma_zparilut_set_thrs_randomselect( num_rmU, &UT, 0, &thrsU, queue );
        end = magma_sync_wtime( queue ); t_selectrm=end-start;
        magma_zmfree( &oneL, queue );
        magma_zmfree( &oneU, queue );
        start = magma_sync_wtime( queue );
        magma_zparilut_thrsrm( 1, &L_new, &thrsL, queue );//printf("done...");fflush(stdout);
        magma_zparilut_thrsrm( 1, &U_new, &thrsU, queue );//printf("done...");fflush(stdout);
        // magma_zparilut_thrsrm_semilinked( &U_new, &UT, &thrsU, queue );//printf("done.\n");fflush(stdout);
        CHECK( magma_zmatrix_swap( &L_new, &L, queue) );
        CHECK( magma_zmatrix_swap( &U_new, &U, queue) );
        magma_zmfree( &L_new, queue );
        magma_zmfree( &U_new, queue );
        end = magma_sync_wtime( queue ); t_rm=end-start;

        
        start = magma_sync_wtime( queue );
        magma_free_cpu( UT.row ); UT.row = NULL;
        magma_free_cpu( UT.list ); UT.list = NULL;
        CHECK( magma_zparilut_create_collinkedlist( U, &UT, queue) );
        end = magma_sync_wtime( queue ); t_transpose1+=end-start;
        start = magma_sync_wtime( queue );
        CHECK( magma_zparilut_sweep_semilinked( &A0, &L, &UT, queue ) );
        end = magma_sync_wtime( queue ); t_sweep2+=end-start;

        start = magma_sync_wtime( queue );

        end = magma_sync_wtime( queue ); t_rm+=end-start;
        // end using linked list
        
        if( timing == 1 ){
            t_total = t_cand+t_res+t_nrm+t_selectadd+t_add+t_transpose1+t_sweep1+t_selectrm+t_rm+t_sweep2+t_transpose2;
            accum = accum + t_total;
            printf("%5lld %5lld %5lld  %.4e   %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e    %.2e\n",
                    (long long) iters, (long long) L.nnz, (long long) U.nnz, (double) sum, 
                    t_cand, t_res, t_nrm, t_selectadd, t_add, t_transpose1, t_sweep1, t_selectrm, t_rm, t_sweep2, t_transpose2, t_total, accum );
            fflush(stdout);
        }
        

        sprintf(filenameL, "LT_rm%03d_step%d.m", (int)(precond->rtol*1000), iters+1);
        sprintf(filenameU, "UT_rm%03d_step%d.m", (int)(precond->rtol*1000), iters+1);

        // write to file
        // CHECK( magma_zwrite_csrtomtx( L, filenameL, queue ));
        // CHECK( magma_zwrite_csrtomtx( U, filenameU, queue ));
        
      
    }

    if (timing == 1) {
        printf("]; \n");
    }
    //##########################################################################



    //printf("%% check L:\n"); fflush(stdout);
    //magma_zdiagcheck_cpu( hL, queue );
    //printf("%% check U:\n"); fflush(stdout);
    //magma_zdiagcheck_cpu( hU, queue );

    // for CUSPARSE
    CHECK( magma_zmtransfer( L, &precond->L, Magma_CPU, Magma_DEV , queue ));
    CHECK( magma_zmtransfer( U, &precond->U, Magma_CPU, Magma_DEV , queue ));

    // CUSPARSE context //
    CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrL ));
    CHECK_CUSPARSE( cusparseSetMatType( descrL, CUSPARSE_MATRIX_TYPE_TRIANGULAR ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrL, CUSPARSE_DIAG_TYPE_UNIT ));
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

    if( precond->trisolver != 0 && precond->trisolver != Magma_CUSOLVE ){
        //prepare for iterative solves

        // extract the diagonal of L into precond->d
        CHECK( magma_zjacobisetup_diagscal( precond->L, &precond->d, queue ));
        CHECK( magma_zvinit( &precond->work1, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, queue ));

        // extract the diagonal of U into precond->d2
        CHECK( magma_zjacobisetup_diagscal( precond->U, &precond->d2, queue ));
        CHECK( magma_zvinit( &precond->work2, Magma_DEV, hA.num_rows, 1, MAGMA_Z_ZERO, queue ));
    }

    if( precond->trisolver == Magma_JACOBI && precond->pattern == 1 ){
        // dirty workaround for Jacobi trisolves....
        magma_zmfree( &hL, queue );
        magma_zmfree( &hU, queue );
        CHECK( magma_zmtransfer( precond->U, &hU, Magma_DEV, Magma_CPU , queue ));
        CHECK( magma_zmtransfer( precond->L, &hL, Magma_DEV, Magma_CPU , queue ));
        magma_zmfree( &hAT, queue );
        hAT.diagorder_type = Magma_VALUE;
        CHECK( magma_zmconvert( hL, &hAT , Magma_CSR, Magma_CSRU, queue ));
        #pragma omp parallel for
        for (magma_int_t i=0; i<hAT.nnz; i++) {
            hAT.val[i] = MAGMA_Z_ONE/hAT.val[i];
        }
        CHECK( magma_zmtransfer( hAT, &(precond->LD), Magma_CPU, Magma_DEV, queue ));

        magma_zmfree( &hAT, queue );
        hAT.diagorder_type = Magma_VALUE;
        CHECK( magma_zmconvert( hU, &hAT , Magma_CSR, Magma_CSRL, queue ));
        #pragma omp parallel for
        for (magma_int_t i=0; i<hAT.nnz; i++) {
            hAT.val[i] = MAGMA_Z_ONE/hAT.val[i];
        }
        CHECK( magma_zmtransfer( hAT, &(precond->UD), Magma_CPU, Magma_DEV, queue ));
    }

    cleanup:

    cusparseDestroy( cusparseHandle );
    cusparseDestroyMatDescr( descrL );
    cusparseDestroyMatDescr( descrU );
    cusparseHandle=NULL;
    descrL=NULL;
    descrU=NULL;
    magma_zmfree( &hA, queue );
    magma_zmfree( &hAT, queue );
    magma_zmfree( &A0, queue );
    magma_zmfree( &L0, queue );
    magma_zmfree( &U0, queue );
    magma_zmfree( &hAT, queue );
    magma_zmfree( &hL, queue );
    magma_zmfree( &L, queue );
    magma_zmfree( &L_new, queue );
    magma_zmfree( &hU, queue );
    magma_zmfree( &U, queue );
    magma_zmfree( &U_new, queue );
    //magma_zmfree( &UT, queue );
#endif
    return info;
}
