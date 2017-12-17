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
magma_zparilut(
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
    L.diagorder_type = Magma_VALUE;
    magma_zmatrix_tril( hA, &L, queue );
    magma_zmtranspose(hA, &hAT, queue );
    U.diagorder_type = Magma_UNITY;
    magma_zmatrix_tril( hAT, &U, queue );
    for ( magma_int_t z=0; z<U.num_rows; z++ ){
        U.val[U.row[z+1]-1] = MAGMA_Z_ONE;        
    }

    CHECK( magma_zmtranspose( U, &UT, queue) );
    L.rowidx = NULL;
    UT.rowidx = NULL;
    magma_zmatrix_addrowindex( &L, queue ); 
    magma_zmatrix_addrowindex( &U, queue ); 
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
    CHECK( magma_zmtransfer( L, &oneL, A.memory_location, Magma_CPU, queue ));
    CHECK( magma_zmtransfer( UT, &U0, A.memory_location, Magma_CPU, queue ));
    magma_zmatrix_addrowindex( &U, queue );
    magma_zmfree(&UT, queue );
    //magma_free_cpu( UT.row ); UT.row = NULL;
    //magma_free_cpu( UT.list ); UT.list = NULL;
    //CHECK( magma_zparilut_create_collinkedlist( U, &UT, queue) );

    if (timing == 1) {
        printf("ilut_fill_ratio = %.6f;\n\n", precond->atol ); 
        
        printf("performance_%d = [\n%%iter L.nnz U.nnz    ILU-Norm     candidat  resid     ILU-norm  selectad  add       transp1   sweep1    selectrm  remove    sweep2    transp2   total       accum\n", (int) num_threads);
    }

    //##########################################################################

    for( magma_int_t iters =0; iters<precond->sweeps; iters++ ) {
    t_rm=0.0; t_add=0.0; t_res=0.0; t_sweep1=0.0; t_sweep2=0.0; t_cand=0.0;
                        t_transpose1=0.0; t_transpose2=0.0; t_selectrm=0.0;
                        t_selectadd=0.0; t_nrm=0.0; t_total = 0.0;
     
        num_rmL = max( (L_new.nnz-L0nnz*(1+precond->atol*(iters+1)/precond->sweeps)), 0 );
        num_rmU = max( (U_new.nnz-U0nnz*(1+precond->atol*(iters+1)/precond->sweeps)), 0 );
        // magma_free_cpu( UT.row ); UT.row = NULL;
        // magma_free_cpu( UT.list ); UT.list = NULL;
        // CHECK( magma_zparilut_create_collinkedlist( U, &UT, queue) );
        start = magma_sync_wtime( queue );
        magma_zmfree(&UT, queue );
        //magma_zmtransposestruct_cpu( U, &UT, queue );
        magma_zcsrcoo_transpose( U, &UT, queue );
        end = magma_sync_wtime( queue ); t_transpose1+=end-start;
        start = magma_sync_wtime( queue );
        magma_zparilut_candidates( L0, U0, L, UT, &hL, &hU, queue );
        end = magma_sync_wtime( queue ); t_cand=+end-start;
        
        if( precond->rtol == 1.0 ){
            
            start = magma_sync_wtime( queue );
            magma_zparilut_residuals( hA, L, U, &hL, queue );
            magma_zparilut_residuals( hA, L, U, &hU, queue );
            end = magma_sync_wtime( queue ); t_res=+end-start;
            start = magma_sync_wtime( queue );
            magma_zmatrix_abssum( hL, &sumL, queue );
            magma_zmatrix_abssum( hU, &sumU, queue );
            sum = sumL + sumU;
            end = magma_sync_wtime( queue ); t_nrm+=end-start;
            
            
            start = magma_sync_wtime( queue );
            magma_zparilut_transpose_select_one( hU, &oneU, queue );
            magma_zparilut_selectoneperrow( 1, &hL, &oneL, queue );
            magma_zmfree(&hL, queue );
            magma_zmfree(&hU, queue );
            end = magma_sync_wtime( queue ); t_selectadd+=end-start;
            
        } else if( precond->rtol > 1.0 ) {
            start = magma_sync_wtime( queue );
            magma_zparilut_residuals( hA, L, U, &hL, queue );
            magma_zparilut_residuals( hA, L, U, &hU, queue );
            end = magma_sync_wtime( queue ); t_res=+end-start;
            start = magma_sync_wtime( queue );
            magma_zmatrix_abssum( hL, &sumL, queue );
            magma_zmatrix_abssum( hU, &sumU, queue );
            sum = sumL + sumU;
            end = magma_sync_wtime( queue ); t_nrm+=end-start;
            CHECK( magma_zmatrix_swap(  &hL, &oneL, queue) );
            magma_zmfree(&hL, queue );
            start = magma_sync_wtime( queue );
            if( precond->pattern == 2 ){
                // align residuals in U
                magma_zparilut_align_residuals( L, U, &hL, &hU, queue );
            } else if(precond->pattern == 0){
                for(magma_int_t z=0; z<hL.nnz; z++)
                    hL.val[z] = MAGMA_Z_ZERO;
                for(magma_int_t z=0; z<hU.nnz; z++)
                    hU.val[z] = MAGMA_Z_ZERO;
            }
            magma_zcsrcoo_transpose( hU, &oneU, queue );
            end = magma_sync_wtime( queue ); t_transpose2+=end-start;
            magma_zmfree(&hU, queue );
            magma_zmfree(&UT, queue );
                
        } else {
            
            start = magma_sync_wtime( queue );
            magma_zparilut_residuals( hA, L, U, &hL, queue );
            magma_zparilut_residuals( hA, L, U, &hU, queue );
            end = magma_sync_wtime( queue ); t_res=+end-start;
            start = magma_sync_wtime( queue );
            magma_zmatrix_abssum( hL, &sumL, queue );
            magma_zmatrix_abssum( hU, &sumU, queue );
            sum = sumL + sumU;
            end = magma_sync_wtime( queue ); t_nrm+=end-start;
            
            
            start = magma_sync_wtime( queue );
            magma_zparilut_transpose_select_one( hU, &oneU, queue );
            
            magma_zmfree(&hU, queue );
            magma_zmfree(&UT, queue );
            // magma_zmatrix_addrowindex( &hU, queue );
            // CHECK( magma_zmatrix_swap( &oneU, &hU, queue) );
            // magma_zmfree(&oneU, queue );
            // magma_zmatrix_addrowindex( &hU, queue );
            
            
            end = magma_sync_wtime( queue ); t_transpose2+=end-start;
            
        
            magma_zparilut_selectoneperrow( 1, &hL, &oneL, queue );
            //CHECK( magma_zmatrix_swap( &oneL, &hL, queue) );
            //CHECK( magma_zmatrix_swap( &oneU, &hU, queue) );  
        
            // use only a subset of the candidates
            //magma_zmfree( &oneL, queue );
            //magma_zmfree( &oneU, queue );
            num_rmL = max(oneL.nnz * ( precond->rtol ),0);
            num_rmU = max(oneU.nnz * ( precond->rtol ),0);
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
                        thrsL = 1e6;
                    }
                //} 
                //if( id == num_threads-1 ){
                    if( num_rmU>0 ){
                        magma_zparilut_set_thrs_randomselect( num_rmU, &hU, 1, &thrsU, queue );
                    } else {
                        thrsU = 1e6;
                    }
                //}
            }
            magma_zparilut_thrsrm( 1, &oneL, &thrsL, queue );
            magma_zparilut_thrsrm( 1, &oneU, &thrsU, queue );
            
        }
        
        end = magma_sync_wtime( queue ); t_selectadd+=end-start;
        
        start = magma_sync_wtime( queue );
        #pragma omp parallel        
        for(int row=0; row<hL.num_rows; row++){
            magma_zindexsort( &hL.col[hL.row[row]], 0, hL.row[row+1]-hL.row[row]-1, queue );
        }

        #pragma omp parallel  
        for(int row=0; row<hL.num_rows; row++){
            magma_zindexsort( &hU.col[hU.row[row]], 0, hU.row[row+1]-hU.row[row]-1, queue );
        }
        CHECK( magma_zmatrix_cup(  L, oneL, &L_new, queue ) );   
        CHECK( magma_zmatrix_cup(  U, oneU, &U_new, queue ) );
        //magma_zmatrix_addrowindex( &U, queue );
        end = magma_sync_wtime( queue ); t_add=+end-start;
        magma_zmfree( &oneL, queue );
        magma_zmfree( &oneU, queue );
       
        // using linked list
       // start = magma_sync_wtime( queue );
       // magma_free_cpu( UT.row ); UT.row = NULL;
       // magma_free_cpu( UT.list ); UT.list = NULL;
       // CHECK( magma_zparilut_create_collinkedlist( U_new, &UT, queue) );
       // end = magma_sync_wtime( queue ); t_transpose2+=end-start;
        start = magma_sync_wtime( queue );
        // CHECK( magma_zparilut_sweep( &A0, &L_new, &U_new, queue ) );
        
         CHECK( magma_zparilut_sweep_sync( &A0, &L_new, &U_new, queue ) );
        end = magma_sync_wtime( queue ); t_sweep1+=end-start;
        num_rmL = max( (L_new.nnz-L0nnz*(1+(precond->atol-1.)*(iters+1)/precond->sweeps)), 0 );
        num_rmU = max( (U_new.nnz-U0nnz*(1+(precond->atol-1.)*(iters+1)/precond->sweeps)), 0 );
        start = magma_sync_wtime( queue );
        // pre-select: ignore the diagonal entries
        magma_zparilut_preselect( 0, &L_new, &oneL, queue );
        magma_zparilut_preselect( 0, &U_new, &oneU, queue );
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
        
        // if(thrsL > thrsL_old){
        //     thrsL=thrsL*2;
        // } else {
        //     thrsL_old=thrsL;
        // }
        // if(thrsU > thrsU_old){
        //     thrsU=thrsU*2;
        // } else {
        //     thrsU_old=thrsU;
        // }
        
        // magma_zparilut_set_thrs_randomselect( num_rmL, &L_new, 0, &thrsL, queue );
        // magma_zparilut_set_thrs_randomselect( num_rmU, &UT, 0, &thrsU, queue );
        end = magma_sync_wtime( queue ); t_selectrm=end-start;
        magma_zmfree( &oneL, queue );
        magma_zmfree( &oneU, queue );
        start = magma_sync_wtime( queue );
        
        magma_zparilut_thrsrm( 1, &L_new, &thrsL, queue );//printf("done...");fflush(stdout);
        magma_zparilut_thrsrm( 1, &U_new, &thrsU, queue );//printf("done...");fflush(stdout);

        
        // magma_zparilut_thrsrm_U( 1, L_new, &U_new, &thrsU, queue );
        // for(int z=0; z<L_new.nnz; z++){
        //     if(MAGMA_Z_ABS(L_new.val[z])<thrsL){
        //      printf("invalid element here:%.4e  < %.4e  <%.4e> \n",MAGMA_Z_ABS(L_new.val[z]),thrsL, MAGMA_Z_REAL(L_new.val[z]));    
        //     }
        // }
        // 
        // for(int z=0; z<U_new.nnz; z++){
        //     if(MAGMA_Z_ABS(U_new.val[z])<thrsU){
        //      printf("invalid element here:%.4e  < %.4e  <%.4e> \n",MAGMA_Z_ABS(U_new.val[z]),thrsU, MAGMA_Z_REAL(U_new.val[z]));    
        //     }
        // }
        
        // magma_zparilut_thrsrm_semilinked( &U_new, &UT, &thrsU, queue );//printf("done.\n");fflush(stdout);
        CHECK( magma_zmatrix_swap( &L_new, &L, queue) );
        CHECK( magma_zmatrix_swap( &U_new, &U, queue) );
        magma_zmfree( &L_new, queue );
        magma_zmfree( &U_new, queue );
        end = magma_sync_wtime( queue ); t_rm=end-start;
        
        start = magma_sync_wtime( queue );
        // magma_free_cpu( UT.row ); UT.row = NULL;
        // magma_free_cpu( UT.list ); UT.list = NULL;
        // CHECK( magma_zparilut_create_collinkedlist( U, &UT, queue) );
        // end = magma_sync_wtime( queue ); t_transpose1+=end-start;
        
        start = magma_sync_wtime( queue );
        
        //magma_free_cpu( U.rowidx ); U.rowidx = NULL;
        
        // CHECK( magma_zparilut_sweep( &A0, &L, &U, queue ) );
        CHECK( magma_zparilut_sweep_sync( &A0, &L, &U, queue ) );
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
    magma_zcsrcoo_transpose( U, &UT, queue );
    //magma_zmtranspose(U, &UT, queue );
    CHECK( magma_zmtransfer( UT, &precond->U, Magma_CPU, Magma_DEV , queue ));

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
        precond->L.dval, precond->L.drow, precond->L.dcol, precond->cuinfoL ));
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrU ));
    CHECK_CUSPARSE( cusparseSetMatType( descrU, CUSPARSE_MATRIX_TYPE_TRIANGULAR ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrU, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrU, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE( cusparseSetMatFillMode( descrU, CUSPARSE_FILL_MODE_UPPER ));
    CHECK_CUSPARSE( cusparseCreateSolveAnalysisInfo( &precond->cuinfoU ));
    CHECK_CUSPARSE( cusparseZcsrsv_analysis( cusparseHandle,
        CUSPARSE_OPERATION_NON_TRANSPOSE, precond->U.num_rows,
        precond->U.nnz, descrU,
        precond->U.dval, precond->U.drow, precond->U.dcol, precond->cuinfoU ));

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
    magma_zmfree( &UT, queue );
    magma_zmfree( &U_new, queue );
    //magma_zmfree( &UT, queue );
#endif
    return info;
}
