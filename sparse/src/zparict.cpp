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

    Prepares the iterative threshold Incomplete Cholesky preconditioner. 
    The strategy is interleaving a parallel fixed-point iteration that 
    approximates an incomplete factorization for a given nonzero pattern with a 
    procedure that adaptively changes the pattern. Much of this new algorithm 
    has fine-grained parallelism, and we show that it can efficiently exploit 
    the compute power of shared memory architectures.

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
magma_zparict(
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
                    
    double sum, sumL;//, sumU, thrsL_old=1e9, thrsU_old=1e9;

    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrL=NULL;
    cusparseMatDescr_t descrU=NULL;
    magma_z_matrix hA={Magma_CSR}, A0={Magma_CSR}, hAT={Magma_CSR}, hL={Magma_CSR}, hU={Magma_CSR},
                    oneL={Magma_CSR}, LT={Magma_CSR},
                    L={Magma_CSR}, L_new={Magma_CSR};
    magma_z_matrix L0={Magma_CSR};  
    magma_int_t num_rmL;
    double thrsL = 0.0;

    magma_int_t num_threads, timing = 1; // print timing
    magma_int_t L0nnz;

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
    L.diagorder_type = Magma_VALUE;
    magma_zmatrix_tril( hA, &L, queue );
    L.rowidx = NULL;
    magma_zmatrix_addrowindex( &L, queue ); 
    L0nnz=L.nnz;
        
    // need only lower triangular
    CHECK( magma_zmtransfer( L, &L0, A.memory_location, Magma_CPU, queue ));
    
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
        start = magma_sync_wtime( queue );
        magma_zmfree(&LT, queue );
        magma_zcsrcoo_transpose( L, &LT, queue );
        end = magma_sync_wtime( queue ); t_transpose1+=end-start;
        start = magma_sync_wtime( queue ); 
        magma_zparict_candidates( L0, L, LT, &hL, queue );
        #pragma omp parallel        
        for(int row=0; row<hL.num_rows; row++){
            magma_zindexsort( &hL.col[hL.row[row]], 0, hL.row[row+1]-hL.row[row]-1, queue );
        }
        end = magma_sync_wtime( queue ); t_cand=+end-start;
        
        start = magma_sync_wtime( queue );
        magma_zparilut_residuals( hA, L, L, &hL, queue );
        end = magma_sync_wtime( queue ); t_res=+end-start;
        start = magma_sync_wtime( queue );
        magma_zmatrix_abssum( hL, &sumL, queue );
        sum = sumL*2;
        end = magma_sync_wtime( queue ); t_nrm+=end-start;
        
        start = magma_sync_wtime( queue );
        CHECK( magma_zmatrix_cup(  L, hL, &L_new, queue ) );  
        end = magma_sync_wtime( queue ); t_add=+end-start;
        magma_zmfree( &hL, queue );
       
        start = magma_sync_wtime( queue );
         CHECK( magma_zparict_sweep_sync( &A0, &L_new, queue ) );
        end = magma_sync_wtime( queue ); t_sweep1+=end-start;
        num_rmL = max( (L_new.nnz-L0nnz*(1+(precond->atol-1.)*(iters+1)/precond->sweeps)), 0 );
        start = magma_sync_wtime( queue );
        magma_zparilut_preselect( 0, &L_new, &oneL, queue );
        //#pragma omp parallel
        {
            if( num_rmL>0 ){
                magma_zparilut_set_thrs_randomselect( num_rmL, &oneL, 0, &thrsL, queue );
            } else {
                thrsL = 0.0;
            }
        }
        end = magma_sync_wtime( queue ); t_selectrm=end-start;
        magma_zmfree( &oneL, queue );
        start = magma_sync_wtime( queue );
        magma_zparilut_thrsrm( 1, &L_new, &thrsL, queue );//printf("done...");fflush(stdout);
        CHECK( magma_zmatrix_swap( &L_new, &L, queue) );
        magma_zmfree( &L_new, queue );
        end = magma_sync_wtime( queue ); t_rm=end-start;
        
        start = magma_sync_wtime( queue );
        CHECK( magma_zparict_sweep_sync( &A0, &L, queue ) );
        end = magma_sync_wtime( queue ); t_sweep2+=end-start;

        if( timing == 1 ){
            t_total = t_cand+t_res+t_nrm+t_selectadd+t_add+t_transpose1+t_sweep1+t_selectrm+t_rm+t_sweep2+t_transpose2;
            accum = accum + t_total;
            printf("%5lld %5lld %5lld  %.4e   %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e  %.2e    %.2e\n",
                    (long long) iters, (long long) L.nnz, (long long) L.nnz, (double) sum, 
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
    CHECK( magma_zmtransfer( L, &precond->M, Magma_CPU, Magma_DEV , queue ));

    // CUSPARSE context //
    // lower triangular factor
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
        precond->M.dval, precond->M.drow, precond->M.dcol, precond->cuinfoL ));
    
    // upper triangular factor
    CHECK_CUSPARSE( cusparseCreateMatDescr( &descrU ));
    CHECK_CUSPARSE( cusparseSetMatType( descrU, CUSPARSE_MATRIX_TYPE_TRIANGULAR ));
    CHECK_CUSPARSE( cusparseSetMatDiagType( descrU, CUSPARSE_DIAG_TYPE_NON_UNIT ));
    CHECK_CUSPARSE( cusparseSetMatIndexBase( descrU, CUSPARSE_INDEX_BASE_ZERO ));
    CHECK_CUSPARSE( cusparseSetMatFillMode( descrU, CUSPARSE_FILL_MODE_LOWER ));
    CHECK_CUSPARSE( cusparseCreateSolveAnalysisInfo( &precond->cuinfoU ));
    CHECK_CUSPARSE( cusparseZcsrsm_analysis( cusparseHandle,
        CUSPARSE_OPERATION_TRANSPOSE, precond->M.num_rows,
        precond->M.nnz, descrU,
        precond->M.dval, precond->M.drow, precond->M.dcol, precond->cuinfoU ));
    
    
    if( precond->trisolver != 0 && precond->trisolver != Magma_CUSOLVE ){
        //prepare for iterative solves
        
        // copy the matrix to precond->L and (transposed) to precond->U
        CHECK( magma_zmtransfer(precond->M, &(precond->L), Magma_DEV, Magma_DEV, queue ));
        CHECK( magma_zmtranspose( precond->L, &(precond->U), queue ));
        
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
        CHECK( magma_zmtransfer( precond->L, &hL, Magma_DEV, Magma_CPU , queue ));
        hAT.diagorder_type = Magma_VALUE;
        CHECK( magma_zmconvert( hL, &hAT , Magma_CSR, Magma_CSRU, queue ));
        #pragma omp parallel for
        for (magma_int_t i=0; i<hAT.nnz; i++) {
            hAT.val[i] = MAGMA_Z_ONE/hAT.val[i];
        }
        CHECK( magma_zmtransfer( hAT, &(precond->LD), Magma_CPU, Magma_DEV, queue ));

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
    magma_zmfree( &hAT, queue );
    magma_zmfree( &hL, queue );
    magma_zmfree( &L, queue );
    magma_zmfree( &L_new, queue );
#endif
    return info;
}
