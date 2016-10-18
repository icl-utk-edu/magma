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
magma_zparilutsetup(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    real_Double_t start, end;
    real_Double_t t_rm=0.0, t_res=0.0, t_sweep1=0.0, t_sweep2=0.0, t_cand=0.0,
                    t_thres=0.0, t_reorder1=0.0, t_reorder2=0.0, t_rowmajor=0.0,
                    t_select=0.0, t_insert=0.0, accum=0.0;
#ifdef _OPENMP

    cusparseHandle_t cusparseHandle=NULL;
    cusparseMatDescr_t descrL=NULL;
    cusparseMatDescr_t descrU=NULL;
    magma_index_t *rm_locL = NULL;
    magma_index_t *rm_locU = NULL;
    magma_int_t num_rmLt, num_rmUt;
    magma_z_matrix hA={Magma_CSR}, hAT={Magma_CSR}, hL={Magma_CSR}, hU={Magma_CSR},
                    L={Magma_CSR}, U={Magma_CSR}, L_new={Magma_CSR}, U_new={Magma_CSR},
                    UR={Magma_CSR};
    magma_int_t num_rmL, num_rmU, num_rm_glL, num_rm_glU;
    magmaDoubleComplex thrsL = MAGMA_Z_ZERO;
    magmaDoubleComplex thrsU = MAGMA_Z_ZERO;

    magma_int_t num_threads, timing = 1; // print timing
    magma_int_t nnzL, nnzU;

    magma_int_t reorder = precond->restart;
    // use the linked-list variant for sweeps, residuals, candidate search
    // reorder = 0 never reorder
    // reorder = 1 reorder only once per loop (default)
    // reorder = 2 reorder twice per loop

    CHECK( magma_index_malloc_cpu( &rm_locL, A.nnz ) );
    CHECK( magma_index_malloc_cpu( &rm_locU, A.nnz ) );
    num_rm_glL = A.nnz*precond->rtol;
    num_rm_glU = A.nnz*precond->rtol;
    num_rmL = num_rm_glL;
    num_rmU = num_rm_glU;


    CHECK( magma_zmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));

        // in case using fill-in
    if( precond->levels > 0 ){
        CHECK( magma_zsymbilu( &hA, precond->levels, &hL, &hU , queue ));
    }
    // need only lower triangular
    magma_zmfree(&hU, queue );
    magma_zmfree(&hL, queue );

    magma_zmconvert( hA, &hL, Magma_CSR, Magma_CSRL, queue );
    magma_zmconvert( hL, &L, Magma_CSR, Magma_CSRLIST, queue );

    magma_zmtranspose(hA, &hAT, queue );
    magma_zmconvert( hAT, &hU, Magma_CSR, Magma_CSRL, queue );
    magma_zmconvert( hU, &U, Magma_CSR, Magma_CSRLIST, queue );

    // num_rm_glL = hL.nnz*precond->rtol;
    // num_rm_glU = hU.nnz*precond->rtol;



    start = magma_sync_wtime( queue );
    magma_zparilut_colmajor( U, &UR, queue );
    end = magma_sync_wtime( queue );
    //magma_free_cpu( UR.row );
    //magma_free_cpu( UR.list );


    //magma_z_mvisu( L, queue );

   // magma_zmfree( &hAT, queue );
   // magma_zmconvert( UR, &hAT, Magma_CSRLIST, Magma_CSR, queue );
    //magma_z_mvisu( UR, queue );

    magma_zmalloc_cpu( &L_new.val, L.num_rows*100 );
    magma_index_malloc_cpu( &L_new.rowidx, L.num_rows*100 );
    magma_index_malloc_cpu( &L_new.col, L.num_rows*100 );
    magma_index_malloc_cpu( &L_new.row, L.num_rows*100 );
    magma_index_malloc_cpu( &L_new.list, L.num_rows*100 );
    L_new.num_rows = L.num_rows;
    L_new.num_cols = L.num_cols;
    L_new.true_nnz = L.num_rows*100;
    L_new.blocksize = L.nnz;
    L_new.storage_type = Magma_COO;
    L_new.memory_location = Magma_CPU;

    magma_zmalloc_cpu( &U_new.val, U.num_rows*100 );
    magma_index_malloc_cpu( &U_new.rowidx, U.num_rows*100 );
    magma_index_malloc_cpu( &U_new.col, U.num_rows*100 );
    magma_index_malloc_cpu( &U_new.row, U.num_rows*100 );
    magma_index_malloc_cpu( &U_new.list, U.num_rows*100 );
    U_new.num_rows = U.num_rows;
    U_new.num_cols = U.num_cols;
    U_new.true_nnz = U.num_rows*100;
    U_new.blocksize = U.nnz;
    U_new.storage_type = Magma_COO;
    U_new.memory_location = Magma_CPU;

    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }

    #pragma omp parallel for
    for( magma_int_t z=0; z<A.nnz; z++ ){
        rm_locL[z] = L.nnz+z;
        rm_locU[z] = U.nnz+z;
    }

    magma_zparilut_sweep_list( &hA, &L, &U, queue );
    magma_zparilut_sweep_list( &hA, &L, &U, queue );
    magma_zparilut_sweep_list( &hA, &L, &U, queue );
    magma_zparilut_sweep_list( &hA, &L, &U, queue );
    magma_zparilut_sweep_list( &hA, &L, &U, queue );

    nnzL=L.nnz;
    nnzU=U.nnz;

    start = magma_sync_wtime( queue );


     start = magma_sync_wtime( queue );
     magma_zparilut_colmajorup( U, &UR, queue );
     end = magma_sync_wtime( queue );

      for( magma_int_t z=0; z<A.nnz; z++ ){
          rm_locL[z] = L.nnz+z;
          rm_locU[z] = U.nnz+z;
      }

     if( timing == 1 ){
         printf("performance_%d = \[\n%%iter\tL.nnz\tU.nnz\trm L\trm U\trowmajor\tcandidates\tresiduals\tselect\t\tinsert\t\treorder\t\tsweep\t\tthreshold\tremove\t\treorder\t\tsweeep\t\ttotal\t\t\taccum\n", num_threads);
     }

    //##########################################################################

    for( magma_int_t iters =0; iters<precond->sweeps; iters++ ) {
        L_new.nnz = 0;
        U_new.nnz = 0;

        start = magma_sync_wtime( queue );
        magma_zparilut_colmajorup( U, &UR, queue );
        end = magma_sync_wtime( queue ); t_rowmajor=end-start;

        start = magma_sync_wtime( queue );
        if( reorder == 0 ){
            magma_zparilut_candidates_linkedlist(
            L,
            U,
            UR,
            &L_new,
            &U_new,
            queue );
        } else {
            magma_zparilut_candidates_linkedlist(
            L,
            U,
            UR,
            &L_new,
            &U_new,
            queue );
        }
        end = magma_sync_wtime( queue ); t_cand=end-start;

        if( reorder == 1 ){
            // start = magma_sync_wtime( queue );
            // magma_zparilut_residuals_linkedlist( hA, L, U, &L_new, queue );
            // magma_zparilut_residuals_linkedlist( hA, L, U, &U_new, queue );
            // end = magma_sync_wtime( queue ); t_res=end-start;
            // printf("residuals unordered:%.2e\n", t_res);
            start = magma_sync_wtime( queue );
            magma_zparilut_residuals_list( hA, L, U, &L_new, queue );
            magma_zparilut_residuals_list( hA, L, U, &U_new, queue );
            end = magma_sync_wtime( queue ); t_res=end-start;
            // printf("residuals ordered:%.2e\n", t_res);
        } else {
            //magma_z_mvisu( U, queue );
            // magma_zparilut_reorder( &L, queue );
            // magma_zparilut_reorder( &U, queue );
            //printf("nnz:%d %d\n",L.nnz, U.nnz);
            // magma_zparilut_randlist( &L, queue );
            // magma_zparilut_randlist( &U, queue );

            start = magma_sync_wtime( queue );
            magma_zparilut_residuals_linkedlist( hA, L, U, &L_new, queue );
            magma_zparilut_residuals_linkedlist( hA, L, U, &U_new, queue );
            end = magma_sync_wtime( queue ); t_res=end-start;
        }

        num_rmL = max( nnzL*(1+precond->atol*(iters+1)/precond->sweeps) - L.nnz + num_rm_glL,0 );
        num_rmU = max( nnzU*(1+precond->atol*(iters+1)/precond->sweeps) - U.nnz + num_rm_glU,0 );
        // num_rmL = num_rm_glL;
        // num_rmU = num_rm_glU;

        num_rmLt = num_rmL;
        num_rmUt = num_rmU;

        start = magma_sync_wtime( queue );
        magma_zparilut_select_candidates_L( &num_rmL, rm_locL, &L_new, queue );
        magma_zparilut_select_candidates_U( &num_rmU, rm_locU, &U_new, queue );
        end = magma_sync_wtime( queue ); t_select=end-start;

        start = magma_sync_wtime( queue );
        magma_zparilut_insert(
             &num_rmL,
             &num_rmU,
             rm_locL,
             rm_locU,
             &L_new,
             &U_new,
             &L,
             &U,
             &UR,
             queue );
        end = magma_sync_wtime( queue ); t_insert=end-start;

        if( reorder > 1 ){
            // L.nnz = L.nnz + num_rmL;
            // U.nnz = U.nnz + num_rmU;
            // start = magma_sync_wtime( queue );
            // magma_zparilut_sweep_linkedlist( &hA, &L, &U, queue );
            // end = magma_sync_wtime( queue ); t_sweep1=end-start;
            // printf(" only sweep time: %.2e\n", t_sweep1 );
            start = magma_sync_wtime( queue );
            magma_zparilut_reorder( &L, queue );
            magma_zparilut_reorder( &U, queue );
            end = magma_sync_wtime( queue ); t_reorder1=end-start;
            // printf(" reorder time: %.2e\n", t_reorder1 );
            start = magma_sync_wtime( queue );
            magma_zparilut_sweep_list( &hA, &L, &U, queue );
            end = magma_sync_wtime( queue ); t_sweep1=end-start;
            // printf(" only ordered sweep time: %.2e\n", t_sweep1 );
            // printf(" reorder +sweep time: %.2e\n\n", t_reorder1+t_sweep1 );
        } else {
            L.nnz = L.nnz + num_rmL;
            U.nnz = U.nnz + num_rmU;
            //magma_z_mvisu( U, queue );
            // start = magma_sync_wtime( queue );
            //  magma_zparilut_reorder( &L, queue );
            //  magma_zparilut_reorder( &U, queue );
            // end = magma_sync_wtime( queue ); t_reorder1=end-start;
            //printf("nnz:%d %d\n",L.nnz, U.nnz);
            // magma_zparilut_randlist( &L, queue );
            // magma_zparilut_randlist( &U, queue );
            //printf("nnz:%d %d\n",L.nnz, U.nnz);
            //magma_zparilut_reorder( &L, queue );
            //magma_zparilut_reorder( &U, queue );
            //magma_z_mvisu( U, queue );
            start = magma_sync_wtime( queue );
            magma_zparilut_sweep_linkedlist( &hA, &L, &U, queue );
            end = magma_sync_wtime( queue ); t_sweep1=end-start;
            start = magma_sync_wtime( queue );
            end = magma_sync_wtime( queue );
        }

        //num_rmL = abs(L.nnz - nnzL*(1+precond->atol*(iters+1)/precond->sweeps) );
        //num_rmU = abs(U.nnz - nnzU*(1+precond->atol*(iters+1)/precond->sweeps) );

        num_rmL = max(L.nnz - nnzL*(1+precond->atol*(iters+1)/precond->sweeps),0 );
        num_rmU = max(U.nnz - nnzU*(1+precond->atol*(iters+1)/precond->sweeps),0 );

        start = magma_sync_wtime( queue );
        info = magma_zparilut_set_approx_thrs( num_rmL, &L, 0, &thrsL, queue );
        if( info !=0 ){
            printf("%% error: breakdown in iteration :%d. fallback.\n\n", iters+1);fflush(stdout);
            info = 0;
            break;
        }//printf("done thrs L\n");fflush(stdout);
        info = magma_zparilut_set_approx_thrs( num_rmU, &U, 0, &thrsU, queue );
        if( info !=0 ){
            printf("%% error: breakdown in iteration :%d. fallback.\n\n", iters+1);fflush(stdout);
            info = 0;
            break;
        }//printf("done thrs U\n");fflush(stdout);
        end = magma_sync_wtime( queue ); t_thres=end-start;

        // magma_zparilut_LU_approx_thrs( num_rmL+num_rmU, &L, &U, 0, &thrsL, queue );
        // thrsU=thrsL;
   // magma_int_t num_rm,
   // magma_z_matrix *L,
   // magma_z_matrix *U,
   // magma_int_t order,
   // magmaDoubleComplex *thrs,
   // magma_queue_t queue )

        start = magma_sync_wtime( queue );
        magma_zparilut_rm_thrs( &thrsL, &num_rmL, &L, &L_new, rm_locL, queue );
        magma_zparilut_rm_thrs( &thrsU, &num_rmU, &U, &U_new, rm_locU, queue );
        end = magma_sync_wtime( queue ); t_rm=end-start;


        if( reorder == 1 || reorder > 1 ){
            // start = magma_sync_wtime( queue );
            // L.nnz = L.nnz - num_rmL;
            // U.nnz = U.nnz - num_rmU;
            // magma_zparilut_sweep_linkedlist( &hA, &L, &U, queue );
            // end = magma_sync_wtime( queue ); t_sweep2=end-start;
            // printf(" only sweep time: %.2e\n", t_sweep2 );
            start = magma_sync_wtime( queue );
            magma_zparilut_reorder( &L, queue );
            magma_zparilut_reorder( &U, queue );
            end = magma_sync_wtime( queue ); t_reorder2=end-start;
            // printf(" reorder time: %.2e\n", t_reorder2 );
            start = magma_sync_wtime( queue );
            magma_zparilut_sweep_list( &hA, &L, &U, queue );
            end = magma_sync_wtime( queue ); t_sweep2=end-start;
            // printf(" only ordered sweep time: %.2e\n", t_sweep2 );
            // printf(" reorder +sweep time: %.2e\n", t_reorder2+t_sweep2 );
        } else {
            L.nnz = L.nnz - num_rmL;
            U.nnz = U.nnz - num_rmU;

            //magma_z_mvisu( U, queue );
            // start = magma_sync_wtime( queue );
            // magma_zparilut_reorder( &L, queue );
            // magma_zparilut_reorder( &U, queue );
            // end = magma_sync_wtime( queue ); t_reorder2=end-start;
            // //printf("nnz:%d %d\n",L.nnz, U.nnz);
            // magma_zparilut_randlist( &L, queue );
            // magma_zparilut_randlist( &U, queue );

            start = magma_sync_wtime( queue );
            magma_zparilut_sweep_linkedlist( &hA, &L, &U, queue );
            end = magma_sync_wtime( queue ); t_sweep2=end-start;
            start = magma_sync_wtime( queue );
            magma_zparilut_reorder( &L, queue );
            magma_zparilut_reorder( &U, queue );
            end = magma_sync_wtime( queue );
        }

        start = magma_sync_wtime( queue );

        if( timing == 1 ){
            accum = accum + t_cand+t_res+t_select+t_insert+t_reorder1+t_sweep1+t_thres+t_rm+t_reorder2+t_rowmajor+t_sweep2;
            printf("%d\t%d\t%d\t%d\t%d\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t%.2e\t\t%.2e\n",
                    iters, L.nnz, U.nnz, num_rmLt, num_rmUt, t_rowmajor, t_cand, t_res, t_select, t_insert, t_reorder1, t_sweep1, t_thres, t_rm, t_reorder2, t_sweep2,
                    t_cand+t_res+t_select+t_insert+t_reorder1+t_sweep1+t_thres+t_rm+t_reorder2+t_rowmajor+t_sweep2, accum);
            fflush(stdout);
        }

    }

    if( timing == 1 ){
         printf("];\n");
    }
    //##########################################################################

    magma_zparilut_sweep_list( &hA, &L, &U, queue );
    magma_zparilut_sweep_list( &hA, &L, &U, queue );

    L.nnz=L.nnz-num_rmL;
    U.nnz=U.nnz-num_rmU;
    magma_zparilut_count( L, &L.nnz, queue);
    magma_zparilut_count( U, &U.nnz, queue);


    magma_zmfree( &hAT, queue );
    magma_zparilut_reorder( &L, queue );
    magma_zparilut_reorder( &U, queue );

    magma_zmfree( &hL, queue );
    magma_zmfree( &hU, queue );
    magma_zmconvert( L, &hL, Magma_CSRLIST, Magma_CSR, queue );
    magma_zmconvert( U, &hU, Magma_CSRLIST, Magma_CSR, queue );
    magma_zmtranspose(hU, &hAT, queue );


    //printf("%% check L:\n");fflush(stdout);
    //magma_zdiagcheck_cpu( hL, queue );
    //printf("%% check U:\n");fflush(stdout);
    //magma_zdiagcheck_cpu( hU, queue );

    // for CUSPARSE
    CHECK( magma_zmtransfer( hL, &precond->L, Magma_CPU, Magma_DEV , queue ));
    CHECK( magma_zmtransfer( hAT, &precond->U, Magma_CPU, Magma_DEV , queue ));

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

    magma_free_cpu( rm_locL );
    magma_free_cpu( rm_locU );
    magma_free_cpu( UR.row );
    magma_free_cpu( UR.list );
    magma_zmfree( &L_new, queue );
    magma_zmfree( &U_new, queue );
    cusparseDestroy( cusparseHandle );
    cusparseDestroyMatDescr( descrL );
    cusparseDestroyMatDescr( descrU );
    cusparseHandle=NULL;
    descrL=NULL;
    descrU=NULL;
    magma_zmfree( &hA, queue );
    magma_zmfree( &hAT, queue );
    magma_zmfree( &hL, queue );
    magma_zmfree( &L, queue );
    magma_zmfree( &hU, queue );
    magma_zmfree( &U, queue );
#endif
    return info;
}


