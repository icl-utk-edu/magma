/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt
       @author Stephen Wood
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime_api.h>
#include <cublas.h>
#include <cusparse_v2.h>
#include <cuda_profiler_api.h>

// includes, project
#include "magma_v2.h"
#include "magmasparse.h"
#include "testings.h"

#define PRECISION_z


int main( int argc, char** argv)
{
    // generates the exact preconditioner for system IC
    // and uses then update sweeps for all others
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    
    real_Double_t start, end;
    real_Double_t t_cusparse, t_chow;

    magma_z_matrix hA={Magma_CSR}, hAL={Magma_CSR},
    hAU={Magma_CSR},  hAUt={Magma_CSR}, hLCU={Magma_CSR}, hUCU={Magma_CSR},
    hAcusparse={Magma_CSR}, hAtmp={Magma_CSR}, dA={Magma_CSR}, hLU={Magma_CSR}, 
    dL={Magma_CSR}, dU={Magma_CSR}, hL={Magma_CSR}, hU={Magma_CSR}, 
    hUT={Magma_CSR};

    int inp=1;
    int dwitermax = 1;
    magmaDoubleComplex diagweight = MAGMA_Z_ZERO;
    magma_scale_t scaling = Magma_UNITDIAG;

    for (inp=1; inp < argc; ++inp) {
        if (strcmp("--dwitermax", argv[inp]) == 0 && inp+1 < argc) {
            dwitermax = atoi( argv[++inp] );
        }
        else if (strcmp("--dweight", argv[inp]) == 0 && inp+1 < argc) {
            double dwtmp = 0.0;
            sscanf( argv[++inp], "%lf", &dwtmp );
            diagweight = MAGMA_Z_MAKE( dwtmp, 0.0 );
        }
        else if ( strcmp("--mscale", argv[inp]) == 0 && inp+1 < argc ) {
            inp++;
            if ( strcmp("NOSCALE", argv[inp]) == 0 ) {
                scaling = Magma_NOSCALE;
            }
            else if ( strcmp("UNITDIAG", argv[inp]) == 0 ) {
                scaling = Magma_UNITDIAG;
            }
            else if ( strcmp("UNITROW", argv[inp]) == 0 ) {
                scaling = Magma_UNITROW;
            }
            else {
                printf( "%%error: invalid scaling, use default.\n" );
            }
        }
        else {
            break;
        }
    }
    printf("dwitermax=%d dweight=%e mscale=%d\n", dwitermax, MAGMA_Z_REAL(diagweight), scaling );
    
        //################################################################//
        //                      read matrix from file                     //
        //################################################################//
    while( inp < argc ) {
        if ( strcmp("LAPLACE2D", argv[inp]) == 0 && inp+1 < argc ) {   // Laplace test
            inp++;
            magma_int_t laplace_size = atoi( argv[inp] );
            TESTING_CHECK( magma_zm_5stencil(  laplace_size, &hA, queue ));
            inp++;
        } else {                        // file-matrix test
            TESTING_CHECK( magma_z_csr_mtx( &hA,  argv[inp], queue ));
            inp++;
        }

    // optional scaling
    if (scaling != Magma_NOSCALE) {
        TESTING_CHECK( magma_zmscale( &hA, scaling, queue ) );
    }

        //################################################################//
        //                  cuSPARSE reference ILU                        //
        //################################################################//
        
        real_Double_t cunonlinres = 0.0;
        real_Double_t cuilures = 0.0;

        magma_z_mtransfer( hA, &dA, Magma_CPU, Magma_DEV, queue );
        // CUSPARSE context //
        cusparseHandle_t cusparseHandle;
        cusparseStatus_t cusparseStatus;
        cusparseStatus = cusparseCreate(&cusparseHandle);
        if (cusparseStatus != 0)    printf("error in Handle.\n");
        cusparseMatDescr_t descrA;
        cusparseStatus = cusparseCreateMatDescr(&descrA);
        if (cusparseStatus != 0)    printf("error in MatrDescr.\n");
        cusparseStatus =
        cusparseSetMatType(descrA,CUSPARSE_MATRIX_TYPE_GENERAL);
        if (cusparseStatus != 0)    printf("error in MatrType.\n");
        cusparseStatus =
        cusparseSetMatDiagType (descrA, CUSPARSE_DIAG_TYPE_NON_UNIT);
        if (cusparseStatus != 0)    printf("error in DiagType.\n");
        cusparseStatus =
        cusparseSetMatIndexBase(descrA,CUSPARSE_INDEX_BASE_ZERO);
        if (cusparseStatus != 0)    printf("error in IndexBase.\n");
        cusparseSolveAnalysisInfo_t info;
        cusparseStatus =
        cusparseCreateSolveAnalysisInfo(&info);
        if (cusparseStatus != 0)    printf("error in info.\n");
        start = magma_sync_wtime( queue );
        cusparseStatus =
        cusparseZcsrsv_analysis( cusparseHandle, 
                                 CUSPARSE_OPERATION_NON_TRANSPOSE, 
                                 dA.num_rows, dA.nnz, descrA,
                                 dA.val, dA.row, dA.col, info); 
        if (cusparseStatus != 0)    printf("error in analysis.\n");
        cusparseStatus =
        cusparseZcsrilu0( cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, 
                          dA.num_rows, descrA, 
                         (magmaDoubleComplex*) dA.val, (const int *) dA.row, 
                         (const int *) dA.col, info);
        if (cusparseStatus != 0)    printf("error in ILU.\n");
                  //magma_zprint_matrix(dA, queue); getchar();
        end = magma_sync_wtime( queue );
        t_cusparse = end-start;

        cusparseDestroySolveAnalysisInfo( info );
        if (cusparseStatus != 0)    printf("error in info-free.\n");

        // end CUSPARSE context //
        magma_z_mtransfer( dA, &hAcusparse, Magma_DEV, Magma_CPU, queue );
        magma_z_mconvert( hAcusparse, &hLCU, Magma_CSR, Magma_CSRL, queue );
        magma_z_mconvert( hAcusparse, &hUCU, Magma_CSR, Magma_CSRU, queue );
        
        magma_zilures(   hA, hLCU, hUCU, &hLU, &cuilures, &cunonlinres, queue ); 
        magma_z_mfree( &hLCU, queue );
        magma_z_mfree( &hUCU, queue );
        magma_z_mfree( &hLU, queue );

        //################################################################//
        //                  end cuSPARSE reference ILU                    //
        //################################################################//
    // reorder the matrix determining the update processing order
    magma_z_sparse_matrix hAcopy, hACSRCOO, dAinitguess;
    magma_z_mtransfer( hA, &hAcopy, Magma_CPU, Magma_CPU, queue );
    // --------------- initial error and residual -----------//
    magma_zsymbilu( &hAcopy, 0, &hAL, &hAU, queue ); 
    real_Double_t initres = 0.0;
    real_Double_t initilures = 0.0;
    real_Double_t initnonlinres = 0.0;
    magma_zmlumerge( hAL, hAU, &hAtmp, queue );
    // frobenius norm of error
    magma_zfrobenius( hAcusparse, hAtmp, &initres, queue );
    // ilu residual
    magma_zilures(   hA, hAL, hAU, &hLU, &initilures, &initnonlinres, queue ); 
    // free what we don't need any more
    magma_z_mfree( &hAtmp, queue );
    magma_z_mfree( &hAL, queue );
    magma_z_mfree( &hAU, queue );
    for( int localiters = 1; localiters < 2; localiters++){ //local iterations

//    printf("\n\nLaplace3D_%d = [\n", 32*(matrix-7));
    // ---------------- iteration matrices ------------------- //
    // possibility to increase fill-in in ILU-(m)
    //ILU-m levels
    for( int levels = 0; levels < 1; levels++){ //ILU-m levels
    //{int levels = atoi( argv[1]);
    magma_z_mtransfer( hA, &hAcopy, Magma_CPU, Magma_CPU, queue );
    magma_zsymbilu( &hAcopy, levels, &hAL, &hAUt, queue ); 
    printf("ILU%d = [", levels);
    printf("\n%%#=======================================================================================================#\n");
    printf("%%#\t#nnz\titers\tbs\tILU-time\tParILU-time\tILU-ILUres\tParILU-ILUres\tParILU-nonlinres\t\tscaled \n");
    // add a unit diagonal to L for the algorithm
    magma_zmLdiagadd( &hAL, queue ); 

    // transpose U for the algorithm
    magma_z_cucsrtranspose(  hAUt, &hAU, queue );
    magma_z_mfree( &hAUt, queue );
    // scale to unit diagonal
    //magma_zmscale( &hAU, Magma_UNITDIAG, queue );


/*
    // need only lower triangular
    magma_z_mfree(&hAL);
    hAL.diagorder_type == Magma_UNITY;
    magma_z_mconvert( hA, &hAL, Magma_CSR, Magma_CSRL, queue );
*/

    // ---------------- initial guess ------------------- //
    magma_z_mconvert( hAcopy, &hACSRCOO, Magma_CSR, Magma_CSRCOO, queue );
    int blocksize = 1;
    //magma_zmreorder( hACSRCOO, n, blocksize, blocksize, blocksize, &hAinitguess, queue );
    //magma_z_mfree(&hAinitguess);
    magma_z_mtransfer( hACSRCOO, &dAinitguess, Magma_CPU, Magma_DEV, queue );
    magma_z_mfree(&hACSRCOO, queue );


        //################################################################//
        //                        iterative ILU                           //
        //################################################################//
    
    // weight diagonal    
    magma_zmdiagadd( &hAU, diagweight, queue);    
    // number of AILU sweeps
    for(int iters=0; iters<101; iters+=1){
        // reduce diagonal weight
        if (iters > 0 && iters < dwitermax) {
            magmaDoubleComplex dwincr = MAGMA_Z_MAKE( -MAGMA_Z_REAL(diagweight) / (double)(dwitermax), 0.0 );
            magma_zmdiagadd( &hAU, dwincr, queue); 
        }
    // take average results for residuals
    real_Double_t resavg = 0.0;
    real_Double_t iluresavg = 0.0;
    real_Double_t nonlinresavg = 0.0;
    int nnz, numavg = 1;
    //multiple runs
    for (int z = 0; z < numavg; z++) {
        real_Double_t res = 0.0;
        real_Double_t ilures = 0.0;
        real_Double_t nonlinres = 0.0;

        // transfer the factor L and U
        magma_z_mtransfer( hAL, &dL, Magma_CPU, Magma_DEV, queue );
        magma_z_mtransfer( hAU, &dU, Magma_CPU, Magma_DEV, queue );

        // iterative ILU embedded in timing
        start = magma_sync_wtime( queue );
        for(int i=0; i<iters; i++){
cudaProfilerStart();
            magma_zparilu_csr( dAinitguess, dL, dU, queue );
cudaProfilerStop();
        }
        end = magma_sync_wtime( queue );
        t_chow = end-start;

        // check the residuals
        magma_z_mtransfer( dL, &hL, Magma_DEV, Magma_CPU, queue );
        magma_z_mtransfer( dU, &hU, Magma_DEV, Magma_CPU, queue );
        magma_z_cucsrtranspose(  hU, &hUT, queue );

        magma_z_mfree(&dL, queue );
        magma_z_mfree(&dU, queue );

        magma_zmlumerge( hL, hUT, &hAtmp, queue );
        // frobenius norm of error
        magma_zfrobenius( hAcusparse, hAtmp, &res, queue );
         //magma_zprint_matrix(hAtmp, queue); getchar();
        // ilu residual
        magma_zilures(   hA, hL, hUT, &hLU, &ilures, &nonlinres, queue ); 

        iluresavg += ilures;
        resavg += res;
        nonlinresavg += nonlinres;
        nnz = hAtmp.nnz;

        magma_z_mfree( &hL, queue );
        magma_z_mfree( &hU, queue );
        magma_z_mfree( &hUT, queue );
        magma_z_mfree( &hAtmp, queue );
    }//multiple runs

    iluresavg = iluresavg/numavg;
    resavg = resavg/numavg;
    nonlinresavg = nonlinresavg/numavg;

    printf(" %d\t%d\t%d\t%d\t%.2e\t",
                              levels, nnz, 1* iters, blocksize, t_cusparse);
    printf(" %.2e\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t\n", 
    t_chow, cuilures, iluresavg, nonlinresavg, iluresavg/initilures, nonlinresavg/initnonlinres);


 //   printf(" %.2e  &  ", iluresavg);    
    }// iters
    magma_z_mfree( &hAcopy, queue );
    printf("\n%%#=======================================================================================================#\n]; \n");
    }// levels

    }// localiters

    // free all memory
    magma_z_mfree( &hAL, queue );
    magma_z_mfree( &hAU, queue );
    magma_z_mfree( &hAcusparse, queue );
    magma_z_mfree( &dA, queue );
    magma_z_mfree( &dAinitguess, queue );
    magma_z_mfree( &hA, queue );

    }// multiple matrices

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return 0;
}
