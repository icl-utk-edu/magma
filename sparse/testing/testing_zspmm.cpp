/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef MAGMA_WITH_MKL
    #include "mkl_spblas.h"
    
    #define PRECISION_z
    #if defined(PRECISION_z)
    #define MKL_ADDR(a) (MKL_Complex16*)(a)
    #elif defined(PRECISION_c)
    #define MKL_ADDR(a) (MKL_Complex8*)(a)
    #else
    #define MKL_ADDR(a) (a)
    #endif
#endif

// includes, project
#include "magma_v2.h"
#include "magmasparse.h"
#include "magma_lapack.h"
#include "testings.h"

#if CUDA_VERSION >= 12000
  #define CUSPARSE_CSRMV_ALG2 CUSPARSE_SPMV_CSR_ALG2
  #define CUSPARSE_CSRMV_ALG1 CUSPARSE_SPMV_CSR_ALG1
  #define CUSPARSE_CSRMM_ALG1 CUSPARSE_SPMM_CSR_ALG1
#endif

#if CUDA_VERSION >= 11000
#define cusparseZcsrmm(handle, op, rows, num_vecs, cols, nnz, alpha, descr, dval, drow, dcol,   \
                       x, ldx, beta, y, ldy)                                                    \
    {                                                                                           \
        cusparseSpMatDescr_t descrA;                                                            \
        cusparseDnMatDescr_t descrX, descrY;                                                    \
        cusparseCreateCsr(&descrA, rows, cols, nnz,                                             \
                          (void *)drow, (void *)dcol, (void *)dval,                             \
                          CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,                               \
                          CUSPARSE_INDEX_BASE_ZERO, CUDA_C_64F);                                \
        cusparseCreateDnMat(&descrX, cols, num_vecs, ldx, x, CUDA_C_64F, CUSPARSE_ORDER_COL);   \
        cusparseCreateDnMat(&descrY, cols, num_vecs, ldy, y, CUDA_C_64F, CUSPARSE_ORDER_COL);   \
                                                                                                \
        size_t bufsize;                                                                         \
        void *buf;                                                                              \
        cusparseSpMM_bufferSize(handle, op, CUSPARSE_OPERATION_NON_TRANSPOSE,                   \
                                (void *)alpha, descrA, descrX, beta, descrY, CUDA_C_64F,        \
                                CUSPARSE_CSRMM_ALG1, &bufsize);                                 \
        if (bufsize > 0)                                                                        \
           magma_malloc(&buf, bufsize);                                                         \
        cusparseSpMM(handle, op, CUSPARSE_OPERATION_NON_TRANSPOSE,                              \
                     (void *)alpha, descrA, descrX, beta, descrY, CUDA_C_64F,                   \
                     CUSPARSE_CSRMM_ALG1, buf);                                                 \
        if (bufsize > 0)                                                                        \
           magma_free(buf);                                                                     \
    }
#endif

#define PRECISION_z

/* For hipSPARSE, they use a separate complex type than for hipBLAS */
#if defined(MAGMA_HAVE_HIP)
  #ifdef PRECISION_z
    #define hipblasDoubleComplex hipDoubleComplex
#elif defined(PRECISION_c)
    #define hipblasComplex hipComplex
  #endif
#endif

/* ////////////////////////////////////////////////////////////////////////////
   -- testing sparse matrix vector product
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    
    magma_z_matrix hA={Magma_CSR}, hA_SELLP={Magma_CSR}, 
    dA={Magma_CSR}, dA_SELLP={Magma_CSR};
    
    magma_z_matrix hx={Magma_CSR}, hy={Magma_CSR}, dx={Magma_CSR}, 
    dy={Magma_CSR}, hrefvec={Magma_CSR}, hcheck={Magma_CSR};
        
    hA_SELLP.blocksize = 8;
    hA_SELLP.alignment = 8;
    real_Double_t start, end, res;
    #ifdef MAGMA_WITH_MKL
        magma_int_t *pntre=NULL;
    #endif
    #ifdef MAGMA_HAVE_CUDA
    cusparseHandle_t cusparseHandle = NULL;
    cusparseMatDescr_t descr = NULL;
    #endif
    magmaDoubleComplex c_one  = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex c_zero = MAGMA_Z_MAKE(0.0, 0.0);
    
    double accuracy = 1e-10;
    
    #define PRECISION_z
    #if defined(PRECISION_c)
        accuracy = 1e-4;
    #endif
    #if defined(PRECISION_s)
        accuracy = 1e-4;
    #endif
    
    magma_int_t i, j;
    for( i = 1; i < argc; ++i ) {
        if ( strcmp("--blocksize", argv[i]) == 0 ) {
            hA_SELLP.blocksize = atoi( argv[++i] );
        } else if ( strcmp("--alignment", argv[i]) == 0 ) {
            hA_SELLP.alignment = atoi( argv[++i] );
        } else
            break;
    }
    printf("\n#    usage: ./run_zspmm"
           " [ --blocksize %lld --alignment %lld (for SELLP) ] matrices\n\n",
           (long long) hA_SELLP.blocksize, (long long) hA_SELLP.alignment );

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_zm_5stencil(  laplace_size, &hA, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_z_csr_mtx( &hA,  argv[i], queue ));
        }

        printf("%% matrix info: %lld-by-%lld with %lld nonzeros\n",
                (long long) hA.num_rows, (long long) hA.num_cols, (long long) hA.nnz );

        real_Double_t FLOPS = 2.0*hA.nnz/1e9;



        // m - number of rows for the sparse matrix
        // n - number of vectors to be multiplied in the SpMM product
        magma_int_t m, n;

        m = hA.num_rows;
        n = 48;

        // init CPU vectors
        TESTING_CHECK( magma_zvinit( &hx, Magma_CPU, m, n, c_one, queue ));
        TESTING_CHECK( magma_zvinit( &hy, Magma_CPU, m, n, c_zero, queue ));

        // init DEV vectors
        TESTING_CHECK( magma_zvinit( &dx, Magma_DEV, m, n, c_one, queue ));
        TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, m, n, c_zero, queue ));


        // calling MKL with CSR
        #ifdef MAGMA_WITH_MKL
            TESTING_CHECK( magma_imalloc_cpu( &pntre, m + 1 ) );
            pntre[0] = 0;
            for (j=0; j < m; j++ ) {
                pntre[j] = hA.row[j+1];
            }

            MKL_INT num_rows = hA.num_rows;
            MKL_INT num_cols = hA.num_cols;
            MKL_INT nnz = hA.nnz;
            MKL_INT num_vecs = n;

            MKL_INT *col;
            TESTING_CHECK( magma_malloc_cpu( (void**) &col, nnz * sizeof(MKL_INT) ));
            for( magma_int_t t=0; t < hA.nnz; ++t ) {
                col[ t ] = hA.col[ t ];
            }
            MKL_INT *row;
            TESTING_CHECK( magma_malloc_cpu( (void**) &row, num_rows * sizeof(MKL_INT) ));
            for( magma_int_t t=0; t < hA.num_rows; ++t ) {
                row[ t ] = hA.col[ t ];
            }

            // === Call MKL with consecutive SpMVs, using mkl_zcsrmv ===
            // warmp up
            mkl_zcsrmv( "N", &num_rows, &num_cols,
                        MKL_ADDR(&c_one), "GFNC", MKL_ADDR(hA.val), col, row, pntre,
                                                  MKL_ADDR(hx.val),
                        MKL_ADDR(&c_zero),        MKL_ADDR(hy.val) );
    
            start = magma_wtime();
            for (j=0; j < 10; j++ ) {
                mkl_zcsrmv( "N", &num_rows, &num_cols,
                            MKL_ADDR(&c_one), "GFNC", MKL_ADDR(hA.val), col, row, pntre,
                                                      MKL_ADDR(hx.val),
                            MKL_ADDR(&c_zero),        MKL_ADDR(hy.val) );
            }
            end = magma_wtime();
            printf( "\n > MKL SpMVs : %.2e seconds %.2e GFLOP/s    (CSR).\n",
                                            (end-start)/10, FLOPS*10/(end-start) );
    
            // === Call MKL with blocked SpMVs, using mkl_zcsrmm ===
            char transa = 'n';
            MKL_INT ldb = n, ldc=n;
            char matdescra[6] = {'g', 'l', 'n', 'c', 'x', 'x'};
    
            // warm up
            mkl_zcsrmm( &transa, &num_rows, &num_vecs, &num_cols, MKL_ADDR(&c_one), matdescra,
                        MKL_ADDR(hA.val), col, row, pntre,
                        MKL_ADDR(hx.val), &ldb,
                        MKL_ADDR(&c_zero),
                        MKL_ADDR(hy.val), &ldc );
    
            start = magma_wtime();
            for (j=0; j < 10; j++ ) {
                mkl_zcsrmm( &transa, &num_rows, &num_vecs, &num_cols, MKL_ADDR(&c_one), matdescra,
                            MKL_ADDR(hA.val), col, row, pntre,
                            MKL_ADDR(hx.val), &ldb,
                            MKL_ADDR(&c_zero),
                            MKL_ADDR(hy.val), &ldc );
            }
            end = magma_wtime();
            printf( "\n > MKL SpMM  : %.2e seconds %.2e GFLOP/s    (CSR).\n",
                    (end-start)/10, FLOPS*10.*n/(end-start) );

            magma_free_cpu( row );
            magma_free_cpu( col );
            row = NULL;
            col = NULL;

        #endif // MAGMA_WITH_MKL

        // copy matrix to GPU
        TESTING_CHECK( magma_zmtransfer( hA, &dA, Magma_CPU, Magma_DEV, queue ));
        // SpMV on GPU (CSR)
        start = magma_sync_wtime( queue );
        for (j=0; j < 10; j++) {
            TESTING_CHECK( magma_z_spmv( c_one, dA, dx, c_zero, dy, queue ));
        }
        end = magma_sync_wtime( queue );
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (standard CSR).\n",
                                        (end-start)/10, FLOPS*10.*n/(end-start) );

        TESTING_CHECK( magma_zmtransfer( dy, &hrefvec , Magma_DEV, Magma_CPU, queue ));
        magma_zmfree(&dA, queue );

        #ifdef MAGMA_HAVE_CUDA
        // convert to SELLP and copy to GPU
        TESTING_CHECK( magma_zmconvert(  hA, &hA_SELLP, Magma_CSR, Magma_SELLP, queue ));
        TESTING_CHECK( magma_zmtransfer( hA_SELLP, &dA_SELLP, Magma_CPU, Magma_DEV, queue ));
        magma_zmfree(&hA_SELLP, queue );
        magma_zmfree( &dy, queue );
        TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, dx.num_rows, dx.num_cols, c_zero, queue ));
        // SpMV on GPU (SELLP)
        start = magma_sync_wtime( queue );
        for (j=0; j < 10; j++) {
            TESTING_CHECK( magma_z_spmv( c_one, dA_SELLP, dx, c_zero, dy, queue ));
        }
        end = magma_sync_wtime( queue );
        printf( " > MAGMA: %.2e seconds %.2e GFLOP/s    (SELLP).\n",
                                        (end-start)/10, FLOPS*10.*n/(end-start) );

        TESTING_CHECK( magma_zmtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue ));
        res = 0.0;
        for(magma_int_t k=0; k < hA.num_rows; k++ ) {
            res=res + MAGMA_Z_REAL(hcheck.val[k]) - MAGMA_Z_REAL(hrefvec.val[k]);
        }
        printf("%% |x-y|_F = %8.2e\n", res);
        if ( res < accuracy )
            printf("%% tester spmm SELL-P:  ok\n");
        else
            printf("%% tester spmm SELL-P:  failed\n");
        magma_zmfree( &hcheck, queue );
        magma_zmfree(&dA_SELLP, queue );


        // SpMV on GPU (CUSPARSE - CSR)
        // CUSPARSE context //
        magma_zmfree( &dy, queue );
        TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, dx.num_rows, dx.num_cols, c_zero, queue ));
        //#ifdef PRECISION_d
        start = magma_sync_wtime( queue );
        TESTING_CHECK( cusparseCreate( &cusparseHandle ));
        TESTING_CHECK( cusparseSetStream( cusparseHandle, magma_queue_get_cuda_stream(queue) ));
        TESTING_CHECK( cusparseCreateMatDescr( &descr ));
        TESTING_CHECK( cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL ));
        TESTING_CHECK( cusparseSetMatIndexBase( descr, CUSPARSE_INDEX_BASE_ZERO ));
        magmaDoubleComplex alpha = c_one;
        magmaDoubleComplex beta = c_zero;

        // copy matrix to GPU
        TESTING_CHECK( magma_zmtransfer( hA, &dA, Magma_CPU, Magma_DEV, queue) );

        for (j=0; j < 10; j++) {
            cusparseZcsrmm(cusparseHandle,
                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                           dA.num_rows,   n, dA.num_cols, dA.nnz,
                           (cuDoubleComplex*)&alpha, descr, (cuDoubleComplex*)dA.dval, dA.drow, dA.dcol,
                           (cuDoubleComplex*)dx.dval, dA.num_cols, (cuDoubleComplex*)&beta, 
                           (cuDoubleComplex*)dy.dval, dA.num_cols);
        }
        end = magma_sync_wtime( queue );
        printf( " > CUSPARSE: %.2e seconds %.2e GFLOP/s    (CSR).\n",
                                        (end-start)/10, FLOPS*10*n/(end-start) );

        TESTING_CHECK( magma_zmtransfer( dy, &hcheck , Magma_DEV, Magma_CPU, queue ));
        res = 0.0;
        for(magma_int_t k=0; k < hA.num_rows; k++ ) {
            res = res + MAGMA_Z_REAL(hcheck.val[k]) - MAGMA_Z_REAL(hrefvec.val[k]);
        }
        printf("%% |x-y|_F = %8.2e\n", res);
        if ( res < accuracy )
            printf("%% tester spmm cuSPARSE:  ok\n");
        else
            printf("%% tester spmm cuSPARSE:  failed\n");
        magma_zmfree( &hcheck, queue );

        cusparseDestroyMatDescr( descr ); 
        cusparseDestroy( cusparseHandle );
        descr = NULL;
        cusparseHandle = NULL;
        //#endif

        printf("\n\n");
        #endif

        // free CPU memory
        magma_zmfree( &hA, queue );
        magma_zmfree( &hx, queue );
        magma_zmfree( &hy, queue );
        magma_zmfree( &hrefvec, queue );
        // free GPU memory
        magma_zmfree( &dx, queue );
        magma_zmfree( &dy, queue );
        magma_zmfree( &dA, queue);

        #ifdef MAGMA_WITH_MKL
            magma_free_cpu( pntre );
        #endif
        
        i++;
    }

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
