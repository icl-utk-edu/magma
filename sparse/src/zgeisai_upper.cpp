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

#define PRECISION_z


/***************************************************************************//**
    Purpose
    -------

    Prepares Incomplete LU preconditioner using a sparse approximate inverse
    instead of sparse triangular solves.
    
    This routine only handles the upper triangular part.


    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input matrix A

    @param[in]
    S           magma_z_matrix
                pattern for the ISAI preconditioner

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
magma_ziluisaisetup_upper(
    magma_z_matrix A,
    magma_z_matrix S,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // real_Double_t start, end;

    magma_index_t *sizes_h = NULL;
    magmaDoubleComplex *rhs_d = NULL;
    magma_index_t *sizes_d = NULL, *locations_d = NULL;
    magma_int_t maxsize, nnzloc, nnzL=0, nnzU=0;
    magma_z_matrix MT={Magma_CSR};
    magmaDoubleComplex *trisystems_d = NULL;

    int warpsize=32;
    int offset = 0; // can be changed to better match the matrix structure
    magma_int_t z;

#if (CUDA_VERSION <= 6000) // this won't work, just to have something...
    printf( "%% error: ISAI preconditioner requires CUDA > 6.0.\n" );
    info = MAGMA_ERR_NOT_SUPPORTED;
    goto cleanup;
#endif

   // we need this in any case as the ISAI matrix is generated in transpose fashion
   CHECK( magma_zmtranspose( S, &MT, queue ) );

    CHECK( magma_index_malloc_cpu( &sizes_h, A.num_rows+1 ) );
    // only needed in case the systems are generated in GPU main memory
    // CHECK( magma_index_malloc( &sizes_d, A.num_rows ) );
    // CHECK( magma_index_malloc( &locations_d, A.num_rows*warpsize ) );
    // CHECK( magma_zmalloc( &trisystems_d, min(320000,A.num_rows) *warpsize*warpsize ) ); // fixed size - go recursive
    // CHECK( magma_zmalloc( &rhs_d, A.num_rows*warpsize ) );
    #pragma omp parallel for
    for( magma_int_t i=0; i<A.num_rows; i++ ){
            maxsize = sizes_h[i] = 0;
    }
    magma_index_getvector( S.num_rows+1, S.drow, 1, sizes_h, 1, queue );
    maxsize = 0;
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        nnzloc = sizes_h[i+1]-sizes_h[i];
        nnzL+= nnzloc;
        if( nnzloc > maxsize ){
            maxsize = sizes_h[i+1]-sizes_h[i];
        }
        if( maxsize > warpsize ){
            printf("%%   error for ISAI L: size of system %d is too large by %d\n", (int) i, (int) (maxsize-32));
            printf("%% fallback: use exact triangular solve (cuSOLVE)\n");
            precond->trisolver = Magma_CUSOLVE;
            goto cleanup;
        }
    }

    printf("%% nnz in L-ISAI (total max/row): %d %d\n", (int) nnzL, (int) maxsize);
    // via registers
     CHECK( magma_zisai_generator_regs( MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                    precond->U, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );

    CHECK( magma_zmtranspose( MT, &precond->UD, queue ) );

cleanup:
    magma_free_cpu( sizes_h );
    magma_zmfree( &MT, queue );
    return info;
}

