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
magma_ziluisaisetup(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // real_Double_t start, end;

    magmaDoubleComplex *trisystems_d = NULL;
    magmaDoubleComplex *rhs_d = NULL;
    magma_index_t *sizes_d = NULL, *locations_d = NULL;
    magma_index_t *sizes_h = NULL;
    magma_int_t maxsize, nnzloc, nnzL=0, nnzU=0;
    int warpsize=32;
    int offset = 0; // can be changed to better match the matrix structure
    magma_z_matrix LT={Magma_CSR}, MT={Magma_CSR}, QT={Magma_CSR};
    magma_int_t z;
    // magma_int_t timing = 1;

#if (CUDA_VERSION <= 6000) // this won't work, just to have something...
    printf( "%% error: ISAI preconditioner requires CUDA > 6.0.\n" );
    info = MAGMA_ERR_NOT_SUPPORTED;
    goto cleanup;
#endif

    CHECK( magma_index_malloc_cpu( &sizes_h, A.num_rows+1 ) );
    // only needed in case the systems are generated in GPU main memory
    // CHECK( magma_index_malloc( &sizes_d, A.num_rows ) );
    // CHECK( magma_index_malloc( &locations_d, A.num_rows*warpsize ) );
    // CHECK( magma_zmalloc( &trisystems_d, min(320000,A.num_rows) *warpsize*warpsize ) ); // fixed size - go recursive
    // CHECK( magma_zmalloc( &rhs_d, A.num_rows*warpsize ) );

    for( magma_int_t i=0; i<A.num_rows; i++ ){
            maxsize = sizes_h[i] = 0;
    }
    // ILU setup
    // CHECK( magma_zcumilusetup( A, precond, queue ) );

    // we need this in any case
    CHECK( magma_zmtranspose( precond->L, &MT, queue ) );

    // SPAI for L
    if( precond->trisolver == Magma_JACOBI ){ // block diagonal structure
        if( precond->pattern == 0 ){
            precond->pattern = 1;
        }
        CHECK( magma_zmisai_blockstruct( A.num_rows, precond->pattern, offset, MagmaLower, &MT, queue ) );
        CHECK( magma_z_mtransfer( MT, &QT, Magma_CPU, Magma_DEV, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmtranspose( QT, &MT, queue ) );
        magma_zmfree( &QT, queue );
    } else if( precond->trisolver == Magma_VBJACOBI ){ // block diagonal structure with variable blocksize
        CHECK( magma_z_mtransfer( A, &QT, A.memory_location, Magma_CPU, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmsupernodal( &precond->pattern, QT, &MT, queue ) );
        magma_zmfree( &QT, queue );
        CHECK( magma_zmconvert( MT, &QT, Magma_CSR, Magma_CSRL, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmconvert( QT, &MT, Magma_CSR, Magma_CSR, queue ) );
        magma_zmfree( &QT, queue );
        CHECK( magma_z_mtransfer( MT, &QT, Magma_CPU, Magma_DEV, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmtranspose( QT, &MT, queue ) );
        magma_zmfree( &QT, queue );

    } else if( precond->trisolver == Magma_ISAI ){
        if( precond->pattern == 100 ){
            CHECK( magma_z_mtransfer( MT, &LT, Magma_DEV, Magma_DEV, queue ) );
            CHECK( magma_zgeisai_maxblock( LT, &MT, queue ) );
        } else {
            // pattern L^x
            if( precond->pattern > 1 ){
                CHECK( magma_z_mtransfer( MT, &LT, Magma_DEV, Magma_DEV, queue ) );
                z = 1;
                while( z<precond->pattern ){
                    CHECK( magma_z_spmm( MAGMA_Z_ONE, LT, MT, &QT, queue ) );
                    magma_zmfree( &MT, queue );
                    CHECK( magma_z_mtransfer( QT, &MT, Magma_DEV, Magma_DEV, queue ) );
                    magma_zmfree( &QT, queue );
                    z++;
                }
            }
        }
    } else{
        printf("%% error: pattern not supported.\n" );
    }
    magma_index_getvector( A.num_rows+1, MT.drow, 1, sizes_h, 1, queue );
    maxsize = 0;
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        nnzloc = sizes_h[i+1]-sizes_h[i];
        nnzL+= nnzloc;
        if( nnzloc > maxsize ){
            maxsize = sizes_h[i+1]-sizes_h[i];
        }
        if( maxsize > warpsize ){
            printf("%%   error for ISAI: size of system %d is too large by %d\n", i, maxsize-32);
            break;
        }
    }

    printf("%% nnz in L-ISAI (total max/row): %d %d\n", nnzL, maxsize);
    // this can be modified to the thread-block-size
    if( maxsize > warpsize ){
       info = -(maxsize - warpsize);
       goto cleanup;
    }
    // via main memory
    //  CHECK( magma_z_mtransfer( MT, &LT, Magma_DEV, Magma_DEV, queue ) );
    // if( maxsize <= 8 ){
    //     CHECK( magma_zisaigenerator_8_gpu( MagmaLower, MagmaNoTrans, MagmaNonUnit,
    //                 LT, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );
    // } else if( maxsize <= 16 ){
    //     CHECK( magma_zisaigenerator_16_gpu( MagmaLower, MagmaNoTrans, MagmaNonUnit,
    //                 LT, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );
    // } else {
    //     CHECK( magma_zisaigenerator_32_gpu( MagmaLower, MagmaNoTrans, MagmaNonUnit,
    //                 LT, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );
    // }
    // via registers
     CHECK( magma_zisai_generator_regs( MagmaLower, MagmaNoTrans, MagmaNonUnit,
                    precond->L, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );

    CHECK( magma_zmtranspose( MT, &precond->LD, queue ) );
    magma_zmfree( &LT, queue );
    magma_zmfree( &MT, queue );
    // magma_z_mvisu(precond->LD, queue);

   // we need this in any case
   CHECK( magma_zmtranspose( precond->U, &MT, queue ) );

    // SPAI for U
    if( precond->trisolver == Magma_JACOBI ){ // block diagonal structure
        if( precond->pattern == 0 ){
            precond->pattern = 1;
        }
        CHECK( magma_zmisai_blockstruct( A.num_rows, precond->pattern, offset, MagmaUpper, &MT, queue ) );
        CHECK( magma_z_mtransfer( MT, &QT, Magma_CPU, Magma_DEV, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmtranspose( QT, &MT, queue ) );
        magma_zmfree( &QT, queue );

    } else if( precond->trisolver == Magma_VBJACOBI ){ // block diagonal structure with variable blocksize
        CHECK( magma_z_mtransfer( A, &QT, A.memory_location, Magma_CPU, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmsupernodal( &precond->pattern, QT, &MT, queue ) );
        magma_zmfree( &QT, queue );
        CHECK( magma_zmconvert( MT, &QT, Magma_CSR, Magma_CSRU, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmconvert( QT, &MT, Magma_CSR, Magma_CSR, queue ) );
        magma_zmfree( &QT, queue );
        CHECK( magma_z_mtransfer( MT, &QT, Magma_CPU, Magma_DEV, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmtranspose( QT, &MT, queue ) );
        magma_zmfree( &QT, queue );
    } else if( precond->trisolver == Magma_ISAI ){
        if( precond->pattern == 100 ){
            CHECK( magma_z_mtransfer( MT, &LT, Magma_DEV, Magma_DEV, queue ) );
            CHECK( magma_zgeisai_maxblock( LT, &MT, queue ) );
        } else {
        // pattern U^x
            if( precond->pattern > 1 ){
                CHECK( magma_z_mtransfer( MT, &LT, Magma_DEV, Magma_DEV, queue ) );
                z = 1;
                while( z<precond->pattern ){
                    CHECK( magma_z_spmm( MAGMA_Z_ONE, LT, MT, &QT, queue ) );
                    magma_zmfree( &MT, queue );
                    CHECK( magma_z_mtransfer( QT, &MT, Magma_DEV, Magma_DEV, queue ) );
                    magma_zmfree( &QT, queue );
                    z++;
                }
            }
        }
    }
    magma_index_getvector( A.num_rows+1, MT.drow, 1, sizes_h, 1, queue );
    maxsize = 0;
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        nnzloc = sizes_h[i+1]-sizes_h[i];
        nnzU+= nnzloc;
        if( nnzloc > maxsize ){
            maxsize = sizes_h[i+1]-sizes_h[i];
        }
        if( maxsize > warpsize ){
            printf("%%   error for ISAI: size of system %d is too large by %d\n", i, maxsize-32);
            break;
        }
    }
    printf("%% nnz in U-ISAI (total max/row): %d %d\n", nnzU, maxsize);
    // this can be modified to the thread-block-size
    if( maxsize > warpsize ){
       info = -(maxsize - warpsize);
       goto cleanup;
    }
    // via main memory
    // CHECK( magma_z_mtransfer( MT, &LT, Magma_DEV, Magma_DEV, queue ) );
    // if( maxsize <= 8 ){
    //     CHECK( magma_zisaigenerator_8_gpu( MagmaUpper, MagmaNoTrans, MagmaNonUnit,
    //                 LT, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );
    // } else if( maxsize <= 16 ){
    //     CHECK( magma_zisaigenerator_16_gpu( MagmaUpper, MagmaNoTrans, MagmaNonUnit,
    //                 LT, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );
    // } else {
    //     CHECK( magma_zisaigenerator_32_gpu( MagmaUpper, MagmaNoTrans, MagmaNonUnit,
    //                 LT, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );
    // }
    // via registers
     CHECK( magma_zisai_generator_regs( MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                    precond->U, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );

     CHECK( magma_zmtranspose( MT, &precond->UD, queue ) );
     // magma_z_mvisu( precond->UD, queue );

cleanup:
    // magma_free( sizes_d );
    magma_free_cpu( sizes_h );
    // magma_free( locations_d );
    // magma_free( trisystems_d );
    // magma_free( rhs_d );
    magma_zmfree( &LT, queue );
    magma_zmfree( &MT, queue );
    magma_zmfree( &QT, queue );

    return info;
}




/***************************************************************************//**
    Purpose
    -------

    Prepares Incomplete LU preconditioner using a sparse approximate inverse
    instead of sparse triangular solves. This is the transpose preconditioner
    setup needed e.g. for BiCG, QMR, LSQR...


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
magma_ziluisaisetup_t(
    magma_z_matrix A,
    magma_z_matrix b,
    magma_z_preconditioner *precond,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    // real_Double_t start, end;

    magmaDoubleComplex *trisystems_d = NULL;
    magmaDoubleComplex *rhs_d = NULL;
    magma_index_t *sizes_d = NULL, *locations_d = NULL;
    magma_index_t *sizes_h = NULL;
    magma_int_t maxsize, nnzloc, nnzL=0, nnzU=0;
    int warpsize=32;
    int offset = 0; // can be changed to better match the matrix structure
    magma_z_matrix LT={Magma_CSR}, MT={Magma_CSR}, QT={Magma_CSR};
    magma_int_t z;
    // magma_int_t timing = 1;

#if (CUDA_VERSION <= 6000) // this won't work, just to have something...
    printf( "%% error: ISAI preconditioner requires CUDA > 6.0.\n" );
    info = MAGMA_ERR_NOT_SUPPORTED;
    goto cleanup;
#endif

    CHECK( magma_index_malloc_cpu( &sizes_h, A.num_rows+1 ) );

    for( magma_int_t i=0; i<A.num_rows; i++ ){
            maxsize = sizes_h[i] = 0;
    }
    // ILU setup

    // we need this in any case
    CHECK( magma_zmtranspose( precond->L, &MT, queue ) );

    // SPAI for L
    if( precond->trisolver == Magma_JACOBI ){ // block diagonal structure
        if( precond->pattern == 0 ){
            precond->pattern = 1;
        }
        CHECK( magma_zmisai_blockstruct( A.num_rows, precond->pattern, offset, MagmaLower, &MT, queue ) );
        CHECK( magma_z_mtransfer( MT, &QT, Magma_CPU, Magma_DEV, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmtranspose( QT, &MT, queue ) );
        magma_zmfree( &QT, queue );
    } else if( precond->trisolver == Magma_VBJACOBI ){ // block diagonal structure with variable blocksize
        CHECK( magma_z_mtransfer( A, &QT, A.memory_location, Magma_CPU, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmsupernodal( &precond->pattern, QT, &MT, queue ) );
        magma_zmfree( &QT, queue );
        CHECK( magma_zmconvert( MT, &QT, Magma_CSR, Magma_CSRL, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmconvert( QT, &MT, Magma_CSR, Magma_CSR, queue ) );
        magma_zmfree( &QT, queue );
        CHECK( magma_z_mtransfer( MT, &QT, Magma_CPU, Magma_DEV, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmtranspose( QT, &MT, queue ) );
        magma_zmfree( &QT, queue );
    } else if( precond->trisolver == Magma_ISAI ){
        if( precond->pattern == 100 ){
            CHECK( magma_z_mtransfer( MT, &LT, Magma_DEV, Magma_DEV, queue ) );
            CHECK( magma_zgeisai_maxblock( LT, &MT, queue ) );
        } else {
            // pattern L^x
            if( precond->pattern > 1 ){
                CHECK( magma_z_mtransfer( MT, &LT, Magma_DEV, Magma_DEV, queue ) );
                z = 1;
                while( z<precond->pattern ){
                    CHECK( magma_z_spmm( MAGMA_Z_ONE, LT, MT, &QT, queue ) );
                    magma_zmfree( &MT, queue );
                    CHECK( magma_z_mtransfer( QT, &MT, Magma_DEV, Magma_DEV, queue ) );
                    magma_zmfree( &QT, queue );
                    z++;
                }
            }
        }
    }
    magma_index_getvector( A.num_rows+1, MT.drow, 1, sizes_h, 1, queue );
    maxsize = 0;
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        nnzloc = sizes_h[i+1]-sizes_h[i];
        nnzL+= nnzloc;
        if( nnzloc > maxsize ){
            maxsize = sizes_h[i+1]-sizes_h[i];
        }
        if( maxsize > warpsize ){
            printf("%%   error for ISAI: size of system %d is too large by %d\n", i, maxsize-32);
            break;
        }
    }
    // this can be modified to the thread-block-size
    if( maxsize > warpsize ){
       info = -(maxsize - warpsize);
       goto cleanup;
    }
    // via registers
     CHECK( magma_zisai_generator_regs( MagmaLower, MagmaNoTrans, MagmaNonUnit,
                    precond->L, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );


    CHECK( magma_z_mtransfer( MT, &precond->LDT, Magma_CPU, Magma_DEV, queue ) );
    //CHECK( magma_zmtranspose( MT, &precond->LDT, queue ) );
    magma_zmfree( &LT, queue );
    magma_zmfree( &MT, queue );

   // we need this in any case
   CHECK( magma_zmtranspose( precond->U, &MT, queue ) );

    // SPAI for U
    if( precond->trisolver == Magma_JACOBI ){ // block diagonal structure

        if( precond->pattern == 0 ){
            precond->pattern = 1;
        }
        CHECK( magma_zmisai_blockstruct( A.num_rows, precond->pattern, offset, MagmaUpper, &MT, queue ) );
        CHECK( magma_z_mtransfer( MT, &QT, Magma_CPU, Magma_DEV, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmtranspose( QT, &MT, queue ) );
        magma_zmfree( &QT, queue );

    } else if( precond->trisolver == Magma_VBJACOBI ){ // block diagonal structure with variable blocksize
        CHECK( magma_z_mtransfer( A, &QT, A.memory_location, Magma_CPU, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmsupernodal( &precond->pattern, QT, &MT, queue ) );
        magma_zmfree( &QT, queue );
        CHECK( magma_zmconvert( MT, &QT, Magma_CSR, Magma_CSRU, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmconvert( QT, &MT, Magma_CSR, Magma_CSR, queue ) );
        magma_zmfree( &QT, queue );
        CHECK( magma_z_mtransfer( MT, &QT, Magma_CPU, Magma_DEV, queue ) );
        magma_zmfree( &MT, queue );
        CHECK( magma_zmtranspose( QT, &MT, queue ) );
        magma_zmfree( &QT, queue );
    } else if( precond->trisolver == Magma_ISAI ){
        if( precond->pattern == 100 ){
            CHECK( magma_z_mtransfer( MT, &LT, Magma_DEV, Magma_DEV, queue ) );
            CHECK( magma_zgeisai_maxblock( LT, &MT, queue ) );
        } else {
        // pattern U^x
            if( precond->pattern > 1 ){
                CHECK( magma_z_mtransfer( MT, &LT, Magma_DEV, Magma_DEV, queue ) );
                z = 1;
                while( z<precond->pattern ){
                    CHECK( magma_z_spmm( MAGMA_Z_ONE, LT, MT, &QT, queue ) );
                    magma_zmfree( &MT, queue );
                    CHECK( magma_z_mtransfer( QT, &MT, Magma_DEV, Magma_DEV, queue ) );
                    magma_zmfree( &QT, queue );
                    z++;
                }
            }
        }
    }
    magma_index_getvector( A.num_rows+1, MT.drow, 1, sizes_h, 1, queue );
    maxsize = 0;
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        nnzloc = sizes_h[i+1]-sizes_h[i];
        nnzU+= nnzloc;
        if( nnzloc > maxsize ){
            maxsize = sizes_h[i+1]-sizes_h[i];
        }
        if( maxsize > warpsize ){
            printf("%%   error for ISAI: size of system %d is too large by %d\n", i, maxsize-32);
            break;
        }
    }
    // this can be modified to the thread-block-size
    if( maxsize > warpsize ){
       info = -(maxsize - warpsize);
       goto cleanup;
    }
    // via registers
     CHECK( magma_zisai_generator_regs( MagmaUpper, MagmaNoTrans, MagmaNonUnit,
                    precond->U, &MT, sizes_d, locations_d, trisystems_d, rhs_d, queue ) );

     CHECK( magma_z_mtransfer( MT, &precond->UDT, Magma_CPU, Magma_DEV, queue ) );
    //CHECK( magma_zmtranspose( MT, &precond->UDT, queue ) );

cleanup:
    magma_free_cpu( sizes_h );
    magma_zmfree( &LT, queue );
    magma_zmfree( &MT, queue );
    magma_zmfree( &QT, queue );

    return info;
}





/***************************************************************************//**
    Purpose
    -------

    Left-hand-side application of ISAI preconditioner.


    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                input RHS b

    @param[in,out]
    x           magma_z_matrix
                solution x

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
magma_zisai_l(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue ){
    magma_int_t info = 0;

    if( precond->maxiter == 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->LD, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
    } else if( precond->maxiter > 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->LD, b, MAGMA_Z_ZERO, precond->d, queue ); // d=L_d^(-1)b
        magma_z_spmv( MAGMA_Z_ONE, precond->LD, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
        for( int z=0; z<precond->maxiter; z++ ){
            magma_z_spmv( MAGMA_Z_ONE, precond->L, *x, MAGMA_Z_ZERO, precond->work1, queue ); // work1 = L * x
            magma_z_spmv( MAGMA_Z_ONE, precond->LD, precond->work1, MAGMA_Z_ZERO, precond->work2, queue ); // work2 = L_d^(-1)work1
            magma_zaxpy( b.num_rows*b.num_cols, -MAGMA_Z_ONE, precond->work2.dval, 1 , x->dval, 1, queue );        // x = - work2
            magma_zaxpy( b.num_rows*b.num_cols, MAGMA_Z_ONE, precond->d.dval, 1 , x->dval, 1, queue );        // x = d + x = L_d^(-1)b - work2 = L_d^(-1)b - L_d^(-1) * L * x
        }
    }

    return info;
}


/***************************************************************************//**
    Purpose
    -------

    Right-hand-side application of ISAI preconditioner.


    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                input RHS b

    @param[in,out]
    x           magma_z_matrix
                solution x

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
magma_zisai_r(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue ){
    magma_int_t info = 0;

    if( precond->maxiter == 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->UD, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
    } else if( precond->maxiter > 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->UD, b, MAGMA_Z_ZERO, precond->d, queue ); // d=L^(-1)b
        magma_z_spmv( MAGMA_Z_ONE, precond->UD, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
        for( int z=0; z<precond->maxiter; z++ ){
            magma_z_spmv( MAGMA_Z_ONE, precond->U, *x, MAGMA_Z_ZERO, precond->work1, queue ); // work1=b+Lb
            magma_z_spmv( MAGMA_Z_ONE, precond->UD, precond->work1, MAGMA_Z_ZERO, precond->work2, queue ); // x=x+L^(-1)work1
            magma_zaxpy( b.num_rows*b.num_cols, -MAGMA_Z_ONE, precond->work2.dval, 1 , x->dval, 1, queue );        // t = t + c
            magma_zaxpy( b.num_rows*b.num_cols, MAGMA_Z_ONE, precond->d.dval, 1 , x->dval, 1, queue );        // t = t + c
        }
    }

    return info;
}


/***************************************************************************//**
    Purpose
    -------

    Left-hand-side application of ISAI preconditioner. Transpose.


    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                input RHS b

    @param[in,out]
    x           magma_z_matrix
                solution x

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
magma_zisai_l_t(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue ){
    magma_int_t info = 0;

    if( precond->maxiter == 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->LDT, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
    } else if( precond->maxiter > 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->LDT, b, MAGMA_Z_ZERO, precond->d, queue ); // d=L^(-1)b
        magma_z_spmv( MAGMA_Z_ONE, precond->LDT, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
        for( int z=0; z<precond->maxiter; z++ ){
            magma_z_spmv( MAGMA_Z_ONE, precond->LT, *x, MAGMA_Z_ZERO, precond->work1, queue ); // work1=b+Lb
            magma_z_spmv( MAGMA_Z_ONE, precond->LDT, precond->work1, MAGMA_Z_ZERO, precond->work2, queue ); // x=x+L^(-1)work1
            magma_zaxpy( b.num_rows*b.num_cols, -MAGMA_Z_ONE, precond->work2.dval, 1 , x->dval, 1, queue );        // t = t + c
            magma_zaxpy( b.num_rows*b.num_cols, MAGMA_Z_ONE, precond->d.dval, 1 , x->dval, 1, queue );        // t = t + c
        }
    }

    return info;
}


/***************************************************************************//**
    Purpose
    -------

    Right-hand-side application of ISAI preconditioner. Transpose.


    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                input RHS b

    @param[in,out]
    x           magma_z_matrix
                solution x

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
magma_zisai_r_t(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue ){
    magma_int_t info = 0;

    if( precond->maxiter == 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->UDT, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
    } else if( precond->maxiter > 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->UDT, b, MAGMA_Z_ZERO, precond->d, queue ); // d=L^(-1)b
        magma_z_spmv( MAGMA_Z_ONE, precond->UDT, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
        for( int z=0; z<precond->maxiter; z++ ){
            magma_z_spmv( MAGMA_Z_ONE, precond->UT, *x, MAGMA_Z_ZERO, precond->work1, queue ); // work1=b+Lb
            magma_z_spmv( MAGMA_Z_ONE, precond->UDT, precond->work1, MAGMA_Z_ZERO, precond->work2, queue ); // x=x+L^(-1)work1
            magma_zaxpy( b.num_rows*b.num_cols, -MAGMA_Z_ONE, precond->work2.dval, 1 , x->dval, 1, queue );        // t = t + c
            magma_zaxpy( b.num_rows*b.num_cols, MAGMA_Z_ONE, precond->d.dval, 1 , x->dval, 1, queue );        // t = t + c
        }
    }

    return info;
}

