/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
       @author Stephen Wood

*/
#include "magmasparse_internal.h"

#define RTOLERANCE     lapackf77_dlamch( "E" )
#define ATOLERANCE     lapackf77_dlamch( "E" )


/**
    Purpose
    -------

    Scales a matrix.

    Arguments
    ---------

    @param[in,out]
    A           magma_z_matrix*
                input/output matrix

    @param[in]
    scaling     magma_scale_t
                scaling type (unit rownorm / unit diagonal)

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmscale(
    magma_z_matrix *A,
    magma_scale_t scaling,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaDoubleComplex *tmp=NULL;
    
    magma_z_matrix hA={Magma_CSR}, CSRA={Magma_CSR};
    
    if( A->num_rows != A->num_cols && scaling != Magma_NOSCALE ){
        printf("%% warning: non-square matrix.\n");
        printf("%% Fallback: no scaling.\n");
        scaling = Magma_NOSCALE;
    } 
        
   
    if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSRCOO ) {
        if ( scaling == Magma_NOSCALE ) {
            // no scale
            ;
        }
        else if( A->num_rows == A->num_cols ){
            if ( scaling == Magma_UNITROW ) {
                // scale to unit rownorm
                CHECK( magma_zmalloc_cpu( &tmp, A->num_rows ));
                for( magma_int_t z=0; z<A->num_rows; z++ ) {
                    magmaDoubleComplex s = MAGMA_Z_MAKE( 0.0, 0.0 );
                    for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ )
                        s+= MAGMA_Z_REAL(A->val[f])*MAGMA_Z_REAL(A->val[f]);
                    tmp[z] = MAGMA_Z_MAKE( 1.0/sqrt(  MAGMA_Z_REAL( s )  ), 0.0 );
                }        
                for( magma_int_t z=0; z<A->nnz; z++ ) {
                    A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
                }
            }
            else if (scaling == Magma_UNITDIAG ) {
                // scale to unit diagonal
                CHECK( magma_zmalloc_cpu( &tmp, A->num_rows ));
                for( magma_int_t z=0; z<A->num_rows; z++ ) {
                    magmaDoubleComplex s = MAGMA_Z_MAKE( 0.0, 0.0 );
                    for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ ) {
                        if ( A->col[f]== z ) {
                            // add some identity matrix
                            //A->val[f] = A->val[f] +  MAGMA_Z_MAKE( 100000.0, 0.0 );
                            s = A->val[f];
                        }
                    }
                    if ( s == MAGMA_Z_MAKE( 0.0, 0.0 ) ){
                        printf("%%error: zero diagonal element.\n");
                        info = MAGMA_ERR;
                    }
                    tmp[z] = MAGMA_Z_MAKE( 1.0/sqrt(  MAGMA_Z_REAL( s )  ), 0.0 );
                }
                for( magma_int_t z=0; z<A->nnz; z++ ) {
                    A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
                }
            }
            else {
                printf( "%%error: scaling not supported.\n" );
                info = MAGMA_ERR_NOT_SUPPORTED;
            }
        }
        else {
            printf( "%%error: scaling not supported.\n" );
            info = MAGMA_ERR_NOT_SUPPORTED;
        }
    }
    else {
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        CHECK( magma_zmtransfer( *A, &hA, A->memory_location, Magma_CPU, queue ));
        CHECK( magma_zmconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO, queue ));

        CHECK( magma_zmscale( &CSRA, scaling, queue ));

        magma_zmfree( &hA, queue );
        magma_zmfree( A, queue );
        CHECK( magma_zmconvert( CSRA, &hA, Magma_CSRCOO, A_storage, queue ));
        CHECK( magma_zmtransfer( hA, A, Magma_CPU, A_location, queue ));
    }
    
cleanup:
    magma_free_cpu( tmp );
    magma_zmfree( &hA, queue );
    magma_zmfree( &CSRA, queue );
    return info;
}

/**
    Purpose
    -------

    Scales a matrix and a right hand side vector of a Ax = b system.

    Arguments
    ---------

    @param[in,out]
    A           magma_z_matrix*
                input/output matrix
                
    @param[in,out]
    b           magma_z_matrix*
                input/output right hand side vector    
                
    @param[out]
    scaling_factors   magma_z_matrix*
                output scaling factors vector   

    @param[in]
    scaling     magma_scale_t
                scaling type (unit rownorm / unit diagonal)

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmscale_matrix_rhs(
    magma_z_matrix *A,
    magma_z_matrix *b,
    magma_z_matrix *scaling_factors,
    magma_scale_t scaling,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaDoubleComplex *tmp=NULL;
    
    magma_z_matrix hA={Magma_CSR}, CSRA={Magma_CSR};
    
    printf("%% scaling = %d\n", scaling);
    
    if( A->num_rows != A->num_cols && scaling != Magma_NOSCALE ){
        printf("%% warning: non-square matrix.\n");
        printf("%% Fallback: no scaling.\n");
        scaling = Magma_NOSCALE;
    } 
        
   
    if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSRCOO ) {
        if ( scaling == Magma_NOSCALE ) {
            // no scale
            ;
        }
        else if( A->num_rows == A->num_cols ){
            if ( scaling == Magma_UNITROW ) {
                // scale to unit rownorm by rows
                CHECK( magma_zmalloc_cpu( &tmp, A->num_rows ));
                for( magma_int_t z=0; z<A->num_rows; z++ ) {
                    magmaDoubleComplex s = MAGMA_Z_MAKE( 0.0, 0.0 );
                    for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ )
                        s+= MAGMA_Z_MAKE( MAGMA_Z_REAL(A->val[f])*MAGMA_Z_REAL(A->val[f]), 0.0 );
                    tmp[z] = MAGMA_Z_MAKE( 1.0/sqrt(  MAGMA_Z_REAL( s )  ), 0.0 );
                }        
                for( magma_int_t z=0; z<A->nnz; z++ ) {
                    A->val[z] = A->val[z] * tmp[A->rowidx[z]];
                }
                for ( int i=0; i<A->num_rows; i++ ) {
                  b->val[i] = b->val[i] * tmp[i];
                }
            }
            else if (scaling == Magma_UNITDIAG ) {
                // scale to unit diagonal by rows
                CHECK( magma_zmalloc_cpu( &tmp, A->num_rows ));
                for( magma_int_t z=0; z<A->num_rows; z++ ) {
                    magmaDoubleComplex s = MAGMA_Z_MAKE( 0.0, 0.0 );
                    for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ ) {
                        if ( A->col[f]== z ) {
                            s = A->val[f];
                        }
                    }
                    if ( s == MAGMA_Z_MAKE( 0.0, 0.0 ) ){
                        printf("%%error: zero diagonal element.\n");
                        info = MAGMA_ERR;
                    }
                    tmp[z] = MAGMA_Z_MAKE( 1.0/MAGMA_Z_REAL( s ), 0.0 );
                }
                for( magma_int_t z=0; z<A->nnz; z++ ) {
                    A->val[z] = A->val[z] * tmp[A->rowidx[z]];
                }
                for ( int i=0; i<A->num_rows; i++ ) {
                  b->val[i] = b->val[i] * tmp[i];
                }
            }
            else if ( scaling == Magma_UNITROWCOL ) {
                // scale to unit rownorm by rows and columns
                CHECK( magma_zmalloc_cpu( &tmp, A->num_rows ));
                for( magma_int_t z=0; z<A->num_rows; z++ ) {
                    magmaDoubleComplex s = MAGMA_Z_MAKE( 0.0, 0.0 );
                    for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ )
                        s+= MAGMA_Z_MAKE( MAGMA_Z_REAL(A->val[f])*MAGMA_Z_REAL(A->val[f]), 0.0 );
                    tmp[z] = MAGMA_Z_MAKE( 1.0/sqrt(  MAGMA_Z_REAL( s )  ), 0.0 );
                }        
                for( magma_int_t z=0; z<A->nnz; z++ ) {
                    A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
                }
                scaling_factors->num_rows = A->num_rows;
                scaling_factors->num_cols = 1;
                scaling_factors->ld = 1;
                scaling_factors->nnz = A->num_rows;
                scaling_factors->val = NULL;
                CHECK( magma_zmalloc_cpu( &scaling_factors->val, A->num_rows ));
                for ( int i=0; i<A->num_rows; i++ ) {
                  scaling_factors->val[i] = tmp[i];
                  b->val[i] = b->val[i] * tmp[i];
                }
            }
            else if (scaling == Magma_UNITDIAGCOL ) {
                // scale to unit diagonal by rows and columns
                CHECK( magma_zmalloc_cpu( &tmp, A->num_rows ));
                for( magma_int_t z=0; z<A->num_rows; z++ ) {
                    magmaDoubleComplex s = MAGMA_Z_MAKE( 0.0, 0.0 );
                    for( magma_int_t f=A->row[z]; f<A->row[z+1]; f++ ) {
                        if ( A->col[f]== z ) {
                            s = A->val[f];
                        }
                    }
                    if ( s == MAGMA_Z_MAKE( 0.0, 0.0 ) ){
                        printf("%%error: zero diagonal element.\n");
                        info = MAGMA_ERR;
                    }
                    tmp[z] = MAGMA_Z_MAKE( 1.0/sqrt(  MAGMA_Z_REAL( s )  ), 0.0 );
                }
                for( magma_int_t z=0; z<A->nnz; z++ ) {
                    A->val[z] = A->val[z] * tmp[A->col[z]] * tmp[A->rowidx[z]];
                }
                scaling_factors->num_rows = A->num_rows;
                scaling_factors->num_cols = 1;
                scaling_factors->ld = 1;
                scaling_factors->nnz = A->num_rows;
                scaling_factors->val = NULL;
                CHECK( magma_zmalloc_cpu( &scaling_factors->val, A->num_rows ));
                for ( int i=0; i<A->num_rows; i++ ) {
                  scaling_factors->val[i] = tmp[i];
                  b->val[i] = b->val[i] * tmp[i];
                }
            }
            else {
                printf( "%%error: scaling %d not supported line = %d.\n", 
                  scaling, __LINE__ );
                info = MAGMA_ERR_NOT_SUPPORTED;
            }
        }
        else {
            printf( "%%error: scaling %d not supported line = %d.\n", 
                  scaling, __LINE__ );
            info = MAGMA_ERR_NOT_SUPPORTED;
        }
    }
    else {
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        CHECK( magma_zmtransfer( *A, &hA, A->memory_location, Magma_CPU, queue ));
        CHECK( magma_zmconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO, queue ));

        CHECK( magma_zmscale_matrix_rhs( &CSRA, b, scaling_factors, scaling, queue ));

        magma_zmfree( &hA, queue );
        magma_zmfree( A, queue );
        CHECK( magma_zmconvert( CSRA, &hA, Magma_CSRCOO, A_storage, queue ));
        CHECK( magma_zmtransfer( hA, A, Magma_CPU, A_location, queue ));
    }
    
cleanup:
    magma_free_cpu( tmp );
    magma_zmfree( &hA, queue );
    magma_zmfree( &CSRA, queue );
    return info;
}

/**
    Purpose
    -------

    Adds a multiple of the Identity matrix to a matrix: A = A+add * I

    Arguments
    ---------

    @param[in,out]
    A           magma_z_matrix*
                input/output matrix

    @param[in]
    add         magmaDoubleComplex
                scaling for the identity matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmdiagadd(
    magma_z_matrix *A,
    magmaDoubleComplex add,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix hA={Magma_CSR}, CSRA={Magma_CSR};
    
    if ( A->memory_location == Magma_CPU && A->storage_type == Magma_CSRCOO ) {
        for( magma_int_t z=0; z<A->nnz; z++ ) {
            if ( A->col[z]== A->rowidx[z] ) {
                // add some identity matrix
                A->val[z] = A->val[z] +  add;
            }
        }
    }
    else {
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        CHECK( magma_zmtransfer( *A, &hA, A->memory_location, Magma_CPU, queue ));
        CHECK( magma_zmconvert( hA, &CSRA, hA.storage_type, Magma_CSRCOO, queue ));

        CHECK( magma_zmdiagadd( &CSRA, add, queue ));

        magma_zmfree( &hA, queue );
        magma_zmfree( A, queue );
        CHECK( magma_zmconvert( CSRA, &hA, Magma_CSRCOO, A_storage, queue ));
        CHECK( magma_zmtransfer( hA, A, Magma_CPU, A_location, queue ));
    }
    
cleanup:
    magma_zmfree( &hA, queue );
    magma_zmfree( &CSRA, queue );
    return info;
}
