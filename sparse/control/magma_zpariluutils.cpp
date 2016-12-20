/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/
#include "magmasparse_internal.h"

#define PRECISION_z

/**
    Purpose
    -------

    Computes the Frobenius norm of the difference between the CSR matrices A
    and B. They need to share the same sparsity pattern!


    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                sparse matrix in CSR

    @param[in]
    B           magma_z_matrix
                sparse matrix in CSR
                
    @param[out]
    res         real_Double_t*
                Frobenius norm of difference
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_zfrobenius(
    magma_z_matrix A,
    magma_z_matrix B,
    real_Double_t *res,
    magma_queue_t queue )
{
    real_Double_t tmp2;
    magma_int_t i,j,k;
    *res = 0.0;
    
    for(i=0; i<A.num_rows; i++){
        for(j=A.row[i]; j<A.row[i+1]; j++){
            magma_index_t localcol = A.col[j];
            for( k=B.row[i]; k<B.row[i+1]; k++){
                if(B.col[k] == localcol){
                    tmp2 = (real_Double_t) fabs( MAGMA_Z_REAL(A.val[j] )
                                                    - MAGMA_Z_REAL(B.val[k]) );

                    (*res) = (*res) + tmp2* tmp2;
                }
            }
        }
    }

    (*res) =  sqrt((*res));

    return MAGMA_SUCCESS;
}



/**
    Purpose
    -------

    Computes the nonlinear residual A - LU and returns the difference as
    well es the Frobenius norm of the difference


    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input sparse matrix in CSR

    @param[in]
    L           magma_z_matrix
                input sparse matrix in CSR

    @param[in]
    U           magma_z_matrix
                input sparse matrix in CSR

    @param[out]
    LU          magma_z_matrix*
                output sparse matrix in A-LU in CSR

    @param[out]
    res         real_Double_t*
                Frobenius norm of difference
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_znonlinres(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *LU,
    real_Double_t *res,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    real_Double_t tmp2;
    magma_int_t i,j,k;
        
    magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0 );

    magma_z_matrix dL={Magma_CSR}, dU={Magma_CSR}, dLU={Magma_CSR}, A_t={Magma_CSR};
    
    // make sure the target structure is empty
    magma_zmfree( LU, queue );

    CHECK( magma_zmtransfer( L, &dL, Magma_CPU, Magma_DEV, queue  ));
    CHECK( magma_zmtransfer( U, &dU, Magma_CPU, Magma_DEV, queue  ));
    CHECK( magma_zmtransfer( A, &A_t, Magma_CPU, Magma_CPU, queue  ));
    CHECK( magma_z_spmm( one, dL, dU, &dLU, queue ));

    CHECK( magma_zmtransfer(dLU, LU, Magma_DEV, Magma_CPU, queue ));
    magma_zmfree( &dL, queue  );
    magma_zmfree( &dU, queue  );
    magma_zmfree( &dLU, queue  );

    // compute Frobenius norm of A-LU
    for(i=0; i<A.num_rows; i++){
        for(j=A.row[i]; j<A.row[i+1]; j++){
            magma_index_t lcol = A.col[j];
            magmaDoubleComplex newval = MAGMA_Z_MAKE(0.0, 0.0);
            for(k=LU->row[i]; k<LU->row[i+1]; k++){
                if( LU->col[k] == lcol ){
                    newval = MAGMA_Z_MAKE(
                        MAGMA_Z_REAL( LU->val[k] )- MAGMA_Z_REAL( A.val[j] )
                                                , 0.0 );
                }
            }
            A_t.val[j] = newval;
        }
    }

    for(i=0; i<A.num_rows; i++){
        for(j=A.row[i]; j<A.row[i+1]; j++){
            tmp2 = (real_Double_t) fabs( MAGMA_Z_REAL(A_t.val[j]) );
            (*res) = (*res) + tmp2* tmp2;
        }
    }

    magma_zmfree( LU, queue  );
    magma_zmfree( &A_t, queue  );

    (*res) =  sqrt((*res));
    
cleanup:
    if( info !=0 ){
        magma_zmfree( LU, queue  );
    }
    magma_zmfree( &A_t, queue  );
    magma_zmfree( &dL, queue  );
    magma_zmfree( &dU, queue  );
    magma_zmfree( &dLU, queue  );
    return info;
}

/**
    Purpose
    -------

    Computes the ILU residual A - LU and returns the difference as
    well es the Frobenius norm of the difference


    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input sparse matrix in CSR

    @param[in]
    L           magma_z_matrix
                input sparse matrix in CSR

    @param[in]
    U           magma_z_matrix
                input sparse matrix in CSR

    @param[out]
    LU          magma_z_matrix*
                output sparse matrix in A-LU in CSR
                
    @param[out]
    res         real_Double_t*
                Frobenius norm of difference
                
    @param[out]
    nonlinres   real_Double_t*
                Frobenius norm of difference
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_zilures(
    magma_z_matrix A,
    magma_z_matrix L,
    magma_z_matrix U,
    magma_z_matrix *LU,
    real_Double_t *res,
    real_Double_t *nonlinres,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magmaDoubleComplex tmp;
    real_Double_t tmp2;
    magma_int_t i, j, k;
    
    magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0 );

    magma_z_matrix LL={Magma_CSR}, dL={Magma_CSR}, dU={Magma_CSR}, dLU={Magma_CSR};
    
    // make sure the target structure is empty
    magma_zmfree( LU, queue );

    if( L.row[1]==1 ){        // lower triangular with unit diagonal
        //printf("L lower triangular.\n");
        LL.diagorder_type = Magma_UNITY;
        CHECK( magma_zmconvert( L, &LL, Magma_CSR, Magma_CSRL, queue ));
    }
    else if ( L.row[1]==0 ){ // strictly lower triangular
        //printf("L strictly lower triangular.\n");
        CHECK( magma_zmtransfer( L, &LL, Magma_CPU, Magma_CPU, queue ));
        magma_free_cpu( LL.col );
        magma_free_cpu( LL.val );
        LL.nnz = L.nnz+L.num_rows;
        CHECK( magma_zmalloc_cpu( &LL.val, LL.nnz ));
        CHECK( magma_index_malloc_cpu( &LL.col, LL.nnz ));
        magma_int_t z=0;
        for (i=0; i < L.num_rows; i++) {
            LL.row[i] = z;
            for (j=L.row[i]; j < L.row[i+1]; j++) {
                LL.val[z] = L.val[j];
                LL.col[z] = L.col[j];
                z++;
            }
            // add unit diagonal
            LL.val[z] = MAGMA_Z_MAKE(1.0, 0.0);
            LL.col[z] = i;
            z++;
        }
        LL.row[LL.num_rows] = z;
    }
    else {
        printf("error: L neither lower nor strictly lower triangular!\n");
    }

    CHECK( magma_zmtransfer( LL, &dL, Magma_CPU, Magma_DEV, queue  ));
    CHECK( magma_zmtransfer( U, &dU, Magma_CPU, Magma_DEV, queue  ));
    magma_zmfree( &LL, queue );
    CHECK( magma_z_spmm( one, dL, dU, &dLU, queue ));



    CHECK( magma_zmtransfer(dLU, LU, Magma_DEV, Magma_CPU, queue ));
    magma_zmfree( &dL, queue );
    magma_zmfree( &dU, queue );
    magma_zmfree( &dLU, queue );

    // compute Frobenius norm of A-LU
    for(i=0; i<A.num_rows; i++){
        for(j=A.row[i]; j<A.row[i+1]; j++){
            magma_index_t lcol = A.col[j];
            for(k=LU->row[i]; k<LU->row[i+1]; k++){
                if( LU->col[k] == lcol ){
                    tmp = MAGMA_Z_MAKE(
                        MAGMA_Z_REAL( LU->val[k] )- MAGMA_Z_REAL( A.val[j] )
                                                , 0.0 );
                    LU->val[k] = tmp;

                    tmp2 = (real_Double_t) fabs( MAGMA_Z_REAL(tmp) );
                    (*nonlinres) = (*nonlinres) + tmp2*tmp2;
                }
            }
        }
    }

    for(i=0; i<LU->num_rows; i++){
        for(j=LU->row[i]; j<LU->row[i+1]; j++){
            tmp2 = (real_Double_t) fabs( MAGMA_Z_REAL(LU->val[j]) );
            (*res) = (*res) + tmp2* tmp2;
        }
    }

    (*res) =  sqrt((*res));
    (*nonlinres) =  sqrt((*nonlinres));

cleanup:
    if( info !=0 ){
        magma_zmfree( LU, queue  );
    }
    magma_zmfree( &LL, queue );
    magma_zmfree( &dL, queue  );
    magma_zmfree( &dU, queue  );
    magma_zmfree( &dLU, queue  );
    return info;
}



/**
    Purpose
    -------

    Computes the IC residual A - CC^T and returns the difference as
    well es the Frobenius norm of the difference


    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input sparse matrix in CSR

    @param[in]
    C           magma_z_matrix
                input sparse matrix in CSR

    @param[in]
    CT          magma_z_matrix
                input sparse matrix in CSR

    @param[in]
    LU          magma_z_matrix*
                output sparse matrix in A-LU in CSR

    @param[out]
    res         real_Double_t*
                IC residual
                
    @param[out]
    nonlinres   real_Double_t*
                nonlinear residual
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_zicres(
    magma_z_matrix A,
    magma_z_matrix C,
    magma_z_matrix CT,
    magma_z_matrix *LU,
    real_Double_t *res,
    real_Double_t *nonlinres,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magmaDoubleComplex tmp;
    real_Double_t tmp2;
    magma_int_t i,j,k;

    magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0 );
    
    magma_z_matrix dL={Magma_CSR}, dU={Magma_CSR}, dLU={Magma_CSR};
    
    // make sure the target structure is empty
    magma_zmfree( LU, queue );
    
    *res = 0.0;
    *nonlinres = 0.0;

    CHECK( magma_zmtransfer( C, &dL, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_zmtransfer( CT, &dU, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_z_spmm( one, dL, dU, &dLU, queue ));
    CHECK( magma_zmtransfer(dLU, LU, Magma_DEV, Magma_CPU, queue ));

    magma_zmfree( &dLU, queue );

    // compute Frobenius norm of A-LU
    for(i=0; i<A.num_rows; i++){
        for(j=A.row[i]; j<A.row[i+1]; j++){
            magma_index_t lcol = A.col[j];
            for(k=LU->row[i]; k<LU->row[i+1]; k++){
                if( LU->col[k] == lcol ){
                    tmp = MAGMA_Z_MAKE(
                        MAGMA_Z_REAL( LU->val[k] )- MAGMA_Z_REAL( A.val[j] )
                                                , 0.0 );
                    LU->val[k] = tmp;

                    tmp2 = (real_Double_t) fabs( MAGMA_Z_REAL(tmp) );
                    (*nonlinres) = (*nonlinres) + tmp2*tmp2;
                }
            }
        }
    }

    for(i=0; i<LU->num_rows; i++){
        for(j=LU->row[i]; j<LU->row[i+1]; j++){
            tmp2 = (real_Double_t) fabs( MAGMA_Z_REAL(LU->val[j]) );
            (*res) = (*res) + tmp2* tmp2;
        }
    }


    (*res) =  sqrt((*res));
    (*nonlinres) =  sqrt((*nonlinres));

cleanup:
    if( info !=0 ){
        magma_zmfree( LU, queue  );
    }
    magma_zmfree( &dL, queue  );
    magma_zmfree( &dU, queue  );
    magma_zmfree( &dLU, queue  );
    return info;
}



/**
    Purpose
    -------

    Computes an initial guess for the ParILU/ParIC


    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                sparse matrix in CSR

    @param[out]
    L           magma_z_matrix*
                sparse matrix in CSR

    @param[out]
    U           magma_z_matrix*
                sparse matrix in CSR
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.


    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_zinitguess(
    magma_z_matrix A,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magmaDoubleComplex one = MAGMA_Z_MAKE( 1.0, 0.0 );
    
    magma_z_matrix hAL={Magma_CSR}, hAU={Magma_CSR}, dAL={Magma_CSR}, 
    dAU={Magma_CSR}, dALU={Magma_CSR}, hALU={Magma_CSR}, hD={Magma_CSR}, 
    dD={Magma_CSR}, dL={Magma_CSR}, hL={Magma_CSR};
    magma_int_t i,j;
    
    magma_int_t offdiags = 0;
    magma_index_t *diag_offset;
    magmaDoubleComplex *diag_vals=NULL;
    
    // make sure the target structure is empty
    magma_zmfree( L, queue );
    magma_zmfree( U, queue );

    // need only lower triangular
    hAL.diagorder_type = Magma_VALUE;
    CHECK( magma_zmconvert( A, &hAL, Magma_CSR, Magma_CSRL, queue ));
    //magma_zmconvert( hAL, &hALCOO, Magma_CSR, Magma_CSRCOO );

    // need only upper triangular
    //magma_zmconvert( A, &hAU, Magma_CSR, Magma_CSRU );
    CHECK( magma_z_cucsrtranspose(  hAL, &hAU, queue ));
    //magma_zmconvert( hAU, &hAUCOO, Magma_CSR, Magma_CSRCOO );
    CHECK( magma_zmtransfer( hAL, &dAL, Magma_CPU, Magma_DEV, queue ));
    CHECK( magma_z_spmm( one, dAL, dAU, &dALU, queue ));
    CHECK( magma_zmtransfer( dALU, &hALU, Magma_DEV, Magma_CPU, queue ));

    magma_zmfree( &dAU, queue);
    magma_zmfree( &dALU, queue);


    CHECK( magma_zmalloc_cpu( &diag_vals, offdiags+1 ));
    CHECK( magma_index_malloc_cpu( &diag_offset, offdiags+1 ));
    diag_offset[0] = 0;
    diag_vals[0] = MAGMA_Z_MAKE( 1.0, 0.0 );
    CHECK( magma_zmgenerator( hALU.num_rows, offdiags, diag_offset, diag_vals, &hD, queue ));
    magma_zmfree( &hALU, queue );

    
    for(i=0; i<hALU.num_rows; i++){
        for(j=hALU.row[i]; j<hALU.row[i+1]; j++){
            if( hALU.col[j] == i ){
                //printf("%d %d  %d == %d -> %f   -->", i, j, hALU.col[j], i, hALU.val[j]);
                hD.val[i] = MAGMA_Z_MAKE(
                        1.0 / sqrt(fabs(MAGMA_Z_REAL(hALU.val[j])))  , 0.0 );
                //printf("insert %f at %d\n", hD.val[i], i);
            }
        }
    }


    CHECK( magma_zmtransfer( hD, &dD, Magma_CPU, Magma_DEV, queue ));
    magma_zmfree( &hD, queue);

    CHECK( magma_z_spmm( one, dD, dAL, &dL, queue ));
    magma_zmfree( &dAL, queue );
    magma_zmfree( &dD, queue );



/*
    // check for diagonal = 1
    magma_z_matrix dLt={Magma_CSR}, dLL={Magma_CSR}, LL={Magma_CSR};
    CHECK( magma_z_cucsrtranspose(  dL, &dLt ));
    CHECK( magma_zcuspmm( dL, dLt, &dLL ));
    CHECK( magma_zmtransfer( dLL, &LL, Magma_DEV, Magma_CPU ));
    //for(i=0; i < hALU.num_rows; i++) {
    for(i=0; i < 100; i++) {
        for(j=hALU.row[i]; j < hALU.row[i+1]; j++) {
            if( hALU.col[j] == i ){
                printf("%d %d -> %f   -->", i, i, LL.val[j]);
            }
        }
    }
*/
    CHECK( magma_zmtransfer( dL, &hL, Magma_DEV, Magma_CPU, queue ));
    CHECK( magma_zmconvert( hL, L, Magma_CSR, Magma_CSRCOO, queue ));



cleanup:
    if( info !=0 ){
        magma_zmfree( L, queue  );
        magma_zmfree( U, queue  );
    }
    magma_zmfree( &dAU, queue);
    magma_zmfree( &dALU, queue);
    magma_zmfree( &dL, queue );
    magma_zmfree( &hL, queue );
    magma_zmfree( &dAL, queue );
    magma_zmfree( &dD, queue );
    magma_zmfree( &hD, queue);
    magma_zmfree( &hALU, queue );
    return info;
}



/**
    Purpose
    -------

    Using the iterative approach of computing ILU factorizations with increasing
    fill-in, it takes the input matrix A, containing the approximate factors,
    ( L and U as well )
    computes a matrix with one higher level of fill-in, inserts the original
    approximation as initial guess, and provides the factors L and U also
    filled with the scaled initial guess.


    Arguments
    ---------

    @param[in]
    A           magma_z_matrix*
                sparse matrix in CSR

    @param[out]
    B           magma_z_matrix*
                sparse matrix in CSR
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.


    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_zinitrecursiveLU(
    magma_z_matrix A,
    magma_z_matrix *B,
    magma_queue_t queue )
{
    magma_int_t i,j,k;
    
    // make sure the target structure is empty
    magma_zmfree( B, queue );

    for(i=0; i<A.num_rows; i++){
        for(j=B->row[i]; j<B->row[i+1]; j++){
            B->val[j] = MAGMA_Z_MAKE(0.0, 0.0);
            magma_index_t localcol = B->col[j];
            for( k=A.row[i]; k<A.row[i+1]; k++){
                if(A.col[k] == localcol){
                    B->val[j] = A.val[k];
                }
            }
        }
    }

    return MAGMA_SUCCESS; 
}



/**
    Purpose
    -------

    Checks for a lower triangular matrix whether it is strictly lower triangular
    and in the negative case adds a unit diagonal. It does this in-place.


    Arguments
    ---------

    @param[in,out]
    L           magma_z_matrix*
                sparse matrix in CSR
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

magma_int_t
magma_zmLdiagadd(
    magma_z_matrix *L,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_z_matrix LL={Magma_CSR};

    if( L->row[1]==1 ){        // lower triangular with unit diagonal
        //printf("L lower triangular.\n");
        LL.diagorder_type = Magma_UNITY;
        CHECK( magma_zmconvert( *L, &LL, Magma_CSR, Magma_CSRL, queue ));
    }
    else if ( L->row[1]==0 ){ // strictly lower triangular
        //printf("L strictly lower triangular.\n");
        CHECK( magma_zmtransfer( *L, &LL, Magma_CPU, Magma_CPU, queue ));
        magma_free_cpu( LL.col );
        magma_free_cpu( LL.val );
        LL.nnz = L->nnz+L->num_rows;
        CHECK( magma_zmalloc_cpu( &LL.val, LL.nnz ));
        CHECK( magma_index_malloc_cpu( &LL.col, LL.nnz ));
        magma_int_t z=0;
        for( magma_int_t i=0; i<L->num_rows; i++){
            LL.row[i] = z;
            for( magma_int_t j=L->row[i]; j<L->row[i+1]; j++){
                LL.val[z] = L->val[j];
                LL.col[z] = L->col[j];
                z++;
            }
            // add unit diagonal
            LL.val[z] = MAGMA_Z_MAKE(1.0, 0.0);
            LL.col[z] = i;
            z++;
        }
        LL.row[LL.num_rows] = z;
        LL.nnz = z;
    }
    else {
        printf("error: L neither lower nor strictly lower triangular!\n");
    }
    magma_zmfree( L, queue );
    CHECK( magma_zmtransfer(LL, L, Magma_CPU, Magma_CPU, queue ));

cleanup:
    if( info != 0 ){
        magma_zmfree( L, queue );
    }
    magma_zmfree( &LL, queue );
    return info;
}
