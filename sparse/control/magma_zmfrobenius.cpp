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

    Computes the Frobenius norm || A - B ||_S on the sparsity pattern of S.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                input sparse matrix in CSR

    @param[in]
    B           magma_z_matrix
                input sparse matrix in CSR

    @param[in]
    S           magma_z_matrix
                input sparsity pattern in CSR

    @param[out]
    norm        double*
                Frobenius norm of difference on sparsity pattern S
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmfrobenius(
    magma_z_matrix A,
    magma_z_matrix B,
    magma_z_matrix S,
    double *norm,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    double tmp;
    magma_int_t i,j,k;
        
    magma_z_matrix hA={Magma_CSR}, hB={Magma_CSR}, hS={Magma_CSR};

    CHECK( magma_zmtransfer( A, &hA, A.memory_location, Magma_CPU, queue  ));
    CHECK( magma_zmtransfer( B, &hB, B.memory_location, Magma_CPU, queue  ));
    CHECK( magma_zmtransfer( S, &hS, S.memory_location, Magma_CPU, queue  ));
    
    if( hA.num_rows == hB.num_rows && hA.num_rows == hS.num_rows ) {
        for(i=0; i<hS.num_rows; i++){
            for(j=hS.row[i]; j<hS.row[i+1]; j++){
                magma_index_t lcol = hS.col[j];
                magmaDoubleComplex Aval = MAGMA_Z_MAKE(0.0, 0.0);
                magmaDoubleComplex Bval = MAGMA_Z_MAKE(0.0, 0.0);
                for(k=hA.row[i]; k<hA.row[i+1]; k++){
                    if( hA.col[k] == lcol ){
                        Aval = hA.val[k];
                    }
                }
                for(k=hB.row[i]; k<hB.row[i+1]; k++){
                    if( hB.col[k] == lcol ){
                        Bval = hB.val[k];
                    }
                }
                tmp = MAGMA_Z_ABS(Aval - Bval) ;
                (*norm) = (*norm) + tmp * tmp;
            }
        }
        
        (*norm) =  sqrt((*norm));
    }
    
    
cleanup:
    magma_zmfree( &hA, queue );
    magma_zmfree( &hB, queue );
    magma_zmfree( &hS, queue );
    
    return info;
}
