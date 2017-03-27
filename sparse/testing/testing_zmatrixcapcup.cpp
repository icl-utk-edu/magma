/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include "magma_v2.h"
#include "magmasparse.h"
#include "magma_operators.h"
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    /* Initialize */
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    magma_zopts zopts;
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );

    magma_z_matrix Z={Magma_CSR};
    magma_z_matrix Z1={Magma_CSR};
    magma_z_matrix Z2={Magma_CSR};
    magma_z_matrix Z3={Magma_CSR};
    magma_z_matrix Z4={Magma_CSR};
    magma_z_matrix Z5={Magma_CSR};
    
    int i=1;
    TESTING_CHECK( magma_zparse_opts( argc, argv, &zopts, &i, queue ));
    printf("matrixinfo = [\n");
    printf("%%   size (n)   ||   nonzeros (nnz)   ||   nnz/n\n");
    printf("%%=============================================================%%\n");
    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_zm_5stencil(  laplace_size, &Z, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_z_csr_mtx( &Z,  argv[i], queue ));
        }

        printf("   %10lld          %10lld          %10lld\n",
               (long long) Z.num_rows, (long long) Z.nnz, (long long) (Z.nnz/Z.num_rows) );
        
        // multiply by itself
        TESTING_CHECK( magma_zmconvert( Z, &Z1,  Magma_CSR, Magma_CSRL, queue ));
        TESTING_CHECK( magma_zmtranspose( Z1, &Z2, queue ) );
        magma_zprint_matrix( Z, queue );
        magma_zprint_matrix( Z1, queue );
        printf("B = tril(A)^T:\n");
        magma_zprint_matrix( Z2, queue );
        
        // now the negcap:
        printf("C = B cup B^T :\n");
        TESTING_CHECK( magma_zmatrix_cup( Z1, Z2, &Z3, queue ));
        magma_zprint_matrix( Z3, queue );
        
        // now the negcap:
        printf("C = B cap B^T :\n");
        TESTING_CHECK( magma_zmatrix_cap( Z1, Z2, &Z4, queue ));
        magma_zprint_matrix( Z4, queue );
        
        
        // now the negcap:
        printf("C = B negcap B^T :\n");
        TESTING_CHECK( magma_zmatrix_negcap( Z1, Z2, &Z5, queue ));
        magma_zprint_matrix( Z5, queue );
        
        magma_zmfree(&Z, queue );
        magma_zmfree(&Z1, queue );
        magma_zmfree(&Z2, queue );
        magma_zmfree(&Z3, queue );
        magma_zmfree(&Z4, queue );
        magma_zmfree(&Z5, queue );

        i++;
    }
    printf("%%=============================================================%%\n");
    printf("];\n");
    
    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
