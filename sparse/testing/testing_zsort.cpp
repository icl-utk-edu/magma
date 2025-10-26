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
#include <time.h>

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
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );

    magma_int_t i, n=100;
    magma_index_t *x=NULL;
    magmaDoubleComplex *y=NULL;
    
    magma_z_matrix A={Magma_CSR};

    TESTING_CHECK( magma_index_malloc_cpu( &x, n ));
    printf("unsorted:\n");
    srand(time(NULL));
    for(i = 0; i < n; i++ ){
        int r = rand()%100;
        x[i] = r;
        printf("%d  ", x[i]);
    }
    printf("\n\n");
    
    printf("sorting...");
    TESTING_CHECK( magma_zindexsort(x, 0, n-1, queue ));
    printf("done.\n\n");
    
    printf("sorted:\n");
    for(i = 0; i < n; i++ ){
        printf("%d  ", x[i]);
    }
    printf("\n\n");

    magma_free_cpu( x );
    
    
    TESTING_CHECK( magma_zmalloc_cpu( &y, n ));
    printf("unsorted:\n");
    srand(time(NULL));
    for(i = 0; i < n; i++ ){
        double r = (double) rand()/(double) 10.;
        y[i] = MAGMA_Z_MAKE( r, 0.0);
        if (i % 5 == 0)
            y[i] = - y[i];
        printf("%2.2f + %2.2f  ", MAGMA_Z_REAL(y[i]), MAGMA_Z_IMAG(y[i]) );
    }
    printf("\n\n");
    
    printf("sorting...");
    TESTING_CHECK( magma_zsort(y, 0, n-1, queue ));
    printf("done.\n\n");
    
    printf("sorted:\n");
    for(i = 0; i < n; i++ ){
        printf("%2.2f + %2.2f  ", MAGMA_Z_REAL(y[i]), MAGMA_Z_IMAG(y[i]) );
    }
    printf("\n\n");

    magma_free_cpu( y );
    
    magma_queue_destroy( queue );
    magma_finalize();
    return info;
}
