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
#include "testings.h"


/* ////////////////////////////////////////////////////////////////////////////
   -- testing threshold selection GPU kernel
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    magma_zopts zopts;
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    
    
    real_Double_t start, end;
    
    double thrs;
    int m = 10000;
    int n = 100;
    int count = 0;
    magmaDoubleComplex *val, *d_val;
    TESTING_CHECK(magma_zmalloc_cpu(&val, m));
    TESTING_CHECK(magma_zmalloc(&d_val, m));
    // fill the values with random numbers
    for (int z=0; z<m; z++){
        val[z] = MAGMA_Z_MAKE((double)(rand()%m)/(double)m, 0.0);        
    }
    
    // copy over
    magma_zsetvector( m, val, 1, d_val, 1, queue );
    
    start = magma_sync_wtime( queue );
    for(int i=0; i<10; i++)
        TESTING_CHECK(magma_zthrsholdselect(m, n, d_val, &thrs, queue));
    end = magma_sync_wtime( queue );
    printf( " > %.2e seconds.\n", (end-start)/10 );
    
    for(int z=0; z<m; z++) {
        if (MAGMA_Z_ABS(val[z])<thrs) {
            count++;    
        }
    }
    printf("selecting threshold for %d of %d elements: threshold selected: %.8e -> exact to %d elements.\n", n, m, thrs, count);
    
    magma_free(d_val);
    magma_free_cpu(val);
    
    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
