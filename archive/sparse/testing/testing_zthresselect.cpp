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

    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    
    magma_z_matrix A={Magma_CSR};
    real_Double_t start, end, t_gpu=0.0, t_cpu=0.0;
    magma_int_t sampling = 16;
    double thrs;
    for( int m = 1000; m<10000001; m=m*2) {
        for( int n = 320; n<m; n=n*2){
        int count = 0;
        sampling = m/327680+1;
        sampling = 1;
        magmaDoubleComplex *val, *d_val;
        TESTING_CHECK(magma_zmalloc_cpu(&val, m));
        TESTING_CHECK(magma_zmalloc(&d_val, m));
        // fill the values with random numbers
        for (int z=0; z<m; z++){
            val[z] = MAGMA_Z_MAKE(55.0*(double)(rand()%m)/(double)m, 0.0);        
        }
        
        // copy over
        magma_zsetvector( m, val, 1, d_val, 1, queue );
        
        start = magma_sync_wtime( queue );
        for(int i=0; i<10; i++)
            TESTING_CHECK(magma_zthrsholdselect(sampling, m, n, d_val, &thrs, queue));
        end = magma_sync_wtime( queue );
        t_gpu = (end-start) / 10.0;
        count = 0;
        for(int z=0; z<m; z++) {
            if (MAGMA_Z_ABS(val[z])<thrs) {
                count++;    
            }
        }
        printf("%% m n thrs count absolute-acc relative-acc time-gpu m n thrs count absolute-acc relative-acc time-cpu\n");

        printf( " %10d  %10d  %.8e  %10d %.4e %.4e\t\t %.3e", m, n, thrs, count, fabs(1.0-(float)count/(float)n), fabs((float)(n-count)/(float)m), t_gpu );
        
        // cpu reference for comparison
        A.nnz = m;
        A.val = val;
        start = magma_sync_wtime( queue );
        for(int i=0; i<10; i++)
            magma_zselectrandom( A.val, m, n, queue );
        end = magma_sync_wtime( queue );
        t_cpu = (end-start) / 10.0;
        thrs = MAGMA_Z_ABS(A.val[n]);
        count = 0;
        for(int z=0; z<m; z++) {
            if (MAGMA_Z_ABS(val[z])<thrs) {
                count++;    
            }
        }
                
        magma_free(d_val);
        magma_free_cpu(val);

        printf( " %10d  %10d  %.8e  %10d %.4e %.4e\t\t %.3e\n", m, n, thrs, count, fabs(1.0-(float)count/(float)n), fabs((float)(n-count)/(float)m), t_cpu );
    }
    }
    
    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
