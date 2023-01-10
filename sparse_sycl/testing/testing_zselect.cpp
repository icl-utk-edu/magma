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


const int SEED = 352302;
void makeRandomArray(magmaDoubleComplex* a, int size) {
    srand(SEED);
    //int rand_max = 100;
    for (int i=0; i<size; i++) {
        a[i] = MAGMA_Z_MAKE( ((double) rand() / (RAND_MAX)), 0.0 );
    }
}

/* ////////////////////////////////////////////////////////////////////////////
   -- testing for the magma_zselect magma_zselectrandom magma_zselectsort functions
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    /* Initialize */
    TESTING_CHECK( magma_init() );
    magma_print_environment();
    magma_queue_t queue=NULL;
    magma_queue_create( 0, &queue );
    // using std::swap;
    real_Double_t start, end, t_select, t_selectrandom, t_selectbitonic;
    
    int size = atoi(argv[1]);
    int selectset = atoi(argv[2]);
    // not sure whether this shoudl go here...
    // selectset--;
    magmaDoubleComplex* a = NULL;
    TESTING_CHECK( magma_zmalloc_cpu( &a, size ) );
    
    makeRandomArray(a, size);
#if defined(DEBUG)
    for(int i=0; i<size; i++)
        printf("%.2f\t", a[i] );
#endif
    start = magma_sync_wtime( queue );
    magma_zselect(a, size, selectset, queue);
    end = magma_sync_wtime( queue );
    t_select = end-start;
    magmaDoubleComplex selectResult = a[selectset];
//#if defined(DEBUG)
    printf("\n selected by select: %.2f\n\n", MAGMA_Z_ABS(selectResult) );
//#endif

    makeRandomArray(a, size);
    start = magma_sync_wtime( queue );
    magma_zselectrandom(a, size, selectset, queue);
    end = magma_sync_wtime( queue );
    t_selectrandom = end-start;
    magmaDoubleComplex selectRandomResult = a[selectset];
//#if defined(DEBUG)
    printf("\n selected by ranomized select: %.2f\n\n", MAGMA_Z_ABS(selectResult) );
//#endif
/*
    makeRandomArray(a, size);
    clock_t sortBegin = clock();
    std::sort(a, a+size);
    clock_t sortEnd = clock();
    double sortResult = a[selectset];
#if defined(DEBUG)
    printf("\n selected: %.2f\n\n", sortResult );
#endif
*/
    if (!(selectResult == selectRandomResult) ){
        printf(" Inconsistent result.\n");
    }
    
    makeRandomArray(a, size);
    start = magma_sync_wtime( queue );
    magma_int_t flag =0;
    magma_zbitonic_sort(0, size, a, flag, queue);
    end = magma_sync_wtime( queue );
    t_selectbitonic = end-start;
    magmaDoubleComplex BitonicResult = a[selectset];
//#if defined(DEBUG)
    printf("\n selected by bitonic sort: %.2f\n\n", MAGMA_Z_ABS(selectResult) );
//#endif
/*
    makeRandomArray(a, size);
    clock_t sortBegin = clock();
    std::sort(a, a+size);
    clock_t sortEnd = clock();
    double sortResult = a[selectset];
#if defined(DEBUG)
    printf("\n selected: %.2f\n\n", sortResult );
#endif
*/
    if (!(BitonicResult == selectRandomResult) ){
        printf(" Inconsistent result.\n");
    }
    
    
    printf(" Select time (ms): %.4f\n", double(t_select)*1000 );
    printf(" Randomized select time (ms): %.4f\n", double(t_selectrandom)*1000 );
    printf(" Bitonicsort time (ms): %.4f\n", double(t_selectbitonic)*1000 );

    // magma_free_cpu( &a );
    
    magma_queue_destroy( queue );
    magma_finalize();
    return info;
}
