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
   -- testing any solver
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    magma_queue_t queue = NULL;
    magma_queue_create( 0, &queue );
    /*
    magma_z_matrix Z={Magma_CSR};
    
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

        magma_zmfree(&Z, queue );

        i++;
    }
    printf("%%=============================================================%%\n");
    printf("];\n");
   */ 
    //const magma_index_t* rowidx = new magma_index_t*[10];
    magma_index_t* rowidx={NULL}, *rowptr={NULL};// = {2,6,22,8,33,12,5,1,0,6};

    const magma_int_t num_rows = 10;
    magma_index_malloc_cpu(&rowidx,num_rows);
    magma_index_malloc_cpu(&rowptr, num_rows+1);
  
    magma_int_t nnz=0;

    magma_index_t* d_rowidx=NULL;
    magma_index_t* d_rowptr=NULL;
    
    for(int i = 0;i<10;i++)
    {
        rowidx[i] = (rand()%10);
    }

    magma_index_malloc(&d_rowidx,num_rows);
    magma_index_malloc(&d_rowptr,num_rows+1);
    //magma_index_malloc_cpu(&rowptr,num_rows+1);

    magma_index_setvector(num_rows,rowidx,1,d_rowidx,1,queue);

    TESTING_CHECK(magma_zget_row_ptr(num_rows,&nnz,d_rowidx,d_rowptr,queue));
    
    magma_index_getvector(num_rows+1,d_rowptr,1,rowptr,1,queue);
    int j =0;
    for(int i = 0;i<num_rows;++i)
    {   
        if(i==num_rows){
            j  = num_rows-1;
        }
        else{
            j = i;
        }
        printf("Row %d has %d nnz and rowptr of %d\n", i,rowidx[j],rowptr[i]);
    }
    printf("total number of nz:%d\n", rowptr[num_rows]);
    magma_free(d_rowidx);
    magma_free(d_rowptr);

    magma_free_cpu(rowptr);
    magma_free_cpu(rowidx);
    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
