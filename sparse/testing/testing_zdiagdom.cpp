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

    magma_zopts zopts;
    magma_queue_t queue;
    magma_queue_create( 0, &queue );
    
    magma_z_matrix A={Magma_CSR};
    magma_z_matrix x={Magma_CSR}, b={Magma_CSR};
    double min_dd, max_dd, avg_dd;
    char *filename = "blocksizes";
    int nlength = 675200000;
    
    int i=1;
    TESTING_CHECK( magma_zparse_opts( argc, argv, &zopts, &i, queue ));

    TESTING_CHECK( magma_zsolverinfo_init( &zopts.solver_par, &zopts.precond_par, queue ));

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_zm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_z_csr_mtx( &A,  argv[i], queue ));
        }

        // for the eigensolver case
        zopts.solver_par.ev_length = A.num_cols;
        TESTING_CHECK( magma_zeigensolverinfo_init( &zopts.solver_par, queue ));

        // scale matrix
        TESTING_CHECK( magma_zmscale( &A, zopts.scaling, queue ));
        
        // preconditioner
        if ( zopts.solver_par.solver != Magma_ITERREF ) {
            TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        }

        printf( "\n%% matrix info: %lld-by-%lld with %lld nonzeros\n\n",
                            (long long) A.num_rows, (long long) A.num_cols, (long long) A.nnz );
        
        printf("matrixinfo = [\n");
        printf("%%   size   (m x n)     ||   nonzeros (nnz)   ||   nnz/m   ||   stored nnz\n");
        printf("%%============================================================================%%\n");
        printf("  %8lld  %8lld      %10lld             %4lld        %10lld\n",
               (long long) A.num_rows, (long long) A.num_cols, (long long) A.true_nnz,
               (long long) (A.true_nnz/A.num_rows), (long long) A.nnz );
        printf("%%============================================================================%%\n");
        printf("];\n");
        //TESTING_CHECK( magma_zvread( &x, nlength, filename, queue ) );
        TESTING_CHECK( magma_zmdiagdom( zopts.precond_par.M, &min_dd, &max_dd, &avg_dd, queue ) );
        printf("diagonaldominance = [\n");
        printf("%%   min     ||   max   ||   avg\n");
        printf("%%============================================================================%%\n");
        printf("  %4.4f      %4.4f      %4.4f\n",
                  min_dd, max_dd, avg_dd );
        printf("%%============================================================================%%\n");
        printf("];\n");
    // begin ugly workaround
    magma_int_t *blocksizes=NULL, *blocksizes2=NULL, *start=NULL, *v=NULL;
    magma_int_t blockcount=0, blockcount2=0;
    
    int maxblocksize = 12;
    int current_size = 0;
    int prev_matches = 0;

    TESTING_CHECK( magma_imalloc_cpu( &v, A.num_rows+10 ));
    TESTING_CHECK( magma_imalloc_cpu( &start, A.num_rows+10 ));
    TESTING_CHECK( magma_imalloc_cpu( &blocksizes, A.num_rows+10 ));
    TESTING_CHECK( magma_imalloc_cpu( &blocksizes2, A.num_rows+10 ));
// begin ugly workaround
    maxblocksize = maxblocksize - 1;
    v[0] = 1;

    for( magma_int_t i=1; i<A.num_rows; i++ ){
        // pattern matches the pattern of the previous row
        int match = 0; // 0 means match!
        if( prev_matches == maxblocksize ){ // bounded by maxblocksize
            match = 1; // no match
            prev_matches = 0;
        } else if( ((A.row[i+1]-A.row[i])-(A.row[i]-A.row[i-1])) != 0 ){
            match = 1; // no match
            prev_matches = 0;
        } else {
            magma_index_t length = (A.row[i+1]-A.row[i]);
            magma_index_t start1 = A.row[i-1];
            magma_index_t start2 = A.row[i];
            for( magma_index_t j=0; j<length; j++ ){
                if( A.col[ start1+j ] != A.col[ start2+j ] ){
                    match = 1;
                    prev_matches = 0;
                }
            }
            if( match == 0 ){
                prev_matches++; // add one match to the block
            }
        }
        v[ i ] = match;
    }

    // start = find[v];
    blockcount = 0;
    for( magma_int_t i=0; i<A.num_rows; i++ ){
        if( v[i] == 1 ){
            start[blockcount] = i;
            blockcount++;
        }
    }
    start[blockcount] = A.num_rows;

    for( magma_int_t i=0; i<blockcount; i++ ){
        blocksizes[i] = start[i+1] - start[i];
        if( blocksizes[i] > maxblocksize ){
            // maxblocksize = blocksizes[i];
            // printf("%% warning: at i=%5lld blocksize required is %5lld\n",
            //                                                (long long) i, (long long) blocksizes[i] );
        }
    }

    current_size = 0;
    blockcount2=0;
    for( magma_int_t i=0; i<blockcount; i++ ){
        if( current_size + blocksizes[i] > maxblocksize ){
            blocksizes2[ blockcount2 ] = current_size;
            blockcount2++;
            current_size = blocksizes[i]; // form new block
        } else {
            current_size = current_size + blocksizes[i]; // add to previous block
        }
        blocksizes[i] = start[i+1] - start[i];
    }
    blocksizes2[ blockcount2 ] = current_size;
    blockcount2++;
    
    TESTING_CHECK( magma_zvinit( &x, Magma_CPU, blockcount2, 1, MAGMA_Z_ZERO, queue ));
    for( magma_int_t i=0; i<blockcount2; i++ ){
        x.val[i] = MAGMA_Z_MAKE(blocksizes2[i], 0.0 );
       // printf("bsz=%d\n", blocksizes2[i]);
    }
// end ugly workaround
        
        
        TESTING_CHECK( magma_zmbdiagdom( zopts.precond_par.M, x, &min_dd, &max_dd, &avg_dd, queue ) );
        printf("blockdiagonaldominance = [\n");
        printf("%%   min     ||   max   ||   avg\n");
        printf("%%============================================================================%%\n");
        printf("  %4.4f      %4.4f      %4.4f\n",
                  min_dd, max_dd, avg_dd );
        printf("%%============================================================================%%\n");
        printf("];\n");

        magma_zmfree(&A, queue );
        magma_zmfree(&x, queue );
        magma_zmfree(&b, queue );
        i++;
    }

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
