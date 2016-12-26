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
    
    magmaDoubleComplex one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex zero = MAGMA_Z_MAKE(0.0, 0.0);
    magmaDoubleComplex mone = MAGMA_Z_MAKE(-1.0, 0.0);
    magma_z_matrix A={Magma_CSR}, a={Magma_CSR}, b={Magma_CSR};
    magma_z_matrix c={Magma_CSR}, d={Magma_CSR};
    magma_int_t dofs;
    double res;
    
    //Chronometry
    real_Double_t tempo1, tempo2;
    
    int i=1;
    TESTING_CHECK( magma_zparse_opts( argc, argv, &zopts, &i, queue ));
    zopts.solver_par.solver = Magma_PBICGSTAB;

    TESTING_CHECK( magma_zsolverinfo_init( &zopts.solver_par, &zopts.precond_par, queue ));

    while( i < argc ) {
        if ( strcmp("LAPLACE2D", argv[i]) == 0 && i+1 < argc ) {   // Laplace test
            i++;
            magma_int_t laplace_size = atoi( argv[i] );
            TESTING_CHECK( magma_zm_5stencil(  laplace_size, &A, queue ));
        } else {                        // file-matrix test
            TESTING_CHECK( magma_z_csr_mtx( &A,  argv[i], queue ));
        }
        dofs = A.num_rows;
        
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
        
        // preconditioner
        zopts.precond_par.solver = Magma_ILU;
        TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );

        // vectors and initial guess
        TESTING_CHECK( magma_zvinit( &a, Magma_DEV, A.num_rows, 1, one, queue ));
        TESTING_CHECK( magma_zvinit( &b, Magma_DEV, A.num_rows, 1, zero, queue ));
        TESTING_CHECK( magma_zvinit( &c, Magma_DEV, A.num_rows, 1, zero, queue ));
        TESTING_CHECK( magma_zvinit( &d, Magma_DEV, A.num_rows, 1, zero, queue ));
        
        
        // b = sptrsv(L,a)
        // c = L*b
        // d = a-c
        // res = norm(d)
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_applyprecond_left( MagmaNoTrans, A, a, &b, &zopts.precond_par, queue ));
        tempo2 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_spmv( one, zopts.precond_par.L, b, zero, c, queue ));   
        magma_zcopy( dofs, a.dval, 1 , d.dval, 1, queue );
        magma_zaxpy( dofs, mone, c.dval, 1 , d.dval, 1, queue );
        res = magma_dznrm2( dofs, d.dval, 1, queue );
        
        printf("residual_L = %.6e\n", res );
        printf("time_L = %.6e\n",tempo2-tempo1 );
        
        // b = sptrsv(U,a)
        // c = L*b
        // d = a-c
        // res = norm(d)
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_applyprecond_left( MagmaNoTrans, A, a, &b, &zopts.precond_par, queue ));
        tempo2 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_spmv( one, zopts.precond_par.L, b, zero, c, queue ));   
        magma_zcopy( dofs, a.dval, 1 , d.dval, 1, queue );
        magma_zaxpy( dofs, mone, c.dval, 1 , d.dval, 1, queue );
        res = magma_dznrm2( dofs, d.dval, 1, queue );
        
        printf("residual_U = %.6e\n", res );
        printf("time_U = %.6e\n",tempo2-tempo1 );
        
        
        magma_zmfree(&A, queue );
        magma_zmfree(&b, queue );
        magma_zmfree(&c, queue );
        magma_zmfree(&d, queue );
        i++;
    }

    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
