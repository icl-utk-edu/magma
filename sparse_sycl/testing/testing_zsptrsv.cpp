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

#include <cuda_runtime.h>

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
    
    int debug = 0;
    
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
        
        
        if(debug)printf("%% --- debug mode ---");
        else { printf("prec_info = [\n");
               printf("%% row-wise: cuSOLVE, sync-free, BJ(1)-3, BJ(1)-5, BJ(12)-3, BJ(12)-5, BJ(24)-3, BJ(24)-5, ISAI(1)-0, ISAI(2)-0, ISAI(3)-0\n");
               printf("%% col-wise: prec-setup res_L time_L res_U time_U\n");
        }
        // preconditioner with cusparse trisolve
        printf("%% --- Now use cuSPARSE trisolve ---\n");
        zopts.precond_par.solver = Magma_ILU;
        zopts.precond_par.trisolver = Magma_CUSOLVE;
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        tempo2 = magma_sync_wtime( queue );
        if(debug)printf("%% time_magma_z_precondsetup = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );

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
        if(debug)printf("%% residual_L = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_L = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );
        
        // b = sptrsv(U,a)
        // c = L*b
        // d = a-c
        // res = norm(d)
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_applyprecond_right( MagmaNoTrans, A, a, &b, &zopts.precond_par, queue ));
        tempo2 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_spmv( one, zopts.precond_par.U, b, zero, c, queue ));   
        magma_zcopy( dofs, a.dval, 1 , d.dval, 1, queue );
        magma_zaxpy( dofs, mone, c.dval, 1 , d.dval, 1, queue );
        res = magma_dznrm2( dofs, d.dval, 1, queue );
        if(debug)printf("%% residual_U = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_U = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\n",tempo2-tempo1 );
        magma_zmfree(&a, queue );
        magma_zmfree(&b, queue );
        magma_zmfree(&c, queue );
        magma_zmfree(&d, queue );
        magma_zprecondfree( &zopts.precond_par , queue );

        // preconditioner with sync-free trisolve
        printf("\n%% --- Now use sync-free trisolve (under construction) ---\n");
        zopts.precond_par.solver = Magma_ILU;
        zopts.precond_par.trisolver = Magma_SYNCFREESOLVE;
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        tempo2 = magma_sync_wtime( queue );
        if(debug)printf("%% time_magma_z_precondsetup = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );

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
        if(debug)printf("%% residual_L = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_L = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );
        
        // b = sptrsv(U,a)
        // c = L*b
        // d = a-c
        // res = norm(d)
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_applyprecond_right( MagmaNoTrans, A, a, &b, &zopts.precond_par, queue ));
        tempo2 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_spmv( one, zopts.precond_par.U, b, zero, c, queue ));   
        magma_zcopy( dofs, a.dval, 1 , d.dval, 1, queue );
        magma_zaxpy( dofs, mone, c.dval, 1 , d.dval, 1, queue );
        res = magma_dznrm2( dofs, d.dval, 1, queue );
        if(debug)printf("%% residual_U = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_U = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\n",tempo2-tempo1 );
        magma_zmfree(&a, queue );
        magma_zmfree(&b, queue );
        magma_zmfree(&c, queue );
        magma_zmfree(&d, queue );
        magma_zprecondfree( &zopts.precond_par , queue );


        
                // preconditioner with blcok-Jacobi trisolve
        printf("\n%% --- Now use block-Jacobi trisolve ---\n");
        zopts.precond_par.solver = Magma_ILU;
        zopts.precond_par.trisolver = Magma_JACOBI;
        zopts.precond_par.pattern = 1;
        zopts.precond_par.maxiter = 3;
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        tempo2 = magma_sync_wtime( queue );
        if(debug)printf("%% time_magma_z_precondsetup = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );

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
        if(debug)printf("%% residual_L = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_L = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );
        
        // b = sptrsv(U,a)
        // c = L*b
        // d = a-c
        // res = norm(d)
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_applyprecond_right( MagmaNoTrans, A, a, &b, &zopts.precond_par, queue ));
        tempo2 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_spmv( one, zopts.precond_par.U, b, zero, c, queue ));   
        magma_zcopy( dofs, a.dval, 1 , d.dval, 1, queue );
        magma_zaxpy( dofs, mone, c.dval, 1 , d.dval, 1, queue );
        res = magma_dznrm2( dofs, d.dval, 1, queue );
        if(debug)printf("%% residual_U = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_U = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\n",tempo2-tempo1 );
        magma_zmfree(&a, queue );
        magma_zmfree(&b, queue );
        magma_zmfree(&c, queue );
        magma_zmfree(&d, queue );
        magma_zprecondfree( &zopts.precond_par , queue );

        
        
        
                 // preconditioner with blcok-Jacobi trisolve
        printf("\n%% --- Now use block-Jacobi trisolve ---\n");
        zopts.precond_par.solver = Magma_ILU;
        zopts.precond_par.trisolver = Magma_JACOBI;
        zopts.precond_par.pattern = 1;
        zopts.precond_par.maxiter = 5;
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        tempo2 = magma_sync_wtime( queue );
        if(debug)printf("%% time_magma_z_precondsetup = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );

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
        if(debug)printf("%% residual_L = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_L = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );
        
        // b = sptrsv(U,a)
        // c = L*b
        // d = a-c
        // res = norm(d)
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_applyprecond_right( MagmaNoTrans, A, a, &b, &zopts.precond_par, queue ));
        tempo2 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_spmv( one, zopts.precond_par.U, b, zero, c, queue ));   
        magma_zcopy( dofs, a.dval, 1 , d.dval, 1, queue );
        magma_zaxpy( dofs, mone, c.dval, 1 , d.dval, 1, queue );
        res = magma_dznrm2( dofs, d.dval, 1, queue );
        if(debug)printf("%% residual_U = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_U = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\n",tempo2-tempo1 );
        magma_zmfree(&a, queue );
        magma_zmfree(&b, queue );
        magma_zmfree(&c, queue );
        magma_zmfree(&d, queue );
        magma_zprecondfree( &zopts.precond_par , queue );

        
        
        
                 // preconditioner with blcok-Jacobi trisolve
        printf("\n%% --- Now use block-Jacobi trisolve ---\n");
        zopts.precond_par.solver = Magma_ILU;
        zopts.precond_par.trisolver = Magma_JACOBI;
        zopts.precond_par.pattern = 12;
        zopts.precond_par.maxiter = 3;
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        tempo2 = magma_sync_wtime( queue );
        if(debug)printf("%% time_magma_z_precondsetup = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );

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
        if(debug)printf("%% residual_L = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_L = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );
        
        // b = sptrsv(U,a)
        // c = L*b
        // d = a-c
        // res = norm(d)
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_applyprecond_right( MagmaNoTrans, A, a, &b, &zopts.precond_par, queue ));
        tempo2 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_spmv( one, zopts.precond_par.U, b, zero, c, queue ));   
        magma_zcopy( dofs, a.dval, 1 , d.dval, 1, queue );
        magma_zaxpy( dofs, mone, c.dval, 1 , d.dval, 1, queue );
        res = magma_dznrm2( dofs, d.dval, 1, queue );
        if(debug)printf("%% residual_U = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_U = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\n",tempo2-tempo1 );
        magma_zmfree(&a, queue );
        magma_zmfree(&b, queue );
        magma_zmfree(&c, queue );
        magma_zmfree(&d, queue );
        magma_zprecondfree( &zopts.precond_par , queue );

        
        
        
                       // preconditioner with blcok-Jacobi trisolve
        printf("\n%% --- Now use block-Jacobi trisolve ---\n");
        zopts.precond_par.solver = Magma_ILU;
        zopts.precond_par.trisolver = Magma_JACOBI;
        zopts.precond_par.pattern = 12;
        zopts.precond_par.maxiter = 5;
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        tempo2 = magma_sync_wtime( queue );
        if(debug)printf("%% time_magma_z_precondsetup = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );

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
        if(debug)printf("%% residual_L = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_L = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );
        
        // b = sptrsv(U,a)
        // c = L*b
        // d = a-c
        // res = norm(d)
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_applyprecond_right( MagmaNoTrans, A, a, &b, &zopts.precond_par, queue ));
        tempo2 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_spmv( one, zopts.precond_par.U, b, zero, c, queue ));   
        magma_zcopy( dofs, a.dval, 1 , d.dval, 1, queue );
        magma_zaxpy( dofs, mone, c.dval, 1 , d.dval, 1, queue );
        res = magma_dznrm2( dofs, d.dval, 1, queue );
        if(debug)printf("%% residual_U = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_U = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\n",tempo2-tempo1 );
        magma_zmfree(&a, queue );
        magma_zmfree(&b, queue );
        magma_zmfree(&c, queue );
        magma_zmfree(&d, queue );
        magma_zprecondfree( &zopts.precond_par , queue );

        
        
        
                        // preconditioner with blcok-Jacobi trisolve
        printf("\n%% --- Now use block-Jacobi trisolve ---\n");
        zopts.precond_par.solver = Magma_ILU;
        zopts.precond_par.trisolver = Magma_JACOBI;
        zopts.precond_par.pattern = 24;
        zopts.precond_par.maxiter = 3;
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        tempo2 = magma_sync_wtime( queue );
        if(debug)printf("%% time_magma_z_precondsetup = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );

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
        if(debug)printf("%% residual_L = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_L = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );
        
        // b = sptrsv(U,a)
        // c = L*b
        // d = a-c
        // res = norm(d)
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_applyprecond_right( MagmaNoTrans, A, a, &b, &zopts.precond_par, queue ));
        tempo2 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_spmv( one, zopts.precond_par.U, b, zero, c, queue ));   
        magma_zcopy( dofs, a.dval, 1 , d.dval, 1, queue );
        magma_zaxpy( dofs, mone, c.dval, 1 , d.dval, 1, queue );
        res = magma_dznrm2( dofs, d.dval, 1, queue );
        if(debug)printf("%% residual_U = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_U = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\n",tempo2-tempo1 );
        magma_zmfree(&a, queue );
        magma_zmfree(&b, queue );
        magma_zmfree(&c, queue );
        magma_zmfree(&d, queue );
        magma_zprecondfree( &zopts.precond_par , queue );

        
        
        
                        // preconditioner with blcok-Jacobi trisolve
        printf("\n%% --- Now use block-Jacobi trisolve ---\n");
        zopts.precond_par.solver = Magma_ILU;
        zopts.precond_par.trisolver = Magma_JACOBI;
        zopts.precond_par.pattern = 24;
        zopts.precond_par.maxiter = 5;
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        tempo2 = magma_sync_wtime( queue );
        if(debug)printf("%% time_magma_z_precondsetup = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );

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
        if(debug)printf("%% residual_L = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_L = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );
        
        // b = sptrsv(U,a)
        // c = L*b
        // d = a-c
        // res = norm(d)
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_applyprecond_right( MagmaNoTrans, A, a, &b, &zopts.precond_par, queue ));
        tempo2 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_spmv( one, zopts.precond_par.U, b, zero, c, queue ));   
        magma_zcopy( dofs, a.dval, 1 , d.dval, 1, queue );
        magma_zaxpy( dofs, mone, c.dval, 1 , d.dval, 1, queue );
        res = magma_dznrm2( dofs, d.dval, 1, queue );
        if(debug)printf("%% residual_U = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_U = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\n",tempo2-tempo1 );
        magma_zmfree(&a, queue );
        magma_zmfree(&b, queue );
        magma_zmfree(&c, queue );
        magma_zmfree(&d, queue );
        magma_zprecondfree( &zopts.precond_par , queue );

        
        
                // preconditioner with sync-free trisolve
        printf("\n%% --- Now use ISAI trisolve ---\n");
        zopts.precond_par.solver = Magma_ILU;
        zopts.precond_par.trisolver = Magma_ISAI;
        zopts.precond_par.pattern = 1;
        zopts.precond_par.maxiter = 0;
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        // insert here a check wether we went for the fallback
        if( zopts.precond_par.trisolver == Magma_ISAI ){
        tempo2 = magma_sync_wtime( queue );
        if(debug)printf("%% time_magma_z_precondsetup = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );

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
        if(debug)printf("%% residual_L = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_L = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );
        
        // b = sptrsv(U,a)
        // c = L*b
        // d = a-c
        // res = norm(d)
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_applyprecond_right( MagmaNoTrans, A, a, &b, &zopts.precond_par, queue ));
        tempo2 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_spmv( one, zopts.precond_par.U, b, zero, c, queue ));   
        magma_zcopy( dofs, a.dval, 1 , d.dval, 1, queue );
        magma_zaxpy( dofs, mone, c.dval, 1 , d.dval, 1, queue );
        res = magma_dznrm2( dofs, d.dval, 1, queue );
        if(debug)printf("%% residual_U = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_U = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\n",tempo2-tempo1 );
        } else {
            printf("NaN\tNaN\tNaN\tNaN\tNaN\n" );
        }
        magma_zmfree(&a, queue );
        magma_zmfree(&b, queue );
        magma_zmfree(&c, queue );
        magma_zmfree(&d, queue );
        magma_zprecondfree( &zopts.precond_par , queue );

        
        
        
                        // preconditioner with sync-free trisolve
        printf("\n%% --- Now use ISAI trisolve ---\n");
        zopts.precond_par.solver = Magma_ILU;
        zopts.precond_par.trisolver = Magma_ISAI;
        zopts.precond_par.pattern = 2;
        zopts.precond_par.maxiter = 0;
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        // insert here a check wether we went for the fallback
        if( zopts.precond_par.trisolver == Magma_ISAI ){
        tempo2 = magma_sync_wtime( queue );
        if(debug)printf("%% time_magma_z_precondsetup = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );

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
        if(debug)printf("%% residual_L = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_L = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );
        
        // b = sptrsv(U,a)
        // c = L*b
        // d = a-c
        // res = norm(d)
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_applyprecond_right( MagmaNoTrans, A, a, &b, &zopts.precond_par, queue ));
        tempo2 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_spmv( one, zopts.precond_par.U, b, zero, c, queue ));   
        magma_zcopy( dofs, a.dval, 1 , d.dval, 1, queue );
        magma_zaxpy( dofs, mone, c.dval, 1 , d.dval, 1, queue );
        res = magma_dznrm2( dofs, d.dval, 1, queue );
        if(debug)printf("%% residual_U = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_U = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\n",tempo2-tempo1 );
        } else {
            printf("NaN\tNaN\tNaN\tNaN\tNaN\n" );
        }
        magma_zmfree(&a, queue );
        magma_zmfree(&b, queue );
        magma_zmfree(&c, queue );
        magma_zmfree(&d, queue );
        magma_zprecondfree( &zopts.precond_par , queue );

        
        
        
                        // preconditioner with sync-free trisolve
        printf("\n%% --- Now use ISAI trisolve ---\n");
        zopts.precond_par.solver = Magma_ILU;
        zopts.precond_par.trisolver = Magma_ISAI;
        zopts.precond_par.pattern = 3;
        zopts.precond_par.maxiter = 0;
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_precondsetup( A, b, &zopts.solver_par, &zopts.precond_par, queue ) );
        // insert here a check wether we went for the fallback
        if( zopts.precond_par.trisolver == Magma_ISAI ){
        tempo2 = magma_sync_wtime( queue );
        if(debug)printf("%% time_magma_z_precondsetup = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );

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
        if(debug)printf("%% residual_L = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_L = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\t",tempo2-tempo1 );
        
        // b = sptrsv(U,a)
        // c = L*b
        // d = a-c
        // res = norm(d)
        tempo1 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_applyprecond_right( MagmaNoTrans, A, a, &b, &zopts.precond_par, queue ));
        tempo2 = magma_sync_wtime( queue );
        TESTING_CHECK( magma_z_spmv( one, zopts.precond_par.U, b, zero, c, queue ));   
        magma_zcopy( dofs, a.dval, 1 , d.dval, 1, queue );
        magma_zaxpy( dofs, mone, c.dval, 1 , d.dval, 1, queue );
        res = magma_dznrm2( dofs, d.dval, 1, queue );
        if(debug)printf("%% residual_U = %.6e\n", res );
        else printf("%.6e\t", res );
        if(debug)printf("%% time_U = %.6e\n",tempo2-tempo1 );
        else printf("%.6e\n",tempo2-tempo1 );
        } else {
            printf("NaN\tNaN\tNaN\tNaN\tNaN\n" );
        }
        magma_zmfree(&a, queue );
        magma_zmfree(&b, queue );
        magma_zmfree(&c, queue );
        magma_zmfree(&d, queue );
        magma_zprecondfree( &zopts.precond_par , queue );

        
        if(debug)printf("%% --- completed ---");
        else printf("];\n");
        
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
