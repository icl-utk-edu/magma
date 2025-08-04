/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Stephen Wood
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

/*******************************************************************************
    Purpose
    -------
    
    ZGEMV  performs one of the matrix-vector operations on the CPU

    y := alpha*A*x + beta*y,   or   y := alpha*A^T*x + beta*y,   or

    y := alpha*A^H*x + beta*y,

    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.
    
    Arguments
    ---------
    @param[in]
    transA      transA is CHARACTER*1
                On entry, TRANS specifies the operation to be performed as
                follows:

                transA = 'N' or 'n'   y := alpha*A*x + beta*y.

                transA = 'T' or 't'   y := alpha*A^T*x + beta*y.

                transA = 'C' or 'c'   y := alpha*A^H*x + beta*y.
    @param[in]
    flip        magma_int_t
                0: transA unchanged, 1: transA reversed
                
    @param[in]
    m          magma_int_t
                number of rows of the matrix A.

    @param[in]
    n          magma_int_t
                number of columns of the matrix A.                
                
    @param[in]
    alpha       magmaDoubleComplex
                scalar.
                
    @param[in]
    dA          magma_z_matrix
                input matrix dA.
                
    @param[in]
    aoff        magma_int_t
                the offset for the elements of dA.  
                
    @param[in]
    ldda        magma_int_t
                the increment for the elements of dA.     
                
    @param[in]
    dx          magma_z_matrix
                input vector dx.            
                
    @param[in]
    xoff        magma_int_t
                the offset for the elements of dx.
                
    @param[in]
    incx        magma_int_t
                the increment for the elements of dx.
                
    @param[in]
    beta        magmaDoubleComplex
                scalar.            
                
    @param[in,out]
    dy          magma_z_matrix *
                input vector dy.  
                
    @param[in]
    yoff        magma_int_t
                the offset for the elements of dy.
                
    @param[in]
    incy        magma_int_t
                the increment for the elements of dy.        
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
*******************************************************************************/

extern "C" 
void
magmablas_zgemv_cpu(
    magma_trans_t transA, magma_int_t flip,
    magma_int_t m, magma_int_t n, 
    magmaDoubleComplex alpha, 
    magma_z_matrix dA, magma_int_t aoff, magma_int_t ldda,
    magma_z_matrix dx, magma_int_t xoff, magma_int_t incx, 
    magmaDoubleComplex beta,
    magma_z_matrix *dy, magma_int_t yoff, magma_int_t incy, 
    magma_queue_t queue )
{
    magma_int_t info = MAGMA_NOTCONVERGED;
    magma_z_matrix A = {Magma_CSR}, x = {Magma_CSR}, y = {Magma_CSR};
    
    TESTING_CHECK( magma_zmtransfer( dA, &A, Magma_DEV, Magma_CPU, queue ));
    TESTING_CHECK( magma_zmtransfer( dx, &x, Magma_DEV, Magma_CPU, queue ));
    TESTING_CHECK( magma_zmtransfer( *dy, &y, Magma_DEV, Magma_CPU, queue ));
    printf("\nmagmablas_zgemv_cpu m=%d, n=%d ldda=%d\n", 
      A.num_rows, A.num_cols, A.ld);
    
    if (flip==0) {
        blasf77_zgemv( lapack_trans_const(transA), 
            &m, &n, &alpha, &A.val[aoff], &ldda, &x.val[xoff], 
            &incx, &beta, &y.val[yoff], &incy); 
    }
    else if (flip==1) {
      magma_trans_t transtmp;
      if (transA==MagmaNoTrans)  
        transtmp = MagmaTrans;
      else if (transA==MagmaTrans)
        transtmp = MagmaNoTrans;
      else if (transA==MagmaConjTrans) {
        transtmp = MagmaNoTrans; 
        // TODO: congugate A.
      }
      
      blasf77_zgemv( lapack_trans_const(transtmp), 
          &m, &n, &alpha, &A.val[aoff], &ldda, &x.val[xoff], 
          &incx, &beta, &y.val[yoff], &incy); 
      
    }
    
    TESTING_CHECK( magma_zmtransfer( y, dy, Magma_CPU, Magma_DEV, queue ));
    
cleanup:
    // free resources
    magma_zmfree( &A, queue );
    magma_zmfree( &x, queue );
    magma_zmfree( &y, queue );
}


/*******************************************************************************
    Purpose
    -------
    
    Prints the residual of an input vector with respect to two reference 
    vectors.
    
    Arguments
    ---------
    @param[in]
    dofs        magma_int_t
                number of elements in vectors.
                
    @param[in]
    dy          magmaDoubleComplex_ptr
                input vector dx.

    @param[in]
    dyTrans     magmaDoubleComplex_ptr
                first reference vector dx.                
                
    @param[in]
    dyNoTrans   magmaDoubleComplex_ptr
                second reference vector dx.
                
    @param[in]
    dytmp       magmaDoubleComplex_ptr
                workspace.
                
    @param[in]
    ref         magmaDoubleComplex
                reference residual.  
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
*******************************************************************************/

extern "C" 
void
magma_print_residual(
    magma_int_t dofs,
    magmaDoubleComplex_ptr dy,
    magmaDoubleComplex_ptr dyTrans,
    magmaDoubleComplex_ptr dyNoTrans,
    magmaDoubleComplex_ptr dytmp,
    magmaDoubleComplex ref, 
    magma_queue_t queue )
{
#define PRECISION_z
#if defined(PRECISION_d)  
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    magmaDoubleComplex res;
    magma_zcopy( dofs, dy, 1, dytmp, 1, queue ); // dy to dytmp
    magma_zaxpy( dofs, c_neg_one, dyTrans, 1, dy, 1, queue );  // dy = dy - y_ref
    res = magma_dznrm2( dofs, dy, 1, queue );                // res = ||dy||
    res /= ref;
    printf("|y-yTrans|/|yTrans| = %20.16e\n", res);
    magma_zaxpy( dofs, c_neg_one, dyNoTrans, 1, dytmp, 1, queue );  // dy = dy - y_ref
    res = magma_dznrm2( dofs, dytmp, 1, queue );                // res = ||dy||
    res /= ref;
    printf("|y-yNoTrans|/|yNoTrans| = %20.16e\n", res);
#endif  
}


/* ////////////////////////////////////////////////////////////////////////////
   -- testing_zgemv() to determine interaction of row-major and column-major 
   ordering between zvinit(), magmablas_zgemv() on GPU, and blasf77_zgemv() on 
   CPU through hard-coded A matrix of doubles, a unit vector, x, and reference 
   solutions yNoTrans = A*x, yTrans = A'*x.
   
   TODO: implement tests for complex A and x. 
*/
int main(  int argc, char** argv )
{
    magma_int_t info = 0;
    TESTING_CHECK( magma_init() );
    magma_print_environment();

    magma_zopts zopts;
    magma_queue_t queue;
    magma_queue_create( 0, &queue );
    
    magmaDoubleComplex c_one = MAGMA_Z_MAKE(1.0, 0.0);
    magmaDoubleComplex c_zero = MAGMA_Z_MAKE(0.0, 0.0);
    const magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    const magma_int_t c_flip = 0;
    
    magmaDoubleComplex ref, refsub, refsublr, res;
    magma_z_matrix A={Magma_CSR}, dA={Magma_CSR}; 
    magma_z_matrix dx={Magma_CSR}, dy={Magma_CSR}, dytmp={Magma_CSR};
    magma_z_matrix yNoTrans={Magma_CSR}, dyNoTrans={Magma_CSR};
    magma_z_matrix yTrans={Magma_CSR}, dyTrans={Magma_CSR};
    magma_z_matrix yNoTranssub={Magma_CSR}, dyNoTranssub={Magma_CSR};
    magma_z_matrix yTranssub={Magma_CSR}, dyTranssub={Magma_CSR};
    magma_z_matrix yNoTranssublr={Magma_CSR}, dyNoTranssublr={Magma_CSR};
    magma_z_matrix yTranssublr={Magma_CSR}, dyTranssublr={Magma_CSR};
    
#define PRECISION_z
#if defined(PRECISION_d) 
    
    TESTING_CHECK( magma_zvinit( &A, Magma_CPU, 4, 4, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dA, Magma_DEV, 4, 4, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dx, Magma_DEV, 4, 1, c_one, queue ));
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 4, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dytmp, Magma_DEV, 4, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &yNoTrans, Magma_CPU, 4, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dyNoTrans, Magma_DEV, 4, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &yTrans, Magma_CPU, 4, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dyTrans, Magma_DEV, 4, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &yNoTranssub, Magma_CPU, 4, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dyNoTranssub, Magma_DEV, 4, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &yTranssub, Magma_CPU, 4, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dyTranssub, Magma_DEV, 4, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &yNoTranssublr, Magma_CPU, 4, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dyNoTranssublr, Magma_DEV, 4, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &yTranssublr, Magma_CPU, 4, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dyTranssublr, Magma_DEV, 4, 1, c_zero, queue ));

    printf("+++++++++++++++++++++++++++++++++\n");
    printf("ZGEMV operation on GPU and CPU compared for 4x4 matrix\n");   
    printf("+++++++++++++++++++++++++++++++++\n");
    for (magma_int_t k=0; k<A.num_rows*A.num_cols; k++) {
      A.val[k] = MAGMA_Z_MAKE(k, 0.);    
    }
    printf("A set on cpu\n");
    yTrans.val[0]=24.;   yTrans.val[1]=28.;    yTrans.val[2]=32.;    yTrans.val[3]=36.;
    yNoTrans.val[0]=6.;  yNoTrans.val[1]=22.;  yNoTrans.val[2]=38.;  yNoTrans.val[3]=54.;
    yTranssub.val[2]=38.;     yTranssub.val[3]=54.;
    yNoTranssub.val[0]=20.;   yNoTranssub.val[1]=22.;   yNoTranssub.val[2]=24.;   yNoTranssub.val[3]=26.;
    yTranssublr.val[2]=24.;     yTranssublr.val[3]=26.;
    yNoTranssublr.val[2]=21.;   yNoTranssublr.val[3]=29.;
    
    printf("y refs set on cpu\n");  
    TESTING_CHECK( magma_zmtransfer( A, &dA, Magma_CPU, Magma_DEV, queue ));
    TESTING_CHECK( magma_zmtransfer( yNoTrans, &dyNoTrans, Magma_CPU, Magma_DEV, queue ));
    TESTING_CHECK( magma_zmtransfer( yTrans, &dyTrans, Magma_CPU, Magma_DEV, queue ));
    TESTING_CHECK( magma_zmtransfer( yNoTranssub, &dyNoTranssub, Magma_CPU, Magma_DEV, queue ));
    TESTING_CHECK( magma_zmtransfer( yTranssub, &dyTranssub, Magma_CPU, Magma_DEV, queue ));
    TESTING_CHECK( magma_zmtransfer( yNoTranssublr, &dyNoTranssublr, Magma_CPU, Magma_DEV, queue ));
    TESTING_CHECK( magma_zmtransfer( yTranssublr, &dyTranssublr, Magma_CPU, Magma_DEV, queue ));
    
    printf("Reference answer yTrans:\n");
    magma_zprint_matrix(yTrans, queue);
    printf("Reference answer yNoTrans:\n");
    magma_zprint_matrix(yNoTrans, queue);
    
    ref = magma_dznrm2( A.num_rows, dyNoTrans.dval, 1, queue );
    refsub = magma_dznrm2( 2, &dyNoTranssub.dval[2], 1, queue );
    refsublr = magma_dznrm2( 2, &dyNoTranssublr.dval[2], 1, queue );
    
    printf("=======\tbefore magma_zgemv NoTrans line %d\n", __LINE__);
    printf("A:\n");
    magma_zprint_matrix(dA, queue);
    printf("x:\n");
    magma_zprint_matrix(dx, queue);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magmablas_zgemv( MagmaNoTrans, dA.num_rows, dA.num_cols, c_one, dA.dval, dA.ld, dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv NoTrans row x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    magmablas_zgemv_cpu( MagmaNoTrans, c_flip, dA.num_rows, dA.num_cols, c_one, dA, 0, dA.ld, dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu NoTrans row x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 4, 1, c_zero, queue ));
    
    printf("=======\tbefore magma_zgemv Trans line %d\n", __LINE__);
    magmablas_zgemv( MagmaTrans, dA.num_rows, dA.num_cols, c_one, dA.dval, dA.ld, dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv Trans row x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    magmablas_zgemv_cpu( MagmaTrans, c_flip, dA.num_rows, dA.num_cols, c_one, dA, 0, dA.ld, dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu Trans line row x col; ld %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 4, 1, c_zero, queue ));
    
    printf("=======\tbefore magma_zgemv ConjTrans line %d\n", __LINE__);
    magmablas_zgemv( MagmaConjTrans, dA.num_rows, dA.num_cols, c_one, dA.dval, dA.ld, dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv ConjTrans row x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    magmablas_zgemv_cpu( MagmaConjTrans, c_flip, dA.num_rows, dA.num_cols, c_one, dA, 0, dA.ld, dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu ConjTrans row x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 4, 1, c_zero, queue ));
    
    printf("=======\tbefore magma_zgemv NoTrans 2x4 lower submatrix line %d\n", __LINE__);
    magmablas_zgemv( MagmaNoTrans, 2, dA.num_cols, c_one, &dA.dval[2*dA.ld], dA.ld, 
      dx.dval, 1, c_zero, &dy.dval[2], 1, queue );
    printf("\tafter magma_zgemv NoTrans subrow x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( 2, &dy.dval[2], 
      &dyTranssub.dval[2], &dyNoTranssub.dval[2], 
      &dytmp.dval[2], refsub, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 4, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaNoTrans, c_flip, 2, dA.num_cols, c_one, dA, 2*dA.ld, dA.ld, 
      dx, 0, 1, c_zero, &dy, 2, 1, queue );
    printf("\tafter magma_zgemv_cpu NoTrans subrow x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( 2, &dy.dval[2], 
      &dyTranssub.dval[2], &dyNoTranssub.dval[2], 
      &dytmp.dval[2], refsub, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 4, 1, c_zero, queue ));
    magmablas_zgemv( MagmaNoTrans, dA.num_cols, 2, c_one, &dA.dval[2*dA.ld], dA.ld, 
      dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv NoTrans col x subrow; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( 2, &dy.dval[2], 
      &dyTranssub.dval[2], &dyNoTranssub.dval[2], 
      &dytmp.dval[2], refsub, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 4, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaNoTrans, c_flip, dA.num_cols, 2, c_one, dA, 2*dA.ld, dA.ld, 
      dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu NoTrans col x subrow; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( 2, &dy.dval[2], 
      &dyTranssub.dval[2], &dyNoTranssub.dval[2], 
      &dytmp.dval[2], refsub, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 4, 1, c_zero, queue ));
    magmablas_zgemv( MagmaTrans, dA.num_cols, 2, c_one, &dA.dval[2*dA.ld], dA.ld, 
      dx.dval, 1, c_zero, &dy.dval[2], 1, queue );
    printf("\tafter magma_zgemv Trans col x subrow; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( 2, &dy.dval[2], 
      &dyTranssub.dval[2], &dyNoTranssub.dval[2], 
      &dytmp.dval[2], refsub, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 4, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaTrans, c_flip, dA.num_cols, 2, c_one, dA, 2*dA.ld, dA.ld, 
      dx, 0, 1, c_zero, &dy, 2, 1, queue );
    printf("\tafter magma_zgemv_cpu Trans col x subrow; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( 2, &dy.dval[2], 
      &dyTranssub.dval[2], &dyNoTranssub.dval[2], 
      &dytmp.dval[2], refsub, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 4, 1, c_zero, queue ));
    
    printf("=======\tbefore magma_zgemv NoTrans 2x2 lower right submatrix line %d\n", __LINE__);
    magmablas_zgemv( MagmaNoTrans, 2, 2, c_one, &dA.dval[2*dA.ld+2], dA.ld, 
      &dx.dval[2], 1, c_zero, &dy.dval[2], 1, queue );
    printf("\tafter magma_zgemv NoTrans subrow x subcol; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( 2, &dy.dval[2], 
      &dyTranssublr.dval[2], &dyNoTranssublr.dval[2], 
      &dytmp.dval[2], refsublr, queue );
    
    magmablas_zgemv_cpu( MagmaNoTrans, c_flip, 2, 2, c_one, dA, 2*dA.ld+2, dA.ld, 
      dx, 2, 1, c_zero, &dy, 2, 1, queue );
    printf("\tafter magma_zgemv_cpu NoTrans subrow x subcol; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( 2, &dy.dval[2], 
      &dyTranssublr.dval[2], &dyNoTranssublr.dval[2], 
      &dytmp.dval[2], refsublr, queue );
    
    magma_zmfree( &A, queue );
    magma_zmfree( &dA, queue );
    magma_zmfree( &dx, queue );
    magma_zmfree( &dy, queue );
    magma_zmfree( &yNoTrans, queue );
    magma_zmfree( &dyNoTrans, queue );
    magma_zmfree( &yTrans, queue );
    magma_zmfree( &dyTrans, queue );
    magma_zmfree( &yNoTranssub, queue );
    magma_zmfree( &dyNoTranssub, queue );
    magma_zmfree( &yTranssub, queue );
    magma_zmfree( &dyTranssub, queue );
    magma_zmfree( &yNoTranssublr, queue );
    magma_zmfree( &dyNoTranssublr, queue );
    magma_zmfree( &yTranssublr, queue );
    magma_zmfree( &dyTranssublr, queue );
    
    printf("\n\n+++++++++++++++++++++++++++++++++\n");
    printf("ZGEMV operation on GPU and CPU compared for 5x3 matrix\n"); 
    printf("+++++++++++++++++++++++++++++++++\n");
    
    TESTING_CHECK( magma_zvinit( &A, Magma_CPU, 5, 3, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dA, Magma_DEV, 5, 3, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dx, Magma_DEV, 5, 1, c_one, queue ));
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dytmp, Magma_DEV, 5, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &yNoTrans, Magma_CPU, 5, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dyNoTrans, Magma_DEV, 5, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &yTrans, Magma_CPU, 5, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dyTrans, Magma_DEV, 5, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &yNoTranssub, Magma_CPU, 5, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dyNoTranssub, Magma_DEV, 5, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &yTranssub, Magma_CPU, 5, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dyTranssub, Magma_DEV, 5, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &yNoTranssublr, Magma_CPU, 5, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dyNoTranssublr, Magma_DEV, 5, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &yTranssublr, Magma_CPU, 5, 1, c_zero, queue ));
    TESTING_CHECK( magma_zvinit( &dyTranssublr, Magma_DEV, 5, 1, c_zero, queue ));
    
    for (magma_int_t k=0; k<A.num_rows*A.num_cols; k++) {
      A.val[k] = MAGMA_Z_MAKE(k, 0.);    
    }
    
    yTrans.val[0]=30.;   yTrans.val[1]=35.;    yTrans.val[2]=40.;    yTrans.val[3]=0.;    yTrans.val[4]=0.;
    yNoTrans.val[0]=3.;  yNoTrans.val[1]=12.;  yNoTrans.val[2]=21.;  yNoTrans.val[3]=30.;  yNoTrans.val[4]=39.;
    yTranssub.val[0]=21.;   yTranssub.val[1]=23.;    yTranssub.val[2]=25.;    yTranssub.val[3]=0.;    yTranssub.val[4]=0.;
    yNoTranssub.val[0]=30.;  yNoTranssub.val[1]=39.;  yNoTranssub.val[2]=0.;  yNoTranssub.val[3]=0.;  yNoTranssub.val[4]=0.;
    yTranssublr.val[0]=30.;   yTranssublr.val[1]=33.;    yTranssublr.val[2]=0.;    yTranssublr.val[3]=0.;    yTranssublr.val[4]=0.;
    yNoTranssublr.val[0]=15.;  yNoTranssublr.val[1]=21.;  yNoTranssublr.val[2]=27.;  yNoTranssublr.val[3]=0.;  yNoTranssublr.val[4]=0.;
    
    TESTING_CHECK( magma_zmtransfer( A, &dA, Magma_CPU, Magma_DEV, queue ));
    TESTING_CHECK( magma_zmtransfer( yNoTrans, &dyNoTrans, Magma_CPU, Magma_DEV, queue ));
    TESTING_CHECK( magma_zmtransfer( yTrans, &dyTrans, Magma_CPU, Magma_DEV, queue ));
    TESTING_CHECK( magma_zmtransfer( yNoTranssub, &dyNoTranssub, Magma_CPU, Magma_DEV, queue ));
    TESTING_CHECK( magma_zmtransfer( yTranssub, &dyTranssub, Magma_CPU, Magma_DEV, queue ));
    TESTING_CHECK( magma_zmtransfer( yNoTranssublr, &dyNoTranssublr, Magma_CPU, Magma_DEV, queue ));
    TESTING_CHECK( magma_zmtransfer( yTranssublr, &dyTranssublr, Magma_CPU, Magma_DEV, queue ));
    
    printf("yTrans:\n");
    magma_zprint_matrix(yTrans, queue);
    printf("yNoTrans:\n");
    magma_zprint_matrix(yNoTrans, queue);
    
    ref = magma_dznrm2( A.num_rows, dyNoTrans.dval, 1, queue );
    refsub = magma_dznrm2( A.num_rows, dyNoTranssub.dval, 1, queue );
    refsublr = magma_dznrm2( A.num_rows, dyNoTranssublr.dval, 1, queue );
    
    printf("=======\tbefore magma_zgemv NoTrans line %d\n", __LINE__);
    printf("A:\n");
    magma_zprint_matrix(A, queue);
    printf("x:\n");
    magma_zprint_matrix(dx, queue);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);

    printf("A.num_rows=%d A.num_cols=%d A.ld=%d\n", dA.num_rows, dA.num_cols, dA.ld);
    magmablas_zgemv( MagmaNoTrans, dA.num_rows, dA.num_cols, c_one, dA.dval, dA.ld, dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv NoTrans row x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaNoTrans, c_flip, dA.num_rows, dA.num_cols, c_one, dA, 0, dA.ld, dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu NoTrans row x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaNoTrans, c_flip, dA.num_cols, dA.num_rows, c_one, dA, 0, dA.ld, dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu NoTrans col x row; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv( MagmaNoTrans, dA.num_cols, dA.num_rows, c_one, dA.dval, dA.num_cols, dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv NoTrans col x row; col line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaNoTrans, c_flip, dA.num_cols, dA.num_rows, c_one, dA, 0, dA.num_cols, dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu NoTrans col x row; col line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
   
    printf("=======\tbefore magma_zgemv NoTrans 2x3 lower submatrix line %d\n", __LINE__);
    
    printf("subrow addressing by ld:\n");
    for (int p=0; p<dA.num_rows; p++) {
      printf("%d %e\n", p, A.val[p*dA.ld]);  
    }
    printf("subrow addressing by num_cols:\n");
    for (int p=0; p<dA.num_rows; p++) {
      printf("%d %e\n", p, A.val[p*dA.num_cols]);  
    }
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv( MagmaNoTrans, 2, dA.num_cols, c_one, &dA.dval[3*dA.num_cols], dA.ld, 
      dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv NoTrans subrow x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTranssub.dval, dyNoTranssub.dval, 
      dytmp.dval, refsub, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaNoTrans, c_flip, 2, dA.num_cols, c_one, dA, 3*dA.num_cols, dA.ld,
      dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu NoTrans subrow x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTranssub.dval, dyNoTranssub.dval, 
      dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv( MagmaNoTrans, dA.num_cols, 2, c_one, &dA.dval[3*dA.num_cols], dA.ld, 
      dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv NoTrans col x subrow; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTranssub.dval, dyNoTranssub.dval, 
      dytmp.dval, refsub, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaNoTrans, c_flip, dA.num_cols, 2, c_one, dA, 3*dA.num_cols, dA.ld,
      dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu NoTrans col x subrow; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTranssub.dval, dyNoTranssub.dval, 
      dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv( MagmaNoTrans, dA.num_cols, 2, c_one, &dA.dval[3*dA.num_cols], dA.num_cols, 
      dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv NoTrans col x subrow; col line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTranssub.dval, dyNoTranssub.dval, 
      dytmp.dval, refsub, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaNoTrans, c_flip, dA.num_cols, 2, c_one, dA, 3*dA.num_cols, dA.num_cols,
      dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu NoTrans col x subrow; col line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTranssub.dval, dyNoTranssub.dval, 
      dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv( MagmaTrans, dA.num_cols, 2, c_one, &dA.dval[3*dA.num_cols], dA.num_cols, 
      dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv Trans col x subrow; col line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTranssub.dval, dyNoTranssub.dval, 
      dytmp.dval, refsub, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaTrans, c_flip, dA.num_cols, 2, c_one, dA, 3*dA.num_cols, dA.num_cols,
      dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu Trans col x subrow; col line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTranssub.dval, dyNoTranssub.dval, 
      dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv( MagmaTrans, dA.num_cols, 2, c_one, &dA.dval[3*dA.num_cols], dA.ld, 
      dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv Trans col x subrow; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTranssub.dval, dyNoTranssub.dval, 
      dytmp.dval, refsub, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaTrans, c_flip, dA.num_cols, 2, c_one, dA, 3*dA.num_cols, dA.ld,
      dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu Trans col x subrow; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTranssub.dval, dyNoTranssub.dval, 
      dytmp.dval, ref, queue );
    
    printf("=======\tbefore magma_zgemv NoTrans 3x2 lower right submatrix line %d\n", __LINE__);
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv( MagmaNoTrans, 2, 3, c_one, &dA.dval[2*dA.num_cols+1], dA.num_cols, 
      dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv NoTrans col x subrow; col line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTranssublr.dval, dyNoTranssublr.dval, 
      dytmp.dval, refsub, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaNoTrans, c_flip, 2, 3, c_one, dA, 2*dA.num_cols+1, dA.num_cols,
      dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu NoTrans col x subrow; col line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTranssublr.dval, dyNoTranssublr.dval, 
      dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv( MagmaTrans, 2, 3, c_one, &dA.dval[2*dA.num_cols+1], dA.num_cols, 
      dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv Trans col x subrow; col line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTranssublr.dval, dyNoTranssublr.dval, 
      dytmp.dval, refsub, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaTrans, c_flip, 2, 3, c_one, dA, 2*dA.num_cols+1, dA.num_cols,
      dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu Trans col x subrow; col line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTranssublr.dval, dyNoTranssublr.dval, 
      dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    
    printf("=======\tbefore magma_zgemv Trans line %d\n", __LINE__);
    printf("A:\n");
    magma_zprint_matrix(A, queue);
    printf("x:\n");
    magma_zprint_matrix(dx, queue);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    
    printf("A.num_rows=%d A.num_cols=%d A.ld=%d\n", A.num_rows, A.num_cols, A.ld);
    magmablas_zgemv( MagmaTrans, dA.num_rows, dA.num_cols, c_one, dA.dval, dA.ld, dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv Trans row x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaTrans, c_flip, dA.num_rows, dA.num_cols, c_one, dA, 0, dA.ld, dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu Trans row x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    
    printf("=======\tbefore magma_zgemv Trans line %d\n", __LINE__);
    printf("A:\n");
    magma_zprint_matrix(dA, queue);
    printf("x:\n");
    magma_zprint_matrix(dx, queue);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    
    printf("A.num_rows=%d A.num_cols=%d A.ld=%d\n", A.num_rows, A.num_cols, A.ld);
    magmablas_zgemv( MagmaTrans, dA.num_cols, dA.num_rows, c_one, dA.dval, dA.num_cols, dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv Trans col x row; col line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaTrans, c_flip, dA.num_cols, dA.num_rows, c_one, dA, 0, dA.num_cols, dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv Trans col x row; col line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    
    printf("=======\tbefore magma_zgemv ConjTrans line %d\n", __LINE__);
    printf("A:\n");
    magma_zprint_matrix(A, queue);
    printf("x:\n");
    magma_zprint_matrix(dx, queue);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    
    printf("A.num_rows=%d A.num_cols=%d A.ld=%d\n", A.num_rows, A.num_cols, A.ld);
    magmablas_zgemv( MagmaConjTrans, dA.num_rows, dA.num_cols, c_one, dA.dval, dA.ld, dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv ConjTrans row x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaConjTrans, c_flip, dA.num_rows, dA.num_cols, c_one, dA, 0, dA.ld, dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv ConjTrans row x col; ld line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    
    printf("\tbefore magma_zgemv Trans line %d\n", __LINE__);
    printf("A:\n");
    magma_zprint_matrix(A, queue);
    printf("x:\n");
    magma_zprint_matrix(dx, queue);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    
    printf("A.num_rows=%d A.num_cols=%d A.ld=%d\n", A.num_rows, A.num_cols, A.ld);
    magmablas_zgemv( MagmaConjTrans, dA.num_cols, dA.num_rows, c_one, dA.dval, dA.num_cols, dx.dval, 1, c_zero, dy.dval, 1, queue );
    printf("\tafter magma_zgemv ConjTrans col x row; col line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
    TESTING_CHECK( magma_zvinit( &dy, Magma_DEV, 5, 1, c_zero, queue ));
    magmablas_zgemv_cpu( MagmaConjTrans, c_flip, dA.num_cols, dA.num_rows, c_one, dA, 0, dA.num_cols, dx, 0, 1, c_zero, &dy, 0, 1, queue );
    printf("\tafter magma_zgemv_cpu ConjTrans col x row; col line %d\n", __LINE__);
    printf("y:\n");
    magma_zprint_matrix(dy, queue);
    magma_print_residual( A.num_rows, dy.dval, dyTrans.dval, dyNoTrans.dval, dytmp.dval, ref, queue );
    
#endif   

    magma_zmfree( &A, queue );
    magma_zmfree( &dA, queue );
    magma_zmfree( &dx, queue );
    magma_zmfree( &dy, queue );
    magma_zmfree( &yNoTrans, queue );
    magma_zmfree( &dyNoTrans, queue );
    magma_zmfree( &yTrans, queue );
    magma_zmfree( &dyTrans, queue );
    magma_zmfree( &yNoTranssub, queue );
    magma_zmfree( &dyNoTranssub, queue );
    magma_zmfree( &yTranssub, queue );
    magma_zmfree( &dyTranssub, queue );
    magma_zmfree( &yNoTranssublr, queue );
    magma_zmfree( &dyNoTranssublr, queue );
    magma_zmfree( &yTranssublr, queue );
    magma_zmfree( &dyTranssublr, queue );
    
    magma_queue_destroy( queue );
    TESTING_CHECK( magma_finalize() );
    return info;
}
