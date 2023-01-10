/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Hartwig Anzt

       @precisions normal z -> s d c
*/
#include "magmasparse_internal.h"

#include <cuda.h>  // for CUDA_VERSION

#define PRECISION_z


/***************************************************************************//**
    Purpose
    -------

    Left-hand-side application of ISAI preconditioner.


    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                input RHS b

    @param[in,out]
    x           magma_z_matrix
                solution x

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_zisai_l(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue ){
    magma_int_t info = 0;

    if( precond->maxiter == 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->LD, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
    } else if( precond->maxiter > 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->LD, b, MAGMA_Z_ZERO, precond->d, queue ); // d=L_d^(-1)b
        magma_z_spmv( MAGMA_Z_ONE, precond->LD, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
        for( int z=0; z<precond->maxiter; z++ ){
            magma_z_spmv( MAGMA_Z_ONE, precond->L, *x, MAGMA_Z_ZERO, precond->work1, queue ); // work1 = L * x
            magma_z_spmv( MAGMA_Z_ONE, precond->LD, precond->work1, MAGMA_Z_ZERO, precond->work2, queue ); // work2 = L_d^(-1)work1
            magma_zaxpy( b.num_rows*b.num_cols, -MAGMA_Z_ONE, precond->work2.dval, 1 , x->dval, 1, queue );        // x = - work2
            magma_zaxpy( b.num_rows*b.num_cols, MAGMA_Z_ONE, precond->d.dval, 1 , x->dval, 1, queue );        // x = d + x = L_d^(-1)b - work2 = L_d^(-1)b - L_d^(-1) * L * x
        }
    }

    return info;
}


/***************************************************************************//**
    Purpose
    -------

    Right-hand-side application of ISAI preconditioner.


    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                input RHS b

    @param[in,out]
    x           magma_z_matrix
                solution x

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_zisai_r(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue ){
    magma_int_t info = 0;

    if( precond->maxiter == 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->UD, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
    } else if( precond->maxiter > 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->UD, b, MAGMA_Z_ZERO, precond->d, queue ); // d=L^(-1)b
        magma_z_spmv( MAGMA_Z_ONE, precond->UD, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
        for( int z=0; z<precond->maxiter; z++ ){
            magma_z_spmv( MAGMA_Z_ONE, precond->U, *x, MAGMA_Z_ZERO, precond->work1, queue ); // work1=b+Lb
            magma_z_spmv( MAGMA_Z_ONE, precond->UD, precond->work1, MAGMA_Z_ZERO, precond->work2, queue ); // x=x+L^(-1)work1
            magma_zaxpy( b.num_rows*b.num_cols, -MAGMA_Z_ONE, precond->work2.dval, 1 , x->dval, 1, queue );        // t = t + c
            magma_zaxpy( b.num_rows*b.num_cols, MAGMA_Z_ONE, precond->d.dval, 1 , x->dval, 1, queue );        // t = t + c
        }
    }

    return info;
}


/***************************************************************************//**
    Purpose
    -------

    Left-hand-side application of ISAI preconditioner. Transpose.


    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                input RHS b

    @param[in,out]
    x           magma_z_matrix
                solution x

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_zisai_l_t(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue ){
    magma_int_t info = 0;

    if( precond->maxiter == 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->LDT, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
    } else if( precond->maxiter > 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->LDT, b, MAGMA_Z_ZERO, precond->d, queue ); // d=M_L*b
        magma_z_spmv( MAGMA_Z_ONE, precond->LDT, b, MAGMA_Z_ZERO, *x, queue ); // x = M_L*b
        for( int z=0; z<precond->maxiter; z++ ){
            magma_z_spmv( MAGMA_Z_ONE, precond->LT, *x, MAGMA_Z_ZERO, precond->work1, queue ); // work1=L*M_L*b
            magma_z_spmv( MAGMA_Z_ONE, precond->LDT, precond->work1, MAGMA_Z_ZERO, precond->work2, queue ); // work2 = M_L*L*M_L*b
            magma_zaxpy( b.num_rows*b.num_cols, -MAGMA_Z_ONE, precond->work2.dval, 1 , x->dval, 1, queue );        // x = M_L*x -M_L*L*M_L*b
            magma_zaxpy( b.num_rows*b.num_cols, MAGMA_Z_ONE, precond->d.dval, 1 , x->dval, 1, queue );        // x = M_L*x + M_L*b - M_L *L*M_L *b 
        }
    }

    return info;
}


/***************************************************************************//**
    Purpose
    -------

    Right-hand-side application of ISAI preconditioner. Transpose.


    Arguments
    ---------

    @param[in]
    b           magma_z_matrix
                input RHS b

    @param[in,out]
    x           magma_z_matrix
                solution x

    @param[in,out]
    precond     magma_z_preconditioner*
                preconditioner parameters

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgepr
    ********************************************************************/
extern "C"
magma_int_t
magma_zisai_r_t(
    magma_z_matrix b,
    magma_z_matrix *x,
    magma_z_preconditioner *precond,
    magma_queue_t queue ){
    magma_int_t info = 0;

    if( precond->maxiter == 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->UDT, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
    } else if( precond->maxiter > 0 ){
        magma_z_spmv( MAGMA_Z_ONE, precond->UDT, b, MAGMA_Z_ZERO, precond->d, queue ); // d=L^(-1)b
        magma_z_spmv( MAGMA_Z_ONE, precond->UDT, b, MAGMA_Z_ZERO, *x, queue ); // SPAI
        for( int z=0; z<precond->maxiter; z++ ){
            magma_z_spmv( MAGMA_Z_ONE, precond->UT, *x, MAGMA_Z_ZERO, precond->work1, queue ); // work1=b+Lb
            magma_z_spmv( MAGMA_Z_ONE, precond->UDT, precond->work1, MAGMA_Z_ZERO, precond->work2, queue ); // x=x+L^(-1)work1
            magma_zaxpy( b.num_rows*b.num_cols, -MAGMA_Z_ONE, precond->work2.dval, 1 , x->dval, 1, queue );        // t = t + c
            magma_zaxpy( b.num_rows*b.num_cols, MAGMA_Z_ONE, precond->d.dval, 1 , x->dval, 1, queue );        // t = t + c
        }
    }

    return info;
}
