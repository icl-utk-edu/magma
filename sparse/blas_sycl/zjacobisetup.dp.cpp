/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magmasparse_internal.h"

#define BLOCK_SIZE 512


#define PRECISION_z

void 
zvjacobisetup_gpu(  int num_rows, 
                    int num_vecs,
                    magmaDoubleComplex *b, 
                    magmaDoubleComplex *d, 
                    magmaDoubleComplex *c,
                    magmaDoubleComplex *x,
                    sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);

    if(row < num_rows ){
        for( int i=0; i<num_vecs; i++ ){
            c[row+i*num_rows] = b[row+i*num_rows] / d[row];
            x[row+i*num_rows] = c[row+i*num_rows];
        }
    }
}


/**
    Purpose
    -------

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Returns the vector c. It calls a GPU kernel

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                number of rows
                
    @param[in]
    b           magma_z_matrix
                RHS b

    @param[in]
    d           magma_z_matrix
                vector with diagonal entries

    @param[out]
    c           magma_z_matrix*
                c = D^(-1) * b

    @param[out]
    x           magma_z_matrix*
                iteration vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zgegpuk
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobisetup_vector_gpu(
    magma_int_t num_rows, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix c,
    magma_z_matrix *x,
    magma_queue_t queue )
{
    sycl::range<3> grid(1, 1, magma_ceildiv(num_rows, BLOCK_SIZE));
    int num_vecs = b.num_rows / num_rows;
    magma_int_t threads = BLOCK_SIZE;
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto x_val_ct5 = x->val;

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                           sycl::range<3>(1, 1, threads)),
                         [=](sycl::nd_item<3> item_ct1) {
                             zvjacobisetup_gpu(num_rows, num_vecs, b.dval,
                                               d.dval, c.dval, x_val_ct5,
                                               item_ct1);
                         });
    });

    return MAGMA_SUCCESS;
}


void 
zjacobidiagscal_kernel(  int num_rows,
                         int num_vecs, 
                    magmaDoubleComplex *b, 
                    magmaDoubleComplex *d, 
                    magmaDoubleComplex *c,
                    sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);

    if(row < num_rows ){
        for( int i=0; i<num_vecs; i++)
            c[row+i*num_rows] = b[row+i*num_rows] * d[row];
    }
}


/**
    Purpose
    -------

    Prepares the Jacobi Iteration according to
       x^(k+1) = D^(-1) * b - D^(-1) * (L+U) * x^k
       x^(k+1) =      c     -       M        * x^k.

    Returns the vector c. It calls a GPU kernel

    Arguments
    ---------

    @param[in]
    num_rows    magma_int_t
                number of rows
                
    @param[in]
    b           magma_z_matrix
                RHS b

    @param[in]
    d           magma_z_matrix
                vector with diagonal entries

    @param[out]
    c           magma_z_matrix*
                c = D^(-1) * b
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobi_diagscal(
    magma_int_t num_rows, 
    magma_z_matrix d, 
    magma_z_matrix b, 
    magma_z_matrix *c,
    magma_queue_t queue )
{
    sycl::range<3> grid(1, 1, magma_ceildiv(num_rows, 512));
    int num_vecs = b.num_rows*b.num_cols/num_rows;
    magma_int_t threads = 512;
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto c_val_ct4 = c->val;

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                           sycl::range<3>(1, 1, threads)),
                         [=](sycl::nd_item<3> item_ct1) {
                             zjacobidiagscal_kernel(num_rows, num_vecs, b.dval,
                                                    d.dval, c_val_ct4,
                                                    item_ct1);
                         });
    });

    return MAGMA_SUCCESS;
}


void 
zjacobiupdate_kernel(  int num_rows,
                       int num_cols, 
                    magmaDoubleComplex *t, 
                    magmaDoubleComplex *b, 
                    magmaDoubleComplex *d, 
                    magmaDoubleComplex *x,
                    sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);

    if(row < num_rows ){
        for( int i=0; i<num_cols; i++)
            x[row+i*num_rows] += (b[row+i*num_rows]-t[row+i*num_rows]) * d[row];
    }
}


/**
    Purpose
    -------

    Updates the iteration vector x for the Jacobi iteration
    according to
        x=x+d.*(b-t)
    where d is the diagonal of the system matrix A and t=Ax.

    Arguments
    ---------
                
    @param[in]
    t           magma_z_matrix
                t = A*x
                
    @param[in]
    b           magma_z_matrix
                RHS b
                
    @param[in]
    d           magma_z_matrix
                vector with diagonal entries

    @param[out]
    x           magma_z_matrix*
                iteration vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobiupdate(
    magma_z_matrix t, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix *x,
    magma_queue_t queue )
{
    sycl::range<3> grid(1, 1, magma_ceildiv(t.num_rows, BLOCK_SIZE));
    magma_int_t threads = BLOCK_SIZE;
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto x_dval_ct5 = x->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                           sycl::range<3>(1, 1, threads)),
                         [=](sycl::nd_item<3> item_ct1) {
                             zjacobiupdate_kernel(t.num_rows, t.num_cols,
                                                  t.dval, b.dval, d.dval,
                                                  x_dval_ct5, item_ct1);
                         });
    });

    return MAGMA_SUCCESS;
}


void 
zjacobispmvupdate_kernel(  
    int num_rows,
    int num_cols, 
    magmaDoubleComplex * dval, 
    magma_index_t * drowptr, 
    magma_index_t * dcolind,
    magmaDoubleComplex *t, 
    magmaDoubleComplex *b, 
    magmaDoubleComplex *d, 
    magmaDoubleComplex *x ,
    sycl::nd_item<3> item_ct1)
{
    int row = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
    int j;

    if(row<num_rows){
        /*
        DPCT1064:365: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int start = drowptr[ row ];
        int end = drowptr[ row+1 ];
        for( int i=0; i<num_cols; i++){
            for( j=start; j<end; j++){
                dot += dval[ j ] * x[ dcolind[j]+i*num_rows ];
            }
            x[row+i*num_rows] += (b[row+i*num_rows]-dot) * d[row];
        }
    }
}


/**
    Purpose
    -------

    Updates the iteration vector x for the Jacobi iteration
    according to
        x=x+d.*(b-Ax)


    Arguments
    ---------

    @param[in]
    maxiter     magma_int_t
                number of Jacobi iterations   
                
    @param[in]
    A           magma_z_matrix
                system matrix
                
    @param[in]
    t           magma_z_matrix
                workspace
                
    @param[in]
    b           magma_z_matrix
                RHS b
                
    @param[in]
    d           magma_z_matrix
                vector with diagonal entries

    @param[out]
    x           magma_z_matrix*
                iteration vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobispmvupdate(
    magma_int_t maxiter,
    magma_z_matrix A,
    magma_z_matrix t, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix *x,
    magma_queue_t queue )
{
    // local variables
    //magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    //magmaDoubleComplex c_one = MAGMA_Z_ONE;

    sycl::range<3> grid(1, 1, magma_ceildiv(t.num_rows, BLOCK_SIZE));
    magma_int_t threads = BLOCK_SIZE;

    for( magma_int_t i=0; i<maxiter; i++ ) {
        // distinct routines imply synchronization
        // magma_z_spmv( c_one, A, *x, c_zero, t, queue );                // t =  A * x
        // zjacobiupdate_kernel<<< grid, threads, 0, queue->sycl_stream()>>>( t.num_rows, t.num_cols, t.dval, b.dval, d.dval, x->dval );
        // merged in one implies asynchronous update
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                auto x_dval_ct8 = x->dval;

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        zjacobispmvupdate_kernel(t.num_rows, t.num_cols, A.dval,
                                                 A.drow, A.dcol, t.dval, b.dval,
                                                 d.dval, x_dval_ct8, item_ct1);
                    });
            });
    }

    return MAGMA_SUCCESS;
}
