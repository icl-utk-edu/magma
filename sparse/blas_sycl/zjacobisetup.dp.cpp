/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s
       @author Hartwig Anzt

*/
#include <CL/sycl.hpp>
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
    /*
    DPCT1049:362: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
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
    /*
    DPCT1049:363: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
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
    /*
    DPCT1049:364: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
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
        /*
        DPCT1049:366: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
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


void 
zjacobispmvupdate_bw_kernel(  
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
    int row_tmp = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);
    int row = num_rows-1 - row_tmp;
    int j;

    if( row>-1 ){
        /*
        DPCT1064:367: Migrated make_cuDoubleComplex call is used in a macro
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
    This kernel processes the thread blocks in reversed order.

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
magma_zjacobispmvupdate_bw(
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
        /*
        DPCT1049:368: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                auto x_dval_ct8 = x->dval;

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        zjacobispmvupdate_bw_kernel(
                            t.num_rows, t.num_cols, A.dval, A.drow, A.dcol,
                            t.dval, b.dval, d.dval, x_dval_ct8, item_ct1);
                    });
            });
    }

    return MAGMA_SUCCESS;
}


void 
zjacobispmvupdateselect_kernel(  
    int num_rows,
    int num_cols, 
    int num_updates, 
    magma_index_t * indices, 
    magmaDoubleComplex * dval, 
    magma_index_t * drowptr, 
    magma_index_t * dcolind,
    magmaDoubleComplex *t, 
    magmaDoubleComplex *b, 
    magmaDoubleComplex *d, 
    magmaDoubleComplex *x,
    magmaDoubleComplex *y ,
    sycl::nd_item<3> item_ct1,
    const sycl::stream &stream_ct1)
{
    int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);
    int j;

    if(  idx<num_updates){
        int row = indices[ idx ];
        stream_ct1 << " ";
        //if( row < num_rows ){
        /*
        DPCT1064:369: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        magmaDoubleComplex dot = MAGMA_Z_ZERO;
        int start = drowptr[ row ];
        int end = drowptr[ row+1 ];
        for( int i=0; i<num_cols; i++){
            for( j=start; j<end; j++){
                dot += dval[ j ] * x[ dcolind[j]+i*num_rows ];
            }
            x[row+i*num_rows] = x[row+i*num_rows] + (b[row+i*num_rows]-dot) * d[row];
            
            //magmaDoubleComplex add = (b[row+i*num_rows]-dot) * d[row];
            //#if defined(PRECISION_s) //|| defined(PRECISION_d)
            //    atomicAdd( x + row + i*num_rows, add );  
            //#endif
            // ( unsigned int* address, unsigned int val);
        //}
        }
    }
}


/**
    Purpose
    -------

    Updates the iteration vector x for the Jacobi iteration
    according to
        x=x+d.*(b-Ax)
        
    This kernel allows for overlapping domains: the indices-array contains
    the locations that are updated. Locations may be repeated to simulate
    overlapping domains.


    Arguments
    ---------

    @param[in]
    maxiter     magma_int_t
                number of Jacobi iterations
                
    @param[in]
    num_updates magma_int_t
                number of updates - length of the indices array
                    
    @param[in]
    indices     magma_index_t*
                indices, which entries of x to update
                
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
   
    @param[in]
    tmp         magma_z_matrix
                workspace

    @param[out]
    x           magma_z_matrix*
                iteration vector
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zjacobispmvupdateselect(
    magma_int_t maxiter,
    magma_int_t num_updates,
    magma_index_t *indices,
    magma_z_matrix A,
    magma_z_matrix t, 
    magma_z_matrix b, 
    magma_z_matrix d, 
    magma_z_matrix tmp, 
    magma_z_matrix *x,
    magma_queue_t queue )
{
    // local variables
    //magmaDoubleComplex c_zero = MAGMA_Z_ZERO
    //magmaDoubleComplex c_one = MAGMA_Z_ONE;
    
    //magma_z_matrix swp;

    sycl::range<3> grid(1, 1, magma_ceildiv(num_updates, BLOCK_SIZE));
    magma_int_t threads = BLOCK_SIZE;
    printf("num updates:%d %d %d\n", int(num_updates), int(threads),
           int(grid[2]));

    for( magma_int_t i=0; i<maxiter; i++ ) {
        /*
        DPCT1049:370: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::stream stream_ct1(64 * 1024, 80, cgh);

                auto x_dval_ct10 = x->dval;

                cgh.parallel_for(
                    sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                      sycl::range<3>(1, 1, threads)),
                    [=](sycl::nd_item<3> item_ct1) {
                        zjacobispmvupdateselect_kernel(
                            t.num_rows, t.num_cols, num_updates, indices,
                            A.dval, A.drow, A.dcol, t.dval, b.dval, d.dval,
                            x_dval_ct10, tmp.dval, item_ct1, stream_ct1);
                    });
            });
        //swp.dval = x->dval;
        //x->dval = tmp.dval;
        //tmp.dval = swp.dval;
    }
    
    return MAGMA_SUCCESS;
}


void 
zftjacobicontractions_kernel(
    int num_rows,
    magmaDoubleComplex * xkm2val, 
    magmaDoubleComplex * xkm1val, 
    magmaDoubleComplex * xkval, 
    magmaDoubleComplex * zval,
    magmaDoubleComplex * cval ,
    sycl::nd_item<3> item_ct1)
{
    int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);

    if(  idx<num_rows ){
        /*
        DPCT1064:371: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        zval[idx] = MAGMA_Z_MAKE(MAGMA_Z_ABS(xkm1val[idx] - xkval[idx]), 0.0);
        /*
        DPCT1064:372: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        cval[idx] = MAGMA_Z_MAKE(MAGMA_Z_ABS(xkm2val[idx] - xkm1val[idx]) /
                                     MAGMA_Z_ABS(xkm1val[idx] - xkval[idx]),
                                 0.0);
    }
}


/**
    Purpose
    -------

    Computes the contraction coefficients c_i:
    
    c_i = z_i^{k-1} / z_i^{k} 
        
        = | x_i^{k-1} - x_i^{k-2} | / |  x_i^{k} - x_i^{k-1} |

    Arguments
    ---------

    @param[in]
    xkm2        magma_z_matrix
                vector x^{k-2}
                
    @param[in]
    xkm1        magma_z_matrix
                vector x^{k-2}
                
    @param[in]
    xk          magma_z_matrix
                vector x^{k-2}
   
    @param[out]
    z           magma_z_matrix*
                ratio
                
    @param[out]
    c           magma_z_matrix*
                contraction coefficients
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zftjacobicontractions(
    magma_z_matrix xkm2,
    magma_z_matrix xkm1, 
    magma_z_matrix xk, 
    magma_z_matrix *z,
    magma_z_matrix *c,
    magma_queue_t queue )
{
    sycl::range<3> grid(1, 1, magma_ceildiv(xk.num_rows, BLOCK_SIZE));
    magma_int_t threads = BLOCK_SIZE;

    /*
    DPCT1049:373: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto z_dval_ct4 = z->dval;
        auto c_dval_ct5 = c->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                           sycl::range<3>(1, 1, threads)),
                         [=](sycl::nd_item<3> item_ct1) {
                             zftjacobicontractions_kernel(
                                 xkm2.num_rows, xkm2.dval, xkm1.dval, xk.dval,
                                 z_dval_ct4, c_dval_ct5, item_ct1);
                         });
    });

    return MAGMA_SUCCESS;
}


void 
zftjacobiupdatecheck_kernel(
    int num_rows,
    double delta,
    magmaDoubleComplex * xold, 
    magmaDoubleComplex * xnew, 
    magmaDoubleComplex * zprev,
    magmaDoubleComplex * cval, 
    magma_int_t *flag_t,
    magma_int_t *flag_fp ,
    sycl::nd_item<3> item_ct1)
{
    int idx = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
              item_ct1.get_local_id(2);

    if(  idx<num_rows ){
        double t1 = delta * MAGMA_Z_ABS(cval[idx]);
        double  vkv = 1.0;
        for( magma_int_t i=0; i<min( flag_fp[idx], 100 ); i++){
            vkv = vkv*2;
        }
        magmaDoubleComplex xold_l = xold[idx];
        magmaDoubleComplex xnew_l = xnew[idx];
        /*
        DPCT1064:374: Migrated make_cuDoubleComplex call is used in a macro
        definition and is not valid for all macro uses. Adjust the code.
        */
        magmaDoubleComplex znew =
            MAGMA_Z_MAKE(max(MAGMA_Z_ABS(xold_l - xnew_l), 1e-15), 0.0);

        magmaDoubleComplex znr = zprev[idx] / znew; 
        double t2 = MAGMA_Z_ABS( znr - cval[idx] );
        
        //% evaluate fp-cond
        magma_int_t fpcond = 0;
        if( MAGMA_Z_ABS(znr)>vkv ){
            fpcond = 1;
        }
        
        // % combine t-cond and fp-cond + flag_t == 1
        magma_int_t cond = 0;
        if( t2<t1 || (flag_t[idx]>0 && fpcond > 0 ) ){
            cond = 1;
        }
        flag_fp[idx] = flag_fp[idx]+1;
        if( fpcond>0 ){
            flag_fp[idx] = 0;
        }
        if( cond > 0 ){
            flag_t[idx] = 0;
            zprev[idx] = znew;
            xold[idx] = xnew_l;
        } else {
            flag_t[idx] = 1;
            xnew[idx] = xold_l;
        }
    }
}


/**
    Purpose
    -------

    Checks the Jacobi updates accorting to the condition in the ScaLA'15 paper.

    Arguments
    ---------
    
    @param[in]
    delta       double
                threshold

    @param[in,out]
    xold        magma_z_matrix*
                vector xold
                
    @param[in,out]
    xnew        magma_z_matrix*
                vector xnew
                
    @param[in,out]
    zprev       magma_z_matrix*
                vector z = | x_k-1 - x_k |
   
    @param[in]
    c           magma_z_matrix
                contraction coefficients
                
    @param[in,out]
    flag_t      magma_int_t
                threshold condition
                
    @param[in,out]
    flag_fp     magma_int_t
                false positive condition
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_z
    ********************************************************************/

extern "C" magma_int_t
magma_zftjacobiupdatecheck(
    double delta,
    magma_z_matrix *xold,
    magma_z_matrix *xnew, 
    magma_z_matrix *zprev, 
    magma_z_matrix c,
    magma_int_t *flag_t,
    magma_int_t *flag_fp,
    magma_queue_t queue )
{
    sycl::range<3> grid(1, 1, magma_ceildiv(xnew->num_rows, BLOCK_SIZE));
    magma_int_t threads = BLOCK_SIZE;

    /*
    DPCT1049:375: The work-group size passed to the SYCL kernel may exceed the
    limit. To get the device limit, query info::device::max_work_group_size.
    Adjust the work-group size if needed.
    */
    ((sycl::queue *)(queue->sycl_stream()))->submit([&](sycl::handler &cgh) {
        auto xold_num_rows_ct0 = xold->num_rows;
        auto xold_dval_ct2 = xold->dval;
        auto xnew_dval_ct3 = xnew->dval;
        auto zprev_dval_ct4 = zprev->dval;

        cgh.parallel_for(sycl::nd_range<3>(grid * sycl::range<3>(1, 1, threads),
                                           sycl::range<3>(1, 1, threads)),
                         [=](sycl::nd_item<3> item_ct1) {
                             zftjacobiupdatecheck_kernel(
                                 xold_num_rows_ct0, delta, xold_dval_ct2,
                                 xnew_dval_ct3, zprev_dval_ct4, c.dval, flag_t,
                                 flag_fp, item_ct1);
                         });
    });

    return MAGMA_SUCCESS;
}
