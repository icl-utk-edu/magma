/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c

       @author Stan Tomov
       @author Mark Gates
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"

#define NB 16

// nearly same code in ztranspose_inplace.cu

/******************************************************************************/
// grid is (n/nb) x ((n/nb)/2 + 1), where n/nb is odd.
// lower indicates blocks in lower triangle of grid, including diagonal.
// lower blocks cover left side of matrix, including diagonal.
// upper blocks swap block indices (x,y) and shift by grid width (or width-1)
// to cover right side of matrix.
//      [ A00 A01 A02 ]                  [ A00  .   .  |  .   .  ]
//      [ A10 A11 A12 ]                  [ A10 A11  .  |  .   .  ]
// grid [ A20 A21 A22 ] covers matrix as [ A20 A21 A22 |  .   .  ]
//      [ A30 A31 A32 ]                  [ A30 A31 A32 | A01  .  ]
//      [ A40 A41 A42 ]                  [ A40 A41 A42 | A02 A12 ]
//
// See ztranspose_conj_inplace_even for description of threads.

void ztranspose_conj_inplace_odd( int n, magmaDoubleComplex *matrix, int lda ,
                                  sycl::nd_item<3> item_ct1,
                                  sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
                                  sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{

    int i = item_ct1.get_local_id(2);
    int j = item_ct1.get_local_id(1);

    bool lower = (item_ct1.get_group(2) >= item_ct1.get_group(1));
    int ii =
        (lower ? item_ct1.get_group(2)
               : (item_ct1.get_group(1) + item_ct1.get_group_range(1) - 1));
    int jj = (lower ? item_ct1.get_group(1)
                    : (item_ct1.get_group(2) + item_ct1.get_group_range(1)));

    ii *= NB;
    jj *= NB;

    magmaDoubleComplex *A = matrix + ii+i + (jj+j)*lda;
    if ( ii == jj ) {
        if ( ii+i < n && jj+j < n ) {
            sA[j][i] = MAGMA_Z_CONJ( *A );
        }
        /*
        DPCT1065:1404: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if ( ii+i < n && jj+j < n ) {
            *A = sA[i][j];
        }
    }
    else {
        magmaDoubleComplex *B = matrix + jj+i + (ii+j)*lda;
        if ( ii+i < n && jj+j < n ) {
            sA[j][i] = MAGMA_Z_CONJ( *A );
        }
        if ( jj+i < n && ii+j < n ) {
            sB[j][i] = MAGMA_Z_CONJ( *B );
        }
        /*
        DPCT1065:1405: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if ( ii+i < n && jj+j < n ) {
            *A = sB[i][j];
        }
        if ( jj+i < n && ii+j < n ) {
            *B = sA[i][j];
        }
    }
}


/******************************************************************************/
// grid is ((n/nb) + 1) x (n/nb)/2, where n/nb is even.
// lower indicates blocks in strictly lower triangle of grid, excluding diagonal.
// lower blocks shift up by one to cover left side of matrix including diagonal.
// upper blocks swap block indices (x,y) and shift by grid width
// to cover right side of matrix.
//      [ A00  A01 ]                  [ A10  .  |  .   .  ]
//      [ A10  A11 ]                  [ A20 A21 |  .   .  ]
// grid [ A20  A21 ] covers matrix as [ A30 A31 | A00  .  ]
//      [ A30  A31 ]                  [ A40 A41 | A01 A11 ]
//      [ A40  A41 ]
//
// Each block is NB x NB threads.
// For non-diagonal block A, block B is symmetric block.
// Thread (i,j) loads A(i,j) into sA(j,i) and B(i,j) into sB(j,i), i.e., transposed,
// syncs, then saves sA(i,j) to B(i,j) and sB(i,j) to A(i,j).
// Threads outside the matrix do not touch memory.

void ztranspose_conj_inplace_even( int n, magmaDoubleComplex *matrix, int lda ,
                                   sycl::nd_item<3> item_ct1,
                                   sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write, sycl::access::target::local> sA,
                                   sycl::accessor<magmaDoubleComplex, 2, sycl::access_mode::read_write, sycl::access::target::local> sB)
{

    int i = item_ct1.get_local_id(2);
    int j = item_ct1.get_local_id(1);

    bool lower = (item_ct1.get_group(2) > item_ct1.get_group(1));
    int ii = (lower ? (item_ct1.get_group(2) - 1)
                    : (item_ct1.get_group(1) + item_ct1.get_group_range(1)));
    int jj = (lower ? (item_ct1.get_group(1))
                    : (item_ct1.get_group(2) + item_ct1.get_group_range(1)));

    ii *= NB;
    jj *= NB;

    magmaDoubleComplex *A = matrix + ii+i + (jj+j)*lda;
    if ( ii == jj ) {
        if ( ii+i < n && jj+j < n ) {
            sA[j][i] = MAGMA_Z_CONJ( *A );
        }
        /*
        DPCT1065:1406: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if ( ii+i < n && jj+j < n ) {
            *A = sA[i][j];
        }
    }
    else {
        magmaDoubleComplex *B = matrix + jj+i + (ii+j)*lda;
        if ( ii+i < n && jj+j < n ) {
            sA[j][i] = MAGMA_Z_CONJ( *A );
        }
        if ( jj+i < n && ii+j < n ) {
            sB[j][i] = MAGMA_Z_CONJ( *B );
        }
        /*
        DPCT1065:1407: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        if ( ii+i < n && jj+j < n ) {
            *A = sB[i][j];
        }
        if ( jj+i < n && ii+j < n ) {
            *B = sA[i][j];
        }
    }
}


/***************************************************************************//**
    Purpose
    -------
    ztranspose_conj_inplace_q conjugate-transposes a square N-by-N matrix in-place.
    
    Same as ztranspose_conj_inplace, but adds queue argument.
    
    Arguments
    ---------
    @param[in]
    n       INTEGER
            The number of rows & columns of the matrix dA.  N >= 0.
    
    @param[in]
    dA      COMPLEX_16 array, dimension (LDDA,N)
            The N-by-N matrix dA.
            On exit, dA(j,i) = dA_original(i,j), for 0 <= i,j < N.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array dA.  LDDA >= N.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.
    
    @ingroup magma_transpose
*******************************************************************************/
extern "C" void
magmablas_ztranspose_conj_inplace(
    magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( n < 0 )
        info = -1;
    else if ( ldda < n )
        info = -3;
    
    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }

    sycl::range<3> threads(1, NB, NB);
    int nblock = magma_ceildiv( n, NB );
    
    // need 1/2 * (nblock+1) * nblock to cover lower triangle and diagonal of matrix.
    // block assignment differs depending on whether nblock is odd or even.
    if ( nblock % 2 == 1 ) {
        sycl::range<3> grid(1, (nblock + 1) / 2, nblock);
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 2,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sA_acc_ct1(sycl::range<2>(NB, NB+1), cgh);
                sycl::accessor<magmaDoubleComplex, 2,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sB_acc_ct1(sycl::range<2>(NB, NB+1), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     ztranspose_conj_inplace_odd(
                                         n, dA, ldda, item_ct1, sA_acc_ct1,
                                         sB_acc_ct1);
                                 });
            });
    }
    else {
        sycl::range<3> grid(1, nblock / 2, nblock + 1);
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<magmaDoubleComplex, 2,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sA_acc_ct1(sycl::range<2>(NB, NB+1), cgh);
                sycl::accessor<magmaDoubleComplex, 2,
                               sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sB_acc_ct1(sycl::range<2>(NB, NB+1), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     ztranspose_conj_inplace_even(
                                         n, dA, ldda, item_ct1, sA_acc_ct1,
                                         sB_acc_ct1);
                                 });
            });
    }
}
