/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Mark Gates
*/
#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"
#include <cmath>

#define COMPLEX

#define NB_X 64

/* Computes row sums dwork[i] = sum( abs( A(i,:) )), i=0:m-1, for || A ||_inf,
 * where m and n are any size.
 * Has ceil( m/NB_X ) blocks of NB_X threads. Each thread does one row.
 * See also zlange_max_kernel code, below. */
extern "C" void
zlange_inf_kernel(
    int m, int n,
    const magmaDoubleComplex * __restrict__ A, int lda,
    double * __restrict__ dwork , sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * NB_X + item_ct1.get_local_id(2);
    double rsum[4] = {0, 0, 0, 0};
    int n_mod_4 = n % 4;
    n -= n_mod_4;
    
    // if beyond last row, skip row
    if ( i < m ) {
        A += i;
        
        if ( n >= 4 ) {
            const magmaDoubleComplex *Aend = A + lda*n;
            magmaDoubleComplex rA[4] = { A[0], A[lda], A[2*lda], A[3*lda] };
            A += 4*lda;
            
            while( A < Aend ) {
                rsum[0] += MAGMA_Z_ABS( rA[0] );  rA[0] = A[0];
                rsum[1] += MAGMA_Z_ABS( rA[1] );  rA[1] = A[lda];
                rsum[2] += MAGMA_Z_ABS( rA[2] );  rA[2] = A[2*lda];
                rsum[3] += MAGMA_Z_ABS( rA[3] );  rA[3] = A[3*lda];
                A += 4*lda;
            }
            
            rsum[0] += MAGMA_Z_ABS( rA[0] );
            rsum[1] += MAGMA_Z_ABS( rA[1] );
            rsum[2] += MAGMA_Z_ABS( rA[2] );
            rsum[3] += MAGMA_Z_ABS( rA[3] );
        }
    
        /* clean up code */
        switch( n_mod_4 ) {
            case 0:
                break;
    
            case 1:
                rsum[0] += MAGMA_Z_ABS( A[0] );
                break;
    
            case 2:
                rsum[0] += MAGMA_Z_ABS( A[0]   );
                rsum[1] += MAGMA_Z_ABS( A[lda] );
                break;
    
            case 3:
                rsum[0] += MAGMA_Z_ABS( A[0]     );
                rsum[1] += MAGMA_Z_ABS( A[lda]   );
                rsum[2] += MAGMA_Z_ABS( A[2*lda] );
                break;
        }
    
        /* compute final result */
        dwork[i] = rsum[0] + rsum[1] + rsum[2] + rsum[3];
    }
}


/* Computes max of row dwork[i] = max( abs( A(i,:) )), i=0:m-1, for || A ||_max,
 * where m and n are any size.
 * Has ceil( m/NB_X ) blocks of NB_X threads. Each thread does one row.
 * Based on zlange_inf_kernel code, above. */
extern "C" void
zlange_max_kernel(
    int m, int n,
    const magmaDoubleComplex * __restrict__ A, int lda,
    double * __restrict__ dwork , sycl::nd_item<3> item_ct1)
{
    int i = item_ct1.get_group(2) * NB_X + item_ct1.get_local_id(2);
    double rmax[4] = {0, 0, 0, 0};
    int n_mod_4 = n % 4;
    n -= n_mod_4;
    
    // if beyond last row, skip row
    if ( i < m ) {
        A += i;
        
        if ( n >= 4 ) {
            const magmaDoubleComplex *Aend = A + lda*n;
            magmaDoubleComplex rA[4] = { A[0], A[lda], A[2*lda], A[3*lda] };
            A += 4*lda;
            
            while( A < Aend ) {
                rmax[0] = max_nan( rmax[0], MAGMA_Z_ABS( rA[0] ));  rA[0] = A[0];
                rmax[1] = max_nan( rmax[1], MAGMA_Z_ABS( rA[1] ));  rA[1] = A[lda];
                rmax[2] = max_nan( rmax[2], MAGMA_Z_ABS( rA[2] ));  rA[2] = A[2*lda];
                rmax[3] = max_nan( rmax[3], MAGMA_Z_ABS( rA[3] ));  rA[3] = A[3*lda];
                A += 4*lda;
            }
            
            rmax[0] = max_nan( rmax[0], MAGMA_Z_ABS( rA[0] ));
            rmax[1] = max_nan( rmax[1], MAGMA_Z_ABS( rA[1] ));
            rmax[2] = max_nan( rmax[2], MAGMA_Z_ABS( rA[2] ));
            rmax[3] = max_nan( rmax[3], MAGMA_Z_ABS( rA[3] ));
        }
    
        /* clean up code */
        switch( n_mod_4 ) {
            case 0:
                break;
    
            case 1:
                rmax[0] = max_nan( rmax[0], MAGMA_Z_ABS( A[0] ));
                break;                          
                                                
            case 2:                             
                rmax[0] = max_nan( rmax[0], MAGMA_Z_ABS( A[  0] ));
                rmax[1] = max_nan( rmax[1], MAGMA_Z_ABS( A[lda] ));
                break;                          
                                                
            case 3:                             
                rmax[0] = max_nan( rmax[0], MAGMA_Z_ABS( A[    0] ));
                rmax[1] = max_nan( rmax[1], MAGMA_Z_ABS( A[  lda] ));
                rmax[2] = max_nan( rmax[2], MAGMA_Z_ABS( A[2*lda] ));
                break;
        }
    
        /* compute final result */
        dwork[i] = max_nan( max_nan( max_nan( rmax[0], rmax[1] ), rmax[2] ), rmax[3] );
    }
}


/* Computes col sums dwork[j] = sum( abs( A(:,j) )), j=0:n-1, for || A ||_one,
 * where m and n are any size.
 * Has n blocks of NB threads each. Block j sums one column, A(:,j) into dwork[j].
 * Thread i accumulates A(i,j) + A(i+NB,j) + A(i+2*NB,j) + ... into ssum[i],
 * then threads collectively do a sum-reduction of ssum,
 * and finally thread 0 saves to dwork[j]. */
extern "C" void
zlange_one_kernel(
    int m, int n,
    const magmaDoubleComplex * __restrict__ A, int lda,
    double * __restrict__ dwork , sycl::nd_item<3> item_ct1, double *ssum)
{

    int tx = item_ct1.get_local_id(2);

    A += item_ct1.get_group(2) * lda; // column j

    ssum[tx] = 0;
    for( int i = tx; i < m; i += NB_X ) {
        ssum[tx] += MAGMA_Z_ABS( A[i] );
    }
    magma_sum_reduce<NB_X>(tx, ssum, item_ct1);
    if ( tx == 0 ) {
        dwork[item_ct1.get_group(2)] = ssum[0];
    }
}

/* Based on zlange_one_kernel code, above.
 * Computes col sums dwork[j] = sum( abs( A(:,j) )^2 ), j=0:n-1, for || A ||_F,
 * where m and n are any size.
 * Has n blocks of NB threads each. Block j sums one column, A(:,j) into dwork[j].
 * Thread i accumulates A(i,j) + A(i+NB,j) + A(i+2*NB,j) + ... into ssum[i],
 * then threads collectively do a sum-reduction of ssum,
 * and finally thread 0 saves to dwork[j]. */
extern "C" void
zlange_fro_kernel(
    int m, int n,
    const magmaDoubleComplex * __restrict__ A, int lda,
    double * __restrict__ dwork , sycl::nd_item<3> item_ct1, double *ssum)
{

    int tx = item_ct1.get_local_id(2);

    A += item_ct1.get_group(2) * lda; // column j

    ssum[tx] = 0;
    for( int i = tx; i < m; i += NB_X ) {
#ifdef COMPLEX
        double a = MAGMA_Z_ABS( A[i] );
#else
        double a = A[i];
#endif
        ssum[tx] += a*a;
    }
    magma_sum_reduce<NB_X>(tx, ssum, item_ct1);
    if ( tx == 0 ) {
        dwork[item_ct1.get_group(2)] = ssum[0];
    }
}

/***************************************************************************//**
    Purpose
    -------
    ZLANGE  returns the value of the one norm, or the Frobenius norm, or
    the  infinity norm, or the  element of  largest absolute value  of a
    real matrix A.
    
    Description
    -----------
    ZLANGE returns the value
    
       ZLANGE = ( max(abs(A(i,j))), NORM = MagmaMaxNorm
                (
                ( norm1(A),         NORM = MagmaOneNorm
                (
                ( normI(A),         NORM = MagmaInfNorm
                (
                ( normF(A),         NORM = MagmaFrobeniusNorm
    
    where norm1 denotes the one norm of a matrix (maximum column sum),
    normI denotes the infinity norm of a matrix (maximum row sum) and
    normF denotes the Frobenius norm of a matrix (square root of sum of
    squares). Note that max(abs(A(i,j))) is not a consistent matrix norm.
    
    Arguments
    ---------
    @param[in]
    norm    magma_norm_t
            Specifies the value to be returned in ZLANGE as described
            above.
    
    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.  When M = 0,
            ZLANGE is set to zero.
    
    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.  When N = 0,
            ZLANGE is set to zero.
    
    @param[in]
    dA      DOUBLE PRECISION array on the GPU, dimension (LDDA,N)
            The m by n matrix A.
    
    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(M,1).
    
    @param
    dwork   (workspace) DOUBLE PRECISION array on the GPU, dimension (LWORK).
    
    @param[in]
    lwork   INTEGER
            The dimension of the array WORK.
            If NORM = MagmaInfNorm or MagmaMaxNorm, LWORK >= max( 1, M ).
            If NORM = MagmaOneNorm,                 LWORK >= max( 1, N ).
            Note this is different than LAPACK, which requires WORK only for
            NORM = MagmaInfNorm, and does not pass LWORK.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_lange
*******************************************************************************/
extern "C" double
magmablas_zlange(
    magma_norm_t norm, magma_int_t m, magma_int_t n,
    magmaDoubleComplex_const_ptr dA, magma_int_t ldda,
    magmaDouble_ptr dwork, magma_int_t lwork,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    if ( ! (norm == MagmaInfNorm || norm == MagmaMaxNorm ||
            norm == MagmaOneNorm || norm == MagmaFrobeniusNorm) )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < m )
        info = -5;
    else if ( ((norm == MagmaInfNorm || norm == MagmaMaxNorm) && (lwork < m)) ||
              ((norm == MagmaOneNorm || norm == MagmaFrobeniusNorm ) && (lwork < n)) )
        info = -7;

    if ( info != 0 ) {
        magma_xerbla( __func__, -(info) );
        return info;
    }
    
    /* Quick return */
    if ( m == 0 || n == 0 )
        return 0;
    
    //int i;
    sycl::range<3> threads(1, 1, NB_X);
    double result = -1;
    if ( norm == MagmaInfNorm ) {
        sycl::range<3> grid(1, 1, magma_ceildiv(m, NB_X));
        /*
        DPCT1049:1098: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zlange_inf_kernel(m, n, dA, ldda, dwork,
                                                 item_ct1);
                           });
        /*
        DPCT1049:1099: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_acc_ct1(sycl::range<1>(512), cgh);

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 512),
                                                   sycl::range<3>(1, 1, 512)),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_max_nan_kernel(
                                         m, dwork, item_ct1,
                                         (double *)smax_acc_ct1.get_pointer());
                                 });
            });
    }
    else if ( norm == MagmaMaxNorm ) {
        sycl::range<3> grid(1, 1, magma_ceildiv(m, NB_X));
        /*
        DPCT1049:1100: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->parallel_for(sycl::nd_range<3>(grid * threads, threads),
                           [=](sycl::nd_item<3> item_ct1) {
                               zlange_max_kernel(m, n, dA, ldda, dwork,
                                                 item_ct1);
                           });
        /*
        DPCT1049:1101: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_acc_ct1(sycl::range<1>(512), cgh);

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 512),
                                                   sycl::range<3>(1, 1, 512)),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_max_nan_kernel(
                                         m, dwork, item_ct1,
                                         (double *)smax_acc_ct1.get_pointer());
                                 });
            });
    }
    else if ( norm == MagmaOneNorm ) {
        sycl::range<3> grid(1, 1, n);
        /*
        DPCT1049:1102: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    ssum_acc_ct1(sycl::range<1>(NB_X), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zlange_one_kernel(
                                         m, n, dA, ldda, dwork, item_ct1,
                                         ssum_acc_ct1.get_pointer());
                                 });
            });
        /*
        DPCT1049:1103: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    smax_acc_ct1(sycl::range<1>(512), cgh);

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 512),
                                                   sycl::range<3>(1, 1, 512)),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_max_nan_kernel(
                                         n, dwork, item_ct1,
                                         (double *)smax_acc_ct1.get_pointer());
                                 });
            }); // note n instead of m
    }
    else if ( norm == MagmaFrobeniusNorm ) {
        sycl::range<3> grid(1, 1, n);
        /*
        DPCT1049:1104: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    ssum_acc_ct1(sycl::range<1>(NB_X), cgh);

                cgh.parallel_for(sycl::nd_range<3>(grid * threads, threads),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     zlange_fro_kernel(
                                         m, n, dA, ldda, dwork, item_ct1,
                                         ssum_acc_ct1.get_pointer());
                                 });
            });
        /*
        DPCT1049:1105: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        ((sycl::queue *)(queue->sycl_stream()))
            ->submit([&](sycl::handler &cgh) {
                sycl::accessor<double, 1, sycl::access_mode::read_write,
                               sycl::access::target::local>
                    sum_acc_ct1(sycl::range<1>(512), cgh);

                cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 512),
                                                   sycl::range<3>(1, 1, 512)),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     magma_sum_reduce_kernel(
                                         n, dwork, item_ct1,
                                         (double *)sum_acc_ct1.get_pointer());
                                 });
            }); // note n instead of m
    }
 
    magma_dgetvector( 1, &dwork[0], 1, &result, 1, queue );
    if( norm == MagmaFrobeniusNorm ) {
        result = sqrt(result); // Square root for final result.
    }
    
    return result;
}
