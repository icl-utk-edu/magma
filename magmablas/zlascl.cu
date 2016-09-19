/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Mark Gates
*/
#include "magma_internal.h"

#define NB 64


// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right.
__global__ void
zlascl_full(
    int m, int n, double mul,
    magmaDoubleComplex* A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    A += ind;
    if (ind < m) {
        for (int j=0; j < n; j++ )
            A[j*lda] *= mul;
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from left edge and moving right to diagonal.
__global__ void
zlascl_lower(
    int m, int n, double mul,
    magmaDoubleComplex* A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    int break_d = (ind < n) ? ind : n-1;

    A += ind;
    if (ind < m) {
        for (int j=0; j <= break_d; j++ )
            A[j*lda] *= mul;
    }
}


// each thread block does one NB x n block row of A.
// each thread does one row, starting from right edge and moving left to diagonal.
__global__ void
zlascl_upper(
    int m, int n, double mul,
    magmaDoubleComplex* A, int lda)
{
    int ind = blockIdx.x * NB + threadIdx.x;

    A += ind;
    if (ind < m) {
        for (int j=n-1; j >= ind; j--)
            A[j*lda] *= mul;
    }
}


/***************************************************************************//**
    Purpose
    -------
    ZLASCL multiplies the M by N complex matrix A by the real scalar
    CTO/CFROM.  This is done without over/underflow as long as the final
    result CTO*A(I,J)/CFROM does not over/underflow. TYPE specifies that
    A may be full, upper triangular, lower triangular.

    Arguments
    ---------
    @param[in]
    type    magma_type_t
            TYPE indices the storage type of the input matrix A.
            = MagmaFull:   full matrix.
            = MagmaLower:  lower triangular matrix.
            = MagmaUpper:  upper triangular matrix.
            Other formats that LAPACK supports, MAGMA does not currently support.

    @param[in]
    kl      INTEGER
            Unused, for LAPACK compatability.

    @param[in]
    ku      KU is INTEGER
            Unused, for LAPACK compatability.

    @param[in]
    cfrom   DOUBLE PRECISION

    @param[in]
    cto     DOUBLE PRECISION
    \n
            The matrix A is multiplied by CTO/CFROM. A(I,J) is computed
            without over/underflow if the final result CTO*A(I,J)/CFROM
            can be represented without over/underflow.
            CFROM must be nonzero. CFROM and CTO must not be NAN.

    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in,out]
    dA      COMPLEX*16 array, dimension (LDDA,N)
            The matrix to be multiplied by CTO/CFROM.  See TYPE for the
            storage type.

    @param[in]
    ldda    INTEGER
            The leading dimension of the array A.  LDDA >= max(1,M).

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value.
    
    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_lascl
*******************************************************************************/
extern "C" void
magmablas_zlascl(
    magma_type_t type, magma_int_t kl, magma_int_t ku,
    double cfrom, double cto,
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex_ptr dA, magma_int_t ldda,
    magma_queue_t queue,
    magma_int_t *info )
{
    *info = 0;
    if ( type != MagmaLower && type != MagmaUpper && type != MagmaFull )
        *info = -1;
    else if ( cfrom == 0 || isnan(cfrom) )
        *info = -4;
    else if ( isnan(cto) )
        *info = -5;
    else if ( m < 0 )
        *info = -6;
    else if ( n < 0 )
        *info = -3;
    else if ( ldda < max(1,m) )
        *info = -7;
    
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return;  //info;
    }
    
    dim3 threads( NB );
    dim3 grid( magma_ceildiv( m, NB ) );
    
    double smlnum, bignum, cfromc, ctoc, cto1, cfrom1, mul;
    magma_int_t done = false;
    
    // Uses over/underflow procedure from LAPACK zlascl
    // Get machine parameters
    smlnum = lapackf77_dlamch("s");
    bignum = 1 / smlnum;
    
    cfromc = cfrom;
    ctoc   = cto;
    int cnt = 0;
    while( ! done ) {
        cfrom1 = cfromc*smlnum;
        if ( cfrom1 == cfromc ) {
            // cfromc is an inf.  Multiply by a correctly signed zero for
            // finite ctoc, or a nan if ctoc is infinite.
            mul  = ctoc / cfromc;
            done = true;
            cto1 = ctoc;
        }
        else {
            cto1 = ctoc / bignum;
            if ( cto1 == ctoc ) {
                // ctoc is either 0 or an inf.  In both cases, ctoc itself
                // serves as the correct multiplication factor.
                mul  = ctoc;
                done = true;
                cfromc = 1;
            }
            else if ( fabs(cfrom1) > fabs(ctoc) && ctoc != 0 ) {
                mul  = smlnum;
                done = false;
                cfromc = cfrom1;
            }
            else if ( fabs(cto1) > fabs(cfromc) ) {
                mul  = bignum;
                done = false;
                ctoc = cto1;
            }
            else {
                mul  = ctoc / cfromc;
                done = true;
            }
        }
        
        if (type == MagmaLower) {
            zlascl_lower <<< grid, threads, 0, queue->cuda_stream() >>> (m, n, mul, dA, ldda);
        }
        else if (type == MagmaUpper) {
            zlascl_upper <<< grid, threads, 0, queue->cuda_stream() >>> (m, n, mul, dA, ldda);
        }
        else if (type == MagmaFull) {
            zlascl_full  <<< grid, threads, 0, queue->cuda_stream() >>> (m, n, mul, dA, ldda);
        }
     
        cnt += 1;
    }
}
