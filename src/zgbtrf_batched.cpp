/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Ahmad Abdelfattah

   @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"

/***************************************************************************//**
    Purpose
    -------
    ZGBTRF computes an LU factorization of a complex m-by-n band matrix AB
    using partial pivoting with row interchanges.

    This is a batched version that factors `batchCount` M-by-N matrices in parallel.
    dAB, dipiv, and info become arrays with one entry per matrix.

   The band storage scheme is illustrated by the following example, when
   M = N = 6, KL = 2, KU = 1:

   On entry:                       On exit:

      *    *    *    +    +    +       *    *    *   u14  u25  u36
      *    *    +    +    +    +       *    *   u13  u24  u35  u46
      *   a12  a23  a34  a45  a56      *   u12  u23  u34  u45  u56
     a11  a22  a33  a44  a55  a66     u11  u22  u33  u44  u55  u66
     a21  a32  a43  a54  a65   *      m21  m32  m43  m54  m65   *
     a31  a42  a53  a64   *    *      m31  m42  m53  m64   *    *

     Note that this behavior is a little different from the standard LAPACK
     routine. Array elements marked * are not read by the routine, but may
     be zeroed out after completion. Elements marked + need not be set on entry,
     but are required by the routine to store elements of U because of fill-in
     resulting from the row interchanges.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of each matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix A.  N >= 0.

    @param[in]
    kl      INTEGER
            The number of subdiagonals within the band of A.  KL >= 0.

    @param[in]
    ku      INTEGER
            The number of superdiagonals within the band of A.  KL >= 0.

    @param[in,out]
    dAB_array    Array of pointers, dimension (batchCount).
            Each is a COMPLEX_16 array on the GPU, dimension (LDAB,N)
            On entry, the matrix A in band storage, in rows KL+1 to
            2*KL+KU+1; rows 1 to KL of the array need not be set.
            The j-th column of A is stored in the j-th column of the
            array AB as follows:
            AB(kl+ku+1+i-j,j) = A(i,j) for max(1,j-ku)<=i<=min(m,j+kl)

            On exit, details of the factorization: U is stored as an
            upper triangular band matrix with KL+KU superdiagonals in
            rows 1 to KL+KU+1, and the multipliers used during the
            factorization are stored in rows KL+KU+2 to 2*KL+KU+1.
            See above for details about the band storage.

    @param[in]
    lddab   INTEGER
            The leading dimension of each array AB.  LDDAB >= (2*KL+KU+1).

    @param[out]
    dipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgbtrf_batched(
        magma_int_t m, magma_int_t n,
        magma_int_t kl, magma_int_t ku,
        magmaDoubleComplex **dAB_array, magma_int_t lddab,
        magma_int_t **dipiv_array, magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)
{
    return 0;
}
