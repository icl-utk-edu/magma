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

    @param[in,out]
    device_work  Workspace, allocated on device memory

    @param[in,out]
    lwork        INTEGER pointer
                 The size of the workspace (device_work) in bytes
                 - lwork[0] < 0: a workspace query is assumed, the routine
                   calculates the required amount of workspace and returns
                   it in lwork. The workspace is not referenced, and no
                   factorization is performed.
                -  lwork[0] >= 0: the routine assumes that the user has provided
                   a workspace with the size in lwork.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_getrf_batched
*******************************************************************************/
extern "C" magma_int_t
magma_zgbtrf_batched_work(
        magma_int_t m, magma_int_t n,
        magma_int_t kl, magma_int_t ku,
        magmaDoubleComplex **dAB_array, magma_int_t lddab,
        magma_int_t **dipiv_array, magma_int_t *info_array,
        void* device_work, magma_int_t *lwork,
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0, nb = 8, nthreads = kl+1;
    magma_int_t minmn = min(m, n);
    magma_int_t kv    = kl + ku;

    if( m < 0 )
        arginfo = -1;
    else if ( n < 0 )
        arginfo = -2;
    else if ( kl < 0 )
        arginfo = -3;
    else if ( ku < 0 )
        arginfo = -4;
    else if ( lddab < (kl+kv+1) )
        arginfo = -6;
    else if ( batchCount < 0)
        arginfo = -11;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( m == 0 || n == 0 || batchCount == 0) return 0;

    // get tuning parameters for fused and sliding window routines
    magma_int_t ntcol = 1;
    magma_get_zgbtrf_batched_params(m, n, kl, ku, &nb, &nthreads);

    // calculate workspace for generic implementation
    magma_int_t lwork_bytes = 0;
    lwork_bytes += batchCount * sizeof(int); // no need for magma_int_t here

    if( *lwork < 0) {
        *lwork = lwork_bytes;
        arginfo = 0;
        return arginfo;
    }

    if( *lwork < lwork_bytes ) {
        arginfo = -13;
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // first try the fully fused factorization
    if(minmn <= 64) {
        magma_int_t info_fused = 0;
        info_fused = magma_zgbtrf_batched_fused_sm(
                        m,  n, kl, ku,
                        dAB_array, lddab, dipiv_array,
                        info_array, nthreads, ntcol,
                        batchCount, queue );
        if(info_fused == 0) return arginfo;
    }

    // try the sliding window implementation
    magma_int_t info_sliding_window = 0;
    if(nb >= n) {
        info_sliding_window = magma_zgbtrf_batched_sliding_window_loopout(
                            m, n, kl, ku,
                            dAB_array, lddab, dipiv_array, info_array,
                            device_work, lwork, batchCount, queue );
    }
    else{
        info_sliding_window = magma_zgbtrf_batched_sliding_window_loopin(
                            m, n, kl, ku,
                            dAB_array, lddab, dipiv_array, info_array,
                            batchCount, queue );
    }
    if(info_sliding_window == 0) return arginfo;

    // generic implementation (currently unblocked)
    // TODO: implement blocked version to use level-3 BLAS
    // ju_array holds (per problem) the index of the last column affected
    // by the previous factorization stage
    int* ju_array = (int*)device_work;
    // set info to zero
    magma_memset_async(info_array, 0, batchCount*sizeof(magma_int_t), queue);
    for(magma_int_t j = 0; j < minmn; j++) {
        // izamax
        magma_int_t km = 1 + min( kl, m-j-1 ); // diagonal and subdiagonal(s)
        magma_izamax_batched(
            km, dAB_array, kv, j, lddab, 1,
            dipiv_array, j,
            0, 0, info_array, batchCount, queue);

        // adjust ju_array
        magma_zgbtrf_set_fillin(n, kl, ku, dAB_array, lddab, dipiv_array, ju_array, j, batchCount, queue);
        magma_gbtrf_adjust_ju(n, ku, dipiv_array, ju_array, j, batchCount, queue);

        // swap (right only)
        magma_zgbtf2_zswap_batched(
            kl, ku, dAB_array, kv, j, lddab,
            dipiv_array, j, ju_array, j, batchCount, queue);

        // adjust pivot
        adjust_ipiv_batched(dipiv_array, j, 1, j, batchCount, queue);

        // scal and ger
        magma_zgbtf2_scal_ger_batched(
            m, n, kl, ku,
            dAB_array, kv, j, lddab,
            ju_array, j, info_array,
            batchCount, queue);
    }

    return arginfo;
}


////////////////////////////////////////////////////////////////////////////////
/// @see magma_zgbtrf_batched_work.
/// This is the (pointer + stride) interface of magma_zgbtrf_batched_work
extern "C" magma_int_t
magma_zgbtrf_batched_strided_work(
        magma_int_t m, magma_int_t n,
        magma_int_t kl, magma_int_t ku,
        magmaDoubleComplex* dAB, magma_int_t lddab, magma_int_t strideAB,
        magma_int_t* dipiv, magma_int_t stride_piv,
        magma_int_t *info_array,
        void* device_work, magma_int_t *lwork,
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    magma_int_t kv = kl + ku;

    if( m < 0 )
        arginfo = -1;
    else if ( n < 0 )
        arginfo = -2;
    else if ( kl < 0 )
        arginfo = -3;
    else if ( ku < 0 )
        arginfo = -4;
    else if ( lddab < (kl+kv+1) )
        arginfo = -6;
    else if ( strideAB < (lddab * n))
        arginfo = -7;
    else if ( stride_piv < min(m, n))
        arginfo = -9;
    else if ( batchCount < 0 )
        arginfo = -13;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    // quick return if possible
    if( m == 0 || n == 0 || batchCount == 0) return 0;

    magmaDoubleComplex** dAB_array   = (magmaDoubleComplex**)queue->get_dAarray();
    magma_int_t**      dipiv_array   = (magma_int_t**)queue->get_dBarray();

    // query workspace
    magma_int_t my_work[1] = {-1};
    magma_zgbtrf_batched_work(m, n, kl, ku, NULL, lddab, NULL, NULL, NULL, my_work, batchCount, queue);

    if( *lwork < 0 ) {
        *lwork = my_work[0];
        return arginfo;
    }

    if( *lwork < my_work[0] ) {
        arginfo = -12;
        return arginfo;
    }

    magma_int_t max_batchCount   = queue->get_maxBatch();
    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount){
        magma_int_t batch = min(max_batchCount, batchCount-i);
        magma_zset_pointer(dAB_array,   (magmaDoubleComplex*)(dAB + i * strideAB), lddab, 0, 0, strideAB,   batch, queue);
        magma_iset_pointer(dipiv_array, (magma_int_t*)(dipiv + i * stride_piv),        1, 0, 0, stride_piv, batch, queue);

        magma_zgbtrf_batched_work(
            m, n, kl, ku,
            dAB_array, lddab,
            dipiv_array, info_array + i,
            device_work, lwork,
            batch, queue );
    }
    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    ZGBTRF computes an LU factorization of a COMPLEX m-by-n band matrix A
    using partial pivoting with row interchanges.

    This is the batched version of the algorithm, which performs the factorization
    on a batch of matrices with the same size and lower/upper bandwidths.

    Arguments
    ---------
    @param[in]
    M     INTEGER
          The number of rows of the matrix A.  M >= 0.

    @param[in]
    N     INTEGER
          The number of columns of the matrix A.  N >= 0.

    @param[in]
    KL    INTEGER
          The number of subdiagonals within the band of A.  KL >= 0.

    @param[in]
    KU    INTEGER
          The number of superdiagonals within the band of A.  KU >= 0.

    @param[in,out]
    dAB_array    Array of pointers, dimension (batchCount).
          Each is a COMPLEX_16 array, dimension (LDDAB,N)
          On entry, the matrix AB in band storage, in rows KL+1 to
          2*KL+KU+1; rows 1 to KL of the array need not be set.
          The j-th column of A is stored in the j-th column of the
          array AB as follows:
          AB(kl+ku+1+i-j,j) = A(i,j) for max(1,j-ku)<=i<=min(m,j+kl)

          On exit, details of the factorization: U is stored as an
          upper triangular band matrix with KL+KU superdiagonals in
          rows 1 to KL+KU+1, and the multipliers used during the
          factorization are stored in rows KL+KU+2 to 2*KL+KU+1.
          See below for further details.

    @param[in]
    LDDAB INTEGER
          The leading dimension of the array AB.  LDAB >= 2*KL+KU+1.

    @param[out]
    dIPIV_array    Array of pointers, dimension (batchCount).
          Each is an INTEGER array, dimension (min(M,N))
          The pivot indices; for 1 <= i <= min(M,N), row i of the
          matrix was interchanged with row IPIV(i).

    @param[out]
    dINFO_array    INTEGER array, dimension (batchCount)
          Each is the INFO output for a given matrix
          = 0: successful exit
          < 0: if INFO = -i, the i-th argument had an illegal value
          > 0: if INFO = +i, U(i,i) is exactly zero. The factorization
               has been completed, but the factor U is exactly
               singular, and division by zero will occur if it is used
               to solve a system of equations.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

  Further Details
  ===============

  The band storage scheme is illustrated by the following example, when
  M = N = 6, KL = 2, KU = 1:

  On entry:                       On exit:

      *    *    *    +    +    +       *    *    *   u14  u25  u36
      *    *    +    +    +    +       *    *   u13  u24  u35  u46
      *   a12  a23  a34  a45  a56      *   u12  u23  u34  u45  u56
     a11  a22  a33  a44  a55  a66     u11  u22  u33  u44  u55  u66
     a21  a32  a43  a54  a65   *      m21  m32  m43  m54  m65   *
     a31  a42  a53  a64   *    *      m31  m42  m53  m64   *    *

  Array elements marked * are not used by the routine; elements marked
  + need not be set on entry, but are required by the routine to store
  elements of U because of fill-in resulting from the row interchanges.


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
    magma_int_t arginfo = 0;
    magma_int_t kv    = kl + ku;

    if( m < 0 )
        arginfo = -1;
    else if ( n < 0 )
        arginfo = -2;
    else if ( kl < 0 )
        arginfo = -3;
    else if ( ku < 0 )
        arginfo = -4;
    else if ( lddab < (kl+kv+1) )
        arginfo = -6;
    else if ( batchCount < 0 )
        arginfo = -9;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( m == 0 || n == 0 || batchCount == 0) return 0;

    magma_int_t lwork[1] = {-1};

    // query workspace
    magma_zgbtrf_batched_work(
        m, n, kl, ku,
        NULL, lddab,
        NULL, NULL,
        NULL, lwork,
        batchCount, queue);

    void* device_work = NULL;
    magma_malloc((void**)&device_work, lwork[0]);

    // call generic implementation
    magma_zgbtrf_batched_work(
        m, n, kl, ku,
        dAB_array, lddab,
        dipiv_array, info_array,
        device_work, lwork, batchCount, queue);

    magma_free(device_work);
    return arginfo;
}


/// @see magma_zgbtrf_batched. This is the (pointer + stride) interface of magma_zgbtrf_batched
extern "C" magma_int_t
magma_zgbtrf_batched_strided(
        magma_int_t m, magma_int_t n,
        magma_int_t kl, magma_int_t ku,
        magmaDoubleComplex* dAB, magma_int_t lddab, magma_int_t strideAB,
        magma_int_t* dipiv, magma_int_t stride_piv,
        magma_int_t *info_array,
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = 0;
    magma_int_t kv    = kl + ku;

    if( m < 0 )
        arginfo = -1;
    else if ( n < 0 )
        arginfo = -2;
    else if ( kl < 0 )
        arginfo = -3;
    else if ( ku < 0 )
        arginfo = -4;
    else if ( lddab < (kl+kv+1) )
        arginfo = -6;
    else if ( strideAB < (lddab * n))
        arginfo = -7;
    else if ( stride_piv < min(m, n))
        arginfo = -9;
    else if ( batchCount < 0 )
        arginfo = -11;

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    if( m == 0 || n == 0 || batchCount == 0) return 0;

    magma_int_t lwork[1] = {-1};

    // query workspace
    magma_zgbtrf_batched_strided_work(
        m, n, kl, ku,
        NULL, lddab, strideAB,
        NULL, stride_piv,
        NULL, NULL, lwork,
        batchCount, queue);

    void* device_work = NULL;
    magma_malloc((void**)&device_work, lwork[0]);

    // call generic implementation
    magma_zgbtrf_batched_strided_work(
        m, n, kl, ku,
        dAB, lddab, strideAB,
        dipiv, stride_piv,
        info_array,
        device_work, lwork,
        batchCount, queue);

    magma_free(device_work);
    return arginfo;
}
