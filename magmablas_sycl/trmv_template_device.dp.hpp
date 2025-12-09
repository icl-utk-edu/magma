/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#ifndef TRMV_TEMPLATE_DEVICE_CUH
#define TRMV_TEMPLATE_DEVICE_CUH

#include <sycl/sycl.hpp>
///////////////////////////////////////////////////////////////////////////////////////////////////
// op<trans>( x ) returns x or conj(x).
template<typename T, const int CONJA>
static inline T OP( T& x )
{
    if(CONJA == 1) return conj(x);
    else return x;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
/*
trmv modes (trans - uplo) -- diag is handled in every case
NL: NoTrans - Lower
NU: NoTrans - Upper
TL: Trans   - Lower
TU: Trans   - Upper
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
// NL, NU
template<typename T, const int NB, int CONJA>
static 
void trmv_small_template_device(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        int n,    // n must be <= NB
        T* A, int ldda,
        T* X, int incx, const sycl::nd_item<3> &item_ct1, T *sA, T *sX)
{
    const int tx = item_ct1.get_local_id(2);
    const int slda = NB+1;

    // init sA and to zero
    for(int j = 0; j < NB; j++) {
        sA(tx,j) = make_FloatingPoint(0.0, 0.0);
    }

    // load A and X
    if(tx < n) {
        if(transA == MagmaNoTrans) {
            for(int j = 0; j < n; j++) {
                sA(tx,j) = A[j * ldda + tx];
            }
        }
        else {
            for(int j = 0; j < n; j++) {
                sA(j,tx) = OP<T,CONJA>( A[j * ldda + tx] );
            }
        }
        sX[ tx ] = X[tx * incx];
    }
    item_ct1.barrier();

    // handle diag -- no need to sync before that because every thread is updating the row it read
    if(diag == MagmaUnit){
        sA(tx,tx) = make_FloatingPoint(1.0, 0.0);
    }

    // handle uplo
    if(uplo == MagmaUpper){
        for(int j = 0; j < n; j++) {
            sA(tx,j) = (tx > j) ? make_FloatingPoint(0.0, 0.0) : sA(tx,j);
        }
    }
    else {
        for(int j = 0; j < n; j++) {
            sA(tx,j) = (tx < j) ? make_FloatingPoint(0.0, 0.0) : sA(tx,j);
        }
    }
    item_ct1.barrier();

    // multiply
    T rx = make_FloatingPoint(0.0, 0.0);
    for(int j = 0; j < NB; j++)
        rx += sA(tx,j) * sX[j];

    // write B
    if(tx < n) X[ tx * incx ] = rx;
}

#endif //TRMV_TEMPLATE_DEVICE_CUH
