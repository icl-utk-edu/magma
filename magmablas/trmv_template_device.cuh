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

///////////////////////////////////////////////////////////////////////////////////////////////////
// op<trans>( x ) returns x or conj(x).
template<typename T, const int CONJA>
__device__ static inline T OP( T& x )
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
static __device__
void trmv_small_template_device(
        magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        int n,    // n must be <= NB
        T* A, int ldda,
        T* X, int incx)
{
#define sA(i, j, slda) sA[j * slda + i]
    const int tx   = threadIdx.x;
    const int slda = NB+1;

    __shared__ T sA[slda*NB];
    __shared__ T sX[NB];

    // init sA and to zero
    for(int j = 0; j < NB; j++) {
        sA(tx,j,slda) = make_FloatingPoint(0.0, 0.0);
    }

    // load A and X
    if(tx < n) {
        if(transA == MagmaNoTrans) {
            for(int j = 0; j < n; j++) {
                sA(tx,j,slda) = A[j * ldda + tx];
            }
        }
        else {
            for(int j = 0; j < n; j++) {
                sA(j,tx,slda) = OP<T,CONJA>( A[j * ldda + tx] );
            }
        }
        sX[ tx ] = X[tx * incx];
    }
    __syncthreads();

    // handle diag -- no need to sync before that because every thread is updating the row it read
    if(diag == MagmaUnit){
        sA(tx,tx,slda) = make_FloatingPoint(1.0, 0.0);
    }

    // handle uplo
    if(uplo == MagmaUpper){
        for(int j = 0; j < n; j++) {
            sA(tx,j,slda) = (tx > j) ? make_FloatingPoint(0.0, 0.0) : sA(tx,j,slda);
        }
    }
    else {
        for(int j = 0; j < n; j++) {
            sA(tx,j,slda) = (tx < j) ? make_FloatingPoint(0.0, 0.0) : sA(tx,j,slda);
        }
    }
    __syncthreads();

    // multiply
    T rx = make_FloatingPoint(0.0, 0.0);
    for(int j = 0; j < NB; j++)
        rx += sA(tx,j,slda) * sX[j];

    // write B
    if(tx < n) X[ tx * incx ] = rx;
#undef sA
}

#endif //TRMV_TEMPLATE_DEVICE_CUH
