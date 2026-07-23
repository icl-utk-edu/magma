/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
       @author Natalie Beams
*/

#ifndef TRSV_TEMPLATE_DEVICE_CUH
#define TRSV_TEMPLATE_DEVICE_CUH

///////////////////////////////////////////////////////////////////////////////////////////////////
/* common functions */
///////////////////////////////////////////////////////////////////////////////////////////////////
#define LPACKED(i_, j_, N_) (N_*(j_) - (j_)*(j_+1)/2 + i_)
#define UPACKED(i_, j_) ((j_)*(j_+1)/2 + i_)

template<typename T, const int NB, const int CONJA>
__device__ __inline__
void read_sA(int tx, T* A, T* sA, int n, int ldda, int slda)
{
    if(n == NB){
        #pragma unroll
        for(int i = 0; i < NB; i++){
            sA[i * slda + tx] = (CONJA == 0) ? A[i * ldda + tx] : conj(A[i * ldda + tx]);
        }
    }
    else{
       for(int i = 0; i < n; i++){
           sA[i * slda + tx] = (CONJA == 0) ? A[i * ldda + tx] : conj(A[i * ldda + tx]);
        }
    }
}

template<typename T, const int NB, const int CONJA, const int LOWER>
__device__ __inline__
void read_packed_sA(int tx, T* A, int coffA, int roffA, T* sA, int n, int ldda, int slda)
{
  int index = 0;
  if (LOWER > 0) {
    // Only read in lower portion; global row >= global column
    if (n == NB) {
        #pragma unroll
        for(int i = 0; i < NB; i++){
           if ((roffA + tx >= coffA + i) && (roffA + tx < ldda)) {
              index = LPACKED(roffA + tx, coffA + i, ldda);
              sA[i * slda + tx] = (CONJA == 0) ? A[index] : conj(A[index]);
	   }
        }
    }
    else {
        for(int i = 0; i < n; i++){
           if ((roffA + tx >= coffA + i) && (roffA + tx < ldda)) {
              index = LPACKED(roffA + tx, coffA + i, ldda);
              sA[i * slda + tx] = (CONJA == 0) ? A[index] : conj(A[index]);
	   }
        }
    }
  }
  else {
    // Only read in upper portion; global row <= global column
    if (n == NB) {
        #pragma unroll
        for(int i = 0; i < NB; i++){
           if ((roffA + tx <= coffA + i) && (coffA + i < ldda)) {
              index = UPACKED(roffA + tx, coffA + i);
              sA[i * slda + tx] = (CONJA == 0) ? A[index] : conj(A[index]);
	   }
        }
    }
    else {
        for(int i = 0; i < n; i++){
           if ((roffA + tx <= coffA + i) && (coffA + i < ldda)) {
              index = UPACKED(roffA + tx, coffA + i);
              sA[i * slda + tx] = (CONJA == 0) ? A[index] : conj(A[index]);
	   }
        }
    }
  }
}

template<typename T, const int NB, const int CONJA, const int LOWER, const int PACKEDA>
__device__ __inline__
void trsv_init_data( int tx, int n,
                     magma_diag_t diag,
                     T* A, int coffA, int roffA, int ldda,
                     T* x, int incx,
                     T* sA, int slda,
                     T* sx)
{
    const T c_zero = make_FloatingPoint(0.0, 0.0);
    const T c_one  = make_FloatingPoint(1.0, 0.0);

    // init sA and sx
    if(tx < NB){
        #pragma unroll
        for(int i = 0; i < NB; i++){
            sA[i * slda + tx] = c_zero;
        }
        sA[tx * slda + tx] = c_one;
        sx[tx] = c_zero;
    }

    if( tx < n ){
        // load A
	if (PACKEDA > 0) {
          read_packed_sA<T, NB, CONJA, LOWER>(tx, A, coffA, roffA, sA, n, ldda, slda);
	}
	else {
          read_sA<T, NB, CONJA>(tx, A + coffA * ldda + roffA, sA, n, ldda, slda);
	}

        // handle diag
        if(diag == MagmaNonUnit){
            sA[tx * slda + tx] = div(c_one, sA[tx * slda + tx]);
        }else{
            sA[tx * slda + tx] = c_one;
        }

        // load x
        sx[ tx ] = x[ tx * incx ];
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, const int NB>
__device__ __inline__
void trsv_write_x( int tx, int n,
                   T*  x, int incx,
                   T* sx )
{
    if(tx < n){
        x[ tx * incx ] = sx[ tx ];
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
/* trsv device functions */
///////////////////////////////////////////////////////////////////////////////////////////////////
/*
trsv modes
NL: NoTrans - Lower
NU: NoTrans - Upper
TL: Trans   - Lower
TU: Trans   - Upper
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
// NL
template<typename T, const int NB, const int PACKEDA>
static __device__
void trsv_template_device_NL(
        magma_diag_t diag, int n,
        T* A, int coffA, int roffA, int ldda,
        T* x, int incx)
{
#define sA(i, j) sA[(j)*slda + (i)]

    constexpr int slda = NB;
    const     int   tx = threadIdx.x;

    __shared__ T sA[slda * NB];
    __shared__ T sx[NB];

    trsv_init_data<T, NB, 0, 1, PACKEDA>(tx, n, diag, A, coffA, roffA, ldda, x, incx, sA, slda, sx);
    __syncthreads();

    // solve
    #pragma unroll
    for(int i = 0; i < NB; i++) {
        if(tx == 0) {
            sx[i] *= sA(i, i);
        }
        __syncthreads();

        if(tx > i) {
            sx[tx] -= sx[i] * sA(tx, i);
        }
        __syncthreads();
    }

    // write x
    trsv_write_x<T, NB>( tx, n, x, incx, sx );

#undef sA
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// NU
template<typename T, const int NB, const int PACKEDA>
static __device__
void trsv_template_device_NU(
        magma_diag_t diag, int n,
        T* A, int coffA, int roffA, int ldda,
        T* x, int incx)
{
#define sA(i, j) sA[(j)*slda + (i)]

    constexpr int slda = NB;
    const     int   tx = threadIdx.x;

    __shared__ T sA[slda * NB];
    __shared__ T sx[NB];

    trsv_init_data<T, NB, 0, 0, PACKEDA>(tx, n, diag, A, coffA, roffA, ldda, x, incx, sA, slda, sx);
    __syncthreads();

    // solve
    #pragma unroll
    for(int i = NB-1; i >= 0; i--) {
        if(tx == 0) {
            sx[i] *= sA(i, i);
        }
        __syncthreads();

        if(tx < i) {
            sx[tx] -= sx[i] * sA(tx, i);
        }
        __syncthreads();
    }

    // write x
    trsv_write_x<T, NB>( tx, n, x, incx, sx );

#undef sA
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// TL, CL
template<typename T, const int NB, const int CONJA, const int PACKEDA>
static __device__
void trsv_template_device_TL(
        magma_diag_t diag, int n,
        T* A, int coffA, int roffA, int ldda,
        T* x, int incx)
{
#define sA(i, j) sA[(j)*slda + (i)]

    constexpr int slda = NB;
    const     int   tx = threadIdx.x;

    __shared__ T sA[slda * NB];
    __shared__ T sx[NB];

    trsv_init_data<T, NB, CONJA, 1, PACKEDA>(tx, n, diag, A, coffA, roffA, ldda, x, incx, sA, slda, sx);
    __syncthreads();

    // solve
    #pragma unroll
    for(int i = NB-1; i >= 0; i--) {
        if(tx == 0) {
            sx[i] *= sA(i, i);
        }
        __syncthreads();

        if(tx < i) {
            sx[tx] -= sx[i] * sA(i, tx);
        }
        __syncthreads();
    }

    // write x
    trsv_write_x<T, NB>( tx, n, x, incx, sx );

#undef sA
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// TU, CU
template<typename T, const int NB, const int CONJA, const int PACKEDA>
static __device__
void trsv_template_device_TU(
        magma_diag_t diag, int n,
        T* A, int coffA, int roffA, int ldda,
        T* x, int incx)
{
#define sA(i, j) sA[(j)*slda + (i)]

    constexpr int slda = NB;
    const     int   tx = threadIdx.x;

    __shared__ T sA[slda * NB];
    __shared__ T sx[NB];

    trsv_init_data<T, NB, CONJA, 0, PACKEDA>(tx, n, diag, A, coffA, roffA, ldda, x, incx, sA, slda, sx);
    __syncthreads();

    // solve
    #pragma unroll
    for(int i = 0; i < NB; i++) {
        if(tx == 0) {
            sx[i] *= sA(i, i);
        }
        __syncthreads();

        if(tx > i) {
            sx[tx] -= sx[i] * sA(i, tx);
        }
        __syncthreads();
    }

    // write x
    trsv_write_x<T, NB>( tx, n, x, incx, sx );

#undef sA
}

#endif //TRSV_TEMPLATE_DEVICE_CUH
