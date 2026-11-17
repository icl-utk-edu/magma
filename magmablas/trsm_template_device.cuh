/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
       @author Natalie Beams
*/

#ifndef TRSM_TEMPLATE_DEVICE_CUH
#define TRSM_TEMPLATE_DEVICE_CUH

///////////////////////////////////////////////////////////////////////////////////////////////////
/* common functions */
///////////////////////////////////////////////////////////////////////////////////////////////////
#define LPACKED(i_, j_, N_) (N_*(j_) - (j_)*(j_+1)/2 + i_)
#define UPACKED(i_, j_) ((j_)*(j_+1)/2 + i_)

template<typename T, const int NB, const int CONJA>
__device__ __inline__
void left_read_sA(int tx, T* A, T* sA, int m, int ldda)
{
    if (m == NB) {
      #pragma unroll
      for(int i = 0; i < NB; i++){
          sA[i * NB + tx] = (CONJA == 0) ? A[i * ldda + tx] : conj(A[i * ldda + tx]);
      }
    }
    else {
      for(int i = 0; i < m; i++){
          sA[i * NB + tx] = (CONJA == 0) ? A[i * ldda + tx] : conj(A[i * ldda + tx]);
      }
    }
}

template<typename T, const int NB, const int CONJA, const int LOWER>
__device__ __inline__
void left_read_packed_sA(int tx, T* A, int coffA, int roffA, T* sA, int m, int ldda)
{
    int index = 0;
    if (LOWER > 0) {
      // Only read in lower portion; global row >= global column
      if (m == NB) {
        #pragma unroll
        for(int i = 0; i < NB; i++){
          if ((roffA + tx >= coffA + i) && (roffA + tx < ldda)) {
            index = LPACKED(roffA + tx, coffA + i, ldda);
            sA[i * NB + tx] = (CONJA == 0) ? A[index] : conj(A[index]);
          }
        }
      }
      else {
        for(int i = 0; i < m; i++){
          if ((roffA + tx >= coffA + i) && (roffA + tx < ldda)) {
            index = LPACKED(roffA + tx, coffA + i, ldda);
            sA[i * NB + tx] = (CONJA == 0) ? A[index] : conj(A[index]);
          }
        }
      }
    }
    else {
      // Only read in upper portion; global row <= global column
      if (m == NB) {
        #pragma unroll
        for(int i = 0; i < NB; i++){
          if ((roffA + tx <= coffA + i) && (coffA + i < ldda)) {
            index = UPACKED(roffA + tx, coffA + i);
            sA[i * NB + tx] = (CONJA == 0) ? A[index] : conj(A[index]);
          }
        }
      }
      else {
        for(int i = 0; i < m; i++){
          if ((roffA + tx <= coffA + i) && (coffA + i < ldda)) {
            index = UPACKED(roffA + tx, coffA + i);
            sA[i * NB + tx] = (CONJA == 0) ? A[index] : conj(A[index]);
          }
        }
      }
    }
}

template<typename T, const int NB, const int CONJA>
__device__ __inline__
void right_read_sA(int tx, T* A, T* sA, int n, int ldda, int slda)
{
    if (n == NB) {
      #pragma unroll
      for(int i = 0; i < NB; i++){
          sA[i * slda + tx] = (CONJA == 0) ? A[i * ldda + tx] : conj(A[i * ldda + tx]);
      }
    }
    else {
      for(int i = 0; i < n; i++){
          sA[i * slda + tx] = (CONJA == 0) ? A[i * ldda + tx] : conj(A[i * ldda + tx]);
      }
    }
}

template<typename T, const int NB, const int CONJA, const int LOWER>
__device__ __inline__
void right_read_packed_sA(int tx, T* A, int coffA, int roffA, T* sA, int n, int ldda, int slda)
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

template<typename T, const int NB, const int NRHS, const int CONJA, const int LOWER, const int PACKEDA>
__device__ __inline__
void trsm_left_init_data( int tx, int m, int nn, T alpha,
                          magma_diag_t diag,
                          T* A, int coffA, int roffA, int ldda,
                          T* B, int lddb,
                          T* sA, T* sB)
{
    // init sA and sB
    if(tx < NB){
        #pragma unroll
        for(int i = 0; i < NB; i++){
            sA[i * NB + tx] = make_FloatingPoint(0.0, 0.0);
        }
        #pragma unroll
        for(int i = 0; i < NRHS; i++){
            sB[i * (NB+1) + tx] = make_FloatingPoint(0.0, 0.0);
        }
        sA[tx * NB + tx] = make_FloatingPoint(1.0, 0.0);
    }

    if( tx < m ){
        // load A
        if (PACKEDA > 0) {
          left_read_packed_sA<T, NB, CONJA, LOWER>(tx, A, coffA, roffA, sA, m, ldda);
        }
        else {
          left_read_sA<T, NB, CONJA>(tx, A + coffA * ldda + roffA, sA, m, ldda);
        }

        // handle diag
        if(diag == MagmaNonUnit){
            sA[tx * NB + tx] = div(make_FloatingPoint(1.0, 0.0), sA[tx * NB + tx]);
        }else{
            sA[tx * NB + tx] = make_FloatingPoint(1.0, 0.0);
        }

        // load B
        if(nn == NRHS){
            #pragma unroll
            for(int i = 0; i < NRHS; i++){
                sB[ i * (NB+1) + tx ] = alpha * B[i * lddb + tx];
            }
        }
        else{
            #pragma unroll
            for(int i = 0; i < nn; i++){
                sB[ i * (NB+1) + tx ] = alpha * B[i * lddb + tx];
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, const int NB, const int NRHS>
__device__ __inline__
void trsm_left_write_B( int tx, int m, int nn,
                        T* B, int lddb, T* sB)
{
    if(tx < m){
        if(nn == NRHS){
            #pragma unroll
            for(int i = 0; i < NRHS; i++){
                B[i * lddb + tx] = sB[ i * (NB+1) + tx ];
            }
        }
        else{
            #pragma unroll
            for(int i = 0; i < nn; i++){
                B[i * lddb + tx] = sB[ i * (NB+1) + tx ];
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, const int NB, const int NRHS, const int CONJA, const int LOWER, const int PACKEDA>
__device__ __inline__
void trsm_right_init_data( int tx, int mm, int n, T alpha,
                           magma_diag_t diag,
                           T* A, int coffA, int roffA, int ldda,
                           T* B, int lddb,
                           T* sA, int slda,
                           T* sB, int sldb)
{
    // init sA
    if(tx < NB){
        #pragma unroll
        for(int i = 0; i < NB; i++){
            sA[i * slda + tx] = make_FloatingPoint(0.0, 0.0);
        }
        sA[tx * slda + tx] = make_FloatingPoint(1.0, 0.0);
    }

    // init sB
    if(tx < mm){
        #pragma unroll
        for(int i = 0; i < NB; i++){
            sB[i * sldb + tx] = make_FloatingPoint(0.0, 0.0);
        }
    }

    // no need to sync because each thread updates the same row it initialized above
    // load A
    if(tx < n){
        if (PACKEDA > 0) {
          right_read_packed_sA<T, NB, CONJA, LOWER>(tx, A, coffA, roffA, sA, n, ldda, slda);
        }
	else {
          right_read_sA<T, NB, CONJA>(tx, A + coffA * ldda + roffA, sA, n, ldda, slda);
        }

        // handle diag
        if(diag == MagmaNonUnit){
            sA[tx * slda + tx] = div(make_FloatingPoint(1.0, 0.0), sA[tx * slda + tx]);
        }else{
            sA[tx * slda + tx] = make_FloatingPoint(1.0, 0.0);
        }
    }

    // load B
    if(tx < mm){
        if(n == NB){
            #pragma unroll
            for(int i = 0; i < NB; i++){
                sB[i * sldb + tx] = alpha * B[i * lddb + tx];
            }
        }
        else{
            #pragma unroll
            for(int i = 0; i < n; i++){
                sB[i * sldb + tx] = alpha * B[i * lddb + tx];
            }
        }
    }
    __syncthreads();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, const int NB, const int NRHS>
__device__ __inline__
void trsm_right_write_B( int tx, int mm, int n,
                         T* B, int lddb, T* sB, int sldb)
{
    if(tx < mm){
        if(n == NB){
            #pragma unroll
            for(int i = 0; i < NB; i++){
                B[i * lddb + tx] = sB[i * sldb + tx];
            }
        }
        else{
            #pragma unroll
            for(int i = 0; i < n; i++){
                B[i * lddb + tx] = sB[i * sldb + tx];
            }
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
/* trsm device functions */
///////////////////////////////////////////////////////////////////////////////////////////////////
/*
trsm modes
lNL: left  - NoTrans - Lower
lNU: left  - NoTrans - Upper
lTL: left  - Trans   - Lower
lTU: left  - Trans   - Upper
rNL: right - NoTrans - Lower
rNU: right - NoTrans - Upper
rTL: right - Trans   - Lower
rTU: right - Trans   - Upper
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
// lNL
template<typename T, const int NB, const int NRHS, const int PACKEDA>
static __device__
void trsm_template_device_lNL(
        magma_diag_t diag, int m, int n,
        T alpha, T* A, int coffA, int roffA, int ldda,
                 T* B, int lddb)
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    B += bx * NRHS * lddb;

    __shared__ T sA[NB * NB];
    __shared__ T sB[ (NB+1) * NRHS];
    T rB[NB] = {make_FloatingPoint(0.0, 0.0)};
    const int nblocks = magma_ceildiv(n, NRHS);
    const int nn = (bx < nblocks-1) ? NRHS : ( n - (nblocks-1) * NRHS );

    trsm_left_init_data<T, NB, NRHS, 0, 1, PACKEDA>(tx, m, nn, alpha, diag, A, coffA, roffA, ldda, B, lddb, sA, sB);
    __syncthreads();

    // B: sB -> rB
    #pragma unroll
    for(int i = 0; i < NB; i++){
       rB[i] = sB[tx * (NB+1) + i];
    }

    // solve in reg
    #pragma unroll
    for(int i = 0; i < NB; i++){
        #pragma unroll
        for(int j = 0; j < i; j++){
            rB[i] -= rB[j] * sA[j * NB + i];
        }
        rB[i] *= sA[i * NB + i];
    }

    // B: rB -> sB
    #pragma unroll
    for(int i = 0; i < NB; i++){
        sB[tx * (NB+1) + i] = rB[i];
    }

    __syncthreads();

    // write B
    trsm_left_write_B<T, NB, NRHS>(tx, m, nn, B, lddb, sB);

}
///////////////////////////////////////////////////////////////////////////////////////////////////
// lNU
template<typename T, const int NB, const int NRHS, const int PACKEDA>
static __device__
void trsm_template_device_lNU(
        magma_diag_t diag, int m, int n,
        T alpha, T* A, int coffA, int roffA, int ldda,
                 T* B, int lddb)
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    B += bx * NRHS * lddb;

    __shared__ T sA[NB * NB];
    __shared__ T sB[(NB+1) * NRHS];
    T rB[NB] = {make_FloatingPoint(0.0, 0.0)};
    const int nblocks = magma_ceildiv(n, NRHS);
    const int nn = (bx < nblocks-1) ? NRHS : n - (nblocks-1)*NRHS ;

    trsm_left_init_data<T, NB, NRHS, 0, 0, PACKEDA>(tx, m, nn, alpha, diag, A, coffA, roffA, ldda, B, lddb, sA, sB);
    __syncthreads();

    // B: sB -> rB
    #pragma unroll
    for(int i = 0; i < NB; i++){
       rB[i] = sB[tx * (NB+1) + i];
    }

    // solve in reg
    #pragma unroll
    for(int i = NB-1; i >= 0; i--){
        #pragma unroll
        for(int j = NB-1; j > i; j--){
            rB[i] -= rB[j] * sA[j * NB + i];
        }
        rB[i] *= sA[i * NB + i];
    }

    // B: rB -> sB
    #pragma unroll
    for(int i = 0; i < NB; i++){
        sB[tx * (NB+1) + i] = rB[i];
    }

    __syncthreads();

    // write B
    trsm_left_write_B<T, NB, NRHS>(tx, m, nn, B, lddb, sB);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// lTL, lCL
template<typename T, const int NB, const int NRHS, const int CONJA, const int PACKEDA>
static __device__
void trsm_template_device_lTL(
        magma_diag_t diag, int m, int n,
        T alpha, T* A, int coffA, int roffA, int ldda,
                 T* B, int lddb)
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    B += bx * NRHS * lddb;

    __shared__ T sA[NB * NB];
    __shared__ T sB[(NB+1) * NRHS];
    const int nblocks = magma_ceildiv(n, NRHS);
    const int nn = (bx < nblocks-1) ? NRHS : n - (nblocks-1)*NRHS ;
    T rB[NB] = {make_FloatingPoint(0.0, 0.0)};

    trsm_left_init_data<T, NB, NRHS, CONJA, 1, PACKEDA>(tx, m, nn, alpha, diag, A, coffA, roffA, ldda, B, lddb, sA, sB);
    __syncthreads();

    // B: sB -> rB
    #pragma unroll
    for(int i = 0; i < NB; i++){
       rB[i] = sB[tx * (NB+1) + i];
    }

    // solve in reg
    #pragma unroll
    for(int i = NB-1; i >= 0; i--){
        #pragma unroll
        for(int j = NB-1; j > i; j--){
            rB[i] -= rB[j] * sA[i * NB + j];
        }
        rB[i] *= sA[i * NB + i];
    }

    // B: rB -> sB
    #pragma unroll
    for(int i = 0; i < NB; i++){
        sB[tx * (NB+1) + i] = rB[i];
    }

    __syncthreads();

    // write B
    trsm_left_write_B<T, NB, NRHS>(tx, m, nn, B, lddb, sB);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// lTU, lCU
template<typename T, const int NB, const int NRHS, const int CONJA, const int PACKEDA>
static __device__
void trsm_template_device_lTU(
        magma_diag_t diag, int m, int n,
        T alpha, T* A, int coffA, int roffA, int ldda,
                 T* B, int lddb)
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    B += bx * NRHS * lddb;

    __shared__ T sA[NB * NB];
    __shared__ T sB[(NB+1) * NRHS];
    const int nblocks = magma_ceildiv(n, NRHS);
    const int nn = (bx < nblocks-1) ? NRHS : n - (nblocks-1)*NRHS ;
    T rB[NB] = {make_FloatingPoint(0.0, 0.0)};

    trsm_left_init_data<T, NB, NRHS, CONJA, 0, PACKEDA>(tx, m, nn, alpha, diag, A, coffA, roffA, ldda, B, lddb, sA, sB);
    __syncthreads();

    // B: sB -> rB
    #pragma unroll
    for(int i = 0; i < NB; i++){
       rB[i] = sB[tx * (NB+1) + i];
    }

    // solve in reg
    #pragma unroll
    for(int i = 0; i < NB; i++){
        #pragma unroll
        for(int j = 0; j < i; j++){
            rB[i] -= rB[j] * sA[i * NB + j];
        }
        rB[i] *= sA[i * NB + i];
    }

    // B: rB -> sB
    #pragma unroll
    for(int i = 0; i < NB; i++){
        sB[tx * (NB+1) + i] = rB[i];
    }

    __syncthreads();

    // write B
    trsm_left_write_B<T, NB, NRHS>(tx, m, nn, B, lddb, sB);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// rNL
template<typename T, const int NB, const int NRHS, const int PACKEDA>
static __device__
void trsm_template_device_rNL(
        magma_diag_t diag, int m, int n,
        T alpha, T* A, int coffA, int roffA, int ldda,
                 T* B, int lddb)
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int slda = NB;
    const int sldb = NRHS+1;
    B += bx * NRHS;

    __shared__ T sA[slda * NB];
    __shared__ T sB[sldb * NB];
    const int nblocks = magma_ceildiv(m, NRHS);
    const int mm = (bx < nblocks-1) ? NRHS : m - (nblocks-1)*NRHS ;

    trsm_right_init_data<T, NB, NRHS, 0, 1, PACKEDA>(tx, mm, n, alpha, diag, A, coffA, roffA, ldda, B, lddb, sA, slda, sB, sldb);

    // solve in sB
    #pragma unroll
    for(int i = NB-1; i >= 0; i--){
        #pragma unroll
        for(int j = NB-1; j > i; j--){
            sB[i * sldb + tx] -= sB[j * sldb + tx] * sA[i * slda + j];
            //rB[i] -= rB[j] * sA[i * NB + j];
        }
        //rB[i] *= sA[i * NB + i];
        sB[i * sldb + tx] *= sA[i * slda + i];
    }

    // write B
    trsm_right_write_B<T, NB, NRHS>(tx, mm, n, B, lddb, sB, sldb);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// rNU
template<typename T, const int NB, const int NRHS, const int PACKEDA>
static __device__
void trsm_template_device_rNU(
        magma_diag_t diag, int m, int n,
        T alpha, T* A, int coffA, int roffA, int ldda,
                 T* B, int lddb)
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int slda = NB;
    const int sldb = NRHS+1;
    B += bx * NRHS;

    __shared__ T sA[slda * NB];
    __shared__ T sB[sldb * NB];
    const int nblocks = magma_ceildiv(m, NRHS);
    const int mm = (bx < nblocks-1) ? NRHS : m - (nblocks-1)*NRHS ;
    //T rB[NB] = {make_FloatingPoint(0.0, 0.0)};

    trsm_right_init_data<T, NB, NRHS, 0, 0, PACKEDA>(tx, mm, n, alpha, diag, A, coffA, roffA, ldda, B, lddb, sA, slda, sB, sldb);

    // solve in sB
    #pragma unroll
    for(int i = 0; i < NB; i++){
        #pragma unroll
        for(int j = 0; j < i; j++){
            sB[i * sldb + tx] -= sB[j * sldb + tx] * sA[i * slda + j];
            //rB[i] -= rB[j] * sA[i * NB + j];
        }
        //rB[i] *= sA[i * NB + i];
        sB[i * sldb + tx] *= sA[i * slda + i];
    }

    // write B
    trsm_right_write_B<T, NB, NRHS>(tx, mm, n, B, lddb, sB, sldb);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// rTL, rCL
template<typename T, const int NB, const int NRHS, const int CONJA, const int PACKEDA>
static __device__
void trsm_template_device_rTL(
        magma_diag_t diag, int m, int n,
        T alpha, T* A, int coffA, int roffA, int ldda,
                 T* B, int lddb)
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int slda = NB;
    const int sldb = NRHS+1;
    B += bx * NRHS;

    __shared__ T sA[slda * NB];
    __shared__ T sB[sldb * NB];
    const int nblocks = magma_ceildiv(m, NRHS);
    const int mm = (bx < nblocks-1) ? NRHS : m - (nblocks-1)*NRHS ;
    //T rB[NB] = {make_FloatingPoint(0.0, 0.0)};

    trsm_right_init_data<T, NB, NRHS, CONJA, 1, PACKEDA>(tx, mm, n, alpha, diag, A, coffA, roffA, ldda, B, lddb, sA, slda, sB, sldb);

    // solve in sB
    #pragma unroll
    for(int i = 0; i < NB; i++){
        #pragma unroll
        for(int j = 0; j < i; j++){
            sB[i * sldb + tx] -= sB[j * sldb + tx] * sA[j * slda + i];
            //rB[i] -= rB[j] * sA[j * NB + i];
        }
        //rB[i] *= sA[i * NB + i];
        sB[i * sldb + tx] *= sA[i * slda + i];
    }

    // write B
    trsm_right_write_B<T, NB, NRHS>(tx, mm, n, B, lddb, sB, sldb);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// rTU, rCU
template<typename T, const int NB, const int NRHS, const int CONJA, const int PACKEDA>
static __device__
void trsm_template_device_rTU(
        magma_diag_t diag, int m, int n,
        T alpha, T* A, int coffA, int roffA, int ldda,
                 T* B, int lddb)
{
    const int tx = threadIdx.x;
    const int bx = blockIdx.x;
    const int slda = NB;
    const int sldb = NRHS+1;
    B += bx * NRHS;

    __shared__ T sA[slda * NB];
    __shared__ T sB[sldb * NB];
    const int nblocks = magma_ceildiv(m, NRHS);
    const int mm = (bx < nblocks-1) ? NRHS : m - (nblocks-1)*NRHS ;
    //T rB[NB] = {make_FloatingPoint(0.0, 0.0)};

    trsm_right_init_data<T, NB, NRHS, CONJA, 0, PACKEDA>(tx, mm, n, alpha, diag, A, coffA, roffA, ldda, B, lddb, sA, slda, sB, sldb);

    // solve in sB
    #pragma unroll
    for(int i = NB-1; i >= 0; i--){
        #pragma unroll
        for(int j = NB-1; j > i; j--){
            sB[i * sldb + tx] -= sB[j * sldb + tx] * sA[j * slda + i];
            //rB[i] -= rB[j] * sA[j * NB + i];
        }
        sB[i * sldb + tx] *= sA[i * slda + i];
        //rB[i] *= sA[i * NB + i];
    }

    // write B
    trsm_right_write_B<T, NB, NRHS>(tx, mm, n, B, lddb, sB, sldb);
}
///////////////////////////////////////////////////////////////////////////////////////////////////
#endif //TRSM_TEMPLATE_DEVICE_CUH
