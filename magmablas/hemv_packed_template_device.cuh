/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
       @author Natalie Beams

       Originally based on an implementation by KBLAS (https://ecrc.kaust.edu.sa/Pages/Res-kblas.aspx)
       This version is for matrices stored in a packed format (only lower or upper triangular part).
*/

#ifndef HEMV_PACKED_TEMPLATE_DEVICE_CUH
#define HEMV_PACKED_TEMPLATE_DEVICE_CUH

#define EPT    (NB/TY)
#define LPACKED(i_, j_, N_) (N_*(j_) - (j_)*(j_+1)/2 + i_)
#define UPACKED(i_, j_) ((j_)*(j_+1)/2 + i_)
/******************************************************************************/
template <typename T, const int NB, const int TY, const int LOWER>
__device__ __inline__ void
hemv_packed_diag_device( int N,
                         T alpha, T *A, int offA, int ldda,
                         T *X, int incx,
                         T beta , T *Y, int incy )
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int n  = min(NB, N - bx * NB);

    T res = make_FloatingPoint(0.0, 0.0);
    T ry  = make_FloatingPoint(0.0, 0.0);

    __shared__ T sA[NB * NB];
    __shared__ T sX[NB];

    int row = bx * NB + tx + offA;
    int col = bx * NB + ty + offA; 
    X += bx * NB * incx;
    Y += bx * NB * incy;

    // init sA/sX to zeros
    #pragma unroll
    for(int i = 0; i < NB; i += TY){
        sA[(i + ty) * NB + tx] = make_FloatingPoint(0.0, 0.0);
    }
    if(ty == 0){
        sX[tx] = make_FloatingPoint(0.0, 0.0);
    }
    if(tx >= n) return;

    // load x/y
    if(ty == 0 && tx < n){
        sX[tx] = X[tx * incx];
        ry = Y[tx * incy] * beta;
    }

    // read sA
    if(n < NB){
        int i;
        #pragma unroll
        for(i = 0; i < n-TY; i+=TY){
            sA[(i+ty) * NB + tx] = A[(LOWER > 0) ? LPACKED(row, col + i, ldda) : UPACKED(row, col + i)];
        }
        if(ty < (n-i)){
            sA[(i+ty) * NB + tx] = A[(LOWER > 0) ? LPACKED(row, col + i, ldda) : UPACKED(row, col + i)];
        }
    }else{
        #pragma unroll
        for(int i = 0; i < NB; i+= TY)
            sA[(i + ty) * NB + tx] = A[(LOWER > 0) ? LPACKED(row, col + i, ldda) : UPACKED(row, col + i)];
    }
    __syncthreads();

    // mirror
    if( LOWER > 0 ){
        #pragma unroll
        for(int i = 0; i < NB; i+=TY){
            if(tx < ty+i){
                sA[(i + ty) * NB + tx] = conj( sA[ tx * NB + (i+ty)] );
            }
        }
    }else{
        #pragma unroll
        for(int i = 0; i < NB; i+=TY){
            if(tx > ty+i){
                sA[(i+ty) * NB + tx] = conj( sA[tx * NB + (i+ty)] );
            }
        }
    }
    __syncthreads();

    // ignore imaginary part of diagonal elements
    if(ty == 0){
        sA[ tx * NB + tx ] = make_FloatingPoint( real(sA[ tx * NB + tx ]), 0.0 );
    }
    __syncthreads();

    // compute
    #pragma unroll
    for(int i = 0; i < NB; i += TY){
        res += sA[ (i + ty) * NB + tx ] * sX[i + ty];
    }

    __syncthreads();
    sA[ty * NB + tx] = res;
    __syncthreads();

    if (ty == 0) {
        res = make_FloatingPoint( 0.0, 0.0 );
        #pragma unroll
        for(int i = 0; i < TY; i++)
            res += sA[i * NB + tx];
        res *= alpha;
        if(tx < n){
            Y[tx * incy] = res + ry;
        }
    }
}

/******************************************************************************/
template <typename T, const int NB, const int TY>
__device__ __inline__ void
hemv_packed_lower_device( int N, T alpha,
                          T *A, int offA, int ldda,
                          T *X, int incx,
                          T *Y, int incy )
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    T *X_, *Y_;
    T rA[EPT], rB[EPT];
    T rv[EPT];
    T rh = make_FloatingPoint(0.0, 0.0);

    __shared__ T sA[NB * (NB+1)];
    __shared__ T sX[NB];

    const int gridx = magma_ceildiv(N, NB);    // do not use gridDim.x (doesn't work for vbatched)
    const int nfull = (gridx-bx-2); // exclude the diagonal block and the last full/partial block
    const int start = by * (nfull/gridDim.y) + min(by, nfull%gridDim.y);
    const int count = nfull/gridDim.y + ( by < (nfull%gridDim.y) );
    if( (bx == gridx-1) || (by < gridDim.y-1 && count == 0))return;

    int row = bx * NB + start * NB + tx + offA;
    int col = bx * NB + ty + offA;
    X += bx * NB * incx;
    X_ = X;
    X += start * NB * incx;
    Y += bx * NB * incy;
    Y_ = Y;
    Y_+= start * NB * incy;

    if(ty == 0){
        sX[tx] = X_[tx * incx];
    }
    __syncthreads();

    #pragma unroll
    for(int i = 0; i < EPT; i++){
        rv[i] = make_FloatingPoint(0.0, 0.0);
    }

    row  += NB;
    X  += NB * incx;
    Y_ += NB * incy;

    if(count > 0){
        #pragma unroll
        for(int k = 0; k < EPT; k++){
            rB[k] = A[LPACKED(row, col + k * TY, ldda)];
        }
    }
    #pragma unroll
    for(int i = 0; i < count; i++){
        #pragma unroll
        for(int k = 0; k < EPT; k++){
            rA[k] = rB[k];
        }

        row  += NB;
        if(i < count-1){
            #pragma unroll
            for(int k = 0; k < EPT; k++){
                rB[k] = A[LPACKED(row, col + k * TY, ldda)];
            }
        }

        rh = make_FloatingPoint(0.0, 0.0);
        #pragma unroll
        for(int k = 0; k < EPT; k++){
            rh += rA[k] * sX[k * TY + ty];
            rv[k] += conj( rA[k] ) * X[tx * incx];
        }

        // Horizontal block should be stored in global memory
        __syncthreads();
        sA[ty * (NB+1) + tx] = rh;
        __syncthreads();
        if(ty == 0)
        {
            rh = make_FloatingPoint(0.0, 0.0);
            #pragma unroll
            for (int k = 0; k < TY; k++) {
                rh += sA[k * (NB+1) + tx];
            }
            rh *= alpha;

            magmablas_atomic_add(&Y_[incy * tx], rh);
        }
        X  += NB * incx;
        Y_ += NB * incy;
    }

    // last irregular block
    const int n = N - (bx+nfull+1)*NB;    // size of remaining full/partial block
    if(by == gridDim.y-1){
        if(tx < n) {
            #pragma unroll
            for(int k = 0; k < EPT; k++){
                rA[k] = A[LPACKED(row, col + k * TY, ldda)];
            }

            rh = make_FloatingPoint(0.0, 0.0);
            #pragma unroll
            for(int k = 0; k < EPT; k++){
                rh += rA[k] * sX[k * TY + ty];
                rv[k] += conj( rA[k] ) * X[tx * incx];
            }
        }
        // Horizontal block should be stored in global memory
        __syncthreads();
        if(tx < n) {
            sA[ty * (NB+1) + tx] = rh;
        }
        __syncthreads();
        if(ty == 0 && tx < n) {
            rh = make_FloatingPoint(0.0, 0.0);
            #pragma unroll
            for (int k = 0; k < TY; k++) {
                rh += sA[k * (NB+1) + tx];
            }
            rh *= alpha;

            magmablas_atomic_add(&Y_[incy * tx], rh);
        }
    }

    __syncthreads();
    #pragma unroll
    for(int k = 0; k < EPT; k++){
        sA[(k * TY + ty) * (NB+1) + tx] = rv[k];
    }
    __syncthreads();

    if(ty == 0){
        rv[0] = make_FloatingPoint(0.0, 0.0);
        #pragma unroll
        for(int k = 0; k < NB; k++){
            rv[0] += sA[tx * (NB+1) + k];
        }
           rv[0] *= alpha;
           magmablas_atomic_add(&Y[incy * tx], rv[0]);
    }
}

/******************************************************************************/
template <typename T, const int NB, const int TY>
__device__ __inline__ void
hemv_packed_upper_device( int N, T alpha,
                          T *A, int offA, int ldda,
                          T *X, int incx,
                          T *Y, int incy )
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    T *X_, *Y_;
    T rA[EPT], rB[EPT];
    T rv[EPT];
    int addr[EPT];
    T rh = make_FloatingPoint(0.0, 0.0);

    __shared__ T sA[NB * (NB+1)];
    __shared__ T sX[NB];

    const int gridx = magma_ceildiv(N, NB);    // do not use gridDim.x (doesn't work for vbatched)
    const int nr = N - (gridx-1) * NB;
    const int nblocks = bx;
    const int start = by * (nblocks/gridDim.y) + min(by, nblocks%gridDim.y);
    const int count = nblocks/gridDim.y + ( by < (nblocks%gridDim.y) );
    if( bx == 0 || count == 0)return;

    int row = start * NB + tx + offA;
    int col_base = bx * NB + ty + offA;

    X_ = X + bx * NB * incx;
    X += start * NB * incx;
    Y_ = Y + start * NB * incy;
    Y += bx * NB * incy;

    // init
    if(ty == 0) sX[tx] = make_FloatingPoint(0.0, 0.0);
    if(bx == gridx-1 && nr < NB){
        #pragma unroll
        for(int i = 0; i < EPT; i++){
            rv[i] = make_FloatingPoint(0.0, 0.0);
            addr[i] = min(i*TY, nr-1);
        }
    }
    else{
        #pragma unroll
        for(int i = 0; i < EPT; i++){
            rv[i] = make_FloatingPoint(0.0, 0.0);
            addr[i] = i * TY;
        }
    }

    if(bx == gridx-1 && nr < NB){
        if(ty == 0 && tx < nr)
            sX[tx] = X_[tx * incx];
    }
    else{
        if(ty == 0)
            sX[tx] = X_[tx * incx];
    }
    __syncthreads();

    #pragma unroll
    for(int k = 0; k < EPT; k++)
        rB[k] = A[UPACKED(row, col_base + addr[k])];

    #pragma unroll
    for(int i = 0; i < count; i++){
        #pragma unroll
        for(int k = 0; k < EPT; k++)
            rA[k] = rB[k];

        row  += NB;
        if(i < count-1){
            #pragma unroll
            for(int k = 0; k < EPT; k++)
                rB[k] = A[UPACKED(row, col_base + addr[k])];
        }

        rh = make_FloatingPoint(0.0, 0.0);
        #pragma unroll
        for(int k = 0; k < EPT; k++){
            rh += rA[k] * sX[k * TY + ty];
            rv[k] += conj( rA[k] ) * X[tx * incx];
        }

        // Horizontal block should be stored in global memory
        __syncthreads();
        sA[ty * (NB+1) + tx] = rh;
        __syncthreads();
        if(ty == 0)
        {
            rh = make_FloatingPoint(0.0, 0.0);
            #pragma unroll
            for (int k = 0; k < TY; k++)
                rh += sA[k * (NB+1) + tx];

            rh *= alpha;

            magmablas_atomic_add(&Y_[incy * tx], rh);
        }
        X  += NB * incx;
        Y_ += NB * incy;
    }

    __syncthreads();
    #pragma unroll
    for(int k = 0; k < EPT; k++){
        sA[(k * TY + ty) * (NB+1) + tx] = rv[k];
    }
    __syncthreads();

    if(ty == 0){
        rv[0] = make_FloatingPoint(0.0, 0.0);
        #pragma unroll
        for(int k = 0; k < NB; k++){
            rv[0] += sA[tx * (NB+1) + k];
        }
        rv[0] *= alpha;
        if (bx == gridx-1 && nr < NB) {
            if (tx < nr)
                magmablas_atomic_add(&Y[incy * tx], rv[0]);
        }
        else {
            magmablas_atomic_add(&Y[incy * tx], rv[0]);
        }
    }
}

/******************************************************************************/
#endif // HEMV_PACKED_TEMPLATE_DEVICE_CUH
