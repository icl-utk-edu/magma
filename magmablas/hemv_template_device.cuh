/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Ahmad Abdelfattah
       
       Originally based on an implementation by KBLAS (https://ecrc.kaust.edu.sa/Pages/Res-kblas.aspx)
*/

#ifndef HEMV_TEMPLATE_DEVICE_CUH
#define HEMV_TEMPLATE_DEVICE_CUH

#define EPT    (NB/TY)
/******************************************************************************/
template <typename T, const int NB, const int TY>
__device__ __inline__ void
hemv_diag_device( magma_uplo_t uplo, int N, 
                  T alpha, T *A, int ldda, 
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
    
    A += bx * NB * (ldda + 1) + ty * ldda + tx;
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
            sA[(i+ty) * NB + tx] = A[i * ldda];
        }
        if(ty < (n-i)){
            sA[(i+ty) * NB + tx] = A[i * ldda];
        }
    }else{
        #pragma unroll
        for(int i = 0; i < NB; i+= TY)
            sA[(i + ty) * NB + tx] = A[i * ldda];    
    }
    __syncthreads();
    
    // mirror 
    if(uplo == MagmaLower){
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
hemv_lower_device( int N, T alpha, 
                   T *A, int ldda, 
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
    
    A += bx * NB * (ldda + 1) + start * NB + ty * ldda + tx;
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
    
    A  += NB;
    X  += NB * incx;
    Y_ += NB * incy;
    
    if(count > 0){
        #pragma unroll
        for(int k = 0; k < EPT; k++){
            rB[k] = A[k * TY * ldda];
        }
    }
    #pragma unroll
    for(int i = 0; i < count; i++){
        #pragma unroll
        for(int k = 0; k < EPT; k++){
            rA[k] = rB[k];
        }
        
        A  += NB;
        if(i < count-1){
            #pragma unroll
            for(int k = 0; k < EPT; k++){
                rB[k] = A[k * TY * ldda];
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
    if(by == gridDim.y-1 && tx < n){
        #pragma unroll
        for(int k = 0; k < EPT; k++){
            rA[k] = A[k * TY * ldda];
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
hemv_upper_device( int N, T alpha, 
                   T *A, int ldda, 
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
    
    if(bx == gridx-1 && nr < NB)
        A += bx * NB * ldda + start * NB; // + ty * ldda + tx;
    else 
        A += bx * NB * ldda + start * NB + ty * ldda + tx;
    
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
            addr[i] = min(i*TY + ty, nr-1) * ldda + tx; 
        }
    }
    else{
        #pragma unroll
        for(int i = 0; i < EPT; i++){
            rv[i] = make_FloatingPoint(0.0, 0.0);
            addr[i] = i * TY * ldda; 
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
        rB[k] = A[ addr[k] ];
    
    #pragma unroll
    for(int i = 0; i < count; i++){
        #pragma unroll
        for(int k = 0; k < EPT; k++)
            rA[k] = rB[k];

        A  += NB;
        if(i < count-1){
            #pragma unroll
            for(int k = 0; k < EPT; k++)
                rB[k] = A[ addr[k] ];
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
#endif // HEMV_TEMPLATE_DEVICE_CUH
