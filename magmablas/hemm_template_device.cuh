/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah

*/

#ifndef HEMM_TEMPLATE_DEVICE_CUH
#define HEMM_TEMPLATE_DEVICE_CUH
/******************************************************************************/
// op<trans>( x ) returns x or conj(x).
template<typename T, const int CONJA>
__device__ static inline T OP( T& x )
{
    if(CONJA == 1) return conj(x);
    else return x;
}

/******************************************************************************/
template<class T, const int DIM, const int BLK_M, const int BLK_N, 
         const int THR_M, const int THR_N, const int CONJA>
static __device__ 
void hemm_template_device_ll(
    int M, int N, 
    const T* __restrict__ A, int LDA,
    const T* __restrict__ B, int LDB,
    T*       __restrict__ C, int LDC,
    T alpha, T beta )
{
    const int tx = threadIdx.x;  // thread's m dimension
    const int ty = threadIdx.y;  // thread's n dimension

    const int bx = blockIdx.x;   // block's m dimension
    const int by = blockIdx.y;   // block's n dimension

    __shared__ T sA[BLK_M][BLK_M+1];
    __shared__ T sB[BLK_N][BLK_M+1];

    // Registers for the innermost loop
    T rC[THR_N][THR_M];
    T tmp; 
    
    const T *offs_dA = A + bx*BLK_M + ty*LDA + tx;
    ptrdiff_t boundA = (LDA*(M-1) + M) - ( bx*BLK_M  + ty*LDA + tx ) -1;
    
    const T *offs_dB = B + by*BLK_N*LDB + ty*LDB + tx;
    ptrdiff_t boundB = (LDB*(N-1) + M) - ( by*BLK_N*LDB + ty*LDB + tx ) -1;
    
    // Zero C
    #pragma unroll
    for (int n = 0; n < THR_N; n++)
        #pragma unroll
        for (int m = 0; m < THR_M; m++)
            rC[n][m] = make_FloatingPoint(0.0, 0.0);
    
    const int part_1 = BLK_M * bx;
    const int part_2 = min(BLK_M, M - part_1);
    const int part_3 = M - ( part_1 + part_2 ); 
    int kk;
    
    // part 1
    for (kk = 0; kk < part_1; kk += BLK_M){
        #pragma unroll
        for (int n = 0; n < BLK_M; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sA[n+ty][m+tx] = fetch(A, m, n, boundA);
        
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
        __syncthreads();

        offs_dA += BLK_M*LDA;
        boundA  -= BLK_M*LDA;

        offs_dB += BLK_M;
        boundB  -= BLK_M;
        
        // Multiply
        #pragma unroll
        for (int k = 0; k < BLK_M; k++)
        {
            #pragma unroll
            for (int n = 0; n < THR_N; n++) {
                #pragma unroll
                for (int m = 0; m < THR_M; m++) {
                    fma( sA[k][m*DIM+tx], sB[n*DIM+ty][k], rC[n][m]);
                }
            }
        }
        __syncthreads();
    }
    
    //part 2
    if(part_2 > 0){
        // read diagonal A block
        #pragma unroll
        for (int n = 0; n < BLK_M; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sA[n+ty][m+tx] = fetch(A, m, n, boundA);    
        __syncthreads();
        
        // read B block
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
        // mirror A block - copy lower to upper
        #pragma unroll
        for (int n = 0; n < BLK_M; n += DIM){
            if(ty > tx){
                sA[n+ty][n+tx] = OP<T, CONJA>( sA[n+tx][n+ty] );
            }else if(ty == tx){
                sA[n+ty][n+tx] = make_FloatingPoint( real(sA[n+ty][n+tx]), 0.0 );
            }
            #pragma unroll
            for (int m = n+DIM; m < BLK_M; m += DIM)
                sA[m+ty][n+tx] = OP<T, CONJA>( sA[n+tx][m+ty] );
        }
        
        // advance pointers 
        offs_dA += BLK_M;
        boundA  -= BLK_M;
        
        offs_dB += BLK_M;
        boundB  -= BLK_M;
        
        __syncthreads();
        // Multiply - account for irregular sizes
        if(part_2 < BLK_M){
            #pragma unroll
            for (int k = 0; k < part_2; k++){
                #pragma unroll
                for (int n = 0; n < THR_N; n++) {
                    #pragma unroll
                    for (int m = 0; m < THR_M; m++) {
                        fma( sA[k][m*DIM+tx], sB[n*DIM+ty][k], rC[n][m]);
                    }
                }
            }
        }else{
            #pragma unroll
            for (int k = 0; k < BLK_M; k++){
                #pragma unroll
                for (int n = 0; n < THR_N; n++) {
                    #pragma unroll
                    for (int m = 0; m < THR_M; m++) {
                        fma( sA[k][m*DIM+tx], sB[n*DIM+ty][k], rC[n][m]);
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // part three
    for (kk = 0; kk < part_3-BLK_M; kk += BLK_M){
        #pragma unroll
        for (int n = 0; n < BLK_M; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM){
                tmp = fetch(A, m, n, boundA); 
                sA[m+tx][n+ty] = OP<T, CONJA>( tmp );
            }
        
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
        __syncthreads();

        offs_dA += BLK_M;
        boundA  -= BLK_M;
        
        offs_dB += BLK_M;
        boundB  -= BLK_M;
        
        // Multiply
        #pragma unroll
        for (int k = 0; k < BLK_M; k++)
        {
            #pragma unroll
            for (int n = 0; n < THR_N; n++) {
                #pragma unroll
                for (int m = 0; m < THR_M; m++) {
                    fma( sA[k][m*DIM+tx], sB[n*DIM+ty][k], rC[n][m]);
                }
            }
        }
        __syncthreads();
    }
    
    // Multiply last full (BLK_M) or partial block
    kk = part_3 - kk;
    #pragma unroll
    for (int n = 0; n < BLK_M; n += DIM)
        #pragma unroll
        for (int m = 0; m < BLK_M; m += DIM){
            tmp = fetch(A, m, n, boundA);
            sA[m+tx][n+ty] = OP<T, CONJA>( tmp );
        }
    
    #pragma unroll
    for (int n = 0; n < BLK_N; n += DIM)
        #pragma unroll
        for (int m = 0; m < BLK_M; m += DIM)
            sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
    __syncthreads();
    // Multiply
    #pragma unroll
    for (int k = 0; k < kk; k++){
        #pragma unroll
        for (int n = 0; n < THR_N; n++) {
            #pragma unroll
            for (int m = 0; m < THR_M; m++) {
                fma( sA[k][m*DIM+tx], sB[n*DIM+ty][k], rC[n][m] );
            }
        }
    }

    // Store C regs->dev
    #pragma unroll
    for (int n = 0; n < THR_N; n++) {
        int coord_dCn = by*BLK_N + n*DIM + ty;
        #pragma unroll
        for (int m = 0; m < THR_M; m++) {
            int coord_dCm = bx*BLK_M + m*DIM + tx;
            if (coord_dCm < M && coord_dCn < N) {
                int offsC = coord_dCn*LDC + coord_dCm;

                T &regC = rC[n][m];
                T &memC = C[offsC];

                memC = add(mul(alpha, regC), mul(beta, memC));
            }
        }
    }
}

/******************************************************************************/
template<class T, const int DIM, const int BLK_M, const int BLK_N, 
         const int THR_M, const int THR_N, const int CONJA>
static __device__ 
void hemm_template_device_lu(
    int M, int N, 
    const T* __restrict__ A, int LDA,
    const T* __restrict__ B, int LDB,
    T*       __restrict__ C, int LDC,
    T alpha, T beta )
{
    const int tx = threadIdx.x;  // thread's m dimension
    const int ty = threadIdx.y;  // thread's n dimension

    const int bx = blockIdx.x;   // block's m dimension
    const int by = blockIdx.y;   // block's n dimension

    __shared__ T sA[BLK_M][BLK_M+1];
    __shared__ T sB[BLK_N][BLK_M+1];

    // Registers for the innermost loop
    T rC[THR_N][THR_M];
    T tmp; 
    
    const T *offs_dA = A + bx*BLK_M*LDA + ty*LDA + tx;
    ptrdiff_t boundA = (LDA*(M-1) + M) - ( bx*BLK_M*LDA  + ty*LDA + tx ) -1;
    
    const T *offs_dB = B + by*BLK_N*LDB + ty*LDB + tx;
    ptrdiff_t boundB = (LDB*(N-1) + M) - ( by*BLK_N*LDB + ty*LDB + tx ) -1;
    
    // Zero C
    #pragma unroll
    for (int n = 0; n < THR_N; n++)
        #pragma unroll
        for (int m = 0; m < THR_M; m++)
            rC[n][m] = make_FloatingPoint(0.0, 0.0);
    
    const int part_1 = BLK_M * bx;
    const int part_2 = min(BLK_M, M - part_1);
    const int part_3 = M - ( part_1 + part_2 ); 
    int kk;

    
    // part 1
    for (kk = 0; kk < part_1; kk += BLK_M)
    {
        #pragma unroll
        for (int n = 0; n < BLK_M; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM){
                tmp = fetch(A, m, n, boundA); 
                sA[m+tx][n+ty] = OP<T, CONJA>( tmp );
            }
        
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
        __syncthreads();

        offs_dA += BLK_M;
        boundA  -= BLK_M;

        offs_dB += BLK_M;
        boundB  -= BLK_M;
        
        // Multiply
        #pragma unroll
        for (int k = 0; k < BLK_M; k++)
        {
            #pragma unroll
            for (int n = 0; n < THR_N; n++) {
                #pragma unroll
                for (int m = 0; m < THR_M; m++) {
                    fma( sA[k][m*DIM+tx], sB[n*DIM+ty][k], rC[n][m]);
                }
            }
        }
        __syncthreads();
    }
    
    //part 2
    if(part_2 > 0){
        // read diagonal A block
        #pragma unroll
        for (int n = 0; n < BLK_M; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sA[n+ty][m+tx] = fetch(A, m, n, boundA);
        __syncthreads();
        
        // read B block
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
        // mirror A block - copy upper to lower
        #pragma unroll
        for (int n = 0; n < BLK_M; n += DIM){
            if(ty < tx){
                sA[n+ty][n+tx] = OP<T, CONJA>( sA[n+tx][n+ty] );
            }else if(ty == tx){
                sA[n+ty][n+tx] = make_FloatingPoint( real(sA[n+ty][n+tx]), 0.0 );
            }
            #pragma unroll
            for (int m = n+DIM; m < BLK_M; m += DIM)
                sA[n+ty][m+tx] = OP<T, CONJA>( sA[m+tx][n+ty] );
        }
        
        // advance pointers 
        offs_dA += BLK_M*LDA;
        boundA  -= BLK_M*LDA;

        offs_dB += BLK_M;
        boundB  -= BLK_M;
        
        __syncthreads();
        // Multiply
        if(part_2 < BLK_M){
            #pragma unroll
            for (int k = 0; k < part_2; k++){
                #pragma unroll
                for (int n = 0; n < THR_N; n++) {
                    #pragma unroll
                    for (int m = 0; m < THR_M; m++) {
                        fma( sA[k][m*DIM+tx], sB[n*DIM+ty][k], rC[n][m]);
                    }
                }
            }
        }else{
            #pragma unroll
            for (int k = 0; k < BLK_M; k++){
                #pragma unroll
                for (int n = 0; n < THR_N; n++) {
                    #pragma unroll
                    for (int m = 0; m < THR_M; m++) {
                        fma( sA[k][m*DIM+tx], sB[n*DIM+ty][k], rC[n][m]);
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // part three
    for (kk = 0; kk < part_3-BLK_M; kk += BLK_M){
        #pragma unroll
        for (int n = 0; n < BLK_M; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sA[n+ty][m+tx] = fetch(A, m, n, boundA);
        
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sB[n+ty][m+tx] = fetch(B, m, n, boundB);

        offs_dA += BLK_M*LDA;
        boundA  -= BLK_M*LDA;
        
        offs_dB += BLK_M;
        boundB  -= BLK_M;
        
        __syncthreads();
        // Multiply
        #pragma unroll
        for (int k = 0; k < BLK_M; k++)
        {
            #pragma unroll
            for (int n = 0; n < THR_N; n++) {
                #pragma unroll
                for (int m = 0; m < THR_M; m++) {
                    fma( sA[k][m*DIM+tx], sB[n*DIM+ty][k], rC[n][m]);
                }
            }
        }
        __syncthreads();
    }
    
    // Multiply last full (BLK_M) or partial block
    kk = part_3 - kk;
    #pragma unroll
    for (int n = 0; n < BLK_M; n += DIM)
        #pragma unroll
        for (int m = 0; m < BLK_M; m += DIM)
            sA[n+ty][m+tx] = fetch(A, m, n, boundA);
    
    #pragma unroll
    for (int n = 0; n < BLK_N; n += DIM)
        #pragma unroll
        for (int m = 0; m < BLK_M; m += DIM)
            sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
    __syncthreads();
    // Multiply
    #pragma unroll
    for (int k = 0; k < kk; k++){
        #pragma unroll
        for (int n = 0; n < THR_N; n++) {
            #pragma unroll
            for (int m = 0; m < THR_M; m++) {
                fma( sA[k][m*DIM+tx], sB[n*DIM+ty][k], rC[n][m] );
            }
        }
    }

    // Store C regs->dev
    #pragma unroll
    for (int n = 0; n < THR_N; n++) {
        int coord_dCn = by*BLK_N + n*DIM + ty;
        #pragma unroll
        for (int m = 0; m < THR_M; m++) {
            int coord_dCm = bx*BLK_M + m*DIM + tx;
            if (coord_dCm < M && coord_dCn < N) {
                int offsC = coord_dCn*LDC + coord_dCm;

                T &regC = rC[n][m];
                T &memC = C[offsC];

                memC = add(mul(alpha, regC), mul(beta, memC));
            }
        }
    }
}

/******************************************************************************/
template<class T, const int DIM, const int BLK_M, const int BLK_N, 
         const int THR_M, const int THR_N, const int CONJA>
static __device__ 
void hemm_template_device_rl(
    int M, int N, 
    const T* __restrict__ A, int LDA,
    const T* __restrict__ B, int LDB,
    T*       __restrict__ C, int LDC,
    T alpha, T beta )
{
    const int tx = threadIdx.x;  // thread's m dimension
    const int ty = threadIdx.y;  // thread's n dimension

    const int bx = blockIdx.x;   // block's m dimension
    const int by = blockIdx.y;   // block's n dimension

    __shared__ T sA[BLK_N][BLK_N+1];
    __shared__ T sB[BLK_N][BLK_M+1];

    // Registers for the innermost loop
    T rC[THR_N][THR_M];
    T tmp; 
    
    const T *offs_dA = A + by*BLK_N + ty*LDA + tx;
    ptrdiff_t boundA = (LDA*(N-1) + N) - ( by*BLK_N  + ty*LDA + tx ) -1;
    
    const T *offs_dB = B + bx*BLK_M + ty*LDB + tx;
    ptrdiff_t boundB = (LDB*(N-1) + M) - ( bx*BLK_M + ty*LDB + tx ) -1;
    
    // Zero C
    #pragma unroll
    for (int n = 0; n < THR_N; n++)
        #pragma unroll
        for (int m = 0; m < THR_M; m++)
            rC[n][m] = make_FloatingPoint(0.0, 0.0);
    
    const int part_1 = BLK_N * by;
    const int part_2 = min(BLK_N, N - part_1);
    const int part_3 = N - ( part_1 + part_2 ); 
    int kk;
    
    // part 1
    for (kk = 0; kk < part_1; kk += BLK_N){
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_N; m += DIM){
                tmp = fetch(A, m, n, boundA); 
                sA[m+tx][n+ty] = OP<T, CONJA>( tmp );
            }
        
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
        __syncthreads();

        offs_dA += BLK_N*LDA;
        boundA  -= BLK_N*LDA;

        offs_dB += BLK_N*LDB;
        boundB  -= BLK_N*LDB;
        
        // Multiply
        #pragma unroll
        for (int k = 0; k < BLK_N; k++)
        {
            #pragma unroll
            for (int n = 0; n < THR_N; n++) {
                #pragma unroll
                for (int m = 0; m < THR_M; m++) {
                    fma( sB[k][m*DIM+tx], sA[n*DIM+ty][k], rC[n][m] );
                }
            }
        }
        __syncthreads();
    }
    
    //part 2
    if(part_2 > 0){
        // read diagonal A block
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_N; m += DIM)
                sA[n+ty][m+tx] = fetch(A, m, n, boundA);
        __syncthreads();
        
        // read B block
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
        // mirror A block - copy lower to upper
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM){
            if(ty > tx){
                sA[n+ty][n+tx] = OP<T, CONJA>( sA[n+tx][n+ty] );
            }else if(ty == tx){
                sA[n+ty][n+tx] = make_FloatingPoint( real(sA[n+ty][n+tx]), 0.0 );
            }
            #pragma unroll
            for (int m = n+DIM; m < BLK_N; m += DIM)
                sA[m+ty][n+tx] = OP<T, CONJA>( sA[n+tx][m+ty] );
        }
        
        // advance pointers 
        offs_dA += BLK_N;
        boundA  -= BLK_N;
        
        offs_dB += BLK_N*LDB;
        boundB  -= BLK_N*LDB;
        
        __syncthreads();
        // Multiply - account for irregular sizes
        if(part_2 < BLK_N){
            #pragma unroll
            for (int k = 0; k < part_2; k++){
                #pragma unroll
                for (int n = 0; n < THR_N; n++) {
                    #pragma unroll
                    for (int m = 0; m < THR_M; m++) {
                        fma( sB[k][m*DIM+tx], sA[n*DIM+ty][k], rC[n][m] );
                    }
                }
            }
        }else{
            #pragma unroll
            for (int k = 0; k < BLK_N; k++){
                #pragma unroll
                for (int n = 0; n < THR_N; n++) {
                    #pragma unroll
                    for (int m = 0; m < THR_M; m++) {
                        fma( sB[k][m*DIM+tx], sA[n*DIM+ty][k], rC[n][m] );
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // part three
    for (kk = 0; kk < part_3-BLK_N; kk += BLK_N){
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_N; m += DIM){ 
                sA[n+ty][m+tx] = fetch(A, m, n, boundA);
            }
        
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
        __syncthreads();

        offs_dA += BLK_N;
        boundA  -= BLK_N;
        
        offs_dB += BLK_N*LDB;
        boundB  -= BLK_N*LDB;
        
        // Multiply
        #pragma unroll
        for (int k = 0; k < BLK_N; k++)
        {
            #pragma unroll
            for (int n = 0; n < THR_N; n++) {
                #pragma unroll
                for (int m = 0; m < THR_M; m++) {
                    fma( sB[k][m*DIM+tx], sA[n*DIM+ty][k], rC[n][m] );
                }
            }
        }
        __syncthreads();
    }
    
    // Multiply last full (BLK_N) or partial block
    kk = part_3 - kk;
    #pragma unroll
    for (int n = 0; n < BLK_N; n += DIM)
        #pragma unroll
        for (int m = 0; m < BLK_N; m += DIM){
            sA[n+ty][m+tx] = fetch(A, m, n, boundA);
        }
    
    #pragma unroll
    for (int n = 0; n < BLK_N; n += DIM)
        #pragma unroll
        for (int m = 0; m < BLK_M; m += DIM)
            sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
    __syncthreads();
    // Multiply
    #pragma unroll
    for (int k = 0; k < kk; k++){
        #pragma unroll
        for (int n = 0; n < THR_N; n++) {
            #pragma unroll
            for (int m = 0; m < THR_M; m++) {
                fma( sB[k][m*DIM+tx], sA[n*DIM+ty][k], rC[n][m] );
            }
        }
    }

    // Store C regs->dev
    #pragma unroll
    for (int n = 0; n < THR_N; n++) {
        int coord_dCn = by*BLK_N + n*DIM + ty;
        #pragma unroll
        for (int m = 0; m < THR_M; m++) {
            int coord_dCm = bx*BLK_M + m*DIM + tx;
            if (coord_dCm < M && coord_dCn < N) {
                int offsC = coord_dCn*LDC + coord_dCm;

                T &regC = rC[n][m];
                T &memC = C[offsC];

                memC = add(mul(alpha, regC), mul(beta, memC));
            }
        }
    }
}

/******************************************************************************/
template<class T, const int DIM, const int BLK_M, const int BLK_N, 
         const int THR_M, const int THR_N, const int CONJA>
static __device__ 
void hemm_template_device_ru(
    int M, int N, 
    const T* __restrict__ A, int LDA,
    const T* __restrict__ B, int LDB,
    T*       __restrict__ C, int LDC,
    T alpha, T beta )
{
    const int tx = threadIdx.x;  // thread's m dimension
    const int ty = threadIdx.y;  // thread's n dimension

    const int bx = blockIdx.x;   // block's m dimension
    const int by = blockIdx.y;   // block's n dimension

    __shared__ T sA[BLK_N][BLK_N+1];
    __shared__ T sB[BLK_N][BLK_M+1];

    // Registers for the innermost loop
    T rC[THR_N][THR_M];
    T tmp; 
    
    const T *offs_dA = A + by*BLK_N*LDA + ty*LDA + tx;
    ptrdiff_t boundA = (LDA*(N-1) + N) - ( by*BLK_N*LDA  + ty*LDA + tx ) -1;
    
    const T *offs_dB = B + bx*BLK_M + ty*LDB + tx;
    ptrdiff_t boundB = (LDB*(N-1) + M) - ( bx*BLK_M + ty*LDB + tx ) -1;
    
    // Zero C
    #pragma unroll
    for (int n = 0; n < THR_N; n++)
        #pragma unroll
        for (int m = 0; m < THR_M; m++)
            rC[n][m] = make_FloatingPoint(0.0, 0.0);
    
    const int part_1 = BLK_N * by;
    const int part_2 = min(BLK_N, N - part_1);
    const int part_3 = N - ( part_1 + part_2 ); 
    int kk;
    
    // part 1
    for (kk = 0; kk < part_1; kk += BLK_N){
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_N; m += DIM){ 
                sA[n+ty][m+tx] = fetch(A, m, n, boundA);
            }
        
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
        __syncthreads();

        offs_dA += BLK_N;
        boundA  -= BLK_N;

        offs_dB += BLK_N*LDB;
        boundB  -= BLK_N*LDB;
        
        // Multiply
        #pragma unroll
        for (int k = 0; k < BLK_N; k++)
        {
            #pragma unroll
            for (int n = 0; n < THR_N; n++) {
                #pragma unroll
                for (int m = 0; m < THR_M; m++) {
                    fma( sB[k][m*DIM+tx], sA[n*DIM+ty][k], rC[n][m] );
                }
            }
        }
        __syncthreads();
    }
    
    //part 2
    if(part_2 > 0){
        // read diagonal A block
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_N; m += DIM)
                sA[n+ty][m+tx] = fetch(A, m, n, boundA);
        __syncthreads();
        
        // read B block
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
        // mirror A block - copy upper to lower
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM){
            if(ty < tx){
                sA[n+ty][n+tx] = OP<T, CONJA>( sA[n+tx][n+ty] );
            }else if(ty == tx){
                sA[n+ty][n+tx] = make_FloatingPoint( real(sA[n+ty][n+tx]), 0.0 );
            }
            #pragma unroll
            for (int m = n+DIM; m < BLK_N; m += DIM)
                sA[n+ty][m+tx] = OP<T, CONJA>( sA[m+tx][n+ty] );
        }
        
        // advance pointers 
        offs_dA += BLK_N*LDA;
        boundA  -= BLK_N*LDA;
        
        offs_dB += BLK_N*LDB;
        boundB  -= BLK_N*LDB;
        
        __syncthreads();
        // Multiply - account for irregular sizes
        if(part_2 < BLK_N){
            #pragma unroll
            for (int k = 0; k < part_2; k++){
                #pragma unroll
                for (int n = 0; n < THR_N; n++) {
                    #pragma unroll
                    for (int m = 0; m < THR_M; m++) {
                        fma( sB[k][m*DIM+tx], sA[n*DIM+ty][k], rC[n][m] );
                    }
                }
            }
        }else{
            #pragma unroll
            for (int k = 0; k < BLK_N; k++){
                #pragma unroll
                for (int n = 0; n < THR_N; n++) {
                    #pragma unroll
                    for (int m = 0; m < THR_M; m++) {
                        fma( sB[k][m*DIM+tx], sA[n*DIM+ty][k], rC[n][m] );
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // part three
    for (kk = 0; kk < part_3-BLK_N; kk += BLK_N){
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_N; m += DIM){
                tmp = fetch(A, m, n, boundA); 
                sA[m+tx][n+ty] = OP<T, CONJA>( tmp );
            }
        
        #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_M; m += DIM)
                sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
        __syncthreads();

        offs_dA += BLK_N*LDA;
        boundA  -= BLK_N*LDA;
        
        offs_dB += BLK_N*LDB;
        boundB  -= BLK_N*LDB;
        
        // Multiply
        #pragma unroll
        for (int k = 0; k < BLK_N; k++)
        {
            #pragma unroll
            for (int n = 0; n < THR_N; n++) {
                #pragma unroll
                for (int m = 0; m < THR_M; m++) {
                    fma( sB[k][m*DIM+tx], sA[n*DIM+ty][k], rC[n][m] );
                }
            }
        }
        __syncthreads();
    }
    
    // Multiply last full (BLK_N) or partial block
    kk = part_3 - kk;
    #pragma unroll
        for (int n = 0; n < BLK_N; n += DIM)
            #pragma unroll
            for (int m = 0; m < BLK_N; m += DIM){
                tmp = fetch(A, m, n, boundA); 
                sA[m+tx][n+ty] = OP<T, CONJA>( tmp );
            }

    #pragma unroll
    for (int n = 0; n < BLK_N; n += DIM)
        #pragma unroll
        for (int m = 0; m < BLK_M; m += DIM)
            sB[n+ty][m+tx] = fetch(B, m, n, boundB);
    
    __syncthreads();
    // Multiply
    #pragma unroll
    for (int k = 0; k < kk; k++){
        #pragma unroll
        for (int n = 0; n < THR_N; n++) {
            #pragma unroll
            for (int m = 0; m < THR_M; m++) {
                fma( sB[k][m*DIM+tx], sA[n*DIM+ty][k], rC[n][m] );
            }
        }
    }

    // Store C regs->dev
    #pragma unroll
    for (int n = 0; n < THR_N; n++) {
        int coord_dCn = by*BLK_N + n*DIM + ty;
        #pragma unroll
        for (int m = 0; m < THR_M; m++) {
            int coord_dCm = bx*BLK_M + m*DIM + tx;
            if (coord_dCm < M && coord_dCn < N) {
                int offsC = coord_dCn*LDC + coord_dCm;

                T &regC = rC[n][m];
                T &memC = C[offsC];

                memC = add(mul(alpha, regC), mul(beta, memC));
            }
        }
    }
}

/******************************************************************************/
#endif //HEMM_TEMPLATE_DEVICE_CUH
