/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

////////////////////////////////////////////////////////////////////////////////
#define SLDA(n)              ( (((n)+1)%4) == 0 ? (n) : (n+1) )
#define sA(i,j)              sA[(j) * slda + (i)]
#define NTCOL(M)             ((M > 32) ? 1 : 2)
#define _TPC_                (16)

////////////////////////////////////////////////////////////////////////////////
//             For sm kernel
////////////////////////////////////////////////////////////////////////////////
static __device__ __inline__
void zgeqr2_compute_vtA_device(
        int &m, int &n, int &j,
        magmaDoubleComplex *sA, const int &slda,
        magmaDoubleComplex *sY, magmaDoubleComplex &tau,
        magmaDoubleComplex *sTmp,
        const int &tx, const int &ntx)
{
    magmaDoubleComplex zsum = MAGMA_Z_ZERO;

    const int ncols= n-j-1;
    const int tpc  = ntx / ncols; // threads-per-column
    const int nath = ncols * tpc; // # active threads
    const int tx_  = tx % tpc;
    const int ty_  = tx / tpc;
    sTmp += tpc * ty_;
    if( tx < nath ) {
        for(int i = tx_+j; i < m; i+=tpc) {
            zsum += MAGMA_Z_CONJ( sA(i,j) ) * sA(i,ty_+j+1);
        }
        sTmp[ tx_ ] = zsum;
    }
    __syncthreads();
    // reduce
    if( tx < nath && tx_ == 0) {
        zsum = MAGMA_Z_ZERO;
        for(int i = 0; i < tpc; i++) {
            zsum += sTmp[i];
        }

        sY[ty_+j+1] = zsum * MAGMA_Z_CONJ( tau );; // sTmp differs based on the value of ty_;
    }
}

////////////////////////////////////////////////////////////////////////////////
//             For sm kernel
////////////////////////////////////////////////////////////////////////////////
static __device__ __inline__
void zgeqr2_compute_norm(
        int n,
        magmaDoubleComplex* x, double* dx,
        const int &tx, const int &ntx)
{
    double sum = MAGMA_D_ZERO;
    for(int itx = tx; itx < n; itx+=ntx) {
        sum += MAGMA_Z_REAL( x[itx] ) * MAGMA_Z_REAL( x[itx] ) +
               MAGMA_Z_IMAG( x[itx] ) * MAGMA_Z_IMAG( x[itx] ) ;
    }
    dx[ tx ] = sum;
    // there is a sync at the beginning & end of magma_sum_reduce_n
    __syncthreads();
    // at this point the length of dx is <= ntx (which is 1024 max.)
    if ( ntx >  512 ) { if ( tx <  512 && tx +  512 < ntx ) { dx[tx] += dx[tx+ 512]; }  __syncthreads(); }
    if ( ntx >  256 ) { if ( tx <  256 && tx +  256 < ntx ) { dx[tx] += dx[tx+ 256]; }  __syncthreads(); }
    if ( ntx >  128 ) { if ( tx <  128 && tx +  128 < ntx ) { dx[tx] += dx[tx+ 128]; }  __syncthreads(); }
    if ( ntx >   64 ) { if ( tx <   64 && tx +   64 < ntx ) { dx[tx] += dx[tx+  64]; }  __syncthreads(); }
    // continue with serial sum
    sum = MAGMA_D_ZERO;
    if( tx == 0 ) {
        for( int i = 0; i < min(ntx,64); i++ ) {
            sum += dx[i];
        }
        dx[0] = sum;
    }
    __syncthreads();
}


////////////////////////////////////////////////////////////////////////////////
//             Reg. kernel
////////////////////////////////////////////////////////////////////////////////
template<int M32, int N>
__global__
__launch_bounds__(M32*NTCOL(M32))
void
zgeqr2_fused_reg_kernel_batched(
    int m,
    magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t lwork_tmp, magma_int_t *info_array, magma_int_t check_launch_only, magma_int_t batchCount )
{
    extern __shared__ magmaDoubleComplex zdata[];

    // if check_launch_only = 1, then return immediately
    // this is only to check if the kernel has been launched
    // successfully
    if(check_launch_only == 1) return;

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int nty = blockDim.y;
    const int batchid = blockIdx.x * nty + ty;
    if(batchid >= batchCount) return;

    magmaDoubleComplex* dA   = dA_array[batchid] + Aj * ldda + Ai;
    magmaDoubleComplex* dtau = dtau_array[batchid] + taui;
    magma_int_t* info        = &info_array[batchid];

    magmaDoubleComplex rA[N] = {MAGMA_Z_ZERO};
    const int slda = SLDA(M32);
    const int sldt = SLDA(_TPC_);

    // shared memory pointers
    magmaDoubleComplex* sA    = (magmaDoubleComplex*)(zdata);
    magmaDoubleComplex* sY    = sA   + (nty * slda * N);
    magmaDoubleComplex* stau  = sY   + (nty * N);
    magmaDoubleComplex* sTmp  = stau + nty * N;
    sA    += ty * slda * N;
    sY    += ty * N;
    stau  += ty * N;
    sTmp  += ty * lwork_tmp;
    double* snorm = (double*) (sTmp); // must be set after offsetting w.r.t. ty

    magmaDoubleComplex alpha, tau, tmp = MAGMA_Z_ZERO, scale = MAGMA_Z_ZERO;
    double sum = MAGMA_D_ZERO, norm = MAGMA_D_ZERO, norm_no_alpha = MAGMA_D_ZERO, beta;
    int i = 0;

    if( tx == 0 ){
        (*info) = 0;
    }

    // init sA to zero
    #pragma unroll
    for(int j = 0; j < N; j++) {
        sA(tx,j) = MAGMA_Z_ZERO;
    }

    // init tau
    if(tx < N) {
        stau[tx] = MAGMA_Z_ZERO;
    }

    // read and prepare for the norm of the first column
    if( tx < m ) {
        rA[0] = dA[ 0 * ldda + tx ];
        #pragma unroll
        for(int j = 1; j < N; j++) {
            rA[j] = dA[ j * ldda + tx ];
        }
        sA(tx, 0) = rA[0];
        tmp       = (tx < 1) ? MAGMA_Z_ZERO : rA[0];    // exclude first element (alpha)
        sum += MAGMA_Z_REAL( tmp ) * MAGMA_Z_REAL( tmp ) +
               MAGMA_Z_IMAG( tmp ) * MAGMA_Z_IMAG( tmp ) ;
    }
    snorm[ tx ] = sum;
    __syncthreads();

    #pragma unroll
    for(int j = 0; j < N; j++) {
        alpha = sA(j,j);

        // compute the norm of the current column
        {
            // we assume that this kernel is launch for m <= 1024
            if ( M32 >  512 ) { if ( tx <  512 && tx +  512 < M32 ) { snorm[tx] += snorm[tx+ 512]; }  __syncthreads(); }
            if ( M32 >  256 ) { if ( tx <  256 && tx +  256 < M32 ) { snorm[tx] += snorm[tx+ 256]; }  __syncthreads(); }
            if ( M32 >  128 ) { if ( tx <  128 && tx +  128 < M32 ) { snorm[tx] += snorm[tx+ 128]; }  __syncthreads(); }
            if ( M32 >   64 ) { if ( tx <   64 && tx +   64 < M32 ) { snorm[tx] += snorm[tx+  64]; }  __syncthreads(); }
            if ( M32 >   32 ) { if ( tx <   32 && tx +   32 < M32 ) { snorm[tx] += snorm[tx+  32]; }  __syncthreads(); }
            if ( M32 >   16 ) { if ( tx <   16 && tx +   16 < M32 ) { snorm[tx] += snorm[tx+  16]; }  __syncthreads(); }
            if ( M32 >    8 ) { if ( tx <    8 && tx +    8 < M32 ) { snorm[tx] += snorm[tx+   8]; }  __syncthreads(); }

            // continue with serial sum
            sum = MAGMA_D_ZERO;
            if( tx == 0 ) {
                #pragma unroll
                for( i = 0; i < 8; i++ ) {
                    sum += snorm[i];
                }
                #if 0
                snorm[0] = sqrt( sum );
                sum      = sum - (MAGMA_Z_REAL(alpha) * MAGMA_Z_REAL(alpha)) - (MAGMA_Z_IMAG(alpha) * MAGMA_Z_IMAG(alpha));
                snorm[1] = sum;
                #else
                snorm[1] = sum;  // store norm^2 without alpha
                sum     += (MAGMA_Z_REAL(alpha) * MAGMA_Z_REAL(alpha)) + (MAGMA_Z_IMAG(alpha) * MAGMA_Z_IMAG(alpha));
                snorm[0] = sqrt( sum );
                #endif
            }
            __syncthreads();
        } // end of computing the norm

        norm          = snorm[0];
        norm_no_alpha = snorm[1];
        bool zero_nrm = (norm_no_alpha == 0) && (MAGMA_Z_IMAG(alpha) == 0);

        tmp   = rA[j];
        tau   = MAGMA_Z_ZERO;
        scale = MAGMA_Z_ONE;
        if(!zero_nrm ) {
            beta  = -copysign(norm, real(alpha));
            scale = (tx > j) ? MAGMA_Z_DIV( MAGMA_Z_ONE,  alpha - MAGMA_Z_MAKE(beta, 0)) : scale;
            tau   = MAGMA_Z_MAKE( (beta - real(alpha))  / beta, -imag(alpha) / beta );
            rA[j] *= scale;
            tmp    = (tx == j) ? MAGMA_Z_MAKE(beta, MAGMA_D_ZERO) : rA[j];
            if(tx == j) {
                stau[j] = tau;
                rA[j]   = MAGMA_Z_ONE;
            }
        }

        rA[j]  = (tx == i) ? MAGMA_Z_ONE  : rA[j];
        rA[j]  = (tx <  j) ? MAGMA_Z_ZERO : rA[j];

        // write the column into global memory
        if( tx < m ) {
            dA[j * ldda + tx] = tmp;
        }

        // now compute (I - tau * v * v') A
        // first: y = tau * v' * A (row vector)
        {
            #pragma unroll
            for(int jj = j+1; jj < N; jj++) {
                sA(tx, jj) = MAGMA_Z_CONJ( rA[j] ) * rA[jj];
            }
            __syncthreads();

            const int NCOLS = N-j-1;
            const int TPC   = _TPC_;
            const int NGRP  = M32/ TPC;
            const int tx_   = tx % TPC;
            const int ty_   = tx / TPC;

            magmaDoubleComplex zsum = MAGMA_Z_ZERO;
            magmaDoubleComplex* sT  = sTmp;

            int ig = 0;

            #pragma unroll
            for(ig = 0; ig < NCOLS-NGRP; ig+=NGRP) {
                zsum  = MAGMA_Z_ZERO;
                sT = sTmp + (ty_+ig+j+1) * sldt;
                #pragma unroll
                for(i = 0; i < M32; i+=TPC) {
                    zsum += sA(tx_+i,ty_+ig+j+1);
                }
                sT[tx_] = zsum;
            }

            if(ty_ < NCOLS-ig) {
                zsum  = MAGMA_Z_ZERO;
                sT = sTmp + (ty_+ig+j+1) * sldt;
                #pragma unroll
                for(i = 0; i < M32; i+=TPC) {
                    zsum += sA(tx_+i,ty_+ig+j+1);
                }
                sT[tx_] = zsum;
            }
            __syncthreads();

            // reduce
            if( tx < NCOLS ) {
                zsum = MAGMA_Z_ZERO;
                sT = sTmp + (tx+j+1) * sldt;
                #pragma unroll
                for(int i = 0; i < TPC; i++) {
                    zsum += sT[i];
                }
                sY[tx+j+1] = zsum * MAGMA_Z_CONJ( tau );
            }
        }
        __syncthreads();

        // now compute: A = A - v * y
        {
            // compute the next column and prepare for the norm
            if(j < N-1) {
                rA[j+1]    -= rA[j] * sY[j+1];
                sA(tx,j+1)  = rA[j+1]; // for alpha next iteration
                tmp         = (tx < (j+2)) ? MAGMA_Z_ZERO : rA[j+1];    // exclude first element (alpha)
                snorm[ tx ] = MAGMA_Z_REAL(tmp) * MAGMA_Z_REAL(tmp) +
                              MAGMA_Z_IMAG(tmp) * MAGMA_Z_IMAG(tmp) ;
            }

            // the rest of the columns
            #pragma unroll
            for(int jj = j+2; jj < N; jj++) {
                rA[jj] -= rA[j] * sY[jj];
            }
        }
        __syncthreads();
    }

    // write tau
    if(tx < N) {
        dtau[tx] = stau[tx];
    }
}

////////////////////////////////////////////////////////////////////////////////
//             Reg. kernel driver
////////////////////////////////////////////////////////////////////////////////
template<int M32, int N>
static magma_int_t
magma_zgeqr2_fused_reg_kernel_driver_batched(
    magma_int_t m,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t* info_array, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;
    magma_int_t nthreads = M32;
    const magma_int_t ntcol = NTCOL(M32);

    magma_int_t lwork_norm = magma_roundup( M32 * sizeof(double), sizeof(magmaDoubleComplex));
    magma_int_t lwork_vtA  = (N) * SLDA(_TPC_);
    magma_int_t lwork_tmp  = max(lwork_norm, lwork_vtA);

    magma_int_t shmem = 0;
    shmem += SLDA(M32) * N * sizeof(magmaDoubleComplex);  // sA
    shmem += N             * sizeof(magmaDoubleComplex);  // sY
    shmem += N             * sizeof(magmaDoubleComplex);  // stau
    shmem += lwork_tmp     * sizeof(magmaDoubleComplex);  // for norm and w = v' * A
    shmem *= ntcol;
    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    dim3 grid(gridx, 1, 1);
    dim3 threads( nthreads, ntcol, 1);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, shmem_max = 0;
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zgeqr2_fused_reg_kernel_batched<M32, N>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
    }
    #else
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlock, device);
    #endif    // CUDA_VERSION >= 9000

    magma_int_t total_threads = nthreads * ntcol;
    if ( total_threads > nthreads_max || shmem > shmem_max ) {
        //printf("error: kernel %s requires too many threads or too much shared memory\n", __func__);
        arginfo = -100;
        return arginfo;
    }

    //if(check_launch_only == 1) return arginfo;
    void *kernel_args[] = {&m, &dA_array, &Ai, &Aj, &ldda, &dtau_array, &taui, &lwork_tmp, &info_array, &check_launch_only, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)zgeqr2_fused_reg_kernel_batched<M32, N>, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
//             Reg. kernel driver instantiation based on n
////////////////////////////////////////////////////////////////////////////////
// instantiates the kernel driver based on n
template<int M32>
static magma_int_t
magma_zgeqr2_fused_reg_N_batched(
    magma_int_t m, magma_int_t n,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t* info_array, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    switch(n) {
        case  1: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32, 1>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case  2: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32, 2>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case  3: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32, 3>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case  4: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32, 4>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case  5: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32, 5>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case  6: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32, 6>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case  7: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32, 7>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case  8: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32, 8>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        #if defined(MAGMA_HAVE_CUDA) && !defined(PRECISION_z)
        case  9: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32, 9>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 10: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32,10>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 11: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32,11>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 12: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32,12>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 13: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32,13>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 14: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32,14>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 15: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32,15>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        case 16: arginfo = magma_zgeqr2_fused_reg_kernel_driver_batched<M32,16>( m, dA_array, Ai, Aj, ldda, dtau_array, taui, info_array, check_launch_only, batchCount, queue ); break;
        #endif
        default: arginfo = -100;
    }
    return arginfo;
}
