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
#define sA(i,j)               sA[(j) * slda + (i)]
#define sV(i,j)               sV[(j) * sldv + (i)]
#define sT(i,j)               sT[(j) * sldt + (i)]
#define NTCOL(M)             ((M > 32) ? 1 : 2)

////////////////////////////////////////////////////////////////////////////////
//              ZLARF fused register kernel
////////////////////////////////////////////////////////////////////////////////
template<int M32, int NB, int TPC>
__global__
__launch_bounds__(M32*NTCOL(M32))
void
zunm2r_reg_kernel_batched(
    magma_side_t side, magma_trans_t trans,
    int m, int n, int ib,
    magmaDoubleComplex **dA_array, int Ai, int Aj, int ldda,
    magmaDoubleComplex **dV_array, int Vi, int Vj, int lddv,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only, magma_int_t batchCount )
{
    extern __shared__ magmaDoubleComplex zdata[];

    // if check_launch_only = 1, then return immediately
    // this is only to check if the kernel has been launched
    // successfully
    if(check_launch_only == 1) return;

    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int ty_ = tx / TPC;
    const int tx_ = tx % TPC;
    const int nty = blockDim.y;
    const int batchid = blockIdx.x * nty + ty;
    if(batchid >= batchCount) return;

    magmaDoubleComplex* dA   = dA_array[batchid] + Aj * ldda + Ai;
    magmaDoubleComplex* dV   = dV_array[batchid] + Vj * lddv + Vi;
    magmaDoubleComplex* dtau = dtau_array[batchid] + taui;

    magmaDoubleComplex rA[NB] = {MAGMA_Z_ZERO};
    const int slda = SLDA(M32);
    const int sldv = SLDA(M32);
    const int sldt = SLDA(TPC);

    // shared memory pointers
    magmaDoubleComplex* sV   = (magmaDoubleComplex*)(zdata);
    magmaDoubleComplex* sA   = sV + (nty * sldv * NB);
    magmaDoubleComplex* sT   = sA + (nty * slda * NB);
    magmaDoubleComplex* stau = sT + (nty * sldt * NB);
    sV    += ty * sldv * NB;
    sA    += ty * slda * NB;
    sT    += ty * sldt * NB;
    stau  += ty * NB;

    magmaDoubleComplex zsum;
    int i, iib;

    // init sA,sV to zero
    #pragma unroll
    for(int j = 0; j < NB; j++) {
        sA(tx,j) = MAGMA_Z_ZERO;
        sV(tx,j) = MAGMA_Z_ZERO;
    }

    // read tau and init diag(sV)
    // if trans == MagmaNoTrans, we read V and tau in reverse order
    if(tx < ib) {
        int rtx    = (trans == MagmaNoTrans) ? NB-tx-1 : tx;
        sV(tx,rtx) = MAGMA_Z_ONE; // does not need a sync before it

        rtx = (trans == MagmaNoTrans) ? NB-ib+tx  : tx;
        stau[rtx] = (trans == MagmaNoTrans) ? dtau[ib-tx-1] : MAGMA_Z_CONJ( dtau[tx] );
    }

    // read into sV
    if( tx < m ) {
        if(trans == MagmaNoTrans) {
            for(int j = NB-1; j >= NB-ib; j--) {
                sV(tx,j) = (tx > NB-j-1) ? dV[(NB-j-1) * lddv + tx] : sV(tx,j);
            }
        }
        else {
            for(int j = 0; j < ib; j++) {
                sV(tx,j) = (tx > j) ? dV[j * lddv + tx] : sV(tx,j);
            }
        }
    }

    //////////// main loop ////////////////
    for(iib = 0; iib < (n/NB)*NB; iib+=NB) {
        // read A
        if(tx < m) {
            #pragma unroll
            for(int j = 0; j < NB; j++) {
                rA[j]    = dA[ j * ldda + tx ];
            }
        }

        // apply loop
        #pragma unroll
        for(int j = 0; j < NB; j++) {
            // compute v' * rA -> sA
            #pragma unroll
            for(int jj = 0; jj < NB; jj++) {
                sA(tx,jj) = MAGMA_Z_CONJ( sV(tx,j) ) * rA[jj];
            }
            __syncthreads();

            // reduce (1-of-2)
            zsum = MAGMA_Z_ZERO;
            if(ty_ < NB) {
                #pragma unroll
                for(i = 0; i < M32-TPC; i+=TPC) {
                    zsum += sA(tx_+i,ty_);
                }

                if(tx_ < M32-i) {
                    zsum += sA(tx_+i,ty_);
                }

                sT(tx_,ty_) = zsum;
            }
            __syncthreads();

            // reduce (2-of-2)
            zsum = MAGMA_Z_ZERO;
            if(tx < NB) {
                #pragma unroll
                for(i = 0; i < TPC; i++) {
                    zsum += sT(i,tx);
                }
                sT(0,tx) = stau[j] * zsum;
            }
            __syncthreads();

            // rank update
            #pragma unroll
            for(int jj = 0; jj < NB; jj++) {
                rA[jj] -= sV(tx,j) * sT(0,jj);
            }
        }    // end of apply loop

        // write rA
        if(tx < m) {
            #pragma unroll
            for(int j = 0; j < NB; j++) {
                dA[ j * ldda + tx ] = rA[j];
            }
        }

        // advance dA
        dA += NB*ldda;
    }    // end of main loop

    //////////// cleanup section ////////////////
    if(n - iib > 0) {
        int nn = n - iib;
        // read A
        if(tx < m) {
            for(int j = 0; j < nn; j++) {
                sA(tx,j) = dA[ j * ldda + tx ];
            }
        }
        __syncthreads();

        // apply loop
        #pragma unroll
        for(int j = 0; j < NB; j++) {
            // reduce (1-of-2)
            zsum = MAGMA_Z_ZERO;
            if(ty_ < nn) {
                #pragma unroll
                for(i = 0; i < M32-TPC; i+=TPC) {
                    zsum += sA(tx_+i,ty_) * MAGMA_Z_CONJ( sV(tx_+i,j) );
                }

                if(tx_ < M32-i) {
                    zsum += sA(tx_+i,ty_) * MAGMA_Z_CONJ( sV(tx_+i,j) );
                }

                sT(tx_,ty_) = zsum;
            }
            __syncthreads();

            // reduce (2-of-2)
            zsum = MAGMA_Z_ZERO;
            if(tx < nn) {
                #pragma unroll
                for(i = 0; i < TPC; i++) {
                    zsum += sT(i,tx);
                }
                sT(0,tx) = stau[j] * zsum;
            }
            __syncthreads();

            // rank update
            for(int jj = 0; jj < nn; jj++) {
                sA(tx,jj) -= sV(tx,j) * sT(0,jj);
            }
            __syncthreads();

        }    // end of apply loop

        // write rA
        if(tx < m) {
            for(int j = 0; j < nn; j++) {
                dA[ j * ldda + tx ] = sA(tx,j);
            }
        }
    }    // end of cleanup section

}

////////////////////////////////////////////////////////////////////////////////
//              ZLARF fused register kernel driver
////////////////////////////////////////////////////////////////////////////////
template<int M32, int NB>
static magma_int_t
magma_zunm2r_reg_kernel_driver_batched(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t ib,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;
    magma_int_t nthreads = M32;
    const magma_int_t ntcol = NTCOL(M32);
    const magma_int_t TPC   = M32 / NB;

    magma_int_t shmem = 0;
    shmem += SLDA(M32) * NB * sizeof(magmaDoubleComplex);  // sA
    shmem += SLDA(M32) * NB * sizeof(magmaDoubleComplex);  // sV
    shmem += SLDA(TPC) * NB * sizeof(magmaDoubleComplex);  // sT
    shmem += NB             * sizeof(magmaDoubleComplex);  // stau
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
        cudaFuncSetAttribute(zunm2r_reg_kernel_batched<M32, NB, TPC>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
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
    void *kernel_args[] = {&side, &trans, &m, &n, &ib, &dA_array, &Ai, &Aj, &ldda, &dV_array, &Vi, &Vj, &lddv, &dtau_array, &taui, &check_launch_only, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)zunm2r_reg_kernel_batched<M32, NB, TPC>, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
//              ZUNM2R register kernel driver
//              instantiations based on nb
//              ib <= nb (an outside loop breaks down ib if ib > nb)
////////////////////////////////////////////////////////////////////////////////
template<int M32>
static magma_int_t
magma_zunm2r_reg_NB_batched(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t ib,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    switch(nb) {
        case 1: arginfo = magma_zunm2r_reg_kernel_driver_batched<M32, 1>( side, trans, m, n, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 2: arginfo = magma_zunm2r_reg_kernel_driver_batched<M32, 2>( side, trans, m, n, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 4: arginfo = magma_zunm2r_reg_kernel_driver_batched<M32, 4>( side, trans, m, n, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        case 8: arginfo = magma_zunm2r_reg_kernel_driver_batched<M32, 8>( side, trans, m, n, ib, dA_array, Ai, Aj, ldda, dV_array, Vi, Vj, lddv, dtau_array, taui, check_launch_only, batchCount, queue ); break;
        default: arginfo = -100;
    }
    return arginfo;
}
