/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
       @author Azzam Haidar

       @precisions normal z -> s d c
*/

#include <cuda.h>    // for CUDA_VERSION
#include "magma_internal.h"
#include "magma_templates.h"
#include "batched_kernel_param.h"

#define PRECISION_z

////////////////////////////////////////////////////////////////////////////////
#define SLDA(n)              ( (((n)+1)%4) == 0 ? (n) : (n+1) )
#define sA(i,j)               sA[(j) * slda + (i)]
#define sV(i,j)               sV[(j) * sldv + (i)]
#define sT(i,j)               sT[(j) * sldt + (i)]
#define MAX_THREADS          (256)

////////////////////////////////////////////////////////////////////////////////
template<int NB>
__global__
__launch_bounds__(MAX_THREADS)
void
zunm2r_sm_kernel_batched(
    magma_side_t side, magma_trans_t trans,
    int m, int n, int ib,
    magmaDoubleComplex **dA_array, int Ai, int Aj, int ldda,
    magmaDoubleComplex **dV_array, int Vi, int Vj, int lddv,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t batchCount )
{
    extern __shared__ magmaDoubleComplex zdata[];
    const int tx  = threadIdx.x;
    const int ty  = threadIdx.y;
    const int ntx = blockDim.x;
    const int nty = blockDim.y;
    const int tpc = ntx / NB; // a minimum of NB threads is required
    const int ty_ = tx / tpc;
    const int tx_ = tx % tpc;
    const int batchid = blockIdx.x * nty + ty;
    if(batchid >= batchCount) return;

    magmaDoubleComplex* dA   = dA_array[batchid] + Aj * ldda + Ai;
    magmaDoubleComplex* dV   = dV_array[batchid] + Vj * lddv + Vi;
    magmaDoubleComplex* dtau = dtau_array[batchid] + taui;

    const int slda = SLDA(m);
    const int sldv = SLDA(m);
    const int sldt = SLDA(tpc);

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
    int iib;

    // init sA,sV to zero
    for(int i = tx; i < m; i+=ntx) {
        #pragma unroll
        for(int j = 0; j < NB; j++) {
            sA(i,j) = MAGMA_Z_ZERO;
            sV(i,j) = MAGMA_Z_ZERO;
        }
    }

    // read tau and init diag(sV)
    // if trans == MagmaNoTrans, we read V and tau in reverse order
    if(tx < ib) {
        int rtx   = (trans == MagmaNoTrans) ? NB-tx-1 : tx;
        sV(tx,rtx) = MAGMA_Z_ONE; // does not need a sync before it

        rtx = (trans == MagmaNoTrans) ? NB-ib+tx  : tx;
        stau[rtx] = (trans == MagmaNoTrans) ? dtau[ib-tx-1] : MAGMA_Z_CONJ( dtau[tx] );
    }

    // read into sV
    if(trans == MagmaNoTrans) {
        // first loop over NB checks against the diagonal
        for(int j = NB-1; j >= NB-ib; j--) {
            sV(tx,j) = (tx > NB-j-1) ? dV[(NB-j-1) * lddv + tx] : sV(tx,j);
        }

        // read rest of dV
        for(int i = tx+ntx; i < m; i+=ntx) {
            for(int j = NB-1; j >= NB-ib; j--) {
                sV(i,j) = dV[(NB-j-1) * lddv + i];
            }
        }
    }
    else {
        // first loop over NB checks against the diagonal
        for(int j = 0; j < ib; j++) {
            sV(tx,j) = (tx > j) ? dV[j * lddv + tx] : sV(tx,j);
        }

        // read rest of dV
        for(int i = tx+ntx; i < m; i+=ntx) {
            for(int j = 0; j < ib; j++) {
                sV(i,j) = dV[j * lddv + i];
            }
        }
    }
    // end of reading in SV

    //////////// main loop ////////////////
    for(iib = 0; iib < (n/NB)*NB; iib+=NB) {
        // read A
        for(int i = tx; i < m; i+=ntx) {
            #pragma unroll
            for(int j = 0; j < NB; j++) {
                sA(i, j) = dA[ j * ldda + i ];
            }
        }
        __syncthreads();

        // apply loop
        #pragma unroll
        for(int j = 0; j < NB; j++) {
            // compute v' * A and reduce (1-of-2)
            zsum = MAGMA_Z_ZERO;
            if(ty_ < NB) {
                for(int i = tx_; i < m; i+=tpc) {
                    zsum += sA(i,ty_) * MAGMA_Z_CONJ( sV(i,j) );
                }
                sT(tx_,ty_) = zsum;
            }
            __syncthreads();

            // reduce (2-of-2)
            zsum = MAGMA_Z_ZERO;
            if(tx < NB) {
                for(int i = 0; i < tpc; i++) {
                    zsum += sT(i,tx);
                }
                sT(0,tx) = stau[j] * zsum;
            }
            __syncthreads();

            // rank update
            for(int i = tx; i < m; i+=ntx) {
                #pragma unroll
                for(int jj = 0; jj < NB; jj++) {
                    sA(i,jj) -= sV(i,j) * sT(0,jj);
                }
            }
            __syncthreads();
        }    // end of apply loop

        // write sA
        for(int i = tx; i < m; i+=ntx) {
            #pragma unroll
            for(int j = 0; j < NB; j++) {
                dA[ j * ldda + i ] = sA(i, j);
            }
        }

        // advance dA
        dA += NB*ldda;
    }    // end of main loop

    //////////// cleanup section ////////////////
    if(n - iib > 0) {
        int nn = n - iib;
        // read A
        for(int i = tx; i < m; i+= ntx) {
            for(int j = 0; j < nn; j++) {
                sA(i,j) = dA[ j * ldda + i ];
            }
        }
        __syncthreads();

        // apply loop
        #pragma unroll
        for(int j = 0; j < NB; j++) {
            // reduce (1-of-2)
            zsum = MAGMA_Z_ZERO;
            if(ty_ < nn) {
                for(int i = tx_; i < m; i+=tpc) {
                    zsum += sA(i,ty_) * MAGMA_Z_CONJ( sV(i,j) );
                }

                sT(tx_,ty_) = zsum;
            }
            __syncthreads();

            // reduce (2-of-2)
            zsum = MAGMA_Z_ZERO;
            if(tx < nn) {
                for(int i = 0; i < tpc; i++) {
                    zsum += sT(i,tx);
                }
                sT(0,tx) = stau[j] * zsum;
            }
            __syncthreads();

            // rank update
            for(int i = tx; i < m; i+=ntx) {
                for(int jj = 0; jj < nn; jj++) {
                    sA(i,jj) -= sV(i,j) * sT(0,jj);
                }
            }
            __syncthreads();

        }    // end of apply loop

        // write rA
        for(int i = tx; i < m; i+=ntx) {
            for(int j = 0; j < nn; j++) {
                dA[ j * ldda + i ] = sA(i,j);
            }
        }
    }    // end of cleanup section

}

////////////////////////////////////////////////////////////////////////////////
// calculates the shared memory requirement of zunm2r_sm_kernel_batched
// (m, n) are the dimensions of the 'C' matrix, to be updated by the 'Q' matrix
// ib is the number of elementary reflectors defining 'Q'
// nb is the blocking size used internally by the kernel
//    - it can be considered the maximum value of ib, so (nb >= ib)
extern "C"
magma_int_t magma_zunm2r_batched_kernel_sm_size(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t ib, magma_int_t nb)
{
    magma_int_t nthreads = magma_get_zgeqr2_fused_sm_batched_nthreads(m, ib);
    magma_int_t ntcol    = max(1, 32/nthreads);
    magma_int_t TPC      = nthreads / nb;

    magma_int_t shmem = 0;
    shmem += SLDA(m)   * nb * sizeof(magmaDoubleComplex);  // sA
    shmem += SLDA(m)   * nb * sizeof(magmaDoubleComplex);  // sV
    shmem += SLDA(TPC) * nb * sizeof(magmaDoubleComplex);  // sT
    shmem += nb             * sizeof(magmaDoubleComplex);  // stau
    shmem *= ntcol;

    return shmem;
}

////////////////////////////////////////////////////////////////////////////////
template<int NB>
static magma_int_t
magma_zunm2r_sm_kernel_driver_batched(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t ib,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t nthreads, magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_device_t device;
    magma_getdevice( &device );
    magma_int_t arginfo = 0;
    const magma_int_t ntcol = max(1, 32/nthreads);

    magma_int_t shmem = magma_zunm2r_batched_kernel_sm_size(side, trans, m, n, ib, NB);
    magma_int_t gridx = magma_ceildiv(batchCount, ntcol);
    nthreads = min(nthreads, m);           // nthreads should not be greater than m
    nthreads = max(nthreads, NB);          // nthreads should not be less than NB
    nthreads = min(nthreads, MAX_THREADS); // nthreads should not be greater than MAX_THREADS

    dim3 grid(gridx, 1, 1);
    dim3 threads( nthreads, ntcol, 1);

    // get max. dynamic shared memory on the GPU
    int nthreads_max, shmem_max = 0;
    cudaDeviceGetAttribute (&nthreads_max, cudaDevAttrMaxThreadsPerBlock, device);
    #if CUDA_VERSION >= 9000
    cudaDeviceGetAttribute (&shmem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (shmem <= shmem_max) {
        cudaFuncSetAttribute(zunm2r_sm_kernel_batched<NB>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem);
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

    if( check_launch_only == 1 ) return arginfo;

    void *kernel_args[] = {&side, &trans, &m, &n, &ib, &dA_array, &Ai, &Aj, &ldda, &dV_array, &Vi, &Vj, &lddv, &dtau_array, &taui, &batchCount};
    cudaError_t e = cudaLaunchKernel((void*)zunm2r_sm_kernel_batched<NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());
    if( e != cudaSuccess ) {
        //printf("error in %s : failed to launch kernel %s\n", __func__, cudaGetErrorString(e));
        arginfo = -100;
    }

    return arginfo;
}

/***************************************************************************//**
    Purpose
    -------
    ZUNM2R overwrites the general complex m-by-n matrix C with

       Q * C    if SIDE = MagmaLeft and TRANS = 'N', or
       Q**H* C  if SIDE = MagmaLeft and TRANS = 'C', or

       C * Q    if SIDE = MagmaRight and TRANS = 'N', or
       C * Q**H if SIDE = MagmaRight and TRANS = 'C',

   where Q is a complex unitary matrix defined as the product of k
   elementary reflectors

       Q = H(1) H(2) . . . H(k)

   as returned by ZGEQRF. Q is of order m if SIDE = MagmaLeft and of order n
   if SIDE = MagmaRight.

  - This is an internal batch implementation of ZUNM2R
  - The implementation uses shared memory blocking
  - Only SIDE = MagmaLeft is currently supported

    @ingroup magma_unmqr
*******************************************************************************/
extern "C"
magma_int_t
magma_zunm2r_sm_batched(
    magma_side_t side, magma_trans_t trans,
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t k,
    magmaDoubleComplex** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t ldda,
    magmaDoubleComplex** dV_array, magma_int_t Vi, magma_int_t Vj, magma_int_t lddv,
    magmaDoubleComplex **dtau_array, magma_int_t taui,
    magma_int_t check_launch_only,
    magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t arginfo = 0;
    magma_int_t m32 = magma_roundup(m, 32);

    if (side != MagmaLeft) {
        printf("%s currently supports side = MagmaLeft only\n", __func__);
        arginfo = -1;
    } else if (trans != MagmaNoTrans && trans != Magma_ConjTrans) {
        arginfo = -2;
    } else if (m < 0 || m32 < nb) {
        arginfo = -3;
    } else if (n < 0) {
        arginfo = -4;
    } else if (nb <= 0) {
        arginfo = -5;
    } else if (k < 0) {
        arginfo = -6;
    } else if (ldda < max(1,m)) {
        arginfo = -10;
    } else if (lddv < max(1,m)) {
        arginfo = -14;
    } else if (batchCount < 0) {
        arginfo = -18;
    }

    if (arginfo != 0) {
        magma_xerbla( __func__, -(arginfo) );
        return arginfo;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0 || k == 0 || batchCount == 0)
        return arginfo;

    magma_int_t istart, istop, istep;
    if( (side == MagmaLeft  && trans != MagmaNoTrans) ||
        (side == MagmaRight && trans == MagmaNoTrans) ) {
        istart = 0;
        istop  = k;
        istep  = nb;
    }
    else {
        istart = magma_roundup(k-1,nb);
        istop  = 0;
        istep  = -nb;
    }

    magma_int_t locinfo = 0;
    for(magma_int_t i = istart; (istep < 0 ? i >= istop : i < istop); i+=istep) {
        magma_int_t ib       = min(nb, k-i);
        magma_int_t nthreads = magma_get_zgeqr2_fused_sm_batched_nthreads(m-i, ib);
        switch(nb) {
            case 1: locinfo = magma_zunm2r_sm_kernel_driver_batched<1>( side, trans, m-i, n, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, nthreads, check_launch_only, batchCount, queue ); break;
            case 2: locinfo = magma_zunm2r_sm_kernel_driver_batched<2>( side, trans, m-i, n, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, nthreads, check_launch_only, batchCount, queue ); break;
            case 4: locinfo = magma_zunm2r_sm_kernel_driver_batched<4>( side, trans, m-i, n, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, nthreads, check_launch_only, batchCount, queue ); break;
            case 8: locinfo = magma_zunm2r_sm_kernel_driver_batched<8>( side, trans, m-i, n, ib, dA_array, Ai+i, Aj, ldda, dV_array, Vi+i, Vj+i, lddv, dtau_array, taui+i, nthreads, check_launch_only, batchCount, queue ); break;
            default: locinfo = -5; // unsupported nb
        }

        // accumulate locinfo (useful in checking launch only)
        // if arginfo = 0 at the end, it means that check launch was successful at every iteration
        arginfo += locinfo;
    }

    return arginfo;
}
