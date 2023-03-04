/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah

       @precisions normal z -> s d c
*/

#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"

#define PRECISION_z

#define GBTRS_SWAP_THREADS (128)

#define GBTRS_GERU_THREADS_X (32)
#define GBTRS_GERU_THREADS_Y (4)

#define GBTRS_UPPER_THREADS (128)

#ifdef PRECISION_z
#define GBTRS_LOWER_NB      (4)
#define GBTRS_UPPER_NB      (4)
#elif defined(PRECISION_s)
#define GBTRS_LOWER_NB      (16)
#define GBTRS_UPPER_NB      (16)
#else // d, c
#define GBTRS_LOWER_NB      (8)
#define GBTRS_UPPER_NB      (8)
#endif

//#define DBG

////////////////////////////////////////////////////////////////////////////////
template<typename T>
__device__ void print_memory(
                const char* msg,
                int m, int n, T* sA, int lda,
                int tx, int ty, int tz,
                int bx, int by, int bz)
{
#if defined(PRECISION_d) && defined(DBG)
    __syncthreads();
    if(threadIdx.x == tx && threadIdx.y == ty && threadIdx.z == tz &&
       blockIdx.x  == bx && blockIdx.y  == by && blockIdx.z  == bz) {
        printf("%s = [ \n", msg);
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                printf("%8.4f  ", (double)(sA[j*lda+i]));
            }
            printf("\n");
        }
        printf("]; \n");
    }
    __syncthreads();
#endif
}

////////////////////////////////////////////////////////////////////////////////
__global__
__launch_bounds__(GBTRS_SWAP_THREADS)
void zgbtrs_swap_kernel_batched(
        int n,
        magmaDoubleComplex** dA_array, int ldda,
        magma_int_t** dipiv_array, int j)
{
    const int ntx     = blockDim.x;
    const int tx      = threadIdx.x;
    const int batchid = blockIdx.x;

    magmaDoubleComplex* dA    = dA_array[batchid];
    magma_int_t*        dipiv = dipiv_array[batchid];

    int jp = dipiv[j] - 1; // undo fortran indexing
    if( j != jp ) {
        for(int i = tx; i < n; i+=ntx) {
            magmaDoubleComplex tmp = dA[i * ldda +  j];
            dA[i * ldda +  j]      = dA[i * ldda + jp];
            dA[i * ldda + jp]      = tmp;
        }
    }
}


////////////////////////////////////////////////////////////////////////////////
__global__
__launch_bounds__(GBTRS_GERU_THREADS_X*GBTRS_GERU_THREADS_Y)
void zgeru_kernel_batched(
        int m, int n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex** dX_array, int xi, int xj, int lddx, int incx,
        magmaDoubleComplex** dY_array, int yi, int yj, int lddy, int incy,
        magmaDoubleComplex** dA_array, int ai, int aj, int ldda )
{
    const int tx      = threadIdx.x;
    const int ty      = threadIdx.y;
    const int ntx     = blockDim.x;
    const int nty     = blockDim.y;
    const int gtx     = blockIdx.x * ntx + tx;
    const int batchid = blockIdx.z;

    magmaDoubleComplex* dX    = dX_array[batchid] + xj * lddx + xi;
    magmaDoubleComplex* dY    = dY_array[batchid] + yj * lddy + yi;
    magmaDoubleComplex* dA    = dA_array[batchid] + aj * ldda + ai;

    if(gtx < m) {
        for(int j = ty; j < n; j += nty) {
            dA[j * ldda + gtx] += alpha * dX[gtx * incx] * dY[j * incy];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
__global__
__launch_bounds__(GBTRS_UPPER_THREADS)
void zgbtrs_upper_columnwise_kernel_batched(
        int n, int kl, int ku, int nrhs, int j,
        magmaDoubleComplex** dA_array, int ldda,
        magmaDoubleComplex** dB_array, int lddb)
{
#define dA(i,j) dA[(j)*ldda + (i)]
#define dB(i,j) dB[(j)*lddb + (i)]

    const int kv      = kl + ku;
    const int tx      = threadIdx.x;
    const int ntx     = blockDim.x;
    const int batchid = blockIdx.x;
    //const int je      = (n-1) - j;

    magmaDoubleComplex* dA = dA_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];

    // advance dA/dB based on j
    dA += j * ldda + kv;
    dB += j;

    const int nupdates = min(kv, j);
    magmaDoubleComplex s;
    for(int rhs = 0; rhs < nrhs; rhs++) {
        s = dB(0,rhs) * MAGMA_Z_DIV(MAGMA_Z_ONE, dA(0,0));
        __syncthreads();

        if(tx == 0) dB(0,rhs) = s;
        for(int i = tx; i < nupdates ; i+= ntx) {
            dB(-i-1,rhs) -= s * dA(-i-1,0);
        }
    }

#undef dA
#undef dB
}

////////////////////////////////////////////////////////////////////////////////
template<int MAX_THREADS, int NB>
__global__
__launch_bounds__(MAX_THREADS)
void zgbtrs_lower_blocked_kernel_batched(
        int n, int kl, int ku, int nrhs, int nrhs_nb,
        magmaDoubleComplex** dA_array, int ldda, magma_int_t** dipiv_array,
        magmaDoubleComplex** dB_array, int lddb )
{
#define dB(i, j)  dB[(j)*lddb + (i)]
#define sB(i, j)  sB[(j)*sldb + (i)]

    extern __shared__ magmaDoubleComplex zdata[];
    const int kv      = kl + ku;
    const int tx      = threadIdx.x;
    const int ntx     = blockDim.x;
    const int bx      = blockIdx.x;
    const int by      = blockIdx.y;
    const int batchid = bx;
    const int my_rhs  = min(nrhs_nb, nrhs - by * nrhs_nb);
    const int sldb    = (NB+kl);

    magmaDoubleComplex* dA = dA_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];
    magma_int_t* dipiv     = dipiv_array[batchid];

    magmaDoubleComplex rA[NB] = {MAGMA_Z_ZERO};
    magmaDoubleComplex* sB    = (magmaDoubleComplex*)zdata;
    int* sipiv                = (int*)( sB + nrhs_nb * sldb );

    // advance dA and dB
    dA += kv+1;
    dB += by * nrhs_nb * lddb;

    int b_elements_1     = min(NB, n);

    for(int itx = tx; itx < b_elements_1; itx+=ntx) {
        for(int jb = 0; jb < my_rhs; jb++) {
            sB(itx, jb) = dB(itx, jb);
        }
    }

    for(int j = 0; j < n/*n1*/; j+=NB) {
        int nb = min(NB, n-j);
        // read A
        if(nb == NB) {
            #pragma unroll
            for(int ja = 0; ja < NB; ja++) {
                rA[ja] = dA[ja * ldda + tx];
            }
        }
        else{
            #pragma unroll
            for(int ja = 0; ja < NB; ja++) {
                rA[ja] = (ja < nb) ? dA[ja * ldda + tx] : MAGMA_Z_ZERO;
            }
        }


        // read pivot info
        for(int ip = tx; ip < nb; ip+=ntx) {
            sipiv[ip] = (int)( dipiv[ip] );
        }

        // read extra B elements to have a total of (nb + kl) elements
        int b_elements_2 = min(nb+kl-b_elements_1, n-j-b_elements_1);

        #ifdef DBG0
        __syncthreads();
        if(tx == 0) {
            printf("nb = %d, b1 = %d, b2 = %d\n", nb, b_elements_1, b_elements_2);
        }
        __syncthreads();
        #endif

        for(int itx = tx; itx < b_elements_2; itx+=ntx) {
            for(int jb = 0; jb < my_rhs; jb++) {
                sB(itx+b_elements_1, jb) = dB(itx+b_elements_1, jb);
            }
        }
        __syncthreads();

        //print_memory<magmaDoubleComplex>("sB1", sldb, my_rhs, sB, sldb, 0, 0, 0, 0, 0, 0);


        // swap & rank-1 update
        #pragma unroll
        for(int ja = 0; ja < NB; ja++) {
            // swap: note that the swap only affects the segment we read from B
            // since we always read extra KL elements
            int jp = sipiv[ja] - j - 1;
            if(ja < nb && jp != ja) {
                for(int jb = tx; jb < my_rhs; jb+=ntx) {
                    magmaDoubleComplex ztmp = sB(jp, jb);
                    sB(jp, jb)              = sB(ja, jb);
                    sB(ja, jb)              = ztmp;
                }
            }
            __syncthreads();

            //print_memory<magmaDoubleComplex>("tB1", sldb, my_rhs, sB, sldb, 0, 0, 0, 0, 0, 0);

            // apply
            for(int jb = 0; jb < my_rhs; jb++) {
                #if defined(DBG0) && defined(PRECISION_d)
                printf("[%d]: %.4f -= %.4f * %.4f\n", tx, sB(tx+ja+1, jb), rA[ja], sB(ja,jb));
                __syncthreads();
                #endif
                sB(tx+ja+1, jb) -= rA[ja] * sB(ja,jb);
            }
            __syncthreads();
            //print_memory<magmaDoubleComplex>("tB2", sldb, my_rhs, sB, sldb, 0, 0, 0, 0, 0, 0);

        } // end of swap & rank-1 updates

        // write part of B that is finished and shift the the rest up

        for(int itx = tx; itx < nb; itx+=ntx) {
            for(int jb = 0; jb < my_rhs; jb++) {
                dB(itx, jb) = sB(itx, jb);
            }
        }
        __syncthreads();

        //print_memory<magmaDoubleComplex>("sB2", sldb, my_rhs, sB, sldb, 0, 0, 0, 0, 0, 0);

        // shift up
        int shift_size = b_elements_1 + b_elements_2 - nb;
        for(int itx = tx; itx < shift_size; itx+=ntx) {
            for(int jb = 0; jb < my_rhs; jb++) {
                sB(itx, jb) = sB(itx+nb, jb);
            }
        }
        __syncthreads();
        //print_memory<magmaDoubleComplex>("sB3", sldb, my_rhs, sB, sldb, 0, 0, 0, 0, 0, 0);
        #ifdef DBG0
        __syncthreads();
        if(tx == 0) {
            printf("shift = %d\n", shift_size);
        }
        __syncthreads();
        #endif

        b_elements_1 = shift_size; /*b_elements_2*/;
        dA    += nb * ldda;
        dB    += nb;
        dipiv += nb;
    }
}

////////////////////////////////////////////////////////////////////////////////
template<int MAX_THREADS, int NB>
__global__
__launch_bounds__(MAX_THREADS)
void zgbtrs_upper_blocked_kernel_batched(
        int n, int kl, int ku, int nrhs, int nrhs_nb,
        magmaDoubleComplex** dA_array, int ldda,
        magmaDoubleComplex** dB_array, int lddb )
{
#define dA(i, j)  dA[(j)*ldda + (i)]
#define dB(i, j)  dB[(j)*lddb + (i)]
#define sB(i, j)  sB[(j)*sldb + (i)]
#define sBr(i, j) sBr[(j)*sldb + (i)]

    extern __shared__ magmaDoubleComplex zdata[];
    const int kv      = kl + ku;
    const int kb      = NB + kv;
    const int tx      = threadIdx.x;
    const int ntx     = blockDim.x;
    const int rtx     = ntx-1-tx;  // reverse tx
    const int bx      = blockIdx.x;
    const int by      = blockIdx.y;
    const int batchid = bx;
    const int my_rhs  = min(nrhs_nb, nrhs - by * nrhs_nb);
    const int sldb    = kb;

    magmaDoubleComplex* dA = dA_array[batchid];
    magmaDoubleComplex* dB = dB_array[batchid];

    magmaDoubleComplex rA[NB] = {MAGMA_Z_ZERO};
    magmaDoubleComplex* sB    = (magmaDoubleComplex*)zdata;
    magmaDoubleComplex* stmp  = sB + nrhs_nb * sldb;

    // advance dA, dB, sB
    dA += (n-1) * ldda + kv;             // backwards
    dB += (by * nrhs_nb * lddb) + (n-1); // backwards

    //#if defined(DBG) && defined(PRECISION_d)
    //printf("%.4f\n", dB[0]);
    //__syncthreads();
    for(int itx = tx; itx < kb; itx+=ntx) {sB[itx] = MAGMA_Z_ZERO;}
    __syncthreads();
    //#endif
    //print_memory<magmaDoubleComplex>("sB-1", sldb, my_rhs, sB, sldb, 0, 0, 0, 0, 0, 0);


    magmaDoubleComplex* sBr = sB + (kb-1);
    // we need (NB+kv) elements in one sweep
    int b_elements_1     = min(NB, n);
    for(int itx = rtx; itx < b_elements_1; itx+=ntx) {
        for(int jb = 0; jb < my_rhs; jb++) {
            sBr(-itx, jb) = dB(-itx, jb);
            #if defined(DBG) && defined(PRECISION_d)
            printf("[%d, %d]: sBr(-itx, jb) = %.4f\n", tx, rtx, sBr(-itx, jb));
            #endif
        }
    }
    __syncthreads();
    print_memory<magmaDoubleComplex>("sB0", sldb, my_rhs, sB, sldb, 0, 0, 0, 0, 0, 0);

    for(int fj = 0; fj < n; fj+=NB) {
        int nb = min(NB, n-fj);
        //int j  = (n-1) - fj;

        // read A
        if(nb == NB) {
            #pragma unroll
            for(int ja = 0; ja < NB; ja++) {
                rA[NB-1-ja] = dA(-rtx, -ja);
            }
        }
        else{
            #pragma unroll
            for(int ja = 0; ja < NB; ja++) {
                rA[NB-1-ja] = (ja < nb) ? dA(-rtx, -ja) : MAGMA_Z_ZERO;
            }
        }

        // read extra B elements to have a total of (nb + kl) elements
        int b_elements_2 = min(kb-b_elements_1, n-fj-b_elements_1);

        #ifdef DBG
        __syncthreads();
        if(tx == 0) {
            printf("nb = %d, b1 = %d, b2 = %d\n", nb, b_elements_1, b_elements_2);
        }
        __syncthreads();
        #endif

        for(int itx = rtx; itx < b_elements_2; itx+=ntx) {
            for(int jb = 0; jb < my_rhs; jb++) {
                sBr(-(b_elements_1+itx), jb) = dB(-(b_elements_1+itx), jb);
            }
        }
        __syncthreads();

        print_memory<magmaDoubleComplex>("sB1", sldb, my_rhs, sB, sldb, 0, 0, 0, 0, 0, 0);


        // apply block of A (divide + rank-1 updates)
        #pragma unroll
        for(int ja = NB-1; ja >= 0; ja--) {
            int jj = (NB-1) - ja;
            if(rtx == 0) {
                stmp[0] = MAGMA_Z_DIV(MAGMA_Z_ONE, rA[ja]);
            }
            __syncthreads();

            #if defined(DBG) && defined(PRECISION_d)
            __syncthreads();
            if(rtx == 0) {
                printf("[%d]: 1 / %.4f = %4f\n", tx, rA[ja], stmp[0]);
            }
            __syncthreads();
            #endif


            for(int jb = tx; jb < my_rhs; jb+=ntx) {
                #if defined(DBG) && defined(PRECISION_d)
                printf("[%d]: sB[%d] ( = %.4f ) * %.4f = \n", tx, kb-1-jj, sB(kb-1-jj, jb), stmp[0]);
                #endif
                sB(kb-1-jj, jb) *= stmp[0];
                //sB(ja, jb) *= stmp[0];
                #if defined(DBG) && defined(PRECISION_d)
                printf("[%d]: %.4f\n", tx, sB(kb-1-jj, jb));
                #endif
            }
            __syncthreads();

            // rank-1 update
            magmaDoubleComplex ztmp = (rtx == 0) ? MAGMA_Z_ZERO : rA[ja];
            for(int jb = 0; jb < my_rhs; jb++) {
                #if defined(DBG) && defined(PRECISION_d)
                __syncthreads();
                printf("[%d]: %.4f -= %4f * %.4f\n", tx, sBr(-jj-rtx, jb), ztmp, sB(kb-1-jj,jb));
                __syncthreads();
                #endif

                sBr(-jj-rtx, jb) -= ztmp * sB(kb-1-jj,jb);
                //sB(tx+ja+1, jb) -= rA[ja] * sB(ja,jb);
            }
            __syncthreads();
            print_memory<magmaDoubleComplex>("tB", sldb, my_rhs, sB, sldb, 0, 0, 0, 0, 0, 0);

        } // end of swap & rank-1 updates

        // write part of B that is finished and shift the the rest down

        for(int itx = rtx; itx < nb; itx+=ntx) {
            for(int jb = 0; jb < my_rhs; jb++) {
                dB(-itx, jb) = sBr(-itx, jb);
            }
        }
        __syncthreads();

        print_memory<magmaDoubleComplex>("sB2", sldb, my_rhs, sB, sldb, 0, 0, 0, 0, 0, 0);

        // shift down
        int shift_size = b_elements_1 + b_elements_2 - nb;
        for(int itx = rtx; itx < shift_size; itx+=ntx) {
            for(int jb = 0; jb < my_rhs; jb++) {
                sBr(-itx, jb) = sBr(-itx-nb, jb);
            }
        }
        __syncthreads();

        print_memory<magmaDoubleComplex>("sB3", sldb, my_rhs, sB, sldb, 0, 0, 0, 0, 0, 0);
        #ifdef DBG
        __syncthreads();
        if(tx == 0) {
            printf("shift = %d\n", shift_size);
        }
        __syncthreads();
        #endif

        b_elements_1 = shift_size; /*b_elements_2*/;
        dA    -= nb * ldda;
        dB    -= nb;
    }
}

////////////////////////////////////////////////////////////////////////////////
extern "C"
void magmablas_zgbtrs_swap_batched(
        magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t ldda,
        magma_int_t** dipiv_array, magma_int_t j,
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t nthreads = min(n, GBTRS_SWAP_THREADS);
    magma_int_t nblocks  = batchCount;

    dim3 grid(nblocks, 1, 1);
    dim3 threads(nthreads, 1, 1);
    zgbtrs_swap_kernel_batched<<<grid, threads, 0, queue->cuda_stream()>>>
    (n, dA_array, ldda, dipiv_array, j);
}

////////////////////////////////////////////////////////////////////////////////
extern "C"
void magmablas_zgeru_batched_core(
        magma_int_t m, magma_int_t n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex** dX_array, magma_int_t xi, magma_int_t xj, magma_int_t lddx, magma_int_t incx,
        magmaDoubleComplex** dY_array, magma_int_t yi, magma_int_t yj, magma_int_t lddy, magma_int_t incy,
        magmaDoubleComplex** dA_array, magma_int_t ai, magma_int_t aj, magma_int_t ldda,
        magma_int_t batchCount, magma_queue_t queue )
{
    if(m == 0 || n == 0 || batchCount == 0) return;

    magma_int_t ntx     = min(m, GBTRS_GERU_THREADS_X);
    magma_int_t nty     = min(n, GBTRS_GERU_THREADS_Y);
    magma_int_t nblocks = magma_ceildiv(m, GBTRS_GERU_THREADS_X);

    dim3 threads(ntx, nty, 1);

    magma_int_t max_batchCount = queue->get_maxBatch();
    for(magma_int_t ib = 0; ib < batchCount; ib += max_batchCount){
        magma_int_t ibatch = min(max_batchCount, batchCount - ib);
        dim3 grid(nblocks, 1, ibatch);

        zgeru_kernel_batched<<<grid, threads, 0, queue->cuda_stream()>>>
        (m, n, alpha, dX_array + ib, xi, xj, lddx, incx, dY_array + ib, yi, yj, lddy, incy, dA_array + ib, ai, aj, ldda);
    }
}

////////////////////////////////////////////////////////////////////////////////
extern "C"
void magmablas_zgbtrs_upper_columnwise_batched(
        magma_int_t n, magma_int_t kl, magma_int_t ku,
        magma_int_t nrhs, magma_int_t j,
        magmaDoubleComplex** dA_array, magma_int_t ldda,
        magmaDoubleComplex** dB_array, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t kv       = kl + ku;
    magma_int_t nthreads = min(GBTRS_UPPER_THREADS, kv+1);
    magma_int_t nblocks  = batchCount;

    dim3 grid(nblocks, 1, 1);
    dim3 threads(nthreads, 1, 1);
    zgbtrs_upper_columnwise_kernel_batched<<<grid, threads, 0, queue->cuda_stream()>>>
    (n, kl, ku, nrhs, j, dA_array, ldda, dB_array, lddb);
}

////////////////////////////////////////////////////////////////////////////////
extern "C"
magma_int_t
magmablas_zgbtrs_lower_blocked_batched(
        magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
        magmaDoubleComplex** dA_array, magma_int_t ldda, magma_int_t** dipiv_array,
        magmaDoubleComplex** dB_array, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t nb         = GBTRS_LOWER_NB;
    magma_int_t nrhs_nb    = 4;
    magma_int_t nthreads   = kl;
    magma_int_t nthreads32 = magma_roundup(nthreads, 32);
    magma_int_t nblocks_x  = batchCount;
    magma_int_t nblocks_y  = magma_ceildiv(nrhs, nrhs_nb);
    magma_int_t sldb       = (nb + kl);


    magma_int_t shmem = 0;
    shmem += sldb * nrhs_nb * sizeof(magmaDoubleComplex);
    shmem += nb * sizeof(int);

    dim3 threads(nthreads, 1, 1);
    dim3 grid(nblocks_x, nblocks_y);
    void *kernel_args[] = {&n, &kl, &ku, &nrhs, &nrhs_nb, &dA_array, &ldda, &dipiv_array, &dB_array, &lddb};

    magma_int_t arginfo = 0;
    cudaError_t e;
    switch( nthreads32 ) {
        case   32: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched<  32, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case   64: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched<  64, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case   96: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched<  96, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  128: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 128, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  160: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 160, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  192: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 192, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  224: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 224, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  256: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 256, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  288: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 288, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  320: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 320, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  352: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 352, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  384: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 384, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  416: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 416, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  448: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 448, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  480: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 480, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  512: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 512, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  544: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 544, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  576: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 576, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  608: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 608, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  640: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 640, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  672: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 672, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  704: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 704, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  736: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 736, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  768: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 768, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  800: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 800, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  832: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 832, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  864: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 864, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  896: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 896, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  928: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 928, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  960: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 960, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  992: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched< 992, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case 1024: e = cudaLaunchKernel((void*)zgbtrs_lower_blocked_kernel_batched<1024, GBTRS_LOWER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        default: arginfo = -100;
    }

    if(e != cudaSuccess) arginfo = -100;

    return arginfo;
}

////////////////////////////////////////////////////////////////////////////////
extern "C"
magma_int_t
magmablas_zgbtrs_upper_blocked_batched(
        magma_int_t n, magma_int_t kl, magma_int_t ku, magma_int_t nrhs,
        magmaDoubleComplex** dA_array, magma_int_t ldda,
        magmaDoubleComplex** dB_array, magma_int_t lddb,
        magma_int_t batchCount, magma_queue_t queue )
{
    magma_int_t kv         = kl + ku;
    magma_int_t nb         = GBTRS_UPPER_NB;
    magma_int_t nrhs_nb    = 4;
    magma_int_t nthreads   = kv + 1;
    magma_int_t nthreads32 = magma_roundup(nthreads, 32);
    magma_int_t nblocks_x  = batchCount;
    magma_int_t nblocks_y  = magma_ceildiv(nrhs, nrhs_nb);
    magma_int_t sldb       = (nb + kv);


    magma_int_t shmem = 0;
    shmem += sldb * nrhs_nb * sizeof(magmaDoubleComplex);  // sB
    shmem += 1 * sizeof(magmaDoubleComplex);  // stmp

    dim3 threads(nthreads, 1, 1);
    dim3 grid(nblocks_x, nblocks_y);
    void *kernel_args[] = {&n, &kl, &ku, &nrhs, &nrhs_nb, &dA_array, &ldda, &dB_array, &lddb};

    magma_int_t arginfo = 0;
    cudaError_t e;
    switch( nthreads32 ) {
        case   32: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched<  32, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case   64: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched<  64, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case   96: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched<  96, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  128: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 128, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  160: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 160, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  192: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 192, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  224: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 224, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  256: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 256, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  288: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 288, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  320: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 320, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  352: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 352, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  384: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 384, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  416: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 416, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  448: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 448, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  480: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 480, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  512: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 512, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  544: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 544, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  576: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 576, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  608: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 608, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  640: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 640, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  672: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 672, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  704: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 704, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  736: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 736, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  768: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 768, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  800: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 800, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  832: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 832, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  864: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 864, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  896: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 896, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  928: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 928, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  960: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 960, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case  992: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched< 992, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        case 1024: e = cudaLaunchKernel((void*)zgbtrs_upper_blocked_kernel_batched<1024, GBTRS_UPPER_NB>, grid, threads, kernel_args, shmem, queue->cuda_stream());break;
        default: arginfo = -100;
    }

    if(e != cudaSuccess) arginfo = -100;

    return arginfo;
}

