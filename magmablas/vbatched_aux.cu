/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah
       
*/

#include "magma_internal.h"
#include "magma_templates.h"

#define BLK_X    (128)

// =============================================================================
// Auxiliary functions for vbatched routines


/******************************************************************************/
// max reduce kernel
// 1) set overwrite to 0 ==> result is written to y and x is untouched
//    set overwrite to 1 ==> result is written to x (x is destroyed) 
// Each thread block gets the max of <MAX_REDUCE_SEGMENT> elements and 
// writes it to the workspace
#define MAX_REDUCE_SEGMENT    (512)    // must be even
#define MAX_REDUCE_TX         (MAX_REDUCE_SEGMENT/2)
    
__global__ 
void magma_ivec_max_kernel( int vecsize, 
                              magma_int_t* x, magma_int_t* y, 
                              int overwrite)
{
    const int tx  = threadIdx.x;
    const int bx  = blockIdx.x;
    const int gtx = bx * MAX_REDUCE_SEGMENT + tx;
    
    __shared__ int swork[MAX_REDUCE_SEGMENT];
        
    // init shmem
    swork[tx] = 0;
    swork[tx + MAX_REDUCE_TX] = 0;
    
    // read the input segment into swork
    if(gtx < vecsize)swork[tx] = (int)x[gtx];
    if( (gtx + MAX_REDUCE_TX) < vecsize ) swork[tx + MAX_REDUCE_TX] = (int)x[gtx + MAX_REDUCE_TX];
    __syncthreads();
    magma_max_reduce<MAX_REDUCE_SEGMENT, int>(tx, swork);
    __syncthreads();
    // write the result back
    if(overwrite == 0)
    {
        if(tx == 0) y[bx] = (magma_int_t)swork[0];
    }
    else
    {
        if(tx == 0) x[bx] = (magma_int_t)swork[0];
    }
}

//----------------
// kernel driver
//----------------
extern "C" 
magma_int_t magma_ivec_max( magma_int_t vecsize, 
                              magma_int_t* x, 
                              magma_int_t* work, magma_int_t lwork, magma_queue_t queue)
{
    dim3 threads(MAX_REDUCE_TX, 1, 1);
    dim3 grid( magma_ceildiv( vecsize, MAX_REDUCE_SEGMENT ), 1, 1);
    if (lwork < (magma_int_t)grid.x) {
        printf("error in %s: lwork must be at least %lld, input is %lld\n",
               __func__, (long long) grid.x, (long long) lwork );
    }
    
    magma_ivec_max_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, x, work, 0);
    magma_int_t new_vecsize = grid.x;
    
    while(new_vecsize > 1)
    {
        grid.x = magma_ceildiv( new_vecsize, MAX_REDUCE_SEGMENT );
        magma_ivec_max_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(new_vecsize, work, NULL, 1);
        new_vecsize = grid.x;
    }
    
    // copy the result to cpu and return it
    magma_int_t vecmax = 0;
    magma_getvector(1, sizeof(magma_int_t), work, 1, &vecmax, 1, queue);
    return (magma_int_t)vecmax;
}


/******************************************************************************/
// integer sum (isum) reduce kernel
// initially needed for vbatched trsm
// 1) set overwrite to 0 ==> result is written to y and x is untouched
//    set overwrite to 1 ==> result is written to x (x is destroyed) 
// Each thread block gets the custom sum of <ISUM_REDUCE_SEGMENT> elements and 
// writes it to the workspace
#define ISUM_REDUCE_SEGMENT    (512)    // must be even
#define ISUM_REDUCE_TX         (ISUM_REDUCE_SEGMENT/2)

__global__ 
void magma_isum_reduce_kernel( int vecsize, 
                              magma_int_t* x, magma_int_t* y, 
                              int overwrite)
{
    const int tx  = threadIdx.x;
    const int bx  = blockIdx.x;
    const int gtx = bx * ISUM_REDUCE_SEGMENT + tx;
    
    __shared__ int swork[ISUM_REDUCE_SEGMENT];
        
    // init shmem
    swork[tx] = 0;
    swork[tx + ISUM_REDUCE_TX] = 0;
    
    // read the input segment into swork
    if(gtx < vecsize)swork[tx] = (int)(x[gtx]);
    if( (gtx + ISUM_REDUCE_TX) < vecsize ) swork[tx + ISUM_REDUCE_TX] = (int)(x[gtx + ISUM_REDUCE_TX]);
    __syncthreads();
    magma_sum_reduce<ISUM_REDUCE_SEGMENT, int>(tx, swork);
    __syncthreads();
    // write the result back
    if(overwrite == 0)
    {
        if(tx == 0) y[bx] = (magma_int_t)swork[0];
    }
    else
    {
        if(tx == 0) x[bx] = (magma_int_t)swork[0];
    }
}

//----------------
// kernel driver
//----------------
extern "C" 
magma_int_t magma_isum_reduce( magma_int_t vecsize, 
                              magma_int_t* x, 
                              magma_int_t* work, magma_int_t lwork, magma_queue_t queue)
{
    dim3 threads(ISUM_REDUCE_TX, 1, 1);
    dim3 grid( magma_ceildiv( vecsize, ISUM_REDUCE_SEGMENT ), 1, 1);
    if (lwork < (magma_int_t)grid.x) {
        printf("error in %s: lwork must be at least %lld, input is %lld\n",
               __func__, (long long) grid.x, (long long) lwork );
    }
    
    magma_isum_reduce_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, x, work, 0);
    magma_int_t new_vecsize = grid.x;
    
    while(new_vecsize > 1)
    {
        grid.x = magma_ceildiv( new_vecsize, ISUM_REDUCE_SEGMENT );
        magma_isum_reduce_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(new_vecsize, work, NULL, 1);
        new_vecsize = grid.x;
    }
    
    // copy the result to cpu and return it
    magma_int_t isum = 0;
    magma_getvector(1, sizeof(magma_int_t), work, 1, &isum, 1, queue);
    return (magma_int_t)isum;
}


/******************************************************************************/
// y[i] = a1 * x1[i] + a2 * x2[i]
__global__ 
void magma_ivec_add_kernel( int vecsize, 
                                  magma_int_t a1, magma_int_t *x1, 
                                  magma_int_t a2, magma_int_t *x2, 
                                  magma_int_t *y)
{
    const int indx = blockIdx.x * blockDim.x + threadIdx.x;
    if(indx < vecsize)
    {
        y[indx] = a1 * x1[indx] + a2 * x2[indx];
    }
}

//----------------
// kernel driver
//----------------
extern "C" 
void magma_ivec_add( magma_int_t vecsize, 
                           magma_int_t a1, magma_int_t *x1, 
                           magma_int_t a2, magma_int_t *x2, 
                           magma_int_t *y, magma_queue_t queue)
{
    dim3 threads(BLK_X, 1, 1);
    dim3 grid( magma_ceildiv( vecsize, BLK_X ), 1, 1);
    magma_ivec_add_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, a1, x1, a2, x2, y);
}


/******************************************************************************/
// y[i] = x1[i] * x2[i]
__global__ 
void magma_ivec_mul_kernel( int vecsize, 
                                  magma_int_t *x1, 
                                  magma_int_t *x2, 
                                  magma_int_t *y)
{
    const int indx = blockIdx.x * blockDim.x + threadIdx.x;
    if(indx < vecsize)
    {
        y[indx] = x1[indx] * x2[indx];
    }
}

//----------------
// kernel driver
//----------------
extern "C" 
void magma_ivec_mul( magma_int_t vecsize, 
                           magma_int_t *x1, magma_int_t *x2, 
                           magma_int_t *y, magma_queue_t queue)
{
    dim3 threads(BLK_X, 1, 1);
    dim3 grid( magma_ceildiv( vecsize, BLK_X ), 1, 1);
    magma_ivec_mul_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, x1, x2, y);
}


/******************************************************************************/
// ceildiv
__global__ void magma_ivec_ceildiv_kernel(int vecsize, magma_int_t *x, int nb, magma_int_t *y)
{
    const int indx = blockIdx.x * blockDim.x + threadIdx.x;
    if(indx < vecsize)
    {
        y[indx] = (magma_int_t)magma_ceildiv(x[indx], (magma_int_t)nb);
    }
}

//----------------
// kernel driver
//----------------
extern "C" 
void magma_ivec_ceildiv( magma_int_t vecsize, 
                        magma_int_t *x, 
                        magma_int_t nb, 
                        magma_int_t *y, magma_queue_t queue)
{
    dim3 threads(BLK_X);
    dim3 grid( magma_ceildiv( vecsize, BLK_X ), 1);
    
    magma_ivec_ceildiv_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, x, nb, y);
}


/******************************************************************************/
// roundup
__global__ 
void magma_ivec_roundup_kernel(int vecsize, magma_int_t *x, int nb, magma_int_t *y)
{
    const int indx = blockIdx.x * blockDim.x + threadIdx.x;
    if(indx < vecsize)
    {
        y[indx] = (magma_int_t)magma_roundup(x[indx], (magma_int_t)nb);
    }
}

//----------------
// kernel driver
//----------------
extern "C" 
void magma_ivec_roundup( magma_int_t vecsize, 
                        magma_int_t *x, 
                        magma_int_t nb, 
                        magma_int_t *y, magma_queue_t queue)
{
    dim3 threads(BLK_X);
    dim3 grid( magma_ceildiv( vecsize, BLK_X ), 1);
    
    magma_ivec_roundup_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, x, nb, y);
}


/******************************************************************************/
// set vector to a const value
template<typename T>
__global__ 
void magma_setvector_const_gpu_kernel(int vecsize, T *x, T value)
{
    const int indx = blockIdx.x * blockDim.x + threadIdx.x;
    if(indx < vecsize)
    {
        x[indx] = value;
    }
}

//----------------
// kernel drivers
//----------------
extern "C" 
void magma_ivec_setc( magma_int_t vecsize, 
                                magma_int_t *x, 
                                magma_int_t value, 
                                magma_queue_t queue)
{
    dim3 threads(BLK_X);
    dim3 grid( magma_ceildiv( vecsize, BLK_X ), 1);
    
    magma_setvector_const_gpu_kernel<magma_int_t><<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, x, value);
}

//---------------
extern "C" 
void magma_zsetvector_const( magma_int_t vecsize, 
                                magmaDoubleComplex *x, 
                                magmaDoubleComplex value, 
                                magma_queue_t queue)
{
    dim3 threads(BLK_X);
    dim3 grid( magma_ceildiv( vecsize, BLK_X ), 1);
    
    magma_setvector_const_gpu_kernel<magmaDoubleComplex><<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, x, value);
}

//---------------
extern "C" 
void magma_csetvector_const( magma_int_t vecsize, 
                                magmaFloatComplex *x, 
                                magmaFloatComplex value, 
                                magma_queue_t queue)
{
    dim3 threads(BLK_X);
    dim3 grid( magma_ceildiv( vecsize, BLK_X ), 1);
    
    magma_setvector_const_gpu_kernel<magmaFloatComplex><<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, x, value);
}

//---------------
extern "C" 
void magma_dsetvector_const( magma_int_t vecsize, 
                                double *x, 
                                double value, 
                                magma_queue_t queue)
{
    dim3 threads(BLK_X);
    dim3 grid( magma_ceildiv( vecsize, BLK_X ), 1);
    
    magma_setvector_const_gpu_kernel<double><<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, x, value);
}

//---------------
extern "C" 
void magma_ssetvector_const( magma_int_t vecsize, 
                                float *x, 
                                float value, 
                                magma_queue_t queue)
{
    dim3 threads(BLK_X);
    dim3 grid( magma_ceildiv( vecsize, BLK_X ), 1);
    
    magma_setvector_const_gpu_kernel<float><<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, x, value);
}


/******************************************************************************/
// performs addition with a const value
__global__ 
void magma_ivec_addc_kernel(int vecsize, magma_int_t *x, int value, magma_int_t *y)
{
    const int indx = blockIdx.x * blockDim.x + threadIdx.x;
    if(indx < vecsize)
    {
        y[indx] = (x[indx] + (magma_int_t)value); 
    }
}

//----------------
// kernel driver
//----------------
extern "C" 
void magma_ivec_addc(magma_int_t vecsize, magma_int_t *x, magma_int_t value, magma_int_t *y, magma_queue_t queue)
{
    dim3 threads(BLK_X);
    dim3 grid( magma_ceildiv( vecsize, BLK_X ), 1);
    
    magma_ivec_addc_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, x, value, y);
}


/******************************************************************************/
// performs multiplication with a const value
__global__ 
void magma_ivec_mulc_kernel(int vecsize, magma_int_t *x, int value, magma_int_t *y)
{
    const int indx = blockIdx.x * blockDim.x + threadIdx.x;
    if(indx < vecsize)
    {
        y[indx] = (x[indx] * (magma_int_t)value); 
    }
}

//----------------
// kernel driver
//----------------
extern "C" 
void magma_ivec_mulc(magma_int_t vecsize, magma_int_t *x, magma_int_t value, magma_int_t *y, magma_queue_t queue)
{
    dim3 threads(BLK_X);
    dim3 grid( magma_ceildiv( vecsize, BLK_X ), 1);
    
    magma_ivec_mulc_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, x, value, y);
}


/******************************************************************************/
// performs a min. operation against a const value
__global__ 
void magma_ivec_minc_kernel(int vecsize, magma_int_t *x, int value, magma_int_t *y)
{
    const int indx = blockIdx.x * blockDim.x + threadIdx.x;
    const magma_int_t value_l = (magma_int_t)value; 
    if(indx < vecsize)
    {
        y[indx] = ( value_l < x[indx] )? value_l : x[indx];
    }
}

//----------------
// kernel driver
//----------------
extern "C" 
void magma_ivec_minc(magma_int_t vecsize, magma_int_t *x, magma_int_t value, magma_int_t *y, magma_queue_t queue)
{
    dim3 threads(BLK_X, 1, 1);
    dim3 grid( magma_ceildiv( vecsize, BLK_X ), 1, 1);
    
    magma_ivec_minc_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, x, value, y);
}


/******************************************************************************/
// performs a max. operation against a const value
__global__ 
void magma_ivec_maxc_kernel(int vecsize, magma_int_t *x, int value, magma_int_t *y)
{
    const int indx = blockIdx.x * blockDim.x + threadIdx.x;
    const magma_int_t value_l = (magma_int_t)value; 
    if(indx < vecsize)
    {
        y[indx] = ( value_l > x[indx] )? value_l : x[indx];
    }
}

//----------------
// kernel driver
//----------------
extern "C" 
void magma_ivec_maxc(magma_int_t vecsize, magma_int_t* x, magma_int_t value, magma_int_t* y, magma_queue_t queue)
{
    dim3 threads(BLK_X, 1, 1);
    dim3 grid( magma_ceildiv( vecsize, BLK_X ), 1, 1);
    
    magma_ivec_maxc_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, x, value, y);
}


/******************************************************************************/
// This kernel is for the vbatched trsm routine
// auxiliary kernel to compute jb = (m % TRI_BATCHED_NB == 0) ? TRI_BATCHED_NB : (m % TRI_BATCHED_NB)
// This kernel is specific to trsm, so it is not in vbatched_aux.h
__global__ void magma_compute_trsm_jb_kernel(int vecsize, magma_int_t *m, int tri_nb, magma_int_t *jbv)
//(int vecsize, int* m, int tri_nb, int* jbv)
{
    const int indx = blockIdx.x * blockDim.x + threadIdx.x;
    const int my_m = (magma_int_t)m[indx];
    if(indx < vecsize)
    {
        int my_jb;
        if(my_m % tri_nb == 0) my_jb = tri_nb;
        else my_jb = (my_m % tri_nb);
        
        jbv[indx] = (magma_int_t)my_jb; 
    }
}

//--------------
// Kernel Driver
//--------------
extern "C"
void magma_compute_trsm_jb(magma_int_t vecsize, magma_int_t* m, magma_int_t tri_nb, magma_int_t* jbv, magma_queue_t queue)
{
    const int nthreads = 128;
    dim3 threads(nthreads);
    dim3 grid( magma_ceildiv( vecsize, nthreads ), 1);
    
    magma_compute_trsm_jb_kernel<<<grid, threads, 0, queue->cuda_stream()>>>(vecsize, m, tri_nb, jbv);
}


/******************************************************************************/
// A max-reduce kernel specific for computing the max M/N/K for launching vbatched kernels
#define AUX_MAX_SEGMENT    (256)    // must be even
#define AUX_MAX_TX         (AUX_MAX_SEGMENT)
__global__ void magma_imax_size_kernel_1(magma_int_t *n, int l)
{
    magma_int_t *vec; 
    const int tx = threadIdx.x; 
    int i, value, lmax = 0;
    const int L = (l/AUX_MAX_SEGMENT) * AUX_MAX_SEGMENT;
    
    __shared__ int swork[AUX_MAX_SEGMENT];
    
    vec = n;
    for(i = 0; i < L; i+= AUX_MAX_SEGMENT){
        value = (int)vec[i + tx];
        lmax = ( value > lmax ) ? value : lmax;
    }
    
    // last incomplete segment
    if(tx < l - L){
        value = (int)vec[L + tx];
        lmax = ( value > lmax ) ? value : lmax;
    }
    
    swork[tx] = lmax;
    __syncthreads();
    magma_max_reduce<AUX_MAX_SEGMENT, int>(tx, swork);
    // no need to sync
    if(tx == 0){
        vec[l] = (magma_int_t)(swork[0]);
    }
}

//--------------
// Kernel Driver
//--------------
extern "C"
void magma_imax_size_1(magma_int_t *n, magma_int_t l, magma_queue_t queue)
{
    dim3 grid(1, 1, 1);
    dim3 threads(AUX_MAX_TX, 1, 1);
    magma_imax_size_kernel_1<<< grid, threads, 0, queue->cuda_stream() >>>(n, l);
}


/******************************************************************************/
__global__ void magma_imax_size_kernel_2(magma_int_t *m, magma_int_t *n, int l)
{
    magma_int_t *vec; 
    const int bx = blockIdx.x; 
    const int tx = threadIdx.x; 
    int i, value, lmax = 0;
    const int L = (l/AUX_MAX_SEGMENT) * AUX_MAX_SEGMENT;
    
    __shared__ int swork[AUX_MAX_SEGMENT];
    
    if     (bx == 0) vec = m;
    else if(bx == 1) vec = n; 
    
    for(i = 0; i < L; i+= AUX_MAX_SEGMENT){
        value = (int)vec[i + tx];
        lmax = ( value > lmax ) ? value : lmax;
    }
    
    // last incomplete segment
    if(tx < l - L){
        value = (int)vec[L + tx];
        lmax = ( value > lmax ) ? value : lmax;
    }
    
    swork[tx] = lmax;
    __syncthreads();
    magma_max_reduce<AUX_MAX_SEGMENT, int>(tx, swork);
    // no need to sync
    if(tx == 0){
        vec[l] = (magma_int_t)(swork[0]);
    }
}

//--------------
// Kernel Driver
//--------------
extern "C"
void magma_imax_size_2(magma_int_t *m, magma_int_t *n, magma_int_t l, magma_queue_t queue)
{
    dim3 grid(2, 1, 1);
    dim3 threads(AUX_MAX_TX, 1, 1);
    magma_imax_size_kernel_2<<< grid, threads, 0, queue->cuda_stream() >>>(m, n, l);
}


/******************************************************************************/
__global__ void magma_imax_size_kernel_3(magma_int_t *m, magma_int_t *n, magma_int_t *k, int l)
{
    magma_int_t *vec; 
    const int bx = blockIdx.x; 
    const int tx = threadIdx.x; 
    int i, value, lmax = 0;
    const int L = (l/AUX_MAX_SEGMENT) * AUX_MAX_SEGMENT;
    
    __shared__ int swork[AUX_MAX_SEGMENT];
    
    if     (bx == 0) vec = m;
    else if(bx == 1) vec = n; 
    else if(bx == 2) vec = k;
    
    for(i = 0; i < L; i+= AUX_MAX_SEGMENT){
        value = (int)vec[i + tx];
        lmax = ( value > lmax ) ? value : lmax;
    }
    
    // last incomplete segment
    if(tx < l - L){
        value = (int)vec[L + tx];
        lmax = ( value > lmax ) ? value : lmax;
    }
    
    swork[tx] = lmax;
    __syncthreads();
    magma_max_reduce<AUX_MAX_SEGMENT, int>(tx, swork);
    // no need to sync
    if(tx == 0){
        vec[l] = (magma_int_t)(swork[0]);
    }
}

//--------------
// Kernel Driver
//--------------
extern "C"
void magma_imax_size_3(magma_int_t *m, magma_int_t *n, magma_int_t *k, magma_int_t l, magma_queue_t queue)
{
    dim3 grid(3, 1, 1);
    dim3 threads(AUX_MAX_TX, 1, 1);
    magma_imax_size_kernel_3<<< grid, threads, 0, queue->cuda_stream() >>>(m, n, k, l);
}
