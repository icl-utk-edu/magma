
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


#ifndef MAGMABLAS_ZGETF2_DEVICES_Z_H
#define MAGMABLAS_ZGETF2_DEVICES_Z_H

/******************************************************************************/
static __device__ __inline__ int
izamax_devfunc(int length, const magmaDoubleComplex *x, int incx, double *shared_x, int *shared_idx)
{
    int tx = threadIdx.x;
    magmaDoubleComplex res;
    double  res1;
    int nchunk = magma_ceildiv( length, zamax );

    if ( tx < zamax ) {
        shared_x[tx]   = 0.0;
        shared_idx[tx] = tx; //-1; // -1 will crash the code in case matrix is singular, better is to put =tx and make check info at output
    }
    __syncthreads();

    for (int s =0; s < nchunk; s++)
    {
        if ( (tx + s * zamax < length) && (tx < zamax) )
        {
            res = x[(tx + s * zamax) * incx];
            res1 = fabs(MAGMA_Z_REAL(res)) + fabs(MAGMA_Z_IMAG(res));

            if ( res1  > shared_x[tx] )
            {
                shared_x[tx] = res1;
                shared_idx[tx] = tx + s * zamax;
            }
        }
    }
    __syncthreads();

    if (length >= zamax) // there are more than 128 threads working ==> all shared_x shared_idx are initialized here so I can call the fixed getidmax
        magma_getidmax<zamax>(tx, shared_x, shared_idx);
    else
        magma_getidmax_n(min(zamax,length), tx, shared_x, shared_idx);
    return shared_idx[0];
}

/******************************************************************************/
static __device__ __inline__
void zswap_device( magma_int_t n,
                   magmaDoubleComplex_ptr x, magma_int_t incx,
                   magma_int_t step, magma_int_t* ipiv)
{
    const int tx = threadIdx.x;

    __shared__ int jp;

    if (tx == 0){
        jp = ipiv[step] - 1;
    }
    __syncthreads();

    if (jp == step) return; // no pivot

    if (tx < n) {
        magmaDoubleComplex tmp = x[jp + tx * incx];
        x[jp + tx * incx] = x[step + tx * incx];
        x[step + tx * incx] = tmp;
    }
}

/******************************************************************************/
// This version swaps two rows that are specified at the input
// the logic deciding these two rows is assumed to be at the
// kernel level (unlike zswap_device)
static __device__ __inline__
void zswap_device_v2(
            magma_int_t n,
            magmaDoubleComplex_ptr x1, magma_int_t incx1,
            magmaDoubleComplex_ptr x2, magma_int_t incx2 )
{
    const int tx = threadIdx.x;

    if (tx < n) {
        magmaDoubleComplex tmp  = x1[tx * incx1];
        x1[tx * incx1]          = x2[tx * incx2];
        x2[tx * incx2]          = tmp;
    }
}

/******************************************************************************/
template<int N>
static __device__ __inline__
void zscal_zgeru_device( int m,
                         magmaDoubleComplex_ptr dA, int lda,
                         magma_int_t *info, int step, int gbstep)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * blockDim.x + tx;
    // checkinfo to avoid computation of the singular matrix
    if( (*info) != 0 ) return;

    magmaDoubleComplex rA[N], reg;
    __shared__ magmaDoubleComplex shared_y[N];

    if (tx < N) {
        shared_y[tx] = dA[lda * tx];
    }
    __syncthreads();

    if (shared_y[0] == MAGMA_Z_ZERO) {
        (*info) = step + gbstep + 1;
        return;
    }

    // terminate threads that are out of the range
    if (gtx == 0 || gtx >= m) return;

    reg = MAGMA_Z_DIV(MAGMA_Z_ONE, shared_y[0]);
    #pragma unroll
    for(int i = 0; i < N; i++)
        rA[i] = dA[ i* lda + gtx ];

    rA[0] *= reg;

    #pragma unroll
    for(int i = 1; i < N; i++)
        rA[i] -= rA[0] * shared_y[i];

    #pragma unroll
    for(int i = 0; i < N; i++)
        dA[gtx + i * lda] = rA[i];
}

/******************************************************************************/
static __device__ __inline__
void zscal_zgeru_generic_device( int m, int n,
                         magmaDoubleComplex_ptr dA, int lda,
                         magma_int_t *info, int step, int gbstep)
{
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * blockDim.x + tx;
    // checkinfo to avoid computation of the singular matrix
    if( (*info) != 0 ) return;
    if (gtx == 0 || gtx >= m) return;

    magmaDoubleComplex rA, reg;

    if (dA[0] == MAGMA_Z_ZERO) {
        (*info) = step + gbstep + 1;
        return;
    }

    reg = MAGMA_Z_DIV(MAGMA_Z_ONE, dA[0]);
    rA  = dA[ gtx ];
    rA *= reg;

    dA[ gtx ] = rA;
    #pragma unroll
    for(int i = 1; i < n; i++)
        dA[i * lda + gtx] -= rA * dA[i * lda + 0];

}

/******************************************************************************/
static __device__ __inline__
void
zupdate_device(int m, int step, magmaDoubleComplex* x, int ldx,  magmaDoubleComplex *A, int lda)
{
    int tid = threadIdx.x;
    int nchunk = magma_ceildiv( m, MAX_NTHREADS );
    int indx;
    //magmaDoubleComplex reg = MAGMA_Z_ZERO;

    // update the current column by all the previous one
    #pragma unroll
    for (int i=0; i < step; i++) {
        for (int s=0; s < nchunk; s++)
        {
            indx = tid + s * MAX_NTHREADS;
            if ( indx > i  && indx < m ) {
                A[indx] -=  A[i] * x[indx + i*ldx];
                //printf("         @ step %d tid %d updating x[tid]*y[i]=A %5.3f %5.3f = %5.3f  at i %d\n", step, tid, x[tid + i*ldx], A[i], A[tid],i);
            }
        }
        __syncthreads();
    }

    //printf("         @ step %d tid %d adding %5.3f to A %5.3f make it %5.3f\n",step,tid,-reg,A[tid],A[tid]-reg);
}


/******************************************************************************/
static __device__ __inline__
void
zscal5_device(int m, magmaDoubleComplex* x, magmaDoubleComplex alpha)
{
    int tid = threadIdx.x;
    int nchunk = magma_ceildiv( m, MAX_NTHREADS );

    for (int s=0; s < nchunk; s++)
    {
        if ( (tid + s * MAX_NTHREADS) < m ) {
            #if 0
            x[tid + s * MAX_NTHREADS] *= MAGMA_Z_DIV(MAGMA_Z_ONE, alpha);
            #else
            x[tid + s * MAX_NTHREADS] = x[tid + s * MAX_NTHREADS]/alpha;
            #endif
        }
    }
    __syncthreads();
}

/******************************************************************************/
template<int WIDTH>
static __device__ __inline__
void
zgetf2_fused_device( int m, int minmn, magmaDoubleComplex rA[WIDTH], magma_int_t* dipiv,
                     magmaDoubleComplex* swork, int &linfo, int gbstep, int &rowid)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    magmaDoubleComplex reg       = MAGMA_Z_ZERO;
    magmaDoubleComplex update    = MAGMA_Z_ZERO;

    int max_id;
    double rx_abs_max = MAGMA_D_ZERO;

    magmaDoubleComplex *sx = (magmaDoubleComplex*)(swork);
    double* dsx = (double*)(sx + blockDim.y * WIDTH);
    int* isx    = (int*)(dsx + blockDim.y * m);
    int* sipiv  = (int*)(isx + blockDim.y * m);
    sx    += ty * WIDTH;
    dsx   += ty * m;
    isx   += ty * m;
    sipiv += ty * WIDTH;

    rowid = tx;

    // init sipiv
    if(tx < WIDTH){
        sipiv[tx] = 0;
    }

    #pragma unroll
    for(int i = 0; i < WIDTH; i++){
        // izamax and find pivot
        dsx[ rowid ] = fabs(MAGMA_Z_REAL( rA[i] )) + fabs(MAGMA_Z_IMAG( rA[i] ));
        isx[ tx ] = tx;
        __syncthreads();
        magma_getidmax_n(m-i, tx, dsx+i, isx+i); // this devfunc has syncthreads at the end
        rx_abs_max = dsx[i];
        max_id = isx[i];
        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (gbstep+i+1) : linfo;
        update = ( rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ZERO : MAGMA_Z_ONE;
        __syncthreads();

        if(rowid == max_id){
            sipiv[i] = max_id;
            rowid = i;
            #pragma unroll
            for(int j = 0; j < WIDTH; j++){
                sx[j] = update * rA[j];
            }
        }
        else if(rowid == i){
            rowid = max_id;
        }
        __syncthreads();

        reg = (linfo == 0 ) ? MAGMA_Z_DIV(MAGMA_Z_ONE, sx[i] ) : MAGMA_Z_ONE;
        // scal and ger
        if( rowid > i ){
            rA[i] *= reg;
            #pragma unroll
            for(int j = i+1; j < WIDTH; j++){
                rA[j] -= rA[i] * sx[j];
            }
        }
    }

    // write
    if(tx < minmn){
        dipiv[tx] = (magma_int_t)(sipiv[tx] + 1); // fortran indexing
        //printf("--- ipiv[%d] --- = %d\n", tx, dipiv[tx]);
    }
}


#endif // MAGMABLAS_ZGETF2_DEVICES_Z_H
