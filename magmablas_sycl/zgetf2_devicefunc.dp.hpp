#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
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
static __inline__ int
izamax_devfunc(int length, const magmaDoubleComplex *x, int incx, double *shared_x, int *shared_idx,
               const sycl::nd_item<3> &item_ct1)
{
    int tx = item_ct1.get_local_id(2);
    magmaDoubleComplex res;
    double  res1;
    int nchunk = magma_ceildiv( length, zamax );

    if ( tx < zamax ) {
        shared_x[tx]   = 0.0;
        shared_idx[tx] = tx; //-1; // -1 will crash the code in case matrix is singular, better is to put =tx and make check info at output
    }
    /*
    DPCT1065:143: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    for (int s =0; s < nchunk; s++)
    {
        if ( (tx + s * zamax < length) && (tx < zamax) )
        {
            res = x[(tx + s * zamax) * incx];
            res1 =
                sycl::fabs(MAGMA_Z_REAL(res)) + sycl::fabs(MAGMA_Z_IMAG(res));

            if ( res1  > shared_x[tx] )
            {
                shared_x[tx] = res1;
                shared_idx[tx] = tx + s * zamax;
            }
        }
    }
    /*
    DPCT1065:144: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (length >= zamax) // there are more than 128 threads working ==> all shared_x shared_idx are initialized here so I can call the fixed getidmax
        magma_getidmax<zamax>(tx, shared_x, shared_idx, item_ct1);
    else
        magma_getidmax_n(min(zamax, length), tx, shared_x, shared_idx,
                         item_ct1);
    return shared_idx[0];
}

/******************************************************************************/
static __inline__
void zswap_device( magma_int_t n,
                   magmaDoubleComplex_ptr x, magma_int_t incx,
                   magma_int_t step, magma_int_t* ipiv,
                   const sycl::nd_item<3> &item_ct1, int *jp)
{
    const int tx = item_ct1.get_local_id(2);

    if (tx == 0){
        *jp = ipiv[step] - 1;
    }
    /*
    DPCT1065:145: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (*jp == step) return; // no pivot

    if (tx < n) {
        magmaDoubleComplex tmp = x[*jp + tx * incx];
        x[*jp + tx * incx] = x[step + tx * incx];
        x[step + tx * incx] = tmp;
    }
}

/******************************************************************************/
// This version swaps two rows that are specified at the input
// the logic deciding these two rows is assumed to be at the
// kernel level (unlike zswap_device)
static __inline__
void zswap_device_v2(
            magma_int_t n,
            magmaDoubleComplex_ptr x1, magma_int_t incx1,
            magmaDoubleComplex_ptr x2, magma_int_t incx2 ,
            const sycl::nd_item<3> &item_ct1)
{
    const int tx = item_ct1.get_local_id(2);

    if (tx < n) {
        magmaDoubleComplex tmp  = x1[tx * incx1];
        x1[tx * incx1]          = x2[tx * incx2];
        x2[tx * incx2]          = tmp;
    }
}

/******************************************************************************/
template<int N>
static __inline__
void zscal_zgeru_device( int m,
                         magmaDoubleComplex_ptr dA, int lda,
                         magma_int_t *info, int step, int gbstep,
                         const sycl::nd_item<3> &item_ct1,
                         magmaDoubleComplex *shared_y)
{
    const int tx = item_ct1.get_local_id(2);
    const int gtx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + tx;

    magmaDoubleComplex rA[N], reg;

    if (tx < N) {
        shared_y[tx] = dA[lda * tx];
    }
    /*
    DPCT1065:146: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    // terminate threads that are out of the range
    if (gtx == 0 || gtx >= m) return;

    double rTmp = sycl::fabs(MAGMA_Z_REAL( shared_y[0] ) ) + sycl::fabs( MAGMA_Z_IMAG( shared_y[0] ) );

    reg = (rTmp == MAGMA_D_ZERO) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, shared_y[0]);

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
static __inline__
void zscal_zgeru_generic_device( int m, int n,
                         magmaDoubleComplex_ptr dA, int lda,
                         magma_int_t *info, int step, int gbstep,
                         const sycl::nd_item<3> &item_ct1)
{
    const int tx = item_ct1.get_local_id(2);
    const int gtx = item_ct1.get_group(2) * item_ct1.get_local_range(2) + tx;
    if (gtx == 0 || gtx >= m) return;

    magmaDoubleComplex rA, reg;
    double rTmp;
    rA   = dA[0];
    rTmp = sycl::fabs(MAGMA_Z_REAL(rA)) + sycl::fabs(MAGMA_Z_IMAG(rA));

    reg = (rTmp == MAGMA_D_ZERO) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, rA);
    rA  = dA[ gtx ];
    rA *= reg;

    dA[ gtx ] = rA;
    #pragma unroll
    for(int i = 1; i < n; i++)
        dA[i * lda + gtx] -= rA * dA[i * lda + 0];
}

/******************************************************************************/
static __inline__
void
zupdate_device(int m, int step, magmaDoubleComplex* x, int ldx,  magmaDoubleComplex *A, int lda,
               const sycl::nd_item<3> &item_ct1)
{
    int tid = item_ct1.get_local_id(2);
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
        /*
        DPCT1065:147: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
    }

    //printf("         @ step %d tid %d adding %5.3f to A %5.3f make it %5.3f\n",step,tid,-reg,A[tid],A[tid]-reg);
}


/******************************************************************************/
static __inline__
void
zscal5_device(int m, magmaDoubleComplex* x, magmaDoubleComplex alpha,
              const sycl::nd_item<3> &item_ct1)
{
    int tid = item_ct1.get_local_id(2);
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
    /*
    DPCT1065:148: Consider replacing sycl::nd_item::barrier() with
    sycl::nd_item::barrier(sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();
}

/******************************************************************************/
template<int WIDTH>
static __inline__
void
zgetf2_fused_device( int m, int minmn, magmaDoubleComplex rA[WIDTH], magma_int_t* dipiv,
                     magmaDoubleComplex* swork, int &linfo, int gbstep, int &rowid,
                     const sycl::nd_item<3> &item_ct1)
{
    const int tx = item_ct1.get_local_id(2);
    const int ty = item_ct1.get_local_id(1);

    magmaDoubleComplex reg       = MAGMA_Z_ZERO;

    int max_id;
    double rx_abs_max = MAGMA_D_ZERO;

    magmaDoubleComplex *sx = (magmaDoubleComplex*)(swork);
    double *dsx = (double *)(sx + item_ct1.get_local_range(1) * WIDTH);
    int *isx = (int *)(dsx + item_ct1.get_local_range(1) * m);
    int *sipiv = (int *)(isx + item_ct1.get_local_range(1) * m);
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
        dsx[ rowid ] = sycl::fabs(MAGMA_Z_REAL( rA[i] )) + sycl::fabs(MAGMA_Z_IMAG( rA[i] ));
        isx[ tx ] = tx;
        /*
        DPCT1065:149: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        magma_getidmax_n(m - i, tx, dsx + i, isx + i,
                         item_ct1); // this devfunc has syncthreads at the end
        rx_abs_max = dsx[i];
        max_id = isx[i];
        linfo  = ( rx_abs_max == MAGMA_D_ZERO && linfo == 0) ? (gbstep+i+1) : linfo;
        if(tx == 0) {
            sipiv[i] = max_id;
        }
        /*
        DPCT1065:150: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if( rowid == max_id ) {
            #pragma unroll
            for(int j = 0; j < WIDTH; j++){
                sx[j] = rA[j];
            }
        }
        /*
        DPCT1065:151: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        if( rx_abs_max != MAGMA_D_ZERO ) {
            if(rowid == max_id){
                rowid = i;
            }
            else if(rowid == i){
                rowid = max_id;
            }
        }
        /*
        DPCT1065:152: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        reg = (rx_abs_max == MAGMA_D_ZERO ) ? MAGMA_Z_ONE : MAGMA_Z_DIV(MAGMA_Z_ONE, sx[i] );
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
