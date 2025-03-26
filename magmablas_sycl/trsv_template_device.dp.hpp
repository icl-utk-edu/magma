#include <sycl/sycl.hpp>
/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Ahmad Abdelfattah
*/

#ifndef TRSV_TEMPLATE_DEVICE_DP_HPP
#define TRSV_TEMPLATE_DEVICE_DP_HPP

///////////////////////////////////////////////////////////////////////////////////////////////////
/* common functions */
///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, const int NB, const int CONJA>
__inline__
void trsv_init_data( int tx, int n,
                     magma_diag_t diag,
                     T* A, int ldda,
                     T* x, int incx,
                     T* sA, int slda,
                     T* sx)
{
    const T c_zero = make_FloatingPoint(0.0, 0.0);
    const T c_one  = make_FloatingPoint(1.0, 0.0);

    // init sA and sx
    if(tx < NB){
        #pragma unroll
        for(int i = 0; i < NB; i++){
            sA[i * slda + tx] = c_zero;
        }
        sA[tx * slda + tx] = c_one;
        sx[tx] = c_zero;
    }

    if( tx < n ){
        // load A
        if(n == NB){
            #pragma unroll
            for(int i = 0; i < NB; i++){
                sA[i * slda + tx] = (CONJA == 0) ? A[i * ldda + tx] : conj(A[i * ldda + tx]);
            }
        }
        else{
            for(int i = 0; i < n; i++){
                sA[i * slda + tx] = (CONJA == 0) ? A[i * ldda + tx] : conj(A[i * ldda + tx]);
            }
        }

        // handle diag
        if(diag == MagmaNonUnit){
            sA[tx * slda + tx] = div(c_one, sA[tx * slda + tx]);
        }else{
            sA[tx * slda + tx] = c_one;
        }

        // load x
        sx[ tx ] = x[ tx * incx ];
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
template<typename T, const int NB>
__inline__
void trsv_write_x( int tx, int n,
                   T*  x, int incx,
                   T* sx )
{
    if(tx < n){
        x[ tx * incx ] = sx[ tx ];
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////////
/* trsv device functions */
///////////////////////////////////////////////////////////////////////////////////////////////////
/*
trsv modes
NL: NoTrans - Lower
NU: NoTrans - Upper
TL: Trans   - Lower
TU: Trans   - Upper
*/
///////////////////////////////////////////////////////////////////////////////////////////////////
// NL
template<typename T, const int NB>
static 
void trsv_template_device_NL(
        magma_diag_t diag, int n,
        T* A, int ldda,
        T* x, int incx, const sycl::nd_item<3> &item_ct1, T *sA, T *sx)
{
#define sA(i, j) sA[(j)*slda + (i)]

    constexpr int slda = NB;
    const int tx = item_ct1.get_local_id(2);

    trsv_init_data<T, NB, 0>(tx, n, diag, A, ldda, x, incx, sA, slda, sx);
    item_ct1.barrier();

    // solve
    #pragma unroll
    for(int i = 0; i < NB; i++) {
        if(tx == 0) {
            sx[i] *= sA(i, i);
        }
        item_ct1.barrier();

        if(tx > i) {
            sx[tx] -= sx[i] * sA(tx, i);
        }
        item_ct1.barrier();
    }

    // write x
    trsv_write_x<T, NB>( tx, n, x, incx, sx );

#undef sA
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// NU
template<typename T, const int NB>
static 
void trsv_template_device_NU(
        magma_diag_t diag, int n,
        T* A, int ldda,
        T* x, int incx, const sycl::nd_item<3> &item_ct1, T *sA, T *sx)
{
#define sA(i, j) sA[(j)*slda + (i)]

    constexpr int slda = NB;
    const int tx = item_ct1.get_local_id(2);

    trsv_init_data<T, NB, 0>(tx, n, diag, A, ldda, x, incx, sA, slda, sx);
    item_ct1.barrier();

    // solve
    #pragma unroll
    for(int i = NB-1; i >= 0; i--) {
        if(tx == 0) {
            sx[i] *= sA(i, i);
        }
        item_ct1.barrier();

        if(tx < i) {
            sx[tx] -= sx[i] * sA(tx, i);
        }
        item_ct1.barrier();
    }

    // write x
    trsv_write_x<T, NB>( tx, n, x, incx, sx );

#undef sA
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// TL, CL
template<typename T, const int NB, const int CONJA>
static 
void trsv_template_device_TL(
        magma_diag_t diag, int n,
        T* A, int ldda,
        T* x, int incx, const sycl::nd_item<3> &item_ct1, T *sA, T *sx)
{
#define sA(i, j) sA[(j)*slda + (i)]

    constexpr int slda = NB;
    const int tx = item_ct1.get_local_id(2);

    trsv_init_data<T, NB, CONJA>(tx, n, diag, A, ldda, x, incx, sA, slda, sx);
    item_ct1.barrier();

    // solve
    #pragma unroll
    for(int i = NB-1; i >= 0; i--) {
        if(tx == 0) {
            sx[i] *= sA(i, i);
        }
        item_ct1.barrier();

        if(tx < i) {
            sx[tx] -= sx[i] * sA(i, tx);
        }
        item_ct1.barrier();
    }

    // write x
    trsv_write_x<T, NB>( tx, n, x, incx, sx );

#undef sA
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// TU, CU
template<typename T, const int NB, const int CONJA>
static 
void trsv_template_device_TU(
        magma_diag_t diag, int n,
        T* A, int ldda,
        T* x, int incx, const sycl::nd_item<3> &item_ct1, T *sA, T *sx)
{
#define sA(i, j) sA[(j)*slda + (i)]

    constexpr int slda = NB;
    const int tx = item_ct1.get_local_id(2);

    trsv_init_data<T, NB, CONJA>(tx, n, diag, A, ldda, x, incx, sA, slda, sx);
    item_ct1.barrier();

    // solve
    #pragma unroll
    for(int i = 0; i < NB; i++) {
        if(tx == 0) {
            sx[i] *= sA(i, i);
        }
        item_ct1.barrier();

        if(tx > i) {
            sx[tx] -= sx[i] * sA(i, tx);
        }
        item_ct1.barrier();
    }

    // write x
    trsv_write_x<T, NB>( tx, n, x, incx, sx );

#undef sA
}

#endif //TRSV_TEMPLATE_DEVICE_DP_HPP
