/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong
       @author Azzam Haidar

*/
#include "magma_internal.h"
#include "magma_templates.h"

#define PRECISION_s

#include "gemv_template_kernel_batched.cuh"
#include "gemv_config/gemvn_param.h"
#include "gemv_config/gemvt_param.h"

#define version(s,v) s ## _V_ ## v


/***************************************************************************//**
    Purpose
    -------
    SGEMV performs one of the matrix-vector operations
    
        y := alpha*A*x    + beta*y,   or
        y := alpha*A**T*x + beta*y,   or
        y := alpha*A**H*x + beta*y,
    
    where alpha and beta are scalars, x and y are vectors and A is an
    m by n matrix.

    Arguments
    ----------
    @param[in]
    trans   magma_trans_t
            On entry, TRANS specifies the operation to be performed as
            follows:
      -     = MagmaNoTrans:    y := alpha*A  *x + beta*y
      -     = MagmaTrans:      y := alpha*A^T*x + beta*y
      -     = MagmaConjTrans:  y := alpha*A^H*x + beta*y

    @param[in]
    m       INTEGER
            On entry, m specifies the number of rows of the matrix A.

    @param[in]
    n       INTEGER
            On entry, n specifies the number of columns of the matrix A
 
    @param[in]
    alpha   REAL
            On entry, ALPHA specifies the scalar alpha.


    @param[in]
    dA_array     Array of pointers, dimension (batchCount).
             Each is a REAL array A of DIMENSION ( ldda, n ) on the GPU
   
    @param[in]
    ldda    INTEGER
            LDDA specifies the leading dimension of A.

    @param[in]
    dx_array     Array of pointers, dimension (batchCount).
            Each is a REAL array of dimension
            n if trans == MagmaNoTrans
            m if trans == MagmaTrans or MagmaConjTrans
     
    @param[in]
    incx    Specifies the increment for the elements of X.
            INCX must not be zero.
  
    @param[in]
    beta    REAL
            On entry, ALPHA specifies the scalar beta. When BETA is
            supplied as zero then Y need not be set on input.

    @param[out]
    dy_array     Array of pointers, dimension (batchCount).
            Each is a REAL array of dimension
            m if trans == MagmaNoTrans
            n if trans == MagmaTrans or MagmaConjTrans

    @param[in]
    incy    Specifies the increment for the elements of Y.
            INCY must not be zero.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    @ingroup magma_gemv_batched
*******************************************************************************/
extern "C" void
magmablas_sgemv_batched(
    magma_trans_t trans, magma_int_t m, magma_int_t n, 
    float alpha,
    magmaFloat_ptr dA_array[], magma_int_t ldda, 
    magmaFloat_ptr dx_array[], magma_int_t incx,
    float beta,
    magmaFloat_ptr dy_array[], magma_int_t incy, 
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t info = 0;
    if ( trans != MagmaNoTrans && trans != MagmaTrans && trans != MagmaConjTrans )
        info = -1;
    else if ( m < 0 )
        info = -2;
    else if ( n < 0 )
        info = -3;
    else if ( ldda < m )
        info = -6;
    else if ( incx == 0 )
        info = -8;
    else if ( incy == 0 )
        info = -11;
    
    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return;  //info;
    }    
                                                                                   
    if ( trans == MagmaNoTrans ) {                                                   
        if (max(m, n) <= 96) { // small size                         
            if (m < n) { // Fat matrix
                if ( m <= 8) 
                {
                    gemvn_template_batched<float, version(N, 32)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else if ( m <= 16)            
                {
                    gemvn_template_batched<float, version(N, 72)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else if ( m <= 32)            
                {
                    gemvn_template_batched<float, version(N, 97)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else if ( m <= 64)            
                {
                    gemvn_template_batched<float, version(N, 120)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else
                {
                    gemvn_template_batched<float, version(N, 130)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }  
            }
            else {   // Tall matrix
                if ( n <= 16) 
                {    
                    gemvn_template_batched<float, version(N, 118)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else 
                {
                    gemvn_template_batched<float, version(N, 120)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }  
            }
        }
        else { // big size
            if (m < n) { // Fat matrix
                if (m <= 16)
                {
                    gemvn_template_batched<float, version(N, 79)>              
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else if (m <= 32)
                {
                    gemvn_template_batched<float, version(N, 103)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else if (m <= 64)
                {
                    gemvn_template_batched<float, version(N, 126)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else
                {
                    gemvn_template_batched<float, version(N, 135)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
            }
            else { // Tall or square matrix
                if (m <= 256)
                {
                    gemvn_template_batched<float, version(N, 137)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else
                {
                    gemvn_template_batched<float, version(N, 140)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
            }
        }// big size        
    }
    else {
        if (max(m, n) <= 96) // small size
        {                              
            gemvc_template_batched<float, version(T, 46)>             
                ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );    
        }
        else // big size
        {
            if (m <= n) //  Fat or square matrix
            {    
                if (m <= 64)
                {
                    gemvc_template_batched<float, version(T, 47)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else
                {
                    gemvc_template_batched<float, version(T, 133)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
            }                           
            else// (m > n) Tall matrix
            {
                if (n <= 8)
                {
                    gemvc_template_batched<float, version(T, 130)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
                else
                {
                    gemvc_template_batched<float, version(T, 131)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, batchCount, queue );
                }
            }
        }        
    }                   
}
