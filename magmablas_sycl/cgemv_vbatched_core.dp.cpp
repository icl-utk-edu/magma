/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong
       @author Azzam Haidar

*/
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "magma_internal.h"
#include "magma_templates.h"

#define PRECISION_c

#include "gemv_template_kernel_vbatched.dp.hpp"
#include "gemv_config/gemvn_param.h"
#include "gemv_config/gemvt_param.h"

#define version(s,v) s ## _V_ ## v

/******************************************************************************/
extern "C" void
magmablas_cgemv_vbatched_max_nocheck(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, 
    magmaFloatComplex alpha,
    magmaFloatComplex_ptr dA_array[], magma_int_t* ldda, 
    magmaFloatComplex_ptr dx_array[], magma_int_t* incx,
    magmaFloatComplex beta,
    magmaFloatComplex_ptr dy_array[], magma_int_t* incy, 
    magma_int_t batchCount, 
    magma_int_t max_m, magma_int_t max_n, magma_queue_t queue)
{
    if ( trans == MagmaNoTrans ) {                                                   
        if (max(max_m, max_n) <= 96) { // small size                         
            if (max_m < max_n) { // Fat matrix
                if ( max_m <= 16) 
                {    
                    gemvn_template_vbatched<magmaFloatComplex, version(N, 70)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else if ( max_m <= 32) 
                {
                    gemvn_template_vbatched<magmaFloatComplex, version(N, 100)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else if ( max_m <= 64)            
                {
                    gemvn_template_vbatched<magmaFloatComplex, version(N, 117)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else
                {
                    gemvn_template_vbatched<magmaFloatComplex, version(N, 131)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }  
            }
            else {  // Tall or square matrix
                if ( max_n <= 32) 
                {    
                    gemvn_template_vbatched<magmaFloatComplex, version(N, 129)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else
                {
                    gemvn_template_vbatched<magmaFloatComplex, version(N, 131)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }  
            }
        }
        else { // big size
            if (max_m < max_n) { // Fat matrix
                if (max_m <= 8)
                {
                    gemvn_template_vbatched<magmaFloatComplex, version(N, 36)>              
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else if (max_m <= 16)
                {
                    gemvn_template_vbatched<magmaFloatComplex, version(N, 70)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else if (max_m <= 32)
                {
                    gemvn_template_vbatched<magmaFloatComplex, version(N, 100)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else if (max_m <= 32)
                {
                    gemvn_template_vbatched<magmaFloatComplex, version(N, 116)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else
                {
                    gemvn_template_vbatched<magmaFloatComplex, version(N, 133)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
            }
            else { // Tall or square matrix
                if (max_m <= 256)
                {
                    gemvn_template_vbatched<magmaFloatComplex, version(N, 137)>             
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else
                {
                    gemvn_template_vbatched<magmaFloatComplex, version(N, 140)>               
                        ( m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
            }
        }// big size        
    } else {
        if (max(max_m, max_n) <= 96) // small size
        {
            if (max_n <= 8)
            {
                gemvc_template_vbatched<magmaFloatComplex, version(T, 42)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
            }
            else
            {
                gemvc_template_vbatched<magmaFloatComplex, version(T, 46)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
            }
        }
        else // big size
        {
            if (max_m <= max_n) //  Fat or square matrix
            {    
                if (max_m <= 64)
                {
                    gemvc_template_vbatched<magmaFloatComplex, version(T, 47)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else
                {
                    gemvc_template_vbatched<magmaFloatComplex, version(T, 90)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
            }                           
            else// (max_m > n) Tall matrix
            {
                if (max_n <= 8)
                {
                    gemvc_template_vbatched<magmaFloatComplex, version(T, 130)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
                else
                {
                    gemvc_template_vbatched<magmaFloatComplex, version(T, 90)>             
                        ( trans, m, n, alpha, dA_array, ldda, dx_array, incx, beta, dy_array, incy, max_m, max_n, batchCount, queue );
                }
            }
        }        
    }
}
