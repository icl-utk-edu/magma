/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

*/
#include "magma_internal.h"

#include "trace.h"
#include "magma_timer.h"
//#include "../testing/flops.h"

/***************************************************************************//**
    Purpose
    -------
    ZGETRF computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
       A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.
    Use two buffer to send panels.

    Arguments
    ---------
    @param[in]
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    m       INTEGER
            The number of rows of the matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of the matrix A.  N >= 0.

    @param[in]
    nb      INTEGER
            The block size used for the matrix distribution.

    @param[in]
    offset  INTEGER
            The first row and column indices of the submatrix that
            this routine will factorize.

    @param[in,out]
    d_lAT   COMPLEX_16 array of pointers on the GPU, dimension (ngpu).
            On entry, the M-by-N matrix A distributed over GPUs
            (d_lAT[d] points to the local matrix on d-th GPU).
            It uses a 1D block column cyclic format (with the block size
            nb), and each local matrix is stored by row.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    lddat   INTEGER
            The leading dimension of the array d_lAT[d]. LDDA >= max(1,M).

    @param[out]
    ipiv    INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[in] 
    d_lAP   COMPLEX_16 array of pointers on the GPU, dimension (ngpu).
            d_lAP[d] is the workspace on d-th GPU. Each local workspace
            must be of size (3+ngpu)*nb*maxm, where maxm is m rounded
            up to a multiple of 32 and nb is the block size.

    @param[in] 
    W       COMPLEX_16 array, dimension (ngpu*nb*maxm).
            It is used to store panel on CPU.

    @param[in]
    ldw     INTEGER
            The leading dimension of the workspace w.

    @param[in]
    queues  magma_queue_t
            queues[d] points to the queues for the d-th GPU to execute
            in. Each GPU require two queues.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @ingroup magma_getrf
*******************************************************************************/
extern "C" magma_int_t
magma_zgetrf2_mgpu(
    magma_int_t ngpu,
    magma_int_t m, magma_int_t n, magma_int_t nb, magma_int_t offset,
    magmaDoubleComplex_ptr d_lAT[], magma_int_t lddat, magma_int_t *ipiv,
    magmaDoubleComplex_ptr d_lAP[],
    magmaDoubleComplex *W, magma_int_t ldw,
    magma_queue_t queues[][2],
    magma_int_t *info)
{
#define dAT(id,i,j)  (d_lAT[(id)] + ((offset)+(i)*nb)*lddat + (j)*nb)
#define W(j) (W + ((j)%ngpu)*nb*ldw)

    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;

    magma_int_t block_size = 32;
    magma_int_t iinfo, n_local[MagmaMaxGPUs];
    magma_int_t maxm, mindim;
    magma_int_t i, j, d, dd, rows, cols, s, ldpan[MagmaMaxGPUs];
    magma_int_t id, j_local, j_local2, nb0, nb1, h = 2+ngpu;
    magmaDoubleComplex *d_panel[MagmaMaxGPUs], *panel_local[MagmaMaxGPUs];

    /* Check arguments */
    *info = 0;
    if (m < 0)
        *info = -2;
    else if (n < 0)
        *info = -3;
    else if (ngpu*lddat < max(1,n))
        *info = -5;

    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    /* Quick return if possible */
    if (m == 0 || n == 0)
        return *info;

    /* Function Body */
    mindim = min(m, n);
    if ( ngpu > magma_ceildiv( n, nb )) {
        *info = -1;
        return *info;
    }

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    
    /* Use hybrid blocked code. */
    maxm  = magma_roundup( m, block_size );

    /* some initializations */
    for (d=0; d < ngpu; d++) {
        magma_setdevice(d);
        
        n_local[d] = ((n/nb)/ngpu)*nb;
        if (d < (n/nb) % ngpu)
            n_local[d] += nb;
        else if (d == (n/nb) % ngpu)
            n_local[d] += n % nb;
        
        /* workspaces */
        d_panel[d] = &(d_lAP[d][h*nb*maxm]);   /* temporary panel storage */
    }
    trace_init( 1, ngpu, 2, (magma_queue_t *)queues );

    /* start sending the panel to cpu */
    nb0 = min(mindim, nb);
    magma_setdevice(0);
    trace_gpu_start( 0, 1, "comm", "get" );
    magmablas_ztranspose( nb0, m, dAT(0,0,0), lddat, d_lAP[0], maxm, queues[0][1] );
    magma_zgetmatrix_async( m, nb0,
                            d_lAP[0], maxm,
                            W(0),     ldw, 
                            queues[0][1] );
    trace_gpu_end( 0, 1 );

    /* ---------------------------------------------------------------------- */
    magma_timer_t time=0;
    timer_start( time );

    s = mindim / nb;
    for( j=0; j < s; j++ ) {
        /* Set the GPU number that holds the current panel */
        id = j % ngpu;
        magma_setdevice(id);
        
        /* Set the local index where the current panel is */
        j_local = j/ngpu;
        cols  = maxm - j*nb;
        rows  = m - j*nb;
        
        /* synchronize j-th panel from id-th gpu into work */
        magma_queue_sync( queues[id][1] );
        
        /* j-th panel factorization */
        trace_cpu_start( 0, "getrf", "getrf" );
        lapackf77_zgetrf( &rows, &nb, W(j), &ldw, ipiv+j*nb, &iinfo);
        if ( (*info == 0) && (iinfo > 0) ) {
            *info = iinfo + j*nb;
        }
        trace_cpu_end( 0 );
        
        /* start sending the panel to all the gpus */
        d = (j+1) % ngpu;
        for( dd=0; dd < ngpu; dd++ ) {
            magma_setdevice(d);
            trace_gpu_start( 0, 1, "comm", "set" );
            magma_zsetmatrix_async( rows, nb,
                                    W(j),     ldw,
                                    &d_lAP[d][(j%h)*nb*maxm], cols,
                                    queues[d][1] );
            trace_gpu_end( 0, 1 );
            d = (d+1) % ngpu;
        }
        
        /* apply the pivoting */
        d = (j+1) % ngpu;
        for( dd=0; dd < ngpu; dd++ ) {
            magma_setdevice(d);
            
            trace_gpu_start( d, 1, "pivot", "pivot" );
            if ( dd == 0 ) {
                for( i=j*nb; i < j*nb + nb; ++i ) {
                    ipiv[i] += j*nb;
                }
            }
            magmablas_zlaswp( lddat, dAT(d,0,0), lddat, j*nb + 1, j*nb + nb, ipiv, 1, 
                              queues[d][0] );
            trace_gpu_end( d, 1 );
            d = (d+1) % ngpu;
        }
    
    
        /* update the trailing-matrix/look-ahead */
        d = (j+1) % ngpu;
        for( dd=0; dd < ngpu; dd++ ) {
            magma_setdevice(d);
            
            /* storage for panel */
            if ( d == id ) {
                /* the panel belongs to this gpu */
                panel_local[d] = dAT(d,j,j_local);
                ldpan[d] = lddat;
                /* next column */
                j_local2 = j_local+1;
            } else {
                /* the panel belongs to another gpu */
                panel_local[d] = d_panel[d];
                ldpan[d] = nb;
                /* next column */
                j_local2 = j_local;
                if ( d < id ) j_local2 ++;
            }
            /* the size of the next column */
            if ( s > (j+1) ) {
                nb0 = nb;
            } else {
                nb0 = n_local[d]-nb*(s/ngpu);
                if ( d < s % ngpu ) nb0 -= nb;
            }
            magma_queue_t queue = NULL;
            if ( d == (j+1) % ngpu) {
                /* owns the next column, look-ahead the column */
                nb1 = nb0;
                queue = queues[d][1];
                
                /* make sure all the pivoting has been applied */
                magma_queue_sync(queues[d][0]);
                trace_gpu_start( d, 1, "gemm", "gemm" );
                /* transpose panel on GPU */
                magmablas_ztranspose( rows, nb, &d_lAP[d][(j%h)*nb*maxm], cols, panel_local[d], ldpan[d], 
                                      queue );
                /* sync for remaining update */
                magma_queue_sync(queues[d][1]);
            } else {
                /* update the entire trailing matrix */
                nb1 = n_local[d] - j_local2*nb;
                queue = queues[d][0];
                
                /* synchronization to make sure panel arrived on gpu */
                magma_queue_sync(queues[d][1]);
                trace_gpu_start( d, 0, "gemm", "gemm" );
                /* transpose panel on GPU */
                magmablas_ztranspose( rows, nb, &d_lAP[d][(j%h)*nb*maxm], cols, panel_local[d], ldpan[d], 
                                      queue );
            }
            
            /* gpu updating the trailing matrix */
            magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                         nb1, nb, c_one,
                         panel_local[d],       ldpan[d],
                         dAT(d, j, j_local2), lddat,
                         queue );
            magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                         nb1, m-(j+1)*nb, nb,
                         c_neg_one, dAT(d, j,   j_local2),         lddat,
                                    &(panel_local[d][nb*ldpan[d]]), ldpan[d],
                         c_one,     dAT(d, j+1, j_local2),         lddat,
                         queue );
        
            if ( d == (j+1) % ngpu ) {
                /* Set the local index where the current panel is */
                magma_int_t loff    = j+1;
                magma_int_t ldda    = maxm - (j+1)*nb;
                magma_int_t j_local_lookahead = (j+1)/ngpu;
                magma_int_t cols_lookahead    = m - (j+1)*nb;
                nb0 = min(nb, mindim - (j+1)*nb); /* size of the diagonal block */
                trace_gpu_end( d, 1 );
                
                if ( nb0 > 0 ) {
                    /* transpose the panel for sending it to cpu */
                    trace_gpu_start( d, 1, "comm", "get" );
                    magmablas_ztranspose( nb0, m-(j+1)*nb, dAT(d,loff,j_local_lookahead), lddat, 
                                          &d_lAP[d][((j+1)%h)*nb*maxm], ldda,
                                          queue );
             
                    /* send the panel to cpu */
                    magma_zgetmatrix_async( cols_lookahead, nb0,
                                            &d_lAP[d][((j+1)%h)*nb*maxm], ldda,
                                            W(j+1), ldw, 
                                            queues[d][1] );

                    trace_gpu_end( d, 1 );
                }
            } else {
                trace_gpu_end( d, 0 );
            }
            
            d = (d+1) % ngpu;
        }
    
        /* update the remaining matrix by gpu owning the next panel */
        if ( (j+1) < s ) {
            magma_int_t j_local_rest = (j+1)/ngpu;
            magma_int_t rows_rest  = m - (j+1)*nb;
            
            d = (j+1) % ngpu;
            magma_setdevice(d);
            trace_gpu_start( d, 0, "gemm", "gemm" );

            magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                         n_local[d] - (j_local_rest+1)*nb, nb,
                         c_one, panel_local[d],       ldpan[d],
                                dAT(d,j,j_local_rest+1),  lddat,
                        queues[d][0] );
            magma_zgemm( MagmaNoTrans, MagmaNoTrans,
                         n_local[d]-(j_local_rest+1)*nb, rows_rest, nb,
                         c_neg_one, dAT(d,j,j_local_rest+1),            lddat,
                                    &(panel_local[d][nb*ldpan[d]]), ldpan[d],
                         c_one,     dAT(d,j+1,  j_local_rest+1),        lddat,
                         queues[d][0] );
            trace_gpu_end( d, 0 );
        }
    } /* end of for j=1..s */
    /* ---------------------------------------------------------------------- */
    
    /* Set the GPU number that holds the last panel */
    id = s % ngpu;
    
    /* Set the local index where the last panel is */
    j_local = s/ngpu;
    
    /* size of the last diagonal-block */
    nb0 = min(m - s*nb, n - s*nb);
    rows = m    - s*nb;
    cols = maxm - s*nb;
    
    if ( nb0 > 0 ) {
        magma_setdevice(id);
        
        /* wait for the last panel on cpu */
        magma_queue_sync( queues[id][1] );
    
        /* factor on cpu */
        lapackf77_zgetrf( &rows, &nb0, W(s), &ldw, ipiv+s*nb, &iinfo);
        if ( (*info == 0) && (iinfo > 0) )
            *info = iinfo + s*nb;
        
        /* send the factor to gpus */
        for( d=0; d < ngpu; d++ ) {
            magma_setdevice(d);
            j_local2 = j_local;
            if ( d < id ) j_local2 ++;
            
            if ( d == id || n_local[d] > j_local2*nb ) {
                magma_zsetmatrix_async( rows, nb0,
                                        W(s), ldw,
                                        &d_lAP[d][(s%h)*nb*maxm], cols, 
                                        queues[d][1] );
            }
        }
        
        for( d=0; d < ngpu; d++ ) {
            magma_setdevice(d);
            if ( d == 0 ) {
                for( i=s*nb; i < s*nb + nb0; ++i ) {
                    ipiv[i] += s*nb;
                }
            }
            magmablas_zlaswp( lddat, dAT(d,0,0), lddat, s*nb + 1, s*nb + nb0, ipiv, 1, 
                              queues[d][0] );
        }
        
        for( d=0; d < ngpu; d++ ) {
            magma_setdevice(d);
            
            /* wait for the pivoting to be done */
            magma_queue_sync( queues[d][0] );
            
            j_local2 = j_local;
            if ( d < id ) j_local2++;
            if ( d == id ) {
                /* the panel belongs to this gpu */
                panel_local[d] = dAT(d,s,j_local);
                
                /* next column */
                nb1 = n_local[d] - j_local*nb-nb0;
                
                magmablas_ztranspose( rows, nb0, &d_lAP[d][(s%h)*nb*maxm], cols, panel_local[d], lddat,
                                      queues[d][1] );
                
                if ( nb1 > 0 ) {
                    magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                                 nb1, nb0, c_one,
                                 panel_local[d],        lddat,
                                 dAT(d,s,j_local)+nb0, lddat,
                                 queues[d][1] );
                }
            } else if ( n_local[d] > j_local2*nb ) {
                /* the panel belongs to another gpu */
                panel_local[d] = d_panel[d];
                
                /* next column */
                nb1 = n_local[d] - j_local2*nb;
                
                magmablas_ztranspose( rows, nb0, &d_lAP[d][(s%h)*nb*maxm], cols, panel_local[d], nb, 
                                      queues[d][1] );
                magma_ztrsm( MagmaRight, MagmaUpper, MagmaNoTrans, MagmaUnit,
                             nb1, nb0, c_one,
                             panel_local[d],     nb,
                             dAT(d,s,j_local2), lddat,
                             queues[d][1] );
            }
        }
    } /* if ( nb0 > 0 ) */
    
    /* clean up */
    trace_finalize( "zgetrf_mgpu.svg","trace.css" );
    for( d=0; d < ngpu; d++ ) {
        magma_setdevice(d);
        magma_queue_sync( queues[d][0] );
        magma_queue_sync( queues[d][1] );
    }
    magma_setdevice( orig_dev );
    
    timer_start( time );
    //timer_printf("\n Performance %f GFlop/s\n", FLOPS_ZGETRF(m,n) / 1e9 / time );

    return *info;
} /* magma_zgetrf2_mgpu */

#undef dAT
