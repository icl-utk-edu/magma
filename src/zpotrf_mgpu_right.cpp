/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
       
       @author Azzam Haidar
       @author Ichi Yamazaki

       @precisions normal z -> s d c

*/
#ifndef MAGMA_NO_V1

#include "magma_internal.h"
#include "trace.h"

#if 0 // needs updating to MAGMA v2 interfaces

/***************************************************************************//**
    Purpose
    -------
    ZPOTRF computes the Cholesky factorization of a complex Hermitian
    positive definite matrix dA.

    The factorization has the form
       dA = U**H * U,   if UPLO = MagmaUpper, or
       dA = L  * L**H,  if UPLO = MagmaLower,
    where U is an upper triangular matrix and L is lower triangular.

    This is the block version of the algorithm, calling Level 3 BLAS.

    Arguments
    ---------
    @param[in]
    ngpu    INTEGER
            Number of GPUs to use. ngpu > 0.

    @param[in]
    uplo    magma_uplo_t
      -     = MagmaUpper:  Upper triangle of dA is stored;
      -     = MagmaLower:  Lower triangle of dA is stored.

    @param[in]
    n       INTEGER
            The order of the matrix dA.  N >= 0.

    @param[in,out]
    d_lA    COMPLEX_16 array of pointers on the GPU, dimension (ngpu)
            On entry, the Hermitian matrix dA distributed over GPUs
            (dl_A[d] points to the local matrix on the d-th GPU).
            It is distributed in 1D block column or row cyclic (with the
            block size of nb) if UPLO = MagmaUpper or MagmaLower, respectively.
            If UPLO = MagmaUpper, the leading N-by-N upper triangular
            part of dA contains the upper triangular part of the matrix dA,
            and the strictly lower triangular part of dA is not referenced.
            If UPLO = MagmaLower, the leading N-by-N lower triangular part
            of dA contains the lower triangular part of the matrix dA, and
            the strictly upper triangular part of dA is not referenced.
    \n
            On exit, if INFO = 0, the factor U or L from the Cholesky
            factorization dA = U**H * U or dA = L * L**H.

    @param[in]
    ldda     INTEGER
            The leading dimension of the array dA.  LDDA >= max(1,N).
            To benefit from coalescent memory accesses LDDA must be
            divisible by 16.

    @param[out]
    info    INTEGER
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
      -     > 0:  if INFO = i, the leading minor of order i is not
                  positive definite, and the factorization could not be
                  completed.

    @ingroup magma_potrf
*******************************************************************************/
extern "C" magma_int_t
magma_zpotrf_mgpu_right(
    magma_int_t ngpu,
    magma_uplo_t uplo, magma_int_t n,
    magmaDoubleComplex_ptr d_lA[], magma_int_t ldda,
    magma_int_t *info )
{
    #define dlA(id, i, j)  (d_lA[(id)] + (j) * ldda + (i))
    #define dlP(id, i, j)  (d_lP[(id)] + (j) * ldda + (i))

    #define panel(j)  (panel + (j))
    #define tmppanel(j)  (tmppanel + (j))
    #define tmpprevpanel(j)  (tmpprevpanel + (j))
    #define STREAM_ID(i) (nqueue > 1 ? 1+((i)/nb)%(nqueue-1) : 0)

    magmaDoubleComplex c_one     = MAGMA_Z_ONE;
    magmaDoubleComplex c_neg_one = MAGMA_Z_NEG_ONE;
    double             d_one     =  1.0;
    double             d_neg_one = -1.0;
    const char* uplo_ = lapack_uplo_const( uplo );

    magma_int_t j, nb, d, id, j_local, blkid, crosspoint, prevtrsmrows=0, nqueue = 5;
    magmaDoubleComplex *panel, *tmppanel0, *tmppanel1, *tmppanel, *tmpprevpanel;
    magmaDoubleComplex *d_lP[MagmaMaxGPUs], *dlpanel, *dlpanels[MagmaMaxGPUs];
    magma_int_t rows, trsmrows, igpu, n_local[MagmaMaxGPUs], ldpanel;
    magma_queue_t queues[MagmaMaxGPUs][10];

    *info = 0;
    if ( uplo != MagmaUpper && uplo != MagmaLower ) {
        *info = -1;
    } else if (n < 0) {
        *info = -2;
    } else if (ldda < max(1,n)) {
        *info = -4;
    }
    if (*info != 0) {
        magma_xerbla( __func__, -(*info) );
        return *info;
    }

    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );
    magma_queue_t orig_stream;
    magmablasGetKernelStream( &orig_stream );

    nb = magma_get_zpotrf_nb(n);

    ldpanel = ldda;
    magma_setdevice(0);
    if (MAGMA_SUCCESS != magma_zmalloc_pinned( &panel, 2 * nb * ldpanel )) {
        *info = MAGMA_ERR_HOST_ALLOC;
        return *info;
    }

    tmppanel0 = panel;
    tmppanel1 = tmppanel0 + nb * ldpanel;

    if ((nb <= 1) || (nb >= n)) {
        // Use unblocked code.
        magma_zgetmatrix( n, n, dlA(0, 0, 0), ldda, panel, ldpanel);
        lapackf77_zpotrf( uplo_, &n, panel, &ldpanel, info);
        magma_zsetmatrix( n, n, panel, ldpanel, dlA(0, 0, 0), ldda );
    } else {
        for( d = 0; d < ngpu; d++ ) {
            // local-n and local-ld
            n_local[d] = ((n / nb) / ngpu) * nb;
            if (d < (n / nb) % ngpu)
                n_local[d] += nb;
            else if (d == (n / nb) % ngpu)
                n_local[d] += n % nb;

            magma_setdevice(d);
            magma_device_sync();
            if (MAGMA_SUCCESS != magma_zmalloc( &d_lP[d], nb * ldda )) {
                for( j = 0; j < d; j++ ) {
                    magma_setdevice(j);
                    magma_free( d_lP[d] );
                }
                *info = MAGMA_ERR_DEVICE_ALLOC;
                return *info;
            }
            for( j=0; j < nqueue; j++ ) {
                magma_queue_create( &queues[d][j] );
            }
        }

        //#define ENABLE_TIMER
        #if defined (ENABLE_TIMER)
        real_Double_t therk[4], tmtc, tcchol, tctrsm, tctm, tmnp, tcnp;
        real_Double_t ttot_herk[4] = {0,0,0,0}, ttot_mtc = 0, ttot_cchol = 0, ttot_ctrsm = 0, ttot_ctm = 0, ttot_mnp = 0, ttot_cnp = 0;
        printf("\n\n %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s %10s\n",
                "j", "nb", "row", "mtc", "CPU_np", "panel", "ctrsm", "CH+TRSM", "CPU", "dsyrk[0]", "dsyrk[1]", "dsyrk[2]", "dsyrk[3]", "ctm P", "gpu_np");
        printf("     ====================================================================================================\n");
        #endif

        // Use blocked code.
        if (uplo == MagmaUpper) {
            printf( " === not supported, yet ===\n" );
        } else {
            blkid = -1;
            if (ngpu == 4)
                crosspoint = n;
            else if (ngpu == 3)
                crosspoint = n;
            else if (ngpu == 2)
                crosspoint = 20160;
            else
                crosspoint = 0;
            crosspoint = 0; //n; //n -- > gpu always does next panel, 0 --> cpu always does next panel
            crosspoint = n;

            #if defined (ENABLE_TIMER)
            real_Double_t tget = magma_wtime(), tset = 0.0, ttot = 0.0;
            #endif
            if ( n > nb ) {
                // send first panel to cpu
                magma_setdevice(0);
                tmppanel = tmppanel0;
                magma_zgetmatrix_async(n, nb,
                        dlA(0, 0, 0), ldda,
                        tmppanel(0),  ldpanel,
                        queues[0][0] );
            }
            #if defined (ENABLE_TIMER)
            for( d=0; d < ngpu; d++ ) {
                magma_setdevice(d);
                magma_device_sync();
            }
            tget = magma_wtime()-tget;
            #endif

            // Compute the Cholesky factorization A = L*L'
            for (j = 0; (j + nb) < n; j += nb) {
                #if defined (ENABLE_TIMER)
                therk[0] = therk[1] = therk[2] = therk[3] = tmtc = tcchol = tctrsm = tctm = tmnp = tcnp = 0.0;
                #endif

                blkid += 1;
                tmppanel = (blkid % 2 == 0) ? tmppanel0 : tmppanel1;
                // Set the gpu number that holds the current panel
                id = (j / nb) % ngpu;
                magma_setdevice(id);

                // Set the local index where the current panel is
                j_local = j / (nb * ngpu) * nb;
                
                rows = n - j;
                // Wait for the panel on cpu
                magma_queue_sync( queues[id][0] );
                if (j > 0 && prevtrsmrows > crosspoint) {
                    #if defined (ENABLE_TIMER)
                    tcnp = magma_wtime();
                    #endif

                    tmpprevpanel = ((blkid - 1) % 2) == 0 ? tmppanel0 : tmppanel1;

                    blasf77_zgemm( MagmaNoTransStr, MagmaConjTransStr,
                            &rows, &nb, &nb,
                            &c_neg_one, tmpprevpanel(j), &ldpanel,
                                        tmpprevpanel(j), &ldpanel,
                            &c_one,     tmppanel(j),     &ldpanel );

                    #if defined (ENABLE_TIMER)
                    tcnp = magma_wtime() - tcnp;
                    ttot_cnp += tcnp;
                    #endif
                }

                #if defined (ENABLE_TIMER)
                tcchol = magma_wtime();
                #endif
                lapackf77_zpotrf(MagmaLowerStr, &nb, tmppanel(j), &ldpanel, info);
                if (*info != 0) {
                    *info = *info + j;
                    break;
                }

                #if defined (ENABLE_TIMER)
                tcchol = magma_wtime() - tcchol;
                ttot_cchol += tcchol;
                tctrsm = magma_wtime();
                #endif

                trsmrows = rows - nb;

                if (trsmrows > 0) {
                    blasf77_ztrsm(MagmaRightStr, MagmaLowerStr, MagmaConjTransStr, MagmaNonUnitStr,
                                  &trsmrows, &nb,
                                  &c_one, tmppanel(j), &ldpanel,
                                          tmppanel(j + nb), &ldpanel);
                }

                #if defined (ENABLE_TIMER)
                tctrsm = magma_wtime() - tctrsm;
                ttot_ctrsm += tctrsm;
                tctm = magma_wtime();
                #endif

                d = (id + 1) % ngpu;
                // send current panel to gpus
                for (igpu = 0; igpu < ngpu; igpu++, d = (d + 1) % ngpu ) {
                    magma_int_t myrows = 0;
                    magma_int_t row_offset = 0;
                    if ( d == id ) {
                        dlpanel = dlA(d, j, j_local);
                        myrows = rows;
                        row_offset = 0;
                    } else {
                        dlpanel = dlP(d, 0, 0);
                        myrows = trsmrows;
                        row_offset = nb;
                    }

                    if (myrows > 0) {
                        magma_setdevice(d);
                        magma_zsetmatrix_async(myrows, nb,
                                tmppanel(j + row_offset),    ldpanel,
                                dlpanel, ldda, queues[d][0] );
                    }
                }
                /* make sure panel is on GPUs */
                d = (id + 1) % ngpu;
                for (igpu = 0; igpu < ngpu; igpu++, d = (d + 1) % ngpu ) {
                    magma_setdevice(d);
                    magma_queue_sync( queues[d][0] );
                }

                #if defined (ENABLE_TIMER)
                tctm = magma_wtime() - tctm;
                ttot_ctm += tctm;
                #endif

                if ( (j + nb) < n) {
                    magma_int_t offset = 0;
                    magma_int_t row_offset = 0;
                    if (j + nb + nb < n) {
                        d = (id + 1) % ngpu;
                        magma_setdevice(d);
                        magma_int_t j_local2 = (j + nb) / (nb * ngpu) * nb;
                        if (trsmrows <= crosspoint) {
                            #if defined (ENABLE_TIMER)
                            tmnp = magma_wtime();
                            #endif

                            // do gemm on look ahead panel
                            if ( d == id ) {
                                dlpanel = dlA(d, j + nb, j_local);
                            } else {
                                dlpanel = dlP(d, 0, 0);
                            }

                            magmablasSetKernelStream( queues[d][STREAM_ID(j_local2)] );
                            #define ZHERK_ON_DIAG
                            #ifdef  ZHERK_ON_DIAG
                            magma_zherk( MagmaLower, MagmaNoTrans,
                                         nb, nb,
                                         d_neg_one, dlpanel, ldda,
                                         d_one,     dlA(d, j + nb, j_local2), ldda);
                            magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                         trsmrows-nb, nb, nb,
                                         c_neg_one, dlpanel+nb, ldda,
                                                    dlpanel,    ldda,
                                         c_one,     dlA(d, j + nb +nb, j_local2), ldda);
                            #else
                            magma_zgemm( MagmaNoTrans, MagmaConjTrans,
                                         trsmrows, nb, nb,
                                         c_neg_one, dlpanel, ldda,
                                                    dlpanel, ldda,
                                         c_one,     dlA(d, j + nb, j_local2), ldda);
                            #endif

                            #if defined (ENABLE_TIMER)
                            magma_device_sync();
                            tmnp = magma_wtime() - tmnp;
                            ttot_mnp += tmnp;
                            #endif
                        }
                        // send next panel to cpu
                        magma_queue_sync( queues[d][STREAM_ID(j_local2)] ); // make sure lookahead is done
                        tmppanel = ((blkid+1) % 2 == 0) ? tmppanel0 : tmppanel1;
                        magma_zgetmatrix_async(rows-nb, nb,
                                dlA(d, j+nb, j_local2), ldda,
                                tmppanel(j+nb),  ldpanel,
                                queues[d][0] );
                        tmppanel = (blkid % 2 == 0) ? tmppanel0 : tmppanel1;

                        offset = j + nb + nb;
                        row_offset = nb;
                    } else {
                        offset = j + nb;
                        row_offset = 0;
                    }

                    if (n - offset > 0) {
                        // syrk on multiple gpu
                        for (d = 0; d < ngpu; d++ ) {
                            if ( d == id ) {
                                dlpanels[d] = dlA(d, j + nb + row_offset, j_local);
                            } else {
                                dlpanels[d] = dlP(d, row_offset, 0);
                            }
                        }

                        #if defined (ENABLE_TIMER)
                        for( d=0; d < ngpu; d++ ) {
                            therk[d] = magma_wtime();
                        }
                        #endif

                        //magmablasSetKernelStream( queues[d] );
                        //magma_zherk( MagmaLower, MagmaNoTrans, n - offset, nb,
                        //             d_neg_one, dlpanel, ldda,
                        //             d_one,     &d_lA[d][offset + offset*ldda], ldda );
                        #ifdef  ZHERK_ON_DIAG
                        magma_zherk_mgpu
                        #else
                        magma_zherk_mgpu2
                        #endif
                                        (ngpu, MagmaLower, MagmaNoTrans,
                                         nb, n - offset, nb,
                                         d_neg_one, dlpanels, ldda, 0,
                                         d_one,     d_lA,     ldda, offset,
                                         nqueue, queues );
                        #if defined (ENABLE_TIMER)
                        for( d=0; d < ngpu; d++ ) {
                            magma_setdevice(d);
                            magma_device_sync();
                            therk[d] = magma_wtime() - therk[d];
                            ttot_herk[d] += therk[d];
                        }
                        #endif
                    }

                    prevtrsmrows = trsmrows;

                    #if defined (ENABLE_TIMER)
                    ttot += (tcnp + tcchol + tctrsm + therk[0] + therk[1] + therk[2] + tctm + tmnp);
                    printf("%10lld %10lld %10lld %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf(%lld) %10.3lf\n",
                            (long long) j, (long long) nb, (long long) rows, tmtc,
                            tcnp,     // gemm
                            tcchol,   // potrf
                            tctrsm,   // trsm
                            (tcchol + tctrsm),
                            (tmtc + tcnp + tcchol + tctrsm),
                            therk[0], therk[1], therk[2], therk[3], // syrk
                            tctm, // copy panel to GPU
                            tmnp, // lookahead on GPU
                            (long long)((id + 1) % ngpu),
                            (tcnp + tcchol + tctrsm + therk[0] + therk[1] + therk[2] + tctm + tmnp));
                    fflush(0);
                    #endif
                }
            }
            for( d = 0; d < ngpu; d++ ) {
                magma_setdevice(d);
                for( id=0; id < nqueue; id++ ) {
                    magma_queue_sync( queues[d][id] );
                }
            }
            #if defined (ENABLE_TIMER)
            printf("\n%10lld %10lld %10lld %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf(-) %10.3lf\n",
                    (long long) n, (long long) n, (long long) 0, ttot_mtc,
                    ttot_cnp,     // gemm
                    ttot_cchol,   // potrf
                    ttot_ctrsm,   // trsm
                    (ttot_cchol + ttot_ctrsm),
                    (ttot_mtc + ttot_cnp + ttot_cchol + ttot_ctrsm),
                    ttot_herk[0], ttot_herk[1], ttot_herk[2], ttot_herk[3], // syrk
                    ttot_ctm, // copy panel to GPU
                    ttot_mnp, // lookahead on GPU
                    (ttot_cnp + ttot_cchol + ttot_ctrsm + ttot_herk[0] + ttot_herk[1] + ttot_herk[2] + ttot_ctm + ttot_mnp));
            printf("%10lld %10lld %10lld %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf %10.3lf(-) %10.3lf (ratio)\n",
                    (long long) n, (long long) n, (long long) 0, ttot_mtc/ttot,
                    ttot_cnp/ttot,     // gemm
                    ttot_cchol/ttot,   // potrf
                    ttot_ctrsm/ttot,   // trsm
                    (ttot_cchol + ttot_ctrsm)/ttot,
                    (ttot_mtc + ttot_cnp + ttot_cchol + ttot_ctrsm)/ttot,
                    ttot_herk[0]/ttot, ttot_herk[1]/ttot, ttot_herk[2]/ttot, ttot_herk[3]/ttot, // syrk
                    ttot_ctm/ttot, // copy panel to GPU
                    ttot_mnp/ttot, // lookahead on GPU
                    (ttot_cnp + ttot_cchol + ttot_ctrsm + ttot_herk[0] + ttot_herk[1] + ttot_herk[2] + ttot_ctm + ttot_mnp)/ttot);
            #endif

            // cholesky for the last block
            if (j < n && *info == 0) {
                rows = n - j;
                id = (j / nb) % ngpu;

                // Set the local index where the current panel is
                j_local = j / (nb * ngpu) * nb;
                
                magma_setdevice(id);
                #if defined (ENABLE_TIMER)
                tset = magma_wtime();
                #endif
                magma_zgetmatrix(rows, rows, dlA(id, j, j_local), ldda, panel(j), ldpanel);
                lapackf77_zpotrf(MagmaLowerStr, &rows, panel(j), &ldpanel, info);
                magma_zsetmatrix(rows, rows, panel(j), ldpanel, dlA(id, j, j_local), ldda);
                #if defined (ENABLE_TIMER)
                tset = magma_wtime() - tset;
                #endif
            }
            #if defined (ENABLE_TIMER)
            printf( " matrix_get,set: %10.3lf %10.3lf -> %10.3lf\n", tget, tset, ttot + tget + tset );
            #endif
        } // end of else not upper

        // clean up
        for( d = 0; d < ngpu; d++ ) {
            magma_setdevice(d);
            for( j=0; j < nqueue; j++ ) {
                magma_queue_destroy( queues[d][j] );
            }
            magma_free( d_lP[d] );
        }
    } // end of not lapack

    // free workspace
    magma_free_pinned( panel );
    magma_setdevice( orig_dev );
    magmablasSetKernelStream( orig_stream );

    return *info;
} /* magma_zpotrf_mgpu_right */

#undef dlA
#undef dlP
#undef panel
#undef tmppanel
#undef tmpprevpanel
#undef STREAM_ID

#endif // needs updating to MAGMA v2 interfaces
#endif // MAGMA_NO_V1
