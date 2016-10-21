/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Mark Gates
       @author Azzam Haidar
*/
#include <cuda_runtime.h>

#include "magma_internal.h"
#include "magma_bulge.h"
//#include "trace.h"

/***************************************************************************//**
    Purpose
    -------
    ZHEMM performs one of the matrix-matrix operations

        C := alpha*A*B + beta*C,
    or
        C := alpha*B*A + beta*C,

    where alpha and beta are scalars, A is a Hermitian matrix, and
    B and C are m by n matrices.

    Arguments
    ---------
    \param[in]
    side    magma_side_t
            On entry, SIDE specifies whether the Hermitian matrix A
            appears on the left or right in the operation as follows:

            SIDE = MagmaLeft    C := alpha*A*B + beta*C,

            SIDE = MagmaRight   C := alpha*B*A + beta*C.

            *** Currently, only MagmaLeft is implemented ***

    \param[in]
    uplo    magma_uplo_t
            On entry, UPLO specifies whether the upper or lower
            triangular part of the Hermitian matrix A is to be
            referenced as follows:

            UPLO = MagmaUpper   Only the upper triangular part of the
                                Hermitian matrix is to be referenced.

            UPLO = MagmaLower   Only the lower triangular part of the
                                Hermitian matrix is to be referenced.

            *** Currently, only MagmaLower is implemented ***

    \param[in]
    m       INTEGER
            On entry, M specifies the number of rows of the matrix dC.
            M >= 0.

    \param[in]
    n       INTEGER
            On entry, N specifies the number of columns of the matrix dC.
            N >= 0.

    \param[in]
    alpha   COMPLEX*16
            On entry, ALPHA specifies the scalar alpha.

    \param[in]
    dA      COMPLEX*16 array of DIMENSION ( LDDA, ka ), where ka is
            m when SIDE = MagmaLower and is n otherwise.
            Before entry with SIDE = MagmaLeft, the m by m part of
            the array A must contain the Hermitian matrix, such that
            when UPLO = MagmaUpper, the leading m by m upper triangular
            part of the array A must contain the upper triangular part
            of the Hermitian matrix and the strictly lower triangular
            part of A is not referenced, and when UPLO = MagmaLower,
            the leading m by m lower triangular part of the array A
            must contain the lower triangular part of the Hermitian
            matrix and the strictly upper triangular part of A is not
            referenced.
            Before entry with SIDE = MagmaRight, the n by n part of
            the array A must contain the Hermitian matrix, such that
            when UPLO = MagmaUpper, the leading n by n upper triangular
            part of the array A must contain the upper triangular part
            of the Hermitian matrix and the strictly lower triangular
            part of A is not referenced, and when UPLO = MagmaLower,
            the leading n by n lower triangular part of the array A
            must contain the lower triangular part of the Hermitian
            matrix and the strictly upper triangular part of A is not
            referenced.
            Note that the imaginary parts of the diagonal elements need
            not be set, they are assumed to be zero.

    \param[in]
    ldda    INTEGER
            On entry, LDDA specifies the first dimension of A as declared
            in the calling (sub) program.
            When SIDE = MagmaLower then LDDA >= max( 1, m ),
            otherwise                   LDDA >= max( 1, n ).

    \param[in]
    dB      COMPLEX*16 array of DIMENSION ( LDDB, n ).
            Before entry, the leading m by n part of the array B must
            contain the matrix B.

    \param[in]
    lddb    INTEGER
            On entry, LDDB specifies the first dimension of B as declared
            in the calling (sub) program. LDDB >= max( 1, m ).

    \param[in]
    beta    COMPLEX*16
            On entry, BETA specifies the scalar beta. When BETA is
            supplied as zero then C need not be set on input.

    \param[in,out]
    dC      COMPLEX*16 array of DIMENSION ( LDDC, n ).
            Before entry, the leading m by n part of the array C must
            contain the matrix C, except when beta is zero, in which
            case C need not be set on entry.
            On exit, the array C is overwritten by the m by n updated
            matrix.

    \param[in]
    lddc    INTEGER
            On entry, LDDC specifies the first dimension of C as declared
            in the calling (sub) program. LDDC >= max( 1, m ).

    @ingroup magma_hemm
*******************************************************************************/
extern "C"
void magmablas_zhemm_mgpu(
    magma_side_t side, magma_uplo_t uplo, magma_int_t m, magma_int_t n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex_ptr dA[],    magma_int_t ldda,  magma_int_t offset,
    magmaDoubleComplex_ptr dB[],    magma_int_t lddb,
    magmaDoubleComplex beta,
    magmaDoubleComplex_ptr dC[],    magma_int_t lddc,
    magmaDoubleComplex_ptr dwork[], magma_int_t dworksiz,
    magma_int_t ngpu, magma_int_t nb,
    magma_queue_t queues[][20], magma_int_t nqueue,
    magma_event_t events[][MagmaMaxGPUs*MagmaMaxGPUs + 10], magma_int_t nevents,
    magma_int_t gnode[MagmaMaxGPUs][MagmaMaxGPUs + 2], magma_int_t ncmplx )
{
    #define dA(dev, i, j) (dA[dev] + (i) + (j)*ldda)
    #define dB(dev, i, j) (dB[dev] + (i) + (j)*lddb)
    #define dC(dev, i, j) (dC[dev] + (i) + (j)*lddc)
    #define dwork(dev, i, j) (dwork[dev] + (i) + (j)*lddwork)
    
    magma_int_t nrowa = (side == MagmaLeft ? m : n);
    magma_int_t info = 0;
    if (side != MagmaLeft) {
        info = -1;
    } else if (uplo != MagmaLower) {
        info = -2;
    } else if ( m < 0 ) {
        info = -3;
    } else if ( n < 0 ) {
        info = -4;
    } else if (ldda < max(1,nrowa)) {
        info = -7;
    } else if (lddb < max(1,m)) {
        info = -10;
    } else if (lddc < max(1,m)) {
        info = -13;
    } else if (dworksiz < lddc*n + m*n*ngpu) {
        info = -15;
    } else if (ngpu < 1) {
        info = -16;
    } else if (nb < 1) {
        info = -17;
    } else if (nqueue < ngpu) {
        info = -19;
    } else if (nevents < ngpu*ngpu) {
        info = -21;
    } else if (ncmplx < 1) {
        info = -23;
    }

    if (info != 0) {
        magma_xerbla( __func__, -(info) );
        return; //info;
    }
    
    magmaDoubleComplex c_one  = MAGMA_Z_ONE;

    magmaDoubleComplex_ptr dwork2[MagmaMaxGPUs];

    magma_int_t maxgsize = n*m;
    magma_int_t lddwork = lddc;
    magma_int_t dev, devperm;
    
    for( dev = 0; dev < ngpu; ++dev ) {
        dwork2[dev] = dwork[dev] + n*lddwork;  // size of dwork2 is maxgsize*ngpu
    }
    
    magma_device_t orig_dev;
    magma_getdevice( &orig_dev );

    magma_int_t lcblki, gbblki;
    magma_int_t ib, ioff, iblock, di;
    magma_int_t myrowsize, mycolsize, mycoloffset, mynbblkrow;
    magma_int_t myblk, myblkoffset, mynbblk, mynbblkoffset, mynblstblks, mydevlstblk;
    magma_int_t gmaster;
    magma_int_t masterdev, lcdev, lccolsize, myngpu;

    magma_int_t stdev       = (offset/nb)%ngpu;
    magma_int_t blockoffset = offset % nb;
    magma_int_t fstblksiz   = 0;
    if (blockoffset > 0) {
        fstblksiz = min( m, nb - blockoffset );
    }
    //magma_int_t nbblk       = magma_ceildiv(m, nb);
    magma_int_t nbblk       = magma_ceildiv( m + blockoffset, nb );
    magma_int_t remm        = m - fstblksiz;
    magma_int_t nbblkoffset = offset/nb;

    magma_int_t nblstblks = -1;
    magma_int_t devlstblk = -1;
    magma_int_t lstblksiz = remm%nb;
    if (lstblksiz > 0) {
        nblstblks = nbblk%ngpu;
        devlstblk = (nblstblks - 1 + ngpu)%ngpu;
    }

    magma_int_t ncmplxactive =  0;
    magma_int_t cmplxisactive[MagmaMaxGPUs];
    magma_int_t gpuisactive[MagmaMaxGPUs];
    memset( gpuisactive,   0, MagmaMaxGPUs*sizeof(magma_int_t) );
    memset( cmplxisactive, 0, MagmaMaxGPUs*sizeof(magma_int_t) );

    for( dev = 0; dev < ngpu; ++dev ) {
        magma_setdevice( dev );
        cudaMemset( dwork(dev,0,0), 0, (lddwork)*(n)*sizeof(magmaDoubleComplex) );
        // put all dC on all dev to 0 except the one which
        // hold i == 0 because this one has to multiply by beta.
        if (dev != stdev) {
            cudaMemset( dC(dev,0,0), 0, (lddc)*(n)*sizeof(magmaDoubleComplex) );
        }
    }

    // 1. symmetrize
    magma_int_t newstdev, newoffset = offset;
    if (blockoffset > 0) {
        newoffset   = offset + fstblksiz; // newoffset is adjusted over nb
        myblkoffset = (nbblkoffset/ngpu) + (nbblkoffset%ngpu > stdev ? 1 : 0);
        //printf("STDEV %d  voici offset %d remm %d   myblockoffset %d    siz %d \n", stdev, offset, remm, myblkoffset, fstblksiz);
        magma_setdevice( stdev );
        //printf( "symm1 dev %d, %2d, dA(%d, %d, %d) %p, queues %p\n",
        //        stdev, fstblksiz, stdev, offset, myblkoffset*nb + blockoffset,
        //        (void*)dA(stdev, offset, myblkoffset*nb + blockoffset),
        //        (void*)queues[stdev][0] );
        //fflush(0);
        magmablas_zsymmetrize_tiles( MagmaLower, fstblksiz,
            dA(stdev, offset, myblkoffset*nb + blockoffset), ldda,
            1, ngpu*nb, nb, queues[stdev][0] );
    }

    for( dev = 0; dev < ngpu; ++dev ) {
        newstdev      = (newoffset/nb)%ngpu;
        mynbblk       = remm/nb; // number of block of size nb. if m%nb > 0 then a last block exist and is of size ib=m%nb
        myblk         = (mynbblk/ngpu) + (mynbblk%ngpu > ((dev - newstdev + ngpu)%ngpu) ? 1 : 0);
        devperm       = (dev - newstdev + ngpu)%ngpu;
        mynbblkoffset = newoffset/nb;
        myblkoffset   = (mynbblkoffset/ngpu) + (mynbblkoffset%ngpu > dev ? 1 : 0);
        //printf("dev %d  devperm %d   newoffset %d  rowoff %d    coloff %d    myblk %d  \n", dev, devperm, newoffset, newoffset + devperm*nb, myblkoffset*nb, myblk);
        magma_setdevice( dev );
        //printf( "symm2 dev %d, %2d, dA(%d, %d, %d) %p, queues %p\n",
        //        dev, nb, dev, newoffset + devperm*nb, myblkoffset*nb,
        //        (void*)dA(dev, newoffset + devperm*nb, myblkoffset*nb),
        //        (void*)queues[dev][0] );
        //fflush(0);
        magmablas_zsymmetrize_tiles( MagmaLower, nb,
            dA(dev, newoffset + devperm*nb, myblkoffset*nb), ldda,
            myblk, ngpu*nb, nb, queues[dev][0] );
        if (remm%nb > 0) {
            mynblstblks = (mynbblk + 1)%ngpu;
            mydevlstblk = (mynblstblks - 1 + ngpu)%ngpu;
            //printf("==> siz %d devperm %d,    mydevlstblk %d,    newoffset + mynbblk*nb %d,   myblkoffset*nb + myblk*nb %d\n", remm % nb, devperm, mydevlstblk, newoffset + mynbblk*nb, myblkoffset*nb + myblk*nb);
            if (devperm == mydevlstblk) {
                //printf( "symm3 dev %2d, %d, dA(%d, %d, %d) %p, queues %p\n",
                //        dev, remm % nb, dev, newoffset + mynbblk*nb, myblkoffset*nb + myblk*nb,
                //        (void*)dA(dev, newoffset + devperm*nb, myblkoffset*nb),
                //        (void*)queues[dev][0] );
                //fflush(0);
                magmablas_zsymmetrize( MagmaLower, remm % nb,
                    dA(dev, newoffset + mynbblk*nb, myblkoffset*nb + myblk*nb), ldda,
                    queues[dev][0] );  // last partial tile
            }
        }
    }

    // ROW GEMM transpose a row and make a gemm with a block
    // if only 1 GPU used the ROW GEMM is integrated with the
    // COL GEMM (better accuracy observed) and better perf
    if (ngpu > 1) {
        for( magma_int_t i = fstblksiz; i < m; i += nb ) {
            ib            = min( nb, m - i );      // block size
            ioff          = i + offset;          // start global index in parent matrix
            mynbblkoffset = offset/nb;
            mynbblk       = magma_ceildiv(i, nb);
            for( dev = 0; dev < ngpu; ++dev ) {
                myblk       = (mynbblk/ngpu) + (mynbblk%ngpu > ((dev - stdev + ngpu)%ngpu) ?  1:0 );
                myblkoffset = (mynbblkoffset/ngpu) + (mynbblkoffset%ngpu > dev ? 1 : 0);
                myrowsize   = myblk * nb;
                mycoloffset = myblkoffset*nb;
                if (dev == stdev) {
                    myrowsize = myrowsize - blockoffset;
                    mycoloffset = myblkoffset*nb + blockoffset;
                }
                //printf("ROW GEMM: voici i %d   ib %d    ioff %d   mynbblkoffset %d stdev %d  dev %d myblk %d  myblkoffset %d  mycoloffset %d  rowsize %d\n", i, ib, ioff, mynbblkoffset, stdev, dev, myblk, myblkoffset, mycoloffset, myrowsize);
                if (myrowsize > 0) {
                    magma_setdevice( dev );
                    magma_zgemm( MagmaConjTrans, MagmaNoTrans, myrowsize, n, ib,
                                 alpha, dA(dev,ioff,mycoloffset), ldda,
                                        dB(dev,i,0),    lddb,
                                 c_one, dwork(dev,0,0), lddwork, queues[dev][1] );
                }
            }
        }
        for( dev = 0; dev < ngpu; ++dev ) {
            magma_setdevice( dev );
            magma_event_record( events[dev][1], queues[dev][1] );
        }
    }
    
    // COL GEMM
    // blockoffset is offset within first block; for subsequent blocks it is 0
    if (blockoffset > 0) {
        ib     = min( nb - blockoffset, m );    // block size
        iblock = (offset / nb) / ngpu;          // local block id
        di     = iblock*nb + blockoffset;       // local index in parent matrix
        magma_setdevice( stdev );
        //printf("DEV %d COL GEMM first   ioff %d  di %d   m %d   n %d   ib %d \n", stdev, offset, di, m, n, ib);
        magma_zgemm( MagmaNoTrans, MagmaNoTrans, m, n, ib,
                     alpha, dA(stdev,offset,di), ldda,
                            dB(stdev,0,0),       lddb,
                     beta,  dC(stdev,0,0),       lddc, queues[stdev][0] );
    }
    
    // COL GEMM
    for( magma_int_t i = fstblksiz; i < m; i += nb ) {
        ib     = min( nb, m - i );    // block size
        ioff   = i + offset;          // start global index in parent matrix
        iblock = (ioff / nb) / ngpu;  // local block id
        dev    = (ioff / nb) % ngpu;
        di     = iblock*nb;           // local index in parent matrix
        
        //printf("DEV %d COL GEMM i %d      ioff %d  di %d m-i %d    n %d   ib %d \n", dev, i, ioff, di, m-i, n, ib);
        
        magma_setdevice( dev );
        if (i == 0) {
            magma_zgemm( MagmaNoTrans, MagmaNoTrans, m-i, n, ib,
                         alpha, dA(dev,ioff,di), ldda,
                                dB(dev,i,0),     lddb,
                         beta,  dC(dev,i,0),     lddc, queues[dev][0] );
        }
        else {
            magma_zgemm( MagmaNoTrans, MagmaNoTrans, m-i, n, ib,
                         alpha, dA(dev,ioff,di), ldda,
                                dB(dev,i,0),     lddb,
                         c_one, dC(dev,i,0),     lddc, queues[dev][0] );
        }
        magma_event_record( events[dev][0], queues[dev][0] );
        // if only 1 GPU is used, do the ROW GEMM
        if (ngpu == 1) {
            // NOTE THAT because the COL gemm write dC below the diagonal (i)
            // and the ROW GEMM write dC from 0 to diag-1, so they could
            // run in parallel on different queues.
            //
            // NO NO NO because
            // it might happen that col finished i and strated i+1 while row still at i
            magma_zgemm( MagmaConjTrans, MagmaNoTrans, i, n, ib,
                         alpha, dA(dev,ioff,offset), ldda,
                                dB(dev,i,0),         lddb,
                         c_one, dC(dev,0,0),         lddc, queues[dev][0] );
        }
    }
    
    if (ngpu > 1) {
        for( dev = 0; dev < ngpu; ++dev ) {
            mynbblk   = magma_ceildiv( m + blockoffset, nb);
            mynbblkrow  = mynbblk - 1;
            devperm   = (dev - stdev + ngpu)%ngpu;
            myblk     = (mynbblkrow/ngpu) + (mynbblkrow%ngpu > devperm ?  1:0 );
            myrowsize = myblk * nb;
            if (dev == stdev) {
                myrowsize = myrowsize - blockoffset;
            }
            
            //printf("blockoffset %d mynbblkrow %d devperm %d  DEV %d RECEIVING myblk %d  myrowsize %d\n", blockoffset, mynbblkrow, devperm, dev, myblk, myrowsize);
            if (myrowsize > 0) {
                magma_setdevice( dev );
                magma_queue_wait_event( queues[dev][0], events[dev][1] );
                //magma_queue_sync( queues[dev][1] );
                // for each dev add the computed ROW block each on its placment with dC
                for( magma_int_t blki = 0; blki < myblk; ++blki) {
                    gbblki = (blki*ngpu + devperm)*nb - blockoffset;
                    lcblki = blki*nb;
                    ib     = nb; // min(nb, m-gbblki);
                    if (dev == stdev) {
                        lcblki = blki*nb - blockoffset;
                        if (blki == 0) {
                            gbblki = 0;
                            lcblki = 0;
                            ib     = nb - blockoffset;
                        }
                    }
                    magmablas_zgeadd( ib, n, c_one,
                                      &dwork[dev][lcblki], lddwork,
                                      &dC[dev][gbblki],    lddc, queues[dev][0] );
                }
                magma_event_record( events[dev][0], queues[dev][0] );
            }
        }
    }

    // ===========================================================
    //             COMMUNICATION ALL_REDUCE_SUM
    // ===========================================================
    if (ngpu == 1) {
        return;
    }
    // INITIALIZE COMM
    for( magma_int_t cmplxid = 0; cmplxid < ncmplx; ++cmplxid ) {
        masterdev     = -1;
        gnode[cmplxid][MagmaMaxGPUs+1] = -1;
        myngpu = gnode[cmplxid][MagmaMaxGPUs];
        for( magma_int_t idev = 0; idev < myngpu; ++idev ) {
            dev         = gnode[cmplxid][idev];
            devperm     = (dev - stdev + ngpu)%ngpu;
            myblk       = (nbblk/ngpu) + (nbblk%ngpu > devperm ? 1 : 0 );
            mycolsize   = myblk*nb;
            myblkoffset = nb*((nbblkoffset/ngpu) + (nbblkoffset%ngpu > dev ? 1 : 0));
            if (dev == stdev) {
                mycolsize   -= blockoffset;
                myblkoffset += blockoffset;     // local index in parent matrix
            }
            if ((devperm == devlstblk) && (lstblksiz > 0)) {
                mycolsize -= (nb - (remm%nb));
            }
            mycolsize = min(mycolsize, m);
            if (mycolsize > 0) {
                gpuisactive[dev] = mycolsize;
                if (masterdev == -1) {
                    masterdev    = dev;
                    ncmplxactive = ncmplxactive + 1;
                    cmplxisactive[cmplxid] = 1;
                    gnode[cmplxid][MagmaMaxGPUs+1] = masterdev;
                }
            }
        }
    }
    
    //*******************************
    //  each GPU send its result
    //  to its master. The master make
    //  the addition and then send to
    //  to the masters of other complex
    //  and receive from the masters of
    //  other complex make the addition
    //  and broadcast locally the final
    //  result.
    //*******************************
    //printf("=======================================================================\n");
    //printf("                     sending to my master                             \n");
    //printf("=======================================================================\n");
    for( magma_int_t cmplxid = 0; cmplxid < ncmplx; ++cmplxid ) {
        myngpu    = gnode[cmplxid][MagmaMaxGPUs];
        masterdev = gnode[cmplxid][MagmaMaxGPUs+1];
        // check if complex is active
        if (masterdev != -1) {
            for( magma_int_t idev = 0; idev < myngpu; ++idev ) {
                dev         = gnode[cmplxid][idev];
                mycolsize   = gpuisactive[dev];
                if (mycolsize > 0) {
                    // I am an active GPU. if I am not the master, then send my result to my master.
                    // store result on dwork[masterdev][dev*maxgsize]
                    if (dev != masterdev) {
                        magma_setdevice( dev );
                        //printf("             GPU %d sending to my master %d\n", dev, masterdev);
                        // wait the geadd of my ROW and COL GEMM is done
                        magma_queue_wait_event( queues[dev][0], events[dev][0] );
                        // sending to the master of my complex
                        magma_zcopymatrix_async(
                            m, n,
                            &dC[dev][0], lddc,
                            &dwork2[masterdev][maxgsize*dev], m, queues[dev][0] );
                        magma_event_record( events[dev][masterdev], queues[dev][0] );
                    } // end I am not the masterdev
                } // end if mycolsize > 0
            } // for idev
        } // end of if masterdev != -1 meaning complex is active
    } // for cmplxid

    //printf("=======================================================================\n");
    //printf(" each master do addition of local result and broadcast to other masters \n");
    //printf("=======================================================================\n");
    for( magma_int_t cmplxid = 0; cmplxid < ncmplx; ++cmplxid ) {
        myngpu    = gnode[cmplxid][MagmaMaxGPUs];
        masterdev = gnode[cmplxid][MagmaMaxGPUs+1];
        // check if complex is active
        if (masterdev != -1) {
            magma_setdevice( masterdev );
            // addition is done on stream 0 sequentially
            // wait the geadd of my ROW and COL GEMM is done
            magma_queue_wait_event( queues[masterdev][0], events[masterdev][0] );
            // ========================================
            //     local addition
            // ========================================
            for( magma_int_t l = 0; l < myngpu; ++l ) {
                lcdev         = gnode[cmplxid][l];
                lccolsize     = gpuisactive[lcdev];
                if ((lcdev != masterdev) && (lccolsize > 0)) {
                    //printf("             master %d receiving from %d and adding \n", masterdev, lcdev);
                    // this is an active GPU of my complex.
                    // wait I received what he send it to me and then do addition.
                    magma_queue_wait_event( queues[masterdev][0], events[lcdev][masterdev] );
                    magmablas_zgeadd( m, n, c_one,
                                      &dwork2[masterdev][maxgsize*lcdev], m,
                                      &dC[masterdev][0], lddc, queues[masterdev][0] );
                }
            } // for l=1:myngpu
            // because addition is done sequentially on stream 0,
            // I have to record this to be able to synch using it
            magma_event_record( events[masterdev][masterdev], queues[masterdev][0] );
            // ========================================
            //
            // ========================================
            //      send to other masters
            // ========================================
            for( magma_int_t k = 0; k < ncmplx; ++k ) {
                if (k != cmplxid) {
                    gmaster = gnode[k][MagmaMaxGPUs+1];
                    if (gmaster != -1) { // complex is active
                        // Master has to  wait until finish the local addition then send using gmaster stream.
                        // use stream 0 to make it sequential or stream gmaster to make it parallel.
                        // Now both re the same.
                        //printf("             master %d from cmplx %d sending to other master %d on cmplx %d \n", masterdev, cmplxid, gmaster, k);
                        magma_queue_wait_event( queues[masterdev][gmaster], events[masterdev][masterdev] );
                        magma_zcopymatrix_async(
                            m, n,
                            &dC[masterdev][0], lddc,
                            &dwork2[gmaster][maxgsize*masterdev], m, queues[masterdev][gmaster] );
                        magma_event_record( events[masterdev][gmaster],   queues[masterdev][gmaster] );
                        magma_event_record( events[masterdev][masterdev], queues[masterdev][gmaster] );
                    } // end of gmaster != -1
                } // end of k != cmplxid
            } // for k = 0: ncmplx
            // ========================================
        } // end of if masterdev != -1 meaning complex is active
    } // for cmplxid

    //printf("=======================================================================\n");
    //printf(" each master wait receiving other masters results, do the addition and broadcast locally \n");
    //printf("=======================================================================\n");
    for( magma_int_t cmplxid = 0; cmplxid < ncmplx; ++cmplxid ) {
        myngpu    = gnode[cmplxid][MagmaMaxGPUs];
        masterdev = gnode[cmplxid][MagmaMaxGPUs+1];
        // check if complex is active
        if (masterdev != -1) {
            magma_setdevice( masterdev );
            // addition is done on stream 0 sequentially
            // master has to wait until finishing all the send to other masters.
            magma_queue_wait_event( queues[masterdev][0], events[masterdev][masterdev] );
            // ========================================
            //  addition of results from other masters
            // ========================================
            for( magma_int_t k = 0; k < ncmplx; ++k ) {
                if (k != cmplxid) {
                    gmaster = gnode[k][MagmaMaxGPUs+1];
                    if (gmaster != -1) { // complex is active
                        // Master has to  wait until receiving from gmaster, then do addition using stream 0
                        //printf("             master %d from cmplx %d receiving from other master %d on cmplx %d and adding \n", masterdev, cmplxid, gmaster, k);
                        magma_queue_wait_event( queues[masterdev][0], events[gmaster][masterdev] );
                        magmablas_zgeadd( m, n, c_one,
                                          &dwork2[masterdev][maxgsize*gmaster], m,
                                          &dC[masterdev][0], lddc, queues[ masterdev ][0] );
                    } // end of gmaster != -1
                } // end of k != cmplxid
            } // for k = 0: ncmplx
            // because addition is done sequentially on stream 0,
            // I have to record this to be able to synch using it
            magma_event_record( events[masterdev][masterdev], queues[masterdev][0] );
            // ========================================
            // ========================================
            //     local broadcast of final results
            // ========================================
            for( magma_int_t l = 0; l < myngpu; ++l ) {
                lcdev         = gnode[cmplxid][l];
                lccolsize     = gpuisactive[lcdev];
                if ((lcdev != masterdev) && (lccolsize > 0)) {
                    // this is an active GPU of my complex.
                    // wait the previous addition is done meaning stream 0 is finished and broadcast sequentially for now.
                    // to make it parallel put stream lcdev instead of stream 0
                    //printf("             master %d broadcasting local to %d  \n", masterdev, lcdev);
                    magma_queue_wait_event( queues[masterdev][0], events[masterdev][masterdev] );
                    magma_zcopymatrix_async(
                        m, n,
                        &dC[masterdev][0], lddc,
                        &dC[lcdev][0],     lddc, queues[masterdev][0] );
                    magma_event_record( events[masterdev][lcdev], queues[masterdev][0] );
                }
            } // for l=1:myngpu
            // ========================================
        } // end of if masterdev != -1 meaning complex is active
    } // for cmplxid

    for( magma_int_t cmplxid = 0; cmplxid < ncmplx; ++cmplxid ) {
        myngpu    = gnode[cmplxid][MagmaMaxGPUs];
        masterdev = gnode[cmplxid][MagmaMaxGPUs+1];
        // check if complex is active
        if (masterdev != -1) {
            for( magma_int_t l = 0; l < myngpu; ++l ) {
                lcdev         = gnode[cmplxid][l];
                lccolsize     = gpuisactive[lcdev];
                if (lccolsize > 0) {
                    magma_setdevice( lcdev );
                    magma_queue_wait_event( queues[lcdev][0], events[lcdev][0] );
                    magma_queue_wait_event( queues[lcdev][0], events[masterdev][lcdev] );
                }
            } // for l=1:myngpu
        } // end of if masterdev != -1 meaning complex is active
    } // for cmplxid

    //printf("****************************************************\n");
    //printf("                      finish zhemm                   \n");
    //printf("****************************************************\n");

    magma_setdevice( orig_dev );
}
