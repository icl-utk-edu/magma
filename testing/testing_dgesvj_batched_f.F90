!!
!!   -- MAGMA (version 2.0)
!!      Univ. of Tennessee, Knoxville
!!      Univ. of California, Berkeley
!!      Univ. of Colorado, Denver
!!      @date
!!
program testing_dgesvj_batched_f
    use magma
    implicit none

    external dlange, dgemm, dgesv, dlamch, dscal, dlacpy, dlaset

    double precision dlange, dlamch

    double precision              :: rnumber(2), Anorm, err, uerr, verr, error
    double precision, allocatable :: work(:)
    double precision, allocatable :: hA(:), hU(:), hV(:), hS(:), hUS(:), hI(:)
    magma_devptr_t                :: dinfo_array
    magma_devptr_t                :: dA, dU, dV, dS
    magma_devptr_t                :: queue
    integer                       :: dev, m, n, info, lda, batch, minmn, maxmn
    integer                       :: ldu, ldv, strideA, strideU, strideV
    integer                       :: i, j, b, iter, niter
    real(kind=8)                  :: t, tstart, tend, Uorth, Vorth
    double precision              :: c_one, c_neg_one
    character                     :: jobu, jobv
    parameter                       (batch = 100., niter = 3)
    parameter                       (c_one = 1., c_neg_one = -1.)

    !! Initialize MAGMA
    call magmaf_init()
    dev = 0
    call magmaf_queue_create( dev, queue )

    !! Print header
    print '(a5, "   ", a5, "   ", a5, "   ", a8, "   ", a12, "   ", a12, "   ", a12)', &
          "Batch", "m", "n", "Time(ms)", "|A-USV^H|",  "|I-UU^H|", "|I-VV^H|"
    print *,'------------------------------------------------------------------------------'

    do m = 32, 512, 32
        do iter = 1, niter, 1
            jobu  = 's'
            jobv  = 's'
            n     = m
            minmn = min(m,n)
            maxmn = max(m,n)
            lda   = m    ! A is m x n
            ldu   = m    ! U is m x minmn
            ldv   = n    ! V is n x minmn

            strideA = lda*n;
            strideU = ldu*minmn;
            strideV = ldv*minmn;

            !! Allocate CPU memory
            allocate( hA(  batch*strideA ) )
            allocate( hU(  batch*strideU ) )
            allocate( hV(  batch*strideV ) )
            allocate( hS(  batch*minmn   ) )
            allocate( hUS( batch*strideU ) )
            allocate( hI(  minmn*minmn   ) )
            allocate( work( n ) )

            !! Allocate GPU memory
            info = magmaf_dmalloc( dA, batch*lda*n )
            if (info .ne. 0) then
                print *, "Error: magmaf_dmalloc( dA ) failed, info = ", info
                stop
            endif

            info = magmaf_dmalloc( dU, batch*ldu*m )
            if (info .ne. 0) then
                print *, "Error: magmaf_dmalloc( dU  ) failed, info = ", info
                stop
            endif

            info = magmaf_dmalloc( dV, batch*ldv*minmn )
            if (info .ne. 0) then
                print *, "Error: magmaf_dmalloc( dV  ) failed, info = ", info
                stop
            endif

            info = magmaf_dmalloc( dS, batch*minmn )
            if (info .ne. 0) then
                print *, "Error: magmaf_dmalloc( dS  ) failed, info = ", info
                stop
            endif

            info = magmaf_imalloc( dinfo_array, batch )
            if (info .ne. 0) then
                print *, "Error: magmaf_dmalloc( dinfo_array  ) failed, info = ", info
                stop
            endif

            !! Initialize the matrix
            do i = 1, batch*lda*n
                call random_number(rnumber)
                hA(i) = rnumber(1)
            end do

            !! dA = hA
            call magmaf_dsetmatrix( m, batch*n, hA, lda, dA, lda, queue )

            !! Call magma batch gesvj
            call magmaf_wtime( tstart )
            call magmaf_dgesvj_batched_strided(jobu, jobv, m, n, dA, lda, strideA, &
                                    dS, minmn, dU, ldu, strideU, dV, ldv, strideV, &
                                    dinfo_array, batch, queue)
            call magmaf_queue_sync( queue )  !! probably not necessary
            call magmaf_wtime( tend )
            t = (tend - tstart) * 1000. !! milliseconds

            !! hU = dU, hV = dV, hS = dS
            call magmaf_dgetmatrix( m, batch*minmn, dU, ldu, hU, ldu, queue )
            call magmaf_dgetmatrix( minmn, batch*n, dV, ldv, hV, ldv, queue )
            call magmaf_dgetmatrix( minmn, batch, dS, minmn, hS, minmn, queue )

            !! Compute svd error (rough version)
            error = 0.
            Uorth = 0.
            Vorth = 0.
            do b = 1, batch
                !! check orthogonality of u, set hI to identity
                call dlaset('F', minmn, minmn, 0., c_one, hI(1), minmn)
                call dgemm( 't', 'n', minmn, minmn, m, &
                            c_one,      hU(1+(b-1)*ldu*minmn), ldu, &
                                        hU(1+(b-1)*ldu*minmn), ldu, &
                            c_neg_one,  hI(1),                 minmn )
                uerr  = dlange( 'F', minmn, minmn, hI(1), minmn, work ) / m

                !! check orthogonality of v, set hI to identity
                call dlaset('F', minmn, minmn, 0., c_one, hI(1), minmn)
                call dgemm( 't', 'n', minmn, minmn, n, &
                            c_one,      hV(1+(b-1)*ldv*minmn), ldv, &
                                        hV(1+(b-1)*ldv*minmn), ldv, &
                            c_neg_one,  hI(1),                 minmn )
                verr  = dlange( 'F', minmn, minmn, hI(1), minmn, work ) / n

                !! check decomposition ||A - USVt || / (max(m,n)*||A||)
                Anorm = dlange( 'F', m, n, hA(1+(b-1)*lda*n), lda, work )
                !! scale u by sigma
                do j = 1,minmn
                    call dscal(m, hS(1+(b-1)*minmn+(j-1)), hU(1 + (b-1)*ldu*minmn + (j-1)*ldu), 1)
                end do

                call dgemm( 'n', 't', m, n, minmn, &
                            c_one,      hU(1+(b-1)*ldu*minmn), ldu, &
                                        hV(1+(b-1)*ldv*minmn), ldv, &
                            c_neg_one,  hA(1+(b-1)*lda*n),     lda )
                err   = dlange( 'F', m, n, hA(1+(b-1)*lda*n), lda, work ) / (maxmn * Anorm)
                Uorth = max(Uorth, uerr)
                Vorth = max(Vorth, verr)
                error = max(error, err)
            end do
            print '(i5"   ", i5"   ", i5"   ", f8.2"   ", e12.4"   ", e12.4"   ", e12.4)', &
                   batch, m, n, t, error, Uorth, Vorth

            !! Free CPU memory
            deallocate( hA, hU, hV, hS, hUS, hI, work)

            !! Free GPU memory
            info = magmaf_free( dA )
            if (info .ne. 0) then
                print *, 'Error: magmaf_free( dA ) failed, info = ', info
                stop
            endif

            info = magmaf_free( dU )
            if (info .ne. 0) then
                print *, 'Error: magmaf_free( dU ) failed, info = ', info
                stop
            endif

            info = magmaf_free( dV )
            if (info .ne. 0) then
                print *, 'Error: magmaf_free( dV ) failed, info = ', info
                stop
            endif
        enddo
        print *,''
    enddo

    !! Cleanup
    call magmaf_queue_destroy( queue );
    call magmaf_finalize()
end
