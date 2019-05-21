!!
!!   -- MAGMA (version 2.0)
!!      Univ. of Tennessee, Knoxville
!!      Univ. of California, Berkeley
!!      Univ. of Colorado, Denver
!!      @date
!!
!!  @precisions normal z -> c d s
!!
program testing_zgetrf_gpu_f
    use magma
    implicit none

    external zlange, zgemm, zgesv, dlamch

    double precision zlange, dlamch

    double precision              :: rnumber(2), Anorm, Bnorm, Rnorm, Xnorm, error, tol
    double precision              :: dAnorm, dBnorm, dRnorm, dXnorm, derror
    double precision, allocatable :: work(:)
    complex*16, allocatable       :: hA(:), hB(:), hX(:)
    magma_devptr_t                :: dA, dB, dX, dwork
    magma_devptr_t                :: queue
    integer, allocatable          :: ipiv(:)
    integer                       :: dev, i, n, nb, info, lda, ldda, nrhs, lwork
    character(len=16)             :: okay
    !! magma's precision generator messes up "double precision", so use "real(kind=8)"
    real(kind=8)                  :: gflops, t, tstart, tend

    complex*16                    :: c_one, c_neg_one
    parameter                       (c_one = 1., c_neg_one = -1.)

    !! Initialize MAGMA
    call magmaf_init()
    dev = 0
    call magmaf_queue_create( dev, queue )

    !! Print header
    print '(a5, "  ", a5, "  ", a5, "  ", a12, "  ", a12, "  ", a12, "  ", a)', &
          "n", "nrhs", "nb", "error", "error2", "Gflop/s", "status"

    do n = 100, 1000, 100
        nrhs = 10
        lda  = n
        ldda = ((n+31)/32)*32  !! round up to multiple of 32

        !! get nb, just to check that the Fortran nb interface works
        nb = magmaf_get_zgetrf_nb( n, n );

        !! Allocate CPU memory
        allocate( hA( lda*n ) )
        allocate( hB( lda*nrhs ) )
        allocate( hX( lda*nrhs ) )
        allocate( work( n ) )
        allocate( ipiv( n ) )

        !! Allocate GPU memory
        info = magmaf_zmalloc( dA, ldda*n )
        if (info .ne. 0) then
            print *, "Error: magmaf_zmalloc( dA ) failed, info = ", info
            stop
        endif

        info = magmaf_zmalloc( dB, ldda*nrhs )
        if (info .ne. 0) then
            print *, "Error: magmaf_zmalloc( dB  ) failed, info = ", info
            stop
        endif

        info = magmaf_zmalloc( dX, ldda*nrhs )
        if (info .ne. 0) then
            print *, "Error: magmaf_zmalloc( dX  ) failed, info = ", info
            stop
        endif

        lwork = n
        info = magmaf_zmalloc( dwork, lwork )
        if (info .ne. 0) then
            print *, "Error: magmaf_zmalloc( dwork  ) failed, info = ", info
            stop
        endif

        !! Initializa the matrix
        do i = 1, lda*n
            call random_number(rnumber)
            hA(i) = rnumber(1)
        end do

        do i = 1, lda*nrhs
          call random_number(rnumber)
          hB(i) = rnumber(1)
        end do
        hX(:) = hB(:)

        !! dA = hA
        call magmaf_zsetmatrix( n, n, hA, lda, dA, ldda, queue )

        !! dB = hB
        !! dX = dB
        call magmaf_zsetmatrix( n, nrhs, hB, lda, dB, ldda, queue )
        call magmaf_zcopymatrix( n, nrhs, dB, ldda, dX, ldda, queue )

        !! Call magma LU
        call magmaf_wtime( tstart )
        call magmaf_zgetrf_gpu( n, n, dA, ldda, ipiv, info )
        call magmaf_wtime( tend )
        if (info .ne. 0) then
            print *, "Error: magmaf_zgetrf_gpu failed, info = ", info
            stop
        endif

        t = tend - tstart
        gflops = 2./3. * n * n * n * 1e-9 / t

        !! Call magma solve
        call magmaf_zgetrs_gpu( 'n', n, nrhs, dA, ldda, ipiv, dX, ldda, info )
        if (info .ne. 0) then
            print *, "Error: magmaf_zgetrs_gpu failed, info = ", info
            stop
        endif

        !! hX = dX
        call magmaf_zgetmatrix( n, nrhs, dX, ldda, hX, lda, queue )

        !! Compute residual, using LAPACK
        Anorm = zlange( '1', n, n,    hA, lda, work )
        Bnorm = zlange( '1', n, nrhs, hB, lda, work )
        Xnorm = zlange( '1', n, nrhs, hX, lda, work )
        call zgemm( 'n', 'n', n,  nrhs, n, &
                    c_one,     hA, lda, &
                               hX, lda, &
                    c_neg_one, hB, lda )
        Rnorm = zlange( '1', n, nrhs, hB, lda, work )
        error = Rnorm / ((Anorm*Xnorm + Bnorm) * n)

        !! Compute residual, using MAGMA, to demo their use
        !! reset dA = hA
        call magmaf_zsetmatrix( n, n, hA, lda, dA, ldda, queue )
        dAnorm = magmablasf_zlange( '1', n, n,    dA, ldda, dwork, lwork, queue )
        dBnorm = magmablasf_zlange( '1', n, nrhs, dB, ldda, dwork, lwork, queue )
        dXnorm = magmablasf_zlange( '1', n, nrhs, dX, ldda, dwork, lwork, queue )
        call magmablasf_zgemm( 'n', 'n', n, nrhs, n, &
                               c_one,     dA, ldda, &
                                          dX, ldda, &
                               c_neg_one, dB, ldda, queue )
        dRnorm = magmablasf_zlange( '1', n, nrhs, dB, ldda, dwork, lwork, queue )
        derror = dRnorm / ((dAnorm*dXnorm + dBnorm) * n)

        tol = 60 * dlamch('E')
        if (error < tol .and. derror < tol) then
            okay = "ok"
        else
            okay = "FAILED"
        endif

        print '(i5, "  ", i5, "  ", i5, "  ", e12.4, "  ", e12.4, "  ", f12.4, "  ", a)', &
              n, nrhs, nb, error, derror, gflops, okay

        !! Free CPU memory
        deallocate( hA, hX, hB, work, ipiv )

        !! Free GPU memory
        info = magmaf_free( dA )
        if (info .ne. 0) then
            print *, 'Error: magmaf_free( dA ) failed, info = ', info
            stop
        endif

        info = magmaf_free( dB )
        if (info .ne. 0) then
            print *, 'Error: magmaf_free( dB ) failed, info = ', info
            stop
        endif

        info = magmaf_free( dX )
        if (info .ne. 0) then
            print *, 'Error: magmaf_free( dX ) failed, info = ', info
            stop
        endif

        info = magmaf_free( dwork )
        if (info .ne. 0) then
            print *, 'Error: magmaf_free( dwork ) failed, info = ', info
            stop
        endif
    enddo

    !! Cleanup
    call magmaf_queue_destroy( queue );
    call magmaf_finalize()
end
