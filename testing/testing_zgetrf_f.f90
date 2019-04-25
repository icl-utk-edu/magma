!
!   -- MAGMA (version 2.0) --
!      Univ. of Tennessee, Knoxville
!      Univ. of California, Berkeley
!      Univ. of Colorado, Denver
!      @date
!
!  @precisions normal z -> c d s
!
      program testing_zgetrf_f

      use magma

      external dlamch, zlange, zgemm, zgesv, zgetrs

      double precision zlange, dlamch

      double precision              :: rnumber(2), Anorm, Bnorm, Xnorm, Rnorm
      double precision, allocatable :: work(:)
      complex*16, allocatable       :: A(:), B(:), X(:)
      complex*16, allocatable       :: A2(:)
      integer,    allocatable       :: ipiv(:)

      complex*16                    :: c_one, c_neg_one
      integer                       :: i, n, info, lda
      integer                       :: nrhs
      real(kind=8)                  :: flops, t, tstart, tend

      PARAMETER          ( nrhs = 1, c_one = 1., c_neg_one = -1. )
      
      n   = 2048
      lda = n
 
!------ Allocate CPU memory
      allocate(A(lda*n))
      allocate(A2(lda*n))
      allocate(B(lda*nrhs))
      allocate(X(lda*nrhs))
      allocate(ipiv(n))
      allocate(work(n))

!---- Initializa the matrix
      do i=1,n*n
         call random_number(rnumber)
         A(i) = rnumber(1)
      end do
      A2(:) = A(:)

      do i=1,n*nrhs
         call random_number(rnumber)
         B(i) = rnumber(1)
      end do
      X(:) = B(:)

!---- Call magma LU ----------------
      call magmaf_wtime(tstart)
      call magmaf_zgetrf(n, n, A, lda, ipiv, info)
      call magmaf_wtime(tend)

      if ( info .ne. 0 )  then
         write(*,*) "Info : ", info
      end if

!---- Call solve -------------
      call zgetrs('n', n, nrhs, A, lda, ipiv, X, lda, info)

      if ( info .ne. 0 )  then
         write(*,*) "Info : ", info
      end if

!---- Compare the two results ------
      Anorm = zlange('I', n, n,    A2, lda, work)
      Bnorm = zlange('I', n, nrhs, B,  lda, work)
      Xnorm = zlange('I', n, nrhs, X,  lda, work)

      call zgemm('n', 'n', n,  nrhs, n, c_one, A2, lda, X, lda, c_neg_one, B, lda)
      Rnorm = zlange('I', n, nrhs, B, lda, work)
            
      write(*,*)
      write(*,*  ) 'Solving A x = b using LU factorization:'
      write(*,105) '  || A || = ', Anorm
      write(*,105) '  || b || = ', Bnorm
      write(*,105) '  || x || = ', Xnorm
      write(*,105) '  || b - A x || = ', Rnorm
      
      flops = 2. * n * n * n / 3.
      t = tend - tstart

      write(*,*)   '  Gflops  = ',  flops / t / 1e9
      write(*,*)

      Rnorm = Rnorm / ( (Anorm*Xnorm+Bnorm) * n * dlamch('E') )

      if ( Rnorm > 60. ) then
         write(*,105) '  Solution is suspicious, ', Rnorm
      else
         write(*,105) '  Solution is CORRECT'
      end if

!---- Free CPU memory
      deallocate(A, A2, B, X, ipiv, work)

 105  format((a35,es10.3))

      end
