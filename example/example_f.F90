!! This is a simple standalone example. See README.txt

module example_f
use magma
implicit none
contains


!! ------------------------------------------------------------
!! Replace with your code to initialize the A matrix.
!! This simply initializes it to random values.
!! Note that A is stored column-wise, not row-wise.
!!
!! m   - number of rows,    m >= 0.
!! n   - number of columns, n >= 0.
!! A   - m-by-n array of size lda*n.
!! lda - leading dimension of A, lda >= m.
!!
!! When lda > m, rows (m, ..., lda-1) below the bottom of the matrix are ignored.
!! This is helpful for working with sub-matrices, and for aligning the top
!! of columns to memory boundaries (or avoiding such alignment).
!! Significantly better memory performance is achieved by having the outer loop
!! over columns (j), and the inner loop over rows (i), than the reverse.
subroutine zfill_matrix( m, n, A, lda )
    integer :: m, n, lda
    complex*16 :: A(:,:)
    
    integer :: i, j
    real*16 :: re, im
    
    do j = 1, n
        do i = 1, m
            call random_number( re )
            call random_number( im )
            A(i,j) = complex( re, im )
        end do
    end do
end subroutine


!! ------------------------------------------------------------
!! Replace with your code to initialize the X rhs.
subroutine zfill_rhs( m, nrhs, X, ldx )
    integer :: m, nrhs, ldx
    complex*16 :: X(:,:)
    call zfill_matrix( m, nrhs, X, ldx )
end subroutine


!! ------------------------------------------------------------
!! Replace with your code to initialize the dA matrix on the GPU device.
!! This simply leverages the CPU version above to initialize it to random values,
!! and copies the matrix to the GPU.
subroutine zfill_matrix_gpu( m, n, dA, lda )
    integer :: m, n, lda
    magma_devptr_t :: dA
    
    complex*16, allocatable :: A(:,:)
    integer :: sizeof_complex=16
    
    allocate( A(lda,n) )
    call zfill_matrix( m, n, A, lda )
    call cublas_set_matrix( m, n, sizeof_complex, A, lda, dA, lda )
    deallocate( A )
end subroutine


!! ------------------------------------------------------------
!! Replace with your code to initialize the dX rhs on the GPU device.
subroutine zfill_rhs_gpu( m, nrhs, dX, ldx )
    integer :: m, nrhs, ldx
    magma_devptr_t :: dX
    call zfill_matrix_gpu( m, nrhs, dX, ldx )
end subroutine


!! ------------------------------------------------------------
!! Solve A * X = B, where A and X are stored in CPU host memory.
!! Internally, MAGMA transfers data to the GPU device
!! and uses a hybrid CPU + GPU algorithm.
subroutine cpu_interface( n, nrhs )
    integer :: n, nrhs
    
    complex*16, allocatable :: A(:,:), X(:,:)
    integer,    allocatable :: ipiv(:)
    integer :: lda, ldx, info
    
    lda  = n
    ldx  = lda
    info = 0
    
    !! allocate CPU memory
    !! no magma Fortran routines for this, so use Fortran's normal mechanism
    allocate( A(lda, n) )
    allocate( X(ldx, nrhs) )
    allocate( ipiv(n) )
    
    !! Replace these with your code to initialize A and X
    call zfill_matrix( n, n, A, lda )
    call zfill_rhs( n, nrhs, X, ldx )
    
    ! Solve using LU factorization
    call magmaf_zgesv( n, 1, A, lda, ipiv, X, lda, info )
    if (info .ne. 0) then
        print "(a,i5)", "magma_zgesv failed with info=", info
    end if
    
    ! Instead, for least squares, or if preferred over LU, use QR
    !call magmaf_zgels( MagmaNoTrans, n, n, 1, A, lda, X, lda, work, lwork, info )
    
    ! Instead, if A is SPD (symmetric/Hermitian positive definite), use Cholesky
    !call magmaf_zposv( MagmaLower, n, 1, A, lda, X, lda, info )
    
    !! TODO: use result in X
    
!! cleanup:
    deallocate( A )
    deallocate( X )
    deallocate( ipiv )
end subroutine


!! ------------------------------------------------------------
!! Solve dA * dX = dB, where dA and dX are stored in GPU device memory.
!! Internally, MAGMA uses a hybrid CPU + GPU algorithm.
subroutine gpu_interface( n, nrhs )
    integer :: n, nrhs
    
    magma_devptr_t :: dA, dX
    integer, allocatable :: ipiv(:)
    integer :: ldda, lddx, info
    integer :: sizeof_complex=16

    ldda = ceiling(real(n)/32)*32
    lddx = ldda
    info = 0
        
    !! allocate GPU memory
    !! no magma Fortran routines for this, so use cublas
    call cublas_alloc( ldda*n,    sizeof_complex, dA )
    call cublas_alloc( lddx*nrhs, sizeof_complex, dX )
    allocate( ipiv(n) )  !! ipiv always on CPU
    if (dA == 0 .or. dX == 0) then
        print "(a)", "malloc failed"
        goto 1000
    endif
    
    !! Replace these with your code to initialize A and X
    call zfill_matrix_gpu( n, n, dA, ldda )
    call zfill_rhs_gpu( n, nrhs, dX, lddx )
    
    call magmaf_zgesv_gpu( n, 1, dA, ldda, ipiv, dX, ldda, info )
    if (info .ne. 0) then
        print "(a,i5)", "magma_zgesv_gpu failed with info=", info
    endif
    
    !! TODO: use result in dX
    
!! cleanup:
1000 continue
    call cublas_free( dA )
    call cublas_free( dX )
    deallocate( ipiv )
end subroutine

end module


!! ------------------------------------------------------------
program main
    use magma
    use example_f
    implicit none
    
    integer :: n=1000, nrhs=1
    
    call magmaf_init()
    
    print "(a)", "using MAGMA CPU interface"
    call cpu_interface( n, nrhs )

    print "(a)", "using MAGMA GPU interface"
    call gpu_interface( n, nrhs )
    
    call magmaf_finalize()
end program
