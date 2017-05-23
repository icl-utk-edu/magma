program main
    use iso_c_binding
    use magma2
    implicit none
    
    integer :: dev
    type(c_ptr) :: queue  !! magma_queue_t
    
    print "(a)", "--------------- init"
    call magma_init()
    
    print "(a)", "--------------- create queue"
    dev = 0
    call magma_queue_create( dev, queue )

    call test_aux( queue )
    call test_blas_lapack( queue )
    call test_batched( queue )

    print "(a)", "--------------- destroy queue"
    call magma_queue_destroy( queue )
    
    print "(a)", "--------------- finalize"
    call magma_finalize()
    print "(a)", "done"

contains

!! -----------------------------------------------------------------------------
!! test auxiliary routines, found in magma_common.F90 and magma2.F90
subroutine test_aux( queue )
    type(c_ptr) :: queue  !! magma_queue_t

    integer :: num, dev, arch, info

    integer :: major, minor, micro
    type(c_ptr) :: x, xpin
    type(c_ptr) :: dx
    double precision :: time
    integer(c_size_t) :: mem
    
    print "(a)", "------------------------------ aux tests"
    print "(a)", "---------- version"
    call magma_version( major, minor, micro )
    print "(a,i3,i3,i3)", "version", major, minor, micro
    
    call magma_print_environment()
    print *
    
    print "(a)", "---------- malloc GPU memory"
    info = magma_malloc( dx, 1000*sizeof_double )
    print "(a,z16)", "malloc", dx
    
    info = magma_free( dx )
    print "(a,z16)", "free", dx
    print *
    
    print "(a)", "---------- malloc CPU memory"
    info = magma_malloc_cpu( x, 1000*sizeof_double )
    print "(a,z16)", "malloc_cpu", x
    
    info = magma_free_cpu( x )
    print "(a,z16)", "free_cpu", x
    print *
    
    print "(a)", "---------- malloc CPU pinned memory"
    info = magma_malloc_pinned( xpin, 1000*sizeof_double )
    print "(a,z16)", "malloc_pinned", xpin
    
    info = magma_free_pinned( xpin )
    print "(a,z16)", "free_pinned", xpin
    print *
    
    print "(a)", "---------- device support"
    num = magma_num_gpus()
    print "(a,i2)", "num gpus", num
    
    arch = magma_get_device_arch()
    print "(a,i5)", "arch", arch
    
    dev = 0
    call magma_set_device( dev )
    print "(a,i2)", "set_device", dev
    
    call magma_get_device( dev )
    print "(a,i2)", "get_device", dev
    
    mem = magma_mem_size( queue )
    print "(a,i16,a,i16,a)", "mem_size", mem, " bytes", mem/1024/1024, " MiB"
    print *
    
    print "(a)", "---------- queue support"
    call magma_queue_sync( queue )
    print "(a)", "queue sync"
    
    dev = magma_queue_get_device( queue )
    print "(a,i6)", "queue_get_device", dev
    print *
    
    print "(a)", "---------- timing"
    time = magma_wtime()
    !call sleep( 1 )  !! GNU extension
    time = magma_wtime() - time
    print "(a,f9.4)", "time     ", time
    
    time = magma_sync_wtime( queue )
    !call sleep( 1 )  !! GNU extension
    time = magma_sync_wtime( queue ) - time
    print "(a,f9.4)", "sync_time", time
    
    print "(a)", "------------------------------ end aux tests"
end subroutine test_aux


!! -----------------------------------------------------------------------------
!! test BLAS and LAPACK style MAGMA routines, found in magma_zfortran.F90
!! (excluding batched)
subroutine test_blas_lapack( queue )
    type(c_ptr) :: queue  !! magma_queue_t
    
    !! parts of this probably won't work unless m = n = k
    integer, parameter :: m = 3, n = 3, k = 3, maxn = 3, &
                          lda = 3, ldb = 3, ldc = 3, ldr = 3
    double precision :: A(lda,k), B(ldb,n), C(ldc,n), R(ldc,maxn)
    integer :: ipiv(m), info
    type(c_ptr) :: dA, dB, dC
    
    print "(a)", "------------------------------ blas/lapack tests"
    print "(a)", "---------- setup matrices"
    A = reshape( (/ 1, 2, 3,   3, 1, 2,   2, 3, 1 /), (/ lda, k /) )
    B = reshape( (/ 4, 5, 6,   6, 4, 5,   5, 6, 4 /), (/ ldb, n /) )
    C = reshape( (/ 7, 8, 9,   9, 7, 8,   8, 9, 7 /), (/ ldc, n /) )
    
    print "(a)", "A ="
    call print_matrix( A )
    
    print "(a)", "B ="
    call print_matrix( B )
    
    print "(a)", "C ="
    call print_matrix( C )
    
    info = magma_dmalloc( dA, int(lda, c_size_t)*k )
    info = magma_dmalloc( dB, int(ldb, c_size_t)*n )
    info = magma_dmalloc( dC, int(ldc, c_size_t)*n )
    print "(a,z16)", "malloc(dA)", dA
    print "(a,z16)", "malloc(dB)", dB
    print "(a,z16)", "malloc(dC)", dC
    
    call magma_dsetmatrix( m, k, A, lda, dA, lda, queue )
    call magma_dsetmatrix( k, n, B, ldb, dB, ldb, queue )
    call magma_dsetmatrix( m, n, C, ldc, dC, ldc, queue )
    print "(a)", "setmatrix A, B, C => dA, dB, dC"
    
    print "(a)", "---------- axpy"
    call magma_daxpy( min(m,k), 2.0d0, dA, 1, dB, 1, queue )
    print "(a)", "daxpy"
    
    call magma_dgetmatrix( k, n, dB, ldb, R, ldr, queue )
    print "(a)", "B(:,1) = 2*A(:,1) + B(:,1)"
    call print_matrix( R )
    
    B(:,1) = 2*A(:,1) + B(:,1)
    print "(a)", "expect B ="
    call print_matrix( B )
    
    print "(a)", "---------- gemv"
    call magma_dgemv( MagmaNoTrans, m, k, 1.5d0, dA, lda, dB, 1, 2.5d0, dC, 1, queue )
    print "(a)", "dgemv"
    
    call magma_dgetmatrix( m, n, dC, lda, R, ldr, queue )
    print "(a)", "C(:,1) = 1.5*A*B(:,1) + 2.5*C(:,1)"
    call print_matrix( R )
    
    C(:,1) = 1.5*matmul(A, B(:,1)) + 2.5*C(:,1)
    print "(a)", "expect C ="
    call print_matrix( C )
    
    print "(a)", "---------- gemm"
    call magma_dgemm( MagmaNoTrans, MagmaNoTrans, m, n, k, &
                      0.5d0, dA, lda, &
                             dB, ldb, &
                      0.8d0, dC, ldc, queue )
    print "(a)", "gemm"
    
    call magma_dgetmatrix( m, n, dC, ldc, R, ldr, queue )
    print "(a)", "C = 0.5*A*B + 0.8*C ="
    call print_matrix( R )
    
    C = 0.5*matmul(A,B) + 0.8*C
    print "(a)", "expect C ="
    call print_matrix( C )
    
    print "(a)", "---------- getrf"
    ipiv(:) = -1  !! reset
    R = A  !! save
    call magma_dgetrf( m, k, A, lda, ipiv, info )
    if (info .ne. 0) then
        print "(a,i5)", "magma_dgetrf error: info =", info
    endif
    
    print "(a)", "magma_dgetrf, A ="
    call print_matrix( A )
    print "(a)", "ipiv ="
    print *, ipiv
    print *
    
    ipiv(:) = -1  !! reset
    A = R  !! restore
    call magma_dsetmatrix( m, k, A, lda, dA, lda, queue )
    call magma_dgetrf_gpu( m, k, dA, lda, ipiv, info )
    if (info .ne. 0) then
        print "(a,i5)", "magma_dgetrf error: info =", info
    endif
    call magma_dgetmatrix( m, k, dA, lda, A, lda, queue )
    
    print "(a)", "magma_dgetrf_gpu, A ="
    call print_matrix( A )
    print "(a)", "ipiv ="
    print *, ipiv
    print *
    
    ipiv(:) = -1  !! reset
    A = R  !! restore
    call dgetrf( m, k, A, lda, ipiv, info )
    if (info .ne. 0) then
        print "(a,i5)", "lapack dgetrf error: info =", info
    endif
    
    print "(a)", "lapack dgetrf, A ="
    call print_matrix( A )
    print "(a)", "ipiv ="
    print *, ipiv
    print *
    
    print "(a)", "---------- potrf"
    A = reshape( (/ 4, 1, 2,   1, 3, 1,   2, 1, 5 /), (/ m, k /) )
    
    R = A  !! save
    call magma_dpotrf( MagmaLower, m, A, lda, info )
    if (info .ne. 0) then
        print "(a,i5)", "magma_dpotrf error: info =", info
    endif
    
    print "(a)", "magma_dpotrf, A ="
    call print_matrix( A )

    A = R  !! restore
    call magma_dsetmatrix( m, k, A, lda, dA, lda, queue )
    call magma_dpotrf_gpu( MagmaLower, m, dA, lda, info )
    if (info .ne. 0) then
        print "(a,i5)", "magma_dpotrf error: info =", info
    endif
    call magma_dgetmatrix( m, k, dA, lda, A, lda, queue )
    
    print "(a)", "magma_dpotrf_gpu, A ="
    call print_matrix( A )
    
    A = R  !! restore
    call dpotrf( "lower", k, A, lda, info )
    if (info .ne. 0) then
        print "(a,i5)", "lapack dpotrf error: info =", info
    endif
    
    print "(a)", "lapack dpotrf, A ="
    call print_matrix( A )
    
    print "(a)", "---------- cleanup"
    info = magma_free( dA )
    info = magma_free( dB )
    info = magma_free( dC )
    print "(a)", "free dA, dB, dC"
    print *
    
    print "(a)", "------------------------------ end blas/lapack test"
end subroutine test_blas_lapack
    

!! -----------------------------------------------------------------------------
!! test MAGMA batched routine (dgetrf_batched)
subroutine test_batched( queue )
    type(c_ptr) :: queue
    
    integer, parameter :: n = 4, lda = n, batchcount = 5
    integer :: b, i, j, info
    type(c_ptr) :: dA           !! on GPU, double*  (array of n * n * batchcount doubles)
    type(c_ptr) :: dipiv        !! on GPU, int*     (array of n * batchcount integers)
    type(c_ptr) ::    dA_array  !! on GPU, double** (array of batchcount double* pointers)
    type(c_ptr) :: dipiv_array  !! on GPU, int**    (array of batchcount int*    pointers)
    type(c_ptr) :: dinfo_array  !! on GPU, int*     (array of batchcount integers)
    
    double precision :: A(n, n, batchcount)  !! on CPU
    double precision :: R(n, n, batchcount)  !! on CPU, for MAGMA results
    integer          :: ipiv(n)              !! on CPU, for LAPACK
    integer          :: ipiv_r(n*batchcount) !! on CPU, for MAGMA results
    
    type(c_ptr) ::    A_array(batchcount)  !! on CPU, double** (array of batchcount double* pointers)
    type(c_ptr) :: ipiv_array(batchcount)  !! on CPU, int**    (array of batchcount int*    pointers)
    integer     :: info_array(batchcount)  !! on CPU, int*
        
    print "(a)", "------------------------------ batched"
    info = magma_malloc( dA,          batchcount * n * n * sizeof_double )
    info = magma_malloc( dipiv,       batchcount * n * sizeof_int )
    info = magma_malloc(    dA_array, batchcount * sizeof_ptr )
    info = magma_malloc( dipiv_array, batchcount * sizeof_ptr )
    info = magma_malloc( dinfo_array, batchcount * sizeof_int )
    
    A_array = c_null_ptr
    ipiv_array = c_null_ptr
    info_array = -1
    
    !! setup A matrices on CPU, then copy to GPU
    print "(a)", "---------- setup matrices"
    do b = 1, batchcount
        do j = 1, n
            do i = 1, n
                call random_number( A(i,j,b) )
                A(i,j,b) = A(i,j,b) + b
            enddo
        enddo
    enddo
    print "(a)", "A ="
    do b = 1, batchcount
        call print_matrix( A(:,:,b) )
    enddo
    
    call magma_dsetmatrix( n, n*batchcount, A, lda, dA, lda, queue )
    
    !! setup pointer arrays on CPU, then copy to GPU
    print "(a)", "---------- setup pointer arrays"
    do b = 1, batchcount
        A_array(b)    = magma_doffset_2d( dA, n, 1, (b-1)*n + 1 )  !! = dA + (b-1)*n*n*sizeof_double
        ipiv_array(b) = magma_ioffset_1d( dipiv, 1, (b-1)*n + 1 )  !! = dipiv + (b-1)*n*sizeof_int
    enddo
    call magma_psetvector( batchcount,    A_array, 1,    dA_array, 1, queue )
    call magma_psetvector( batchcount, ipiv_array, 1, dipiv_array, 1, queue )
    
    print "(a,i20)", "dA", dA
    print "(a,i4,a)", "A_array (first should match dA, rest should be n * n * sizeof(double) = ", &
          n*n*sizeof_double, " apart)"
    print *, A_array
    print *
    
    print "(a,i20)", "dipiv", dipiv
    print "(a,i4,a)", "ipiv_array (first should match dipiv; rest should be n * sizeof(int) = ", &
          n*sizeof_int, " apart)"
    print *, ipiv_array
    print *
    
    !! call batched routine
    print "(a)", "---------- call dgetrf_batched"
    call magma_dgetrf_batched( n, n, dA_array, lda, dipiv_array, &
                               dinfo_array, batchcount, queue )
    
    !! get results
    print "(a)", "---------- get results"
    call magma_igetvector( batchcount, dinfo_array, 1, info_array, 1, queue )
    print "(a)", "info_array ="
    print *, info_array
    
    call magma_igetvector( batchcount*n, dipiv, 1, ipiv_r, 1, queue )
    
    call magma_dgetmatrix( n, n*batchcount, dA, lda, R, lda, queue )
    do b = 1, batchcount
        print "(a)", "-----"
        !! call lapack for comparison
        call dgetrf( n, n, A(1,1,b), lda, ipiv, info )
        if (info .ne. 0) then
            print "(a)", "dgetrf error, info =", info
        endif
        
        print "(a,i2,a)", "magma A(:,:,", b, ")"
        call print_matrix( R(:,:,b) )
        print "(a)", "magma ipiv"
        print *, ipiv_r( (b-1)*n + 1 : b*n )
        
        print "(a)", "lapack"
        call print_matrix( A(:,:,b) )
        print "(a)", "lapack ipiv"
        print *, ipiv
        print *
    enddo
    
    print "(a)", "---------- cleanup"
    info = magma_free( dA )
    info = magma_free( dipiv )
    info = magma_free(    dA_array )
    info = magma_free( dipiv_array )
    info = magma_free( dinfo_array )
    
    print "(a)", "------------------------------ end batched"
end subroutine test_batched


!! -----------------------------------------------------------------------------
subroutine print_matrix( A )
    double precision :: A(:,:)
    
    integer :: i
    do i = 1, ubound(A,1)
        print *, A(i,:)
    enddo
    print *
end subroutine print_matrix

end program
