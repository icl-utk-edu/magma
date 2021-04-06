Conversion from MAGMA v1 to v2.

This uses the Hessenberg (zgehrd, zlahr2, zlahru) and QR (zgeqrf_gpu) as
examples. To see changes, diffs of these 3 files are in:
    convert_v2_zgehrd.html
    convert_v2_zlahr2.html
    convert_v2_zlahru.html
    convert_v2_zgeqrf.html


----------------------------------------
In kernel files (*.cu):
Many kernels have already been converted, but this is how to fix one that it
isn't yet converted.

1)  Add an _q version that takes a queue, e.g., magmablas_zswap_q( ..., queue ).
    The kernel launch changes from
        <<< ..., magma_stream >>>
    to
        <<< ..., queue->cuda_stream() >>>

    The prototype for the _q version goes in magmablas_z_q.h; most already exist.

2)  Have the non-q version call the _q version, using magmablasGetQueue().
    See magmablas/zswap.cu for a simple example.

3)  Automated partial conversion using scripts/convert_v2_kernel.pl
    This replaces queue with queue->cuda_stream(),
    and magma_stream with magmablasGetQueue().
    Steps (1) and (2) are manual, though.

    In most cases, you do not currently need to change the header, unless the
    driver calls other magma functions that now take a queue, such as ztrsm.
    This gets more complicated, so ask Mark.



----------------------------------------
In src files:

0)  Run testers to verify that the code is working before changes.

        cd testing
        ./run_tests.py -s -m testing_*gehrd

    It's possible that something failed trying to make things backwards
    compatibile. Of course, this should be resolved before changing the code.

    The testers now run on a non-NULL stream, by default. This breaks any code
    that assumed extra synchronization from the NULL stream. To achieve the old
    behavior, use --null-stream:

        ./testing_dgeqrf_gpu --range 100:300:100 -c --version 3

        %   M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)    |b - A*x|
        %===============================================================
          100   100     ---   (  ---  )      1.44 (   0.00)     5.38e-18   ok
          200   200     ---   (  ---  )      3.54 (   0.00)     7.33e-05   failed
          300   300     ---   (  ---  )      5.33 (   0.01)     3.35e-05   failed


        ./testing_dgeqrf_gpu --range 100:300:100 -c --version 3 --null-stream

        %   M     N   CPU GFlop/s (sec)   GPU GFlop/s (sec)    |b - A*x|
        %===============================================================
          100   100     ---   (  ---  )      1.45 (   0.00)     5.38e-18   ok
          200   200     ---   (  ---  )      3.29 (   0.00)     3.40e-18   ok
          300   300     ---   (  ---  )      5.01 (   0.01)     2.58e-18   ok

    In the case of QR, an extra queue sync needed to be added after larfb.

1)  Automated partial conversion with scripts/convert_v2_src.pl.

        cd src
        ../scripts/convert_v2_src.pl zgehrd.cpp
        ../scripts/convert_v2_src.pl zlahr2.cpp
        ../scripts/convert_v2_src.pl zlahru.cpp

    This should do much of the tedious search and replace, but that's all it
    does -- it doesn't parse the code, so it can easily make mistakes.

    - Renames "stream[...]" to "queues[...]" (plural).

    - Replaces the #include header
      from common_magma.h   (which includes magma.h)
      to   magma_internal.h (which includes magma_v2.h)

    - Adds device to magma_queue_create.
      This may add redundant getdevice calls that need to be removed.
      If you already know the device (e.g., in a multi-GPU code that loops
      over the devices), you can remove the getdevice call entirely.

    - Comments out Set/GetKernelStream.
      You should then delete these, but you may need them to verify that it
      picked the correct queue for subsequent magmablas calls.

    - Add queue argument to magmablas calls.
      If the code called magmablasSetKernelStream( queue ), the script *assumes*
      that is the queue to use. Otherwise, it will put in UNKNOWN.

2)  Manually cleanup and verify code.

    - Remove any redundant or commented out code from step (1).

    - Create & destroy any queues that may be needed.
      For auxiliary routines such as larfb, lahr2, lahru, latrd, add queue(s)
      as an argument so it isn't created & destroyed for each panel.

      For Hessenberg, I create a queue in zgehrd, and pass that into zlahr2 and
      zlahru.

3)  For better OpenCL compatability, it is preferred to always use dA(i,j) syntax
    for all GPU arrays. For example, instead of passing "dA, ldda",
    pass "dA(0,0), ldda". In OpenCL, this is mapped to a pointer and offset, as
    two arguments. Then a simple #ifdef switches between the CUDA and OpenCL versions:

        #ifdef MAGMA_HAVE_OPENCL
        #define dA(i_,j_)  dA, ((i_) + (j_)*ldda + dA_offset)
        #else
        #define dA(i_,j_) (dA + (i_) + (j_)*ldda)
        #endif

    Also, all GPU arrays should be declared as magmaDoubleComplex_ptr
    instead of magmaDoubleComplex* (and similar). In OpenCL, these _ptr types
    are mapped to cl_mem.

4)  Run testers to verify that it still works. Here is running small (-s) and
    medium (-m) tests for all 4 precisions.

        cd testing
        ./run_tests.py -s -m testing_*gehrd

        [ lots of output ]

        ****************************************************************************************************
        summary
        ****************************************************************************************************
        all 472 tests in 8 commands passed!
