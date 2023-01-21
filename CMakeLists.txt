cmake_minimum_required( VERSION 3.18 )

# ----------------------------------------
# to disable Fortran, set this to "off"
# see also -DADD_ below
option( USE_FORTRAN "Fortran is required for some tester checks, but can be disabled with reduced functionality" ON )

if (USE_FORTRAN)
    project( MAGMA C CXX Fortran )
else()
    project( MAGMA C CXX )
endif()


# ----------------------------------------
# to show compile commands, set this here or use 'make VERBOSE=1'
#set(CMAKE_VERBOSE_MAKEFILE on)

# ----------------------------------------
# MAGMA requires one backend to be enabled
option(MAGMA_ENABLE_CUDA     "Enable the CUDA backend"  OFF)
option(MAGMA_ENABLE_HIP      "Enable the HIP  backend"  OFF)

# check if one backend has been enabled
if (NOT MAGMA_ENABLE_CUDA AND
    NOT MAGMA_ENABLE_HIP
    )
  message(STATUS "MAGMA requires one enabled backend!")
  message(STATUS "Building CUDA backend")
  set( MAGMA_ENABLE_CUDA ON )
endif()

# ----------------------------------------
# don't regenerate files during make.
# (I think this means you have to manually re-run CMake if CMakeLists changes.
# It fixes the huge problems with CMake interrupting Visual Studio.)
set(CMAKE_SUPPRESS_REGENERATION on)


# ----------------------------------------
# force an out-of-source build, to not overwrite the existing Makefiles
# (out-of-source is cleaner, too)
string( COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}" MAGMA_COMPILE_INPLACE )
if (MAGMA_COMPILE_INPLACE)
    message( FATAL_ERROR "Compiling MAGMA with CMake requires an out-of-source build. To proceed:
    rm -rf CMakeCache.txt CMakeFiles/   # delete files in ${CMAKE_SOURCE_DIR}
    mkdir build
    cd build
    cmake ..
    make" )
endif()


# ----------------------------------------
# prefer shared libraries
option( BUILD_SHARED_LIBS "If on, build shared libraries, otherwise build static libraries" ON )

# prefer /usr/local/magma, instead of /usr/local.
if (UNIX AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "/usr/local/magma" CACHE PATH "..." FORCE)
endif()

# ----------------------------------------
# use C++11 and C99
# see http://stackoverflow.com/questions/10851247/how-to-activate-c-11-in-cmake
include(CheckCXXCompilerFlag)
include(CheckCCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
CHECK_CXX_COMPILER_FLAG("-fPIC" COMPILER_SUPPORTS_FPIC)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
else()
    message( WARNING "The compiler ${CMAKE_CXX_COMPILER} doesn't support the -std=c++11 flag. Some code may not compile.")
endif()

CHECK_C_COMPILER_FLAG("-std=c99" COMPILER_SUPPORTS_C99)
if (COMPILER_SUPPORTS_C99)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
else()
    message( WARNING "The compiler ${CMAKE_C_COMPILER} doesn't support the -std=c99 flag. Some code may not compile.")
endif()


# ----------------------------------------
# check Fortran name mangling
if (USE_FORTRAN)
    include( FortranCInterface )
    FortranCInterface_HEADER( ${CMAKE_SOURCE_DIR}/include/magma_mangling_cmake.h MACRO_NAMESPACE MAGMA_ )
else()
    # set one of -DADD_, -DUPCASE, or -DNOCHANGE. See README.
    message( STATUS "Building without Fortran compiler" )
    set( FORTRAN_CONVENTION "-DADD_" CACHE STRING "Fortran calling convention, one of -DADD_, -DNOCHANGE, -DUPCASE" )
    set_property( CACHE FORTRAN_CONVENTION PROPERTY STRINGS -DADD_ -DNOCHANGE -DUPCASE )
    message( STATUS "    Using ${FORTRAN_CONVENTION} for Fortran calling convention" )
    set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   ${FORTRAN_CONVENTION}" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FORTRAN_CONVENTION}" )
    # see also NVCC_FLAGS below
endif()


# ----------------------------------------
# locate OpenMP
find_package( OpenMP )
if (OPENMP_FOUND)
    message( STATUS "Found OpenMP" )
    message( STATUS "    OpenMP_C_FLAGS   ${OpenMP_C_FLAGS}" )
    message( STATUS "    OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS}" )
    set( CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}" )
endif()

if (MAGMA_ENABLE_CUDA)
  enable_language(CUDA)

  # ----------------------------------------
  # locate CUDA libraries
  set( GPU_TARGET "Kepler Maxwell Pascal" CACHE STRING "CUDA architectures to compile for; one or more of Fermi, Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Hopper, or valid sm_[0-9][0-9]" )
  find_package( CUDAToolkit )
  if (CUDAToolkit_FOUND)
    message( STATUS "Found CUDA ${CUDA_VERSION}" )
    message( STATUS "    CUDA_CUDART_LIBRARY: CUDA::cudart" )
    #message( STATUS "    CUDA_CUBLAS_LIBRARIES: CUDA::cublas" )

    # NVCC options for the different cards
    # sm_xx is binary, compute_xx is PTX for forward compatability
    # MIN_ARCH is lowest requested version

    if(WIN32)
        # Disable separable compilation on Windows because object linking list
        # becomes too long when building multiple archs and MSVC throws errors
        set(CUDA_SEPARABLE_COMPILATION OFF)
    else()
        set(CUDA_SEPARABLE_COMPILATION ON)
    endif()

    set(__cuda_architectures)

    include_directories( ${CUDAToolkit_INCLUDE_DIRS} )

    if (GPU_TARGET MATCHES Fermi)
        set( GPU_TARGET "${GPU_TARGET} sm_20" )
    endif()

    if (GPU_TARGET MATCHES Kepler)
        set( GPU_TARGET "${GPU_TARGET} sm_30 sm_35 sm_37" )
    endif()

    if (GPU_TARGET MATCHES Maxwell)
        set( GPU_TARGET "${GPU_TARGET} sm_50" )
    endif()

    if (GPU_TARGET MATCHES Pascal)
        set( GPU_TARGET "${GPU_TARGET} sm_60" )
    endif()

    if (GPU_TARGET MATCHES Volta)
        set( GPU_TARGET "${GPU_TARGET} sm_70" )
    endif()

    if (GPU_TARGET MATCHES Turing)
        set( GPU_TARGET "${GPU_TARGET} sm_75" )
    endif()

    if (GPU_TARGET MATCHES Ampere)
        set( GPU_TARGET "${GPU_TARGET} sm_80" )
    endif()

    if (GPU_TARGET MATCHES Hopper)
        set( GPU_TARGET "${GPU_TARGET} sm_90" )
    endif()

    if (GPU_TARGET MATCHES sm_20)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 200 )
        endif()
        list(APPEND __cuda_architectures 20)
        message( STATUS "    compile for CUDA arch 2.0 (Fermi)" )
    endif()

    if (GPU_TARGET MATCHES sm_30)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 300 )
        endif()
        list(APPEND __cuda_architectures 30)
        message( STATUS "    compile for CUDA arch 3.0 (Kepler)" )
    endif()

    if (GPU_TARGET MATCHES sm_35)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 300 )
        endif()
        list(APPEND __cuda_architectures 35)
        message( STATUS "    compile for CUDA arch 3.5 (Kepler)" )
    endif()

    if (GPU_TARGET MATCHES sm_37)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 300 )
        endif()
        list(APPEND __cuda_architectures 37)
        message( STATUS "    compile for CUDA arch 3.7 (Kepler)" )
    endif()

    if (GPU_TARGET MATCHES sm_50)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 500 )
        endif()
        list(APPEND __cuda_architectures 50)
        message( STATUS "    compile for CUDA arch 5.0 (Maxwell)" )
    endif()

    if (GPU_TARGET MATCHES sm_52)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 520 )
        endif()
        list(APPEND __cuda_architectures 52)
        message( STATUS "    compile for CUDA arch 5.2 (Maxwell)" )
    endif()

    if (GPU_TARGET MATCHES sm_53)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 530 )
        endif()
        list(APPEND __cuda_architectures 53)
        message( STATUS "    compile for CUDA arch 5.3 (Maxwell)" )
    endif()

    if (GPU_TARGET MATCHES sm_60)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 600 )
        endif()
        list(APPEND __cuda_architectures 60)
        message( STATUS "    compile for CUDA arch 6.0 (Pascal)" )
    endif()

    if (GPU_TARGET MATCHES sm_61)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 610 )
        endif()
        list(APPEND __cuda_architectures 61)
        message( STATUS "    compile for CUDA arch 6.1 (Pascal)" )
    endif()

    if (GPU_TARGET MATCHES sm_62)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 620 )
        endif()
        list(APPEND __cuda_architectures 62)
        message( STATUS "    compile for CUDA arch 6.2 (Pascal)" )
    endif()

    if (GPU_TARGET MATCHES sm_70)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 700 )
        endif()
        list(APPEND __cuda_architectures 70)
        message( STATUS "    compile for CUDA arch 7.0 (Volta)" )
    endif()

    if (GPU_TARGET MATCHES sm_71)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 710 )
        endif()
        list(APPEND __cuda_architectures 71)
        message( STATUS "    compile for CUDA arch 7.1 (Volta)" )
    endif()

    if (GPU_TARGET MATCHES sm_75)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 750 )
        endif()
        list(APPEND __cuda_architectures 75)
        message( STATUS "    compile for CUDA arch 7.5 (Turing)" )
    endif()

    if (GPU_TARGET MATCHES sm_80)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 800 )
        endif()
        list(APPEND __cuda_architectures 80)
        message( STATUS "    compile for CUDA arch 8.0 (Ampere)" )
    endif()

    if (GPU_TARGET MATCHES sm_90)
        if (NOT MIN_ARCH)
            set( MIN_ARCH 900 )
        endif()
        list(APPEND __cuda_architectures 90)
        message( STATUS "    compile for CUDA arch 9.0 (Hopper)" )
    endif()

    if (NOT MIN_ARCH)
        message( FATAL_ERROR "GPU_TARGET must contain one or more of Fermi, Kepler, Maxwell, Pascal, Volta, Turing, Ampere, or valid sm_[0-9][0-9]" )
    endif()

    set(CUDA_ARCHITECTURES "${__cuda_architectures}")

    add_library(magma_nvcc_flags INTERFACE)
    if (COMPILER_SUPPORTS_FPIC)
        target_compile_options(magma_nvcc_flags
            INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options;-fPIC,${FORTRAN_CONVENTION}>
        )
    else()
        # No Position Independent Code on Windows. Compiler will complain if you add that flag.
        target_compile_options(magma_nvcc_flags
            INTERFACE
            $<$<COMPILE_LANGUAGE:CUDA>:--compiler-options;${FORTRAN_CONVENTION}>
        )
    endif(COMPILER_SUPPORTS_FPIC)

    set(MAGMA_HAVE_CUDA "1")
    set(MAGMA_CUDA_ARCH_MIN "${MIN_ARCH}")
    message( STATUS "Define -DMAGMA_HAVE_CUDA -DMAGMA_CUDA_ARCH_MIN=${MIN_ARCH}" )
  else()
    message( STATUS "Could not find CUDA" )
  endif()

endif()

if (MAGMA_ENABLE_HIP)
  set( GPU_TARGET "gfx900" CACHE STRING "HIP architectures to compile for" )
  list(APPEND CMAKE_PREFIX_PATH /opt/rocm /opt/rocm/hip)
  find_package( HIP )
  if (HIP_FOUND)
    message( STATUS "Found HIP ${HIP_VERSION}" )
    message( STATUS "    HIP_INCLUDE_DIRS:   ${HIP_INCLUDE_DIRS}"   )
    message( STATUS "GPU_TARGET:  ${GPU_TARGET}"   )

    include_directories( ${HIP_INCLUDE_DIRS} )

    set(HIP_SEPARABLE_COMPILATION ON)

    if (GPU_TARGET MATCHES kaveri)
      set( GPU_TARGET ${GPU_TARGET} gfx700 )
    endif()

    if (GPU_TARGET MATCHES hawaii)
      set( GPU_TARGET ${GPU_TARGET} gfx701 )
    endif()

    if (GPU_TARGET MATCHES kabini)
      set( GPU_TARGET ${GPU_TARGET} gfx703 )
    endif()

    if (GPU_TARGET MATCHES mullins)
      set( GPU_TARGET ${GPU_TARGET} gfx703 )
    endif()

    if (GPU_TARGET MATCHES bonaire)
      set( GPU_TARGET ${GPU_TARGET} gfx704 )
    endif()

    if (GPU_TARGET MATCHES carrizo)
      set( GPU_TARGET ${GPU_TARGET} gfx801 )
    endif()

    if (GPU_TARGET MATCHES iceland)
      set( GPU_TARGET ${GPU_TARGET} gfx802 )
    endif()

    if (GPU_TARGET MATCHES tonga)
      set( GPU_TARGET ${GPU_TARGET} gfx802 )
    endif()

    if (GPU_TARGET MATCHES fiji)
      set( GPU_TARGET ${GPU_TARGET} gfx803 )
    endif()

    if (GPU_TARGET MATCHES polaris10)
      set( GPU_TARGET ${GPU_TARGET} gfx803 )
    endif()

    if (GPU_TARGET MATCHES tongapro)
      set( GPU_TARGET ${GPU_TARGET} gfx805 )
    endif()

    if (GPU_TARGET MATCHES stoney)
      set( GPU_TARGET ${GPU_TARGET} gfx810 )
    endif()

    set( DEVCCFLAGS  "" )
    set(VALID_GFXS "700;701;702;703;704;705;801;802;803;805;810;900;902;904;906;908;909;90c;1010;1011;1012;1030;1031;1032;1033")
    foreach( GFX ${VALID_GFXS} )
      if ( GPU_TARGET MATCHES gfx${GFX} )
	set( DEVCCFLAGS ${DEVCCFLAGS} --amdgpu-target=gfx${GFX} )
      endif()
    endforeach()

    set( DEVCCFLAGS ${DEVCCFLAGS} -fPIC ${FORTRAN_CONVENTION} )
    set(MAGMA_HAVE_HIP "1")
    message( STATUS "Define -DMAGMA_HAVE_HIP" )

    set_property(TARGET hip::device APPEND PROPERTY COMPATIBLE_INTERFACE_BOOL INTERFACE_HIP_DEVICE_COMPILE)
    set_property(TARGET hip::device PROPERTY INTERFACE_HIP_DEVICE_COMPILE ON)
    set(GPU_ARCH_FLAGS ${DEVCCFLAGS})
    
    #add_compile_options(${GPU_ARCH_FLAGS})
  else()
    message( STATUS "Could not find HIP" )
  endif()
endif()

# ----------------------------------------
# locate LAPACK libraries

set(BLA_VENDOR "" CACHE STRING
    "Use specified BLAS library. See https://cmake.org/cmake/help/latest/module/FindBLAS.html")

# List from CMake 3.17, minus some obsolete ones:
# PhiPACK, Compaq CXML, DEC Alpha DXML, SunPerf, SGI SCSL, SGIMATH,
# Intel, NAS (Apple veclib).
# FLAME is BLIS.
set_property(CACHE BLA_VENDOR PROPERTY STRINGS
    "" "All" "Goto" "OpenBLAS" "FLAME" "ATLAS" "IBMESSL"
    "Intel10_64lp" "Intel10_64lp_seq" "Intel10_64ilp" "Intel10_64ilp_seq"
    "ACML" "ACML_MP" "ACML_GPU"
    "Apple"
    "Arm" "Arm_mp" "Arm_ilp64" "Arm_ilp64_mp"
    "Generic")

set( LAPACK_LIBRARIES "" CACHE STRING "Libraries for LAPACK and BLAS, to manually override search" )
if (LAPACK_LIBRARIES STREQUAL "")
    message( STATUS "Searching for BLAS and LAPACK. To override, set LAPACK_LIBRARIES using ccmake." )
    find_package( LAPACK )
    # force showing updated LAPACK_LIBRARIES in ccmake / cmake-gui.
    set( LAPACK_LIBRARIES ${LAPACK_LIBRARIES} CACHE STRING "Libraries for LAPACK and BLAS, to manually override search" FORCE )
else()
    message( STATUS "User set LAPACK_LIBRARIES. To change, edit LAPACK_LIBRARIES using ccmake (set to empty to enable search)." )
    # Check either -lname syntax or file existence
    foreach( LIB ${LAPACK_LIBRARIES} )
        if (NOT LIB MATCHES "^-l[a-zA-Z0-9_-]+$")
        if (NOT EXISTS ${LIB})
            message( WARNING "\n      Warning: file ${LIB} does not exist.\n" )
        endif()
        endif()
    endforeach()
endif()

# If using MKL, add it to includes and define MAGMA_WITH_MKL
# Initially, this gets MKLROOT from environment, but then the user can edit it.
if (LAPACK_LIBRARIES MATCHES mkl_core)
    set( MKLROOT $ENV{MKLROOT} CACHE STRING "MKL installation directory" )
    if (MKLROOT STREQUAL "")
        message( WARNING "LAPACK_LIBRARIES has MKL, but MKLROOT not set; can't add include directory." )
    else()
        message( STATUS "MKLROOT set to ${MKLROOT}. To change, edit MKLROOT using ccmake." )
        if (NOT EXISTS ${MKLROOT})
            message( FATAL_ERROR "MKLROOT ${MKLROOT} directory does not exist." )
        endif()
        include_directories( ${MKLROOT}/include )
        add_definitions( -DMAGMA_WITH_MKL )
        message( STATUS "Define -DMAGMA_WITH_MKL" )
    endif()
endif()


# ----------------------------------------
# save magma.lib, magma_sparse.lib, etc. in lib/
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY lib )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY lib )


# ----------------------------------------
# list of sources
if (MAGMA_ENABLE_CUDA)
  include( ${CMAKE_SOURCE_DIR}/CMake.src.cuda )
else()
  include( ${CMAKE_SOURCE_DIR}/CMake.src.hip )
endif()

# ----------------------------------------
# common flags

if (WIN32)
    # On Windows:
    #     Strip out /W3; we will use -W4
    #     -Wall is way too verbose; use -W4
    #     -MP enables parallel builds
    #     -std=c99 is not implemented, so skip that
    string( REGEX REPLACE " */W3" "" CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}"   )
    string( REGEX REPLACE " */W3" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" )
    set( CMAKE_C_FLAGS     "${CMAKE_C_FLAGS} -W4 -MP -DMAGMA_NOAFFINITY" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W4 -MP -DMAGMA_NOAFFINITY" )
else()
    # Primarily for gcc / nvcc:
    # Ignore unused static functions in headers.
    set( CMAKE_C_FLAGS     "${CMAKE_C_FLAGS} -Wall -Wno-unused-function" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-unused-function" )
endif()

if (CMAKE_HOST_APPLE)
    # Use rpaths, which is on by default in CMake 3.
    set( CMAKE_MACOSX_RPATH 1 )

    # 64-bit veclib (Accelerate) has issues; substitute correct functions from LAPACK.
    # (The issue is single precision functions that return doubles;
    # if a consistent prototype is used, the problem goes away in C,
    # but this is not feasible in Fortran.)
    if (LAPACK_LIBRARIES MATCHES "Accelerate")
        if (USE_FORTRAN)
            message( STATUS "MacOS X: adding blas_fix library" )
            add_library( blas_fix ${libblas_fix_src} )
            target_link_libraries( blas_fix
                ${LAPACK_LIBRARIES}
            )
            set( blas_fix blas_fix )
            set( blas_fix_lib -lblas_fix )
        else()
            message( WARNING "\n      Warning: cannot compile blas_fix library for MacOS X without Fortran compiler.\n" )
        endif()
    endif()

    set( CMAKE_C_FLAGS     "${CMAKE_C_FLAGS} -DMAGMA_NOAFFINITY" )
    set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DMAGMA_NOAFFINITY" )

    # previously, just compile as 32-bit, but CUDA 6.5 no longer has 32-bit FAT libraries
    ## set( CMAKE_C_FLAGS       "${CMAKE_C_FLAGS} -m32" )
    ## set( CMAKE_CXX_FLAGS     "${CMAKE_CXX_FLAGS} -m32" )
    ## set( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -m32" )
    ## set( CUDA_64_BIT_DEVICE_CODE OFF )
endif()

include_directories( "${CMAKE_BINARY_DIR}/include" )

include_directories( include )
include_directories( control )
if (MAGMA_ENABLE_CUDA)
  include_directories( magmablas )  # e.g., shuffle.cuh
else()
  include_directories( magmablas_hip )  # e.g., shuffle.cuh
endif()
  
# Need to check sizeof(void*) after setting flags above;
# CMAKE_SIZEOF_VOID_P can be wrong.
include( CheckTypeSize )
CHECK_TYPE_SIZE( void* SIZEOF_VOID_PTR )
if (USE_FORTRAN)
    set( CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -Dmagma_devptr_t=\"integer\(kind=${SIZEOF_VOID_PTR}\)\"" )
endif()

# Configure config
configure_file(${CMAKE_SOURCE_DIR}/include/magma_config.h.in  ${CMAKE_BINARY_DIR}/include/magma_config.h)

# ----------------------------------------
# compile MAGMA library
if (WIN32)
    # Windows seems to have a problem mixing C, CUDA, and Fortran files
    # Currently ignores .f90 and .F90 files, because it doesn't seem to
    # understand that .F90 files should be pre-processed.

    # separate Fortran and C/C++/CUDA files
    foreach( filename ${libmagma_all} )
        if (filename MATCHES "\\.(f)$")  # |f90|F90
            list( APPEND libmagma_all_f   ${filename} )
        elseif (filename MATCHES "\\.(c|cu|cpp)$")
            list( APPEND libmagma_all_cpp ${filename} )
        endif()
    endforeach()
    #message( "libmagma_all_cpp ${libmagma_all_cpp}" )
    #message( "libmagma_all_f   ${libmagma_all_f}"   )

    # on Windows, Fortran files aren't compiled if listed here...
    add_library( magma ${libmagma_all_cpp} )
    target_link_libraries( magma
        ${LAPACK_LIBRARIES}
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        magma_nvcc_flags
    )

    # no Fortran files at the moment (how to test libmagma_all_f is not empty?),
    # but keep this around for future reference.
    #
    ##  ...so make a separate library out of Fortran files. Ugh.
    ## add_library( magmaf ${libmagma_all_f} )
    ## target_link_libraries( magmaf
    ##     ${LAPACK_LIBRARIES}
    ##     CUDA::cudart
    ##     CUDA::cublas
    ##     CUDA::cusparse
    ## )
    ## make list of Fortran .mod files to install, as below
else()
    # Unix doesn't seem to have a problem with mixing C, CUDA, and Fortran files
    if (MAGMA_ENABLE_CUDA)
    #message(FATAL_ERROR "${libmagma_all}")
      add_library( magma ${libmagma_all} )
      target_link_libraries( magma
        ${blas_fix}
        ${LAPACK_LIBRARIES}
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
        magma_nvcc_flags
	)
    else()
      find_package( hipBLAS )
      if (hipBLAS_FOUND)
	message( STATUS "Found rocBLAS ${rocBLAS_VERSION}" )
      endif()
      find_package( hipSPARSE )
      if (hipSPARSE_FOUND)
	message( STATUS "Found rocSPARSE ${rocSPARSE_VERSION}" )
      endif()
      add_library( magma ${libmagma_all} )
      target_link_libraries( magma
	hip::host
        ${blas_fix}
        ${LAPACK_LIBRARIES}
	roc::hipblas
	roc::hipsparse
	)
    endif()
    
    if (USE_FORTRAN)
        # make list of Fortran .mod files to install
        foreach( filename ${libmagma_all} )
            if (filename MATCHES "\\.(f90|F90)$")
                # mod files seem to wind up in root build directory
                get_filename_component( fmod ${filename} NAME_WE )
                list( APPEND modules "${CMAKE_BINARY_DIR}/${fmod}.mod" )
            endif()
        endforeach()
    endif()
endif()
add_custom_target( lib DEPENDS magma )


# ----------------------------------------
# compile lapacktest library
# If use fortran, compile only Fortran files, not magma_[sdcz]_no_fortran.cpp
# else,           compile only C++     files, not Fortran files
if (USE_FORTRAN)
    foreach( filename ${liblapacktest_all} )
        if (filename MATCHES "\\.(f|f90|F90)$")
            list( APPEND liblapacktest_all_f ${filename} )
        endif()
    endforeach()
    add_library( lapacktest ${liblapacktest_all_f} )
else()
    # alternatively, use only C/C++/CUDA files, including magma_[sdcz]_no_fortran.cpp
    foreach( filename ${liblapacktest_all} )
        if (filename MATCHES "\\.(c|cu|cpp)$")
            list( APPEND liblapacktest_all_cpp ${filename} )
        endif()
    endforeach()
    add_library( lapacktest ${liblapacktest_all_cpp} )
endif()
target_link_libraries( lapacktest
    ${blas_fix}
    ${LAPACK_LIBRARIES}
)


# ----------------------------------------
# compile tester library
add_library( tester ${libtest_all} )
target_link_libraries( tester
    magma
    lapacktest
    ${blas_fix}
    ${LAPACK_LIBRARIES}
)


# ----------------------------------------
# compile MAGMA sparse library

# sparse doesn't have Fortran at the moment, so no need for above shenanigans
if (MAGMA_ENABLE_CUDA)
  include_directories( sparse/include )
  include_directories( sparse/control )
else()
  include_directories( sparse_hip/include )
  include_directories( sparse_hip/control )
endif()
include_directories( testing )

if (MAGMA_ENABLE_CUDA)
  add_library( magma_sparse ${libsparse_all} )
  set_property(TARGET magma_sparse PROPERTY CUDA_STANDARD 11)
  target_link_libraries( magma_sparse
    magma
    ${blas_fix}
    ${LAPACK_LIBRARIES}
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
    magma_nvcc_flags
    )
else()
  add_library( magma_sparse ${libsparse_all} )
  target_link_libraries( magma_sparse
    magma
    ${blas_fix}
    ${LAPACK_LIBRARIES}
    hip::device
    roc::hipblas
    roc::hipsparse
    )
endif()
add_custom_target( sparse-lib DEPENDS magma_sparse )


# ----------------------------------------
# compile each tester

# save testers to testing/
# save tester lib files to testing_lib/ to avoid cluttering lib/
set( CMAKE_RUNTIME_OUTPUT_DIRECTORY testing )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY testing_lib )
set( CMAKE_LIBRARY_OUTPUT_DIRECTORY testing_lib )

# skip Fortran testers, which require an extra file from CUDA
foreach( filename ${testing_all} )
    if (filename MATCHES "\\.(c|cu|cpp)$")
        list( APPEND testing_all_cpp ${filename} )
    endif()
endforeach()
foreach( TEST ${testing_all_cpp} )
    string( REGEX REPLACE "\\.(cpp|f90|F90)" "" EXE ${TEST} )
    string( REGEX REPLACE "testing/" "" EXE ${EXE} )
    #message( "${TEST} --> ${EXE}" )
    add_executable( ${EXE} ${TEST} )
    target_link_libraries( ${EXE} tester lapacktest magma )
    list( APPEND testing ${EXE} )
endforeach()
add_custom_target( testing DEPENDS ${testing} )


# ----------------------------------------
# compile each sparse tester

if (MAGMA_ENABLE_CUDA)
  set(SPARSE_TEST_DIR "sparse/testing")
else()
  set(SPARSE_TEST_DIR "sparse_hip/testing")
endif()


set( CMAKE_RUNTIME_OUTPUT_DIRECTORY "${SPARSE_TEST_DIR}" )
cmake_policy( SET CMP0037 OLD)
foreach( TEST ${sparse_testing_all} )
    string( REGEX REPLACE "\\.(cpp|f90|F90)"     "" EXE ${TEST} )
    string( REGEX REPLACE "${SPARSE_TEST_DIR}/" "" EXE ${EXE} )
    #message( "${TEST} --> ${EXE}" )
    add_executable( ${EXE} ${TEST} )
    target_link_libraries( ${EXE} magma_sparse magma )
    list( APPEND sparse-testing ${EXE} )
endforeach()
add_custom_target( sparse-testing DEPENDS ${sparse-testing} )


# ----------------------------------------
# what to install
install( TARGETS magma magma_sparse ${blas_fix}
         RUNTIME DESTINATION bin
         LIBRARY DESTINATION lib
         ARCHIVE DESTINATION lib )
if (MAGMA_ENABLE_CUDA)
  file( GLOB headers include/*.h sparse/include/*.h "${CMAKE_BINARY_DIR}/include/*.h" )
else()
  file( GLOB headers include/*.h sparse_hip/include/*.h "${CMAKE_BINARY_DIR}/include/*.h" )
endif()
if (USE_FORTRAN)
    install( FILES ${headers} ${modules}
             DESTINATION include )
else()
    install( FILES ${headers} DESTINATION include )
endif()

# ----------------------------------------
# pkg-config
get_target_property(MAGMA_INCLUDE magma INCLUDE_DIRECTORIES)
foreach(dir ${MAGMA_INCLUDE})
    string(APPEND INCLUDE_COMPILER_STRING "-I${dir} ")
endforeach()
set( MAGMA_INCLUDE "${INCLUDE_COMPILER_STRING}" )
set( pkgconfig lib/pkgconfig/magma.pc )
message( STATUS "pkgconfig ${pkgconfig}" )
set( INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}" )
set( CFLAGS "${CMAKE_C_FLAGS}" )
set( CXXFLAGS "${CMAKE_CXX_FLAGS}" )
# CMake finds the Accelerate directory; we want -framework Accelerate for linking.
string( REPLACE "/System/Library/Frameworks/Accelerate.framework" "-framework Accelerate" LAPACK_LIBS "${LAPACK_LIBRARIES}" )
if (MAGMA_ENABLE_CUDA)
  string( REPLACE ";" " " LIBS
    "${blas_fix_lib} ${LAPACK_LIBS} -L${CUDAToolkit_LIBRARY_DIR} -lcudart -lcublas -lcusparse")
else()
  string( REPLACE ";" " " LIBS
     "${blas_fix_lib} ${LAPACK_LIBS} ${HIP} ${rocBLAS} ${rocSPARSE}" )
#    "${blas_fix_lib} ${LAPACK_LIBS} hip::device roc::hipblas roc::hipsparse" )
endif()
set( MAGMA_REQUIRED "" )
configure_file( "${pkgconfig}.in" "${pkgconfig}" @ONLY )
install( FILES "${CMAKE_BINARY_DIR}/${pkgconfig}"
         DESTINATION lib/pkgconfig )

# ----------------------------------------
get_directory_property( compile_definitions COMPILE_DEFINITIONS )

message( STATUS "Flags" )
message( STATUS "    CMAKE_INSTALL_PREFIX:  ${CMAKE_INSTALL_PREFIX}" )
message( STATUS "    CFLAGS:                ${CMAKE_C_FLAGS}" )
message( STATUS "    CXXFLAGS:              ${CMAKE_CXX_FLAGS}" )
if (MAGMA_ENABLE_CUDA)
  message( STATUS "    NVCCFLAGS:             ${CUDA_NVCC_FLAGS}" )
else()
  message( STATUS "    DEVCCFLAGS:            ${DEVCCFLAGS}" )
endif()
message( STATUS "    FFLAGS:                ${CMAKE_Fortran_FLAGS}" )
message( STATUS "    LIBS:                  ${LIBS}" )
message( STATUS "    blas_fix:              ${blas_fix}  (MacOS Accelerate only)" )
message( STATUS "    LAPACK_LIBRARIES:      ${LAPACK_LIBRARIES}"      )
message( STATUS "    INCLUDE_DIRECTORIES:   ${MAGMA_INCLUDE}"   )
if (MAGMA_ENABLE_CUDA)
  message( STATUS "    CUDA_CUDART_LIBRARY:   CUDA::cudart"   )
  message( STATUS "    CUDA_CUBLAS_LIBRARIES: CUDA::cublas" )
  message( STATUS "    CUDA_cusparse_LIBRARY: CUDA::cusparse" )
else()
  message( STATUS "    HIP_LIBRARY:   hip::device"   )
  message( STATUS "    HIP_BLAS_LIBRARIES: roc::hipblas" )
  message( STATUS "    HIP_sparse_LIBRARY: roc::hipsparse" )
endif()
message( STATUS "    Fortran modules:       ${modules}" )
