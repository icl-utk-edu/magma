#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 2.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date
#//////////////////////////////////////////////////////////////////////////////

# --------------------
# MAGMA configuration: ROCM, GCC, OpenBLAS

# -*-
# Most things here are similar to CUDA, and this script is designed to build
#   with CUDA and without ROCM (by setting BACKEND=cuda).

# -*-
# If there are any problems building with this new ROCM setup specifically,
#   you can email Cade Brown <cbrow216@vols.utk.edu> with your issues

# -*-
# note: when creating an application on top of hipMAGMA, you may need to use
#   the ROCM compiler (or some C++) compiler, even if your application is
#   expecting the C interface of MAGMA. This is due to the way ROCM defines
#   its types as C++ types. So, you can replace `CC=gcc` with `CC=hipcc`,
#   and your project should compile fine.


# --------------------
# configuration

# should MAGMA be built on CUDA (NVIDIA only) or ROCM (AMD or NVIDIA)
# enter 'cuda' or 'hip' respectively
BACKEND      = hip

# set these to their real paths
OPENBLASDIR ?= /usr/local/openblas
ROCM_PATH   ?= /opt/rocm
CUDADIR     ?= /usr/local/cuda

# require either hip or cuda
ifeq (,$(findstring $(BACKEND),hip cuda))
    $(error "'BACKEND' should be either 'cuda' or 'hip' (got $$BACKEND=$(BACKEND))")
endif

# --------------------
# programs

# set compilers
# (for ROCM, set all compilers to 'hipcc')
CC           = hipcc
CXX          = hipcc
FORT        ?= gfortran
HIPCC       ?= hipcc
NVCC        ?= nvcc
DEVCC       ?= NONE

# set from 'BACKEND'
ifeq ($(BACKEND),cuda)
    DEVCC    = $(NVCC)
else ifeq ($(BACKEND),hip)
    DEVCC    = $(HIPCC)
endif

# and utilities
ARCH        ?= ar
ARCHFLAGS   ?= cr
RANLIB      ?= ranlib

# sparse directory (uncomment to try & build MAGMA sparse/ on ROCM
#SPARSE_DIR  = sparse_hip

# --------------------
# flags/settings

# set our GPU targets
ifeq ($(BACKEND),cuda)
    # See a full table: https://nouveau.freedesktop.org/wiki/CodeNames/
    # note: provided by freedesktop, which is reliable, but not from the vendor
    GPU_TARGET = Volta Turing Ampere
else ifeq ($(BACKEND),hip)
    # See a full table:
    #   https://llvm.org/docs/AMDGPUUsage.html#amdgpu-processor-table
    # note: provided by LLVM, so not technically from the vendor.
    # Here's ROCm's source: https://rocm.github.io/hardware.html , but it is
    # not specific enough. TODO: find better link?
    GPU_TARGET ?= gfx900 gfx906 gfx908
endif

# Use -fPIC to make shared (.so) and static (.a) library;
# can be commented out if making only static library.
FPIC        = -fPIC

# use -fopenmp to support openmp builds
# comment out to build without openmp support
# NOTE: For some reason, using `-fopenmp` is not enough to add linker flags to
#   the compiler sometimes, so we need to add -L here too if there are problems
FOPENMP     = -fopenmp
#FOPENMP     =
#FOPENMP     = -fopenmp -L$(ROCM_PATH)/lib -I$(ROCM_PATH)/include

# now, generate our flags
CFLAGS      = -O3 $(FPIC) $(FOPENMP) -DNDEBUG -DADD_ -Wall -std=c99
CXXFLAGS    = -O3 $(FPIC) $(FOPENMP) -DNDEBUG -DADD_ -Wall -std=c++11
FFLAGS      = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument
F90FLAGS    = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument -x f95-cpp-input
LDFLAGS     =     $(FPIC) $(FOPENMP)

# add in the default for device compiling
DEVCCFLAGS  = -O3 -DNDEBUG -DADD_

# Add ROCM_VERSION for device compilation
DEVCCFLAGS += -DROCM_VERSION=$(shell ./tools/get-rocm-version.sh)

# add the flags in a backend-specific way
ifeq ($(BACKEND),cuda)
    DEVCCFLAGS += -Xcompiler "$(FPIC)" -Xcompiler "$(FOPENMP)" -std=c++11
else ifeq ($(BACKEND),hip)
    DEVCCFLAGS += $(FPIC) $(FOPENMP) -std=c++11
    # check for older versions of ROCM
    ifeq ($(shell hipconfig --version | cut -b -3),3.0)
        $(info Building with HIP 3.0)
        # they don't have hipblasComplex yet, so replace it manually
        DEVCCFLAGS += -DhipblasComplex=hipComplex -DhipblasDoubleComplex=hipDoubleComplex
    endif
endif


# --------------------
# libraries

LIB       =

# OpenBLAS
LIB      += -lopenblas

ifeq ($(BACKEND),cuda)
    LIB   += -lcublas -lcusparse -lcudart -lcudadevrt
else ifeq ($(BACKEND),hip)
    LIB   += -lhipblas -lhipsparse
endif

# --------------------
# directories

# define library directories preferably in your environment, or here.
LIBDIR    += -L$(OPENBLASDIR)/lib

# include each's directories
# note: we don't need to include the library directories, since nvcc or hipcc
#   should already know about them. But, `gcc` might not know about include
#   folders
ifeq ($(BACKEND),cuda)
    INC    += -I$(CUDADIR)/include
    LIBDIR += -L$(CUDADIR)/lib64
else ifeq ($(BACKEND),hip)
    INC    += -I$(ROCM_PATH)/include
    LIBDIR += -L$(ROCM_PATH)/lib
endif


# --------------------
# checks

# check for openblas
-include make.check-openblas

# checks for the specified backend
ifeq ($(BACKEND),cuda)
    -include make.check-cuda
else ifeq ($(BACKEND),hip)
    -include make.check-hip
endif


