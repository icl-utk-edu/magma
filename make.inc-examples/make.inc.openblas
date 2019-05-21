#//////////////////////////////////////////////////////////////////////////////
#   -- MAGMA (version 2.0) --
#      Univ. of Tennessee, Knoxville
#      Univ. of California, Berkeley
#      Univ. of Colorado, Denver
#      @date
#//////////////////////////////////////////////////////////////////////////////

# GPU_TARGET contains one or more of Fermi, Kepler, Maxwell, Pascal, Volta
# to specify for which GPUs you want to compile MAGMA:
#     Fermi   - NVIDIA compute capability 2.x cards
#     Kepler  - NVIDIA compute capability 3.x cards
#     Maxwell - NVIDIA compute capability 5.x cards
#     Pascal  - NVIDIA compute capability 6.x cards
#     Volta   - NVIDIA compute capability 7.x cards
# The default is "Kepler Maxwell Pascal".
# Note that NVIDIA no longer supports 1.x cards, as of CUDA 6.5.
# See http://developer.nvidia.com/cuda-gpus
#
#GPU_TARGET ?= Kepler Maxwell Pascal

# --------------------
# programs

CC        = gcc
CXX       = g++
NVCC      = nvcc
FORT      = gfortran

ARCH      = ar
ARCHFLAGS = cr
RANLIB    = ranlib


# --------------------
# flags

# Use -fPIC to make shared (.so) and static (.a) library;
# can be commented out if making only static library.
FPIC      = -fPIC

CFLAGS    = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -fopenmp
FFLAGS    = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument
F90FLAGS  = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument -x f95-cpp-input
NVCCFLAGS = -O3         -DNDEBUG -DADD_       -Xcompiler "$(FPIC)" -std=c++11
LDFLAGS   =     $(FPIC)                       -fopenmp

# C++11 (gcc >= 4.7) is not required, but has benefits like atomic operations
CXXFLAGS := $(CFLAGS) -std=c++11
CFLAGS   += -std=c99


# --------------------
# libraries

# gcc with OpenBLAS (includes LAPACK)
LIB       = -lopenblas

LIB      += -lcublas -lcusparse -lcudart -lcudadevrt


# --------------------
# directories

# define library directories preferably in your environment, or here.
#OPENBLASDIR ?= /usr/local/openblas
#CUDADIR ?= /usr/local/cuda
-include make.check-openblas
-include make.check-cuda

LIBDIR    = -L$(CUDADIR)/lib64 \
            -L$(OPENBLASDIR)/lib

INC       = -I$(CUDADIR)/include
