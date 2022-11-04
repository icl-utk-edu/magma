# --------------------
# configuration

# should MAGMA be built on CUDA (NVIDIA only) or HIP (AMD or NVIDIA)
# enter 'cuda' or 'hip' respectively
BACKEND      = dpcpp

CC           = dpcpp
CXX          = dpcpp
FORT         = ifort
DEVCC       = dpcpp

# and utilities
ARCH        ?= ar
ARCHFLAGS   ?= cr
RANLIB      ?= ranlib

# Use -fPIC to make shared (.so) and static (.a) library;
# can be commented out if making only static library.
FPIC        = -fPIC

# use -fopenmp to support openmp builds
# comment out to build without openmp support
# NOTE: For some reason, using `-fopenmp` is not enough to add linker flags to
#   the compiler, so we need to add -L here too
#FOPENMP     =  -fopenmp-targets=amdgcn-amd-amdhsa
FOPENMP =  -fopenmp
#DEVOPENMP   = -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx906

# now, generate our flags
CFLAGS      = -O3 $(FPIC) $(FOPENMP) -DNDEBUG -DADD_ -Wall -std=c99 -DENABLE_TIMER
CXXFLAGS    = -O3 $(FPIC) $(FOPENMP) -DNDEBUG -DADD_ -Wall -std=c++17 -DENABLE_TIMER
FFLAGS      = -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument
F90FLAGS    = -O3 $(FPIC) -DNDEBUG -DADD_ #-Wall -Wno-unused-dummy-argument -x f95-cpp-input
LDFLAGS     =     $(FPIC) -fsycl-device-lib=all $(FOPENMP)

# add in the default for device compiling
DEVCCFLAGS  = -O3 -DNDEBUG -DADD_

# --------------------
# libraries

#LIB      = -lmkl_sycl -lmkl_intel_ilp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl
LIB      = -lmkl_sycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl


# --------------------
# directories
INC += -I$(MKLROOT)/include -L$(MKLROOT)/lib/intel64 -I$(ONEAPI_ROOT)/dpcpp-ct/latest/include -I$(CMPLR_ROOT)/linux/include/sycl
