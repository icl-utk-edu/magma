# ------------------------------------------------------------------------------
# build process
#
# For hipMAGMA (branch of MAGMA that builds on HIP), the build process is basically:
# 0: Clone the repo, or download a release
# 1: Copy your `make.inc` for your specific platform
# 2: If its the repo, then run `make -f make.gen.interface_hip`, and `make -f make.gen.magmablas_hip`, 
#      and `make -f make.gen.testing_hip`
# 3: Now, run `make`, like normal.
# 4: You can make specific testers if the builds are failing
#



# ------------------------------------------------------------------------------
# programs
#
# Users should make all changes in make.inc
# It should not be necesary to change anything in here.

include make.inc

# --------------------
# configuration

# should MAGMA be built on CUDA (NVIDIA only) or HIP (AMD or NVIDIA)
# enter 'cuda' or 'hip' respectively
BACKEND     ?= cuda

# set these to their real paths
CUDADIR     ?= /usr/local/cuda
HIPDIR      ?= /opt/rocm/hip

# require either hip or cuda
ifeq (,$(findstring $(BACKEND),"hip cuda"))
    $(error "'BACKEND' should be either 'cuda' or 'hip' (got '$(BACKEND)')")
endif

# --------------------
# programs

# set compilers
CC          ?= gcc
CXX         ?= g++
FORT        ?= gfortran
HIPCC       ?= hipcc
NVCC        ?= nvcc
DEVCC       ?= NONE


# set from 'BACKEND'
ifeq ($(BACKEND),cuda)
    DEVCC = $(NVCC)

else ifeq ($(BACKEND),hip)
    DEVCC = $(HIPCC)

    # if we are using HIP, make sure generated sources are up to date
    # Technically, this 'recursive' make which we don't like to do, but also this is a simple solution
    #   that allows that file to handle all code generation
    # Another reason is that I don't want to flood the namespace (for example, that file also 
    #   defines an 'all' and 'clean' target as phonies)
    # So, in the future that whole file may be integrated, but for now this seems simplest
	# Detect number of jobs here, so it runs at an appropriate speed
    MAKE_PID := $(shell echo $$PPID)
    JOB_FLAG := $(filter -j%, $(subst -j ,-j,$(shell ps T | grep "^\s*$(MAKE_PID).*$(MAKE)")))
    JOBS     := $(subst -j,,$(JOB_FLAG))
    tmp := $(shell $(MAKE) -j$(JOBS) -f make.gen.hipMAGMA 1>&2)
else
    $(warning BACKEND: $(BACKEND) not recognized)
endif

# and utilities
ARCH        ?= ar
ARCHFLAGS   ?= cr
RANLIB      ?= ranlib


# --------------------
# flags/settings

# Use -fPIC to make shared (.so) and static (.a) library;
# can be commented out if making only static library.
FPIC        ?= -fPIC

# now, generate our flags
CFLAGS      ?= -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -fopenmp -std=c99
CXXFLAGS    ?= -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -fopenmp -std=c++11
FFLAGS      ?= -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument
F90FLAGS    ?= -O3 $(FPIC) -DNDEBUG -DADD_ -Wall -Wno-unused-dummy-argument -x f95-cpp-input
LDFLAGS     ?=     $(FPIC)                       -fopenmp

ifndef $(and DEVCCFLAGS, NVCCFLAGS)
    DEVCCFLAGS  ?= -O3         -DNDEBUG -DADD_
endif
# DEVCCFLAGS are populated later in `backend-specific`

# Extension for object files: o for unix, obj for Windows?
o_ext      ?= o

# where to install to?
prefix     ?= /usr/local/magma

# ------------------------------------------------------------------------------
# MAGMA-specific programs & flags

ifeq ($(blas_fix),1)
    # prepend -lblas_fix to LIB (it must come before LAPACK library/framework)
    LIB := -lblas_fix $(LIB)
endif

LIBS       = $(LIBDIR) $(LIB)

# preprocessor flags. See below for MAGMA_INC
CPPFLAGS   = $(INC) $(MAGMA_INC)

# where testers look for MAGMA libraries
RPATH      = -Wl,-rpath,${abspath ./lib}

codegen    = ./tools/codegen.py

ifeq ($(BACKEND),cuda)

	# Add legacy flags
	DEVCCFLAGS += $(NVCCFLAGS)
	
	# ------------------------------------------------------------------------------
	# NVCC options for the different cards
	# First, add smXX for architecture names
	# Internal CUDA architectures we support
	# TODO: Filter on regex to discard the named architectures?
	CUDA_ARCH_ := $(GPU_TARGET)
	ifneq ($(findstring Kepler, $(GPU_TARGET)),)
		CUDA_ARCH_ += sm_30
		CUDA_ARCH_ += sm_35
	endif
	ifneq ($(findstring Maxwell, $(GPU_TARGET)),)
		CUDA_ARCH_ += sm_50
	endif
	ifneq ($(findstring Pascal, $(GPU_TARGET)),)
		CUDA_ARCH_ += sm_60
	endif
	ifneq ($(findstring Volta, $(GPU_TARGET)),)
		CUDA_ARCH_ += sm_70
	endif
	ifneq ($(findstring Turing, $(GPU_TARGET)),)
		CUDA_ARCH_ += sm_75
	endif
	ifneq ($(findstring Ampere, $(GPU_TARGET)),)
		CUDA_ARCH_ += sm_80
	endif


        # Remember to add to CMakeLists.txt too!

        # Next, add compile options for specific smXX
        # sm_xx is binary, compute_xx is PTX for forward compatability
        # MIN_ARCH is lowest requested version
        #          Use it ONLY in magma_print_environment; elsewhere use __CUDA_ARCH__ or magma_getdevice_arch()
        # NV_SM    accumulates sm_xx for all requested versions
        # NV_COMP  is compute_xx for highest requested version
        #
        # See also $(info compile for ...) in Makefile

        ## Suggestion by Mark (from SLATE)
        # Valid architecture numbers
        # TODO: remove veryold ones?
    VALID_SMS = 30 32 35 37 50 52 53 60 61 62 70 72 75 80

    # code=sm_XX is binary, code=compute_XX is PTX
    GENCODE_SM      = -gencode arch=compute_$(sm),code=sm_$(sm)
    GENCODE_COMP    = -gencode arch=compute_$(sm),code=compute_$(sm)

    # Get gencode options for all sm_XX in cuda_arch_.
    NV_SM      := $(filter %, $(foreach sm, $(VALID_SMS),$(if $(findstring sm_$(sm), $(CUDA_ARCH_)),$(GENCODE_SM))))
    NV_COMP    := $(filter %, $(foreach sm, $(VALID_SMS),$(if $(findstring sm_$(sm), $(CUDA_ARCH_)),$(GENCODE_COMP))))

    ifeq ($(NV_SM),)
        $(error GPU_TARGET, currently $(GPU_TARGET), must contain one or more of Fermi, Kepler, Maxwell, Pascal, Volta, Turing, or valid sm_[0-9][0-9]. Please edit your make.inc file)
    else
        # Get last option (last 2 words) of nv_compute.
        nwords := $(words $(NV_COMP))
        nwords_1 := $(shell expr $(nwords) - 1)
        NV_COMP_LAST := $(wordlist $(nwords_1), $(nwords), $(NV_COMP))
    endif

    # Use all sm_XX (binary), and the last compute_XX (PTX) for forward compatibility.
    DEVCCFLAGS += $(NV_SM) $(NV_COMP_LAST)
    LIBS += -lcublas -lcudart

    # Get first (minimum) architecture
    MIN_ARCH := $(wordlist 1, 1, $(foreach sm, $(VALID_SMS),$(if $(findstring sm_$(sm), $(CUDA_ARCH_)),$(sm)0)))
	ifeq ($(MIN_ARCH),)
		$(error GPU_TARGET, currently $(GPU_TARGET), must contain one or more of Fermi, Kepler, Maxwell, Pascal, Volta, Turing, or valid sm_[0-9][0-9]. Please edit your make.inc file)
	endif

	DEVCCFLAGS += -DHAVE_CUDA -DHAVE_CUBLAS -DMIN_CUDA_ARCH=$(MIN_ARCH)


	CFLAGS    += -DMIN_CUDA_ARCH=$(MIN_ARCH)
	CXXFLAGS  += -DMIN_CUDA_ARCH=$(MIN_ARCH)

	CFLAGS    += -DHAVE_CUDA -DHAVE_CUBLAS
	CXXFLAGS  += -DHAVE_CUDA -DHAVE_CUBLAS
else ifeq ($(BACKEND),hip)

	# ------------------------------------------------------------------------------
	# hipcc backend
	# Source: https://llvm.org/docs/AMDGPUUsage.html#target-triples

	# Filter our human readable names and replace with numeric names
	HIP_ARCH_ := $(GPU_TARGET)
	ifneq ($(findstring kaveri, $(GPU_TARGET)),)
		HIP_ARCH_ += gfx700
	endif
	ifneq ($(findstring hawaii, $(GPU_TARGET)),)
		HIP_ARCH_ += gfx701
	endif
	ifneq ($(findstring kabini, $(GPU_TARGET)),)
		HIP_ARCH_ += gfx703
	endif
	ifneq ($(findstring mullins, $(GPU_TARGET)),)
		HIP_ARCH_ += gfx703
	endif
	ifneq ($(findstring bonaire, $(GPU_TARGET)),)
		HIP_ARCH_ += gfx704
	endif
	ifneq ($(findstring carrizo, $(GPU_TARGET)),)
		HIP_ARCH_ += gfx801
	endif
	ifneq ($(findstring iceland, $(GPU_TARGET)),)
		HIP_ARCH_ += gfx802
	endif
	ifneq ($(findstring tonga, $(GPU_TARGET)),)
		HIP_ARCH_ += gfx802
	endif
	ifneq ($(findstring fiji, $(GPU_TARGET)),)
		HIP_ARCH_ += gfx803
	endif
	# These are in the documentation, and the leftmost column *seems* like a continuation
	#   of gfx803
	ifneq ($(findstring polaris10, $(GPU_TARGET)),)
		HIP_ARCH_ += gfx803
	endif
	ifneq ($(findstring polaris11, $(GPU_TARGET)),)
		HIP_ARCH_ += gfx803
	endif

	ifneq ($(findstring tongapro, $(GPU_TARGET)),)
		HIP_ARCH_ += gfx805
	endif
	ifneq ($(findstring stoney, $(GPU_TARGET)),)
		HIP_ARCH_ += gfx810
	endif

    ## Suggestion by Mark (from SLATE)
    # Valid architecture numbers
    # TODO: remove veryold ones?
    VALID_GFXS = 600 601 602 700 701 702 703 704 705 801 802 803 805 810 900 902 904 906 908 909 90c 1010 1011 1012 1030 1031 1032 1033


	# Generated GFX option
    TARGET_GFX      = --amdgpu-target=gfx$(gfx)

    # Get gencode options for all sm_XX in cuda_arch_.
    AMD_GFX    := $(filter %, $(foreach gfx, $(VALID_GFXS),$(if $(findstring gfx$(gfx), $(HIP_ARCH_)),$(TARGET_GFX))))

    ifeq ($(AMD_GFX),)
        $(error GPU_TARGET, currently $(GPU_TARGET), must contain one or more of the targets for AMDGPUs (https://llvm.org/docs/AMDGPUUsage.html#target-triples), or valid gfx[0-9][0-9][0-9][0-9]?. Please edit your make.inc file)
    else
    endif

    # Use all sm_XX (binary), and the last compute_XX (PTX) for forward compatibility.
    DEVCCFLAGS += $(AMD_GFX)

    # Get first (minimum) architecture
    MIN_ARCH := $(wordlist 1, 1, $(foreach gfx, $(VALID_GFXS),$(if $(findstring gfx$(gfx), $(HIP_ARCH_)),$(gfx))))
	ifeq ($(MIN_ARCH),)
		$(error GPU_TARGET, currently $(GPU_TARGET), did not contain a minimum arch)
	endif

    # just so we know
    CFLAGS     += -DHAVE_HIP
    CXXFLAGS   += -DHAVE_HIP
    DEVCCFLAGS += -DHAVE_HIP
endif


# ------------------------------------------------------------------------------
# Define the pointer size for fortran compilation
# If there's an issue compiling sizeptr, assume 8 byte (64 bit) pointers
PTRFILE = control/sizeptr.c
PTROBJ  = control/sizeptr.$(o_ext)
PTREXEC = control/sizeptr
PTRSIZE = $(shell if [ -x $(PTREXEC) ]; then $(PTREXEC); else echo 8; fi)
PTROPT  = -Dmagma_devptr_t="integer(kind=$(PTRSIZE))"

$(PTREXEC): $(PTROBJ)
	-$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<
	touch $@

$(PTROBJ): $(PTRFILE)
	-$(CC) $(CFLAGS) -c -o $@ $<
	touch $@


# ------------------------------------------------------------------------------
# include sub-directories

# variables that multiple sub-directories add to.
# these MUST be := defined, not = defined, for $(cdir) to work.
hdr                  :=
libmagma_src         :=
libmagma_dynamic_src :=
testing_src          :=
libsparse_src        :=
libsparse_dynamic_src:=
sparse_testing_src   :=

subdirs := \
	blas_fix            \
	control             \
	include             \
	src                 \
	testing/lin         \

# the directory in which the MAGMA sparse source is located
# change to sparse_hip for hipified sources
# right now, just use old one so the dense section still builds

ifeq ($(BACKEND),cuda)
	SPARSE_DIR ?= sparse
	subdirs += interface_cuda
	subdirs += testing
	subdirs += magmablas

    # add all sparse folders
	# Don't do it for HIP yet
    subdirs += $(SPARSE_DIR) $(SPARSE_DIR)/blas $(SPARSE_DIR)/control $(SPARSE_DIR)/include $(SPARSE_DIR)/src $(SPARSE_DIR)/testing

else ifeq ($(BACKEND),hip)
	SPARSE_DIR ?= ./sparse_hip
	subdirs += interface_hip
	subdirs += magmablas_hip
	subdirs += testing

    subdirs += $(SPARSE_DIR) $(SPARSE_DIR)/blas $(SPARSE_DIR)/control $(SPARSE_DIR)/include $(SPARSE_DIR)/src $(SPARSE_DIR)/testing

endif




Makefiles := $(addsuffix /Makefile.src, $(subdirs))

#$(info $$Makefiles=$(Makefiles))

include $(Makefiles)

-include Makefile.internal
-include Makefile.local
-include Makefile.gen


#$(info $$libmagma_src=$(libmagma_src))
#$(info $$libmagma_all=$(libmagma_all))

# ------------------------------------------------------------------------------
# objects

ifeq ($(FORT),)
    liblapacktest_all2 := $(filter     %_no_fortran.cpp, $(liblapacktest_all))
else
    liblapacktest_all2 := $(filter-out %_no_fortran.cpp, $(liblapacktest_all))
endif

ifeq ($(FORT),)
    libmagma_all := $(filter-out %.f %.f90 %.F90, $(libmagma_all))
    testing_all  := $(filter-out %.f %.f90 %.F90, $(testing_all))
endif

libmagma_obj       := $(addsuffix .$(o_ext), $(basename $(libmagma_all)))
libblas_fix_obj    := $(addsuffix .$(o_ext), $(basename $(libblas_fix_src)))
libtest_obj        := $(addsuffix .$(o_ext), $(basename $(libtest_all)))
liblapacktest_obj  := $(addsuffix .$(o_ext), $(basename $(liblapacktest_all2)))
testing_obj        := $(addsuffix .$(o_ext), $(basename $(testing_all)))
libsparse_obj      := $(addsuffix .$(o_ext), $(basename $(libsparse_all)))
sparse_testing_obj := $(addsuffix .$(o_ext), $(basename $(sparse_testing_all)))

ifneq ($(libmagma_dynamic_src),)
libmagma_dynamic_obj := $(addsuffix .$(o_ext),      $(basename $(libmagma_dynamic_all)))
  ifeq ($(BACKEND),cuda)
    libmagma_dlink_obj   := magmablas/dynamic.link.o
  else ifeq ($(BACKEND),hip)
    libmagma_dlink_obj   := magmablas_hip/dynamic.link.o
  endif

libmagma_obj         += $(libmagma_dynamic_obj) $(libmagma_dlink_obj)
endif

ifneq ($(libsparse_dynamic_src),)
libsparse_dynamic_obj := $(addsuffix .$(o_ext),      $(basename $(libsparse_dynamic_all)))

ifeq ($(BACKEND),cuda)
  libsparse_dlink_obj   := $(SPARSE_DIR)/blas/dynamic.link.o
else ifeq ($(BACKEND),hip)
  # No dynamic parallelism support in HIP
  #libsparse_dlink_obj   := $(SPARSE_DIR)/blas/dynamic.link.o
endif


libsparse_obj         += $(libsparse_dynamic_obj) $(libsparse_dlink_obj)
endif

deps :=
deps += $(addsuffix .d, $(basename $(libmagma_all)))
deps += $(addsuffix .d, $(basename $(libblas_fix_src)))
deps += $(addsuffix .d, $(basename $(libtest_all)))
deps += $(addsuffix .d, $(basename $(lapacktest_all2)))
deps += $(addsuffix .d, $(basename $(testing_all)))
deps += $(addsuffix .d, $(basename $(libsparse_all)))
deps += $(addsuffix .d, $(basename $(sparse_testing_all)))

# headers must exist before compiling objects, but we don't want to require
# re-compiling the whole library for every minor header change,
# so use order-only prerequisite (after "|").
$(libmagma_obj):       | $(header_all)
$(libtest_obj):        | $(header_all)
$(testing_obj):        | $(header_all)
$(libsparse_obj):      | $(header_all)
$(sparse_testing_obj): | $(header_all)

# changes to testings.h require re-compiling, e.g., if magma_opts changes
$(testing_obj):        testing/testings.h
$(libtest_obj):        testing/testings.h
$(sparse_testing_obj): testing/testings.h

# this allows "make force=force" to force re-compiling
$(libmagma_obj):       $(force)
$(libblas_fix_obj):    $(force)
$(libtest_obj):        $(force)
$(liblapacktest_obj):  $(force)
$(testing_obj):        $(force)
$(libsparse_obj):      $(force)
$(sparse_testing_obj): $(force)

force: ;


# ----- include paths
MAGMA_INC  = -I./include -I./testing

$(libmagma_obj):       MAGMA_INC += -I./control
$(libtest_obj):        MAGMA_INC += -I./testing
$(testing_obj):        MAGMA_INC += -I./testing

ifeq ($(BACKEND),cuda) 
$(libsparse_obj):      MAGMA_INC += -I./control -I./magmablas -I./sparse/include -I./sparse/control
$(sparse_testing_obj): MAGMA_INC += -I./sparse/include -I./sparse/control -I./testing
else ifeq ($(BACKEND),hip)
$(libsparse_obj):      MAGMA_INC += -I./control -I./magmablas_hip -I$(SPARSE_DIR)/include -I$(SPARSE_DIR)/control
$(sparse_testing_obj): MAGMA_INC += -I$(SPARSE_DIR)/include -I$(SPARSE_DIR)/control -I./testing
endif


# ----- headers
# to test that headers are self-contained,
# pre-compile each into a header.h.gch file using "g++ ... -c header.h"
header_gch := $(addsuffix .gch, $(filter-out %.cuh, $(header_all)))

test_headers: $(header_gch)

%.h.gch: %.h
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<


# ----- libraries
libmagma_a      := lib/libmagma.a
libmagma_so     := lib/libmagma.so
libblas_fix_a   := lib/libblas_fix.a
libtest_a       := testing/libtest.a
liblapacktest_a := testing/lin/liblapacktest.a
libsparse_a     := lib/libmagma_sparse.a
libsparse_so    := lib/libmagma_sparse.so

# static libraries
libs_a := \
	$(libmagma_a)		\
	$(libtest_a)		\
	$(liblapacktest_a)	\
	$(libblas_fix_a)	\
	$(libsparse_a)		\

# shared libraries
libs_so := \
	$(libmagma_so)		\
	$(libsparse_so)		\

#$(info $$libmagma_obj=$(libmagma_obj))

# add objects to libraries
$(libmagma_a):      $(libmagma_obj)
$(libmagma_so):     $(libmagma_obj)
$(libblas_fix_a):   $(libblas_fix_obj)
$(libtest_a):       $(libtest_obj)
$(liblapacktest_a): $(liblapacktest_obj)
$(libsparse_a):     $(libsparse_obj)
$(libsparse_so):    $(libsparse_obj)

# sparse requires libmagma
$(libsparse_so): | $(libmagma_so)


# ----- testers
testing_c_src := $(filter %.c %.cpp,       $(testing_all))
testing_f_src := $(filter %.f %.f90 %.F90, $(testing_all))
testers       := $(basename $(testing_c_src))
testers_f     := $(basename $(testing_f_src))

sparse_testers := $(basename $(sparse_testing_all))

# depend on static libraries
# see below for libmagma, which is either static or shared
$(testers):        $(libtest_a) $(liblapacktest_a)
$(testers_f):      $(libtest_a) $(liblapacktest_a)
$(sparse_testers): $(libtest_a)  # doesn't use liblapacktest

# ----- blas_fix
# if using blas_fix (e.g., on MacOS), libmagma requires libblas_fix
ifeq ($(blas_fix),1)
    $(libmagma_a):     | $(libblas_fix_a)
    $(libmagma_so):    | $(libblas_fix_a)
    $(testers):        | $(libblas_fix_a)
    $(testers_f):      | $(libblas_fix_a)
    $(sparse_testers): | $(libblas_fix_a)
endif


# ------------------------------------------------------------------------------
# MacOS (darwin) needs shared library's path set
# $OSTYPE may not be exported from the shell, so echo it
ostype = ${shell echo $${OSTYPE}}
ifneq ($(findstring darwin, ${ostype}),)
    $(libmagma_so):  LDFLAGS += -install_name @rpath/$(notdir $(libmagma_so))
    $(libsparse_so): LDFLAGS += -install_name @rpath/$(notdir $(libsparse_so))
endif


# ------------------------------------------------------------------------------
# targets

.PHONY: all lib static shared clean test dense sparse

.DEFAULT_GOAL := all

all: dense sparse

dense: lib test

sparse: sparse-lib sparse-test

# lib defined below in shared libraries, depending on fPIC

test: testing

testers_f: $(testers_f)

ifeq ($(BACKEND),cuda)
sparse-test: sparse/testing
sparse-testing: sparse/testing
else ifeq ($(BACKEND),hip)
sparse-test: $(SPARSE_DIR)/testing
sparse-testing: $(SPARSE_DIR)/testing
endif
# cleangen is defined in Makefile.gen; cleanall also does cleanmake in Makefile.internal
cleanall: clean cleangen

# TODO: should this do all $(subdirs) clean?
clean: lib/clean testing/clean
	-rm -f $(deps)


# ------------------------------------------------------------------------------
# shared libraries

# check whether all FLAGS have -fPIC
have_fpic = $(and $(findstring -fPIC, $(CFLAGS)),   \
                  $(findstring -fPIC, $(CXXFLAGS)), \
                  $(findstring -fPIC, $(FFLAGS)),   \
                  $(findstring -fPIC, $(F90FLAGS)), \
                  $(findstring -fPIC, $(DEVCCFLAGS)))

# --------------------
# if all flags have -fPIC: compile shared & static
ifneq ($(have_fpic),)

    lib: static shared

    sparse-lib: sparse-static sparse-shared

    shared: $(libmagma_so)

    sparse-shared: $(libsparse_so)

    # as a shared library, changing libmagma.so does NOT require re-linking testers,
    # so use order-only prerequisite (after "|").
    $(testers):        | $(libmagma_a) $(libmagma_so)
    $(testers_f):      | $(libmagma_a) $(libmagma_so)
    $(sparse_testers): | $(libmagma_a) $(libmagma_so) $(libsparse_a) $(libsparse_so)

    libs := $(libmagma_a) $(libmagma_so) $(libsparse_a) $(libsparse_so)

# --------------------
# else: some flags are missing -fPIC: compile static only
else

    lib: static

    sparse-lib: sparse-static

    shared:
	@echo "Error: 'make shared' requires CFLAGS, CXXFLAGS, FFLAGS, F90FLAGS, and NVCCFLAGS to have -fPIC."
	@echo "This is now the default in most example make.inc.* files, except atlas."
	@echo "Please edit your make.inc file and uncomment FPIC."
	@echo "After updating make.inc, please 'make clean && make shared && make test'."
	@echo "To compile only a static library, use 'make static'."

    sparse-shared: shared

    # as a static library, changing libmagma.a does require re-linking testers,
    # so use regular prerequisite.
    $(testers):        $(libmagma_a)
    $(testers_f):      $(libmagma_a)
    $(sparse_testers): $(libmagma_a) $(libsparse_a)

    libs := $(libmagma_a) $(libsparse_a)

endif
# --------------------

ifeq ($(blas_fix),1)
    libs += $(libblas_fix_a)
endif


# ------------------------------------------------------------------------------
# static libraries

static: $(libmagma_a)

sparse-static: $(libsparse_a)


# ------------------------------------------------------------------------------
# sub-directory targets

control_obj          := $(filter          control/%.o, $(libmagma_obj))
src_obj              := $(filter              src/%.o, $(libmagma_obj))

sparse_control_obj   := $(filter   $(SPARSE_DIR)/control/%.o, $(libsparse_obj))
sparse_blas_obj      := $(filter      $(SPARSE_DIR)/blas/%.o, $(libsparse_obj))
sparse_src_obj       := $(filter       $(SPARSE_DIR)/src/%.o, $(libsparse_obj))


ifeq ($(BACKEND),cuda)
  interface_cuda_obj   := $(filter   interface_cuda/%.o, $(libmagma_obj))
  magmablas_obj        := $(filter        magmablas/%.o, $(libmagma_obj))
else ifeq ($(BACKEND),hip)
  interface_hip_obj   := $(filter     interface_hip/%.o, $(libmagma_obj))
  magmablas_hip_obj   := $(filter     magmablas_hip/%.o, $(libmagma_obj))
  #$(info $$magmablas_hip_obj=$(magmablas_hip_obj))
endif



# ----------
# sub-directory builds
include:             $(header_all)

blas_fix:            $(libblas_fix_a)

control:             $(control_obj)



ifeq ($(BACKEND),cuda)
	interface_cuda:      $(interface_cuda_obj)
	magmablas:           $(magmablas_obj)
else ifeq ($(BACKEND),hip)
	interface_hip:       $(interface_hip_obj)
	magmablas_hip:       $(magmablas_hip_obj)
endif


src:                 $(src_obj)

testing:             $(testers)

sparse/blas:    $(sparse_blas_obj)

sparse/control: $(sparse_control_obj)

sparse/src:     $(sparse_src_obj)

sparse/testing: $(sparse_testers)

run_test: test
	cd testing && ./run_tests.py

# ----------
# sub-directory clean
include/clean:
	-rm -f $(shdr) $(dhdr) $(chdr)

blas_fix/clean:
	-rm -f $(libblas_fix_a) $(libblas_fix_obj)

control/clean:
	-rm -f $(control_obj) include/*.mod control/*.mod

ifeq ($(BACKEND),cuda)
interface_cuda/clean:
	-rm -f $(interface_cuda_obj)

magmablas/clean:
	-rm -f $(magmablas_obj)

else ifeq ($(BACKEND),hip)

interface_hip/clean:
	-rm -f $(interface_hip_obj)

magmablas_hip/clean:
	-rm -f $(magmablas_hip_obj)

endif

src/clean:
	-rm -f $(src_obj)

testing/cleanexe:
	-rm -f $(testers) $(testers_f)

testing/clean: testing/lin/clean
	-rm -f $(testers) $(testers_f) $(testing_obj) \
		$(libtest_a) $(libtest_obj)

testing/lin/clean:
	-rm -f $(liblapacktest_a) $(liblapacktest_obj)

# hmm... what should lib/clean do? just the libraries, not objects?
lib/clean: blas_fix/clean sparse/clean
	-rm -f $(libmagma_a) $(libmagma_so) $(libmagma_obj)

sparse/clean: sparse/testing/clean
	-rm -f $(libsparse_a) $(libsparse_so) $(libsparse_obj)

sparse/blas/clean:
	-rm -f $(sparse_blas_obj)

sparse/control/clean:
	-rm -f $(sparse_control_obj)

sparse/src/clean:
	-rm -f $(sparse_src_obj)

sparse/testing/clean:
	-rm -f $(sparse_testers) $(sparse_testing_obj)


# ------------------------------------------------------------------------------
# rules

.DELETE_ON_ERROR:

.SUFFIXES:

# object file rules

%.$(o_ext): %.f
	$(FORT) $(FFLAGS) -c -o $@ $<

%.$(o_ext): %.f90
	$(FORT) $(F90FLAGS) $(CPPFLAGS) -c -o $@ $<
	-mv $(notdir $(basename $@)).mod include/

%.$(o_ext): %.F90 $(PTREXEC)
	$(FORT) $(F90FLAGS) $(CPPFLAGS) $(PTROPT) -c -o $@ $<
	-mv $(notdir $(basename $@)).mod include/
 
%.$(o_ext): %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) -c -o $@ $<



# ------------------------------------------------------------------------------
# DEVICE kernels

# set the device extension
ifeq ($(BACKEND),cuda)
d_ext := cu
else ifeq ($(BACKEND),hip)
d_ext := cpp
CXXFLAGS += -D__HIP_PLATFORM_HCC__
endif


ifeq ($(BACKEND),cuda)

%.i: %.$(d_ext)
	$(DEVCC) -E $(DEVCCFLAGS) $(CPPFLAGS) -c -o $@ $<

%.$(o_ext): %.$(d_ext)
	$(DEVCC) $(DEVCCFLAGS) $(CPPFLAGS)  -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

else ifeq ($(BACKEND),hip)

%.hip.o: %.hip.cpp
	$(DEVCC) $(DEVCCFLAGS) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

# use `hipcc` for all .cpp's. It may be a bit slower (althought I haven't tested it)
# but there's no good way to tell whether or not it fails for some reason. (buggy
# hipcc is probably the culprit)
#%.o: %.cpp
#	$(DEVCC) $(DEVCCFLAGS) $(CPPFLAGS) -c -o $@ $<

endif

# assume C++ for headers; needed for Fortran wrappers
%.i: %.h
	$(CXX) -E $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<

%.i: %.c
	$(CC) -E $(CFLAGS) $(CPPFLAGS) -c -o $@ $<

%.i: %.cpp
	$(CXX) -E $(CXXFLAGS) $(CPPFLAGS) -c -o $@ $<


ifeq ($(BACKEND),cuda)
$(libmagma_dynamic_obj): %.$(o_ext): %.$(d_ext)
	$(DEVCC) $(DEVCCFLAGS) $(CPPFLAGS) -I$(SPARSE_DIR)/include -dc -o $@ $<

$(libmagma_dlink_obj): $(libmagma_dynamic_obj)
	$(DEVCC) $(DEVCCFLAGS) $(CPPFLAGS) -dlink -I$(SPARSE_DIR)/include -o $@ $^

$(libsparse_dynamic_obj): %.$(o_ext): %.$(d_ext)
	$(DEVCC) $(DEVCCFLAGS) $(CPPFLAGS) -I$(SPARSE_DIR)/include -dc -o $@ $<

$(libsparse_dlink_obj): $(libsparse_dynamic_obj)
	$(DEVCC) $(DEVCCFLAGS) $(CPPFLAGS) -dlink -I$(SPARSE_DIR)/include -o $@ $^

else ifeq ($(BACKEND),hip)

$(libmagma_dynamic_obj): %.$(o_ext): %.$(d_ext)
	$(DEVCC) $(DEVCCFLAGS) $(CPPFLAGS) -I$(SPARSE_DIR)/include -c -o $@ $<

$(libmagma_dlink_obj): $(libmagma_dynamic_obj)
	$(DEVCC) $(DEVCCFLAGS) $(CPPFLAGS) -dlink -I$(SPARSE_DIR)/include -o $@ $^

$(libsparse_dynamic_obj): %.$(o_ext): %.$(d_ext)
	$(DEVCC) $(DEVCCFLAGS) $(CPPFLAGS) -I$(SPARSE_DIR)/include -c -o $@ $<

$(libsparse_dlink_obj): $(libsparse_dynamic_obj)
	$(DEVCC) $(DEVCCFLAGS) $(CPPFLAGS) -I$(SPARSE_DIR)/include -c -o $@ $^

endif
# ------------------------------------------------------------------------------
# library rules

$(libs_a):
	@echo "===== static library $@"
	$(ARCH) $(ARCHFLAGS) $@ $^
	$(RANLIB) $@
	@echo

ifneq ($(have_fpic),)
    $(libmagma_so):
	@echo "===== shared library $@"
	$(CXX) $(LDFLAGS) -shared -o $@ \
		$^ \
		-L./lib $(LIBS)
	@echo

    # Can't add -Llib -lmagma to LIBS, because that would apply to libsparse_so's
    # prerequisites, namely libmagma_so. So libmagma and libsparse need different rules.
    # See Make section 6.11 Target-specific Variable Values.
    $(libsparse_so):
	@echo "===== shared library $@"
	$(CXX) $(LDFLAGS) -shared -o $@ \
		$^ \
		-L./lib $(LIBS) -lmagma
	@echo
else
    # missing -fPIC: "make shared" prints warning
    $(libs_so): shared
endif


# ------------------------------------------------------------------------------
# testers

# link testing_foo from testing_foo.o
$(testers): %: %.$(o_ext)
	$(CXX) $(LDFLAGS) $(RPATH) \
	-o $@ $< \
	-L./testing -ltest \
	-L./lib -lmagma \
	-L./testing/lin -llapacktest \
	$(LIBS)

# link Fortran testing_foo from testing_foo.o
$(testers_f): %: %.$(o_ext)
	$(FORT) $(LDFLAGS) $(RPATH) \
	-o $@ $< \
	-L./testing -ltest \
	-L./testing/lin -llapacktest \
	-L./lib -lmagma \
	$(LIBS)

# link sparse testing_foo from testing_foo.o
$(sparse_testers): %: %.$(o_ext)
	$(CXX) $(LDFLAGS) $(RPATH) \
	-o $@ $< \
	-L./testing -ltest \
	-L./lib -lmagma_sparse -lmagma \
	$(LIBS)


# ------------------------------------------------------------------------------
# filter out MAGMA-specific options for pkg-config
#TODO: add hip specific ones
INSTALL_FLAGS := $(filter-out \
	-DMAGMA_NOAFFINITY -DMAGMA_SETAFFINITY -DMAGMA_WITH_ACML -DMAGMA_WITH_MKL -DUSE_FLOCK \
	-DMIN_CUDA_ARCH=100 -DMIN_CUDA_ARCH=200 -DMIN_CUDA_ARCH=300 \
	-DMIN_CUDA_ARCH=350 -DMIN_CUDA_ARCH=500 -DMIN_CUDA_ARCH=600 -DMIN_CUDA_ARCH=610 \
	-DHAVE_CUBLAS -DHAVE_clBLAS -DHAVE_HIP \
	-fno-strict-aliasing -fPIC -O0 -O1 -O2 -O3 -pedantic -std=c99 -stdc++98 -stdc++11 \
	-Wall -Wshadow -Wno-long-long, $(CFLAGS))

INSTALL_LDFLAGS := $(filter-out -fPIC -Wall, $(LDFLAGS))

install_dirs:
	mkdir -p $(DESTDIR)$(prefix)
	mkdir -p $(DESTDIR)$(prefix)/include
	mkdir -p $(DESTDIR)$(prefix)/lib$(LIB_SUFFIX)
	mkdir -p $(DESTDIR)$(prefix)/lib$(LIB_SUFFIX)/pkgconfig

install: lib sparse-lib install_dirs
        # MAGMA
	cp include/*.h         $(DESTDIR)$(prefix)/include
	cp include/*.mod       $(DESTDIR)$(prefix)/include
	cp $(SPARSE_DIR)/include/*.h  $(DESTDIR)$(prefix)/include
	cp $(libs)             $(DESTDIR)$(prefix)/lib$(LIB_SUFFIX)
	${MAKE} pkgconfig

pkgconfig:
        # pkgconfig
	mkdir -p $(DESTDIR)$(prefix)/lib$(LIB_SUFFIX)/pkgconfig
	cat lib/pkgconfig/magma.pc.in                   | \
	sed -e s:@INSTALL_PREFIX@:"$(prefix)":          | \
	sed -e s:@CFLAGS@:"$(INSTALL_FLAGS) $(INC)":    | \
	sed -e s:@LIBS@:"$(INSTALL_LDFLAGS) $(LIBS)":   | \
	sed -e s:@MAGMA_REQUIRED@::                       \
	    > $(DESTDIR)$(prefix)/lib$(LIB_SUFFIX)/pkgconfig/magma.pc


# ------------------------------------------------------------------------------
# files.txt is nearly all (active) files in SVN, excluding directories. Useful for rsync, etc.
# files-doxygen.txt is all (active) source files in SVN, used by Doxyfile-fast

# excludes non-active directories like obsolete.
# excludes directories by matching *.* files (\w\.\w) and some exceptions like Makefile.
files.txt: force
	hg st -m -a -c \
		| perl -pe 's/^. +//' | sort \
		| egrep -v '^\.$$|obsolete|deprecated|contrib\b|^exp' \
		| egrep '\w\.\w|Makefile|docs|run' \
		> files.txt
	egrep -v '(\.html|\.css|\.f|\.in|\.m|\.mtx|\.pl|\.png|\.sh|\.txt)$$|checkdiag|COPYRIGHT|docs|example|make\.|Makefile|quark|README|Release|results|testing_|testing/lin|testing/matgen|tools' files.txt \
		| perl -pe 'chomp; $$_ = sprintf("\t../%-57s\\\n", $$_);' \
		> files-doxygen.txt

# files.txt per sub-directory
subdir_files = $(addsuffix /files.txt,$(subdirs) $(sparse_subdirs))

$(subdir_files): force
	cd $(dir $@) && hg st -m -a -c -X '*/*' . \
		| perl -pe 's/^. +//' | sort \
		| egrep -v '^\.$$|obsolete|deprecated|contrib\b|^exp' \
		| egrep '\w\.\w|Makefile|docs|run' \
		> files.txt


# ------------------------------------------------------------------------------
echo:
	@echo "====="
	@echo "hdr                $(hdr)\n"
	@echo "header_all         $(header_all)\n"
	@echo "header_gch         $(header_gch)\n"
	@echo "====="
	@echo "libmagma_src       $(libmagma_src)\n"
	@echo "libmagma_all       $(libmagma_all)\n"
	@echo "libmagma_obj       $(libmagma_obj)\n"
	@echo "libmagma_a         $(libmagma_a)"
	@echo "libmagma_so        $(libmagma_so)"
	@echo "====="
	@echo "libmagma_dynamic_src $(libmagma_dynamic_src)\n"
	@echo "libmagma_dynamic_all $(libmagma_dynamic_all)\n"
	@echo "libmagma_dynamic_obj $(libmagma_dynamic_obj)\n"
	@echo "libmagma_dlink_obj   $(libmagma_dlink_obj)\n"
	@echo "====="
	@echo "libsparse_src      $(libsparse_src)\n"
	@echo "libsparse_all      $(libsparse_all)\n"
	@echo "libsparse_obj      $(libsparse_obj)\n"
	@echo "libsparse_a        $(libsparse_a)"
	@echo "libsparse_so       $(libsparse_so)"
	@echo "====="
	@echo "blas_fix           $(blas_fix)"
	@echo "libblas_fix_src    $(libblas_fix_src)"
	@echo "libblas_fix_a      $(libblas_fix_a)"
	@echo "====="
	@echo "libtest_src        $(libtest_src)\n"
	@echo "libtest_all        $(libtest_all)\n"
	@echo "libtest_obj        $(libtest_obj)\n"
	@echo "libtest_a          $(libtest_a)\n"
	@echo "====="
	@echo "liblapacktest_src  $(liblapacktest_src)\n"
	@echo "liblapacktest_all  $(liblapacktest_all)\n"
	@echo "liblapacktest_all2 $(liblapacktest_all2)\n"
	@echo "liblapacktest_obj  $(liblapacktest_obj)\n"
	@echo "liblapacktest_a    $(liblapacktest_a)\n"
	@echo "====="
	@echo "testing_src        $(testing_src)\n"
	@echo "testing_all        $(testing_all)\n"
	@echo "testing_obj        $(testing_obj)\n"
	@echo "testers            $(testers)\n"
	@echo "testers_f          $(testers_f)\n"
	@echo "====="
	@echo "sparse_testing_src $(sparse_testing_src)\n"
	@echo "sparse_testing_all $(sparse_testing_all)\n"
	@echo "sparse_testing_obj $(sparse_testing_obj)\n"
	@echo "sparse_testers     $(sparse_testers)\n"
	@echo "====="
	@echo "dep     $(dep)"
	@echo "deps    $(deps)\n"
	@echo "====="
	@echo "libs    $(libs)"
	@echo "libs_a  $(libs_a)"
	@echo "libs_so $(libs_so)"
	@echo "====="
	@echo "LIBS    $(LIBS)"


# ------------------------------------------------------------------------------
cleandep:
	-rm -f $(deps)

ifeq ($(dep),1)
    -include $(deps)
endif
