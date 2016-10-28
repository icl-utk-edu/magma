# This is a simple standalone example. See README.txt
# Initially it is setup to use OpenBLAS.
# See magma/make.inc for alternate BLAS and LAPACK libraries,
# or use pkg-config as described below.

# Paths where MAGMA, CUDA, and OpenBLAS are installed
MAGMADIR     ?= ..
CUDADIR      ?= /usr/local/cuda
OPENBLASDIR  ?= /usr/local/openblas

CC            = gcc
FORT          = gfortran
LD            = gcc
CFLAGS        = -Wall
LDFLAGS       = -Wall

MAGMA_CFLAGS   := -DADD_ -I$(MAGMADIR)/include -I$(MAGMADIR)/sparse/include  \
                  -I$(CUDADIR)/include
MAGMA_F90FLAGS := -I$(MAGMADIR)/include -Dmagma_devptr_t="integer(kind=8)"

MAGMA_LIBS   := -L$(MAGMADIR)/lib -L$(CUDADIR)/lib \
                -lmagma -lcublas -lcudart
                
MAGMASPARSE_LIBS   := -L$(MAGMADIR)/lib -L$(CUDADIR)/lib64 -L$(OPENBLASDIR)/lib \
                      -lmagma_sparse -lcusparse -lcusolver

# Alternatively, using pkg-config (see README.txt):
# MAGMA_CFLAGS := $(shell pkg-config --cflags magma)
# MAGMA_LIBS   := $(shell pkg-config --libs   magma)


# ----------------------------------------
default:
	@echo "Available make targets are:"
	@echo "  make all       # compiles both example_v1, example_v2, example_sparse and example_f"
	@echo "  make c         # compiles example_v1, example_v2 and example_sparse"
	@echo "  make fortran   # compiles example_f"
	@echo "  make clean     # deletes executables and object files"

all: c fortran

c: example_v1 example_v2 example_sparse

fortran: example_f

clean:
	-rm -f example_v1 example_v2 example_sparse example_f *.o *.mod

.SUFFIXES:


# ----------------------------------------
# C example
%.o: %.c
	$(CC) $(CFLAGS) $(MAGMA_CFLAGS) -c -o $@ $<

example_v1: example_v1.o
	$(LD) $(LDFLAGS) -o $@ $^ $(MAGMA_LIBS)

example_v2: example_v2.o
	$(LD) $(LDFLAGS) -o $@ $^ $(MAGMA_LIBS)
	
example_sparse: example_sparse.o
	$(LD) $(LDFLAGS) -o $@ $^ $(MAGMA_LIBS) $(MAGMASPARSE_LIBS)


# ----------------------------------------
# Fortran example
# this uses capital .F90 to preprocess to define magma_devptr_t
%.o: %.F90
	$(FORT) $(F90FLAGS) $(MAGMA_F90FLAGS) -c -o $@ $<

fortran.o: $(CUDADIR)/src/fortran.c
	$(CC) $(CFLAGS) $(MAGMA_CFLAGS) -DCUBLAS_GFORTRAN -c -o $@ $<

example_f: example_f.o fortran.o
	$(FORT) $(LDFLAGS) -o $@ $^ $(MAGMA_LIBS)
