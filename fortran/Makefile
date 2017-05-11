# 1) setup in environment or in this Makefile
# e.g., in sh (bash):
# export MAGMADIR=/path/to/magma where MAGMA is installed
# export CUDADIR=/path/to/cuda   where CUDA  is installed
#
# 2) adjust BLAS & LAPACK libraries below as needed
#
# 3) make should compile test; run ./test to see if it works

CC         = gcc
CXX        = g++
FC         = gfortran
CFLAGS     = -Wall -I$(MAGMADIR)/include -I$(CUDADIR)/include
CXXFLAGS   = -Wall -I$(MAGMADIR)/include -I$(CUDADIR)/include
FCFLAGS    = -Wall -Wno-unused-function
LDFLAGS    =
LIBS       = -L$(MAGMADIR)/lib -lmagma -Wl,-rpath,$(MAGMADIR)/lib

# BLAS and LAPACK libraries
# framework Accelerate is for MacOS
LIBS      += -framework Accelerate

obj = \
	test.o			\
	offset.o		\
	magma2.o		\
	magma2_sfortran.o	\
	magma2_dfortran.o	\
	magma2_cfortran.o	\
	magma2_zfortran.o	\
	magma2_common.o		\

all: test

# generate precisions
magma2_sfortran.F90: magma2_zfortran.stamp
magma2_dfortran.F90: magma2_zfortran.stamp
magma2_cfortran.F90: magma2_zfortran.stamp
magma2_zfortran.stamp: magma2_zfortran.F90
	$(MAGMADIR)/tools/codegen.py magma2_zfortran.F90
	touch $@

# force modules to compile in correct order
test.o: magma2.o
magma2.o: magma2_sfortran.o magma2_dfortran.o magma2_cfortran.o magma2_zfortran.o
magma2_sfortran.o: magma2_common.o
magma2_dfortran.o: magma2_common.o
magma2_cfortran.o: magma2_common.o
magma2_zfortran.o: magma2_common.o

test: $(obj)
	$(FC) $(LDFLAGS) -o $@ $^ \
		$(LIBS)

%.o: %.F90
	$(FC) $(FCFLAGS) -c -o $@ $<

%.o: %.f90
	$(FC) $(FCFLAGS) -c -o $@ $<

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	-rm -f *.o *.mod test
