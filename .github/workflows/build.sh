#!/bin/bash -xe

source /etc/profile

maker=$1
device=$2
blas=$3
compiler=$4

module load cmake
if [ "$blas" = "openblas" ]; then
   module load openblas
   export OPENBLASDIR=$ICL_OPENBLAS_ROOT
else
   module load intel-oneapi-mkl
fi
[ "$compiler" = "intel" ] && module load intel-parallel-studio

export CUDADIR=/usr/local/cuda
export PATH=$PATH:$CUDADIR/bin
export LIBRARY_PATH=$LIBRARY_PATH:$CUDADIR/lib64
export GPU_TARGET=Volta

if [ "$maker" = "make" ]; then
   cd make.inc-examples
   perl -pi -e 's/GPU_TARGET/#/' *
   cp make.inc.mkl-icc       make.inc.mkl-intel
   cp make.inc.mkl-gcc-ilp64 make.inc.mkl-ilp64-gcc
   cp make.inc.mkl-icc-ilp64 make.inc.mkl-ilp64-intel
   cp make.inc.openblas      make.inc.openblas-gcc
   cp make.inc.openblas      make.inc.openblas-intel
   cd ..
   ln -s make.inc-examples/make.inc.$blas-$compiler make.inc
else # maker="cmake"
   if [ "$blas" = "mkl-ilp64" ]; then
      echo CMake needs to be configured to use ilp64
      exit 1 
   fi
   echo "FORT = true" > make.inc
   make generate
   mkdir build
   cd build
   cmake -DGPU_TARGET=$GPU_TARGET ..
fi

make -j8 || make -j1

