#!/bin/bash -xe

source /etc/profile

maker=$1
device=$2
blas=$3
compiler=$4

module load cmake
module load openblas
module load intel-oneapi-mkl
module load intel-oneapi-compilers

export CUDADIR=/usr/local/cuda
export GPU_TARGET=Volta
export OPENBLASDIR=$ICL_OPENBLAS_DIR

if [ "$maker" = "make" ]; then
   cd make.inc-examples
   cp make.inc.mkl-gcc-ilp64 make.inc.mkl-ilp64-gcc
   cp make.inc.mkl-icc-ilp64 make.inc.mkl-ilp64-icc
   cp make.inc.openblas make.inc.openblas-gcc
   cp make.inc.openblas make.inc.openblas-icc
   cd ..
   ln -s make.inc-examples/make.inc.$blas-$compiler make.inc
else # maker="cmake"
   echo "FORT = true" > make.inc
   make generate
   mkdir build
   cd build
   cmake -DGPU_TARGET=$GPU_TARGET ..
fi

make -j8

