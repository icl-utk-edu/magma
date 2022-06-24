#!/bin/bash -xe 

maker=$1
device=$2
blas=$3
compiler=$4


source ~/spack_setup
GCC=gcc@8.5.0
sload cmake %$GCC
sload openblas %$GCC
sload cuda@11.4.3 %$GCC
sload intel-oneapi-mkl %$GCC
sload intel-oneapi-compilers %$GCC

export CUDADIR=$CUDA_HOME
export GPU_TARGET=Volta
export OPENBLASDIR=`slocation openblas %$GCC`

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

