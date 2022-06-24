#!/bin/bash -xe 

maker=$1
device=$2
blas=$3
compiler=$4


source spack_setup
GCC=gcc@8.5.0
sload cmake %$GCC
sload openblas@0.3.10 %$GCC
sload $cuda %$GCC
sload intel-oneapi-mkl %$GCC
sload intel-oneapi-compilers %$GCC

export CUDADIR=$CUDA_HOME
export GPU_TARGET=Volta
export OPENBLASDIR=`slocation openblas@0.3.10 %$GCC`

(
cd make.inc-examples
cp make.inc.mkl-gcc-ilp64 make.inc.mkl-ilp64-gcc
cp make.inc.mkl-icc-ilp64 make.inc.mkl-ilp64-icc
cp make.inc.openblas make.inc.openblas-gcc
cp make.inc.openblas make.inc.openblas-icc
)

ln -s make.inc-examples/make.inc.$blas-$compiler make.inc

LIBDIR=-L$CUDADIR/lib64 make -j8

