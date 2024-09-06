#!/bin/bash

source .github/workflows/setup.sh

cd make.inc-examples
cp make.inc.mkl-icc       make.inc.mkl-intel
cp make.inc.mkl-gcc-ilp64 make.inc.mkl-ilp64-gcc
cp make.inc.mkl-icc-ilp64 make.inc.mkl-ilp64-intel
cp make.inc.openblas      make.inc.openblas-gcc
cp make.inc.openblas      make.inc.openblas-intel
cd ..
ln -sf make.inc-examples/make.inc.$BLAS-$COMPILER make.inc

make BACKEND=$BACKEND generate
