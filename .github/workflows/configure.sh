#!/bin/bash

source .github/workflows/setup.sh

if [ "$MAKER" = "make" ]; then
   cd make.inc-examples
   cp make.inc.mkl-icx       make.inc.mkl-intel
   cp make.inc.mkl-gcc-ilp64 make.inc.mkl-ilp64-gcc
   cp make.inc.mkl-icx-ilp64 make.inc.mkl-ilp64-intel
   cp make.inc.openblas      make.inc.openblas-gcc
   cp make.inc.openblas      make.inc.openblas-intel
   cd ..
   ln -sf make.inc-examples/make.inc.$BLAS-$COMPILER make.inc
else # "$MAKER" = "cmake"
   mkdir -p build
   cd build
   cmake -DGPU_TARGET=$GPU_TARGET -DMAGMA_ENABLE_${BACKEND^^}=ON $OPTIONS ..
fi
