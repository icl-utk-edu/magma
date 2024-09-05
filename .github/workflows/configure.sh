#!/bin/bash

source .github/workflows/setup.sh

if [ "$MAKER" = "make" ]; then
   cd make.inc-examples
   perl -pi -e 's/GPU_TARGET/#/' *
   cp make.inc.mkl-icc       make.inc.mkl-intel
   cp make.inc.mkl-gcc-ilp64 make.inc.mkl-ilp64-gcc
   cp make.inc.mkl-icc-ilp64 make.inc.mkl-ilp64-intel
   cp make.inc.openblas      make.inc.openblas-gcc
   cp make.inc.openblas      make.inc.openblas-intel
   cd ..
   ln -sf make.inc-examples/make.inc.$BLAS-$COMPILER make.inc
else # MAKER="cmake"
   if [ "$BLAS" = "mkl-ilp64" ]; then
      echo CMake needs to be configured to use ilp64
      exit 1 
   fi
   echo -e "FORT = true\nGPU_TARGET = $GPU_TARGET\nBACKEND = $BACKEND\n" > make.inc
   make generate
   mkdir -p build
   cd build
   cmake -DGPU_TARGET=$GPU_TARGET -DMAGMA_ENABLE_${BACKEND^^}=ON $OPTIONS ..
fi
