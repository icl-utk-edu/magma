#!/bin/bash

source .github/workflows/setup.sh

if [ "$MAKER" = "cmake" ]; then
   mkdir -p build
   cd build
   cmake -DGPU_TARGET=$GPU_TARGET -DMAGMA_ENABLE_${BACKEND^^}=ON $OPTIONS ..
fi
