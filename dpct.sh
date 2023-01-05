#!/bin/bash

MAGMA_HOME=$HOME/magma

dpct --cuda-include-path="$CUDA_HOME/include"\
    --extra-arg="-I $MAGMA_HOME/include"\
    --extra-arg="-I  $MAGMA_HOME/sparse/include"\
    --extra-arg="-I $MAGMA_HOME/control"\
    --extra-arg="-I $MAGMA_HOME/sparse/control"\
    --extra-arg="-I ${CMPLR_ROOT}/${INTEL_TARGET_PLATFORM}/include/sycl"\
    --extra-arg="-I ${CMPLR_ROOT}/${INTEL_TARGET_PLATFORM}/include"\
    --extra-arg="-DADD_"\
    --in-root="$MAGMA_HOME/sparse"\
    --out-root="$MAGMA_HOME/sparse_sycl"\
    --report-file-prefix="sparse_dpct_out"\
    --extra-arg="-std=c++17"\
    --extra-arg="-ferror-limit=3"\
    --format-style=llvm\
    --process-all\
    -p compile_commands.json
#    --gen-build-script\

