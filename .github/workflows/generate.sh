#!/bin/bash

source .github/workflows/setup.sh

# Temporarily create a make.inc for the generate step
echo -e "FORT = true\nGPU_TARGET = $GPU_TARGET\nBACKEND = $BACKEND\n" > make.inc

make generate

rm make.inc
