#!/bin/bash

source .github/workflows/setup.sh

# Get the test harness from the source dir if it wasn't copied
[ -e testing/run_tests.py ] || cp ../testing/run_tests.py testing/
cd testing
./run_tests.py -s testing_sgemm
