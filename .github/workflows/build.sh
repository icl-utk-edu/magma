#!/bin/bash

source .github/workflows/setup.sh

make -j8 || make -j1
