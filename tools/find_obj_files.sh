#!/bin/sh
#
# Finds all object files (.o, .mod)
# Used in build.sh
#
# @author Mark Gates

find . -name build   -prune -o \
       -name builds  -prune -o \
       \(    -name '*.o'   \
          -o -name '*.mod' \
          -not -type d \)  \
       -print
