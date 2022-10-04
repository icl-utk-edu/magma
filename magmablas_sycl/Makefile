# See Makefile.src for list of files in this directory.
# This makefile simply forwards commands to the top-level makefile.

top  := ..
pwd  := $(shell pwd)
cdir := $(notdir $(pwd))

default: $(cdir)

include $(top)/Makefile.subdir
