#!/usr/bin/env python
#
# Compares testing/Makefile.src and testing/run_tests.py
# to ensure that all routines are being tested.
# Assumes the order is the same in both files.
#
# Usage: ./tools/checklist_run_tests.py
# works from MAGMA root or testing directory.

import re
import os

if (os.path.exists('testing')):
	os.chdir( "testing" )


# ----------------------------------------
# parse Makefile.src > tmp-src.txt

# some testers in Makefile that are excluded from run_tests.py
exclude = (
	'testing_auxiliary',
	'testing_blas_z',
	'testing_dgeev',  # handled by zgeev
	'testing_z',
)

infile  = open( "Makefile.src" )
outfile = open( "tmp-src.txt", "w" )

for line in infile:
	m = re.search( '^\t\$\(cdir\)/(testing_\w+)\.cpp', line )
	if m and m.group(1) not in exclude:
		print >>outfile, m.group(1)

outfile.close()


# ----------------------------------------
# parse run_tests.py > tmp-tests.txt
infile  = open( "run_tests.py" )
outfile = open( "tmp-tests.txt", "w" )

last = ''
for line in infile:
	m = re.search( "^\t\('(testing_\w\w+)',", line )
	if m and m.group(1) != last:
		print >>outfile, m.group(1)
		last = m.group(1)

outfile.close()


# ----------------------------------------
cmd = "diff tmp-src.txt tmp-tests.txt"
print cmd
os.system( cmd )
