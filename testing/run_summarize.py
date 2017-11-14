#!/usr/bin/env python
#
# MAGMA (version 2.0) --
# Univ. of Tennessee, Knoxville
# Univ. of California, Berkeley
# Univ. of Colorado, Denver
# @date

## @file run_summarize.py
#  @author Mark Gates
#
# Usage:
# First run tests, saving output in tests.txt:
#     ./run_tests.py [options] > tests.txt
#     or
#     ./testing_xyz  [options] > tests.txt
#
# Then parse their output:
#     ./run_summarize.py test.txt
#
# Parses the output of MAGMA testers and sorts tests into categories:
#     ok:              test passed
#     suspect:         test failed, but  error < tol2*eps, so it's probably ok
#     failed:          test failed, with error > tol2*eps
#     error:           segfault, etc.
#     known failures:  commands that we already know have issues
#     ignore:          ignore issues like using --ngpu 2 with only 1 gpu.
#
# tol  is tolerance given when the tester was run, by default tol = 30.
# tol2 is specified here, using --tol2, by default tol2 = 100.
#
# For each suspect or failed command, prints the command and suspect or failed
# tests. Also adds the ratio {error/eps} in braces after each error.
# Tests that passed are not output by default.
#
# This is helpful to re-parse and summarize output from \ref run_tests.py, and
# to apply a second tolerence to separate true failures from borderline cases.
#
# The --rerun [123] option is helpful to generate a shell script to re-run
# failed cases.
#
# Example usage. First run tests, directing output to a file:
#
#     magma/testing> python ./run_tests.py --tol 30 --medium testing_sgeqr2x_gpu > sgeqr2x.txt
#     testing_sgeqr2x_gpu --version 1 -c                ok
#     testing_sgeqr2x_gpu --version 2 -c                ** 1 tests failed
#     testing_sgeqr2x_gpu --version 3 -c                ok
#     testing_sgeqr2x_gpu --version 4 -c                ** 1 tests failed
#     ----------------------------------------------------------------------------------------------------
#     summary
#     ----------------------------------------------------------------------------------------------------
#        34 tests in 4 commands passed
#         2 tests failed accuracy test
#         0 errors detected (crashes, CUDA errors, etc.)
#     routines with failures:
#         testing_sgeqr2x_gpu --version 2 -c
#         testing_sgeqr2x_gpu --version 4 -c
#
# Then post-process the output:
#
#     magma/testing> ./run_summarize.py sgeqr2x.txt
#     single epsilon 5.96e-08,  tol2 100,  tol2*eps 5.96e-06,
#     double epsilon 1.11e-16,  tol2 100,  tol2*eps 1.11e-14,
#     ------------------------------------------------------------------------------------------------------------------------
#     okay tests:                                          2 commands,     68 tests
#
#
#     ------------------------------------------------------------------------------------------------------------------------
#     errors (segfault, etc.):                             0 commands,      0 tests
#
#
#     ------------------------------------------------------------------------------------------------------------------------
#     failed tests (error > tol2*eps):                     0 commands,      0 tests
#
#
#     ------------------------------------------------------------------------------------------------------------------------
#     suspicious tests (tol2*eps > error > tol*eps):       0 commands,      0 tests
#
#
#     ------------------------------------------------------------------------------------------------------------------------
#     known failures:                                      2 commands,      2 tests
#     ./testing_sgeqr2x_gpu --version 2 --tol 30
#     failed tests (error > tol2*eps):
#       100   100      4.01 (   0.34)      0.07 (  20.73)  5.23e-06 {   87.8}     2.08e-03 {34899.3}     2.29e-07 {    3.8}       9.05e-05 { 1518.5}   failed
#
#     ./testing_sgeqr2x_gpu --version 4 --tol 30
#     failed tests (error > tol2*eps):
#       100   100      3.99 (   0.34)      0.09 (  15.68)  5.58e-06 {   93.6}     2.21e-03 {37080.5}     2.19e-07 {    3.7}       9.65e-05 { 1619.1}   failed
#
#     ------------------------------------------------------------------------------------------------------------------------
#     ignored errors (e.g., malloc failed):                0 commands,      0 tests
#
#
#     ------------------------------------------------------------------------------------------------------------------------
#     other (lines that did not get matched):              0 commands,      0 tests
#

import re
import sys
import os
import math
from math import isnan, isinf

from optparse import OptionParser

parser = OptionParser()
parser.add_option( '--tol2',  action='store',      help='set tolerance (tol2)', default='100' )
parser.add_option( '--okay',  action='store_true', help='print okay tests',     default=False )
parser.add_option( '--rerun', action='store',      type=int, default=0,
	help='generate script to re-run failed tests. Values:\n'
		+'    1 - re-run exact command;\n'
		+'    2 - re-run using run_tests.py;\n'
		+'    3 - re-run testers with known bugs.' )

(opts, args) = parser.parse_args()


# --------------------
tol2     = int(opts.tol2)

seps     = 5.96e-08
deps     = 1.11e-16

stol_30  = 30.0   * seps
dtol_30  = 30.0   * deps

stol_100 = 100.0  * seps
dtol_100 = 100.0  * deps

stol_1k  = 1000.0 * seps
dtol_1k  = 1000.0 * deps

epsilons = {
	's': seps,
	'c': seps,
	'd': deps,
	'z': deps,
}


# --------------------
# hash of cmd: row, where each row is an array of fields.
# Most fields are an array of text lines.
# The known field is a boolean flag.
data = {}


# --------------------
# fields in each row
Field_CmdLine = 0
Field_Okay    = 1
Field_Suspect = 2
Field_Failed  = 3
Field_Error   = 4
Field_Ignore  = 5
Field_Other   = 6
Field_Known   = 7

# labels for each field
labels = [
	'',
	'okay tests',
	'suspicious tests (tol2*eps > error > tol*eps)',
	'failed tests (error > tol2*eps)',
	'errors (segfault, etc.)',
	'ignored errors (e.g., malloc failed)',
	'other (lines that did not get matched)',
	'known failures',
]


# --------------------
# errors to ignore
ignore_regexp = r'malloc failed|returned error -11[23]'

# errors (segfaults, etc.)
error_regexp = r'exit|memory leak|memory mapping error|CUDA runtime error|illegal value|returned error|ERROR SUMMARY: [1-9]'

# testers with known bugs or issues
known_regexp = '|'.join((
	r'geqr2x_gpu.*--version +[24]',
	r'gegqr_gpu.*--version +[34]',   # N=95, specifically
	r'gesv_rbt',                     # RBT known to fail on some matrices
	r'trsm',                         # needs more rigorous error bound or better conditioned matrices
))

# problem size, possibly with couple words before it, e.g.:
# "1234 ..."
# "upper  1234 ..."
# "vector  upper  1234 ..."
size_regexp  = r'^ *([a-zA-Z]\w* +){0,2}\d+ '


# --------------------
# input:  re match object of floating point error
# output: string "error {error/eps}"
# effects: sets g_failed
def add_ratio( match ):
	global g_failed, eps
	s     = match.group(1)
	error = float( s )
	ratio = error / eps
	g_failed |= (isnan(error) or isinf(error) or ratio >= tol2)
	return s + ' {%7.1f}' % (ratio)
# end


# --------------------
# input:  line
# output: line with errors replaced by "error {error/eps}"
# effects: sets g_failed, g_suspect, g_okay
def find_ok_failed( line ):
	global g_failed, g_suspect, g_okay
	if ( re.search( r'failed', line )):
		line = re.sub( r' (\d\.\d+e[+-]\d+|-?nan|-?inf)', add_ratio, line )
		if ( not g_failed ):
			line = re.sub( r'failed', 'suspect', line )
			g_suspect = True
		# end
	elif ( re.search( r'\b(ok|skipping|error check only for)\b', line )):
		g_okay = True
	return line
# end


# --------------------
# saves g_context into data[ g_cmd ], depending on flags g_ignore, g_error, g_failed, g_suspect, g_okay.
# resets g_context and above flags.
def save():
	global g_cmd, g_context, g_ignore, g_error, g_failed, g_suspect, g_okay
	if ( g_cmd ):
		if ( not data.has_key( g_cmd )):
			known = (re.search( known_regexp, g_cmd ) != None)
			# fields:       [ cmdline,   okay, susp, fail, err,  ignr, othr, known ]
			data[ g_cmd ] = [ g_cmdline, [],   [],   [],   [],   [],   [],   known ]
		# end
		if ( g_context ):
			if   ( g_ignore  ):  data[ g_cmd ][ Field_Ignore  ].append( g_context )
			elif ( g_error   ):  data[ g_cmd ][ Field_Error   ].append( g_context )
			elif ( g_failed  ):  data[ g_cmd ][ Field_Failed  ].append( g_context )
			elif ( g_suspect ):  data[ g_cmd ][ Field_Suspect ].append( g_context )
			elif ( g_okay    ):  data[ g_cmd ][ Field_Okay    ].append( g_context )
			else:                data[ g_cmd ][ Field_Other   ].append( g_context )
		# end
	# end
	
	# reset globals to accumulate results for next problem size
	g_context = ''
	g_ignore  = False
	g_error   = False
	g_failed  = False
	g_suspect = False
	g_okay    = False
# end


# --------------------
# set globals
g_cmd     = None
g_cmdline = None
g_size    = ''
save()


# --------------------
# This uses a finite state machine to track what the current line is.
# Diagrams are provided in comments below. Some transitions save and reset the
# current problem size, as noted by # save(). States are:
# Start: before seeing command
# Cmd:   testing_xyz command line:       ./testing_zgetrf
# Pre:   errors before a problem size
# Size:  problem size
# Post:  checks after a problem size
State_Start = 0
State_Cmd   = 1
State_Pre   = 2
State_Size  = 3
State_Post  = 4
State_End   = 5

for filename in args:
	(d, f) = os.path.split( filename )
	match = re.search( '^(xs|s|m|l|xl)-', f )
	if (match):
		g_size = match.group(1)
	
	fopen = open( filename )
	state = State_Start
	for line in fopen:
		# ignore anything starting with % comment, ***** separators, and blank lines
		if ( re.search( r'^%|^\*{5,}$|^\s*$', line )):
			continue
		
		# end:
		# start     cmd     pre     size     post
		#  /\        |       |        |        |
		#   '----<---'---<---'----<---'----<---'    # save()
		if ( re.search( '^(summary|  \*\* \d+ tests failed)$', line )):
			save()
			state = State_Start
			g_cmd     = None
			g_cmdline = None
			continue
		
		# command:
		# start --> cmd     pre     size     post
		#            /\       |       |        |
		#             '---<---'---<---'----<---'    # save()
		# cmd is everything except -c, -n, --range, -N problem sizes,
		# to show options like -L, -U.
		m = re.search( r'^(?:cuda-memcheck +)?(./testing_(\w).*)', line )
		if ( m ):
			save()
			state = State_Cmd
			g_cmd = re.sub( ' -c\d*|(-n|-N|--range) +\S+', '', m.group(1) )
			g_cmdline = line
			# select appropriate epsilon for precision
			p = m.group(2)
			if ( p in 'sdcz' ):
				eps = epsilons[ p ]
			continue
		
		# pre-errors (segfaults, illegal arguments, etc.) that appear before size:
		#                   ,-.
		#                   | v
		# start     cmd --> pre     size     post
		#                    /\       |        |
		#                     '---<---'----<---'    # save()
		m1 = re.search( error_regexp,  line )
		m2 = re.search( ignore_regexp, line )
		if ( m1 or m2 ):
			if ( state != State_Pre ):
				save()
				state = State_Pre
			g_context += line
			g_error  = True
			g_ignore = (m2 != None)
			continue
		
		# problem size
		#            ,--->---.--->---.
		#            |       |       v
		# start     cmd     pre     size     post
		#                           /\ |       |
		#                            '-'---<---'    # save()
		if ( re.search( size_regexp, line )):
			if ( state == State_Size or state == State_Post ):
				save()
			state = State_Size
			line = find_ok_failed( line )
			g_context += line
			continue
		
		# otherwise, default action depends on state:
		#  ,-.              ,-.              ,-.
		#  | v              | v              | v
		# start     cmd --> pre     size --> post
		if ( state == State_Start ):
			pass
			
		elif ( state == State_Cmd or state == State_Pre ):
			state = State_Pre
			line = find_ok_failed( line )  # shouldn't have ok or failed
			g_context += line
			
		elif ( state == State_Size or state == State_Post ):
			state = State_Post
			line = find_ok_failed( line )
			g_context += line
			
		else:
			print 'unhandled', state, line
	# end
	save()
# end


# ------------------------------------------------------------------------------
# Processes commands that have tests in the given field.
# If output is true, prints commands and tests in given field,
# otherwise just prints count of the commands and tests.
# If field is KNOWN, prints tests that are suspect, failed, or error.
def output( field, output ):
	cmds  = 0
	tests = 0
	result = ''
	for cmd in sorted( data.keys() ):
		row = data[cmd]
		if ( field == Field_Known ):
			if ( row[Field_Known] ):
				#result += row[Field_CmdLine]
				#result += cmd + '\n'
				num = len(row[Field_Suspect]) + len(row[Field_Failed]) + len(row[Field_Error])
				#if ( num == 0 ):
				#	result += 'no failures (has ' + cmd + ' been fixed?)\n\n'
				if (num > 0 and output):
					result += cmd + '\n'
					if ( len(row[Field_Okay]) > 0 and opts.okay ):
						result += labels[Field_Okay]    + ':\n' + ''.join( row[Field_Okay]    )
					if ( len(row[Field_Error]) > 0 ):
						result += labels[Field_Error]   + ':\n' + ''.join( row[Field_Error]   )
					if ( len(row[Field_Failed]) > 0 ):
						result += labels[Field_Failed]  + ':\n' + ''.join( row[Field_Failed]  )
					if ( len(row[Field_Suspect]) > 0 ):
						result += labels[Field_Suspect] + ':\n' + ''.join( row[Field_Suspect] )
					if ( len(row[Field_Ignore]) > 0 and opts.okay ):
						result += labels[Field_Ignore]  + ':\n' + ''.join( row[Field_Ignore]  )
					result += '\n'
				# end
				cmds  += 1
				tests += num
		else:
			num = len( row[field] )
			if ( num > 0 and not row[Field_Known] ):
				if ( output ):
					#result += row[Field_CmdLine]
					result += cmd + '\n'
					result += ''.join( row[field] ) + '\n'
				# end
				cmds  += 1
				tests += num
			# end
		# end
	# end
	print '#' * 120
	print '%-50s %3d commands, %6d tests' % (labels[field]+':', cmds, tests )
	print result
	print
# end


# ------------------------------------------------------------------------------
# Prints shell script to re-run failed commands exactly as in the output.
# For example, if   ./testing_dpotrf -L -c -n 1000   failed,
# it would re-run   ./testing_dpotrf -L -c -n 1000
def rerun1():
	output = 'errors.txt'
	if (g_size):
		output = g_size + '-' + output
	
	print '#!/bin/sh'
	print
	print 'touch', output
	print
	for cmd in sorted( data.keys() ):
		row = data[cmd]
		if ((row[Field_Error] or row[Field_Failed] or row[Field_Suspect]) and not row[Field_Known]):
			test = row[Field_CmdLine].rstrip()
			print 'echo    ', test, '>>', output
			print '(set -x;', test, '>>', output, ')'
			print
	# end
# end


# ------------------------------------------------------------------------------
# Prints shell script to re-run failed commands, using run_tests.py.
# For example, if   ./testing_dpotrf -L -c -n 1000   failed,
# it would re-run   ./run_tests.py testing_dpotrf
# which would re-run both -L and -U options (unlike rerun1).
def rerun2():
	output = 'errors.txt'
	size = ''
	if (g_size):
		output = g_size + '-' + output
		size = '--' + g_size
	seen = {}
	
	print '#!/bin/sh'
	print
	print 'FUNCS="',
	for cmd in sorted( data.keys() ):
		row = data[cmd]
		if ((row[Field_Error] or row[Field_Failed] or row[Field_Suspect]) and not row[Field_Known]):
			match = re.search( '^\./(testing\w+)', cmd )
			test = match.group(1)
			if (not seen.has_key( test )):
				seen[test] = True
				print test,
			# end
		# end
	# end
	print '"'
	print
	print './run_tests.py', size, '$FUNCS >', output
# end


# ------------------------------------------------------------------------------
# Same as rerun2 but with known failures.
def rerun_known():
	output = 'errors.txt'
	size = ''
	if (g_size):
		output = g_size + '-' + output
		size = '--' + g_size
	seen = {}
	
	print '#!/bin/sh'
	print
	print 'FUNCS="',
	for cmd in sorted( data.keys() ):
		row = data[cmd]
		if (row[Field_Known]):
			match = re.search( '^\./(testing\w+)', cmd )
			test = match.group(1)
			if (not seen.has_key( test )):
				seen[test] = True
				print test,
			# end
		# end
	# end
	print '"'
	print
	print './run_tests.py', size, '$FUNCS >', output
# end


# ------------------------------------------------------------------------------
if   (opts.rerun == 1):
	rerun1()
elif (opts.rerun == 2):
	rerun2()
elif (opts.rerun == 3):
	rerun_known()
else:
	print 'single epsilon %.2e,  tol2 %.0f,  tol2*eps %.2e,' % (seps, tol2, tol2*seps)
	print 'double epsilon %.2e,  tol2 %.0f,  tol2*eps %.2e,' % (deps, tol2, tol2*deps)
	output( Field_Okay,    opts.okay )
	output( Field_Error,   True   )
	output( Field_Failed,  True   )
	output( Field_Suspect, True   )
	output( Field_Known,   True   )
	output( Field_Ignore,  True   )
	output( Field_Other,   True   )  # tests that didn't have "ok" or "failed"
# end
