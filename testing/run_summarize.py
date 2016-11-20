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
# This is helpful to re-parse and summarize output from run-tests.py, and
# to apply a second tolerence to separate true failures from borderline cases.

import re
import sys
import math
from math import isnan, isinf

from optparse import OptionParser

parser = OptionParser()
parser.add_option( '--tol2', action='store',      dest='tol2', help='set tolerance (tol2)', default='100' )
parser.add_option( '--okay', action='store_true', dest='okay', help='print okay tests',     default=False )

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

print 'single epsilon %.2e,  tol2 %.0f,  tol2*eps %.2e,  30*eps %.2e,  100*eps %.2e,  1000*eps %.2e' % (seps, tol2, tol2*seps, 30*seps, 100*seps, 1000*seps)
print 'double epsilon %.2e,  tol2 %.0f,  tol2*eps %.2e,  30*eps %.2e,  100*eps %.2e,  1000*eps %.2e' % (deps, tol2, tol2*deps, 30*deps, 100*deps, 1000*deps)

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
CmdLine = 0
Okay    = 1
Suspect = 2
Failed  = 3
Error   = 4
Ignore  = 5
Other   = 6
Known   = 7

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
	r'hemm_mgpu',                    # tester is broken
	r'symm_mgpu',                    # tester is broken
	r'gesv_rbt',                     # RBT known to fail on some matrices
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
			if   ( g_ignore  ):  data[ g_cmd ][ Ignore  ].append( g_context )
			elif ( g_error   ):  data[ g_cmd ][ Error   ].append( g_context )
			elif ( g_failed  ):  data[ g_cmd ][ Failed  ].append( g_context )
			elif ( g_suspect ):  data[ g_cmd ][ Suspect ].append( g_context )
			elif ( g_okay    ):  data[ g_cmd ][ Okay    ].append( g_context )
			else:                data[ g_cmd ][ Other   ].append( g_context )
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
save()


# --------------------
# This uses a finite state machine to track what the current line is.
# Diagrams are provided in comments below. Some transitions save and reset the
# current problem size, as noted by # save(). States are:
# Start: before seeing command
# Cmd:   testing_xyz command line:       ./testing_zgetrf
# Pre:   errors before a problem size:   
# Size:  problem size
# Post:  checks after a problem size
Start = 0
Cmd   = 1
Pre   = 2
Size  = 3
Post  = 4
End   = 5


for filename in args:
	fopen = open( filename )
	state = Start
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
			state = Start
			g_cmd     = None
			g_cmdline = None
			continue
		
		# command:
		# start --> cmd     pre     size     post
		#            /\       |       |        |
		#             '---<---'---<---'----<---'    # save()
		m = re.search( r'^(?:cuda-memcheck +)?(./testing_(\w).*?) (-c|--range|-N)', line )
		if ( m ):
			save()
			state = Cmd
			g_cmd     = m.group(1)
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
			if ( state != Pre ):
				save()
				state = Pre
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
			if ( state == Size or state == Post ):
				save()
			state = Size
			line = find_ok_failed( line )
			g_context += line
			continue
		
		# otherwise, default action depends on state:
		#  ,-.              ,-.              ,-.
		#  | v              | v              | v
		# start     cmd --> pre     size --> post
		if ( state == Start ):
			pass
			
		elif ( state == Cmd or state == Pre ):
			state = Pre
			line = find_ok_failed( line )  # shouldn't have ok or failed
			g_context += line
			
		elif ( state == Size or state == Post ):
			state = Post
			line = find_ok_failed( line )
			g_context += line
			
		else:
			print 'unhandled', state, line
	# end
	save()
# end


# ------------------------------------------------------------
# Processes commands that have tests in the given field.
# If output is true, prints commands and tests in given field,
# otherwise just prints count of the commands and tests.
# If field is KNOWN, prints tests that are suspect, failed, or error.
def output( field, output ):
	cmds  = 0
	tests = 0
	result = ''
	for cmd in sorted( data.keys()):
		row = data[cmd]
		if ( field == Known ):
			if ( row[Known] ):
				#result += row[CmdLine]
				result += cmd + '\n'
				num = len(row[Suspect]) + len(row[Failed]) + len(row[Error])
				if ( num == 0 ):
					result += 'no failures (has ' + cmd + ' been fixed?)\n\n'
				elif ( output ):
					#result += cmd + '\n'
					if ( len(row[Okay]) > 0 and opts.okay ):
						result += labels[Okay]    + ':\n' + ''.join( row[Okay]    )
					if ( len(row[Error]) > 0 ):
						result += labels[Error]   + ':\n' + ''.join( row[Error]   )
					if ( len(row[Failed]) > 0 ):
						result += labels[Failed]  + ':\n' + ''.join( row[Failed]  )
					if ( len(row[Suspect]) > 0 ):
						result += labels[Suspect] + ':\n' + ''.join( row[Suspect] )
					if ( len(row[Ignore]) > 0 and opts.okay ):
						result += labels[Ignore]  + ':\n' + ''.join( row[Ignore]  )
					result += '\n'
				# nd
				cmds  += 1
				tests += num
		else:
			num = len( row[field] )
			if ( num > 0 and not row[Known] ):
				if ( output ):
					#result += row[CmdLine]
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


output( Okay,    opts.okay )
output( Error,   True   )
output( Failed,  True   )
output( Suspect, True   )
output( Known,   True   )
output( Ignore,  True   )
output( Other,   True   )  # tests that didn't have "ok" or "failed"
