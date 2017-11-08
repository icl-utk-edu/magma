#!/usr/bin/env python
#
# Rejects commits that don't follow certain stylistic rules.
# see https://bitbucket.org/icl/style/wiki/ICL_C_CPP_Coding_Style_Guide
# Only some rules are enforced here (those being easy to check with regexp).
# See check() calls below.
#
# Enable by adding to .hg/hgrc:
# [hooks]
# pretxncommit.whitespace = hg export tip | tools/check-style.py
#
# Run standalone:
# tools/check-style.py files
# Files must be under Mercurial control so they show up in 'hg diff'.
#
# Adapted from:
# http://hgbook.red-bean.com/read/handling-repository-events-with-hooks.html

from __future__ import print_function

import re
import os
import sys

# ------------------------------------------------------------------------------
# ANSI codes
esc     = chr(0x1B) + '['

red     = esc + '31m'
green   = esc + '32m'
yellow  = esc + '33m'
blue    = esc + '34m'
magenta = esc + '35m'
cyan    = esc + '36m'
white   = esc + '37m'

bg_red     = esc + '41m'
bg_green   = esc + '42m'
bg_yellow  = esc + '43m'
bg_blue    = esc + '44m'
bg_magenta = esc + '45m'
bg_cyan    = esc + '46m'
bg_white   = esc + '47m'

font_bold    = esc + '1m'
font_normal  = esc + '0m'

# ------------------------------------------------------------------------------
# .sh -- issues with ${foo}
src_ext = (
    '.c', '.h', '.hh', '.cc', '.hpp', '.cpp', '.cu', '.cuh',
    '.py', '.pl',
    '.f', '.f90', '.F', '.F90',
)

# ------------------------------------------------------------------------------
class Error( Exception ):
    def __init__( self, msg, line ):
        Exception.__init__( self, msg )
        self.line = line
# end

# ------------------------------------------------------------------------------
# prints msg and returns 1 if line matches regexp, and doesn't match exclude.
def check( regexp, line, filename, linenum, msg, exclude=None ):
    regexp = '(' + regexp + ')'
    s = re.search( regexp, line )
    if (s and (exclude is None or not re.search( exclude, line ))):
        line2 = re.sub( regexp, bg_yellow + r'\1' + font_normal, line )
        #line2 = line.replace( s.group(0), bg_red + s.group(0) + font_normal )
        #raise Error( msg, line2 )
        print( '%s%s:%d:%s %s\n%s' % (
            font_bold, filename, linenum, font_normal,
            msg, line2), end='', file=sys.stderr )
        return 1
    # end
    return 0
# end

# ------------------------------------------------------------------------------
# check diff output for style
# lines is an iterator of 'hg diff' or 'hg export' output.
# returns number of errors found.
def check_style( lines ):
    errors = 0
    linenum = 0
    header = False
    filename = None
    is_src = False
    for line in lines:
        # header - get filename
        if (header):
            m = re.search( r'^(?:---|\+\+\+) ([^\t\n]+)', line )
            if (m and m.group(1) != '/dev/null'):
                filename = m.group(1)

                (base, ext) = os.path.splitext( filename )
                is_src = (ext in src_ext)
            if (line.startswith('+++ ')):
                header = False
            continue
        # end

        # new diff starts new header
        if (line.startswith( 'diff ' )):
            header = True
            continue

        # hunk header - save line number
        m = re.search( r'^@@ -\d+,\d+ \+(\d+),', line )
        if (m):
            linenum = int( m.group(1) )
            continue

        # hunk body - check added lines for style errors
        if (line.startswith( '+' )):
            line = line[1:]  # chop off '+'
            errors += check( r'[ \t]+$', line, filename, linenum, 'remove trailing whitespace' )
            errors += check( r'\r',      line, filename, linenum, 'use Unix newlines only; no Windows returns!' )
            if (is_src):
                errors += check( r'\t', line, filename, linenum, 'remove tab' )
                errors += check( r'^ +(?:if|else +if|for|while|switch|catch)(?:|  +)\(', line, filename, linenum, 'one space after if, for, while, switch, ...' )
                errors += check( r'^ +\} *else', line, filename, linenum, "add newline between '} else' (don't cuddle)" )
            # end
        # end

        ##print( 'line[%4d]' % (linenum), line, end='' )

        # increment line number on unchanged (' ') or added ('+') lines
        if (line and line[0] in ' +'):
            linenum += 1
    # end

    return errors
# end

# ------------------------------------------------------------------------------
# called with files, runs 'hg diff' on them
# otherwise, reads from stdin
if (__name__ == '__main__'):
    if (len(sys.argv) > 1):
        errors = check_style( os.popen( 'hg diff ' + ' '.join( sys.argv[1:] )))
    else:
        # assume stdin is a diff, e.g., from hg export tip
        errors = check_style( sys.stdin )

    if (errors):
        os.system( 'hg tip --template "{desc}" > .hg/commit.save' )
        print( 'message saved; use `hg commit -l .hg/commit.save`' )
        sys.exit(1)
    # end
# end
