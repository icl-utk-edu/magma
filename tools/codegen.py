#!/usr/bin/env python3
#
# ===========================================================================
# PLASMA's codegen.py is considered the definitive source.
# Please make changes in PLASMA, then propogate to other projects.
# ===========================================================================
#
# Tested with python 2.7.9 and 3.4.3.
#
# @author Mark Gates

from __future__ import print_function

description = '''\
Generates different precisions of files based on substitutions in subs.py
Also generates the Makefile rules used to invoke it as needed.'''

help = '''\
----------------------------------------------------------------------
Example uses:

  codegen.py zgemm.c zherk.c
      generates sgemm.c, dgemm.c, cgemm.c, ssyrk.c, dsyrk.c, cherk.c

  codegen.py -p s zgemm.c zherk.c
      generates sgemm.c, ssyrk.c

  codegen.py --output zgemm.c zherk.c
      prints "sgemm.c dgemm.c cgemm.c ssyrk.c dsyrk.c cherk.c"

  codegen.py --make --prefix blas zgemm.c
      prints Makefile.gen, using "blas" as prefix on Makefile variables

----------------------------------------------------------------------
Example use in Makefile:

default: all

codegen = tools/codegen.py

-include Makefile.blas.gen  # defines $(blas_all)

blas_obj = $(addsuffix .o, $(basename $(blas_all)))

all: $(blas_obj)

clean:
	-rm -f $(blas_obj)

distclean: clean cleangen
	-rm -f Makefile.blas.gen

.DELETE_ON_ERROR:

# ----------
# Create rules to do precision generation.
blas_src = src/zgemm.c

Makefile.blas.gen:
	$(codegen) --make --prefix blas $(blas_src) > $@

# force re-generating Makefile.blas.gen if $(blas_src) changes
ifneq ($(blas_src),$(blas_old))
ifneq ($(filter-out $(blas_generated),$(blas_src)),$(blas_templates))
Makefile.blas.gen: force_gen
endif
endif

force_gen: ;

----------------------------------------------------------------------
There can be multiple Makefile.xyz.gen files using different prefixes.

Makefile.gen defines several variables; the default prefix is "src":
  src_old        # Files that "codegen --make" was called with, e.g., $(blas_src) above.
                 # This can include generated files.
  src_templates  # Files that are not generated (whether or not it has @precision).
  src_generated  # Files that are generated.
  src_all        # $(src_templates) $(src_generated)

Makefile.gen defines two Makefile rules:
  make generate  # Generates all precisions (sgemm.c, dgemm.c, cgemm.c)
  make cleangen  # Removes generated files  (sgemm.c, dgemm.c, cgemm.c)
'''

import os
import re
import sys
import traceback
import argparse
from datetime import datetime


# ------------------------------------------------------------
# command line options
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawDescriptionHelpFormatter,
    description=description,
    epilog=help )
parser.add_argument( '-v', '--verbose',   action='store_true', help='Print verbose output to stderr' )
parser.add_argument( '-o', '--output',    action='store_true', help='Generate list of output files' )
parser.add_argument( '-m', '--make',      action='store_true', help='Generate Makefile rules' )
parser.add_argument(       '--prefix',    action='store',      help='Prefix for variables in Makefile', default='src')
parser.add_argument( '-p', '--precision', action='append',     help='Generate only given precision (s d c z ds zc ...). Repeatable.' )
parser.add_argument( 'args', nargs='*',   action='store',      help='Files to process' )
opts = parser.parse_args()

if opts.verbose:
    print( "opts", opts, file=sys.stderr )


# ------------------------------------------------------------
# load substitution tables
from magmasubs import subs

# Fill in subs_search with same structure as subs, but containing None values.
# Later in substitute(), we'll cache compiled regexps in subs_search.
# We could pre-compile them here, but that would compile many unneeded ones.
#
# Fill in subs_replace with pre-processed version of subs, removing regexp escapes.
try:
    subs_search  = {}
    subs_replace = {}
    for key in subs.keys():
        nrow = len( subs[key]    )
        ncol = len( subs[key][0] )
        subs_search [key] = [ [ None for j in range(ncol) ] for i in range(nrow) ]
        subs_replace[key] = [ [ None for j in range(ncol) ] for i in range(nrow) ]
        for (i, row) in enumerate( subs[key] ):
            for (j, sub) in enumerate( row ):
                sub = sub.replace( r'\b',  r''  )
                sub = sub.replace( r'\*',  r'*' )
                sub = sub.replace( r'\(',  r'(' )
                sub = sub.replace( r'\)',  r')' )
                sub = sub.replace( r'\.',  r'.' )
                sub = sub.replace( r'\^',  r'^' )
                subs_replace[key][i][j] = sub
            # end
        # end
    # end
except Exception as err:
    print( "Error: in subs:", file=sys.stderr )
    if 'key' in locals() and 'i' in locals():
        print( "row %d of substitution table '%s': %s" %
               (i, key, row), file=sys.stderr )
    traceback.print_exc()
    exit(1)


# ------------------------------------------------------------
class SourceFile( object ):
    '''SourceFile encapsulates a single file.
    If the file contains @precisions, it is a generator file.
    If the file contains @generated,  it is a generated file.
    It handles determining output files, Makefile rules, and generating other precisions.'''

    # --------------------
    # matches "@precisions z -> s d c"
    #                                          ($1_)  ($2_)      ($3_________)
    precisions_re = re.compile( r"@precisions +(\w+) +(\w+) +-> +(\w+( +\w+)*)" )

    generated_re  = re.compile( r"@generated" )

    # --------------------
    def __init__( self, filename ):
        '''Creates a single file.
        Determines whether it can do precision generation and if so,
        its substitution table, source and destination precisions.
        '''
        self._filename = filename
        fd = open( filename, 'r' )
        self._text = fd.read()
        fd.close()
        m = self.precisions_re.search( self._text )
        if m:
            self._is_generated = False
            self._table = m.group(1)          # e.g.:  normal or mixed
            self._src   = m.group(2)          # e.g.:  z
            self._dsts  = m.group(3).split()  # e.g.:  s, d, c
        else:
            m = self.generated_re.search( self._text )
            self._is_generated = (m != None)
            self._table = None
            self._src   = None
            self._dsts  = []
    # end

    # --------------------
    def is_generator( self ):
        '''True if this file can generate other precisions (has @precisions).'''
        return (self._table != None)

    # --------------------
    def is_generated( self ):
        '''True if this file was generated (has @generated).'''
        return self._is_generated

    # --------------------
    def get_filenames( self, precision=None ):
        '''Returns (files, precs) for the given precisions.
        files is list of generated filenames,
        precs is list of corresponding precisions.
        If precision is None, returns for all of file's precisions.
        If file is not a generator, returns empty lists.'''
        files = []
        ps    = []
        if self.is_generator():
            for prec in self._get_precisions( precision ):
                outname = self._substitute( self._filename, prec )
                if outname == self._filename:
                    print( "Error: no change in filename '%s' for %s -> %s; skipping" % (self._filename, self._src, prec), file=sys.stderr )
                else:
                    files.append( outname )
                    ps   .append( prec )
            # end
        # end
        return (files, ps)
    # end

    # --------------------
    def get_make_rules( self, precision=None ):
        '''Returns (files, precs, rules) for the given precisions.
        files is list of generated filenames,
        precs is list of corresponding precisions,
        rules is Makefile rules to make those files.
        If precision is None, returns for all of file's precisions.
        If file is not a generator, returns empty list and empty rules.'''
        (files, precs) = self.get_filenames( precision )
        rules = ""
        for (outname, prec) in zip( files, precs ):
            rules += "%s: %s\n\t$(codegen) -p %s $<\n\n" % (outname, self._filename, prec)
        # end
        return (files, precs, rules)
    # end

    # --------------------
    def generate_files( self, precision=None ):
        '''Generates files for the given precisions.
        If precision is None, generates for all of file's precisions.
        If file is not a generator, does nothing.'''
        (files, precs) = self.get_filenames( precision )
        for (outname, prec) in zip( files, precs ):
            if opts.verbose:
                print( "generating", outname, file=sys.stderr )
            output = self._substitute( self._text, prec )
            fd = open( outname, 'w' )
            fd.write( output )
            fd.close()
        # end
    # end

    # --------------------
    def _get_precisions( self, precision ):
        '''Given a precision or list of precisions,
        returns list of those that apply to this file.
        If precision is None, returns all of file's precisions.
        '''
        if precision:
            if isinstance( precision, (list, tuple) ):
                ps = filter( lambda x: x in self._dsts, precision )
            elif precision in self._dsts:
                ps = [ precision ]
            else:
                ps = []
        else:
            ps = self._dsts
        return ps
    # end

    # --------------------
    def _substitute( self, text, precision ):
        '''Apply substitutions to text for given precision.'''
        try:
            # Get substitution table based on self._table
            subs_o = subs[         self._table ]  # original
            subs_s = subs_search[  self._table ]  # compiled as search regexp
            subs_r = subs_replace[ self._table ]  # with regexp removed for replacement

            # Get which column is from and to.
            header = subs_o[0]
            jfrom = header.index( self._src )
            jto   = header.index( precision )
        except Exception as err:
            print( "Error: bad table or precision in '%s', @precisions %s %s -> %s:" %
                   (self._filename, self._table, self._src, self._dsts), file=sys.stderr )
            traceback.print_exc()
            exit(1)

        # Apply substitutions
        try:
            line = 0
            for (orig, search, replace) in zip( subs_o[1:], subs_s[1:], subs_r[1:] ):
                line += 1
                if orig[jfrom] == '':  # skip blank substitutions
                    continue
                if search[jfrom] is None:
                    search[jfrom] = re.compile( orig[jfrom] )
                text = re.sub( search[jfrom], replace[jto], text )
            # end
        except Exception as err:
            print( "Error: in row %d of substitution table '%s': %s" %
                   (line, self._table, subs_o[line]), file=sys.stderr )
            traceback.print_exc()
            exit(1)

        # Replace @precision with @generated, file, rule, and timestamp
        gen = "@generated from %s, %s %s -> %s, %s" % (
            self._filename, self._table, self._src, precision, datetime.now().ctime())
        text = re.sub( self.precisions_re, gen, text )
        return text
    # end
# end SourceFile


# ------------------------------------------------------------
def main():
    if opts.make:
        print( "# auto-generated by codegen.py $(%s_old), %s" % (opts.prefix, datetime.now().ctime()) )
        print( "%s_old := " % (opts.prefix) + ' '.join( opts.args ) + "\n" )
        templates = []
        generated = []
        for filename in opts.args:
            src = SourceFile( filename )
            if src.is_generator():
                (files, precs, rules) = src.get_make_rules( opts.precision )
                print( rules, end='' )
                generated += files
            # templates is all non-generated files, whether a generator or not
            if not src.is_generated():
                templates.append( filename )
        # end
        # print footer for Makefile.gen
        print( "%s_templates := \\\n\t" % (opts.prefix) + " \\\n\t".join( templates ) + "\n" )
        print( "%s_generated := \\\n\t" % (opts.prefix) + " \\\n\t".join( generated ) + "\n" )
        print( "%s_all := $(%s_templates) $(%s_generated)\n" % (opts.prefix, opts.prefix, opts.prefix) )
        print( "%s_generate: $(%s_generated)\n" % (opts.prefix, opts.prefix) )
        print( "%s_cleangen:"                   % (opts.prefix) )
        print( "\trm -f $(%s_generated)\n"      % (opts.prefix) )
        print( "generate: %s_generate\n"        % (opts.prefix) )
        print( "cleangen: %s_cleangen\n"        % (opts.prefix) )

    elif opts.output:
        generated = []
        for filename in opts.args:
            src = SourceFile( filename )
            (files, precs) = src.get_filenames( opts.precision )
            generated += files
        # end
        print( " ".join( generated ) )

    else:
        # default is to generate files
        for filename in opts.args:
            src = SourceFile( filename )
            src.generate_files( opts.precision )
        # end
    # end
# end main

if __name__ == "__main__":
    main()
