#!/usr/bin/perl -0777 -p
#
# Used as INPUT_FILTER in Doxyfile to strips out "Arguments" and "Purpose"
# from MAGMA docs, as doxygen provides its own sections.
# Also strips trailing _q from v2 names (which are accessed through macros without the _q.)
#
# -0777 slurps whole files; see 'man perlrun'

use strict;

s/^ +Arguments *\n +[-=]{2,}$/\n/gm;
s/^ +Purpose *\n +[-=]{2,}$/\n/gm;

# strip "_q" from end of function names, but keep "magma_zpanel_to_q"
if (not m/magma_\wpanel_to_q/ ) {
    s/\b(magma\w+)_q\(/$1(/g;
}

#--------------------
# Recognize $...$ and \[...\] Latex syntax.
# See slate/tools/doxygen-filter.pl
sub dollar
{
    my( $pre ) = @_;
    if ($pre eq '\\') {
        # change \$ to $
        return '$';
    }
    elsif ($pre eq '\\f') {
        # don't change \f$
        return '\\f$';
    }
    else {
        # change $ to \f$
        return $pre . '\\f$';
    }
}

# replace \[ and \] by \f[ and \f]
s/\\([\[\]])/\\f$1/g;

# replace         $  by  \f$
# replace        \$  by  $
# don't change  \f$
s/(\\f|\\|)\$/dollar($1)/eg;
