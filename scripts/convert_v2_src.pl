#!/usr/bin/perl -i
#
# Does some basic conversion from MAGMA v1 to v2.
# 1) Change "stream[...]" to "queues[...]"
#
# 2) Change header to magma_v2.h or magma_internal.h
#
# 3) Add device to magma_queue_create.
#    This may add extraneous get_device calls that should be manually removed.
#
# 4) Comment out Set/GetKernelStream.
#    These should be manually removed after verifying the code.
#
# 5) Add queue argument to MAGMA kernel calls.
#    It gets this queue from the last seen SetKernelStream -- but this totally
#    ignores the actual control flow of the program, so it may be wrong!
#
# Modified code must be manually verified for correctness.
#
# @author Mark Gates

use strict;

# MAGMA kernels, in alphabetical order
my @kernels = qw(
	amax
	asum
	axpy
	caxpycp
	cnrm2
	copy
	dot
	gemm
	gemv
	hemm
	hemv
	her2k
	herk
	lacpy
	lag2c
	lag2z
	lange
	lanhe
	lansy
	lantr
	larfb
	laset
	laswp
	lat2c
	lat2z
	nrm2
	scal
	swap
	symm
	symv
	transpose
	trmm
	trsm
	trsv
	zamax
	zasum
	znrm2
);

my @setget = qw(
	copyvector
	copymatrix
	getvector
	getmatrix
	setvector
	setmatrix
);

my $kernels = join( "|", @kernels );
my $setget  = join( "|", @setget  );

$/ = ";";  # slurp C statements

my $queue;
my $last;
while( <> ) {
	if ($ARGV ne $last) {
		$queue = "UNKNOWN";
		$last = $ARGV;
	}
	##print "<<<< $_ >>>>\n\n";
	
	# update version
	s/MAGMA \(version \d\.\d\)/MAGMA (version 2.0)/;
	
	# rename stream array to queues
	# make arrays plural
	s/\bstream *\[/queues[/g;
	s/\bevent *\[/events[/g;
	
	# fix headers
	s/#include "magma.h"/#include "magma_v2.h"/;
	s/#include "common_magma.h"/#include "magma_internal.h"/;
	s/#include "magmasparse_internal.h"/#include "magmasparse_internal.h"/;
	
	# change UpperLower to Full, for consistency with LAPACK
	s/UpperLower/Full/g;
	
	# fix queue_create
	#  ($1)                      ($2 )
	s/^( +)magma_queue_create\( *(.*?) *\);/${1}magma_device_t cdev;\n${1}magma_getdevice( \&cdev );\n${1}magma_queue_create( cdev, $2 );/mg;
	
	# comment out Set/GetKernelStream & record queue
	if ( s@(magmablasSetKernelStream\( *(.*?) *\);)@//$1@g ) {
		##print "[[[[ queue $queue ]]]]\n";
		$queue = $2;
	}
	s@(magmablasGetKernelStream\( *(.*?) *\);)@//$1@g;
	s@(magma_queue_t +orig_(stream|queue))@//$1@g;
	
	# add queue to BLAS/GPU kernels
	#   ($1 ..............................................)      ($2 ...)
	s/\b(magma(?:blas)?_[isdcz](?:(?:$kernels)\w*|$setget)) *\( ?([^;]*?)\s*\);/$1( $2, $queue );/g;
	s/\b(magma(?:blas)?_index_(?:(?:$kernels)\w*|$setget)) *\( ?([^;]*?)\s*\);/$1( $2, $queue );/g;
	
	print;
}
