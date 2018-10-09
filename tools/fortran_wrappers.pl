#!/usr/bin/perl
#
# Processes include/magma_[sdcz].h header files (pre-processed as magma_[sdcz].i)
# to generate
# control/magma_zf77.cpp     Fortran wrappers and
# control/magma_zfortran.F90 Fortran interfaces.
#
# @author Mark Gates

my $usage = "Usage: $0 [--opencl] [--wrapper file] [--interface file] magma_z.i\n";

use strict;
use Text::Balanced qw( extract_bracketed );
use Getopt::Long;
use Text::Wrap;

# --------------------
# parse options
my $verbose = 0;
my $opencl = 0;
my( $file_wrapper, $file_interface );
GetOptions(
	"verbose"     => \$verbose,
	"opencl"      => \$opencl,
	"wrapper=s"   => \$file_wrapper,
	"interface=s" => \$file_interface,
) or die( "$!$usage" );

if ($verbose) {
	print "ARGV: @ARGV\n";
}

if ($#ARGV != 0) {
	die("Specify one header file as input.\n$usage");
}

my( $blas, $precision ) = $ARGV[0] =~ m/magma(blas)?_(s|d|c|z|zc|ds)\.i/;
if (not $precision) {
	die("Specify one of magma_[sdcz].i pre-processed header files.\n$usage");
}

# default output files
if (not $file_wrapper  ) { $file_wrapper   = "control/magma${blas}_${precision}f77.cpp"; }
if (not $file_interface) { $file_interface = "control/magma${blas}_${precision}fortran.F90"; }

print "file_wrapper   $file_wrapper\n";
print "file_interface $file_interface\n";


# --------------------
# declare constants and variables
my( $pre, $return, $prefix, $func, $is_gpu, $text, $rest, $comment,
	$funcf,
	$wrapper, $call, $interface, $vars,
	$args, @args, $arg, $type, $type2, $base_type, $var, $first_arg, $is_ptr,
	%seen );

# ignore auxiliary functions which the user doesn't need
# ignore PLASMA functions (tstrf, incpiv)
# ignore misc functions in headers but not in library (larfg, getrf2, geqr2) (as of 2012-04-16)
my @ignore = qw(
	[sdcz]geqrf_msub
	[sdcz]getrf_msub
	[sdcz]getrf2_msub
	[sdcz]potrf_msub
	[sdcz]potrf2_msub
	
	[cz]lahef
	[sd]lasyf
	[sd]sygvr
	[sd]sygvx
	[sdcz]geqr2
	[sdcz]getrf2
	[sdcz]getrf_incpiv
	[sdcz]labrd
	[sdcz]laex\d
	[sdcz]laex\d_m
	[sdcz]lahr2
	[sdcz]lahr2_m
	[sdcz]lahru
	[sdcz]lahru_m
	[sdcz]laqps\d?
	[sdcz]larfg
	[sdcz]latrd
	[sdcz]latrd2
	[sdcz]tsqrt
	[sdcz]tstrf
	
	[sdcz]getmatrix_1D_row_bcyclic
	[sdcz]setmatrix_1D_row_bcyclic
	[sdcz]getmatrix_1D_col_bcyclic
	[sdcz]setmatrix_1D_col_bcyclic
	[sdcz]htodhe
	[sdcz]htodpo
	[sdcz]dtohhe
	[sdcz]dtohpo
);
my $ignore = join( "|", @ignore );
if ($verbose) {
	print STDERR "ignore: $ignore\n";
}

# map C types to name for magma_<name>_const(...) call
my %constants = (
	'magma_order_t'      => 'order'  ,
	'magma_trans_t'      => 'trans'  ,
	'magma_uplo_t'       => 'uplo'   ,
	'magma_diag_t'       => 'diag'   ,
	'magma_side_t'       => 'side'   ,
	'magma_type_t'       => 'uplo'   ,  # type == uplo
	'magma_norm_t'       => 'norm'   ,
	'magma_dist_t'       => 'dist'   ,
	'magma_pack_t'       => 'pack'   ,
	'magma_vec_t'        => 'vec'    ,
	'magma_range_t'      => 'range'  ,
	'magma_vect_t'       => 'vect'   ,
	'magma_direct_t'     => 'direct' ,
	'magma_storev_t'     => 'storev' ,
	'magma_bool_t'       => 'bool'   ,
);

# map C base types to Fortran types
my %f90types = (
	'size_t'             => 'integer         ',  # does Fortran have unsigned?
	'char'               => 'character       ',
	'int'                => 'integer         ',
	'magma_int_t'        => 'integer         ',
	'float'              => 'real            ',
	'double'             => 'double precision',
	'cuFloatComplex'     => 'complex         ',
	'cuDoubleComplex'    => 'complex*16      ',
	'magmaFloatComplex'  => 'complex         ',
	'magmaDoubleComplex' => 'complex*16      ',
	'magma_queue_t'      => 'magma_devptr_t  ',  # really host pointers, not device pointers,
	'magma_event_t'      => 'magma_devptr_t  ',  # but reusing devptr is an easy solution.
	
	'magma_ptr'              => 'magma_devptr_t',
	'magma_const_ptr'        => 'magma_devptr_t',
	'magmaInt_ptr'           => 'magma_devptr_t',
	'magmaFloat_ptr'         => 'magma_devptr_t',
	'magmaDouble_ptr'        => 'magma_devptr_t',
	'magmaFloatComplex_ptr'  => 'magma_devptr_t',
	'magmaDoubleComplex_ptr' => 'magma_devptr_t',
);

# add all symbolic constant types (magma_uplo_t, etc.), mapping to character
for my $type ( keys %constants ) {
	$f90types{ $type } = 'character       ';
}

my %devptrs = (
	'magma_ptr'              => 'magma_devptr',
	'magma_const_ptr'        => 'magma_devptr',
	
	'magma_int_t'            => 'magma_idevptr',
	'float'                  => 'magma_sdevptr',
	'double'                 => 'magma_ddevptr',
	
	'cuFloatComplex'         => 'magma_cdevptr',
	'cuDoubleComplex'        => 'magma_zdevptr',
	'magmaFloatComplex'      => 'magma_cdevptr',
	'magmaDoubleComplex'     => 'magma_zdevptr',
	
	'magmaInt_ptr'           => 'magma_idevptr',
	'magmaFloat_ptr'         => 'magma_sdevptr',
	'magmaDouble_ptr'        => 'magma_ddevptr',
	'magmaFloatComplex_ptr'  => 'magma_cdevptr',
	'magmaDoubleComplex_ptr' => 'magma_zdevptr',
	
	'magmaInt_const_ptr'           => 'magma_idevptr',
	'magmaFloat_const_ptr'         => 'magma_sdevptr',
	'magmaDouble_const_ptr'        => 'magma_ddevptr',
	'magmaFloatComplex_const_ptr'  => 'magma_cdevptr',
	'magmaDoubleComplex_const_ptr' => 'magma_zdevptr',
);

# Fortran 90 has 132 line length limit, so wrap text
$Text::Wrap::separator = "  &\n";
$Text::Wrap::columns   = 90;
$Text::Wrap::unexpand  = 0;  # no tabs


# --------------------
# header for magma_*f77.cpp wrappers

my $output_wrapper = <<EOT;
/*******************************************************************************
 *  This file is AUTOMATICALLY GENERATED by:
 *  $0 @ARGV
 *  Do not edit.
 ******************************************************************************/

#include <stdint.h>  // for uintptr_t

#include "magma_v2.h"
#include "magma_mangling.h"

// typedef from CUDADIR/src/fortran.h directory
typedef size_t devptr_t;

#ifdef PGI_FORTRAN
    #define magma_devptr(ptr_)  ((void*)              (ptr_))
    #define magma_idevptr(ptr_) ((magma_int_t*)       (ptr_))
    #define magma_zdevptr(ptr_) ((magmaDoubleComplex*)(ptr_))
    #define magma_cdevptr(ptr_) ((magmaFloatComplex*) (ptr_))
    #define magma_ddevptr(ptr_) ((double*)            (ptr_))
    #define magma_sdevptr(ptr_) ((float*)             (ptr_))
#else
    #define magma_devptr(ptr_)  ((void*)              (uintptr_t)(*(ptr_)))
    #define magma_idevptr(ptr_) ((magma_int_t*)       (uintptr_t)(*(ptr_)))
    #define magma_zdevptr(ptr_) ((magmaDoubleComplex*)(uintptr_t)(*(ptr_)))
    #define magma_cdevptr(ptr_) ((magmaFloatComplex*) (uintptr_t)(*(ptr_)))
    #define magma_ddevptr(ptr_) ((double*)            (uintptr_t)(*(ptr_)))
    #define magma_sdevptr(ptr_) ((float*)             (uintptr_t)(*(ptr_)))
#endif

#ifdef __cplusplus
extern "C" {
#endif

EOT


# --------------------
# header for magma_*fortran.F90 interfaces
my $output_interface = <<EOT;
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!  This file is AUTOMATICALLY GENERATED by:
!!  $0 @ARGV
!!  Do not edit.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module magma${blas}_${precision}fortran

use magma_param

implicit none

!---- Fortran interfaces to MAGMA subroutines ----
interface

EOT


# --------------------
# slurp whole file
undef $/;
$_ = <>;

# strip out everything preceding the last extern "C", and the extern "C" itself.
# This is our crude method to tell what is magma stuff and what is from other included headers.
s/^.*extern "C" \{\s*//s;
s/\}\s*$//;

# strip out pre-processor directives (i.e., include file line numbers)
s/^#.*//mg;

# strip out extern consts
s/extern const .*;\n//g;

# strip out static, inline
s/\b(static|inline)\s+//g;

my $i = 1;
while ($_) {
	#print STDERR $i, ': ', substr( $_, 0, 10 ), "\n";
	#$i += 1;
	
	# look for function, e.g.: "magma_int_t magma_foo_gpu("
	if (m/(.*?)^(magma_int_t|float|double|int|void)\s+(magma|magmablas)_(\w+?)(_gpu)?\s*(\(.*)/ms) {
		# parse magma function
		$pre    = $1;
		$return = $2;
		$prefix = $3;
		$func   = $4;
		$is_gpu = $5;
		$text   = $6;
		
		# get arguments: "( ... )"
		($args, $rest) = extract_bracketed( $text, '()' );
		$args =~ s/\n/ /g;
		$args =~ s/^\( *//;
		$args =~ s/ *\)$//;
		
		# Most MAGMA functions return info; ignore the return value.
		if ($return eq 'magma_int_t' and $args =~ m/\binfo\b/) {
			$return = '';
		}
		if ($return eq 'void') {
			$return = ''
		}
		
		if ($func =~ s/_internal//) {
			print STDERR "Dropping internal args (func, file, line) on $func:\n    $args\n";
			$args =~ s/,\s+const char\*\s+func,\s+const char\*\s+file,\s+int\s+line//;
			print STDERR "=>  $args\n\n";
		}
		
		@args = split( /, */, $args );
		
		$funcf = "${prefix}f_$func$is_gpu";
		$wrapper = sprintf( "#define %s FORTRAN_NAME( %s, %s )\n",
			${funcf}, $funcf, uc($funcf) );
		
		my $match = $func =~ m/^($ignore)$/;
		if ($verbose) {
			print STDERR "FUNC $func $match\n";
		}
		if ($opencl and $is_gpu) {
			# ignore OpenCL GPU functions, since
			# we haven't dealt with passing cl_mem objects in Fortran yet
			$wrapper   = "";
			$interface = "";
		}
		elsif ($func =~ m/^($ignore)$/ or $func =~ m/(_mgpu|_batch|_vbatch)/) {
			# ignore auxiliary functions and multi-GPU functions, since
			# we haven't dealt with passing arrays of pointers in Fortran yet
			$wrapper   = "";
			$interface = "";
		}
		elsif ($seen{ $funcf }) {
			print STDERR "WARNING: ignoring duplicate $func\n";
			$wrapper   = "";
			$interface = "";
		}
		else {
			$seen{ $funcf } = 1;
			
			if ($return) {
				# function
				# build up wrapper and the call inside the wrapper, argument by argument
				$wrapper .= "$return ${funcf}(\n    ";
				$call     = "return ${prefix}_$func$is_gpu(\n        ";
				
				# build up Fortran interface and variable definitions, argument by argument
				my $t = $f90types{$return};
				$t =~ s/ +$//;
				$interface = "$t function $funcf( ";
				$vars      = "";
			}
			else {
				# subroutine
				# build up wrapper and the call inside the wrapper, argument by argument
				$wrapper .= "void ${funcf}(\n    ";
				$call     = "${prefix}_$func$is_gpu(\n        ";
				
				# build up Fortran interface and variable definitions, argument by argument
				$interface = "subroutine $funcf( ";
				$vars      = "";
			}
			
			$first_arg = 1;
			foreach $arg (@args) {
				$arg =~ s/\bconst +//g;
				($type, $var, $type2) = $arg =~ m/^((?:const +)?\w+(?: *\*+)?) *(\w+)([\[0-9\]]*)$/;
				if (not $type) {
					print STDERR "\nFAILED: func $func, arg $arg\n";
				}
				$base_type = $type;
				$base_type =~ s/\*//;
				$base_type =~ s/const +//;
				$base_type =~ s/^ +//;
				$base_type =~ s/ +$//;
				#print STDERR "base_type $base_type\n";
				
				$is_ptr = ($type =~ m/\*/ or $type =~ m/_ptr$/ or $type2 ne "");
				#print STDERR "  type $type, type2 $type2, var $var, is_ptr $is_ptr\n";
				if ($is_ptr) {
					unless( $first_arg ) {
						$wrapper   .= ",\n    ";
						$call      .= ",\n        ";
						$interface .= ", ";
					}
					if ($type =~ m/_ptr$/) {  # or ($is_gpu and $var =~ m/^d\w+/) or $var eq "dT") {
						# The _ptr types identify device pointers.
						# OLD: For _gpu interfaces, assume ptrs that start with "d" are device pointers.
						# OLD: CPU interface for geqrf, etc., passes dT as device pointer (weirdly).
						if ($type !~ m/_ptr$/) {
							print STDERR "WARNING: assuming in $func$is_gpu: $var is devptr, but lacks _ptr!\n";
						}
						if (not defined $devptrs{$base_type}) {
							print STDERR "ERROR: devptrs{$base_type} not defined\n";
						}
						$wrapper .= "devptr_t *$var";
						$call    .= "$devptrs{$base_type}($var)";
						$vars    .= "    magma_devptr_t   :: $var\n";
					}
					else {
						$wrapper .= $arg;
						$call    .= $var;
						if ($var =~ m/^(info|iter|m)$/) {
							# special case for variables passed as pointers in C,
							# but are single values, not arrays.
							# e.g., see zhegvd and zcgesv
							$vars .= "    $f90types{$base_type} :: $var\n";
						}
						else {
							$vars .= "    $f90types{$base_type} :: $var(*)\n";
						}
					}
					$interface .= $var;
				}
				else {
					unless( $first_arg ) {
						$wrapper   .= ", ";
						$call      .= ", ";
						$interface .= ", ";
					}
					# convert scalars to pointers for Fortran interface
					# map string constants => symbolic constants
					if ($constants{$type}) {
						$wrapper   .= "const char* $var";
						$call      .= "magma_$constants{$type}_const(*$var)";
					}
					else {
						$wrapper   .= "$type *$var";
						$call      .= "*$var";
					}
					$interface .= $var;
					$vars      .= "    $f90types{$base_type} :: $var\n";
				}
				$first_arg = 0;
			}
			$wrapper .= " )\n{\n    $call );\n}\n\n";
			
			$interface .= " )\n";
			$interface  = Text::Wrap::wrap( "", "        ", $interface );
			$interface .= "${vars}end\n\n";
			#if ($return) {
			#	$interface .= "end function $funcf\n\n";
			#}
			#else {
			#	$interface .= "end subroutine $funcf\n\n";
			#}
		}
		
		if ($pre and $verbose) {
			print STDERR "WARNING: ignoring unrecognized text before function: <<<\n$pre>>>\n";
		}
		
		$output_wrapper   .= $wrapper;
		$output_interface .= $interface;
		
		$_ = $rest;
		s/^ *;//;
	}
	else {
		if ($verbose) {
			print STDERR "WARNING: ignoring unrecognized text at end of file: <<<\n$_>>>\n";
		}
		last;
	}
	s/^\n+//;
}


# --------------------
# footer for magma_zf77.cpp wrappers
$output_wrapper .= <<EOT;

#ifdef __cplusplus
}
#endif
EOT


# --------------------
# footer for magma_zfortran.F90 interfaces
my %datatype = (
	s => 'real',
	d => 'double',
	c => 'complex',
	z => 'complex_16',
);

$output_interface .= <<EOT;
end interface

end module magma${blas}_${precision}fortran
EOT


# --------------------
# output both files
open( OUT, ">$file_wrapper" ) or die( "Can't open $file_wrapper: $!" );
print OUT $output_wrapper;
close( OUT );

open( OUT, ">$file_interface" ) or die( "Can't open $file_interface: $!" );
print OUT $output_interface;
close( OUT );
