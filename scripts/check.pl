#!/usr/bin/env perl -i
#
# Adds CHECK around functions calls.
# CHECK is a macro defined in magmasparse_common.h that
# sets info to error & does goto 
#
# @author Mark Gates

use strict;

# read in whole file, so functions can span multiple lines
undef $/;

# functions to CHECK
my @funcs = (
    'magma_zparse_opts', 
    'magma_z_csr_mtx', 
    'magma_zcsrset', 
    'magma_zcsrget', 
    'magma_zvset', 
    'magma_zvget', 
    'magma_zvset_dev', 
    'magma_zvget_dev', 
    'magma_z_csr_mtxsymm', 
    'magma_z_csr_compressor', 
    'magma_zmcsrcompressor', 
    'magma_zmcsrcompressor_gpu', 
    'magma_zvtranspose', 
    'magma_z_cucsrtranspose', 
    'magma_zcsrsplit', 
    'magma_zmscale', 
    'magma_zmdiff', 
    'magma_zmdiagadd', 
    'magma_zmsort', 
    'magma_zindexsort',
    'magma_zdomainoverlap',
    'magma_zsymbilu', 
    'magma_zwrite_csr_mtx', 
    'magma_zwrite_csrtomtx', 
    'magma_zprint_csr', 
    'magma_zprint_csr_mtx', 
    'magma_zmtranspose',
    'magma_zmtransfer',
    'magma_zmconvert',
    'magma_zvinit',
    'magma_zprint_vector',
    'magma_zvread',
    'magma_zvspread',
    'magma_zprint_matrix',
    'magma_zdiameter',
    'magma_zrowentries',
    'magma_zresidual',
    'magma_zmgenerator',
    'magma_zm_27stencil',
    'magma_zm_5stencil',
    'magma_zsolverinfo_init',
    'magma_zeigensolverinfo_init',
    'magma_ziterilusetup', 
    'magma_zitericsetup', 
    'magma_zitericupdate', 
    'magma_zapplyiteric_l', 
    'magma_zapplyiteric_r', 
    'magma_ziterilu_csr', 
    'magma_ziteric_csr', 
    'magma_zfrobenius', 
    'magma_znonlinres',   
    'magma_zilures',   
    'magma_zicres',       
    'magma_zinitguess', 
    'magma_zinitrecursiveLU', 
    'magma_zmLdiagadd', 
    'magma_zcg',
    'magma_zcg_res',
    'magma_zcg_merge',
    'magma_zgmres',
    'magma_zbicgstab',
    'magma_zbicgstab_merge',
    'magma_zbicgstab_merge2',
    'magma_zpcg',
    'magma_zbpcg',
    'magma_zpbicgstab',
    'magma_zpgmres',
    'magma_zfgmres',
    'magma_zjacobi',
    'magma_zjacobidomainoverlap',
    'magma_zbaiter',
    'magma_ziterref',
    'magma_zilu',
    'magma_zbcsrlu',
    'magma_zbcsrlutrf',
    'magma_zbcsrlusv',
    'magma_zilucg',
    'magma_zilugmres',
    'magma_zlobpcg_shift',
    'magma_zlobpcg_res',
    'magma_zlobpcg_maxpy',
    'magma_zlobpcg',
    'magma_zjacobisetup',
    'magma_zjacobisetup_matrix',
    'magma_zjacobisetup_vector',
    'magma_zjacobiiter',
    'magma_zjacobiiter_precond', 
    'magma_zjacobiiter_sys',
    'magma_zpastixsetup',
    'magma_zapplypastix',
    'magma_zapplycustomprecond_l',
    'magma_zapplycustomprecond_r',
    'magma_zcuilusetup',
    'magma_zapplycuilu_l',
    'magma_zapplycuilu_r',
    'magma_zcuiccsetup',
    'magma_zapplycuicc_l',
    'magma_zapplycuicc_r',
    'magma_zcumilusetup',
    'magma_zcumilugeneratesolverinfo',
    'magma_zapplycumilu_l',
    'magma_zapplycumilu_r',
    'magma_zcumiccsetup',
    'magma_zcumicgeneratesolverinfo',
    'magma_zapplycumicc_l',
    'magma_zapplycumicc_r',
    'magma_zbajac_csr',
    'magma_z_spmv',
    'magma_zcustomspmv',
    'magma_z_spmv_shift',
    'magma_zcuspmm',
    'magma_z_spmm',
    'magma_zsymbilu', 
    'magma_zcuspaxpy',
    'magma_z_precond',
    'magma_z_solver',
    'magma_z_precondsetup',
    'magma_z_applyprecond',
    'magma_z_applyprecond_left',
    'magma_z_applyprecond_right',
    'magma_z_initP2P',
    'magma_zcompact',
    'magma_zcompactActive',
    'magma_zmlumerge',    
    'magma_zgecsrmv',
    'magma_zgecsrmv_shift',
    'magma_zmgecsrmv',
    'magma_zgeellmv',
    'magma_zgeellmv_shift',
    'magma_zmgeellmv',
    'magma_zgeelltmv',
    'magma_zgeelltmv_shift',
    'magma_zmgeelltmv',
    'magma_zgeellrtmv',
    'magma_zgesellcmv',
    'magma_zgesellpmv',
    'magma_zmgesellpmv',
    'magma_zmgesellpmv_blocked',
    'magma_zmergedgs',
    'magma_zcopyscale',    
    'magma_zjacobisetup_vector_gpu',
    'magma_zjacobi_diagscal',    
    'magma_zjacobiupdate',
    'magma_zjacobispmvupdate',
    'magma_zjacobispmvupdateselect',
    'magma_zjacobisetup_diagscal',
    'magma_zbicgmerge1',    
    'magma_zbicgmerge2',
    'magma_zbicgmerge3',
    'magma_zbicgmerge4',
    'magma_zcgmerge_spmv1', 
    'magma_zcgmerge_xrbeta', 
    'magma_zmdotc',
    'magma_zgemvmdot',
    'magma_zbicgmerge_spmv1', 
    'magma_zbicgmerge_spmv2', 
    'magma_zbicgmerge_xrbeta', 
    'magma_zbcsrswp',
    'magma_zbcsrtrsv',
    'magma_zbcsrvalcpy',
    'magma_zbcsrluegemm',
    'magma_zbcsrlupivloc',
    'magma_zbcsrblockinfo5',
    'magma_zmalloc',
    'magma_index_malloc',
    'magma_imalloc',
    'magma_dmalloc',
    'magma_zmalloc_cpu',
    'magma_index_malloc_cpu',
    'magma_imalloc_cpu',
    'magma_dmalloc_cpu',
    'magma_zmalloc_pinned',
    'magma_index_malloc_pinned',
    'magma_imalloc_pinned',
    'magma_dmalloc_pinned',
    'magma_malloc',
    'magma_malloc_cpu',
    'magma_malloc_pinned',
);

my $funcs = join( "|", @funcs );

my @cusparse = (
	'cusparseCreate',
	'cusparseCreateMatDescr',
	'cusparseCreateSolveAnalysisInfo',
	'cusparseDestroy',
	'cusparseDestroyMatDescr',
	'cusparseDestroySolveAnalysisInfo',
	'cusparseSetMatDiagType',
	'cusparseSetMatFillMode',
	'cusparseSetMatIndexBase',
	'cusparseSetMatType',
	'cusparseSetPointerMode',
	'cusparseSetStream',
	'cusparseXcsrgeamNnz',
	'cusparseXcsrgemmNnz',
	'cusparseZcsrgeam',
	'cusparseZcsrgemm',
	'cusparseZcsric0',
	'cusparseZcsrilu0',
	'cusparseZcsrsm_analysis',
	'cusparseZcsrsm_solve',
	'cusparseZcsrsv_analysis',
);

my $cusparse = join( "|", @cusparse );

# insert spaces after commas
sub args
{
	my($args) = @_;
	$args =~ s/,(\S)/, $1/g;
	return $args;
}

while( <> ) {
	# cleanup includes
	s|#include "../include/|#include "|g;
	
	# cleanup trailing spaces
	s|(\S) +$|$1|mg;
	
	# add include magmasparse_common.h, either replacing common_magma.h, or as last include
	if ( not m/magmasparse_internal\.h/ ) {
		if ( not s/#include "common_magma.h"/#include "magmasparse_internal.h"/ ) {
			s/(.*include.*?\n)/$1#include "magmasparse_common.h"\n/s;
		}
	}
	s/#include "magmasparse.h"\n//;  # included via magmasparse_internal.h
	
	# -----
	# CHECK functions
	s/^( +)($funcs) *(\([^;]*?\)) *;( ?) {0,8}/$1CHECK( $2$3);$4/mg;
	
	# change old stat += functions to CHECK
	s/^( +)stat\w* *\+?= *($funcs) *(\([^;]*?\)) *;( ?) {0,8}/$1CHECK( $2$3);$4/mg;
	
	# -----
	# CHECK cusparse
	s/^( +)\w*[Ss]tat\w*\s*=\s*($cusparse) *\( *([^;]*?\S) *\) *;/"$1CHECK_CUSPARSE( $2( ".args($3)." ));"/mge;
	s/^( +)($cusparse) *\( *([^;]*?\S) *\) *;/"$1CHECK_CUSPARSE( $2( ".args($3)." ));"/mge;
	
	# strip out old status checks
	s/^ +if *\( *cusparseStatus *!= *0 *\)\s*printf\([^;]+\); *\n//mg;
	s/^ +cusparseStatus_t +\w+ *; *\n//mg;
	
	# add note to manually check Destroy statements (e.g., go in cleanup?)
	if ( not m/TODO put destroy into cleanup/ ) {
		s|(.*cusparseDestroy.*)|$1 // TODO put destroy into cleanup?|mg;
	}
	
	# -----
	# add note to manually check all return statements
	if ( not m/TODO change to goto cleanup/ ) {
		s|^( *return.*)|$1 // TODO change to goto cleanup?|mg;
	}
	
	print;
}
