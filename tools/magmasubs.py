# Substitutions used in codegen.py
#
# Substitutions are applied in the order listed. This is important in cases
# where multiple substitutions could match, or when one substitution matches
# the result of a previous substitution. For example, these rules are correct
# in this order:
#
#    ('real',   'double precision',  'real',   'double precision' ),  # before double
#    ('float',  'double',            'float',  'double'           ),
#
# but if switched would translate 'double precision' -> 'float precision',
# which is wrong.
#
# @author Mark Gates



# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
#                                                                             #
#          DO NOT EDIT      OpenCL and MIC versions      DO NOT EDIT          #
#          DO NOT EDIT      OpenCL and MIC versions      DO NOT EDIT          #
#          DO NOT EDIT      OpenCL and MIC versions      DO NOT EDIT          #
#                                                                             #
# Please edit the CUDA MAGMA version, then copy it to MIC and OpenCL MAGMA.   #
# Otherwise they get out-of-sync and it's really hard to sync them up again.  #
#                                                                             #
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #


# ===========================================================================
# utilitiy functions

# ----------------------------------------
def upper( table ):
    '''
    maps double-nested list of strings to upper case.
    [ ['Foo', 'bar'], ['baz', 'ZAB'] ]
    becomes
    [ ['FOO', 'BAR'], ['BAZ', 'ZAB'] ]
    '''
    ucase = [ [ x.upper() for x in row ] for row in table ]
    return ucase
# end


# ----------------------------------------
def lower( table ):
    '''
    maps double-nested list of strings to lower case.
    [ ['Foo', 'BAR'], ['BAZ', 'zab'] ]
    becomes
    [ ['foo', 'bar'], ['baz', 'zab'] ]
    '''
    lcase = [ [ x.lower() for x in row ] for row in table ]
    return lcase
# end


# ----------------------------------------
def title( table ):
    '''
    Maps double-nested list of strings to Title case. Useful for cuBLAS.
    [ ['FOO', 'bar'], ['Baz', 'Zab'] ]
    becomes
    [ ['Foo', 'Bar'], ['Baz', 'Zab'] ]
    '''
    tcase = [ [ x.title() for x in row ] for row in table ]
    return tcase
# end


# ===========================================================================
# BLAS and LAPACK routines need both lower and uppercase, for example:
# in filenames:              zgetrf.cpp
# in magma_zlapack.h:        FORTRAN_NAME( zaxpy, ZAXAPY )
# in doxygen documentation:  ZGETRF computes ...
# BLAS also needs Titlecase: cublasZaxpy
# The easiest way to maintain this is to separate these lists here,
# and use them later with the above lower, upper, and title routines.

# ----------------------------------------
blas_mixed = [
    # BLAS and LAPACK, lowercase, alphabetic order
    # for mixed precision
    ('daxpy',          'zaxpy'           ),
    ('ddot',           'zdotc'           ),
    ('dgemm',          'zgemm'           ),
    ('dgeqrf',         'zgeqrf'          ),
    ('dgeqrs',         'zgeqrs'          ),
    ('dgesv',          'zgesv'           ),
    ('dgetrf',         'zgetrf'          ),
    ('dgetrs',         'zgetrs'          ),
    ('dlacpy',         'zlacpy'          ),
    ('dlag2s',         'zlag2c'          ),
    ('dlagsy',         'zlaghe'          ),
    ('dlange',         'zlange'          ),
    ('dlansy',         'zlanhe'          ),
    ('dlansy',         'zlansy'          ),
    ('dlarnv',         'zlarnv'          ),
    ('dlat2s',         'zlat2c'          ),
    ('dnrm2',          'dznrm2'          ),
    ('dormqr',         'zunmqr'          ),
    ('dpotrf',         'zpotrf'          ),
    ('dpotrs',         'zpotrs'          ),
    ('dsymm',          'zhemm'           ),
    ('dsymv',          'zhemv'           ),
    ('dsyrk',          'zherk'           ),
    ('dsytrf',         'zhetrf'          ),
    ('dsytrs',         'zhetrs'          ),
    ('dtrmm',          'ztrmm'           ),
    ('dtrsm',          'ztrsm'           ),
    ('dtrsv',          'ztrsv'           ),
    ('idamax',         'izamax'          ),
    ('slag2d',         'clag2z'          ),
    ('slansy',         'clanhe'          ),
    ('slat2d',         'clat2z'          ),
    ('spotrf',         'cpotrf'          ),
    ('ssysv',          'chesv'           ),
    ('ssysv',          'csysv'           ),
    ('ssytrf',         'chetrf'          ),
    ('ssytrs',         'chetrs'          ),
    ('strmm',          'ctrmm'           ),
    ('strsm',          'ctrsm'           ),
    ('strsv',          'ctrsv'           ),
]


# ----------------------------------------
blas = [
    # BLAS, lowercase, alphabetic order
    ('isamax',         'idamax',         'icamax',         'izamax'          ),
    ('isamax',         'idamax',         'isamax',         'idamax'          ),
    ('isamin',         'idamin',         'icamin',         'izamin'          ),
    ('sasum',          'dasum',          'scasum',         'dzasum'          ),
    ('saxpy',          'daxpy',          'caxpy',          'zaxpy'           ),
    ('scopy',          'dcopy',          'ccopy',          'zcopy'           ),
    ('scopy',          'dcopy',          'scopy',          'dcopy'           ),
    ('sdot',           'ddot',           'cdotc',          'zdotc'           ),
    ('sdot',           'ddot',           'cdotu',          'zdotu'           ),
    ('sgemm',          'dgemm',          'cgemm',          'zgemm'           ),
    ('sgemv',          'dgemv',          'cgemv',          'zgemv'           ),
    ('sger',           'dger',           'cgerc',          'zgerc'           ),
    ('sger',           'dger',           'cgeru',          'zgeru'           ),
    ('snrm2',          'dnrm2',          'scnrm2',         'dznrm2'          ),
    ('srot',           'drot',           'crot',           'zrot'            ),
    ('srot',           'drot',           'csrot',          'zdrot'           ),
    ('srot',           'drot',           'srot',           'drot'            ),
    ('sscal',          'dscal',          'cscal',          'zscal'           ),
    ('sscal',          'dscal',          'csscal',         'zdscal'          ),
    ('sscal',          'dscal',          'sscal',          'dscal'           ),
    ('sswap',          'dswap',          'cswap',          'zswap'           ),
    ('ssymm',          'dsymm',          'chemm',          'zhemm'           ),
    ('ssymm',          'dsymm',          'csymm',          'zsymm'           ),
    ('ssymv',          'dsymv',          'chemv',          'zhemv'           ),
    ('ssymv',          'dsymv',          'csymv',          'zsymv'           ),
    ('ssyr',           'dsyr',           'cher',           'zher'            ),  # also does zher2, zher2k, zherk
    ('ssyr',           'dsyr',           'csyr',           'zsyr'            ),  # also does zsyrk, zsyr2k
    ('strmm',          'dtrmm',          'ctrmm',          'ztrmm'           ),
    ('strmv',          'dtrmv',          'ctrmv',          'ztrmv'           ),
    ('strsm',          'dtrsm',          'ctrsm',          'ztrsm'           ),
    ('strsv',          'dtrsv',          'ctrsv',          'ztrsv'           ),
]


# ----------------------------------------
lapack = [
    # LAPACK, lowercase, alphabetic order
    ('sbdsdc',         'dbdsdc',         'sbdsdc',         'dbdsdc'          ),
    ('sbdsqr',         'dbdsqr',         'cbdsqr',         'zbdsqr'          ),
    ('sbdt01',         'dbdt01',         'cbdt01',         'zbdt01'          ),
    ('sdiinertia',     'ddiinertia',     'cdiinertia',     'zdiinertia'      ),
    ('ssidi',          'dsidi',          'ssidi',          'dsidi'           ),
    ('sgbbrd',         'dgbbrd',         'cgbbrd',         'zgbbrd'          ),
    ('sgbsv',          'dgbsv',          'cgbsv',          'zgbsv'           ),
    ('sgebak',         'dgebak',         'cgebak',         'zgebak'          ),
    ('sgebal',         'dgebal',         'cgebal',         'zgebal'          ),
    ('sgebd2',         'dgebd2',         'cgebd2',         'zgebd2'          ),
    ('sgebrd',         'dgebrd',         'cgebrd',         'zgebrd'          ),
    ('sgeev',          'dgeev',          'cgeev',          'zgeev'           ),
    ('sgegqr',         'dgegqr',         'cgegqr',         'zgegqr'          ),
    ('sgehd2',         'dgehd2',         'cgehd2',         'zgehd2'          ),
    ('sgehrd',         'dgehrd',         'cgehrd',         'zgehrd'          ),
    ('sgejsv',         'dgejsv',         'cgejsv',         'zgejsv'          ),
    ('sgelq2',         'dgelq2',         'cgelq2',         'zgelq2'          ),
    ('sgelqf',         'dgelqf',         'cgelqf',         'zgelqf'          ),
    ('sgelqs',         'dgelqs',         'cgelqs',         'zgelqs'          ),
    ('sgels',          'dgels',          'cgels',          'zgels'           ),
    ('sgeqlf',         'dgeqlf',         'cgeqlf',         'zgeqlf'          ),
    ('sgeqp3',         'dgeqp3',         'cgeqp3',         'zgeqp3'          ),
    ('sgeqr2',         'dgeqr2',         'cgeqr2',         'zgeqr2'          ),
    ('sgeqrf',         'dgeqrf',         'cgeqrf',         'zgeqrf'          ),
    ('sgerqf',         'dgerqf',         'cgerqf',         'zgerqf'          ),
    ('sgeqrs',         'dgeqrs',         'cgeqrs',         'zgeqrs'          ),
    ('sgeqrt',         'dgeqrt',         'cgeqrt',         'zgeqrt'          ),
    ('sgerfs',         'dgerfs',         'cgerfs',         'zgerfs'          ),
    ('sgesdd',         'dgesdd',         'cgesdd',         'zgesdd'          ),
    ('sgessm',         'dgessm',         'cgessm',         'zgessm'          ),
    ('sgesv',          'dgesv',          'cgesv',          'zgesv'           ),  # also does zgesvd, zgesvj
    ('sget22',         'dget22',         'cget22',         'zget22'          ),
    ('sgetf2',         'dgetf2',         'cgetf2',         'zgetf2'          ),
    ('sgetmi',         'dgetmi',         'cgetmi',         'zgetmi'          ),
    ('sgetmo',         'dgetmo',         'cgetmo',         'zgetmo'          ),
    ('sgetrf',         'dgetrf',         'cgetrf',         'zgetrf'          ),
    ('sgetri',         'dgetri',         'cgetri',         'zgetri'          ),
    ('sgetrs',         'dgetrs',         'cgetrs',         'zgetrs'          ),
    ('sgglse',         'dgglse',         'cgglse',         'zgglse'          ),
    ('sggrqf',         'dggrqf',         'cggrqf',         'zggrqf'          ),
    ('shseqr',         'dhseqr',         'chseqr',         'zhseqr'          ),
    ('shst01',         'dhst01',         'chst01',         'zhst01'          ),
    ('slabad',         'dlabad',         'slabad',         'dlabad'          ),
    ('slabrd',         'dlabrd',         'clabrd',         'zlabrd'          ),
    ('slacgv',         'dlacgv',         'clacgv',         'zlacgv'          ),
    ('slacp2',         'dlacp2',         'clacp2',         'zlacp2'          ),
    ('slacpy',         'dlacpy',         'clacpy',         'zlacpy'          ),
    ('slacrm',         'dlacrm',         'clacrm',         'zlacrm'          ),
    ('sladiv',         'dladiv',         'cladiv',         'zladiv'          ),
    ('slaed',          'dlaed',          'slaed',          'dlaed'           ),
    ('slaex',          'dlaex',          'slaex',          'dlaex'           ),
    ('slag2d',         'dlag2s',         'clag2z',         'zlag2c'          ),
    ('slagsy',         'dlagsy',         'claghe',         'zlaghe'          ),
    ('slagsy',         'dlagsy',         'clagsy',         'zlagsy'          ),
    ('slahr',          'dlahr',          'clahr',          'zlahr'           ),
    ('slaln2',         'dlaln2',         'slaln2',         'dlaln2'          ),
    ('slamc3',         'dlamc3',         'slamc3',         'dlamc3'          ),
    ('slamch',         'dlamch',         'slamch',         'dlamch'          ),
    ('slamrg',         'dlamrg',         'slamrg',         'dlamrg'          ),
    ('slange',         'dlange',         'clange',         'zlange'          ),
    ('slanst',         'dlanst',         'clanht',         'zlanht'          ),
    ('slansy',         'dlansy',         'clanhe',         'zlanhe'          ),
    ('slansy',         'dlansy',         'clansy',         'zlansy'          ),
    ('slantr',         'dlantr',         'clantr',         'zlantr'          ),
    ('slapy3',         'dlapy3',         'slapy3',         'dlapy3'          ),
    ('slaqp2',         'dlaqp2',         'claqp2',         'zlaqp2'          ),
    ('slaqps',         'dlaqps',         'claqps',         'zlaqps'          ),
    ('slaqtrs',        'dlaqtrs',        'claqtrs',        'zlaqtrs'         ),
    ('slarcm',         'dlarcm',         'clarcm',         'zlarcm'          ),
    ('slarf',          'dlarf',          'clarf',          'zlarf'           ),  # also does zlarfb, zlarfg, etc.
    ('slarnv',         'dlarnv',         'clarnv',         'zlarnv'          ),
    ('slarnv',         'dlarnv',         'slarnv',         'dlarnv'          ),
    ('slartg',         'dlartg',         'clartg',         'zlartg'          ),
    ('slascl',         'dlascl',         'slascl',         'dlascl'          ),
    ('slascl',         'dlascl',         'clascl',         'zlascl'          ),
    ('slaset',         'dlaset',         'claset',         'zlaset'          ),
    ('slasrt',         'dlasrt',         'slasrt',         'dlasrt'          ),
    ('slaswp',         'dlaswp',         'claswp',         'zlaswp'          ),
    ('slasyf',         'dlasyf',         'clahef',         'zlahef'          ),
    ('slatms',         'dlatms',         'clatms',         'zlatms'          ),
    ('slatrd',         'dlatrd',         'clatrd',         'zlatrd'          ),
    ('slatrs',         'dlatrs',         'clatrs',         'zlatrs'          ),
    ('slauum',         'dlauum',         'clauum',         'zlauum'          ),
    ('slavsy',         'dlavsy',         'clavhe',         'zlavhe'          ),
    ('sorg2r',         'dorg2r',         'cung2r',         'zung2r'          ),
    ('sorgbr',         'dorgbr',         'cungbr',         'zungbr'          ),
    ('sorghr',         'dorghr',         'cunghr',         'zunghr'          ),
    ('sorglq',         'dorglq',         'cunglq',         'zunglq'          ),
    ('sorgql',         'dorgql',         'cungql',         'zungql'          ),
    ('sorgqr',         'dorgqr',         'cungqr',         'zungqr'          ),
    ('sorgtr',         'dorgtr',         'cungtr',         'zungtr'          ),
    ('sorm2r',         'dorm2r',         'cunm2r',         'zunm2r'          ),
    ('sormbr',         'dormbr',         'cunmbr',         'zunmbr'          ),
    ('sormlq',         'dormlq',         'cunmlq',         'zunmlq'          ),
    ('sormql',         'dormql',         'cunmql',         'zunmql'          ),
    ('sormqr',         'dormqr',         'cunmqr',         'zunmqr'          ),
    ('sormrq',         'dormrq',         'cunmrq',         'zunmrq'          ),
    ('sormr2',         'dormr2',         'cunmr2',         'zunmr2'          ),
    ('sormtr',         'dormtr',         'cunmtr',         'zunmtr'          ),
    ('sort01',         'dort01',         'cunt01',         'zunt01'          ),
    ('spack',          'dpack',          'cpack',          'zpack'           ),
    ('splgsy',         'dplgsy',         'cplghe',         'zplghe'          ),
    ('splgsy',         'dplgsy',         'cplgsy',         'zplgsy'          ),
    ('splrnt',         'dplrnt',         'cplrnt',         'zplrnt'          ),
    ('sposv',          'dposv',          'cposv',          'zposv'           ),
    ('spotf2',         'dpotf2',         'cpotf2',         'zpotf2'          ),
    ('spotrf',         'dpotrf',         'cpotrf',         'zpotrf'          ),
    ('spotri',         'dpotri',         'cpotri',         'zpotri'          ),
    ('spotrs',         'dpotrs',         'cpotrs',         'zpotrs'          ),
    ('sqpt01',         'dqpt01',         'cqpt01',         'zqpt01'          ),
    ('sqrt02',         'dqrt02',         'cqrt02',         'zqrt02'          ),
    ('ssbtrd',         'dsbtrd',         'chbtrd',         'zhbtrd'          ),
    ('sshift',         'dshift',         'cshift',         'zshift'          ),
    ('sssssm',         'dssssm',         'cssssm',         'zssssm'          ),
    ('sstebz',         'dstebz',         'sstebz',         'dstebz'          ),
    ('sstedc',         'dstedc',         'cstedc',         'zstedc'          ),
    ('sstedx',         'dstedx',         'cstedx',         'zstedx'          ),
    ('sstedx',         'dstedx',         'sstedx',         'dstedx'          ),
    ('sstegr',         'dstegr',         'cstegr',         'zstegr'          ),
    ('sstein',         'dstein',         'cstein',         'zstein'          ),
    ('sstemr',         'dstemr',         'cstemr',         'zstemr'          ),
    ('ssteqr',         'dsteqr',         'csteqr',         'zsteqr'          ),
    ('ssterf',         'dsterf',         'ssterf',         'dsterf'          ),
    ('ssterm',         'dsterm',         'csterm',         'zsterm'          ),
    ('sstt21',         'dstt21',         'cstt21',         'zstt21'          ),
    ('ssyev',          'dsyev',          'cheev',          'zheev'           ),
    ('ssyevd',         'dsyevd',         'cheevd',         'zheevd'          ),
    ('ssygs2',         'dsygs2',         'chegs2',         'zhegs2'          ),
    ('ssygst',         'dsygst',         'chegst',         'zhegst'          ),
    ('ssygv',          'dsygv',          'chegv',          'zhegv'           ),
    ('ssysv',          'dsysv',          'csysv',          'zsysv'           ),
    ('ssyt21',         'dsyt21',         'chet21',         'zhet21'          ),
    ('ssyt22',         'dsyt22',         'chet22',         'zhet22'          ),
    ('ssytd2',         'dsytd2',         'chetd2',         'zhetd2'          ),
    ('ssytf2',         'dsytf2',         'chetf2',         'zhetf2'          ),
    ('ssytf2',         'dsytf2',         'csytf2',         'zsytf2'          ),
    ('ssytrd',         'dsytrd',         'chetrd',         'zhetrd'          ),
    ('ssytrf',         'dsytrf',         'chetrf',         'zhetrf'          ),
    ('ssysv',          'dsysv',          'chesv',          'zhesv'           ),
    ('ssysv',          'dsysv',          'csysv',          'zsysv'           ),
    ('ssytrf',         'dsytrf',         'csytrf',         'zsytrf'          ),
    ('ssytrs',         'dsytrs',         'chetrs',         'zhetrs'          ),
    ('ssytrs',         'dsytrs',         'csytrs',         'zsytrs'          ),
    ('strevc',         'dtrevc',         'ctrevc',         'ztrevc'          ),
    ('strsmpl',        'dtrsmpl',        'ctrsmpl',        'ztrsmpl'         ),
    ('strtri',         'dtrtri',         'ctrtri',         'ztrtri'          ),
    ('stsmqr',         'dtsmqr',         'ctsmqr',         'ztsmqr'          ),
    ('stsqrt',         'dtsqrt',         'ctsqrt',         'ztsqrt'          ),
    ('ststrf',         'dtstrf',         'ctstrf',         'ztstrf'          ),
]


# ===========================================================================
# Dictionary is keyed on substitution type (mixed, normal, etc.)
subs = {
  # ------------------------------------------------------------
  # replacements applied to mixed precision files.
  'mixed' : [
    # ----- header
    ('ds',                        'zc'                      ),

    # ----- special cases
    ('dcopy',                     'zcopy'                   ),  # before zc
    ('dssysv',                    'zchesv'                   ),  # before zc

    # ----- Mixed precision prefix
    # TODO drop these two -- they are way too general
    ('DS',                        'ZC'                      ),
    ('ds',                        'zc'                      ),

    # ----- Preprocessor
    ('#define PRECISION_d',       '#define PRECISION_z'     ),
    ('#define PRECISION_s',       '#define PRECISION_c'     ),
    ('#undef PRECISION_d',        '#undef PRECISION_z'      ),
    ('#undef PRECISION_s',        '#undef PRECISION_c'      ),
    ('#define REAL',              '#define COMPLEX'         ),  # deprecated
    ('#undef REAL',               '#undef COMPLEX'          ),  # deprecated
    ('#define MAGMA_REAL',        '#define MAGMA_COMPLEX'   ),
    ('#undef MAGMA_REAL',         '#undef MAGMA_COMPLEX'    ),

    # ----- Text
    ('symmetric',                 'hermitian'               ),
    ('symmetric',                 'Hermitian'               ),
    ('orthogonal',                'unitary'                 ),

    # ----- CBLAS
    ('',                          'CBLAS_SADDR'             ),

    # ----- Complex numbers
    ('(double)',                  'cuComplexFloatToDouble'  ),
    ('(float)',                   'cuComplexDoubleToFloat'  ),
    ('',                          'cuCrealf'                ),
    ('',                          'cuCimagf'                ),
    ('',                          'cuCreal'                 ),
    ('',                          'cuCimag'                 ),
    ('',                          'cuConj'                  ),
    ('abs',                       'cuCabs'                  ),
    ('absf',                      'cuCabsf'                 ),

    # ----- Constants
    # see note in "normal" section below about ConjTrans
    ('MagmaTrans',                 'Magma_ConjTrans'         ),

    # ----- BLAS & LAPACK
    ]
    + title( blas_mixed )  # e.g., Dgemm, as in cuBLAS, before lowercase (e.g., for Zdrot)
    + lower( blas_mixed )  # e.g., dgemm
    + upper( blas_mixed )  # e.g., DGEMM
    + [

    # ----- PLASMA / MAGMA data types
    ('double',                    'double2'                 ),
    ('float',                     'float2'                  ),
    ('double',                    'cuDoubleComplex'         ),
    ('float',                     'cuFloatComplex'          ),
    ('double',                    'MKL_Complex16'           ),
    ('float',                     'MKL_Complex8'            ),
    ('magmaFloat_const_ptr',      'magmaFloatComplex_const_ptr' ),  # before magmaDoubleComplex
    ('magmaDouble_const_ptr',     'magmaDoubleComplex_const_ptr'),
    ('magmaFloat_ptr',            'magmaFloatComplex_ptr'   ),
    ('magmaDouble_ptr',           'magmaDoubleComplex_ptr'  ),
    ('double',                    'magmaDoubleComplex'      ),
    ('float',                     'magmaFloatComplex'       ),
    ('DOUBLE PRECISION',          'COMPLEX_16'              ),
    ('SINGLE PRECISION',          'COMPLEX'                 ),
    ('real',                      'complex'                 ),

    # ----- PLASMA / MAGMA functions, alphabetic order
    ('dsaxpy',                    'zcaxpy'                  ),
    ('dslaswp',                   'zclaswp'                 ),
    ('magma_sdgetrs',             'magma_czgetrs'           ),

    # ----- Sparse Stuff
    ('dspgmres',                  'zcpgmres'                ),
    ('dspbicgstab',               'zcpbicgstab'             ),
    ('dsir',                      'zcir'                    ),
    ('dspir',                     'zcpir'                   ),

    # ----- Prefixes
    ('blasf77_d',                 'blasf77_z'               ),
    ('blasf77_s',                 'blasf77_c'               ),
    ('cublasId',                  'cublasIz'                ),
    ('cublasD',                   'cublasZ'                 ),
    ('cublasS',                   'cublasC'                 ),
    ('clblasD',                   'clblasZ'                 ),
    ('clblasS',                   'clblasC'                 ),
    ('lapackf77_d',               'lapackf77_z'             ),
    ('lapackf77_s',               'lapackf77_c'             ),
    ('MAGMA_D',                   'MAGMA_Z'                 ),
    ('MAGMA_S',                   'MAGMA_C'                 ),
    ('magmablas_d',               'magmablas_z'             ),
    ('magmablas_s',               'magmablas_c'             ),
    ('magma_d',                   'magma_z'                 ),
    ('magma_s',                   'magma_c'                 ),
    ('magma_get_d',               'magma_get_z'             ),
    ('magma_get_s',               'magma_get_c'             ),
    ('magmasparse_ds',            'magmasparse_zc'          ),

    # magma_ceildiv -> magma_seildiv, so revert
    ('magma_ceildiv',             'magma_seildiv'           ),
    ('magma_copy',                'magma_sopy'              ),
    
  ], # end mixed

  # ------------------------------------------------------------
  # replacements applied to most files.
  'normal' : [
    # ----- header
    ('s',              'd',              'c',              'z'               ),

    # ----- Preprocessor
    ('#define PRECISION_s',  '#define PRECISION_d',  '#define PRECISION_c',   '#define PRECISION_z'  ),
    ('#undef PRECISION_s',   '#undef PRECISION_d',   '#undef PRECISION_c',    '#undef PRECISION_z'   ),
    ('#define REAL',         '#define REAL',         '#define COMPLEX',       '#define COMPLEX'      ),  # deprecated
    ('#undef REAL',          '#undef REAL',          '#undef COMPLEX',        '#undef COMPLEX'       ),  # deprecated
    ('#define MAGMA_REAL',   '#define MAGMA_REAL',   '#define MAGMA_COMPLEX', '#define MAGMA_COMPLEX'),
    ('#undef MAGMA_REAL',    '#undef MAGMA_REAL',    '#undef MAGMA_COMPLEX',  '#undef MAGMA_COMPLEX' ),
    ('#define SINGLE',       '#define DOUBLE',       '#define SINGLE',        '#define DOUBLE'       ),  # deprecated
    ('#undef SINGLE',        '#undef DOUBLE',        '#undef SINGLE',         '#undef DOUBLE'        ),  # deprecated
    ('#define MAGMA_SINGLE', '#define MAGMA_DOUBLE', '#define MAGMA_SINGLE',  '#define MAGMA_DOUBLE' ),
    ('#undef MAGMA_SINGLE',  '#undef MAGMA_DOUBLE',  '#undef MAGMA_SINGLE',   '#undef MAGMA_DOUBLE'  ),

    # ----- Text
    ('symmetric',      'symmetric',      'hermitian',      'hermitian'       ),
    ('symmetric',      'symmetric',      'Hermitian',      'Hermitian'       ),
    ('orthogonal',     'orthogonal',     'unitary',        'unitary'         ),
    ('%f',             '%lf',            '%f',             '%lf'             ),  # for scanf

    # ----- CBLAS
    ('',               '',               'CBLAS_SADDR',    'CBLAS_SADDR'     ),

    # ----- Complex numbers
    # \b regexp here avoids conjugate -> conjfugate, and fabs -> fabsf -> fabsff.
    # Note r for raw string literals, otherwise \b is a bell character.
    # The \b is deleted from replacement strings.
    # conj() and fabs() are overloaded in MAGMA, so don't need substitution.
    #(r'',             r'',              r'\bconjf\b',     r'\bconj\b'        ),
    #(r'\bfabsf\b',    r'\bfabs\b',      r'\bfabsf\b',     r'\bfabs\b'        ),
    #(r'\bfabsf\b',    r'\bfabs\b',      r'\bcabsf\b',     r'\bcabs\b'        ),
    ('',               '',               'cuCrealf',       'cuCreal'         ),
    ('',               '',               'cuCimagf',       'cuCimag'         ),
    ('',               '',               'cuConjf',        'cuConj'          ),
    ('fabsf',         r'\bfabs\b',       'cuCabsf',        'cuCabs'          ),

    # ----- Constants
    # Do not convert ConjTrans to Trans, since in most cases ConjTrans
    # must be a valid option to real-precision functions.
    # E.g., dgemm( ConjTrans, ConjTrans, ... ) should be valid; if ConjTrans is
    # converted, then dgemm will have 2 Trans cases and no ConjTrans case.
    # Only for zlarfb and zunm*, convert it using special Magma_ConjTrans alias.
    ('MagmaTrans',     'MagmaTrans',     'Magma_ConjTrans', 'Magma_ConjTrans'  ),

    # ----- BLAS & LAPACK
    ]
    + title( blas )    # e.g., Dgemm, as in cuBLAS, before lowercase (e.g., for Zdrot)
    + lower( blas )    # e.g., dgemm
    + upper( blas )    # e.g., DGEMM
    + lower( lapack )  # e.g., dgetrf
    + upper( lapack )  # e.g., DGETRF
    + [

    # ----- PLASMA / MAGMA data types
    ('float',                'double',               'std::complex<float>',  'std::complex<double>'),
    ('REAL',                 'DOUBLE PRECISION',     'REAL',                 'DOUBLE PRECISION'    ),
    ('real',                 'double precision',     'real',                 'double precision'    ),  # before double
    ('float',                'double',               'float _Complex',       'double _Complex'     ),
    ('float',                'double',               'cuFloatComplex',       'cuDoubleComplex'     ),
    ('CUDA_R_32F',           'CUDA_R_64F',           'CUDA_C_32F',           'CUDA_C_64F'          ),
    ('float',                'double',               'MKL_Complex8',         'MKL_Complex16'       ),
    ('magmaFloat_const_ptr', 'magmaDouble_const_ptr','magmaFloatComplex_const_ptr', 'magmaDoubleComplex_const_ptr'),  # before magmaDoubleComplex
    ('magmaFloat_const_ptr', 'magmaDouble_const_ptr','magmaFloat_const_ptr',        'magmaDouble_const_ptr'       ),  # before magmaDoubleComplex
    ('magmaFloat_ptr',       'magmaDouble_ptr',      'magmaFloatComplex_ptr',       'magmaDoubleComplex_ptr'      ),  # before magmaDoubleComplex
    ('magmaFloat_ptr',       'magmaDouble_ptr',      'magmaFloat_ptr',              'magmaDouble_ptr'             ),  # before magmaDoubleComplex
    ('float',                'double',               'magmaFloatComplex',    'magmaDoubleComplex'  ),
    ('float',                'double',               'PLASMA_Complex32_t',   'PLASMA_Complex64_t'  ),
    ('PlasmaRealFloat',      'PlasmaRealDouble',     'PlasmaComplexFloat',   'PlasmaComplexDouble' ),
    ('real',                 'double precision',     'complex',              'complex\*16'         ),
    ('REAL',                 'DOUBLE PRECISION',     'COMPLEX',              'COMPLEX_16'          ),
    ('REAL',                 'DOUBLE PRECISION',     'COMPLEX',              'COMPLEX\*16'         ),
    ('sizeof_real',          'sizeof_double',        'sizeof_complex',       'sizeof_complex_16'   ),  # before complex
    ('real',                 'real',                 'complex',              'complex'             ),
    ('float',                'double',               'float2',               'double2'             ),
    ('float',                'double',               'float',                'double'              ),

    # ----- PLASMA / MAGMA functions, alphabetic order
    ('bsy2trc',        'bsy2trc',        'bhe2trc',        'bhe2trc'         ),
    ('magma_ssqrt',    'magma_dsqrt',    'magma_csqrt',    'magma_zsqrt'     ),
    ('magma_ssqrt',    'magma_dsqrt',    'magma_ssqrt',    'magma_dsqrt'     ),
    ('SAUXILIARY',     'DAUXILIARY',     'CAUXILIARY',     'ZAUXILIARY'      ),
    ('sauxiliary',     'dauxiliary',     'cauxiliary',     'zauxiliary'      ),
    ('sb2st',          'sb2st',          'hb2st',          'hb2st'           ),
    ('sbcyclic',       'dbcyclic',       'cbcyclic',       'zbcyclic'        ),
    ('SBULGE',         'DBULGE',         'CBULGE',         'ZBULGE'          ),
    ('sbulge',         'dbulge',         'cbulge',         'zbulge'          ),
    ('scheck',         'dcheck',         'ccheck',         'zcheck'          ),
    ('SCODELETS',      'DCODELETS',      'CCODELETS',      'ZCODELETS'       ),
    ('sgeadd',         'dgeadd',         'cgeadd',         'zgeadd'          ),
    ('sgeam',          'dgeam',          'cgeam',          'zgeam'           ),
    ('sgecfi',         'dgecfi',         'cgecfi',         'zgecfi'          ),
    ('SGERBT',         'DGERBT',         'CGERBT',         'ZGERBT'          ),
    ('sgerbt',         'dgerbt',         'cgerbt',         'zgerbt'          ),
    ('sgetmatrix',     'dgetmatrix',     'cgetmatrix',     'zgetmatrix'      ),
    ('sgetmatrix',     'dgetmatrix',     'sgetmatrix',     'dgetmatrix'      ),
    ('sgetrl',         'dgetrl',         'cgetrl',         'zgetrl'          ),
    ('sgetvector',     'dgetvector',     'cgetvector',     'zgetvector'      ),
    ('sgetvector',     'dgetvector',     'sgetvector',     'dgetvector'      ),
    ('slocality',      'dlocality',      'clocality',      'zlocality'       ),
    ('smalloc',        'dmalloc',        'cmalloc',        'zmalloc'         ),
    ('smalloc',        'dmalloc',        'smalloc',        'dmalloc'         ),
    ('smove',          'dmove',          'smove',          'dmove'           ),
    ('spanel_to_q',    'dpanel_to_q',    'cpanel_to_q',    'zpanel_to_q'     ),
    ('spermute',       'dpermute',       'cpermute',       'zpermute'        ),
    ('SPRBT',          'DPRBT',          'CPRBT',          'ZPRBT'           ),
    ('sprbt',          'dprbt',          'cprbt',          'zprbt'           ),
    ('SPRINT',         'DPRINT',         'CPRINT',         'ZPRINT'          ),
    ('sprint',         'dprint',         'cprint',         'zprint'          ),
    ('sprint',         'dprint',         'sprint',         'dprint'          ),
    ('sprofiling',     'dprofiling',     'cprofiling',     'zprofiling'      ),
    ('sq_to_panel',    'dq_to_panel',    'cq_to_panel',    'zq_to_panel'     ),
    ('sset',           'dset',           'cset',           'zset'            ),
    ('ssign',          'dsign',          'ssign',          'dsign'           ),
    ('SSIZE',          'DSIZE',          'CSIZE',          'ZSIZE'           ),
    ('ssplit',         'dsplit',         'csplit',         'zsplit'          ),
    ('ssyrbt',         'dsyrbt',         'cherbt',         'zherbt'          ),
    ('stile',          'dtile',          'ctile',          'ztile'           ),
    ('STRANSPOSE',     'DTRANSPOSE',     'CTRANSPOSE',     'ZTRANSPOSE'      ),
    ('stranspose',     'dtranspose',     'ctranspose_conj','ztranspose_conj' ),  # before ztranspose
    ('stranspose',     'dtranspose',     'ctranspose',     'ztranspose'      ),
    ('strdtype',       'dtrdtype',       'ctrdtype',       'ztrdtype'        ),
    ('sy2sb',          'sy2sb',          'he2hb',          'he2hb'           ),
    ('szero',          'dzero',          'czero',          'zzero'           ),

    # ----- special cases for d -> s that need complex (e.g., testing_dgeev)
    # c/z precisions are effectively disabled for these rules
    ('caxpy',             'zaxpy',              'cccccccc', 'zzzzzzzz' ),
    ('clange',            'zlange',             'cccccccc', 'zzzzzzzz' ),
    ('cuFloatComplex',    'cuDoubleComplex',    'cccccccc', 'zzzzzzzz' ),
    ('magmaFloatComplex', 'magmaDoubleComplex', 'cccccccc', 'zzzzzzzz' ),
    ('MAGMA_C',           'MAGMA_Z',            'cccccccc', 'zzzzzzzz' ),
    ('scnrm2',            'dznrm2',             'cccccccc', 'zzzzzzzz' ),
    ('magma_c',           'magma_z',            'ccccccc',  'zzzzzzz'  ),

    # ----- SPARSE BLAS
    ('cusparseS',      'cusparseD',      'cusparseC',      'cusparseZ'       ),
    ('sgeaxpy',        'dgeaxpy',        'cgeaxpy',        'zgeaxpy'         ),
    ('sgedense',       'dgedense',       'cgedense',       'zgedense'        ),
    ('sgecsr',         'dgecsr',         'cgecsr',         'zgecsr'          ),
    ('sgecsrmv',       'dgecsrmv',       'cgecsrmv',       'zgecsrmv'        ),
    ('smgecsrmv',      'dmgecsrmv',      'cmgecsrmv',      'zmgecsrmv'       ),
    ('sgeellmv',       'dgeellmv',       'cgeellmv',       'zgeellmv'        ),
    ('smgeellmv',      'dmgeellmv',      'cmgeellmv',      'zmgeellmv'       ),
    ('sgeelltmv',      'dgeelltmv',      'cgeelltmv',      'zgeelltmv'       ),
    ('smgeelltmv',     'dmgeelltmv',     'cmgeelltmv',     'zmgeelltmv'      ),
    ('sgeellrtmv',     'dgeellrtmv',     'cgeellrtmv',     'zgeellrtmv'      ),
    ('sgesellcm',      'dgesellcm',      'cgesellcm',      'zgesellcm'       ),
    ('smgesellcm',     'dmgesellcm',     'cmgesellcm',     'zmgesellcm'      ),
    ('smdot',          'dmdot',          'cmdot',          'zmdot'           ),
    ('smzdotc',        'dmzdotc',        'cmzdotc',        'zmzdotc'         ),
    ('smt',            'dmt',            'cmt',            'zmt'             ),
    ('spipelined',     'dpipelined',     'cpipelined',     'zpipelined'      ),
    ('mkl_scsrmv',     'mkl_dcsrmv',     'mkl_ccsrmv',     'mkl_zcsrmv'      ),
    ('mkl_scsrmm',     'mkl_dcsrmm',     'mkl_ccsrmm',     'mkl_zcsrmm'      ),
    ('mkl_sbsrmv',     'mkl_dbsrmv',     'mkl_cbsrmv',     'mkl_zbsrmv'      ),
    ('scsrgemv',       'dcsrgemv',       'ccsrgemv',       'zcsrgemv'        ),
    ('SCSRGEMV',       'DCSRGEMV',       'CCSRGEMV',       'ZCSRGEMV'        ),
    ('mic_scsrmm',     'mic_dcsrmm',     'mic_ccsrmm',     'mic_zcsrmm'      ),
    ('mic_sbsrmv',     'mic_dbsrmv',     'mic_cbsrmv',     'mic_zbsrmv'      ),
    ('smerge',         'dmerge',         'cmerge',         'zmerge'          ),
    ('sbcsr',          'dbcsr',          'cbcsr',          'zbcsr'           ),
    ('siterilu',       'diterilu',       'citerilu',       'ziterilu'        ),
    ('siteric',        'diteric',        'citeric',        'ziteric'         ),
    ('sdummy',         'ddummy',         'cdummy',         'zdummy'          ),
    ('stest',          'dtest',          'ctest',          'ztest'           ),
    ('sgeisai',        'dgeisai',        'cgeisai',        'zgeisai'         ),
    ('ssyisai',        'dsyisai',        'csyisai',        'zsyisai'         ),
    ('silu',           'dilu',           'cilu',           'zilu'            ),
    ('sgeblock',       'dgeblock',       'cilugeblock',    'zgeblock'        ),
    ('sge3pt',         'dge3pt',         'cge3pt',         'zge3pt'          ),    
    ('sgecscsyncfreetrsm',  'dgecscsyncfreetrsm',  'cgecscsyncfreetrsm',  'zgecscsyncfreetrsm'),   


    # ----- SPARSE Iterative Solvers
    ('scg',            'dcg',            'ccg',            'zcg'             ),
    ('slsqr',          'dlsqr',          'clsqr',          'zlsqr'           ),
    ('sgmres',         'dgmres',         'cgmres',         'zgmres'          ),
    ('sbicg',          'dbicg',          'cbicg',          'zbicg'           ),
    ('sqmr',           'dqmr',           'cqmr',           'zqmr'            ),
    ('spqmr',          'dpqmr',          'cpqmr',          'zpqmr'           ),
    ('stfqmr',         'dtfqmr',         'ctfqmr',         'ztfqmr'          ),
    ('sptfqmr',        'dptfqmr',        'cptfqmr',        'zptfqmr'         ),
    ('spcg',           'dpcg',           'cpcg',           'zpcg'            ),
    ('sbpcg',          'dbpcg',          'cbpcg',          'zbpcg'           ),
    ('spbicg',         'dpbicg',         'cpbicg',         'zpbicg'          ),
    ('spgmres',        'dpgmres',        'cpgmres',        'zpgmres'         ),
    ('sfgmres',        'dfgmres',        'cfgmres',        'zfgmres'         ),
    ('sbfgmres',       'dbfgmres',       'cbfgmres',       'zbfgmres'        ),
    ('sidr',           'didr',           'cidr',           'zidr'            ),
    ('spidr',          'dpidr',          'cpidr',          'zpidr'           ),
    ('sp1gmres',       'dp1gmres',       'cp1gmres',       'zp1gmres'        ),
    ('sjacobi',        'djacobi',        'cjacobi',        'zjacobi'         ),
    ('sftjacobi',      'dftjacobi',      'cftjacobi',      'zftjacobi'       ),
    ('siterref',       'diterref',       'citerref',       'ziterref'        ),
    ('silu',           'dilu',           'cilu',           'zilu'            ),
    ('sailu',          'dailu',          'cailu',          'zailu'           ),
    ('scuilu',         'dcuilu',         'ccuilu',         'zcuilu'          ),
    ('scumilu',        'dcumilu',        'ccumilu',        'zcumilu'         ),
    ('sbailu',         'dbailu',         'cbailu',         'zbailu'          ),
    ('spastix',        'dpastix',        'cpastix',        'zpastix'         ),
    ('slobpcg',        'dlobpcg',        'clobpcg',        'zlobpcg'         ),
    ('sbajac',         'dbajac',         'cbajac',         'zbajac'          ),
    ('sbaiter',        'dbaiter',        'cbaiter',        'zbaiter'         ),
    ('sbombard',       'dbombard',       'cbombard',       'zbombard'        ),
    ('scustom',        'dcustom',        'ccustom',        'zcustom'         ),
    ('sparilu',        'dparilu',        'cparilu',        'zparilu'         ),
    ('sparic',         'dparic',         'cparic',         'zparic'          ),

    # ----- SPARSE Iterative Eigensolvers
    ('slobpcg',        'dlobpcg',        'clobpcg',        'zlobpcg'         ),

    # ----- SPARSE Big data analytics
    ('sjaccard',       'djaccard',       'cjaccard',       'zjaccard'        ),

    # ----- SPARSE direct solver interface (PARDISO)
    ('spardiso',       'dpardiso',       'cpardiso',       'zpardiso'        ),

    # ----- SPARSE auxiliary tools
    ('matrix_s',       'matrix_d',       'matrix_c',       'matrix_z'        ),
    ('svjacobi',       'dvjacobi',       'cvjacobi',       'zvjacobi'        ),
    ('s_csr2array',    'd_csr2array',    'c_csr2array',    'z_csr2array'     ),
    ('s_array2csr',    'd_array2csr',    'c_array2csr',    'z_array2csr'     ),
    ('read_s_csr',     'read_d_csr',     'read_c_csr',     'read_z_csr'      ),
    ('print_s_csr',    'print_d_csr',    'print_c_csr',    'print_z_csr'     ),
    ('write_s_csr',    'write_d_csr',    'write_c_csr',    'write_z_csr'     ),
    ('s_transpose',    'd_transpose',    'c_transpose',    'z_transpose'     ),
    ('SPARSE_S_H',     'SPARSE_D_H',     'SPARSE_C_H',     'SPARSE_Z_H'      ),
    ('_TYPES_S_H',     '_TYPES_D_H',     '_TYPES_C_H',     '_TYPES_Z_H'      ),
    ('sresidual',      'dresidual',      'cresidual',      'zresidual'       ),
    ('scompact',       'dcompact',       'ccompact',       'zcompact'        ),
    ('sortho',         'dortho',         'cortho',         'zortho'          ),

    # ----- SPARSE runfiles
    ('run_s',          'run_d',          'run_c',          'run_z'           ),


    # ----- Xeon Phi (MIC) specific, alphabetic order unless otherwise required
    ('SREG_WIDTH',                  'DREG_WIDTH',                  'CREG_WIDTH',                  'ZREG_WIDTH' ),
    ('_MM512_I32LOEXTSCATTER_PPS',  '_MM512_I32LOEXTSCATTER_PPD',  '_MM512_I32LOEXTSCATTER_PPC',  '_MM512_I32LOEXTSCATTER_PPZ' ),
    ('_MM512_LOAD_PPS',             '_MM512_LOAD_PPD',             '_MM512_LOAD_PPC',             '_MM512_LOAD_PPZ' ),
    ('_MM512_STORE_PPS',            '_MM512_STORE_PPD',            '_MM512_STORE_PPC',            '_MM512_STORE_PPZ' ),
    ('_MM_DOWNCONV_PS_NONE',        '_MM_DOWNCONV_PD_NONE',        '_MM_DOWNCONV_PC_NONE',        '_MM_DOWNCONV_PZ_NONE' ),
    ('__M512S',                     '__M512D',                     '__M512C',                     '__M512Z' ),
    ('somatcopy',                   'domatcopy',                   'comatcopy',                   'zomatcopy'),

    # ----- Prefixes
    # Most routines have already been renamed by above BLAS/LAPACK rules.
    # Functions where real == complex name can be handled here;
    # if real != complex name, it must be handled above.
    ('blasf77_s',      'blasf77_d',      'blasf77_c',      'blasf77_z'       ),
    ('blasf77_s',      'blasf77_d',      'blasf77_s',      'blasf77_d'       ),
    ('BLAS_S',         'BLAS_D',         'BLAS_C',         'BLAS_Z'          ),
    ('BLAS_s',         'BLAS_d',         'BLAS_c',         'BLAS_z'          ),
    ('BLAS_s',         'BLAS_d',         'BLAS_s',         'BLAS_d'          ),
    ('blas_is',        'blas_id',        'blas_ic',        'blas_iz'         ),
    ('blas_s',         'blas_d',         'blas_c',         'blas_z'          ),
    ('cl_ps',          'cl_pd',          'cl_pc',          'cl_pz'           ),
    ('cl_s',           'cl_d',           'cl_c',           'cl_z'            ),
    ('compute_s',      'compute_d',      'compute_c',      'compute_z'       ),
    ('control_s',      'control_d',      'control_c',      'control_z'       ),
    ('coreblas_s',     'coreblas_d',     'coreblas_c',     'coreblas_z'      ),
    ('core_ssb',       'core_dsb',       'core_chb',       'core_zhb'        ),
    ('CORE_S',         'CORE_D',         'CORE_C',         'CORE_Z'          ),
    ('CORE_s',         'CORE_d',         'CORE_c',         'CORE_z'          ),
    ('core_s',         'core_d',         'core_c',         'core_z'          ),
    ('CORE_s',         'CORE_d',         'CORE_s',         'CORE_d'          ),
    ('cpu_gpu_s',      'cpu_gpu_d',      'cpu_gpu_c',      'cpu_gpu_z'       ),
    ('cublasIs',       'cublasId',       'cublasIs',       'cublasId'        ),
    ('cublasIs',       'cublasId',       'cublasIc',       'cublasIz'        ),
    ('cublasS',        'cublasD',        'cublasC',        'cublasZ'         ),
    ('clblasiS',       'clblasiD',       'clblasiC',       'clblasiZ'        ),
    ('clblasS',        'clblasD',        'clblasC',        'clblasZ'         ),
    ('example_s',      'example_d',      'example_c',      'example_z'       ),
    ('ipt_s',          'ipt_d',          'ipt_c',          'ipt_z'           ),
    ('LAPACKE_s',      'LAPACKE_d',      'LAPACKE_c',      'LAPACKE_z'       ),
    ('lapackf77_s',    'lapackf77_d',    'lapackf77_c',    'lapackf77_z'     ),
    ('lapackf77_s',    'lapackf77_d',    'lapackf77_s',    'lapackf77_d'     ),
    ('lapack_s',       'lapack_d',       'lapack_c',       'lapack_z'        ),
    ('lapack_s',       'lapack_d',       'lapack_s',       'lapack_d'        ),
    ('MAGMABLAS_S',    'MAGMABLAS_D',    'MAGMABLAS_C',    'MAGMABLAS_Z'     ),
    ('magmablas_s',    'magmablas_d',    'magmablas_c',    'magmablas_z'     ),
    ('magmablas_s',    'magmablas_d',    'magmablas_s',    'magmablas_d'     ),
    ('magmaf_s',       'magmaf_d',       'magmaf_c',       'magmaf_z'        ),
    ('magma_get_s',    'magma_get_d',    'magma_get_c',    'magma_get_z'     ),
    ('magma_ps',       'magma_pd',       'magma_pc',       'magma_pz'        ),
    ('magma_ssb',      'magma_dsb',      'magma_chb',      'magma_zhb'       ),
    ('MAGMA_S',        'MAGMA_D',        'MAGMA_C',        'MAGMA_Z'         ),
    ('MAGMA_s',        'MAGMA_d',        'MAGMA_c',        'MAGMA_z'         ),
    ('magma_s',        'magma_d',        'magma_c',        'magma_z'         ),
    ('magma_s',        'magma_d',        'magma_sc',       'magma_dz'        ),
    ('magma_s',        'magma_d',        'magma_s',        'magma_d'         ),
    ('magmasparse_s',  'magmasparse_d',  'magmasparse_c',  'magmasparse_z'   ),
    ('plasma_ps',      'plasma_pd',      'plasma_pc',      'plasma_pz'       ),
    ('PLASMA_S',       'PLASMA_D',       'PLASMA_C',       'PLASMA_Z'        ),
    ('PLASMA_sor',     'PLASMA_dor',     'PLASMA_cun',     'PLASMA_zun'      ),
    ('PLASMA_s',       'PLASMA_d',       'PLASMA_c',       'PLASMA_z'        ),
    ('plasma_s',       'plasma_d',       'plasma_c',       'plasma_z'        ),
    ('PROFILE_S',      'PROFILE_D',      'PROFILE_C',      'PROFILE_Z'       ),
    ('profile_s',      'profile_d',      'profile_c',      'profile_z'       ),
    ('SCHED_s',        'SCHED_d',        'SCHED_c',        'SCHED_z'         ),
    ('starpu_s',       'starpu_d',       'starpu_c',       'starpu_z'        ),
    ('testing_ds',     'testing_ds',     'testing_zc',     'testing_zc'      ),
    ('testing_s',      'testing_d',      'testing_c',      'testing_z'       ),
    ('time_s',         'time_d',         'time_c',         'time_z'          ),
    ('WRAPPER_S',      'WRAPPER_D',      'WRAPPER_C',      'WRAPPER_Z'       ),
    ('wrapper_s',      'wrapper_d',      'wrapper_c',      'wrapper_z'       ),
    ('Workspace_s',    'Workspace_d',    'Workspace_c',    'Workspace_z'     ),
    ('workspace_s',    'workspace_d',    'workspace_c',    'workspace_z'     ),

    # magma_[get_]d -> magma_[get_]s, so revert _sevice to _device, etc.
    ('_device',        '_sevice',        '_device',        '_sevice'         ),
    ('magma_devptr_t', 'magma_sevptr_t', 'magma_devptr_t', 'magma_sevptr_t'  ),
    ('magma_diag',     'magma_siag',     'magma_diag',     'magma_siag'      ),
    ('magma_direct',   'magma_sirect',   'magma_direct',   'magma_sirect'    ),
    ('lapack_diag',    'lapack_siag',    'lapack_diag',    'lapack_siag'     ),
    ('lapack_direct',  'lapack_sirect',  'lapack_direct',  'lapack_sirect'   ),
    ('magma_copy',     'magma_sopy',     'magma_copy',     'magma_sopy'      ),
    
  ], # end normal

  # ------------------------------------------------------------
  # replacements applied to Fortran files.
  'fortran' : [
    # ----- header                                                             
    ('s',              'd',              'c',              'z'               ),

    # ----- Text
    ('symmetric',      'symmetric',      'hermitian',      'hermitian'       ),
    ('symmetric',      'symmetric',      'Hermitian',      'Hermitian'       ),
    ('orthogonal',     'orthogonal',     'unitary',        'unitary'         ),
    
    # ----- data types    
    ('REAL',           'DOUBLE PRECISION', 'REAL',         'DOUBLE PRECISION'),
    ('real',           'double precision', 'real',         'double precision'),
    ('REAL',           'DOUBLE PRECISION', 'COMPLEX',      'COMPLEX\*16'     ),
    ('real',           'double precision', 'complex',      'complex\*16'     ),
    ('real',           'double',           'complex',      'complex16'       ),
    ('c_float',        'c_double',         'c_float_complex', 'c_double_complex'),
    ('real',           'real',             'complex',      'complex'         ),
    
    # ----- constants                        
    ('\.0E',           '\.0D',             '\.0E',         '\.0D'            ),
    
    # ----- BLAS & LAPACK
    ]
    + lower( blas )    # e.g., dgemm
    + upper( blas )    # e.g., DGEMM
    + lower( lapack )  # e.g., dgetrf
    + upper( lapack )  # e.g., DGETRF
    + [
    
    # ----- Prefixes
    ('magma_s',        'magma_d',        'magma_c',        'magma_z'         ),
    ('magma2_s',       'magma2_d',       'magma2_c',       'magma2_z'        ),
  ], # end fortran
  
  # ------------------------------------------------------------
  # replacements applied for profiling with tau
  'tracing' :[
    # ----- Special line indicating column types
    ['plain', 'tau'],

    # ----- Replacements
    ('(\w+\*?)\s+(\w+)\s*\(([a-z* ,A-Z_0-9]*)\)\s*{\s+(.*)\s*#pragma tracing_start\s+(.*)\s+#pragma tracing_end\s+(.*)\s+}',
      r'\1 \2(\3){\n\4tau("\2");\5tau();\6}'),
    ('\.c','.c.tau'),
  ],
};
