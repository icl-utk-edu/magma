#include <assert.h>
#include <stdio.h>

#include "magma_types.h"

// =============================================================================
/// @addtogroup magma_const
/// Convert LAPACK character constants to MAGMA constants.
/// This is a one-to-many mapping, requiring multiple translators
/// (e.g., "N" can be NoTrans or NonUnit or NoVec).
/// Matching is case-insensitive.
/// @{

// These functions and cases are in the same order as the constants are
// declared in magma_types.h

/******************************************************************************/
/// @retval MagmaFalse if lapack_char = 'N'
/// @retval MagmaTrue  if lapack_char = 'Y'
extern "C"
magma_bool_t   magma_bool_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return MagmaFalse;
        case 'Y': case 'y': return MagmaTrue;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaFalse;
    }
}

/******************************************************************************/
/// @retval MagmaRowMajor if lapack_char = 'R'
/// @retval MagmaColMajor if lapack_char = 'C'
extern "C"
magma_order_t  magma_order_const ( char lapack_char )
{
    switch( lapack_char ) {
        case 'R': case 'r': return MagmaRowMajor;
        case 'C': case 'c': return MagmaColMajor;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaRowMajor;
    }
}

/******************************************************************************/
/// @retval MagmaNoTrans   if lapack_char = 'N'
/// @retval MagmaTrans     if lapack_char = 'T'
/// @retval MagmaConjTrans if lapack_char = 'C'
extern "C"
magma_trans_t  magma_trans_const ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return MagmaNoTrans;
        case 'T': case 't': return MagmaTrans;
        case 'C': case 'c': return MagmaConjTrans;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaNoTrans;
    }
}

/******************************************************************************/
/// @retval MagmaUpper if lapack_char = 'U'
/// @retval MagmaLower if lapack_char = 'L'
/// @retval MagmaFull  otherwise
extern "C"
magma_uplo_t   magma_uplo_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'U': case 'u': return MagmaUpper;
        case 'L': case 'l': return MagmaLower;
        default:            return MagmaFull;        // see laset
    }
}

/******************************************************************************/
/// @retval MagmaNonUnit if lapack_char = 'N'
/// @retval MagmaUnit    if lapack_char = 'U'
extern "C"
magma_diag_t   magma_diag_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return MagmaNonUnit;
        case 'U': case 'u': return MagmaUnit;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaNonUnit;
    }
}

/******************************************************************************/
/// @retval MagmaLeft      if lapack_char = 'L'
/// @retval MagmaRight     if lapack_char = 'R'
/// @retval MagmaBothSides if lapack_char = 'B'
extern "C"
magma_side_t   magma_side_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'L': case 'l': return MagmaLeft;
        case 'R': case 'r': return MagmaRight;
        case 'B': case 'b': return MagmaBothSides;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaLeft;
    }
}

/******************************************************************************/
/// @retval MagmaOneNorm       if lapack_char = '1' or 'O'
/// @retval MagmaTwoNorm       if lapack_char = '2'
/// @retval MagmaFrobeniusNorm if lapack_char = 'F' or 'E'
/// @retval MagmaInfNorm       if lapack_char = 'I'
/// @retval MagmaMaxNorm       if lapack_char = 'M'
extern "C"
magma_norm_t   magma_norm_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'O': case 'o': case '1': return MagmaOneNorm;
        case '2':           return MagmaTwoNorm;
        case 'F': case 'f': case 'E': case 'e': return MagmaFrobeniusNorm;
        case 'I': case 'i': return MagmaInfNorm;
        case 'M': case 'm': return MagmaMaxNorm;
        // MagmaRealOneNorm
        // MagmaRealInfNorm
        // MagmaRealMaxNorm
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaOneNorm;
    }
}

/******************************************************************************/
/// @retval MagmaDistUniform   if lapack_char = 'U'
/// @retval MagmaDistSymmetric if lapack_char = 'S'
/// @retval MagmaDistNormal    if lapack_char = 'N'
extern "C"
magma_dist_t   magma_dist_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'U': case 'u': return MagmaDistUniform;
        case 'S': case 's': return MagmaDistSymmetric;
        case 'N': case 'n': return MagmaDistNormal;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaDistUniform;
    }
}

/******************************************************************************/
/// @retval MagmaHermGeev   if lapack_char = 'H'
/// @retval MagmaHermPoev   if lapack_char = 'P'
/// @retval MagmaNonsymPosv if lapack_char = 'N'
/// @retval MagmaSymPosv    if lapack_char = 'S'
extern "C"
magma_sym_t    magma_sym_const   ( char lapack_char )
{
    switch( lapack_char ) {
        case 'H': case 'h': return MagmaHermGeev;
        case 'P': case 'p': return MagmaHermPoev;
        case 'N': case 'n': return MagmaNonsymPosv;
        case 'S': case 's': return MagmaSymPosv;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaHermGeev;
    }
}

/******************************************************************************/
/// @retval MagmaNoPacking     if lapack_char = 'N'
/// @retval MagmaPackSubdiag   if lapack_char = 'U'
/// @retval MagmaPackSupdiag   if lapack_char = 'L'
/// @retval MagmaPackColumn    if lapack_char = 'C'
/// @retval MagmaPackRow       if lapack_char = 'R'
/// @retval MagmaPackLowerBand if lapack_char = 'B'
/// @retval MagmaPackUpeprBand if lapack_char = 'Q'
/// @retval MagmaPackAll       if lapack_char = 'Z'
extern "C"
magma_pack_t   magma_pack_const  ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return MagmaNoPacking;
        case 'U': case 'u': return MagmaPackSubdiag;
        case 'L': case 'l': return MagmaPackSupdiag;
        case 'C': case 'c': return MagmaPackColumn;
        case 'R': case 'r': return MagmaPackRow;
        case 'B': case 'b': return MagmaPackLowerBand;
        case 'Q': case 'q': return MagmaPackUpeprBand;
        case 'Z': case 'z': return MagmaPackAll;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaNoPacking;
    }
}

/******************************************************************************/
/// @retval MagmaNoVec        if lapack_char = 'N'
/// @retval MagmaVec          if lapack_char = 'V'
/// @retval MagmaIVec         if lapack_char = 'I'
/// @retval MagmaAllVec       if lapack_char = 'A'
/// @retval MagmaSomeVec      if lapack_char = 'S'
/// @retval MagmaOverwriteVec if lapack_char = 'O'
extern "C"
magma_vec_t    magma_vec_const   ( char lapack_char )
{
    switch( lapack_char ) {
        case 'N': case 'n': return MagmaNoVec;
        case 'V': case 'v': return MagmaVec;
        case 'I': case 'i': return MagmaIVec;
        case 'A': case 'a': return MagmaAllVec;
        case 'S': case 's': return MagmaSomeVec;
        case 'O': case 'o': return MagmaOverwriteVec;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaNoVec;
    }
}

/******************************************************************************/
/// @retval MagmaRangeAll if lapack_char = 'A'
/// @retval MagmaRangeV   if lapack_char = 'V'
/// @retval MagmaRangeI   if lapack_char = 'I'
extern "C"
magma_range_t  magma_range_const ( char lapack_char )
{
    switch( lapack_char ) {
        case 'A': case 'a': return MagmaRangeAll;
        case 'V': case 'v': return MagmaRangeV;
        case 'I': case 'i': return MagmaRangeI;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaRangeAll;
    }
}

/******************************************************************************/
/// @retval MagmaQ if lapack_char = 'Q'
/// @retval MagmaP if lapack_char = 'P'
extern "C"
magma_vect_t magma_vect_const( char lapack_char )
{
    switch( lapack_char ) {
        case 'Q': case 'q': return MagmaQ;
        case 'P': case 'p': return MagmaP;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaQ;
    }
}

/******************************************************************************/
/// @retval MagmaForward  if lapack_char = 'F'
/// @retval MagmaBackward if lapack_char = 'B'
extern "C"
magma_direct_t magma_direct_const( char lapack_char )
{
    switch( lapack_char ) {
        case 'F': case 'f': return MagmaForward;
        case 'B': case 'b': return MagmaBackward;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaForward;
    }
}

/******************************************************************************/
/// @retval MagmaColumnwise if lapack_char = 'C'
/// @retval MagmaRowwise    if lapack_char = 'R'
extern "C"
magma_storev_t magma_storev_const( char lapack_char )
{
    switch( lapack_char ) {
        case 'C': case 'c': return MagmaColumnwise;
        case 'R': case 'r': return MagmaRowwise;
        default:
            fprintf( stderr, "Error in %s: unexpected value %c\n", __func__, lapack_char );
            return MagmaColumnwise;
    }
}

// =============================================================================
/// @}
// end group magma_const


// =============================================================================
/// @addtogroup lapack_const
/// Convert MAGMA constants to LAPACK constants.
/// Though LAPACK only cares about the first character,
/// the string is generally descriptive, such as "Upper".
/// @{

// The magma2lapack_constants table has an entry for each MAGMA constant,
// enumerated on the right, with a corresponding LAPACK string.
// The lapack_*_const() functions return entries from this table.
// The lapacke_*_const() functions defined in magma_types.h
// return a single character (e.g., 'U' for "Upper").

const char *magma2lapack_constants[] =
{
    "No",                                    //  0: MagmaFalse
    "Yes",                                   //  1: MagmaTrue (zlatrs)
    "", "", "", "", "", "", "", "", "",      //  2-10
    "", "", "", "", "", "", "", "", "", "",  // 11-20
    "", "", "", "", "", "", "", "", "", "",  // 21-30
    "", "", "", "", "", "", "", "", "", "",  // 31-40
    "", "", "", "", "", "", "", "", "", "",  // 41-50
    "", "", "", "", "", "", "", "", "", "",  // 51-60
    "", "", "", "", "", "", "", "", "", "",  // 61-70
    "", "", "", "", "", "", "", "", "", "",  // 71-80
    "", "", "", "", "", "", "", "", "", "",  // 81-90
    "", "", "", "", "", "", "", "", "", "",  // 91-100
    "Row",                                   // 101: MagmaRowMajor
    "Column",                                // 102: MagmaColMajor
    "", "", "", "", "", "", "", "",          // 103-110
    "No transpose",                          // 111: MagmaNoTrans
    "Transpose",                             // 112: MagmaTrans
    "Conjugate transpose",                   // 113: MagmaConjTrans
    "", "", "", "", "", "", "",              // 114-120
    "Upper",                                 // 121: MagmaUpper
    "Lower",                                 // 122: MagmaLower
    "General",                               // 123: MagmaFull; see lascl for "G"
    "", "", "", "", "", "", "",              // 124-130
    "Non-unit",                              // 131: MagmaNonUnit
    "Unit",                                  // 132: MagmaUnit
    "", "", "", "", "", "", "", "",          // 133-140
    "Left",                                  // 141: MagmaLeft
    "Right",                                 // 142: MagmaRight
    "Both",                                  // 143: MagmaBothSides (dtrevc)
    "", "", "", "", "", "", "",              // 144-150
    "", "", "", "", "", "", "", "", "", "",  // 151-160
    "", "", "", "", "", "", "", "", "", "",  // 161-170
    "1 norm",                                // 171: MagmaOneNorm
    "",                                      // 172: MagmaRealOneNorm
    "2 norm",                                // 173: MagmaTwoNorm
    "Frobenius norm",                        // 174: MagmaFrobeniusNorm
    "Infinity norm",                         // 175: MagmaInfNorm
    "",                                      // 176: MagmaRealInfNorm
    "Maximum norm",                          // 177: MagmaMaxNorm
    "",                                      // 178: MagmaRealMaxNorm
    "", "",                                  // 179-180
    "", "", "", "", "", "", "", "", "", "",  // 181-190
    "", "", "", "", "", "", "", "", "", "",  // 191-200
    "Uniform",                               // 201: MagmaDistUniform
    "Symmetric",                             // 202: MagmaDistSymmetric
    "Normal",                                // 203: MagmaDistNormal
    "", "", "", "", "", "", "",              // 204-210
    "", "", "", "", "", "", "", "", "", "",  // 211-220
    "", "", "", "", "", "", "", "", "", "",  // 221-230
    "", "", "", "", "", "", "", "", "", "",  // 231-240
    "Hermitian",                             // 241 MagmaHermGeev
    "Positive ev Hermitian",                 // 242 MagmaHermPoev
    "NonSymmetric pos sv",                   // 243 MagmaNonsymPosv
    "Symmetric pos sv",                      // 244 MagmaSymPosv
    "", "", "", "", "", "",                  // 245-250
    "", "", "", "", "", "", "", "", "", "",  // 251-260
    "", "", "", "", "", "", "", "", "", "",  // 261-270
    "", "", "", "", "", "", "", "", "", "",  // 271-280
    "", "", "", "", "", "", "", "", "", "",  // 281-290
    "No Packing",                            // 291 MagmaNoPacking
    "U zero out subdiag",                    // 292 MagmaPackSubdiag
    "L zero out superdiag",                  // 293 MagmaPackSupdiag
    "C",                                     // 294 MagmaPackColumn
    "R",                                     // 295 MagmaPackRow
    "B",                                     // 296 MagmaPackLowerBand
    "Q",                                     // 297 MagmaPackUpeprBand
    "Z",                                     // 298 MagmaPackAll
    "", "",                                  // 299-300
    "No vectors",                            // 301 MagmaNoVec
    "Vectors needed",                        // 302 MagmaVec
    "I",                                     // 303 MagmaIVec
    "All",                                   // 304 MagmaAllVec
    "Some",                                  // 305 MagmaSomeVec
    "Overwrite",                             // 306 MagmaOverwriteVec
    "", "", "", "",                          // 307-310
    "All",                                   // 311 MagmaRangeAll
    "V",                                     // 312 MagmaRangeV
    "I",                                     // 313 MagmaRangeI
    "", "", "", "", "", "", "",              // 314-320
    "",                                      // 321
    "Q",                                     // 322
    "P",                                     // 323
    "", "", "", "", "", "", "",              // 324-330
    "", "", "", "", "", "", "", "", "", "",  // 331-340
    "", "", "", "", "", "", "", "", "", "",  // 341-350
    "", "", "", "", "", "", "", "", "", "",  // 351-360
    "", "", "", "", "", "", "", "", "", "",  // 361-370
    "", "", "", "", "", "", "", "", "", "",  // 371-380
    "", "", "", "", "", "", "", "", "", "",  // 381-390
    "Forward",                               // 391: MagmaForward
    "Backward",                              // 392: MagmaBackward
    "", "", "", "", "", "", "", "",          // 393-400
    "Columnwise",                            // 401: MagmaColumnwise
    "Rowwise",                               // 402: MagmaRowwise
    "", "", "", "", "", "", "", ""           // 403-410
    // Remember to add a comma!
};

/******************************************************************************/
/// maps any MAGMA constant to its corresponding LAPACK string
extern "C"
const char* lapack_const_str( int magma_const )
{
    assert( magma_const >= Magma2lapack_Min );
    assert( magma_const <= Magma2lapack_Max );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_bool_const()
extern "C"
const char* lapack_bool_const( magma_bool_t magma_const )
{
    assert( magma_const >= MagmaFalse );
    assert( magma_const <= MagmaTrue  );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_order_const()
extern "C"
const char* lapack_order_const( magma_order_t magma_const )
{
    assert( magma_const >= MagmaRowMajor );
    assert( magma_const <= MagmaColMajor );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_trans_const()
extern "C"
const char* lapack_trans_const( magma_trans_t magma_const )
{
    assert( magma_const >= MagmaNoTrans   );
    assert( magma_const <= MagmaConjTrans );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_uplo_const()
extern "C"
const char* lapack_uplo_const ( magma_uplo_t magma_const )
{
    assert( magma_const >= MagmaUpper );
    assert( magma_const <= MagmaFull  );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_diag_const()
extern "C"
const char* lapack_diag_const ( magma_diag_t magma_const )
{
    assert( magma_const >= MagmaNonUnit );
    assert( magma_const <= MagmaUnit    );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_side_const()
extern "C"
const char* lapack_side_const ( magma_side_t magma_const )
{
    assert( magma_const >= MagmaLeft  );
    assert( magma_const <= MagmaBothSides );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_norm_const()
extern "C"
const char* lapack_norm_const  ( magma_norm_t   magma_const )
{
    assert( magma_const >= MagmaOneNorm     );
    assert( magma_const <= MagmaRealMaxNorm );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_dist_const()
extern "C"
const char* lapack_dist_const  ( magma_dist_t   magma_const )
{
    assert( magma_const >= MagmaDistUniform );
    assert( magma_const <= MagmaDistNormal );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_sym_const()
extern "C"
const char* lapack_sym_const   ( magma_sym_t    magma_const )
{
    assert( magma_const >= MagmaHermGeev );
    assert( magma_const <= MagmaSymPosv  );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_pack_const()
extern "C"
const char* lapack_pack_const  ( magma_pack_t   magma_const )
{
    assert( magma_const >= MagmaNoPacking );
    assert( magma_const <= MagmaPackAll   );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_vec_const()
extern "C"
const char* lapack_vec_const   ( magma_vec_t    magma_const )
{
    assert( magma_const >= MagmaNoVec );
    assert( magma_const <= MagmaOverwriteVec );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_range_const()
extern "C"
const char* lapack_range_const ( magma_range_t  magma_const )
{
    assert( magma_const >= MagmaRangeAll );
    assert( magma_const <= MagmaRangeI   );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_vect_const()
extern "C"
const char* lapack_vect_const( magma_vect_t magma_const )
{
    assert( magma_const >= MagmaQ );
    assert( magma_const <= MagmaP );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_direct_const()
extern "C"
const char* lapack_direct_const( magma_direct_t magma_const )
{
    assert( magma_const >= MagmaForward );
    assert( magma_const <= MagmaBackward );
    return magma2lapack_constants[ magma_const ];
}

/******************************************************************************/
/// inverse of magma_storev_const()
extern "C"
const char* lapack_storev_const( magma_storev_t magma_const )
{
    assert( magma_const >= MagmaColumnwise );
    assert( magma_const <= MagmaRowwise    );
    return magma2lapack_constants[ magma_const ];
}

// =============================================================================
/// @}
// end group lapack_const


#ifdef MAGMA_HAVE_clBLAS
// =============================================================================
/// @addtogroup clblas_const
/// Convert MAGMA constants to AMD clBLAS constants.
/// Available if MAGMA_HAVE_OPENCL was defined when MAGMA was compiled.
/// TODO: we do not currently provide inverse converters (clBLAS => MAGMA).
/// @{

// The magma2clblas_constants table has an entry for each MAGMA constant,
// enumerated on the right, with a corresponding clBLAS constant.

const int magma2clblas_constants[] =
{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,                      // 100
    clblasRowMajor,         // 101: MagmaRowMajor
    clblasColumnMajor,      // 102: MagmaColMajor
    0, 0, 0, 0, 0, 0, 0, 0,
    clblasNoTrans,          // 111: MagmaNoTrans
    clblasTrans,            // 112: MagmaTrans
    clblasConjTrans,        // 113: MagmaConjTrans
    0, 0, 0, 0, 0, 0, 0,
    clblasUpper,            // 121: MagmaUpper
    clblasLower,            // 122: MagmaLower
    0, 0, 0, 0, 0, 0, 0, 0,
    clblasNonUnit,          // 131: MagmaNonUnit
    clblasUnit,             // 132: MagmaUnit
    0, 0, 0, 0, 0, 0, 0, 0,
    clblasLeft,             // 141: MagmaLeft
    clblasRight,            // 142: MagmaRight
    0, 0, 0, 0, 0, 0, 0, 0
};

/******************************************************************************/
/// @retval clblasRowMajor    if magma_const = MagmaRowMajor
/// @retval clblasColumnMajor if magma_const = MagmaColMajor
extern "C"
clblasOrder       clblas_order_const( magma_order_t magma_const )
{
    assert( magma_const >= MagmaRowMajor );
    assert( magma_const <= MagmaColMajor );
    return (clblasOrder)     magma2clblas_constants[ magma_const ];
}

/******************************************************************************/
/// @retval clblasNoTrans   if magma_const = MagmaNoTrans
/// @retval clblasTrans     if magma_const = MagmaTrans
/// @retval clblasConjTrans if magma_const = MagmaConjTrans
extern "C"
clblasTranspose   clblas_trans_const( magma_trans_t magma_const )
{
    assert( magma_const >= MagmaNoTrans   );
    assert( magma_const <= MagmaConjTrans );
    return (clblasTranspose) magma2clblas_constants[ magma_const ];
}

/******************************************************************************/
/// @retval clblasUpper if magma_const = MagmaUpper
/// @retval clblasLower if magma_const = MagmaLower
extern "C"
clblasUplo        clblas_uplo_const ( magma_uplo_t magma_const )
{
    assert( magma_const >= MagmaUpper );
    assert( magma_const <= MagmaLower );
    return (clblasUplo)      magma2clblas_constants[ magma_const ];
}

/******************************************************************************/
/// @retval clblasNonUnit if magma_const = MagmaNonUnit
/// @retval clblasUnit    if magma_const = MagmaUnit
extern "C"
clblasDiag        clblas_diag_const ( magma_diag_t magma_const )
{
    assert( magma_const >= MagmaNonUnit );
    assert( magma_const <= MagmaUnit    );
    return (clblasDiag)      magma2clblas_constants[ magma_const ];
}

/******************************************************************************/
/// @retval clblasLeft  if magma_const = MagmaLeft
/// @retval clblasRight if magma_const = MagmaRight
extern "C"
clblasSide        clblas_side_const ( magma_side_t magma_const )
{
    assert( magma_const >= MagmaLeft  );
    assert( magma_const <= MagmaRight );
    return (clblasSide)      magma2clblas_constants[ magma_const ];
}

// =============================================================================
/// @}
// end group clblas_const
#endif  // MAGMA_HAVE_OPENCL


#ifdef MAGMA_HAVE_CUDA
// =============================================================================
/// @addtogroup cublas_const
/// Convert MAGMA constants to NVIDIA cuBLAS constants.
/// Available if MAGMA_HAVE_CUDA was defined when MAGMA was compiled.
/// TODO: we do not currently provide inverse converters (cuBLAS => MAGMA).
/// @{

// The magma2cublas_constants table has an entry for each MAGMA constant,
// enumerated on the right, with a corresponding cuBLAS constant.

const int magma2cublas_constants[] =
{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,                      // 100
    0,                      // 101: MagmaRowMajor
    0,                      // 102: MagmaColMajor
    0, 0, 0, 0, 0, 0, 0, 0,
    CUBLAS_OP_N,            // 111: MagmaNoTrans
    CUBLAS_OP_T,            // 112: MagmaTrans
    CUBLAS_OP_C,            // 113: MagmaConjTrans
    0, 0, 0, 0, 0, 0, 0,
    CUBLAS_FILL_MODE_UPPER, // 121: MagmaUpper
    CUBLAS_FILL_MODE_LOWER, // 122: MagmaLower
    0, 0, 0, 0, 0, 0, 0, 0,
    CUBLAS_DIAG_NON_UNIT,   // 131: MagmaNonUnit
    CUBLAS_DIAG_UNIT,       // 132: MagmaUnit
    0, 0, 0, 0, 0, 0, 0, 0,
    CUBLAS_SIDE_LEFT,       // 141: MagmaLeft
    CUBLAS_SIDE_RIGHT,      // 142: MagmaRight
    0, 0, 0, 0, 0, 0, 0, 0
};

/******************************************************************************/
/// @retval CUBLAS_OP_N if magma_const = MagmaNoTrans
/// @retval CUBLAS_OP_T if magma_const = MagmaTrans
/// @retval CUBLAS_OP_C if magma_const = MagmaConjTrans
extern "C"
cublasOperation_t    cublas_trans_const ( magma_trans_t magma_const )
{
    assert( magma_const >= MagmaNoTrans   );
    assert( magma_const <= MagmaConjTrans );
    return (cublasOperation_t)  magma2cublas_constants[ magma_const ];
}

/******************************************************************************/
/// @retval CUBLAS_FILL_MODE_UPPER if magma_const = MagmaUpper
/// @retval CUBLAS_FILL_MODE_LOWER if magma_const = MagmaLower
extern "C"
cublasFillMode_t     cublas_uplo_const  ( magma_uplo_t magma_const )
{
    assert( magma_const >= MagmaUpper );
    assert( magma_const <= MagmaLower );
    return (cublasFillMode_t)   magma2cublas_constants[ magma_const ];
}

/******************************************************************************/
/// @retval CUBLAS_DIAG_NONUNIT if magma_const = MagmaNonUnit
/// @retval CUBLAS_DIAG_UNIT    if magma_const = MagmaUnit
extern "C"
cublasDiagType_t     cublas_diag_const  ( magma_diag_t magma_const )
{
    assert( magma_const >= MagmaNonUnit );
    assert( magma_const <= MagmaUnit    );
    return (cublasDiagType_t)   magma2cublas_constants[ magma_const ];
}

/******************************************************************************/
/// @retval CUBLAS_SIDE_LEFT  if magma_const = MagmaLeft
/// @retval CUBLAS_SIDE_RIGHT if magma_const = MagmaRight
extern "C"
cublasSideMode_t     cublas_side_const  ( magma_side_t magma_const )
{
    assert( magma_const >= MagmaLeft  );
    assert( magma_const <= MagmaRight );
    return (cublasSideMode_t)   magma2cublas_constants[ magma_const ];
}

// =============================================================================
/// @}
#endif  // MAGMA_HAVE_CUDA



#ifdef MAGMA_HAVE_HIP

const int magma2hipblas_constants[] =
{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,                      // 100
    0,                      // 101: MagmaRowMajor
    0,                      // 102: MagmaColMajor
    0, 0, 0, 0, 0, 0, 0, 0,
    HIPBLAS_OP_N,           // 111: MagmaNoTrans
    HIPBLAS_OP_T,           // 112: MagmaTrans
    HIPBLAS_OP_C,           // 113: MagmaConjTrans
    0, 0, 0, 0, 0, 0, 0,
    HIPBLAS_FILL_MODE_UPPER,// 121: MagmaUpper
    HIPBLAS_FILL_MODE_LOWER,// 122: MagmaLower
    0, 0, 0, 0, 0, 0, 0, 0,
    HIPBLAS_DIAG_NON_UNIT,  // 131: MagmaNonUnit
    HIPBLAS_DIAG_UNIT,      // 132: MagmaUnit
    0, 0, 0, 0, 0, 0, 0, 0,
    HIPBLAS_SIDE_LEFT,      // 141: MagmaLeft
    HIPBLAS_SIDE_RIGHT,     // 142: MagmaRight
    0, 0, 0, 0, 0, 0, 0, 0
};

extern "C"
hipblasOperation_t    hipblas_trans_const ( magma_trans_t magma_const )
{
    assert( magma_const >= MagmaNoTrans   );
    assert( magma_const <= MagmaConjTrans );
    return (hipblasOperation_t)  magma2hipblas_constants[ magma_const ];
}

extern "C"
hipblasFillMode_t     hipblas_uplo_const  ( magma_uplo_t magma_const )
{
    assert( magma_const >= MagmaUpper );
    assert( magma_const <= MagmaLower );
    return (hipblasFillMode_t)   magma2hipblas_constants[ magma_const ];
}

extern "C"
hipblasDiagType_t     hipblas_diag_const  ( magma_diag_t magma_const )
{
    assert( magma_const >= MagmaNonUnit );
    assert( magma_const <= MagmaUnit    );
    return (hipblasDiagType_t)   magma2hipblas_constants[ magma_const ];
}

extern "C"
hipblasSideMode_t     hipblas_side_const  ( magma_side_t magma_const )
{
    assert( magma_const >= MagmaLeft  );
    assert( magma_const <= MagmaRight );
    return (hipblasSideMode_t)   magma2hipblas_constants[ magma_const ];
}


#endif // MAGMA_HAVE_HIP


#ifdef MAGMA_HAVE_SYCL

const char magma2syclblas_constants[] =
{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0,                      // 100
    0,                      // 101: MagmaRowMajor
    0,                      // 102: MagmaColMajor
    0, 0, 0, 0, 0, 0, 0, 0,
    (char) oneapi::mkl::transpose::nontrans,  // 111: MagmaNoTrans
    (char) oneapi::mkl::transpose::trans,     // 112: MagmaTrans
    (char) oneapi::mkl::transpose::conjtrans, // 113: MagmaConjTrans
    0, 0, 0, 0, 0, 0, 0,
    (char) oneapi::mkl::uplo::upper,// 121: MagmaUpper
    (char) oneapi::mkl::uplo::lower,// 122: MagmaLower
    0, 0, 0, 0, 0, 0, 0, 0,
    (char) oneapi::mkl::diag::nonunit,// 131: MagmaNonUnit
    (char) oneapi::mkl::diag::unit,   // 132: MagmaUnit
    0, 0, 0, 0, 0, 0, 0, 0,
    (char) oneapi::mkl::side::left,   // 141: MagmaLeft
    (char) oneapi::mkl::side::right,  // 142: MagmaRight
    0, 0, 0, 0, 0, 0, 0, 0
};

extern "C"
oneapi::mkl::transpose    syclblas_trans_const ( magma_trans_t magma_const )
{
    assert( magma_const >= MagmaNoTrans   );
    assert( magma_const <= MagmaConjTrans );
    oneapi::mkl::transpose trans_const{magma2syclblas_constants[ magma_const ]};
    return trans_const;
}

extern "C"
oneapi::mkl::uplo     syclblas_uplo_const  ( magma_uplo_t magma_const )
{
    assert( magma_const >= MagmaUpper );
    assert( magma_const <= MagmaLower );
    oneapi::mkl::uplo uplo_const{magma2syclblas_constants[ magma_const ]};
    return uplo_const;
}

extern "C"
oneapi::mkl::diag    syclblas_diag_const  ( magma_diag_t magma_const )
{
    assert( magma_const >= MagmaNonUnit );
    assert( magma_const <= MagmaUnit    );
    oneapi::mkl::diag diag_const{magma2syclblas_constants[ magma_const ]};
    return diag_const;
}

extern "C"
oneapi::mkl::side   syclblas_side_const  ( magma_side_t magma_const )
{
    assert( magma_const >= MagmaLeft  );
    assert( magma_const <= MagmaRight );
    oneapi::mkl::side side_const{magma2syclblas_constants[ magma_const ]};
    return side_const;
}


#endif // MAGMA_HAVE_SYCL

#ifdef HAVE_CBLAS
// =============================================================================
/// @addtogroup cblas_const
/// Convert MAGMA constants to CBLAS constants.
/// Available if HAVE_CBLAS was defined when MAGMA was compiled.
/// MAGMA constants have the same value as CBLAS constants,
/// which these routines verify by asserts.
/// TODO: we do not currently provide inverse converters (CBLAS => MAGMA),
/// though it is a trivial cast since the values are the same.
/// @{

/******************************************************************************/
/// @retval CblasRowMajor if magma_const = MagmaRowMajor
/// @retval CblasColMajor if magma_const = MagmaColMajor
extern "C"
enum CBLAS_ORDER     cblas_order_const  ( magma_order_t magma_const )
{
    assert( magma_const >= MagmaRowMajor );
    assert( magma_const <= MagmaColMajor );
    assert( (int)MagmaRowMajor == CblasRowMajor );
    return (enum CBLAS_ORDER)     magma_const;
}

/******************************************************************************/
/// @retval CblasNoTrans   if magma_const = MagmaNoTrans
/// @retval CblasTrans     if magma_const = MagmaTrans
/// @retval CblasConjTrans if magma_const = MagmaConjTrans
extern "C"
enum CBLAS_TRANSPOSE cblas_trans_const  ( magma_trans_t magma_const )
{
    assert( magma_const >= MagmaNoTrans   );
    assert( magma_const <= MagmaConjTrans );
    assert( (int)MagmaNoTrans == CblasNoTrans );
    return (enum CBLAS_TRANSPOSE) magma_const;
}

/******************************************************************************/
/// @retval CblasUpper if magma_const = MagmaUpper
/// @retval CblasLower if magma_const = MagmaLower
extern "C"
enum CBLAS_UPLO      cblas_uplo_const   ( magma_uplo_t magma_const )
{
    assert( magma_const >= MagmaUpper );
    assert( magma_const <= MagmaLower );
    assert( (int)MagmaUpper == CblasUpper );
    return (enum CBLAS_UPLO)      magma_const;
}

/******************************************************************************/
/// @retval CblasNonUnit if magma_const = MagmaNonUnit
/// @retval CblasUnit    if magma_const = MagmaUnit
extern "C"
enum CBLAS_DIAG      cblas_diag_const   ( magma_diag_t magma_const )
{
    assert( magma_const >= MagmaNonUnit );
    assert( magma_const <= MagmaUnit    );
    assert( (int)MagmaUnit == CblasUnit );
    return (enum CBLAS_DIAG)      magma_const;
}

/******************************************************************************/
/// @retval CblasLeft  if magma_const = MagmaLeft
/// @retval CblasRight if magma_const = MagmaRight
extern "C"
enum CBLAS_SIDE      cblas_side_const   ( magma_side_t magma_const )
{
    assert( magma_const >= MagmaLeft  );
    assert( magma_const <= MagmaRight );
    assert( (int)MagmaLeft == CblasLeft );
    return (enum CBLAS_SIDE)      magma_const;
}

// =============================================================================
/// @}
// end group cblas_const

#endif  // HAVE_CBLAS
