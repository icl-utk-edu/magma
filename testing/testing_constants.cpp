/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Mark Gates
*/
#include <stdio.h>

#include "magma_v2.h"
#include "testings.h"

int gStatus;

void check_( bool flag, const char* msg, int line )
{
    if ( ! flag ) {
        gStatus += 1;
        printf( "line %d: %s failed\n", line, msg );
    }
}

#define check( flag ) check_( flag, #flag, __LINE__ )


/* ////////////////////////////////////////////////////////////////////////////
   -- Testing lapack_xxxxx_const and related
*/
int main( int argc, char** argv )
{
    gStatus = 0;
    int s;

    // ------------------------------------------------------------
    s = gStatus;
    check( lapack_bool_const(   MagmaFalse         )[0] == 'N' );
    check( lapack_bool_const(   MagmaTrue          )[0] == 'Y' );

    check( lapack_order_const(  MagmaRowMajor      )[0] == 'R' );
    check( lapack_order_const(  MagmaColMajor      )[0] == 'C' );

    check( lapack_trans_const(  MagmaNoTrans       )[0] == 'N' );
    check( lapack_trans_const(  MagmaTrans         )[0] == 'T' );
    check( lapack_trans_const(  MagmaConjTrans     )[0] == 'C' );

    check( lapack_uplo_const(   MagmaUpper         )[0] == 'U' );
    check( lapack_uplo_const(   MagmaLower         )[0] == 'L' );
    check( lapack_uplo_const(   MagmaFull          )[0] == 'G' );

    check( lapack_diag_const(   MagmaNonUnit       )[0] == 'N' );
    check( lapack_diag_const(   MagmaUnit          )[0] == 'U' );

    check( lapack_side_const(   MagmaLeft          )[0] == 'L' );
    check( lapack_side_const(   MagmaRight         )[0] == 'R' );
    check( lapack_side_const(   MagmaBothSides     )[0] == 'B' );

    check( lapack_norm_const(   MagmaOneNorm       )[0] == '1' );
    check( lapack_norm_const(   MagmaTwoNorm       )[0] == '2' );
    check( lapack_norm_const(   MagmaFrobeniusNorm )[0] == 'F' );
    check( lapack_norm_const(   MagmaInfNorm       )[0] == 'I' );
    check( lapack_norm_const(   MagmaMaxNorm       )[0] == 'M' );

    check( lapack_dist_const(   MagmaDistUniform   )[0] == 'U' );
    check( lapack_dist_const(   MagmaDistSymmetric )[0] == 'S' );
    check( lapack_dist_const(   MagmaDistNormal    )[0] == 'N' );

    check( lapack_sym_const(    MagmaHermGeev      )[0] == 'H' );
    check( lapack_sym_const(    MagmaHermPoev      )[0] == 'P' );
    check( lapack_sym_const(    MagmaNonsymPosv    )[0] == 'N' );
    check( lapack_sym_const(    MagmaSymPosv       )[0] == 'S' );

    check( lapack_pack_const(   MagmaNoPacking     )[0] == 'N' );
    check( lapack_pack_const(   MagmaPackSubdiag   )[0] == 'U' );
    check( lapack_pack_const(   MagmaPackSupdiag   )[0] == 'L' );
    check( lapack_pack_const(   MagmaPackColumn    )[0] == 'C' );
    check( lapack_pack_const(   MagmaPackRow       )[0] == 'R' );
    check( lapack_pack_const(   MagmaPackLowerBand )[0] == 'B' );
    check( lapack_pack_const(   MagmaPackUpeprBand )[0] == 'Q' );
    check( lapack_pack_const(   MagmaPackAll       )[0] == 'Z' );

    check( lapack_vec_const(    MagmaNoVec         )[0] == 'N' );
    check( lapack_vec_const(    MagmaVec           )[0] == 'V' );
    check( lapack_vec_const(    MagmaIVec          )[0] == 'I' );
    check( lapack_vec_const(    MagmaAllVec        )[0] == 'A' );
    check( lapack_vec_const(    MagmaSomeVec       )[0] == 'S' );
    check( lapack_vec_const(    MagmaOverwriteVec  )[0] == 'O' );

    check( lapack_range_const(  MagmaRangeAll      )[0] == 'A' );
    check( lapack_range_const(  MagmaRangeV        )[0] == 'V' );
    check( lapack_range_const(  MagmaRangeI        )[0] == 'I' );

    check( lapack_vect_const(   MagmaQ             )[0] == 'Q' );
    check( lapack_vect_const(   MagmaP             )[0] == 'P' );

    check( lapack_direct_const( MagmaForward       )[0] == 'F' );
    check( lapack_direct_const( MagmaBackward      )[0] == 'B' );

    check( lapack_storev_const( MagmaColumnwise    )[0] == 'C' );
    check( lapack_storev_const( MagmaRowwise       )[0] == 'R' );
    printf( "MAGMA  -> lapack_xxxxx_const    %s\n", (s == gStatus ? "ok" : "failed"));

    // ------------------------------------------------------------
    s = gStatus;
    check( lapacke_bool_const(   MagmaFalse         ) == 'N' );
    check( lapacke_bool_const(   MagmaTrue          ) == 'Y' );

    check( lapacke_order_const(  MagmaRowMajor      ) == 'R' );
    check( lapacke_order_const(  MagmaColMajor      ) == 'C' );

    check( lapacke_trans_const(  MagmaNoTrans       ) == 'N' );
    check( lapacke_trans_const(  MagmaTrans         ) == 'T' );
    check( lapacke_trans_const(  MagmaConjTrans     ) == 'C' );

    check( lapacke_uplo_const(   MagmaUpper         ) == 'U' );
    check( lapacke_uplo_const(   MagmaLower         ) == 'L' );
    check( lapacke_uplo_const(   MagmaFull          ) == 'G' );

    check( lapacke_diag_const(   MagmaNonUnit       ) == 'N' );
    check( lapacke_diag_const(   MagmaUnit          ) == 'U' );

    check( lapacke_side_const(   MagmaLeft          ) == 'L' );
    check( lapacke_side_const(   MagmaRight         ) == 'R' );
    check( lapacke_side_const(   MagmaBothSides     ) == 'B' );

    check( lapacke_norm_const(   MagmaOneNorm       ) == '1' );
    check( lapacke_norm_const(   MagmaTwoNorm       ) == '2' );
    check( lapacke_norm_const(   MagmaFrobeniusNorm ) == 'F' );
    check( lapacke_norm_const(   MagmaInfNorm       ) == 'I' );
    check( lapacke_norm_const(   MagmaMaxNorm       ) == 'M' );

    check( lapacke_dist_const(   MagmaDistUniform   ) == 'U' );
    check( lapacke_dist_const(   MagmaDistSymmetric ) == 'S' );
    check( lapacke_dist_const(   MagmaDistNormal    ) == 'N' );

    check( lapacke_sym_const(    MagmaHermGeev      ) == 'H' );
    check( lapacke_sym_const(    MagmaHermPoev      ) == 'P' );
    check( lapacke_sym_const(    MagmaNonsymPosv    ) == 'N' );
    check( lapacke_sym_const(    MagmaSymPosv       ) == 'S' );

    check( lapacke_pack_const(   MagmaNoPacking     ) == 'N' );
    check( lapacke_pack_const(   MagmaPackSubdiag   ) == 'U' );
    check( lapacke_pack_const(   MagmaPackSupdiag   ) == 'L' );
    check( lapacke_pack_const(   MagmaPackColumn    ) == 'C' );
    check( lapacke_pack_const(   MagmaPackRow       ) == 'R' );
    check( lapacke_pack_const(   MagmaPackLowerBand ) == 'B' );
    check( lapacke_pack_const(   MagmaPackUpeprBand ) == 'Q' );
    check( lapacke_pack_const(   MagmaPackAll       ) == 'Z' );

    check( lapacke_vec_const(    MagmaNoVec         ) == 'N' );
    check( lapacke_vec_const(    MagmaVec           ) == 'V' );
    check( lapacke_vec_const(    MagmaIVec          ) == 'I' );
    check( lapacke_vec_const(    MagmaAllVec        ) == 'A' );
    check( lapacke_vec_const(    MagmaSomeVec       ) == 'S' );
    check( lapacke_vec_const(    MagmaOverwriteVec  ) == 'O' );

    check( lapacke_range_const(  MagmaRangeAll      ) == 'A' );
    check( lapacke_range_const(  MagmaRangeV        ) == 'V' );
    check( lapacke_range_const(  MagmaRangeI        ) == 'I' );

    check( lapacke_vect_const(   MagmaQ             ) == 'Q' );
    check( lapacke_vect_const(   MagmaP             ) == 'P' );

    check( lapacke_direct_const( MagmaForward       ) == 'F' );
    check( lapacke_direct_const( MagmaBackward      ) == 'B' );

    check( lapacke_storev_const( MagmaColumnwise    ) == 'C' );
    check( lapacke_storev_const( MagmaRowwise       ) == 'R' );
    printf( "MAGMA  -> lapacke_xxxxx_const   %s\n", (s == gStatus ? "ok" : "failed"));

    // ------------------------------------------------------------
    s = gStatus;
    check( lapack_const_str( MagmaFalse         )[0] == 'N' );
    check( lapack_const_str( MagmaTrue          )[0] == 'Y' );

    check( lapack_const_str( MagmaRowMajor      )[0] == 'R' );
    check( lapack_const_str( MagmaColMajor      )[0] == 'C' );

    check( lapack_const_str( MagmaNoTrans       )[0] == 'N' );
    check( lapack_const_str( MagmaTrans         )[0] == 'T' );
    check( lapack_const_str( MagmaConjTrans     )[0] == 'C' );

    check( lapack_const_str( MagmaUpper         )[0] == 'U' );
    check( lapack_const_str( MagmaLower         )[0] == 'L' );
    check( lapack_const_str( MagmaFull          )[0] == 'G' );

    check( lapack_const_str( MagmaNonUnit       )[0] == 'N' );
    check( lapack_const_str( MagmaUnit          )[0] == 'U' );

    check( lapack_const_str( MagmaLeft          )[0] == 'L' );
    check( lapack_const_str( MagmaRight         )[0] == 'R' );
    check( lapack_const_str( MagmaBothSides     )[0] == 'B' );

    check( lapack_const_str( MagmaOneNorm       )[0] == '1' );
    check( lapack_const_str( MagmaTwoNorm       )[0] == '2' );
    check( lapack_const_str( MagmaFrobeniusNorm )[0] == 'F' );
    check( lapack_const_str( MagmaInfNorm       )[0] == 'I' );
    check( lapack_const_str( MagmaMaxNorm       )[0] == 'M' );

    check( lapack_const_str( MagmaDistUniform   )[0] == 'U' );
    check( lapack_const_str( MagmaDistSymmetric )[0] == 'S' );
    check( lapack_const_str( MagmaDistNormal    )[0] == 'N' );

    check( lapack_const_str( MagmaHermGeev      )[0] == 'H' );
    check( lapack_const_str( MagmaHermPoev      )[0] == 'P' );
    check( lapack_const_str( MagmaNonsymPosv    )[0] == 'N' );
    check( lapack_const_str( MagmaSymPosv       )[0] == 'S' );

    check( lapack_const_str( MagmaNoPacking     )[0] == 'N' );
    check( lapack_const_str( MagmaPackSubdiag   )[0] == 'U' );
    check( lapack_const_str( MagmaPackSupdiag   )[0] == 'L' );
    check( lapack_const_str( MagmaPackColumn    )[0] == 'C' );
    check( lapack_const_str( MagmaPackRow       )[0] == 'R' );
    check( lapack_const_str( MagmaPackLowerBand )[0] == 'B' );
    check( lapack_const_str( MagmaPackUpeprBand )[0] == 'Q' );
    check( lapack_const_str( MagmaPackAll       )[0] == 'Z' );

    check( lapack_const_str( MagmaNoVec         )[0] == 'N' );
    check( lapack_const_str( MagmaVec           )[0] == 'V' );
    check( lapack_const_str( MagmaIVec          )[0] == 'I' );
    check( lapack_const_str( MagmaAllVec        )[0] == 'A' );
    check( lapack_const_str( MagmaSomeVec       )[0] == 'S' );
    check( lapack_const_str( MagmaOverwriteVec  )[0] == 'O' );

    check( lapack_const_str( MagmaRangeAll      )[0] == 'A' );
    check( lapack_const_str( MagmaRangeV        )[0] == 'V' );
    check( lapack_const_str( MagmaRangeI        )[0] == 'I' );

    check( lapack_const_str( MagmaQ             )[0] == 'Q' );
    check( lapack_const_str( MagmaP             )[0] == 'P' );

    check( lapack_const_str( MagmaForward       )[0] == 'F' );
    check( lapack_const_str( MagmaBackward      )[0] == 'B' );

    check( lapack_const_str( MagmaColumnwise    )[0] == 'C' );
    check( lapack_const_str( MagmaRowwise       )[0] == 'R' );
    printf( "MAGMA  -> lapack_const_str      %s\n", (s == gStatus ? "ok" : "failed"));

    // ------------------------------------------------------------
    s = gStatus;
    check( lapacke_const( MagmaFalse         ) == 'N' );
    check( lapacke_const( MagmaTrue          ) == 'Y' );

    check( lapacke_const( MagmaRowMajor      ) == 'R' );
    check( lapacke_const( MagmaColMajor      ) == 'C' );

    check( lapacke_const( MagmaNoTrans       ) == 'N' );
    check( lapacke_const( MagmaTrans         ) == 'T' );
    check( lapacke_const( MagmaConjTrans     ) == 'C' );

    check( lapacke_const( MagmaUpper         ) == 'U' );
    check( lapacke_const( MagmaLower         ) == 'L' );
    check( lapacke_const( MagmaFull          ) == 'G' );

    check( lapacke_const( MagmaNonUnit       ) == 'N' );
    check( lapacke_const( MagmaUnit          ) == 'U' );

    check( lapacke_const( MagmaLeft          ) == 'L' );
    check( lapacke_const( MagmaRight         ) == 'R' );
    check( lapacke_const( MagmaBothSides     ) == 'B' );

    check( lapacke_const( MagmaOneNorm       ) == '1' );
    check( lapacke_const( MagmaTwoNorm       ) == '2' );
    check( lapacke_const( MagmaFrobeniusNorm ) == 'F' );
    check( lapacke_const( MagmaInfNorm       ) == 'I' );
    check( lapacke_const( MagmaMaxNorm       ) == 'M' );

    check( lapacke_const( MagmaDistUniform   ) == 'U' );
    check( lapacke_const( MagmaDistSymmetric ) == 'S' );
    check( lapacke_const( MagmaDistNormal    ) == 'N' );

    check( lapacke_const( MagmaHermGeev      ) == 'H' );
    check( lapacke_const( MagmaHermPoev      ) == 'P' );
    check( lapacke_const( MagmaNonsymPosv    ) == 'N' );
    check( lapacke_const( MagmaSymPosv       ) == 'S' );

    check( lapacke_const( MagmaNoPacking     ) == 'N' );
    check( lapacke_const( MagmaPackSubdiag   ) == 'U' );
    check( lapacke_const( MagmaPackSupdiag   ) == 'L' );
    check( lapacke_const( MagmaPackColumn    ) == 'C' );
    check( lapacke_const( MagmaPackRow       ) == 'R' );
    check( lapacke_const( MagmaPackLowerBand ) == 'B' );
    check( lapacke_const( MagmaPackUpeprBand ) == 'Q' );
    check( lapacke_const( MagmaPackAll       ) == 'Z' );

    check( lapacke_const( MagmaNoVec         ) == 'N' );
    check( lapacke_const( MagmaVec           ) == 'V' );
    check( lapacke_const( MagmaIVec          ) == 'I' );
    check( lapacke_const( MagmaAllVec        ) == 'A' );
    check( lapacke_const( MagmaSomeVec       ) == 'S' );
    check( lapacke_const( MagmaOverwriteVec  ) == 'O' );

    check( lapacke_const( MagmaRangeAll      ) == 'A' );
    check( lapacke_const( MagmaRangeV        ) == 'V' );
    check( lapacke_const( MagmaRangeI        ) == 'I' );

    check( lapacke_const( MagmaQ             ) == 'Q' );
    check( lapacke_const( MagmaP             ) == 'P' );

    check( lapacke_const( MagmaForward       ) == 'F' );
    check( lapacke_const( MagmaBackward      ) == 'B' );

    check( lapacke_const( MagmaColumnwise    ) == 'C' );
    check( lapacke_const( MagmaRowwise       ) == 'R' );
    printf( "MAGMA  -> lapacke_const         %s\n", (s == gStatus ? "ok" : "failed"));

    // ------------------------------------------------------------
    s = gStatus;
    check( magma_bool_const('N') == MagmaFalse );
    check( magma_bool_const('n') == MagmaFalse );
    check( magma_bool_const('Y') == MagmaTrue  );
    check( magma_bool_const('y') == MagmaTrue  );

    check( magma_order_const( 'R' ) == MagmaRowMajor  );
    check( magma_order_const( 'r' ) == MagmaRowMajor  );
    check( magma_order_const( 'C' ) == MagmaColMajor  );
    check( magma_order_const( 'c' ) == MagmaColMajor  );

    check( magma_trans_const( 'N' ) == MagmaNoTrans   );
    check( magma_trans_const( 'n' ) == MagmaNoTrans   );
    check( magma_trans_const( 'T' ) == MagmaTrans     );
    check( magma_trans_const( 't' ) == MagmaTrans     );
    check( magma_trans_const( 'C' ) == MagmaConjTrans );
    check( magma_trans_const( 'c' ) == MagmaConjTrans );

    check( magma_uplo_const( 'U' ) == MagmaUpper      );
    check( magma_uplo_const( 'u' ) == MagmaUpper      );
    check( magma_uplo_const( 'L' ) == MagmaLower      );
    check( magma_uplo_const( 'l' ) == MagmaLower      );
    check( magma_uplo_const( 'A' ) == MagmaFull       );  // anything else
    check( magma_uplo_const( 'a' ) == MagmaFull       );
    check( magma_uplo_const( 'G' ) == MagmaFull       );
    check( magma_uplo_const( 'g' ) == MagmaFull       );
    check( magma_uplo_const( 'F' ) == MagmaFull       );
    check( magma_uplo_const( 'f' ) == MagmaFull       );

    check( magma_diag_const( 'N' ) == MagmaNonUnit    );
    check( magma_diag_const( 'n' ) == MagmaNonUnit    );
    check( magma_diag_const( 'U' ) == MagmaUnit       );
    check( magma_diag_const( 'u' ) == MagmaUnit       );

    check( magma_side_const( 'L' ) == MagmaLeft       );
    check( magma_side_const( 'l' ) == MagmaLeft       );
    check( magma_side_const( 'R' ) == MagmaRight      );
    check( magma_side_const( 'r' ) == MagmaRight      );

    check( magma_norm_const( 'O' ) == MagmaOneNorm       );
    check( magma_norm_const( 'o' ) == MagmaOneNorm       );
    check( magma_norm_const( '1' ) == MagmaOneNorm       );
    check( magma_norm_const( '2' ) == MagmaTwoNorm       );
    check( magma_norm_const( 'F' ) == MagmaFrobeniusNorm );
    check( magma_norm_const( 'f' ) == MagmaFrobeniusNorm );
    check( magma_norm_const( 'E' ) == MagmaFrobeniusNorm );
    check( magma_norm_const( 'e' ) == MagmaFrobeniusNorm );
    check( magma_norm_const( 'I' ) == MagmaInfNorm       );
    check( magma_norm_const( 'i' ) == MagmaInfNorm       );
    check( magma_norm_const( 'M' ) == MagmaMaxNorm       );
    check( magma_norm_const( 'm' ) == MagmaMaxNorm       );

    check( magma_dist_const( 'U' ) == MagmaDistUniform   );
    check( magma_dist_const( 'u' ) == MagmaDistUniform   );
    check( magma_dist_const( 'S' ) == MagmaDistSymmetric );
    check( magma_dist_const( 's' ) == MagmaDistSymmetric );
    check( magma_dist_const( 'N' ) == MagmaDistNormal    );
    check( magma_dist_const( 'n' ) == MagmaDistNormal    );

    //check( magma_xxxx_const( 'H' ) == MagmaHermGeev      );
    //check( magma_xxxx_const( 'P' ) == MagmaHermPoev      );
    //check( magma_xxxx_const( 'N' ) == MagmaNonsymPosv    );
    //check( magma_xxxx_const( 'S' ) == MagmaSymPosv       );

    check( magma_pack_const( 'N' ) == MagmaNoPacking     );
    check( magma_pack_const( 'n' ) == MagmaNoPacking     );
    check( magma_pack_const( 'U' ) == MagmaPackSubdiag   );
    check( magma_pack_const( 'u' ) == MagmaPackSubdiag   );
    check( magma_pack_const( 'L' ) == MagmaPackSupdiag   );
    check( magma_pack_const( 'l' ) == MagmaPackSupdiag   );
    check( magma_pack_const( 'C' ) == MagmaPackColumn    );
    check( magma_pack_const( 'c' ) == MagmaPackColumn    );
    check( magma_pack_const( 'R' ) == MagmaPackRow       );
    check( magma_pack_const( 'r' ) == MagmaPackRow       );
    check( magma_pack_const( 'B' ) == MagmaPackLowerBand );
    check( magma_pack_const( 'b' ) == MagmaPackLowerBand );
    check( magma_pack_const( 'Q' ) == MagmaPackUpeprBand );
    check( magma_pack_const( 'q' ) == MagmaPackUpeprBand );
    check( magma_pack_const( 'Z' ) == MagmaPackAll       );
    check( magma_pack_const( 'z' ) == MagmaPackAll       );

    check( magma_vec_const( 'N' )  == MagmaNoVec         );
    check( magma_vec_const( 'n' )  == MagmaNoVec         );
    check( magma_vec_const( 'V' )  == MagmaVec           );
    check( magma_vec_const( 'v' )  == MagmaVec           );
    check( magma_vec_const( 'I' )  == MagmaIVec          );
    check( magma_vec_const( 'i' )  == MagmaIVec          );
    check( magma_vec_const( 'A' )  == MagmaAllVec        );
    check( magma_vec_const( 'a' )  == MagmaAllVec        );
    check( magma_vec_const( 'S' )  == MagmaSomeVec       );
    check( magma_vec_const( 's' )  == MagmaSomeVec       );
    check( magma_vec_const( 'O' )  == MagmaOverwriteVec  );
    check( magma_vec_const( 'o' )  == MagmaOverwriteVec  );

    check( magma_range_const( 'A' )  == MagmaRangeAll    );
    check( magma_range_const( 'a' )  == MagmaRangeAll    );
    check( magma_range_const( 'V' )  == MagmaRangeV      );
    check( magma_range_const( 'v' )  == MagmaRangeV      );
    check( magma_range_const( 'I' )  == MagmaRangeI      );
    check( magma_range_const( 'i' )  == MagmaRangeI      );

    check( magma_vect_const( 'Q' )   == MagmaQ           );
    check( magma_vect_const( 'q' )   == MagmaQ           );
    check( magma_vect_const( 'P' )   == MagmaP           );
    check( magma_vect_const( 'p' )   == MagmaP           );

    check( magma_direct_const( 'F' ) == MagmaForward     );
    check( magma_direct_const( 'f' ) == MagmaForward     );
    check( magma_direct_const( 'B' ) == MagmaBackward    );
    check( magma_direct_const( 'b' ) == MagmaBackward    );

    check( magma_storev_const( 'C' ) == MagmaColumnwise  );
    check( magma_storev_const( 'c' ) == MagmaColumnwise  );
    check( magma_storev_const( 'R' ) == MagmaRowwise     );
    check( magma_storev_const( 'r' ) == MagmaRowwise     );
    printf( "LAPACK -> magma_xxxxx_const     %s\n", (s == gStatus ? "ok" : "failed"));

    // ------------------------------------------------------------
    #ifdef MAGMA_HAVE_OPENCL
    s = gStatus;
    check( clblas_order_const( MagmaRowMajor      ) == clblasRowMajor    );
    check( clblas_order_const( MagmaColMajor      ) == clblasColumnMajor );

    check( clblas_trans_const( MagmaNoTrans       ) == clblasNoTrans     );
    check( clblas_trans_const( MagmaTrans         ) == clblasTrans       );
    check( clblas_trans_const( MagmaConjTrans     ) == clblasConjTrans   );

    check( clblas_uplo_const(  MagmaUpper         ) == clblasUpper       );
    check( clblas_uplo_const(  MagmaLower         ) == clblasLower       );

    check( clblas_diag_const(  MagmaNonUnit       ) == clblasNonUnit     );
    check( clblas_diag_const(  MagmaUnit          ) == clblasUnit        );

    check( clblas_side_const(  MagmaLeft          ) == clblasLeft        );
    check( clblas_side_const(  MagmaRight         ) == clblasRight       );
    printf( "MAGMA  -> clblas_xxxxx_const   %s\n", (s == gStatus ? "ok" : "failed"));
    #endif


    // ------------------------------------------------------------
    #ifdef CUBLAS_V2_H_
    s = gStatus;
    check( cublas_trans_const( MagmaNoTrans       ) == CUBLAS_OP_N            );
    check( cublas_trans_const( MagmaTrans         ) == CUBLAS_OP_T            );
    check( cublas_trans_const( MagmaConjTrans     ) == CUBLAS_OP_C            );

    check( cublas_uplo_const(  MagmaUpper         ) == CUBLAS_FILL_MODE_UPPER );
    check( cublas_uplo_const(  MagmaLower         ) == CUBLAS_FILL_MODE_LOWER );

    check( cublas_diag_const(  MagmaNonUnit       ) == CUBLAS_DIAG_NON_UNIT   );
    check( cublas_diag_const(  MagmaUnit          ) == CUBLAS_DIAG_UNIT       );

    check( cublas_side_const(  MagmaLeft          ) == CUBLAS_SIDE_LEFT       );
    check( cublas_side_const(  MagmaRight         ) == CUBLAS_SIDE_RIGHT      );
    printf( "MAGMA  -> cublas_xxxxx_const    %s\n", (s == gStatus ? "ok" : "failed"));
    #endif


    // ------------------------------------------------------------
    #ifdef HAVE_CBLAS
    s = gStatus;
    check( cblas_order_const( MagmaRowMajor      ) == CblasRowMajor  );
    check( cblas_order_const( MagmaColMajor      ) == CblasColMajor  );

    check( cblas_trans_const( MagmaNoTrans       ) == CblasNoTrans   );
    check( cblas_trans_const( MagmaTrans         ) == CblasTrans     );
    check( cblas_trans_const( MagmaConjTrans     ) == CblasConjTrans );

    check( cblas_uplo_const(  MagmaUpper         ) == CblasUpper     );
    check( cblas_uplo_const(  MagmaLower         ) == CblasLower     );

    check( cblas_diag_const(  MagmaNonUnit       ) == CblasNonUnit   );
    check( cblas_diag_const(  MagmaUnit          ) == CblasUnit      );

    check( cblas_side_const(  MagmaLeft          ) == CblasLeft      );
    check( cblas_side_const(  MagmaRight         ) == CblasRight     );
    printf( "MAGMA  -> cblas_xxxxx_const     %s\n", (s == gStatus ? "ok" : "failed"));
    #endif

    return gStatus;
}
