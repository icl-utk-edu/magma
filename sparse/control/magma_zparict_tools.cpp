/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt

*/

#include "magmasparse_internal.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define SWAP(a, b)  { tmp = a; a = b; b = tmp; }
#define SWAP_INT(a, b)  { tmpi = a; a = b; b = tmpi; }

#define AVOID_DUPLICATES
//#define NANCHECK



/***************************************************************************//**
    Purpose
    -------
    This function identifies the candidates like they appear as ILU1 fill-in.
    In this version, the matrices are assumed unordered,
    the linked list is traversed to acces the entries of a row.

    Arguments
    ---------

    @param[in]
    L0          magma_z_matrix
                tril( ILU(0) ) pattern of original system matrix.
                
    @param[in]
    L           magma_z_matrix
                Current lower triangular factor.

    @param[in]
    LT          magma_z_matrix
                Transose of the lower triangular factor.

    @param[in,out]
    L_new       magma_z_matrix*
                List of candidates for L in COO format.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/

extern "C" magma_int_t
magma_zparict_candidates(
    magma_z_matrix L0,
    magma_z_matrix L,
    magma_z_matrix LT,
    magma_z_matrix *L_new,
    magma_queue_t queue )
{
    
    magma_int_t info = 0;
    magma_index_t *insertedL;
    double thrs = 1e-8;
    
    magma_int_t orig = 1; // the pattern L0 and U0 is considered
    magma_int_t existing = 0; // existing elements are also considered
    magma_int_t ilufill = 1;
    
    // magma_int_t num_threads;
    // 
    // #pragma omp parallel
    // {
    //     num_threads = omp_get_max_threads();
    // }
    
    // for now: also some part commented out. If it turns out
    // this being correct, I need to clean up the code.

    CHECK( magma_index_malloc_cpu( &L_new->row, L.num_rows+1 ));
    CHECK( magma_index_malloc_cpu( &insertedL, L.num_rows+1 ));
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<L.num_rows+1; i++ ){
        L_new->row[i] = 0;
        insertedL[i] = 0;
    }
    L_new->num_rows = L.num_rows;
    L_new->num_cols = L.num_cols;
    L_new->storage_type = Magma_CSR;
    L_new->memory_location = Magma_CPU;
    
    // go over the original matrix - this is the only way to allow elements to come back...
    if( orig == 1 ){
       #pragma omp parallel for
        for( magma_index_t row=0; row<L0.num_rows; row++){
            magma_int_t numaddrowL = 0;
            magma_int_t ilu0 = L0.row[row];
            magma_int_t ilut = L.row[row];
            magma_int_t endilu0 = L0.row[ row+1 ];
            magma_int_t endilut = L.row[ row+1 ]; 
            magma_int_t ilu0col;
            magma_int_t ilutcol;
            do{
                ilu0col = L0.col[ ilu0 ];
                ilutcol = L.col[ ilut ];
                if( ilu0col == ilutcol ){
                    ilu0++;
                    ilut++;
                    if( existing==1 )
                        numaddrowL++;
                }
                else if( ilutcol<ilu0col ){
                    ilut++;
                    if( existing==1 )
                        numaddrowL++;
                }
                else {
                    // this element is missing in the current approximation
                    // mark it as candidate
                    numaddrowL++;
                    ilu0++;
                }
            }while( ilut<endilut && ilu0<endilu0 );
            // do the rest if existing
            if( ilu0<endilu0 ){
                do{
                    numaddrowL++;
                    ilu0++;
                }while( ilu0<endilu0 ); 
            }
            L_new->row[ row+1 ] = L_new->row[ row+1 ]+numaddrowL;
        }
    } // end original
    
    if( ilufill == 1 ){
        // how to determine candidates:
        // for each node i, look at any "intermediate" neighbor nodes numbered
        // less, and then see if this neighbor has another neighbor j numbered
        // more than the intermediate; if so, fill in is (i,j) if it is not
        // already nonzero
        #pragma omp parallel for
        for( magma_index_t row=0; row<L.num_rows; row++){
            magma_int_t numaddrowL = 0;
            // loop first element over row - only for elements smaller the diagonal
            for( magma_index_t el1=L.row[row]; el1<L.row[row+1]-1; el1++ ){
                magma_index_t col1 = L.col[ el1 ];
                // now check the upper triangular
                // second loop first element over row - only for elements larger the intermediate
                for( magma_index_t el2 = LT.row[ col1 ]+1; el2 < LT.row[ col1+1 ]; el2++ ){
                    magma_index_t col2 = LT.col[ el2 ];
                    magma_index_t cand_row = row;
                    magma_index_t cand_col = col2;
                    // check whether this element already exists
                    // first case: part of L
                    if( cand_col < row ){
                        // check whether this element already exists in L
                        magma_int_t exist = 0;
                        if( existing == 0 ){
                            for(magma_index_t k=L.row[cand_row]; k<L.row[cand_row+1]; k++ ){
                                if( L.col[ k ] == cand_col ){
                                        exist = 1;
                                        break;
                                }
                            }
                        }
                        // if it does not exist, increase counter for this location
                        // use the entry one further down to allow for parallel insertion
                        if( exist == 0 ){
                            //printf("checked row: %d this element does not yet exist in L: (%d,%d)\n", cand_row, cand_col);
                            numaddrowL++;
                            //numaddL[ row+1 ]++;
                        }
                    } else {
                        ;
                        // we don't check if row>col. The motivation is that the element
                        // will in this case be identified as candidate from a differen row.
                        // Due to symmetry we have:
                        // Assume we have a fill-in element (i,j) with i<j
                        // Then there exists a k such that A(i,k)\neq 0 and A(k,j)\neq 0.
                        // Due to symmetry, we then also have A(k,i)\neq 0 and A(j,k)\neq 0.
                        // as a result, A(j,i) will be identified as fill-in also from row j
                        // in which i i the column and i<j. 
                    }
                }
    
            }
            L_new->row[ row+1 ] = L_new->row[ row+1 ]+numaddrowL;
        }
    } // end ilu-fill
    //end = magma_sync_wtime( queue ); printf("llop 1.2 : %.2e\n", end-start);
    // #########################################################################

    // get the total candidate count
    L_new->nnz = 0;
    L_new->row[ 0 ] = L_new->nnz;

    #pragma omp parallel
    {
#ifdef _OPENMP
        magma_int_t id = omp_get_thread_num();
#else
        magma_int_t id = 0;
#endif
        if( id == 0 ){
            for( magma_int_t i = 0; i<L.num_rows; i++ ){
                L_new->nnz = L_new->nnz + L_new->row[ i+1 ];
                L_new->row[ i+1 ] = L_new->nnz;
            }
        }
    }
    
    magma_zmalloc_cpu( &L_new->val, L_new->nnz );
    magma_index_malloc_cpu( &L_new->rowidx, L_new->nnz );
    magma_index_malloc_cpu( &L_new->col, L_new->nnz );
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<L_new->nnz; i++ ){
        L_new->val[i] = MAGMA_Z_ZERO;
    }
    
    #pragma omp parallel for
    for( magma_int_t i=0; i<L_new->nnz; i++ ){
        L_new->col[i] = -1;
        L_new->rowidx[i] = -1;
    }

    // #########################################################################

    
    if( orig == 1 ){
        
       #pragma omp parallel for
        for( magma_index_t row=0; row<L0.num_rows; row++){
            magma_int_t laddL = 0;
            magma_int_t offsetL = L_new->row[row];
            magma_int_t ilu0 = L0.row[row];
            magma_int_t ilut = L.row[row];
            magma_int_t endilu0 = L0.row[ row+1 ];
            magma_int_t endilut = L.row[ row+1 ]; 
            magma_int_t ilu0col;
            magma_int_t ilutcol;
            do{
                ilu0col = L0.col[ ilu0 ];
                ilutcol = L.col[ ilut ];
                if( ilu0col == ilutcol ){
                    ilu0++;
                    ilut++;
                    if( existing==1 ){
                        L_new->col[ offsetL + laddL ] = ilu0col;
                        L_new->rowidx[ offsetL + laddL ] = row;
                        L_new->val[ offsetL + laddL ] = MAGMA_Z_ONE;
                        laddL++;
                    }
                }
                else if( ilutcol<ilu0col ){
                    if( existing==1 ){
                        L_new->col[ offsetL + laddL ] = ilutcol;
                        L_new->rowidx[ offsetL + laddL ] = row;
                        L_new->val[ offsetL + laddL ] = MAGMA_Z_ONE;
                        laddL++;
                    }
                    ilut++;
                }
                else {
                    // this element is missing in the current approximation
                    // mark it as candidate
                    L_new->col[ offsetL + laddL ] = ilu0col;
                    L_new->rowidx[ offsetL + laddL ] = row;
                    L_new->val[ offsetL + laddL ] = MAGMA_Z_ONE + MAGMA_Z_ONE + MAGMA_Z_ONE;
                    laddL++;
                    ilu0++;
                }
            }while( ilut<endilut && ilu0<endilu0 );
            if( ilu0<endilu0 ){
                do{
                    ilu0col = L0.col[ ilu0 ];
                    L_new->col[ offsetL + laddL ] = ilu0col;
                    L_new->rowidx[ offsetL + laddL ] = row;
                    L_new->val[ offsetL + laddL ] = MAGMA_Z_ONE + MAGMA_Z_ONE + MAGMA_Z_ONE;
                    laddL++;
                    ilu0++;
                }while( ilu0<endilu0 ); 
            }
            insertedL[row] = laddL;
        }
        
    } // end original
    
    if( ilufill==1 ){
        #pragma omp parallel for
        for( magma_index_t row=0; row<L.num_rows; row++){
            magma_int_t laddL = 0;
            magma_int_t offsetL = L_new->row[row] + insertedL[row];
            // loop first element over row - only for elements smaller the diagonal
            for( magma_index_t el1=L.row[row]; el1<L.row[row+1]-1; el1++ ){
                
                magma_index_t col1 = L.col[ el1 ];
                // now check the upper triangular
                // second loop first element over row - only for elements larger the intermediate
                for( magma_index_t el2 = LT.row[ col1 ]+1; el2 < LT.row[ col1+1 ]; el2++ ){
                    magma_index_t col2 = LT.col[ el2 ];
                    magma_index_t cand_row = row;
                    magma_index_t cand_col = col2;
                    //$########### we now have the candidate cand_row cand_col
                    
                    
                    // check whether this element already exists
                    // first case: part of L
                    if( cand_col < row ){
                        magma_int_t exist = 0;
                        if( existing == 0 ){
                            for(magma_index_t k=L.row[cand_row]; k<L.row[cand_row+1]; k++ ){
                                if( L.col[ k ] == cand_col ){
                                        exist = 1;
                                        break;
                                }
                            }
                        }
    #ifdef AVOID_DUPLICATES
                        for( magma_int_t k=L_new->row[cand_row]; k<L_new->row[cand_row+1]; k++){
                            if( L_new->col[ k ] == cand_col ){
                                // element included in LU and nonzero
                                exist = 1;
                                break;
                            }
                        }
    #endif
                        // if it does not exist, increase counter for this location
                        // use the entry one further down to allow for parallel insertion
                        if( exist == 0 ){
                            //  printf("---------------->>>  candidate in L at (%d, %d)\n", cand_row, cand_col);
                            //add in the next location for this row
                            // L_new->val[ numaddL[row] + laddL ] =  MAGMA_Z_MAKE(1e-14,0.0);
                            L_new->rowidx[ offsetL + laddL ] = cand_row;
                            L_new->col[ offsetL + laddL ] = cand_col;
                            L_new->val[ offsetL + laddL ] = MAGMA_Z_ONE;
                            // L_new->list[ numaddL[row] + laddL ] = -1;
                            // L_new->row[ numaddL[row] + laddL ] = -1;
                            laddL++;
                        }
                    } else {
                        ;
                    }
                }
            }
        }
    } //end ilufill
    
#ifdef AVOID_DUPLICATES
        // #####################################################################
        
        CHECK( magma_zparilut_thrsrm( 1, L_new, &thrs, queue ) );

        // #####################################################################
#endif

cleanup:
    magma_free_cpu( insertedL );
    return info;
}






/***************************************************************************//**
    Purpose
    -------
    This function does one synchronized ParILU sweep. Input and output are 
    different arrays.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix*
                System matrix.

    @param[in]
    L           magma_z_matrix*
                Current approximation for the lower triangular factor
                The format is sorted CSR.

    @param[in]
    U           magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is sorted CSC.
                
    @param[out]
    L_new       magma_z_matrix*
                Current approximation for the lower triangular factor
                The format is unsorted CSR.

    @param[out]
    U_new       magma_z_matrix*
                Current approximation for the upper triangular factor
                The format is unsorted CSC.

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
*******************************************************************************/


extern "C" magma_int_t
magma_zparict_sweep_sync(
    magma_z_matrix *A,
    magma_z_matrix *L,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    //printf("\n"); fflush(stdout);
    // parallel for using openmp
    
    // temporary vectors to swap the col/rowidx later
    // magma_index_t *tmpi;
    
    magmaDoubleComplex *L_new_val = NULL, *val_swap = NULL;
    
    CHECK( magma_zmalloc_cpu( &L_new_val, L->nnz ));
    
    #pragma omp parallel for
    for( magma_int_t e=0; e<L->nnz; e++){
        magma_int_t i,j,icol,jcol;//,jold;

        magma_index_t row = L->rowidx[ e ];
        magma_index_t col = L->col[ e ];

        magmaDoubleComplex A_e = MAGMA_Z_ZERO;
        // check whether A contains element in this location
        for( i = A->row[row]; i<A->row[row+1]; i++){
            if( A->col[i] == col ){
                A_e = A->val[i];
                break;
            }
        }

        //now do the actual iteration
        i = L->row[ row ];
        j = L->row[ col ];
        magma_int_t endi = L->row[ row+1 ];
        magma_int_t endj = L->row[ col+1 ];
        magmaDoubleComplex sum = MAGMA_Z_ZERO;
        magmaDoubleComplex lsum = MAGMA_Z_ZERO;
        while( i<endi && j<endj ){
            lsum = MAGMA_Z_ZERO;
            //jold = j;
            icol = L->col[i];
            jcol = L->col[j];
            if( icol == jcol ){
                lsum = L->val[i] * L->val[j];
                sum = sum + lsum;
                i++;
                j++;
            }
            else if( icol<jcol ){
                i++;
            }
            else {
                j++;
            }
        }
        sum = sum - lsum;
        if( row == col ){
            // write back to location e
            L_new_val[ e ] = MAGMA_Z_MAKE( sqrt( fabs( MAGMA_Z_REAL(A_e - sum) )), 0.0 );
        } else {
            // write back to location e
            L_new_val[ e ] =  ( A_e - sum ) / L->val[endj-1];
        }
    }// end omp parallel section
    
    val_swap = L_new_val;
    L_new_val = L->val;
    L->val = val_swap;
    
    magma_free_cpu( L_new_val );
    
cleanup:
    return info;
}

