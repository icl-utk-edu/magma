/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/

//  in this file, many routines are taken from
//  the IO functions provided by MatrixMarket

#include "magmasparse_internal.h"


/******************************************************************************
 * ILU mex function from MATLAB:
 *
 * [l, u] = ilu_mex(a, level, omega, storage);
 *****************************************************************************/

#define mwIndex magma_index_t



void magma_zshell_sort(
    const magma_int_t n, magma_index_t *x)
{
    magma_int_t m, max, j, k, itemp;

    m = n/2;

    while (m > 0) {
        max = n - m;
        for (j=0; j<max; j++)
        {
            for (k=j; k>=0; k-=m)
            {
                if (x[k+m] >= x[k])
                    break;
                itemp = x[k+m];
                x[k+m] = x[k];
                x[k] = itemp;
            }
        }
        m = m/2;
    }
}

/*
// symbolic level ILU
// factors magma_int_to separate upper and lower parts
// sorts the entries in each row of A by index
// assumes no zero rows
*/

extern "C"
magma_int_t
magma_zsymbolic_ilu(
    const magma_int_t levfill,                 /* level of fill */
    const magma_int_t n,                       /* order of matrix */
    magma_int_t *nzl,                          /* input-output */
    magma_int_t *nzu,                          /* input-output */
    const mwIndex *ia,
    const mwIndex *ja,    /* input */
    mwIndex *ial,
    mwIndex *jal,              /* output lower factor structure */
    mwIndex *iau,
    mwIndex *jau)              /* output upper factor structure */
{
    magma_int_t info = 0;
    
    magma_int_t i;
    magma_index_t *lnklst=NULL;
    magma_index_t *curlev=NULL;
    magma_index_t *levels=NULL;
    magma_index_t *iwork=NULL;
    
    magma_int_t knzl = 0;
    magma_int_t knzu = 0;

    CHECK( magma_index_malloc_cpu( &lnklst, n ));
    CHECK( magma_index_malloc_cpu( &curlev, n ));
    CHECK( magma_index_malloc_cpu( &levels, *nzu ));
    CHECK( magma_index_malloc_cpu( &iwork, n ));
    
    for(magma_int_t t=0; t<n; t++){
        lnklst[t] = 0;
        curlev[t] = 0;
        iwork[t] = 0;
    }

    for(magma_int_t t=0; t<*nzu; t++){
        levels[t] = 0;
    }

    ial[0] = 0;
    iau[0] = 0;

    for (i=0; i<n; i++) {
        //printf("check line %d\n", i);
        magma_int_t first, next, j;

        /* copy column indices of row into workspace and sort them */

        magma_int_t len = ia[i+1] - ia[i];
        next = 0;
        for (j=ia[i]; j<ia[i+1]; j++)
            iwork[next++] = ja[j];
        magma_zshell_sort(len, iwork);
        //printf("check2 line %d\n", i);
        /* construct implied linked list for row */

        first = iwork[0];
        curlev[first] = 0;

        for (j=0; j<=len-2; j++)
        {
            lnklst[iwork[j]] = iwork[j+1];
            curlev[iwork[j]] = 0;
        }
        // printf("check3 line %d iwork[len-1]:%d\n", i, iwork[len-1]);
        lnklst[iwork[len-1]] = n;
        curlev[iwork[len-1]] = 0;

        /* merge with rows in U */
        // printf("check4 line %d lnklst[iwork[len-1]]:%d\n", i, lnklst[iwork[len-1]]);
        next = first;
        // printf("next:%d (!<) first:%d\n", next, i);
        while (next < i)
        {
          //  printf("check line %d while %d\n", i, next);
            magma_int_t oldlst = next;
            magma_int_t nxtlst = lnklst[next];
            magma_int_t row = next;
            magma_int_t ii;

            /* scan row */

            for (ii=iau[row]+1; ii<iau[row+1]; /*nop*/)
            {
                if (jau[ii] < nxtlst)
                {
                    /* new fill-in */
                    magma_int_t newlev = curlev[row] + levels[ii] + 1;
                    if (newlev <= levfill)
                    {
                        lnklst[oldlst]  = jau[ii];
                        lnklst[jau[ii]] = nxtlst;
                        oldlst = jau[ii];
                        curlev[jau[ii]] = newlev;
                    }
                    ii++;
                }
                else if (jau[ii] == nxtlst)
                {
                    magma_int_t newlev;
                    oldlst = nxtlst;
                    nxtlst = lnklst[oldlst];
                    newlev = curlev[row] + levels[ii] + 1;
                    curlev[jau[ii]] = min( curlev[jau[ii]], newlev );
                    ii++;
                }
                else /* (jau[ii] > nxtlst) */
                {
                    oldlst = nxtlst;
                    nxtlst = lnklst[oldlst];
                }
            }
            next = lnklst[next];
        }
        
        /* gather the pattern magma_int_to L and U */
       // printf("check line5 %d\n", i);
        next = first;
        while (next < i)
        {
            if (knzl >= *nzl) {
                printf("ILU: STORAGE parameter value %d<%d too small.\n", int(*nzl), int(knzl));
                printf("Increase STORAGE parameter.\n");
                info = -1;
                goto cleanup;
            }
            jal[knzl++] = next;
            next = lnklst[next];
        }
        ial[i+1] = knzl;
        //  printf("check line6 %d\n", i);
        if (next != i)
        {
            printf("ILU structurally singular.\n");
            /*
                assert(knzu < *nzu);
                levels[knzu] = 2*n;
                jau[knzu++] = i;
            */
        }
        //  printf("check line7 %d\n", i);
        //                 printf("next:%d  n:%d \n", next, n);
        while (next < n)
        {
            if (knzu >= *nzu) {
                printf("ILU: STORAGE parameter value %d < %d too small.\n", int(*nzu), int(knzu));
                printf("Increase STORAGE parameter.\n");
                info = -1;
                goto cleanup;
            }
            // printf("1 knzu:%d  next:%d \n", knzu, next );
            levels[knzu] = curlev[next];
            //  printf("2 knzu:%d  next:%d \n", knzu, next );
            jau[knzu++] = next;
            //  printf("3 knzu:%d  next:%d \n", knzu, next );
            next = lnklst[next];
            //  printf("4 next:%d  n:%d \n", next, n);
        }
        iau[i+1] = knzu;
    }

    *nzl = knzl;
    *nzu = knzu;

    // printf("ende\n");

#if 0
    printf( "Actual nnz for ILU: %d\n", *nzl + *nzu );
#endif

cleanup:
    magma_free_cpu(lnklst);
    magma_free_cpu(curlev);
    magma_free_cpu(levels);
    magma_free_cpu(iwork);
    
    return info;
}



/******************************************************************************
 *
 * MEX function
 *
 *****************************************************************************/

void magma_zmexFunction(magma_int_t nlhs, magma_int_t n, magmaDoubleComplex omega,
                 magma_int_t levfill, magma_int_t storage,
                magma_index_t * ial, magma_index_t *jal, magmaDoubleComplex *al,
                magma_index_t * iau, magma_index_t *jau, magmaDoubleComplex *au,
                magma_int_t nrhs,
                magma_index_t * ia, magma_index_t *ja, magmaDoubleComplex *a )
{
    /* matrix is stored in CSC format, 0-based */

    magma_int_t nzl, nzu;

    
    nzl = storage;
    nzu = storage;


    /* the following will fail and return to matlab if insufficient storage */
    magma_zsymbolic_ilu(levfill, n, &nzl, &nzu, ia, ja, ial, jal, iau, jau);
}

/* shell sort
// stable, so it is fast if already sorted
// sorts x[0:n-1] in place, ascending order.
*/



/**
    Purpose
    -------

    This routine performs a symbolic ILU factorization.
    The algorithm is taken from an implementation written by Edmond Chow.

    Arguments
    ---------
    @param[in,out]
    A           magma_z_matrix*
                matrix in magma sparse matrix format containing the original
                matrix on input, and L,U on output

    @param[in]
    levels      magma_magma_int_t_t
                fill in level

    @param[out]
    L           magma_z_matrix*
                output lower triangular matrix in magma sparse matrix format
                empty on function call

    @param[out]
    U           magma_z_matrix*
                output upper triangular matrix in magma sparse matrix format
                empty on function call
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zsymbilu(
    magma_z_matrix *A,
    magma_int_t levels,
    magma_z_matrix *L,
    magma_z_matrix *U,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix A_copy={Magma_CSR}, B={Magma_CSR};
    magma_z_matrix hA={Magma_CSR}, CSRCOOA={Magma_CSR};
    
    // make sure the target structure is empty
    magma_zmfree( L, queue );
    magma_zmfree( U, queue );
    
    if( A->memory_location == Magma_CPU && A->storage_type == Magma_CSR ){
        CHECK( magma_zmtransfer( *A, &A_copy, Magma_CPU, Magma_CPU, queue ));
        CHECK( magma_zmtransfer( *A, &B, Magma_CPU, Magma_CPU, queue ));

        // possibility to scale to unit diagonal
        //magma_zmscale( &B, Magma_UNITDIAG );

        CHECK( magma_zmconvert( B, L, Magma_CSR, Magma_CSR , queue));
        CHECK( magma_zmconvert( B, U, Magma_CSR, Magma_CSR, queue ));

        magma_int_t num_lnnz = (levels > 0 ) ? B.nnz/2*(2*levels+50) : B.nnz;
        magma_int_t num_unnz = (levels > 0 ) ? B.nnz/2*(2*levels+50) : B.nnz;

        magma_free_cpu( L->col );
        magma_free_cpu( U->col );
        CHECK( magma_index_malloc_cpu( &L->col, num_lnnz ));
        CHECK( magma_index_malloc_cpu( &U->col, num_unnz ));

        magma_zsymbolic_ilu( levels, A->num_rows, &num_lnnz, &num_unnz, B.row, B.col,
                                            L->row, L->col, U->row, U->col );
        L->nnz = num_lnnz;
        U->nnz = num_unnz;
        magma_free_cpu( L->val );
        magma_free_cpu( U->val );
        CHECK( magma_zmalloc_cpu( &L->val, L->nnz ));
        CHECK( magma_zmalloc_cpu( &U->val, U->nnz ));
        for( magma_int_t i=0; i<L->nnz; i++ )
            L->val[i] = MAGMA_Z_MAKE( 0.0, 0.0 );

        for( magma_int_t i=0; i<U->nnz; i++ )
            U->val[i] = MAGMA_Z_MAKE( 0.0, 0.0 );
        // take the original values (scaled) as initial guess for L
        for(magma_int_t i=0; i<L->num_rows; i++){
            for(magma_int_t j=B.row[i]; j<B.row[i+1]; j++){
                magma_index_t lcol = B.col[j];
                for(magma_int_t k=L->row[i]; k<L->row[i+1]; k++){
                    if( L->col[k] == lcol ){
                        L->val[k] =  B.val[j];
                    }
                }
            }
        }

        // take the original values (scaled) as initial guess for U
        for(magma_int_t i=0; i<U->num_rows; i++){
            for(magma_int_t j=B.row[i]; j<B.row[i+1]; j++){
                magma_index_t lcol = B.col[j];
                for(magma_int_t k=U->row[i]; k<U->row[i+1]; k++){
                    if( U->col[k] == lcol ){
                        U->val[k] =  B.val[j];
                    }
                }
            }
        }
        magma_zmfree( &B, queue );
        // fill A with the new structure;
        magma_free_cpu( A->col );
        magma_free_cpu( A->val );
        CHECK( magma_index_malloc_cpu( &A->col, L->nnz+U->nnz ));
        CHECK( magma_zmalloc_cpu( &A->val, L->nnz+U->nnz ));
        A->nnz = L->nnz+U->nnz;
        
        magma_int_t z = 0;
        for(magma_int_t i=0; i<A->num_rows; i++){
            A->row[i] = z;
            for(magma_int_t j=L->row[i]; j<L->row[i+1]; j++){
                A->col[z] = L->col[j];
                A->val[z] = L->val[j];
                z++;
            }
            for(magma_int_t j=U->row[i]; j<U->row[i+1]; j++){
                A->col[z] = U->col[j];
                A->val[z] = U->val[j];
                z++;
            }
        }
        A->row[A->num_rows] = z;
        // reset the values of A to the original entries
        for(magma_int_t i=0; i<A->num_rows; i++){
            for(magma_int_t j=A_copy.row[i]; j<A_copy.row[i+1]; j++){
                magma_index_t lcol = A_copy.col[j];
                for(magma_int_t k=A->row[i]; k<A->row[i+1]; k++){
                    if( A->col[k] == lcol ){
                        A->val[k] =  A_copy.val[j];
                    }
                }
            }
        }
    }
    else {
        magma_storage_t A_storage = A->storage_type;
        magma_location_t A_location = A->memory_location;
        CHECK( magma_zmtransfer( *A, &hA, A->memory_location, Magma_CPU, queue ));
        CHECK( magma_zmconvert( hA, &CSRCOOA, hA.storage_type, Magma_CSR, queue ));

        CHECK( magma_zsymbilu( &CSRCOOA, levels, L, U, queue ));

        magma_zmfree( &hA, queue );
        magma_zmfree( A, queue );
        CHECK( magma_zmconvert( CSRCOOA, &hA, Magma_CSR, A_storage, queue ));
        CHECK( magma_zmtransfer( hA, A, Magma_CPU, A_location, queue ));
    }
    
cleanup:
    if( info != 0 ){
        magma_zmfree( L, queue );
        magma_zmfree( U, queue );
    }
    magma_zmfree( &A_copy, queue );
    magma_zmfree( &B, queue );
    magma_zmfree( &hA, queue );
    magma_zmfree( &CSRCOOA, queue );
    return info;
}
