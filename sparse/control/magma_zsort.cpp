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


#define SWAP(a, b)  { tmp = val[a]; val[a] = val[b]; val[b] = tmp; }
#define SWAPM(a, b) { tmpv = val[a]; val[a] = val[b]; val[b] = tmpv;  \
                      tmpc = col[a]; col[a] = col[b]; col[b] = tmpc;  \
                      tmpr = row[a]; row[a] = row[b]; row[b] = tmpr; }

#define UP 0
#define DOWN 1

/**
    Purpose
    -------

    Sorts an array of values in increasing order.

    Arguments
    ---------

    @param[in,out]
    x           magmaDoubleComplex*
                array to sort

    @param[in]
    first       magma_int_t
                pointer to first element

    @param[in]
    last        magma_int_t
                pointer to last element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zsort(
    magmaDoubleComplex *x,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magmaDoubleComplex temp;
    magma_index_t pivot,j,i;

    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while( MAGMA_Z_ABS(x[i]) <= MAGMA_Z_ABS(x[pivot]) && i<last )
                i++;
            while( MAGMA_Z_ABS(x[j]) > MAGMA_Z_ABS(x[pivot]) )
                j--;
            if( i<j ){
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;
        CHECK( magma_zsort( x, first, j-1, queue ));
        CHECK( magma_zsort( x, j+1, last, queue ));
    }
cleanup:
    return info;
}


/**
    Purpose
    -------

    Sorts an array of values in increasing order.

    Arguments
    ---------

    @param[in,out]
    x           magmaDoubleComplex*
                array to sort

    @param[in,out]
    col         magma_index_t*
                Target array, will be modified during operation.

    @param[in,out]
    row         magma_index_t*
                Target array, will be modified during operation.

    @param[in]
    first       magma_int_t
                pointer to first element

    @param[in]
    last        magma_int_t
                pointer to last element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zmsort(
    magmaDoubleComplex *x,
    magma_index_t *col,
    magma_index_t *row,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magmaDoubleComplex temp;
    magma_index_t pivot,j,i, tmpcol, tmprow;

    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while( MAGMA_Z_ABS(x[i]) <= MAGMA_Z_ABS(x[pivot]) && i<last )
                i++;
            while( MAGMA_Z_ABS(x[j]) > MAGMA_Z_ABS(x[pivot]) )
                j--;
            if( i<j ){
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
                tmpcol = col[i];
                col[i] = col[j];
                col[j] = tmpcol;
                tmprow = row[i];
                row[i] = row[j];
                row[j] = tmprow;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;
        CHECK( magma_zmsort( x, col, row, first, j-1, queue ));
        CHECK( magma_zmsort( x, col, row, j+1, last, queue ));
    }
cleanup:
    return info;
}


/**
    Purpose
    -------

    Sorts an array of integers in increasing order.

    Arguments
    ---------

    @param[in,out]
    x           magma_index_t*
                array to sort

    @param[in]
    first       magma_int_t
                pointer to first element

    @param[in]
    last        magma_int_t
                pointer to last element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zindexsort(
    magma_index_t *x,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_index_t pivot,j,temp,i;

    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while( x[i]<=x[pivot] && i<last )
                i++;
            while( x[j]>x[pivot] )
                j--;
            if( i<j ){
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;
        CHECK( magma_zindexsort( x, first, j-1, queue ));
        CHECK( magma_zindexsort( x, j+1, last, queue ));
    }
cleanup:
    return info;
}


/**
    Purpose
    -------

    Sorts an array of integers, updates a respective array of values.

    Arguments
    ---------

    @param[in,out]
    x           magma_index_t*
                array to sort

    @param[in,out]
    y           magmaDoubleComplex*
                array to sort

    @param[in]
    first       magma_int_t
                pointer to first element

    @param[in]
    last        magma_int_t
                pointer to last element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zindexsortval(
    magma_index_t *x,
    magmaDoubleComplex *y,
    magma_int_t first,
    magma_int_t last,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_index_t pivot,j,temp,i;
    magmaDoubleComplex tempval;

    if(first<last){
         pivot=first;
         i=first;
         j=last;

        while(i<j){
            while( x[i]<=x[pivot] && i<last )
                i++;
            while( x[j]>x[pivot] )
                j--;
            if( i<j ){
                temp = x[i];
                x[i] = x[j];
                x[j] = temp;
                tempval = y[i];
                y[i] = y[j];
                y[j] = tempval;
            }
        }

        temp=x[pivot];
        x[pivot]=x[j];
        x[j]=temp;

        tempval=y[pivot];
        y[pivot]=y[j];
        y[j]=tempval;
        CHECK( magma_zindexsortval( x, y, first, j-1, queue ));
        CHECK( magma_zindexsortval( x, y, j+1, last, queue ));
    }
cleanup:
    return info;
}



/**
    Purpose
    -------

    Identifies the kth smallest/largest element in an array and reorders
    such that these elements come to the front. The related arrays col and row
    are also reordered.

    Arguments
    ---------

    @param[in,out]
    val         magmaDoubleComplex*
                Target array, will be modified during operation.

    @param[in,out]
    col         magma_index_t*
                Target array, will be modified during operation.

    @param[in,out]
    row         magma_index_t*
                Target array, will be modified during operation.

    @param[in]
    length      magma_int_t
                Length of the target array.

    @param[in]
    k           magma_int_t
                Element to be identified (largest/smallest).

    @param[in]
    r           magma_int_t
                rule how to sort: '1' -> largest, '0' -> smallest

    @param[out]
    element     magmaDoubleComplex*
                location of the respective element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zmorderstatistics(
    magmaDoubleComplex *val,
    magma_index_t *col,
    magma_index_t *row,
    magma_int_t length,
    magma_int_t k,
    magma_int_t r,
    magmaDoubleComplex *element,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    magma_int_t i, st;
    magmaDoubleComplex tmpv;
    magma_index_t tmpc, tmpr;
    if( r == 0 ){
        for ( st = i = 0; i < length - 1; i++ ) {
            if ( magma_z_isnan_inf( val[i]) ) {
                printf("%% error: array contains %f + %fi.\n", MAGMA_Z_REAL(val[i]), MAGMA_Z_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_Z_ABS(val[i]) > MAGMA_Z_ABS(val[length-1]) ){
                continue;
            }
            SWAPM(i, st);
            st++;
        }

        SWAPM(length-1, st);

        if ( k == st ){
            *element = val[st];
        }
        else if ( st > k ) {
            CHECK( magma_zmorderstatistics( val, col, row, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_zmorderstatistics( val+st, col+st, row+st, length-st, k-st, r, element, queue ));
        }
    } else {
        for ( st = i = 0; i < length - 1; i++ ) {
            if ( magma_z_isnan_inf( val[i]) ) {
                printf("%% error: array contains %f + %fi.\n", MAGMA_Z_REAL(val[i]), MAGMA_Z_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_Z_ABS(val[i]) < MAGMA_Z_ABS(val[length-1]) ){
                continue;
            }
            SWAPM(i, st);
            st++;
        }

        SWAPM(length-1, st);

        if ( k == st ){
            *element = val[st];
        }
        else if ( st > k ) {
            CHECK( magma_zmorderstatistics( val, col, row, st, k, r, element, queue ));
        }
        else {
            CHECK( magma_zmorderstatistics( val+st, col+st, row+st, length-st, k-st, r, element, queue ));
        }
    }

cleanup:
    return info;
}



/**
    Purpose
    -------

    Identifies the kth smallest/largest element in an array.

    Arguments
    ---------

    @param[in,out]
    val         magmaDoubleComplex*
                Target array, will be modified during operation.

    @param[in]
    length      magma_int_t
                Length of the target array.

    @param[in]
    k           magma_int_t
                Element to be identified (largest/smallest).

    @param[in]
    r           magma_int_t
                rule how to sort: '1' -> largest, '0' -> smallest

    @param[out]
    element     magmaDoubleComplex*
                location of the respective element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zorderstatistics(
    magmaDoubleComplex *val,
    magma_int_t length,
    magma_int_t k,
    magma_int_t r,
    magmaDoubleComplex *element,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t i, st;
    magmaDoubleComplex tmp;
    if( r == 0 ){
        for ( st = i = 0; i < length - 1; i++ ) {
            if ( magma_z_isnan_inf( val[i]) ) {
                printf("%%error: array contains %f + %fi.\n", MAGMA_Z_REAL(val[i]), MAGMA_Z_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_Z_ABS(val[i]) > MAGMA_Z_ABS(val[length-1]) ){
                continue;
            }
            SWAP(i, st);
            st++;
        }

        SWAP(length-1, st);

        if ( k == st ){
            *element = val[st];
        }
        else if ( st > k ) {
            CHECK( magma_zorderstatistics( val, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_zorderstatistics( val+st, length-st, k-st, r, element, queue ));
        }
    } else {
        for ( st = i = 0; i < length - 1; i++ ) {
            if ( magma_z_isnan_inf( val[i]) ) {
                printf("%%error: array contains %f + %fi.\n", MAGMA_Z_REAL(val[i]), MAGMA_Z_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_Z_ABS(val[i]) < MAGMA_Z_ABS(val[length-1]) ){
                continue;
            }
            SWAP(i, st);
            st++;
        }

        SWAP(length-1, st);

        if ( k == st ){
            *element = val[st];
        }
        else if ( st > k ) {
            CHECK( magma_zorderstatistics( val, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_zorderstatistics( val+st, length-st, k-st, r, element, queue ));
        }
    }

cleanup:
    return info;
}



/**
    Purpose
    -------

    Approximates the k-th smallest element in an array by
    using order-statistics with step-size inc.

    Arguments
    ---------

    @param[in,out]
    val         magmaDoubleComplex*
                Target array, will be modified during operation.

    @param[in]
    length      magma_int_t
                Length of the target array.

    @param[in]
    k           magma_int_t
                Element to be identified (largest/smallest).

    @param[in]
    inc         magma_int_t
                Stepsize in the approximation.

    @param[in]
    r           magma_int_t
                rule how to sort: '1' -> largest, '0' -> smallest

    @param[out]
    element     magmaDoubleComplex*
                location of the respective element

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zorderstatistics_inc(
    magmaDoubleComplex *val,
    magma_int_t length,
    magma_int_t k,
    magma_int_t inc,
    magma_int_t r,
    magmaDoubleComplex *element,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_int_t i, st;
    magmaDoubleComplex tmp;
    if( r == 0 ){
        for ( st = i = 0; i < length - inc; i=i+inc ) {
            if ( magma_z_isnan_inf( val[i]) ) {
                printf("%%error: array contains %f + %fi.\n", MAGMA_Z_REAL(val[i]), MAGMA_Z_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_Z_ABS(val[i]) > MAGMA_Z_ABS(val[length-inc]) ){
                continue;
            }
            SWAP(i, st);
            st=st+inc;
        }

        SWAP(length-inc, st);

        if ( k == st ){
            *element = val[st];
        }
        else if ( st > k ) {
            CHECK( magma_zorderstatistics( val, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_zorderstatistics( val+st, length-st, k-st, r, element, queue ));
        }
    } else {
        for ( st = i = 0; i < length - inc; i=i+inc ) {
            if ( magma_z_isnan_inf( val[i]) ) {
                printf("%%error: array contains %f + %fi.\n", MAGMA_Z_REAL(val[i]), MAGMA_Z_IMAG(val[i]) );
                info = MAGMA_ERR_NAN;
                goto cleanup;
            }
            if ( MAGMA_Z_ABS(val[i]) < MAGMA_Z_ABS(val[length-1]) ){
                continue;
            }
            SWAP(i, st);
            st=st+inc;
        }

        SWAP(length-inc, st);

        if ( k == st ){
            *element = val[st];
        }
        else if ( st > k ) {
            CHECK( magma_zorderstatistics( val, st, k, r, element, queue ));
        }
        else {
             CHECK( magma_zorderstatistics( val+st, length-st, k-st, r, element, queue ));
        }
    }

cleanup:
    return info;
}




void swap(magmaDoubleComplex *a, magmaDoubleComplex *b)
{
    magmaDoubleComplex t;
    t = *a;
    *a = *b;
    *b = t;
}


/**
    Purpose
    -------

    Approximates the k-th smallest element in an array by
    using order-statistics with step-size inc.

    Arguments
    ---------

    @param[in]
    start       magma_int_t
                Start position of the target array.

    @param[in]
    length      magma_int_t
                Length of the target array.

    @param[in,out]
    seq         magmaDoubleComplex*
                Target array, will be modified during operation.

    @param[in]
    flag        magma_int_t
                ???

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zbitonic_sort(
    magma_int_t start,
    magma_int_t length,
    magmaDoubleComplex *seq,
    magma_int_t flag,
    magma_queue_t queue )
{

    magma_int_t info =0;

    magma_int_t m, i, num_threads=1;
    magma_int_t split_length;

#ifdef _OPENMP
    #pragma omp parallel
    {
        num_threads = omp_get_max_threads();
    }
#endif

    m = length/num_threads;


    if (length == 1)
        return 0;

    if (length % 2 !=0 )
    {
        printf("The length of a (sub)sequence can not be divided by 2.\n");
        info = MAGMA_ERR;
        goto cleanup;
    }

    split_length = length / 2;

    // bitonic split
    #pragma omp parallel for shared(seq, flag, start, split_length) private(i)
    for (i = start; i < start + split_length; i++)
    {
        if (flag == UP)
        {
            if (MAGMA_Z_ABS(seq[i]) > MAGMA_Z_ABS(seq[i + split_length]))
                swap(&seq[i], &seq[i + split_length]);
        }
        else
        {
            if (MAGMA_Z_ABS(seq[i]) < MAGMA_Z_ABS(seq[i + split_length]))
                swap(&seq[i], &seq[i + split_length]);
        }
    }

    if (split_length > m)
    {
        // m is the size of sub part-> n/numThreads
        magma_zbitonic_sort(start, split_length, seq, flag, queue);
        magma_zbitonic_sort(start + split_length, split_length, seq, flag, queue);
    }

cleanup:
    return info;
}
