/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c
       @author Hartwig Anzt
*/
#include "magmasparse_internal.h"

#define COMPLEX
#define PRECISION_z

/**
    Purpose
    -------

    Visualizes part of a vector of type magma_z_matrix.
    With input vector x , offset, visulen, the entries
    offset - (offset +  visulen) of x are visualized.

    Arguments
    ---------

    @param[in]
    x           magma_z_matrix
                vector to visualize

    @param[in]
    offset      magma_int_t
                start inex of visualization

    @param[in]
    visulen     magma_int_t
                number of entries to visualize

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zprint_vector(
    magma_z_matrix x,
    magma_int_t offset,
    magma_int_t  visulen,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix y={Magma_CSR};
    
    //**************************************************************
    magmaDoubleComplex c_zero = MAGMA_Z_ZERO;
    
    #ifdef COMPLEX
    #define magma_zprintval( tmp )       {                                  \
        if ( MAGMA_Z_EQUAL( tmp, c_zero )) {                                \
            printf( "   0.              \n" );                                \
        }                                                                   \
        else {                                                              \
            printf( " %8.4f+%8.4fi\n",                                        \
                    MAGMA_Z_REAL( tmp ), MAGMA_Z_IMAG( tmp ));              \
        }                                                                   \
    }
    #else
    #define magma_zprintval( tmp )       {                                  \
        if ( MAGMA_Z_EQUAL( tmp, c_zero )) {                                \
            printf( "   0.    \n" );                                          \
        }                                                                   \
        else {                                                              \
            printf( " %8.4f\n", MAGMA_Z_REAL( tmp ));                         \
        }                                                                   \
    }
    #endif
    //**************************************************************
    
    printf("visualize entries %d - %d of vector ",
                    int(offset), int(offset + visulen) );
    fflush(stdout);
    if ( x.memory_location == Magma_CPU ) {
        printf("located on CPU:\n");
        for( magma_int_t i=offset; i<offset + visulen; i++ )
            magma_zprintval(x.val[i]);
    }
    else if ( x.memory_location == Magma_DEV ) {
        printf("located on DEV:\n");
        CHECK( magma_zmtransfer( x, &y, Magma_DEV, Magma_CPU, queue ));
        for( magma_int_t i=offset; i<offset +  visulen; i++ )
            magma_zprintval(y.val[i]);
    }

cleanup:
    magma_free_cpu(y.val);
    return info;
}


/**
    Purpose
    -------

    Reads in a double vector of length "length".

    Arguments
    ---------

    @param[out]
    x           magma_z_matrix *
                vector to read in

    @param[in]
    length      magma_int_t
                length of vector
    @param[in]
    filename    char*
                file where vector is stored
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zvread(
    magma_z_matrix *x,
    magma_int_t length,
    char * filename,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_int_t nnz=0, i=0;
    FILE *fid;
    char buff[BUFSIZ]={0};
    int count=0;
    char *p;
    
    // make sure the target structure is empty
    magma_zmfree( x, queue );
    x->ownership = MagmaTrue;
    
    x->memory_location = Magma_CPU;
    x->storage_type = Magma_DENSE;
    x->num_rows = length;
    x->num_cols = 1;
    x->major = MagmaColMajor;
    
    fid = fopen(filename, "r");

    if(NULL==fgets(buff, BUFSIZ, fid))
        return -1;
    rewind(fid);
    for( p=buff; NULL != strtok(p, " \t\n"); p=NULL)
        count++;
    while( !feof(fid) )
    {
        double VAL1;

        magmaDoubleComplex VAL;
        
        #if defined(PRECISION_z) || defined(PRECISION_d)
            double VAL2;
            if( count == 2 ){
                fscanf(fid, "%lg %lg\n", &VAL1, &VAL2);
                VAL = MAGMA_Z_MAKE(VAL1, VAL2);
            }else{
                fscanf(fid, "%lg\n", &VAL1);
                VAL = MAGMA_Z_MAKE(VAL1, 0.0);  
            }
        #else // single-complex or single
            double VAL2;
            if( count == 2 ){
                fscanf(fid, "%g %g\n", &VAL1, &VAL2);
                VAL = MAGMA_Z_MAKE(VAL1, VAL2);
            }else{
                fscanf(fid, "%g\n", &VAL1);
                VAL = MAGMA_Z_MAKE(VAL1, 0.0);  
            }
        #endif
        
        if ( VAL != MAGMA_Z_ZERO )
            nnz++;
        i++;
    }
    
    x->num_rows = i;
    x->nnz = i;
    i=0; nnz=0; count=0;
    // printf("n:%d\n");
    
    CHECK( magma_zmalloc_cpu( &x->val, length ));
    
    rewind(fid);
    for( p=buff; NULL != strtok(p, " \t\n"); p=NULL)
        count++;
    //while( i<length )  // eof() is 'true' at the end of data
    while( !feof(fid) )
    {
        double VAL1;

        magmaDoubleComplex VAL;
        
        #if defined(PRECISION_z) || defined(PRECISION_d)
            double VAL2;
            if( count == 2 ){
                fscanf(fid, "%lg %lg\n", &VAL1, &VAL2);
                VAL = MAGMA_Z_MAKE(VAL1, VAL2);
            }else{
                fscanf(fid, "%lg\n", &VAL1);
                VAL = MAGMA_Z_MAKE(VAL1, 0.0);  
            }
        #else // single-complex or single
            double VAL2;
            if( count == 2 ){
                fscanf(fid, "%g %g\n", &VAL1, &VAL2);
                VAL = MAGMA_Z_MAKE(VAL1, VAL2);
            }else{
                fscanf(fid, "%g\n", &VAL1);
                VAL = MAGMA_Z_MAKE(VAL1, 0.0);  
            }
        #endif
        
        if ( VAL != MAGMA_Z_ZERO )
            nnz++;
        x->val[i] = VAL;
        i++;
    }
    fclose(fid);
    
    // magma_zprint_vector( *x, 0, x->num_rows, queue );
    
    
cleanup:
    return info;
}


/**
    Purpose
    -------

    Reads in a sparse vector-block stored in COO format.

    Arguments
    ---------

    @param[out]
    x           magma_z_matrix *
                vector to read in

    @param[in]
    filename    char*
                file where vector is stored
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zvspread(
    magma_z_matrix *x,
    const char * filename,
    magma_queue_t queue )
{
    magma_int_t info = 0;
    
    magma_z_matrix A={Magma_CSR}, B={Magma_CSR};
    magma_int_t entry=0;
    magma_zmfree( x, queue );
    x->ownership = MagmaTrue;
     //   char *vfilename[] = {"/mnt/sparse_matrices/mtx/rail_79841_B.mtx"};
    CHECK( magma_z_csr_mtx( &A,  filename, queue  ));
    CHECK( magma_zmconvert( A, &B, Magma_CSR, Magma_DENSE, queue ));
    CHECK( magma_zvinit( x, Magma_CPU, A.num_cols, A.num_rows, MAGMA_Z_ZERO, queue ));
    x->major = MagmaRowMajor;
    for(magma_int_t i=0; i<A.num_cols; i++) {
        for(magma_int_t j=0; j<A.num_rows; j++) {
            x->val[i*A.num_rows+j] = B.val[ i+j*A.num_cols ];
            entry++;
        }
    }
    x->num_rows = A.num_rows;
    x->num_cols = A.num_cols;
    
cleanup:
    magma_zmfree( &A, queue );
    magma_zmfree( &B, queue );
    return info;
}

/**
    Purpose
    -------

    Writes a vector to a file.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                matrix to write out

    @param[in]
    filename    const char*
                output-filname of the mtx matrix
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C"
magma_int_t
magma_zwrite_vector(
    magma_z_matrix A,
    const char *filename,
    magma_queue_t queue )
{
    magma_int_t i, info = 0;
    
    FILE *fp;
    
    fp = fopen(filename, "w");
    if ( fp == NULL ){
        printf("\n%% error writing vector: file exists or missing write permission\n");
        info = -1;
        goto cleanup;
    }
            
    #define COMPLEX

    #ifdef COMPLEX
    // complex case
    for(i=0; i < A.num_rows; i++) {
        fprintf( fp, "%.16g %.16g\n",
            MAGMA_Z_REAL((A.val)[i]),
            MAGMA_Z_IMAG((A.val)[i]) );
    }
    #else
    for(i=0; i < A.num_rows; i++) {
        fprintf( fp, "%.16g\n",
            MAGMA_Z_REAL((A.val)[i]) );
    }
    #endif
    
    if (fclose(fp) != 0)
        printf("\n%% error: writing matrix failed\n");
    else
        info = 0;

cleanup:
    return info;
}
