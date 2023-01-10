/* 
*   Matrix Market I/O library for ANSI C
*
*   See http://math.nist.gov/MatrixMarket for details.
*
*
*/
#include "magmasparse_internal.h"
#include "magmasparse_mmio.h"

int mm_read_unsymmetric_sparse(
    const char *fname, 
    magma_index_t *M_, 
    magma_index_t *N_, 
    magma_index_t *nz_,
    double **val_, 
    magma_index_t **I_, 
    magma_index_t **J_)
{
    char buffer[ 1024 ];
    FILE *f;
    MM_typecode matcode;
    magma_index_t M, N, nz;
    int i;
    double *val;
    magma_index_t *I, *J;
    
    magma_int_t info = 0;
 
    if ((f = fopen(fname, "r")) == NULL)
            info = -1;

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("%% mm_read_unsymetric: Could not process Matrix Market banner ");
        printf("%% in file [%s]\n", fname);
        info = -1;
    }
 
    if ( !(mm_is_real(matcode) && mm_is_matrix(matcode) &&
            mm_is_sparse(matcode)))
    {
        mm_snprintf_typecode( buffer, sizeof(buffer), matcode );
        fprintf(stderr, "%% Sorry, MAGMA-sparse does not support ");
        fprintf(stderr, "%% Market Market type: [%s]\n", buffer );
        info = -1;
    }
 
    /* find out size of sparse matrix: M, N, nz .... */
 
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) !=0)
    {
        fprintf(stderr, "%% read_unsymmetric_sparse(): could not parse matrix size.\n");
        info = -1;
    }
 
    *M_ = M;
    *N_ = N;
    *nz_ = nz;
 
    /* reseve memory for matrices */
    CHECK( magma_index_malloc_cpu( &I, nz ) );
    CHECK( magma_index_malloc_cpu( &J, nz ) );
    CHECK( magma_dmalloc_cpu( &val, nz ) );
 
    *val_ = val;
    *I_ = I;
    *J_ = J;
 
    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */
 
    for (i=0; i<nz; i++)
    {
        fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
        I[i]--;  /* adjust from 1-based to 0-based */
        J[i]--;
    }
    fclose(f);
 
cleanup:
    if( info != 0 ){
        magma_free_cpu( I );
        magma_free_cpu( J );
        magma_free_cpu( val );
    }
    return info;
}

int mm_is_valid(MM_typecode matcode)
{
    magma_int_t info = 1;
    
    if (!mm_is_matrix(matcode)) info = 0;
    if (mm_is_dense(matcode) && mm_is_pattern(matcode)) info = 0;
    if (mm_is_real(matcode) && mm_is_hermitian(matcode)) info = 0;
    if (mm_is_pattern(matcode) && (mm_is_hermitian(matcode) || 
                mm_is_skew(matcode))) info = 0;
    return info;
}

int mm_read_banner(FILE *f, MM_typecode *matcode)
{
    magma_int_t info = 0;
        
    char line[MM_MAX_LINE_LENGTH];
    char banner[MM_MAX_TOKEN_LENGTH];
    char mtx[MM_MAX_TOKEN_LENGTH]; 
    char crd[MM_MAX_TOKEN_LENGTH];
    char data_type[MM_MAX_TOKEN_LENGTH];
    char storage_scheme[MM_MAX_TOKEN_LENGTH];
    char *p;


    mm_clear_typecode(matcode);  

    if (fgets(line, MM_MAX_LINE_LENGTH, f) == NULL) 
        info = MM_PREMATURE_EOF;

    if (sscanf(line, "%s %s %s %s %s", banner, mtx, crd, data_type, 
        storage_scheme) != 5)
        info = MM_PREMATURE_EOF;

    /* convert to lower case */
    for (p=mtx; *p != '\0'; p++) {
        *p = tolower(*p);
    }
    for (p=crd; *p != '\0'; p++) {
        *p = tolower(*p);
    }
    for (p=data_type; *p != '\0'; p++) {
        *p = tolower(*p);
    }
    for (p=storage_scheme; *p!='\0'; p++) {
        *p = tolower(*p);
    }

    /* check for banner */
    if (strncmp(banner, MatrixMarketBanner, strlen(MatrixMarketBanner)) != 0)
        info = MM_NO_HEADER;

    /* first field should be "mtx" */
    if (strcmp(mtx, MM_MTX_STR) != 0)
        info =  MM_UNSUPPORTED_TYPE;
    mm_set_matrix(matcode);


    /* second field describes whether this is a sparse matrix (in coordinate
            storgae) or a dense array */


    if (strcmp(crd, MM_SPARSE_STR) == 0)
        mm_set_sparse(matcode);
    else
    if (strcmp(crd, MM_DENSE_STR) == 0)
            mm_set_dense(matcode);
    else
        info = MM_UNSUPPORTED_TYPE;
    

    /* third field */

    if (strcmp(data_type, MM_REAL_STR) == 0)
        mm_set_real(matcode);
    else
    if (strcmp(data_type, MM_COMPLEX_STR) == 0)
        mm_set_complex(matcode);
    else
    if (strcmp(data_type, MM_PATTERN_STR) == 0)
        mm_set_pattern(matcode);
    else
    if (strcmp(data_type, MM_INT_STR) == 0)
        mm_set_integer(matcode);
    else
        info = MM_UNSUPPORTED_TYPE;
    

    /* fourth field */

    if (strcmp(storage_scheme, MM_GENERAL_STR) == 0)
        mm_set_general(matcode);
    else
    if (strcmp(storage_scheme, MM_SYMM_STR) == 0)
        mm_set_symmetric(matcode);
    else
    if (strcmp(storage_scheme, MM_HERM_STR) == 0)
        mm_set_hermitian(matcode);
    else
    if (strcmp(storage_scheme, MM_SKEW_STR) == 0)
        mm_set_skew(matcode);
    else
        info = MM_UNSUPPORTED_TYPE;

    return info;
}

int mm_write_mtx_crd_size(FILE *f, magma_index_t M, magma_index_t N, magma_index_t nz)
{
    magma_int_t info = 0;
    
    if (fprintf(f, "%d %d %d\n", M, N, nz) != 3)
        info = MM_COULD_NOT_WRITE_FILE;
    else 
        info = 0;

    return info;
}

int mm_read_mtx_crd_size(FILE *f, magma_index_t *M, magma_index_t *N, 
                                                    magma_index_t *nz )
{
    magma_int_t info = 0;
    
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;

    /* set info = null parameter values, in case we exit with errors */
    *M = *N = *nz = 0;

    /* now continue scanning until you reach the end-of-comments */
    do 
    {
        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL) 
            info = MM_PREMATURE_EOF;
    }while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d %d", M, N, nz) == 3)
        info = 0;
        
    else
    do
    { 
        num_items_read = fscanf(f, "%d %d %d", M, N, nz); 
        if (num_items_read == EOF) info = MM_PREMATURE_EOF;
    }
    while (num_items_read != 3);

    return info;
}


int mm_read_mtx_array_size(FILE *f, magma_index_t *M, magma_index_t *N)
{
    magma_int_t info = 0;
    
    char line[MM_MAX_LINE_LENGTH];
    int num_items_read;
    /* set info = null parameter values, in case we exit with errors */
    *M = *N = 0;
  
    /* now continue scanning until you reach the end-of-comments */
    do 
    {
        if (fgets(line,MM_MAX_LINE_LENGTH,f) == NULL) 
            info = MM_PREMATURE_EOF;
    }while (line[0] == '%');

    /* line[] is either blank or has M,N, nz */
    if (sscanf(line, "%d %d", M, N) == 2)
        info = 0;
        
    else /* we have a blank line */
    do
    { 
        num_items_read = fscanf(f, "%d %d", M, N); 
        if (num_items_read == EOF) info = MM_PREMATURE_EOF;
    }
    while (num_items_read != 2);

    return info;
}

int mm_write_mtx_array_size(FILE *f, int M, int N)
{
    magma_int_t info = 0;
    
    if (fprintf(f, "%d %d\n", M, N) != 2)
        info = MM_COULD_NOT_WRITE_FILE;
    else 
        info = 0;

    return info;
}



/*-------------------------------------------------------------------------*/

/******************************************************************/
/* use when I[], J[], and val[]J, and val[] are already allocated */
/******************************************************************/

int mm_read_mtx_crd_data(FILE *f, magma_index_t M, magma_index_t N, magma_index_t nz, 
    magma_index_t I[], magma_index_t J[], double val[], MM_typecode matcode)
{
    magma_int_t info = 0;
    
    int i;
    if (mm_is_complex(matcode))
    {
        for (i=0; i<nz; i++)
            if (fscanf(f, "%d %d %lg %lg", &I[i], &J[i], &val[2*i], &val[2*i+1])
                != 4) info = MM_PREMATURE_EOF;
    }
    else if (mm_is_real(matcode))
    {
        for (i=0; i<nz; i++)
        {
            if (fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i])
                != 3) info = MM_PREMATURE_EOF;
        }
    }

    else if (mm_is_pattern(matcode))
    {
        for (i=0; i<nz; i++)
            if (fscanf(f, "%d %d", &I[i], &J[i])
                != 2) info = MM_PREMATURE_EOF;
    }
    else
        info = MM_UNSUPPORTED_TYPE;

    return info;
}

int mm_read_mtx_crd_entry(FILE *f, magma_index_t *I, magma_index_t *J,
        double *real, double *imag, MM_typecode matcode)
{
    magma_int_t info = 0;
    
    if (mm_is_complex(matcode))
    {
            if (fscanf(f, "%d %d %lg %lg", I, J, real, imag)
                != 4) info = MM_PREMATURE_EOF;
    }
    else if (mm_is_real(matcode))
    {
            if (fscanf(f, "%d %d %lg\n", I, J, real)
                != 3) info = MM_PREMATURE_EOF;
    }

    else if (mm_is_pattern(matcode))
    {
            if (fscanf(f, "%d %d", I, J) != 2) info = MM_PREMATURE_EOF;
    }
    else
        info = MM_UNSUPPORTED_TYPE;

    return info;
}


/************************************************************************
    mm_read_mtx_crd()  fills M, N, nz, array of values, and return
                        type code, e.g. 'MCRS'

                        if matrix is complex, values[] is of size 2*nz,
                            (nz pairs of real/imaginary values)
************************************************************************/

int mm_read_mtx_crd(char *fname, magma_index_t *M, magma_index_t *N, 
                            magma_index_t *nz, magma_index_t **I, 
                    magma_index_t **J, double **val, MM_typecode *matcode)
{
    magma_int_t info = 0;
    
    int ret_code;
    FILE *f;

    if (strcmp(fname, "stdin") == 0) f=stdin;
    else
    if ((f = fopen(fname, "r")) == NULL)
        info = MM_COULD_NOT_READ_FILE;


    if ((ret_code = mm_read_banner(f, matcode)) != 0)
        info = ret_code;

    if (!(mm_is_valid(*matcode) && mm_is_sparse(*matcode) && 
            mm_is_matrix(*matcode)))
        info = MM_UNSUPPORTED_TYPE;

    if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) != 0)
        info = ret_code;

    CHECK( magma_index_malloc_cpu( I, *nz ) );
    CHECK( magma_index_malloc_cpu( J, *nz ) );
    *val = NULL;

    if (mm_is_complex(*matcode))
    {
        CHECK( magma_dmalloc_cpu( val, *nz*2 ) );
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, 
                *matcode);
        if (ret_code != 0) info = ret_code;
    }
    else if (mm_is_real(*matcode))
    {
        CHECK( magma_dmalloc_cpu( val, *nz ) );
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, 
                *matcode);
        if (ret_code != 0) info = ret_code;
    }

    else if (mm_is_pattern(*matcode))
    {
        ret_code = mm_read_mtx_crd_data(f, *M, *N, *nz, *I, *J, *val, 
                *matcode);
        if (ret_code != 0) info = ret_code;
    }

    if (f != stdin) fclose(f);
cleanup:
    if( info != 0 ){
        magma_free_cpu( *I );
        magma_free_cpu( *J );
        magma_free_cpu( *val );
    }
    return info;
}

int mm_write_banner(FILE *f, MM_typecode matcode)
{
    magma_int_t info = 0;
    
    char buffer[ 1024 ];
    mm_snprintf_typecode( buffer, sizeof(buffer), matcode );
    int ret_code;

    ret_code = fprintf(f, "%s %s\n", MatrixMarketBanner, buffer);
    if (ret_code !=2 )
        info = MM_COULD_NOT_WRITE_FILE;
    else
        info = 0;
    
    return info;
}

int mm_write_mtx_crd(char fname[], magma_index_t M, magma_index_t N, magma_index_t nz, 
        magma_index_t I[], magma_index_t J[], double val[], MM_typecode matcode)
{
    char buffer[ 1024 ];
    magma_int_t info = 0;
        
    FILE *f;
    int i;

    if (strcmp(fname, "stdout") == 0) 
        f = stdout;
    else
    if ((f = fopen(fname, "w")) == NULL)
        info = MM_COULD_NOT_WRITE_FILE;
    
    /* print banner followed by typecode */
    mm_snprintf_typecode( buffer, sizeof(buffer), matcode );
    fprintf(f, "%s ", MatrixMarketBanner);
    fprintf(f, "%s\n", buffer );

    /* print matrix sizes and nonzeros */
    fprintf(f, "%d %d %d\n", M, N, nz);

    /* print values */
    if (mm_is_pattern(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d\n", I[i], J[i]);
    else
    if (mm_is_real(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d %20.16g\n", I[i], J[i], val[i]);
    else
    if (mm_is_complex(matcode))
        for (i=0; i<nz; i++)
            fprintf(f, "%d %d %20.16g %20.16g\n", I[i], J[i], val[2*i], 
                        val[2*i+1]);
    else
    {
        if (f != stdout) fclose(f);
        info = MM_UNSUPPORTED_TYPE;
    }

    if (f !=stdout) fclose(f);

    return info;
}
  

void mm_snprintf_typecode( char *buffer, size_t buflen, MM_typecode matcode )
{
    const char *types[4];
    //int error =0;

    buffer[0] = '\0';
    
    /* check for MTX type */
    if (mm_is_matrix(matcode)) 
        types[0] = MM_MTX_STR;
    else
        types[0] = MM_UNKNOWN;

    /* check for CRD or ARR matrix */
    if (mm_is_sparse(matcode))
        types[1] = MM_SPARSE_STR;
    else
    if (mm_is_dense(matcode))
        types[1] = MM_DENSE_STR;
    else
        types[1] = MM_UNKNOWN;

    /* check for element data type */
    if (mm_is_real(matcode))
        types[2] = MM_REAL_STR;
    else
    if (mm_is_complex(matcode))
        types[2] = MM_COMPLEX_STR;
    else
    if (mm_is_pattern(matcode))
        types[2] = MM_PATTERN_STR;
    else
    if (mm_is_integer(matcode))
        types[2] = MM_INT_STR;
    else
        types[2] = MM_UNKNOWN;


    /* check for symmetry type */
    if (mm_is_general(matcode))
        types[3] = MM_GENERAL_STR;
    else
    if (mm_is_symmetric(matcode))
        types[3] = MM_SYMM_STR;
    else 
    if (mm_is_hermitian(matcode))
        types[3] = MM_HERM_STR;
    else 
    if (mm_is_skew(matcode))
        types[3] = MM_SKEW_STR;
    else
        types[3] = MM_UNKNOWN;

    snprintf( buffer, buflen, "%s %s %s %s", types[0], types[1], types[2], types[3] );
}
