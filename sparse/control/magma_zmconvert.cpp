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

#include <cuda.h>  // for CUDA_VERSION

/**
    Purpose
    -------

    Helper function to compress CSR containing zero-entries.


    Arguments
    ---------

    @param[in]
    val         magmaDoubleComplex**
                input val pointer to compress

    @param[in]
    row         magma_int_t**
                input row pointer to modify

    @param[in]
    col         magma_int_t**
                input col pointer to compress

    @param[in]
    valn        magmaDoubleComplex**
                output val pointer

    @param[out]
    rown        magma_int_t**
                output row pointer

    @param[out]
    coln        magma_int_t**
                output col pointer

    @param[out]
    n           magma_int_t*
                number of rows in matrix

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_z_csr_compressor(
    magmaDoubleComplex ** val,
    magma_index_t ** row,
    magma_index_t ** col,
    magmaDoubleComplex ** valn,
    magma_index_t ** rown,
    magma_index_t ** coln,
    magma_int_t *n,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_index_t i,j, nnz_new=0, (*row_nnz)=NULL, nnz_this_row;
    CHECK( magma_index_malloc_cpu( &(row_nnz), (*n) ));
    CHECK( magma_index_malloc_cpu( rown, *n+1 ));
    for( i=0; i<*n; i++ ) {
        (*rown)[i] = nnz_new;
        nnz_this_row = 0;
        for( j=(*row)[i]; j<(*row)[i+1]; j++ ) {
            if ( (MAGMA_Z_REAL((*val)[j]) != 0) || (MAGMA_Z_IMAG((*val)[j]) != 0) ) {
                nnz_new++;
                nnz_this_row++;
            }
        }
        row_nnz[i] = nnz_this_row;
    }
    (*rown)[*n] = nnz_new;

    CHECK( magma_zmalloc_cpu( valn, nnz_new ));
    CHECK( magma_index_malloc_cpu( coln, nnz_new ));

    nnz_new = 0;
    for( i=0; i<*n; i++ ) {
        for( j=(*row)[i]; j<(*row)[i+1]; j++ ) {
            if ( (MAGMA_Z_REAL((*val)[j]) != 0) || (MAGMA_Z_IMAG((*val)[j]) != 0) ) {
                (*valn)[nnz_new]= (*val)[j];
                (*coln)[nnz_new]= (*col)[j];
                nnz_new++;
            }
        }
    }


cleanup:
    if ( info != 0 ) {
        magma_free_cpu( valn );
        magma_free_cpu( coln );
        magma_free_cpu( rown );
    }
    magma_free_cpu( row_nnz );
    row_nnz = NULL;
    return info;
}


/**
    Purpose
    -------

    Converter between different sparse storage formats.

    Arguments
    ---------

    @param[in]
    A           magma_z_matrix
                sparse matrix A

    @param[out]
    B           magma_z_matrix*
                copy of A in new format

    @param[in]
    old_format  magma_storage_t
                original storage format

    @param[in]
    new_format  magma_storage_t
                new storage format

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmconvert(
    magma_z_matrix A,
    magma_z_matrix *B,
    magma_storage_t old_format,
    magma_storage_t new_format,
    magma_queue_t queue )
{
    magma_int_t info = 0;

    magma_index_t *length=NULL;

    magma_z_matrix hA={Magma_CSR}, hB={Magma_CSR};
    magma_z_matrix dA={Magma_CSR}, dB={Magma_CSR};
    magma_index_t *row_tmp=NULL, *col_tmp=NULL;
    magmaDoubleComplex *val_tmp = NULL;
    magma_index_t *row_tmp2=NULL, *col_tmp2=NULL;
    magmaDoubleComplex *val_tmp2 = NULL;
    magmaDoubleComplex *transpose=NULL;
    magma_index_t *nnz_per_row=NULL;

    cusparseHandle_t cusparseHandle = 0;
    cusparseMatDescr_t descr = 0;
    
    // make sure the target structure is empty
    magma_zmfree( B, queue );
    B->ownership = MagmaTrue;

    B->val = NULL;
    B->col = NULL;
    B->row = NULL;
    B->rowidx = NULL;
    B->list = NULL;
    B->blockinfo = NULL;
    B->diag = NULL;
    B->dval = NULL;
    B->dcol = NULL;
    B->drow = NULL;
    B->drowidx = NULL;
    B->ddiag = NULL;
    B->dlist = NULL;
    B->tile_ptr = NULL;
    B->dtile_ptr = NULL;
    B->tile_desc = NULL;
    B->dtile_desc = NULL;
    B->tile_desc_offset_ptr = NULL;
    B->dtile_desc_offset_ptr = NULL;
    B->tile_desc_offset = NULL;
    B->dtile_desc_offset = NULL;
    B->calibrator = NULL;
    B->dcalibrator = NULL;

    magmaDoubleComplex zero = MAGMA_Z_MAKE( 0.0, 0.0 );

    // check whether matrix on CPU
    if ( A.memory_location == Magma_CPU )
    {
        // CSR to anything
        if ( old_format == Magma_CSR )
        {
            // CSR to CSR
            if ( new_format == Magma_CSR ) {
                // fill in information for B
                B->storage_type = Magma_CSR;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;

                CHECK( magma_zmalloc_cpu( &B->val, A.nnz ));
                CHECK( magma_index_malloc_cpu( &B->row, A.num_rows+1 ));
                CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));

                for( magma_int_t i=0; i < A.nnz; i++) {
                    B->val[i] = A.val[i];
                    B->col[i] = A.col[i];
                }
                for( magma_int_t i=0; i < A.num_rows+1; i++) {
                    B->row[i] = A.row[i];
                }
            }
            // CSR to CUCSR
            else if ( new_format == Magma_CUCSR ){
                CHECK(magma_zmconvert(A, B, Magma_CSR, Magma_CSR, queue));
                B->storage_type = Magma_CUCSR;
            }

            // CSR to CSRL
            else if ( new_format == Magma_CSRL ) {
                // fill in information for B
                B->storage_type = Magma_CSR;
                B->memory_location = A.memory_location;
                B->fill_mode = MagmaLower;
                B->num_rows = A.num_rows;
                B->num_cols = A.num_cols;
                B->true_nnz = A.true_nnz;
                B->diameter = A.diameter;

                magma_int_t numzeros=0;
                for( magma_int_t i=0; i < A.num_rows; i++) {
                    for( magma_int_t j=A.row[i]; j < A.row[i+1]; j++) {
                        if ( A.col[j] <= i) {
                            numzeros++;
                        }
                    }
                }
                B->nnz = numzeros;
                CHECK( magma_zmalloc_cpu( &B->val, numzeros ));
                CHECK( magma_index_malloc_cpu( &B->row, A.num_rows+1 ));
                CHECK( magma_index_malloc_cpu( &B->col, numzeros ));

                numzeros=0;
                for( magma_int_t i=0; i < A.num_rows; i++) {
                    B->row[i]=numzeros;
                    for( magma_int_t j=A.row[i]; j < A.row[i+1]; j++) {
                        if ( A.col[j] < i) {
                            B->val[numzeros] = A.val[j];
                            B->col[numzeros] = A.col[j];
                            numzeros++;
                        }
                        else if ( A.col[j] == i &&
                                        B->diagorder_type == Magma_UNITY) {
                            B->val[numzeros] = MAGMA_Z_MAKE(1.0, 0.0);
                            B->col[numzeros] = A.col[j];
                            numzeros++;
                        }
                        else if ( A.col[j] == i ) {
                            B->val[numzeros] = A.val[j];
                            B->col[numzeros] = A.col[j];
                            numzeros++;
                        }
                    }
                }
                B->row[B->num_rows] = numzeros;
            }

            // CSR to CSRU
            else if (  new_format == Magma_CSRU ) {
                // fill in information for B
                B->storage_type = Magma_CSR;
                B->memory_location = A.memory_location;
                B->fill_mode = MagmaLower;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->diameter = A.diameter;
                B->fill_mode = MagmaUpper;
                magma_int_t numzeros=0;
                for( magma_int_t i=0; i < A.num_rows; i++) {
                    for( magma_int_t j=A.row[i]; j < A.row[i+1]; j++) {
                        if ( A.col[j] >= i) {
                            numzeros++;
                        }
                    }
                }
                B->nnz = numzeros;
                CHECK( magma_zmalloc_cpu( &B->val, numzeros ));
                CHECK( magma_index_malloc_cpu( &B->row, A.num_rows+1 ));
                CHECK( magma_index_malloc_cpu( &B->col, numzeros ));

                numzeros=0;
                for( magma_int_t i=0; i < A.num_rows; i++) {
                    B->row[i]=numzeros;
                    for( magma_int_t j=A.row[i]; j < A.row[i+1]; j++) {
                        if ( A.col[j] >= i) {
                            B->val[numzeros] = A.val[j];
                            B->col[numzeros] = A.col[j];
                            numzeros++;
                        }
                    }
                }
                B->row[B->num_rows] = numzeros;
            }

            // CSR to CSRD (diagonal elements first)
            else if ( new_format == Magma_CSRD ) {
                // fill in information for B
                B->storage_type = Magma_CSRD;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;

                CHECK( magma_zmalloc_cpu( &B->val, A.nnz ));
                CHECK( magma_index_malloc_cpu( &B->row, A.num_rows+1 ));
                CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));

                for(magma_int_t i=0; i < A.num_rows; i++) {
                    magma_int_t count = 1;
                    for(magma_int_t j=A.row[i]; j < A.row[i+1]; j++) {
                        if ( A.col[j] == i ) {
                            B->col[A.row[i]] = A.col[j];
                            B->val[A.row[i]] = A.val[j];
                        } else {
                            B->col[A.row[i]+count] = A.col[j];
                            B->val[A.row[i]+count] = A.val[j];
                            count++;
                        }
                    }
                }
                for( magma_int_t i=0; i < A.num_rows+1; i++) {
                    B->row[i] = A.row[i];
                }
            }

            // CSR to COO
            else if ( new_format == Magma_COO ) {
                CHECK( magma_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
                B->storage_type = Magma_COO;

                magma_free_cpu( B->row );
                CHECK( magma_index_malloc_cpu( &B->row, A.nnz ));

                for(magma_int_t i=0; i < A.num_rows; i++) {
                    for(magma_int_t j=A.row[i]; j < A.row[i+1]; j++) {
                        B->row[j] = i;
                    }
                }
            }

            // CSR to CSRCOO
            else if ( new_format == Magma_CSRCOO ) {
                CHECK( magma_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
                B->storage_type = Magma_CSRCOO;

                CHECK( magma_index_malloc_cpu( &B->rowidx, A.nnz ));

                for(magma_int_t i=0; i < A.num_rows; i++) {
                    for(magma_int_t j=A.row[i]; j < A.row[i+1]; j++) {
                        B->rowidx[j] = i;
                    }
                }
            }

            // CSR to CSRLIST
            else if ( new_format == Magma_CSRLIST ) {
                CHECK( magma_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
                B->storage_type = Magma_CSRLIST;
                magma_free_cpu( B->val );
                magma_free_cpu( B->col );

                CHECK( magma_zmalloc_cpu( &B->val, A.nnz+A.num_rows*2 ));
                CHECK( magma_index_malloc_cpu( &B->col, A.nnz+A.num_rows*2 ));
                CHECK( magma_index_malloc_cpu( &B->rowidx, A.nnz+A.num_rows*2 ));
                CHECK( magma_index_malloc_cpu( &B->list, A.nnz+A.num_rows*2 ));

                for(magma_int_t i=0; i < A.nnz; i++) {
                    B->col[i] = A.col[i];
                    B->val[i] = A.val[i];
                }

                for(magma_int_t i=0; i < A.num_rows; i++) {
                    for(magma_int_t j=A.row[i]; j < A.row[i+1]; j++) {
                        B->rowidx[j] = i;
                    }
                    for(magma_int_t j=A.row[i]; j < A.row[i+1]-1; j++) {
                        B->list[j] = j+1;
                    }
                    B->list[A.row[i+1]-1] = 0;
                }
                for(magma_int_t i=A.nnz; i < A.nnz+A.num_rows*2; i++) {
                    B->list[i] = -1;
                }
                B->true_nnz = A.nnz+A.num_rows*2;
            }

            // CSR to ELLPACKT (using row-major storage)
            else if (  new_format == Magma_ELLPACKT ) {
                // fill in information for B
                B->storage_type = Magma_ELLPACKT;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;
                // conversion
                magma_index_t i, j, maxrowlength=0;
                CHECK( magma_index_malloc_cpu( &length, A.num_rows));

                for( i=0; i < A.num_rows; i++ ) {
                    length[i] = A.row[i+1]-A.row[i];
                    if (length[i] > maxrowlength)
                        maxrowlength = length[i];
                }
                //printf( "Conversion to ELLPACK with %d elements per row: ",
                                                                // maxrowlength );
                //fflush(stdout);
                CHECK( magma_zmalloc_cpu( &B->val, maxrowlength*A.num_rows ));
                CHECK( magma_index_malloc_cpu( &B->col, maxrowlength*A.num_rows ));

                for( i=0; i < (maxrowlength*A.num_rows); i++) {
                    B->val[i] = MAGMA_Z_MAKE(0., 0.);
                    B->col[i] =  -1;
                }
                for( i=0; i < A.num_rows; i++ ) {
                    magma_int_t offset = 0;
                    for( j=A.row[i]; j < A.row[i+1]; j++ ) {
                        B->val[i*maxrowlength+offset] = A.val[j];
                        B->col[i*maxrowlength+offset] = A.col[j];
                        offset++;
                    }
                }
                B->max_nnz_row = maxrowlength;
            }

            // CSR to ELL
            else if ( new_format == Magma_ELL ) {
                // fill in information for B
                B->storage_type = Magma_ELL;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;

                // conversion
                magma_index_t i, j, maxrowlength=0;
                CHECK( magma_index_malloc_cpu( &length, A.num_rows));

                for( i=0; i < A.num_rows; i++ ) {
                    length[i] = A.row[i+1]-A.row[i];
                    if (length[i] > maxrowlength)
                        maxrowlength = length[i];
                }
                //printf( "Conversion to ELL with %d elements per row: ",
                                                               // maxrowlength );
                //fflush(stdout);
                CHECK( magma_zmalloc_cpu( &B->val, maxrowlength*A.num_rows ));
                CHECK( magma_index_malloc_cpu( &B->col, maxrowlength*A.num_rows ));

                for( i=0; i < (maxrowlength*A.num_rows); i++) {
                    B->val[i] = MAGMA_Z_MAKE(0., 0.);
                    B->col[i] = 0;
                }

                for( i=0; i < A.num_rows; i++ ) {
                    magma_int_t offset = 0;
                    for( j=A.row[i]; j < A.row[i+1]; j++ ) {
                        B->val[offset*A.num_rows+i] = A.val[j];
                        B->col[offset*A.num_rows+i] = A.col[j];
                        offset++;
                    }
                }
                B->max_nnz_row = maxrowlength;
                //printf( "done\n" );
            }

            // CSR to ELLD (ELLPACK with diagonal element first)
            else if ( new_format == Magma_ELLD ) {
                // fill in information for B
                B->storage_type = Magma_ELLD;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;

                // conversion
                magma_index_t i, j, maxrowlength=0;
                CHECK( magma_index_malloc_cpu( &length, A.num_rows));

                for( i=0; i < A.num_rows; i++ ) {
                    length[i] = A.row[i+1]-A.row[i];
                    if (length[i] > maxrowlength)
                        maxrowlength = length[i];
                }
                //printf( "Conversion to ELL with %d elements per row: ",
                                                               // maxrowlength );
                //fflush(stdout);
                CHECK( magma_zmalloc_cpu( &B->val, maxrowlength*A.num_rows ));
                CHECK( magma_index_malloc_cpu( &B->col, maxrowlength*A.num_rows ));


                for( i=0; i < (maxrowlength*A.num_rows); i++) {
                    B->val[i] = MAGMA_Z_MAKE(0., 0.);
                    B->col[i] =  -1;
                }

                for( i=0; i < A.num_rows; i++ ) {
                    magma_int_t offset = 1;
                    for( j=A.row[i]; j < A.row[i+1]; j++ ) {
                        if ( A.col[j] == i ) { // diagonal case
                            B->val[i*maxrowlength] = A.val[j];
                            B->col[i*maxrowlength] = A.col[j];
                        } else {
                            B->val[i*maxrowlength+offset] = A.val[j];
                            B->col[i*maxrowlength+offset] = A.col[j];
                            offset++;
                        }
                    }
                }
                B->max_nnz_row = maxrowlength;
            }

            // CSR to ELLRT (also ELLPACKRT)
            else if (  new_format == Magma_ELLRT ) {
                // fill in information for B
                B->storage_type = Magma_ELLRT;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;

                // conversion
                magma_index_t i, j, maxrowlength=0;
                CHECK( magma_index_malloc_cpu( &length, A.num_rows));

                for( i=0; i < A.num_rows; i++ ) {
                    length[i] = A.row[i+1]-A.row[i];
                    if (length[i] > maxrowlength)
                        maxrowlength = length[i];
                }

                //printf( "Conversion to ELLRT with %d elements per row: ",
                //                                                   maxrowlength );

                magma_int_t threads_per_row = B->alignment;
                magma_int_t rowlength = magma_roundup( maxrowlength, threads_per_row );

                CHECK( magma_zmalloc_cpu( &B->val, rowlength*A.num_rows ));
                CHECK( magma_index_malloc_cpu( &B->col, rowlength*A.num_rows ));
                CHECK( magma_index_malloc_cpu( &B->row, A.num_rows ));

                for( i=0; i < rowlength*A.num_rows; i++) {
                    B->val[i] = MAGMA_Z_MAKE(0., 0.);
                    B->col[i] =  0;
                }

                for( i=0; i < A.num_rows; i++ ) {
                    magma_int_t offset = 0;
                    for( j=A.row[i]; j < A.row[i+1]; j++ ) {
                        B->val[i*rowlength+offset] = A.val[j];
                        B->col[i*rowlength+offset] = A.col[j];
                        offset++;
                    }
                    B->row[i] = A.row[i+1] - A.row[i];
                }
                B->max_nnz_row = maxrowlength;
                //printf( "done\n" );
            }

            // CSR to SELLP
            // SELLC is SELLP using alignment 1
            // see paper by M. KREUTZER, G. HAGER, G WELLEIN, H. FEHSKE A. BISHOP
            // A UNIFIED SPARSE MATRIX DATA FORMAT
            // FOR MODERN PROCESSORS WITH WIDE SIMD UNITS
            // in SELLP we modify SELLC:
            // alignment is posible such that multiple threads can be used for SpMV
            // so the rowlength is padded (SELLP) to a multiple of the alignment
            else if ( new_format == Magma_SELLP ) {
                if( 256%(B->blocksize) !=0 ){
                    printf("error: blocksize not supported!\n");
                    info = MAGMA_ERR_NOT_SUPPORTED;
                    goto cleanup;
                }

                // fill in information for B
                B->storage_type = new_format;
                if (B->alignment > 1)
                    B->storage_type = Magma_SELLP;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->diameter = A.diameter;
                B->max_nnz_row = 0;
                magma_int_t C = B->blocksize;
                magma_int_t slices = ( A.num_rows+C-1)/(C);
                B->numblocks = slices;
                magma_int_t alignedlength, alignment = B->alignment;
                // conversion
                magma_index_t i, j, k, maxrowlength=0;
                CHECK( magma_index_malloc_cpu( &length, C));
                // B-row points to the start of each slice
                CHECK( magma_index_malloc_cpu( &B->row, slices+1 ));


                B->row[0] = 0;
                for( i=0; i < slices; i++ ) {
                    maxrowlength = 0;
                    for(j=0; j < C; j++) {
                        if (i*C+j < A.num_rows) {
                            length[j] = A.row[i*C+j+1]-A.row[i*C+j];
                        }
                        else
                            length[j]=0;
                        if (length[j] > maxrowlength) {
                            maxrowlength = length[j];
                        }
                    }
                    alignedlength = magma_roundup( maxrowlength, alignment );
                    B->row[i+1] = B->row[i] + alignedlength * C;
                    if ( alignedlength > B->max_nnz_row )
                        B->max_nnz_row = alignedlength;
                }
                B->nnz = B->row[slices];
                //printf( "Conversion to SELLC with %d slices of size %d and"
                //       " %d nonzeros.\n", slices, C, B->nnz );

                //fflush(stdout);
                CHECK( magma_zmalloc_cpu( &B->val, B->row[slices] ));
                CHECK( magma_index_malloc_cpu( &B->col, B->row[slices] ));

                // zero everything
                for( i=0; i < B->row[slices]; i++ ) {
                    B->val[ i ] = MAGMA_Z_MAKE(0., 0.);
                    B->col[ i ] =  0;
                }
                // fill in values
                for( i=0; i < slices; i++ ) {
                    for(j=0; j < C; j++) {
                        magma_int_t line = i*C+j;
                        magma_int_t offset = 0;
                        if ( line < A.num_rows) {
                            for( k=A.row[line]; k < A.row[line+1]; k++ ) {
                                B->val[ B->row[i] + j +offset*C ] = A.val[k];
                                B->col[ B->row[i] + j +offset*C ] = A.col[k];
                                offset++;
                            }
                        }
                    }
                }
                //B->nnz = A.nnz;
            }

            // CSR to DENSE
            else if ( new_format == Magma_DENSE ) {
                //printf( "Conversion to DENSE: " );
                // fill in information for B
                B->storage_type = Magma_DENSE;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;

                // conversion
                CHECK( magma_zmalloc_cpu( &B->val, A.num_rows*A.num_cols ));

                for( magma_int_t i=0; i<(A.num_rows)*(A.num_cols); i++) {
                    B->val[i] = MAGMA_Z_MAKE(0., 0.);
                }

                for(magma_int_t i=0; i < A.num_rows; i++ ) {
                    for(magma_int_t j=A.row[i]; j < A.row[i+1]; j++ )
                        B->val[i * (A.num_cols) + A.col[j] ] = A.val[ j ];
                }

                //printf( "done\n" );
            }

            // CSR to BCSR
            else if ( new_format == Magma_BCSR ) {
                CHECK( magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, queue ) );
                dB.blocksize = B->blocksize;
                CHECK( magma_zmconvert(dA, &dB, Magma_CSR, Magma_BCSR, queue ) );
                CHECK( magma_zmtransfer(dB, B, Magma_DEV, Magma_CPU, queue ) );
            }

            // CSR to CSR5
            else if ( new_format == Magma_CSR5 ) {
                //printf( "Conversion to CSR5: " );
                // fill in information for B
                B->storage_type = Magma_CSR5;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;

                CHECK( magma_zmalloc_cpu( &B->val, A.nnz ));
                CHECK( magma_index_malloc_cpu( &B->row, A.num_rows+1 ));
                CHECK( magma_index_malloc_cpu( &B->col, A.nnz ));

                for( magma_int_t i=0; i < A.num_rows+1; i++) {
                    B->row[i] = A.row[i];
                }

                // compute sigma
                int r = 4;
                int s = 32;
                int t = 256;
                int u = 6;

                int csr_nnz_per_row = B->nnz / B->num_rows;
                if (csr_nnz_per_row <= r)
                    B->csr5_sigma = r;
                else if (csr_nnz_per_row > r && csr_nnz_per_row <= s)
                    B->csr5_sigma = csr_nnz_per_row;
                else if (csr_nnz_per_row <= t && csr_nnz_per_row > s)
                    B->csr5_sigma = s;
                else // csr_nnz_per_row > t
                    B->csr5_sigma = u;

                // conversion
                // compute #bits required for `y_offset' and `scansum_offset'
                int base = 2;
                B->csr5_bit_y_offset = 1;
                while (base < MAGMA_CSR5_OMEGA * B->csr5_sigma)
                { base *= 2; B->csr5_bit_y_offset++; }

                base = 2;
                B->csr5_bit_scansum_offset = 1;
                while (base < MAGMA_CSR5_OMEGA)
                { base *= 2; B->csr5_bit_scansum_offset++; }

                if ( (size_t) B->csr5_bit_y_offset + B->csr5_bit_scansum_offset >
                    sizeof(magma_uindex_t) * 8 - 1)
                {
                    printf("error: csr5-omega not supported.\n");
                    info = MAGMA_ERR_NOT_SUPPORTED;
                }

                int bit_all = B->csr5_bit_y_offset + B->csr5_bit_scansum_offset
                              + B->csr5_sigma;
                B->csr5_num_packets = ceil((double)bit_all
                                           /(double)(sizeof(magma_uindex_t)*8));

                // calculate the number of tiles
                B->csr5_p = ceil((double)B->nnz
                                 / (double)(MAGMA_CSR5_OMEGA * B->csr5_sigma));
                //printf("sigma = %i, p = %i\n", B->csr5_sigma, B->csr5_p);
                // malloc the newly added arrays for CSR5
                CHECK( magma_uindex_malloc_cpu( &B->tile_ptr, B->csr5_p+1 ));
                for( magma_int_t i=0; i<B->csr5_p+1; i++) {
                    B->tile_ptr[i] = 0;
                }

                CHECK( magma_uindex_malloc_cpu( &B->tile_desc,
                          B->csr5_p * MAGMA_CSR5_OMEGA * B->csr5_num_packets ));
                for( magma_int_t i=0; i<B->csr5_p * MAGMA_CSR5_OMEGA
                                        * B->csr5_num_packets; i++) {
                    B->tile_desc[i] = 0;
                }


                CHECK( magma_zmalloc_cpu( &B->calibrator, B->csr5_p ));
                for( magma_int_t i=0; i<B->csr5_p; i++) {
                    B->calibrator[i] = MAGMA_Z_MAKE(0., 0.);
                }

                CHECK( magma_index_malloc_cpu( &B->tile_desc_offset_ptr,
                                               B->csr5_p+1 ));
                for( magma_int_t i=0; i<B->csr5_p+1; i++) {
                    B->tile_desc_offset_ptr[i] = 0;
                }


                // convert csr data to csr5 data (3 steps)
                // step 1 generate tile pointer
                // step 1.1 binary search row pointer
                for (magma_index_t global_id = 0; global_id <= B->csr5_p;
                     global_id++)
                {
                    // compute tile boundaries by tile of size sigma * omega
                    magma_index_t boundary = global_id * B->csr5_sigma
                                             * MAGMA_CSR5_OMEGA;

                    // clamp tile boundaries to [0, nnz]
                    boundary = boundary > B->nnz ? B->nnz : boundary;

                    // binary search
                    magma_index_t start = 0, stop = B->num_rows, median;
                    magma_index_t key_median;
                    while (stop >= start)
                    {
                        median = (stop + start) / 2;
                        key_median = B->row[median];
                        if (boundary >= key_median)
                            start = median + 1;
                        else
                            stop = median - 1;
                    }
                    B->tile_ptr[global_id] = start-1;
                }
                
                // step 1.2 check empty rows
                for (magma_index_t group_id = 0; group_id < B->csr5_p; group_id++) {
                    int dirty = 0;
                
                    magma_uindex_t start = B->tile_ptr[group_id];
                    magma_uindex_t stop  = B->tile_ptr[group_id+1];
                    start = (start << 1) >> 1;
                    stop  = (stop << 1) >> 1;
                
                    if (start == stop)
                        continue;
                
                    for (magma_uindex_t row_idx = start; row_idx <= stop; row_idx++) {
                        if (B->row[row_idx] == B->row[row_idx+1]) {
                            dirty = 1;
                            break;
                        }
                    }
                
                    if (dirty) {
                        start |= sizeof(magma_uindex_t) == 4
                                           ? 0x80000000 : 0x8000000000000000;
                        B->tile_ptr[group_id] = start;
                    }
                }
                B->csr5_tail_tile_start = (B->tile_ptr[B->csr5_p-1] << 1) >> 1;
                
                // step 2. generate tile descriptor
                
                int bit_all_offset = B->csr5_bit_y_offset
                                     + B->csr5_bit_scansum_offset;
                
                //generate_tile_descriptor_s1_kernel
                for (int par_id = 0; par_id < B->csr5_p-1; par_id++) {
                    const magma_index_t row_start = B->tile_ptr[par_id]
                                                    & 0x7FFFFFFF;
                    const magma_index_t row_stop  = B->tile_ptr[par_id + 1]
                                                    & 0x7FFFFFFF;
                
                    for (int rid = row_start; rid <= row_stop; rid++) {
                        int ptr = B->row[rid];
                        int pid = ptr / (MAGMA_CSR5_OMEGA * B->csr5_sigma);
                
                        if (pid == par_id) {
                            int lx = (ptr / B->csr5_sigma) % MAGMA_CSR5_OMEGA;
                
                            const int glid = ptr%B->csr5_sigma+bit_all_offset;
                            const int ly = glid / 32;
                            const int llid = glid % 32;
                
                            const magma_uindex_t val = 0x1 << (31 - llid);
                
                            const int location = pid * MAGMA_CSR5_OMEGA
                                * B->csr5_num_packets
                                + ly * MAGMA_CSR5_OMEGA + lx;
                            B->tile_desc[location] |= val;
                        }
                    }
                }
                
                //generate_tile_descriptor_s2_kernel
                int num_thread = 1; //omp_get_max_threads();
                magma_index_t *s_segn_scan_all, *s_present_all;
                
                CHECK( magma_index_malloc_cpu( &s_segn_scan_all,
                                           2 * MAGMA_CSR5_OMEGA * num_thread ));
                CHECK( magma_index_malloc_cpu( &s_present_all,
                                           2 * MAGMA_CSR5_OMEGA * num_thread ));
                
                
                //int *s_segn_scan_all = (int *)malloc(2 * MAGMA_CSR5_OMEGA
                //                                   * sizeof(int) * num_thread);
                //int *s_present_all   = (int *)malloc(2 * MAGMA_CSR5_OMEGA
                //                                   * sizeof(int) * num_thread);
                for (magma_index_t i = 0; i < num_thread; i++)
                    s_present_all[i * 2 * MAGMA_CSR5_OMEGA + MAGMA_CSR5_OMEGA]
                        = 1;
                
                //const int bit_all_offset = bit_y_offset + bit_scansum_offset;
                
                //#pragma omp parallel for
                for (int par_id = 0; par_id < B->csr5_p-1; par_id++) {
                    int tid = 0; //omp_get_thread_num();
                    int *s_segn_scan = &s_segn_scan_all[tid * 2
                                                        * MAGMA_CSR5_OMEGA];
                    int *s_present = &s_present_all[tid * 2
                                                        * MAGMA_CSR5_OMEGA];
                
                    memset(s_segn_scan, 0, (MAGMA_CSR5_OMEGA + 1)*sizeof(int));
                    memset(s_present, 0, MAGMA_CSR5_OMEGA * sizeof(int));
                
                    bool with_empty_rows = (B->tile_ptr[par_id] >> 31) & 0x1;
                    magma_index_t row_start       = B->tile_ptr[par_id]
                                                    & 0x7FFFFFFF;
                    const magma_index_t row_stop  = B->tile_ptr[par_id + 1]
                                                    & 0x7FFFFFFF;
                
                    if (row_start == row_stop)
                        continue;
                
                    //#pragma simd
                    for (int lane_id = 0; lane_id < MAGMA_CSR5_OMEGA; lane_id++) {
                        int start = 0, stop = 0, segn = 0;
                        bool present = 0;
                        magma_uindex_t bitflag = 0;
                
                        present |= !lane_id;
                
                        // extract the first bit-flag packet
                        int ly = 0;
                        magma_uindex_t first_packet = B->tile_desc[par_id
                            * MAGMA_CSR5_OMEGA * B->csr5_num_packets+lane_id];
                        bitflag = (first_packet << bit_all_offset)
                                   | ((magma_uindex_t)present << 31);
                        start = !((bitflag >> 31) & 0x1);
                        present |= (bitflag >> 31) & 0x1;
                
                        for (int i = 1; i < B->csr5_sigma; i++) {
                            if ((!ly && i == 32 - bit_all_offset)
                                || (ly && (i - (32 - bit_all_offset)) % 32==0))
                            {
                                ly++;
                                bitflag = B->tile_desc[par_id
                                          * MAGMA_CSR5_OMEGA
                                          * B->csr5_num_packets
                                          + ly * MAGMA_CSR5_OMEGA + lane_id];
                            }
                            const int norm_i = !ly ? i
                                               : i - (32 - bit_all_offset);
                            stop += (bitflag >> (31 - norm_i % 32) ) & 0x1;
                            present |= (bitflag >> (31 - norm_i % 32)) & 0x1;
                        }
                
                        // compute y_offset for all tiles
                        segn = stop - start + present;
                        segn = segn > 0 ? segn : 0;
                
                        s_segn_scan[lane_id] = segn;
                
                        // compute scansum_offset
                        s_present[lane_id] = present;
                    }
                
                    //scan_single<int>(s_segn_scan, MAGMA_CSR5_OMEGA + 1);
                    int old_val, new_val;
                    old_val = s_segn_scan[0];
                    s_segn_scan[0] = 0;
                    for (int i = 1; i < MAGMA_CSR5_OMEGA + 1; i++) {
                        new_val = s_segn_scan[i];
                        s_segn_scan[i] = old_val + s_segn_scan[i-1];
                        old_val = new_val;
                    }
                
                    if (with_empty_rows) {
                        B->tile_desc_offset_ptr[par_id]
                            = s_segn_scan[MAGMA_CSR5_OMEGA];
                        B->tile_desc_offset_ptr[B->csr5_p] = 1;
                    }
                
                    //#pragma simd
                    for (int lane_id = 0; lane_id < MAGMA_CSR5_OMEGA; lane_id++) {
                        int y_offset = s_segn_scan[lane_id];
                
                        int scansum_offset = 0;
                        int next1 = lane_id + 1;
                        if (s_present[lane_id]) {
                            while ( ! s_present[next1] && next1 < MAGMA_CSR5_OMEGA)
                            {
                                scansum_offset++;
                                next1++;
                            }
                        }
                
                        magma_uindex_t first_packet = B->tile_desc[par_id
                           * MAGMA_CSR5_OMEGA * B->csr5_num_packets + lane_id];
                
                        y_offset = lane_id ? y_offset - 1 : 0;
                
                        first_packet |= y_offset << (32-B->csr5_bit_y_offset);
                        first_packet |= scansum_offset << (32-bit_all_offset);
                
                        B->tile_desc[par_id * MAGMA_CSR5_OMEGA
                              * B->csr5_num_packets + lane_id] = first_packet;
                    }
                }
                
                magma_free_cpu(s_segn_scan_all);
                magma_free_cpu(s_present_all);
                
                if (B->tile_desc_offset_ptr[B->csr5_p]) {
                    //scan_single(B->tile_desc_offset_ptr, p+1);
                    int old_val, new_val;
                    old_val = B->tile_desc_offset_ptr[0];
                    B->tile_desc_offset_ptr[0] = 0;
                    for (int i = 1; i < B->csr5_p+1; i++)
                    {
                        new_val = B->tile_desc_offset_ptr[i];
                        B->tile_desc_offset_ptr[i] = old_val
                                                + B->tile_desc_offset_ptr[i-1];
                        old_val = new_val;
                    }
                }
                
                B->csr5_num_offsets = B->tile_desc_offset_ptr[B->csr5_p];
                
                if (B->csr5_num_offsets) {
                    CHECK( magma_index_malloc_cpu( &B->tile_desc_offset
                                                   , B->csr5_num_offsets ));
                
                    //err = generate_tile_descriptor_offset
                    const int bit_bitflag = 32 - bit_all_offset;
                
                    //#pragma omp parallel for
                    for (int par_id = 0; par_id < B->csr5_p-1; par_id++) {
                        bool with_empty_rows = (B->tile_ptr[par_id] >> 31)&0x1;
                        if (!with_empty_rows)
                            continue;
                
                        magma_index_t row_start       = B->tile_ptr[par_id]
                                                        & 0x7FFFFFFF;
                        const magma_index_t row_stop  = B->tile_ptr[par_id + 1]
                                                        & 0x7FFFFFFF;
                
                        int offset_pointer = B->tile_desc_offset_ptr[par_id];
                        //#pragma simd
                        for (int lane_id = 0; lane_id < MAGMA_CSR5_OMEGA; lane_id++) {
                            bool local_bit;
                
                            // extract the first bit-flag packet
                            int ly = 0;
                            magma_uindex_t descriptor = B->tile_desc[par_id
                                * MAGMA_CSR5_OMEGA * B->csr5_num_packets
                                + lane_id];
                            int y_offset = descriptor
                                           >> (32 - B->csr5_bit_y_offset);
                
                            descriptor = descriptor << bit_all_offset;
                            descriptor = lane_id ? descriptor
                                         : descriptor | 0x80000000;
                
                            local_bit = (descriptor >> 31) & 0x1;
                
                            if (local_bit && lane_id) {
                                const magma_index_t idx = par_id
                                    * MAGMA_CSR5_OMEGA * B->csr5_sigma
                                    + lane_id * B->csr5_sigma;
                                // binary search
                                magma_index_t start = 0;
                                magma_index_t stop = row_stop - row_start - 1;
                                magma_index_t median, key_median;
                                while (stop >= start) {
                                    median = (stop + start) / 2;
                                    key_median = B->row[row_start+1+median];
                                    if (idx >= key_median)
                                        start = median + 1;
                                    else
                                        stop = median - 1;
                                }
                                const magma_index_t y_index = start-1;
                
                                B->tile_desc_offset[offset_pointer + y_offset]
                                    = y_index;
                
                                y_offset++;
                            }
                
                            for (int i = 1; i < B->csr5_sigma; i++) {
                                if ((!ly && i == bit_bitflag)
                                    || (ly && !(31 & (i - bit_bitflag))))
                                {
                                    ly++;
                                    descriptor = B->tile_desc[par_id
                                        * MAGMA_CSR5_OMEGA
                                        * B->csr5_num_packets
                                        + ly * MAGMA_CSR5_OMEGA + lane_id];
                                }
                                const int norm_i = 31 & (!ly
                                                        ? i : i - bit_bitflag);
                
                                local_bit = (descriptor >> (31 - norm_i))&0x1;
                
                                if (local_bit) {
                                    const magma_index_t idx = par_id
                                        * MAGMA_CSR5_OMEGA * B->csr5_sigma
                                        + lane_id * B->csr5_sigma + i;
                                    // binary search
                                    magma_index_t start = 0;
                                    magma_index_t stop = row_stop-row_start-1;
                                    magma_index_t median, key_median;
                                    while (stop >= start) {
                                        median = (stop + start) / 2;
                                        key_median=B->row[row_start+1+median];
                                        if (idx >= key_median)
                                            start = median + 1;
                                        else
                                            stop = median - 1;
                                    }
                                    const magma_index_t y_index = start-1;
                
                                    B->tile_desc_offset[offset_pointer
                                                        + y_offset] = y_index;
                
                                    y_offset++;
                                }
                            }
                        }
                    }
                }
                
                // step 3. transpose column_index and value arrays
                //#pragma omp parallel for
                for (int par_id = 0; par_id < B->csr5_p; par_id++) {
                    // if this is fast track tile, do not transpose it
                    if (B->tile_ptr[par_id] == B->tile_ptr[par_id + 1]) {
                        for (int idx = 0; idx < MAGMA_CSR5_OMEGA * B->csr5_sigma; idx++) {
                            int src_idx = par_id * MAGMA_CSR5_OMEGA
                                          * B->csr5_sigma + idx;
                            B->col[src_idx] = A.col[src_idx];
                            B->val[src_idx] = A.val[src_idx];
                        }
                        continue;
                    }
                    //#pragma simd
                    if (par_id < B->csr5_p-1) {
                        for (int idx = 0; idx < MAGMA_CSR5_OMEGA * B->csr5_sigma; idx++) {
                            int idx_y = idx % B->csr5_sigma;
                            int idx_x = idx / B->csr5_sigma;
                            int src_idx = par_id * MAGMA_CSR5_OMEGA
                                          * B->csr5_sigma + idx;
                            int dst_idx = par_id * MAGMA_CSR5_OMEGA
                                          * B->csr5_sigma + idx_y
                                          * MAGMA_CSR5_OMEGA + idx_x;
                
                            B->col[dst_idx] = A.col[src_idx];
                            B->val[dst_idx] = A.val[src_idx];
                        }
                    }
                    else { // the last tile
                        for (int idx = par_id * MAGMA_CSR5_OMEGA * B->csr5_sigma; idx < B->nnz; idx++) {
                            B->col[idx] = A.col[idx];
                            B->val[idx] = A.val[idx];
                        }
                    }
                }

                //printf( "done\n" );
            }

            else {
                printf("error: format not supported.\n");
                info = MAGMA_ERR_NOT_SUPPORTED;
            }
        }
        // anything to CSR
        else if ( new_format == Magma_CSR ) {
            // CSRU/CSRCSCU to CSR
            if ( old_format == Magma_CSRU ) {
                CHECK( magma_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
            }

            // CUCSR to CSR
            else if ( old_format == Magma_CUCSR ){
                CHECK(magma_zmconvert(A, B, Magma_CSR, Magma_CSR, queue));
            }

            // CSRD to CSR (diagonal elements first)
            else if ( old_format == Magma_CSRD ) {
                CHECK( magma_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
                for( magma_int_t i=0; i < A.num_rows; i++) {
                    magma_zindexsortval(
                    B->col,
                    B->val,
                    B->row[i],
                    B->row[i+1]-1,
                    queue );
                }
            }

            // CSRCOO to CSR
            else if ( old_format == Magma_CSRCOO ) {
                CHECK( magma_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
            }

            // CSRLIST to CSR
            else if ( old_format == Magma_CSRLIST ) {
                CHECK( magma_zmconvert( A, B, Magma_CSR, Magma_CSR, queue ));
                magma_int_t element, row, numnnz;

                numnnz = 0;
                // fill the rowpointer
                B->row[0] = 0;
                for( row=0; row<A.num_rows; row++ ){
                    element = A.row[row];
                    do{
                        B->val[ numnnz ] = A.val[ element ];
                        B->col[ numnnz ] = A.col[ element ];
                        numnnz++;
                        element = A.list[ element ];
                    }while( element != 0 );
                    B->row[ row+1 ] = numnnz;
                }
                // sort elements in every row according to col
                for( magma_int_t i=0; i < A.num_rows; i++) {
                    magma_zindexsortval(
                    B->col,
                    B->val,
                    B->row[i],
                    B->row[i+1]-1,
                    queue );
                }
            }

            // ELL/ELLPACK to CSR
            else if ( old_format == Magma_ELLPACKT ) {
                //printf( "Conversion to CSR: " );
                // fill in information for B
                B->storage_type = Magma_CSR;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;

                // conversion

                CHECK( magma_index_malloc_cpu( &row_tmp, A.num_rows+1 ));
                //fill the row-pointer
                for( magma_int_t i=0; i < A.num_rows+1; i++ )
                    row_tmp[i] = i*A.max_nnz_row;
                //now use AA_ELL, IA_ELL, row_tmp as CSR with some zeros.
                //The CSR compressor removes these
                CHECK( magma_z_csr_compressor(&A.val, &row_tmp, &A.col,
                           &B->val, &B->row, &B->col, &B->num_rows, queue ));
                B->nnz = B->row[B->num_rows];
            }

            // ELL (column-major) to CSR
            else if ( old_format == Magma_ELL ) {
                //printf( "Conversion to CSR: " );
                //fflush(stdout);
                // fill in information for B
                B->storage_type = Magma_CSR;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;

                // conversion
                CHECK( magma_zmalloc_cpu( &val_tmp, A.num_rows*A.max_nnz_row ));
                CHECK( magma_index_malloc_cpu( &row_tmp, A.num_rows+1 ));
                CHECK( magma_index_malloc_cpu( &col_tmp, A.num_rows*A.max_nnz_row ));

                //fill the row-pointer
                for( magma_int_t i=0; i < A.num_rows+1; i++ )
                    row_tmp[i] = i*A.max_nnz_row;
                //transform RowMajor to ColMajor
                for( magma_int_t j=0; j < A.max_nnz_row; j++ ) {
                    for( magma_int_t i=0; i < A.num_rows; i++ ) {
                        col_tmp[i*A.max_nnz_row+j] = A.col[j*A.num_rows+i];
                        val_tmp[i*A.max_nnz_row+j] = A.val[j*A.num_rows+i];
                    }
                }
                //now use AA_ELL, IA_ELL, row_tmp as CSR with some zeros.
                //The CSR compressor removes these
                CHECK( magma_z_csr_compressor(&val_tmp, &row_tmp, &col_tmp,
                           &B->val, &B->row, &B->col, &B->num_rows, queue ));

                B->nnz = B->row[B->num_rows];
            }

            // ELLD (ELLPACK with diagonal element first) to CSR
            else if ( old_format == Magma_ELLD ) {
          /*      CHECK( magma_zmconvert( A, B, Magma_ELL, Magma_CSR, queue ));
                for( magma_int_t i=0; i < A.num_rows; i++) {
                    magma_zindexsortval(
                    B->col,
                    B->val,
                    B->row[i],
                    B->row[i+1]-1,
                    queue );
                }
            */

                //printf( "Conversion to CSR: " );
                //fflush(stdout);
                // fill in information for B
                B->storage_type = Magma_CSR;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;

                // conversion
                CHECK( magma_index_malloc_cpu( &row_tmp, A.num_rows+1 ));
                //fill the row-pointer
                for( magma_int_t i=0; i < A.num_rows+1; i++ )
                    row_tmp[i] = i*A.max_nnz_row;
                // sort the diagonal element into the right place
                CHECK( magma_zmalloc_cpu( &val_tmp2, A.num_rows*A.max_nnz_row ));
                CHECK( magma_index_malloc_cpu( &col_tmp2, A.num_rows*A.max_nnz_row ));

                for( magma_int_t j=0; j < A.num_rows; j++ ) {
                    magma_index_t diagcol = A.col[j*A.max_nnz_row];
                    magma_int_t smaller = 0;
                    for( magma_int_t i=1; i < A.max_nnz_row; i++ ) {
                        if ( (A.col[j*A.max_nnz_row+i] < diagcol)
                             && (A.val[j*A.max_nnz_row+i] !=  zero) )
                            smaller++;
                    }
                    for( magma_int_t i=0; i < smaller; i++ ) {
                        col_tmp2[j*A.max_nnz_row+i] = A.col[j*A.max_nnz_row+i+1];
                        val_tmp2[j*A.max_nnz_row+i] = A.val[j*A.max_nnz_row+i+1];
                    }
                    col_tmp2[j*A.max_nnz_row+smaller] = A.col[j*A.max_nnz_row];
                    val_tmp2[j*A.max_nnz_row+smaller] = A.val[j*A.max_nnz_row];
                    for( magma_int_t i=smaller+1; i < A.max_nnz_row; i++ ) {
                        col_tmp2[j*A.max_nnz_row+i] = A.col[j*A.max_nnz_row+i];
                        val_tmp2[j*A.max_nnz_row+i] = A.val[j*A.max_nnz_row+i];
                    }
                }

                //now use AA_ELL, IA_ELL, row_tmp as CSR with some zeros.
                //The CSR compressor removes these
                CHECK( magma_z_csr_compressor(&val_tmp2, &row_tmp, &col_tmp2,
                           &B->val, &B->row, &B->col, &B->num_rows, queue ));
                B->nnz = B->row[B->num_rows];
            }

            // ELLRT to CSR
            else if ( old_format == Magma_ELLRT ) {
                //printf( "Conversion to CSR: " );
                //fflush(stdout);
                // fill in information for B
                B->storage_type = Magma_CSR;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;

                magma_int_t threads_per_row = A.alignment;
                magma_int_t rowlength = magma_roundup( A.max_nnz_row, threads_per_row );
                // conversion
                CHECK( magma_index_malloc_cpu( &row_tmp, A.num_rows+1 ));
                //fill the row-pointer
                for( magma_int_t i=0; i < A.num_rows+1; i++ )
                    row_tmp[i] = i*rowlength;
                //now use AA_ELL, IA_ELL, row_tmp as CSR with some zeros.
                //The CSR compressor removes these
                CHECK( magma_z_csr_compressor(&A.val, &row_tmp, &A.col,
                       &B->val, &B->row, &B->col, &B->num_rows, queue ));
                B->nnz = B->row[B->num_rows];
                //printf( "done\n" );
            }

            // SELLP to CSR
            else if ( old_format == Magma_SELLP ) {
                // printf( "Conversion to CSR: " );
                // fill in information for B
                B->storage_type = Magma_CSR;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;
                magma_int_t C = A.blocksize;
                magma_int_t slices = A.numblocks;
                B->blocksize = A.blocksize;
                B->numblocks = A.numblocks;
                // conversion
                CHECK( magma_zmalloc_cpu( &val_tmp,
                                          A.max_nnz_row*(A.num_rows+C) ));
                CHECK( magma_index_malloc_cpu( &row_tmp, A.num_rows+C ));
                CHECK( magma_index_malloc_cpu( &col_tmp,
                                               A.max_nnz_row*(A.num_rows+C) ));
                // zero everything
                for(magma_int_t i=0; i < A.max_nnz_row*(A.num_rows+C); i++ ) {
                    val_tmp[ i ] = MAGMA_Z_MAKE(0., 0.);
                    col_tmp[ i ] =  0;
                }

                //fill the row-pointer
                for( magma_int_t i=0; i < A.num_rows+1; i++ ) {
                    row_tmp[i] = A.max_nnz_row*i;
                }

                //transform RowMajor to ColMajor
                for( magma_int_t k=0; k < slices; k++) {
                    magma_int_t blockinfo = (A.row[k+1]-A.row[k])/A.blocksize;
                    for( magma_int_t j=0; j < C; j++ ) {
                        for( magma_int_t i=0; i < blockinfo; i++ ) {
                            col_tmp[ (k*C+j)*A.max_nnz_row+i ] =
                                                    A.col[A.row[k]+i*C+j];
                            val_tmp[ (k*C+j)*A.max_nnz_row+i ] =
                                                    A.val[A.row[k]+i*C+j];
                        }
                    }
                }

                //now use AA_ELL, IA_ELL, row_tmp as CSR with some zeros.
                //The CSR compressor removes these

                CHECK( magma_z_csr_compressor(&val_tmp, &row_tmp, &col_tmp,
                           &B->val, &B->row, &B->col, &B->num_rows, queue ));
                B->nnz = B->row[B->num_rows];
                //printf( "done\n" );
            }

            // CSR5 to CSR
            else if ( old_format == Magma_CSR5 ) {
                // printf( "Conversion to CSR: " );
                // fill in information for B
                B->storage_type = Magma_CSR;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;

                // conversion
                CHECK( magma_zmalloc_cpu( &B->val, B->nnz));
                CHECK( magma_index_malloc_cpu( &B->row, B->num_rows+1 ));
                CHECK( magma_index_malloc_cpu( &B->col, B->nnz ));

                for( magma_int_t i=0; i < A.num_rows+1; i++) {
                    B->row[i] = A.row[i];
                }

                // step 1. transpose column_index and value arrays
                //#pragma omp parallel for
                for (int par_id = 0; par_id < A.csr5_p; par_id++)
                {
                    // if this is fast track tile, do not transpose it
                    if (A.tile_ptr[par_id] == A.tile_ptr[par_id + 1])
                    {
                        for (int idx = 0; idx < MAGMA_CSR5_OMEGA * A.csr5_sigma; idx++) {
                            int src_idx = par_id * MAGMA_CSR5_OMEGA
                                          * A.csr5_sigma + idx;
                            B->col[src_idx] = A.col[src_idx];
                            B->val[src_idx] = A.val[src_idx];
                        }
                        continue;
                    }
                    if (par_id < A.csr5_p-1) {
                        //#pragma simd
                        for (int idx = 0; idx < MAGMA_CSR5_OMEGA * A.csr5_sigma;
                                idx++)
                        {
                            int idx_y = idx % MAGMA_CSR5_OMEGA;
                            int idx_x = idx / MAGMA_CSR5_OMEGA;
                            int src_idx = par_id * MAGMA_CSR5_OMEGA*A.csr5_sigma
                                      + idx;
                            int dst_idx = par_id * MAGMA_CSR5_OMEGA*A.csr5_sigma
                                      + idx_y * A.csr5_sigma + idx_x;
                            B->col[dst_idx] = A.col[src_idx];
                            B->val[dst_idx] = A.val[src_idx];
                        }
                    }
                    else // the last tile
                    {
                        for (int idx = par_id * MAGMA_CSR5_OMEGA
                                       * A.csr5_sigma; idx < A.nnz; idx++)
                        {
                            B->col[idx] = A.col[idx];
                            B->val[idx] = A.val[idx];
                        }
                    }
                }

                //printf( "done\n" );
            }

            // DENSE to CSR
            else if ( old_format == Magma_DENSE ) {
                //printf( "Conversion to CSR: " );
                // fill in information for B
                B->storage_type = Magma_CSR;
                B->memory_location = A.memory_location;
                B->fill_mode = A.fill_mode;
                B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
                B->num_cols = A.num_cols;
                B->nnz = A.nnz;
                B->max_nnz_row = A.max_nnz_row;
                B->diameter = A.diameter;

                // conversion

                B->nnz=0;
                for( magma_int_t i=0; i<(A.num_rows)*(A.num_cols); i++ ) {
                    if ( MAGMA_Z_REAL(A.val[i]) != 0.0 || MAGMA_Z_IMAG(A.val[i]) != 0.0 )
                        (B->nnz)++;
                }
                CHECK( magma_zmalloc_cpu( &B->val, B->nnz));
                CHECK( magma_index_malloc_cpu( &B->row, B->num_rows+1 ));
                CHECK( magma_index_malloc_cpu( &B->col, B->nnz ));

                magma_int_t i = 0;
                magma_int_t j = 0;
                magma_int_t k = 0;

                for(i=0; i<(A.num_rows)*(A.num_cols); i++)
                {
                    if ( i%(B->num_cols) == 0 )
                    {
                        (B->row)[k] = j;
                        k++;
                    }
                    if ( MAGMA_Z_REAL(A.val[i]) != 0 || MAGMA_Z_IMAG(A.val[i]) != 0)
                    {
                        (B->val)[j] = A.val[i];
                        (B->col)[j] = i%(B->num_cols);
                        j++;
                    }
                }
                (B->row)[B->num_rows]=B->nnz;

                //printf( "done\n" );
            }

            // BCSR to CSR
            else if ( old_format == Magma_BCSR ) {
                CHECK( magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, queue ) );
                CHECK( magma_zmconvert(dA, &dB, Magma_BCSR, Magma_CSR, queue ) );
                magma_zmfree( &dA, queue );
                CHECK( magma_zmtransfer(dB, B, Magma_DEV, Magma_CPU, queue ) );
                magma_zmfree( &dB, queue );
            }

            // COO to CSR
            else if ( old_format == Magma_COO ) {
                CHECK( magma_zmtransfer(A, &dA, Magma_CPU, Magma_DEV, queue ) );
                CHECK( magma_zmconvert(dA, &dB, Magma_COO, Magma_CSR, queue ) );
                magma_zmfree( &dA, queue );
                CHECK( magma_zmtransfer(dB, B, Magma_DEV, Magma_CPU, queue ) );
                magma_zmfree( &dB, queue );
            }

            else {
                printf("error: format not supported.\n");
                //magmablasSetKernelStream( queue );
                info = MAGMA_ERR_NOT_SUPPORTED;
            }
        }
        else {
            printf("error: conversion not supported.\n");
            //magmablasSetKernelStream( queue );
            info = MAGMA_ERR_NOT_SUPPORTED;
        }
    } // end CPU case
    else if ( A.memory_location == Magma_DEV ) {
        // CSR to CSR
        if ( old_format == Magma_CSR && new_format == Magma_CSR ) {
            CHECK( magma_zmtransfer( A, B, Magma_DEV, Magma_DEV, queue ));
        }
        // CSR to DENSE
        if ( old_format == Magma_CSR && new_format == Magma_DENSE ) {
            // use for now the workaround of using the CPU
            printf("%% warning: format not supported on GPU. "
            "Conversion handled by CPU.\n");
            CHECK( magma_zmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
            CHECK( magma_zmconvert( hA, &hB, old_format, new_format, queue ));
            CHECK( magma_zmtransfer( hB, B, Magma_CPU, A.memory_location, queue ));
            
            // // fill in information for B
            //B->storage_type = Magma_DENSE;
            //B->memory_location = A.memory_location;
            //B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
            //B->num_cols = A.num_cols;
            //B->nnz = A.nnz;
            //B->max_nnz_row = A.max_nnz_row;
            //B->diameter = A.diameter;
            //
            // // CUSPARSE context //
            //CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
            //CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
            //CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));
            //CHECK_CUSPARSE( cusparseSetMatType( descr, CUSPARSE_MATRIX_TYPE_GENERAL ));
            //CHECK_CUSPARSE( cusparseSetMatIndexBase( descr, CUSPARSE_INDEX_BASE_ZERO ));
            //// end CUSPARSE context //
            //
            //CHECK( magma_zmalloc( &B->dval, A.num_rows*A.num_cols ));
            //
            //
            // // conversion using CUSPARSE
            //cusparseZcsr2dense( cusparseHandle, A.num_rows, A.num_cols,
            //                    descr, A.dval, A.drow, A.dcol,
            //                    B->dval, max(A.num_rows, A.num_cols) );
        }
        // DENSE to CSR
        else if ( old_format == Magma_DENSE && new_format == Magma_CSR ) {
            // use for now the workaround of using the CPU
            printf("%% warning: format not supported on GPU. "
            "Conversion handled by CPU.\n");
            CHECK( magma_zmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
            CHECK( magma_zmconvert( hA, &hB, old_format, new_format, queue ));
            CHECK( magma_zmtransfer( hB, B, Magma_CPU, A.memory_location, queue ));
            
            //  // fill in information for B
            // B->storage_type = Magma_CSR;
            // B->memory_location = A.memory_location;
            // B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
            // B->num_cols = A.num_cols;
            // B->max_nnz_row = A.max_nnz_row;
            // B->diameter = A.diameter;
            // 
            //  // CUSPARSE context //
            // CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
            // CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
            // CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));
            //  // end CUSPARSE context //
            // 
            // 
            // intnnz = B->nnz;
            // CHECK( magma_index_malloc( &nnz_per_row, A.num_rows ));
            // cusparseZnnz( cusparseHandle, CUSPARSE_DIRECTION_COLUMN,
            //               A.num_rows, A.num_cols,
            //               descr,
            //               A.dval, A.num_rows, nnz_per_row, &intnnz );
            // 
            // CHECK( magma_zmalloc( &B->dval, B->nnz ));
            // CHECK( magma_index_malloc( &B->drow, B->num_rows+1 ));
            // CHECK( magma_index_malloc( &B->dcol, B->nnz ));
            // 
            // 
            // // conversion using CUSPARSE
            // cusparseZdense2csr( cusparseHandle, A.num_rows, A.num_cols,
            //                     descr,
            //                     A.dval, A.num_rows, nnz_per_row,
            //                     B->dval, B->drow, B->dcol );
        }
        // CSR to BCSR
        else if ( old_format == Magma_CSR && new_format == Magma_BCSR ) {
            //printf( "Conversion to BCSR: " );
            // fill in information for B
            B->storage_type = Magma_BCSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            magma_int_t size_b = B->blocksize;

            // CUSPARSE context //
            CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
            CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
            CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));
            // end CUSPARSE context //

            magma_index_t base, nnzb;
            magma_int_t mb = magma_ceildiv( A.num_rows, size_b );
            // nnzTotalDevHostPtr points to host memory
            magma_index_t *nnzTotalDevHostPtr = &nnzb;

            CHECK( magma_index_malloc( &B->drow, mb+1 ));
            cusparseXcsr2bsrNnz( cusparseHandle, CUSPARSE_DIRECTION_COLUMN,
                                 A.num_rows, A.num_cols, descr,
                                 A.drow, A.dcol, size_b,
                                 descr, B->drow, nnzTotalDevHostPtr );


            if (NULL != nnzTotalDevHostPtr) {
                nnzb = *nnzTotalDevHostPtr;
            } else {
                magma_index_getvector( 1, B->row+mb, 1, &nnzb, 1, queue );
                magma_index_getvector( 1, B->row, 1, &base, 1, queue );
                nnzb -= base;
            }
            B->numblocks = nnzb; // number of blocks

            CHECK( magma_zmalloc( &B->dval, nnzb*size_b*size_b ));
            CHECK( magma_index_malloc( &B->dcol, nnzb ));


            // conversion using CUSPARSE
            cusparseZcsr2bsr( cusparseHandle, CUSPARSE_DIRECTION_ROW,
                              A.num_rows, A.num_cols, descr,
                              A.dval, A.drow, A.dcol,
                              size_b, descr,
                              B->dval, B->drow, B->dcol);
        }
        // BCSR to CSR
        else if ( old_format == Magma_BCSR && new_format == Magma_CSR ) {
            //printf( "Conversion to CSR: " );
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->diameter = A.diameter;

            magma_int_t size_b = A.blocksize;

            // CUSPARSE context //
            CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
            CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
            CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));
            // end CUSPARSE context //

            magma_int_t mb = magma_ceildiv( A.num_rows, size_b );
            magma_int_t nb = magma_ceildiv( A.num_cols, size_b );
            magma_int_t nnzb = A.numblocks; // number of blocks
            B->nnz  = nnzb * size_b * size_b; // number of elements
            B->num_rows = mb * size_b;
            B->num_cols = nb * size_b;

            CHECK( magma_zmalloc( &B->dval, B->nnz ));
            CHECK( magma_index_malloc( &B->drow, B->num_rows+1 ));
            CHECK( magma_index_malloc( &B->dcol, B->nnz ));


            // conversion using CUSPARSE
            cusparseZbsr2csr( cusparseHandle, CUSPARSE_DIRECTION_ROW,
                              mb, nb, descr, A.dval, A.drow, A.dcol,
                              size_b, descr,
                              B->dval, B->drow, B->dcol );
        }
        // CSR to CSC
        else if ( old_format == Magma_CSR && new_format == Magma_CSC ) {
            // fill in information for B
            B->storage_type = Magma_CSC;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // CUSPARSE context //
            CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
            CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
            CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));
            // end CUSPARSE context //

            CHECK( magma_zmalloc( &B->dval, B->nnz ));
            CHECK( magma_index_malloc( &B->drow, B->nnz ));
            CHECK( magma_index_malloc( &B->dcol, B->num_cols+1 ));


            // conversion using CUSPARSE
            cusparseZcsr2csc(cusparseHandle, A.num_rows, A.num_cols, A.nnz,
                             A.dval, A.drow, A.dcol,
                             B->dval, B->drow, B->dcol,
                             CUSPARSE_ACTION_NUMERIC,
                             CUSPARSE_INDEX_BASE_ZERO);
        }
        // CSC to CSR
        else if ( old_format == Magma_CSC && new_format == Magma_CSR ) {
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // CUSPARSE context //
            CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
            CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
            CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));
            // end CUSPARSE context //

            CHECK( magma_zmalloc( &B->dval, B->nnz ));
            CHECK( magma_index_malloc( &B->drow, B->num_rows+1 ));
            CHECK( magma_index_malloc( &B->dcol, B->nnz ));


            // conversion using CUSPARSE
            cusparseZcsr2csc(cusparseHandle, A.num_cols, A.num_rows, A.nnz,
                             A.dval, A.dcol, A.drow,
                             B->dval, B->dcol, B->drow,
                             CUSPARSE_ACTION_NUMERIC,
                             CUSPARSE_INDEX_BASE_ZERO);
        }
        // CSR to COO
        else if ( old_format == Magma_CSR && new_format == Magma_COO ) {
            // fill in information for B
            B->storage_type = Magma_COO;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;

            // CUSPARSE context //
            CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
            CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
            CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));
            // end CUSPARSE context //

            CHECK( magma_zmalloc( &B->dval, B->nnz ));
            CHECK( magma_index_malloc( &B->drow, B->nnz ));
            CHECK( magma_index_malloc( &B->dcol, B->nnz ));


            magma_zcopyvector( A.nnz, A.dval, 1, B->dval, 1, queue );
            magma_index_copyvector( A.nnz, A.dcol, 1, B->dcol, 1, queue );

            // conversion using CUSPARSE
            cusparseXcsr2coo( cusparseHandle, A.drow,
                              A.nnz, A.num_rows, B->drow,
                              CUSPARSE_INDEX_BASE_ZERO );
        }
        // COO to CSR
        else if ( old_format == Magma_COO && new_format == Magma_CSR ) {
#if CUDA_VERSION >= 7000
            // fill in information for B
            B->storage_type = Magma_CSR;
            B->memory_location = A.memory_location;
            B->num_rows = A.num_rows; B->true_nnz = A.true_nnz;
            B->num_cols = A.num_cols;
            B->nnz = A.nnz;
            B->max_nnz_row = A.max_nnz_row;
            B->diameter = A.diameter;
            int m = A.num_rows;
            int n = A.num_cols;
            int nnz = A.nnz;

            // CUSPARSE context //

            size_t pBufferSizeInBytes = 0;
            void *pBuffer = NULL;
            int *P = NULL;

            CHECK_CUSPARSE( cusparseCreate( &cusparseHandle ));
            CHECK_CUSPARSE( cusparseSetStream( cusparseHandle, queue->cuda_stream() ));
            CHECK_CUSPARSE( cusparseCreateMatDescr( &descr ));
            // end CUSPARSE context //

            CHECK( magma_zmalloc( &B->dval, B->nnz ));
            CHECK( magma_index_malloc( &B->drowidx, B->nnz ));
            CHECK( magma_index_malloc( &B->dcol, B->nnz ));
            CHECK( magma_index_malloc( &B->drow, B->num_rows + 1 ));

            magma_zcopyvector( A.nnz, A.dval, 1, B->dval, 1, queue );
            magma_index_copyvector( A.nnz, A.dcol, 1, B->dcol, 1, queue );
            magma_index_copyvector( A.nnz, A.drowidx, 1, B->drowidx, 1, queue );

            // step 1: allocate buffer
            cusparseXcoosort_bufferSizeExt(cusparseHandle, m, n, nnz, B->drowidx, B->dcol, &pBufferSizeInBytes);
            //cudaMalloc( &pBuffer, sizeof(char)* pBufferSizeInBytes);
            CHECK( magma_malloc( &pBuffer, sizeof(char)* pBufferSizeInBytes ));
            // step 2: setup permutation vector P to identity
            CHECK( magma_index_malloc( &P, nnz ));
            //magma_( &P, sizeof(int)*nnz);
            cusparseCreateIdentityPermutation(cusparseHandle, nnz, P);

            // step 3: sort COO format by Row
            cusparseXcoosortByRow(cusparseHandle, m, n, nnz, B->drowidx, B->dcol, P, pBuffer);

            // step 4: gather sorted cooVals
            cusparseZgthr(cusparseHandle, nnz, A.dval, B->dval, P, CUSPARSE_INDEX_BASE_ZERO);


            // conversion using CUSPARSE
            cusparseXcoo2csr( cusparseHandle, B->drowidx,
                              A.nnz, A.num_rows, B->drow,
                              CUSPARSE_INDEX_BASE_ZERO );
            magma_free( B->drowidx );
            magma_free( pBuffer );
            magma_free( P );
#else
                printf("error: conversion on GPU only supported for CUDA version >= 7.0.\n");

#endif
        }
        else {
            printf("%% warning: format not supported on GPU. "
            "Conversion handled by CPU.\n");
            CHECK( magma_zmtransfer( A, &hA, A.memory_location, Magma_CPU, queue ));
            CHECK( magma_zmconvert( hA, &hB, old_format, new_format, queue ));
            CHECK( magma_zmtransfer( hB, B, Magma_CPU, A.memory_location, queue ));
        }
    }

cleanup:
    cusparseDestroyMatDescr(descr);
    cusparseDestroy(cusparseHandle);
    descr = NULL;
    cusparseHandle = NULL;
    magma_free( nnz_per_row );
    magma_free_cpu( row_tmp );
    magma_free_cpu( col_tmp );
    magma_free_cpu( val_tmp );
    magma_free_cpu( row_tmp2 );
    magma_free_cpu( col_tmp2 );
    magma_free_cpu( val_tmp2 );
    row_tmp = NULL;
    col_tmp = NULL;
    val_tmp = NULL;
    row_tmp2 = NULL;
    col_tmp2 = NULL;
    val_tmp2 = NULL;
    magma_free( transpose );
    magma_free_cpu( length );
    length = NULL;
    magma_zmfree( &hA, queue );
    magma_zmfree( &hB, queue );
    magma_zmfree( &dA, queue );
    magma_zmfree( &dB, queue );
    if ( info != 0 ) {
        magma_zmfree( B, queue );
    }
    return info;
}
