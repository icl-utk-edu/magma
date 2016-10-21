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


/**
    Purpose
    -------

    Free the memory of a magma_z_matrix.


    Arguments
    ---------

    @param[in,out]
    A           magma_z_matrix*
                matrix to free
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zmfree(
    magma_z_matrix *A,
    magma_queue_t queue )
{
    if ( A->memory_location == Magma_CPU ) {
        if (A->storage_type == Magma_ELL || A->storage_type == Magma_ELLPACKT) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if (A->storage_type == Magma_ELLD ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_ELLRT ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->row );
            magma_free_cpu( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_SELLP ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->row );
            magma_free_cpu( A->col );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_CSR5 ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->row );
            magma_free_cpu( A->col );
            magma_free_cpu( A->tile_ptr );
            magma_free_cpu( A->tile_desc );
            magma_free_cpu( A->tile_desc_offset_ptr );
            magma_free_cpu( A->tile_desc_offset );
            magma_free_cpu( A->calibrator );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
            A->csr5_sigma = 0;
            A->csr5_bit_y_offset = 0;
            A->csr5_bit_scansum_offset = 0;
            A->csr5_num_packets = 0;
            A->csr5_p = 0;
            A->csr5_num_offsets = 0;
            A->csr5_tail_tile_start = 0;
        }
        if ( A->storage_type == Magma_CSRLIST ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->row );
            magma_free_cpu( A->col );
            magma_free_cpu( A->list );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_CSR  ||
             A->storage_type == Magma_CSC  ||
             A->storage_type == Magma_CSRD ||
             A->storage_type == Magma_CSRL ||
             A->storage_type == Magma_CSRU )
        {
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            magma_free_cpu( A->row );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if (  A->storage_type == Magma_CSRCOO ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            magma_free_cpu( A->row );
            magma_free_cpu( A->rowidx );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_BCSR ) {
            magma_free_cpu( A->val );
            magma_free_cpu( A->col );
            magma_free_cpu( A->row );
            magma_free_cpu( A->blockinfo );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
            A->blockinfo = 0;
        }
        if ( A->storage_type == Magma_DENSE ) {
            magma_free_cpu( A->val );
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        A->val = NULL;
        A->col = NULL;
        A->row = NULL;
        A->rowidx = NULL;
        A->blockinfo = NULL;
        A->diag = NULL;
        A->dval = NULL;
        A->dcol = NULL;
        A->drow = NULL;
        A->drowidx = NULL;
        A->ddiag = NULL;
        A->dlist = NULL;
        A->list = NULL;
        A->tile_ptr = NULL;
        A->dtile_ptr = NULL;
        A->tile_desc = NULL;
        A->dtile_desc = NULL;
        A->tile_desc_offset_ptr = NULL;
        A->dtile_desc_offset_ptr = NULL;
        A->tile_desc_offset = NULL;
        A->dtile_desc_offset = NULL;
        A->calibrator = NULL;
        A->dcalibrator = NULL;
    }

    if ( A->memory_location == Magma_DEV ) {
        if (A->storage_type == Magma_ELL || A->storage_type == Magma_ELLPACKT) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_ELLD ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_ELLRT ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_SELLP ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_CSR5 ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dtile_ptr ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dtile_desc ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dtile_desc_offset_ptr ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dtile_desc_offset ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcalibrator ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
            A->csr5_sigma = 0;
            A->csr5_bit_y_offset = 0;
            A->csr5_bit_scansum_offset = 0;
            A->csr5_num_packets = 0;
            A->csr5_p = 0;
            A->csr5_num_offsets = 0;
            A->csr5_tail_tile_start = 0;
        }
        if ( A->storage_type == Magma_CSRLIST ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dlist ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_CSR  ||
             A->storage_type == Magma_CSC  ||
             A->storage_type == Magma_CSRD ||
             A->storage_type == Magma_CSRL ||
             A->storage_type == Magma_CSRU )
        {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if (  A->storage_type == Magma_CSRCOO ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drowidx ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_BCSR ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->drow ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }
            magma_free_cpu( A->blockinfo );
            A->blockinfo = NULL;
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_DENSE ) {
            if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                printf("Memory Free Error.\n");
                return MAGMA_ERR_INVALID_PTR; 
            }

            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        A->val = NULL;
        A->col = NULL;
        A->row = NULL;
        A->rowidx = NULL;
        A->blockinfo = NULL;
        A->diag = NULL;
        A->dval = NULL;
        A->dcol = NULL;
        A->drow = NULL;
        A->drowidx = NULL;
        A->ddiag = NULL;
        A->dlist = NULL;
        A->list = NULL;
        A->tile_ptr = NULL;
        A->dtile_ptr = NULL;
        A->tile_desc = NULL;
        A->dtile_desc = NULL;
        A->tile_desc_offset_ptr = NULL;
        A->dtile_desc_offset_ptr = NULL;
        A->tile_desc_offset = NULL;
        A->dtile_desc_offset = NULL;
        A->calibrator = NULL;
        A->dcalibrator = NULL;
    }

    else {
        // printf("Memory Free Error.\n");
        return MAGMA_ERR_INVALID_PTR;
    }
    return MAGMA_SUCCESS;
}
