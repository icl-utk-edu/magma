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
            if (A->ownership) {
                magma_free_cpu( A->val );
                magma_free_cpu( A->col );
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if (A->storage_type == Magma_ELLD ) {
            if (A->ownership) {
                magma_free_cpu( A->val );
                magma_free_cpu( A->col );
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_ELLRT ) {
            if (A->ownership) {
                magma_free_cpu( A->val );
                magma_free_cpu( A->row );
                magma_free_cpu( A->col );
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_SELLP ) {
            if (A->ownership) {
                magma_free_cpu( A->val );
                magma_free_cpu( A->row );
                magma_free_cpu( A->col );
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_CSR5 ) {
            if (A->ownership) {
                magma_free_cpu( A->val );
                magma_free_cpu( A->row );
                magma_free_cpu( A->col );
                magma_free_cpu( A->tile_ptr );
                magma_free_cpu( A->tile_desc );
                magma_free_cpu( A->tile_desc_offset_ptr );
                magma_free_cpu( A->tile_desc_offset );
                magma_free_cpu( A->calibrator );
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
            if (A->ownership) {
                magma_free_cpu( A->val );
                magma_free_cpu( A->row );
                magma_free_cpu( A->col );
                magma_free_cpu( A->list );
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
            if (A->ownership) {
                magma_free_cpu( A->val );
                magma_free_cpu( A->col );
                magma_free_cpu( A->row );
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if (  A->storage_type == Magma_CSRCOO ) {
            if (A->ownership) {
                magma_free_cpu( A->val );
                magma_free_cpu( A->col );
                magma_free_cpu( A->row );
                magma_free_cpu( A->rowidx );
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_BCSR ) {
            if (A->ownership) {
                magma_free_cpu( A->val );
                magma_free_cpu( A->col );
                magma_free_cpu( A->row );
                magma_free_cpu( A->blockinfo );
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
            A->blockinfo = 0;
        }
        if ( A->storage_type == Magma_DENSE ) {
            if (A->ownership) {
                magma_free_cpu( A->val );
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

    if ( A->memory_location == Magma_DEV ) {
        if (A->storage_type == Magma_ELL || A->storage_type == Magma_ELLPACKT) {
            if (A->ownership) {
                if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                    printf("Memory Free Error.\n");
                    return MAGMA_ERR_INVALID_PTR; 
                }
                if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                    printf("Memory Free Error.\n");
                    return MAGMA_ERR_INVALID_PTR; 
                }
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_ELLD ) {
            if (A->ownership) {
                if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                    printf("Memory Free Error.\n");
                    return MAGMA_ERR_INVALID_PTR; 
                }
                if ( magma_free( A->dcol ) != MAGMA_SUCCESS ) {
                    printf("Memory Free Error.\n");
                    return MAGMA_ERR_INVALID_PTR; 
                }
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_ELLRT ) {
            if (A->ownership) {
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
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_SELLP ) {
            if (A->ownership) {
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
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_CSR5 ) {
            if (A->ownership) {
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
            if (A->ownership) {
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
            if (A->ownership) {
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
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if (  A->storage_type == Magma_CSRCOO ) {
            if (A->ownership) {
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
            }
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_BCSR ) {
            if (A->ownership) {
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
            }
            A->blockinfo = NULL;
            A->num_rows = 0;
            A->num_cols = 0;
            A->nnz = 0; A->true_nnz = 0;
        }
        if ( A->storage_type == Magma_DENSE ) {
            if (A->ownership) {
                if ( magma_free( A->dval ) != MAGMA_SUCCESS ) {
                    printf("Memory Free Error.\n");
                    return MAGMA_ERR_INVALID_PTR; 
                }
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





/**
    Purpose
    -------

    Free a preconditioner.


    Arguments
    ---------

    @param[in,out]
    precond_par magma_z_preconditioner*
                structure containing all preconditioner information
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zaux
    ********************************************************************/

extern "C" magma_int_t
magma_zprecondfree(
    magma_z_preconditioner *precond_par,
    magma_queue_t queue ){

    if ( precond_par->d.val != NULL ) {
        magma_free( precond_par->d.val );
        precond_par->d.val = NULL;
    }
    if ( precond_par->d2.val != NULL ) {
        magma_free( precond_par->d2.val );
        precond_par->d2.val = NULL;
    }
    if ( precond_par->work1.val != NULL ) {
        magma_free( precond_par->work1.val );
        precond_par->work1.val = NULL;
    }
    if ( precond_par->work2.val != NULL ) {
        magma_free( precond_par->work2.val );
        precond_par->work2.val = NULL;
    }
    if ( precond_par->M.val != NULL ) {
        if ( precond_par->M.memory_location == Magma_DEV )
            magma_free( precond_par->M.dval );
        else
            magma_free_cpu( precond_par->M.val );
        precond_par->M.val = NULL;
    }
    if ( precond_par->M.col != NULL ) {
        if ( precond_par->M.memory_location == Magma_DEV )
            magma_free( precond_par->M.dcol );
        else
            magma_free_cpu( precond_par->M.col );
        precond_par->M.col = NULL;
    }
    if ( precond_par->M.row != NULL ) {
        if ( precond_par->M.memory_location == Magma_DEV )
            magma_free( precond_par->M.drow );
        else
            magma_free_cpu( precond_par->M.row );
        precond_par->M.row = NULL;
    }
    if ( precond_par->M.blockinfo != NULL ) {
        magma_free_cpu( precond_par->M.blockinfo );
        precond_par->M.blockinfo = NULL;
    }
    if ( precond_par->L.val != NULL ) {
        if ( precond_par->L.memory_location == Magma_DEV )
            magma_free( precond_par->L.dval );
        else
            magma_free_cpu( precond_par->L.val );
        precond_par->L.val = NULL;
    }
    if ( precond_par->L.col != NULL ) {
        if ( precond_par->L.memory_location == Magma_DEV )
            magma_free( precond_par->L.col );
        else
            magma_free_cpu( precond_par->L.dcol );
        precond_par->L.col = NULL;
    }
    if ( precond_par->L.row != NULL ) {
        if ( precond_par->L.memory_location == Magma_DEV )
            magma_free( precond_par->L.drow );
        else
            magma_free_cpu( precond_par->L.row );
        precond_par->L.row = NULL;
    }
    if ( precond_par->L.blockinfo != NULL ) {
        magma_free_cpu( precond_par->L.blockinfo );
        precond_par->L.blockinfo = NULL;
    }
    if ( precond_par->LT.val != NULL ) {
        if ( precond_par->LT.memory_location == Magma_DEV )
            magma_free( precond_par->LT.dval );
        else
            magma_free_cpu( precond_par->LT.val );
        precond_par->LT.val = NULL;
    }
    if ( precond_par->LT.col != NULL ) {
        if ( precond_par->LT.memory_location == Magma_DEV )
            magma_free( precond_par->LT.col );
        else
            magma_free_cpu( precond_par->LT.dcol );
        precond_par->LT.col = NULL;
    }
    if ( precond_par->LT.row != NULL ) {
        if ( precond_par->LT.memory_location == Magma_DEV )
            magma_free( precond_par->LT.drow );
        else
            magma_free_cpu( precond_par->LT.row );
        precond_par->LT.row = NULL;
    }
    if ( precond_par->LT.blockinfo != NULL ) {
        magma_free_cpu( precond_par->LT.blockinfo );
        precond_par->LT.blockinfo = NULL;
    }
    if ( precond_par->U.val != NULL ) {
        if ( precond_par->U.memory_location == Magma_DEV )
            magma_free( precond_par->U.dval );
        else
            magma_free_cpu( precond_par->U.val );
        precond_par->U.val = NULL;
    }
    if ( precond_par->U.col != NULL ) {
        if ( precond_par->U.memory_location == Magma_DEV )
            magma_free( precond_par->U.dcol );
        else
            magma_free_cpu( precond_par->U.col );
        precond_par->U.col = NULL;
    }
    if ( precond_par->U.row != NULL ) {
        if ( precond_par->U.memory_location == Magma_DEV )
            magma_free( precond_par->U.drow );
        else
            magma_free_cpu( precond_par->U.row );
        precond_par->U.row = NULL;
    }
    if ( precond_par->U.blockinfo != NULL ) {
        magma_free_cpu( precond_par->U.blockinfo );
        precond_par->U.blockinfo = NULL;
    }
    if ( precond_par->UT.val != NULL ) {
        if ( precond_par->UT.memory_location == Magma_DEV )
            magma_free( precond_par->UT.dval );
        else
            magma_free_cpu( precond_par->UT.val );
        precond_par->UT.val = NULL;
    }
    if ( precond_par->UT.col != NULL ) {
        if ( precond_par->UT.memory_location == Magma_DEV )
            magma_free( precond_par->UT.col );
        else
            magma_free_cpu( precond_par->UT.dcol );
        precond_par->UT.col = NULL;
    }
    if ( precond_par->UT.row != NULL ) {
        if ( precond_par->UT.memory_location == Magma_DEV )
            magma_free( precond_par->UT.drow );
        else
            magma_free_cpu( precond_par->UT.row );
        precond_par->UT.row = NULL;
    }
    if ( precond_par->UT.blockinfo != NULL ) {
        magma_free_cpu( precond_par->UT.blockinfo );
        precond_par->UT.blockinfo = NULL;
    }
    if ( precond_par->solver == Magma_ILU ||
        precond_par->solver == Magma_PARILU ||
        precond_par->solver == Magma_ICC||
        precond_par->solver == Magma_PARIC ) {
        if( precond_par->cuinfoL != NULL )
            cusparseDestroySolveAnalysisInfo( precond_par->cuinfoL ); 
        if( precond_par->cuinfoU != NULL )
            cusparseDestroySolveAnalysisInfo( precond_par->cuinfoU ); 
        precond_par->cuinfoL = NULL;
        precond_par->cuinfoU = NULL;
        if( precond_par->cuinfoLT != NULL )
            cusparseDestroySolveAnalysisInfo( precond_par->cuinfoLT ); 
        if( precond_par->cuinfoUT != NULL )
            cusparseDestroySolveAnalysisInfo( precond_par->cuinfoUT ); 
        precond_par->cuinfoLT = NULL;
        precond_par->cuinfoUT = NULL;
    }
    if ( precond_par->LD.val != NULL ) {
        if ( precond_par->LD.memory_location == Magma_DEV )
            magma_free( precond_par->LD.dval );
        else
            magma_free_cpu( precond_par->LD.val );
        precond_par->LD.val = NULL;
    }
    if ( precond_par->LD.col != NULL ) {
        if ( precond_par->LD.memory_location == Magma_DEV )
            magma_free( precond_par->LD.dcol );
        else
            magma_free_cpu( precond_par->LD.col );
        precond_par->LD.col = NULL;
    }
    if ( precond_par->LD.row != NULL ) {
        if ( precond_par->LD.memory_location == Magma_DEV )
            magma_free( precond_par->LD.drow );
        else
            magma_free_cpu( precond_par->LD.row );
        precond_par->LD.row = NULL;
    }
    if ( precond_par->LD.blockinfo != NULL ) {
        magma_free_cpu( precond_par->LD.blockinfo );
        precond_par->LD.blockinfo = NULL;
    }
    if ( precond_par->UD.val != NULL ) {
        if ( precond_par->UD.memory_location == Magma_DEV )
            magma_free( precond_par->UD.dval );
        else
            magma_free_cpu( precond_par->UD.val );
        precond_par->UD.val = NULL;
    }
    if ( precond_par->UD.col != NULL ) {
        if ( precond_par->UD.memory_location == Magma_DEV )
            magma_free( precond_par->UD.dcol );
        else
            magma_free_cpu( precond_par->UD.col );
        precond_par->UD.col = NULL;
    }
    if ( precond_par->UD.row != NULL ) {
        if ( precond_par->UD.memory_location == Magma_DEV )
            magma_free( precond_par->UD.drow );
        else
            magma_free_cpu( precond_par->UD.row );
        precond_par->UD.row = NULL;
    }
    if ( precond_par->UD.blockinfo != NULL ) {
        magma_free_cpu( precond_par->UD.blockinfo );
        precond_par->UD.blockinfo = NULL;
    }
    if ( precond_par->LDT.val != NULL ) {
        if ( precond_par->LDT.memory_location == Magma_DEV )
            magma_free( precond_par->LDT.dval );
        else
            magma_free_cpu( precond_par->LDT.val );
        precond_par->LDT.val = NULL;
    }
    if ( precond_par->LDT.col != NULL ) {
        if ( precond_par->LDT.memory_location == Magma_DEV )
            magma_free( precond_par->LDT.dcol );
        else
            magma_free_cpu( precond_par->LDT.col );
        precond_par->LDT.col = NULL;
    }
    if ( precond_par->LDT.row != NULL ) {
        if ( precond_par->LDT.memory_location == Magma_DEV )
            magma_free( precond_par->LDT.drow );
        else
            magma_free_cpu( precond_par->LDT.row );
        precond_par->LDT.row = NULL;
    }
    if ( precond_par->LDT.blockinfo != NULL ) {
        magma_free_cpu( precond_par->LDT.blockinfo );
        precond_par->LDT.blockinfo = NULL;
    }
    if ( precond_par->UDT.val != NULL ) {
        if ( precond_par->UDT.memory_location == Magma_DEV )
            magma_free( precond_par->UDT.dval );
        else
            magma_free_cpu( precond_par->UDT.val );
        precond_par->UDT.val = NULL;
    }
    if ( precond_par->UDT.col != NULL ) {
        if ( precond_par->UDT.memory_location == Magma_DEV )
            magma_free( precond_par->UDT.dcol );
        else
            magma_free_cpu( precond_par->UDT.col );
        precond_par->UDT.col = NULL;
    }
    if ( precond_par->UDT.row != NULL ) {
        if ( precond_par->UDT.memory_location == Magma_DEV )
            magma_free( precond_par->UDT.drow );
        else
            magma_free_cpu( precond_par->UDT.row );
        precond_par->UDT.row = NULL;
    }
    if ( precond_par->UDT.blockinfo != NULL ) {
        magma_free_cpu( precond_par->UDT.blockinfo );
        precond_par->UDT.blockinfo = NULL;
    }
    if ( precond_par->L_dgraphindegree != NULL ) {
        magma_free( precond_par->L_dgraphindegree );
        precond_par->L_dgraphindegree = NULL;
    }
    if ( precond_par->L_dgraphindegree_bak != NULL ) {
        magma_free( precond_par->L_dgraphindegree_bak );
        precond_par->L_dgraphindegree_bak = NULL;
    }
    if ( precond_par->U_dgraphindegree != NULL ) {
        magma_free( precond_par->U_dgraphindegree );
        precond_par->U_dgraphindegree = NULL;
    }
    if ( precond_par->U_dgraphindegree_bak != NULL ) {
        magma_free( precond_par->U_dgraphindegree_bak );
        precond_par->U_dgraphindegree_bak = NULL;
    }

    precond_par->solver = Magma_NONE;
    
    return MAGMA_SUCCESS;

}


