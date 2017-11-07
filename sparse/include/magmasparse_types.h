/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date
*/

#ifndef MAGMASPARSE_TYPES_H
#define MAGMASPARSE_TYPES_H


#if defined(HAVE_PASTIX)
//PaStiX include
#include <stdint.h>
/* to access functions from the libpastix, respect this order */
#include <pastix.h>
#include <read_matrix.h>
#include <get_options.h>
#include <assert.h>
#include <omp.h>
#endif

// includes CUDA
#include <cusparse_v2.h>



#ifdef __cplusplus
extern "C" {
#endif


#define MAGMA_CSR5_OMEGA 32

typedef struct magma_z_matrix
{
    magma_storage_t    storage_type;            // matrix format - CSR, ELL, SELL-P, CSR5
    magma_location_t   memory_location;         // CPU or DEV
    magma_symmetry_t   sym;                     // opt: indicate symmetry
    magma_diagorder_t  diagorder_type;          // opt: only needed for factorization matrices
    magma_uplo_t   fill_mode;               // fill mode full/lower/upper
    magma_int_t        num_rows;                // number of rows
    magma_int_t        num_cols;                // number of columns
    magma_int_t        nnz;                     // opt: number of nonzeros
    magma_int_t        max_nnz_row;             // opt: max number of nonzeros in one row
    magma_int_t        diameter;                // opt: max distance of entry from main diagonal
    magma_int_t        true_nnz;              // opt: true nnz
    magma_bool_t       ownership;               // does MAGMA own the arrays of this matrix structure
    union {
        magmaDoubleComplex      *val;           // array containing values in CPU case
        magmaDoubleComplex_ptr  dval;           // array containing values in DEV case
    };
    union {
        magmaDoubleComplex      *diag;          // opt: diagonal entries in CPU case
        magmaDoubleComplex_ptr  ddiag;          // opt: diagonal entries in DEV case
    };
    union {
        magma_index_t           *row;           // opt: row pointer CPU case
        magmaIndex_ptr          drow;           // opt: row pointer DEV case
    };
    union {
        magma_index_t           *rowidx;        // opt: array containing row indices CPU case
        magmaIndex_ptr          drowidx;        // opt: array containing row indices DEV case
    };
    union {
        magma_index_t           *col;           // opt: array containing col indices CPU case
        magmaIndex_ptr          dcol;           // opt: array containing col indices DEV case
    };
    union {
        magma_index_t           *list;          // opt: linked list pointing to next element
        magmaIndex_ptr          dlist;          // opt: linked list pointing to next element
    };
    union {
        magma_uindex_t           *tile_ptr;              // opt: CSR5 tile pointer CPU case
        magmaUIndex_ptr          dtile_ptr;              // opt: CSR5 tile pointer DEV case
    };
    union {
        magma_uindex_t           *tile_desc;             // opt: CSR5 tile descriptor CPU case
        magmaUIndex_ptr          dtile_desc;             // opt: CSR5 tile descriptor DEV case
    };
    union {
        magma_index_t            *tile_desc_offset_ptr;  // opt: CSR5 tile descriptor offset pointer CPU case
        magmaIndex_ptr           dtile_desc_offset_ptr;  // opt: CSR5 tile descriptor offset pointer DEV case
    };
    union {
        magma_index_t            *tile_desc_offset;      // opt: CSR5 tile descriptor offset CPU case
        magmaIndex_ptr           dtile_desc_offset;      // opt: CSR5 tile descriptor offset DEV case
    };
    union {
        magmaDoubleComplex       *calibrator;            // opt: CSR5 calibrator CPU case
        magmaDoubleComplex_ptr   dcalibrator;            // opt: CSR5 calibrator DEV case
    };
    magma_index_t      *blockinfo;              // opt: for BCSR format CPU case
    magma_int_t        blocksize;               // opt: info for SELL-P/BCSR
    magma_int_t        numblocks;               // opt: info for SELL-P/BCSR
    magma_int_t        alignment;               // opt: info for SELL-P/BCSR
    magma_int_t        csr5_sigma;              // opt: info for CSR5
    magma_int_t        csr5_bit_y_offset;       // opt: info for CSR5
    magma_int_t        csr5_bit_scansum_offset; // opt: info for CSR5
    magma_int_t        csr5_num_packets;        // opt: info for CSR5
    magma_index_t      csr5_p;                  // opt: info for CSR5
    magma_index_t      csr5_num_offsets;        // opt: info for CSR5
    magma_index_t      csr5_tail_tile_start;    // opt: info for CSR5
    magma_order_t      major;                   // opt: row/col major for dense matrices
    magma_int_t        ld;                      // opt: leading dimension for dense
} magma_z_matrix;

typedef struct magma_c_matrix
{
    magma_storage_t    storage_type;            // matrix format - CSR, ELL, SELL-P, CSR5
    magma_location_t   memory_location;         // CPU or DEV
    magma_symmetry_t   sym;                     // opt: indicate symmetry
    magma_diagorder_t  diagorder_type;          // opt: only needed for factorization matrices
    magma_uplo_t   fill_mode;               // fill mode full/lower/upper
    magma_int_t        num_rows;                // number of rows
    magma_int_t        num_cols;                // number of columns
    magma_int_t        nnz;                     // opt: number of nonzeros
    magma_int_t        max_nnz_row;             // opt: max number of nonzeros in one row
    magma_int_t        diameter;                // opt: max distance of entry from main diagonal
    magma_int_t        true_nnz;              // opt: true nnz
    magma_bool_t       ownership;               // does MAGMA own the arrays of this matrix structure
    union {
        magmaFloatComplex       *val;           // array containing values in CPU case
        magmaFloatComplex_ptr   dval;           // array containing values in DEV case
    };
    union {
        magmaFloatComplex       *diag;          // opt: diagonal entries in CPU case
        magmaFloatComplex_ptr   ddiag;          // opt: diagonal entries in DEV case
    };
    union {
        magma_index_t           *row;           // row pointer CPU case
        magmaIndex_ptr          drow;           // row pointer DEV case
    };
    union {
        magma_index_t           *rowidx;        // opt: array containing row indices CPU case
        magmaIndex_ptr          drowidx;        // opt: array containing row indices DEV case
    };
    union {
        magma_index_t           *col;           // array containing col indices CPU case
        magmaIndex_ptr          dcol;           // array containing col indices DEV case
    };
    union {
        magma_index_t           *list;          // opt: linked list pointing to next element
        magmaIndex_ptr          dlist;          // opt: linked list pointing to next element
    };
    union {
        magma_uindex_t           *tile_ptr;              // opt: CSR5 tile pointer CPU case
        magmaUIndex_ptr          dtile_ptr;              // opt: CSR5 tile pointer DEV case
    };
    union {
        magma_uindex_t           *tile_desc;             // opt: CSR5 tile descriptor CPU case
        magmaUIndex_ptr          dtile_desc;             // opt: CSR5 tile descriptor DEV case
    };
    union {
        magma_index_t            *tile_desc_offset_ptr;  // opt: CSR5 tile descriptor offset pointer CPU case
        magmaIndex_ptr           dtile_desc_offset_ptr;  // opt: CSR5 tile descriptor offset pointer DEV case
    };
    union {
        magma_index_t            *tile_desc_offset;      // opt: CSR5 tile descriptor offset CPU case
        magmaIndex_ptr           dtile_desc_offset;      // opt: CSR5 tile descriptor offset DEV case
    };
    union {
        magmaFloatComplex        *calibrator;            // opt: CSR5 calibrator CPU case
        magmaFloatComplex_ptr    dcalibrator;            // opt: CSR5 calibrator DEV case
    };
    magma_index_t      *blockinfo;              // opt: for BCSR format CPU case
    magma_int_t        blocksize;               // opt: info for SELL-P/BCSR
    magma_int_t        numblocks;               // opt: info for SELL-P/BCSR
    magma_int_t        alignment;               // opt: info for SELL-P/BCSR
    magma_int_t        csr5_sigma;              // opt: info for CSR5
    magma_int_t        csr5_bit_y_offset;       // opt: info for CSR5
    magma_int_t        csr5_bit_scansum_offset; // opt: info for CSR5
    magma_int_t        csr5_num_packets;        // opt: info for CSR5
    magma_index_t      csr5_p;                  // opt: info for CSR5
    magma_index_t      csr5_num_offsets;        // opt: info for CSR5
    magma_index_t      csr5_tail_tile_start;    // opt: info for CSR5
    magma_order_t      major;                   // opt: row/col major for dense matrices
    magma_int_t        ld;                      // opt: leading dimension for dense
} magma_c_matrix;


typedef struct magma_d_matrix
{
    magma_storage_t    storage_type;            // matrix format - CSR, ELL, SELL-P, CSR5
    magma_location_t   memory_location;         // CPU or DEV
    magma_symmetry_t   sym;                     // opt: indicate symmetry
    magma_diagorder_t  diagorder_type;          // opt: only needed for factorization matrices
    magma_uplo_t   fill_mode;               // fill mode full/lower/upper
    magma_int_t        num_rows;                // number of rows
    magma_int_t        num_cols;                // number of columns
    magma_int_t        nnz;                     // opt: number of nonzeros
    magma_int_t        max_nnz_row;             // opt: max number of nonzeros in one row
    magma_int_t        diameter;                // opt: max distance of entry from main diagonal
    magma_int_t        true_nnz;              // opt: true nnz
    magma_bool_t       ownership;               // does MAGMA own the arrays of this matrix structure
    union {
        double                  *val;           // array containing values in CPU case
        magmaDouble_ptr         dval;           // array containing values in DEV case
    };
    union {
        double                  *diag;          // opt: diagonal entries in CPU case
        magmaDouble_ptr         ddiag;          // opt: diagonal entries in DEV case
    };
    union {
        magma_index_t           *row;           // row pointer CPU case
        magmaIndex_ptr          drow;           // row pointer DEV case
    };
    union {
        magma_index_t           *rowidx;        // opt: array containing row indices CPU case
        magmaIndex_ptr          drowidx;        // opt: array containing row indices DEV case
    };
    union {
        magma_index_t           *col;           // array containing col indices CPU case
        magmaIndex_ptr          dcol;           // array containing col indices DEV case
    };
    union {
        magma_index_t           *list;          // opt: linked list pointing to next element
        magmaIndex_ptr          dlist;          // opt: linked list pointing to next element
    };
    union {
        magma_uindex_t           *tile_ptr;              // opt: CSR5 tile pointer CPU case
        magmaUIndex_ptr          dtile_ptr;              // opt: CSR5 tile pointer DEV case
    };
    union {
        magma_uindex_t           *tile_desc;             // opt: CSR5 tile descriptor CPU case
        magmaUIndex_ptr          dtile_desc;             // opt: CSR5 tile descriptor DEV case
    };
    union {
        magma_index_t            *tile_desc_offset_ptr;  // opt: CSR5 tile descriptor offset pointer CPU case
        magmaIndex_ptr           dtile_desc_offset_ptr;  // opt: CSR5 tile descriptor offset pointer DEV case
    };
    union {
        magma_index_t            *tile_desc_offset;      // opt: CSR5 tile descriptor offset CPU case
        magmaIndex_ptr           dtile_desc_offset;      // opt: CSR5 tile descriptor offset DEV case
    };
    union {
        double                   *calibrator;            // opt: CSR5 calibrator CPU case
        magmaDouble_ptr          dcalibrator;            // opt: CSR5 calibrator DEV case
    };
    magma_index_t      *blockinfo;              // opt: for BCSR format CPU case
    magma_int_t        blocksize;               // opt: info for SELL-P/BCSR
    magma_int_t        numblocks;               // opt: info for SELL-P/BCSR
    magma_int_t        alignment;               // opt: info for SELL-P/BCSR
    magma_int_t        csr5_sigma;              // opt: info for CSR5
    magma_int_t        csr5_bit_y_offset;       // opt: info for CSR5
    magma_int_t        csr5_bit_scansum_offset; // opt: info for CSR5
    magma_int_t        csr5_num_packets;        // opt: info for CSR5
    magma_index_t      csr5_p;                  // opt: info for CSR5
    magma_index_t      csr5_num_offsets;        // opt: info for CSR5
    magma_index_t      csr5_tail_tile_start;    // opt: info for CSR5
    magma_order_t      major;                   // opt: row/col major for dense matrices
    magma_int_t        ld;                      // opt: leading dimension for dense
} magma_d_matrix;


typedef struct magma_s_matrix
{
    magma_storage_t    storage_type;            // matrix format - CSR, ELL, SELL-P, CSR5
    magma_location_t   memory_location;         // CPU or DEV
    magma_symmetry_t   sym;                     // opt: indicate symmetry
    magma_diagorder_t  diagorder_type;          // opt: only needed for factorization matrices
    magma_uplo_t   fill_mode;               // fill mode full/lower/upper
    magma_int_t        num_rows;                // number of rows
    magma_int_t        num_cols;                // number of columns
    magma_int_t        nnz;                     // opt: number of nonzeros
    magma_int_t        max_nnz_row;             // opt: max number of nonzeros in one row
    magma_int_t        diameter;                // opt: max distance of entry from main diagonal
    magma_int_t        true_nnz;              // opt: true nnz
    magma_bool_t       ownership;               // does MAGMA own the arrays of this matrix structure
    union {
        float                   *val;           // array containing values in CPU case
        magmaFloat_ptr          dval;           // array containing values in DEV case
    };
    union {
        float                   *diag;          // opt: diagonal entries in CPU case
        magmaFloat_ptr          ddiag;          // opt: diagonal entries in DEV case
    };
    union {
        magma_index_t           *row;           // opt: row pointer CPU case
        magmaIndex_ptr          drow;           // opt: row pointer DEV case
    };
    union {
        magma_index_t           *rowidx;        // opt: array containing row indices CPU case
        magmaIndex_ptr          drowidx;        // opt: array containing row indices DEV case
    };
    union {
        magma_index_t           *col;           // opt: array containing col indices CPU case
        magmaIndex_ptr          dcol;           // opt: array containing col indices DEV case
    };
    union {
        magma_index_t           *list;          // opt: linked list pointing to next element
        magmaIndex_ptr          dlist;          // opt: linked list pointing to next element
    };
    union {
        magma_uindex_t           *tile_ptr;              // opt: CSR5 tile pointer CPU case
        magmaUIndex_ptr          dtile_ptr;              // opt: CSR5 tile pointer DEV case
    };
    union {
        magma_uindex_t           *tile_desc;             // opt: CSR5 tile descriptor CPU case
        magmaUIndex_ptr          dtile_desc;             // opt: CSR5 tile descriptor DEV case
    };
    union {
        magma_index_t            *tile_desc_offset_ptr;  // opt: CSR5 tile descriptor offset pointer CPU case
        magmaIndex_ptr           dtile_desc_offset_ptr;  // opt: CSR5 tile descriptor offset pointer DEV case
    };
    union {
        magma_index_t            *tile_desc_offset;      // opt: CSR5 tile descriptor offset CPU case
        magmaIndex_ptr           dtile_desc_offset;      // opt: CSR5 tile descriptor offset DEV case
    };
    union {
        float                    *calibrator;            // opt: CSR5 calibrator CPU case
        magmaFloat_ptr           dcalibrator;            // opt: CSR5 calibrator DEV case
    };
    magma_index_t      *blockinfo;              // opt: for BCSR format CPU case
    magma_int_t        blocksize;               // opt: info for SELL-P/BCSR
    magma_int_t        numblocks;               // opt: info for SELL-P/BCSR
    magma_int_t        alignment;               // opt: info for SELL-P/BCSR
    magma_int_t        csr5_sigma;              // opt: info for CSR5
    magma_int_t        csr5_bit_y_offset;       // opt: info for CSR5
    magma_int_t        csr5_bit_scansum_offset; // opt: info for CSR5
    magma_int_t        csr5_num_packets;        // opt: info for CSR5
    magma_index_t      csr5_p;                  // opt: info for CSR5
    magma_index_t      csr5_num_offsets;        // opt: info for CSR5
    magma_index_t      csr5_tail_tile_start;    // opt: info for CSR5
    magma_order_t      major;                   // opt: row/col major for dense matrices
    magma_int_t        ld;                      // opt: leading dimension for dense
} magma_s_matrix;


// for backwards compatability, make these aliases.
typedef magma_s_matrix magma_s_sparse_matrix;
typedef magma_d_matrix magma_d_sparse_matrix;
typedef magma_c_matrix magma_c_sparse_matrix;
typedef magma_z_matrix magma_z_sparse_matrix;

typedef magma_s_matrix magma_s_vector;
typedef magma_d_matrix magma_d_vector;
typedef magma_c_matrix magma_c_vector;
typedef magma_z_matrix magma_z_vector;

/*
typedef struct magma_z_vector
{
    magma_location_t   memory_location;         // CPU or DEV
    magma_int_t        num_rows;                // number of rows
    magma_int_t        num_cols;                // number of columns (in case of a block of vectors)
    magma_int_t        nnz;                     // opt: number of nonzeros
    union {
        magmaDoubleComplex      *val;           // array containing values in CPU case
        magmaDoubleComplex_ptr  dval;           // array containing values in DEV case
    };
    magma_order_t      major;                   // storage type:Row/Column-Major
} magma_z_vector;

typedef struct magma_c_vector
{
    magma_location_t   memory_location;         // CPU or DEV
    magma_int_t        num_rows;                // number of rows
    magma_int_t        num_cols;                // number of columns (in case of a block of vectors)
    magma_int_t        nnz;                     // opt: number of nonzeros
    union {
        magmaFloatComplex       *val;           // array containing values in CPU case
        magmaFloatComplex_ptr   dval;           // array containing values in DEV case
    };
    magma_order_t      major;                   // storage type:Row/Column-Major
} magma_c_vector;


typedef struct magma_d_vector
{
    magma_location_t   memory_location;         // CPU or DEV
    magma_int_t        num_rows;                // number of rows
    magma_int_t        num_cols;                // number of columns (in case of a block of vectors)
    magma_int_t        nnz;                     // opt: number of nonzeros
    union {
        double                  *val;           // array containing values in CPU case
        magmaDouble_ptr         dval;           // array containing values in DEV case
    };
    magma_order_t      major;                   // storage type:Row/Column-Major
} magma_d_vector;


typedef struct magma_s_vector
{
    magma_location_t   memory_location;         // CPU or DEV
    magma_int_t        num_rows;                // number of rows
    magma_int_t        num_cols;                // number of columns (in case of a block of vectors)
    magma_int_t        nnz;                     // opt: number of nonzeros
    union {
        float                   *val;           // array containing values in CPU case
        magmaFloat_ptr          dval;           // array containing values in DEV case
    };
    magma_order_t      major;                   // storage type:Row/Column-Major
} magma_s_vector;
*/


//*****************     solver parameters     ********************************//

typedef struct magma_z_solver_par
{
    magma_solver_type  solver;                  // solver type
    magma_int_t        version;                 // sometimes there are different versions
    double             atol;                    // absolute residual stopping criterion
    double             rtol;                    // relative residual stopping criterion
    magma_int_t        maxiter;                 // upper iteration limit
    magma_int_t        restart;                 // for GMRES
    magma_ortho_t      ortho;                   // for GMRES
    magma_int_t        numiter;                 // feedback: number of needed iterations
    magma_int_t        spmv_count;              // feedback: number of needed SpMV - can be different to iteration count
    double             init_res;                // feedback: initial residual
    double             final_res;               // feedback: final residual
    double             iter_res;                // feedback: iteratively computed residual
    real_Double_t      runtime;                 // feedback: runtime needed
    real_Double_t      *res_vec;                // feedback: array containing residuals
    real_Double_t      *timing;                 // feedback: detailed timing
    magma_int_t        verbose;                 // print residual every 'verbose' iterations
    magma_int_t        num_eigenvalues;         // number of EV for eigensolvers
    magma_int_t        ev_length;               // needed for framework
    double             *eigenvalues;            // feedback: array containing eigenvalues
    magmaDoubleComplex_ptr      eigenvectors;   // feedback: array containing eigenvectors on DEV
    magma_int_t        info;                    // feedback: did the solver converge etc.

    //---------------------------------
    // the input for verbose is:
    // 0 = production mode
    // k>0 = convergence and timing is monitored in *res_vec and *timeing every  
    // k-th iteration 
    //
    // the output of info is:
    //  0 = convergence (stopping criterion met)
    // -1 = no convergence
    // -2 = convergence but stopping criterion not met within maxiter
    //--------------------------------
} magma_z_solver_par;



typedef struct magma_c_solver_par
{
    magma_solver_type  solver;                  // solver type
    magma_int_t        version;                 // sometimes there are different versions
    float              atol;                     // absolute residual stopping criterion
    float              rtol;                    // relative residual stopping criterion
    magma_int_t        maxiter;                 // upper iteration limit
    magma_int_t        restart;                 // for GMRES
    magma_ortho_t      ortho;                   // for GMRES
    magma_int_t        numiter;                 // feedback: number of needed iterations
    magma_int_t        spmv_count;              // feedback: number of needed SpMV - can be different to iteration count
    float              init_res;                // feedback: initial residual
    float              final_res;               // feedback: final residual
    float              iter_res;                // feedback: iteratively computed residual
    real_Double_t      runtime;                 // feedback: runtime needed
    real_Double_t      *res_vec;                // feedback: array containing residuals
    real_Double_t      *timing;                 // feedback: detailed timing
    magma_int_t        verbose;                 // print residual ever 'verbose' iterations
    magma_int_t        num_eigenvalues;         // number of EV for eigensolvers
    magma_int_t        ev_length;               // needed for framework
    float              *eigenvalues;            // feedback: array containing eigenvalues
    magmaFloatComplex_ptr       eigenvectors;   // feedback: array containing eigenvectors on DEV
    magma_int_t        info;                    // feedback: did the solver converge etc.

    //---------------------------------
    // the input for verbose is:
    // 0 = production mode
    // k>0 = convergence and timing is monitored in *res_vec and *timeing every  
    // k-th iteration 
    //
    // the output of info is:
    //  0 = convergence (stopping criterion met)
    // -1 = no convergence
    // -2 = convergence but stopping criterion not met within maxiter
    //--------------------------------
} magma_c_solver_par;



typedef struct magma_d_solver_par
{
    magma_solver_type  solver;                  // solver type
    magma_int_t        version;                 // sometimes there are different versions
    double             atol;                     // absolute residual stopping criterion
    double             rtol;                    // relative residual stopping criterion
    magma_int_t        maxiter;                 // upper iteration limit
    magma_int_t        restart;                 // for GMRES
    magma_ortho_t      ortho;                   // for GMRES
    magma_int_t        numiter;                 // feedback: number of needed iterations
    magma_int_t        spmv_count;              // feedback: number of needed SpMV - can be different to iteration count
    double             init_res;                // feedback: initial residual
    double             final_res;               // feedback: final residual
    double             iter_res;                // feedback: iteratively computed residual
    real_Double_t      runtime;                 // feedback: runtime needed
    real_Double_t      *res_vec;                // feedback: array containing residuals
    real_Double_t      *timing;                 // feedback: detailed timing
    magma_int_t        verbose;                 // print residual ever 'verbose' iterations
    magma_int_t        num_eigenvalues;         // number of EV for eigensolvers
    magma_int_t        ev_length;               // needed for framework
    double             *eigenvalues;            // feedback: array containing eigenvalues
    magmaDouble_ptr             eigenvectors;   // feedback: array containing eigenvectors on DEV
    magma_int_t        info;                    // feedback: did the solver converge etc.

    //---------------------------------
    // the input for verbose is:
    // 0 = production mode
    // k>0 = convergence and timing is monitored in *res_vec and *timeing every  
    // k-th iteration 
    //
    // the output of info is:
    //  0 = convergence (stopping criterion met)
    // -1 = no convergence
    // -2 = convergence but stopping criterion not met within maxiter
    //--------------------------------
} magma_d_solver_par;



typedef struct magma_s_solver_par
{
    magma_solver_type  solver;                  // solver type
    magma_int_t        version;                 // sometimes there are different versions
    float              atol;                     // absolute residual stopping criterion
    float              rtol;                    // relative residual stopping criterion
    magma_int_t        maxiter;                 // upper iteration limit
    magma_int_t        restart;                 // for GMRES
    magma_ortho_t      ortho;                   // for GMRES
    magma_int_t        numiter;                 // feedback: number of needed iterations
    magma_int_t        spmv_count;              // feedback: number of needed SpMV - can be different to iteration count
    float              init_res;                // feedback: initial residual
    float              final_res;               // feedback: final residual
    float              iter_res;                // feedback: iteratively computed residual
    real_Double_t      runtime;                 // feedback: runtime needed
    real_Double_t      *res_vec;                // feedback: array containing residuals
    real_Double_t      *timing;                 // feedback: detailed timing
    magma_int_t        verbose;                 // print residual ever 'verbose' iterations
    magma_int_t        num_eigenvalues;         // number of EV for eigensolvers
    magma_int_t        ev_length;               // needed for framework
    float              *eigenvalues;            // feedback: array containing eigenvalues
    magmaFloat_ptr              eigenvectors;   // feedback: array containing eigenvectors on DEV
    magma_int_t        info;                    // feedback: did the solver converge etc.

    //---------------------------------
    // the input for verbose is:
    // 0 = production mode
    // k>0 = convergence and timing is monitored in *res_vec and *timeing every  
    // k-th iteration 
    //
    // the output of info is:
    //       0          Success.
    //      -117        Not supported.
    //      -201        No convergence within iteration limit. 
    //      -202        No convergence.
    //      -203        Operator A is not positive definite.
    //--------------------------------
} magma_s_solver_par;



//************            preconditioner parameters       ********************//

typedef struct magma_z_preconditioner
{
    magma_solver_type       solver;
    magma_solver_type       trisolver;
    magma_int_t             levels;
    magma_int_t             sweeps;
    magma_int_t             pattern;
    magma_int_t             bsize;
    magma_int_t             offset;
    magma_precision         format;
    double                  atol;                
    double                  rtol;    
    magma_int_t             maxiter;
    magma_int_t             restart; 
    magma_int_t             numiter;
    magma_int_t             spmv_count;  
    double                  init_res;
    double                  final_res;
    real_Double_t      runtime;                 // feedback: preconditioner runtime needed
    real_Double_t      setuptime;               // feedback: preconditioner setup time needed
    magma_z_matrix   M;
    magma_z_matrix   L;
    magma_z_matrix   LT;
    magma_z_matrix   U;
    magma_z_matrix   UT;
    magma_z_matrix   LD;
    magma_z_matrix   UD;
    magma_z_matrix   LDT;
    magma_z_matrix   UDT;
    magma_index_t *Lnz;
    magma_index_t *Unz;
    magma_index_t **Lja;
    magma_index_t **Uja;
    magmaDoubleComplex **Lma;
    magmaDoubleComplex **Uma;
    magma_z_matrix          d;
    magma_z_matrix          d2;
    magma_z_matrix          work1;
    magma_z_matrix          work2;
    magma_int_t*            int_array_1;
    magma_int_t*            int_array_2;
    magma_index_t*            L_dgraphindegree;     // for sync-free trisolve
    magma_index_t*            L_dgraphindegree_bak; // for sync-free trisolve
    magma_index_t*            U_dgraphindegree;     // for sync-free trisolve
    magma_index_t*            U_dgraphindegree_bak; // for sync-free trisolve
    cusparseSolveAnalysisInfo_t cuinfo;
    cusparseSolveAnalysisInfo_t cuinfoL;
    cusparseSolveAnalysisInfo_t cuinfoLT;
    cusparseSolveAnalysisInfo_t cuinfoU;
    cusparseSolveAnalysisInfo_t cuinfoUT;
    magma_bool_t            transpose;                 // need the transpose for the solver?
#if defined(HAVE_PASTIX)
    pastix_data_t*          pastix_data;
    magma_int_t*            iparm;
    double*                 dparm;
#endif
} magma_z_preconditioner;

typedef struct magma_c_preconditioner
{
    magma_solver_type       solver;
    magma_solver_type       trisolver;
    magma_int_t             levels;
    magma_int_t             sweeps;
    magma_int_t             pattern;
    magma_int_t             bsize;
    magma_int_t             offset;
    magma_precision         format;
    float                   atol;                
    float                   rtol;               
    magma_int_t             maxiter;
    magma_int_t             restart; 
    magma_int_t             numiter;
    magma_int_t             spmv_count;
    float                   init_res;
    float                   final_res;
    real_Double_t      runtime;                // feedback: preconditioner runtime needed
    real_Double_t      setuptime;           // feedback: preconditioner setup time needed
    magma_c_matrix   M;
    magma_c_matrix   L;
    magma_c_matrix   LT;
    magma_c_matrix   U;
    magma_c_matrix   UT;
    magma_c_matrix   LD;
    magma_c_matrix   UD;
    magma_c_matrix   LDT;
    magma_c_matrix   UDT;
    magma_index_t *Lnz;
    magma_index_t *Unz;
    magma_index_t **Lja;
    magma_index_t **Uja;
    magmaFloatComplex **Lma;
    magmaFloatComplex **Uma;
    magma_c_matrix          d;
    magma_c_matrix          d2;
    magma_c_matrix          work1;
    magma_c_matrix          work2;
    magma_int_t*            int_array_1;
    magma_int_t*            int_array_2;
    magma_index_t*            L_dgraphindegree;     // for sync-free trisolve
    magma_index_t*            L_dgraphindegree_bak; // for sync-free trisolve
    magma_index_t*            U_dgraphindegree;     // for sync-free trisolve
    magma_index_t*            U_dgraphindegree_bak; // for sync-free trisolve
    cusparseSolveAnalysisInfo_t cuinfo;
    cusparseSolveAnalysisInfo_t cuinfoL;
    cusparseSolveAnalysisInfo_t cuinfoLT;
    cusparseSolveAnalysisInfo_t cuinfoU;
    cusparseSolveAnalysisInfo_t cuinfoUT;
    magma_bool_t            transpose;                 // need the transpose for the solver?
#if defined(HAVE_PASTIX)
    pastix_data_t*          pastix_data;
    magma_int_t*            iparm;
    float*                  dparm;
#endif
} magma_c_preconditioner;


typedef struct magma_d_preconditioner
{
    magma_solver_type       solver;
    magma_solver_type       trisolver;
    magma_int_t             levels;
    magma_int_t             sweeps;
    magma_int_t             pattern;
    magma_int_t             bsize;
    magma_int_t             offset;
    magma_precision         format;
    double                  atol;                
    double                  rtol; 
    magma_int_t             maxiter;
    magma_int_t             restart; 
    magma_int_t             numiter;
    magma_int_t             spmv_count;
    double                  init_res;
    double                  final_res;
    real_Double_t      runtime;                // feedback: preconditioner runtime needed
    real_Double_t      setuptime;           // feedback: preconditioner setup time needed
    magma_d_matrix   M;
    magma_d_matrix   L;
    magma_d_matrix   LT;
    magma_d_matrix   U;
    magma_d_matrix   UT;
    magma_d_matrix   LD;
    magma_d_matrix   UD;
    magma_d_matrix   LDT;
    magma_d_matrix   UDT;
    magma_index_t *Lnz;
    magma_index_t *Unz;
    magma_index_t **Lja;
    magma_index_t **Uja;
    double **Lma;
    double **Uma;
    magma_d_matrix          d;
    magma_d_matrix          d2;
    magma_d_matrix          work1;
    magma_d_matrix          work2;
    magma_int_t*            int_array_1;
    magma_int_t*            int_array_2;
    magma_index_t*            L_dgraphindegree;     // for sync-free trisolve
    magma_index_t*            L_dgraphindegree_bak; // for sync-free trisolve
    magma_index_t*            U_dgraphindegree;     // for sync-free trisolve
    magma_index_t*            U_dgraphindegree_bak; // for sync-free trisolve
    cusparseSolveAnalysisInfo_t cuinfo;
    cusparseSolveAnalysisInfo_t cuinfoL;
    cusparseSolveAnalysisInfo_t cuinfoLT;
    cusparseSolveAnalysisInfo_t cuinfoU;
    cusparseSolveAnalysisInfo_t cuinfoUT;
    magma_bool_t            transpose;                 // need the transpose for the solver?
#if defined(HAVE_PASTIX)
    pastix_data_t*          pastix_data;
    magma_int_t*            iparm;
    double*                 dparm;
#endif
} magma_d_preconditioner;


typedef struct magma_s_preconditioner
{
    magma_solver_type       solver;
    magma_solver_type       trisolver;
    magma_int_t             levels;
    magma_int_t             sweeps;
    magma_int_t             pattern;
    magma_int_t             bsize;
    magma_int_t             offset;
    magma_precision         format;
    float                   atol;                
    float                   rtol;    
    magma_int_t             maxiter;
    magma_int_t             restart; 
    magma_int_t             numiter;
    magma_int_t             spmv_count;
    float                   init_res;
    float                   final_res;
    real_Double_t      runtime;                // feedback: preconditioner runtime needed
    real_Double_t      setuptime;           // feedback: preconditioner setup time needed
    magma_s_matrix   M;
    magma_s_matrix   L;
    magma_s_matrix   LT;
    magma_s_matrix   U;
    magma_s_matrix   UT;
    magma_s_matrix   LD;
    magma_s_matrix   UD;
    magma_s_matrix   LDT;
    magma_s_matrix   UDT;
    magma_index_t *Lnz;
    magma_index_t *Unz;
    magma_index_t **Lja;
    magma_index_t **Uja;
    float **Lma;
    float **Uma;
    magma_s_matrix          d;
    magma_s_matrix          d2;
    magma_s_matrix          work1;
    magma_s_matrix          work2;
    magma_int_t*            int_array_1;
    magma_int_t*            int_array_2;
    magma_index_t*            L_dgraphindegree;     // for sync-free trisolve
    magma_index_t*            L_dgraphindegree_bak; // for sync-free trisolve
    magma_index_t*            U_dgraphindegree;     // for sync-free trisolve
    magma_index_t*            U_dgraphindegree_bak; // for sync-free trisolve
    cusparseSolveAnalysisInfo_t cuinfo;
    cusparseSolveAnalysisInfo_t cuinfoL;
    cusparseSolveAnalysisInfo_t cuinfoLT;
    cusparseSolveAnalysisInfo_t cuinfoU;
    cusparseSolveAnalysisInfo_t cuinfoUT;
    magma_bool_t            transpose;                 // need the transpose for the solver?
#if defined(HAVE_PASTIX)
    pastix_data_t*          pastix_data;
    magma_int_t*            iparm;
    float*                  dparm;
#endif
} magma_s_preconditioner;


//##############################################################################
//
//              opts for the testers
//
//##############################################################################

typedef struct magma_zopts
{
    magma_operation_t       operation;
    magma_location_t        compute_location;
    magma_z_solver_par      solver_par;
    magma_z_preconditioner  precond_par;
    magma_storage_t         input_format;
    magma_trans_t           trans;
    magma_int_t             blocksize;
    magma_int_t             alignment;
    magma_storage_t         output_format;
    magma_location_t        input_location;
    magma_location_t        output_location;
    magma_scale_t           scaling;
} magma_zopts;

typedef struct magma_copts
{
    magma_operation_t       operation;
    magma_location_t        compute_location;
    magma_c_solver_par      solver_par;
    magma_c_preconditioner  precond_par;
    magma_storage_t         input_format;
    magma_trans_t           trans;
    magma_int_t             blocksize;
    magma_int_t             alignment;
    magma_storage_t         output_format;
    magma_location_t        input_location;
    magma_location_t        output_location;
    magma_scale_t           scaling;
} magma_copts;

typedef struct magma_dopts
{
    magma_operation_t       operation;
    magma_location_t        compute_location;
    magma_d_solver_par      solver_par;
    magma_d_preconditioner  precond_par;
    magma_storage_t         input_format;
    magma_trans_t           trans;
    magma_int_t             blocksize;
    magma_int_t             alignment;
    magma_storage_t         output_format;
    magma_location_t        input_location;
    magma_location_t        output_location;
    magma_scale_t           scaling;
} magma_dopts;

typedef struct magma_sopts
{
    magma_operation_t       operation;
    magma_location_t        compute_location;
    magma_s_solver_par      solver_par;
    magma_s_preconditioner  precond_par;
    magma_storage_t         input_format;
    magma_trans_t           trans;
    magma_int_t             blocksize;
    magma_int_t             alignment;
    magma_storage_t         output_format;
    magma_location_t        input_location;
    magma_location_t        output_location;
    magma_scale_t           scaling;
} magma_sopts;

#ifdef __cplusplus
}
#endif

#endif        //  #ifndef MAGMASPARSE_TYPES_H
