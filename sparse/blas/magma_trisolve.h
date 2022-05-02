/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tobias Ribizel
*/
#include "magmasparse_internal.h"


#ifndef MAGMA_TRISOLVE_H
#define MAGMA_TRISOLVE_H

/**
    Purpose
    -------

    Frees the analysis data for a triangular solve.

    Arguments
    ---------

    @param[in,out]
    solve_info  magma_solve_info_t*
                analysis data to free.

    ********************************************************************/
void magma_trisolve_free(magma_solve_info_t *solve_info);

/**
    Purpose
    -------

    Performs a triangular solve analysis for the given system matrix.
    Abstracts away interface for cuSPARSE/hipSPARSE.

    Arguments
    ---------

    @param[in]
    M           magma_z_matrix    
                triangular system matrix

    @param[in]
    solve_info  magma_solve_info_t
                analysis data produced by trisolve_analysis.

    @param[in]
    upper_triangular bool
                true if the system matrix is upper triangular,
                false if it is lower triangular.

    @param[in]
    unit_diagonal bool
                true if the system matrix is assumed to have a unit diagonal,
                false otherwise.

    @param[in]
    transpose   bool
                true if the system matrix should be transposed for the solve.
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    ********************************************************************/
magma_int_t magma_ztrisolve_analysis(
    magma_z_matrix M, 
    magma_solve_info_t *solve_info,
    bool upper_triangular,
    bool unit_diagonal,
    bool transpose,
    magma_queue_t queue);

/**
    Purpose
    -------

    Performs a triangular solve with the given solve info.
    Abstracts away interface for cuSPARSE/hipSPARSE.

    Arguments
    ---------

    @param[in]
    M           magma_z_matrix    
                triangular system matrix

    @param[in]
    solve_info  magma_solve_info_t
                analysis data produced by trisolve_analysis.

    @param[in]
    upper_triangular bool
                true if the system matrix is upper triangular,
                false if it is lower triangular.

    @param[in]
    unit_diagonal bool
                true if the system matrix is assumed to have a unit diagonal,
                false otherwise.

    @param[in]
    transpose   bool
                true if the system matrix should be transposed for the solve.

    @param[in]
    b           magma_z_matrix    
                input matrix for the solve

    @param[in]
    x           magma_z_matrix    
                output matrix for the solve
                
    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    ********************************************************************/
magma_int_t magma_ztrisolve(
    magma_z_matrix M,
    magma_solve_info_t solve_info,
    bool upper_triangular,
    bool unit_diagonal,
    bool transpose,
    magma_z_matrix b,
    magma_z_matrix x,
    magma_queue_t queue);

magma_int_t magma_ctrisolve_analysis(
    magma_c_matrix M, 
    magma_solve_info_t *solve_info,
    bool upper_triangular,
    bool unit_diagonal,
    bool transpose,
    magma_queue_t queue);

magma_int_t magma_ctrisolve(
    magma_c_matrix M,
    magma_solve_info_t solve_info,
    bool upper_triangular,
    bool unit_diagonal,
    bool transpose,
    magma_c_matrix b,
    magma_c_matrix x,
    magma_queue_t queue);


magma_int_t magma_dtrisolve_analysis(
    magma_d_matrix M, 
    magma_solve_info_t *solve_info,
    bool upper_triangular,
    bool unit_diagonal,
    bool transpose,
    magma_queue_t queue);

magma_int_t magma_dtrisolve(
    magma_d_matrix M,
    magma_solve_info_t solve_info,
    bool upper_triangular,
    bool unit_diagonal,
    bool transpose,
    magma_d_matrix b,
    magma_d_matrix x,
    magma_queue_t queue);
    
magma_int_t magma_strisolve_analysis(
    magma_s_matrix M, 
    magma_solve_info_t *solve_info,
    bool upper_triangular,
    bool unit_diagonal,
    bool transpose,
    magma_queue_t queue);

magma_int_t magma_strisolve(
    magma_s_matrix M,
    magma_solve_info_t solve_info,
    bool upper_triangular,
    bool unit_diagonal,
    bool transpose,
    magma_s_matrix b,
    magma_s_matrix x,
    magma_queue_t queue);


#endif