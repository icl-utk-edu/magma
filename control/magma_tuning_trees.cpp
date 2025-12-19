/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Natalie Beams

*/

#include "magma_tuning_trees.h"
#include "gemm_batched_decision_tree.h"

/***************************************************************************
     Given m, n, and k (dimensions for a GEMM operation), evaluate the
     tuning tree defined by the features, thresholds, left_children and
     right_children arrays. These arrays are nearly identical to those
     used to define decision trees in scikit-learn, with one exception: in
     the thresholds array, if a node is a leaf node, then the value is
     the configuration number associated with that leaf. In the features,
     left_children, and right_children arrays, a negative value indicates
     that the node is a leaf node.

     The tree is traversed as follows:
       For current_node (not a leaf node), the value of
       features[current_node] tells us whether the next decision is based
       on m, n, or k. The relevant dimension is checked against the value
       in thresholds[current_node]. If it is less than or equal to the
       threshold, then update current_node to left_children[current_node];
       otherwise, update current_node to right_children[current_node].
       The process continues until we reach a leaf, where the configuration
       number is read from thresholds[leaf].

    Arguments
    ---------
    @param[in]
    m                INTEGER.
                     The number of rows of the A matrix

    @param[in]
    n                INTEGER.
                     The number of columns of the A matrix

    @param[in]
    k                INTEGER.
                     The number of columns of the B matrix

    @param[in]
    features         std::vector<magma_int_t>*.
                     The features array defining which parameter (m = 0,
                     n = 1, k = 2) is the basis for the decision at a given
                     node in the tree

    @param[in]
    thresholds       std::vector<magma_int_t>*.
                     The array defining the threshold for the decision at a
                     given node, or the value of the selected configuration 
                     for a leaf node

    @param[in]
    left_children    std::vector<magma_int_t>*.
                     The array defining the left children of a given (non-
                     leaf) node

    @param[in]
    right_children   std::vector<magma_int_t>*.
                     The array defining the right children of a given (non-
                     leaf) node

    @param[in]
    max_depth        INTEGER.
                     The maximum depth of the tree (used to prevent an
                     infinite loop in the case of an error during tree
                     traversal). The depth is the maximum number of 
                     decisions required to reach a leaf for all leaves.
******************************************************************************/
magma_int_t
evaluate_gemm_tree(magma_int_t m, magma_int_t n, magma_int_t k,
                               std::vector<magma_int_t>* features,
	                       std::vector<magma_int_t>* thresholds,
                               std::vector<magma_int_t>* left_children,
                               std::vector<magma_int_t>* right_children,
                               magma_int_t max_depth)
{
    magma_int_t num_decisions_made = 0;
    magma_int_t config = -1;
    magma_int_t current_node = 0;
    magma_int_t current_feature = -1;
    magma_int_t feature_array[3] = { m, n, k };

    while ((config < 0) && (num_decisions_made <= max_depth))
    {
        current_feature = (*features)[current_node];
	if (current_feature < 0) // we have reached a leaf node
	{
	   config = (*thresholds)[current_node];
	}
	else // move to child node based on threshold
	{
            if (feature_array[current_feature] <= (*thresholds)[current_node])
            {
                current_node = (*left_children)[current_node];
	    }
	    else
            {
                current_node = (*right_children)[current_node];
	    }
        }
        num_decisions_made++;
    }
    if (config < 0)
        printf("Error processing GEMM tuning tree! Invalid configuration chosen\n");

    return config;
}

/***************************************************************************
    Return the preferred configuration (template parameters for the kernel)
    for this batched ZGEMM operation, by evaluating the tuning tree.

    Arguments
    ---------
    @param[in]
    transA  magma_trans_t.
            transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( A ) = A.
      -     = MagmaTrans:      op( A ) = A**T.
      -     = MagmaConjTrans:  op( A ) = A**H.
    
    @param[in]
    transB  magma_trans_t.
            transB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( B ) = B.
      -     = MagmaTrans:      op( B ) = B**T.
      -     = MagmaConjTrans:  op( B ) = B**H.
    
    @param[in]
    m       INTEGER.
            The number of rows of the A matrix

    @param[in]
    n       INTEGER.
            The number of columns of the A matrix

    @param[in]
    k       INTEGER.
            The number of columns of the B matrix

******************************************************************************/
magma_int_t
magma_zgemm_batched_get_config(magma_trans_t transA, magma_trans_t transB,
		magma_int_t m, magma_int_t n, magma_int_t k)
{
    std::vector<magma_int_t> *features;
    std::vector<magma_int_t> *thresholds;
    std::vector<magma_int_t> *left_children;
    std::vector<magma_int_t> *right_children;
    magma_int_t *max_depth;

    if (transA == MagmaNoTrans)
    {
         if (transB == MagmaNoTrans) // NN
	 {
             features = &zgemm_batched_NN_features_pvc;
	     thresholds = &zgemm_batched_NN_thresholds_pvc;
	     left_children = &zgemm_batched_NN_children_left_pvc;
	     right_children = &zgemm_batched_NN_children_right_pvc;
	     max_depth = &zgemm_batched_NN_max_depth_pvc;
	 }
	 else // NT, NC
	 {
             features = &zgemm_batched_NT_features_pvc;
	     thresholds = &zgemm_batched_NT_thresholds_pvc;
	     left_children = &zgemm_batched_NT_children_left_pvc;
	     right_children = &zgemm_batched_NT_children_right_pvc;
	     max_depth = &zgemm_batched_NT_max_depth_pvc;
	 }
    }
    else
    {
        if (transB == MagmaNoTrans) // TN, CN
	{
             features = &zgemm_batched_TN_features_pvc;
	     thresholds = &zgemm_batched_TN_thresholds_pvc;
	     left_children = &zgemm_batched_TN_children_left_pvc;
	     right_children = &zgemm_batched_TN_children_right_pvc;
	     max_depth = &zgemm_batched_TN_max_depth_pvc;
	}
	else // TT, CT, TC, CC
	{
             features = &zgemm_batched_TT_features_pvc;
	     thresholds = &zgemm_batched_TT_thresholds_pvc;
	     left_children = &zgemm_batched_TT_children_left_pvc;
	     right_children = &zgemm_batched_TT_children_right_pvc;
	     max_depth = &zgemm_batched_TT_max_depth_pvc;
	}
    }
    magma_int_t config = evaluate_gemm_tree(m, n, k, features, thresholds,
		    left_children, right_children, *max_depth);

    return config;
}

/***************************************************************************
    Return the preferred configuration (template parameters for the kernel)
    for this batched CGEMM operation, by evaluating the tuning tree.

    Arguments
    ---------
    @param[in]
    transA  magma_trans_t.
            transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( A ) = A.
      -     = MagmaTrans:      op( A ) = A**T.
      -     = MagmaConjTrans:  op( A ) = A**H.
    
    @param[in]
    transB  magma_trans_t.
            transB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( B ) = B.
      -     = MagmaTrans:      op( B ) = B**T.
      -     = MagmaConjTrans:  op( B ) = B**H.
    
    @param[in]
    m       INTEGER.
            The number of rows of the A matrix

    @param[in]
    n       INTEGER.
            The number of columns of the A matrix

    @param[in]
    k       INTEGER.
            The number of columns of the B matrix

******************************************************************************/
magma_int_t
magma_cgemm_batched_get_config(magma_trans_t transA, magma_trans_t transB,
		magma_int_t m, magma_int_t n, magma_int_t k)
{
    std::vector<magma_int_t> *features;
    std::vector<magma_int_t> *thresholds;
    std::vector<magma_int_t> *left_children;
    std::vector<magma_int_t> *right_children;
    magma_int_t *max_depth;

    if (transA == MagmaNoTrans)
    {
         if (transB == MagmaNoTrans) // NN
	 {
             features = &cgemm_batched_NN_features_pvc;
	     thresholds = &cgemm_batched_NN_thresholds_pvc;
	     left_children = &cgemm_batched_NN_children_left_pvc;
	     right_children = &cgemm_batched_NN_children_right_pvc;
	     max_depth = &cgemm_batched_NN_max_depth_pvc;
	 }
	 else // NT, NC
	 {
             features = &cgemm_batched_NT_features_pvc;
	     thresholds = &cgemm_batched_NT_thresholds_pvc;
	     left_children = &cgemm_batched_NT_children_left_pvc;
	     right_children = &cgemm_batched_NT_children_right_pvc;
	     max_depth = &cgemm_batched_NT_max_depth_pvc;
	 }
    }
    else
    {
        if (transB == MagmaNoTrans) // TN, CN
	{
             features = &cgemm_batched_TN_features_pvc;
	     thresholds = &cgemm_batched_TN_thresholds_pvc;
	     left_children = &cgemm_batched_TN_children_left_pvc;
	     right_children = &cgemm_batched_TN_children_right_pvc;
	     max_depth = &cgemm_batched_TN_max_depth_pvc;
	}
	else // TT, CT, TC, CC
	{
             features = &cgemm_batched_TT_features_pvc;
	     thresholds = &cgemm_batched_TT_thresholds_pvc;
	     left_children = &cgemm_batched_TT_children_left_pvc;
	     right_children = &cgemm_batched_TT_children_right_pvc;
	     max_depth = &cgemm_batched_TT_max_depth_pvc;
	}
    }
    magma_int_t config = evaluate_gemm_tree(m, n, k, features, thresholds,
		    left_children, right_children, *max_depth);

    return config;
}

/***************************************************************************
    Return the preferred configuration (template parameters for the kernel)
    for this batched DGEMM operation, by evaluating the tuning tree.

    Arguments
    ---------
    @param[in]
    transA  magma_trans_t.
            transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( A ) = A.
      -     = MagmaTrans:      op( A ) = A**T.
    
    @param[in]
    transB  magma_trans_t.
            transB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( B ) = B.
      -     = MagmaTrans:      op( B ) = B**T.

    @param[in]
    m       INTEGER.
            The number of rows of the A matrix

    @param[in]
    n       INTEGER.
            The number of columns of the A matrix

    @param[in]
    k       INTEGER.
            The number of columns of the B matrix

******************************************************************************/
magma_int_t
magma_dgemm_batched_get_config(magma_trans_t transA, magma_trans_t transB,
		magma_int_t m, magma_int_t n, magma_int_t k)
{
    std::vector<magma_int_t> *features;
    std::vector<magma_int_t> *thresholds;
    std::vector<magma_int_t> *left_children;
    std::vector<magma_int_t> *right_children;
    magma_int_t *max_depth;

    if (transA == MagmaNoTrans)
    {
         if (transB == MagmaNoTrans) // NN
	 {
             features = &dgemm_batched_NN_features_pvc;
	     thresholds = &dgemm_batched_NN_thresholds_pvc;
	     left_children = &dgemm_batched_NN_children_left_pvc;
	     right_children = &dgemm_batched_NN_children_right_pvc;
	     max_depth = &dgemm_batched_NN_max_depth_pvc;
	 }
	 else // NT
	 {
             features = &dgemm_batched_NT_features_pvc;
	     thresholds = &dgemm_batched_NT_thresholds_pvc;
	     left_children = &dgemm_batched_NT_children_left_pvc;
	     right_children = &dgemm_batched_NT_children_right_pvc;
	     max_depth = &dgemm_batched_NT_max_depth_pvc;
	 }
    }
    else
    {
        if (transB == MagmaNoTrans) // TN
	{
             features = &dgemm_batched_TN_features_pvc;
	     thresholds = &dgemm_batched_TN_thresholds_pvc;
	     left_children = &dgemm_batched_TN_children_left_pvc;
	     right_children = &dgemm_batched_TN_children_right_pvc;
	     max_depth = &dgemm_batched_TN_max_depth_pvc;
	}
	else // TT
	{
             features = &dgemm_batched_TT_features_pvc;
	     thresholds = &dgemm_batched_TT_thresholds_pvc;
	     left_children = &dgemm_batched_TT_children_left_pvc;
	     right_children = &dgemm_batched_TT_children_right_pvc;
	     max_depth = &dgemm_batched_TT_max_depth_pvc;
	}
    }
    magma_int_t config = evaluate_gemm_tree(m, n, k, features, thresholds,
		    left_children, right_children, *max_depth);

    return config;
}

/***************************************************************************
    Return the preferred configuration (template parameters for the kernel)
    for this batched SGEMM operation, by evaluating the tuning tree.

    Arguments
    ---------
    @param[in]
    transA  magma_trans_t.
            transA specifies the form of op( A ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( A ) = A.
      -     = MagmaTrans:      op( A ) = A**T.
    
    @param[in]
    transB  magma_trans_t.
            transB specifies the form of op( B ) to be used in
            the matrix multiplication as follows:
      -     = MagmaNoTrans:    op( B ) = B.
      -     = MagmaTrans:      op( B ) = B**T.

    @param[in]
    m       INTEGER.
            The number of rows of the A matrix

    @param[in]
    n       INTEGER.
            The number of columns of the A matrix

    @param[in]
    k       INTEGER.
            The number of columns of the B matrix

******************************************************************************/
magma_int_t
magma_sgemm_batched_get_config(magma_trans_t transA, magma_trans_t transB,
		magma_int_t m, magma_int_t n, magma_int_t k)
{
    std::vector<magma_int_t> *features;
    std::vector<magma_int_t> *thresholds;
    std::vector<magma_int_t> *left_children;
    std::vector<magma_int_t> *right_children;
    magma_int_t *max_depth;

    if (transA == MagmaNoTrans)
    {
         if (transB == MagmaNoTrans) // NN
	 {
             features = &sgemm_batched_NN_features_pvc;
	     thresholds = &sgemm_batched_NN_thresholds_pvc;
	     left_children = &sgemm_batched_NN_children_left_pvc;
	     right_children = &sgemm_batched_NN_children_right_pvc;
	     max_depth = &sgemm_batched_NN_max_depth_pvc;
	 }
	 else // NT
	 {
             features = &sgemm_batched_NT_features_pvc;
	     thresholds = &sgemm_batched_NT_thresholds_pvc;
	     left_children = &sgemm_batched_NT_children_left_pvc;
	     right_children = &sgemm_batched_NT_children_right_pvc;
	     max_depth = &sgemm_batched_NT_max_depth_pvc;
	 }
    }
    else
    {
        if (transB == MagmaNoTrans) // TN
	{
             features = &sgemm_batched_TN_features_pvc;
	     thresholds = &sgemm_batched_TN_thresholds_pvc;
	     left_children = &sgemm_batched_TN_children_left_pvc;
	     right_children = &sgemm_batched_TN_children_right_pvc;
	     max_depth = &sgemm_batched_TN_max_depth_pvc;
	}
	else // TT
	{
             features = &sgemm_batched_TT_features_pvc;
	     thresholds = &sgemm_batched_TT_thresholds_pvc;
	     left_children = &sgemm_batched_TT_children_left_pvc;
	     right_children = &sgemm_batched_TT_children_right_pvc;
	     max_depth = &sgemm_batched_TT_max_depth_pvc;
	}
    }
    magma_int_t config = evaluate_gemm_tree(m, n, k, features, thresholds,
		    left_children, right_children, *max_depth);

    return config;
}
