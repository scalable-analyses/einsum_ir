#ifndef EINSUM_IR_FRONTEND_EINSUM_EXPRESSION
#define EINSUM_IR_FRONTEND_EINSUM_EXPRESSION

#include <cstdint>
#include "../backend/EinsumNode.h"

namespace einsum_ir {
  namespace frontend {
    class EinsumExpression;
  }
}

class einsum_ir::frontend::EinsumExpression {
  public:
    //! number of dimensions
    int64_t m_num_dims = 0;
    //! sizes of the dimensions
    int64_t const * m_dim_sizes = nullptr;

    //! number of binary contractions
    int64_t m_num_conts = 0;

    //! external dimension ids of the einsum string
    //! example: 01,234->124 (ab,cde->bce) gets 01234124
    int64_t const * m_string_dim_ids_ext = nullptr;

    //! sizes of the tensors in the external einsum string
    //! example: 01,234->124 has sizes 233
    int64_t const * m_string_num_dims_ext = nullptr;

    //! datatype of all tensors
    data_t m_dtype = data_t::UNDEFINED_DTYPE;
    //! data points of the tensors 
    void * const * m_data_ptrs = nullptr;

    //! external contraction path
    //! tensors are assumed to be removed after every contraction
    int64_t const * m_path_ext = nullptr;

    //! internal contraction path
    //! tensors are not removed after the contraction, i.e., they have unique ids
    std::vector< int64_t > m_path_int;

    //! internal dimension ids of the einsum string
    //! includes intermediate tensors
    std::vector< int64_t > m_string_dim_ids_int;

    //! sizes of the tensors in the internal einsum string
    std::vector< int64_t > m_string_num_dims_int;

    //! mapping from dim ids to sizes
    //! TODO: remove
    std::map< int64_t, int64_t > m_map_dim_sizes;

    //! nodes of the resulting einsum tree
    std::vector< backend::EinsumNode > m_nodes;

    //! true if the expression was compiled
    bool m_compiled = false;

    /**
     * Derives a histogram showing how often the dimensions appear in the einsum string.
     *
     * @param i_num_dims number of dimensions.
     * @param i_string_size size of the einsum string 
     * @param i_string_dim_ids einsum string containing only the dimension ids.
     * @param o_histogram will be set to number of dimension occurrences in string.
     **/
    static void histogram( int64_t         i_num_dims,
                           int64_t         i_string_size,
                           int64_t const * i_string_dim_ids,
                           int64_t       * o_histogram );

    /**
     * Derives an output string of a binary contraction and updates the histogram accordingly.
     *
     * @param i_num_dims_left number of dimensions in the left string.
     * @param i_num_dims_right number of dimensions in the right string.
     * @param i_dim_ids_left dimension ids of the left tensor.
     * @param i_dim_ids_right dimension ids of the right tensor.
     * @param io_histogram histogram encoding the current state of the einsum tree, will be updated.
     * @param o_substring_out will be set to the output dimension ids (ascending order). 
     **/
    static void substring_out( int64_t                        i_num_dims_left,
                               int64_t                        i_num_dims_right,
                               int64_t                const * i_dim_ids_left,
                               int64_t                const * i_dim_ids_right,
                               int64_t                      * io_histogram,
                               std::vector< int64_t >       & o_substring_out );

    /**
     * Translates the standard contraction path to one with unique tensor ids.
     *
     * A "standard" einsum evaluation, assumes that used tensors are removed from the einsum string.
     * For example the contraction path of the einsum string ea,fb,abcd,gc,hd->efgh
     * could first perform a binary contraction of inputs 0 and 2.
     * This would be the binary contraction ea,abcd->ebcd.
     * In the remaining einsum string the inputs are removed and ebcd gets appended:
     *   fb,gc,hd,ebcd->efgh.
     * Now, the next binary contraction is formulated by means of the new string.
     * For example, tuple (0, 3) would mean the binary contraction fb,ebcd->fcde.
     *
     * In contrast, this routine assumes that all inputs remain as ghost entries in the einsum string.
     * For example, after the first binary contraction with tuple (0,2), we obtain:
     *   >ea<,fb,>abcd<,gc,hd,ebcd->efgh
     * Here, the ghost entries are marked through ">[...]<".
     * Correspondingly, the tuple matching the next contraction fb,ebcd->fcde is given as (1,5).
     *
     * @param i_num_conts number of binary contractions.
     * @param i_path contraction path in the standard formulation.
     * @param o_path contraction path which assumes ghost entries where each tensor id is unique.
     **/
    static void unique_tensor_ids( int64_t         i_num_conts,
                                   int64_t const * i_path,
                                   int64_t       * o_path );

    /**
     * Initializes the einsum expression.
     *
     * @param i_num_dims number of dimensions.
     * @param i_dim_sizes sizes of the dimensions. 
     * @param i_num_conts number of binary contractions.
     * @param i_string_num_ids sizes of the substrings describing the input tensors and output tensor.
     * @param i_string_dim_ids einsum string containing the dimension ids.
     * @param i_path contraction path.
     * @param i_dtype datatype of all tensors.
     * @param i_data_ptr pointers to the tensor's data.
     **/
    void init( int64_t                 i_num_dims,
               int64_t const         * i_dim_sizes,
               int64_t                 i_num_conts,
               int64_t const         * i_string_num_dims,
               int64_t const         * i_string_dim_ids,
               int64_t const         * i_path,
               data_t                  i_dtype,
               void          * const * i_data_ptrs );

    /**
     * Compiles the einsum expression. 
     **/
    err_t compile();

    /**
     * Stores the data of the given tensor internally and locks it.
     * In following execution the stored data is used.
     *
     * @param i_tensor_id id of the the tensor in the einsum string.
     **/
    err_t store_and_lock_data( int64_t i_tensor_id );

    /**
     * Unlocks the data of the given tensor.
     * In following executions the provided data pointer is used.
     **/
    err_t unlock_data( int64_t i_tensor_id );

    /**
     * Evaluates the einsum expression.
     */
    void eval();

    /**
     * Gets the number of scalar operations required to evaluate the expression.
     **/
    int64_t num_ops();
};

#endif