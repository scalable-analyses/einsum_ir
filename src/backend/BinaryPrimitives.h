#ifndef EINSUM_IR_BACKEND_BINARY_DIMENSION_ORDERING
#define EINSUM_IR_BACKEND_BINARY_DIMENSION_ORDERING

#include <cstdint>
#include <vector>
#include <map>
#include "../constants.h"

namespace einsum_ir {
  namespace backend {
    class BinaryPrimitives;
  }
}

class einsum_ir::backend::BinaryPrimitives {
  private:
    //! minimum size of the blocked C dimension.
    int64_t m_size_cb_min = 0;
    //! maximum size of the blocked C dimension.
    int64_t m_size_cb_max = 0;
    //! minimum size of the blocked M dimension.
    int64_t m_size_mb_min = 0;
    //! maximum size of the blocked M dimension.
    int64_t m_size_mb_max = 0;
    //! minimum size of the blocked N dimension.
    int64_t m_size_nb_min = 0;
    //! maximum size of the blocked N dimension.
    int64_t m_size_nb_max = 0;
    //! minimum size of the blocked K dimension.
    int64_t m_size_kb_min = 0;
    //! maximum size of the blocked K dimension.
    int64_t m_size_kb_max = 0;

    // target size for inner k loops
    int64_t m_size_inner_k_loops = 32;
    // target size for m loops
    int64_t m_size_inner_m_loops = 32;
    // target size for n loops
    int64_t m_size_inner_n_loops = 32;

    int64_t m_next_free_id = 100000;

    /**
     * Derives the primitive blocking in the following format:
     *  Left tensor:   kb x mb cb
     *  Right tensor:  nb x kb cb
     *  Output tensor: nb x mb cb
     * (x denotes an arbitrary number of dimensions which are not blocked)
     *
     * @param i_size_mb_min minimum size of the blocked M dimension.
     * @param i_size_mb_max maximum size of the blocked M dimension (min has priority).
     * @param i_size_nb_min minimum size of the blocked N dimension.
     * @param i_size_nb_max maximum size of the blocked N dimension (min has priority).
     * @param i_size_kb_min minimum size of the blocked K dimension.
     * @param i_size_kb_max maximum size of the blocked K dimension (min has priority).
     * @param i_num_dims_left number of dimensions in the left tensor.
     * @param i_num_dims_right number of dimensions in the right tensor.
     * @param i_num_dims_out number of dimensions in the output tensor.
     * @param i_dim_ids_left array of dimension IDs in the left tensor.
     * @param i_dim_ids_right array of dimension IDs in the right tensor.
     * @param i_dim_ids_out array of dimension IDs in the output tensor.
     * @param i_dim_sizes map of dimension sizes.
     * @param i_strides_left map of strides for the left tensor.
     * @param i_strides_right map of strides for the right tensor.
     * @param i_strides_out map of strides for the output tensor.
     * @param o_dim_ids_cb will be set to array of dimension IDs building the blocked C dimension.
     * @param o_dim_ids_mb will be set to array of dimension IDs building the blocked M dimension.
     * @param o_dim_ids_nb will be set to array of dimension IDs building the blocked N dimension.
     * @param o_dim_ids_kb will be set to array of dimension IDs building the blocked K dimension.
     * @return error code.
     **/
    err_t static blocking_left_kb_x_mb_cb_right_nb_x_kb_cb_out_nb_x_mb_cb( int64_t                              i_size_mb_min,
                                                                           int64_t                              i_size_mb_max,
                                                                           int64_t                              i_size_nb_min,
                                                                           int64_t                              i_size_nb_max,
                                                                           int64_t                              i_size_kb_min,
                                                                           int64_t                              i_size_kb_max,
                                                                           int64_t                              i_num_dims_left,
                                                                           int64_t                              i_num_dims_right,
                                                                           int64_t                              i_num_dims_out,
                                                                           int64_t                      const * i_dim_ids_left,
                                                                           int64_t                      const * i_dim_ids_right,
                                                                           int64_t                      const * i_dim_ids_out,
                                                                           std::map< int64_t, int64_t > const * i_dim_sizes,
                                                                           std::map< int64_t, int64_t > const * i_strides_left,
                                                                           std::map< int64_t, int64_t > const * i_strides_right,
                                                                           std::map< int64_t, int64_t > const * i_strides_out,
                                                                           std::vector< int64_t >             * o_dim_ids_cb,
                                                                           std::vector< int64_t >             * o_dim_ids_mb,
                                                                           std::vector< int64_t >             * o_dim_ids_nb,
                                                                           std::vector< int64_t >             * o_dim_ids_kb );

    /**
     * Derives the primitive blocking in the following format:
     * Left tensor:   x cb kb mb
     * Right tensor:  x cb nb kb
     * Output tensor: nb x mb cb
     * (x denotes an arbitrary number of dimensions which are not blocked)
     *
     * @param i_size_cb_min minimum size of the blocked C dimension.
     * @param i_size_cb_max maximum size of the blocked C dimension (min has priority).
     * @param i_size_nb_min minimum size of the blocked N dimension.
     * @param i_size_nb_max maximum size of the blocked N dimension (min has priority).
     * @param i_size_kb_min minimum size of the blocked K dimension.
     * @param i_size_kb_max maximum size of the blocked K dimension (min has priority).
     * @param i_num_dims_left number of dimensions in the left tensor.
     * @param i_num_dims_right number of dimensions in the right tensor.
     * @param i_num_dims_out number of dimensions in the output tensor.
     * @param i_dim_ids_left array of dimension IDs in the left tensor.
     * @param i_dim_ids_right array of dimension IDs in the right tensor.
     * @param i_dim_ids_out array of dimension IDs in the output tensor.
     * @param i_dim_sizes map of dimension sizes.
     * @param i_strides_left map of strides for the left tensor.
     * @param i_strides_right map of strides for the right tensor.
     * @param i_strides_out map of strides for the output tensor.
     * @param o_dim_ids_cb will be set to array of dimension IDs building the blocked C dimension.
     * @param o_dim_ids_mb will be set to array of dimension IDs building the blocked M dimension.
     * @param o_dim_ids_nb will be set to array of dimension IDs building the blocked N dimension.
     * @param o_dim_ids_kb will be set to array of dimension IDs building the blocked K dimension.
     * @return error code.
     **/
    err_t static blocking_left_x_cb_kb_mb_right_x_cb_nb_kb_out_nb_x_mb_cb( int64_t                              i_size_cb_min,
                                                                           int64_t                              i_size_cb_max,
                                                                           int64_t                              i_size_nb_min,
                                                                           int64_t                              i_size_nb_max,
                                                                           int64_t                              i_size_kb_min,
                                                                           int64_t                              i_size_kb_max,
                                                                           int64_t                              i_num_dims_left,
                                                                           int64_t                              i_num_dims_right,
                                                                           int64_t                              i_num_dims_out,
                                                                           int64_t                      const * i_dim_ids_left,
                                                                           int64_t                      const * i_dim_ids_right,
                                                                           int64_t                      const * i_dim_ids_out,
                                                                           std::map< int64_t, int64_t > const * i_dim_sizes,
                                                                           std::map< int64_t, int64_t > const * i_strides_left,
                                                                           std::map< int64_t, int64_t > const * i_strides_right,
                                                                           std::map< int64_t, int64_t > const * i_strides_out,
                                                                           std::vector< int64_t >             * o_dim_ids_cb,
                                                                           std::vector< int64_t >             * o_dim_ids_mb,
                                                                           std::vector< int64_t >             * o_dim_ids_nb,
                                                                           std::vector< int64_t >             * o_dim_ids_kb );

    /**
     * Reorders the dimensions of the left and right input tensors such they following format is obtained:
     *   Left tensor:                  bc bm bk bi kb mb cb
     *   Right tensor:                 bc bn bk bj nb kb cb
     *   Output tensor (not reordered) x nb x mb cb
     *  
     *   - x denotes an arbitrary number of dimensions which are not blocked
     *   - the order of bm and bn follows that of the output tensor
     *   - dimensions I, J and K are ordered by size with the smallest dimension leftmost
     *
     * @param i_size_cb_min minimum size of the blocked C dimension.
     * @param i_size_cb_max maximum size of the blocked C dimension (min has priority).
     * @param i_size_mb_min minimum size of the blocked M dimension.
     * @param i_size_mb_max maximum size of the blocked M dimension (min has priority).
     * @param i_size_nb_min minimum size of the blocked N dimension.
     * @param i_size_nb_max maximum size of the blocked N dimension (min has priority).
     * @param i_size_kb_min minimum size of the blocked K dimension.
     * @param i_size_kb_max maximum size of the blocked K dimension (min has priority).
     * @param i_num_dims_left number of dimensions in the left tensor.
     * @param i_num_dims_right number of dimensions in the right tensor.
     * @param i_num_dims_out number of dimensions in the output tensor.
     * @param i_dim_sizes map of dimension sizes.
     * @param io_dim_ids_left will be set to array of ordered dimension IDs in the left tensor.
     * @param io_dim_ids_right will be set to array of ordered dimension IDs in the right tensor.
     * @param io_dim_ids_out will be set to array of ordered dimension IDs in the output tensor.
     * @return error code.
     **/
    err_t static reorder_left_bc_bm_bk_bi_kb_mb_cb_right_bc_bn_bk_bj_nb_kb_cb_out_native( int64_t                              i_size_cb_min,
                                                                                          int64_t                              i_size_cb_max,
                                                                                          int64_t                              i_size_mb_min,
                                                                                          int64_t                              i_size_mb_max,
                                                                                          int64_t                              i_size_nb_min,
                                                                                          int64_t                              i_size_nb_max,
                                                                                          int64_t                              i_size_kb_min,
                                                                                          int64_t                              i_size_kb_max,
                                                                                          int64_t                              i_num_dims_left,
                                                                                          int64_t                              i_num_dims_right,
                                                                                          int64_t                              i_num_dims_out,
                                                                                          std::map< int64_t, int64_t > const * i_dim_sizes,
                                                                                          int64_t                            * io_dim_ids_left,
                                                                                          int64_t                            * io_dim_ids_right,
                                                                                          int64_t                            * io_dim_ids_out );

    /**
     * Reorders the dimensions of the left and right input tensors such they following format is obtained:
     *   Left tensor:   bc bm bk bi kb mb
     *   Right tensor:  bc bn bk bj nb kb
     *   Output tensor: x nb x mb
     *
     *  - x denotes an arbitrary number of dimensions which are not blocked.
     *  - the order of bm and bn follows that of the output tensor.
     *  - dimensions I, J and K are ordered by size with the smallest dimension leftmost.
     *
     * @param i_size_mb_min minimum size of the blocked M dimension.
     * @param i_size_mb_max maximum size of the blocked M dimension (min has priority).
     * @param i_size_nb_min minimum size of the blocked N dimension.
     * @param i_size_nb_max maximum size of the blocked N dimension (min has priority).
     * @param i_size_kb_min minimum size of the blocked K dimension.
     * @param i_size_kb_max maximum size of the blocked K dimension (min has priority).
     * @param i_num_dims_left number of dimensions in the left tensor.
     * @param i_num_dims_right number of dimensions in the right tensor.
     * @param i_num_dims_out number of dimensions in the output tensor.
     * @param i_dim_sizes map of dimension sizes.
     * @param io_dim_ids_left will be set to array of ordered dimension IDs in the left tensor.
     * @param io_dim_ids_right will be set to array of ordered dimension IDs in the right tensor.
     * @param io_dim_ids_out will be set to array of ordered dimension IDs in the output tensor.
     * @return error code.
     **/
    err_t static reorder_left_bc_bm_bk_bi_kb_mb_right_bc_bn_bk_bj_nb_kb_out_native( int64_t                              i_size_mb_min,
                                                                                    int64_t                              i_size_mb_max,
                                                                                    int64_t                              i_size_nb_min,
                                                                                    int64_t                              i_size_nb_max,
                                                                                    int64_t                              i_size_kb_min,
                                                                                    int64_t                              i_size_kb_max,
                                                                                    int64_t                              i_num_dims_left,
                                                                                    int64_t                              i_num_dims_right,
                                                                                    int64_t                              i_num_dims_out,
                                                                                    std::map< int64_t, int64_t > const * i_dim_sizes,
                                                                                    int64_t                            * io_dim_ids_left,
                                                                                    int64_t                            * io_dim_ids_right,
                                                                                    int64_t                            * io_dim_ids_out );

    /**
     * Reorders the dimensions of the left and right input tensors such they following format is obtained:
     *   Left tensor:                  bc bm bk bi cb kb mb
     *   Right tensor:                 bc bn bk bj cb nb kb
     *   Output tensor (not reordered) x nb x mb cb
     *
     *  - x denotes an arbitrary number of dimensions which are not blocked.
     *  - the order of bm and bn follows that of the output tensor.
     *  - dimensions I, J and K are ordered by size with the smallest dimension leftmost.
     *
     * @param i_size_mb_min minimum size of the blocked M dimension.
     * @param i_size_mb_max maximum size of the blocked M dimension (min has priority).
     * @param i_size_nb_min minimum size of the blocked N dimension.
     * @param i_size_nb_max maximum size of the blocked N dimension (min has priority).
     * @param i_size_kb_min minimum size of the blocked K dimension.
     * @param i_size_kb_max maximum size of the blocked K dimension (min has priority).
     * @param i_num_dims_left number of dimensions in the left tensor.
     * @param i_num_dims_right number of dimensions in the right tensor.
     * @param i_num_dims_out number of dimensions in the output tensor.
     * @param i_dim_sizes map of dimension sizes.
     * @param io_dim_ids_left will be set to array of ordered dimension IDs in the left tensor.
     * @param io_dim_ids_right will be set to array of ordered dimension IDs in the right tensor.
     * @param io_dim_ids_out will be set to array of ordered dimension IDs in the output tensor.
     */
    err_t static reorder_left_bc_bm_bk_bi_cb_kb_mb_right_bc_bn_bk_bj_cb_nb_kb_out_native( int64_t                              i_size_mb_min,
                                                                                          int64_t                              i_size_mb_max,
                                                                                          int64_t                              i_size_nb_min,
                                                                                          int64_t                              i_size_nb_max,
                                                                                          int64_t                              i_size_kb_min,
                                                                                          int64_t                              i_size_kb_max,
                                                                                          int64_t                              i_num_dims_left,
                                                                                          int64_t                              i_num_dims_right,
                                                                                          int64_t                              i_num_dims_out,
                                                                                          std::map< int64_t, int64_t > const * i_dim_sizes,
                                                                                          int64_t                            * io_dim_ids_left,
                                                                                          int64_t                            * io_dim_ids_right,
                                                                                          int64_t                            * io_dim_ids_out );

  public:
    /**
     * Initializes the primitive blocking and reordering.
     *
     * @param i_size_cb_min minimum size of the blocked C dimension.
     * @param i_size_cb_max maximum size of the blocked C dimension.
     * @param i_size_mb_min minimum size of the blocked M dimension.
     * @param i_size_mb_max maximum size of the blocked M dimension.
     * @param i_size_nb_min minimum size of the blocked N dimension.
     * @param i_size_nb_max maximum size of the blocked N dimension.
     * @param i_size_kb_min minimum size of the blocked K dimension.
     * @param i_size_kb_max maximum size of the blocked K dimension.
     **/
    void init( int64_t i_size_cb_min,
               int64_t i_size_cb_max,
               int64_t i_size_mb_min,
               int64_t i_size_mb_max,
               int64_t i_size_nb_min,
               int64_t i_size_nb_max,
               int64_t i_size_kb_min,
               int64_t i_size_kb_max );

    /**
     * Initializes the primitive blocking and reordering.
     *
     * @param i_data_type data type.
     * @param i_backend_type backend type.
     * @return error code.
     **/
    err_t init( data_t    i_data_type,
                backend_t i_backend_type );

    /**
     * Determines if the left and right tensors should be swapped for better blocking properties.
     *
     * @param i_num_dims_left number of dimensions in the left tensor.
     * @param i_num_dims_right number of dimensions in the right tensor.
     * @param i_num_dims_out number of dimensions in the output tensor.
     * @param i_dim_ids_left array of dimension IDs in the left tensor.
     * @param i_dim_ids_right array of dimension IDs in the right tensor.
     * @param i_dim_ids_out array of dimension IDs in the output tensor.
     **/
    bool static swap_inputs( int64_t                              i_num_dims_left,
                             int64_t                              i_num_dims_right,
                             int64_t                              i_num_dims_out,
                             int64_t                      const * i_dim_ids_left,
                             int64_t                      const * i_dim_ids_right,
                             int64_t                      const * i_dim_ids_out );

    /**
     * Derives the primitive blocking for the given tensors.
     * If any of the strides is not provided, a contiguous generalized row-major layout is assumed.
     *
     * @param i_primitive_blocking primitive blocking to derive.
     * @param i_num_dims_left number of dimensions in the left tensor.
     * @param i_num_dims_right number of dimensions in the right tensor.
     * @param i_num_dims_out number of dimensions in the output tensor.
     * @param i_dim_ids_left array of dimension IDs in the left tensor.
     * @param i_dim_ids_right array of dimension IDs in the right tensor.
     * @param i_dim_ids_out array of dimension IDs in the output tensor.
     * @param i_dim_sizes map of inner dimension sizes.
     * @param i_strides_left map of strides for the left tensor (optional).
     * @param i_strides_right map of strides for the right tensor (optional).
     * @param i_strides_out map of strides for the output tensor (optional).
     * @param o_dim_ids_cb will be set to array of dimension IDs building the blocked C dimension.
     * @param o_dim_ids_mb will be set to array of dimension IDs building the blocked M dimension.
     * @param o_dim_ids_nb will be set to array of dimension IDs building the blocked N dimension.
     * @param o_dim_ids_kb will be set to array of dimension IDs building the blocked K dimension.
     * @return error code.
     **/
    err_t blocking( primblo_t                            i_primitive_blocking,
                    int64_t                              i_num_dims_left,
                    int64_t                              i_num_dims_right,
                    int64_t                              i_num_dims_out,
                    int64_t                      const * i_dim_ids_left,
                    int64_t                      const * i_dim_ids_right,
                    int64_t                      const * i_dim_ids_out,
                    std::map< int64_t, int64_t > const * i_dim_sizes,
                    std::map< int64_t, int64_t > const * i_strides_left,
                    std::map< int64_t, int64_t > const * i_strides_right,
                    std::map< int64_t, int64_t > const * i_strides_out,
                    std::vector< int64_t >             * o_dim_ids_cb,
                    std::vector< int64_t >             * o_dim_ids_mb,
                    std::vector< int64_t >             * o_dim_ids_nb,
                    std::vector< int64_t >             * o_dim_ids_kb );

    /**
     * Reorders the dimensions of the left, right and output tensors such that the given tensor ordering is obtained.
     *
     * @param i_primitive_ordering primitive ordering to derive.
     * @param i_num_dims_left number of dimensions in the left tensor.
     * @param i_num_dims_right number of dimensions in the right tensor.
     * @param i_num_dims_out number of dimensions in the output tensor.
     * @param i_dim_sizes map of dimension sizes.
     * @param io_dim_ids_left will be set to array of ordered dimension IDs in the left tensor.
     * @param io_dim_ids_right will be set to array of ordered dimension IDs in the right tensor.
     * @param io_dim_ids_out will be set to array of ordered dimension IDs in the output tensor.
     * @return error code.
     **/
    err_t reorder( tenord_t                             i_primitive_ordering,
                   int64_t                              i_num_dims_left,
                   int64_t                              i_num_dims_right,
                   int64_t                              i_num_dims_out,
                   std::map< int64_t, int64_t > const * i_dim_sizes,
                   int64_t                            * io_dim_ids_left,
                   int64_t                            * io_dim_ids_right,
                   int64_t                            * io_dim_ids_out ) const;

    /**
     * Reorders the dimensions of the left, right and output tensors for the given primitive backend.
     *
     * @param i_backend_type backend type.
     * @param i_num_dims_left number of dimensions in the left tensor.
     * @param i_num_dims_right number of dimensions in the right tensor.
     * @param i_num_dims_out number of dimensions in the output tensor.
     * @param i_dim_sizes map of dimension sizes.
     * @param io_dim_ids_left will be set to array of ordered dimension IDs in the left tensor.
     * @param io_dim_ids_right will be set to array of ordered dimension IDs in the right tensor.
     * @param io_dim_ids_out will be set to array of ordered dimension IDs in the output tensor.
     */
    err_t reorder( backend_t                            i_backend_type,
                   int64_t                              i_num_dims_left,
                   int64_t                              i_num_dims_right,
                   int64_t                              i_num_dims_out,
                   std::map< int64_t, int64_t > const * i_dim_sizes,
                   int64_t                            * io_dim_ids_left,
                   int64_t                            * io_dim_ids_right,
                   int64_t                            * io_dim_ids_out ) const;


    /**
     * Determines a loop execution strategy and split dimensions if necessary
     * 
     * @param io_dim_types map of dimension types.
     * @param io_dim_sizes map of dimension sizes.
     * @param io_strides_left  map of strides of left input
     * @param io_strides_right map of strides of right input
     * @param io_strides_out  map of  strides of output
     * @param i_dim_ids_c array of dimension IDs with type c. 
     * @param i_dim_ids_m array of dimension IDs with type m. 
     * @param i_dim_ids_n array of dimension IDs with type n. 
     * @param i_dim_ids_k array of dimension IDs with type k. 
     * @param i_dim_ids_cb array of dimension IDs building the blocked C dimension.
     * @param i_dim_ids_mb array of dimension IDs building the blocked M dimension.
     * @param i_dim_ids_nb array of dimension IDs building the blocked N dimension.
     * @param i_dim_ids_kb array of dimension IDs building the blocked K dimension.
     * @param o_loop_order heuristic calculated loop execution strategy
     **/
    void compileLoopOrder( std::map< int64_t, dim_t >   & io_dim_types,
                           std::map< int64_t, int64_t > & io_dim_sizes,
                           std::map< int64_t, int64_t > & io_strides_left,
                           std::map< int64_t, int64_t > & io_strides_right,
                           std::map< int64_t, int64_t > & io_strides_out,
                           std::vector< int64_t > const & i_dim_ids_c,
                           std::vector< int64_t > const & i_dim_ids_m,
                           std::vector< int64_t > const & i_dim_ids_n,
                           std::vector< int64_t > const & i_dim_ids_k,
                           std::vector< int64_t > const & i_dim_ids_cb,
                           std::vector< int64_t > const & i_dim_ids_mb,
                           std::vector< int64_t > const & i_dim_ids_nb,
                           std::vector< int64_t > const & i_dim_ids_kb,
                           std::vector< int64_t >       & o_loop_order 
                          );

    /**
     * finds a divisor of an integer as close to target target integer
     * 
     * @param i_dim_size dimension that is split
     * @param i_target_size a target integer
     * 
     * @return best found divisor
     **/
    int64_t splitDimension( int64_t i_dim_size,
                            int64_t i_target_size
                          );

};

#endif