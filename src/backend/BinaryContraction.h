#ifndef EINSUM_IR_BACKEND_BINARY_CONTRACTION
#define EINSUM_IR_BACKEND_BINARY_CONTRACTION

#include <cstdint>
#include <vector>
#include <map>
#include "../constants.h"

namespace einsum_ir {
  namespace backend {
    class BinaryContraction;
  }
}

class einsum_ir::backend::BinaryContraction {

  public:
    //! number of dimensions of the left tensor
    int64_t m_num_dims_left = 0;
    //! number of dimensions of the right tensor
    int64_t m_num_dims_right = 0;
    //! number of dimensions of the output tensor
    int64_t m_num_dims_out = 0;

    //! mapping from the dimension ids to the inner dimension sizes
    std::map< int64_t, int64_t > const * m_dim_sizes_inner = nullptr;
    //! mapping from the dimension ids to the outer dimension sizes of the left tensor
    std::map< int64_t, int64_t > const * m_dim_sizes_outer_left = nullptr;
    //! mapping from the dimension ids to the outer dimension sizes of the right tensor
    std::map< int64_t, int64_t > const * m_dim_sizes_outer_right = nullptr;
    //! mapping from the dimension ids to the outer dimension sizes of the auxiliary output tensor
    std::map< int64_t, int64_t > const * m_dim_sizes_outer_out_aux = nullptr;
    //! mapping from the dimension ids to the outer dimension sizes of the output tensor
    std::map< int64_t, int64_t > const * m_dim_sizes_outer_out = nullptr;

    //! left tensor's native dimension ids, i.e., without any imposed ordering
    int64_t const * m_dim_ids_left_native = nullptr;
    //! right tensor's native dimension ids, i.e., without any imposed ordering
    int64_t const * m_dim_ids_right_native = nullptr;
    //! output tensor's dimension ids
    int64_t const * m_dim_ids_out = nullptr;

    //! link between secondary dimensions and primary dimensions
    std::map< int64_t, int64_t > const * m_dim_link_s_to_p = nullptr;

    //! dimension types of the output tensor
    std::vector< dim_t > m_dim_types_out;

    //! ordered dim ids of the left tensor
    std::vector< int64_t > m_dim_ids_left_ordered;
    //! ordered dim ids of the right tensor
    std::vector< int64_t > m_dim_ids_right_ordered;

    //! number of C dimensions
    int64_t m_num_dims_c = 0;
    //! number of M dimensions
    int64_t m_num_dims_m = 0;
    //! number of N dimensions
    int64_t m_num_dims_n = 0;
    //! number of K dimensions
    int64_t m_num_dims_k = 0;
    //! number of I dimensions
    int64_t m_num_dims_i = 0;
    //! number of J dimensions
    int64_t m_num_dims_j = 0;

    //! C dimension ids
    std::vector< int64_t > m_dim_ids_c;
    //! M dimension ids
    std::vector< int64_t > m_dim_ids_m;
    //! N dimension ids
    std::vector< int64_t > m_dim_ids_n;
    //! K dimension ids
    std::vector< int64_t > m_dim_ids_k;
    //! I dimension ids
    std::vector< int64_t > m_dim_ids_i;
    //! J dimension ids
    std::vector< int64_t > m_dim_ids_j;

    //! sizes of the C dimensions
    std::vector< int64_t > m_sizes_c;
    //! sizes of the M dimensions
    std::vector< int64_t > m_sizes_m;
    //! sizes of the N dimensions
    std::vector< int64_t > m_sizes_n;
    //! sizes of the K dimensions
    std::vector< int64_t > m_sizes_k;
    //! sizes of the I dimensions
    std::vector< int64_t > m_sizes_i;
    //! sizes of the J dimensions
    std::vector< int64_t > m_sizes_j;

    //! C strides of the left tensor
    std::vector< int64_t > m_strides_left_c;
    //! M strides of the left tensor
    std::vector< int64_t > m_strides_left_m;
    //! K strides of the left tensor
    std::vector< int64_t > m_strides_left_k;
    //! I strides of the left tensor
    std::vector< int64_t > m_strides_left_i;

    //! C strides of the right tensor
    std::vector< int64_t > m_strides_right_c;
    //! N strides of the right tensor
    std::vector< int64_t > m_strides_right_n;
    //! K strides of the right tensor
    std::vector< int64_t > m_strides_right_k;
    //! J strides of the right tensor
    std::vector< int64_t > m_strides_right_j;

    //! C strides of the auxiliary output tensor
    std::vector< int64_t > m_strides_out_aux_c;
    //! M strides of the auxiliary output tensor
    std::vector< int64_t > m_strides_out_aux_m;
    //! N strides of the auxiliary output tensor
    std::vector< int64_t > m_strides_out_aux_n;

    //! C strides of the output tensor
    std::vector< int64_t > m_strides_out_c;
    //! M strides of the output tensor
    std::vector< int64_t > m_strides_out_m;
    //! N strides of the output tensor
    std::vector< int64_t > m_strides_out_n;

    //! datatype of the left input
    data_t m_dtype_left = UNDEFINED_DTYPE;

    //! datatype of the right input
    data_t m_dtype_right = UNDEFINED_DTYPE;

    //! datatype used during the computations
    data_t m_dtype_comp = UNDEFINED_DTYPE;

    //! datatype of the output
    data_t m_dtype_out = UNDEFINED_DTYPE;

    //! type of the first touch kernel
    kernel_t m_ktype_first_touch = UNDEFINED_KTYPE;

    //! type of the main kernel
    kernel_t m_ktype_main = UNDEFINED_KTYPE;

    //! type of the last touch kernel
    kernel_t m_ktype_last_touch = UNDEFINED_KTYPE;

    //! true if the input tensors were reordered
    bool m_tensors_in_reordered = false;

    //! true if left and right tensor have been swapped
    bool m_tensors_in_swapped = false;

    //! true if the binary contraction was compiled
    bool m_compiled = false;

    /**
     * Derives the dimension types of tensor t2 w.r.t. tensors t0 and t1.
     *
     * @param i_num_dims_t0 number of t0 dimensions.
     * @param i_num_dims_t1 number of t1 dimensions.
     * @param i_num_dims_t2 number of t2 dimensions.
     * @param i_dim_ids_t0 dimension identifiers of t0's dimensions.
     * @param i_dim_ids_t1 dimension identifiers of t1's dimensions.
     * @param i_dim_ids_t2 dimension identifiers of t2's dimensions.
     * @param i_dim_type_t2 dimension type which will be set if dimension is exclusive to t2.
     * @param i_dim_type_t2_t0 dimension type which will be set if dimension is part of t2 and t0.
     * @param i_dim_type_t2_t1 dimension type which will be set if dimension is part of t2 and t1.
     * @param i_dim_type_t2_t0_t1 dimension type which will be set if dimension is part of t2, t0 and t1.
     * @param o_dim_types_t2 will be set to dimension types of t2.
     **/
    static void dim_types( int64_t         i_num_dims_t0,
                           int64_t         i_num_dims_t1,
                           int64_t         i_num_dims_t2,
                           int64_t const * i_dim_ids_t0,
                           int64_t const * i_dim_ids_t1,
                           int64_t const * i_dim_ids_t2,
                           dim_t           i_dim_type_t2,
                           dim_t           i_dim_type_t2_t0,
                           dim_t           i_dim_type_t2_t1,
                           dim_t           i_dim_type_t2_t0_t1,
                           dim_t         * o_dim_types_t2 );

    /**
     * Filters the dimension ids based on the given dimension type.
     *
     * @param i_num_dims_tensor number of tensor dimensions.
     * @param i_dim_type_filter dimension type which is used as filter.
     * @param i_dim_types_tensor types of the tensor's dimensions.
     * @param i_dim_ids_tensor ids of the tensor's dimensions.
     * @param o_dim_ids_filtered will be set to filtered dimension ids.
     * @return number of found dimension matching the given type. 
     **/
    static int64_t filter_dim_ids( int64_t         i_num_dims_tensor,
                                   dim_t           i_dim_type_filter,
                                   dim_t   const * i_dim_types_tensor,
                                   int64_t const * i_dim_ids_tensor,
                                   int64_t       * o_dim_ids_filtered );

    /**
     * Derives the dimension types and ids.
     *
     * @param i_num_dims_left number of dimensions of the left tensor.
     * @param i_num_dims_right number of dimensions of the right tensor.
     * @param i_num_dims_out number of dimensions of the output tensor.
     * @param i_dim_ids_left dimension ids of the left tensor.
     * @param i_dim_ids_right dimension ids of the right tensor.
     * @param i_dim_ids_out dimensions ids of the output tensor.
     * @param o_dim_types_out will be set to dimension types of the output tensor.
     * @param o_dim_ids_c will be set to ids of the C dimensions.
     * @param o_dim_ids_m will be set to ids of the M dimensions.
     * @param o_dim_ids_n will be set to ids of the N dimensions.
     * @param o_dim_ids_k will be set to ids of the K dimensions.
     * @param o_dim_ids_i will be set to ids of the I dimensions.
     * @param o_dim_ids_j will be set to ids of the J dimensions.
     **/
    static void dim_types_ids( int64_t                                 i_num_dims_left,
                               int64_t                                 i_num_dims_right,
                               int64_t                                 i_num_dims_out,
                               int64_t                         const * i_dim_ids_left,
                               int64_t                         const * i_dim_ids_right,
                               int64_t                         const * i_dim_ids_out,
                               std::vector< einsum_ir::dim_t >       & o_dim_types_out,
                               std::vector<          int64_t >       & o_dim_ids_c,
                               std::vector<          int64_t >       & o_dim_ids_m,
                               std::vector<          int64_t >       & o_dim_ids_n,
                               std::vector<          int64_t >       & o_dim_ids_k,
                               std::vector<          int64_t >       & o_dim_ids_i,
                               std::vector<          int64_t >       & o_dim_ids_j );

    /**
     * Orders the dimensions of the left and right input tensor according to the given kernel type.
     *
     * @param i_kernel_type ordering of the input tensors.
     * @param i_num_dims_c number of C dimensions.
     * @param i_num_dims_m number of M dimensions.
     * @param i_num_dims_n number of N dimensions.
     * @param i_num_dims_k number of K dimensions.
     * @param i_num_dims_i number of I dimensions.
     * @param i_num_dims_j number of J dimensions.
     * @param i_num_dims_cb number of blocked C dimensions.
     * @param i_num_dims_mb number of blocked M dimensions.
     * @param i_num_dims_nb number of blocked N dimensions.
     * @param i_num_dims_kb number of blocked K dimensions.
     * @param i_num_dims_ib number of blocked I dimensions.
     * @param i_num_dims_jb number of blocked J dimensions.
     * @param i_dim_ids_c ids of the C dimensions.
     * @param i_dim_ids_m ids of the M dimensions.
     * @param i_dim_ids_n ids of the N dimensions.
     * @param i_dim_ids_k ids of the K dimensions.
     * @param i_dim_ids_i ids of the I dimensions.
     * @param i_dim_ids_j ids of the J dimensions.
     * @param o_dim_ids_left ordered dimension ids of the left tensor.
     * @param o_dim_ids_right ordered dimensions ids of the right tensor.
     *
     * @return SUCCESS if successful, DIMENSION_ORDERING_FAILED if not.
     **/
    static err_t order_dims_in( tenord_t        i_tensor_ordering,
                                int64_t         i_num_dims_c,
                                int64_t         i_num_dims_m,
                                int64_t         i_num_dims_n,
                                int64_t         i_num_dims_k,
                                int64_t         i_num_dims_i,
                                int64_t         i_num_dims_j,
                                int64_t         i_num_dims_cb,
                                int64_t         i_num_dims_mb,
                                int64_t         i_num_dims_nb,
                                int64_t         i_num_dims_kb,
                                int64_t         i_num_dims_ib,
                                int64_t         i_num_dims_jb,
                                int64_t const * i_dim_ids_c,
                                int64_t const * i_dim_ids_m,
                                int64_t const * i_dim_ids_n,
                                int64_t const * i_dim_ids_k,
                                int64_t const * i_dim_ids_i,
                                int64_t const * i_dim_ids_j,
                                int64_t       * o_dim_ids_left,
                                int64_t       * o_dim_ids_right );

    /**
     * Derives the strides for the dimensions in a tensor.
     *
     * @param i_num_dims number of tensor dimensions.
     * @param i_dim_ids dimension ids of the tensor.
     * @param i_dim_sizes key-value (dim_id-size) sizes of the dimensions.
     * @param o_strides will be set set to key-value (dim_id-stride) strides of the dimensions.
     **/
    static void strides( int64_t                              i_num_dims,
                         int64_t const *                      i_dim_ids,
                         std::map< int64_t, int64_t > const & i_dim_sizes,
                         std::map< int64_t, int64_t >       & o_strides );

    /**
     * Derives the strides based on on the sizes of the dimensions in the respective tensors.
     *
     * @param i_num_dims_left number of dimensions of the left tensor.
     * @param i_num_dims_right number of dimensions of the right tensor.
     * @param i_num_dims_out number of dimensions of the output tensor.
     * @param i_num_dims_c number of C dimensions.
     * @param i_num_dims_m number of M dimensions.
     * @param i_num_dims_n number of N dimensions.
     * @param i_num_dims_k number of K dimensions.
     * @param i_num_dims_i number of I dimensions.
     * @param i_num_dims_j number of J dimensions.
     * @param i_dim_ids_left dimension ids of the left tensor.
     * @param i_dim_ids_right dimension ids of the right tensor.
     * @param i_dim_ids_out dimensions ids of the output tensor.
     * @param i_dim_ids_c ids of the C dimensions.
     * @param i_dim_ids_m ids of the M dimensions.
     * @param i_dim_ids_n ids of the N dimensions.
     * @param i_dim_ids_k ids of the K dimensions.
     * @param i_dim_ids_i ids of the I dimensions.
     * @param i_dim_ids_j ids of the J dimensions.
     * @param i_dim_sizes_left outer dimension sizes of the left tensor.
     * @param i_dim_sizes_right outer dimension sizes of the right tensor.
     * @param i_dim_sizes_out_aux outer dimension sizes of the auxiliary output tensor.
     * @param i_dim_sizes_out outer dimension sizes of the output tensor.
     * @param o_strides_left_c will be set to the strides of the left tensor's C dimensions.
     * @param o_strides_left_m will be set to the strides of the left tensor's M dimensions.
     * @param o_strides_left_k will be set to the strides of the left tensor's K dimensions.
     * @param o_strides_left_i will be set to the strides of the left tensor's I dimensions.
     * @param o_strides_right_c will be set to the strides of the right tensor's C dimensions.
     * @param o_strides_right_n will be set to the strides of the right tensor's N dimensions.
     * @param o_strides_right_k will be set to the strides of the right tensor's K dimensions.
     * @param o_strides_right_j will be set to the strides of the right tensor's J dimensions.
     * @param o_strides_out_aux_c will be set to the strides of the auxiliary output tensor's C dimensions.
     * @param o_strides_out_aux_m will be set to the strides of the auxiliary output tensor's M dimensions.
     * @param o_strides_out_aux_n will be set to the strides of the auxiliary output tensor's N dimensions.
     * @param o_strides_out_c will be set to the strides of the output tensor's C dimensions.
     * @param o_strides_out_m will be set to the strides of the output tensor's M dimensions.
     * @param o_strides_out_n will be set to the strides of the output tensor's N dimensions.
     **/
    void strides( int64_t                              i_num_dims_left,
                  int64_t                              i_num_dims_right,
                  int64_t                              i_num_dims_out,
                  int64_t                              i_num_dims_c,
                  int64_t                              i_num_dims_m,
                  int64_t                              i_num_dims_n,
                  int64_t                              i_num_dims_k,
                  int64_t                              i_num_dims_i,
                  int64_t                              i_num_dims_j,
                  int64_t                      const * i_dim_ids_left,
                  int64_t                      const * i_dim_ids_right,
                  int64_t                      const * i_dim_ids_out,
                  int64_t                      const * i_dim_ids_c,
                  int64_t                      const * i_dim_ids_m,
                  int64_t                      const * i_dim_ids_n,
                  int64_t                      const * i_dim_ids_k,
                  int64_t                      const * i_dim_ids_i,
                  int64_t                      const * i_dim_ids_j,
                  std::map< int64_t, int64_t > const & i_dim_sizes_left,
                  std::map< int64_t, int64_t > const & i_dim_sizes_right,
                  std::map< int64_t, int64_t > const & i_dim_sizes_out_aux,
                  std::map< int64_t, int64_t > const & i_dim_sizes_out,
                  int64_t                            * o_strides_left_c,
                  int64_t                            * o_strides_left_m,
                  int64_t                            * o_strides_left_k,
                  int64_t                            * o_strides_left_i,
                  int64_t                            * o_strides_right_c,
                  int64_t                            * o_strides_right_n,
                  int64_t                            * o_strides_right_k,
                  int64_t                            * o_strides_right_j,
                  int64_t                            * o_strides_out_aux_c,
                  int64_t                            * o_strides_out_aux_m,
                  int64_t                            * o_strides_out_aux_n,
                  int64_t                            * o_strides_out_c,
                  int64_t                            * o_strides_out_m,
                  int64_t                            * o_strides_out_n );

    /**
     * Determines the location of the primary dimension in a tensor corresponding to the given secondary one.
     *
     * @param i_dim_id_s id of the secondary dimension.
     * @param i_num_dims_p number of primary dimensions.
     * @param i_dim_ids_p dimension ids of the primary dimensions.
     * @param i_dim_link_s_to_p dimension link from the secondary to the primary dimensions.
     * @param o_location will be set to location of primary dimension in the tensor.
     **/
    static err_t link_secondary_to_primary( int64_t                              i_dim_id_s,
                                            int64_t                              i_num_dims_p,
                                            int64_t                      const * i_dim_ids_p,
                                            std::map< int64_t, int64_t > const & i_dim_link_s_to_p,
                                            int64_t                            & o_location );

    /**
     * Virtual destructor.
     **/
    virtual ~BinaryContraction(){};

    /**
     * Initializes the binary contraction.
     *
     * @param i_num_dims_left number of dimensions of the left tensor.
     * @param i_num_dims_right number of dimensions of the right tensor.
     * @param i_num_dims_out number of dimensions of the output tensor.
     * @param i_dim_sizes_inner mapping from the dimension ids to the inner sizes.
     * @param i_dim_sizes_outer_left mapping from the dimension ids to the left tensor's outer sizes.
     * @param i_dim_sizes_outer_right mapping from the dimension ids to the right tensor's outer sizes.
     * @param i_dim_sizes_outer_out_aux mapping from the dimension ids to the auxiliary out tensor's outer sizes.
     * @param i_dim_sizes_outer_out mapping from the dimension ids to the out tensor's outer sizes.
     * @param i_dim_ids_left dimension ids of the left tensor.
     * @param i_dim_ids_right dimension ids of the right tensor.
     * @param i_dim_ids_out dimensions ids of the output tensor.
     * @param i_dtype_left datatype of the left input.
     * @param i_dtype_right datatype of the right input.
     * @param i_dtype_comp compute data type.
     * @param i_dtype_out datatype of the output.
     * @param i_ktype_first_touch type of the first-touch kernel.
     * @param i_ktype_main type of the main kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     **/
    void init( int64_t                              i_num_dims_left,
               int64_t                              i_num_dims_right,
               int64_t                              i_num_dims_out,
               std::map< int64_t, int64_t > const & i_dim_sizes_inner,
               std::map< int64_t, int64_t > const & i_dim_sizes_outer_left,
               std::map< int64_t, int64_t > const & i_dim_sizes_outer_right,
               std::map< int64_t, int64_t > const & i_dim_sizes_outer_out_aux,
               std::map< int64_t, int64_t > const & i_dim_sizes_outer_out,
               int64_t                      const * i_dim_ids_left,
               int64_t                      const * i_dim_ids_right,
               int64_t                      const * i_dim_ids_out,
               data_t                               i_dtype_left,
               data_t                               i_dtype_right,
               data_t                               i_dtype_comp,
               data_t                               i_dtype_out,
               kernel_t                             i_ktype_first_touch,
               kernel_t                             i_ktype_main,
               kernel_t                             i_ktype_last_touch );

    /**
     * Initializes the binary contraction.
     *
     * @param i_num_dims_left number of dimensions of the left tensor.
     * @param i_num_dims_right number of dimensions of the right tensor.
     * @param i_num_dims_out number of dimensions of the output tensor.
     * @param i_dim_sizes_inner mapping from the dimension ids to the inner sizes.
     * @param i_dim_sizes_outer_left mapping from the dimension ids to the left tensor's outer sizes.
     * @param i_dim_sizes_outer_right mapping from the dimension ids to the right tensor's outer sizes.
     * @param i_dim_sizes_outer_out mapping from the dimension ids to the out tensor's outer sizes.
     * @param i_dim_ids_left dimension ids of the left tensor.
     * @param i_dim_ids_right dimension ids of the right tensor.
     * @param i_dim_ids_out dimensions ids of the output tensor.
     * @param i_dtype_left datatype of the left input.
     * @param i_dtype_right datatype of the right input.
     * @param i_dtype_comp compute data type.
     * @param i_dtype_out datatype of the output.
     * @param i_ktype_first_touch type of the first-touch kernel.
     * @param i_ktype_main type of the main kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     **/
    void init( int64_t                              i_num_dims_left,
               int64_t                              i_num_dims_right,
               int64_t                              i_num_dims_out,
               std::map< int64_t, int64_t > const & i_dim_sizes_inner,
               std::map< int64_t, int64_t > const & i_dim_sizes_outer_left,
               std::map< int64_t, int64_t > const & i_dim_sizes_outer_right,
               std::map< int64_t, int64_t > const & i_dim_sizes_outer_out,
               int64_t                      const * i_dim_ids_left,
               int64_t                      const * i_dim_ids_right,
               int64_t                      const * i_dim_ids_out,
               data_t                               i_dtype_left,
               data_t                               i_dtype_right,
               data_t                               i_dtype_comp,
               data_t                               i_dtype_out,
               kernel_t                             i_ktype_first_touch,
               kernel_t                             i_ktype_main,
               kernel_t                             i_ktype_last_touch );

    /**
     * Initializes the binary contraction.
     *
     * @param i_num_dims_left number of dimensions of the left tensor.
     * @param i_num_dims_right number of dimensions of the right tensor.
     * @param i_num_dims_out number of dimensions of the output tensor.
     * @param i_dim_sizes mapping from the dimension ids to their tensor's sizes.
     * @param i_dim_ids_left dimension ids of the left tensor.
     * @param i_dim_ids_right dimension ids of the right tensor.
     * @param i_dim_ids_out dimensions ids of the output tensor.
     * @param i_dtype_left datatype of the left input.
     * @param i_dtype_right datatype of the right input.
     * @param i_dtype_comp compute data type.
     * @param i_dtype_out datatype of the output.
     * @param i_ktype_first_touch type of the first-touch kernel.
     * @param i_ktype_main type of the main kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     **/
    void init( int64_t                              i_num_dims_left,
               int64_t                              i_num_dims_right,
               int64_t                              i_num_dims_out,
               std::map< int64_t, int64_t > const & i_dim_sizes,
               int64_t                      const * i_dim_ids_left,
               int64_t                      const * i_dim_ids_right,
               int64_t                      const * i_dim_ids_out,
               data_t                               i_dtype_left,
               data_t                               i_dtype_right,
               data_t                               i_dtype_comp,
               data_t                               i_dtype_out,
               kernel_t                             i_ktype_first_touch,
               kernel_t                             i_ktype_main,
               kernel_t                             i_ktype_last_touch );

    /**
     * Initializes the binary contraction.
     *
     * @param i_num_dims_left number of dimensions of the left tensor.
     * @param i_num_dims_right number of dimensions of the right tensor.
     * @param i_num_dims_out number of dimensions of the output tensor.
     * @param i_dim_sizes_inner mapping from the dimension ids to the inner sizes.
     * @param i_dim_sizes_outer_left mapping from the dimension ids to the left tensor's outer sizes.
     * @param i_dim_sizes_outer_right mapping from the dimension ids to the right tensor's outer sizes.
     * @param i_dim_sizes_outer_out_aux mapping from the dimension ids to the auxiliary out tensor's outer sizes.
     * @param i_dim_sizes_outer_out mapping from the dimension ids to the out tensor's outer sizes.
     * @param i_dim_ids_left dimension ids of the left tensor.
     * @param i_dim_ids_right dimension ids of the right tensor.
     * @param i_dim_ids_out dimensions ids of the output tensor.
     * @param i_dim_link_s_to_p link from secondary dims to primary ones.
     * @param i_dtype_left datatype of the left input.
     * @param i_dtype_right datatype of the right input.
     * @param i_dtype_comp compute data type.
     * @param i_dtype_out datatype of the output.
     * @param i_ktype_first_touch type of the first-touch kernel.
     * @param i_ktype_main type of the main kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     **/
    void init( int64_t                              i_num_dims_left,
               int64_t                              i_num_dims_right,
               int64_t                              i_num_dims_out,
               std::map< int64_t, int64_t > const & i_dim_sizes_inner,
               std::map< int64_t, int64_t > const & i_dim_sizes_outer_left,
               std::map< int64_t, int64_t > const & i_dim_sizes_outer_right,
               std::map< int64_t, int64_t > const & i_dim_sizes_outer_out_aux,
               std::map< int64_t, int64_t > const & i_dim_sizes_outer_out,
               int64_t                      const * i_dim_ids_left,
               int64_t                      const * i_dim_ids_right,
               int64_t                      const * i_dim_ids_out,
               std::map< int64_t, int64_t > const & i_dim_link_s_to_p,
               data_t                               i_dtype_left,
               data_t                               i_dtype_right,
               data_t                               i_dtype_comp,
               data_t                               i_dtype_out,
               kernel_t                             i_ktype_first_touch,
               kernel_t                             i_ktype_main,
               kernel_t                             i_ktype_last_touch );

    /**
     * Initializes the binary contraction.
     *
     * @param i_num_dims_left number of dimensions of the left tensor.
     * @param i_num_dims_right number of dimensions of the right tensor.
     * @param i_num_dims_out number of dimensions of the output tensor.
     * @param i_dim_sizes_inner mapping from the dimension ids to the inner sizes.
     * @param i_dim_sizes_outer_left mapping from the dimension ids to the left tensor's outer sizes.
     * @param i_dim_sizes_outer_right mapping from the dimension ids to the right tensor's outer sizes.
     * @param i_dim_sizes_outer_out mapping from the dimension ids to the out tensor's outer sizes.
     * @param i_dim_ids_left dimension ids of the left tensor.
     * @param i_dim_ids_right dimension ids of the right tensor.
     * @param i_dim_ids_out dimensions ids of the output tensor.
     * @param i_dim_link_s_to_p link from secondary dims to primary ones.
     * @param i_dtype_left datatype of the left input.
     * @param i_dtype_right datatype of the right input.
     * @param i_dtype_comp compute data type.
     * @param i_dtype_out datatype of the output.
     * @param i_ktype_first_touch type of the first-touch kernel.
     * @param i_ktype_main type of the main kernel.
     * @param i_ktype_last_touch type of the last touch kernel.
     **/
    void init( int64_t                              i_num_dims_left,
               int64_t                              i_num_dims_right,
               int64_t                              i_num_dims_out,
               std::map< int64_t, int64_t > const & i_dim_sizes_inner,
               std::map< int64_t, int64_t > const & i_dim_sizes_outer_left,
               std::map< int64_t, int64_t > const & i_dim_sizes_outer_right,
               std::map< int64_t, int64_t > const & i_dim_sizes_outer_out,
               int64_t                      const * i_dim_ids_left,
               int64_t                      const * i_dim_ids_right,
               int64_t                      const * i_dim_ids_out,
               std::map< int64_t, int64_t > const & i_dim_link_s_to_p,
               data_t                               i_dtype_left,
               data_t                               i_dtype_right,
               data_t                               i_dtype_comp,
               data_t                               i_dtype_out,
               kernel_t                             i_ktype_first_touch,
               kernel_t                             i_ktype_main,
               kernel_t                             i_ktype_last_touch );

    /**
     * Compiles the base data.
     *
     * @return SUCCESS if successful, error code otherwise.
     **/
    err_t compile_base();

    /**
     * Compiles the binary contraction. 
     *
     * @return SUCCESS if successful, error code otherwise.
     **/
    virtual err_t compile() = 0;

    /**
     * Performs a contraction on the given input data.
     *
     * @param i_tensor_left left input tensor.
     * @param i_tensor_right right input tensor.
     * @param io_tensor_out output tensor. 
     **/
    virtual void contract( void const * i_tensor_left,
                           void const * i_tensor_right,
                           void       * io_tensor_out ) = 0;

    /**
     * Performs a contraction on the given input data.
     *
     * @param i_tensor_left left input tensor.
     * @param i_tensor_right right input tensor.
     * @param i_tensor_out_aux auxiliary data w.r.t. output tensor.
     * @param io_tensor_out output tensor.
     **/
    virtual void contract( void const * i_tensor_left,
                           void const * i_tensor_right,
                           void const * i_tensor_out_aux,
                           void       * io_tensor_out ) = 0;

    /**
     * Gets the dimensions ids of the inputs in the order assumed by the contraction.
     *
     * @param i_side side for which the dimension are requested (0: left, 1:right).
     * @return ordered dimension ids.
     **/
    int64_t const * dim_ids_in_ordered( int64_t i_side );

    /**
     * Gets the number of operations for a single contraction.
     **/
    int64_t num_ops();

    /**
     * Initializes the threading configuration of the contraction.
     *
     * @param i_num_tasks_target number of targeted tasks.
     **/
    virtual void threading( int64_t i_num_tasks_target  ) = 0;
};

#endif