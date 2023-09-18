#ifndef EINSUM_IR_FRONTEND_RESIDUAL_BLOCK_DOWN_SAMPLE
#define EINSUM_IR_FRONTEND_RESIDUAL_BLOCK_DOWN_SAMPLE

#include <cstdint>
#include "../backend/EinsumNode.h"
#include "ResidualBlock.h"

namespace einsum_ir {
  namespace frontend {
    class ResidualBlockDownSample;
  }
}

class einsum_ir::frontend::ResidualBlockDownSample {
  private:
    //
    // block with downsampling
    //
    //
    //   ab: image dimensions
    //   cd + ef: kernel dimensions
    //   xyz: respective number of features/channels
    //
    //                 output
    //                   | 
    //              _____+_____
    //             /            \
    //            /          ___abz___
    //           /          /         \
    //       __abu__   ___aby___    zyef
    //      /       \ /         \
    //    uxcd      abx        yxcd
    //               |
    //             input
    ResidualBlock m_res_block;

    //! inner dimension sizes
    std::map< int64_t, int64_t > m_dim_sizes_inner;
    //! outer dimension sizes which apply to the input only
    std::map< int64_t, int64_t > m_dim_sizes_outer_input;
    //! outer dimension sizes
    std::map< int64_t, int64_t > m_dim_sizes_outer;
    //! outer auxiliary dimension sizes
    std::map< int64_t, int64_t > m_dim_sizes_aux_outer;
    //! dimension link
    std::map< int64_t, int64_t > m_dim_link_s_to_p;

    //! dimension ids of the activation tensor which are also used for the output
    std::vector< int64_t > m_dim_ids_abx;
    //! dimension ids of the weight tensor
    std::vector< int64_t > m_dim_ids_uxcd;
    //! dimension ids of the the output tensor
    std::vector< int64_t > m_dim_ids_abu;

    //! first activation node
    einsum_ir::backend::EinsumNode m_node_abx;
    //! first weight node
    einsum_ir::backend::EinsumNode m_node_uxcd;
    //! second activation node
    einsum_ir::backend::EinsumNode m_node_abu;

    //! input and output width (excludes padding)
    int64_t m_width                     = 0;
    //! input and output height (excludes padding)
    int64_t m_height                    = 0;
    //! width of the used kernels
    int64_t m_kernel_width              = 0;
    //! height of the used kernels
    int64_t m_kernel_height             = 0;
    //! width of the used kernels in the downsampling part
    int64_t m_kernel_width_down_sample  = 0;
    //! height of the used kernels in the downsampling part
    int64_t m_kernel_height_down_sample = 0;
    //! number of features occurring in the input tensors
    int64_t m_num_features_in           = 0;
    //! number of features occurring in the intermediate and output tensors
    int64_t m_num_features_out          = 0;
    //! stride used in the first convolution
    int64_t m_stride                    = 0;

    //! data pointer to fist activation tensor
    void  * m_data_abx     = nullptr;
    //! data pointer to weight tensor
    void  * m_data_uxcd    = nullptr;
    //! data pointer to bias
    void  * m_data_abu_aux = nullptr;
    //! data pointer to second activation tensor
    void  * m_data_abu     = nullptr;

    //! dimension offsets used to write in the right padded locations
    std::map< int64_t, int64_t > m_offsets;
    //! per-dimension strides
    std::map< int64_t, int64_t > m_strides;

  public:
    /**
     * Initializes the residual block.
     *
     * @param i_width width excluding padding.
     * @param i_height height excluding padding.
     * @param i_kernel_width width of the used kernels.
     * @param i_kernel_height height of the used kernels.
     * @param i_kernel_width_down_sample width of the used kernels in the downsampling part.
     * @param i_kernel_height_down_sample height of the used kernels in the downsampling part.
     * @param i_num_features_in number of input features.
     * @param i_num_features_out number of output features.
     * @param i_stride strides used in the first convolution.
     * @param i_activations_0 input activations. assumed as (height+kernel_height-1, width+kernel_width-1, num_features_in). 
     * @param i_weights_down_sample weights of the downsampling convolution.
     * @param i_bias_down_sample bias of the downsampling convolution.
     * @param i_weights_0 first weight tensor. assumed as (num_features_new, num_features_old, kernel_height, kernel_width).
     * @param i_bias_0 first bias. assumes as (num_features).
     * @param io_activations_1 intermediate activations. assumed to be zero-initialized as (height+kernel_height-1, width+kernel_width-1, num_features_out). 
     * @param i_weights_1 second weight tensor. assumed as (num_features_new, num_features_old, kernel_height, kernel_width).
     * @param i_bias_1 second bias. assumes as (num_features).
     * @param io_activations_2 output activations to which the result is added to. assumed as (height+kernel_height-1, width+kernel_width-1, num_features_out).
     **/
    void init( int64_t   i_width,
               int64_t   i_height,
               int64_t   i_kernel_width,
               int64_t   i_kernel_height,
               int64_t   i_kernel_width_down_sample,
               int64_t   i_kernel_height_down_sample,
               int64_t   i_num_features_in,
               int64_t   i_num_features_out,
               int64_t   i_stride,
               void    * i_activations_0,
               void    * i_weights_down_sample,
               void    * i_bias_down_sample,
               void    * i_weights_0,
               void    * i_bias_0,
               void    * io_activations_1,
               void    * i_weights_1,
               void    * i_bias_1,
               void    * io_activations_2 );

   /**
    * Compiles the residual block.
    *
   * @return SUCCESS if successful, error code otherwise. 
    **/
   err_t compile();

  /**
   * Evaluates the block.
   **/
  void eval();

  /**
   * Derives the number of operations in the residual block.
   *
   * @return number of operations.
   **/
  int64_t num_ops();
};

#endif