#ifndef EINSUM_IR_FRONTEND_EINSUM_EXPRESSION_ASCII
#define EINSUM_IR_FRONTEND_EINSUM_EXPRESSION_ASCII

#include <string>
#include <vector>
#include <map>
#include "../constants.h"

namespace einsum_ir {
  namespace frontend {
    class EinsumExpressionAscii;
  }
}

class einsum_ir::frontend::EinsumExpressionAscii {
  public:
    /**
     * Splits an input string by the given separation string.
     *
     * @param i_input input string.
     * @param i_separation separation string.
     * @param o_output output substrings.
     **/
    static void split_string( std::string                const & i_input,
                              std::string                const & i_separation,
                              std::vector< std::string >       & o_output );

    /**
     * Extracts the tensors from an einsum expression.
     * The einsum expression is expected to be in the following format:
     *   "tensor1,tensor2,...,tensorN->tensorN+1"
     * Example:
     *   "iae,bf,dcba,cg,dh->hgfei"
     *
     * @param i_expr_string einsum expression.
     * @param o_tensors will be set to extracted tensors 
     */
    static void parse_tensors( std::string                const & i_expr_string,
                               std::vector< std::string >       & o_tensors );


    /**
     * Extracts the dimension sizes from the dimension sizes string.
     * The dimension sizes string is expected to be in the following format:
     *  "size1,size2,...,sizeN"
     * where the dimension of tensor with id i has the size size_i.
     * Example:
     *  "32,8,4,2,16,64,8,8,8"
     * where tensor 1 has dimension size 32, tensor 2 has dimension size 8, etc.
     *
     * @param i_dim_sizes_string dimension sizes string.
     * @param o_dim_sizes will be set to extracted dimension sizes.
     **/
    static void parse_dim_sizes( std::string            const & i_dim_sizes_string,
                                 std::vector< int64_t >       & o_dim_sizes );

    /**
     * Extracts the contraction path for an einsum expression.
     * The path is expected to be in the following format:
     *   "(tensor1,tensor2),(tensor2,tensor3),...",
     * where tensor1 and tensor2 are the tensors to be contracted first,
     * tensor2 and tensor3 are the tensors to be contracted second, etc.
     *
     * Example:
     *   "(1,2),(2,3),(0,1),(0,1)"
     *
     * @param i_expr_string einsum expression.
     * @param o_path will be set to extracted path.
     **/
    static void parse_path( std::string                const & i_expr_string,
                            std::vector< int64_t >           & o_path );

    /**
     * Extracts the dimension ids from an einsum expression.
     * In the returned mapping, the key is the dimension name and the value is the dimension id.
     *
     * Example:
     *   "iae,bf,dcba,cg,dh->hgfei"
     * will be parsed to:
     *   'a' -> 0
     *   'b' -> 1
     *   'c' -> 2
     *   'd' -> 3
     *   'e' -> 4
     *   'f' -> 5
     *   'g' -> 6
     *   'h' -> 7
     *   'i' -> 8
     *
     * @param i_expr_string einsum expression.
     * @param o_map_dim_name_to_id will be set to extracted dimension ids.
     **/
    static void parse_dim_ids( std::string               const & i_expr_string,
                               std::map< char, int64_t >       & o_map_dim_name_to_id );

    /**
     * Extracts the data type from a string.
     * The input data type is expected to be in the following format:
     * "FP32" or "FP64" or "CPX_FP32" or "CPX_FP64"
     *
     * @param i_dtype_string data type string.
     * @param o_dtype will be set to extracted data type. 
     **/
    static void parse_dtype( std::string const & i_dtype_string,
                             data_t            & o_dtype );

    /**
     * Extracts the complex type from a string.
     * The input complex type is expected to be in the following format:
     * "FP32" or "FP64" or "CPX_FP32" or "CPX_FP64"
     *
     * @param i_ctype_string complex type string.
     * @param o_ctype will be set to extracted complex type.
     **/
    static void parse_ctype( std::string const & i_ctype_string,
                             complex_t         & o_ctype );
};

#endif