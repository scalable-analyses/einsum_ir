#ifndef EINSUM_IR_FRONTEND_EINSUM_TREE_ASCII
#define EINSUM_IR_FRONTEND_EINSUM_TREE_ASCII

#include <string>
#include <vector>
#include <map>
#include "../constants.h"

namespace einsum_ir {
  namespace frontend {
    class EinsumTreeAscii;
  }
}

class einsum_ir::frontend::EinsumTreeAscii {
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
     * Parses an einsum tree with recursive calls.
     *
     * @param i_string_tree einsum tree in string representation.
     * @param o_dim_ids vector of all tensors with their dimension ids.
     * @param o_children vector of all tensors with their children.
     * @param o_node_id id of last parsed node.
     **/
    static err_t parse_tree( std::string                            const & i_string_tree,
                             std::vector< std::vector < int64_t > >       & o_dim_ids,
                             std::vector< std::vector < int64_t > >       & o_children,
                             int64_t                                      & o_node_id );
    /**
     * Parses a string of comma seperated values to a vector of integer.
     * 
     * @param i_string_vector string of comma seperated values.
     * @param o_int_vector vector of integer values.
     **/
    static void parse_vector( std::string            const & i_string_vector,
                              std::vector< int64_t >       & o_int_vector );

    /**
     * Counts the number of nodes in an einsum tree.
     * 
     * @param i_string_tree einsum tree in string representation.
     *
     * @return number of nodes in the einsum tree
     **/
    static int64_t count_nodes( std::string const & i_string_tree );

    /**
     * Parses the dimension size from a string to a map.
     * 
     * @param i_string_sizes dimnesion sizes in a string of comma seperated values.
     * @param i_dim_ids vector of all tensors with their dimension ids.
     * @param o_map_dim_sizes map from dimension ids to their size.
     **/
    static void parse_dim_size( std::string                            const & i_string_sizes,
                                std::vector< std::vector < int64_t > > const & i_dim_ids, 
                                std::map< int64_t, int64_t>                  & o_map_dim_sizes );

    /**
     * Splits an einsum tree in outputs and inputs of the outermost opperation  
     * 
     * @param i_string_tree einsum tree in string representation.
     * @param o_left_tree einsum tree of first input in string representation.
     * @param o_right_tree einsum tree of second input in string representation.
     * @param o_out_tree einsum tree of output in string representation.
     **/
    static err_t split_outer_operation( std::string const & i_string_tree,
                                        std::string       & o_left_tree,
                                        std::string       & o_right_tree,
                                        std::string       & o_out_tree );

};

#endif