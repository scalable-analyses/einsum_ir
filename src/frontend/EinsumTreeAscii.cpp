#include "EinsumTreeAscii.h"
#include <algorithm>
#include <set>

void einsum_ir::frontend::EinsumTreeAscii::split_string( std::string                const & i_input,
                                                         std::string                const & i_separation,
                                                         std::vector< std::string >       & o_output ) {
  std::string l_string = i_input;
  int64_t l_off = 0;
  int64_t l_size_string = l_string.size();
  while( l_off < l_size_string ) {
    l_off = l_string.find( i_separation );
    if( l_off < 0 ) break;
    o_output.push_back( l_string.substr( 0, l_off ) );
    l_string.erase( 0, l_off + i_separation.size() );
  }
  if( l_string.size() > 0 ) {
    o_output.push_back( l_string );
  }
}


int64_t einsum_ir::frontend::EinsumTreeAscii::count_nodes( std::string const & i_string_tree ){

  int64_t l_count = 1;
  int64_t l_size_string = i_string_tree.size();
  for( int64_t l_off = 0; l_off < l_size_string - 3; l_off++ ) {
    if( i_string_tree[ l_off     ] == ']' && 
        i_string_tree[ l_off + 1 ] == ',' && 
        i_string_tree[ l_off + 2 ] == '['    ) {
      l_count++;
    }
    if( i_string_tree[ l_off     ] == ']' && 
        i_string_tree[ l_off + 1 ] == '-' &&
        i_string_tree[ l_off + 2 ] == '>' && 
        i_string_tree[ l_off + 3 ] == '['    ) {
      l_count++;
    }
  }

  return l_count;
}


einsum_ir::err_t einsum_ir::frontend::EinsumTreeAscii::parse_tree( std::string                            const & i_string_tree,
                                                                   std::vector< std::vector < int64_t > >       & o_dim_ids,
                                                                   std::vector< std::vector < int64_t > >       & o_children,
                                                                   int64_t                                      & o_node_id ) {
  //check if tree is a leaf tensor
  if( i_string_tree[0] != '[' ) {
    parse_vector( i_string_tree,
                  o_dim_ids[o_node_id] );
                                                             
    o_node_id++;
    return einsum_ir::SUCCESS;
  }
  else{
    //find operands of outermost operation
    std::string l_left_tree  = "";
    std::string l_right_tree = "";
    std::string l_out_tree   = "";
    err_t l_err = err_t::UNDEFINED_ERROR;
    l_err = split_outer_operation( i_string_tree,
                                   l_left_tree,
                                   l_right_tree,
                                   l_out_tree );
    
    if( l_err != einsum_ir::SUCCESS ){
      return l_err;
    }
    
    //parse left subtree
    l_err = parse_tree( l_left_tree, o_dim_ids, o_children, o_node_id );
    if( l_err != einsum_ir::SUCCESS ){
      return l_err;
    }
    int64_t l_id_left_child = o_node_id - 1;

    //parse right subtree
    if(l_right_tree.size() != 0){
      l_err = parse_tree( l_right_tree, o_dim_ids, o_children, o_node_id );
      if( l_err != einsum_ir::SUCCESS ){
        return l_err;
      }
    }
    int64_t l_id_right_child = o_node_id - 1;

    //update children of output tensor
    o_children[o_node_id].push_back( l_id_left_child );
    if(l_right_tree.size() != 0){
      o_children[o_node_id].push_back( l_id_right_child );
    }

    //update ids of output tensor
    einsum_ir::frontend::EinsumTreeAscii::parse_vector( l_out_tree,
                                                        o_dim_ids[o_node_id] );
    
    o_node_id++;
    return einsum_ir::SUCCESS;
  }
}

void einsum_ir::frontend::EinsumTreeAscii::parse_vector( std::string            const & i_string_vector,
                                                         std::vector< int64_t >       & o_int_vector ){
  o_int_vector.clear();

  std::string l_string_vector = i_string_vector;
  l_string_vector.erase( std::remove( l_string_vector.begin(),
                                      l_string_vector.end(),
                                      ' '),
                         l_string_vector.end());

  std::vector< std::string > l_vector_tmp;
  split_string( l_string_vector,
                std::string(","),
                l_vector_tmp );
  
  o_int_vector.resize( l_vector_tmp.size() );
  for( size_t l_id = 0; l_id < l_vector_tmp.size(); l_id++ ){
    o_int_vector[l_id] = std::stoi( l_vector_tmp[l_id] );
  }
}

void einsum_ir::frontend::EinsumTreeAscii::parse_dim_size( std::string                            const & i_string_sizes,
                                                           std::vector< std::vector < int64_t > > const & i_dim_ids, 
                                                           std::map< int64_t, int64_t>                  & o_map_dim_sizes ){
  
  std::vector< int64_t > l_dim_sizes;
  parse_vector( i_string_sizes,
                l_dim_sizes );

  //add all dimensions to set
  std::set< int64_t> l_dim_names_set;
  for( size_t l_te = 0; l_te < i_dim_ids.size(); l_te++ ) {
    for( std::size_t l_di = 0; l_di < i_dim_ids[l_te].size(); l_di++ ) {
      l_dim_names_set.insert( i_dim_ids[l_te][l_di] );
    }
  }

  //convert set to vector and sort
  std::vector< int64_t > l_dim_names( l_dim_names_set.begin(),
                                      l_dim_names_set.end() );
  std::sort( l_dim_names.begin(),
             l_dim_names.end(),
             []( int64_t a, int64_t b ) {
                 return a < b;
               }
            );

  //create a map of dim_ids to sizes
  o_map_dim_sizes.clear();
  for( std::size_t l_di = 0; l_di < l_dim_names.size(); l_di++ ) {
    o_map_dim_sizes.insert( { l_dim_names[l_di], l_dim_sizes[l_di] } );
  }
}


einsum_ir::err_t einsum_ir::frontend::EinsumTreeAscii::split_outer_operation( std::string const & i_string_tree,
                                                                              std::string       & o_left_tree,
                                                                              std::string       & o_right_tree,
                                                                              std::string       & o_out_tree ){

  int64_t l_id_split_child = 0;
  int64_t l_id_split_out   = 0;
  int64_t l_open_brackets  = 0;
  int64_t l_size_string = i_string_tree.size();
  for( int64_t l_off = 0; l_off < l_size_string; l_off++ ) {
    char l_char = i_string_tree[l_off];
    if(l_char == '['){
      l_open_brackets++;
    }
    if(l_char == ']'){
      l_open_brackets--;
    }
    if(l_char == ',' && l_open_brackets == 0){
      l_id_split_child = l_off;
    }
    if(l_char == '-' && l_open_brackets == 0){
      l_id_split_out = l_off;
      break;
    }
  }

  //binary contraction
  if(l_id_split_child != 0 && l_id_split_out != 0){
    o_left_tree.append(  i_string_tree, 1                   , l_id_split_child                  - 2 );
    o_right_tree.append( i_string_tree, l_id_split_child + 2, l_id_split_out - l_id_split_child - 3 );
    o_out_tree.append(   i_string_tree, l_id_split_out   + 3, l_size_string  - l_id_split_out   - 4 );
  }
  //unary contraction
  else if(l_id_split_child == 0 && l_id_split_out != 0){
    o_left_tree.append(  i_string_tree, 1                   , l_id_split_out                    - 2 );
    o_out_tree.append(   i_string_tree, l_id_split_out   + 3, l_size_string  - l_id_split_out   - 4 );
  }
  //error
  else{
    return einsum_ir::COMPILATION_FAILED;
  }
  return einsum_ir::SUCCESS;
}