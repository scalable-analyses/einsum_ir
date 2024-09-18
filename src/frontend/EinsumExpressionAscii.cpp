#include "EinsumExpressionAscii.h"

#include <algorithm>
#include <set>

void einsum_ir::frontend::EinsumExpressionAscii::split_string( std::string                const & i_input,
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

void einsum_ir::frontend::EinsumExpressionAscii::parse_tensors( std::string                const & i_expr_string,
                                                                std::vector< std::string >       & o_tensors ) {
  o_tensors.clear();

  std::string l_expr = i_expr_string;

  l_expr.erase( std::remove( l_expr.begin(),
                             l_expr.end(),
                             ' '),
                l_expr.end());
  std::vector< std::string > l_tensors_tmp;

  split_string( l_expr,
                std::string("->"),
                l_tensors_tmp );

  split_string( l_tensors_tmp[0],
                std::string(","),
                o_tensors );
  o_tensors.push_back( l_tensors_tmp[1] );
}

void einsum_ir::frontend::EinsumExpressionAscii::parse_dim_sizes( std::string            const & i_dim_sizes_string,
                                                                  std::vector< int64_t >       & o_dim_sizes ) {
  o_dim_sizes.clear();

  std::string l_sizes = i_dim_sizes_string;

  l_sizes.erase( std::remove( l_sizes.begin(),
                              l_sizes.end(),
                              ' '),
                 l_sizes.end());

  std::vector< std::string > l_dim_sizes_tmp;
  split_string( l_sizes,
                std::string(","),
                l_dim_sizes_tmp );
  
  o_dim_sizes.resize( l_dim_sizes_tmp.size() );
  for( std::size_t l_di = 0; l_di < l_dim_sizes_tmp.size(); l_di++ ) {
    o_dim_sizes[l_di] = std::stoi( l_dim_sizes_tmp[l_di] );
  }
}

void einsum_ir::frontend::EinsumExpressionAscii::parse_path( std::string            const & i_expr_string,
                                                             std::vector< int64_t >       & o_path ) {
  o_path.clear();

  std::string l_expr = i_expr_string;

  l_expr.erase( std::remove( l_expr.begin(),
                             l_expr.end(),
                            ' '),
                l_expr.end());
  l_expr.erase( std::remove( l_expr.begin(),
                             l_expr.end(),
                             '('),
                l_expr.end());
  l_expr.erase( std::remove( l_expr.begin(),
                             l_expr.end(),
                             ')'),
                l_expr.end());

  std::vector< std::string > l_path_tmp;
  split_string( l_expr,
                std::string(","),
                l_path_tmp );

  o_path.resize( l_path_tmp.size() );
  for( std::size_t l_co = 0; l_co < l_path_tmp.size(); l_co++ ) {
    o_path[l_co] = std::stoi( l_path_tmp[l_co] );
  }
}

void einsum_ir::frontend::EinsumExpressionAscii::parse_dim_ids( std::string               const & i_expr_string,
                                                                std::map< char, int64_t >       & o_map_dim_name_to_id ) {
  o_map_dim_name_to_id.clear();

  std::vector< std::string > l_tensors;
  parse_tensors( i_expr_string, l_tensors );
  int64_t l_num_tensors = l_tensors.size();

  std::set< char > l_dim_names_set;
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    std::string l_tensor = l_tensors[l_te];

    for( std::size_t l_ch = 0; l_ch < l_tensor.size(); l_ch++ ) {
      l_dim_names_set.insert( l_tensor[l_ch] );
    }
  }
  std::vector< char > l_dim_names( l_dim_names_set.begin(),
                                   l_dim_names_set.end() );

  for( std::size_t l_di = 0; l_di < l_dim_names.size(); l_di++ ) {
    o_map_dim_name_to_id.insert( { l_dim_names[l_di], l_di } );
  }
}

void einsum_ir::frontend::EinsumExpressionAscii::parse_dtype( std::string const & i_dtype_string,
                                                              data_t            & o_dtype ) {
    if( i_dtype_string == "FP32" ) {
      o_dtype = einsum_ir::FP32;
    }
    else if( i_dtype_string == "FP64" ) {
      o_dtype = einsum_ir::FP64;
    }
    else if( i_dtype_string == "CPX_FP32" ) {
      o_dtype = einsum_ir::FP32;
    }
    else if( i_dtype_string == "CPX_FP64" ) {
      o_dtype = einsum_ir::FP64;
    }
    else {
      o_dtype = einsum_ir::UNDEFINED_DTYPE;
    }
}

void einsum_ir::frontend::EinsumExpressionAscii::parse_ctype( std::string const & i_ctype_string,
                                                              complex_t         & o_ctype ) {
    if( i_ctype_string == "FP32" ) {
      o_ctype = einsum_ir::REAL_ONLY;
    }
    else if( i_ctype_string == "FP64" ) {
      o_ctype = einsum_ir::REAL_ONLY;
    }
    else if( i_ctype_string == "CPX_FP32" ) {
      o_ctype = einsum_ir::BATCH_INNER;
    }
    else if( i_ctype_string == "CPX_FP64" ) {
      o_ctype = einsum_ir::BATCH_INNER;
    }
    else {
      o_ctype = einsum_ir::UNDEFINED_CTYPE;
    }
}

void einsum_ir::frontend::EinsumExpressionAscii::parse_loop_order( std::string               const & i_loop_string,
                                                                   std::map< char, int64_t > const & i_map_dim_name_to_id,
                                                                   std::vector< int64_t >          & o_loop_order ){

  o_loop_order.clear();

  std::string l_loops = i_loop_string;

  l_loops.erase( std::remove( l_loops.begin(),
                              l_loops.end(),
                              ' '),
                 l_loops.end());

  std::vector< std::string > l_loop_dims_tmp;
  split_string( l_loops,
                std::string(","),
                l_loop_dims_tmp );
  
  o_loop_order.reserve( l_loop_dims_tmp.size() );
  for( std::size_t l_di = 0; l_di < l_loop_dims_tmp.size(); l_di++ ) {
    o_loop_order.push_back( i_map_dim_name_to_id.at( l_loop_dims_tmp[l_di][0] ) );
  }
}