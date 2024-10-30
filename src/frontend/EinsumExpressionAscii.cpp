#include "EinsumExpressionAscii.h"

#include <algorithm>
#include <set>
#include <codecvt>
#include <locale>

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

void einsum_ir::frontend::EinsumExpressionAscii::schar_to_standard( std::string const & i_expr_string,
                                                                    std::string       & o_expr_string ) {
  o_expr_string.clear();

  std::string l_expr = i_expr_string;

  l_expr.erase( std::remove( l_expr.begin(),
                             l_expr.end(),
                             ' '),
                l_expr.end());

  std::vector< std::string > l_tensors;
  split_string( l_expr,
                std::string("->"),
                l_tensors );

  std::vector< std::string > l_input_tensors;
  split_string( l_tensors[0],
                std::string(","),
                l_input_tensors );

  for( std::size_t l_te = 0; l_te < l_input_tensors.size(); l_te++ ) {
    std::string l_tensor = l_input_tensors[l_te];
    o_expr_string += "[";
    for( std::size_t l_di = 0; l_di < l_tensor.size(); l_di++ ) {
      o_expr_string += l_tensor[l_di];
      if( l_di < l_tensor.size() - 1 ) {
        o_expr_string += ",";
      }
    }
    o_expr_string += "]";
    if( l_te < l_input_tensors.size() - 1 ) {
      o_expr_string += ",";
    }
  }

  o_expr_string += "->[";

  for( std::size_t l_di = 0; l_di < l_tensors[1].size(); l_di++ ) {
    o_expr_string += l_tensors[1][l_di];
    if( l_di < l_tensors[1].size() - 1 ) {
      o_expr_string += ",";
    }
  }

  o_expr_string += "]";
}

void einsum_ir::frontend::EinsumExpressionAscii::standard_to_schar( std::string const & i_expr_string,
                                                                    std::string       & o_expr_string ) {
  o_expr_string.clear();

  std::string l_expr = i_expr_string;

  l_expr.erase( std::remove( l_expr.begin(),
                             l_expr.end(),
                             ' '),
                l_expr.end());

  l_expr.erase( 0, 1 );
  l_expr.erase( l_expr.size() - 1, 1 );

  std::vector< std::string > l_tensors;
  split_string( l_expr,
                std::string("]->["),
                l_tensors );
  
  if( l_tensors.size() == 1 ) {
    l_tensors.push_back( "" );
  }

  std::vector< std::string > l_input_tensors;
  split_string( l_tensors[0],
                std::string("],["),
                l_input_tensors );

  // map: dimension name -> dimension id
  std::map< std::string, int64_t > l_map_dim_name_to_id;

  for( std::size_t l_te = 0; l_te < l_input_tensors.size(); l_te++ ) {
    std::string l_tensor = l_input_tensors[l_te];

    std::vector< std::string > l_tensor_dim_names;
    split_string( l_tensor,
                  std::string(","),
                  l_tensor_dim_names );

    for( std::size_t l_di = 0; l_di < l_tensor_dim_names.size(); l_di++ ) {
      l_map_dim_name_to_id.insert( { l_tensor_dim_names[l_di], 0 } );
    }
  }

  std::vector< std::string > l_dim_names;
  for( std::map< std::string, int64_t >::iterator l_di = l_map_dim_name_to_id.begin(); l_di != l_map_dim_name_to_id.end(); l_di++ ) {
    l_dim_names.push_back( l_di->first );
  }
  std::sort( l_dim_names.begin(),
             l_dim_names.end(),
             []( std::string const & a, std::string const & b ) {
               try {
                 return std::stoi( a ) < std::stoi( b );
               } catch( ... ) {
                 return a < b;
               }
             } );

  int64_t l_dim_id = 0;
  for( std::size_t l_di = 0; l_di < l_dim_names.size(); l_di++ ) {
    l_map_dim_name_to_id[ l_dim_names[l_di] ] = l_dim_id;
    l_dim_id++;
  }

  // map: dimension name -> ascii or utf8 character
  std::map< std::string, std::string > l_map_dim_name_to_char;
  for( std::map< std::string, int64_t >::iterator l_di = l_map_dim_name_to_id.begin(); l_di != l_map_dim_name_to_id.end(); l_di++ ) {
    std::string l_dim_name = l_di->first;
    l_dim_id = l_di->second;

    if( l_map_dim_name_to_id.size() <= 52 ) {
      if( l_dim_id < 26 ) {
        l_map_dim_name_to_char.insert( { l_dim_name, std::string(1, l_dim_id+65) } );
      }
      else {
        l_map_dim_name_to_char.insert( { l_dim_name, std::string(1, l_dim_id+97-26) } );
      }
    }
    else {
      std::wstring_convert< std::codecvt_utf8<char32_t>, char32_t > l_conv;
      std::string u8str = l_conv.to_bytes( l_dim_id+161 );
      l_map_dim_name_to_char.insert( { l_dim_name, u8str } );
    }
  }

  // assemble the output expression
  for( std::size_t l_te = 0; l_te < l_input_tensors.size(); l_te++ ) {
    std::string l_tensor = l_input_tensors[l_te];

    std::vector< std::string > l_tensor_dim_names;
    split_string( l_tensor,
                  std::string(","),
                  l_tensor_dim_names );

    for( std::size_t l_di = 0; l_di < l_tensor_dim_names.size(); l_di++ ) {
      o_expr_string += l_map_dim_name_to_char[ l_tensor_dim_names[l_di] ];
    }
    if( l_te < l_input_tensors.size() - 1 ) {
      o_expr_string += ",";
    }
  }
  o_expr_string += "->";


  std::vector< std::string > l_output_tensor;
  split_string( l_tensors[1],
                std::string(","),
                l_output_tensor );
  for( std::size_t l_di = 0; l_di < l_output_tensor.size(); l_di++ ) {
    o_expr_string += l_map_dim_name_to_char[ l_output_tensor[l_di] ];
  }
}

void einsum_ir::frontend::EinsumExpressionAscii::parse_tensors( std::string                const & i_expr_string,
                                                                    std::vector< std::string >       & o_tensors ) {
  o_tensors.clear();

  std::string l_expr = i_expr_string;

  l_expr.erase( 0, 1 );
  l_expr.erase( l_expr.size() - 1, 1 );

  l_expr.erase( std::remove( l_expr.begin(),
                             l_expr.end(),
                             ' '),
                l_expr.end());
  std::vector< std::string > l_tensors_tmp;

  split_string( l_expr,
                std::string("]->["),
                l_tensors_tmp );

  if( l_tensors_tmp.size() == 1 ) {
    l_tensors_tmp.push_back( "" );
  }

  split_string( l_tensors_tmp[0],
                std::string("],["),
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

void einsum_ir::frontend::EinsumExpressionAscii::parse_dim_ids( std::string                      const & i_expr_string,
                                                                std::map< std::string, int64_t >       & o_map_dim_name_to_id ) {
  o_map_dim_name_to_id.clear();

  std::vector< std::string > l_tensors;
  parse_tensors( i_expr_string, l_tensors );
  int64_t l_num_tensors = l_tensors.size();

  std::set< std::string > l_dim_names_set;
  for( int64_t l_te = 0; l_te < l_num_tensors; l_te++ ) {
    std::string l_tensor = l_tensors[l_te];

    std::vector< std::string > l_tensor_dim_names;
    split_string( l_tensor,
                  std::string(","),
                  l_tensor_dim_names );

    for( std::size_t l_di = 0; l_di < l_tensor_dim_names.size(); l_di++ ) {
      l_dim_names_set.insert( l_tensor_dim_names[l_di] );
    }
  }
  std::vector< std::string > l_dim_names( l_dim_names_set.begin(),
                                          l_dim_names_set.end() );
  std::sort( l_dim_names.begin(),
             l_dim_names.end(),
             []( std::string const & a, std::string const & b ) {
               try {
                 return std::stoi( a ) < std::stoi( b );
               } catch( ... ) {
                 return a < b;
               }
             } );

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