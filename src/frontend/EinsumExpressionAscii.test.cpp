#include "catch.hpp"
#include "EinsumExpressionAscii.h"

TEST_CASE( "Converts an expression string with single characters as dimension ids to standard format.", "[einsum_exp_ascii]" ) {
  std::string l_expr = "iae,bf,dcba,cg,dh->hgfei";
  std::string l_expr_standardized;
  einsum_ir::frontend::EinsumExpressionAscii::schar_to_standard( l_expr,
                                                                 l_expr_standardized );

  REQUIRE( l_expr_standardized == "[i,a,e],[b,f],[d,c,b,a],[c,g],[d,h]->[h,g,f,e,i]" );
}

TEST_CASE( "Converts an expression string in standard format with strings to one using single characters as dimension ids.", "[einsum_exp_ascii]" ) {
  std::string l_expr = "[xix,a,e],[b,f],[d,c,b,a],[c,g],[d,h]->[h,g,f,e,xix]";
  std::string l_expr_schar;
  einsum_ir::frontend::EinsumExpressionAscii::standard_to_schar( l_expr,
                                                                 l_expr_schar );

  REQUIRE( l_expr_schar == "IAE,BF,DCBA,CG,DH->HGFEI" );
}

TEST_CASE( "Converts an expression string in standard format integers to one using single characters as dimension ids.", "[einsum_exp_ascii]" ) {
  std::string l_expr = "[13,5,9],[6,10],[8,7,6,5],[7,11],[8,12]->[12,11,10,9,13]";
  std::string l_expr_schar;
  einsum_ir::frontend::EinsumExpressionAscii::standard_to_schar( l_expr,
                                                                 l_expr_schar );

  REQUIRE( l_expr_schar == "IAE,BF,DCBA,CG,DH->HGFEI" );
}

TEST_CASE( "Parse tensors from an einsum expression with single characters as dimension ids.", "[einsum_exp_ascii]" ) {
  std::string l_expr = "[i,a,e],[b,f],[d,c,b,a],[c,g],[d,h]->[h,g,f,e,i]";
  std::vector< std::string > l_tensors;
  einsum_ir::frontend::EinsumExpressionAscii::parse_tensors( l_expr,
                                                             l_tensors );

  REQUIRE( l_tensors.size() == 6 );
  REQUIRE( l_tensors[0] == "i,a,e" );
  REQUIRE( l_tensors[1] == "b,f" );
  REQUIRE( l_tensors[2] == "d,c,b,a" );
  REQUIRE( l_tensors[3] == "c,g" );
  REQUIRE( l_tensors[4] == "d,h" );
  REQUIRE( l_tensors[5] == "h,g,f,e,i" );
}

TEST_CASE( "Parse tensors from an einsum expression with integers as dimension ids.", "[einsum_exp_ascii]" ) {
  std::string l_expr = "[13,5,9],[6,10],[8,7,6,5],[7,11],[8,12]->[12,11,10,9,13]";
  std::vector< std::string > l_tensors;
  einsum_ir::frontend::EinsumExpressionAscii::parse_tensors( l_expr,
                                                             l_tensors );

  REQUIRE( l_tensors.size() == 6 );
  REQUIRE( l_tensors[0] == "13,5,9" );
  REQUIRE( l_tensors[1] == "6,10" );
  REQUIRE( l_tensors[2] == "8,7,6,5" );
  REQUIRE( l_tensors[3] == "7,11" );
  REQUIRE( l_tensors[4] == "8,12" );
  REQUIRE( l_tensors[5] == "12,11,10,9,13" );
}

TEST_CASE( "Parse dimension ids from an einsum expression with single characters as dimension ids.", "[einsum_exp_ascii]" ) {
  std::string l_expr = "[i,aaa,e],[b,f],[d,c,b,aaa],[c,g],[d,h]->[h,g,f,e,i]";
  std::map< std::string, int64_t > l_map_dim_name_to_id;
  einsum_ir::frontend::EinsumExpressionAscii::parse_dim_ids( l_expr,
                                                             l_map_dim_name_to_id );

  REQUIRE( l_map_dim_name_to_id.size() == 9 );
  REQUIRE( l_map_dim_name_to_id["aaa"] == 0 );
  REQUIRE( l_map_dim_name_to_id["b"]   == 1 );
  REQUIRE( l_map_dim_name_to_id["c"]   == 2 );
  REQUIRE( l_map_dim_name_to_id["d"]   == 3 );
  REQUIRE( l_map_dim_name_to_id["e"]   == 4 );
  REQUIRE( l_map_dim_name_to_id["f"]   == 5 );
  REQUIRE( l_map_dim_name_to_id["g"]   == 6 );
  REQUIRE( l_map_dim_name_to_id["h"]   == 7 );
  REQUIRE( l_map_dim_name_to_id["i"]   == 8 );
}

TEST_CASE( "Parse dimension ids from an einsum expression with integers as dimension ids.", "[einsum_exp_ascii]" ) {
  std::string l_expr = "[13,5,9],[6,10],[8,7,6,5],[7,11],[8,12]->[12,11,10,9,13]";
  std::map< std::string, int64_t > l_map_dim_name_to_id;
  einsum_ir::frontend::EinsumExpressionAscii::parse_dim_ids( l_expr,
                                                             l_map_dim_name_to_id );

  REQUIRE( l_map_dim_name_to_id.size() == 9 );
  REQUIRE( l_map_dim_name_to_id["5"]  == 0 );
  REQUIRE( l_map_dim_name_to_id["6"]  == 1 );
  REQUIRE( l_map_dim_name_to_id["7"]  == 2 );
  REQUIRE( l_map_dim_name_to_id["8"]  == 3 );
  REQUIRE( l_map_dim_name_to_id["9"]  == 4 );
  REQUIRE( l_map_dim_name_to_id["10"] == 5 );
  REQUIRE( l_map_dim_name_to_id["11"] == 6 );
  REQUIRE( l_map_dim_name_to_id["12"] == 7 );
  REQUIRE( l_map_dim_name_to_id["13"] == 8 );
}