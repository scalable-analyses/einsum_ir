#include <ATen/ATen.h>
#include "catch.hpp"
#include "EinsumTree.h"

#include <string>
#include <vector>
#include <map>
#include <iostream>

TEST_CASE( "creation of a binary contraction", "[einsum_tree]" ) {
  std::string l_string_tree  = "[0,3],[1,2,3]->[0,1,2]";
  std::string l_string_torch = "ad,bcd->abc";

  std::map< int64_t, int64_t> l_dim_sizes = { {0, 2},
                                              {1, 4},
                                              {2, 6},
                                              {3, 8} };

  std::vector< std::vector< int64_t > > l_dim_ids;
  l_dim_ids.push_back({0, 3});
  l_dim_ids.push_back({1, 2, 3});
  l_dim_ids.push_back({0, 1, 2});

  std::vector< std::vector< int64_t > > l_children;
  l_children.push_back({    });
  l_children.push_back({    });
  l_children.push_back({0, 1});

  einsum_ir::data_t l_dtype = einsum_ir::data_t::FP32;

  at::Tensor l_left  = at::rand( {2, 8},    at::ScalarType::Float);
  at::Tensor l_right = at::rand( {4, 6, 8}, at::ScalarType::Float);
  at::Tensor l_out   = at::rand( {2, 4, 6}, at::ScalarType::Float);

  void * l_data_ptrs[] = { l_left.data_ptr(),
                           l_right.data_ptr(),
                           l_out.data_ptr() };

  einsum_ir::frontend::EinsumTree einsum_tree;
  einsum_tree.init( &l_dim_ids,
                    &l_children,
                    &l_dim_sizes,
                    l_dtype,
                    l_data_ptrs );

  einsum_ir::err_t l_err = einsum_tree.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  einsum_tree.eval();

  //refernce
  at::Tensor l_out_ref = at::einsum( l_string_torch,
                                     {l_left, l_right} );

  // check results
  REQUIRE( at::allclose( l_out, l_out_ref )  );

}


TEST_CASE( "creation of unary transposition", "[einsum_tree]" ) {
  std::string l_string_tree = "[0,1]->[1,0]";
  std::string l_string_torch = "ab->ba";

  std::map< int64_t, int64_t> l_dim_sizes = { {0, 7},
                                              {1, 3} };

  std::vector< std::vector< int64_t > > l_dim_ids;
  l_dim_ids.push_back({0, 1});
  l_dim_ids.push_back({1, 0});


  std::vector< std::vector< int64_t > > l_children;
  l_children.push_back({   });
  l_children.push_back({ 0 });

  einsum_ir::data_t l_dtype = einsum_ir::data_t::FP32;

  at::Tensor l_in  = at::rand( {7, 3}, at::ScalarType::Float);
  at::Tensor l_out = at::rand( {3, 7}, at::ScalarType::Float);

  void * l_data_ptrs[] = { l_in.data_ptr(),
                           l_out.data_ptr() };

  einsum_ir::frontend::EinsumTree einsum_tree;
  einsum_tree.init( &l_dim_ids,
                    &l_children,
                    &l_dim_sizes,
                    l_dtype,
                    l_data_ptrs );

  einsum_ir::err_t l_err = einsum_tree.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  einsum_tree.eval();

  //refernce
  at::Tensor l_out_ref = at::einsum( l_string_torch,
                                     {l_in} );

  // check results
  REQUIRE( at::allclose( l_out, l_out_ref )  );
}

TEST_CASE( "creation of a complex einsum tree", "[einsum_tree]" ) {
  std::string l_string_tree = "[[3,0]->[0,3]],[[3,2,4],[1,4,2]->[1,2,3]]->[0,1,2]";
  std::string l_string_torch = "da,dce,bec->abc";

  std::map< int64_t, int64_t> l_dim_sizes = { {0, 2},
                                              {1, 3},
                                              {2, 4},
                                              {3, 5},
                                              {4, 6} };

  std::vector< std::vector< int64_t > > l_dim_ids;
  l_dim_ids.push_back({3, 0});
  l_dim_ids.push_back({0, 3});
  l_dim_ids.push_back({3, 2, 4});
  l_dim_ids.push_back({1, 4, 2});
  l_dim_ids.push_back({1, 2, 3});
  l_dim_ids.push_back({0, 1, 2});

  std::vector< std::vector< int64_t > > l_children;
  l_children.push_back({    });
  l_children.push_back({   0});
  l_children.push_back({    });
  l_children.push_back({    });
  l_children.push_back({2, 3});
  l_children.push_back({1, 4});

  einsum_ir::data_t l_dtype = einsum_ir::data_t::FP64;

  at::Tensor l_in1 = at::rand( {5, 2},    at::ScalarType::Double);
  at::Tensor l_in2 = at::rand( {5, 4, 6}, at::ScalarType::Double);
  at::Tensor l_in3 = at::rand( {3, 6, 4}, at::ScalarType::Double);
  at::Tensor l_out = at::rand( {2, 3, 4}, at::ScalarType::Double);

  void * l_data_ptrs[] = { l_in1.data_ptr(),
                           nullptr,
                           l_in2.data_ptr(),
                           l_in3.data_ptr(),
                           nullptr,
                           l_out.data_ptr() };

  einsum_ir::frontend::EinsumTree einsum_tree;
  einsum_tree.init( &l_dim_ids,
                    &l_children,
                    &l_dim_sizes,
                    l_dtype,
                    l_data_ptrs );

  einsum_ir::err_t l_err = einsum_tree.compile();
  REQUIRE( l_err == einsum_ir::SUCCESS );

  einsum_tree.eval();

  //refernce
  at::Tensor l_out_ref = at::einsum( "da,dce,bec->abc",
                                     {l_in1, l_in2, l_in3} );

  // check results
  REQUIRE( at::allclose( l_out, l_out_ref )  );
}

