#include "BinaryContractionFactory.h"

#include "BinaryContractionScalar.h"

#ifdef PP_EINSUM_IR_HAS_LIBXSMM
#include "BinaryContractionTpp.h"
#include "BinaryContractionSfcTpp.h"
#endif

#ifdef PP_EINSUM_IR_HAS_BLAS
#include "BinaryContractionBlas.h"
#endif

#ifdef PP_EINSUM_IR_HAS_TBLIS
#include "BinaryContractionTblis.h"
#endif

bool einsum_ir::backend::BinaryContractionFactory::supports( einsum_ir::backend_t i_backend ) {
  if( i_backend == einsum_ir::backend_t::SCALAR ) {
    return true;
  }

#ifdef PP_EINSUM_IR_HAS_LIBXSMM
  if( i_backend == einsum_ir::backend_t::TPP ) {
    return true;
  }
#endif

#ifdef PP_EINSUM_IR_HAS_BLAS
  if( i_backend == einsum_ir::backend_t::BLAS ) {
    return true;
  }
#endif

#ifdef PP_EINSUM_IR_HAS_TBLIS
  if( i_backend == einsum_ir::backend_t::TBLIS ) {
    return true;
  }
#endif

  return false;
}

einsum_ir::backend::BinaryContraction * einsum_ir::backend::BinaryContractionFactory::create( einsum_ir::backend_t i_backend ) {
  if( i_backend == einsum_ir::backend_t::SCALAR ) {
    return new BinaryContractionScalar();
  }

#ifdef PP_EINSUM_IR_HAS_LIBXSMM
  if( i_backend == einsum_ir::backend_t::TPP ) {
    return new BinaryContractionSfcTpp();
  }
#endif

#ifdef PP_EINSUM_IR_HAS_BLAS
  if( i_backend == einsum_ir::backend_t::BLAS ) {
    return new BinaryContractionBlas();
  }
#endif

#ifdef PP_EINSUM_IR_HAS_TBLIS
  if( i_backend == einsum_ir::backend_t::TBLIS ) {
    return new BinaryContractionTblis();
  }
#endif

  return nullptr;
}