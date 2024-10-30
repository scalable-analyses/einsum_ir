# Assuming the following contraction:
#                       a  b  c  d  e  f  g
# "bcgf,adeg->abcdef" "24,20,20,24,20,20,24"
#
# Here is simple code in MLIR's linalg dialect:
#
# func.func @binary( %lhs: tensor<20x20x24x20xf32>, %rhs: tensor<24x24x20x24xf32>, %acc: tensor<24x20x20x24x20x20xf32> ) ->  tensor<24x20x20x24x20x20xf32> {
#   %result = linalg.generic {
#     indexing_maps = [ affine_map<(a, b, c, d, e, f, g) -> (b, c, g, f)>,
#                       affine_map<(a, b, c, d, e, f, g) -> (a, d, e, g)>,
#                       affine_map<(a, b, c, d, e, f, g) -> (a, b, c, d, e, f) > ],
#     iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]
#   } ins( %lhs, %rhs : tensor<20x20x24x20xf32>, tensor<24x24x20x24xf32> )
#     outs( %acc : tensor<24x20x20x24x20x20xf32> ) {
#     ^bb0(%lhs_one: f32, %rhs_one: f32, %acc_one: f32):
#       %0 = arith.mulf %lhs_one, %rhs_one : f32
#       %1 = arith.addf %acc_one, %0 : f32
#       linalg.yield %1 : f32
#     } -> tensor<24x20x20x24x20x20xf32>
#
#   return %result: tensor<24x20x20x24x20x20xf32>
# }
#
#
# The following code accepts the contraction string and dimension sizes as inputs
# and generates the MLIR code for the binary contraction.
#
MLIR_BINARY_CONTRACTION = """
func.func @binary( %lhs: tensor<{sizes_left}xf32>, %rhs: tensor<{sizes_right}xf32>, %acc: tensor<{sizes_out}xf32> ) ->  tensor<{sizes_out}xf32> {{
  %result = linalg.generic {{
    indexing_maps = [ affine_map<({all_dims}) -> ({tensor_left})>,
                      affine_map<({all_dims}) -> ({tensor_right})>,
                      affine_map<({all_dims}) -> ({tensor_out})> ],
    iterator_types = [{iterator_types}]
  }} ins( %lhs, %rhs : tensor<{sizes_left}xf32>, tensor<{sizes_right}xf32> )
    outs( %acc : tensor<{sizes_out}xf32> ) {{
    ^bb0(%lhs_one: f32, %rhs_one: f32, %acc_one: f32):
      %0 = arith.mulf %lhs_one, %rhs_one : f32
      %1 = arith.addf %acc_one, %0 : f32
      linalg.yield %1 : f32
    }} -> tensor<{sizes_out}xf32>

  return %result: tensor<{sizes_out}xf32>
}}
"""

def generate_binary_contraction( einsum_string,
                                 dim_sizes ):
  # parse the einsum string
  einsum_str = einsum_string.split( "->" )
  tensors_in = einsum_str[0].split( "," )

  # determine unique dimensions and sort them
  all_dims = "".join( tensors_in ) + einsum_str[1]
  all_dims = list( set( all_dims ) )
  all_dims.sort()

  # determine the dimensions of the tensors
  tensor_left  = []
  tensor_right = []
  tensor_out   = []
  for di in tensors_in[0]:
    tensor_left.append( di )
  for di in tensors_in[1]:
    tensor_right.append( di )
  for di in einsum_str[1]:
    tensor_out.append( di )

  # parse the dimension sizes and convert to dict
  dim_sizes = dim_sizes.split( "," )
  dim_sizes_dict = {}
  for i, dim in enumerate( all_dims ):
    dim_sizes_dict[dim] = dim_sizes[i]

  # determine the sizes of the tensors
  sizes_left  = ""
  sizes_right = ""
  sizes_out   = ""

  for id, di in enumerate( tensor_left ):
    if id != 0:
      sizes_left += "x"
    sizes_left += dim_sizes_dict[di]
    id += 1

  for id, di in enumerate( tensor_right ):
    if id != 0:
      sizes_right += "x"
    sizes_right += dim_sizes_dict[di]
    id += 1

  for id, di in enumerate( tensor_out ):
    if id != 0:
      sizes_out += "x"
    sizes_out += dim_sizes_dict[di]
    id += 1

  # determine iterator types
  iterator_types = ""
  for id, di in enumerate( all_dims ):
    if id != 0:
      iterator_types += ", "
    if di in tensor_out:
      iterator_types += "\"parallel\""
    else:
      iterator_types += "\"reduction\""

  # generate the MLIR code
  mlir_code = MLIR_BINARY_CONTRACTION.format( sizes_left     = sizes_left,
                                              sizes_right    = sizes_right,
                                              sizes_out      = sizes_out,
                                              tensor_left    = ", ".join( tensor_left ),
                                              tensor_right   = ", ".join( tensor_right ),
                                              tensor_out     = ", ".join( tensor_out ),
                                              all_dims       = ", ".join( all_dims ),
                                              iterator_types = iterator_types )

  return mlir_code, sizes_left, sizes_right, sizes_out

import iree.compiler
import iree.runtime
import argparse

if __name__ == "__main__":
  # parse arguments using argparse
  parser = argparse.ArgumentParser()
  parser.add_argument( "--einsum_string",                 type=str, required=True )
  parser.add_argument( "--dim_sizes",                     type=str, required=True )
  parser.add_argument( "--target_backend",                type=str, default="llvm-cpu" )
  parser.add_argument( "--compile_args",                  type=str, nargs='+', default=["--iree-llvmcpu-target-cpu-features=host"] )
  parser.add_argument( "--device",                        type=str, default="local-task" )
  parser.add_argument( "--bench_min_time",                type=str, default=None )
  parser.add_argument( "--task_topology_max_group_count", type=int, default=None )
  parser.add_argument( "--task_topology_cpu_ids",         type=str, default=None )
  args = parser.parse_args()

  einsum_string                 = args.einsum_string
  dim_sizes                     = args.dim_sizes
  target_backend                = args.target_backend
  compile_args                  = args.compile_args
  device                        = args.device
  bench_min_time                = args.bench_min_time
  task_topology_max_group_count = args.task_topology_max_group_count
  task_topology_cpu_ids         = args.task_topology_cpu_ids

  # generate MLIR code and get sizes of tensors
  mlir_code, sizes_left, sizes_right, sizes_out = generate_binary_contraction( einsum_string,
                                                                               dim_sizes )


  # compile MLIR code
  iree_compiled = iree.compiler.compile_str( mlir_code,
                                             target_backends=[ target_backend ],
                                             extra_args=compile_args )

  config = iree.runtime.Config("local-task")
  ctx = iree.runtime.SystemContext(config=config)
  vm_module = iree.runtime.VmModule.copy_buffer(ctx.instance, iree_compiled)

  # benchmark module
  bench_kwargs = {}
  if bench_min_time is not None:
    bench_kwargs["benchmark_min_time"] = bench_min_time
  if task_topology_max_group_count is not None:
    bench_kwargs["task_topology_max_group_count"] = task_topology_max_group_count
  if task_topology_cpu_ids is not None:
    bench_kwargs["task_topology_cpu_ids"] = task_topology_cpu_ids
  if device is not None:
    bench_kwargs["device"] = device

  perf = iree.runtime.benchmark_module( module=vm_module,
                                        entry_function="binary",
                                        inputs=[ sizes_left+"xf32",
                                                 sizes_right+"xf32",
                                                 sizes_out+"xf32" ],
                                        **bench_kwargs )

  num_iters = perf[0].iterations
  iter_per_second = perf[0].user_counters.split( "=" )[1]
  iter_per_second = iter_per_second.split( "/s" )[0]

  # remove "k" from iter_per_second and multiply by 1000 if present
  if "k" in iter_per_second:
    iter_per_second = iter_per_second.replace( "k", "" )
    iter_per_second = float( iter_per_second ) * 1000
    iter_per_second = str( iter_per_second )

  print( perf[0].iterations+','+iter_per_second )
