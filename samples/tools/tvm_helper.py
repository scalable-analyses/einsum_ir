import argparse
import numpy
import torch
import tvm.auto_scheduler
import time

def einsum_str_to_int( einsum_str ):
  ids_str = einsum_str.split("->")
  ids_str = ids_str[0].split(",") + [ ids_str[1] ]

  # extract unique ids of all tensors
  ids_all_str = "".join( ids_str )
  ids_all_str = "".join( sorted( set( ids_all_str ) ) )

  # convert to dictionary char to id
  ids_all_dict = { c: i for i, c in enumerate( ids_all_str ) }

  # covert to int
  for idx, te in enumerate( ids_str ):
    ids_str[idx] = [ ids_all_dict[c] for c in te ]

  return ids_str

def cpu_to_llvm( cpu ):
  if( cpu == "zen4" ):
    return "llvm -mcpu=znver4"
  elif( cpu == "spr" ):
    return "llvm -mcpu=sapphirerapids"
  elif( cpu == "grace" ):
    return "llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mcpu=neoverse-v2"
  elif( cpu == "apple-m2" ):
    return "llvm -device=arm_cpu -mtriple=arm64-apple-darwin -mcpu=apple-m2"
  elif( cpu == "generic" ):
    return "llvm"
  else:
    raise ValueError(f"Unknown CPU type: {cpu}")

def parse_args():
  parser = argparse.ArgumentParser()
  
  parser.add_argument( "--cpu",
                       type=str,
                       default="generic",
                       choices=["zen4", "spr", "grace", "apple-m2", "generic"],
                       help="CPU architecture to target (default: %(default)s)" )

  parser.add_argument( "--num_measure_trials",
                       type=int,
                       default=1000,
                       help="number of trials for auto-tuning (default: %(default)s)" )

  parser.add_argument( "--timeout",
                       type=int,
                       default=10,
                       help="timeout in seconds for each measurement (default: %(default)s)" )

  parser.add_argument( "--dtype",
                       type=str,
                       default="float32",
                       choices=["float32", "float64"],
                       help="data type (default: %(default)s)" )

  parser.add_argument( "--log_file",
                       type=str,
                       default="tvm_tuning.json",
                       help="log file to store tuning results (default: %(default)s)" )

  args = parser.parse_args()

  return args

def count_ops( func,
               sizes,
               dtype,
               hardware_params,
               target ):
  task = tvm.auto_scheduler.SearchTask( func            = func,
                                        args            = (*sizes, dtype),
                                        hardware_params = hardware_params,
                                        target          = target )
  
  return task.compute_dag.flop_ct

def optimize( func,
              sizes,
              dtype,
              hardware_params,
              target,
              num_measure_trials,
              timeout,
              log_file ):
  task = tvm.auto_scheduler.SearchTask( func            = func,
                                        args            = (*sizes, dtype),
                                        hardware_params = hardware_params,
                                        target          = target )

  runner = tvm.auto_scheduler.LocalRunner( timeout = timeout )

  tune_options = tvm.auto_scheduler.TuningOptions(
    num_measure_trials = num_measure_trials,
    measure_callbacks  = [tvm.auto_scheduler.RecordToFile(log_file)],
    runner             = runner,
    verbose            = 0
  )

  tuner = tvm.auto_scheduler.TaskScheduler( [task],
                                            callbacks=[] )

  tuner.tune( tune_options )

  sch, args = task.apply_best( log_file )

  func_opt = tvm.build( sch,
                        args,
                        target )

  return func_opt

def verify( func,
            einsum_str,
            dim_sizes,
            dtype ):
  tensor_ids = einsum_str_to_int( einsum_str )

  # determine tensor sizes
  tensor_sizes = []
  for te in tensor_ids:
    tensor_sizes.append( [ dim_sizes[di] for di in te ] )

  # init input tensors
  tensors_in_np = []
  for ts in tensor_sizes[:-1]:
    tensors_in_np.append( numpy.random.uniform( size=ts ).astype( dtype ) )

  # convert to torch tensors
  tensors_in_torch = []
  for te in tensors_in_np:
    tensors_in_torch.append( torch.tensor( te ) )

  # run torch einsum (numpy is slow)
  result_torch = torch.einsum( einsum_str, *tensors_in_torch )

  # run tvm einsum
  dev = tvm.cpu()
  tensors_in_tvm = []
  for te in tensors_in_np:
    tensors_in_tvm.append( tvm.nd.array( te, device=dev ) )

  result_tvm = tvm.nd.empty( tensor_sizes[-1], dtype=dtype, device=dev )
  func( *tensors_in_tvm, result_tvm )

  # get max diff
  max_diff = numpy.abs( result_torch.numpy() - result_tvm.asnumpy() )
  max_diff = max_diff / numpy.abs( result_torch.numpy() )
  max_diff = numpy.max( max_diff )

  return max_diff

def bench( func,
           einsum_str,
           dim_sizes,
           dtype ):
  tensor_ids = einsum_str_to_int( einsum_str )

  # determine tensor sizes
  tensor_sizes = []
  for te in tensor_ids:
    tensor_sizes.append( [ dim_sizes[di] for di in te ] )

  # init input tensors
  tensors_in_np = []
  for ts in tensor_sizes[:-1]:
    tensors_in_np.append( numpy.random.uniform( size=ts ).astype( dtype ) )

  # run tvm einsum
  dev = tvm.cpu()
  tensors_in_tvm = []
  for te in tensors_in_np:
    tensors_in_tvm.append( tvm.nd.array( te, device=dev ) )

  result_tvm = tvm.nd.empty( tensor_sizes[-1], dtype=dtype, device=dev )

  evaluator = func.time_evaluator( func.entry_name,
                                   dev,
                                   min_repeat_ms=10000 )

  median = numpy.median( evaluator( *tensors_in_tvm, result_tvm ).results * 1000 )

  return median

def run_all( einsum_str,
             func,
             sizes,
             dtype,
             hardware_params,
             target,
             num_measure_trials,
             timeout,
             log_file ):
  print( "einsum_str:", einsum_str )
  print( "  optimizing" )
  start_time = time.time()
  func_opt = optimize( func,
                       sizes,
                       dtype,
                       hardware_params,
                       target,
                       num_measure_trials,
                       timeout,
                       log_file )
  optimization_time = time.time() - start_time
  print("    optimization time:", optimization_time )

  print( "  verifying")
  max_diff = verify( func_opt,
                     einsum_str,
                     sizes,
                     dtype )
  print( "    max_diff (relative):", max_diff )

  print( "  benchmarking")
  num_ops = count_ops( func,
                       sizes,
                       dtype,
                       hardware_params,
                       target )

  median = bench( func_opt,
                  einsum_str,
                  sizes,
                  dtype )

  print( "    num_ops: ", num_ops )
  print( "    median:  ", median )
  print( "    gflops:  ", num_ops / (median * 1e9) )