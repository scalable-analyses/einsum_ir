import tvm.te
import tvm.auto_scheduler

#        _________hgfei_______
#       /                     \
#    _acegi_                _acfh_
#   /       \              /      \
# cigj     iaje          dh      _acdf_
#                               /      \
#                              bf     dcba
#
#         a,  b,  c,  d,  e,  f,  g,  h,  i,  j
# sizes: 24, 48, 12, 56, 32, 64,  8, 84,  8, 72
# ops counts:
#   cigj,iaje -> acegi:  2*i*a*j*e*c*g   = 2*8*24*72*32*12*8    =    84,934,656
#   bf,dcba -> acdf:     2*b*c*d*a*f     = 2*48*12*56*24*64     =    99,090,432
#   dh,acdf -> acfh:     2*d*h*c*a*f     = 2*56*84*12*24*64     =   173,408,256
#   acegi,acfh -> hgfei: 2*f*c*a*h*g*e*i = 2*64*12*24*84*8*32*8 = 6,341,787,648
#
# total ops: 84,934,656 + 99,090,432 + 173,408,256 + 6,341,787,648 = 6,699,220,992
@tvm.auto_scheduler.register_workload
def einsum_tree_initial( A, B, C, D, E, F, G, H, I, J, dtype ):
  CIGJ = tvm.te.placeholder( (C, I, G, J), name='CIGJ', dtype=dtype )
  IAJE = tvm.te.placeholder( (I, A, J, E), name='IAJE', dtype=dtype )
  BF   = tvm.te.placeholder( (B, F),       name='BF',   dtype=dtype )
  DCBA = tvm.te.placeholder( (D, C, B, A), name='DCBA', dtype=dtype )
  DH   = tvm.te.placeholder( (D, H),       name='DH',   dtype=dtype )

  # reduction axes
  a = tvm.te.reduce_axis( (0, A), name='a' )
  b = tvm.te.reduce_axis( (0, B), name='b' )
  c = tvm.te.reduce_axis( (0, C), name='c' )
  d = tvm.te.reduce_axis( (0, D), name='d' )
  j = tvm.te.reduce_axis( (0, J), name='j' )

  # compute
  acegi = tvm.te.compute( (A, C, E, G, I),
                          lambda a, c, e, g, i: tvm.te.sum( CIGJ[c, i, g, j] * IAJE[i, a, j, e], axis=j ),
                          name='acegi' )
  
  acdf = tvm.te.compute( (A, C, D, F),
                         lambda a, c, d, f: tvm.te.sum( BF[b, f] * DCBA[d, c, b, a], axis=b ),
                         name='acdf' )
  
  acfh = tvm.te.compute( (A, C, F, H),
                         lambda a, c, f, h: tvm.te.sum( DH[d, h] * acdf[a, c, d, f], axis=d ),
                         name='acfh' )
  
  hgfei = tvm.te.compute( (H, G, F, E, I),
                          lambda h, g, f, e, i: tvm.te.sum( acegi[a, c, e, g, i] * acfh[a, c, f, h], axis=[a,c] ),
                          name='hgfei' )
  
  return [CIGJ, IAJE, BF, DCBA, DH, hgfei]

#        _________hgfei_______
#       /                     \
#    _gcaei_                _hfca_
#   /       \              /      \
# iaje     cigj         _fdca_    dh
#                      /      \
#                    dcba     bf
@tvm.auto_scheduler.register_workload
def einsum_tree_compiled( A, B, C, D, E, F, G, H, I, J, dtype ):
  IAJE = tvm.te.placeholder( (I, A, J, E), name='IAJE', dtype=dtype )
  CIGJ = tvm.te.placeholder( (C, I, G, J), name='CIGJ', dtype=dtype )
  DCBA = tvm.te.placeholder( (D, C, B, A), name='DCBA', dtype=dtype )
  BF   = tvm.te.placeholder( (B, F),       name='BF',   dtype=dtype )
  DH   = tvm.te.placeholder( (D, H),       name='DH',   dtype=dtype )

  # reduction axes
  a = tvm.te.reduce_axis( (0, A), name='a' )
  b = tvm.te.reduce_axis( (0, B), name='b' )
  c = tvm.te.reduce_axis( (0, C), name='c' )
  d = tvm.te.reduce_axis( (0, D), name='d' )
  j = tvm.te.reduce_axis( (0, J), name='j' )

  # compute
  gcaei = tvm.te.compute( (G, C, A, E, I),
                          lambda g, c, a, e, i: tvm.te.sum( IAJE[i, a, j, e] * CIGJ[c, i, g, j], axis=j ),
                          name='gcaei' )
  
  fdca = tvm.te.compute( (F, D, C, A),
                         lambda f, d, c, a: tvm.te.sum( DCBA[d, c, b, a] * BF[b, f], axis=b ),
                         name='fdca' )
  
  hfca = tvm.te.compute( (H, F, C, A),
                         lambda h, f, c, a: tvm.te.sum( fdca[f, d, c, a] * DH[d, h], axis=d ),
                         name='hfca' )
  
  hgfei = tvm.te.compute( (H, G, F, E, I),
                          lambda h, g, f, e, i: tvm.te.sum( gcaei[g, c, a, e, i] * hfca[h, f, c, a], axis=[a,c] ),
                          name='hgfei' )
  
  return [CIGJ, IAJE, BF, DCBA, DH, hgfei]

if __name__=="__main__":
  # parse command line arguments
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument( "--target", type=str,
                      choices=["zen4", "grace"],
                      help="target architecture: zen4 or grace" )
  parser.add_argument( "--dtype", type=str,
                        default="float32",
                        help="data type: float32 or float64" )
  parser.add_argument( "--num_measure_trials", type=int,
                        default=1000,
                        help="number of measurement trials" )
  parser.add_argument( "--timeout", type=int,
                        default=60,
                        help="timeout in seconds" )
  parser.add_argument( "--num_generated_code_execs", type=int,
                        default=10,
                        help="number of times to run the generated code for taking average. " )
  args = parser.parse_args()

  # target architecture
  if args.target == "zen4":
    target = tvm.target.Target("llvm -mcpu=znver4")
  elif args.target == "grace":
    target = tvm.target.Target("llvm -device=arm_cpu -mtriple=aarch64-linux-gnu -mcpu=neoverse-v2")
  else:
    # show help message
    parser.print_help()
    exit()

  runner = tvm.auto_scheduler.LocalRunner( timeout = args.timeout,
                                           number  = args.num_generated_code_execs )
  tune_option = tvm.auto_scheduler.TuningOptions(
    num_measure_trials = args.num_measure_trials,
    verbose = 2,
    runner = runner
  )

  print( "*** running bechmarks ***" )
  print( "parameters:" )
  print( "  target:", target )
  print( "  dtype:", args.dtype )
  print( "  num_measure_trials:", args.num_measure_trials )
  print( "  timeout:", args.timeout )
  print( "  num_generated_code_execs:", args.num_generated_code_execs )

  print( "***********************************")
  print( "*** running einsum_tree_initial ***")
  print( "***********************************")
  func = einsum_tree_initial
  sizes = (24, 48, 12, 56, 32, 64, 8, 84, 8, 72)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  print( "************************************")
  print( "*** running einsum_tree_compiled ***")
  print( "************************************")
  func = einsum_tree_compiled
  sizes = (24, 48, 12, 56, 32, 64, 8, 84, 8, 72)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )