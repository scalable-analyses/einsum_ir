import tvm.te
import tvm.auto_scheduler
import os

# blocked_reordered 1. "abdfe,cf->abcde" "48,36,24,36,48,36" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_abdfe_cf_abcde( A, B, C, D, E, F, dtype ):
  ABDFE = tvm.te.placeholder( (A, B, D, F, E), name="ABDFE", dtype=dtype )
  CF    = tvm.te.placeholder( (C, F),          name="CF",    dtype=dtype )

  f = tvm.te.reduce_axis( (0, F), name="f" )

  abcde = tvm.te.compute( (A, B, C, D, E),
                      lambda a, b, c, d, e: tvm.te.sum( ABDFE[a, b, d, f, e] * CF[c, f], axis=f ),
                      name="abcde" )

  return [ABDFE, CF, abcde]

# blocked_reordered 2. "acdfe,bf->abcde" "48,24,36,36,48,36" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_acdfe_bf_abcde( A, B, C, D, E, F, dtype ):
  ACDFE = tvm.te.placeholder( (A, C, D, F, E), name="ACDFE", dtype=dtype )
  BF    = tvm.te.placeholder( (B, F),          name="BF",    dtype=dtype )

  f = tvm.te.reduce_axis( (0, F), name="f" )

  abcde = tvm.te.compute( (A, B, C, D, E),
                      lambda a, b, c, d, e: tvm.te.sum( ACDFE[a, c, d, f, e] * BF[b, f], axis=f ),
                      name="abcde" )

  return [ACDFE, BF, abcde]

# blocked_reordered 3. "abed,ce->abcd" "96,84,24,96,96" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_abed_ce_abcd( A, B, C, D, E, dtype ):
  ABED = tvm.te.placeholder( (A, B, E, D), name="ABED", dtype=dtype )
  CE   = tvm.te.placeholder( (C, E),       name="CE",   dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )

  abcd = tvm.te.compute( (A, B, C, D),
                      lambda a, b, c, d: tvm.te.sum( ABED[a, b, e, d] * CE[c, e], axis=e ),
                      name="abcd" )

  return [ABED, CE, abcd]

# blocked_reordered 4. "abcfe,df->abcde" "48,36,36,24,48,48" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_abcfe_df_abcde( A, B, C, D, E, F, dtype ):
  ABCFE = tvm.te.placeholder( (A, B, C, F, E), name="ABCFE", dtype=dtype )
  DF    = tvm.te.placeholder( (D, F),          name="DF",    dtype=dtype )

  f = tvm.te.reduce_axis( (0, F), name="f" )

  abcde = tvm.te.compute( (A, B, C, D, E),
                      lambda a, b, c, d, e: tvm.te.sum( ABCFE[a, b, c, f, e] * DF[d, f], axis=f ),
                      name="abcde" )

  return [ABCFE, DF, abcde]

# blocked_reordered 5. "aced,be->abcd" "96,24,84,96,84" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_aced_be_abcd( A, B, C, D, E, dtype ):
  ACED = tvm.te.placeholder( (A, C, E, D), name="ACED", dtype=dtype )
  BE   = tvm.te.placeholder( (B, E),       name="BE",   dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )

  abcd = tvm.te.compute( (A, B, C, D),
                      lambda a, b, c, d: tvm.te.sum( ACED[a, c, e, d] * BE[b, e], axis=e ),
                      name="abcd" )

  return [ACED, BE, abcd]

# blocked_reordered 6. "jki,efghjk->efghi" "6,64,6,64,24,6,64" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_jki_efghjk_efghi( E, F, G, H, I, J, K, dtype ):
  JKI    = tvm.te.placeholder( (J, K, I),          name="JKI", dtype=dtype )
  EFGHJK = tvm.te.placeholder( (E, F, G, H, J, K), name="EFGHJK", dtype=dtype )

  j = tvm.te.reduce_axis( (0, J), name="j" )
  k = tvm.te.reduce_axis( (0, K), name="k" )

  efghi = tvm.te.compute( (E, F, G, H, I),
                      lambda e, f, g, h, i: tvm.te.sum( JKI[j, k, i] * EFGHJK[e, f, g, h, j, k], axis=[j, k] ),
                      name="efghi" )

  return [JKI, EFGHJK, efghi]

# blocked_reordered 7. "abed,ce->abcd" "96,84,24,84,96" "(0,1)"
# same contraction as blocked_reordered 3, different sizes

# blocked_reordered 8. "bcgf,adeg->abcdef" "24,20,20,24,20,20,24" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_bcgf_adeg_abcdef( A, B, C, D, E, F, G, dtype ):
  BCGF = tvm.te.placeholder( (B, C, G, F), name="BCGF", dtype=dtype )
  ADEG = tvm.te.placeholder( (A, D, E, G), name="ADEG", dtype=dtype )

  g = tvm.te.reduce_axis( (0, G), name="g" )

  abcdef = tvm.te.compute( (A, B, C, D, E, F),
                      lambda a, b, c, d, e, f: tvm.te.sum( BCGF[b, c, g, f] * ADEG[a, d, e, g], axis=g ),
                      name="abcdef" )

  return [BCGF, ADEG, abcdef]

# blocked_reordered 9. "bdgf,aceg->abcdef" "24,20,20,24,20,20,24" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_bdgf_aceg_abcdef( A, B, C, D, E, F, G, dtype ):
  BDGF = tvm.te.placeholder( (B, D, G, F), name="BDGF", dtype=dtype )
  ACEG = tvm.te.placeholder( (A, C, E, G), name="ACEG", dtype=dtype )

  g = tvm.te.reduce_axis( (0, G), name="g" )

  abcdef = tvm.te.compute( (A, B, C, D, E, F),
                      lambda a, b, c, d, e, f: tvm.te.sum( BDGF[b, d, g, f] * ACEG[a, c, e, g], axis=g ),
                      name="abcdef" )

  return [BDGF, ACEG, abcdef]

# blocked_reordered 10. "acgf,bdeg->abcdef" "24,20,20,24,20,20,24" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_acgf_bdeg_abcdef( A, B, C, D, E, F, G, dtype ):
  ACGF = tvm.te.placeholder( (A, C, G, F), name="ACGF", dtype=dtype )
  BDEG = tvm.te.placeholder( (B, D, E, G), name="BDEG", dtype=dtype )

  g = tvm.te.reduce_axis( (0, G), name="g" )

  abcdef = tvm.te.compute( (A, B, C, D, E, F),
                      lambda a, b, c, d, e, f: tvm.te.sum( ACGF[a, c, g, f] * BDEG[b, d, e, g], axis=g ),
                      name="abcdef" )

  return [ACGF, BDEG, abcdef]

# blocked_reordered 11. "abgf,cdeg->abcdef" "24,20,20,24,20,20,24" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_abgf_cdeg_abcdef( A, B, C, D, E, F, G, dtype ):
  ABGF = tvm.te.placeholder( (A, B, G, F), name="ABGF", dtype=dtype )
  CDEG = tvm.te.placeholder( (C, D, E, G), name="CDEG", dtype=dtype )

  g = tvm.te.reduce_axis( (0, G), name="g" )

  abcdef = tvm.te.compute( (A, B, C, D, E, F),
                      lambda a, b, c, d, e, f: tvm.te.sum( ABGF[a, b, g, f] * CDEG[c, d, e, g], axis=g ),
                      name="abcdef" )

  return [ABGF, CDEG, abcdef]

# blocked_reordered 12. "efhjki,gjk->efghi" "6,64,24,4,94,6,64" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_efhjki_gjk_efghi( E, F, G, H, I, J, K, dtype ):
  EFHJKI = tvm.te.placeholder( (E, F, H, J, K, I), name="EFHJKI", dtype=dtype )
  GJK    = tvm.te.placeholder( (G, J, K),          name="GJK",    dtype=dtype )

  j = tvm.te.reduce_axis( (0, J), name="j" )
  k = tvm.te.reduce_axis( (0, K), name="k" )

  efghi = tvm.te.compute( (E, F, G, H, I),
                      lambda e, f, g, h, i: tvm.te.sum( EFHJKI[e, f, h, j, k, i] * GJK[g, j, k], axis=[j, k] ),
                      name="efghi" )

  return [EFHJKI, GJK, efghi]

# blocked_reordered 13. "bced,ae->abcd" "96,84,84,84,96" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_bced_ae_abcd( A, B, C, D, E, dtype ):
  BCED = tvm.te.placeholder( (B, C, E, D), name="BCED", dtype=dtype )
  AE   = tvm.te.placeholder( (A, E),       name="AE",   dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )

  abcd = tvm.te.compute( (A, B, C, D),
                      lambda a, b, c, d: tvm.te.sum( BCED[b, c, e, d] * AE[a, e], axis=e ),
                      name="abcd" )

  return [BCED, AE, abcd]

# blocked_reordered 14. "aced,be->abcd" "96,84,84,84,96" "(0,1)"
# same contraction as blocked_reordered 5, different sizes

# blocked_reordered 15. "abed,ce->abcd" "96,84,84,84,96" "(0,1)"
# same contraction as blocked_reordered 3 and 7, different sizes

# blocked_reordered 16. "aedc,ebd->abc" "96,84,84,84,96" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_aedc_ebd_abc( A, B, C, D, E, dtype ):
  AEDC = tvm.te.placeholder( (A, E, D, C), name="AEDC", dtype=dtype )
  EBD  = tvm.te.placeholder( (E, B, D),    name="EBD",  dtype=dtype )

  d = tvm.te.reduce_axis( (0, D), name="d" )
  e = tvm.te.reduce_axis( (0, E), name="e" )

  abc = tvm.te.compute( (A, B, C),
                      lambda a, b, c: tvm.te.sum( AEDC[a, e, d, c] * EBD[e, b, d], axis=[d,e] ),
                      name="abc" )

  return [AEDC, EBD, abc]

# blocked_reordered 17. "gkiljh,ekilfj->efgh" "6,64,4,94,6,64,6,64" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_gkiljh_ekilfj_efgh( E, F, G, H, I, J, K, L, dtype ):
  GKILJH = tvm.te.placeholder( (G, K, I, L, J, H), name="GKILJH", dtype=dtype )
  EKILFJ = tvm.te.placeholder( (E, K, I, L, F, J), name="EKILFJ", dtype=dtype )

  i = tvm.te.reduce_axis( (0, I), name="i" )
  j = tvm.te.reduce_axis( (0, J), name="j" )
  k = tvm.te.reduce_axis( (0, K), name="k" )
  l = tvm.te.reduce_axis( (0, L), name="l" )

  efgh = tvm.te.compute( (E, F, G, H),
                      lambda e, f, g, h: tvm.te.sum( GKILJH[g, k, i, l, j, h] * EKILFJ[e, k, i, l, f, j], axis=[i, j, k, l] ),
                      name="efgh" )

  return [GKILJH, EKILFJ, efgh]

# blocked_reordered 18. "gikljh,eiklfj->efgh" "6,64,4,94,4,94,6,64" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_gikljh_eiklfj_efgh( E, F, G, H, I, J, K, L, dtype ):
  GIKLJH = tvm.te.placeholder( (G, I, K, L, J, H), name="GIKLJH", dtype=dtype )
  EIKLFJ = tvm.te.placeholder( (E, I, K, L, F, J), name="EIKLFJ", dtype=dtype )

  i = tvm.te.reduce_axis( (0, I), name="i" )
  j = tvm.te.reduce_axis( (0, J), name="j" )
  k = tvm.te.reduce_axis( (0, K), name="k" )
  l = tvm.te.reduce_axis( (0, L), name="l" )

  efgh = tvm.te.compute( (E, F, G, H),
                      lambda e, f, g, h: tvm.te.sum( GIKLJH[g, i, k, l, j, h] * EIKLFJ[e, i, k, l, f, j], axis=[i, j, k, l] ),
                      name="efgh" )

  return [GIKLJH, EIKLFJ, efgh]

# blocked_reordered 19. "efiklj,ghkl->efghij" "6,64,4,94,4,94,6,64" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_efiklj_ghkl_efghij( E, F, G, H, I, J, K, L, dtype ):
  EFIKLJ = tvm.te.placeholder( (E, F, I, K, L, J), name="EFIKLJ", dtype=dtype )
  GHKL   = tvm.te.placeholder( (G, H, K, L),       name="GHKL",   dtype=dtype )

  k = tvm.te.reduce_axis( (0, K), name="k" )
  l = tvm.te.reduce_axis( (0, L), name="l" )

  efghij = tvm.te.compute( (E, F, G, H, I, J),
                      lambda e, f, g, h, i, j: tvm.te.sum( EFIKLJ[e, f, i, k, l, j] * GHKL[g, h, k, l], axis=[k, l] ),
                      name="efghij" )

  return [EFIKLJ, GHKL, efghij]

# blocked_reordered 20. "efiklj,ghkl->efghij" "6,64,6,64,4,94,4,94" "(0,1)"
# same contraction as blocked_reordered 19, different sizes

# blocked_reordered 21. "fihg,dieh->defg" "151,48,181,40,151,48" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_fihg_dieh_defg( D, E, F, G, H, I, dtype ):
  FIHG = tvm.te.placeholder( (F, I, H, G), name="FIHG", dtype=dtype )
  DIEH = tvm.te.placeholder( (D, I, E, H), name="DIEH", dtype=dtype )

  i = tvm.te.reduce_axis( (0, I), name="i" )
  h = tvm.te.reduce_axis( (0, H), name="h" )

  defg = tvm.te.compute( (D, E, F, G),
                      lambda d, e, f, g: tvm.te.sum( FIHG[f, i, h, g] * DIEH[d, i, e, h], axis=[i, h] ),
                      name="defg" )

  return [FIHG, DIEH, defg]

# blocked_reordered 22. "cefd,aebf->abcd" "96,84,84,84,84,96" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_cefd_aebf_abcd( A, B, C, D, E, F, dtype ):
  CEFD = tvm.te.placeholder( (C, E, F, D), name="CEFD", dtype=dtype )
  AEBF = tvm.te.placeholder( (A, E, B, F), name="AEBF", dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )
  f = tvm.te.reduce_axis( (0, F), name="f" )

  abcd = tvm.te.compute( (A, B, C, D),
                      lambda a, b, c, d: tvm.te.sum( CEFD[c, e, f, d] * AEBF[a, e, b, f], axis=[e, f] ),
                      name="abcd" )

  return [CEFD, AEBF, abcd]

# blocked_reordered 23. "aefd,becf->abcd" "96,84,84,84,96,96" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_aefd_becf_abcd( A, B, C, D, E, F, dtype ):
  AEFD = tvm.te.placeholder( (A, E, F, D), name="AEFD", dtype=dtype )
  BECF = tvm.te.placeholder( (B, E, C, F), name="BECF", dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )
  f = tvm.te.reduce_axis( (0, F), name="f" )

  abcd = tvm.te.compute( (A, B, C, D),
                      lambda a, b, c, d: tvm.te.sum( AEFD[a, e, f, d] * BECF[b, e, c, f], axis=[e, f] ),
                      name="abcd" )

  return [AEFD, BECF, abcd]

# blocked_reordered 24. "cfed,afbe->abcd" "96,84,84,96,84,84" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_cfed_afbe_abcd( A, B, C, D, E, F, dtype ):
  CFED = tvm.te.placeholder( (C, F, E, D), name="CFED", dtype=dtype )
  AFBE = tvm.te.placeholder( (A, F, B, E), name="AFBE", dtype=dtype )

  f = tvm.te.reduce_axis( (0, F), name="f" )
  e = tvm.te.reduce_axis( (0, E), name="e" )

  abcd = tvm.te.compute( (A, B, C, D),
                      lambda a, b, c, d: tvm.te.sum( CFED[c, f, e, d] * AFBE[a, f, b, e], axis=[f, e] ),
                      name="abcd" )

  return [CFED, AFBE, abcd]

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

  # blocked_reordered 1. "abdfe,cf->abcde" "48,36,24,36,48,36" "(0,1)"
  print( "************************************" )
  print( "*** benchmark 1: abdfe,cf->abcde ***" )
  print( "************************************" )
  func = einsum_abdfe_cf_abcde
  sizes = (48, 36, 24, 36, 48, 36)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 2. "acdfe,bf->abcde" "48,24,36,36,48,36" "(0,1)"
  print( "************************************" )
  print( "*** benchmark 2: acdfe,bf->abcde ***" )
  print( "************************************" )
  func = einsum_acdfe_bf_abcde
  sizes = (48, 24, 36, 36, 48, 36)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 3. "abed,ce->abcd" "96,84,24,96,96" "(0,1)"
  print( "************************************" )
  print( "*** benchmark 3: abed,ce->abcd ***" )
  print( "************************************" )
  func = einsum_abed_ce_abcd
  sizes = (96, 84, 24, 96, 96)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 4. "abcfe,df->abcde" "48,36,36,24,48,48" "(0,1)"
  print( "************************************" )
  print( "*** benchmark 4: abcfe,df->abcde ***" )
  print( "************************************" )
  func = einsum_abcfe_df_abcde
  sizes = (48, 36, 36, 24, 48, 48)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 5. "aced,be->abcd" "96,24,84,96,84" "(0,1)"
  print( "************************************" )
  print( "*** benchmark 5: aced,be->abcd ***" )
  print( "************************************" )
  func = einsum_aced_be_abcd
  sizes = (96, 24, 84, 96, 84)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 6. "jki,efghjk->efghi" "6,64,6,64,24,6,64" "(0,1)"
  print( "**************************************" )
  print( "*** benchmark 6: jki,efghjk->efghi ***" )
  print( "**************************************" )
  func = einsum_jki_efghjk_efghi
  sizes = (6, 64, 6, 64, 24, 6, 64)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 7. "abed,ce->abcd" "96,84,24,84,96" "(0,1)"
  print( "***********************************" )
  print( "*** benchmark 7: abed,ce->abcd ***" )
  print( "***********************************" )
  func = einsum_abed_ce_abcd
  sizes = (96, 84, 24, 84, 96)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 8. "bcgf,adeg->abcdef" "24,20,20,24,20,20,24" "(0,1)"
  print( "**************************************" )
  print( "*** benchmark 8: bcgf,adeg->abcdef ***" )
  print( "**************************************" )
  func = einsum_bcgf_adeg_abcdef
  sizes = (24, 20, 20, 24, 20, 20, 24)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 9. "bdgf,aceg->abcdef" "24,20,20,24,20,20,24" "(0,1)"
  print( "**************************************" )
  print( "*** benchmark 9: bdgf,aceg->abcdef ***" )
  print( "**************************************" )
  func = einsum_bdgf_aceg_abcdef
  sizes = (24, 20, 20, 24, 20, 20, 24)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 10. "acgf,bdeg->abcdef" "24,20,20,24,20,20,24" "(0,1)"
  print( "***************************************" )
  print( "*** benchmark 10: acgf,bdeg->abcdef ***" )
  print( "***************************************" )
  func = einsum_acgf_bdeg_abcdef
  sizes = (24, 20, 20, 24, 20, 20, 24)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 11. "abgf,cdeg->abcdef" "24,20,20,24,20,20,24" "(0,1)"
  print( "***************************************" )
  print( "*** benchmark 11: abgf,cdeg->abcdef ***" )
  print( "***************************************" )
  func = einsum_abgf_cdeg_abcdef
  sizes = (24, 20, 20, 24, 20, 20, 24)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 12. "efhjki,gjk->efghi" "6,64,24,4,94,6,64" "(0,1)"
  print( "***************************************" )
  print( "*** benchmark 12: efhjki,gjk->efghi ***" )
  print( "***************************************" )
  func = einsum_efhjki_gjk_efghi
  sizes = (6, 64, 24, 4, 94, 6, 64)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 13. "bced,ae->abcd" "96,84,84,84,96" "(0,1)"
  print( "************************************" )
  print( "*** benchmark 13: bced,ae->abcd ***" )
  print( "************************************" )
  func = einsum_bced_ae_abcd
  sizes = (96, 84, 84, 84, 96)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 14. "aced,be->abcd" "96,84,84,84,96" "(0,1)"
  print( "************************************" )
  print( "*** benchmark 14: aced,be->abcd ***" )
  print( "************************************" )
  func = einsum_aced_be_abcd
  sizes = (96, 84, 84, 84, 96)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 15. "abed,ce->abcd" "96,84,84,84,96" "(0,1)"
  print( "***********************************" )
  print( "*** benchmark 15: abed,ce->abcd ***" )
  print( "***********************************" )
  func = einsum_abed_ce_abcd
  sizes = (96, 84, 84, 84, 96)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 16. "aedc,ebd->abc" "96,84,84,84,96" "(0,1)"
  print( "**********************************" )
  print( "*** benchmark 16: aedc,ebd->abc ***" )
  print( "**********************************" )
  func = einsum_aedc_ebd_abc
  sizes = (96, 84, 84, 84, 96)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 17. "gkiljh,ekilfj->efgh" "6,64,4,94,6,64,6,64" "(0,1)"
  print( "*****************************************" )
  print( "*** benchmark 17: gkiljh,ekilfj->efgh ***" )
  print( "*****************************************" )
  func = einsum_gkiljh_ekilfj_efgh
  sizes = (6, 64, 4, 94, 6, 64, 6, 64)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 18. "gikljh,eiklfj->efgh" "6,64,4,94,4,94,6,64" "(0,1)"
  print( "*****************************************" )
  print( "*** benchmark 18: gikljh,eiklfj->efgh ***" )
  print( "*****************************************" )
  func = einsum_gikljh_eiklfj_efgh
  sizes = (6, 64, 4, 94, 4, 94, 6, 64)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 19. "efiklj,ghkl->efghij" "6,64,4,94,4,94,6,64" "(0,1)"
  print( "*****************************************" )
  print( "*** benchmark 19: efiklj,ghkl->efghij ***" )
  print( "*****************************************" )
  func = einsum_efiklj_ghkl_efghij
  sizes = (6, 64, 4, 94, 4, 94, 6, 64)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 20. "efiklj,ghkl->efghij" "6,64,6,64,4,94,4,94" "(0,1)"
  print( "*****************************************" )
  print( "*** benchmark 20: efiklj,ghkl->efghij ***" )
  print( "*****************************************" )
  func = einsum_efiklj_ghkl_efghij
  sizes = (6, 64, 6, 64, 4, 94, 4, 94)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 21. "fihg,dieh->defg" "151,48,181,40,151,48" "(0,1)"
  print( "*************************************" )
  print( "*** benchmark 21: fihg,dieh->defg ***" )
  print( "*************************************" )
  func = einsum_fihg_dieh_defg
  sizes = (151, 48, 181, 40, 151, 48)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 22. "cefd,aebf->abcd" "96,84,84,84,84,96" "(0,1)"
  print( "*************************************" )
  print( "*** benchmark 22: cefd,aebf->abcd ***" )
  print( "*************************************" )
  func = einsum_cefd_aebf_abcd
  sizes = (96, 84, 84, 84, 84, 96)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 23. "aefd,becf->abcd" "96,84,84,84,96,96" "(0,1)"
  print( "*************************************" )
  print( "*** benchmark 23: aefd,becf->abcd ***" )
  print( "*************************************" )
  func = einsum_aefd_becf_abcd
  sizes = (96, 84, 84, 84, 96, 96)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )

  # blocked_reordered 24. "cfed,afbe->abcd" "96,84,84,96,84,84" "(0,1)"
  print( "*************************************" )
  print( "*** benchmark 24: cfed,afbe->abcd ***" )
  print( "*************************************" )
  func = einsum_cfed_afbe_abcd
  sizes = (96, 84, 84, 96, 84, 84)

  task = tvm.auto_scheduler.SearchTask( func = func,
                                        args = (*sizes, args.dtype),
                                        target = target )
  task.tune( tune_option )