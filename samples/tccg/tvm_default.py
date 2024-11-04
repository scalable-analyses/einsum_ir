import tvm.te
import tvm.auto_scheduler
import tvm_helper

# default 1. "efbad,cf->abcde" "48,36,24,36,48,36" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_efbad_cf_abcde( A, B, C, D, E, F, dtype ):
  EFBAD = tvm.te.placeholder( (E, F, B, A, D), name="EFBAD", dtype=dtype )
  CF    = tvm.te.placeholder( (C, F),          name="CF",    dtype=dtype )

  f = tvm.te.reduce_axis( (0, F), name="f" )

  abcde = tvm.te.compute( (A, B, C, D, E),
                      lambda a, b, c, d, e: tvm.te.sum( EFBAD[e, f, b, a, d] * CF[c, f], axis=f ),
                      name="abcde" )

  return [EFBAD, CF, abcde]

# default 2. "efcad,bf->abcde" "48,24,36,36,48,36" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_efcad_bf_abcde( A, B, C, D, E, F, dtype ):
  EFCAD = tvm.te.placeholder( (E, F, C, A, D), name="EFCAD", dtype=dtype )
  BF    = tvm.te.placeholder( (B, F),          name="BF",    dtype=dtype )

  f = tvm.te.reduce_axis( (0, F), name="f" )

  abcde = tvm.te.compute( (A, B, C, D, E),
                      lambda a, b, c, d, e: tvm.te.sum( EFCAD[e, f, c, a, d] * BF[b, f], axis=f ),
                      name="abcde" )

  return [EFCAD, BF, abcde]

# default 3. "dbea,ec->abcd" "96,84,24,96,96" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_dbea_ec_abcd( A, B, C, D, E, dtype ):
  DBEA = tvm.te.placeholder( (D, B, E, A), name="DBEA", dtype=dtype )
  EC   = tvm.te.placeholder( (E, C),       name="EC",   dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )

  abcd = tvm.te.compute( (A, B, C, D),
                     lambda a, b, c, d: tvm.te.sum( DBEA[d, b, e, a] * EC[e, c], axis=e ),
                     name="abcd" )

  return [DBEA, EC, abcd]

# default 4. "ecbfa,fd->abcde" "48,36,36,24,48,48" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_ecbfa_fd_abcde( A, B, C, D, E, F, dtype ):
  ECBFA = tvm.te.placeholder( (E, C, B, F, A), name="ECBFA", dtype=dtype )
  FD    = tvm.te.placeholder( (F, D),          name="FD",    dtype=dtype )

  f = tvm.te.reduce_axis( (0, F), name="f" )

  abcde = tvm.te.compute( (A, B, C, D, E),
                      lambda a, b, c, d, e: tvm.te.sum( ECBFA[e, c, b, f, a] * FD[f, d], axis=f ),
                      name="abcde" )

  return [ECBFA, FD, abcde]

# default 5. "deca,be->abcd" "96,24,84,96,84" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_deca_be_abcd( A, B, C, D, E, dtype ):
  DECA = tvm.te.placeholder( (D, E, C, A), name="DECA", dtype=dtype )
  BE   = tvm.te.placeholder( (B, E),       name="BE",   dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )

  abcd = tvm.te.compute( (A, B, C, D),
                         lambda a, b, c, d: tvm.te.sum( DECA[d, e, c, a] * BE[b, e], axis=e ),
                         name="abcd" )

  return [DECA, BE, abcd]

# default 6. "bda,dc->abc" "384,384,24,384" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_bda_dc_abc( A, B, C, D, dtype ):
  BDA = tvm.te.placeholder( (B, D, A), name="BDA", dtype=dtype )
  DC  = tvm.te.placeholder( (D, C),   name="DC",  dtype=dtype )

  d = tvm.te.reduce_axis( (0, D), name="d" )

  abc = tvm.te.compute( (A, B, C),
                        lambda a, b, c: tvm.te.sum( BDA[b, d, a] * DC[d, c], axis=d ),
                        name="abc" )

  return [BDA, DC, abc]

# default 7. "ebad,ce->abcd" "96,84,24,84,96" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_ebad_ce_abcd( A, B, C, D, E, dtype ):
  EBAD = tvm.te.placeholder( (E, B, A, D), name="EBAD", dtype=dtype )
  CE   = tvm.te.placeholder( (C, E),       name="CE",   dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )

  abcd = tvm.te.compute( (A, B, C, D),
                         lambda a, b, c, d: tvm.te.sum( EBAD[e, b, a, d] * CE[c, e], axis=e ),
                         name="abcd" )

  return [EBAD, CE, abcd]

# default 8. "dega,gfbc->abcdef" "24,20,20,24,20,20,24" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_dega_gfbc_abcdef( A, B, C, D, E, F, G, dtype ):
  DEGA = tvm.te.placeholder( (D, E, G, A), name="DEGA", dtype=dtype )
  GFBC = tvm.te.placeholder( (G, F, B, C), name="GFBC", dtype=dtype )

  g = tvm.te.reduce_axis( (0, G), name="g" )

  abcdef = tvm.te.compute( (A, B, C, D, E, F),
                           lambda a, b, c, d, e, f: tvm.te.sum( DEGA[d, e, g, a] * GFBC[g, f, b, c], axis=g ),
                           name="abcdef" )

  return [DEGA, GFBC, abcdef]

# default 9. "dfgb,geac->abcdef" "24,20,20,24,20,20,24" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_dfgb_geac_abcdef( A, B, C, D, E, F, G, dtype ):
  DFGB = tvm.te.placeholder( (D, F, G, B), name="DFGB", dtype=dtype )
  GEAC = tvm.te.placeholder( (G, E, A, C), name="GEAC", dtype=dtype )

  g = tvm.te.reduce_axis( (0, G), name="g" )

  abcdef = tvm.te.compute( (A, B, C, D, E, F),
                           lambda a, b, c, d, e, f: tvm.te.sum( DFGB[d, f, g, b] * GEAC[g, e, a, c], axis=g ),
                           name="abcdef" )

  return [DFGB, GEAC, abcdef]

# default 10. "degb,gfac->abcdef" "24,20,20,24,20,20,24" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_degb_gfac_abcdef( A, B, C, D, E, F, G, dtype ):
  DEGB = tvm.te.placeholder( (D, E, G, B), name="DEGB", dtype=dtype )
  GFAC = tvm.te.placeholder( (G, F, A, C), name="GFAC", dtype=dtype )

  g = tvm.te.reduce_axis( (0, G), name="g" )

  abcdef = tvm.te.compute( (A, B, C, D, E, F),
                           lambda a, b, c, d, e, f: tvm.te.sum( DEGB[d, e, g, b] * GFAC[g, f, a, c], axis=g ),
                           name="abcdef" )

  return [DEGB, GFAC, abcdef]

# default 11. "degc,gfab->abcdef" "24,20,20,24,20,20,24" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_degc_gfab_abcdef( A, B, C, D, E, F, G, dtype ):
  DEGC = tvm.te.placeholder( (D, E, G, C), name="DEGC", dtype=dtype )
  GFAB = tvm.te.placeholder( (G, F, A, B), name="GFAB", dtype=dtype )

  g = tvm.te.reduce_axis( (0, G), name="g" )

  abcdef = tvm.te.compute( (A, B, C, D, E, F),
                           lambda a, b, c, d, e, f: tvm.te.sum( DEGC[d, e, g, c] * GFAB[g, f, a, b], axis=g ),
                           name="abcdef" )

  return [DEGC, GFAB, abcdef]

# default 12. "dca,bd->abc" "384,24,376,384" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_dca_bd_abc( A, B, C, D, dtype ):
  DCA = tvm.te.placeholder( (D, C, A), name="DCA", dtype=dtype )
  BD  = tvm.te.placeholder( (B, D),   name="BD",  dtype=dtype )

  d = tvm.te.reduce_axis( (0, D), name="d" )

  abc = tvm.te.compute( (A, B, C),
                        lambda a, b, c: tvm.te.sum( DCA[d, c, a] * BD[b, d], axis=d ),
                        name="abc" )

  return [DCA, BD, abc]

# default 13. "ea,ebcd->abcd" "96,84,84,84,96" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_ea_ebcd_abcd( A, B, C, D, E, dtype ):
  EA   = tvm.te.placeholder( (E, A), name="EA",   dtype=dtype )
  EBCD = tvm.te.placeholder( (E, B, C, D), name="EBCD", dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )

  abcd = tvm.te.compute( (A, B, C, D),
                         lambda a, b, c, d: tvm.te.sum( EA[e, a] * EBCD[e, b, c, d], axis=e ),
                         name="abcd" )

  return [EA, EBCD, abcd]

# default 14. "eb,aecd->abcd" "96,84,84,84,96" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_eb_aecd_abcd( A, B, C, D, E, dtype ):
  EB   = tvm.te.placeholder( (E, B), name="EB",   dtype=dtype )
  AECD = tvm.te.placeholder( (A, E, C, D), name="AECD", dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )

  abcd = tvm.te.compute( (A, B, C, D),
                         lambda a, b, c, d: tvm.te.sum( EB[e, b] * AECD[a, e, c, d], axis=e ),
                         name="abcd" )

  return [EB, AECD, abcd]

# default 15. "ec,abed->abcd" "96,84,84,84,96" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_ec_abed_abcd( A, B, C, D, E, dtype ):
  EC   = tvm.te.placeholder( (E, C), name="EC",   dtype=dtype )
  ABED = tvm.te.placeholder( (A, B, E, D), name="ABED", dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )

  abcd = tvm.te.compute( (A, B, C, D),
                         lambda a, b, c, d: tvm.te.sum( EC[e, c] * ABED[a, b, e, d], axis=e ),
                         name="abcd" )

  return [EC, ABED, abcd]

# default 16. "adec,ebd->abc" "96,84,84,84,96" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_adec_ebd_abc( A, B, C, D, E, dtype ):
  ADEC = tvm.te.placeholder( (A, D, E, C), name="ADEC", dtype=dtype )
  EBD  = tvm.te.placeholder( (E, B, D),   name="EBD",  dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )
  d = tvm.te.reduce_axis( (0, D), name="d" )

  abc = tvm.te.compute( (A, B, C),
                        lambda a, b, c: tvm.te.sum( ADEC[a, d, e, c] * EBD[e, b, d], axis=[e, d] ),
                        name="abc" )

  return [ADEC, EBD, abc]

# default 17. "cad,dcb->ab" "384,376,384,384" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_cad_dcb_ab( A, B, C, D, dtype ):
  CAD = tvm.te.placeholder( (C, A, D), name="CAD", dtype=dtype )
  DCB = tvm.te.placeholder( (D, C, B), name="DCB", dtype=dtype )

  d = tvm.te.reduce_axis( (0, D), name="d" )
  c = tvm.te.reduce_axis( (0, C), name="c" )

  ab = tvm.te.compute( (A, B),
                       lambda a, b: tvm.te.sum( CAD[c, a, d] * DCB[d, c, b], axis=[c, d] ),
                       name="ab" )

  return [CAD, DCB, ab]

# default 18. "acd,dbc->ab" "384,376,376,384" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_acd_dbc_ab( A, B, C, D, dtype ):
  ACD = tvm.te.placeholder( (A, C, D), name="ACD", dtype=dtype )
  DBC = tvm.te.placeholder( (D, B, C), name="DBC", dtype=dtype )

  d = tvm.te.reduce_axis( (0, D), name="d" )
  c = tvm.te.reduce_axis( (0, C), name="c" )

  ab = tvm.te.compute( (A, B),
                       lambda a, b: tvm.te.sum( ACD[a, c, d] * DBC[d, b, c], axis=[c, d] ),
                       name="ab" )

  return [ACD, DBC, ab]

# default 19. "acd,db->abc" "384,376,376,384" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_acd_db_abc( A, B, C, D, dtype ):
  ACD = tvm.te.placeholder( (A, C, D), name="ACD", dtype=dtype )
  DB  = tvm.te.placeholder( (D, B),   name="DB",  dtype=dtype )

  d = tvm.te.reduce_axis( (0, D), name="d" )

  abc = tvm.te.compute( (A, B, C),
                        lambda a, b, c: tvm.te.sum( ACD[a, c, d] * DB[d, b], axis=d ),
                        name="abc" )

  return [ACD, DB, abc]

# default 20. "adc,bd->abc" "384,384,376,376" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_adc_bd_abc( A, B, C, D, dtype ):
  ADC = tvm.te.placeholder( (A, D, C), name="ADC", dtype=dtype )
  BD  = tvm.te.placeholder( (B, D),   name="BD",  dtype=dtype )

  d = tvm.te.reduce_axis( (0, D), name="d" )

  abc = tvm.te.compute( (A, B, C),
                        lambda a, b, c: tvm.te.sum( ADC[a, d, c] * BD[b, d], axis=d ),
                        name="abc" )

  return [ADC, BD, abc]

# default 21. "ac,cb->ab" "7248,7240,7248" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_ac_cb_ab( A, B, C, dtype ):
  AC = tvm.te.placeholder( (A, C), name="AC", dtype=dtype )
  CB = tvm.te.placeholder( (C, B), name="CB", dtype=dtype )

  c = tvm.te.reduce_axis( (0, C), name="c" )

  ab = tvm.te.compute( (A, B),
                       lambda a, b: tvm.te.sum( AC[a, c] * CB[c, b], axis=c ),
                       name="ab" )

  return [AC, CB, ab]

# default 22. "aebf,fdec->abcd" "96,84,84,84,84,96" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_aebf_fdec_abcd( A, B, C, D, E, F, dtype ):
  AEBF = tvm.te.placeholder( (A, E, B, F), name="AEBF", dtype=dtype )
  FDEC = tvm.te.placeholder( (F, D, E, C), name="FDEC", dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )
  f = tvm.te.reduce_axis( (0, F), name="f" )

  abcd = tvm.te.compute( (A, B, C, D),
                         lambda a, b, c, d: tvm.te.sum( AEBF[a, e, b, f] * FDEC[f, d, e, c], axis=[e, f] ),
                         name="abcd" )

  return [AEBF, FDEC, abcd]

# default 23. "eafd,fbec->abcd" "96,84,84,84,96,96" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_eafd_fbec_abcd( A, B, C, D, E, F, dtype ):
  EAFD = tvm.te.placeholder( (E, A, F, D), name="EAFD", dtype=dtype )
  FBEC = tvm.te.placeholder( (F, B, E, C), name="FBEC", dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )
  f = tvm.te.reduce_axis( (0, F), name="f" )

  abcd = tvm.te.compute( (A, B, C, D),
                         lambda a, b, c, d: tvm.te.sum( EAFD[e, a, f, d] * FBEC[f, b, e, c], axis=[e, f] ),
                         name="abcd" )

  return [EAFD, FBEC, abcd]

# 24. "aebf,dfce->abcd" "96,84,84,96,84,84" "(0,1)"
@tvm.auto_scheduler.register_workload
def einsum_aebf_dfce_abcd( A, B, C, D, E, F, dtype ):
  AEBF = tvm.te.placeholder( (A, E, B, F), name="AEBF", dtype=dtype )
  DFCE = tvm.te.placeholder( (D, F, C, E), name="DFCE", dtype=dtype )

  e = tvm.te.reduce_axis( (0, E), name="e" )
  f = tvm.te.reduce_axis( (0, F), name="f" )

  abcd = tvm.te.compute( (A, B, C, D),
                         lambda a, b, c, d: tvm.te.sum( AEBF[a, e, b, f] * DFCE[d, f, c, e], axis=[e, f] ),
                         name="abcd" )

  return [AEBF, DFCE, abcd]

if __name__=="__main__":
    args = tvm_helper.parse_args()
    target = tvm.target.Target(tvm_helper.cpu_to_llvm(args.cpu))
    hardware_params = tvm.auto_scheduler.HardwareParams(target=target)

    # Define benchmarks
    benchmarks = [
        ("efbad,cf->abcde", einsum_efbad_cf_abcde, (48, 36, 24, 36, 48, 36)),
        ("efcad,bf->abcde", einsum_efcad_bf_abcde, (48, 24, 36, 36, 48, 36)),
        ("dbea,ec->abcd", einsum_dbea_ec_abcd, (96, 84, 24, 96, 96)),
        ("ecbfa,fd->abcde", einsum_ecbfa_fd_abcde, (48, 36, 36, 24, 48, 48)),
        ("deca,be->abcd", einsum_deca_be_abcd, (96, 24, 84, 96, 84)),
        ("bda,dc->abc", einsum_bda_dc_abc, (384, 384, 24, 384)),
        ("ebad,ce->abcd", einsum_ebad_ce_abcd, (96, 84, 24, 84, 96)),
        ("dega,gfbc->abcdef", einsum_dega_gfbc_abcdef, (24, 20, 20, 24, 20, 20, 24)),
        ("dfgb,geac->abcdef", einsum_dfgb_geac_abcdef, (24, 20, 20, 24, 20, 20, 24)),
        ("degb,gfac->abcdef", einsum_degb_gfac_abcdef, (24, 20, 20, 24, 20, 20, 24)),
        ("degc,gfab->abcdef", einsum_degc_gfab_abcdef, (24, 20, 20, 24, 20, 20, 24)),
        ("dca,bd->abc", einsum_dca_bd_abc, (384, 24, 376, 384)),
        ("ea,ebcd->abcd", einsum_ea_ebcd_abcd, (96, 84, 84, 84, 96)),
        ("eb,aecd->abcd", einsum_eb_aecd_abcd, (96, 84, 84, 84, 96)),
        ("ec,abed->abcd", einsum_ec_abed_abcd, (96, 84, 84, 84, 96)),
        ("adec,ebd->abc", einsum_adec_ebd_abc, (96, 84, 84, 84, 96)),
        ("cad,dcb->ab", einsum_cad_dcb_ab, (384, 376, 384, 384)),
        ("acd,dbc->ab", einsum_acd_dbc_ab, (384, 376, 376, 384)),
        ("acd,db->abc", einsum_acd_db_abc, (384, 376, 376, 384)),
        ("adc,bd->abc", einsum_adc_bd_abc, (384, 384, 376, 376)),
        ("ac,cb->ab", einsum_ac_cb_ab, (7248, 7240, 7248)),
        ("aebf,fdec->abcd", einsum_aebf_fdec_abcd, (96, 84, 84, 84, 84, 96)),
        ("eafd,fbec->abcd", einsum_eafd_fbec_abcd, (96, 84, 84, 84, 96, 96)),
        ("aebf,dfce->abcd", einsum_aebf_dfce_abcd, (96, 84, 84, 96, 84, 84))
    ]

    for i, (einsum_str, func, sizes) in enumerate(benchmarks, 1):
        middle_line = f"*** benchmark {i}: {einsum_str} ***"
        print('*' * len(middle_line))
        print(middle_line)
        print('*' * len(middle_line))
        tvm_helper.run_all(einsum_str,
                          func, 
                          sizes,
                          args.dtype,
                          hardware_params,
                          target,
                          args.num_measure_trials,
                          args.timeout,
                          args.log_file)