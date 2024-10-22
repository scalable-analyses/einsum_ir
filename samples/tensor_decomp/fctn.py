##
# Approach: Fully-Connected Tensor Network Decomposition and Its Application to Higher-Order Tensor Completion
#           https://doi.org/10.1609/aaai.v35i12.17321
#
# Hyperspectral video dimensions: 60 x 60 x 20 x 20
#
# Rank: 8
#
# tensor network decomposition:
#
# I1: 60 (a)
# I2: 60 (b)
# I3: 20 (c)
# I4: 20 (d)
#
# R12 = R21 = 8 (e)
# R13 = R31 = 8 (f)
# R14 = R41 = 8 (g)
#
# R23 = R32 = 8 (h)
# R24 = R42 = 8 (i)
#
# R34 = R43 = 8 (j)
#
# Factor tensors:
#
# G1: 60 x 8 x 8 x 8 (aefg)
# G2: 60 x 8 x 8 x 8 (behi)
# G3: 20 x 8 x 8 x 8 (cfhj)
# G4: 20 x 8 x 8 x 8 (dgij)
#
# einsum string: "aefg,behi,cfhj,dgij->abcd"
##
import argparse
import torch
import opt_einsum

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( "--benchmark",
                         action = "store_true",
                         help = "benchmark opt_einsum" )
    args = parser.parse_args()

    # create factor tensors
    G1 = torch.randn(60, 8, 8, 8)
    G2 = torch.randn(60, 8, 8, 8)
    G3 = torch.randn(20, 8, 8, 8)
    G4 = torch.randn(20, 8, 8, 8)

    # print contraction path
    print( 'opt_einsum contraction path:' )
    print( opt_einsum.contract_path( "aefg,behi,cfhj,dgij->abcd",
                                     G1,
                                     G2,
                                     G3,
                                     G4,
                                     optimize = 'optimal' ) )

    if args.benchmark:
        print( 'compiling opt_einsum expression' )
        expr = opt_einsum.contract_expression( "aefg,behi,cfhj,dgij->abcd",
                                               G1.shape,
                                               G2.shape,
                                               G3.shape,
                                               G4.shape,
                                               optimize = 'optimal' )

        print( 'benchmarking opt_einsum expression' )
        # warm up
        for _ in range(10):
            result = expr( G1, G2, G3, G4, backend='torch' )

        # benchmark expression
        import time
        start = time.time()
        result = expr( G1, G2, G3, G4, backend='torch' )
        end = time.time()
        
        print( '  time: %.4f seconds' % (end - start) )