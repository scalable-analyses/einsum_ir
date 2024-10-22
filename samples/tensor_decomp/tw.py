##
# Tensor Wheel Decomposition and Its Tensor Completion Application
# https://dl.acm.org/doi/10.5555/3600270.3602228
#
# Section 3.4.2 Real-World Data Completion (Video Data), Hyperspectral Video (HSV)
#
# Video Dimensions: 40 x 40 x 20 x 20
#
# I1: 40 (a)
# I2: 40 (b)
# I3: 20 (c)
# I4: 20 (d)
#
# Wheel config:
#   R1 = R2 = R3 = R4 = 6
#    e    f    g    h
#
#   L1 = L2 = L3 = L4 = 4
#    i    j    k    l
#
# Tensors:
#
# G1: I1 x R1 x R2 x L1
#      a    e    f    i
#
# G2: I2 x R2 x R3 x L2
#      b    f    g    j
#
# G3: I3 x R3 x R4 x L3
#      c    g    h    k
#
# G4: I4 x R4 x R1 x L4
#      d    h    e    l
#
#  C: L1 x L2 x L3 x L4
#      i    j    k    l
#
# einsum string: "aefi,bfgj,cghk,dhel,ijkl->abcd"
#
##
import torch
import opt_einsum

# create tensors
G1 = torch.randn(40, 6, 6, 4)
G2 = torch.randn(40, 6, 6, 4)
G3 = torch.randn(20, 6, 6, 4)
G4 = torch.randn(20, 6, 6, 4)

C = torch.randn(4, 4, 4, 4)

einsum_string = "aefi,bfgj,cghk,dhel,ijkl->abcd"

# print contraction path
print( 'opt_einsum contraction path:' )
print( opt_einsum.contract_path( einsum_string,
                                 G1,
                                 G2,
                                 G3,
                                 G4,
                                 C,
                                 optimize = 'optimal' ) )