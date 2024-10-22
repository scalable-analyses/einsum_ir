###
# Approach: Wide Compression: Tensor Ring Nets
#           https://doi.org/10.1109/CVPR.2018.00972
#
# Config (Table 1 and Table 2):
#
#   Rank R in tensor ring decomposition: [3,5,15,50]
#
#   fc1: (4 x 7 x 4 x 7) x (3 x 4 x 5 x 5)
#   fc2 (not shown): (3 x 4 x 5 x 5) x (4 x 5 x 5)
#   fc3 (not shown): (4 x 5 x 5) x (2 x 5)
#
# fc1 dimensions (see Figure 4):
#   I1: 4 (a)
#   I2: 7 (b)
#   I3: 4 (c)
#   I4: 7 (d)
#
#   O1: 3 (e)
#   O2: 4 (f)
#   O3: 5 (g)
#   O4: 5 (h)
#
#   R12 = R21 = R (i)
#   R15 = R51 = R (j)
#   R23 = R32 = R (k)
#   R34 = R43 = R (l)
#   R48 = R84 = R (m)
#
#   R87 = R78 = R (n)
#   R76 = R67 = R (o)
#   R56 = R65 = R (p)
#
# fc1 tensors:
#
#   U1: 4 x R X R (aij)
#   U2: 7 x R X R (bik)
#   U3: 4 x R X R (ckl)
#   U4: 7 x R X R (dlm)
#
#   U5: 3 x R X R (ejp)
#   U6: 4 x R X R (fop)
#   U7: 5 x R X R (gno)
#   U8: 5 x R X R (hmn)
#
# fc1 einsum string: "aij,bik,ckl,dlm,ejp,fop,gno,hmn->efgh"
##
import torch
import opt_einsum

# batch size
B = 128

# rank
R = 50

# create tensors
I = torch.randn(B, 4, 7, 4, 7)

U1 = torch.randn(4, R, R)
U2 = torch.randn(7, R, R)
U3 = torch.randn(4, R, R)
U4 = torch.randn(7, R, R)

U5 = torch.randn(3, R, R)
U6 = torch.randn(4, R, R)
U7 = torch.randn(5, R, R)
U8 = torch.randn(5, R, R)

num_params_original = 4*7*4*7 * 3*4*5*5
num_params_compressed = 4*R*R + 7*R*R + 4*R*R + 7*R*R + 3*R*R + 4*R*R + 5*R*R + 5*R*R

print( f"original number of parameters: {num_params_original}" )
print( f"compressed number of parameters: {num_params_compressed}" )
print( f"compression ratio: {num_params_original / num_params_compressed :.2f}" )

# determine contraction order
einsum_string = "Babcd,aij,bik,ckl,dlm,ejp,fop,gno,hmn->Befgh"

print( 'opt_einsum contraction path:' )
print( opt_einsum.contract_path( einsum_string,
                                 I,
                                 U1,
                                 U2,
                                 U3,
                                 U4,
                                 U5,
                                 U6,
                                 U7,
                                 U8 ) )