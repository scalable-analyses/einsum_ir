##
# Generalizing Tensor Decomposition for N-ary Relational Knowledge Bases
# https://dl.acm.org/doi/10.1145/3366423.3380188
#
# Einsum string:
#   aib,bjc,ckd,dle,ema->ijklm
#   https://github.com/liuyuaa/GETD/blob/master/Nary%20code/model.py#L40
#
# Dimension sizes for JF17K-4:
# 
# python main.py --dataset JF17K-4 --num_iterations 200 --batch_size 128 --edim 25 --rdim 25 --k 5 --n_i 25 --TR_ranks 40 --dr 0.995 --lr 0.0006071265071591076
#
#   a: 40
#   b: 40
#   c: 40
#   d: 40
#   e: 40
#
#   i: 25
#   j: 25
#   k: 25
#   l: 25
#   m: 25
##
import torch
import opt_einsum

# create tensors
Z0 = torch.randn(40, 25, 40)
Z1 = torch.randn(40, 25, 40)
Z2 = torch.randn(40, 25, 40)
Z3 = torch.randn(40, 25, 40)
Z4 = torch.randn(40, 25, 40)

# print contraction path
print( 'opt_einsum contraction path:' )
print( opt_einsum.contract_path( "aib,bjc,ckd,dle,ema->ijklm",
                                 Z0,
                                 Z1,
                                 Z2,
                                 Z3,
                                 Z4,
                                 optimize = 'optimal' ) )