###
# Approach: High-Performance Deep Learning via a Single Building Block
#           https://doi.org/10.1109/IPDPS47924.2020.00032
#
# Config (Algorithm 5):
#
# M = m1*m2
# N = n1*n2
# K = k1*k2
#
# where m1, n1 and k1 are the blocking factors.
#
# einsum string: "[m2,k2,k1,m1],[n2,k2,n1,k1]->[n2,m2,n1,m1]"
###
import argparse

parser = argparse.ArgumentParser(description='Generate blocked linear layers.')
parser.add_argument('--M', type=int, default=2048, help='M dimension size')
parser.add_argument('--N', type=int, default=2048, help='N dimension size')
parser.add_argument('--K', type=int, default=2048, help='K dimension size')
parser.add_argument('--choices-m1', type=int, nargs='+', default=[16, 32, 64, 128, 256],            help='Choices for m1')
parser.add_argument('--choices-n1', type=int, nargs='+', default=[16, 32, 64, 128, 256],            help='Choices for n1')
parser.add_argument('--choices-k1', type=int, nargs='+', default=[16, 32, 64, 128, 256, 512, 1024], help='Choices for k1')
args = parser.parse_args()

str_expr = "[m2,k2,k1,m1],[n2,k2,n1,k1]->[n2,m2,n1,m1]"
str_cont_path = "(0,1)"

# generate all possible combinations of m1, n1 and k1
for m1 in args.choices_m1:
  for n1 in args.choices_n1:
    for k1 in args.choices_k1:
        m2 = args.M // m1
        n2 = args.N // n1
        k2 = args.K // k1
        str_dim_sizes = "{},{},{},{},{},{}".format( k1, k2, m1, m2, n1, n2 )

        print(f'"{str_expr}" "{str_dim_sizes}" "{str_cont_path}"')