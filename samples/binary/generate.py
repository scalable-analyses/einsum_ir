import random
import argparse

def prime_factors(n):
  i = 2
  factors = []
  while i * i <= n:
    if n % i:
      i += 1
    else:
      n //= i
      factors.append(i)
  if n > 1:
    factors.append(n)
  return factors

def split_dim( size,
               num_dims ):
  if num_dims == 0 and size > 1:
    return None
  elif num_dims == 0 and size == 1:
    return []

  factors = prime_factors( size )

  if len( factors ) < num_dims:
    return None

  # shuffle factors
  random.shuffle( factors )

  # init output dimensions
  out_dims = []
  for di in range( num_dims ):
    factor = factors.pop()
    out_dims.append( factor )

  # distribute the remaining factors randomly
  for factor in factors:
    di = random.randint( 0, num_dims - 1 )
    out_dims[di] *= factor

  return out_dims


def binary_from_bgemm( size_c,
                       size_m,
                       size_n,
                       size_k,
                       num_dims_c,
                       num_dims_m,
                       num_dims_n,
                       num_dims_k ):
  split_c = split_dim( size_c,
                       num_dims_c )
  split_m = split_dim( size_m,
                       num_dims_m )
  split_n = split_dim( size_n,
                       num_dims_n )
  split_k = split_dim( size_k,
                       num_dims_k )
  
  if split_c is None or split_m is None or split_n is None or split_k is None:
    return None, None, None, None

  # number of dims in the three tensors
  num_dims_left   = num_dims_c + num_dims_m + num_dims_k
  num_dims_right  = num_dims_c + num_dims_n + num_dims_k
  num_dims_result = num_dims_c + num_dims_m + num_dims_n

  # possible places to put the dimensions
  positions_left   = list( range( num_dims_left ) )
  positions_right  = list( range( num_dims_right ) )
  positions_result = list( range( num_dims_result ) )

  # shuffle the positions
  random.shuffle( positions_left )
  random.shuffle( positions_right )
  random.shuffle( positions_result )

  # assign the dimension ids to the positions
  dim_id = 0

  tensor_left  = [ -1 for _ in range( num_dims_left ) ]
  tensor_right = [ -1 for _ in range( num_dims_right ) ]
  tensor_out   = [ -1 for _ in range( num_dims_result ) ]
  dim_sizes    = []

  for _ in range( num_dims_c ):
    dim_size = split_c.pop()

    tensor_left[positions_left.pop()]   = dim_id
    tensor_right[positions_right.pop()] = dim_id
    tensor_out[positions_result.pop()]  = dim_id
    dim_sizes.append( dim_size )
    dim_id += 1

  for _ in range( num_dims_m ):
    dim_size = split_m.pop()

    tensor_left[positions_left.pop()]   = dim_id
    tensor_out[positions_result.pop()]  = dim_id
    dim_sizes.append( dim_size )
    dim_id += 1

  for _ in range( num_dims_n ):
    dim_size = split_n.pop()

    tensor_right[positions_right.pop()] = dim_id
    tensor_out[positions_result.pop()]  = dim_id
    dim_sizes.append( dim_size )
    dim_id += 1

  for _ in range( num_dims_k ):
    dim_size = split_k.pop()

    tensor_left[positions_left.pop()]   = dim_id
    tensor_right[positions_right.pop()] = dim_id
    dim_sizes.append( dim_size )
    dim_id += 1

  return tensor_left, tensor_right, tensor_out, dim_sizes

if __name__ == "__main__":
  # parse arguments using argparse
  parser = argparse.ArgumentParser()
  parser.add_argument( "--sizes_c",        nargs='+', type=int, default=[ 1, 8, 16, 24, 32 ] )
  parser.add_argument( "--sizes_m",        nargs='+', type=int, default=[ 16, 24, 32, 36, 48, 64, 84, 96, 128, 256, 512, 376, 384, 1024, 2048, 4096 ] )
  parser.add_argument( "--sizes_n",        nargs='+', type=int, default=[ 16, 24, 32, 36, 48, 64, 84, 96, 128, 256, 512, 376, 384, 1024, 2048, 4096 ] )
  parser.add_argument( "--sizes_k",        nargs='+', type=int, default=[ 16, 24, 32, 36, 48, 64, 84, 96, 128, 256, 512, 376, 384, 1024, 2048, 4096 ] )
  parser.add_argument( "--num_dims_c",     nargs='+', type=int, default=[ 0, 1, 2 ] )
  parser.add_argument( "--num_dims_m",     nargs='+', type=int, default=[ 0, 1, 2 ] )
  parser.add_argument( "--num_dims_n",     nargs='+', type=int, default=[ 0, 1, 2 ] )
  parser.add_argument( "--num_dims_k",     nargs='+', type=int, default=[ 0, 1, 2 ] )
  parser.add_argument( "--max_size_left",             type=int, default=2048**2 )
  parser.add_argument( "--max_size_right",            type=int, default=2048**2 )
  parser.add_argument( "--max_size_out",              type=int, default=2048**2 )
  parser.add_argument( "--num_samples",               type=int, default=1 )
  parser.add_argument( "--num_attempts",              type=int, default=100000 )
  parser.add_argument( "--out_format",                type=str, default="einsum_ir" )
  parser.add_argument( "--print_num_ops",             type=bool, default=True )
  parser.add_argument( "--seed",                      type=int, default=None )
  args = parser.parse_args()

  sizes_c_opt = args.sizes_c
  sizes_m_opt = args.sizes_m
  sizes_n_opt = args.sizes_n
  sizes_k_opt = args.sizes_k

  num_dims_c_opt = args.num_dims_c
  num_dims_m_opt = args.num_dims_m
  num_dims_n_opt = args.num_dims_n
  num_dims_k_opt = args.num_dims_k

  max_size_left  = args.max_size_left
  max_size_right = args.max_size_right
  max_size_out   = args.max_size_out

  out_format = args.out_format
  seed = args.seed
  
  # set seed
  if seed is not None:
    random.seed( seed )

  for _ in range( args.num_samples ):
    num_attempts = args.num_attempts

    # generate binary contraction
    while( num_attempts > 0 ):
      num_attempts -= 1

      # pick random BGEMM sizes
      size_c = random.choice( sizes_c_opt )
      size_m = random.choice( sizes_m_opt )
      size_n = random.choice( sizes_n_opt )
      size_k = random.choice( sizes_k_opt )

      # pick random number of dimensions
      num_dims_c = random.choice( num_dims_c_opt )
      num_dims_m = random.choice( num_dims_m_opt )
      num_dims_n = random.choice( num_dims_n_opt )
      num_dims_k = random.choice( num_dims_k_opt )

      # determine sizes of the tensors
      size_left  = size_c * size_m * size_k
      size_right = size_c * size_n * size_k
      size_out   = size_c * size_m * size_n

      # reject if the sizes are too large
      if size_left > max_size_left or size_right > max_size_right or size_out > max_size_out:
        continue


      tensor_left, tensor_right, tensor_out, dim_sizes = binary_from_bgemm( size_c,
                                                                            size_m,
                                                                            size_n,
                                                                            size_k,
                                                                            num_dims_c,
                                                                            num_dims_m,
                                                                            num_dims_n,
                                                                            num_dims_k )

      if tensor_left is not None:
        break

    # determine number of operations
    num_ops = 2 * size_c * size_m * size_n * size_k
    num_ops -= size_c * size_m * size_n

    # format to string
    if out_format == "standard":
      tensor_left_str  = ",".join( [ str(di) for di in tensor_left ] )
      tensor_right_str = ",".join( [ str(di) for di in tensor_right ] )
      tensor_out_str   = ",".join( [ str(di) for di in tensor_out ] )
      dim_sizes_str    = ",".join( [ str(di) for di in dim_sizes ] )

      contraction_str = f"[[{tensor_left_str}],[{tensor_right_str}]]->[{tensor_out_str}] [{dim_sizes_str}]"
    elif out_format == "einsum_ir":
      tensor_left_str  = "".join( [ chr(97+di) for di in tensor_left ] )
      tensor_right_str = "".join( [ chr(97+di) for di in tensor_right ] )
      tensor_out_str   = "".join( [ chr(97+di) for di in tensor_out ] )
      dim_sizes_str    = ",".join( [ str(di) for di in dim_sizes ] )

      contraction_str = f"\"{tensor_left_str},{tensor_right_str}->{tensor_out_str}\" \"{dim_sizes_str}\" \"(0,1)\""

    # print contraction
    print( contraction_str, num_ops )