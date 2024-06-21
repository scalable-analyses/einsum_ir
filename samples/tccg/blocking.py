import functools
import sys

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

if __name__ == "__main__":
  if len(sys.argv) != 4:
    print( "Usage: python blocking.py <einsum_string> <dim_sizes> <cont_path>" )
    sys.exit(1)
  einsum_string = sys.argv[1]
  dim_sizes = sys.argv[2]
  cont_path = sys.argv[3]

  einsum_string = einsum_string.replace("\"", "")
  dim_sizes = dim_sizes.replace("\"", "")
  cont_path = cont_path.replace("\"", "")

  factor_lower_bound = 32
  factor_upper_bound = 96

  # parse all dim chars from the einsum string and sort alphabetically
  dim_chars = sorted(set(einsum_string) - set('->') - set(','))

  # assemble dictionary with dimension sizes
  dim_sizes = dict(zip(dim_chars, map(int, dim_sizes.split(','))))

  # iterate over dim chars and perform blocking
  for dim_char in dim_chars:
    dim_size = dim_sizes[dim_char]

    # prime factorization
    factors = prime_factors(dim_size)
    num_rots = len(factors)

    factors_tmp = factors.copy()

    while( True ):
      # combine prime factors until product reached lower bound
      product_right = 1
      while product_right < factor_lower_bound and len(factors_tmp) > 0:
        product_right *= factors_tmp.pop(0)

      num_rots -= 1
      if( num_rots == 0 ):
        break

      if( product_right > factor_upper_bound ):
        # rotate factors list
        factors.append( factors.pop(0) )
        factors_tmp = factors.copy()
      else:
        # combine prime factors until product reached upper bound
        # do not go over the upper bound
        while product_right < factor_upper_bound and len(factors_tmp) > 0:
          if( product_right * factors_tmp[0] > factor_upper_bound ):
            break
          product_right *= factors_tmp.pop(0)
        break
    
    assert( dim_size % product_right == 0 )
    product_left = dim_size // product_right

    if product_left != 1:
      dim_sizes[dim_char] = (product_left, product_right)


  # derive number of required chars in new einsum string
  num_new_chars = sum(2 if isinstance(size, tuple) else 1 for size in dim_sizes.values())

  # find new unique chars
  new_dim_chars = []
  for i in range(26):
    if chr(97+i) not in dim_sizes:
      new_dim_chars.append(chr(97+i))
    if len(new_dim_chars) == num_new_chars:
      break

  # assemble new einsum string
  new_einsum_string = einsum_string
  for dim_char, size in dim_sizes.items():
    if isinstance(size, tuple):
      new_dims = ''.join(new_dim_chars.pop(0) for _ in range(2) )
      new_einsum_string = new_einsum_string.replace(dim_char, new_dims)
    else:
      new_einsum_string = new_einsum_string.replace(dim_char, new_dim_chars.pop(0))

  # assemble new dimension sizes
  new_dim_sizes = ','.join(str(size) if not isinstance(size, tuple) else ','.join(map(str, size)) for size in dim_sizes.values())
  print( "\""+new_einsum_string+"\"", "\""+new_dim_sizes+"\"", "\""+cont_path+"\"" )