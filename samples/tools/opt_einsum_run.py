import argparse
import torch
import opt_einsum
import time

def normalize_einsum_expression(expression):
    input_subscripts, output_subscript = expression.split('->')
    input_subscripts = input_subscripts.strip('[]')
    output_subscript = output_subscript.strip('[]')
    input_subscripts = input_subscripts.split('],[')

    input_subscripts = [subscript.split(',') for subscript in input_subscripts]
    output_subscript = output_subscript.split(',')


    # try to convert numeric ids to integers
    try:
        input_subscripts = [[int(id) for id in subscript] for subscript in input_subscripts]
        output_subscript = [int(id) for id in output_subscript]
    except:
        pass

    # determine unique ids and sort them
    unique_ids = set()
    for subscript in input_subscripts:
        unique_ids.update(subscript)
    unique_ids.update(output_subscript)
    unique_ids = sorted(unique_ids)

    # create a mapping from numeric ids to alphabetic ids
    if( len(unique_ids) <= 52 ):
        id_map = {id: chr(65 + id) if id < 26 else chr(71 + id) for id in unique_ids}
    else:
        id_map = {id: chr(256 + id) for id in unique_ids}

    # assemble string in "normalized" form
    input_subscripts = [[id_map[id] for id in subscript] for subscript in input_subscripts]
    output_subscript = [id_map[id] for id in output_subscript]

    input_subscripts = [''.join(subscript) for subscript in input_subscripts]
    input_subscripts = ','.join(input_subscripts)

    output_subscript = ''.join(output_subscript)

    expr = input_subscripts + '->' + output_subscript
    
    return expr

def parse_einsum_expression(expression):
    input_subscripts, output_subscript = expression.split('->')
    input_subscripts = input_subscripts.split(',')

    input_subscripts = [list(subscript) for subscript in input_subscripts]

    return input_subscripts, list(output_subscript)

def parse_dim_sizes(expression, dim_sizes):
    chars = ''.join([c for c in expression if c not in ', ->'])
    unique_chars = sorted(set(chars))
    dim_size_map = {char: int(size) for char, size in zip(unique_chars, dim_sizes.split(','))}

    return dim_size_map

def parse_cont_path(cont_path):
    cont_path = cont_path.strip('()').split('),(')
    cont_path = [tuple(map(int, path.split(','))) for path in cont_path]

    return cont_path

def create_tensor( tensor_subscripts,
                   dim_size_map,
                   dtype ):
    shape = [dim_size_map[char] for char in tensor_subscripts]
    tensor = torch.randn( shape, dtype = dtype )

    return tensor

class DummyOptimizer( opt_einsum.paths.PathOptimizer ):
    def __init__( self, cont_path ):
        self.cont_path = cont_path

    def __call__(self, inputs, output, size_dict, memory_limit=None):
        # pylint: disable=unused-argument
        return self.cont_path

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", type=str, help="einsum expression (example: aefi,bfgj,cghk,dhel,ijkl->abcd)", required=True)
    parser.add_argument("--dim_sizes", type=str, help="comma-separated dimension sizes (example: 40,40,20,20,6,6,6,6,4,4,4,4)", required=True)
    parser.add_argument("--cont_path", type=str, help="contraction path (example: (2,3),(2,3),(0,2),(0,1))", required=True)
    parser.add_argument("--backend", type=str, help="backend to use (example: torch)", default='torch')
    parser.add_argument("--dtype", type=str, help="data type FP32 or FP64 (example: FP32)", default='FP32')

    args = parser.parse_args()

    expression = args.expression
    dim_sizes = args.dim_sizes
    cont_path = args.cont_path
    backend = args.backend
    dtype = args.dtype

    print( '*** input parameters ***' )
    print( '  einsum_str:', expression )
    print( '  dim_sizes: ', dim_sizes )
    print( '  cont_path: ', cont_path )
    print( '  backend:   ', backend )
    print( '  dtype:     ', dtype )

    # normalize expression if in array format
    if expression[0] == '[':
        expression = normalize_einsum_expression(expression)
        print( '  normalized einsum_str:', expression )

    input_subscripts, output_subscript = parse_einsum_expression(expression)

    dim_size_map = parse_dim_sizes(expression, dim_sizes)

    cont_path = parse_cont_path(cont_path)

    if dtype == 'FP32':
        dtype = torch.float32
    elif dtype == 'FP64':
        dtype = torch.float64


    # create the input tensors
    tensors = []
    for tensor_subscripts in input_subscripts:
        tensor = create_tensor( tensor_subscripts,
                                dim_size_map,
                                dtype)
        tensors.append( tensor )

    tensor_shapes = [tensor.shape for tensor in tensors]

    # compile the einsum expression
    dummy_optimizer = DummyOptimizer( cont_path )
    start = time.time()
    expr = opt_einsum.contract_expression( expression,
                                           *tensor_shapes,
                                           optimize = dummy_optimizer )
    end = time.time()
    time_compile = end - start

    print( '*** benchmarking opt_einsum expression ***' )

    _ = expr( *tensors, backend = backend )

    # benchmark expression
    start = time.time()
    _ = expr( *tensors, backend = backend )
    end = time.time()
    time_eval = end - start

    _, path_info = opt_einsum.contract_path( expression,
                                             *tensor_shapes,
                                             shapes=True,
                                             optimize = dummy_optimizer)
    num_ops = int(path_info.opt_cost)

    gflops_eval = num_ops / time_eval
    gflops_eval *= 1e-9

    gflops_total = num_ops / (time_eval + time_compile)
    gflops_total *= 1e-9

    print( '  #flops:         %d' % num_ops )
    print( '  time (eval):    %f' % time_eval )
    print( '  time (compile): %f' % time_compile )
    print( '  gflops (eval):  %.2f' % gflops_eval )
    print( '  gflops (total): %.2f' % gflops_total )

    print( 'CSV_DATA: ' + ','.join([ "opt_einsum",
                                     args.backend,
                                     '"' + args.expression + '"',
                                     '"' + args.dim_sizes + '"',
                                     '"' + args.cont_path + '"',
                                     args.dtype,
                                     str(num_ops),
                                     str(time_compile),
                                     str(time_eval),
                                     str(gflops_eval),
                                     str(gflops_total) ]) )