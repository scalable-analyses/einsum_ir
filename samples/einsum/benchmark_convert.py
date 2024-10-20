
def path_to_tree(einsum_string, path):
    """
    Converts an einsum string and a contraction path into a einsum tree representation.

    Parameters:
    einsum_string (str): The einsum string representing the tensor contraction.
    path (list): The contraction path.

    Returns:
    str: The tree representation of the einsum contraction.
    """
    inputs  = einsum_string.split( "->" )[0].split( "," )
    output = einsum_string.split( "->" )[1]

    tree_off = len( inputs )

    tree = inputs.copy()
    # add commas to tensors in tree
    for i in range( len( tree ) ):
        tree[i] = ','.join( tree[i] )

    # assemble einsum tree
    for cont in path:
        # select input tensors
        children = [inputs[cont[0]], inputs[cont[1]]]

        # remove children from list
        max_index = max( cont[0], cont[1] )
        min_index = min( cont[0], cont[1] )
        del inputs[max_index]
        del inputs[min_index]

        # get indices still in any other tensor
        indices_other = set()
        for tensor in inputs+[output]:
            indices_other = indices_other.union( set( tensor ) )

        # get indices which are in the two children and still in any other tensor
        indices_children = set( children[0] ).union( set( children[1] ) )
        indices_parent = indices_children.intersection( indices_other )
        indices_parent = sorted( list( indices_parent ) )

        intermediate = str().join( indices_parent )

        # add intermediate tensor to list
        inputs.append( intermediate )

        # replace children with tree info (incl partial tres)
        children[0] = tree[cont[0]]
        children[1] = tree[cont[1]]

        # remove children from tree
        del tree[max_index]
        del tree[min_index]

        intermediate = ','.join( intermediate )
        
        node = "["+children[0]+"],["+children[1]+"]->["+intermediate+"]"
        tree.append( node )
    
    return tree[0]

import pickle
import argparse
import numpy

if __name__ == "__main__":
    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument( "instance", type=str, help="Path to instance file" )
    # output format either einsum_string+path or tree
    parser.add_argument( "-t", "--tree", action="store_true", help="Output tree instead of einsum_string+path" )
    # replace UTF-8 with ASCII chars instead of integers
    parser.add_argument( "-a", "--ascii", action="store_true", help="Output ASCII instead of integers" )
    args = parser.parse_args()

    # load instance or return error message
    try:
        with open( args.instance, "rb" ) as file:
            einsum_string, tensors, path_meta, _ = pickle.load( file )
    except:
        print( "Error: Could not load instance file" )
        exit(1)

    # path_meta[0] is optimized for intermediate size
    path = path_meta[0][0]

    tree = path_to_tree( einsum_string, path )

    # get set of characters
    chars = set( einsum_string )
    # remove special characters ('-', '>', ',')
    chars = chars.difference( set( [',', '-', '>'] ) )

    # determine dimension sizes
    dim_sizes = {}
    tensors_str = einsum_string.split( '->' )[0].split( ',' )

    for id, tensor_str in enumerate( tensors_str ):
        tensor_sizes = tensors[id].shape

        for i, char in enumerate( tensor_str ):
            dim_sizes[ char ] = tensor_sizes[i]
    
    # sort by key and convert to list without key
    dim_sizes = [ dim_sizes[key] for key in sorted( dim_sizes.keys() ) ]

    # assign indices to characters
    char_dict = {}
    for i, char in enumerate( sorted(chars) ):
        char_dict[char] = i

    # adjust format if not ascii
    if not args.ascii:
        tensors_str_in = einsum_string.split( '->' )[0].split( ',' )
        tensor_str_out = einsum_string.split( '->' )[1]

        # add comma between chars of input tensors
        for i, tensor_str in enumerate( tensors_str_in ):
            tensors_str_in[i] = ','.join( tensor_str )
        # add comma between chars of output tensor
        tensor_str_out = ','.join( tensor_str_out )

        einsum_string = ']->['.join( ['],['.join( tensors_str_in ), tensor_str_out] )
        einsum_string = '[' + einsum_string + ']'

    # convert to ASCII starting at A but jump over special characters
    if args.ascii:
        for char, index in char_dict.items():
            if index > 90-65:
                index += 6
            char_dict[char] = chr( index + 65 )

    # replace characters with indices
    einsum_string_list = []
    for char in einsum_string:
        if char in char_dict:
            einsum_string_list.append( char_dict[char] )
        else:
            einsum_string_list.append( char )
    einsum_string_list = [ str( elem ) for elem in einsum_string_list ]
    einsum_string = ''.join( einsum_string_list )

    tree_list = []
    for char in tree:
        if char in char_dict:
            tree_list.append( char_dict[char] )
        else:
            tree_list.append( char )
    tree_list = [ str( elem ) for elem in tree_list ]
    tree = ''.join( tree_list )

    # assemble output
    output = ""
    if args.tree:
        output += tree
        output += " " + ",".join( [str( size ) for size in dim_sizes] )
    else:
        output += "\"" + einsum_string + "\""
        output += " \"" + ",".join( [str( size ) for size in dim_sizes] ) + "\""
        output +=  " \"" + ",".join( str(path_elem).replace(" ", "") for path_elem in path ) + "\""

    if tensors[0].dtype == numpy.float32:
        output += " FP32"
    elif tensors[0].dtype == numpy.float64:
        output += " FP64"
    elif tensors[0].dtype == numpy.complex64:
        output += " CPX64"

    print( output )