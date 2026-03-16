def find_next_power_of_2(x):
    power = 1
    while power < x:
        power *= 2
    return power

def generate_tensor_config_binary_from_einsum(einsum_string, dim_sizes_input, data_type=None):
    import etops

    backend = "tpp"
    data_type = data_type if data_type is not None else etops.float64
    prim_first = etops.prim.zero
    prim_main = etops.prim.gemm
    prim_last = etops.prim.none

    

    # split einsum into left, right and out
    left_string, right_string = einsum_string.split("->")[0].split(",")
    out_string = einsum_string.split("->")[1]

    # remove any whitespace
    left_string = left_string.replace(" ", "")
    right_string = right_string.replace(" ", "")
    out_string = out_string.replace(" ", "")

    # check that characters appear at most once in each of left, right, and out
    if len(set(left_string)) != len(left_string):
        raise ValueError("Invalid einsum string: Each dimension character can appear at most once in the left tensor")
    if len(set(right_string)) != len(right_string):
        raise ValueError("Invalid einsum string: Each dimension character can appear at most once in the right tensor")
    if len(set(out_string)) != len(out_string):
        raise ValueError("Invalid einsum string: Each dimension character can appear at most once in the output tensor")


    # sort all appearing dimension characters by alphabet
    all_dim_chars = set(left_string + right_string + out_string)
    sorted_dim_chars = sorted(list(all_dim_chars))

    # check if dim_sizes is a dict or a tuple/list
    if isinstance(dim_sizes_input, dict):
        dim_sizes = []
        # build a list of dim sizes, sorted by character
        for dim_char in sorted_dim_chars:
            if dim_char not in dim_sizes_input:
                raise ValueError(f"Dimension size for dimension character {dim_char} not found in dim_sizes dict")
            dim_sizes.append(dim_sizes_input[dim_char])

    else:
        dim_sizes = dim_sizes_input


    if len(sorted_dim_chars) != len(dim_sizes):
        raise ValueError("Number of unique dimension characters in einsum string must match the length of dim_sizes")

    dim_types = []
    exec_types = []
    strides_left = []
    strides_right = []
    strides_out = []

    # create dict of dim sizes
    dim_size_dict = {}
    for i, dim_char in enumerate(sorted_dim_chars):
        dim_size_dict[dim_char] = dim_sizes[i]
    
    strides_dict_left = {}
    strides_dict_right = {}
    strides_dict_out = {}

    # create dict of strides for each tensor
    current_stride = 1
    for i in range(len(left_string) - 1, -1, -1):
        dim_char = left_string[i]
        strides_dict_left[dim_char] = current_stride
        current_stride *= dim_size_dict[dim_char]
    
    current_stride = 1
    for i in range(len(right_string) - 1, -1, -1):
        dim_char = right_string[i]
        strides_dict_right[dim_char] = current_stride
        current_stride *= dim_size_dict[dim_char]
    
    current_stride = 1
    for i in range(len(out_string) - 1, -1, -1):
        dim_char = out_string[i]
        strides_dict_out[dim_char] = current_stride
        current_stride *= dim_size_dict[dim_char]


    for dim_char in sorted_dim_chars:
        if dim_char in left_string and dim_char in right_string and dim_char in out_string:
            dim_types.append(etops.dim.c)
            strides_left.append(strides_dict_left[dim_char])
            strides_right.append(strides_dict_right[dim_char])
            strides_out.append(strides_dict_out[dim_char])

        elif dim_char in left_string and dim_char in out_string:
            dim_types.append(etops.dim.m)
            strides_left.append(strides_dict_left[dim_char])
            strides_right.append(0)
            strides_out.append(strides_dict_out[dim_char])

        elif dim_char in right_string and dim_char in out_string:
            dim_types.append(etops.dim.n)
            strides_left.append(0)
            strides_right.append(strides_dict_right[dim_char])
            strides_out.append(strides_dict_out[dim_char])

        elif dim_char in left_string and dim_char in right_string:
            dim_types.append(etops.dim.k)
            strides_left.append(strides_dict_left[dim_char])
            strides_right.append(strides_dict_right[dim_char])
            strides_out.append(0)

        else:
            raise ValueError("Invalid einsum string: Each dimension character must appear in at least two of the three tensors")

        exec_types.append(etops.exec.seq)

    strides_tensor = [[strides_left, strides_right, strides_out]]


    tensor_operation_config = etops.TensorOperationConfig(
        backend    =   backend,
        data_type  =   data_type,
        prim_first =   prim_first,
        prim_main  =   prim_main,
        prim_last  =   prim_last,
        dim_types  =   tuple(dim_types),
        exec_types =   tuple(exec_types),
        dim_sizes  =   tuple(dim_sizes),
        strides    =   tuple(strides_tensor)
    )

    return tensor_operation_config
