import etops
import os
from .utils import find_next_power_of_2

class JitCompiler:

    def __init__(self, cv):
        self.cv = cv
        self.string_kernel = ""

        self.init_class_variables()

    
    def init_class_variables(self):
        self.indent = "    "
        
        # there is currently no fp16 support in etops, so we use fp16 when dtype is set to fp64
        if self.cv.data_type == etops.float64:
            self.data_type_string = "ct.float16"
        else:
            self.data_type_string = "ct.float32"

    
    def jit_kernel(self):
        string_header = self.generate_header_string()
        string_shared_ids = self.generate_shared_ids_string()
        loop_header_strings = self.get_loop_header_strings()

        (accumulator_string,
        load_left_string,
        load_right_string,
        output_dtype_string,
        reshape_output_string,
        store_output_string,
        mma_string,
        reshape_string_left,
        reshape_string_right,
        permute_string_left,
        permute_string_right,
        permute_string_output) = self.get_loop_body_operation_strings()

        loop_body_strings_before_loop, loop_body_strings_before_next_loop, loop_body_strings_after_next_loop = self.get_loop_body_strings(accumulator_string,
                                                                                                                                          load_left_string,
                                                                                                                                          load_right_string,
                                                                                                                                          output_dtype_string,
                                                                                                                                          reshape_output_string,
                                                                                                                                          store_output_string,
                                                                                                                                          mma_string,
                                                                                                                                          reshape_string_left,
                                                                                                                                          reshape_string_right,
                                                                                                                                          permute_string_left,
                                                                                                                                          permute_string_right,
                                                                                                                                          permute_string_output)

        string_loop = self.generate_loop_string(loop_header_strings,
                                                 loop_body_strings_before_loop,
                                                 loop_body_strings_before_next_loop,
                                                 loop_body_strings_after_next_loop,
                                                 accumulator_string,
                                                 load_left_string,
                                                 load_right_string,
                                                 permute_string_left,
                                                 permute_string_right,
                                                 reshape_string_left,
                                                 reshape_string_right,
                                                 mma_string,
                                                 output_dtype_string,
                                                 reshape_output_string,
                                                 permute_string_output,
                                                 store_output_string)

        self.string_kernel += string_header + "\n"
        self.string_kernel += string_shared_ids
        self.string_kernel += string_loop

    
    def generate_header_string(self):
        strings_imports = []
        strings_imports.append("import cuda.tile as ct")

        kernel_decorator = "@ct.kernel()"
        kernel_name = "contraction_kernel"
        
        kernel_arguments = ["left", "right", "output"]
        
        # define imports
        string_header = ""
        for string_import in strings_imports:
            string_header += string_import + "\n"
        string_header += "\n"

        # function signature
        string_header += kernel_decorator + "\n"
        string_header += "def " + kernel_name + "("
        string_header += ", ".join(kernel_arguments)
        string_header += "):\n"

        # pid
        string_header += f"{self.indent}pid = ct.bid(0)\n"

        return string_header

    
    def generate_shared_ids_string(self):
        strings_shared_ids = []

        for i in range(self.cv.num_shared_outer_loops):
            if i == 0:
                string_loops_remaining = f"{self.indent}loops_remaining = pid"
            else:
                string_loops_remaining = f"{self.indent}loops_remaining -= shared_{i-1} * {self.cv.shared_loop_strides[i-1]}"
            if (i < self.cv.num_shared_outer_loops - 1):
                string_loop_id = f"{self.indent}shared_{i} = loops_remaining // {self.cv.shared_loop_strides[i]}"
            else:
                string_loop_id = f"{self.indent}shared_{i} = loops_remaining"
            string_shared_id = string_loops_remaining + "\n" + string_loop_id + "\n"
            strings_shared_ids.append(string_shared_id)

        string_shared_ids_to_return = ""

        for string_shared_id in strings_shared_ids:
            string_shared_ids_to_return += string_shared_id
            string_shared_ids_to_return += "\n"

        return string_shared_ids_to_return

    
    def get_loop_header_strings(self):
        loop_header_strings = []

        for i in range(self.cv.num_seq_outer_loops):
            loop_var_name = f"seq_{i}"
            loop_bound = self.cv.dim_sizes[self.cv.seq_loop_ids[i]]
            for_loop_string = self.get_string_for_loop_header(loop_var_name, loop_bound, self.cv.dim_types[self.cv.seq_loop_ids[i]])

            for_loop_string_with_indent = self.indent * (i + 1) + for_loop_string
            loop_header_strings.append(for_loop_string_with_indent)
        
        return loop_header_strings


    def get_string_for_loop_header(self, var_name, loop_bound, type):
        type_string = ""
        if type == etops.dim.m:
            type_string = "M"
        elif type == etops.dim.n:
            type_string = "N"
        elif type == etops.dim.k:
            type_string = "K"
        elif type == etops.dim.c:
            type_string = "C"

        return f"for {var_name} in range({loop_bound}):    # {type_string}\n"


    def get_loop_body_operation_strings(self):

        output_buffer_shape = self.get_output_buffer_shape()
        index_string_left, shape_string_left, index_string_right, shape_string_right, index_string_store = self.get_load_and_store_indices()
        permute_map_left, permute_map_right = self.get_permute_maps()
        reshape_tuple_out, permute_map_out = self.get_reshape_and_permute_map_out()

        # strings for each operation
        accumulator_string = f"accumulator = ct.full({output_buffer_shape}, 0, dtype=ct.float32)\n"
        load_left_string = f"tile_left = ct.load(left, index=({index_string_left}), shape=({shape_string_left}), padding_mode=ct.PaddingMode.ZERO)\n"
        load_right_string = f"tile_right = ct.load(right, index=({index_string_right}), shape=({shape_string_right}), padding_mode=ct.PaddingMode.ZERO)\n"
        output_dtype_string = f"out_to_store = ct.astype(accumulator, {self.data_type_string})\n"
        reshape_output_string = f"out_to_store = ct.reshape(out_to_store, {reshape_tuple_out})\n"
        store_output_string = f"ct.store(output, index=({index_string_store}), tile=out_to_store)\n"
        mma_string = f"accumulator = ct.mma(matrix_right, matrix_left, accumulator)\n"
        reshape_string_left = f"matrix_left = ct.reshape(tile_left, ({find_next_power_of_2(self.cv.kernel_shape_k)}, {find_next_power_of_2(self.cv.kernel_shape_m)}))\n"
        reshape_string_right = f"matrix_right = ct.reshape(tile_right, ({find_next_power_of_2(self.cv.kernel_shape_n)}, {find_next_power_of_2(self.cv.kernel_shape_k)}))\n"
        permute_string_left = f"tile_left = ct.permute(tile_left, {permute_map_left})\n" if permute_map_left is not None else None
        permute_string_right = f"tile_right = ct.permute(tile_right, {permute_map_right})\n" if permute_map_right is not None else None
        permute_string_output = f"out_to_store = ct.permute(out_to_store, {permute_map_out})\n" if permute_map_out is not None else None

        return accumulator_string, load_left_string, load_right_string, output_dtype_string, reshape_output_string, store_output_string, mma_string, reshape_string_left, reshape_string_right, permute_string_left, permute_string_right, permute_string_output


    def get_loop_body_strings(self, 
                              accumulator_string,
                              load_left_string,
                              load_right_string,
                              output_dtype_string,
                              reshape_output_string,
                              store_output_string,
                              mma_string,
                              reshape_string_left,
                              reshape_string_right,
                              permute_string_left,
                              permute_string_right,
                              permute_string_output):

        loop_body_strings_before_next_loop = ["" for i in range(self.cv.num_seq_outer_loops)]
        loop_body_strings_after_next_loop = ["" for i in range(self.cv.num_seq_outer_loops)]
        loop_body_strings_before_loop = ["" for i in range(self.cv.num_seq_outer_loops)]

        first_M_loop_depth, last_M_loop_depth, first_N_loop_depth, last_N_loop_depth, first_K_loop_depth, last_K_loop_depth, first_B_loop_depth, last_B_loop_depth = self.get_first_and_last_loop_depths()

        left_tile_load_loop_depth = max(last_B_loop_depth, last_M_loop_depth, last_K_loop_depth)
        right_tile_load_loop_depth = max(last_B_loop_depth, last_N_loop_depth, last_K_loop_depth)

        load_left_before_all_loops = self.cv.num_seq_loops_B == 0 and self.cv.num_seq_loops_M == 0 and self.cv.num_seq_loops_K == 0
        load_right_before_all_loops = self.cv.num_seq_loops_B == 0 and self.cv.num_seq_loops_N == 0 and self.cv.num_seq_loops_K == 0

        if self.cv.num_seq_loops_K > 0:
            buffer_allocation_loop_depth = first_K_loop_depth # should happen before this loop level
            output_store_loop_depth = first_K_loop_depth

        else:
            buffer_allocation_loop_depth = self.cv.num_seq_outer_loops - 1
            output_store_loop_depth = self.cv.num_seq_outer_loops - 1

        for i in range(self.cv.num_seq_outer_loops):
            indent_string = self.indent * (i + 1)
            # for each outer loop, determine the logic that needs to go in the body before the next nested loop 
            # and the logic that needs to go in the body after the next nested loop
            
            # buffer allocation loop
            if i == buffer_allocation_loop_depth:
                if self.cv.num_seq_loops_K > 0:
                    loop_body_strings_before_loop[i] += f"""{indent_string}{accumulator_string}"""
                else:
                    loop_body_strings_before_next_loop[i] += f"""{indent_string}    {accumulator_string}"""

            # load left loop
            if i == left_tile_load_loop_depth:
                if load_left_before_all_loops:
                    loop_body_strings_before_loop[i] += f"""{indent_string}{load_left_string}"""
                    if permute_string_left is not None:
                        loop_body_strings_before_loop[i] += f"""{indent_string}{permute_string_left}"""
                    loop_body_strings_before_loop[i] += f"""{indent_string}{reshape_string_left}"""
                else:
                    loop_body_strings_before_next_loop[i] += f"""{indent_string}    {load_left_string}"""
                    if permute_string_left is not None:
                        loop_body_strings_before_next_loop[i] += f"""{indent_string}    {permute_string_left}"""
                    loop_body_strings_before_next_loop[i] += f"""{indent_string}    {reshape_string_left}"""
            
            # load right loop
            if i == right_tile_load_loop_depth:
                if load_right_before_all_loops:
                    loop_body_strings_before_loop[i] += f"""{indent_string}{load_right_string}"""
                    if permute_string_right is not None:
                        loop_body_strings_before_loop[i] += f"""{indent_string}{permute_string_right}"""
                    loop_body_strings_before_loop[i] += f"""{indent_string}{reshape_string_right}"""
                else:
                    loop_body_strings_before_next_loop[i] += f"""{indent_string}    {load_right_string}"""
                    if permute_string_right is not None:
                        loop_body_strings_before_next_loop[i] += f"""{indent_string}    {permute_string_right}"""
                    loop_body_strings_before_next_loop[i] += f"""{indent_string}    {reshape_string_right}"""

            # mma loop
            if i == self.cv.num_seq_outer_loops - 1:
                loop_body_strings_before_next_loop[i] += f"""{indent_string}    {mma_string}"""
            
            # store output loop
            if i == output_store_loop_depth:
                if self.cv.num_seq_loops_K > 0:
                    loop_body_strings_after_next_loop[i] += f"""{indent_string}{output_dtype_string}"""
                    loop_body_strings_after_next_loop[i] += f"""{indent_string}{reshape_output_string}"""
                    if permute_string_output is not None:
                        loop_body_strings_after_next_loop[i] += f"""{indent_string}{permute_string_output}"""
                    loop_body_strings_after_next_loop[i] += f"""{indent_string}{store_output_string}"""
                else:
                    loop_body_strings_before_next_loop[i] += f"""{indent_string}    {output_dtype_string}"""
                    loop_body_strings_before_next_loop[i] += f"""{indent_string}    {reshape_output_string}"""
                    if permute_string_output is not None:
                        loop_body_strings_before_next_loop[i] += f"""{indent_string}    {permute_string_output}"""
                    loop_body_strings_before_next_loop[i] += f"""{indent_string}    {store_output_string}"""

        return loop_body_strings_before_loop, loop_body_strings_before_next_loop, loop_body_strings_after_next_loop


    def generate_loop_string(self, loop_header_strings,
                              loop_body_strings_before_loop,
                              loop_body_strings_before_next_loop,
                              loop_body_strings_after_next_loop,
                              accumulator_string,
                              load_left_string,
                              load_right_string,
                              permute_string_left,
                              permute_string_right,
                              reshape_string_left,
                              reshape_string_right,
                              mma_string,
                              output_dtype_string,
                              reshape_output_string,
                              permute_string_output,
                              store_output_string):

        loop_string = ""
        for i in range(self.cv.num_seq_outer_loops):
            loop_string += loop_body_strings_before_loop[i]
            loop_string += loop_header_strings[i]
            loop_string += loop_body_strings_before_next_loop[i]

        for i in range(self.cv.num_seq_outer_loops -1, -1, -1):
            loop_string += loop_body_strings_after_next_loop[i]
        
        if self.cv.num_seq_outer_loops == 0:
            loop_string += "    " + accumulator_string
            loop_string += "    " + load_left_string
            loop_string += "    " + load_right_string
            if permute_string_left is not None:
                loop_string += "    " + permute_string_left
            if permute_string_right is not None:
                loop_string += "    " + permute_string_right
            loop_string += "    " + reshape_string_left
            loop_string += "    " + reshape_string_right
            loop_string += "    " + mma_string
            loop_string += "    " + output_dtype_string
            loop_string += "    " + reshape_output_string
            if permute_string_output is not None:
                loop_string += "    " + permute_string_output
            loop_string += "    " + store_output_string

        return loop_string


    def get_first_and_last_loop_depths(self):
        first_M_loop_depth = 0
        last_M_loop_depth = 0
        first_N_loop_depth = 0
        last_N_loop_depth = 0
        first_K_loop_depth = 0
        last_K_loop_depth = 0
        first_B_loop_depth = 0
        last_B_loop_depth = 0

        if len(self.cv.seq_loops_M) > 0:
            first_M_loop_id = self.cv.seq_loops_M[0]
            last_M_loop_id = self.cv.seq_loops_M[-1]

            first_M_loop_depth = self.cv.seq_loop_ids.index(first_M_loop_id)
            last_M_loop_depth = self.cv.seq_loop_ids.index(last_M_loop_id)

        if len(self.cv.seq_loops_N) > 0:
            first_N_loop_id = self.cv.seq_loops_N[0]
            last_N_loop_id = self.cv.seq_loops_N[-1]

            first_N_loop_depth = self.cv.seq_loop_ids.index(first_N_loop_id)
            last_N_loop_depth = self.cv.seq_loop_ids.index(last_N_loop_id)

        if len(self.cv.seq_loops_K) > 0:
            first_K_loop_id = self.cv.seq_loops_K[0]
            last_K_loop_id = self.cv.seq_loops_K[-1]

            first_K_loop_depth = self.cv.seq_loop_ids.index(first_K_loop_id)
            last_K_loop_depth = self.cv.seq_loop_ids.index(last_K_loop_id)

        if len(self.cv.seq_loops_B) > 0:
            first_B_loop_id = self.cv.seq_loops_B[0]
            last_B_loop_id = self.cv.seq_loops_B[-1]

            first_B_loop_depth = self.cv.seq_loop_ids.index(first_B_loop_id)
            last_B_loop_depth = self.cv.seq_loop_ids.index(last_B_loop_id)

        return first_M_loop_depth, last_M_loop_depth, first_N_loop_depth, last_N_loop_depth, first_K_loop_depth, last_K_loop_depth, first_B_loop_depth, last_B_loop_depth

    
    def get_output_buffer_shape(self):
        # padding for each dimension (this might be improved so that padding is only needed once)

        output_buffer_shape = ()
        start_appending_to_buffer_shape = False

        for i in range(self.cv.num_seq_outer_loops):
            if start_appending_to_buffer_shape and not self.cv.seq_loop_ids[i] in self.cv.seq_loops_K:
                output_buffer_shape += (find_next_power_of_2(self.cv.dim_sizes[self.cv.seq_loop_ids[i]]),)

            elif self.cv.seq_loop_ids[i] in self.cv.seq_loops_K:
                start_appending_to_buffer_shape = True

        acc_shape_m = find_next_power_of_2(self.cv.kernel_shape_m)
        acc_shape_n = find_next_power_of_2(self.cv.kernel_shape_n)
        output_buffer_shape += (acc_shape_n, acc_shape_m)

        return output_buffer_shape

    
    def get_load_and_store_indices(self):
        load_indices_left, shape_load_left = self.get_load_and_store_indices_for_tensor(self.cv.config_indices_in_left_tensor)
        load_indices_right, shape_load_right = self.get_load_and_store_indices_for_tensor(self.cv.config_indices_in_right_tensor)
        store_indices, _ = self.get_load_and_store_indices_for_tensor(self.cv.config_indices_in_out_tensor)

        if len(load_indices_left) != self.cv.num_dimensions_left or len(shape_load_left) != self.cv.num_dimensions_left:
            raise ValueError("Error in load indices or shapes for left tensor: Lengths do not match number of dimensions")
        if len(load_indices_right) != self.cv.num_dimensions_right or len(shape_load_right) != self.cv.num_dimensions_right:
            raise ValueError("Error in load indices or shapes for right tensor: Lengths do not match number of dimensions")
        if len(store_indices) != self.cv.num_dimensions_output:
            raise ValueError("Error in store indices or shapes for output tensor: Lengths do not match number of dimensions")

        index_string_left = ", ".join(load_indices_left)
        shape_string_left = ", ".join(shape_load_left)
        index_string_right = ", ".join(load_indices_right)
        shape_string_right = ", ".join(shape_load_right)
        index_string_store = ", ".join(store_indices)

        return index_string_left, shape_string_left, index_string_right, shape_string_right, index_string_store


    def get_load_and_store_indices_for_tensor(self, config_indices_in_tensor):
        load_store_indices = []
        shape_load_store = []

        for i in range(len(config_indices_in_tensor)):
            dim_id_config = config_indices_in_tensor[i]
            
            if self.cv.exec_types[dim_id_config] == etops.exec.shared:
                index_where_shared_id_matches = self.cv.shared_loop_ids.index(dim_id_config)
                load_store_indices.append(f"shared_{index_where_shared_id_matches}")
                shape_load_store.append("1")
            
            elif self.cv.exec_types[dim_id_config] == etops.exec.seq:
                index_where_seq_id_matches = self.cv.seq_loop_ids.index(dim_id_config)
                load_store_indices.append(f"seq_{index_where_seq_id_matches}")
                shape_load_store.append("1")

            elif self.cv.exec_types[dim_id_config] == etops.exec.prim:
                load_store_indices.append("0")
                shape_load_store.append(f"{find_next_power_of_2(self.cv.dim_sizes[dim_id_config])}")
        
        return load_store_indices, shape_load_store


    def get_permute_maps(self):
        permute_map_left = []
        permute_map_right = []
        permute_map_out = []

        # left map
        for i in range(self.cv.num_prim_k):
            prim_dim_id = self.cv.prim_k_ids[i]
            index_where_prim_id_matches = self.cv.config_indices_in_left_tensor.index(prim_dim_id)
            permute_map_left.append(index_where_prim_id_matches)

        for i in range(self.cv.num_prim_m):
            prim_dim_id = self.cv.prim_m_ids[i]
            index_where_prim_id_matches = self.cv.config_indices_in_left_tensor.index(prim_dim_id)
            permute_map_left.append(index_where_prim_id_matches)

        # right map
        for i in range(self.cv.num_prim_n):
            prim_dim_id = self.cv.prim_n_ids[i]
            index_where_prim_id_matches = self.cv.config_indices_in_right_tensor.index(prim_dim_id)
            permute_map_right.append(index_where_prim_id_matches)

        for i in range(self.cv.num_prim_k):
            prim_dim_id = self.cv.prim_k_ids[i]
            index_where_prim_id_matches = self.cv.config_indices_in_right_tensor.index(prim_dim_id)
            permute_map_right.append(index_where_prim_id_matches)


        # check if permute maps are in order; if so, no permute is needed
        if permute_map_left == sorted(permute_map_left):
            permute_map_left = None

        if permute_map_right == sorted(permute_map_right):
            permute_map_right = None


        # if permute maps are not None, inject the other dimension indices (dimensions are all of size 1) to the front
        if permute_map_left is not None:
            append_to_the_front_of_permute_map_left = []
            for i in range(self.cv.num_dimensions_left):
                if not i in permute_map_left:
                    append_to_the_front_of_permute_map_left.append(i)

            permute_map_left = append_to_the_front_of_permute_map_left + permute_map_left

        if permute_map_right is not None:
            append_to_the_front_of_permute_map_right = []
            for i in range(self.cv.num_dimensions_right):
                if not i in permute_map_right:
                    append_to_the_front_of_permute_map_right.append(i)

            permute_map_right = append_to_the_front_of_permute_map_right + permute_map_right


        # generate tupels if not none
        permute_map_left_tuple = None
        permute_map_right_tuple = None

        if permute_map_left is not None:
            permute_map_left_tuple = tuple(permute_map_left)
        if permute_map_right is not None:
            permute_map_right_tuple = tuple(permute_map_right)


        return permute_map_left_tuple, permute_map_right_tuple


    def get_reshape_and_permute_map_out(self):
        # out matrix has shape kernel_n x kernel_m
        # we need to first reshape it to (1, 1, ..., prim_n1, 1, prim_n2, ..., 1, prim_m1, 1, prim_m2)
        #    where 1 are all shared dimensions and prims follow the relative order from the matrix output
        #    the prim dimensions in the primitive dimension slots of the output tensor

        # afterwards we need to permute the primitive dimensions in the reshaped tile so that they are in the same order as the output tensor configuration

        reshape_out = [1 for i in range(self.cv.num_dimensions_output)]
        # ordered for config
        prim_ids_in_out = self.cv.prim_n_ids + self.cv.prim_m_ids
        current_prim_count = 0
        prim_indices = []

        for i in range(self.cv.num_dimensions_output):
            current_out_config_index = self.cv.config_indices_in_out_tensor[i]

            if self.cv.exec_types[current_out_config_index] != etops.exec.prim:
                continue

            prim_indices.append(i)

            current_prim_id = prim_ids_in_out[current_prim_count]
            current_dim_size = self.cv.dim_sizes[current_prim_id]
            reshape_out[i] = find_next_power_of_2(current_dim_size)

            current_prim_count += 1


        prim_indices_reordered = []
        for i in range(len(self.cv.prim_config_indices_output)):
            current_prim_config_index = self.cv.prim_config_indices_output[i]
            index_where_current_prim_config_index_matches_prim_id = prim_ids_in_out.index(current_prim_config_index)
            prim_indices_reordered.append(prim_indices[index_where_current_prim_config_index_matches_prim_id])

        if len(prim_indices_reordered) != len(self.cv.prim_config_indices_output) or len(prim_indices_reordered) != len(prim_indices):
            raise ValueError("Error in permute map for output: Length of reordered prim indices does not match number of prim dimensions in output or length of prim config indices in output")

        permute_map_out = []
        current_prim_reordered_count = 0

        for i in range(self.cv.num_dimensions_output):
            current_out_config_index = self.cv.config_indices_in_out_tensor[i]

            if self.cv.exec_types[current_out_config_index] != etops.exec.prim:
                permute_map_out.append(i)

            else:
                permute_map_out.append(prim_indices_reordered[current_prim_reordered_count])
                current_prim_reordered_count += 1


        reshape_out_tuple = tuple(reshape_out)
        permute_map_out_tuple = tuple(permute_map_out)

        # check if permute map is in order; if so, no permute is needed
        if permute_map_out == sorted(permute_map_out):
            permute_map_out_tuple = None

        return reshape_out_tuple, permute_map_out_tuple


    def print_kernel(self):
        print(self.string_kernel)

    
    def safe_kernel_to_file(self, filename):
        dirname = "generated_kernels"
        path = os.path.join(dirname, filename)
        os.makedirs(dirname, exist_ok=True)


        with open(path, "w") as f:
            f.write(self.string_kernel)
        f.close()
