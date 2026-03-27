import etops
import os

def find_next_power_of_2(x):
    power = 1
    while power < x:
        power *= 2
    return power

# =============================================================================
# Module-level registration for make_tensor_view lowering
# =============================================================================

# Registration state for make_tensor_view lowering
_MAKE_TENSOR_VIEW_REGISTERED = False
_MAKE_TENSOR_VIEW_AVAILABLE = None  # None = not checked, True/False = checked


def _register_make_tensor_view():
    """
    Register make_tensor_view lowering rule with cuda.tile.
    
    Called lazily when JitCompiler is instantiated, not at module import.
    This allows the module to be imported even if cuda.tile is not installed.
    
    Returns:
        True if registration succeeded, False if cuda.tile unavailable.
    """
    global _MAKE_TENSOR_VIEW_REGISTERED, _MAKE_TENSOR_VIEW_AVAILABLE
    
    if _MAKE_TENSOR_VIEW_REGISTERED:
        return True
    
    if _MAKE_TENSOR_VIEW_AVAILABLE is False:
        return False
    
    try:
        import cuda.tile as ct
        from cuda.tile._ir.op_impl import impl
        from cuda.tile._ir.ops import MakeTensorView
        from cuda.tile._ir.ir import Var, Builder, ArrayValue
        from cuda.tile._ir.type import ArrayTy, TupleTy, SizeTy
    except ImportError:
        _MAKE_TENSOR_VIEW_AVAILABLE = False
        return False
    
    _MAKE_TENSOR_VIEW_AVAILABLE = True
    
    # Check if already registered by another module
    if hasattr(ct, 'make_tensor_view'):
        _MAKE_TENSOR_VIEW_REGISTERED = True
        return True
    
    # Define the Python stub
    def make_tensor_view(array, shape, dynamic_strides):
        """
        Create a view of a tensor with specified shape and strides.
        
        Args:
            array: Input tensor (base pointer)
            shape: Tuple of dimension sizes
            dynamic_strides: Tuple of strides for each dimension
        
        Returns:
            A tensor view with the specified layout
        """
        pass
    
    ct.make_tensor_view = make_tensor_view
    
    # Register the lowering implementation
    @impl(ct.make_tensor_view)
    def make_tensor_view_impl(array: Var, shape: Var, dynamic_strides: Var):
        builder = Builder.get_current()
        dtype = getattr(
            array.get_type(),
            'dtype',
            getattr(array.get_type(), 'value_type', None)
        )
        
        arr_val = array.get_aggregate()
        base_ptr = arr_val.base_ptr
        shape_tup = shape.get_aggregate().items
        strides_tup = dynamic_strides.get_aggregate().items

        def _const_val(var):
            """Return the integer value if var is a compile-time constant, else None.

            Uses Var.is_constant() / Var.get_constant() from cuda.tile._ir.ir,
            which is the correct API for inspecting constant scalars at @impl time.
            """
            try:
                if var.is_constant():
                    return int(var.get_constant())
            except Exception:
                pass
            return None

        # Preserve static size information for shapes and strides where the
        # values are compile-time constants.
        shape_size_types  = [SizeTy(_const_val(v)) for v in shape_tup]
        stride_size_types = [SizeTy(_const_val(v)) for v in strides_tup]

        # Create result type, preserving static shape/stride info where available
        res_ty = ArrayTy(
            dtype,
            shape=TupleTy(shape_size_types),
            strides=TupleTy(stride_size_types),
            elements_disjoint=False,
            base_ptr_div_by=None,
            stride_div_by=(None,) * len(strides_tup),
            shape_div_by=(None,) * len(shape_tup)
        )
        
        # MakeTensorView operands must only include dims that remain dynamic.
        # Static dims are encoded in the type itself and must NOT appear as
        # operands or the IR verifier rejects the op.
        dynamic_shape_ops  = tuple(
            v for v, st in zip(shape_tup,   shape_size_types)  if st.value is None
        )
        dynamic_stride_ops = tuple(
            v for v, st in zip(strides_tup, stride_size_types) if st.value is None
        )

        # Add the MakeTensorView operation
        res_var = builder.add_operation(
            MakeTensorView,
            res_ty,
            dict(
                base_ptr=base_ptr,
                shape=dynamic_shape_ops,
                dynamic_strides=dynamic_stride_ops,
            )
        )
        
        # Set aggregate for downstream operations
        res_var.set_aggregate(
            ArrayValue(base_ptr, tuple(shape_tup), tuple(strides_tup))
        )
        return res_var
    
    _MAKE_TENSOR_VIEW_REGISTERED = True
    return True


# =============================================================================
# JitCompiler class
# =============================================================================

class JitCompiler:

    def __init__(self, cv):
        # Ensure make_tensor_view is registered before compilation
        if not _register_make_tensor_view():
            raise ImportError(
                "cuda.tile is required for the cutile backend. "
                "Install with: pip install cuda-tile"
            )
        
        self.cv = cv
        self.string_kernel = ""

        self.init_class_variables()

    
    def init_class_variables(self):
        self.indent = "    "
        
        # Map etops data types to cuda.tile type strings
        dtype_to_cutile = {
            etops.float32: "ct.float32",
            etops.float64: "ct.float64",
            etops.float16: "ct.float16",
            etops.bfloat16: "ct.bfloat16",
            etops.tfloat32: "ct.tfloat32",
        }
        self.data_type_string = dtype_to_cutile.get(self.cv.data_type, "ct.float32")
        
        # Output dtype: TF32 uses float32 storage (TF32 is a compute format, not storage format)
        # For all other types, output dtype matches the data type
        if self.cv.data_type == etops.tfloat32:
            self.output_dtype_string = "ct.float32"
        else:
            self.output_dtype_string = self.data_type_string
        
        # Accumulator dtype: use float64 for float64 inputs, float32 for all others
        # (float32 accumulation for float16/bfloat16/tfloat32 is intentional for numerical stability)
        if self.cv.data_type == etops.float64:
            self.accumulator_dtype_string = "ct.float64"
        else:
            self.accumulator_dtype_string = "ct.float32"

    
    def _validate_binary_operation(self):
        """
        Validate that this is a binary operation (gemm main primitive).
        
        Raises:
            NotImplementedError: If the main primitive is not gemm.
        """
        if self.cv.prim_main != etops.prim.gemm:
            raise NotImplementedError(
                f"Only gemm main primitive is supported, got {self.cv.prim_main}. "
                f"Unary operations and other primitives are not yet implemented."
            )
    
    
    def _get_tensor_view_args(self, stride_sorted_indices, strides, tensor_name):
        """
        Get shape and strides for make_tensor_view, ordered by stride (high to low).
        
        Stride 0 indicates the dimension is NOT part of that tensor.
        This method filters those out and returns the effective shape/strides,
        ordered by stride descending (highest stride first) for GPU DMA optimization.
        
        Args:
            stride_sorted_indices: List of config indices sorted by stride descending.
            strides: Tuple of strides for this tensor (from config.strides).
            tensor_name: Name of tensor for error messages ("in0", "in1", "out").
        
        Returns:
            Tuple of (shape_tuple, strides_tuple) for use with make_tensor_view.
            Dimensions are ordered by stride descending (highest stride leftmost).
        
        Raises:
            ValueError: If all strides are 0 (empty tensor).
        """
        l_shape = []
        l_strides = []
        
        for l_config_idx in stride_sorted_indices:
            l_stride = strides[l_config_idx]
            if l_stride != 0:
                l_shape.append(self.cv.dim_sizes[l_config_idx])
                l_strides.append(l_stride)
        
        if len(l_shape) == 0:
            raise ValueError(
                f"Tensor '{tensor_name}' has all strides = 0, "
                f"which indicates an empty tensor. This is not supported."
            )
        
        return (tuple(l_shape), tuple(l_strides))

    
    def jit_kernel(self):
        string_header = self.generate_header_string()
        string_shared_ids = self.generate_shared_ids_string()
        loop_header_strings = self.get_loop_header_strings()

        (accumulator_string,
        load_in0_string,
        load_in1_string,
        output_dtype_string,
        reshape_output_string,
        store_output_string,
        mma_string,
        reshape_string_in0,
        reshape_string_in1,
        permute_string_in0,
        permute_string_in1,
        permute_string_output) = self.get_loop_body_operation_strings()

        loop_body_strings_before_loop, loop_body_strings_before_next_loop, loop_body_strings_after_next_loop = self.get_loop_body_strings(accumulator_string,
                                                                                                                                          load_in0_string,
                                                                                                                                          load_in1_string,
                                                                                                                                          output_dtype_string,
                                                                                                                                          reshape_output_string,
                                                                                                                                          store_output_string,
                                                                                                                                          mma_string,
                                                                                                                                          reshape_string_in0,
                                                                                                                                          reshape_string_in1,
                                                                                                                                          permute_string_in0,
                                                                                                                                          permute_string_in1,
                                                                                                                                          permute_string_output)

        string_loop = self.generate_loop_string(loop_header_strings,
                                                 loop_body_strings_before_loop,
                                                 loop_body_strings_before_next_loop,
                                                 loop_body_strings_after_next_loop,
                                                 accumulator_string,
                                                 load_in0_string,
                                                 load_in1_string,
                                                 permute_string_in0,
                                                 permute_string_in1,
                                                 reshape_string_in0,
                                                 reshape_string_in1,
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
        
        kernel_arguments = ["in0", "in1", "out"]
        
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
        
        # Validate that this is a binary operation (gemm)
        self._validate_binary_operation()
        
        # make_tensor_view calls to reinterpret input pointers
        # in0 tensor (first input)
        l_shape_in0, l_strides_in0 = self._get_tensor_view_args(
            self.cv.stride_sorted_indices_in0, self.cv.strides_in0, "in0"
        )
        string_header += f"{self.indent}in0 = ct.make_tensor_view(in0, {l_shape_in0}, {l_strides_in0})\n"
        
        # in1 tensor (second input)
        l_shape_in1, l_strides_in1 = self._get_tensor_view_args(
            self.cv.stride_sorted_indices_in1, self.cv.strides_in1, "in1"
        )
        string_header += f"{self.indent}in1 = ct.make_tensor_view(in1, {l_shape_in1}, {l_strides_in1})\n"
        
        # out tensor (output)
        l_shape_out, l_strides_out = self._get_tensor_view_args(
            self.cv.stride_sorted_indices_out, self.cv.strides_out, "out"
        )
        string_header += f"{self.indent}out = ct.make_tensor_view(out, {l_shape_out}, {l_strides_out})\n"

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
        index_string_in0, shape_string_in0, index_string_in1, shape_string_in1, index_string_store = self.get_load_and_store_indices()
        permute_map_in0, permute_map_in1 = self.get_permute_maps()
        reshape_tuple_out, permute_map_out = self.get_reshape_and_permute_map_out()

        # strings for each operation
        accumulator_string = f"accumulator = ct.full({output_buffer_shape}, 0, dtype={self.accumulator_dtype_string})\n"
        load_in0_string = f"tile_in0 = ct.load(in0, index=({index_string_in0}), shape=({shape_string_in0}), padding_mode=ct.PaddingMode.ZERO)\n"
        load_in1_string = f"tile_in1 = ct.load(in1, index=({index_string_in1}), shape=({shape_string_in1}), padding_mode=ct.PaddingMode.ZERO)\n"
        output_dtype_string = f"out_to_store = ct.astype(accumulator, {self.output_dtype_string})\n"
        reshape_output_string = f"out_to_store = ct.reshape(out_to_store, {reshape_tuple_out})\n"
        store_output_string = f"ct.store(out, index=({index_string_store}), tile=out_to_store)\n"
        mma_string = f"accumulator = ct.mma(matrix_in1, matrix_in0, accumulator)\n"
        reshape_string_in0 = f"matrix_in0 = ct.reshape(tile_in0, ({find_next_power_of_2(self.cv.kernel_shape_k)}, {find_next_power_of_2(self.cv.kernel_shape_m)}))\n"
        reshape_string_in1 = f"matrix_in1 = ct.reshape(tile_in1, ({find_next_power_of_2(self.cv.kernel_shape_n)}, {find_next_power_of_2(self.cv.kernel_shape_k)}))\n"
        permute_string_in0 = f"tile_in0 = ct.permute(tile_in0, {permute_map_in0})\n" if permute_map_in0 is not None else None
        permute_string_in1 = f"tile_in1 = ct.permute(tile_in1, {permute_map_in1})\n" if permute_map_in1 is not None else None
        permute_string_output = f"out_to_store = ct.permute(out_to_store, {permute_map_out})\n" if permute_map_out is not None else None

        return accumulator_string, load_in0_string, load_in1_string, output_dtype_string, reshape_output_string, store_output_string, mma_string, reshape_string_in0, reshape_string_in1, permute_string_in0, permute_string_in1, permute_string_output


    def get_loop_body_strings(self,
                              accumulator_string,
                              load_in0_string,
                              load_in1_string,
                              output_dtype_string,
                              reshape_output_string,
                              store_output_string,
                              mma_string,
                              reshape_string_in0,
                              reshape_string_in1,
                              permute_string_in0,
                              permute_string_in1,
                              permute_string_output):

        loop_body_strings_before_next_loop = ["" for i in range(self.cv.num_seq_outer_loops)]
        loop_body_strings_after_next_loop = ["" for i in range(self.cv.num_seq_outer_loops)]
        loop_body_strings_before_loop = ["" for i in range(self.cv.num_seq_outer_loops)]

        first_M_loop_depth, last_M_loop_depth, first_N_loop_depth, last_N_loop_depth, first_K_loop_depth, last_K_loop_depth, first_B_loop_depth, last_B_loop_depth = self.get_first_and_last_loop_depths()

        in0_tile_load_loop_depth = max(last_B_loop_depth, last_M_loop_depth, last_K_loop_depth)
        in1_tile_load_loop_depth = max(last_B_loop_depth, last_N_loop_depth, last_K_loop_depth)

        load_in0_before_all_loops = self.cv.num_seq_loops_B == 0 and self.cv.num_seq_loops_M == 0 and self.cv.num_seq_loops_K == 0
        load_in1_before_all_loops = self.cv.num_seq_loops_B == 0 and self.cv.num_seq_loops_N == 0 and self.cv.num_seq_loops_K == 0

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

            # load in0 loop
            if i == in0_tile_load_loop_depth:
                if load_in0_before_all_loops:
                    loop_body_strings_before_loop[i] += f"""{indent_string}{load_in0_string}"""
                    if permute_string_in0 is not None:
                        loop_body_strings_before_loop[i] += f"""{indent_string}{permute_string_in0}"""
                    loop_body_strings_before_loop[i] += f"""{indent_string}{reshape_string_in0}"""
                else:
                    loop_body_strings_before_next_loop[i] += f"""{indent_string}    {load_in0_string}"""
                    if permute_string_in0 is not None:
                        loop_body_strings_before_next_loop[i] += f"""{indent_string}    {permute_string_in0}"""
                    loop_body_strings_before_next_loop[i] += f"""{indent_string}    {reshape_string_in0}"""
            
            # load in1 loop
            if i == in1_tile_load_loop_depth:
                if load_in1_before_all_loops:
                    loop_body_strings_before_loop[i] += f"""{indent_string}{load_in1_string}"""
                    if permute_string_in1 is not None:
                        loop_body_strings_before_loop[i] += f"""{indent_string}{permute_string_in1}"""
                    loop_body_strings_before_loop[i] += f"""{indent_string}{reshape_string_in1}"""
                else:
                    loop_body_strings_before_next_loop[i] += f"""{indent_string}    {load_in1_string}"""
                    if permute_string_in1 is not None:
                        loop_body_strings_before_next_loop[i] += f"""{indent_string}    {permute_string_in1}"""
                    loop_body_strings_before_next_loop[i] += f"""{indent_string}    {reshape_string_in1}"""

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
                              load_in0_string,
                              load_in1_string,
                              permute_string_in0,
                              permute_string_in1,
                              reshape_string_in0,
                              reshape_string_in1,
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
            loop_string += "    " + load_in0_string
            loop_string += "    " + load_in1_string
            if permute_string_in0 is not None:
                loop_string += "    " + permute_string_in0
            if permute_string_in1 is not None:
                loop_string += "    " + permute_string_in1
            loop_string += "    " + reshape_string_in0
            loop_string += "    " + reshape_string_in1
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
        # Use stride-sorted indices to match tensor view dimension order
        load_indices_in0, shape_load_in0 = self.get_load_and_store_indices_for_tensor(self.cv.stride_sorted_indices_in0)
        load_indices_in1, shape_load_in1 = self.get_load_and_store_indices_for_tensor(self.cv.stride_sorted_indices_in1)
        store_indices, _ = self.get_load_and_store_indices_for_tensor(self.cv.stride_sorted_indices_out)

        if len(load_indices_in0) != self.cv.num_dimensions_in0 or len(shape_load_in0) != self.cv.num_dimensions_in0:
            raise ValueError("Error in load indices or shapes for in0 tensor: Lengths do not match number of dimensions")
        if len(load_indices_in1) != self.cv.num_dimensions_in1 or len(shape_load_in1) != self.cv.num_dimensions_in1:
            raise ValueError("Error in load indices or shapes for in1 tensor: Lengths do not match number of dimensions")
        if len(store_indices) != self.cv.num_dimensions_out:
            raise ValueError("Error in store indices or shapes for out tensor: Lengths do not match number of dimensions")

        index_string_in0 = ", ".join(load_indices_in0)
        shape_string_in0 = ", ".join(shape_load_in0)
        index_string_in1 = ", ".join(load_indices_in1)
        shape_string_in1 = ", ".join(shape_load_in1)
        index_string_store = ", ".join(store_indices)

        return index_string_in0, shape_string_in0, index_string_in1, shape_string_in1, index_string_store


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
        permute_map_in0 = []
        permute_map_in1 = []
        permute_map_out = []

        # in0 map: find positions of prim dimensions in stride-sorted tensor view
        # ct.mma expects (K, M) order for left matrix
        for i in range(self.cv.num_prim_k):
            prim_dim_id = self.cv.prim_k_ids[i]
            index_where_prim_id_matches = self.cv.stride_sorted_indices_in0.index(prim_dim_id)
            permute_map_in0.append(index_where_prim_id_matches)

        for i in range(self.cv.num_prim_m):
            prim_dim_id = self.cv.prim_m_ids[i]
            index_where_prim_id_matches = self.cv.stride_sorted_indices_in0.index(prim_dim_id)
            permute_map_in0.append(index_where_prim_id_matches)

        # in1 map: find positions of prim dimensions in stride-sorted tensor view
        # ct.mma expects (N, K) order for right matrix
        for i in range(self.cv.num_prim_n):
            prim_dim_id = self.cv.prim_n_ids[i]
            index_where_prim_id_matches = self.cv.stride_sorted_indices_in1.index(prim_dim_id)
            permute_map_in1.append(index_where_prim_id_matches)

        for i in range(self.cv.num_prim_k):
            prim_dim_id = self.cv.prim_k_ids[i]
            index_where_prim_id_matches = self.cv.stride_sorted_indices_in1.index(prim_dim_id)
            permute_map_in1.append(index_where_prim_id_matches)


        # check if permute maps are in order; if so, no permute is needed
        if permute_map_in0 == sorted(permute_map_in0):
            permute_map_in0 = None

        if permute_map_in1 == sorted(permute_map_in1):
            permute_map_in1 = None


        # if permute maps are not None, inject the other dimension indices (dimensions are all of size 1) to the front
        if permute_map_in0 is not None:
            append_to_the_front_of_permute_map_in0 = []
            for i in range(self.cv.num_dimensions_in0):
                if not i in permute_map_in0:
                    append_to_the_front_of_permute_map_in0.append(i)

            permute_map_in0 = append_to_the_front_of_permute_map_in0 + permute_map_in0

        if permute_map_in1 is not None:
            append_to_the_front_of_permute_map_in1 = []
            for i in range(self.cv.num_dimensions_in1):
                if not i in permute_map_in1:
                    append_to_the_front_of_permute_map_in1.append(i)

            permute_map_in1 = append_to_the_front_of_permute_map_in1 + permute_map_in1


        # generate tupels if not none
        permute_map_in0_tuple = None
        permute_map_in1_tuple = None

        if permute_map_in0 is not None:
            permute_map_in0_tuple = tuple(permute_map_in0)
        if permute_map_in1 is not None:
            permute_map_in1_tuple = tuple(permute_map_in1)


        return permute_map_in0_tuple, permute_map_in1_tuple


    def get_reshape_and_permute_map_out(self):
        # out matrix has shape kernel_n x kernel_m
        # we need to first reshape it to (1, 1, ..., prim_n1, 1, prim_n2, ..., 1, prim_m1, 1, prim_m2)
        #    where 1 are all shared dimensions and prims follow the relative order from the matrix output
        #    the prim dimensions in the primitive dimension slots of the output tensor

        # afterwards we need to permute the primitive dimensions in the reshaped tile so that they are in the same order as the output tensor configuration

        reshape_out = [1 for i in range(self.cv.num_dimensions_out)]
        # ordered by stride (highest first) to match tensor view dimension order
        prim_ids_in_out = self.cv.prim_n_ids + self.cv.prim_m_ids
        current_prim_count = 0
        prim_indices = []

        for i in range(self.cv.num_dimensions_out):
            current_out_config_index = self.cv.stride_sorted_indices_out[i]

            if self.cv.exec_types[current_out_config_index] != etops.exec.prim:
                continue

            prim_indices.append(i)

            current_prim_id = prim_ids_in_out[current_prim_count]
            current_dim_size = self.cv.dim_sizes[current_prim_id]
            reshape_out[i] = find_next_power_of_2(current_dim_size)

            current_prim_count += 1


        prim_indices_reordered = []
        for i in range(len(self.cv.prim_stride_sorted_indices_out)):
            current_prim_config_index = self.cv.prim_stride_sorted_indices_out[i]
            index_where_current_prim_config_index_matches_prim_id = prim_ids_in_out.index(current_prim_config_index)
            prim_indices_reordered.append(prim_indices[index_where_current_prim_config_index_matches_prim_id])

        if len(prim_indices_reordered) != len(self.cv.prim_stride_sorted_indices_out) or len(prim_indices_reordered) != len(prim_indices):
            raise ValueError("Error in permute map for out: Length of reordered prim indices does not match number of prim dimensions in out or length of prim config indices in out")

        permute_map_out = []
        current_prim_reordered_count = 0

        for i in range(self.cv.num_dimensions_out):
            current_out_config_index = self.cv.stride_sorted_indices_out[i]

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


