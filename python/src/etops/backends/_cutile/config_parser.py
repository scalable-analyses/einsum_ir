import etops

class ConfigParser:
    """
    Class Attributes:
        config
        backend
        data_type
        prim_first
        prim_main
        prim_last
        dim_types
        exec_types
        dim_sizes
        strides
        strides_left
        strides_right
        strides_output
        outer_loops_M
        outer_loops_N
        outer_loops_K
        outer_loops_B
        shared_loops_M
        shared_loops_N
        shared_loops_K
        shared_loops_B
        seq_loops_M
        seq_loops_N
        seq_loops_K
        seq_loops_B
        prim_m_ids
        prim_n_ids
        prim_k_ids
        prim_config_indices_left
        prim_config_indices_right
        prim_config_indices_output
        prim_dim_ids
        shared_loop_ids
        seq_loop_ids
        num_shared_outer_loops
        num_seq_outer_loops
        num_outer_loops
        num_shared_loops_M
        num_shared_loops_N
        num_shared_loops_K
        num_shared_loops_B
        num_prim_m
        num_prim_n
        num_prim_k
        num_prim_dimensions
        num_seq_loops_M
        num_seq_loops_N
        num_seq_loops_K
        num_seq_loops_B
        num_loops_M
        num_loops_N
        num_loops_K
        num_loops_B
        num_dimensions
        num_dimensions_left
        num_dimensions_right
        num_dimensions_output
        kernel_shape_m
        kernel_shape_n
        kernel_shape_k
        shared_loop_strides
        grid_size
        tensor_shape_left
        tensor_shape_right
        tensor_shape_output
        config_indices_in_left_tensor
        config_indices_in_right_tensor
        config_indices_in_out_tensor
    """

    def __init__(self, config, verify_input=True, verify_executable=False):
        self.config = config

        if verify_executable:
            self.verify_executable_config()
        elif verify_input:
            self.verify_input_config()

        self.init_config_arguments()
        self.init_config_loops()
        self.init_dimension_values()
        self.init_loop_strides()
        self.init_tensor_shapes()
        self.init_config_indices_in_tensor()
        self.init_grid()


    def init_config_arguments(self):
        self.backend = self.config.backend
        self.data_type = self.config.data_type
        self.prim_first = self.config.prim_first
        self.prim_main = self.config.prim_main
        self.prim_last = self.config.prim_last
        self.dim_types = self.config.dim_types
        self.exec_types = self.config.exec_types
        self.dim_sizes = self.config.dim_sizes
        self.strides = self.config.strides
        self.strides_left = self.config.strides[0][0]
        self.strides_right = self.config.strides[0][1]
        self.strides_output = self.config.strides[0][2]

    
    def init_config_loops(self):
        # all outher loops (shared and seq) combined; per exec type
        self.outer_loops_M = []
        self.outer_loops_N = []
        self.outer_loops_K = []
        self.outer_loops_B = []

        # shared outer loops; per exec type
        self.shared_loops_M = []
        self.shared_loops_N = []
        self.shared_loops_K = []
        self.shared_loops_B = []

        # seq outer loops; per exec type
        self.seq_loops_M = []
        self.seq_loops_N = []
        self.seq_loops_K = []
        self.seq_loops_B = []

        # outer loops per dim type
        self.shared_loop_ids = []
        self.seq_loop_ids = []

        # prim ids
        self.prim_m_ids = []
        self.prim_n_ids = []
        self.prim_k_ids = []
        self.prim_dim_ids = []

        for i in range(len(self.exec_types)):
            if self.exec_types[i] == etops.exec.prim:
                continue

            if self.exec_types[i] == etops.exec.shared:
                self.shared_loop_ids.append(i)

            elif self.exec_types[i] == etops.exec.seq:
                self.seq_loop_ids.append(i)

            if self.dim_types[i] == etops.dim.m:
                self.outer_loops_M.append(i)
                if self.exec_types[i] == etops.exec.shared:
                    self.shared_loops_M.append(i)
                elif self.exec_types[i] == etops.exec.seq:
                    self.seq_loops_M.append(i)

            elif self.dim_types[i] == etops.dim.n:
                self.outer_loops_N.append(i)
                if self.exec_types[i] == etops.exec.shared:
                    self.shared_loops_N.append(i)
                elif self.exec_types[i] == etops.exec.seq:
                    self.seq_loops_N.append(i)

            elif self.dim_types[i] == etops.dim.k:
                self.outer_loops_K.append(i)
                if self.exec_types[i] == etops.exec.shared:
                    self.shared_loops_K.append(i)
                elif self.exec_types[i] == etops.exec.seq:
                    self.seq_loops_K.append(i)

            elif self.dim_types[i] == etops.dim.c:
                self.outer_loops_B.append(i)
                if self.exec_types[i] == etops.exec.shared:
                    self.shared_loops_B.append(i)
                elif self.exec_types[i] == etops.exec.seq:
                    self.seq_loops_B.append(i)

        # prim IDs
        for i in range(len(self.exec_types)):
            if self.exec_types[i] != etops.exec.prim:
                continue
            
            self.prim_dim_ids.append(i)
            if self.dim_types[i] == etops.dim.m:
                self.prim_m_ids.append(i)
            elif self.dim_types[i] == etops.dim.n:
                self.prim_n_ids.append(i)
            elif self.dim_types[i] == etops.dim.k:
                self.prim_k_ids.append(i)


    def init_dimension_values(self):
        self.num_shared_outer_loops = len(self.shared_loop_ids)
        self.num_seq_outer_loops = len(self.seq_loop_ids)
        self.num_outer_loops = self.num_shared_outer_loops + self.num_seq_outer_loops
        
        self.num_shared_loops_M = len(self.shared_loops_M)
        self.num_shared_loops_N = len(self.shared_loops_N)
        self.num_shared_loops_K = len(self.shared_loops_K)
        self.num_shared_loops_B = len(self.shared_loops_B)
        self.num_seq_loops_M = len(self.seq_loops_M)
        self.num_seq_loops_N = len(self.seq_loops_N)
        self.num_seq_loops_K = len(self.seq_loops_K)
        self.num_seq_loops_B = len(self.seq_loops_B)

        self.num_loops_M = self.num_shared_loops_M + self.num_seq_loops_M
        self.num_loops_N = self.num_shared_loops_N + self.num_seq_loops_N
        self.num_loops_K = self.num_shared_loops_K + self.num_seq_loops_K
        self.num_loops_B = self.num_shared_loops_B + self.num_seq_loops_B

        self.num_prim_m = len(self.prim_m_ids)
        self.num_prim_n = len(self.prim_n_ids)
        self.num_prim_k = len(self.prim_k_ids)
        self.num_prim_dimensions = self.num_prim_m + self.num_prim_n + self.num_prim_k

        self.num_dimensions = len(self.dim_types)
        self.num_dimensions_left = self.num_loops_B + self.num_loops_M + self.num_loops_K + self.num_prim_m + self.num_prim_k
        self.num_dimensions_right = self.num_loops_B + self.num_loops_K + self.num_loops_N + self.num_prim_n + self.num_prim_k
        self.num_dimensions_output = self.num_loops_B + self.num_loops_M + self.num_loops_N + self.num_prim_m + self.num_prim_n

        self.kernel_shape_m = 1
        self.kernel_shape_n = 1
        self.kernel_shape_k = 1

        for i in range(len(self.dim_types)):
            if self.exec_types[i] == etops.exec.prim:
                if self.dim_types[i] == etops.dim.m:
                    self.kernel_shape_m *= self.dim_sizes[i]
                elif self.dim_types[i] == etops.dim.n:
                    self.kernel_shape_n *= self.dim_sizes[i]
                elif self.dim_types[i] == etops.dim.k:
                    self.kernel_shape_k *= self.dim_sizes[i]


    def init_loop_strides(self):
        self.shared_loop_strides = []
        current_loop_stride = 1

        for i in range(self.num_shared_outer_loops - 1, -1, -1):
            # insert current loop stride to the front of the list of loop strides
            self.shared_loop_strides.insert(0, current_loop_stride)
            current_loop_stride *= self.dim_sizes[self.shared_loop_ids[i]]

    
    def init_tensor_shapes(self):
        self.tensor_shape_left = self.get_tensor_shape(self.dim_sizes, self.strides_left)
        self.tensor_shape_right = self.get_tensor_shape(self.dim_sizes, self.strides_right)
        self.tensor_shape_output = self.get_tensor_shape(self.dim_sizes, self.strides_output)

    
    def get_tensor_shape(self, dim_sizes, strides):
        tensor_shape = []
        current_stride = 1
        for i in range(len(dim_sizes)):
            # find dimension ID with stride equal to current_stride
            for j in range(len(dim_sizes)):
                if strides[j] == current_stride:
                    # insert the corresponding dimension size to the front of the shape
                    tensor_shape.insert(0, dim_sizes[j])
                    break
            
            current_stride *= tensor_shape[0]

        return tensor_shape

    
    def init_config_indices_in_tensor(self):
        config_indices_in_tensors = []
        prim_config_indices_in_tensors = []

        for it_tensor in range(3):
            if it_tensor == 0:
                strides = self.strides_left
            elif it_tensor == 1:
                strides = self.strides_right
            else:
                strides = self.strides_output

            config_indices_in_tensor = []
            prim_config_indices_in_tensor = []

            current_stride = 1
        
            for j in range(self.num_dimensions):
                # find dimension ID with stride equal to current_stride
                for i in range(self.num_dimensions):
                    if strides[i] == current_stride:
                        # update load indices and shape for the corresponding dimension
                        config_indices_in_tensor.insert(0, i)

                        if self.exec_types[i] == etops.exec.prim:
                            prim_config_indices_in_tensor.insert(0, i)

                        current_stride *= self.dim_sizes[i]
                        break
            
            config_indices_in_tensors.append(config_indices_in_tensor)
            prim_config_indices_in_tensors.append(prim_config_indices_in_tensor)

        
        self.config_indices_in_left_tensor = config_indices_in_tensors[0]
        self.config_indices_in_right_tensor = config_indices_in_tensors[1]
        self.config_indices_in_out_tensor = config_indices_in_tensors[2]

        self.prim_config_indices_left = prim_config_indices_in_tensors[0]
        self.prim_config_indices_right = prim_config_indices_in_tensors[1]
        self.prim_config_indices_output = prim_config_indices_in_tensors[2]


    
    def init_grid(self):
        self.grid_size = 1
        for i in range(self.num_dimensions):
            if self.exec_types[i] == etops.exec.shared:
                self.grid_size *= self.dim_sizes[i]


    def verify_input_config(self):
        # check that all arrays have the same length
        if not (len(self.config.dim_types) == len(self.config.exec_types) == len(self.config.dim_sizes) == len(self.config.strides[0][0]) == len(self.config.strides[0][1]) == len(self.config.strides[0][2])):
            raise ValueError("Config Parser Error: Length of dim_types, exec_types, dim_sizes, strides_left, strides_right, and strides_output must all be the same")
        
        # check that dim sizes and stides are positive
        for i in range(len(self.config.dim_sizes)):
            if self.config.dim_sizes[i] <= 0:
                raise ValueError("Config Parser Error: Dimension sizes must be positive")
            if self.config.strides[0][0][i] < 0 or self.config.strides[0][1][i] < 0 or self.config.strides[0][2][i] < 0:
                raise ValueError("Config Parser Error: Strides must be non-negative")
        
        # check that at least one dimension of type m, n, and k is present
        if not any(dim_type == etops.dim.m for dim_type in self.config.dim_types):
            raise ValueError("Config Parser Error: At least one dimension of type m must be present")
        if not any(dim_type == etops.dim.n for dim_type in self.config.dim_types):
            raise ValueError("Config Parser Error: At least one dimension of type n must be present")
        if not any(dim_type == etops.dim.k for dim_type in self.config.dim_types):
            raise ValueError("Config Parser Error: At least one dimension of type k must be present")


    def verify_executable_config(self):
        self.verify_input_config()

        # check that exec types are either shared, seq, or prim
        for exec_type in self.config.exec_types:
            if exec_type != etops.exec.shared and exec_type != etops.exec.seq and exec_type != etops.exec.prim:
                raise ValueError("Config Parser Error: Exec types must be either shared, seq, or prim")

        # check that at least one prim dimension is present for dimension types m, n, k
        if sum(1 for i in range(len(self.config.dim_types)) if self.config.exec_types[i] == etops.exec.prim and self.config.dim_types[i] == etops.dim.m) < 1:
            raise ValueError("Config Parser Error: At least one prim dimension of type m must be present")
        if sum(1 for i in range(len(self.config.dim_types)) if self.config.exec_types[i] == etops.exec.prim and self.config.dim_types[i] == etops.dim.n) < 1:
            raise ValueError("Config Parser Error: At least one prim dimension of type n must be present")
        if sum(1 for i in range(len(self.config.dim_types)) if self.config.exec_types[i] == etops.exec.prim and self.config.dim_types[i] == etops.dim.k) < 1:
            raise ValueError("Config Parser Error: At least one prim dimension of type k must be present")
        
        # check that prim dimensions are the innermost dimensions
        non_prim_seen = False
        for i in range(len(self.config.dim_types) - 1, -1, -1):
            if self.config.exec_types[i] == etops.exec.prim:
                if non_prim_seen:
                    raise ValueError("Config Parser Error: Prim dimensions must be the innermost dimensions")
            else:
                non_prim_seen = True
        
        # check that no shared dimension comes after a seq dimension
        seq_seen = False
        for i in range(len(self.config.dim_types)):
            if self.config.exec_types[i] == etops.exec.seq:
                seq_seen = True
            elif self.config.exec_types[i] == etops.exec.shared and seq_seen:
                raise ValueError("Config Parser Error: No shared dimension can come after a sequential dimension")

        # check that all K dimensions are either seq or prim
        for i in range(len(self.config.dim_types)):
            if self.config.dim_types[i] == etops.dim.k and self.config.exec_types[i] == etops.exec.shared:
                raise ValueError("Config Parser Error: K dimensions cannot be shared")

        # check that all seq K dimensions come after all other seq dimensions
        seq_k_seen = False
        for i in range(len(self.config.dim_types)):
            if self.config.exec_types[i] == etops.exec.seq and self.config.dim_types[i] == etops.dim.k:
                seq_k_seen = True
            elif self.config.exec_types[i] == etops.exec.seq and self.config.dim_types[i] != etops.dim.k and seq_k_seen:
                raise ValueError("Config Parser Error: All sequential K dimensions must come after all other sequential dimensions")


    
    def print_config(self):
        print(f"Backend: {self.backend}")
        print(f"Data Type: {self.data_type}")
        print(f"Primitive First: {self.prim_first}")
        print(f"Primitive Main: {self.prim_main}")
        print(f"Primitive Last: {self.prim_last}")
        print(f"Dim Types: {self.dim_types}")
        print(f"Exec Types: {self.exec_types}")
        print(f"Dim Sizes: {self.dim_sizes}")
        print(f"Strides: {self.strides}")


