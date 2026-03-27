import etops
from etops.backends._cutile.config_parser import ConfigParser
import cupy as cp
import math



class Optimizer:
    def __init__(self, config):
        # Initialize GPU properties
        device_id = cp.cuda.Device().id
        props = cp.cuda.runtime.getDeviceProperties(device_id)

        self.sm_count = props['multiProcessorCount']

        # byte values for memory sizes
        self.global_mem_size = props['totalGlobalMem']
        self.l2_cache_size = props['l2CacheSize']

        self.config = config
        self.cv_initial = ConfigParser(self.config, verify_input=True)

        self.data_type = self.config.data_type
        self.prim_main = self.config.prim_main

        self.exec_types = list(self.config.exec_types)
        self.dim_types = list(self.config.dim_types)
        self.dim_sizes = list(self.config.dim_sizes)
        self.divisors = []
        self._init_divisors()

        # check dimensionality of strides
        # currently no support for multiple memory layers
        if len(self.config.strides) > 1:
            raise ValueError("Currently only support for one memory layer for the cuTile backend.")

        self.strides_left = list(self.config.strides[0][0])
        self.strides_right = list(self.config.strides[0][1])
        self.strides_out = list(self.config.strides[0][2])



    def optimize(self):
        """
        Run the optimization pass on the configuration.
        """

        # initialize all exec types to seq
        self.exec_types = [etops.exec.seq] * len(self.exec_types)

        # Fuse all fusable dimensions together
        self._fuse_all_fusable_dimensions(new_exec_type=etops.exec.seq)

        self._find_primitive_dimension()
        self._optimize_for_l2_cache()


    
    def _find_primitive_dimension(self):
        """
        Find the primitive dimension for the main primitive.
        """

        pass
    


    def _optimize_for_l2_cache(self):
        """
        Optimize the configuration for L2 cache size by splitting/fusing dimensions.
        The goal is to maximize L2 cache utilization and reuse.
        """

        pass



    def get_optimized_config(self):
        """
        Returns the optimized configuration
        """

        # convert strides back to tuple of tuples format
        strides = ((tuple(self.strides_left), tuple(self.strides_right), tuple(self.strides_out)),)

        optimized_config = etops.TensorOperationConfig(
            backend    =   self.config.backend,
            data_type  =   self.data_type,
            prim_first =   self.config.prim_first,
            prim_main  =   self.prim_main,
            prim_last  =   self.config.prim_last,
            dim_types  =   tuple(self.dim_types),
            exec_types =   tuple(self.exec_types),
            dim_sizes  =   tuple(self.dim_sizes),
            strides    =   strides
        )

        return optimized_config



    def _are_fusable(self, id_0, id_1):
        """
        Check if two dimensions can be fused together.
        Two dimensions can be fused together if:
        - They have the same dim type
        - They are adjacent to each other in all tensors they appear in
        """

        if id_0 == id_1:
            return False

        dim_type_0 = self.dim_types[id_0]
        dim_type_1 = self.dim_types[id_1]

        if dim_type_0 != dim_type_1:
            return False

        dim_size_0 = self.dim_sizes[id_0]
        dim_size_1 = self.dim_sizes[id_1]

        stride_0_left = self.strides_left[id_0]
        stride_1_left = self.strides_left[id_1]
        stride_0_right = self.strides_right[id_0]
        stride_1_right = self.strides_right[id_1]
        stride_0_out = self.strides_out[id_0]
        stride_1_out = self.strides_out[id_1]

        appear_in_left = stride_0_left != 0 and stride_1_left != 0
        appear_in_right = stride_0_right != 0 and stride_1_right != 0
        appear_in_out = stride_0_out != 0 and stride_1_out != 0

        are_fusable = True

        if appear_in_left:
            are_fusable_left = (stride_0_left == dim_size_1 * stride_1_left) or (stride_1_left == dim_size_0 * stride_0_left)
            are_fusable = are_fusable and are_fusable_left
        
        if appear_in_right:
            are_fusable_right = (stride_0_right == dim_size_1 * stride_1_right) or (stride_1_right == dim_size_0 * stride_0_right)
            are_fusable = are_fusable and are_fusable_right
        
        if appear_in_out:
            are_fusable_out = (stride_0_out == dim_size_1 * stride_1_out) or (stride_1_out == dim_size_0 * stride_0_out)
            are_fusable = are_fusable and are_fusable_out

        return are_fusable



    def _fuse_dimensions(self, id_0, id_1, new_exec_type, error_if_not_fusable=True):
        """
        Fuse two dimensions together.
        New dimension will have size = size_0 * size_1
        New dimension will have stride = min(stride_0, stride_1)
        New dimension will have exec type = new_exec_type
        """

        if not self._are_fusable(id_0, id_1):
            if error_if_not_fusable:
                raise ValueError(f"Dimensions {id_0} and {id_1} cannot be fused together.")
            return

        new_dim_size = self.dim_sizes[id_0] * self.dim_sizes[id_1]
        new_stride_left = min(self.strides_left[id_0], self.strides_left[id_1])
        new_stride_right = min(self.strides_right[id_0], self.strides_right[id_1])
        new_stride_out = min(self.strides_out[id_0], self.strides_out[id_1])
        
        # new dimension ID will be the smaller of the two IDs
        new_id = min(id_0, id_1)

        # remove larger ID from the lists, this also updates the global strides tensor
        remove_id = max(id_0, id_1)
        del self.dim_types[remove_id]
        del self.exec_types[remove_id]
        del self.dim_sizes[remove_id]
        del self.strides_left[remove_id]
        del self.strides_right[remove_id]
        del self.strides_out[remove_id]
        del self.divisors[remove_id]

        # update lists
        self.dim_sizes[new_id] = new_dim_size
        self.strides_left[new_id] = new_stride_left
        self.strides_right[new_id] = new_stride_right
        self.strides_out[new_id] = new_stride_out
        self.divisors[new_id] = self._get_divisors_for_dim(new_id)

    

    def _split_dimension(self, id, dim_size_0, dim_size_1):
        """
        Split a dimension into two dimensions.
        New dimensions will have sizes dim_size_0 and dim_size_1
        New right dimension will have stride = old_stride
        New left dimension will have stride = old_stride * dim_size_1
        New dimensions inherit exec types and dim types from old dimension
        """

        if self.dim_sizes[id] != dim_size_0 * dim_size_1:
            raise ValueError(f"Dimension {id} cannot be split into sizes {dim_size_0} and {dim_size_1}. Size mismatch.")

        old_stride_left = self.strides_left[id]
        old_stride_right = self.strides_right[id]
        old_stride_out = self.strides_out[id]

        # update or insert new dimensions into lists, this also updates the global strides tensor
        self.dim_sizes[id] = dim_size_0
        self.strides_left[id] = old_stride_left * dim_size_1
        self.strides_right[id] = old_stride_right * dim_size_1
        self.strides_out[id] = old_stride_out * dim_size_1
        self.divisors[id] = self._get_divisors_for_dim(id)

        self.dim_types.insert(id + 1, self.dim_types[id])
        self.exec_types.insert(id + 1, self.exec_types[id])
        self.dim_sizes.insert(id + 1, dim_size_1)
        self.strides_left.insert(id + 1, old_stride_left)
        self.strides_right.insert(id + 1, old_stride_right)
        self.strides_out.insert(id + 1, old_stride_out)
        self.divisors.insert(id + 1, self._get_divisors_for_dim(id + 1))



    def _swap_dimensions(self, id_0, id_1):
        """
        Swap two dimensions.
        """

        if id_0 == id_1:
            return

        # Swap the dimensions in all lists
        self.dim_types[id_0], self.dim_types[id_1] = self.dim_types[id_1], self.dim_types[id_0]
        self.exec_types[id_0], self.exec_types[id_1] = self.exec_types[id_1], self.exec_types[id_0]
        self.dim_sizes[id_0], self.dim_sizes[id_1] = self.dim_sizes[id_1], self.dim_sizes[id_0]
        self.strides_left[id_0], self.strides_left[id_1] = self.strides_left[id_1], self.strides_left[id_0]
        self.strides_right[id_0], self.strides_right[id_1] = self.strides_right[id_1], self.strides_right[id_0]
        self.strides_out[id_0], self.strides_out[id_1] = self.strides_out[id_1], self.strides_out[id_0]
        self.divisors[id_0], self.divisors[id_1] = self.divisors[id_1], self.divisors[id_0]



    def _permute_dimensions(self, new_order):
        """
        Permute dimensions according to new_order.
        new_order is a list of dimension IDs in the new order.
        """

        if sorted(new_order) != list(range(len(self.dim_types))):
            raise ValueError(f"Invalid new order {new_order}. Must be a permutation of dimension IDs.")
        
        # Permute the dimensions in all lists according to new_order
        self.dim_types = [self.dim_types[i] for i in new_order]
        self.exec_types = [self.exec_types[i] for i in new_order]
        self.dim_sizes = [self.dim_sizes[i] for i in new_order]
        self.strides_left = [self.strides_left[i] for i in new_order]
        self.strides_right = [self.strides_right[i] for i in new_order]
        self.strides_out = [self.strides_out[i] for i in new_order]
        self.divisors = [self.divisors[i] for i in new_order]



    def _fuse_all_fusable_dimensions(self, exec_type_filter=None, new_exec_type=None):
        """
        Fuse all fusable dimensions together.
        Fused dimensions will be fused together iteratively until no more fusable dimensions can be found.
        Iterates through all pairs of dimensions per dimension type.
        Runtime is O(n^3) per dimension type, so O( (max(num_dims_m, num_dims_n, num_dims_k, num_dims_c))^3 ).
        """

        self._fuse_all_fusable_dimensions_for_dim_type(etops.dim.m, exec_type_filter=exec_type_filter, new_exec_type=new_exec_type)
        self._fuse_all_fusable_dimensions_for_dim_type(etops.dim.n, exec_type_filter=exec_type_filter, new_exec_type=new_exec_type)
        self._fuse_all_fusable_dimensions_for_dim_type(etops.dim.k, exec_type_filter=exec_type_filter, new_exec_type=new_exec_type)
        self._fuse_all_fusable_dimensions_for_dim_type(etops.dim.c, exec_type_filter=exec_type_filter, new_exec_type=new_exec_type)


    
    def _fuse_all_fusable_dimensions_for_dim_type(self, dim_type, exec_type_filter=None, new_exec_type=None):
        """
        Fuse all fusable dimensions of a given dimension type together.
        Fused dimensions will be fused together iteratively until no more fusable dimensions can be found.
        """

        if exec_type_filter is not None and new_exec_type is None:
            new_exec_type = exec_type_filter

        # keep track of whether any dimensions were fused in the current iteration
        fused_in_iteration = True

        while fused_in_iteration:
            dim_ids = [i for i, dt in enumerate(self.dim_types) if dt == dim_type and (exec_type_filter is None or self.exec_types[i] == exec_type_filter)]

            fused_in_iteration = False
            break_for_loops = False
            
            for i in range(len(dim_ids)):
                id_0 = dim_ids[i]

                for j in range(i + 1, len(dim_ids)):
                    id_1 = dim_ids[j]

                    if self._are_fusable(id_0, id_1):
                        self._fuse_dimensions(id_0, id_1, new_exec_type=new_exec_type)

                        fused_in_iteration = True
                        break_for_loops = True
                        break
                    
                if break_for_loops:
                    break



    def _init_divisors(self):
        """
        Update the list of divisors for each dimension based on the current configuration.
        A divisor for a dimension is a number that divides the dimension size and is less than or equal to the dimension size, and larger than 1.
        """

        self.divisors = []

        for i in range(len(self.dim_sizes)):
            dim_divisors = self._get_divisors_for_dim(i)
            self.divisors.append(dim_divisors)



    def _get_divisors_for_dim(self, id):
        """
        Get the list of divisors for a given dimension ID.
        """

        dim_size = self.dim_sizes[id]

        dim_divisors = []
        for i in range(2, math.isqrt(dim_size) + 1):
            if dim_size % i == 0:
                dim_divisors.append(i)
                if i != dim_size // i:
                    dim_divisors.append(dim_size // i)
        
        return dim_divisors
