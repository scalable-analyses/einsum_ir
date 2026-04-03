import etops
import math


class ConfigHelper:
    def __init__(self, config):
        """
        Initializes the ConfigHelper from a TensorOperationConfig.
        Extracts and owns all mutable dimension/stride metadata and provides
        helper methods to fuse, split, swap and permute dimensions.
        """

        # check dimensionality of strides
        # currently no support for multiple memory layers
        if len(config.strides) > 1:
            raise ValueError("Currently only support for one memory layer for the cuTile backend.")

        # Non-list fields needed to reconstruct a TensorOperationConfig
        self.backend    = config.backend
        self.data_type  = config.data_type
        self.prim_first = config.prim_first
        self.prim_main  = config.prim_main
        self.prim_last  = config.prim_last

        self.exec_types = list(config.exec_types)
        self.dim_types  = list(config.dim_types)
        self.dim_sizes  = list(config.dim_sizes)

        self.strides_left  = list(config.strides[0][0])
        self.strides_right = list(config.strides[0][1])
        self.strides_out   = list(config.strides[0][2])

        self.divisors = []
        self._init_divisors()

        self.M_size_total = 1
        self.N_size_total = 1
        self.K_size_total = 1
        self.C_size_total = 1
        self._init_CMNK_sizes()

        self.left_size_total  = self.cfg.C_size_total * self.cfg.M_size_total * self.cfg.K_size_total
        self.right_size_total = self.cfg.C_size_total * self.cfg.K_size_total * self.cfg.N_size_total
        self.out_size_total   = self.cfg.C_size_total * self.cfg.M_size_total * self.cfg.N_size_total

        self.elements_total = self.left_size_total + self.right_size_total + self.out_size_total



    # =========================================================================
    # Divisor helpers
    # =========================================================================

    def _init_divisors(self):
        """
        Rebuild the list of divisors for every dimension from scratch.
        A divisor for a dimension is any integer that divides the dimension
        size, is greater than 0, and is at most the dimension size.
        """

        self.divisors = []

        for i in range(len(self.dim_sizes)):
            dim_divisors = self._get_divisors_for_dim(i)
            self.divisors.append(dim_divisors)



    def _get_divisors_for_dim(self, id):
        """
        Return the sorted list of divisors for dimension *id*.
        """

        dim_size = self.dim_sizes[id]

        dim_divisors = []
        for i in range(1, math.isqrt(dim_size) + 1):
            if dim_size % i == 0:
                dim_divisors.append(i)
                if i != dim_size // i:
                    dim_divisors.append(dim_size // i)

        return dim_divisors



    # =========================================================================
    # CMNK size helpers
    # =========================================================================

    def _init_CMNK_sizes(self):
        """
        (Re-)compute the total number of elements in the M, N, K, and C
        dimensions from the current dim_types / dim_sizes lists.
        """

        self.M_size_total = 1
        self.N_size_total = 1
        self.K_size_total = 1
        self.C_size_total = 1

        for i in range(len(self.dim_types)):
            dim_type = self.dim_types[i]
            dim_size = self.dim_sizes[i]

            if dim_type == etops.dim.m:
                self.M_size_total *= dim_size
            elif dim_type == etops.dim.n:
                self.N_size_total *= dim_size
            elif dim_type == etops.dim.k:
                self.K_size_total *= dim_size
            elif dim_type == etops.dim.c:
                self.C_size_total *= dim_size



    # =========================================================================
    # Fusability check
    # =========================================================================

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

        stride_0_left  = self.strides_left[id_0]
        stride_1_left  = self.strides_left[id_1]
        stride_0_right = self.strides_right[id_0]
        stride_1_right = self.strides_right[id_1]
        stride_0_out   = self.strides_out[id_0]
        stride_1_out   = self.strides_out[id_1]

        appear_in_left  = stride_0_left  != 0 and stride_1_left  != 0
        appear_in_right = stride_0_right != 0 and stride_1_right != 0
        appear_in_out   = stride_0_out   != 0 and stride_1_out   != 0

        # Track whether id_0 is the inner (lower-stride) dimension.
        # None  = not yet determined.
        # True  = id_0 is inner, id_1 is outer  (stride_0 < stride_1).
        # False = id_1 is inner, id_0 is outer  (stride_1 < stride_0).
        # All tensors where both dimensions appear must agree on this ordering;
        # otherwise fusing them would reorder the logical layout across tensors.
        inner_is_id_0 = None

        for appear, s0, s1 in (
            (appear_in_left,  stride_0_left,  stride_1_left),
            (appear_in_right, stride_0_right, stride_1_right),
            (appear_in_out,   stride_0_out,   stride_1_out),
        ):
            if not appear:
                continue

            if s0 == dim_size_1 * s1:
                # id_1 is inner (lower stride), id_0 is outer
                this_inner_is_id_0 = False
            elif s1 == dim_size_0 * s0:
                # id_0 is inner (lower stride), id_1 is outer
                this_inner_is_id_0 = True
            else:
                # dimensions are not adjacent in this tensor
                return False

            if inner_is_id_0 is None:
                inner_is_id_0 = this_inner_is_id_0
            elif inner_is_id_0 != this_inner_is_id_0:
                # ordering is inconsistent across tensors — cannot fuse
                return False

        return True



    # =========================================================================
    # Fuse helpers
    # =========================================================================

    def _fuse_dimensions(self, id_0, id_1, new_exec_type, error_if_not_fusable=True):
        """
        Fuse two dimensions together.
        New dimension will have size  = size_0 * size_1
        New dimension will have stride = min(stride_0, stride_1)
        New dimension will have exec type = new_exec_type
        """

        if not self._are_fusable(id_0, id_1):
            if error_if_not_fusable:
                raise ValueError(f"Dimensions {id_0} and {id_1} cannot be fused together.")
            return

        new_dim_size    = self.dim_sizes[id_0]    * self.dim_sizes[id_1]
        new_stride_left  = min(self.strides_left[id_0],  self.strides_left[id_1])
        new_stride_right = min(self.strides_right[id_0], self.strides_right[id_1])
        new_stride_out   = min(self.strides_out[id_0],   self.strides_out[id_1])

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
        self.dim_sizes[new_id]    = new_dim_size
        self.strides_left[new_id]  = new_stride_left
        self.strides_right[new_id] = new_stride_right
        self.strides_out[new_id]   = new_stride_out
        self.divisors[new_id]     = self._get_divisors_for_dim(new_id)
        self.exec_types[new_id]   = new_exec_type



    def _fuse_all_fusable_dimensions(self, exec_type_filter=None, new_exec_type=None):
        """
        Fuse all fusable dimensions together.
        Fused dimensions will be fused together iteratively until no more
        fusable dimensions can be found.  Iterates through all pairs of
        dimensions per dimension type.
        Runtime is O(n^3) per dimension type.
        """

        self._fuse_all_fusable_dimensions_for_dim_type(etops.dim.m, exec_type_filter=exec_type_filter, new_exec_type=new_exec_type)
        self._fuse_all_fusable_dimensions_for_dim_type(etops.dim.n, exec_type_filter=exec_type_filter, new_exec_type=new_exec_type)
        self._fuse_all_fusable_dimensions_for_dim_type(etops.dim.k, exec_type_filter=exec_type_filter, new_exec_type=new_exec_type)
        self._fuse_all_fusable_dimensions_for_dim_type(etops.dim.c, exec_type_filter=exec_type_filter, new_exec_type=new_exec_type)



    def _fuse_all_fusable_dimensions_for_dim_type(self, dim_type, exec_type_filter=None, new_exec_type=None):
        """
        Fuse all fusable dimensions of a given dimension type together.
        Fused dimensions will be fused together iteratively until no more
        fusable dimensions can be found.
        """

        if exec_type_filter is not None and new_exec_type is None:
            new_exec_type = exec_type_filter

        # keep track of whether any dimensions were fused in the current iteration
        fused_in_iteration = True

        while fused_in_iteration:
            dim_ids = [
                i for i, dt in enumerate(self.dim_types)
                if dt == dim_type and (exec_type_filter is None or self.exec_types[i] == exec_type_filter)
            ]

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



    # =========================================================================
    # Split helpers
    # =========================================================================

    def _split_multiple_dimensions(self, dim_ids, splits, new_exec_type_left=None, new_exec_type_right=None):
        """
        Split multiple dimensions at once.

        Parameters:
        - dim_ids: list of original dimension IDs (positions before any of the
                   splits in this call are applied).
        - splits:  2D list of length len(dim_ids).  splits[i] = [dim_size_0, dim_size_1]
                   are the two sizes that dimension dim_ids[i] is split into, where
                   dim_size_0 * dim_size_1 must equal the current size of that dimension.

        ID-shift handling:
        Splits are applied in ascending order of the original IDs.  Because each split
        inserts one new dimension directly after the split position, every subsequent
        original ID is shifted by one for each earlier split that was performed at a
        strictly lower index.  When processed in ascending order this offset equals the
        running split count, so the actual ID used for split i is original_id + i.
        """

        if len(dim_ids) != len(splits):
            raise ValueError(
                f"_split_multiple_dimensions: dim_ids (len={len(dim_ids)}) and "
                f"splits (len={len(splits)}) must have the same length."
            )

        for i, split in enumerate(splits):
            if len(split) != 2:
                raise ValueError(
                    f"_split_multiple_dimensions: splits[{i}] must contain exactly "
                    f"two values [dim_size_0, dim_size_1], got {split}."
                )

        # Process in ascending order of original IDs so that the offset is simply
        # the number of splits already applied (all of which were at lower indices).
        sorted_pairs = sorted(zip(dim_ids, splits), key=lambda pair: pair[0])

        for offset, (original_id, split) in enumerate(sorted_pairs):
            actual_id = original_id + offset
            self._split_dimension(actual_id, split[0], split[1], new_exec_type_left=new_exec_type_left, new_exec_type_right=new_exec_type_right)



    def _split_dimension(self, id, dim_size_0, dim_size_1, new_exec_type_left=None, new_exec_type_right=None):
        """
        Split a dimension into two dimensions.
        New dimensions will have sizes dim_size_0 and dim_size_1.
        New right dimension will have stride = old_stride.
        New left  dimension will have stride = old_stride * dim_size_1.
        New dimensions inherit exec types and dim types from old dimension.
        """

        if self.dim_sizes[id] != dim_size_0 * dim_size_1:
            raise ValueError(
                f"Dimension {id} cannot be split into sizes {dim_size_0} and "
                f"{dim_size_1}. Size mismatch."
            )

        old_stride_left  = self.strides_left[id]
        old_stride_right = self.strides_right[id]
        old_stride_out   = self.strides_out[id]

        # update or insert new dimensions into lists
        self.dim_sizes[id]     = dim_size_0
        self.strides_left[id]  = old_stride_left  * dim_size_1
        self.strides_right[id] = old_stride_right * dim_size_1
        self.strides_out[id]   = old_stride_out   * dim_size_1
        self.divisors[id]      = self._get_divisors_for_dim(id)
        self.exec_types[id]    = new_exec_type_left if new_exec_type_left is not None else self.exec_types[id]

        self.dim_types.insert(id + 1,     self.dim_types[id])
        self.exec_types.insert(id + 1,    new_exec_type_right if new_exec_type_right is not None else self.exec_types[id])
        self.dim_sizes.insert(id + 1,     dim_size_1)
        self.strides_left.insert(id + 1,  old_stride_left)
        self.strides_right.insert(id + 1, old_stride_right)
        self.strides_out.insert(id + 1,   old_stride_out)
        self.divisors.insert(id + 1,      self._get_divisors_for_dim(id + 1))



    # =========================================================================
    # Swap / permute helpers
    # =========================================================================

    def _swap_dimensions(self, id_0, id_1):
        """
        Swap two dimensions.
        """

        if id_0 == id_1:
            return

        self.dim_types[id_0],    self.dim_types[id_1]    = self.dim_types[id_1],    self.dim_types[id_0]
        self.exec_types[id_0],   self.exec_types[id_1]   = self.exec_types[id_1],   self.exec_types[id_0]
        self.dim_sizes[id_0],    self.dim_sizes[id_1]    = self.dim_sizes[id_1],    self.dim_sizes[id_0]
        self.strides_left[id_0],  self.strides_left[id_1]  = self.strides_left[id_1],  self.strides_left[id_0]
        self.strides_right[id_0], self.strides_right[id_1] = self.strides_right[id_1], self.strides_right[id_0]
        self.strides_out[id_0],   self.strides_out[id_1]   = self.strides_out[id_1],   self.strides_out[id_0]
        self.divisors[id_0],     self.divisors[id_1]     = self.divisors[id_1],     self.divisors[id_0]



    def _permute_dimensions(self, new_order):
        """
        Permute dimensions according to new_order.
        new_order is a list of dimension IDs in the new order.
        """

        if sorted(new_order) != list(range(len(self.dim_types))):
            raise ValueError(f"Invalid new order {new_order}. Must be a permutation of dimension IDs.")

        self.dim_types    = [self.dim_types[i]    for i in new_order]
        self.exec_types   = [self.exec_types[i]   for i in new_order]
        self.dim_sizes    = [self.dim_sizes[i]    for i in new_order]
        self.strides_left  = [self.strides_left[i]  for i in new_order]
        self.strides_right = [self.strides_right[i] for i in new_order]
        self.strides_out   = [self.strides_out[i]   for i in new_order]
        self.divisors     = [self.divisors[i]     for i in new_order]



    # =========================================================================
    # Config reconstruction
    # =========================================================================

    def get_config(self):
        """
        Reconstruct and return a TensorOperationConfig from the current
        (potentially fused / split / permuted) dimension and stride lists.
        """

        strides = ((tuple(self.strides_left), tuple(self.strides_right), tuple(self.strides_out)),)

        return etops.TensorOperationConfig(
            backend    = self.backend,
            data_type  = self.data_type,
            prim_first = self.prim_first,
            prim_main  = self.prim_main,
            prim_last  = self.prim_last,
            dim_types  = tuple(self.dim_types),
            exec_types = tuple(self.exec_types),
            dim_sizes  = tuple(self.dim_sizes),
            strides    = strides
        )
