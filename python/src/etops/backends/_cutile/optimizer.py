import etops
import cupy as cp
import itertools
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

        self.data_type = self.config.data_type
        if self.data_type == etops.float16:
            self.bytes_per_element_load = 2
        elif self.data_type == etops.bfloat16:
            self.bytes_per_element_load = 2
        elif self.data_type == etops.tfloat32:
            self.bytes_per_element_load = 4
        elif self.data_type == etops.float32:
            self.bytes_per_element_load = 4
        elif self.data_type == etops.float64:
            self.bytes_per_element_load = 8

        else:
            raise ValueError(f"Unsupported data type {self.data_type} for cuTile optimizer.")

        self.bytes_per_element_compute = self.bytes_per_element_load


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

        self.M_size_total = 1
        self.N_size_total = 1
        self.K_size_total = 1
        self.C_size_total = 1
        self._init_CMNK_sizes()

        self.left_size_total = self.C_size_total * self.M_size_total * self.K_size_total
        self.right_size_total = self.C_size_total * self.K_size_total * self.N_size_total
        self.out_size_total = self.C_size_total * self.M_size_total * self.N_size_total

        self.elements_total = self.left_size_total + self.right_size_total + self.out_size_total



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



    def _get_all_possible_dim_size_combinations(self, ids_list, strides_list, strides_list_2=None, pruning_rules=[], merge_rules=[]):
        """
        Returns a list of all possible dimension sizes + their corresponding splits that are a combination of the divisors for the given list of dimension IDs.

        pruning_rules determine pruning rules for the generated combinations. Options are:
            - 'always_include_stride_1': always include the stride-1 dimension in the combinations.
              When strides_list_2 is provided, the stride-1 dimension must be non-trivially split
              in both strides lists.
            - '16B_aligned': only include combinations that are aligned to 16 bytes.
              When strides_list_2 is provided, alignment must hold in both strides lists (AND semantics).
            - 'smaller_than_max_prim_size': only include combinations that are smaller than the maximum primitive bytes

        merge_rules determine which combination to keep if the same total size can be achieved by multiple
        divisor combinations. Options are:
            - 'prefer_best_performance_factor': keep the combination with the best performance score,
              evaluated using padding penalty, min-load-size penalty, and split-count penalty.
              Works with one or two strides lists.
        """

        check_always_include_stride_1 = 'always_include_stride_1' in pruning_rules
        check_smaller_than_max_prim_size = 'smaller_than_max_prim_size' in pruning_rules
        check_aligned_to_16B = '16B_aligned' in pruning_rules
        prefer_best_performance_factor = 'prefer_best_performance_factor' in merge_rules

        MAX_PRIM_BYTES = 256
        MAX_PRIM_ELEMENTS = MAX_PRIM_BYTES // self.bytes_per_element_compute

        # stride_1_index_1 is the *positional* index (0..len(ids_list)-1) inside
        # `combination` where strides_list has stride == 1.
        stride_1_index_1 = None
        for pos, dim_id in enumerate(ids_list):
            if strides_list[dim_id] == 1:
                stride_1_index_1 = pos
                break

        # stride_1_index_2 is the positional index where strides_list_2 has stride == 1.
        stride_1_index_2 = None
        if strides_list_2 is not None:
            for pos, dim_id in enumerate(ids_list):
                if strides_list_2[dim_id] == 1:
                    stride_1_index_2 = pos
                    break

        # Validate always_include_stride_1: both lists must have a stride-1 dim when the rule is active.
        if check_always_include_stride_1 and stride_1_index_1 is None:
            raise ValueError("Pruning rule always_include_stride_1 is set but no stride-1 dimension found in the given ids_list and strides_list.")
        if check_always_include_stride_1 and strides_list_2 is not None and stride_1_index_2 is None:
            raise ValueError("Pruning rule always_include_stride_1 is set but no stride-1 dimension found in the given ids_list and strides_list_2.")

        alignment_size = 16 // self.bytes_per_element_load

        list_of_divisors = []
        for i in ids_list:
            list_of_divisors.append(self.divisors[i])

        splits = []
        total_sizes = []

        # iterator over the cartesian product of the list of divisors
        for combination in itertools.product(*list_of_divisors):
            total_size = 1
            for i in range(len(combination)):
                total_size *= combination[i]

            # check pruning rules
            # ===================

            # check smaller_than_max_prim_size
            if check_smaller_than_max_prim_size and total_size > MAX_PRIM_ELEMENTS:
                continue

            # check always_include_stride_1
            # combination[stride_1_index] is the split chosen for the stride-1 dim;
            # skip combinations where that dimension contributes nothing (split==1).
            if stride_1_index_1 is not None:
                if check_always_include_stride_1 and combination[stride_1_index_1] == 1:
                    continue
            if strides_list_2 is not None and stride_1_index_2 is not None:
                if check_always_include_stride_1 and combination[stride_1_index_2] == 1:
                    continue

            # check 16B alignment for strides_list.
            # For the stride-1 dim: the split size itself must be a multiple of alignment_size
            # (elements are contiguous, so the tile width determines the memory alignment boundary).
            # For all other non-trivially split dims: the stride must be a multiple of alignment_size.
            is_aligned_to_16B = True

            if stride_1_index_1 is not None:
                is_aligned_to_16B = is_aligned_to_16B and combination[stride_1_index_1] % alignment_size == 0

            # pos is the positional index into combination; ids_list[pos] is the
            # corresponding dimension ID used to look up strides_list.
            for pos in range(len(combination)):
                if ((stride_1_index_1 is None or pos != stride_1_index_1) and combination[pos] != 1) or sum(combination) == len(combination):
                    is_aligned_to_16B = is_aligned_to_16B and strides_list[ids_list[pos]] % alignment_size == 0

            # check 16B alignment for strides_list_2 (if provided; must also pass — AND semantics).
            if strides_list_2 is not None:
                if stride_1_index_2 is not None:
                    is_aligned_to_16B = is_aligned_to_16B and combination[stride_1_index_2] % alignment_size == 0
                for pos in range(len(combination)):
                    if ((stride_1_index_2 is None or pos != stride_1_index_2) and combination[pos] != 1) or sum(combination) == len(combination):
                        is_aligned_to_16B = is_aligned_to_16B and strides_list_2[ids_list[pos]] % alignment_size == 0

            if check_aligned_to_16B and not is_aligned_to_16B:
                continue


            # check if the same size combination already exists from a different divisor combination, and if so, apply merge rules
            # ====================================================================================================================

            # if merge rules empty or no existing combination has the same total size, add the combination to the list
            if len(merge_rules) == 0 or total_size not in total_sizes:
                splits.append(combination)
                total_sizes.append(total_size)

            elif prefer_best_performance_factor:
                existing_index = total_sizes.index(total_size)
                existing_combination = splits[existing_index]

                current_score = self._get_merge_performance_score_for_split(combination, ids_list, strides_list, strides_list_2)
                existing_score = self._get_merge_performance_score_for_split(existing_combination, ids_list, strides_list, strides_list_2)

                if current_score > existing_score:
                    splits[existing_index] = combination

                elif current_score == existing_score:
                    # if scores are equal, keep the one with a higher split number for lower stride dimensions
                    pass

        return total_sizes, splits


    def _get_all_possible_splits_for_dim_type(self, dim_type, strides_list, strides_list_2=None, pruning_rules=[], merge_rules=[]):
        dim_ids = [i for i, dt in enumerate(self.dim_types) if dt == dim_type]

        total_sizes = []
        splits = []

        total_sizes, splits = self._get_all_possible_dim_size_combinations(dim_ids, strides_list, strides_list_2=strides_list_2, pruning_rules=pruning_rules, merge_rules=merge_rules)
        return total_sizes, splits


    def _get_all_possible_splits_for_dim_with_iterative_pruning_removal(self, dim_type, strides_list, pruning_rules_ordered, merge_rules, strides_list_2=None):
        total_sizes, splits = self._get_all_possible_splits_for_dim_type(dim_type, strides_list, strides_list_2=strides_list_2, pruning_rules=pruning_rules_ordered, merge_rules=merge_rules)

        while len(total_sizes) == 0:
            if len(pruning_rules_ordered) == 0:
                raise ValueError(
                    f"_get_all_possible_splits_for_dim_with_iterative_pruning_removal: "
                    f"No valid splits found for dim_type {dim_type} even after all pruning rules were removed."
                )
            pruning_rules_ordered.pop(0)
            total_sizes, splits = self._get_all_possible_splits_for_dim_type(dim_type, strides_list, strides_list_2=strides_list_2, pruning_rules=pruning_rules_ordered, merge_rules=merge_rules)

        return total_sizes, splits


    def _get_merge_performance_score_for_split(self, split, ids_list, strides_list, strides_list_2=None):
        """
        Compute a scalar performance score for a split combination, used when merging two candidates
        with the same total size under the 'prefer_best_performance_factor' merge rule.

        Scores the three factors that can actually differ between same-total-size candidates:
          - Padding penalty: split_size / next_power_of_2(split_size) for each element in the split.
          - Min-load-size penalty: ×0.65 for each stride-1 dimension in each strides list whose
            split is below the minimum load threshold (64 // bytes_per_element_load).
          - Split-count penalty: ×0.95 if more than one element in the split is != 1.

        Works with one or two strides lists.
        """

        factor = 1.0
        min_load_elements_stride_1 = 64 // self.bytes_per_element_load

        for pos, split_size in enumerate(split):
            dim_id = ids_list[pos]

            # padding penalty
            split_size_pow_2 = self._get_next_power_of_2(split_size)
            factor *= split_size / split_size_pow_2

            # min-load-size penalty for stride-1 dimensions
            if split_size != 1 and split_size < min_load_elements_stride_1:
                if strides_list[dim_id] == 1:
                    factor *= 0.65
                if strides_list_2 is not None and strides_list_2[dim_id] == 1:
                    factor *= 0.65

        # split-count penalty: penalise splits spread across more than one dimension
        non_split_count = sum(1 for size in split if size != 1)
        if non_split_count > 1:
            factor *= 0.95

        return factor



    def _get_next_power_of_2(self, n):
        """
        Get the next power of 2 greater than or equal to n.
        """

        if n <= 0:
            raise ValueError("_get_next_power_of_2: n must be greater than 0.")

        power = 1
        while power < n:
            power *= 2
        return power



    def _get_performance_model_factor_prim_sizes(self, m_split, n_split, k_split):
        
        total_m_size = 1
        total_n_size = 1
        total_k_size = 1

        factor = 1

        for i in range(len(m_split)):
            total_m_size *= m_split[i]
        for i in range(len(n_split)):
            total_n_size *= n_split[i]
        for i in range(len(k_split)):
            total_k_size *= k_split[i]

        total_mn_size = total_m_size * total_n_size

        # =============================================
        # prim sizes not larger than max primitive size
        # =============================================
        max_primitive_size_m = 256 // self.bytes_per_element_compute
        max_primitive_size_n = 256 // self.bytes_per_element_compute
        max_primitive_size_k = 256 // self.bytes_per_element_compute

        max_primitive_size_k = max_primitive_size_k // 2 if total_mn_size > ((64 * 64 * 2) // self.bytes_per_element_compute) else max_primitive_size_k


        if total_m_size > max_primitive_size_m:
            factor *= 0.1
        if total_n_size > max_primitive_size_n:
            factor *= 0.1
        if total_k_size > max_primitive_size_k:
            factor *= 0.2

        # ==================
        # prim size m/n size
        # ==================

        limit_size = 1024 * 2048 // self.bytes_per_element_compute
        
        # we prefer |m_prim| = |n_prim| = 128 for contractions where |M| * |N| is larger than limit_size
        # we prefer |m_prim| = 128 and |n_prim| = 64, or vice versa, for contractions where |M| * |N| is smaller than limit_size

        total_size_M_contraction = 1
        total_size_N_contraction = 1

        for i in range(len(self.dim_types)):
            if self.dim_types[i] == etops.dim.m:
                total_size_M_contraction *= self.dim_sizes[i]
            elif self.dim_types[i] == etops.dim.n:
                total_size_N_contraction *= self.dim_sizes[i]

        total_size_MN_contraction = total_size_M_contraction * total_size_N_contraction

        if total_size_MN_contraction >= limit_size:
            contraction_larger_than_limit = True
        else:
            contraction_larger_than_limit = False

        if contraction_larger_than_limit:
            # compare to prim sizes if 128/128 for m/n in fp16
            log_wanted_prim = math.log2(128)

            log_m_prim = math.ceil(math.log2(total_m_size))
            log_n_prim = math.ceil(math.log2(total_n_size))

            diff_m = abs(log_m_prim - log_wanted_prim)
            diff_n = abs(log_n_prim - log_wanted_prim)

            factor *= 0.75 ** (diff_m + diff_n)

        else:
            # compare to both m_prim = 128 and n_prim = 64, and m_prim = 64 and n_prim = 128, and take the better one
            log_wanted_prim_m_128 = math.log2(128)
            log_wanted_prim_n_64 = math.log2(64)

            log_wanted_prim_m_64 = math.log2(64)
            log_wanted_prim_n_128 = math.log2(128)

            log_m_prim = math.ceil(math.log2(total_m_size))
            log_n_prim = math.ceil(math.log2(total_n_size))

            diff_prim_128_64 = abs(log_m_prim - log_wanted_prim_m_128) + abs(log_n_prim - log_wanted_prim_n_64)
            diff_prim_64_128 = abs(log_m_prim - log_wanted_prim_m_64) + abs(log_n_prim - log_wanted_prim_n_128)

            diff_prim_size = min(diff_prim_128_64, diff_prim_64_128)

            factor *= 0.75 ** diff_prim_size

        
        # ===========
        # prim size k
        # ===========

        # compare k_prim size to 64 or 128 for fp16, depending on the size of total_mn_size
        # if total_mn_size is larger than limit_size_k, we prefer k_prim of size 64 or 32
        # if total_mn_size is smaller than limit_size_k, we prefer k_prim of size 128 or 64


        limit_size_k = 64 * 128 // self.bytes_per_element_compute

        if total_mn_size >= limit_size_k:
            first_compare_k_prim = 64
            second_compare_k_prim = 32
        else:
            first_compare_k_prim = 128
            second_compare_k_prim = 64
        
        log_wanted_prim_k_first = math.log2(first_compare_k_prim)
        log_wanted_prim_k_second = math.log2(second_compare_k_prim)
        log_k_prim = math.ceil(math.log2(total_k_size))

        diff_k_prim_first = abs(log_k_prim - log_wanted_prim_k_first)
        diff_k_prim_second = abs(log_k_prim - log_wanted_prim_k_second)

        diff_k_prim_size = min(diff_k_prim_first, diff_k_prim_second)

        factor *= 0.75 ** diff_k_prim_size

        # ======================================================
        # minimum load sizes of 64 bytes for stride-1 dimensions
        # ======================================================
        min_load_elements_stride_1 = 64 // self.bytes_per_element_load

        split_counter_m = 0
        split_counter_n = 0
        split_counter_k = 0

        for i in range(len(self.dim_types)):
            if self.dim_types[i] == etops.dim.m:
                if m_split[split_counter_m] != 1 and m_split[split_counter_m] < min_load_elements_stride_1:
                    if self.strides_left[i] == 1:
                        factor *= 0.65
                    if self.strides_out[i] == 1:
                        factor *= 0.65
                split_counter_m += 1
            
            elif self.dim_types[i] == etops.dim.n:
                if n_split[split_counter_n] != 1 and n_split[split_counter_n] < min_load_elements_stride_1:
                    if self.strides_right[i] == 1:
                        factor *= 0.65
                    if self.strides_out[i] == 1:
                        factor *= 0.65
                split_counter_n += 1
            
            elif self.dim_types[i] == etops.dim.k:
                if k_split[split_counter_k] != 1 and k_split[split_counter_k] < min_load_elements_stride_1:
                    if self.strides_left[i] == 1:
                        factor *= 0.65
                    if self.strides_right[i] == 1:
                        factor *= 0.65
                split_counter_k += 1



        # ===============
        # padding penalty
        # ===============

        for split_size in m_split:
            split_size_pow_2 = self._get_next_power_of_2(split_size)
            percentage_non_padded = split_size / split_size_pow_2
            factor *= percentage_non_padded

        for split_size in n_split:
            split_size_pow_2 = self._get_next_power_of_2(split_size)
            percentage_non_padded = split_size / split_size_pow_2
            factor *= percentage_non_padded

        for split_size in k_split:
            split_size_pow_2 = self._get_next_power_of_2(split_size)
            percentage_non_padded = split_size / split_size_pow_2
            factor *= percentage_non_padded

        
        # penalty if primitive dimensions are split
        non_split_m = sum(1 for size in m_split if size != 1)
        non_split_n = sum(1 for size in n_split if size != 1)
        non_split_k = sum(1 for size in k_split if size != 1)
        
        if non_split_m > 1:
            factor *= 0.95
        if non_split_n > 1:
            factor *= 0.95
        if non_split_k > 1:
            factor *= 0.95

        return factor



    

    def _get_performance_model_factor_alignment(self, m_split, n_split, k_split, elements_to_align):
        left_tensor_byte_aligned = True
        right_tensor_byte_aligned = True
        output_tensor_byte_aligned = True

        split_counter_m = 0
        split_counter_n = 0
        split_counter_k = 0

        for i in range(len(self.dim_types)):
            if self.dim_types[i] == etops.dim.m:
                if m_split[split_counter_m] != 1:
                    # left
                    if self.strides_left[i] != 1 and self.strides_left[i] % elements_to_align != 0:
                        left_tensor_byte_aligned = False
                    elif self.strides_left[i] == 1 and self.dim_sizes[i] % elements_to_align != 0:
                        left_tensor_byte_aligned = False
                    
                    # output
                    if self.strides_out[i] != 1 and self.strides_out[i] % elements_to_align != 0:
                        output_tensor_byte_aligned = False
                    elif self.strides_out[i] == 1 and self.dim_sizes[i] % elements_to_align != 0:
                        output_tensor_byte_aligned = False
                
                split_counter_m += 1
            
            elif self.dim_types[i] == etops.dim.n:
                if n_split[split_counter_n] != 1:
                    # right
                    if self.strides_right[i] != 1 and self.strides_right[i] % elements_to_align != 0:
                        right_tensor_byte_aligned = False
                    elif self.strides_right[i] == 1 and self.dim_sizes[i] % elements_to_align != 0:
                        right_tensor_byte_aligned = False
                    
                    # output
                    if self.strides_out[i] != 1 and self.strides_out[i] % elements_to_align != 0:
                        output_tensor_byte_aligned = False
                    elif self.strides_out[i] == 1 and self.dim_sizes[i] % elements_to_align != 0:
                        output_tensor_byte_aligned = False

                split_counter_n += 1
            
            elif self.dim_types[i] == etops.dim.k:
                if k_split[split_counter_k] != 1:
                    # left
                    if self.strides_left[i] != 1 and self.strides_left[i] % elements_to_align != 0:
                        left_tensor_byte_aligned = False
                    elif self.strides_left[i] == 1 and self.dim_sizes[i] % elements_to_align != 0:
                        left_tensor_byte_aligned = False

                    # right
                    if self.strides_right[i] != 1 and self.strides_right[i] % elements_to_align != 0:
                        right_tensor_byte_aligned = False
                    elif self.strides_right[i] == 1 and self.dim_sizes[i] % elements_to_align != 0:
                        right_tensor_byte_aligned = False

                split_counter_k += 1

        factor = 1
        if not left_tensor_byte_aligned:
            factor *= 0.25
        if not right_tensor_byte_aligned:
            factor *= 0.25
        if not output_tensor_byte_aligned:
            factor *= 0.25

        return factor

    

    def _performance_model_primitives(self, m_split, n_split, k_split):
        """
        Performance model to estimate the performance of a given primitive configuration.
        """

        ALIGNMENT_BYTES_LOAD = 16
        
        # check split dimension sizes
        factor_prim_sizes = self._get_performance_model_factor_prim_sizes(m_split, n_split, k_split)

        # alignment to 16 bytes for memory accesses
        elements_to_align = ALIGNMENT_BYTES_LOAD // self.bytes_per_element_load
        factor_alignment = self._get_performance_model_factor_alignment(m_split, n_split, k_split, elements_to_align)
        
        performance_score = 100 * factor_prim_sizes * factor_alignment
        return performance_score
        


    def _find_primitive_dimension(self):
        """
        Find the primitive dimension for the main primitive.
        """

        stride_1_type_left = None
        stride_1_type_right = None
        stride_1_type_out = None

        for i in range(len(self.strides_left)):
            if self.strides_left[i] == 1:
                stride_1_type_left = self.dim_types[i]
            if self.strides_right[i] == 1:
                stride_1_type_right = self.dim_types[i]
            if self.strides_out[i] == 1:
                stride_1_type_out = self.dim_types[i]

        if stride_1_type_left is None or stride_1_type_right is None or stride_1_type_out is None:
            raise ValueError("No stride-1 dimension found in left, right, or output tensor.")
    

        total_sizes_m = []
        total_sizes_n = []
        total_sizes_k = []
        splits_m = []
        splits_n = []
        splits_k = []

        pruning_rules_initial = ['16B_aligned', 'smaller_than_max_prim_size', 'always_include_stride_1']
        merge_rules = ['prefer_best_performance_factor']

        # find and split m primitive dimension
        pruning_rules = pruning_rules_initial.copy()

        if stride_1_type_left == etops.dim.m and stride_1_type_out == etops.dim.m:
            strides_list = self.strides_left
            strides_list_2 = self.strides_out
        
        elif stride_1_type_left == etops.dim.m and stride_1_type_out != etops.dim.m:
            strides_list = self.strides_left
            strides_list_2 = None
        
        elif stride_1_type_left != etops.dim.m and stride_1_type_out == etops.dim.m:
            strides_list = self.strides_out
            strides_list_2 = None

        else:
            strides_list = self.strides_left
            strides_list_2 = None
            pruning_rules.remove('always_include_stride_1')

        total_sizes_m, splits_m = self._get_all_possible_splits_for_dim_with_iterative_pruning_removal(etops.dim.m, strides_list, strides_list_2=strides_list_2, pruning_rules_ordered=pruning_rules.copy(), merge_rules=merge_rules)

        # find and split n primitive dimension
        pruning_rules = pruning_rules_initial.copy()
        if stride_1_type_right == etops.dim.n and stride_1_type_out == etops.dim.n:
            strides_list = self.strides_right
            strides_list_2 = self.strides_out

        elif stride_1_type_right == etops.dim.n and stride_1_type_out != etops.dim.n:
            strides_list = self.strides_right
            strides_list_2 = None

        elif stride_1_type_right != etops.dim.n and stride_1_type_out == etops.dim.n:
            strides_list = self.strides_out
            strides_list_2 = None

        else:
            strides_list = self.strides_right
            strides_list_2 = None
            pruning_rules.remove('always_include_stride_1')

        total_sizes_n, splits_n = self._get_all_possible_splits_for_dim_with_iterative_pruning_removal(etops.dim.n, strides_list, strides_list_2=strides_list_2, pruning_rules_ordered=pruning_rules.copy(), merge_rules=merge_rules)

        # find and split k primitive dimension
        pruning_rules = pruning_rules_initial.copy()

        if stride_1_type_left == etops.dim.k and stride_1_type_right == etops.dim.k:
            strides_list = self.strides_left
            strides_list_2 = self.strides_right
        
        elif stride_1_type_left == etops.dim.k and stride_1_type_right != etops.dim.k:
            strides_list = self.strides_left
            strides_list_2 = None
        
        elif stride_1_type_left != etops.dim.k and stride_1_type_right == etops.dim.k:
            strides_list = self.strides_right
            strides_list_2 = None
        
        else:
            strides_list = self.strides_left
            strides_list_2 = None
            pruning_rules.remove('always_include_stride_1')

            
        total_sizes_k, splits_k = self._get_all_possible_splits_for_dim_with_iterative_pruning_removal(etops.dim.k, strides_list, strides_list_2=strides_list_2, pruning_rules_ordered=pruning_rules.copy(), merge_rules=merge_rules)

        # should not happen
        if len(total_sizes_m) == 0 or len(total_sizes_n) == 0 or len(total_sizes_k) == 0:
            raise ValueError("No valid primitive dimension splits found for m, n, or k dimensions.")

        # for each combination of m, n, and k splits, calculate a performance score using the performance model, and keep 20 best performing splits
        best_splits = []
        performance_scores = []

        for split_m in splits_m:
            for split_n in splits_n:
                for split_k in splits_k:
                    performance_score = self._performance_model_primitives(split_m, split_n, split_k)

                    if len(performance_scores) < 20 or performance_score > performance_scores[-1]:
                        performance_scores.append(performance_score)
                        best_splits.append([performance_score, split_m, split_n, split_k])

                        # sort by performance score
                        sorted_pairs = sorted(
                            zip(performance_scores, best_splits),
                            key=lambda pair: pair[0],
                            reverse=True
                        )

                        performance_scores = [score for score, split in sorted_pairs]
                        best_splits = [split for score, split in sorted_pairs]

                        # keep only top 20
                        performance_scores = performance_scores[:20]
                        best_splits = best_splits[:20]

        

    def _optimize_for_l2_cache(self):
        """
        Optimize the configuration for L2 cache size by splitting/fusing dimensions.
        The goal is to maximize L2 cache utilization and reuse.
        """

        l2_cache_size_in_elements = self.l2_cache_size_in_bytes // self.bytes_per_element_load

        # calculate sizes of left and right tensor
        left_tensor_elements_count = 1
        right_tensor_elements_count = 1
        for i in range(len(self.dim_types)):
            if self.dim_types[i] == etops.dim.m or self.dim_types[i] == etops.dim.k or self.dim_types[i] == etops.dim.c:
                left_tensor_elements_count *= self.dim_sizes[i]

        for i in range(len(self.dim_types)):
            if self.dim_types[i] == etops.dim.n or self.dim_types[i] == etops.dim.k or self.dim_types[i] == etops.dim.c:
                right_tensor_elements_count *= self.dim_sizes[i]

        # if left tensor and right tensor both fit in L2 cache, return
        if left_tensor_elements_count + right_tensor_elements_count <= l2_cache_size_in_elements * 0.9:
            return





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
        self.exec_types[new_id] = new_exec_type



    def _split_multiple_dimensions(self, dim_ids, splits):
        """
        Split multiple dimensions at once.

        Parameters:
        - dim_ids: list of original dimension IDs (positions before any of the splits
                   in this call are applied).
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
            self._split_dimension(actual_id, split[0], split[1])



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
        Runtime could be improved by sorting the lists by strides first to O(nlogn), but then the current list structure would be destroyed.
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
        for i in range(1, math.isqrt(dim_size) + 1):
            if dim_size % i == 0:
                dim_divisors.append(i)
                if i != dim_size // i:
                    dim_divisors.append(dim_size // i)
        
        return dim_divisors

    

    def _get_ids_sorted_by_stride(self, strides_list):
        """
        Get the list of dimension IDs sorted by stride in ascending order.
        Drops dimensions with stride equal to 0.
        """

        strides_list_without_zeros = [(i, stride) for i, stride in enumerate(strides_list) if stride != 0]
        sorted_ids = [i for i, stride in sorted(strides_list_without_zeros, key=lambda x: x[1])]

        return sorted_ids

    
    def _init_CMNK_sizes(self):
        """
        Initialize the total number of elements in the M, N, K, and C dimensions.
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

