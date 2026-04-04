import copy
import etops
import cupy as cp
import itertools
import math

from .config_helper import ConfigHelper



class Optimizer:
    def __init__(self, config):
        # Initialize GPU properties
        device_id = cp.cuda.Device().id
        props = cp.cuda.runtime.getDeviceProperties(device_id)

        self.sm_count = props['multiProcessorCount']

        # byte values for memory sizes
        self.global_mem_size = props['totalGlobalMem']
        self.l2_cache_size = props['l2CacheSize']

        self.MAX_L2_CACHE_UTILIZATION = 0.9

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

        # Build config helper — owns all mutable dim/stride metadata
        self.cfg = ConfigHelper(config)

        self.l2_cache_size_in_elements = self.l2_cache_size // self.bytes_per_element_load

        self.optimized_config = None
        self.performance_score_optimized_config = None



    def optimize(self):
        """
        Run the optimization pass on the configuration.
        """

        # initialize all exec types to seq
        self.cfg.exec_types = [etops.exec.seq] * len(self.cfg.exec_types)

        # Fuse all fusable dimensions together
        self.cfg._fuse_all_fusable_dimensions(new_exec_type=etops.exec.seq)

        # Get the primitive dimension candidates
        candidate_cfgs, performance_scores = self._find_primitive_dimension()

        # Reorder non-prim K dimensions immediately to the left of the prim block
        for cfg in candidate_cfgs:
            self._reorder_k_dimensions(cfg)

        candidate_cfgs_after_l2_cache_optimization = []

        for i, cfg in enumerate(candidate_cfgs):
            optimized_cfg, factor = self._optimize_for_l2_cache(cfg)
            candidate_cfgs_after_l2_cache_optimization.append(optimized_cfg)
            performance_scores[i] *= factor

        # get the best config according to the performance scores after L2 cache optimization
        best_index = performance_scores.index(max(performance_scores))
        best_cfg = candidate_cfgs_after_l2_cache_optimization[best_index]

        self.optimized_config = best_cfg.get_config()
        self.performance_score_optimized_config = performance_scores[best_index]




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

        check_always_include_stride_1  = 'always_include_stride_1'       in pruning_rules
        check_smaller_than_max_prim_size = 'smaller_than_max_prim_size' in pruning_rules
        check_aligned_to_16B           = '16B_aligned'                    in pruning_rules
        prefer_best_performance_factor = 'prefer_best_performance_factor' in merge_rules

        MAX_PRIM_BYTES    = 256
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
            list_of_divisors.append(self.cfg.divisors[i])

        splits      = []
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
                existing_index      = total_sizes.index(total_size)
                existing_combination = splits[existing_index]

                current_score  = self._get_merge_performance_score_for_split(combination,          ids_list, strides_list, strides_list_2)
                existing_score = self._get_merge_performance_score_for_split(existing_combination, ids_list, strides_list, strides_list_2)

                if current_score > existing_score:
                    splits[existing_index] = combination

                elif current_score == existing_score:
                    # if scores are equal, keep the one with a higher split number for lower stride dimensions
                    pass

        return total_sizes, splits


    def _get_all_possible_splits_for_dim_type(self, dim_type, strides_list, strides_list_2=None, pruning_rules=[], merge_rules=[]):
        dim_ids = [i for i, dt in enumerate(self.cfg.dim_types) if dt == dim_type]

        total_sizes, splits = self._get_all_possible_dim_size_combinations(
            dim_ids, strides_list,
            strides_list_2=strides_list_2,
            pruning_rules=pruning_rules,
            merge_rules=merge_rules
        )
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

        for i in range(len(self.cfg.dim_types)):
            if self.cfg.dim_types[i] == etops.dim.m:
                total_size_M_contraction *= self.cfg.dim_sizes[i]
            elif self.cfg.dim_types[i] == etops.dim.n:
                total_size_N_contraction *= self.cfg.dim_sizes[i]

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
            log_wanted_prim_n_64  = math.log2(64)

            log_wanted_prim_m_64  = math.log2(64)
            log_wanted_prim_n_128 = math.log2(128)

            log_m_prim = math.ceil(math.log2(total_m_size))
            log_n_prim = math.ceil(math.log2(total_n_size))

            diff_prim_128_64 = abs(log_m_prim - log_wanted_prim_m_128) + abs(log_n_prim - log_wanted_prim_n_64)
            diff_prim_64_128 = abs(log_m_prim - log_wanted_prim_m_64)  + abs(log_n_prim - log_wanted_prim_n_128)

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
            first_compare_k_prim  = 64
            second_compare_k_prim = 32
        else:
            first_compare_k_prim  = 128
            second_compare_k_prim = 64

        log_wanted_prim_k_first  = math.log2(first_compare_k_prim)
        log_wanted_prim_k_second = math.log2(second_compare_k_prim)
        log_k_prim = math.ceil(math.log2(total_k_size))

        diff_k_prim_first  = abs(log_k_prim - log_wanted_prim_k_first)
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

        for i in range(len(self.cfg.dim_types)):
            if self.cfg.dim_types[i] == etops.dim.m:
                if m_split[split_counter_m] != 1 and m_split[split_counter_m] < min_load_elements_stride_1:
                    if self.cfg.strides_left[i] == 1:
                        factor *= 0.65
                    if self.cfg.strides_out[i] == 1:
                        factor *= 0.65
                split_counter_m += 1

            elif self.cfg.dim_types[i] == etops.dim.n:
                if n_split[split_counter_n] != 1 and n_split[split_counter_n] < min_load_elements_stride_1:
                    if self.cfg.strides_right[i] == 1:
                        factor *= 0.65
                    if self.cfg.strides_out[i] == 1:
                        factor *= 0.65
                split_counter_n += 1

            elif self.cfg.dim_types[i] == etops.dim.k:
                if k_split[split_counter_k] != 1 and k_split[split_counter_k] < min_load_elements_stride_1:
                    if self.cfg.strides_left[i] == 1:
                        factor *= 0.65
                    if self.cfg.strides_right[i] == 1:
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
        left_tensor_byte_aligned   = True
        right_tensor_byte_aligned  = True
        output_tensor_byte_aligned = True

        split_counter_m = 0
        split_counter_n = 0
        split_counter_k = 0

        for i in range(len(self.cfg.dim_types)):
            if self.cfg.dim_types[i] == etops.dim.m:
                if m_split[split_counter_m] != 1:
                    # left
                    if self.cfg.strides_left[i] != 1 and self.cfg.strides_left[i] % elements_to_align != 0:
                        left_tensor_byte_aligned = False
                    elif self.cfg.strides_left[i] == 1 and self.cfg.dim_sizes[i] % elements_to_align != 0:
                        left_tensor_byte_aligned = False

                    # output
                    if self.cfg.strides_out[i] != 1 and self.cfg.strides_out[i] % elements_to_align != 0:
                        output_tensor_byte_aligned = False
                    elif self.cfg.strides_out[i] == 1 and self.cfg.dim_sizes[i] % elements_to_align != 0:
                        output_tensor_byte_aligned = False

                split_counter_m += 1

            elif self.cfg.dim_types[i] == etops.dim.n:
                if n_split[split_counter_n] != 1:
                    # right
                    if self.cfg.strides_right[i] != 1 and self.cfg.strides_right[i] % elements_to_align != 0:
                        right_tensor_byte_aligned = False
                    elif self.cfg.strides_right[i] == 1 and self.cfg.dim_sizes[i] % elements_to_align != 0:
                        right_tensor_byte_aligned = False

                    # output
                    if self.cfg.strides_out[i] != 1 and self.cfg.strides_out[i] % elements_to_align != 0:
                        output_tensor_byte_aligned = False
                    elif self.cfg.strides_out[i] == 1 and self.cfg.dim_sizes[i] % elements_to_align != 0:
                        output_tensor_byte_aligned = False

                split_counter_n += 1

            elif self.cfg.dim_types[i] == etops.dim.k:
                if k_split[split_counter_k] != 1:
                    # left
                    if self.cfg.strides_left[i] != 1 and self.cfg.strides_left[i] % elements_to_align != 0:
                        left_tensor_byte_aligned = False
                    elif self.cfg.strides_left[i] == 1 and self.cfg.dim_sizes[i] % elements_to_align != 0:
                        left_tensor_byte_aligned = False

                    # right
                    if self.cfg.strides_right[i] != 1 and self.cfg.strides_right[i] % elements_to_align != 0:
                        right_tensor_byte_aligned = False
                    elif self.cfg.strides_right[i] == 1 and self.cfg.dim_sizes[i] % elements_to_align != 0:
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
        factor_alignment  = self._get_performance_model_factor_alignment(m_split, n_split, k_split, elements_to_align)

        performance_score = 100 * factor_prim_sizes * factor_alignment
        return performance_score



    def _find_primitive_dimension(self):
        """
        Find the primitive dimension for the main primitive.
        """

        stride_1_type_left  = None
        stride_1_type_right = None
        stride_1_type_out   = None

        for i in range(len(self.cfg.strides_left)):
            if self.cfg.strides_left[i] == 1:
                stride_1_type_left = self.cfg.dim_types[i]
            if self.cfg.strides_right[i] == 1:
                stride_1_type_right = self.cfg.dim_types[i]
            if self.cfg.strides_out[i] == 1:
                stride_1_type_out = self.cfg.dim_types[i]

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
            strides_list   = self.cfg.strides_left
            strides_list_2 = self.cfg.strides_out

        elif stride_1_type_left == etops.dim.m and stride_1_type_out != etops.dim.m:
            strides_list   = self.cfg.strides_left
            strides_list_2 = None

        elif stride_1_type_left != etops.dim.m and stride_1_type_out == etops.dim.m:
            strides_list   = self.cfg.strides_out
            strides_list_2 = None

        else:
            strides_list   = self.cfg.strides_left
            strides_list_2 = None
            pruning_rules.remove('always_include_stride_1')

        total_sizes_m, splits_m = self._get_all_possible_splits_for_dim_with_iterative_pruning_removal(etops.dim.m, strides_list, strides_list_2=strides_list_2, pruning_rules_ordered=pruning_rules.copy(), merge_rules=merge_rules)

        # find and split n primitive dimension
        pruning_rules = pruning_rules_initial.copy()
        if stride_1_type_right == etops.dim.n and stride_1_type_out == etops.dim.n:
            strides_list   = self.cfg.strides_right
            strides_list_2 = self.cfg.strides_out

        elif stride_1_type_right == etops.dim.n and stride_1_type_out != etops.dim.n:
            strides_list   = self.cfg.strides_right
            strides_list_2 = None

        elif stride_1_type_right != etops.dim.n and stride_1_type_out == etops.dim.n:
            strides_list   = self.cfg.strides_out
            strides_list_2 = None

        else:
            strides_list   = self.cfg.strides_right
            strides_list_2 = None
            pruning_rules.remove('always_include_stride_1')

        total_sizes_n, splits_n = self._get_all_possible_splits_for_dim_with_iterative_pruning_removal(etops.dim.n, strides_list, strides_list_2=strides_list_2, pruning_rules_ordered=pruning_rules.copy(), merge_rules=merge_rules)

        # find and split k primitive dimension
        pruning_rules = pruning_rules_initial.copy()

        if stride_1_type_left == etops.dim.k and stride_1_type_right == etops.dim.k:
            strides_list   = self.cfg.strides_left
            strides_list_2 = self.cfg.strides_right

        elif stride_1_type_left == etops.dim.k and stride_1_type_right != etops.dim.k:
            strides_list   = self.cfg.strides_left
            strides_list_2 = None

        elif stride_1_type_left != etops.dim.k and stride_1_type_right == etops.dim.k:
            strides_list   = self.cfg.strides_right
            strides_list_2 = None

        else:
            strides_list   = self.cfg.strides_left
            strides_list_2 = None
            pruning_rules.remove('always_include_stride_1')


        total_sizes_k, splits_k = self._get_all_possible_splits_for_dim_with_iterative_pruning_removal(etops.dim.k, strides_list, strides_list_2=strides_list_2, pruning_rules_ordered=pruning_rules.copy(), merge_rules=merge_rules)

        # should not happen
        if len(total_sizes_m) == 0 or len(total_sizes_n) == 0 or len(total_sizes_k) == 0:
            raise ValueError("No valid primitive dimension splits found for m, n, or k dimensions.")

        # for each combination of m, n, and k splits, calculate a performance score using the performance model, and keep 20 best performing splits
        best_splits        = []
        performance_scores = []

        for split_m in splits_m:
            for split_n in splits_n:
                for split_k in splits_k:
                    performance_score = self._performance_model_primitives(split_m, split_n, split_k)

                    if len(performance_scores) < 20 or performance_score > performance_scores[-1]:
                        performance_scores.append(performance_score)
                        best_splits.append([split_m, split_n, split_k])

                        # sort by performance score
                        sorted_pairs = sorted(
                            zip(performance_scores, best_splits),
                            key=lambda pair: pair[0],
                            reverse=True
                        )

                        performance_scores = [score for score, split in sorted_pairs]
                        best_splits        = [split for score, split in sorted_pairs]

                        # keep only top 20
                        performance_scores = performance_scores[:20]
                        best_splits        = best_splits[:20]


        # create new config helpers for the top 20 splits
        # For each candidate, deep-copy the current (post-fused) cfg and apply the
        # M/N/K tile splits so that every dimension is split into
        #   [outer_loop_size, primitive_tile_size]
        # where the primitive tile occupies the inner (lower-stride) sub-dimension.
        # Dimensions whose tile equals their full size (no outer loop needed) or
        # whose tile is 1 (dimension not included in the primitive) are left unsplit.
        prim_cfgs = []

        for split_m, split_n, split_k in best_splits:
            prim_cfg = copy.deepcopy(self.cfg)

            dim_ids_to_split = []
            splits_to_apply  = []

            split_counter_m = 0
            split_counter_n = 0
            split_counter_k = 0

            for i in range(len(prim_cfg.dim_types)):
                dim_type  = prim_cfg.dim_types[i]
                full_size = prim_cfg.dim_sizes[i]

                if dim_type == etops.dim.m:
                    tile_size = split_m[split_counter_m]
                    num_dims_of_type = len(split_m)
                    split_counter_m += 1
                elif dim_type == etops.dim.n:
                    tile_size = split_n[split_counter_n]
                    num_dims_of_type = len(split_n)
                    split_counter_n += 1
                elif dim_type == etops.dim.k:
                    tile_size = split_k[split_counter_k]
                    num_dims_of_type = len(split_k)
                    split_counter_k += 1
                else:
                    # C (batch) dimensions are not split here
                    continue

                # no split
                if tile_size == 1:
                    if num_dims_of_type == 1:
                        prim_cfg.exec_types[i] = etops.exec.prim
                    continue

                # no split, but prim dimension
                if tile_size == full_size:
                    prim_cfg.exec_types[i] = etops.exec.prim
                    continue

                dim_ids_to_split.append(i)
                # split[0] = outer loop size (higher stride, id)
                # split[1] = primitive tile  (lower/original stride, id+1)

                splits_to_apply.append([full_size // tile_size, tile_size])

            prim_cfg._split_multiple_dimensions(dim_ids_to_split, splits_to_apply, new_exec_type_left=etops.exec.seq, new_exec_type_right=etops.exec.prim)

            # -----------------------------------------------------------------------
            # Reorder prim dimensions to the rightmost positions in the order m, n, k.
            # Within each type the prim dim with the largest stride (in the deciding
            # tensor) is placed leftmost.  Deciding tensors:
            #   m → left tensor
            #   n → right tensor
            #   k → right tensor if m_prim_size >= n_prim_size, else left tensor
            # Non-prim dimensions keep their existing relative order at the front.
            # -----------------------------------------------------------------------
            m_prim_size_reorder = 1
            n_prim_size_reorder = 1
            for i in range(len(prim_cfg.dim_types)):
                if prim_cfg.dim_types[i] == etops.dim.m and prim_cfg.exec_types[i] == etops.exec.prim:
                    m_prim_size_reorder *= prim_cfg.dim_sizes[i]
                elif prim_cfg.dim_types[i] == etops.dim.n and prim_cfg.exec_types[i] == etops.exec.prim:
                    n_prim_size_reorder *= prim_cfg.dim_sizes[i]

            k_deciding_strides = prim_cfg.strides_right if m_prim_size_reorder >= n_prim_size_reorder else prim_cfg.strides_left

            non_prim_ids = [
                i for i, et in enumerate(prim_cfg.exec_types)
                if et != etops.exec.prim
            ]
            m_prim_ids = sorted(
                [i for i, (dt, et) in enumerate(zip(prim_cfg.dim_types, prim_cfg.exec_types))
                 if dt == etops.dim.m and et == etops.exec.prim],
                key=lambda i: prim_cfg.strides_left[i],
                reverse=True
            )
            n_prim_ids = sorted(
                [i for i, (dt, et) in enumerate(zip(prim_cfg.dim_types, prim_cfg.exec_types))
                 if dt == etops.dim.n and et == etops.exec.prim],
                key=lambda i: prim_cfg.strides_right[i],
                reverse=True
            )
            k_prim_ids = sorted(
                [i for i, (dt, et) in enumerate(zip(prim_cfg.dim_types, prim_cfg.exec_types))
                 if dt == etops.dim.k and et == etops.exec.prim],
                key=lambda i: k_deciding_strides[i],
                reverse=True
            )

            prim_cfg._permute_dimensions(non_prim_ids + m_prim_ids + n_prim_ids + k_prim_ids)

            prim_cfgs.append(prim_cfg)
        
        return prim_cfgs, performance_scores



    def _reorder_k_dimensions(self, config_object):
        """
        Reorder non-prim K dimensions so they are grouped immediately to the left
        of the primitive block, ordered by descending stride in the deciding tensor.

        The deciding tensor is the larger of the two input tensors by element count,
        using config_object.left_size_total (C×M×K) vs config_object.right_size_total
        (C×N×K).  Left wins ties.

        Within the non-prim K block, the dimension with the largest stride is
        placed leftmost (outermost over K).  Non-K non-prim dimensions (c, m_outer,
        n_outer) keep their existing relative order at the front.  The prim block
        (already ordered m → n → k by _find_primitive_dimension) is left unchanged.

        Modifies config_object in place and returns it.
        """

        k_deciding_strides = config_object.strides_left if config_object.left_size_total >= config_object.right_size_total else config_object.strides_right

        non_k_non_prim_ids = [
            i for i, (dt, et) in enumerate(zip(config_object.dim_types, config_object.exec_types))
            if dt != etops.dim.k and et != etops.exec.prim
        ]
        k_non_prim_ids = sorted(
            [i for i, (dt, et) in enumerate(zip(config_object.dim_types, config_object.exec_types))
             if dt == etops.dim.k and et != etops.exec.prim],
            key=lambda i: k_deciding_strides[i],
            reverse=True
        )
        prim_ids = [
            i for i, et in enumerate(config_object.exec_types)
            if et == etops.exec.prim
        ]

        config_object._permute_dimensions(non_k_non_prim_ids + k_non_prim_ids + prim_ids)
        return config_object



    def _l2_split_wins_by_stride(self, candidate, existing, dim_ids, strides):
        """
        Tie-break between two split combinations that have the same total size and the same
        number of 1s. Compares the split sizes for each dimension in ascending stride order.
        The candidate wins if it has a strictly larger split size at the first position
        where the two differ. Returns False if the candidate does not win (existing stays).
        """

        sorted_positions = sorted(range(len(dim_ids)), key=lambda pos: strides[dim_ids[pos]])

        for pos in sorted_positions:
            if candidate[pos] > existing[pos]:
                return True
            elif candidate[pos] < existing[pos]:
                return False

        return False  # equal in every position — existing stays


    def _get_l2_splits_for_dim_ids(self, dim_ids, config_object, strides, prim_size, total_k_size, l2_limit):
        """
        Generate all valid L2 cache tile splits for the given list of dimension IDs.

        Each element of the returned splits list is a tuple of per-dimension tile sizes
        (one divisor per dimension  in dim_ids). A split is pruned when the tile alone —
        multiplied by the associated primitive size and total K — already exceeds the L2
        limit. When two splits have the same total tile size the merge rule is applied:
        keep the split with more 1s (fewer active tile dimensions); if equal, resolve by
        ascending-stride tie-break via _l2_split_wins_by_stride.

        If no split survives pruning, a trivial all-ones split is returned as fallback
        (no L2 tiling in that dimension).

        Returns (splits, total_sizes).
        """

        if len(dim_ids) == 0:
            return [()], [1]

        splits      = []
        total_sizes = []

        for combination in itertools.product(*[config_object.divisors[i] for i in dim_ids]):
            total_size = 1
            for s in combination:
                total_size *= s

            # Prune: tile * prim * K exceeds the L2 budget for this dimension alone
            if total_size * prim_size * total_k_size > l2_limit:
                continue

            if total_size not in total_sizes:
                splits.append(combination)
                total_sizes.append(total_size)
            else:
                existing_idx  = total_sizes.index(total_size)
                existing      = splits[existing_idx]

                current_ones  = sum(1 for s in combination if s == 1)
                existing_ones = sum(1 for s in existing    if s == 1)

                if current_ones > existing_ones:
                    splits[existing_idx] = combination
                elif current_ones == existing_ones:
                    if self._l2_split_wins_by_stride(combination, existing, dim_ids, strides):
                        splits[existing_idx] = combination

        if not splits:
            # Fallback: trivial split — no L2 tiling for this dimension
            trivial = tuple(1 for _ in dim_ids)
            splits.append(trivial)
            total_sizes.append(1)

        return splits, total_sizes


    def _optimize_for_l2_cache(self, config_object):
        """
        Optimize the configuration for L2 cache size by splitting non-prim dimensions.
        The goal is to maximise L2 cache utilisation and reuse.

        For each non-prim M and N dimension a set of candidate tile sizes is generated
        using _get_l2_splits_for_dim_ids. Individual splits are pruned when their tile
        alone (times the corresponding primitive size times total K) exceeds the L2
        budget. All (M-tile, N-tile) combinations are then evaluated with a simple
        arithmetic-intensity performance model; the combination with the highest score is
        applied to a deep copy of config_object by splitting the relevant dimensions into
        an outer (seq) and inner (seq) loop.

        Returns:
            (optimized_cfg, best_score) — a deep copy of config_object with the best
            L2 splits applied, and the arithmetic-intensity score of that split.
        """

        cfg = copy.deepcopy(config_object)

        # -----------------------------------------------------------------------
        # Prim sizes
        # -----------------------------------------------------------------------
        total_m_prim_size = 1
        total_n_prim_size = 1

        for i in range(len(cfg.dim_types)):
            if cfg.dim_types[i] == etops.dim.m and cfg.exec_types[i] == etops.exec.prim:
                total_m_prim_size *= cfg.dim_sizes[i]
            elif cfg.dim_types[i] == etops.dim.n and cfg.exec_types[i] == etops.exec.prim:
                total_n_prim_size *= cfg.dim_sizes[i]

        # Combined prim footprint (m and n prim) used in the per-dimension pruning.
        # total_k_size (full K, not just prim K) is passed separately and accounts
        # for the iteration depth over K during the L2 tile computation.
        total_mn_prim_size = total_m_prim_size * total_n_prim_size

        total_k_size = cfg.K_size_total
        l2_limit     = self.l2_cache_size_in_elements * self.MAX_L2_CACHE_UTILIZATION

        # -----------------------------------------------------------------------
        # Strides selection: use the strides of the larger tensor for each type
        # -----------------------------------------------------------------------
        left_size  = cfg.C_size_total * cfg.M_size_total * cfg.K_size_total
        right_size = cfg.C_size_total * cfg.K_size_total * cfg.N_size_total
        out_size   = cfg.C_size_total * cfg.M_size_total * cfg.N_size_total

        m_strides = cfg.strides_left  if left_size  >= out_size else cfg.strides_out
        n_strides = cfg.strides_right if right_size >= out_size else cfg.strides_out

        # -----------------------------------------------------------------------
        # Non-prim dimension IDs for M and N
        # -----------------------------------------------------------------------
        m_dim_ids = [
            i for i, (dt, et) in enumerate(zip(cfg.dim_types, cfg.exec_types))
            if dt == etops.dim.m and et != etops.exec.prim
        ]
        n_dim_ids = [
            i for i, (dt, et) in enumerate(zip(cfg.dim_types, cfg.exec_types))
            if dt == etops.dim.n and et != etops.exec.prim
        ]

        # -----------------------------------------------------------------------
        # Generate candidate splits for M and N separately
        # -----------------------------------------------------------------------
        m_splits, m_total_sizes = self._get_l2_splits_for_dim_ids(
            m_dim_ids, cfg, m_strides, total_mn_prim_size, total_k_size, l2_limit
        )
        n_splits, n_total_sizes = self._get_l2_splits_for_dim_ids(
            n_dim_ids, cfg, n_strides, total_mn_prim_size, total_k_size, l2_limit
        )

        # -----------------------------------------------------------------------
        # Evaluate all (M-tile, N-tile) combinations and keep the best
        # -----------------------------------------------------------------------
        best_split_m = m_splits[0]
        best_split_n = n_splits[0]
        best_score   = -math.inf

        for split_m, total_m in zip(m_splits, m_total_sizes):
            for split_n, total_n in zip(n_splits, n_total_sizes):
                # Combined L2 footprint check
                combined_footprint = total_m * total_m_prim_size + total_n * total_n_prim_size
                if combined_footprint > l2_limit:
                    score = 0.0
                else:
                    # Arithmetic-intensity performance model:
                    #   (total_M * M_prim * total_N * N_prim) 
                    #   -------------------------------------------
                    #   (total_M * M_prim * K + total_N * N_prim * K)
                    numerator   =  total_m * total_m_prim_size * total_n * total_n_prim_size
                    denominator = (total_m * total_m_prim_size * total_k_size +
                                   total_n * total_n_prim_size * total_k_size)
                    score = numerator / denominator if denominator > 0 else 0.0

                if score > best_score:
                    best_score   = score
                    best_split_m = split_m
                    best_split_n = split_n

        # -----------------------------------------------------------------------
        # Apply best split to the copy: split each non-prim dim into [outer, inner]
        # Both outer and inner receive exec.seq.
        # -----------------------------------------------------------------------
        dim_ids_to_split = []
        splits_to_apply  = []

        for pos, i in enumerate(m_dim_ids):
            tile_size = best_split_m[pos]
            full_size = cfg.dim_sizes[i]
            if 1 < tile_size < full_size:
                dim_ids_to_split.append(i)
                splits_to_apply.append([full_size // tile_size, tile_size])

        for pos, i in enumerate(n_dim_ids):
            tile_size = best_split_n[pos]
            full_size = cfg.dim_sizes[i]
            if 1 < tile_size < full_size:
                dim_ids_to_split.append(i)
                splits_to_apply.append([full_size // tile_size, tile_size])

        cfg._split_multiple_dimensions(
            dim_ids_to_split, splits_to_apply,
            new_exec_type_left=etops.exec.seq,
            new_exec_type_right=etops.exec.seq
        )

        # -----------------------------------------------------------------------
        # Reorder the newly created L2 inner (tile) dims into the canonical layout:
        #   [front (non-k, non-prim, non-tile) | m_tiles | n_tiles | k_non_prim | prim]
        #
        # Inner dim positions are determined by the same index-shift arithmetic used
        # inside _split_multiple_dimensions: splits are processed in ascending order of
        # original_id, so the i-th inner dim lands at original_id + offset + 1.
        # Non-tile dims keep their existing relative order exactly.
        # -----------------------------------------------------------------------
        sorted_split_pairs = sorted(zip(dim_ids_to_split, splits_to_apply), key=lambda p: p[0])
        l2_inner_ids = [
            original_id + offset + 1
            for offset, (original_id, _) in enumerate(sorted_split_pairs)
        ]

        l2_inner_id_set = set(l2_inner_ids)

        l2_inner_m_ids = sorted(
            [i for i in l2_inner_ids if cfg.dim_types[i] == etops.dim.m],
            key=lambda i: cfg.strides_left[i],
            reverse=True
        )
        l2_inner_n_ids = sorted(
            [i for i in l2_inner_ids if cfg.dim_types[i] == etops.dim.n],
            key=lambda i: cfg.strides_right[i],
            reverse=True
        )

        front_ids = [
            i for i, (dt, et) in enumerate(zip(cfg.dim_types, cfg.exec_types))
            if dt != etops.dim.k and et != etops.exec.prim and i not in l2_inner_id_set
        ]
        k_non_prim_ids_reorder = [
            i for i, (dt, et) in enumerate(zip(cfg.dim_types, cfg.exec_types))
            if dt == etops.dim.k and et != etops.exec.prim
        ]
        prim_ids_reorder = [
            i for i, et in enumerate(cfg.exec_types)
            if et == etops.exec.prim
        ]

        cfg._permute_dimensions(
            front_ids + l2_inner_m_ids + l2_inner_n_ids + k_non_prim_ids_reorder + prim_ids_reorder
        )

        # -----------------------------------------------------------------------
        # Convert best_score to a factor in [WORST_L2_FACTOR, 1.0] based on the best possible score.
        # -----------------------------------------------------------------------
        WORST_L2_FACTOR = 0.4
        
        # best M/N:
        # M * K + N * K = l2_limit
        # M = K => 2 * M * K = l2_limit
        # => M = l2_limit / (2 * K)

        best_possible_M_N = l2_limit / (2 * total_k_size)
        best_possible_score = (best_possible_M_N ** 2) / (2 * best_possible_M_N * total_k_size)


        if best_possible_score > 0:
            raw    = best_score / best_possible_score
            factor = WORST_L2_FACTOR + (1.0 - WORST_L2_FACTOR) * raw
            factor = min(1.0, max(WORST_L2_FACTOR, factor))
        else:
            factor = WORST_L2_FACTOR

        return cfg, factor

    
    def get_optimized_config(self):
        return self.optimized_config



