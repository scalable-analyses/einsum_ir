#!/bin/bash
# parse command line arguments
while getopts 'pe:l:r:b:d:k:n:hv' flag; do
  case "${flag}" in
    p) enable_ht=1 ;;
    e) einsum_ir_exe="${OPTARG}" ;;
    l) log_dir="${OPTARG}" ;;
    r) num_reps="${OPTARG}" ;;
    b) backend="${OPTARG}" ;;
    d) mode="${OPTARG}" ;;
    k) selected_keys="${OPTARG}" ;;
    n) num_cores="${OPTARG}" ;;
    v) verbose=1 ;;
    h) echo "Usage: $0 [-p enable_hyperthreading] [-e einsum_ir_executable] [-l log_directory] [-r num_repetitions] [-b backend] [-d mode] [-k benchmark_keys] [-n num_cores] [-v verbose]"
       echo "Optional arguments:"
       echo "  -p: Enable hyperthreading"
       echo "  -e: Path to einsum_ir executable"
       echo "  -l: Log directory"
       echo "  -r: Number of einsum_ir repetitions"
       echo "  -b: Backend to use (default: tpp)"
       echo "  -d: Mode in which the code is executed (default: 1)"
       echo "      0: Use expr + contraction path, do NOT reorder dimensions"
       echo "      1: Use expr + contraction path, do     reorder dimensions."
       echo "      2: Use pre-optimized einsum tree."
       echo "  -k: Benchmark keys to run (default: all, options: syn,tt,fctn,tw,getd,trn,mera,tnlm,tccg_blocked,tccg_blocked_reordered,fc)"
       echo "  -n: Number of cores to use (default: detected from system)"
       echo "  -v: Verbose output (default: off)"; exit 0 ;;
    *) echo "Unexpected option ${flag}"
       echo "Usage: $0 [-p enable_hyperthreading] [-e einsum_ir_executable] [-l log_directory] [-r num_repetitions] [-b backend] [-d mode] [-k benchmark_keys] [-n num_cores]"
       exit 1 ;;
  esac
done

# determine number of cores if not provided
if [ -z "${num_cores}" ]; then
  num_cores=$(lscpu -p | grep -v '^#' | sort -u -t, -k 2,2 | wc -l)
fi

# default values
einsum_ir_exe="${einsum_ir_exe:-einsum_ir/build/bench_expression}"
log_dir="${log_dir:-logs}"
enable_ht="${enable_ht:-0}"
selected_keys="${selected_keys:-all}"
num_threads=$((num_cores * (1 + enable_ht)))
num_reps=${num_reps:-1}
backend="${backend:-tpp}"
mode="${mode:-1}"
verbose="${verbose:-0}"

# validate backend
case "${backend}" in
  tpp|blas|tblis) ;;
  *) echo "Error: Invalid backend '${backend}'. Must be tpp, blas, or tblis"; exit 1 ;;
esac

# validate mode
case "${mode}" in
  0|1|2) ;;
  *) echo "Error: Invalid mode '${mode}'. Must be 0, 1 or 2"; exit 1 ;;
esac

# check if einsum_ir_exe exists
if [ ! -f "${einsum_ir_exe}" ]; then
  echo "Error: einsum_ir executable not found at ${einsum_ir_exe}"
  exit 1
fi

# ensure log directory exists
if [ ! -d "${log_dir}" ]; then
  mkdir -p "${log_dir}" || { echo "Error: Could not create log directory"; exit 1; }
fi

# determine the KMP_AFFINITY
if [ "${enable_ht}" -eq 1 ]; then
  kmp_affinity="explicit,proclist=[0-$((num_cores * 2 - 1))],granularity=fine"
else
  kmp_affinity="explicit,proclist=[0-$((num_cores - 1))],granularity=fine"
fi

date
echo "Running bench_einsum_ir.sh with the following settings:"
echo "  Number of cores: ${num_cores}"
echo "  Hyperthreading: ${enable_ht}"
echo "  Number of threads: ${num_threads}"
echo "  einsum_ir executable: ${einsum_ir_exe}"
echo "  Log directory: ${log_dir}"
echo "  KMP_AFFINITY: ${kmp_affinity}"
echo "  Number of repetitions: ${num_reps}"
echo "  Backend: ${backend}"
echo "  Mode: ${mode}"
echo "  Selected benchmark keys: ${selected_keys}"
echo ""

declare -A settings
if [ "$mode" = "0" ] || [ "$mode" = "1" ]; then
  settings["syn"]="einsum_benchmark/synthetic.cfg"
  settings["tt"]="tensor_decomp/tt.cfg"
  settings["fctn"]="tensor_decomp/fctn.cfg"
  settings["tw"]="tensor_decomp/tw.cfg"
  settings["getd"]="tensor_decomp/getd.cfg"
  settings["trn"]="tensor_layers/trn.cfg"
  settings["mera"]="einsum_benchmark/str_nw_mera_open_26.cfg"
  settings["tnlm"]="einsum_benchmark/lm_first_last_brackets_4_16d.cfg"
  settings["tccg_blocked_reordered"]="tccg/settings_blocked_reordered.cfg"
  settings["tccg_blocked"]="tccg/settings_blocked.cfg"
  settings["fc"]="tensor_layers/fc.cfg"
else
  settings["syn"]="einsum_benchmark/synthetic_et.cfg"
  settings["tt"]="tensor_decomp/tt_et.cfg"
  settings["fctn"]="tensor_decomp/fctn_et.cfg"
  settings["tw"]="tensor_decomp/tw_et.cfg"
  settings["getd"]="tensor_decomp/getd_et.cfg"
  settings["trn"]="tensor_layers/trn_et.cfg"
  settings["mera"]="einsum_benchmark/str_nw_mera_open_26_et.cfg"
  settings["tnlm"]="einsum_benchmark/lm_first_last_brackets_4_16d_et.cfg"
fi

# get the directory where the script is located
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for key in $(echo "${!settings[@]}" | tr ' ' '\n')
do
  settings[${key}]="$(dirname "$script_dir")/${settings[${key}]}"
done

# check if config files exist
for key in $(echo "${!settings[@]}" | tr ' ' '\n')
do
  if [ ! -f "${settings[${key}]}" ]; then
    echo "Error: Configuration file ${settings[${key}]} not found"
    exit 1
  fi
done

# prepare list of keys to process
keys_to_process=()
if [ "${selected_keys}" = "all" ]; then
    readarray -t keys_to_process < <(printf '%s\n' "${!settings[@]}" | sort)
else
    IFS=',' read -ra keys_to_process <<< "${selected_keys}"
    # Validate keys
    for key in "${keys_to_process[@]}"; do
        if [ -z "${settings[$key]}" ]; then
            echo "Error: Invalid benchmark key '${key}'"
            echo "Available keys: ${!settings[@]}"
            exit 1
        fi
    done
fi

# error tracking variable
any_errors=0

for key in "${keys_to_process[@]}"
do
  date
  echo "Running benchmark setting: ${settings[${key}]}"

  mkdir -p "${log_dir}/${key}"

  if [ "${mode}" = "0" ] || [ "${mode}" = "1" ]; then
    log_file="${log_dir}/${key}/einsum_ir_backend_${backend}_reorder_dims_${mode}.log"
  else
    log_file="${log_dir}/${key}/einsum_ir_backend_${backend}.log"
  fi
  cat /dev/null > "${log_file}"

  total_settings=$(grep -c '^' "${settings[${key}]}")
  current_setting=0
  while IFS= read -r setting || [ -n "$setting" ]; do
    ((current_setting++))
    echo -n "  Processing setting ${current_setting}/${total_settings}"
    if [ "${verbose}" -gt 0 ]; then
      echo ""
    fi
    if [ "${mode}" = "0" ] || [ "${mode}" = "1" ]; then
      command="${einsum_ir_exe} ${setting} FP32 0 1"
    else
      command="${einsum_ir_exe} ${setting} FP32"
    fi
    for rep in $(seq 1 ${num_reps})
    do
      if [ "${verbose}" -gt 0 ]; then
        EINSUM_IR_BACKEND=${backend^^} EINSUM_IR_REORDER_DIMS=${mode} OMP_NUM_THREADS=${num_threads} KMP_AFFINITY=${kmp_affinity} eval ${command} | tee -a ${log_file}
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
          echo "Warning: Command failed: ${command}"
          any_errors=1
        fi
      else
        EINSUM_IR_BACKEND=${backend^^} EINSUM_IR_REORDER_DIMS=${mode} OMP_NUM_THREADS=${num_threads} KMP_AFFINITY=${kmp_affinity} eval ${command} >> ${log_file} 2>&1
        if [ $? -ne 0 ]; then
          echo "Warning: Command failed: ${command}"
          any_errors=1
        fi
        echo -n "."
      fi
    done
    echo ""
  done < "${settings[${key}]}"
  echo ""
done

date
echo "Done"

exit $any_errors
