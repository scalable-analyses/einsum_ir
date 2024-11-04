#!/bin/bash
# parse command line arguments
while getopts 'k:c:l:n:h' flag; do
  case "${flag}" in
    k) selected_keys="${OPTARG}" ;;
    c) cpu_type="${OPTARG}" ;;
    l) log_dir="${OPTARG}" ;;
    n) num_measure_trials="${OPTARG}" ;;
    h) echo "Usage: $0 [-k benchmark_keys] [-c cpu_type] [-l log_directory]"
       echo "Optional arguments:"
       echo "  -k: Comma-separated benchmark keys (default: tccg_default,tccg_blocked_reordered,syn,tt,fctn,tw,getd,trn)"
       echo "      Available: all,syn,tt,fctn,tw,getd,trn,mera,tnlm,tccg_default,tccg_blocked_reordered"
       echo "  -c: CPU type (default: generic)"
       echo "      Available: zen4, spr, grace, apple-m2, generic"
       echo "  -l: Log directory (default: logs)"
       echo "  -n: Number of measure trials (default: 1000)"
       echo "  -h: Show help information"
       echo "Example: $0 -k syn,tt -c zen4 -l custom_logs"; exit 0 ;;
    *) echo "Error: Unexpected option ${flag}"
       echo "Usage: $0 [-k benchmark_keys] [-c cpu_type] [-l log_directory] [-n num_trials] [-h]"
       echo "Use -h for detailed help information"
       echo "Available benchmark keys: all,syn,tt,fctn,tw,getd,trn,mera,tnlm,tccg_default,tccg_blocked_reordered"
       echo "Available CPU types: zen4, spr, grace, apple-m2, generic"
       exit 1 ;;
  esac
done

# Set default log directory
log_dir="${log_dir:-logs}"

# ensure log directory exists
if [ ! -d "${log_dir}" ]; then
  mkdir -p "${log_dir}" || { echo "Error: Could not create log directory"; exit 1; }
fi

# validate CPU type
valid_cpus=("zen4" "spr" "grace" "apple-m2" "generic")
cpu_type=${cpu_type:-"generic"}
if [[ ! " ${valid_cpus[@]} " =~ " ${cpu_type} " ]]; then
    echo "Error: Invalid CPU type '${cpu_type}'"
    echo "Available CPU types: ${valid_cpus[*]}"
    exit 1
fi

selected_keys=${selected_keys:-"tccg_default,tccg_blocked_reordered,syn,tt,fctn,tw,getd,trn"}

num_measure_trials=${num_measure_trials:-1000}

declare -A settings
settings["syn"]="einsum_benchmark/synthetic_tvm.py"
settings["tt"]="tensor_decomp/tt_tvm.py"
settings["fctn"]="tensor_decomp/fctn_tvm.py"
settings["tw"]="tensor_decomp/tw_tvm.py"
settings["getd"]="tensor_decomp/getd_tvm.py"
settings["trn"]="tensor_layers/trn_tvm.py"
settings["mera"]="einsum_benchmark/str_nw_mera_open_26_tvm.py"
settings["tnlm"]="einsum_benchmark/lm_first_last_brackets_4_16d_tvm.py"
settings["tccg_default"]="tccg/tvm_default.py"
settings["tccg_blocked_reordered"]="tccg/tvm_blocked_reordered.py"

for key in $(echo "${!settings[@]}" | tr ' ' '\n')
do
  settings[${key}]="einsum_ir/samples/${settings[${key}]}"
done

# check if settings exist
for key in $(echo "${!settings[@]}" | tr ' ' '\n')
do
  if [ ! -f "${settings[${key}]}" ]; then
    echo "Error: TVM setting ${settings[${key}]} not found"
    exit 1
  fi
done

# prepare list of keys to process
keys_to_process=()
if [ "${selected_keys}" = "all" ]; then
    readarray -t keys_to_process < <(printf '%s\n' "${!settings[@]}" | sort)
else
    IFS=',' read -ra keys_to_process <<< "${selected_keys}"
    for key in "${keys_to_process[@]}"; do
        if [ -z "${settings[$key]}" ]; then
            echo "Error: Invalid benchmark key '${key}'"
            echo "Available keys: ${!settings[@]}"
            exit 1
        fi
    done
fi

# iterate over settings and execute
for key in "${keys_to_process[@]}"; do
    echo "Running ${key} benchmark"
    date
    mkdir -p "${log_dir}/${key}"
    log_file="${log_dir}/${key}/tvm.log"
    tuning_file="${log_dir}/${key}/tvm_tuning.json"

    cat /dev/null > "${log_file}"
    cat /dev/null > "${tuning_file}"

    PYTHONUNBUFFERED=1 PYTHONPATH=$PYTHONPATH:einsum_ir/samples/tools python ${settings[${key}]} --cpu ${cpu_type} --log_file ${tuning_file} --num_measure_trials ${num_measure_trials} >> "${log_file}"
    date
done