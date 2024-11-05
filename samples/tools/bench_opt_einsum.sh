#!/bin/bash
# parse command line arguments
while getopts 'pl:r:b:k:h' flag; do
  case "${flag}" in
    p) enable_ht=1 ;;
    l) log_dir="${OPTARG}" ;;
    r) num_reps="${OPTARG}" ;;
    b) backend="${OPTARG}" ;;
    k) selected_keys="${OPTARG}" ;;
      h) echo "Usage: $0 [-p enable_hyperthreading] [-l log_directory] [-r num_repetitions] [-b backend] [-k benchmark_keys]"
         echo "Optional arguments:"
         echo "  -p: Enable hyperthreading"
         echo "  -l: Log directory"
         echo "  -r: Number of repetitions (default: 1)"
         echo "  -b: Backend to use (default: torch)"
         echo "  -k: Benchmark keys to run (default: all)"; exit 0 ;;
      *) echo "Unexpected option ${flag}"
         echo "Usage: $0 [-p enable_hyperthreading] [-l log_directory] [-r num_repetitions] [-b backend] [-k benchmark_keys]"
         exit 1 ;;
  esac
done

# determine number of cores if not provided
if [ -z "${num_cores}" ]; then
  num_cores=$(lscpu -p | grep -v '^#' | sort -u -t, -k 2,2 | wc -l)
fi

# default values
log_dir="${log_dir:-logs}"
enable_ht="${enable_ht:-0}"
selected_keys="${selected_keys:-all}"
num_threads=$((num_cores * (1 + enable_ht)))
num_reps=${num_reps:-1}
backend="${backend:-torch}"
script_dir="$(dirname "$(readlink -f "$0")")"

# validate backend
case "${backend}" in
  torch) ;;
  *) echo "Error: Invalid backend '${backend}'. Must be torch"; exit 1 ;;
esac

# ensure log directory exists
if [ ! -d "${log_dir}" ]; then
  mkdir -p "${log_dir}" || { echo "Error: Could not create log directory"; exit 1; }
fi

# determine OMP_PLACES for explicit thread pinning
if [ "${enable_ht}" -eq 1 ]; then
  omp_places="$(for i in $(seq 0 $((num_cores * 2 - 1))); do echo "{$i}"; done | paste -sd,)"
else
  omp_places="$(for i in $(seq 0 $((num_cores - 1))); do echo "{$i}"; done | paste -sd,)"
fi


date
echo "Running bench_opt_einsum.sh with the following settings:"
echo "  Number of cores: ${num_cores}"
echo "  Hyperthreading: ${enable_ht}"
echo "  Number of threads: ${num_threads}"
echo "  OMP_PLACES: ${omp_places}"
echo "  Number of repetitions: ${num_reps}"
echo "  Backend: ${backend}"
echo "  Selected benchmark keys: ${selected_keys}"
echo ""

declare -A settings
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

for key in $(echo "${!settings[@]}" | tr ' ' '\n')
do
  settings[${key}]="einsum_ir/samples/${settings[${key}]}"
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

# iterate over the settings and print key and value
for key in "${keys_to_process[@]}"
do
  date
  echo "Running benchmark setting: ${settings[${key}]}"

  mkdir -p "${log_dir}/${key}"
  log_file="${log_dir}/${key}/opt_einsum_backend_${backend}.log"
  cat /dev/null > "${log_file}"

  total_settings=$(grep -c '^' "${settings[${key}]}")
  current_setting=0
  while IFS= read -r setting || [ -n "$setting" ]; do
    ((current_setting++))
    echo -n "  Processing setting ${current_setting}/${total_settings}"
    
    # Parse the einsum expression, dimensions and contraction path from the setting
    read -r expression dim_sizes cont_path <<< "$setting"
    
    command="python ${script_dir}/opt_einsum_run.py --expression ${expression} --dim_sizes ${dim_sizes} --cont_path ${cont_path} --backend ${backend} --dtype FP32"

    for rep in $(seq 1 ${num_reps})
    do
      OMP_NUM_THREADS=${num_threads} OMP_PLACES=${omp_places} OMP_PROC_BIND=true eval ${command} >> ${log_file} 2>&1
      echo -n "."
    done
    echo ""
  done < "${settings[${key}]}"
  echo ""
done

date
echo "Done"
