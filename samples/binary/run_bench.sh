# arg 1: executable
# arg 2: parameter file
# arg 3: precision
# arg 4: store & lock

if [ $# -lt 3 ]
then
  echo "Usage: $0 <executable> <parameter file> <precision>"
  exit 1
fi

# parse the arguments
executable=$1
param_file=$2
precision=$3

# check if the executable exists
if [ ! -f $executable ]
then
  echo "Executable not found: $executable"
  exit 1
fi

# check if the parameter file exists
if [ ! -f $2 ]
then
  echo "Parameter file: $param_file not found"
  exit 1
fi

# check if the precision is valid
if [ $precision != "FP32" ] && [ $precision != "FP64" ]
then
  echo "Invalid precision: $precision"
  exit 1
fi

echo "einsum_string,dim_sizes,cont_path,num_flops,time_compile,time_eval,gflops_eval,gflops_total"

# run configs
while IFS= read -r args || [ -n "$args" ]
do
  # limit args to 3 (einsum_string, dim_sizes, cont_path)
  args=$(echo $args | cut -d' ' -f1-3)

  command="$executable $args $precision $store_lock 2"

  # set output incl error stream
  output=$(eval $command 2>&1)
  # filter for the line with form "CSV_DATA: einsum_ir,"beca,ade->dacb","8,8,2,256,16","(0,1)",1015808,0.00974585,0.00514188,0.197556,0.0682312"
  output=$(echo "$output" | grep "CSV_DATA: einsum_ir")
  # remove "CSV_DATA: einsum_ir," from the line
  output=$(echo "$output" | sed 's/CSV_DATA: einsum_ir,//')

  echo "$output"
done < $param_file