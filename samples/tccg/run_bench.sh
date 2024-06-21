#
# Write an argument parser for the bench_expression program
# arg 1: executable
# arg 2: parameter file
# arg 3: precision
# arg 4: store & lock

if [ $# -lt 4 ]
then
  echo "Usage: $0 <executable> <parameter file> <precision> <store & lock>"
  exit 1
fi

# parse the arguments
executable=$1
param_file=$2
precision=$3
store_lock=$4

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

# check if the store & lock is valid
if [ $store_lock != "0" ] && [ $store_lock != "1" ]
then
  echo "Invalid store & lock: $store_lock"
  exit 1
fi

# run configs
while IFS= read -r args || [ -n "$args" ]
do
  command="$executable $args $precision $store_lock 2"
  eval $command
done < $param_file