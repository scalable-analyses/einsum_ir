l_param_file="$1"

while read -r l_args
do
  if [ $2 ]
  then
    l_command="./build/bench_expression $l_args $2 1"
  else
    l_command="./build/bench_expression $l_args FP32 1"
  fi
  eval $l_command
done < "$l_param_file"