if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_param_file> <output_param_file>"
    exit 1
fi

input_param_file=$1
output_param_file=$2

# check if the input parameter file exists
if [ ! -f $1 ]; then
    echo "Parameter file: $param_file not found"
    exit 1
fi

# run blocking
while read -r args
do
    python blocking.py $args
done < $input_param_file > $output_param_file
