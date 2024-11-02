# this parser create tvm code 
# usage: python main_parser.py > tvm_function.py
#        python main_parser.py '[[c,i,g,j],[i,a,j,e]->[a,c,e,g,i]],[[d,h],[[b,f],[d,c,b,a]->[a,c,d,f]]->[a,c,f,h]]->[h,g,f,e,i]' > tvm_function.py


from parser_backend import *
import sys

def parser_einsum(input_einsum_tree):
    input_einsum_array = []
    input_tensor_array = []
    all_input_tensors = []
    later_input_tensors = []

    print("#######################")
    print("# input_string: "+ input_einsum_tree)
    print("#######################\n")

    pattern = r'(\[(\w,?)+\])\s*->\s*(\[(\w,?)+\])'

    input_einsum_tree = re.sub(pattern, remove_transposiontions, input_einsum_tree)

    dim_count = count_unique_elements(input_einsum_tree)

    print("@tvm.auto_scheduler.register_workload")
    if check_string(input_einsum_tree):
        print(f'def einsum_tree( ' + ", ".join(f"dim_{i}" for i in range(dim_count[0])) + ', dtype):')
    else:
        print(f'def einsum_tree( ' + ", ".join(f"dim_{chr(i)}" for i in range(97, (97+dim_count[0]))) + ', dtype):')
    print_single_expressions(input_einsum_tree, input_einsum_array, later_input_tensors)

    for i in input_einsum_array:
        input_tensor_array.append(i.split('->')[0])


    for i in input_tensor_array:
        split_index = i.find('],[')
        all_input_tensors.append(i[:split_index+1])
        all_input_tensors.append(i[split_index+2:])

    # merge all_input_tensors and later_input_tensors
    all_input_tensors = all_input_tensors + later_input_tensors

    input_arrays = generate_tvm_expressions_placeholder(all_input_tensors)

    print()

    generate_tvm_reduce_axis(input_einsum_tree)

    print()

    generate_tvm_compute(input_einsum_tree, input_arrays)



# main function
# Prüfen, ob das Skript direkt ausgeführt wird
if __name__ == "__main__":
    if len(sys.argv) > 1:
        input_einsum_tree = sys.argv[1]
    else:
        input_einsum_tree = '[[8,4],[7,3,8]->[7,3,4]],[[[[6,3,7]->[3,6,7]],[[5,1,6]->[1,5,6]]->[1,2,5,7]],[0,5]->[0,1,2,7]]->[0,1,2,3,4]'

    parser_einsum(input_einsum_tree)