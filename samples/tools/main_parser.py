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
        input_einsum_tree = '[[[[33,3,4],[[[[15,17,31],[30,31,32,33]->[15,17,30,32,33]],[[[20,23,26],[26,27,28,29]->[20,23,27,28,29]],[[21,27,30],[[13,15,19],[[18,19,20,21],[[12,13],[12,14,18]->[13,14,18]]->[13,14,19,20,21]]->[14,15,20,21]]->[14,15,20,27,30]]->[14,15,23,28,29,30]]->[14,17,23,28,29,32,33]],[[29,32,1,2],[2,3,41]->[1,3,29,32,41]]->[1,3,14,17,23,28,33,41]]->[1,4,14,17,23,28,41]],[[[34,35,38],[[16,34,10],[[14,16,22],[[22,23,24,25],[24,35,36]->[22,23,25,35,36]]->[14,16,23,25,35,36]]->[10,14,23,25,34,35,36]]->[10,14,23,25,36,38]],[[36,37,39],[[25,28,37,0],[0,1,40]->[1,25,28,37,40]]->[1,25,28,36,39,40]]->[1,10,14,23,28,38,39,40]]->[4,10,17,38,39,40,41]],[[4,5,42],[5,6]->[4,6,42]]->[6,10,17,38,39,40,41,42]],[[6,7,43],[[7,8],[[17,9,11],[8,9,44]->[8,11,17,44]]->[7,11,17,44]]->[6,11,17,43,44]]->[10,11,38,39,40,41,42,43,44]'

    parser_einsum(input_einsum_tree)