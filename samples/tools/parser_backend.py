# from parser_num import *
import re

tvm_instructions = []

def count_unique_elements(einsum_string):
    # Find all numbers and individual alphabetic characters in the string
    elements = re.findall(r'\d+|[a-zA-Z]', einsum_string)
    # Convert numbers to integers, leave characters as they are, and find unique elements
    unique_elements = set(int(element) if element.isdigit() else element for element in elements)
    # Return the count of unique elements and the unique elements themselves
    return len(unique_elements), unique_elements


def contains_brackets(s):
    return '[' in s or ']' in s

def check_string(s):
    has_numbers = any(char.isdigit() for char in s)
    return has_numbers

def print_single_expressions(einsum_string, array, array_tensor_later):
    # Stack to store expressions to process
    stack = [einsum_string]
    
    while stack:
        # Get the next expression to process
        expr = stack.pop()
        if not contains_brackets(expr):
            expr = "[" + expr + "]"
            array_tensor_later.append(expr)
        
        # If there's only one "->", print the expression
        if expr.count('->') == 1:    
            array.append(expr)
            continue
        
        # Otherwise, split into root and input expressions
        root_string = expr.split('->')[-1]
        input_string = expr[:-len(root_string)-2]
        
        # Variables for tracking brackets and comma location
        left_bracket_count = 0
        right_bracket_count = 0
        counter = 0
        comma_location = 0 
        
        # Find the splitting point in the input_string
        for char in input_string:
            counter += 1
            if char == '[':
                left_bracket_count += 1
            elif char == ']':
                right_bracket_count += 1
            if left_bracket_count == right_bracket_count:
                comma_location = counter
                break
        
        # Split into left and right expressions
        left_leave = input_string[:comma_location].strip()[1:-1]
        right_leave = input_string[comma_location+1:].strip()[1:-1]
        
        # Add sub-expressions to the stack to process later
        if left_leave:
            stack.append(left_leave)
        if right_leave:
            stack.append(right_leave)


def generate_tvm_expressions_placeholder(arr):
    resulting_inputs = []
    # Loop through each string in the array and enumerate for numbering
    for idx, array_str in enumerate(arr):
        # Remove the brackets and split by commas to get dimensions
        dims = array_str.strip("[]").split(',')
        
        # Format dimensions as "dimX" where X is the value of each dimension
        dim_str = ", ".join(f"dim_{dim.strip()}" for dim in dims)
        input_tensor_names = "_".join(dims)

        print(f"  tensor_{input_tensor_names} = tvm.te.placeholder(({dim_str}), name='tensor_{input_tensor_names}', dtype=dtype)")
        resulting_inputs.append(f"tensor_{input_tensor_names}")
    return resulting_inputs



def find_missing_values(string1, string2_list):
    # Extract all alphanumeric values (both numbers and letters) from string1
    values = re.findall(r'\w+', string1)
    # Convert values to a set for easy comparison
    values_set = set(values)
    
    # Convert the second list of values directly to a set for comparison
    values_in_string2 = set(string2_list)
    
    # Find missing values
    missing_values = values_set - values_in_string2
    
    # Print missing values if any
    if missing_values:
        return missing_values
    else:
        return None

def generate_tvm_reduce_axis(input_einsum):
    root_string = input_einsum.split('->')[-1]
    input_string = input_einsum[:-len(root_string)-2]

    root_dims = root_string.strip("[]").split(',')
    reduce_axis = find_missing_values(input_string, root_dims)
    for i in reduce_axis:
        print(f"  tmp_{i} = tvm.te.reduce_axis((0, dim_{i}), name='tmp_{i}')")



def extract_binary_contraction( einsum_str):
    root_string = einsum_str.split('->')[-1]
    einsum_str = einsum_str[:-len(root_string)-2]

    index = 0

    for i in range(len(einsum_str)-1, 0, -1):
        if einsum_str[i] == ']' and einsum_str[i-1].isalnum():
            index = i
            break

    start_index = -1
    for j in range(index, -1, -1):
        if einsum_str[j] == '[':
            start_index = j
            break

    right_side = einsum_str[start_index:index+1] 

    counter = 0
    comma_location = 0
    left_bracket_count = 0 
    right_bracket_count = 0 

    for char in einsum_str:
        counter += 1
        if char == '[':
            left_bracket_count += 1
        if left_bracket_count == right_bracket_count:
            comma_location = counter

        elif char == ']':
            right_bracket_count += 1
        if left_bracket_count == right_bracket_count:
            comma_location = counter
        if counter == comma_location:
            break

    left_location_start = 0
    for i in range(comma_location, -1, -1):
        if einsum_str[i] == '[':
            left_location_start = i
            break
    left_side = "[" + einsum_str[left_location_start:comma_location].replace('[', '').replace(']', '') + "]"
    
    input_string = left_side + "," + right_side
    root_dims = root_string.strip("[]").split(',')
    reduce_axis = find_missing_values(input_string, root_dims)
    
    new_inst = f"  tensor_{'_'.join(root_string.strip('[]').split(','))} = tvm.te.compute( (dim_{', dim_'.join(root_string.strip('[]').split(','))}), lambda tmp_{', tmp_'.join(root_string.strip('[]').split(','))}: tvm.te.sum( tensor_{'_'.join(left_side.strip('[]').split(','))}[ tmp_{', tmp_'.join(left_side.strip('[]').split(','))} ] * tensor_{'_'.join(right_side.strip('[]').split(','))}[ tmp_{', tmp_'.join(right_side.strip('[]').split(','))} ] , axis=[ tmp_{', tmp_'.join(reduce_axis)} ]), name='tensor_{'_'.join(root_string.strip('[]').split(','))}' )" 
    tvm_instructions.append(new_inst)


    return root_string, left_side, right_side

def split_tree( einsum_tree, root_string, left_string, right_string ):

    sub_string = einsum_tree.replace(root_string, '')

    # remove last two characters
    sub_string = sub_string[:-2]

    # split where the first time open and closed brackets are equal
    index = 0
    counter = 0
    left_bracket_count = 0
    right_bracket_count = 0
    for char in sub_string:
        counter += 1
        if char == '[':
            left_bracket_count += 1
        if left_bracket_count == right_bracket_count:
            index = counter
        elif char == ']':
            right_bracket_count += 1
        if left_bracket_count == right_bracket_count:
            index = counter
        if counter == index:
            break
    
    
    # split the string by the index
    left_tree = sub_string[:index+1]
    right_tree = sub_string[index+1:]

    left_tree = left_tree[1:-2]
    right_tree = right_tree[1:-1]

    return left_tree, right_tree

def traverse_tree( big_string ):

    if( big_string.count('->') == 0 ):
        return
    
    root_string, left_string, right_string = extract_binary_contraction( big_string )
    
    left_tree, right_tree = split_tree( big_string, root_string, left_string, right_string )
    traverse_tree( left_tree )
    traverse_tree( right_tree )


def generate_tvm_compute(input_string, input_array):
    
    traverse_tree( input_string )
    for i in tvm_instructions[::-1]:
        print(i)
    print()
    print( f'  return [ ' +  ", ".join(input_array) + f', {tvm_instructions[0].split()[0] if tvm_instructions[0] else "error"} ]')


