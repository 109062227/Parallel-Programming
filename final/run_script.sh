#!/bin/bash

# Check if the script has been called with at least the program argument
if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  $0 <program> <parameters...>"
    exit 1
fi

# Read the program name and shift the arguments
program="$1"
shift

# change the size here!  it's {h, w, k}
# result need to calc it will be h-k+1 / w-k+1 manually if you want to use tests_seeout
default_test_cases=("{8192, 8192, 96}", "{16384, 16384, 96}") # used "{10,10,3}" "{100,100,5}" "{10000,10000,32}"

# If no parameters are provided, use default cases
if [ $# -eq 0 ]; then
    echo "No parameters provided, using default cases for $program."
    set -- "${default_test_cases[@]}"
fi

# Debugging: Print received program and arguments
echo "Program: $program"
echo "Arguments: $@"

# Process the arguments into an array by removing braces and converting to space-separated format
test_array=()
for arg in "$@"; do
    parsed_arg=$(echo "$arg" | tr -d '{}' | tr ',' ' ')
    test_array+=("$parsed_arg")
done

# Debugging: Print processed array
echo "Processed test cases:"
for params in "${test_array[@]}"; do
    echo "$params"
done

# Loop through the array and call the specified program
for params in "${test_array[@]}"; do
    read -r h w k <<< "$params"
    input_file="./testcases/${h}_${w}_${k}.in"
    output_file="./testcases/created/${h}_${w}_${k}.out"

    if [[ "$program" == "./tests_gen" ]]; then
        echo "Running $program $h $w $k $input_file"
        $program "$h" "$w" "$k" "$input_file"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to execute $program with parameters $h $w $k $input_file."
        fi
    elif [[ "$program" == "./conv_seq" ]]; then
        echo "Running $program $input_file $output_file"
        $program "$input_file" "$output_file"
        if [ $? -ne 0 ]; then
            echo "Error: Failed to execute $program with input $input_file and output $output_file."
        fi
    elif [[ "$program" == "judge" ]]; then
        # echo "Running diff for test cases..."
        original_file="./testcases/${h}_${w}_${k}.out"
        created_file="./testcases/created/${h}_${w}_${k}.out"
        # echo "Comparing $original_file with $created_file..."
        diff "$original_file" "$created_file"
        if [ $? -ne 0 ]; then
            echo "Error: Files $original_file and $created_file differ."
        else
            echo "Success: Files $original_file and $created_file are identical."
        fi
    else
        echo "Error: Unsupported program $program."
        exit 1
    fi
done
