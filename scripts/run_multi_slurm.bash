#!/bin/bash

# Function to scale time based on the time_scaling value
time_scaler() {
    local time=$1
    local time_scaling=$2
    local max_time=72000  # Max allowed time in seconds (20 hours)

    # Convert time to seconds for computation
    IFS=: read -r h m s <<< "$time"
    total_time=$(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))

    # Scale time using awk for floating-point support
    scaled_time=$(awk "BEGIN { printf \"%.0f\", $total_time * $time_scaling }")

    # Check if scaled time exceeds the max allowed time
    if (( scaled_time > max_time )); then
        echo "Error: Computed time exceeds max allowed time."
        exit 1
    fi

    # Convert back to HH:MM:SS format
    new_h=$((scaled_time / 3600))
    new_m=$(( (scaled_time % 3600) / 60 ))
    new_s=$((scaled_time % 60))
    formatted_time=$(printf "%02d:%02d:%02d" "$new_h" "$new_m" "$new_s")

    echo "$formatted_time"
}

# Function to get values from the INI file - simplified version
get_value_from_ini() {
    local section=$1
    local key=$2
    local ini_file=$3
    
    sed -n "/^\[$section\]/,/^\[/p" "$ini_file" | grep "^$key *= *" | sed 's/[^=]*= *\(.*\)/\1/' | sed 's/[[:space:]]*#.*$//' | tr -d '[:space:]'
}

run_multi_slurm() {
    # Default values
    time="00:10:00"
    dataset="Toy_example"
    model_big_set="TEST_SET"
    ini_file="config/models_config.ini"
    delay=0.002

    # Parse named arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --time)
                time="$2"
                shift 2
                ;;
            --dataset)
                dataset="$2"
                shift 2
                ;;
            --model_big_set)
                model_big_set="$2"
                shift 2
                ;;
            --delay)
                delay="$2"
                shift 2
                ;;
            *)
                echo "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    # Handle delay (only echoing first, then actual sleep)
    if (( $(echo "$delay > 0.02" | bc -l) )); then
        echo "Will delay execution by $delay hours..."
    fi
        

    echo "Searching for models in set: $model_big_set"

    # Get all sections from INI file
    declare -a model_sets
    while read -r section; do
        section=$(echo "$section" | sed 's/^\[\([^]]*\)\].*/\1/' | sed 's/[[:space:]]*#.*$//')

        model_big_sets=$(get_value_from_ini "$section" "model_big_set" "$ini_file")

        IFS=',' read -ra sets <<< "$model_big_sets"
        for set in "${sets[@]}"; do
            if [ "$(echo "$set" | tr -d '[:space:]')" = "$model_big_set" ]; then
                model_sets+=("$section")
                break
            fi
        done
    done < <(grep '^\[' "$ini_file")

    echo "Found ${#model_sets[@]} matching models: ${model_sets[*]}"

    # Loop through the models in the set
    for model in "${model_sets[@]}"; do
        time_scaling=$(get_value_from_ini "$model" "time_scaling" "$ini_file")
        nb_of_gpu=$(get_value_from_ini "$model" "nb_of_gpu" "$ini_file")
        batch_size=$(get_value_from_ini "$model" "batch_size" "$ini_file")

        [[ -z "$time_scaling" ]] && time_scaling=1.0
        [[ -z "$nb_of_gpu" ]] && nb_of_gpu=1
        [[ -z "$batch_size" ]] && batch_size=5

        model_short=$(basename "$model")
        formatted_time=$(time_scaler "$time" "$time_scaling")

        echo "-----------------------------------------------------"
        echo "Submitting job for model: $model with time: $formatted_time and GPUs: $nb_of_gpu"
        echo "-----------------------------------------------------"
    done

    # Ask user confirmation before submitting jobs
    read -p "Type 'y' to submit the jobs, anything else to cancel: " confirm
    if [[ "$confirm" != "y" ]]; then
        echo "Job submission cancelled."
        exit 0
    fi


    # Actually submit the jobs
    for model in "${model_sets[@]}"; do
        time_scaling=$(get_value_from_ini "$model" "time_scaling" "$ini_file")
        nb_of_gpu=$(get_value_from_ini "$model" "nb_of_gpu" "$ini_file")
        batch_size=$(get_value_from_ini "$model" "batch_size" "$ini_file")
        working_lib=$(get_value_from_ini "$model" "working_lib" "$ini_file")

        if [[ -n "$working_lib" ]]; then
        module="${working_lib##*_}"
        else
            module="pytorch-gpu/py3/2.6.0"
        fi

        [[ -z "$time_scaling" ]] && time_scaling=1.0
        [[ -z "$nb_of_gpu" ]] && nb_of_gpu=1
        [[ -z "$batch_size" ]] && batch_size=5

        model_short=$(basename "$model")
        formatted_time=$(time_scaler "$time" "$time_scaling")
        cpus_count=$((8 * nb_of_gpu))

        sbatch --time="$formatted_time" \
               --gres=gpu:"$nb_of_gpu" \
               --cpus-per-task="$cpus_count" \
               --job-name="$model" \
            scripts/run_large_LLMs.slurm \
            "$module" \
            scripts/Run_Inferences.py \
            --jz --gpu "$nb_of_gpu" --bs "$batch_size" --dataset "$dataset" --model "$model" --sub_run "$model_short"

        echo "Submitted job for model: $model"
        echo
    done
}

# run_multi_slurm --time 00:30:00 --dataset Graph_Datasets/Toy_example --model_big_set TESTING_SET
