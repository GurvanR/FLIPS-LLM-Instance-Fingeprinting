  # Example SLURM script for HPC clusters (Jean-Zay style). Set --account to your own allocation.
  #!/bin/bash

  #SBATCH --job-name=flips-inference
  #SBATCH --output=runs_outputs/outputs/%x_%j.out
  #SBATCH --error=runs_outputs/errors/%x_%j.out

  #SBATCH --gres=gpu:1
  #SBATCH --ntasks-per-node=1
  #SBATCH --nodes=1
  #SBATCH --hint=nomultithread
  #SBATCH --time=00:30:00 # Default time (can be overridden in sbatch command)
  #SBATCH --partition=<YOUR_PARTITION>
  #SBATCH --cpus-per-task=8
  #SBATCH --account=YOUR_HPC_ACCOUNT
  #SBATCH -C <YOUR_GPU_CONSTRAINT>

  # Load specified module
  module purge
  conda deactivate
  # Load your cluster's PyTorch/GPU modules here.
  # module load arch/a100
  # module load pytorch-gpu/py3/2.8.0

  # Get executable
  EXECUTABLE="$1"
  shift

  if [[ -z "$EXECUTABLE" ]]; then
    echo "Error: an executable must be provided as the first argument."
    echo "Usage: $0 <module> <executable> [additional arguments]"
    exit 1
  fi
  # Redirect caches to $SCRATCH to avoid inode quota exhaustion.
  export XDG_CACHE_HOME=$SCRATCH/.cache
  export HF_HOME=$SCRATCH/.cache/huggingface
  export TORCH_HOME=$SCRATCH/.cache/torch
  export PIP_CACHE_DIR=$SCRATCH/.cache/pip
  export MPLCONFIGDIR=$SCRATCH/.cache/matplotlib

  # Set environment variables
  export OUTLINES_CACHE_DIR=$JOBSCRATCH/cache/
  export VLLM_WORKER_MULTIPROC_METHOD=spawn


  # Enable verbose execution
  set -x

  # Run the job
  srun python "$EXECUTABLE" "$@"

  # Print execution time
  duration=$SECONDS
  echo "$((duration / 60)) min $((duration % 60)) sec"

  # Example:

  #sbatch --time=00:30:00 --gres=gpu:4 scripts/run_large_LLMs.slurm <module_name> scripts/Run_Inferences.py --jz --dataset Toy_example --gpu 1 --model llama_small_2 --sub_run test_example
  #sbatch --time=01:00:00 --gres=gpu:1 --cpus-per-task=8 scripts/run_large_LLMs.slurm pytorch-gpu/py3/2.8.0 "$XP_SCRIPT_PATH\" --xp_config_path \"$file\" --run_name \"$RUN_NAME\" --xp_suffix \"$MULTI_XP_NAME\""


