#!/bin/bash
#SBATCH --job-name="trace_hybrid_qa"
#SBATCH --account="p_gptx"
#SBATCH --mail-user=lena.jurkschat@tu-dresden.de
#SBATCH --mail-type=END
#SBATCH --time=1:00:00
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mincpus=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=10312M
#SBATCH --output=hybrid_qa_trace.out
#SBATCH --error=hybrid_qa_trace.err

module use /beegfs/ws/0/s8916149-scorep-gpu/installed/modenv
module load release/23.04 GCC/11.3.0 OpenMPI/4.1.4 Score-P/gpu-metrics2 Python/3.10

source hybridqa_env/bin/activate
cd app

export HF_HOME=/beegfs/ws/0/s6690609-hybridQA


#Sccore-P Variables
export SCOREP_CUDA_ENABLE=yes
export SCOREP_ENABLE_TRACING=yes
export SCOREP_ENABLE_PROFILING=yes
export SCOREP_EXPERIMENT_DIRECTORY=$HF_HOME/traces
export SCOREP_VERBOSE=false
export SCOREP_CUDA_BUFFER=4G
export SCOREP_TOTAL_MEMORY=4095M
export SCOREP_PROFILING_MAX_CALLPATH_DEPTH=700

srun python -m scorep --cuda --mpp=mpi main.py --dataset qald --model-name 'mosaicml/mpt-7b-8k-chat' --deterministic-prompting True --refined_cache_dir /beegfs/ws/0/s6690609-hybridQA
