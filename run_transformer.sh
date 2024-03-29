#!/bin/bash
#SBATCH --job-name=yh
#SBATCH -o output/tr_%j.out
#SBATCH -e output/tr_%j.err
#SBATCH --mem=500G
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=6  # total number of tasks across all nodes
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-task=4

source ~/.bashrc_dcs
conda activate mmldm_dcs_tempt
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " NGPUs per node:= " $SLURM_GPUS_PER_NODE
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NOD


echo " Running on multiple nodes and GPU devices"
echo ""
echo " Run started at:- "
date

srun python train.py --stage train_caption -b configs/train_caption.yaml -r training_logs/2024-02-19T08-18-55_exp_main/checkpoints/last.ckpt --mm_ckpt /gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/music_motion_diffusion/mm_transformer_logs/2023-11-16T23-46-19_mm_transformer_v15/checkpoints/last.ckpt

echo "Run completed at:- "
date
