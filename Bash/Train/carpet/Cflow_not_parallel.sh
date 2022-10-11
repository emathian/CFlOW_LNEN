#!/bin/bash
#SBATCH --job-name=CFlowCarpet
#SBATCH --nodes=1
#SBATCH --partition=gpu_p2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 # 4
# nombre de taches MPI par noeud
#SBATCH --time=10:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowOnImageNet_not_parallel_carpet_0310.out          # nom du fichier de sortie
#SBATCH --error=CFlowOnImageNet_not_parallel_carpet_0310.error     
#SBATCH --account ohv@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0
srun  python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 512 --dataset mvtec --class-name carpet 
