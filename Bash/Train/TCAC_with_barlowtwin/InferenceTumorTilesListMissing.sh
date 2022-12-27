#!/bin/bash
#SBATCH --job-name=CFlowBT
#SBATCH --qos=qos_cpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
# nombre de taches MPI par noeud
#SBATCH --time=10:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=InferenceTumorTilesListMissing.out          # nom du fichier de sortie
#SBATCH --error=InferenceTumorTilesListMissing.error     
#SBATCH --account uli@cpu

module purge
export PYTHONUSERBASE=/gpfswork/rech/uli/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0
srun python inference_list_tumor_tiles.py
