#!/bin/bash
#SBATCH --job-name=CFlowTCAC
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1
#SBATCH --partition=gpu_p2
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:4 # 4
# nombre de taches MPI par noeud
#SBATCH --time=100:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowOnImageNet_TCAC_segmented.out          # nom du fichier de sortie
#SBATCH --error=CFlowOnImageNet_TCAC_segmented.error     
#SBATCH --account ohv@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0

mkdir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/AC0_train_segmented

srun python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --dataset TCAC   --root-data-path /gpfsscratch/rech/ohv/ueu39kt/Tiles_HE_all_samples_384_384_Vahadane_2 --weights-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/AC0_train_segmented --list-file-train training_ac_0_segment_list.txt --list-file-test validation_ac_5_list.txt --meta-epochs 25 --sub-epochs 8 --parallel --lr 2e-4
