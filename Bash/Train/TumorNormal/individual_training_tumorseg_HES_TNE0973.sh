#!/bin/bash
#SBATCH --job-name=CFlow_ind_seg
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1
#SBATCH --partition=gpu_p13
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1 # 4
# nombre de taches MPI par noeud
#SBATCH --time=40:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowOnImageNet_Tumor_HES_individual_TNE0973.out          # nom du fichier de sortie
#SBATCH --error=CFlowOnImageNet_Tumor_HES_individual_TNE0973.error     
#SBATCH --account ohv@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0
mkdir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/AC0_train_all
srun python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --dataset TumorNormal --class-name Tumor  --root-data-path /gpfsscratch/rech/ohv/ueu39kt/TumoralNormalForFastFlow_difficult_cases --weights-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/TumorSeg_HES_individual_training_TNE0973 --list-file-train TNE0973_test_individual_segmentation_model.txt --list-file-test TNE0973_train_individual_segmentation_model.txt --meta-epochs 30 --sub-epochs 8  --lr 2e-4
