#!/bin/bash
#SBATCH --job-name=DevCFlowTCAC
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH --partition=gpu_p4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:4 # 4
# nombre de taches MPI par noeud
#SBATCH --time=02:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowOnImageNet_TCAC_Dev.out          # nom du fichier de sortie
#SBATCH --error=CFlowOnImageNet_TCAC_Dev.error     
#SBATCH --account ohv@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0
mkdir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/TumorNormalKi67_TNE1977_seg
srun python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --dataset TCAC   --root-data-path /gpfsscratch/rech/ohv/ueu39kt/Tiles_HE_all_samples_384_384_Vahadane_2 --weights-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/AC0_train_dev --list-file-train DEV_training_ac_0_segment_list.txt --list-file-test DEV_validation_ac_5_list.txt --meta-epochs 5 --sub-epochs 1 --parallel
