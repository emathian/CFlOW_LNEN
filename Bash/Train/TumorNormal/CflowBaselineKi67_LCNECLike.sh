#!/bin/bash
#SBATCH --job-name=CFlowKi67
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1
#SBATCH --partition=gpu_p13
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1 # 4
# nombre de taches MPI par noeud
#SBATCH --time=40:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowOnImageNet_Tumor_ki67_LCNECLike.out          # nom du fichier de sortie
#SBATCH --error=CFlowOnImageNet_Tumor_ki67_LCNECLike.error     
#SBATCH --account ohv@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0
python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --dataset TumorNormal --class-name Tumor  --root-data-path /gpfsscratch/rech/ohv/ueu39kt/KI67_Tumoral_Normal_V2 --weights-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/TumorNormalKi67_LCNECLike --list-file-train LCNECLike_train_tiles_list.txt --list-file-test LCNECLike_test_tiles_list.txt