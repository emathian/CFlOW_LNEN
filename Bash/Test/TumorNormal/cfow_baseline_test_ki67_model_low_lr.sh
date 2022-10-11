#!/bin/bash
#SBATCH --job-name=CFlowTKi67
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
##SBATCH -C v100-32g
#SBATCH --partition=gpu_p2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
##SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1 # 4
# nombre de taches MPI par noeud
#SBATCH --time=05:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowOnImageNet_Tumor_ki67_model_low_lr.out          # nom du fichier de sortie
#SBATCH --error=CFlowOnImageNet_Tumor_model_low_lr.error     
#SBATCH --account ohv@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0
python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --action-type norm-test --dataset TumorNormal  --class-name Tumor --checkpoint /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/tumor_normal_ki67_low_lr_3009/Tumor/5/TumorNormal_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_Tumor_5_9.pt --list-file TestTumorNormal.txt --infer_train --viz-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/viz/TumorNormal_Ki67_model_low_lr_ep5_9

