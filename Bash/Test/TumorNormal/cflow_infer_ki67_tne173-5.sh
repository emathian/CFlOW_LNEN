#!/bin/bash
#SBATCH --job-name=CFlowTNE14-5
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1
##SBATCH -C v100-32g
#SBATCH --partition=gpu_p2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1 # 4
# nombre de taches MPI par noeud
#SBATCH --time=30:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowOnImageNet_Ki67_TNE173-5.out          # nom du fichier de sortie
#SBATCH --error=CFlowOnImageNet_Ki67_TNE173-5.error     
#SBATCH --account ohv@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0
python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --action-type norm-test --dataset TumorNormal  --class-name Tumor --checkpoint /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/tumor_normal_ki67_2809/Tumor/10/TumorNormal_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_Tumor_10_4.pt --list-file Tiles_TNE173-5.txt  --viz-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/viz/TumorNormal_Inf_ki67_TNE173-5_2809_1004