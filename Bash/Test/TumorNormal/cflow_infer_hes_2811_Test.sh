#!/bin/bash
#SBATCH --job-name=CF_HES2711_TNE404
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
##SBATCH -C v100-32g
#SBATCH --partition=gpu_p2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1 # 4
# nombre de taches MPI par noeud
#SBATCH --time=05:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowOnImageNet_HES_missing_TNE404_2711.out          # nom du fichier de sortie
#SBATCH --error=CFlowOnImageNet_HES_missing_TNE404_2711.error     
#SBATCH --account uli@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0
python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --action-type norm-test --dataset TumorNormal  --class-name Tumor --checkpoint /gpfsscratch/rech/uli/ueu39kt/CFLOW/weights/TumorNormal/Tumor/11/TumorNormal_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_Tumor_11_2.pt --list-file-test test_individual_segmentation_model.txt --viz-dir /gpfsscratch/rech/uli/ueu39kt/CFLOW/viz/Tiles_HES_TumorNorm_0812_Test --root-data-path /gpfsscratch/rech/uli/ueu39kt/TumoralNormalForFastFlow