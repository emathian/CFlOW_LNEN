#!/bin/bash
#SBATCH --job-name=CFlowLCNECLike_Massimo_VAL1
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1
##SBATCH -C v100-32g
#SBATCH --partition=gpu_p13
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1 # 4
# nombre de taches MPI par noeud
#SBATCH --time=40:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowLCNECLike_Massimo_VAL1.out          # nom du fichier de sortie
#SBATCH --error=CFlowLCNECLike_Massimo_VAL1.error     
#SBATCH --account ohv@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0


python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --action-type norm-test --dataset TumorNormal  --class-name Tumor --checkpoint /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/TumorNormal/Tumor/11/TumorNormal_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_Tumor_11_2.pt --list-file-test Tiles_VAL1.txt --viz-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/viz/TumorNormal_Inf_LCNEC_Massimo_VAL1 --root-data-path  /gpfsscratch/rech/ohv/ueu39kt/Tiles_HE_massimo_LCNEC_384_384_Vahadane
