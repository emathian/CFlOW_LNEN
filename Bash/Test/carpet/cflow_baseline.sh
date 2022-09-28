#!/bin/bash
#SBATCH --job-name=CFlowCarpet
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
##SBATCH -C v100-32g
#SBATCH --partition=gpu_p2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
##SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1 # 4
# nombre de taches MPI par noeud
#SBATCH --time=00:30:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowOnImageNet_carpet.out          # nom du fichier de sortie
#SBATCH --error=CFlowOnImageNet_carpet.error     
#SBATCH --account ohv@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0
python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 512 --action-type norm-test --dataset mvtec  --infer_train --class-name carpet --checkpoint /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/carpet/mvtec_wide_resnet50_2_freia-cflow_pl3_cb8_inp512_run0_carpet_2022-08-04-07:34:07.pt
