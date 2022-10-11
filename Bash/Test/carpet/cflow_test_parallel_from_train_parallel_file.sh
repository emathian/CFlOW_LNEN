#!/bin/bash
#SBATCH --job-name=CFlowCarpet
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH --partition=gpu_p2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1 # 4
#SBATCH --time=00:30:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowOnImageNet_carpet_PAR.out          # nom du fichier de sortie
#SBATCH --error=CFlowOnImageNet_carpet_PAR.error     
#SBATCH --account ohv@v100



module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0
echo ========================================================================
echo  First  test parallel from train parallel file epoch 24
echo ========================================================================
srun python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 512 --action-type norm-test --dataset mvtec --class-name carpet --gpu 0 --checkpoint  /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/carpet_parallel_2809_4/carpet/24/mvtec_wide_resnet50_2_freia-cflow_pl3_cb8_inp512_run0_carpet_24_0.pt

echo ========================================================================
echo  First  test parallel from best checkpoint
echo ========================================================================
srun python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 512 --action-type norm-test --dataset mvtec --class-name carpet --gpu 0 --checkpoint  /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/carpet_parallel_2809_4/carpet/24/mvtec_wide_resnet50_2_freia-cflow_pl3_cb8_inp512_run0_carpet_2022-09-30-14:18:56.pt
