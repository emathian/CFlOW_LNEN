#!/bin/bash
#SBATCH --job-name=CFlowCarpet
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH --partition=gpu_p2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1 # 4
#SBATCH --time=00:30:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowOnImageNet_carpet_parallel_no_distribution_test_set_0710_1004.out          # nom du fichier de sortie
#SBATCH --error=CFlowOnImageNet_carpet_parallel_no_distribution_test_set_0710_1004.error     
#SBATCH --account ohv@v100



module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0
echo ========================================================================
echo  First  test parallel epoch 5
echo ========================================================================
srun python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 512 --action-type norm-test --dataset mvtec --class-name carpet --gpu 0 --checkpoint  /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/carpet_parallel_no_distribution_test_set_0710/carpet/5/mvtec_wide_resnet50_2_freia-cflow_pl3_cb8_inp512_run0_carpet_5_0.pt

