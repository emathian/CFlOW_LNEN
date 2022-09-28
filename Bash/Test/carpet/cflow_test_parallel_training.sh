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
echo  First  test parallel
echo ========================================================================
srun python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 512 --action-type norm-test --dataset mvtec --class-name carpet --gpu 0 --checkpoint  /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/carpet_parallel_2809_2/carpet/1/mvtec_wide_resnet50_2_freia-cflow_pl3_cb8_inp512_run0_carpet_1_0.pt


# echo ========================================================================
# echo  First Bis test
# echo ========================================================================
# python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 512 --action-type norm-test --dataset mvtec   --class-name carpet --checkpoint  /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/carpet_parallel_2809_2/carpet/1/mvtec_wide_resnet50_2_freia-cflow_pl3_cb8_inp512_run0_carpet_1_0.pt


# echo ========================================================================
# echo  Second test
# echo ========================================================================

# python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 512 --action-type norm-test --dataset mvtec   --class-name carpet --checkpoint  /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/carpet_parallel_2809_2/carpet/2/mvtec_wide_resnet50_2_freia-cflow_pl3_cb8_inp512_run0_carpet_2_0.pt
# echo ========================================================================
# echo  Third test
# echo ========================================================================

# python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 512 --action-type norm-test --dataset mvtec   --class-name carpet --checkpoint  /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/carpet_parallel_2809_2/carpet/10/mvtec_wide_resnet50_2_freia-cflow_pl3_cb8_inp512_run0_carpet_10_0.pt

# echo ========================================================================
# echo Not parallel
# echo ========================================================================
# python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 512 --action-type norm-test --dataset mvtec   --class-name carpet --checkpoint  /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/carpet_parallel_2809/carpet/mvtec_wide_resnet50_2_freia-cflow_pl3_cb8_inp512_run0_carpet_2022-09-28-10:25:41.pt


