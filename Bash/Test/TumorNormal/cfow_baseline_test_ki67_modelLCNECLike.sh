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
#SBATCH --time=09:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowOnImageNet_Tumor_ki67_modelLCNECLike.out          # nom du fichier de sortie
#SBATCH --error=CFlowOnImageNet_Tumor_modelLCNECLike.error     
#SBATCH --account ohv@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0

# echo ===============================================================
# echo                        INFER TEST
# echo ===============================================================
# python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --action-type norm-test --dataset TumorNormal  --class-name Tumor --checkpoint /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/TumorNormalKi67_LCNECLike/Tumor/24/TumorNormal_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_Tumor_24_7.pt --list-file-test LCNECLike_test_tiles_list.txt --viz-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/viz/TumorNormal_Ki67_modelLCNECLike --root-data-path /gpfsscratch/rech/ohv/ueu39kt/KI67_Tumoral_Normal_V2


# echo ===============================================================
# echo                        INFER TRAIn
# echo ===============================================================
# python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --action-type norm-test --dataset TumorNormal  --class-name Tumor --checkpoint /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/TumorNormalKi67_LCNECLike/Tumor/24/TumorNormal_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_Tumor_24_7.pt --infer-train --list-file-train LCNECLike_train_tiles_list.txt --viz-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/viz/TumorNormal_Ki67_modelLCNECLike --root-data-path /gpfsscratch/rech/ohv/ueu39kt/KI67_Tumoral_Normal_V2



echo ===============================================================
echo                        INFER all
echo ===============================================================
mkdir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/viz/TumorNormal_Ki67_modelTNE1607_Infer
python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --action-type norm-test --dataset TumorNormal  --class-name Tumor --checkpoint /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/TumorNormalKi67_LCNECLike/Tumor/24/TumorNormal_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_Tumor_24_7.pt --list-file-test LCNECLike_infer_tiles_list.txt --viz-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/viz/TumorNormal_Ki67_modelLCNECLike_Infer --root-data-path  /gpfsscratch/rech/ohv/ueu39kt/KI67_Tiling_256_256_40x
