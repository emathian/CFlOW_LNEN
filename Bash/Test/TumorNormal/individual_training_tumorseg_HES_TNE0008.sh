#!/bin/bash
#SBATCH --job-name=CFlowTHES
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
##SBATCH -C v100-32g
#SBATCH --partition=gpu_p2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
##SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1 # 4
# nombre de taches MPI par noeud
#SBATCH --time=06:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowOnImageNet_Tumor_ki67_modelTNE008.out          # nom du fichier de sortie
#SBATCH --error=CFlowOnImageNet_Tumor_modelTNE008.error     
#SBATCH --account ohv@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0

# mkdir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/viz/TumorNormal_Ki67_modelTNE1895

# echo ===============================================================
# echo                        INFER TEST
# echo ===============================================================
# python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --action-type norm-test --dataset TumorNormal  --class-name Tumor --checkpoint /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/TumorSeg_HES_individual_training_TNE0008/Tumor/29/TumorNormal_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_Tumor_29_7.pt --list-file-test TNE0008_test_individual_segmentation_model.txt --viz-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/viz/TumorNormal_indivuidual_model_TNE0008 --root-data-path  /gpfsscratch/rech/ohv/ueu39kt/TumoralNormalForFastFlow_difficult_cases


# echo ===============================================================
# echo                        INFER TRAIN
# echo ===============================================================
# python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --action-type norm-test --dataset TumorNormal  --class-name Tumor --checkpoint /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/TumorSeg_HES_individual_training_TNE0008/Tumor/29/TumorNormal_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_Tumor_29_7.pt --infer-train --list-file-train TNE0008_train_individual_segmentation_model.txt  --viz-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/viz/TumorNormal_indivuidual_model_TNE0008 --root-data-path  /gpfsscratch/rech/ohv/ueu39kt/TumoralNormalForFastFlow_difficult_cases


echo ===============================================================
echo                        INFER all
echo ===============================================================
mkdir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/viz/TumorNormal_Ki67_modelTNE1895_Infer
python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --action-type norm-test --dataset TumorNormal  --class-name Tumor --checkpoint /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/TumorSeg_HES_individual_training_TNE0008/Tumor/29/TumorNormal_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_Tumor_29_7.pt --list-file-test Inference_individual_model_Tiles_TNE0008.txt --viz-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/viz/TumorNormal_indivuidual_model_TNE0008_Infer --root-data-path  /gpfsscratch/rech/ohv/ueu39kt/Tiles_HE_all_samples_384_384_Vahadane_2