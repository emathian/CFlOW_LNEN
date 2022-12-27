#!/bin/bash
#SBATCH --job-name=CFlow_Infer_AC0_segmented_TNE100-1
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1
#SBATCH --partition=gpu_p2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1 # 4
# nombre de taches MPI par noeud
#SBATCH --time=40:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlow_Infer_AC0_segmented_TNE100-1.out          # nom du fichier de sortie
#SBATCH --error=CFlow_Infer_AC0_segmented_TNE100-1.error     
#SBATCH --account ohv@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/ohv/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0


srun  python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --action-type norm-test --dataset TCAC  --class-name AC0 --checkpoint /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/AC0_train_segmented/none/0/TCAC_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_none_mataepoch_0_subepoch_4_loader_7000_decoder.pt --list-file-test Inference_All_AC0Training_Tiles_TNE100-1.txt --viz-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/viz/CFlow_Infer_AC0_segmented_TNE100-1 --root-data-path  /gpfsscratch/rech/ohv/ueu39kt/Tiles_HE_all_samples_384_384_Vahadane_2  --parallel
