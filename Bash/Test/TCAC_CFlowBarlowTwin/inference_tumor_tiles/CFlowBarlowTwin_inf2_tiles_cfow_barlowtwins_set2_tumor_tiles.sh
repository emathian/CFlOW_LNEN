#!/bin/bash
#SBATCH --job-name=CFlowBT_inf2_tiles_cfow_barlowtwins_set2_tumor_tiles
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1
#SBATCH --partition=gpu_p13
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1 # 4
# nombre de taches MPI par noeud
#SBATCH --time=40:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=CFlowBT_inf2_tiles_cfow_barlowtwins_set2_tumor_tiles.out          # nom du fichier de sortie
#SBATCH --error=CFlowBT_inf2_tiles_cfow_barlowtwins_set2_tumor_tiles.error     
#SBATCH --account uli@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/uli/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0


srun  python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --action-type norm-test --dataset TCAC  --class-name AC0 --checkpoint /gpfsscratch/rech/uli/ueu39kt/CFLOW/weights/TrainCflowBarlowTwin_AC0_tumortiles_set2/AC0/0/TCAC_wide_resnet50_barlowtwin_LNEN_freia-cflow_pl3_cb8_inp384_run0_AC0_mataepoch_0_subepoch_1_loader_0_decoder.pt --list-file-test inf2_tiles_cfow_barlowtwins_set2_tumor_tiles.txt --viz-dir /gpfsscratch/rech/uli/ueu39kt/CFLOW/viz/CFlowBarlowTwin_tumor_inf2_tiles_cfow_barlowtwins_set2_tumor_tiles --root-data-path  /gpfsscratch/rech/uli/ueu39kt/Tiles_HE_all_samples_384_384_Vahadane_2 --parallel