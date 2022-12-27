#!/bin/bash
#SBATCH --job-name=CFlowBT
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1     
#SBATCH --partition=gpu_p13
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:4 # 4
# nombre de taches MPI par noeud
#SBATCH --time=100:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=TrainCflowBarlowTwin_AC0_timortiles_set2_BS256_parallel_v2.out          # nom du fichier de sortie
#SBATCH --error=TrainCflowBarlowTwin_AC0_timortiles_set2_BS256_parallel_v2.error     
#SBATCH --account uli@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/uli/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0
srun python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --dataset TCAC  --root-data-path /gpfsscratch/rech/uli/ueu39kt/Tiles_HE_all_samples_384_384_Vahadane_2 --weights-dir /gpfsscratch/rech/uli/ueu39kt/CFLOW/weights/TrainCflowBarlowTwin_BS256_AC0_tumor_tiles_set2_parralel_v2 --list-file-train train_tiles_cfow_barlowtwins_set2_version2_tumor_tiles.txt  --meta-epochs 25 --sub-epochs 8 --parallel --lr 2e-4 --enc-arch wide_resnet50_barlowtwin_LNEN --class-name AC0 --batch-size 256
