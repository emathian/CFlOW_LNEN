#!/bin/bash
#SBATCH --job-name=CFlowBT
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --partition=gpu_p13
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:4 # 4
# nombre de taches MPI par noeud
#SBATCH --time=15:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=TrainCflowBarlowTwin_AC0_all_bs512_lr16e3.out          # nom du fichier de sortie
#SBATCH --error=TrainCflowBarlowTwin_AC0_all_bs512_lr162e3.error     
#SBATCH --account uli@v100

module purge
export PYTHONUSERBASE=/gpfswork/rech/uli/ueu39kt/.local_base_timm
module load pytorch-gpu/py3/1.9.0
mkdir /gpfsscratch/rech/uli/ueu39kt/CFLOW/weights/TrainCflowBarlowTwin_AC0_all_bs512_lr16e3
srun python /linkhome/rech/genkmw01/ueu39kt/cflow-ad/main_cflow.py --gpu 0 -inp 384 --dataset TCAC  --root-data-path /gpfsscratch/rech/uli/ueu39kt/Tiles_HE_all_samples_384_384_Vahadane_2 --weights-dir /gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/TrainCflowBarlowTwin_AC0_all_bs512_lr16e3 --list-file-train train_tiles_cfow_barlowtwins_dev.txt --list-file-test  test_tiles_cfow_barlowtwins_dev.txt --meta-epochs 25 --sub-epochs 8 --parallel --lr 1.6e-3 --enc-arch wide_resnet50_barlowtwin_LNEN --class-name AC0 --batch-size 512
