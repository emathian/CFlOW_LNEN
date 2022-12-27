#! usr/bin/bash
# CLB Sample 
declare -a StringArray=(    "inf_noAc0_tiles_cfow_barlowtwins_set0_tumor"
"inf_AC0Train_tiles_cfow_barlowtwins_set0_tumor"  "inf_noAc0_tiles_cfow_barlowtwins_set1_tumor"
"inf_AC0Train_tiles_cfow_barlowtwins_set1_tumor"  "inf_noAc0_tiles_cfow_barlowtwins_set2_tumor"
"inf_AC0Train_tiles_cfow_barlowtwins_set2_tumor"  "inf_noAc0_tiles_cfow_barlowtwins_set3_tumor"
"inf_AC0Train_tiles_cfow_barlowtwins_set3_tumor"  "inf_noAc0_tiles_cfow_barlowtwins_set4_tumor"
"inf_AC0Train_tiles_cfow_barlowtwins_set4_tumor"  "inf_noAc0_tiles_cfow_barlowtwins_set5_tumor"
"inf_AC0Train_tiles_cfow_barlowtwins_set7_tumor"  "inf_noAc0_tiles_cfow_barlowtwins_set6_tumor"
)
template_bash_script=inference_tumor_tiles/CFlowBarlowTwin_inf2_tiles_cfow_barlowtwins_set2_tumor_tiles.sh
for val in ${StringArray[@]}; do
    cp $template_bash_script  inference_tumor_tiles/CFlowBarlowTwin_$val.sh
    sed -i "s/CFlowBT_inf2_tiles_cfow_barlowtwins_set2_tumor_tiles/$val/g" inference_tumor_tiles/CFlowBarlowTwin_$val.sh # .out .error jobname
    sed -i "s/inf2_tiles_cfow_barlowtwins_set2_tumor_tiles/$val/g" inference_tumor_tiles/CFlowBarlowTwin_$val.sh # inference files list
    sed -i "s/CFlowBarlowTwin_tumor_inf2_tiles_cfow_barlowtwins_set2_tumor_tiles/$val/g"  inference_tumor_tiles/CFlowBarlowTwin_$val.sh # viz
done;