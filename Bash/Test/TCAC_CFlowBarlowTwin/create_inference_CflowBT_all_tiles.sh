#! usr/bin/bash

# CLB Sample 
declare -a StringArray=("inf_noAc0_tiles_cfow_barlowtwins_set0_all"
"inf_AC0Train_tiles_cfow_barlowtwins_set0_all"   "inf_noAc0_tiles_cfow_barlowtwins_set10_all"
"inf_AC0Train_tiles_cfow_barlowtwins_set11_all"  "inf_noAc0_tiles_cfow_barlowtwins_set1_all"
"inf_AC0Train_tiles_cfow_barlowtwins_set12_all"  "inf_noAc0_tiles_cfow_barlowtwins_set2_all"
"inf_AC0Train_tiles_cfow_barlowtwins_set1_all"   "inf_noAc0_tiles_cfow_barlowtwins_set3_all"
"inf_AC0Train_tiles_cfow_barlowtwins_set2_all"   "inf_noAc0_tiles_cfow_barlowtwins_set4_all"
"inf_AC0Train_tiles_cfow_barlowtwins_set3_all"   "inf_noAc0_tiles_cfow_barlowtwins_set5_all"
"inf_AC0Train_tiles_cfow_barlowtwins_set4_all"   "inf_noAc0_tiles_cfow_barlowtwins_set6_all"
"inf_AC0Train_tiles_cfow_barlowtwins_set5_all"   "inf_noAc0_tiles_cfow_barlowtwins_set7_all"
"inf_AC0Train_tiles_cfow_barlowtwins_set6_all"   "inf_noAc0_tiles_cfow_barlowtwins_set8_all"
"inf_AC0Train_tiles_cfow_barlowtwins_set7_all"   "inf_noAc0_tiles_cfow_barlowtwins_set9_all"
 )
template_bash_script=inference_all_tiles/CFlowBarlowTwin_inf2_tiles_cfow_barlowtwins_set2_all.sh
for val in ${StringArray[@]}; do
    cp $template_bash_script  inference_all_tiles/CFlowBarlowTwin_$val.sh
    sed -i "s/inf2_tiles_cfow_barlowtwins_set2_all/$val/g"  inference_all_tiles/CFlowBarlowTwin_$val.sh

done;