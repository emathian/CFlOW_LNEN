#! usr/bin/bash

# CLB Sample 
declare -a StringArray=("TNE002-5" "TNE006-9" "TNE01-2" "TNE03-4" "TNE050-5" "TNE056-9" 
                        "TNE06-7" "TNE080-6"  "TNE087" "TNE088" "TNE089" "TNE090-7"
                        "TNE098-9" "TNE100-1" "TNE102-7"  "TNE108-9" "TNE11-3" "TNE140-1"
                        "TNE142" "TNE143" "TNE144-9" "TNE15-9"  "TNE2" )
template_bash_script=All_high_lr/CFlow_Infer_AC0_all_TNE000-1.sh
for val in ${StringArray[@]}; do
    cp $template_bash_script  All_high_lr/CFlow_Infer_AC0_all_$val.sh
    sed -i "s/TNE000-1/$val/g"  All_high_lr/CFlow_Infer_AC0_all_$val.sh

done;


# CLB LCNEC Sample 
declare -a StringArray=("TNE[1-2]" )
template_bash_script=All_high_lr/CFlow_Infer_AC0_all_TNE000-1.sh
for val in ${StringArray[@]}; do
    cp $template_bash_script  All_high_lr/CFlow_Infer_AC0_all_LCNEC_CLB_$val.sh
    sed -i "s/Inference_All_AC0Training_Tiles_TNE000-1/Inference_All_AC0Training_Tiles_LCNEC_CLB_$val/g"  All_high_lr/CFlow_Infer_AC0_all_LCNEC_CLB_$val.sh
    sed -i "s/TNE000-1/$val/g"  All_high_lr/CFlow_Infer_AC0_all_LCNEC_CLB_$val.sh
    sed -i "s/Tiles_HE_all_samples_384_384_Vahadane_2/Tiles_HE_LCNEC_384_384_vahadane/g"  All_high_lr/CFlow_Infer_AC0_all_LCNEC_CLB_$val.sh
done;



# Italian LCNEC Sample 
declare -a StringArray=("IT1" "IT20-5"  "IT26-9" "IT30-4" "IT35-9" "IT4-9"  
                            "ITVAL1-2"  "ITVAL3" "ITVAL4-5" "ITVAL6-9"  )
template_bash_script=All_high_lr/CFlow_Infer_AC0_all_TNE000-1.sh
for val in ${StringArray[@]}; do
    cp $template_bash_script  All_high_lr/CFlow_Infer_AC0_all_LCNEC_IT_$val.sh
    sed -i "s/Inference_All_AC0Training_Tiles_TNE000-1/Inference_All_AC0Training_Tiles_LCNEC_IT_$val/g"  All_high_lr/CFlow_Infer_AC0_all_LCNEC_IT_$val.sh
    sed -i "s/TNE000-1/$val/g"  All_high_lr/CFlow_Infer_AC0_all_LCNEC_IT_$val.sh
    sed -i "s/Tiles_HE_all_samples_384_384_Vahadane_2/Tiles_HE_massimo_LCNEC_384_384_Vahadane/g"  All_high_lr/CFlow_Infer_AC0_all_LCNEC_IT_$val.sh
done;