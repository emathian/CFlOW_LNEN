#! usr/bin/bash
declare -a StringArray=("2" "3" "4-9" "VAL1" "VAL2" "VAL3" "VAL4" "VAL5-9" )
template_bash_script=cflow_infer_LCNEC_Massimo_1.sh
for val in ${StringArray[@]}; do
    cp $template_bash_script  cflow_infer_LCNEC_Massimo_$val.sh
    sed -i "s/Massimo_1/Massimo_$val/g"  cflow_infer_LCNEC_Massimo_$val.sh
    sed -i "s/Tiles_1/Tiles_$val/g" cflow_infer_LCNEC_Massimo_$val.sh

done;