#! usr/bin/bash

# CLB Sample 
declare -a StringArray=("TNE0008" "TNE0015" "TNE0017" "TNE0417" "TNE0859" 
                        "TNE0862" "TNE0973"  )
template_bash_script=individual_training_tumorseg_HES_TNE0004.sh
for val in ${StringArray[@]}; do
    cp $template_bash_script  individual_training_tumorseg_HES_$val.sh
    sed -i "s/TNE0004/$val/g" individual_training_tumorseg_HES_$val.sh

done;