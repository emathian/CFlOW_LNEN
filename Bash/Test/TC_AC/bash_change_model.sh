#! usr/bin/bash

find Segmented_low_lr -type f -name '*.sh' -exec sed -i "s+/gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/AC0_train_segmented_low_lr/none/0/TCAC_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_none_mataepoch_0_subepoch_3_loader_9000_decoder_2.pt+/gpfsscratch/rech/ohv/ueu39kt/CFLOW/weights/AC0_train_segmented_low_lr/none/0/TCAC_wide_resnet50_2_freia-cflow_pl3_cb8_inp384_run0_none_mataepoch_0_subepoch_3_loader_9000_decoder.pt+g"  {} \;

# find All_low_lr -type f -name '*.sh' -exec sed -i "s/AC0_all_/AC0_all_low_lr_/g"  {} \;


# find All_low_lr -type f -name '*.sh' -exec sed -i "s/gpu_p2/gpu_p13/g"  {} \;
