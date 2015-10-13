#!/bin/bash

BASE="/home/stzheng/symlinks/robustmultitask_cpp/launcher/settings_bb"
FILEPATHS_JSON="$BASE/filepaths.json"
FEATURES_JSON="$BASE/features_bb_cms_c$1.json"
LABELS_JSON="$BASE/labels_bb.json"
TRAINING_PARAM_JSON="$BASE/training_param.json"
TRAINING_PARAM_3FACTOR_JSON="$BASE/training_param_3factor.json"

echo -e "--- [C] MS --- VERSION \n\n"\
"Using spatial cell of size $1 by $1\n\n"\
"Using files\n"\
"${FILEPATHS_JSON}\n"\
"${FEATURES_JSON}\n"\
"${LABELS_JSON}\n"\
"${TRAINING_PARAM_JSON}\n"\
"${TRAINING_PARAM_3FACTOR_JSON}\n"
"Starting executable\n"\

./train_robust_multitask \
${FILEPATHS_JSON} \
${FEATURES_JSON} \
${LABELS_JSON} \
${TRAINING_PARAM_JSON} \
${TRAINING_PARAM_3FACTOR_JSON}

echo ""
# echo "Emailing a notification of run completion"
# STR=$"CMS Machine"$'\n\n'"StartFromStage: ${StartFromStage}"$'\n\n'"StageThree_StartingLearningRate: ${StageThree_StartingLearningRate}"
# echo $STR

# echo "$STR" | mailx -s "[CMS] Stage 3 sparse + latent factor model :: run $kk is finished" st.t.zheng@gmail.com



| mailx -s "data" st.t.zheng@gmail.com