#!/bin/bash
#
BASE="/home/stzheng/symlinks/robustmultitask_cpp/launcher/settings_bb"
FEATURES_JSON="$BASE/features_bb_cms_c2.json"
LABELS_JSON="$BASE/labels_bb.json"
TRAINING_PARAM_JSON="$BASE/training_param.json"

echo "Bash | Episode $kk (0-indexed)"
echo "--- [C] MS --- VERSION"
./train_robust_multitask \
${FEATURES_JSON} \
${LABELS_JSON} \
${TRAINING_PARAM_JSON} \

echo ""
echo "Emailing a notification of run completion"
STR=$"CMS Machine" $'\n\n'"StartFromStage: ${StartFromStage}"$'\n\n'"StageThree_StartingLearningRate: ${StageThree_StartingLearningRate}"
echo $STR
# echo "$STR" | mailx -s "[CMS] Stage 3 sparse + latent factor model :: run $kk is finished" st.t.zheng@gmail.com

