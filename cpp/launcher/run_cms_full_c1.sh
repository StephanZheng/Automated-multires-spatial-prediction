#!/bin/bash
#
BASE="/home/stzheng/symlinks/robustmultitask_cpp/launcher/settings_bb"
FILEPATHS_JSON="$BASE/cms_filepaths.json"
DATASET_JSON="$BASE/dataset_bb.json"
MODEL_JSON="$BASE/full_c1.json"

echo "Bash | Episode $kk (0-indexed)"
echo "--- [C] MS --- VERSION"
./train_robust_multitask \
${FILEPATHS_JSON} \
${DATASET_JSON} \
${MODEL_JSON} \

echo ""
echo "Emailing a notification of run completion"
STR=$"CMS Machine"$'\n\n'"StartFromStage: ${StartFromStage}"$'\n\n'"StageThree_StartingLearningRate: ${StageThree_StartingLearningRate}""
echo $STR
# echo "$STR" | mailx -s "[CMS] Stage 3 sparse + latent factor model :: run $kk is finished" st.t.zheng@gmail.com

