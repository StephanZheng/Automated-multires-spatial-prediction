#!/bin/bash

# Run like this:
# sudo make clean; sudo make; sudo LD_LIBRARY_PATH="/home/ubuntu/robustmultitask_devstack/lib/boost_1_57_0/stage/lib" ./run_train_macro_AWSStage3.sh

kk=0

StageOne_StartingLearningRate="0.1"
StageThree_StartingLearningRate="0.0005"
LossWeightWeak="1.0"
StartFromStage="1"
#
NumberOfLatentDimensionsBallHandler="10"
#
StageThree_UseSparseMatS="0"
StageThree_Regularization_LFBallhandler="0.1"
StageThree_ApplyMomentumEveryNthMinibatch="50"
StageThree_ResetMomentumEveryNthEpoch="1"
StageThree_TrainOnLatentFactorsEveryBatch="20"
StageThree_UpdateSparseMatS_EveryNthMinibatch="1"
StageThree_AddRegularizationSparseMatS_EveryNthMinibatch="10"
#
StageThree_Regularization_SparseS_Level1="0.1"
#
StageThree_Regularization_SparseS_Level2="0.1"
StageThree_Initialization_SparseS_range="0.1"

SETTINGS_JSON="/home/stzheng/symlinks/robustmultitask_cpp/settingsCMSStage3_bb.json"
PYTHON_LOGGING_ID="1111111"

for NumberOfLatentDimensionsBallHandler_ in $NumberOfLatentDimensionsBallHandler;
do
  for StageThree_Regularization_SparseS_Level1_ in $StageThree_Regularization_SparseS_Level1;
  do
    echo "Bash | Episode $kk (0-indexed)"
    echo "--- [C] MS --- VERSION"
    ./train_robust_multitask \
    ${SETTINGS_JSON} \
    1 \
    $StageOne_StartingLearningRate \
    $StageThree_StartingLearningRate \
    $LossWeightWeak \
    $StartFromStage \
    $NumberOfLatentDimensionsBallHandler_ \
    $StageThree_UseSparseMatS \
    $StageThree_Regularization_LFBallhandler \
    $StageThree_ApplyMomentumEveryNthMinibatch \
    $StageThree_ResetMomentumEveryNthEpoch \
    $StageThree_TrainOnLatentFactorsEveryBatch \
    $StageThree_UpdateSparseMatS_EveryNthMinibatch \
    $StageThree_AddRegularizationSparseMatS_EveryNthMinibatch \
    $StageThree_Regularization_SparseS_Level1_ \
    $StageThree_Regularization_SparseS_Level2 \
    $StageThree_Initialization_SparseS_range \
    $PYTHON_LOGGING_ID

    echo ""
    echo "Emailing a notification of run completion"
    STR=$"CMS Machine"$'\n\n'"StartFromStage: ${StartFromStage}"$'\n\n'"StageThree_StartingLearningRate: ${StageThree_StartingLearningRate}"$'\n\n'"StageThree_AddRegularizationSparseMatS_EveryNthMinibatch: ${StageThree_AddRegularizationSparseMatS_EveryNthMinibatch}"$'\n\n'"StageThree_Regularization_SparseS_Level1: ${StageThree_Regularization_SparseS_Level1}"$'\n\n'"StageThree_Initialization_SparseS_range: ${StageThree_Initialization_SparseS_range}"
    echo $STR
    # echo "$STR" | mailx -s "[CMS] Stage 3 sparse + latent factor model :: run $kk is finished" st.t.zheng@gmail.com
    kk=$(($kk+1))
  done
done
# echo "$STR" | mailx -s "[CMS is done - has done $kk runs" st.t.zheng@gmail.com

