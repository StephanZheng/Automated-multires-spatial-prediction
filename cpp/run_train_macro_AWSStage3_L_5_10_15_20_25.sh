#!/bin/bash

# Run like this:
# sudo make clean; sudo make; sudo LD_LIBRARY_PATH="/home/ubuntu/robustmultitask_devstack/lib/boost_1_57_0/stage/lib" ./run_train_macro_AWSStage3.sh

kk=0

StageOne_StartingLearningRate="0.1"
StageThree_StartingLearningRate="0.01"
LossWeightWeak="1.0"
StartFromStage="3"
#
NumberOfLatentDimensionsBallHandler="5 10 15 20 25"
#
StageThree_UseSparseMatS="0"
StageThree_Regularization_LFBallhandler="0.1"
StageThree_ApplyMomentumEveryNthMinibatch="5"
StageThree_ResetMomentumEveryNthEpoch="1"
StageThree_TrainOnLatentFactorsEveryBatch="1"
StageThree_UpdateSparseMatS_EveryNthMinibatch="1"
StageThree_AddRegularizationSparseMatS_EveryNthMinibatch="10"
#
StageThree_Regularization_SparseS_Level1="0.1"
#
StageThree_Regularization_SparseS_Level2="0.1"
StageThree_Initialization_SparseS_range="0.1"

instanceid=$(ec2metadata --instance-id)
publichostname=$(ec2metadata --public-hostname)

for NumberOfLatentDimensionsBallHandler_ in $NumberOfLatentDimensionsBallHandler;
do
  for StageThree_Regularization_SparseS_Level1_ in $StageThree_Regularization_SparseS_Level1;
  do
    echo "This machine: $(ec2metadata --public-hostname)"
    identifierfile=$instanceid"_Latentdim_"$NumberOfLatentDimensionsBallHandler"_RegularizationLF_"$StageThree_Regularization_LFFlyAction"_RegularizationS_"$StageThree_Regularization_SparseS_Level1"_LearningRate_"$StageThree_StartingLearningRate".dummy"
    touch /home/ubuntu/robustmultitask_devstack/projects/robust_multitask/basketball_prediction/snapshots/rank3/stage3/$identifierfile
    aws s3 mv /home/ubuntu/robustmultitask_devstack/projects/robust_multitask/basketball_prediction/snapshots/rank3/stage3/$identifierfile s3://robustmultitasksnapshots/bball_prediction/snapshots/ --dryrun
    aws s3 mv /home/ubuntu/robustmultitask_devstack/projects/robust_multitask/basketball_prediction/snapshots/rank3/stage3 s3://robustmultitasksnapshots/bball_prediction/snapshots/$instanceid --recursive --dryrun
    echo "Bash | Episode $kk (0-indexed)"
    echo "--- [A] MS --- VERSION"
    ./train_lfm_s3 settingsAWSStage3.json 1 $StageOne_StartingLearningRate $StageThree_StartingLearningRate $LossWeightWeak $StartFromStage $NumberOfLatentDimensionsBallHandler_ $StageThree_UseSparseMatS $StageThree_Regularization_LFBallhandler $StageThree_ApplyMomentumEveryNthMinibatch $StageThree_ResetMomentumEveryNthEpoch $StageThree_TrainOnLatentFactorsEveryBatch $StageThree_UpdateSparseMatS_EveryNthMinibatch $StageThree_AddRegularizationSparseMatS_EveryNthMinibatch $StageThree_Regularization_SparseS_Level1_ $StageThree_Regularization_SparseS_Level2 $StageThree_Initialization_SparseS_range
    echo ""
    echo "Bash | Done with a run! Moving all snapshots to S3 bucket"
    echo ""
    aws s3 mv /home/ubuntu/robustmultitask_devstack/projects/robust_multitask/basketball_prediction/snapshots/rank3/stage3/$identifierfile s3://robustmultitasksnapshots/bball_prediction/snapshots/
    aws s3 mv /home/ubuntu/robustmultitask_devstack/projects/robust_multitask/basketball_prediction/snapshots/rank3/stage3 s3://robustmultitasksnapshots/bball_prediction/snapshots/$instanceid --recursive
    echo ""
    echo "Emailing a notification of run completion"
    STR=$"Instance: ${instanceid}"$'\n\n'"Host: ${publichostname}"$'\n\n'"StartFromStage: ${StartFromStage}"$'\n\n'"StageThree_StartingLearningRate: ${StageThree_StartingLearningRate}"$'\n\n'"StageThree_AddRegularizationSparseMatS_EveryNthMinibatch: ${StageThree_AddRegularizationSparseMatS_EveryNthMinibatch}"$'\n\n'"StageThree_Regularization_SparseS_Level1: ${StageThree_Regularization_SparseS_Level1}"$'\n\n'"StageThree_Initialization_SparseS_range: ${StageThree_Initialization_SparseS_range}"
    echo $STR
    echo "$STR" | mailx -s "[AWS ${instanceid}] Stage 3 sparse + latent factor model :: run $kk is finished" st.t.zheng@gmail.com
    kk=$(($kk+1))
  done
done
echo "$STR" | mailx -s "[AWS ${instanceid}] This machine is done - has done $kk runs" st.t.zheng@gmail.com

