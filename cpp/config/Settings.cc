#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>

#include "config/Settings.h"
#include "util/PrettyOutput.h"

using namespace std;
using namespace rapidjson;

void Settings::loadSettingsJSON(const char * fn_settings_file) {
  char    readBuffer[65536];
  FILE*     fp = fopen(fn_settings_file, "r");
  FileReadStream  frs(fp, readBuffer, sizeof(readBuffer));
  Document  settings_json;
  settings_json.ParseStream(frs);
  fclose(fp);

  static const char* kTypeNames[] = { "Null", "False", "True", "Object", "Array", "String", "Number" };
  for (Value::ConstMemberIterator itr = settings_json.MemberBegin(); itr != settings_json.MemberEnd(); ++itr) {
    string ss = itr->name.GetString();

    if (ss.compare("CapScores")                                      == 0) CapScores = itr->value.GetInt();
    if (ss.compare("DATASET_NAME")                                   == 0) DATASET_NAME = itr->value.GetString();
    if (ss.compare("DesiredSparsityLevel")                           == 0) DesiredSparsityLevel = itr->value.GetDouble();
    if (ss.compare("Dimension_A")                                    == 0) Dimension_A = itr->value.GetInt();
    if (ss.compare("Dimension_B_BiasSlice")                          == 0) Dimension_B_BiasSlice = itr->value.GetInt();
    if (ss.compare("Dimension_C_BiasSlice")                          == 0) Dimension_C_BiasSlice = itr->value.GetInt();
    if (ss.compare("DummyMultiplier")                                == 0) DummyMultiplier = itr->value.GetInt();
    if (ss.compare("EnableDebugPrinter_Level1")                      == 0) EnableDebugPrinter_Level1 = itr->value.GetInt();
    if (ss.compare("EnableDebugPrinter_Level2")                      == 0) EnableDebugPrinter_Level2 = itr->value.GetInt();
    if (ss.compare("EnableDebugPrinter_Level3")                      == 0) EnableDebugPrinter_Level3 = itr->value.GetInt();
    if (ss.compare("FeaturesSubFolder")                              == 0) FeaturesSubFolder = itr->value.GetString();
    if (ss.compare("FeatureThreshold")                               == 0) FeatureThreshold = itr->value.GetDouble();
    if (ss.compare("FILENAME_FEATURES_B")                            == 0) FILENAME_FEATURES_B = itr->value.GetString();
    if (ss.compare("FILENAME_FEATURES_B_window_3by3")                == 0) FILENAME_FEATURES_B_window_3by3 = itr->value.GetString();
    if (ss.compare("FILENAME_FEATURES_C")                            == 0) FILENAME_FEATURES_C = itr->value.GetString();
    if (ss.compare("FILENAME_FEATURES_C_window_3by3")                == 0) FILENAME_FEATURES_C_window_3by3 = itr->value.GetString();
    if (ss.compare("FILENAME_GTRUTH_TEST_STRONG")                    == 0) FILENAME_GTRUTH_TEST_STRONG = itr->value.GetString();
    if (ss.compare("FILENAME_GTRUTH_TEST_WEAK")                      == 0) FILENAME_GTRUTH_TEST_WEAK = itr->value.GetString();
    if (ss.compare("FILENAME_GTRUTH_TRAINVAL_STRONG")                == 0) FILENAME_GTRUTH_TRAINVAL_STRONG = itr->value.GetString();
    if (ss.compare("FILENAME_GTRUTH_TRAINVAL_WEAK")                  == 0) FILENAME_GTRUTH_TRAINVAL_WEAK = itr->value.GetString();
    if (ss.compare("GradientRandomizeAtZero")                        == 0) GradientRandomizeAtZero = itr->value.GetInt();
    if (ss.compare("GroundTruthScoreStrong")                         == 0) GroundTruthScoreStrong = itr->value.GetDouble();
    if (ss.compare("GroundTruthScoreWeak")                           == 0) GroundTruthScoreWeak = itr->value.GetDouble();
    if (ss.compare("GroundTruthSubFolder")                           == 0) GroundTruthSubFolder = itr->value.GetString();

    if (ss.compare("LogFolder")                           == 0) LogFolder = itr->value.GetString();
    if (ss.compare("LogFile_Loss")                           == 0) LogFile_Loss = itr->value.GetString();
    if (ss.compare("LogFile_CellEntropy")                           == 0) LogFile_CellEntropy = itr->value.GetString();


    if (ss.compare("LossWeightStrong")                               == 0) LossWeightStrong = itr->value.GetDouble();
    if (ss.compare("LossWeightWeak")                                 == 0) LossWeightWeak = itr->value.GetDouble();
    if (ss.compare("NumberOfColumnsScoreFunction")                   == 0) NumberOfColumnsScoreFunction = itr->value.GetInt();
    if (ss.compare("NumberOfFrames")                                 == 0) NumberOfFrames = itr->value.GetInt();
    if (ss.compare("NumberOfLatentDimensions")                       == 0) NumberOfLatentDimensions = itr->value.GetInt();
    if (ss.compare("NumberOfStrongLabels_Test")                      == 0) NumberOfStrongLabels_Test = itr->value.GetInt();
    if (ss.compare("NumberOfStrongLabels_Train")                     == 0) NumberOfStrongLabels_Train = itr->value.GetInt();
    if (ss.compare("NumberOfStrongLabels_TrainVal")                  == 0) NumberOfStrongLabels_TrainVal = itr->value.GetInt();
    if (ss.compare("NumberOfStrongLabels_Val")                       == 0) NumberOfStrongLabels_Val = itr->value.GetInt();
    if (ss.compare("NumberOfWeakLabels_Test")                        == 0) NumberOfWeakLabels_Test = itr->value.GetInt();
    if (ss.compare("NumberOfWeakLabels_Train")                       == 0) NumberOfWeakLabels_Train = itr->value.GetInt();
    if (ss.compare("NumberOfWeakLabels_TrainVal")                    == 0) NumberOfWeakLabels_TrainVal = itr->value.GetInt();
    if (ss.compare("NumberOfWeakLabels_Val")                         == 0) NumberOfWeakLabels_Val = itr->value.GetInt();
    if (ss.compare("PYTHON_LOGGING_ID")                              == 0) PYTHON_LOGGING_ID = itr->value.GetString();
    if (ss.compare("ResumeTraining")                                 == 0) ResumeTraining = itr->value.GetInt();
    if (ss.compare("RootFolder")                                     == 0) RootFolder = itr->value.GetString();
    if (ss.compare("ScoreFunctionColumnIndexMod")                    == 0) ScoreFunctionColumnIndexMod = itr->value.GetInt();
    if (ss.compare("SeedSubFolder")                                  == 0) SeedSubFolder = itr->value.GetString();
    if (ss.compare("SnapshotSubFolder")                              == 0) SnapshotSubFolder = itr->value.GetString();
    if (ss.compare("SPARSITY_THRESHOLD")                             == 0) SPARSITY_THRESHOLD = itr->value.GetDouble();
    if (ss.compare("SparsityRunningAverageWindowSize")               == 0) SparsityRunningAverageWindowSize = itr->value.GetInt();
    if (ss.compare("SpatialEntropy_BinWidth")                        == 0) SpatialEntropy_BinWidth = itr->value.GetDouble();
    if (ss.compare("SpatialEntropy_MinValue")                        == 0) SpatialEntropy_MinValue = itr->value.GetDouble();
    if (ss.compare("SpatialEntropy_MaxValue")                        == 0) SpatialEntropy_MaxValue = itr->value.GetDouble();

    if (ss.compare("SpatialEntropy_NumberOfBins")                    == 0) SpatialEntropy_NumberOfBins = itr->value.GetInt();
    if (ss.compare("SpatialRegularization")                          == 0) SpatialRegularization = itr->value.GetInt();
    if (ss.compare("StageOne_ApplyMomentumEveryNthMinibatch")        == 0) StageOne_ApplyMomentumEveryNthMinibatch = itr->value.GetInt();
    if (ss.compare("StageOne_Clamp_AdaptiveLearningRate")            == 0) StageOne_Clamp_AdaptiveLearningRate = itr->value.GetDouble();
    if (ss.compare("StageOne_ComputeLossAfterEveryNthEpoch")         == 0) StageOne_ComputeLossAfterEveryNthEpoch = itr->value.GetInt();
    if (ss.compare("StageOne_CurrentLearningRate")                   == 0) StageOne_CurrentLearningRate = itr->value.GetDouble();
    if (ss.compare("StageOne_Dimension_B")                           == 0) StageOne_Dimension_B = itr->value.GetInt();
    if (ss.compare("StageOne_Dimension_C")                           == 0) StageOne_Dimension_C = itr->value.GetInt();
    if (ss.compare("StageOne_Initialization_Bias_A_range")           == 0) StageOne_Initialization_Bias_A_range = itr->value.GetDouble();
    if (ss.compare("StageOne_Initialization_Weight_U_range")         == 0) StageOne_Initialization_Weight_U_range = itr->value.GetDouble();
    if (ss.compare("StageOne_MiniBatchSize")                         == 0) StageOne_MiniBatchSize = itr->value.GetInt();
    if (ss.compare("StageOne_NumberOfCrossValidationRuns")           == 0) StageOne_NumberOfCrossValidationRuns = itr->value.GetInt();
    if (ss.compare("StageOne_NumberOfEpochs")                        == 0) StageOne_NumberOfEpochs = itr->value.GetInt();
    if (ss.compare("StageOne_NumberOfNonZeroEntries_B")              == 0) StageOne_NumberOfNonZeroEntries_B = itr->value.GetInt();
    if (ss.compare("StageOne_NumberOfNonZeroEntries_C")              == 0) StageOne_NumberOfNonZeroEntries_C = itr->value.GetInt();
    if (ss.compare("StageOne_NumberOfThreads")                       == 0) StageOne_NumberOfThreads = itr->value.GetInt();
    if (ss.compare("StageOne_Regularization_Bias_A")                 == 0) StageOne_Regularization_Bias_A = itr->value.GetDouble();
    if (ss.compare("StageOne_Regularization_Weight_U_Level1")        == 0) StageOne_Regularization_Weight_U_Level1 = itr->value.GetDouble();
    if (ss.compare("StageOne_Regularization_Weight_U_Level2")        == 0) StageOne_Regularization_Weight_U_Level2 = itr->value.GetDouble();
    if (ss.compare("StageOne_Regularization_Weight_U_Level3")        == 0) StageOne_Regularization_Weight_U_Level3 = itr->value.GetDouble();
    if (ss.compare("StageOne_ResetMomentumEveryNthEpoch")            == 0) StageOne_ResetMomentumEveryNthEpoch = itr->value.GetInt();
    if (ss.compare("StageOne_SnapshotMomentum")                      == 0) StageOne_SnapshotMomentum = itr->value.GetString();
    if (ss.compare("StageOne_SnapshotWeightsBiases")                 == 0) StageOne_SnapshotWeightsBiases = itr->value.GetString();
    if (ss.compare("StageOne_StartFromCrossValRun")                  == 0) StageOne_StartFromCrossValRun = itr->value.GetInt();
    if (ss.compare("StageOne_StartFromEpoch")                        == 0) StageOne_StartFromEpoch = itr->value.GetInt();
    if (ss.compare("StageOne_StartingLearningRate")                  == 0) StageOne_StartingLearningRate = itr->value.GetDouble();
    if (ss.compare("StageOne_StatusEveryNBatchesTrain")              == 0) StageOne_StatusEveryNBatchesTrain = itr->value.GetInt();
    if (ss.compare("StageOne_StatusEveryNBatchesValid")              == 0) StageOne_StatusEveryNBatchesValid = itr->value.GetInt();
    if (ss.compare("StageOne_Take_SnapshotAfterEveryNthEpoch")       == 0) StageOne_Take_SnapshotAfterEveryNthEpoch = itr->value.GetInt();
    if (ss.compare("StageOne_TestConditionalScoreThreshold")         == 0) StageOne_TestConditionalScoreThreshold = itr->value.GetDouble();
    if (ss.compare("StageOne_TrainingConditionalScoreThreshold")     == 0) StageOne_TrainingConditionalScoreThreshold = itr->value.GetDouble();
    if (ss.compare("StageOne_UseBias")                               == 0) StageOne_UseBias = itr->value.GetInt();
    if (ss.compare("StageOne_UseSpatialRegularization")              == 0) StageOne_UseSpatialRegularization = itr->value.GetInt();
    if (ss.compare("StageOne_ValidationConditionalScoreThreshold")   == 0) StageOne_ValidationConditionalScoreThreshold = itr->value.GetDouble();
    if (ss.compare("StageOne_ValidationLossDecreaseThreshold")       == 0) StageOne_ValidationLossDecreaseThreshold = itr->value.GetDouble();
    if (ss.compare("StageOne_ValidationLossIncreaseThreshold")       == 0) StageOne_ValidationLossIncreaseThreshold = itr->value.GetDouble();
    if (ss.compare("StageOne_Weight_U_ClampToThreshold")             == 0) StageOne_Weight_U_ClampToThreshold = itr->value.GetInt();
    if (ss.compare("StageOne_Weight_U_RegularizationType")           == 0) StageOne_Weight_U_RegularizationType  = itr->value.GetInt();
    if (ss.compare("StageOne_Weight_U_Threshold")                    == 0) StageOne_Weight_U_Threshold = itr->value.GetDouble();
    if (ss.compare("StageOne_Weight_U_UseSoftThreshold")             == 0) StageOne_Weight_U_UseSoftThreshold = itr->value.GetInt();
    if (ss.compare("StageThree_A_RegularizationStrength_1_Dense")    == 0) StageThree_A_RegularizationStrength_1_Dense = itr->value.GetDouble();
    if (ss.compare("StageThree_A_RegularizationStrength_1_Sparse")   == 0) StageThree_A_RegularizationStrength_1_Sparse = itr->value.GetDouble();
    if (ss.compare("StageThree_A_RegularizationStrength_2_Dense")    == 0) StageThree_A_RegularizationStrength_2_Dense = itr->value.GetDouble();
    if (ss.compare("StageThree_A_RegularizationStrength_2_Sparse")   == 0) StageThree_A_RegularizationStrength_2_Sparse = itr->value.GetDouble();
    if (ss.compare("StageThree_A_SubDimension_2_InitRange")          == 0) StageThree_A_SubDimension_2_InitRange = itr->value.GetDouble();
    if (ss.compare("StageThree_ApplyMomentumEveryNthMinibatch")      == 0) StageThree_ApplyMomentumEveryNthMinibatch = itr->value.GetInt();
    if (ss.compare("StageThree_B_RegularizationStrength_1_Dense")    == 0) StageThree_B_RegularizationStrength_1_Dense = itr->value.GetDouble();
    if (ss.compare("StageThree_B_RegularizationStrength_1_Sparse")   == 0) StageThree_B_RegularizationStrength_1_Sparse = itr->value.GetDouble();
    if (ss.compare("StageThree_B_RegularizationStrength_2_Dense")    == 0) StageThree_B_RegularizationStrength_2_Dense = itr->value.GetDouble();
    if (ss.compare("StageThree_B_RegularizationStrength_2_Sparse")   == 0) StageThree_B_RegularizationStrength_2_Sparse = itr->value.GetDouble();
    if (ss.compare("StageThree_B_SpatReg_Multiplier_Level1")         == 0) StageThree_B_SpatReg_Multiplier_Level1 = itr->value.GetDouble();
    if (ss.compare("StageThree_B_SpatReg_Multiplier_Level2")         == 0) StageThree_B_SpatReg_Multiplier_Level2 = itr->value.GetDouble();
    if (ss.compare("StageThree_B_SubDimension_2_InitRange")          == 0) StageThree_B_SubDimension_2_InitRange = itr->value.GetDouble();
    if (ss.compare("StageThree_C_RegularizationStrength_1_Dense")    == 0) StageThree_C_RegularizationStrength_1_Dense = itr->value.GetDouble();
    if (ss.compare("StageThree_C_RegularizationStrength_1_Sparse")   == 0) StageThree_C_RegularizationStrength_1_Sparse = itr->value.GetDouble();
    if (ss.compare("StageThree_C_RegularizationStrength_2_Dense")    == 0) StageThree_C_RegularizationStrength_2_Dense = itr->value.GetDouble();
    if (ss.compare("StageThree_C_RegularizationStrength_2_Sparse")   == 0) StageThree_C_RegularizationStrength_2_Sparse = itr->value.GetDouble();
    if (ss.compare("StageThree_C_SpatReg_Multiplier_Level1")         == 0) StageThree_C_SpatReg_Multiplier_Level1 = itr->value.GetDouble();
    if (ss.compare("StageThree_C_SpatReg_Multiplier_Level2")         == 0) StageThree_C_SpatReg_Multiplier_Level2 = itr->value.GetDouble();
    if (ss.compare("StageThree_C_SubDimension_2_InitRange")          == 0) StageThree_C_SubDimension_2_InitRange = itr->value.GetDouble();
    if (ss.compare("StageThree_Clamp_AdaptiveLearningRate")          == 0) StageThree_Clamp_AdaptiveLearningRate = itr->value.GetDouble();
    if (ss.compare("StageThree_ComputeLossAfterEveryNthEpoch")       == 0) StageThree_ComputeLossAfterEveryNthEpoch = itr->value.GetInt();
    if (ss.compare("StageThree_CurrentLearningRate")                 == 0) StageThree_CurrentLearningRate = itr->value.GetDouble();
    if (ss.compare("StageThree_Dimension_B")                         == 0) StageThree_Dimension_B = itr->value.GetInt();
    if (ss.compare("StageThree_Dimension_C")                         == 0) StageThree_Dimension_C = itr->value.GetInt();
    if (ss.compare("StageThree_Initialization_Bias_A_range")         == 0) StageThree_Initialization_Bias_A_range = itr->value.GetDouble();
    if (ss.compare("StageThree_Initialization_LF_A_range")           == 0) StageThree_Initialization_LF_A_range = itr->value.GetDouble();
    if (ss.compare("StageThree_Initialization_SLF_B_range")          == 0) StageThree_Initialization_SLF_B_range = itr->value.GetDouble();
    if (ss.compare("StageThree_Initialization_Sparse_S_range")       == 0) StageThree_Initialization_Sparse_S_range = itr->value.GetDouble();
    if (ss.compare("StageThree_Initialization_Weight_W_range")       == 0) StageThree_Initialization_Weight_W_range = itr->value.GetDouble();
    if (ss.compare("StageThree_LF_A_RegularizationType")             == 0) StageThree_LF_A_RegularizationType  = itr->value.GetInt();
    if (ss.compare("StageThree_LF_A_seed")                           == 0) StageThree_LF_A_seed = itr->value.GetString();
    if (ss.compare("StageThree_LF_B_RegularizationType")             == 0) StageThree_LF_B_RegularizationType  = itr->value.GetInt();
    if (ss.compare("StageThree_LF_B_seed")                           == 0) StageThree_LF_B_seed = itr->value.GetString();
    if (ss.compare("StageThree_LF_C_RegularizationType")             == 0) StageThree_LF_C_RegularizationType  = itr->value.GetInt();
    if (ss.compare("StageThree_LF_C_seed")                           == 0) StageThree_LF_C_seed = itr->value.GetString();
    if (ss.compare("StageThree_LF_SA_seed")                          == 0) StageThree_LF_SA_seed = itr->value.GetString();
    if (ss.compare("StageThree_LF_SB_seed")                          == 0) StageThree_LF_SB_seed = itr->value.GetString();
    if (ss.compare("StageThree_LF_SC_seed")                          == 0) StageThree_LF_SC_seed = itr->value.GetString();
    if (ss.compare("StageThree_MiniBatchSize")                       == 0) StageThree_MiniBatchSize = itr->value.GetInt();
    if (ss.compare("StageThree_NumberOfCrossValidationRuns")         == 0) StageThree_NumberOfCrossValidationRuns = itr->value.GetInt();
    if (ss.compare("StageThree_NumberOfEpochs")                      == 0) StageThree_NumberOfEpochs = itr->value.GetInt();
    if (ss.compare("StageThree_NumberOfNonZeroEntries_B")            == 0) StageThree_NumberOfNonZeroEntries_B = itr->value.GetInt();
    if (ss.compare("StageThree_NumberOfNonZeroEntries_C")            == 0) StageThree_NumberOfNonZeroEntries_C = itr->value.GetInt();
    if (ss.compare("StageThree_NumberOfThreads")                     == 0) StageThree_NumberOfThreads = itr->value.GetInt();
    if (ss.compare("StageThree_Regularization_Bias_A")               == 0) StageThree_Regularization_Bias_A = itr->value.GetDouble();
    if (ss.compare("StageThree_Regularization_LF_A")                 == 0) StageThree_Regularization_LF_A = itr->value.GetDouble();
    if (ss.compare("StageThree_Regularization_RepulsionTerm")        == 0) StageThree_Regularization_RepulsionTerm = itr->value.GetDouble();
    if (ss.compare("StageThree_Regularization_S_BC")                 == 0) StageThree_Regularization_S_BC = itr->value.GetDouble();
    if (ss.compare("StageThree_Regularization_SLF_B_Level1")         == 0) StageThree_Regularization_SLF_B_Level1 = itr->value.GetDouble();
    if (ss.compare("StageThree_Regularization_SLF_B_Level2")         == 0) StageThree_Regularization_SLF_B_Level2 = itr->value.GetDouble();
    if (ss.compare("StageThree_Regularization_Sparse_S_Level1")      == 0) StageThree_Regularization_Sparse_S_Level1 = itr->value.GetDouble();
    if (ss.compare("StageThree_Regularization_Sparse_S_Level2")      == 0) StageThree_Regularization_Sparse_S_Level2 = itr->value.GetDouble();
    if (ss.compare("StageThree_Regularization_Sparse_S_Level3")      == 0) StageThree_Regularization_Sparse_S_Level3 = itr->value.GetDouble();
    if (ss.compare("StageThree_RegularizationType_1")                == 0) StageThree_RegularizationType_1 = itr->value.GetInt();
    if (ss.compare("StageThree_RegularizationType_2")                == 0) StageThree_RegularizationType_2 = itr->value.GetInt();
    if (ss.compare("StageThree_RegularizeS_EveryBatch")              == 0) StageThree_RegularizeS_EveryBatch = itr->value.GetInt();
    if (ss.compare("StageThree_ResetMomentumEveryNthEpoch")          == 0) StageThree_ResetMomentumEveryNthEpoch = itr->value.GetInt();
    if (ss.compare("StageThree_S_SpatReg_Multiplier_Level1")         == 0) StageThree_S_SpatReg_Multiplier_Level1 = itr->value.GetDouble();
    if (ss.compare("StageThree_S_SpatReg_Multiplier_Level2")         == 0) StageThree_S_SpatReg_Multiplier_Level2 = itr->value.GetDouble();
    if (ss.compare("StageThree_SLF_B_RegularizationType")            == 0) StageThree_SLF_B_RegularizationType  = itr->value.GetInt();
    if (ss.compare("StageThree_SnapshotMomentum")                    == 0) StageThree_SnapshotMomentum = itr->value.GetString();
    if (ss.compare("StageThree_SnapshotWeightsBiases")               == 0) StageThree_SnapshotWeightsBiases = itr->value.GetString();
    if (ss.compare("StageThree_Sparse_S_ClampToThreshold")           == 0) StageThree_Sparse_S_ClampToThreshold = itr->value.GetInt();
    if (ss.compare("StageThree_Sparse_S_RegularizationType")         == 0) StageThree_Sparse_S_RegularizationType  = itr->value.GetInt();
    if (ss.compare("StageThree_Sparse_S_seed")                       == 0) StageThree_Sparse_S_seed = itr->value.GetString();
    if (ss.compare("StageThree_Sparse_S_UseSoftThreshold")           == 0) StageThree_Sparse_S_UseSoftThreshold = itr->value.GetInt();
    if (ss.compare("StageThree_SparseMatSThreshold")                 == 0) StageThree_SparseMatSThreshold = itr->value.GetDouble();
    if (ss.compare("StageThree_StartFromCrossValRun")                == 0) StageThree_StartFromCrossValRun = itr->value.GetInt();
    if (ss.compare("StageThree_StartFromEpoch")                      == 0) StageThree_StartFromEpoch = itr->value.GetInt();
    if (ss.compare("StageThree_StartFromSeed")                       == 0) StageThree_StartFromSeed = itr->value.GetInt();
    if (ss.compare("StageThree_StartingLearningRate")                == 0) StageThree_StartingLearningRate = itr->value.GetDouble();
    if (ss.compare("StageThree_StatusEveryNBatchesTrain")            == 0) StageThree_StatusEveryNBatchesTrain = itr->value.GetInt();
    if (ss.compare("StageThree_StatusEveryNBatchesValid")            == 0) StageThree_StatusEveryNBatchesValid = itr->value.GetInt();
    if (ss.compare("StageThree_SubDimension_1")                      == 0) StageThree_SubDimension_1 = itr->value.GetInt();
    if (ss.compare("StageThree_SubDimension_2")                      == 0) StageThree_SubDimension_2 = itr->value.GetInt();
    if (ss.compare("StageThree_Take_SnapshotAfterEveryNthEpoch")     == 0) StageThree_Take_SnapshotAfterEveryNthEpoch = itr->value.GetInt();
    if (ss.compare("StageThree_TestConditionalScoreThreshold")       == 0) StageThree_TestConditionalScoreThreshold = itr->value.GetDouble();
    if (ss.compare("StageThree_Train_LF_A_EveryBatch")               == 0) StageThree_Train_LF_A_EveryBatch = itr->value.GetInt();
    if (ss.compare("StageThree_Train_LF_B_EveryBatch")               == 0) StageThree_Train_LF_B_EveryBatch = itr->value.GetInt();
    if (ss.compare("StageThree_Train_LF_C_EveryBatch")               == 0) StageThree_Train_LF_C_EveryBatch = itr->value.GetInt();
    if (ss.compare("StageThree_Train_SLF_B_EveryBatch")              == 0) StageThree_Train_SLF_B_EveryBatch = itr->value.GetInt();
    if (ss.compare("StageThree_TrainBiasEveryBatch")                 == 0) StageThree_TrainBiasEveryBatch = itr->value.GetInt();
    if (ss.compare("StageThree_TrainingConditionalScoreThreshold")   == 0) StageThree_TrainingConditionalScoreThreshold = itr->value.GetDouble();
    if (ss.compare("StageThree_TrainOnLatentFactorsEveryBatch")      == 0) StageThree_TrainOnLatentFactorsEveryBatch = itr->value.GetInt();
    if (ss.compare("StageThree_TrainSubDimension_1")                 == 0) StageThree_TrainSubDimension_1 = itr->value.GetInt();
    if (ss.compare("StageThree_TrainSubDimension_2")                 == 0) StageThree_TrainSubDimension_2 = itr->value.GetInt();
    if (ss.compare("StageThree_TuneSparsity")                        == 0) StageThree_TuneSparsity = itr->value.GetInt();
    if (ss.compare("StageThree_UpdateSparseMatS_EveryNthMinibatch")  == 0) StageThree_UpdateSparseMatS_EveryNthMinibatch = itr->value.GetInt();
    if (ss.compare("StageThree_UseAdaptiveLearningRate")             == 0) StageThree_UseAdaptiveLearningRate = itr->value.GetInt();
    if (ss.compare("StageThree_UseBias")                             == 0) StageThree_UseBias = itr->value.GetInt();
    if (ss.compare("StageThree_UseLatentTerm")                       == 0) StageThree_UseLatentTerm = itr->value.GetInt();
    if (ss.compare("StageThree_UseRepulsion_S_ABC")                  == 0) StageThree_UseRepulsion_S_ABC = itr->value.GetInt();
    if (ss.compare("StageThree_UseRepulsion_S_BC")                   == 0) StageThree_UseRepulsion_S_BC = itr->value.GetInt();
    if (ss.compare("StageThree_UseSparseConcatenation")              == 0) StageThree_UseSparseConcatenation = itr->value.GetInt();
    if (ss.compare("StageThree_UseSparseLatent_B_Term")              == 0) StageThree_UseSparseLatent_B_Term = itr->value.GetInt();
    if (ss.compare("StageThree_UseSparseMatS")                       == 0) StageThree_UseSparseMatS = itr->value.GetInt();
    if (ss.compare("StageThree_UseSpatialRegularization")            == 0) StageThree_UseSpatialRegularization = itr->value.GetInt();
    if (ss.compare("StageThree_ValidationConditionalScoreThreshold") == 0) StageThree_ValidationConditionalScoreThreshold = itr->value.GetDouble();
    if (ss.compare("StageThree_ValidationLossDecreaseThreshold")     == 0) StageThree_ValidationLossDecreaseThreshold = itr->value.GetDouble();
    if (ss.compare("StageThree_ValidationLossIncreaseThreshold")     == 0) StageThree_ValidationLossIncreaseThreshold = itr->value.GetDouble();
    if (ss.compare("StartFromStage")                                 == 0) StartFromStage = itr->value.GetInt();
    if (ss.compare("TrainFrequencyOffset_A")                         == 0) TrainFrequencyOffset_A = itr->value.GetInt();
    if (ss.compare("TrainFrequencyOffset_B")                         == 0) TrainFrequencyOffset_B = itr->value.GetInt();
    if (ss.compare("TrainFrequencyOffset_Bias")                      == 0) TrainFrequencyOffset_Bias = itr->value.GetInt();
    if (ss.compare("TrainFrequencyOffset_C")                         == 0) TrainFrequencyOffset_C = itr->value.GetInt();
    if (ss.compare("TruncatedLasso_Window")                          == 0) TruncatedLasso_Window = itr->value.GetDouble();
    if (ss.compare("TruncatedLasso_Window_ScaleFactor")              == 0) TruncatedLasso_Window_ScaleFactor = itr->value.GetDouble();
    if (ss.compare("TruncatedLasso_Window_UpdateFrequency")          == 0) TruncatedLasso_Window_UpdateFrequency = itr->value.GetInt();
    if (ss.compare("TuneSparsityEveryBatch")                         == 0) TuneSparsityEveryBatch = itr->value.GetInt();
    if (ss.compare("UseFeatureThreshold")                            == 0) UseFeatureThreshold = itr->value.GetInt();
  }
}
void Settings::Stage_One_writeToFile(string fp_snapshot, string session_id) {
}
void Settings::Stage_Three_writeToFile(string fp_snapshot, string session_id) {

  ofstream myfile;
  myfile.open(fp_snapshot + "/rank3/stage3/" + DATASET_NAME + "_r3s3_" + session_id + "_settings.json");
  myfile << "session_id" << ", " << session_id << ",\n";
  myfile.close();
}

void Settings::PrintSettings(int stage) {
  PrintFancy(session_start_time, "Using the following files and filepaths\n");
  PrintDelimiter(0, 0, 40, '                             =');
  cout << "StartFromStage                                = " << StartFromStage << endl;
  PrintDelimiter(0, 0, 40, '                             =');

  if (stage == 1) {
    PrintDelimiter(0, 0, 40, '                             =');
    cout << "Stage 1 settings:" << endl;
    PrintDelimiter(0, 0, 80, '                             =');
    cout << "StageOne_StartingLearningRate                 = " << StageOne_StartingLearningRate << endl;
    cout << "StageOne_Weight_U_RegularizationType          = " << StageOne_Weight_U_RegularizationType << endl;
    PrintDelimiter(0, 0, 40, '                             =');
    cout << "Dimension_A                                   = " << Dimension_A << endl;
    cout << "StageOne_Dimension_B                          = " << StageOne_Dimension_B << endl;
    cout << "StageOne_Dimension_C                          = " << StageOne_Dimension_C << endl;
    cout << "Dimension_B_BiasSlice                         = " << Dimension_B_BiasSlice << endl;
    cout << "Dimension_C_BiasSlice                         = " << Dimension_C_BiasSlice << endl;
  }

  if (stage == 3) {

    cout << "Stage 3 settings:" << endl;

    PrintDelimiter(0, 1, 40, '                             =');
    cout << "NumberOfLatentDimensions_                     = " << NumberOfLatentDimensions << endl;
    cout << "StageThree_SubDimension_1                     = " << StageThree_SubDimension_1 << endl;
    cout << "StageThree_SubDimension_2                     = " << StageThree_SubDimension_2 << endl;
    cout << "StageThree_Dimension_B                        = " << StageThree_Dimension_B << endl;
    cout << "StageThree_Dimension_C                        = " << StageThree_Dimension_C << endl;
    PrintDelimiter(0, 0, 80, '                             =');

    cout << "StageThree_UseSpatialRegularization           = " << StageThree_UseSpatialRegularization << endl;

    PrintDelimiter(1, 0, 40, '                             =');

    cout << "StageThree_StartingLearningRate               = " << StageThree_StartingLearningRate << endl;

    cout << "StageThree_UseSparseMatS                      = " << StageThree_UseSparseMatS << endl;
    cout << "StageThree_UseSparseLatent_B_Term             = " << StageThree_UseSparseLatent_B_Term << endl;
    cout << "StageThree_UseSparseConcatenation             = " << StageThree_UseSparseConcatenation << endl;

    cout << "TruncatedLasso_Window                         = " << TruncatedLasso_Window << endl;
    cout << "TruncatedLasso_Window_UpdateFrequency         = " << TruncatedLasso_Window_UpdateFrequency << endl;
    cout << "TruncatedLasso_Window_ScaleFactor             = " << TruncatedLasso_Window_ScaleFactor << endl;
    cout << "StageThree_Regularization_LF_A                = " << StageThree_Regularization_LF_A << endl;
    cout << "StageThree_B_SpatReg_Multiplier_Level1        = " << StageThree_B_SpatReg_Multiplier_Level1 << endl;
    cout << "StageThree_C_SpatReg_Multiplier_Level1        = " << StageThree_C_SpatReg_Multiplier_Level1 << endl;

    cout << "StageThree_UpdateSparseMatS_EveryNthMinibatch = " << StageThree_UpdateSparseMatS_EveryNthMinibatch << endl;
    cout << "StageThree_RegularizeS_EveryBatch             = " << StageThree_RegularizeS_EveryBatch << endl;
    cout << "StageThree_Regularization_Sparse_S_Level1     = " << StageThree_Regularization_Sparse_S_Level1 << endl;
    // cout << "StageThree_Regularization_Sparse_S_Level2  = " << StageThree_Regularization_Sparse_S_Level2 << endl;

    // PrintDelimiter(0, 0, 40, '                          =');
    // cout << "StageThree_LF_A_RegularizationType         = " << StageThree_LF_A_RegularizationType << endl;
    // cout << "StageThree_LF_B_RegularizationType         = " << StageThree_LF_B_RegularizationType << endl;
    // cout << "StageThree_SLF_B_RegularizationType        = " << StageThree_SLF_B_RegularizationType << endl;
    // cout << "StageThree_LF_C_RegularizationType         = " << StageThree_LF_C_RegularizationType << endl;
    // cout << "StageThree_Sparse_S_RegularizationType     = " << StageThree_Sparse_S_RegularizationType << endl;
    // PrintDelimiter(0, 0, 40, '                          =');
    // cout << "StageThree_UseRepulsion_S_BC               = " << StageThree_UseRepulsion_S_BC << endl;
    // cout << "StageThree_UseRepulsion_S_ABC              = " << StageThree_UseRepulsion_S_ABC << endl;
    // cout << "StageThree_Regularization_RepulsionTerm    = " << StageThree_Regularization_RepulsionTerm << endl;
    // cout << "StageThree_Regularization_S_BC             = " << StageThree_Regularization_S_BC << endl;
    // PrintDelimiter(0, 0, 40, '                          =');
    // cout << "GradientRandomizeAtZero                    = " << GradientRandomizeAtZero << endl;
  }
}