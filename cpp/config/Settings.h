

#ifndef SETTINGS_H
#define SETTINGS_H

#include <boost/unordered_map.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <chrono>
#include <time.h>
#include <random>

#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <queue>

#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/reader.h"

#include "core/DataBlob.h"
#include "util/PrettyOutput.h"
#include "config/GlobalConstants.h"

using namespace std;

class Settings {
public:
  void Stage_One_writeToFile(string fp_snapshot, string session_id);
  void Stage_Three_writeToFile(string fp_snapshot, string session_id);
  void loadSettingsJSON(const char * fn_settings_file);
  void PrintSettings(int stage);

  int EnableDebugPrinter_Level1;
  int EnableDebugPrinter_Level2;
  int EnableDebugPrinter_Level3;


  float TruncatedLasso_Window;
  int TruncatedLasso_Window_UpdateFrequency;
  float TruncatedLasso_Window_ScaleFactor;

  chrono::high_resolution_clock::time_point session_start_time;

  // Use for alternate gradient descent
  int TrainFrequencyOffset_A;
  int TrainFrequencyOffset_B;
  int TrainFrequencyOffset_C;

  int session_id;
  string  PYTHON_LOGGING_ID;

  int CapScores;

  string DATASET_NAME;

  string  RootFolder;
  string  SnapshotSubFolder;
  string  GroundTruthSubFolder;
  string  FeaturesSubFolder;
  string  SeedSubFolder;

  string FILENAME_FEATURES_B;
  string FILENAME_FEATURES_B_window_3by3;
  string FILENAME_FEATURES_C;
  string FILENAME_FEATURES_C_window_3by3;
  string FILENAME_GTRUTH_TRAINVAL_STRONG;
  string FILENAME_GTRUTH_TEST_STRONG;
  string FILENAME_GTRUTH_TRAINVAL_WEAK;
  string FILENAME_GTRUTH_TEST_WEAK;

  // Global settings
  int CurrentNumberOfThreads;
  int NumberOfFrames;
  int NumberOfStrongLabels_TrainVal;
  int NumberOfStrongLabels_Train;
  int NumberOfStrongLabels_Val;
  int NumberOfStrongLabels_Test;
  int NumberOfWeakLabels_TrainVal;
  int NumberOfWeakLabels_Train;
  int NumberOfWeakLabels_Val;
  int NumberOfWeakLabels_Test;

  // Features
  float FeatureThreshold;
  int UseFeatureThreshold;
  int NumberOfLatentDimensions;

  int StageOne_NumberOfNonZeroEntries_B;
  int StageOne_NumberOfNonZeroEntries_C;
  int StageThree_NumberOfNonZeroEntries_B;
  int StageThree_NumberOfNonZeroEntries_C;

  int DummyMultiplier;

  // Ground truth
  float GroundTruthScoreWeak;
  float GroundTruthScoreStrong;

  // Loss-parameters
  float LossWeightWeak;
  float LossWeightStrong;

  VectorBlob TaskWeight;

  // Where to begin?
  int ResumeTraining;
  int StartFromStage;


  int NumberOfColumnsScoreFunction;
  int ScoreFunctionColumnIndexMod;

  int Dimension_A;
  int Dimension_B;
  int Dimension_C;

  int StageOne_Dimension_B;
  int StageOne_Dimension_C;
  int StageThree_Dimension_B;
  int StageThree_Dimension_C;
  int Dimension_B_BiasSlice;
  int Dimension_C_BiasSlice;

  int GradientRandomizeAtZero;

  int StageOne_Weight_U_RegularizationType;
  int StageThree_LF_A_RegularizationType;
  int StageThree_LF_B_RegularizationType;
  int StageThree_SLF_B_RegularizationType;
  int StageThree_LF_C_RegularizationType;
  int StageThree_Sparse_S_RegularizationType;

  // ---------------------------------------------------------------------------
  // Spatial entropy
  // ---------------------------------------------------------------------------
  int SpatialEntropy_NumberOfBins;
  float SpatialEntropy_MinValue;
  float SpatialEntropy_MaxValue;
  float SpatialEntropy_BinWidth;

  // ---------------------------------------------------------------------------
  // Log files for timings
  // ---------------------------------------------------------------------------
  string LogFolder;
  string LogFile_Loss;
  string LogFile_CellEntropy;
  string LogFile_Probabilities;



  // ---------------------------------------------------------------------------
  // Stage 1
  // ---------------------------------------------------------------------------
  float StageOne_LossTrainPreviousEpoch;
  float StageOne_LossValidPreviousEpoch;

  float StageOne_ValidationLossIncreaseThreshold;
  float StageOne_ValidationLossDecreaseThreshold;

  int StageOne_Weight_U_UseSoftThreshold;

  int StageOne_Weight_U_ClampToThreshold;
  float StageOne_Weight_U_Threshold;

  int StageOne_StartFromCrossValRun;
  int StageOne_StartFromEpoch;
  string  StageOne_SnapshotWeightsBiases;
  string  StageOne_SnapshotMomentum;

  float StageOne_StartingLearningRate;
  float StageOne_CurrentLearningRate;
  float StageOne_Cumulative_sum_of_squared_first_derivatives;  // For ADAPTIVE GD
  int StageOne_UseAdaptiveLearningRate;
  float StageOne_Clamp_AdaptiveLearningRate;

  float TEST_train_ExponentialScore;

  // Global training Settings_
  int StageOne_NumberOfThreads;
  int StageOne_NumberOfCrossValidationRuns;
  int StageOne_NumberOfEpochs;
  float StageOne_TrainingConditionalScoreThreshold;
  float StageOne_ValidationConditionalScoreThreshold;
  float StageOne_TestConditionalScoreThreshold;
  int StageOne_ComputeLossAfterEveryNthEpoch;
  int StageOne_Take_SnapshotAfterEveryNthEpoch;
  int StageOne_MiniBatchSize;
  int StageOne_ApplyMomentumEveryNthMinibatch;
  int StageOne_ResetMomentumEveryNthEpoch;
  int StageOne_StatusEveryNBatchesTrain;
  int StageOne_StatusEveryNBatchesValid;
  int StageOne_UseBias;

  // Stage 1 training specific
  float StageOne_Initialization_Weight_U_range;
  float StageOne_Initialization_Bias_A_range;
  float StageOne_Regularization_Weight_U_Level1;
  float StageOne_Regularization_Weight_U_Level2;
  float StageOne_Regularization_Weight_U_Level3;
  float StageOne_Regularization_Bias_A;

  int StageOne_UseSpatialRegularization;
  int SpatialRegularization;

  // ---------------------------------------------------------------------------
  // Stage 3
  // ---------------------------------------------------------------------------
  int StageThree_UseSpatialRegularization;

  float StageThree_LossTrainPreviousEpoch;
  float StageThree_LossValidPreviousEpoch;

  float StageThree_ValidationLossIncreaseThreshold;
  float StageThree_ValidationLossDecreaseThreshold;

  int StageThree_Sparse_S_UseSoftThreshold;
  float StageThree_SparseMatSThreshold;
  int StageThree_Sparse_S_ClampToThreshold;

  float StageThree_S_SpatReg_Multiplier_Level1;
  float StageThree_S_SpatReg_Multiplier_Level2;

  // Sparsity control mechanism
  int StageThree_TuneSparsity;
  int TuneSparsityEveryBatch;
  int SparsityRunningAverageWindowSize;
  float SPARSITY_THRESHOLD;
  float DesiredSparsityLevel;
  float CurrentSparsityLevel_B_SubDim1;
  float CurrentSparsityLevel_B_SubDim2;
  float CurrentSparsityLevel_C_SubDim1;
  float CurrentSparsityLevel_C_SubDim2;


  // Save / resume training settings
  int StageThree_StartFromCrossValRun;
  int StageThree_StartFromEpoch;
  string  StageThree_SnapshotWeightsBiases;
  string  StageThree_SnapshotMomentum;

  // Initial seeds from stage 2
  int StageThree_StartFromSeed;

  string  StageThree_LF_A_seed;
  string  StageThree_LF_B_seed;
  string  StageThree_LF_C_seed;

  string  StageThree_LF_SA_seed;
  string  StageThree_LF_SB_seed;
  string  StageThree_LF_SC_seed;

  string  StageThree_Sparse_S_seed;

  float StageThree_StartingLearningRate;
  float StageThree_CurrentLearningRate;
  float StageThree_Cumulative_sum_of_squared_first_derivatives;  // For ADAPTIVE GD
  int StageThree_UseAdaptiveLearningRate;
  float StageThree_Clamp_AdaptiveLearningRate;

  int StageThree_NumberOfThreads;
  int StageThree_NumberOfCrossValidationRuns;
  int StageThree_NumberOfEpochs;
  float StageThree_TrainingConditionalScoreThreshold;
  float StageThree_ValidationConditionalScoreThreshold;
  float StageThree_TestConditionalScoreThreshold;
  int StageThree_ComputeLossAfterEveryNthEpoch;
  int StageThree_Take_SnapshotAfterEveryNthEpoch;
  int StageThree_MiniBatchSize;
  int StageThree_ApplyMomentumEveryNthMinibatch;
  int StageThree_ResetMomentumEveryNthEpoch;

  int StageThree_UpdateSparseMatS_EveryNthMinibatch;
  int StageThree_RegularizeS_EveryBatch;

  float StageThree_Initialization_SLF_B_range;

  int StageThree_TrainOnLatentFactorsEveryBatch;

  int StageThree_Train_LF_A_EveryBatch;
  int StageThree_Train_LF_B_EveryBatch;
  int StageThree_Train_SLF_B_EveryBatch;
  int StageThree_Train_LF_C_EveryBatch;
  int StageThree_TrainBiasEveryBatch;


  int TrainFrequencyOffset_Bias;

  int StageThree_StatusEveryNBatchesTrain;
  int StageThree_StatusEveryNBatchesValid;
  int StageThree_UseBias;
  int StageThree_UseSparseMatS;
  int StageThree_UseLatentTerm;

  int StageThree_UseSparseLatent_B_Term;

  int StageThree_UseRepulsion_S_BC;
  int StageThree_UseRepulsion_S_ABC;
  float StageThree_Regularization_RepulsionTerm;
  float StageThree_Regularization_S_BC;


  int StageThree_UseSparseConcatenation;
  int StageThree_SubDimension_1;
  int StageThree_SubDimension_2;
  int StageThree_TrainSubDimension_1;
  int StageThree_TrainSubDimension_2;
  float StageThree_A_SubDimension_2_InitRange;
  float StageThree_B_SubDimension_2_InitRange;
  float StageThree_C_SubDimension_2_InitRange;
  int StageThree_RegularizationType_1;
  int StageThree_RegularizationType_2;

  float StageThree_A_RegularizationStrength_1_Dense;
  float StageThree_A_RegularizationStrength_1_Sparse;
  float StageThree_A_RegularizationStrength_2_Dense;
  float StageThree_A_RegularizationStrength_2_Sparse;

  float StageThree_B_RegularizationStrength_1_Dense;
  float StageThree_B_RegularizationStrength_1_Sparse;
  float StageThree_B_RegularizationStrength_2_Dense;
  float StageThree_B_RegularizationStrength_2_Sparse;

  float StageThree_C_RegularizationStrength_1_Dense;
  float StageThree_C_RegularizationStrength_1_Sparse;
  float StageThree_C_RegularizationStrength_2_Dense;
  float StageThree_C_RegularizationStrength_2_Sparse;

  float StageThree_Regularization_repulsion_BB;

  float StageThree_Initialization_Weight_W_range;
  float StageThree_Initialization_LF_A_range;
  float StageThree_Initialization_Sparse_S_range;
  float StageThree_Initialization_Bias_A_range;

  float StageThree_Regularization_LF_A;

  float StageThree_B_SpatReg_Multiplier_Level1;
  float StageThree_B_SpatReg_Multiplier_Level2;

  float StageThree_Regularization_LF_B_Level1;
  float StageThree_Regularization_LF_B_Level2;
  float StageThree_Regularization_LF_C_Level1;
  float StageThree_Regularization_LF_C_Level2;

  float StageThree_Regularization_SLF_B_Level1;
  float StageThree_Regularization_SLF_B_Level2;

  float StageThree_C_SpatReg_Multiplier_Level1;
  float StageThree_C_SpatReg_Multiplier_Level2;

  float StageThree_Regularization_Sparse_S_Level1;
  float StageThree_Regularization_Sparse_S_Level2;
  float StageThree_Regularization_Sparse_S_Level3;
  float StageThree_Regularization_Bias_A;
};



#endif