#include <time.h>
#include <math.h>
#include <chrono>

#include <boost/unordered_map.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/thread.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits.hpp>

#include <stdlib.h>
#include <cstdlib>
#include <cstdio>
#include <string>

#include "config/GlobalConstants.h"
#include "config/Settings.h"

#include "core/Trainer.h"
#include "core/DataBlob.h"
#include "core/FullTensorModelTrainer.h"
#include "core/QueueMessage.h"
#include "core/SpatialEntropy.h"
#include "core/TensorFactorization.h"
#include "core/ThreeMatrixFactorTrainer.h"

#include "util/CurrentStateBlob.h"
#include "util/IOController.h"
#include "util/PerformanceController.h"
#include "util/PrettyOutput.h"

// #define Settings_->DummyMultiplier 1

// #define Settings_->StageOne_NumberOfNonZeroEntries_B (2 * 8)
// #define Settings_->StageOne_NumberOfNonZeroEntries_C (2 * 4)

// #define Settings_->StageThree_NumberOfNonZeroEntries_B (2 * 8)
// #define Settings_->StageThree_NumberOfNonZeroEntries_C (2 * 4)

#define _SCRIPTTITLE \
"[Stage 1 + 2 + 3] This script performs training of a low-rank \
latent factor to recognize semantic relationships \
between entities in images."

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::time_point;
using std::chrono::system_clock;

void LoadJSONfiles(Settings* s, char *argv[]) {
  char * filepaths_json_file;
  char * features_json_file;
  char * labels_json_file;
  char * training_param_json_file;

  filepaths_json_file      = argv[1];
  features_json_file       = argv[2];
  labels_json_file         = argv[3];
  training_param_json_file = argv[4];
  s->loadSettingsJSON(filepaths_json_file);
  s->loadSettingsJSON(features_json_file);
  s->loadSettingsJSON(labels_json_file);
  s->loadSettingsJSON(training_param_json_file);

}

/**
 * Use: ./EXEC_NAME settings.json [1 override_value1 override_value2 ... ...]
 * For override options, see below
 * @param
 * @param
 * @return
 */
int main(int argc, char *argv[]) {

  // Some time logging. Needed for timing of run.
  high_resolution_clock::time_point start_time = high_resolution_clock::now();
  Settings *Settings_                          = new Settings;
  Settings_->session_start_time                = start_time;
  PerformanceController PerformanceController_(Settings_);

  PrintWithDelimiters(Settings_->session_start_time, _SCRIPTTITLE);

  time_t now = system_clock::to_time_t(Settings_->session_start_time);
  char timestamp[80];
  strftime(timestamp, sizeof(timestamp), "%d%b%H%M%S", localtime(&now));

  CurrentStateBlob *CurrentStateBlob_ = new CurrentStateBlob;
  CurrentStateBlob_->start_time       = Settings_->session_start_time;
  CurrentStateBlob_->session_id       = timestamp;

  // ================================================================================================
  // Load training Settings_
  // ================================================================================================

  int _override = 0;
  PrintFancy(Settings_->session_start_time, "Seen " + to_string(argc) + " arguments" );


  assert(argc > 1);  // We need at least a settings -- *.json -- file
  // You can also override JSON settings with command-line settings

  if (argc != 5 && argc != 22) {
    PrintFancy<string>("You should specify either 3 JSON files *or* 3 JSON + 16 override arguments.");
    exit;
  } else if (argc == 5) {
    LoadJSONfiles(Settings_, argv);
  } else if (argc == 22) {
    LoadJSONfiles(Settings_, argv);

    for (int i = 0; i < argc; ++i) {
      if (i == 2) {
        assert(argc == 22);  // If we specify more options than just the settings-file, than we need to specify exactly the right number of override settings
        if (stoi(argv[i]) == 1) {
          PrintFancy(Settings_->session_start_time, "You specified additional arguments, so these will override some settings from those in the JSON file!");
          _override = 1;
        }
      }
      // TODO(stz): make this adaptive, not fixed indices.
      if (_override == 1) {
        cout << argv[i] << endl;
        if (i == 3) {
          Settings_->StageOne_StartingLearningRate       = strtod(argv[i], NULL);
          Settings_->StageOne_Clamp_AdaptiveLearningRate = strtod(argv[i], NULL);
        }
        if (i == 4) {
          Settings_->StageThree_StartingLearningRate       = strtod(argv[i], NULL);
          Settings_->StageThree_Clamp_AdaptiveLearningRate = strtod(argv[i], NULL);
        }

        if (i == 5) Settings_->LossWeightWeak                       = strtod(argv[i], NULL);
        if (i == 6) Settings_->StartFromStage                       = stoi(argv[i], NULL);

        if (i == 7) {
          Settings_->NumberOfLatentDimensions  = stoi(argv[i], NULL);
          // Settings_->StageThree_LF_A_seed = Settings_->SeedSubFolder + "/seed_X_" + to_string(Settings_->NumberOfLatentDimensions) + ".bin";
          // Settings_->StageThree_LF_B_seed = Settings_->SeedSubFolder + "/seed_Y_" + to_string(Settings_->NumberOfLatentDimensions) + ".bin_upscaled";
          // Settings_->StageThree_LF_C_seed = Settings_->SeedSubFolder + "/seed_Z_" + to_string(Settings_->NumberOfLatentDimensions) + ".bin_upscaled";
        }
        if (i == 8) Settings_->StageThree_UseSparseMatS                 = stoi(argv[i], NULL);

        if (i == 9) Settings_->StageThree_Regularization_LF_A               = strtod(argv[i], NULL);
        if (i == 10) Settings_->StageThree_ApplyMomentumEveryNthMinibatch         = stoi(argv[i], NULL);
        if (i == 11) Settings_->StageThree_ResetMomentumEveryNthEpoch           = stoi(argv[i], NULL);
        if (i == 12) Settings_->StageThree_TrainOnLatentFactorsEveryBatch         = stoi(argv[i], NULL);
        if (i == 13) Settings_->StageThree_UpdateSparseMatS_EveryNthMinibatch       = stoi(argv[i], NULL);
        if (i == 14) Settings_->StageThree_RegularizeS_EveryBatch             = stoi(argv[i], NULL);
        if (i == 15) Settings_->StageThree_Regularization_Sparse_S_Level1         = strtod(argv[i], NULL);
        if (i == 16) Settings_->StageThree_Regularization_Sparse_S_Level2         = strtod(argv[i], NULL);
        if (i == 17) Settings_->StageThree_Initialization_Sparse_S_range        = strtod(argv[i], NULL);

        if (i == 18) Settings_->PYTHON_LOGGING_ID                     = stoi(argv[i], NULL);
      }
    }
  }



  Settings_->PrintSettings(Settings_->StartFromStage);

  // --------------------------------------------------------------------------------------------------------------------------
  // Setting filepaths
  // --------------------------------------------------------------------------------------------------------------------------
  string fp_rootfolder                         = Settings_->RootFolder;
  string fp_snapshot                           = Settings_->SnapshotSubFolder;

  string fp_features_B                         = Settings_->FeaturesSubFolder + Settings_->FILENAME_FEATURES_B;
  string fp_features_B_window_3by3             = Settings_->FeaturesSubFolder + Settings_->FILENAME_FEATURES_B_window_3by3;
  string fp_features_C                         = Settings_->FeaturesSubFolder + Settings_->FILENAME_FEATURES_C;
  string fp_features_C_window_3by3             = Settings_->FeaturesSubFolder + Settings_->FILENAME_FEATURES_C_window_3by3;

  string fp_groundtruth_trainval_split_strong  = Settings_->GroundTruthSubFolder + Settings_->FILENAME_GTRUTH_TRAINVAL_STRONG;
  string fp_groundtruth_test_split_strong      = Settings_->GroundTruthSubFolder + Settings_->FILENAME_GTRUTH_TEST_STRONG;

  // string fp_groundtruth_trainval_split_weak = Settings_->GroundTruthSubFolder + "/groundtruth_weak.bin";

  string fp_groundtruth_trainval_split_weak    = Settings_->GroundTruthSubFolder + Settings_->FILENAME_GTRUTH_TRAINVAL_WEAK;
  string fp_groundtruth_test_split_weak        = Settings_->GroundTruthSubFolder + Settings_->FILENAME_GTRUTH_TEST_WEAK;

  cout << fp_rootfolder << endl;
  cout << fp_snapshot << endl;

  cout << fp_features_B << endl;
  cout << fp_features_C << endl;

  cout << fp_groundtruth_trainval_split_strong << endl;
  cout << fp_groundtruth_test_split_strong << endl;
  cout << fp_groundtruth_trainval_split_weak << endl;
  cout << fp_groundtruth_test_split_weak << endl;




  IOController *IOController_ = new IOController(Settings_, CurrentStateBlob_);

  // Load ground-truth labels
  high_resolution_clock::time_point start_time_load_ground_truth = high_resolution_clock::now();

  GroundTruthLabel *GroundTruthLabelsTrainValStrong_ = new GroundTruthLabel[Settings_->NumberOfStrongLabels_TrainVal];
  GroundTruthLabel *GroundTruthLabelsTestStrong_     = new GroundTruthLabel[Settings_->NumberOfStrongLabels_Test];
  GroundTruthLabel *GroundTruthLabelsTrainValWeak_   = new GroundTruthLabel[Settings_->NumberOfWeakLabels_TrainVal];
  GroundTruthLabel *GroundTruthLabelsTestWeak_       = new GroundTruthLabel[Settings_->NumberOfWeakLabels_Test];

  IOController_->LoadGroundTruthLabelsStrong(fp_groundtruth_trainval_split_strong,  GroundTruthLabelsTrainValStrong_,   Settings_->NumberOfStrongLabels_TrainVal);
  IOController_->LoadGroundTruthLabelsWeak(  fp_groundtruth_trainval_split_weak,    GroundTruthLabelsTrainValWeak_,   Settings_->NumberOfWeakLabels_TrainVal);

  PerformanceController_.showTimeElapsed(start_time_load_ground_truth);

  FullTensorModelTrainer   *FullTensorModelTrainer_;
  ThreeMatrixFactorTrainer *ThreeMatrixFactorTrainer_;

  MatrixBlob *Blob_B_;
  MatrixBlob *Blob_BB_;
  MatrixBlob *Blob_BTransposed_;

  MatrixBlob *Blob_C_;
  MatrixBlob *Blob_CC_;
  MatrixBlob *Blob_CTransposed_;

  PrintWithDelimiters(Settings_->session_start_time, "Settings_->StartFromStage: " + to_string(Settings_->StartFromStage));

  // Initialize class-weights
  Settings_->TaskWeight.init(Settings_->Dimension_A);
  for (int i = 0; i < Settings_->Dimension_A; ++i)
  {
    *(Settings_->TaskWeight.att(i)) = 1.0;
  }
  // Manual reweighting
  // *(Settings_->TaskWeight.att(8)) = 0.1;
  // *(Settings_->TaskWeight.att(9)) = 0.1;
  // *(Settings_->TaskWeight.att(18)) = 0.1;
  // *(Settings_->TaskWeight.att(19)) = 0.1;

  // ================================================================================================
  // ================================================================================================
  // Training regulator:
  // The main training routine gives an exit-code.
  // 0 = trained successfully (no blow-up)
  // -1 = blow-up detected. --> drop training-rate and train again until we get exit-code 0.
  int exit_code = -100;
  // ================================================================================================
  // ================================================================================================

  if (Settings_->StartFromStage == 1) {

    // ================================================================================================
    // Stage 1 - train full-rank model on FULL data-set (overfit ok with all strong / weak labels)
    // ================================================================================================

    high_resolution_clock::time_point start_time_load_features = high_resolution_clock::now();

    PrintWithDelimiters(Settings_->session_start_time, "Settings_->NumberOfFrames: " + to_string(Settings_->NumberOfFrames));

    MatrixBlob *Blob_B_ = new MatrixBlob(Settings_->NumberOfFrames, Settings_->StageOne_NumberOfNonZeroEntries_B);
    MatrixBlob *Blob_C_ = new MatrixBlob(Settings_->NumberOfFrames, Settings_->DummyMultiplier * Settings_->StageOne_NumberOfNonZeroEntries_C);

    IOController_->LoadFeaturesFromFile(fp_features_B, Blob_B_, 0, 0, 1);
    // IOController_->LoadFeaturesFromFile(fp_features_coarse_bh, Blob_BTransposed_, 1, 0, 1);
    IOController_->LoadFeaturesFromFile(fp_features_C, Blob_C_, 0, 1, 1);
    // IOController_->LoadFeaturesFromFile(fp_features_coarse_def, Blob_CTransposed_,   1, 1, 1);

    PerformanceController_.showTimeElapsed(start_time_load_features);

    high_resolution_clock::time_point start_time_stage_one = high_resolution_clock::now();

    PrintWithDelimiters(Settings_->session_start_time, "Starting Stage 1");

    FullTensorModelTrainer_ = new FullTensorModelTrainer(IOController_,
                           Blob_B_,
                           Blob_BTransposed_,
                           Blob_C_,
                           Blob_CTransposed_,
                           GroundTruthLabelsTrainValStrong_,
                           GroundTruthLabelsTestStrong_,
                           GroundTruthLabelsTrainValWeak_,
                           GroundTruthLabelsTestWeak_,
                           Settings_,
                           CurrentStateBlob_);

    while(true) {
      exit_code = FullTensorModelTrainer_->TrainStageOne(fp_snapshot);
      if (exit_code != 0) {
        Settings_->StageOne_StartingLearningRate *= LEARNING_RATE_REDUCTION_FACTOR;
        PrintFancy(Settings_->session_start_time, "Train routine exited with code -1. Trying again with learning-rate DOWN to" + to_string(Settings_->StageOne_StartingLearningRate));
      } else break;
    }

    PerformanceController_.showTimeElapsed(start_time_stage_one);

    delete FullTensorModelTrainer_;
  }


  if (Settings_->StartFromStage == 3 or Settings_->StartFromStage == 5) {

    // ================================================================================================
    // Stage 3 - train full-rank model on FULL data-set
    // ================================================================================================

    // ================================================================================================
    // Input to the program - linked interactions, visual features, ground truth labels - fixed!
    // ================================================================================================

    high_resolution_clock::time_point start_time_load_features = high_resolution_clock::now();

    PrintWithDelimiters(Settings_->session_start_time, "Settings_->NumberOfFrames: " + to_string(Settings_->NumberOfFrames));

    Blob_B_       = new MatrixBlob(Settings_->NumberOfFrames, Settings_->StageThree_NumberOfNonZeroEntries_B);
    Blob_BTransposed_ = new MatrixBlob(Settings_->StageThree_NumberOfNonZeroEntries_B, Settings_->NumberOfFrames);

    Blob_C_       = new MatrixBlob(Settings_->NumberOfFrames, Settings_->DummyMultiplier * (Settings_->StageThree_NumberOfNonZeroEntries_C));
    Blob_CTransposed_ = new MatrixBlob(Settings_->DummyMultiplier * Settings_->StageThree_NumberOfNonZeroEntries_C, Settings_->NumberOfFrames);

    IOController_->LoadFeaturesFromFile(fp_features_B, Blob_B_, 0, 0, 3);
    IOController_->LoadFeaturesFromFile(fp_features_C, Blob_C_, 0, 1, 3);

    if (Settings_->StageThree_UseSpatialRegularization == 1 and Settings_->SpatialRegularization == 1) {
      Blob_BB_ = new MatrixBlob(Settings_->NumberOfFrames, Settings_->StageThree_NumberOfNonZeroEntries_B * NUMBER_OF_INTS_WINDOW_3BY3);
      Blob_CC_ = new MatrixBlob(Settings_->NumberOfFrames, Settings_->DummyMultiplier * (Settings_->StageThree_NumberOfNonZeroEntries_C - Settings_->Dimension_C_BiasSlice) * NUMBER_OF_INTS_WINDOW_3BY3);
      IOController_->LoadFeaturesFromFile(fp_features_B_window_3by3, Blob_BB_, 0, 2, 3);
      IOController_->LoadFeaturesFromFile(fp_features_C_window_3by3, Blob_CC_, 0, 3, 3);
    }

    PerformanceController_.showTimeElapsed(start_time_load_features);

    high_resolution_clock::time_point start_time_stage_three = high_resolution_clock::now();

    PrintWithDelimiters(Settings_->session_start_time, "Starting Stage 3");

    ThreeMatrixFactorTrainer_ = new ThreeMatrixFactorTrainer(IOController_,
                             Blob_B_,
                             Blob_BB_,
                             Blob_BTransposed_,
                             Blob_C_,
                             Blob_CC_,
                             Blob_CTransposed_,
                             GroundTruthLabelsTrainValStrong_,
                             GroundTruthLabelsTestStrong_,
                             GroundTruthLabelsTrainValWeak_,
                             GroundTruthLabelsTestWeak_,
                             Settings_,
                             CurrentStateBlob_);

    while(true) {
      exit_code = ThreeMatrixFactorTrainer_->TrainStageThree(fp_snapshot);
      if (exit_code != 0) {
        Settings_->StageThree_StartingLearningRate *= LEARNING_RATE_REDUCTION_FACTOR;
        PrintFancy(Settings_->session_start_time, "Train routine exited with code -1. Trying again with learning-rate DOWN to" + to_string(Settings_->StageThree_StartingLearningRate));
      } else break;
    }

    PerformanceController_.showTimeElapsed(start_time_stage_three);

    delete ThreeMatrixFactorTrainer_;
  }

  // ================================================================================================
  // Clear the heap for neatness
  // ================================================================================================
  delete Settings_;
  delete IOController_;
  delete Blob_B_;
  delete Blob_BTransposed_;
  delete Blob_C_;
  delete Blob_CTransposed_;

  if (Settings_->StageThree_UseSpatialRegularization == 1 and Settings_->SpatialRegularization == 1) {
    delete Blob_BB_;
    delete Blob_CC_;
  }

  delete CurrentStateBlob_;

  delete[] GroundTruthLabelsTrainValStrong_;
  delete[] GroundTruthLabelsTestStrong_;
  delete[] GroundTruthLabelsTrainValWeak_;
  delete[] GroundTruthLabelsTestWeak_;

  return 0;
}




