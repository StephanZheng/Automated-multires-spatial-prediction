

#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <boost/filesystem.hpp>
#include <boost/unordered_map.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <chrono>
#include <time.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <vector>
#include <string>
#include <queue>
#include <iostream>
#include <fstream>

#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/reader.h"

#include "config/GlobalConstants.h"
#include "config/Settings.h"

#include "core/DataBlob.h"
#include "core/QueueMessage.h"
#include "core/SpatialEntropy.h"

#include "util/IOController.h"
#include "util/PerformanceController.h"
#include "util/PrettyOutput.h"
#include "util/TrainUtil.h"

using namespace std;
using namespace boost;
using namespace std::chrono;
using std::chrono::high_resolution_clock;

class Trainer {

public:
  bool debug_mode;
  void EnableDebugMode() {debug_mode = true;};
  void DisableDebugMode() {debug_mode = false;};

  SpatialEntropy spatial_entropy;
  SpatialEntropy spatial_entropy_sign;

  string logfile_loss_filename_;
  string logfile_entropy_filename_;
  string logfile_probs_filename_;
  string logfile_entropy_sign_filename_;
  string logfile_probs_sign_filename_;

  // We use this to store the gradient from the response function.
  // This has to be computed once, so all threads can use it when
  // computing the spatial entropy.
  float dresponsefunction_dB_;

  int         CheckIfFloatIsNan(float x, string y);

  // Pointers to settings / IOController that are needed by all stages of training
  IOController    *IOController_;
  MatrixBlob      *Blob_B_;
  MatrixBlob      *Blob_BB_;
  MatrixBlob      *Blob_BTransposed_;
  MatrixBlob      *Blob_C_;
  MatrixBlob      *Blob_CC_;
  MatrixBlob      *Blob_CTransposed_;

  GroundTruthLabel  *GroundTruthLabelsTrainValStrong_;
  GroundTruthLabel  *GroundTruthLabelsTestStrong_;
  GroundTruthLabel  *GroundTruthLabelsTrainValWeak_;
  GroundTruthLabel  *GroundTruthLabelsTestWeak_;
  Settings          *Settings_;
  CurrentStateBlob  *CurrentStateBlob_;

  std::vector<int>  CurrentTrainIndices_Strong;
  std::vector<int>  CurrentValidIndices_Strong;
  std::vector<int>  CurrentTrainIndices_Weak;
  std::vector<int>  CurrentValidIndices_Weak;

  // At the start of every x-val run, we permute the train-val sets, and store a sorted
  // version of both to minimize lookup time
  std::vector<int>  CurrentTrainIndices_Strong_Sorted;
  std::vector<int>  CurrentTrainIndices_Weak_Sorted;

  std::vector<int>  RandomizedIterator;

  std::vector<float>  NestorovMomentumLambda;
  std::vector<float>  NestorovMomentumPreviousWeight;
  std::vector<float>  NestorovMomentumPreviousLF;
  std::vector<float>  NestorovMomentumPreviousLF_A;
  std::vector<float>  NestorovMomentumPreviousLF_B;
  std::vector<float>  NestorovMomentumPreviousSLF_B;
  std::vector<float>  NestorovMomentumPreviousLF_C;
  std::vector<float>  NestorovMomentumPreviousSparse_S;
  std::vector<float>  NestorovMomentumPreviousBias_A;

  // All stages have the same score array
  MatrixBlob      ConditionalScore;

  // Store computed losses
  std::vector<float>  Results_LossTrain;
  std::vector<float>  Results_LossValid;
  std::vector<float>  Results_LossTest;

  // Multithreading
  TaskQueue task_queue_;

  // Deprecate or move out.
  // boost::mutex              mutex_write_to_bias_A;
  // boost::mutex              mutex_write_to_bias_A_diff;
  // boost::mutex              mutex_write_to_WeightW;
  // boost::mutex              mutex_write_to_WeightU_gradient;

  int global_reset_training;
  int n_victims;
  int _dummy_index_A_to_test_mt;

  Trainer() {}
  Trainer(int n_spatial_cells, int bins, float minval, float maxval) :
    spatial_entropy(n_spatial_cells,
                    bins,
                    minval,
                    maxval), \
    spatial_entropy_sign(n_spatial_cells,
                    2,
                    -1.0,
                    1.0) {
  }

  int         GetExpectedSizeSnapshotFile(Settings * s);

  void        PermuteTrainValIndices();

  FILE *        LoadFile(string fn, int expected_floats_in_file, int n_seek_float);
  void        LoadSnapshotWeight( Tensor3Blob * blob, string fn, int size_of_container, int start_of_container, int slice_start, int slice_end, int row_start, int row_end, int col_start, int col_end );
  void        LoadSnapshotWeight( MatrixBlob * blob, string fn, int size_of_container, int start_of_container, int row_start, int row_end, int col_start, int col_end );
  void        LoadSnapshotWeight( VectorBlob * blob, string fn, int size_of_container, int start_of_container, int col_start, int col_end );

  // Gradient methods
  float         GetLassoGradient(int thread_id, float x);

  float         SoftThresholdUpdate(int thread_id, int sign, float old_value, float update_value);

  void        UpdateTruncatedLassoWindow(int batch);

  // Multithread methods
  int         generateTrainingBatches(int task_type, int batch, int n_batches, int n_datapoints_processed_so_far_this_epoch, int MiniBatchSize);
  int         generateValidationBatches(int task_type, int batch, int n_batches, int n_datapoints_processed_so_far_this_epoch, int MiniBatchSize);

  void        PrintStatus();

  // Shared functions
  void        printDecision(MatrixBlob * blob, int col_lo, int col_hi, float sparsity_level, float running_average, float gradient_curr, float gradient_prev, int exit_code, string message);
  float         getRunningAverage(VectorBlob * blob);
  float         getGradient(VectorBlob * blob, int index);
  void        tuneSparsityLevel(int batch, MatrixBlob * blob, int factor, int subdim);
  int         tuneSparsityAll(int batch, MatrixBlob * LF_A, MatrixBlob * LF_B, MatrixBlob * LF_C);
  int         tuneSparsity(MatrixBlob * blob, int subdim);
  void        checkSparsity(MatrixBlob *blob, VectorBlob * sp_lvls, VectorBlob * sp_nonzeros, int col_lo, int col_hi);
  int         tuneSparsityCore(MatrixBlob * blob, VectorBlob * sp_lvls, VectorBlob * sp_nonzeros, int col_lo, int col_hi);
  float         getSparsityLevel(MatrixBlob * blob, int col_lo, int col_hi, float threshold);
  int         countElementsAboveThreshold(MatrixBlob * blob, int lo, int hi, float threshold);

  // Argument: train, valid, or test set
  float         ComputeLoss(int type);
  float         ComputeLossCore(std::vector<int> * WeakIndices, std::vector<int> * StrongIndices);
  float         boundScore(int frame_id, float score);


  void        ProcessComputedLosses(int cross_val_run, int epoch, float loss_train, float loss_valid, float loss_test);

  float         cap(int thread_id, int index_A, int frame_id, float input, float cap_hi, float cap_lo);

  virtual float     ComputeScore(int thread_id, int index_A, int frame_id) {};
  virtual int     ComputeConditionalScore(int thread_id, int index_A, int frame_id, int ground_truth_label) {};
  void        ComputeConditionalScoreSingleThreaded(int type);

  std::vector<int>  getIndices_B(int frame_id, int stage);
  std::vector<int>  getIndices_C(int frame_id, int stage);

  float         MatrixBlobDotProduct(  int thread_id,
                        int index_1,
                        int index_2,
                        MatrixBlob *Matrix_A,
                        MatrixBlob *Matrix_B
  );

  float         MatrixBlobDotProductThreeWay(int thread_id,
                        int index_1,
                        int index_2,
                        int index_3,
                        MatrixBlob *Matrix_A,
                        MatrixBlob *Matrix_B,
                        MatrixBlob *Matrix_C
  );

  void  BoostParameter( int batch,
              Tensor3Blob *parameter,
              Tensor3Blob *parameter_update,
              Tensor3Blob *parameter_update_squared_cache,
              std::vector<float>  *NestorovMomentumPreviousParameter_,
              int n_elements);

  void  BoostParameter( int batch,
              MatrixBlob *parameter,
              MatrixBlob *parameter_update,
              MatrixBlob *parameter_update_squared_cache,
              std::vector<float>  *NestorovMomentumPreviousParameter_,
              int n_elements);

  void  BoostParameter( int batch,
              VectorBlob *parameter,
              VectorBlob *parameter_update,
              VectorBlob *parameter_update_squared_cache,
              std::vector<float>  *NestorovMomentumPreviousParameter_,
              int n_elements);

  void initLogFiles(string logfile_prefix, string logfile_suffix) {
    // Log files for this session

    // Make directory where log files will be located
    string dir_path = Settings_->LogFolder + "/" + logfile_prefix + logfile_suffix;
    boost::filesystem::path dir(dir_path);
    if(boost::filesystem::create_directory(dir)) {
      std::cout << "Success creating " << dir_path << "\n";
    }

    // Loss
    logfile_loss_filename_ = dir_path + Settings_->LogFile_Loss + logfile_suffix;
    IOController_->OpenNewFile(logfile_loss_filename_);

    // Entropy
    logfile_entropy_filename_ = dir_path + Settings_->LogFile_CellEntropy + logfile_suffix;
    IOController_->OpenNewFile(logfile_entropy_filename_);
    // probs
    logfile_probs_filename_ = dir_path + Settings_->LogFile_Probabilities + logfile_suffix;
    IOController_->OpenNewFile(logfile_probs_filename_);

    // Entropy - sign-only
    logfile_entropy_sign_filename_ = dir_path + Settings_->LogFile_CellEntropy + "_sign" + logfile_suffix;
    IOController_->OpenNewFile(logfile_entropy_sign_filename_);
    // probs - sign-only
    logfile_probs_sign_filename_ = dir_path + Settings_->LogFile_Probabilities + "_sign" + logfile_suffix;
    IOController_->OpenNewFile(logfile_probs_sign_filename_);
  }



  void AddGradientsToHistograms(int thread_id,
                              int index_A,
                              int frame_id,
                              int ground_truth_label,
                              int n_dimension_B, //           = Settings_->StageOne_Dimension_B;
                              int n_dimension_C, //          = Settings_->StageOne_Dimension_C;
                              float dresponsefunction_dB,
                              int MiniBatchSize);

  void ComputeEntropy() {
    // Compute entropy of accumulated gradients so far.
    // TODO(stz): implement streaming version of this: how to deal with renormalization per update?
    spatial_entropy.ComputeEmpiricalDistribution();
    spatial_entropy.ComputeSpatialEntropy();

    spatial_entropy_sign.ComputeEmpiricalDistribution();
    spatial_entropy_sign.ComputeSpatialEntropy();
  }

  void LogEntropy() {

    spatial_entropy.LogToFile(Settings_->session_start_time, logfile_entropy_filename_);
    spatial_entropy_sign.LogToFile(Settings_->session_start_time, logfile_entropy_sign_filename_);

    // spatial_entropy.RecordExtrema();
  }

  void LogProbabilitiesToFile() {
    spatial_entropy.LogProbabilitiesToFile(Settings_->session_start_time, logfile_probs_filename_);
    spatial_entropy_sign.LogProbabilitiesToFile(Settings_->session_start_time, logfile_probs_sign_filename_);
  }

};

#endif



