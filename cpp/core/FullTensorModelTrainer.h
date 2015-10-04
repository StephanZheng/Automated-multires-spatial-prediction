// Copyright 2014 Stephan Zheng

#ifndef FULLTENSORMODELTRAINER_H
#define FULLTENSORMODELTRAINER_H

#include <boost/chrono.hpp>
#include <boost/timer.hpp>
#include <boost/unordered_map.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <boost/math/special_functions/fpclassify.hpp> // isnan

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
#include <assert.h>

#include "rapidjson/rapidjson.h"
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"
#include "rapidjson/reader.h"

#include "config/GlobalConstants.h"
#include "config/Settings.h"

#include "core/Trainer.h"
#include "core/DataBlob.h"
#include "core/QueueMessage.h"
#include "core/SpatialEntropy.h"

#include "util/IOController.h"
#include "util/PerformanceController.h"
#include "util/PrettyOutput.h"
#include "util/TrainUtil.h"

using namespace std;

class FullTensorModelTrainer : public Trainer {

public:
  SpatialEntropy spatial_entropy;

  // These are parameters to be learned
  Tensor3Blob Weight_U;
  Tensor3Blob Weight_U_transposed;
  Tensor3Blob Weight_U_Diff;
  Tensor3Blob Weight_U_DiffSq_Cache;
  Tensor3Blob Weight_U_Snapshot;

  VectorBlob Bias_A;
  VectorBlob Bias_A_Diff;
  VectorBlob Bias_A_DiffSq_Cache;
  VectorBlob Bias_A_Snapshot;

  // Learning hyperparameters
  float regularization_W;
  float regularization_Sparse_S;

  FullTensorModelTrainer(IOController *aIOController_,
                         MatrixBlob *aBlob_B_,
                         MatrixBlob *aBlob_BTransposed_,
                         MatrixBlob *aBlob_C_,
                         MatrixBlob *aBlob_CTransposed_,
                         GroundTruthLabel *aGroundTruthLabelsTrainValStrong_,
                         GroundTruthLabel *aGroundTruthLabelsTestStrong_,
                         GroundTruthLabel *aGroundTruthLabelsTrainValWeak_,
                         GroundTruthLabel *aGroundTruthLabelsTestWeak_,
                         Settings *aSettings_,
                         CurrentStateBlob *aCurrentStateBlob_) : \
  spatial_entropy(aSettings_->StageOne_Dimension_B,
                         aSettings_->SpatialEntropy_NumberOfBins,
                         aSettings_->SpatialEntropy_MinValue,
                         aSettings_->SpatialEntropy_MaxValue) {

    // Tell ThreeMatrixFactorTrainer where the loaded data sits
    IOController_                    = aIOController_;
    Blob_B_                          = aBlob_B_;
    Blob_BTransposed_                = aBlob_BTransposed_;
    Blob_C_                          = aBlob_C_;
    Blob_CTransposed_                = aBlob_CTransposed_;
    GroundTruthLabelsTrainValStrong_ = aGroundTruthLabelsTrainValStrong_;
    GroundTruthLabelsTestStrong_     = aGroundTruthLabelsTestStrong_;
    GroundTruthLabelsTrainValWeak_   = aGroundTruthLabelsTrainValWeak_;
    GroundTruthLabelsTestWeak_       = aGroundTruthLabelsTestWeak_;
    Settings_                        = aSettings_;
    CurrentStateBlob_                = aCurrentStateBlob_;

    int n_dimension_A                = Settings_->Dimension_A;
    int n_dimension_B                = Settings_->StageOne_Dimension_B;
    int n_dimension_C                = Settings_->StageOne_Dimension_C;
    int n_frames                     = Settings_->NumberOfFrames;
    int n_threads                    = Settings_->StageOne_NumberOfThreads;

    ConditionalScore.init(    n_frames,   Settings_->NumberOfColumnsScoreFunction);

    Weight_U.init(        n_dimension_A, n_dimension_B, n_dimension_C);
    // Weight_U_transposed.init(      n_dimension_A, n_dimension_C, n_dimension_B);
    Weight_U_Diff.init(     n_dimension_A, n_dimension_B, n_dimension_C);
    Weight_U_DiffSq_Cache.init( n_dimension_A, n_dimension_B, n_dimension_C);
    Weight_U_Snapshot.init(   n_dimension_A, n_dimension_B, n_dimension_C);

    Bias_A.init(        n_dimension_A);
    Bias_A_Diff.init(       n_dimension_A);
    Bias_A_DiffSq_Cache.init(   n_dimension_A);
    Bias_A_Snapshot.init(     n_dimension_A);

    // Train-val split creation
    // When constructor is called, we shuffle the indices of the strong-weak labels
    // We only randomize the indices ONCE!

    // Every cross-val run, we call the PermuteTrainValIndices method to permute through the train / val indices

    std::vector<int> strong_indices(Settings_->NumberOfStrongLabels_TrainVal);
    std::vector<int> weak_indices(Settings_->NumberOfWeakLabels_TrainVal);
    std::iota(strong_indices.begin(), strong_indices.end(), 0);
    std::iota(weak_indices.begin(), weak_indices.end(), 0);

    mt19937 g(static_cast<uint32_t>(time(0)));
    std::shuffle(strong_indices.begin(), strong_indices.end(), g);
    std::shuffle(weak_indices.begin(), weak_indices.end(), g);

    CurrentTrainIndices_Strong.resize(Settings_->NumberOfStrongLabels_Train);
    CurrentValidIndices_Strong.resize(Settings_->NumberOfStrongLabels_Val);
    CurrentTrainIndices_Weak.resize(Settings_->NumberOfWeakLabels_Train);
    CurrentValidIndices_Weak.resize(Settings_->NumberOfWeakLabels_Val);

    RandomizedIterator.resize(Settings_->NumberOfStrongLabels_Train + Settings_->NumberOfWeakLabels_Train);

    for (int i = 0; i < CurrentTrainIndices_Strong.size(); ++i) CurrentTrainIndices_Strong[i] = strong_indices[i];
    for (int i = 0; i < CurrentValidIndices_Strong.size(); ++i) CurrentValidIndices_Strong[i] = strong_indices[Settings_->NumberOfStrongLabels_Train + i];
    for (int i = 0; i < CurrentTrainIndices_Weak.size(); ++i)   CurrentTrainIndices_Weak[i] = weak_indices[i];
    for (int i = 0; i < CurrentValidIndices_Weak.size(); ++i)   CurrentValidIndices_Weak[i] = weak_indices[Settings_->NumberOfWeakLabels_Train + i];
    for (int i = 0; i < RandomizedIterator.size(); ++i)     RandomizedIterator[i] = i;

    // Momentum update coefficients
    NestorovMomentumLambda.resize(0);
    NestorovMomentumPreviousWeight.resize(0);
    NestorovMomentumPreviousLF.resize(0);
    NestorovMomentumPreviousSparse_S.resize(0);
    NestorovMomentumPreviousBias_A.resize(0);
  }

  int   DecisionUnit(int cross_val_run, int epoch, float loss_train, float loss_valid, float loss_test, int * have_recorded_a_trainloss);
  void  StoreStartingWeights();
  void  RestoreStartingWeights();

  // Routine that contains the stage 1 train logic
  int   TrainStageOne(string fp_snapshot);

  void  ThreadComputer(int thread_id);

  void  InitializeWeightsRandom(float range_W, float range_A);

  float   ComputeInnerproduct(int thread_id, int outer_row, int outer_col, int begin_index, int end_index, MatrixBlob *A, MatrixBlob *B);
  float   ComputeInnerproductTriple(int thread_id, int outer_row, int outer_col, MatrixBlob *A, MatrixBlob *B, MatrixBlob *C);

  float   ComputeTripleProductPsiPsiU(  int thread_id,
                                        int frame_id,
                                        int index_A,
                                        Tensor3Blob *Sparse_S,
                                        std::vector<int> *indices_B,
                                        std::vector<int> *indices_C);

  void  DebuggingTestProbe(int thread_id, int index_A, int frame_id, int ground_truth_label, int index);

  // void  ComputeConditionalScoreSingleThreaded(int type);

  virtual float   ComputeScore(int thread_id, int index_A, int frame_id);
  virtual int  ComputeConditionalScore(int thread_id, int index_A, int frame_id, int ground_truth_label);
  // virtual void  ComputeConditionalScoreSingleThreaded(int type);

  void  ComputeUtranspose();

  void  ComputeWeight_U_Update(int thread_id, int index_A, int frame_id, int ground_truth_label);
  void  ProcessWeight_U_Updates(int thread_id, int index_A);

  void  ComputeBias_A_Update(int thread_id, int index_A, int frame_id, int ground_truth_label);
  void  ProcessBias_A_Updates(int thread_id, int index_A);

  void  ComputeSpatialEntropy(int thread_id, int index_A, int frame_id, int ground_truth_label);

  void  Load_SnapshotWeights(string weights_snapshot_file, string momentum_snapshot_file);

  void  Store_Snapshot(int cross_val_run, int epoch, string fp_snapshot);

  void  ApplyNestorovMomentum(int batch);
};

#endif



